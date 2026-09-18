# PHASE 3 — Extended CNN benchmark + controlled accuracy optimization

Builds on Phase 2 (10-CNN baseline). **Nothing in Phase 1/2 is modified.** The
frozen dataset, patient-level split, labels, test set and Phase-1 preprocessing
are untouched. Selection metric stays **validation Macro-F1**; the **test split
is never loaded** anywhere in Phase 3.

---

## PART A — 5 new architectures

Verified timm/torchvision identifiers in the installed env (timm 1.0.28,
torchvision 0.26.0+cu128):

| benchmark name | backend | id | pretrained | notes |
|---|---|---|---|---|
| `vgg16` | timm | `vgg16` | `vgg16.tv_in1k` | classic VGG-16 (no BN) |
| `vgg19` | timm | `vgg19` | `vgg19.tv_in1k` | classic VGG-19 (no BN) |
| `googlenet` | torchvision | `googlenet` | `GoogLeNet_Weights.IMAGENET1K_V1` | Inception-v1; aux classifiers disabled (no timm equivalent) |
| `squeezenet` | torchvision | `squeezenet1_1` | `SqueezeNet1_1_Weights.IMAGENET1K_V1` | 1×1-conv classifier |
| `shufflenet` | torchvision | `shufflenet_v2_x1_0` | `ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1` | ShuffleNet-V2 1.0× |

`src/benchmark/models.py` now has a **backend registry** (`MODEL_ZOO: name ->
(backend, id)`). timm models use `create_model(num_classes=5, in_chans=3)`;
torchvision models get their head replaced manually (`fc`→Linear(·,5), SqueezeNet
`classifier[1]`→Conv2d(512,5,1)). All take native 3-channel input (grayscale is
replicated upstream), all at 224×224.

**Identical protocol to Phase 2** for every new model: same train/val/test split,
augmentation (`configs/augmentation.yaml`), weighted-CE baseline with **train-only**
class weights, AdamW (lr 1e-4 Stage B / 1e-3 Stage A), cosine schedule + 1-ep
warmup, Stage A 3 ep head-only → Stage B full fine-tune, max 30 ep, early-stop
patience 5 on val Macro-F1, seed 42, AMP, same metrics / GMACs (fvcore) / latency
(batch-1 GPU+CPU) / checkpoint layout.

**OOM control (not silent):** VGG-16/19 are ~135–140 M params with large
activation maps → `configs/benchmark_phase3.yaml:per_model_batch_size` sets
`vgg16: 16`, `vgg19: 16` so they fit 16 GB VRAM. All other models keep batch 64
for parity. This is a memory necessity, documented per-model in `config.json`.

## PART B — extended 15-model report

`scripts/benchmark_phase3.py`:
1. trains **only the 5 new models** → `models/phase3/<m>/{best,last}.pt, config.json, history.json, result.json`
2. **reuses** the 10 Phase-2 `result.json` (frozen, never retrained)
3. merges 10 + 5 → `reports/phase3/extended_benchmark_results.{csv,json}`, `model_complexity.csv`, `pareto_*.png`, `final_phase3_report.md`; copies the Phase-2 confusion matrices / curves / per-class reports into `reports/phase3/` so the folder is self-contained.

CSV columns: model, accuracy, macro_f1, weighted_f1, balanced_accuracy,
quadratic_weighted_kappa, cohen_kappa, macro_precision, macro_recall, KL0–4 F1,
parameters, model_size_mb, gmacs, mean_latency_ms (GPU), cpu_mean_latency_ms,
throughput_images_per_sec, peak_memory_mb, best_epoch, epochs_run, training_time,
+ rankings. Each row tagged `phase` = phase2 / phase3.

## PART C — controlled optimization (`scripts/run_experiments.py`)

**One dimension changed per run** vs `configs/benchmark_phase3.yaml`; everything
else fixed. Every run appends ONE row to `reports/phase3/experiment_results.csv`
(**append-only** — existing `experiment_id`s are never overwritten; `--force` to
redo). Per-run artefacts in `models/phase3/experiments/<experiment_id>/`.

| family | flag | runs per model |
|---|---|---|
| Exp 1 preprocessing | `--experiment preprocessing` | basic, clahe, histeq |
| Exp 2 schedule | `--experiment schedule` | (max30,pat5), (max50,pat8) |
| Exp 3 fine-tuning | `--experiment finetune` | A: head3, B: head5, C: differential LR (backbone 1e-5 / head 1e-4) |
| Exp 4 loss | `--experiment loss` | weighted_cross_entropy, focal (γ=2), class_balanced (effective-number weights, already in `metadata/class_weights.json`) |

Ordinal loss is **not** introduced (deferred until Exp 1–4 are done + documented).
Differential LR uses two AdamW param groups (`split_param_groups()` — backbone vs
classifier head). Class-balanced loss = CE with the Cui et al. effective-number
weights (β=0.9999), computed **train-only** in Phase 1.

## PART D — spatial-attention experiment (`resnet18_spatial_attention`)

`src/benchmark/attention.py`. **Does not touch the plain `resnet18` baseline.**

```
input 3x224x224
   -> ResNet-18 feature backbone (pretrained, classifier+pool removed)  F [B,512,7,7]
   -> SpatialAttention(F)                                               A [B,1,7,7] in (0,1)
   -> F_att = F * (1 + A)          (residual channel-broadcast gate)
   -> AdaptiveAvgPool2d(1) -> flatten                                   [B,512]
   -> Dropout(0.2) -> Linear(512 -> 5)                                  logits [B,5]
```

**SpatialAttention block** (CBAM spatial sub-module, Woo et al. 2018):
- input: `F` `[B, C, h, w]` (ResNet-18 @224 → `C=512, h=w=7`)
- computation: channel avg-pool + channel max-pool → `[B, 2, h, w]`; a single
  `Conv2d(2, 1, kernel_size=7, padding=3)`; `Sigmoid`
- activation: sigmoid → `A ∈ (0, 1)`
- output: `A` `[B, 1, h, w]`; applied as `F_att = F * (1 + A)`
- **parameter count: 99** (`2·1·7·7 + 1` bias) — negligible vs the 11.18 M backbone
- integration point: immediately after the final backbone conv feature map,
  before global pooling and the classifier

Benchmarked against plain `resnet18` under identical config. Stage A trains the
attention block + classifier (2 664 params), backbone frozen; Stage B unfreezes
all. `convnext_tiny_spatial_attention` (feat_dim 768) is also available — run it
only after the ResNet-18 attention run is verified.

Run: `python scripts/run_experiments.py --experiment attention`

## PART E — feature extraction / softmax (as implemented)

- The **CNN backbone is the feature extractor**. Its output feature map is
  globally pooled to a feature vector.
- The **final linear layer produces exactly 5 raw logits** — `[KL0, KL1, KL2,
  KL3, KL4]`. No activation on the logits.
- **Training:** `torch.nn.CrossEntropyLoss(weight=class_weights)` consumes the
  **raw logits** directly. `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`
  internally, so **Softmax is never applied manually before the loss**
  (`src/benchmark/engine.py:train_one_epoch`). Applying softmax first would
  double-count and is a bug — it is not done.
- **Inference / reporting:** `probs = torch.softmax(logits, dim=1)` →
  `[P(KL0..KL4)]`; `pred = argmax(probs)`. Confidence = `max(probs)`.
  (`src/benchmark/engine.py:evaluate`, exported to
  `reports/*/model_predictions/<model>_val_predictions.csv`.)

## PART F — scientific controls (preserved)

Test set untouched · patient-level split unchanged · seed 42 (except Part G) ·
train-only class weights · primary metric val Macro-F1 · QWK + Balanced Accuracy
+ per-class F1 + KL4 F1 + confusion matrix always reported · model selection
never uses the test set.

## PART G — multi-seed robustness

`python scripts/run_experiments.py --experiment multiseed --models <best> --variant <v> --loss <l>`
runs seeds **42, 123, 3407** for the chosen 1–2 configs and writes
`reports/phase3/multi_seed_results.csv` + `multi_seed_summary.csv` (mean ± std of
Macro-F1, QWK, Balanced Accuracy) + `multiseed_comparison.png` (bars with std
error bars). A config is only "better" if the improvement exceeds seed std.

## PART H — experiment tracking

`reports/phase3/experiment_results.csv` — append-only master table, columns:
experiment_id, family, model, preprocessing, loss, optimizer, learning_rate,
stage_a_epochs, max_epochs, patience, differential_lr, augmentation, seed,
best_epoch, epochs_run, val_accuracy, val_macro_f1, val_qwk,
val_balanced_accuracy, KL0–4_F1, parameters, gmacs, latency_ms_gpu,
latency_ms_cpu, training_time_sec, stopped_early, checkpoint.
Per-family views: `preprocessing_ablation.csv`, `training_schedule_ablation.csv`,
`finetune_ablation.csv`, `loss_ablation.csv`, `attention_ablation.csv`,
`multi_seed_results.csv`.

## PART I — final selection logic

Primary **Macro-F1**, then QWK → Balanced Accuracy → KL3/KL4 F1 → parameter count
→ latency. Never highest accuracy alone; never the test set. If two configs are
within one seed-std on Macro-F1 and match on QWK/KL3-KL4, prefer the
smaller/faster one — but do not trade away a substantial Macro-F1 margin for size.

## Commands

```powershell
# A. dry-run the 5 new models (build + few batches, no training)
.\.venv\Scripts\python.exe scripts\benchmark_phase3.py --dry-run

# B. 15-model extended benchmark (trains the 5 new; reuses the 10 Phase-2 results)
.\.venv\Scripts\python.exe scripts\benchmark_phase3.py 2>&1 | Tee-Object reports\phase3\run_extended.txt
#    (+ optional) attention models in the same folder:
.\.venv\Scripts\python.exe scripts\benchmark_phase3.py --models resnet18_spatial_attention,convnext_tiny_spatial_attention --no-merge

# C. controlled experiments on the top-N (replace with your actual top models
#    from extended_benchmark_results.csv):
.\.venv\Scripts\python.exe scripts\run_experiments.py --experiment preprocessing --models convnext_tiny,densenet121,inception_v3
.\.venv\Scripts\python.exe scripts\run_experiments.py --experiment schedule      --models convnext_tiny,densenet121
.\.venv\Scripts\python.exe scripts\run_experiments.py --experiment finetune      --models convnext_tiny,densenet121
.\.venv\Scripts\python.exe scripts\run_experiments.py --experiment loss          --models convnext_tiny,densenet121
.\.venv\Scripts\python.exe scripts\run_experiments.py --experiment attention

# D. multi-seed the winning config
.\.venv\Scripts\python.exe scripts\run_experiments.py --experiment multiseed --models convnext_tiny --variant <best_variant> --loss <best_loss>
```
