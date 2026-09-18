# PHASE 2 — CNN Model Benchmarking

Train and objectively compare **10 ImageNet-pretrained CNN backbones** for 5-class
KL-grade (0–4) knee-OA severity classification, then select ONE model on
**diagnostic performance + deployment efficiency**.

Standard 5-class classification only. No attention / CBAM / Grad-CAM / ordinal
loss / custom architecture in this phase (those are Phases 4–5).

## Fixed experimental controls (identical for all 10 models)

| item | value |
|---|---|
| data | `data/processed/images/basic/` only (Kaggle = OAI-00m ROIs) |
| split | Phase-1 patient-level split, frozen; **test never loaded for selection** |
| input | 3×224×224, grayscale replicated to 3 channels (never colourised), ImageNet norm |
| augmentation (train only) | `configs/augmentation.yaml` — rot ±8°, transl ±5%, scale 0.95–1.05, shear ±4°, hflip 0.5, brightness/contrast 0.90–1.10, Gaussian noise p0.2 σ≤0.02 |
| loss | Weighted Cross-Entropy, **train-only** class weights `metadata/class_weights.json` |
| optimizer | AdamW, lr 1e-4 (Stage B), weight decay 1e-4 |
| scheduler | cosine, 1-epoch warmup |
| transfer learning | Stage A: 3 epochs head-only @ lr 1e-3 → Stage B: unfreeze all @ lr 1e-4 |
| batch size | 64 |
| max epochs | 30 |
| early stopping | monitor `val_macro_f1`, patience 5, restore best |
| selection metric | **validation Macro-F1** (never accuracy alone, never test) |
| seed | 42 (Python/NumPy/torch/CUDA); cuDNN deterministic, `use_deterministic_algorithms(warn_only)` |
| AMP | on (documented as non-deterministic-tolerant) |

Models (`configs/benchmark.yaml`), all via **timm**:
`resnet18, resnet50, densenet121, efficientnet_b0, mobilenet_v2,
mobilenet_v3_large, efficientnet_b1, inception_v3, xception (legacy_xception),
convnext_tiny`.
Inception/Xception run at 224 px (native 299) for fairness — documented per-model.

## Commands

```bash
# 0. deps (Windows + Py3.14 + RTX 5080 / Blackwell sm_120)  -- ORDER MATTERS
pip install "torch==2.11.0" "torchvision==0.26.0" \
    --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps timm==1.0.28          # --no-deps: keep the +cu128 torch
pip install fvcore psutil tabulate huggingface_hub safetensors
# (plain `pip install timm` lets the resolver swap +cu128 torch for a CPU wheel
#  from PyPI and you lose the GPU. Keep torch pinned / use --no-deps for timm.)

# 1. plumbing check — build every model, few batches, shapes/NaN/loss/leakage
python scripts/benchmark_models.py --dry-run

# 2. sanity — ResNet18, 2 real epochs
python scripts/train_model.py --model resnet18 --epochs 2

# 3. full benchmark (all 10)
python scripts/benchmark_models.py

# subset / single
python scripts/benchmark_models.py --models resnet18,resnet50
python scripts/train_model.py --model convnext_tiny
```

## Outputs

```
models/benchmark/<model>/{best.pt, last.pt, config.json, history.json, result.json}
reports/benchmark/
  benchmark_results.csv / .json         # one row per model + rankings
  model_complexity.csv                  # params, GMACs, size
  training_history.csv                  # per-epoch, all models
  reproducibility.json                  # seed + full HW/SW fingerprint
  dry_run_report.json
  final_benchmark_report.md             # spec section 46 (21 sections) + SIH table
  classification_reports/<model>.json   # full metrics incl per-class + adjacent confusion
  confusion_matrices/<model>_confusion_matrix[_normalized].png
  training_curves/<model>_training_curve.png
  latency/<model>_latency.json          # GPU + CPU batch-1 latency, peak memory
  model_predictions/<model>_val_predictions.csv   # patient_id, side, true, pred, prob_kl0..4
  pareto_macroF1_vs_latency.png, pareto_macroF1_vs_size.png
```

## Metric notes

- **GMACs** via `fvcore.FlopCountAnalysis` (one fused multiply-add = 1). FLOPs ≈ 2×GMACs. Column labelled `gmacs`; no invented conversions.
- **Latency**: batch=1, `1×3×224×224`, warmup 20 + 100 timed (CPU: 5 + 30), CUDA-synchronised. GPU latency is development-only — **Raspberry-Pi numbers are NOT inferred here** (Phase 8).
- **QWK**: quadratic-weighted Cohen's κ — penalises far-off ordinal errors (KL0↔KL4 ≫ KL3↔KL4).
- **ROC-AUC**: one-vs-rest, macro-averaged; reported only when all 5 classes present in the eval split.

## Selection framework (spec §35)

Macro-F1 → QWK → Balanced Accuracy → per-class F1 (esp. KL3/KL4) → params → latency.
A much smaller/faster model within a small Macro-F1 margin (matching on QWK and
KL3/KL4 F1) is preferred. Recommendation is evidence-based, not auto-highest-F1.

## Medical framing

Research prototype for **AI-based KL-grade classification from knee X-rays**.
Not a clinical diagnostic claim; does not replace radiologist assessment.
