# AETHER-OA — Knee Osteoarthritis X-ray Grading

A compact, CPU-deployable knee-osteoarthritis (OA) severity grader for the
**Kellgren–Lawrence (KL 0–4)** scale, built by **distilling** two large CNNs
(VGG16 + ConvNeXt-Tiny) into a **MobileNetV2** student, then exporting to ONNX
and serving it from a Raspberry-Pi as a **headless REST inference API**
(FastAPI). The screening UI is a **separate frontend** (the OA-screening kiosk)
that uploads a radiograph and renders the grade, probabilities and CAM overlay.

> **Research prototype / decision-support system — NOT a medical device.**
> It does not perform diagnosis and is **not a substitute for professional
> radiological assessment.** See the disclaimer at the end.

---

## 1. Project title
**AETHER-OA** — X-ray module: automated KL grading of knee OA from plain radiographs.

## 2. Problem statement
Reading the KL grade of a knee radiograph is subjective; inter-reader agreement
is roughly 60–70%. A small, offline model that proposes a grade plus an
OA / no-OA screen — with a visual explanation and a "refer to a clinician" path
when unsure — is useful as a triage / teaching / decision-support aid on
low-cost hardware.

## 3. Objective
Produce **one** model, ≤ ~3 M parameters, that (a) grades KL0–KL4, (b) screens
for radiographic OA (KL≥2), (c) shows where it is looking, (d) calibrates its
confidence and abstains when uncertain, and (e) runs on a Raspberry Pi with **no
PyTorch and no GPU** at inference.

## 4. Dataset
Kaggle *Knee Osteoarthritis Dataset with Severity Grading* (`shashwatwork`) —
per-knee 224×224 grayscale ROIs that are verified to be **OAI baseline-visit (00m)
crops** (all 8,260 KL grades match `OAI_summary.csv` exactly). Working set:
**8,260 knees · 4,130 patients**. Pixels are **not** in this repo — see
[`data/README.md`](data/README.md).

## 5. Data split
Patient-disjoint, stratified by worst-knee grade, seed 42, 70 / 15 / 15:
**train 5,782 · val 1,238 · test 1,240** knees. Pairwise patient overlap = 0.
The **test split is frozen** and was never used for tuning, thresholds, or
checkpoint selection. Stored in `metadata/{train,val,test}.csv`.

## 6. Preprocessing (identical for train / val / test)
`read → grayscale → letterbox to square (fill 0) → resize 224×224 (INTER_AREA on
downscale) → /255 → replicate to 3 channels → ImageNet normalisation`
→ tensor `1 × 3 × 224 × 224`.
Torch-free implementation for deployment: `src/deploy/preprocess.py`.
Training-time augmentation (train only, clinically constrained):
`configs/augmentation_phase7_strong.yaml` — rotation ±12°, translate ±8%, scale
0.90–1.10, shear ±6°, horizontal flip p 0.5, brightness/contrast 0.85–1.15,
Gaussian noise p 0.30 σ≤0.03. Vertical flip, large rotation, elastic warp and
random erasing are **disabled** (they corrupt joint-space / osteophyte cues).

## 7. Model architecture
timm **`mobilenetv2_100`** backbone, 3-channel input, head `Linear(1280 → 5)`.
2,230,277 parameters, 0.313 GMACs.

```
Input 1×3×224×224
 → stem Conv2d(3→32, 3×3, s2) + BN + ReLU6                → 32×112×112
 → 17× inverted-residual bottleneck (expand ×t → 3×3 depthwise → 1×1 linear project, + residual)
   groups: (t,c,n,s) = (1,16,1,1)(6,24,2,2)(6,32,3,2)(6,64,4,2)(6,96,3,1)(6,160,3,2)(6,320,1,1)
 → conv_head Conv2d(320→1280, 1×1) + BN + ReLU6           → F : 1×1280×7×7
 → global average pool                                    → 1280
 → Linear(1280 → 5)                                        → logits z
 → softmax(z / T)                                          → calibrated probabilities p
```

## 8. Knowledge distillation (training only — not needed at inference)
The student learns from the **softened outputs of two frozen teachers**:

| Teacher | timm id | checkpoint | seed | val Macro-F1 | test Macro-F1 (15-model benchmark) |
|---|---|---|---|---|---|
| VGG16 | `vgg16` | `models/phase4/vgg16/best.pt` (Phase 4, epoch 21) | 42 | 0.7077 | **0.7382** |
| ConvNeXt-Tiny | `convnext_tiny` | `models/phase4/convnext_tiny/best.pt` (Phase 4, epoch 14) | 42 | 0.7115 | **0.7169** |

* Teacher class probabilities are **pre-computed once** over train+val and cached as
  `reports/phase8/teacher_logits/{train,val}__{vgg16,convnext_tiny}.npz` (≈305 KB total,
  committed). The KD training loop never loads a teacher model.
* **The teacher `.pt` files are large training-time artifacts and are NOT in this
  repo.** They are re-trainable from `configs/benchmark_phase4.yaml`; the cached
  logits are sufficient to reproduce the distillation.

**Distillation loss** (`src/benchmark/distill.py`):
```
L = α · T² · KL( q ‖ softmax(z_student / T) )  +  (1 − α) · weighted-CE( z_student , y_true )
    q = mean( softmax(z_vgg16 / T) , softmax(z_convnext / T) )
    α = 0.8      T (distillation) = 4      class weights = metadata/class_weights.json (train-split only)
```

## 9. Training methodology
`configs/benchmark_phase8.yaml`. AdamW (lr 1e-4, wd 1e-4). Two-stage transfer:
3 epochs head-only @ lr 1e-3, then full fine-tune @ lr 1e-4. Cosine schedule +
1-epoch warm-up. Batch 64, AMP, grad-clip 5.0. Max 70 epochs, early stop
patience 8 on **validation** Macro-F1. **Weight EMA** (decay 0.999) — the EMA
weights are evaluated and exported. Multi-seed 42 / 123 / 3407.
Deployed checkpoint: **`K6_kd_ens_long__seed123`** (selected on validation),
copied to `models/mobilenet_final/best.pt`.

## 10. Final MobileNetV2 architecture
As §7. Head is a bare `GAP → Linear(1280→5)` — this is what makes the
explainability grad-free (see §13). Deployed as two ONNX graphs:
`mobilenet_v2_oa.onnx` (logits only) and `mobilenet_v2_oa_cam.onnx`
(logits + the `1×1280×7×7` feature map `F`).

## 11. Calibration
Temperature scaling: `p = softmax(z / T)`, **T = 1.3934**, fitted once on the
**validation** set (NLL minimisation). Expected Calibration Error ≈ 0.10 → 0.057.
`argmax` and every rank metric are unchanged; only the confidence numbers move.
Stored in `models/deploy/mobilenet_v2_oa_deploy.json`.

## 12. Test-time augmentation
Horizontal-flip TTA (exact under per-channel normalisation; KL is side-agnostic)
is available in the predictor's eval path. On the deployed seed it is ~neutral
(test Macro-F1 0.6809 → 0.6804); across the 3 seeds it is +0.5–0.8 pt.
The API runs a single forward per request by default.

## 13. Grad-free CAM explainability
Because the head is `GAP → Linear`, Grad-CAM's channel weight collapses
algebraically to the classifier weight, so the map is exactly the original
**Class Activation Map**:

```
CAM_c(x, y) = ReLU( Σ_k  W[c, k] · F_k(x, y) )        W = mobilenet_v2_oa_cam.npz  (5 × 1280)
```

Exact for this architecture, **no gradients**, one forward pass, **+3.2 ms** on
one CPU core. `src/deploy/cam.py` upsamples 7×7 → 224, applies a JET colormap,
α-blends over the film, and derives a coarse focus region
(*central joint line* / *one compartment* / *diffuse–bicompartmental*).

## 14. Clinical outputs (all derived from the one 5-vector `p`)
| Output | Definition |
|---|---|
| KL grade | `argmax p` |
| Calibrated probabilities | `p₀ … p₄` (sum ≈ 1) |
| Expected grade (ordinal) | `Σ c · p_c` |
| Binary OA screen | `P(OA) = p₂ + p₃ + p₄` → OA if ≥ 0.5 |
| 3-class severity band | `[p₀, p₁+p₂, p₃+p₄]` → Normal / Early / Advanced |
| Confidence / abstention | `max p` ; `< 0.73` → REFER (threshold chosen on validation) |
| CAM heatmap | 224×224 overlay (§13) |
| Interpretation & recommendation | grade-banded templates (generic, not patient-specific) |

## 15. Final benchmark (frozen test set, n = 1,240)

**Deployed model — `K6_kd_ens_long__seed123` (KD-MobileNetV2, 2.23 M params):**

| Metric | Value |
|---|---|
| Accuracy | **0.6605** |
| Macro-F1 | **0.6809**  (hflip-TTA 0.6804 · 3-seed mean 0.686 ± 0.007) |
| Weighted-F1 | 0.6559 |
| Balanced accuracy | 0.6804 |
| Quadratic-weighted κ (QWK) | **0.8345** |
| MAE (grades) | 0.3863 |
| Within ±1 grade | 0.954 |
| Per-class F1 | KL0 0.763 · KL1 0.328 · KL2 0.622 · KL3 0.808 · KL4 0.883 |
| Binary OA screen (KL0-1 vs KL2-4) | **ROC-AUC 0.941** |
| 3-class (Normal / Early / Advanced) accuracy | 0.765 |
| ECE (pre → post temperature) | 0.057 → 0.058 |

**Progression of the MobileNetV2 result (test Macro-F1 / accuracy / QWK), authoritative sources in `reports/`:**

| Stage | Macro-F1 | Accuracy | QWK | Source |
|---|---|---|---|---|
| MobileNetV2 **baseline** (15-model benchmark) | 0.6547 | 0.6363 | 0.7958 | `reports/phase5/final_15_model_test_benchmark.csv` |
| MobileNetV2 Phase-7 tuning (strong-aug + hflip TTA) | ~0.696 | ~0.64 | ~0.806 | `reports/phase7/FINAL_MOBILENETV2_BENCHMARK.csv` |
| **KD-MobileNetV2 final (Phase 8)** | **0.6809** | **0.6605** | **0.8345** | `reports/mobilenet_final/FINAL_MOBILENETV2_BENCHMARK_phase8_kd.csv` |
| *(context) VGG16 teacher* | 0.7382 | 0.7258 | 0.8753 | `reports/phase5/…` |
| *(context) ConvNeXt-Tiny teacher* | 0.7169 | 0.6968 | 0.8618 | `reports/phase5/…` |

**Knowledge distillation added +0.026 Macro-F1 and +0.039 QWK to MobileNetV2 at
identical parameter count** (0.6547 → 0.6809 / 0.7958 → 0.8345).

**Label-scheme study** (same KD recipe, *separate* models — not the deployed one;
`reports/mobilenet_final/LABEL_SCHEME_SWEEP.csv`): 5-class Macro-F1 0.694 /
4-class (KL0+KL1 merged) 0.759, acc 0.788 / 3-class 0.713 / binary ROC-AUC 0.943.

Full result documents: `reports/mobilenet_final/` (CSV + `PROJECT_REPORT.html` +
`ARCHITECTURE.html` + `ARCHITECTURE_DETAIL.html`), `reports/phase8/`,
`reports/phase{5,6,7}/`, `docs/`.

## 16. Limitations
* **Single data source** — all data is OAI-derived; no independent external test
  set. Generalisation to other scanners / populations / positioning is unverified.
* **Label ceiling** — grades are single-reader OAI baseline reads; 5-class KL
  accuracy near 90% is not reachable by any model on this data. Published SOTA is
  ~0.70–0.75.
* **KL1 "doubtful" is the weak class** (F1 0.33) — it is the disagreement class by
  construction; the best of 15 benchmarked models reached only 0.49.
* **KL4 has wide confidence intervals** (38 test knees).
* Input is pre-cropped 224² — no full radiograph, no true pixel spacing, no
  joint-space width in mm.

## 17. Deployment architecture
```
X-ray → preprocess (NumPy + OpenCV) → mobilenet_v2_oa[_cam].onnx (ONNX Runtime, CPU)
      → temperature scale → softmax → KL grade / OA screen / 3-class / abstain
      → CAM (NumPy) → heatmap overlay + interpretation + recommendation
      → FastAPI JSON response → external frontend (OA-screening kiosk)
```
**No PyTorch, no CUDA, no timm, no teacher model at inference.** Verified by static
import analysis and by end-to-end execution on real test images.

## 18. Raspberry Pi deployment
Target: a Raspberry Pi running the headless inference API (`scripts/serve_api.py`);
the touchscreen UI is the separate OA-screening kiosk frontend pointed at it.
Runtime dependencies: **Python · ONNX Runtime · NumPy · OpenCV · FastAPI · uvicorn**
(`requirements-runtime.txt`). No CUDA. No PyTorch.

Latency **has only been measured on an x86 CPU** (single ONNX-Runtime thread,
batch 1, 224² input): **~3.5 ms** for grading, **~4.7 ms** with the heatmap;
resident memory ~92 MB; model 8.9 MB. **No Raspberry-Pi latency has been measured
yet** — do not cite one. `scripts/benchmark_edge.py` run *on the Pi* will produce
real numbers.

## 19. Inference API
`scripts/serve_api.py` — a headless FastAPI service (no UI; the frontend is a
separate app). Endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` · `/api/xray/meta` | — | model card, temperature, abstain threshold, CAM availability, frozen test metrics |
| POST | `/api/xray/grade` | `multipart/form-data`: `image=<file>`, optional `explain=true\|false`, `validate=true\|false` | `accepted:true` + KL grade + name, calibrated `dist[5]` (+ uncalibrated), confidence, ordinal expected grade, binary OA screen, 3-class band, abstain flag, templated findings, recommendation, **grad-free CAM overlay as a PNG `data:` URL**, model focus region, `input_check`, inference time |

`explain=true` (default) adds the Grad-CAM heatmap; `explain=false` is grading only.

### Input-validation gate (`src/deploy/validate.py`)

The KL head is a 5-class softmax — it returns a grade for *any* image. Before a
grade or heatmap is shown, `POST /api/xray/grade` runs a two-stage gate:

1. **Pre-flight image statistics** (NumPy/OpenCV, ~1 ms, no model): colour
   content, tonal continuity, contrast, histogram detail, edge/sharpness. A
   *hard* failure (colour photo, screenshot, colour-map, document) is rejected
   **without running the model**.
2. **Model-space OOD** on the same forward pass: free-energy of the logits,
   L2 norm and dead-unit fraction of the penultimate GAP-1280 feature vector.

A rejection returns `HTTP 200` with:
```json
{ "accepted": false, "reason": "colour",
  "message": "This looks like a colour photo or screenshot, not an X-ray. Please upload a knee radiograph.",
  "input_check": { "hard_fail": [...], "soft_fail": [...], "model_fail": [...], "metrics": {...} } }
```
An acceptance carries `"accepted": true` plus an `"input_check"` block of the
observed metrics.

Thresholds (`validate.GATE`) are calibrated against the 80 real knee-X-ray
tiles in `reports/samples/` plus synthetic non-X-ray images; regression test:
`python scripts/test_xray_gate.py` (**ID 160/160 accepted, OOD 9/9 rejected**).
Override per deployment via `mobilenet_v2_oa_deploy.json` → `"input_gate": {...}`,
or bypass entirely with the `validate=false` form field.

**Limitation:** the gate reliably catches the realistic wrong-file cases
(colour images, screenshots, documents, solid/gradient fills). It does *not*
reliably flag a smooth grayscale image or a non-knee radiograph (hand/chest
film) — for those the model's own `abstain` (confidence < 0.73 → "REFER") is
the backstop. A dedicated OOD detector would need real out-of-distribution
training data.

## 20. Installation
```bash
# ---- Inference / API only (no PyTorch) ----
python -m venv .venv && .venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements-runtime.txt

# ---- Full training / benchmarking environment ----
# 1) torch/vision from the CUDA 12.8 index, PINNED (see requirements.txt header):
pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
# 2) then the rest without deps so pip cannot swap in a CPU torch:
pip install --no-deps timm==1.0.28
pip install -r requirements.txt
```

## 21. Inference command (headless)
```bash
python - <<'PY'
from src.deploy.predictor import KneeOAPredictor
p = KneeOAPredictor("models/deploy/mobilenet_v2_oa.onnx")
print(p.predict("path/to/knee.png", explain=True))
PY
```

## 22. API launch command
```bash
python scripts/serve_api.py --host 0.0.0.0 --port 8000
# health check:   curl http://<device-ip>:8000/health
# grade an image: curl -F image=@knee.png http://<device-ip>:8000/api/xray/grade
```
Point the OA-screening kiosk frontend at it via `VITE_XRAY_API=http://<device-ip>:8000`.

Or the self-contained bundle (already path-rewritten, torch-free):
```bash
cd dist/aether_oa_xray_dashboard
pip install -r requirements-edge.txt
python serve_api.py --host 0.0.0.0 --port 8000
```

## 23. Evaluation commands
```bash
# grad-free CAM export + parity check
python scripts/export_onnx_cam.py

# on-device (or x86) latency / memory benchmark of the deploy ONNX
python scripts/benchmark_edge.py --real-images 128

# reproduce the final KD test benchmark (needs the training env + processed data)
python scripts/phase7_final_eval.py --config configs/benchmark_phase8.yaml --tag phase8_kd \
    --experiment-id K6_kd_ens_long__seed123
```

## 24. Repository structure
```
OA/
├── README.md  LICENSE  .gitignore
├── requirements.txt            training / benchmarking (torch + timm, pinned)
├── requirements-runtime.txt    inference / REST API (no torch)
├── configs/                    *.yaml — preprocessing, augmentation, every benchmark phase, KD (benchmark_phase8.yaml)
├── src/
│   ├── common.py
│   ├── datasets/xray_dataset.py
│   ├── preprocessing/          deterministic pipeline + stages
│   ├── benchmark/              engine, models, distill, metrics, experiments, plots … (training + research)
│   └── deploy/                 predictor.py  preprocess.py  cam.py  api.py   ← torch-free inference core + FastAPI
├── scripts/
│   ├── (data)        run_all, audit_dataset, analyze_*, build_manifest, make_splits,
│   │                 compute_class_weights, compute_norm_stats, prepare_dataset, validate_dataset
│   ├── (benchmark)   benchmark_phase{2..8}*, run_experiments, phase{6,7,8}_*, train_model
│   ├── (distill)     phase8_teacher_bank, benchmark_phase8, benchmark_phase8_schemes
│   ├── (deploy)      export_onnx, export_onnx_cam, benchmark_edge, serve_api
├── docs/                       PHASE2_BENCHMARK.md, PHASE3_EXTENDED_BENCHMARK.md
├── metadata/                   master_manifest.csv, {train,val,test}.csv, class_weights.json, norm stats
├── reports/                    every phase's metrics/plots + reports/mobilenet_final/ (the final results)
├── models/
│   ├── deploy/                 mobilenet_v2_oa.onnx  _cam.onnx  _cam.npz  _deploy.json   ← THE PRODUCT
│   ├── mobilenet_final/        best.pt + result.json + config.json + teacher_logits/*.npz
│   └── phase3|4|7/             per-model config/history/result JSON only (benchmark provenance)
├── dist/aether_oa_xray_dashboard/   self-contained edge bundle (Pi drop-in)
└── data/                       README.md only — pixels are NOT distributed
```

## 25. Dataset setup
See [`data/README.md`](data/README.md) — download the Kaggle dataset into a
**sibling** `kaggle/` folder, then `python scripts/run_all.py --quick` and
`python scripts/validate_dataset.py`.

## 26. Model artifact description
| File | Bytes | Purpose | Needed for inference? |
|---|--:|---|:--:|
| `models/deploy/mobilenet_v2_oa.onnx` | 8.89 MB | served model — `input 1×3×224×224 → logits 1×5` | **yes** |
| `models/deploy/mobilenet_v2_oa_deploy.json` | 1.3 KB | temperature (1.3934), abstain threshold (0.73), class names, frozen test metrics | **yes** |
| `models/deploy/mobilenet_v2_oa_cam.onnx` | 8.89 MB | logits + `features 1×1280×7×7` for the grad-free CAM | heatmap only |
| `models/deploy/mobilenet_v2_oa_cam.npz` | 26 KB | classifier `W (5×1280)`, `b (5)` for `ReLU(Σ W[g,k]·Fₖ)` | heatmap only |
| `models/mobilenet_final/best.pt` | 9.14 MB | canonical trainable checkpoint (source of the ONNX; EMA weights) | no — provenance |
| `models/mobilenet_final/teacher_logits/*.npz` | 305 KB | cached teacher soft targets — enough to re-run KD | no — training |

ONNX↔PyTorch parity: max |Δlogit| 1.2e-5, argmax 64/64. Every committed weight file
is < 10 MB — normal Git, no Git-LFS.

## 27. Reproducibility
* Seeds fixed (data split 42; KD final 123; multi-seed 42/123/3407).
* Frozen test split in `metadata/test.csv` — never touched for tuning.
* Every phase's per-run `config.json` + `result.json` + registry CSV is committed
  (`models/phase{3,4,7}/…`, `reports/**/*_REGISTRY.csv`).
* The KD recipe is fully specified in `configs/benchmark_phase8.yaml` +
  `models/mobilenet_final/config.json`; teacher soft targets are committed.
* To reproduce the final model: obtain the data (§25), install the training env
  (§20), then
  `python scripts/benchmark_phase8.py --stage kd` →
  `python scripts/benchmark_phase8.py --stage multiseed` →
  `python scripts/phase8_run_all.py`.

## 28. Disclaimer
This repository is a **research prototype and decision-support experiment**. It is
**not a medical device**, has not been clinically validated, and **must not be
used for diagnosis or to guide patient care**. The Kellgren–Lawrence grade is a
radiographic scale, not a clinical diagnosis of osteoarthritis. Any output is
**not a substitute for professional radiological assessment.** Use for research
and education only.
