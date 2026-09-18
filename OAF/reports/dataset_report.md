# AETHER-OA — X-ray Module: Dataset Report

**Scope:** DATASET → AUDIT → CLEANING → STANDARDIZATION → PATIENT/IMAGE SPLIT →
PREPROCESSING → AUGMENTATION → DATASET VERSION → TRAINING-READY DATA.
**CNN training is NOT started.** Generated 2026-08-27. Seed = 42 everywhere.

---

## 1. Dataset sources

| tag | location | type | images |
|---|---|---|---|
| **Kaggle** — *Knee Osteoarthritis Dataset with Severity Grading* (shashwatwork) | `kaggle/` | knee-ROI PNGs, KL-graded | **9 786** |
| **OAI repo** — denizlab/OAI-KL-Grade-Classification | `OAI-KL-Grade-Classification/` | metadata CSV + code + weights | **0** (no pixel data in repo) |

The OAI repo ships `OAI_summary.csv` (55 191 knee records, 4 508 patients),
classifier/detector split CSVs, reference PyTorch code (ResNet/CBAM, GradCAM) and
`.pt/.pth` weights. It contains **no radiographs** — those require the OAI Data
Use Agreement plus running their detector pipeline. So **no OAI images were
available** for this phase.

## 2. Dataset provenance  (`reports/cross_dataset_overlap.csv`)

Rigorously checked, not assumed:

- 4 130 / 4 130 Kaggle patient IDs are present in `OAI_summary.csv`.
- 8 260 / 8 260 Kaggle `train+val+test` knees match the OAI **baseline visit
  (00m)** KL grade **exactly** — 0 disagreements, 0 not-found.

**Verdict: the Kaggle images ARE OAI 00m per-knee ROIs.** The two sources are
**the same underlying data**, not independent. Consequences:

- Merging OAI + Kaggle would put identical scans on both sides of a train/test
  boundary → **do not merge**.
- OAI cannot serve as a clean *external* test set for these patients (same
  people; other visits would still be patient-leakage).

## 3. Total images & structure

9 786 PNG, **100 % readable, 0 corrupt**, all **224 × 224**, all **1-channel
grayscale, uint8**, all PNG. Folder layout `<split>/<class 0-4>/<file>.png`.

Two filename conventions for the same knees:
`train/ val/ test/` → `<OAI_ID><L|R>.png`; `auto_test/` → `<OAI_ID>_<1|2>.png`
(1 = R, 2 = L).

| folder | KL0 | KL1 | KL2 | KL3 | KL4 | total |
|---|--:|--:|--:|--:|--:|--:|
| train | 2286 | 1046 | 1516 | 757 | 173 | 5778 |
| val | 328 | 153 | 212 | 106 | 27 | 826 |
| test | 639 | 296 | 447 | 223 | 51 | 1656 |
| auto_test | 604 | 275 | 403 | 200 | 44 | 1526 |
| **all** | **3857** | **1770** | **2578** | **1286** | **295** | **9786** |

### `auto_test` is a redundant re-detection of `test`

All 1 526 `auto_test` knees (by patient_id+side) are also in `test`; `test` has
130 extra knees. `auto_test` crops differ from `test` crops (pHash Hamming 8–16)
— it is an **auto-detected re-crop** of the same knees, with **21 KL-label
conflicts** vs `test`. **`auto_test` is excluded** from the working set
(`split = excluded_autotest`) to prevent patient leakage; kept in the manifest
for optional robustness testing.

## 4. Working set (manual, patient-disjoint)

`train + val + test` folders = **8 260 knees = 4 130 patients × 2 knees each**.

| KL | count | % |
|---|--:|--:|
| 0 Normal | 3253 | 39.4 |
| 1 Doubtful | 1495 | 18.1 |
| 2 Minimal | 2175 | 26.3 |
| 3 Moderate | 1086 | 13.1 |
| 4 Severe | 251 | 3.0 |

Imbalance ratio (max/min) ≈ **13.0**; minority/majority ≈ 0.077. **Not
rebalanced** — original distribution preserved; handled at train time via loss
weighting (§12).

## 5–7. Image dimensions / formats / corruption

Uniform 224×224×1 uint8 PNG. **0 corrupt, 0 unreadable, 0 exact (SHA-256)
duplicates** among all 9 786 files.

## 8. Quality analysis  (`reports/suspicious_images.csv`)

Per-image checks (readability, channels, all-black/all-white, dynamic range,
size, aspect ratio, format). **1 image flagged** of 9 786 (0.01 %):

- `kaggle/train/2/9961307R.png` — `LOW_DYNAMIC_RANGE` (std 7.5, range 28–80).
  Low-contrast but a valid film. **Flagged, not removed.** It landed in `train`.

No dark/bright/size/aspect/format anomalies.

## 9. Duplicates  (`reports/duplicates.csv`)

| kind | result |
|---|---|
| Exact (SHA-256) | **0** across all 9 786 |
| Knee-identity across folders | 1 526 knees in both `test` & `auto_test` (→ `auto_test` excluded); **21** carry conflicting KL labels |
| Intra-`train/val/test` duplicates (same knee twice) | **0** |
| Perceptual (pHash Hamming ≤ 4) | 425 clusters. 241 involve `auto_test`. 184 are within the manual set — **all cross-patient look-alikes** (different OAI IDs), i.e. pHash false positives on visually similar normal knees at 224 px, **not** true duplicates and **not** leakage. No action. |

## 10. Patient identifiers & leakage control

- `patient_grouping_available = true`. `patient_id` = OAI participant ID parsed
  from the filename; `knee_side` ∈ {L,R}. Not invented.
- Original Kaggle `train/val/test` folders are **already patient-disjoint**
  (0 shared patients pairwise) — but we still re-split for a controlled,
  reproducible 70/15/15 (the original is ~70/10/20) while keeping
  `original_split` in the manifest.

## 11. Split methodology  (`scripts/make_splits.py`)

- **Unit = patient** (both knees move together). **Stratified by the patient's
  worst-knee KL grade.** `sklearn.train_test_split`, seed 42, 70 / 15 / 15.

| split | patients | knees | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| train | 2891 | 5782 | 2293 | 1048 | 1509 | 756 | 176 |
| val | 619 | 1238 | 484 | 220 | 338 | 159 | 37 |
| test | 620 | 1240 | 476 | 227 | 328 | 171 | 38 |

Knee fractions 0.700 / 0.150 / 0.150. Per-class proportion drift across splits
< 0.013. **Patient overlap train/val/test = 0 / 0 / 0.** (`scripts/validate_dataset.py`
re-verifies all of this and 29 checks total — all pass.)

**Test set is frozen** — not used for any preprocessing/aug/normalization/weight
decision (§ spec 13).

## 12. Class imbalance handling

Class weights computed from **train split only** →
`metadata/class_weights.json`. Baseline = **Weighted Cross-Entropy** with
`balanced` weights:

| KL | 0 | 1 | 2 | 3 | 4 |
|---|--:|--:|--:|--:|--:|
| weight | 0.5043 | 1.1034 | 0.7663 | 1.5296 | 6.5705 |

Also provided: normalised inverse-frequency, class-balanced effective-number
(β = 0.9999). `WeightedRandomSampler` and Focal Loss are wired as config options
in the dataloader but not the default.

## 13. Preprocessing  (`src/preprocessing/`, `configs/preprocessing.yaml`)

Deterministic pipeline (train = val = test):
`read → grayscale → validate → ROI → pad-to-square → resize 224 → contrast`.

- **ROI:** `mode = provided` (no-op). Kaggle images are already isolated knee
  ROIs, so **no knee detector is trained**. `full` / `center_crop` kept
  configurable for future full-radiograph input.
- **Resize:** 224×224, letterbox pad first (images are already square → pad is a
  no-op), `INTER_AREA` on downscale.
- **Channels:** grayscale kept; **3-channel replicated grayscale** for pretrained
  ImageNet CNNs (never colourised). 1-channel also supported.
- **Contrast experiment (A/B/C, not pre-judged):** three full variants
  materialised — `basic` (none), `clahe` (clip 2.0, 8×8), `histeq`. On-disk
  copies stay lossless uint8 PNG.
- **Normalization:** two modes. *dataset_grayscale* (MODE A) stats computed
  **train-only** → `metadata/normalization_stats.json`
  (`basic`: mean 0.60916, std 0.19389; `clahe`: 0.61092 / 0.18866;
  `histeq`: 0.50421 / 0.29106). *imagenet* (MODE B) for pretrained backbones.
  Float scaling + normalization + channel replication happen at **load time**
  (`src/datasets/xray_dataset.py`), not on disk.
- Originals preserved: source untouched; pre-resize grayscale copies in
  `data/interim/cleaned/`; processed in `data/processed/images/<variant>/…`.

## 14. Augmentation  (`configs/augmentation.yaml`) — TRAIN ONLY

Clinically constrained: rotation ±8°, translation ±5 %, scale 0.95–1.05, shear
±4°, horizontal flip p 0.5 (justified — KL grade is side-agnostic, L/R already
normalised per file, matches OAI/DeepKnee reference), brightness/contrast
0.90–1.10, light Gaussian noise (p 0.2, σ ≤ 0.02).
**Disabled:** vertical flip, arbitrary/large rotation, heavy colour jitter,
elastic warp, random erasing, large random crop. Val/test = deterministic, no aug.

## 15. Dataset version / layout

```
data/raw/            pointers only (sources read in place, never modified)
data/interim/cleaned/ pre-resize grayscale copies (8 260)
data/processed/images/{basic,clahe,histeq}/{train,val,test}/{0..4}/  (24 780)
metadata/  master_metadata.csv, master_manifest.csv, train/val/test.csv,
           processed_manifest_{basic,clahe,histeq}.csv,
           class_weights.json, normalization_stats.json
```

`master_manifest.csv` (9 786 rows) columns: sample_id, dataset_source,
patient_id, knee_side, original_path, original_split, kl_grade, class_name,
width, height, channels, file_format, sha256, perceptual_hash, quality_status,
duplicate_status, split.

## 16. Limitations

1. **Single source.** Only OAI-derived data; **no independent external test set**
   is available (OAI repo has no pixels; Kaggle = OAI 00m). External validation
   is deferred until non-OAI radiographs are obtained.
2. **Labels are OAI KL grades** (baseline reading), inheriting OAI reader
   variability; no independent re-grade. 21 KL conflicts exist between `test` and
   the excluded `auto_test`.
3. **KL1 is intrinsically ambiguous** ("doubtful"); kept as its own class (not
   merged) per spec.
4. **Severe class imbalance** — KL4 = 3 % (176 train knees). Weighting helps but
   KL4 metrics will have wide confidence intervals.
5. Images are **pre-cropped at 224²**; original full radiographs and true pixel
   spacing are unavailable → no joint-space-width in mm, limited zoom headroom.
6. pHash near-duplicate scan is advisory only at this resolution.
7. Python 3.14 + very recent wheels (numpy 2.5, pandas 3.0, opencv 5.0); torch is
   **not** installed yet (not needed pre-benchmarking).

## 17. Recommended training strategy

**STRATEGY A — train on Kaggle (OAI-00m) only.**

| strategy | verdict |
|---|---|
| A — Kaggle only | ✅ **Recommended.** 8 260 patient-disjoint knees, all 5 classes, uniform quality, reliable patient IDs, exact KL provenance. |
| B — OAI repo only | ❌ Impossible now — repo has no images. |
| C — combined | ❌ Rejected — Kaggle *is* OAI 00m; merging = guaranteed leakage. |

Use the patient-level 70/15/15 split; baseline = Weighted-CE, 3-ch replicated
grayscale, ImageNet norm for pretrained backbones (dataset-grayscale norm for
from-scratch). Run the `basic` vs `clahe` vs `histeq` contrast comparison as a
first ablation — do **not** assume CLAHE wins. Report Macro-F1, Balanced
Accuracy, Cohen's κ (quadratic), per-class F1 and confusion matrix alongside
accuracy. Keep `test` untouched until final benchmarking. When a non-OAI dataset
is acquired, add it as a held-out external test — never fold it into training.

## Medical framing

This is an **AI-assisted screening / severity-estimation** research prototype,
**not** a diagnostic device. KL grade is a radiographic scale and not a complete
clinical OA diagnosis. Downstream UI wording: *"AI-assisted assessment — clinical
correlation recommended."*

---

### Validation status

`python scripts/validate_dataset.py` → **29 / 29 checks pass.** Dataset is
**ready for CNN benchmarking** (which is intentionally not started).
