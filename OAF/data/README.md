# `data/` — dataset setup

**The knee-radiograph pixels are NOT distributed with this repository.**
This folder ships only documentation and the expected directory layout. Follow
the steps below to place the data locally; everything under `data/` (except this
README) is git-ignored.

---

## 1. Dataset

| | |
|---|---|
| **Name** | *Knee Osteoarthritis Dataset with Severity Grading* (Kaggle, user `shashwatwork`) |
| **What it is** | Per-knee region-of-interest crops from the **Osteoarthritis Initiative (OAI)** baseline visit (00m), 224×224, 8-bit grayscale PNG |
| **Labels** | Kellgren–Lawrence grade: **KL0** normal · **KL1** doubtful · **KL2** minimal · **KL3** moderate · **KL4** severe |
| **Working set** | **8,260 knees · 4,130 patients** (both knees per patient) — the Kaggle `train + val + test` folders. `auto_test/` is a re-detection of the same knees and is **excluded** (leakage). |
| **Approx. download** | ~200 MB (all 9,786 PNGs) |
| **Provenance** | Verified: every Kaggle patient ID is in `OAI_summary.csv`, and all 8,260 KL grades match the OAI 00m reading exactly. The Kaggle images **are** OAI 00m ROIs — see `reports/cross_dataset_overlap.csv`. They must never be merged with any other OAI extract (same scans). |
| **Licence** | OAI Data Use Agreement + Kaggle dataset terms. Redistribution of pixels is not permitted here — obtain it yourself. |

## 2. Class distribution (working set)

| KL | count | share |
|---|--:|--:|
| 0 Normal | 3,253 | 39.4% |
| 1 Doubtful | 1,495 | 18.1% |
| 2 Minimal | 2,175 | 26.3% |
| 3 Moderate | 1,086 | 13.1% |
| 4 Severe | 251 | 3.0% |

## 3. Split methodology (already computed — see `metadata/`)

Patient-disjoint, **stratified by each patient's worst-knee grade**, seed 42, 70 / 15 / 15.
Both knees of a patient always move together. Pairwise patient overlap = 0 / 0 / 0.

| split | patients | knees | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| train | 2,891 | 5,782 | 2,293 | 1,048 | 1,509 | 756 | 176 |
| val | 619 | 1,238 | 484 | 220 | 338 | 159 | 37 |
| **test** (frozen) | 620 | 1,240 | 476 | 227 | 328 | 171 | 38 |

The split is stored in `metadata/{train,val,test}.csv` and `metadata/master_manifest.csv`.
**Never re-split, never evaluate on `test` during development.**

## 4. Preprocessing (deterministic, identical for train/val/test)

`read → grayscale → validate → letterbox to square (fill 0) → resize 224 (INTER_AREA
on downscale) → contrast variant`. For pretrained backbones the grayscale plane is
replicated to 3 channels and ImageNet-normalised at load time. Three contrast variants
can be materialised (`basic`, `clahe`, `histeq`); **`basic` is what the final model uses.**
Implementation: `src/preprocessing/`, config `configs/preprocessing.yaml`.

## 5. How to obtain and place the data

```
projects/
├── OA/                                  ← this repository
└── kaggle/                              ← download target (SIBLING of OA/, git-ignored)
    ├── train/  {0,1,2,3,4}/*.png
    ├── val/    {0,1,2,3,4}/*.png
    ├── test/   {0,1,2,3,4}/*.png
    └── auto_test/ ...                   (present but ignored by the pipeline)
```

1. Kaggle account + CLI: `pip install kaggle` and place `kaggle.json` per Kaggle's docs.
2. Download (into the **sibling** `kaggle/` folder, not inside the repo):
   ```bash
   cd ..                     # parent of OA/
   kaggle datasets download -d shashwatwork/knee-osteoarthritis-dataset-with-severity
   unzip -q knee-osteoarthritis-dataset-with-severity.zip -d kaggle
   ```
3. (Optional, for OAI cross-checks) clone the reference repo as a sibling:
   ```bash
   git clone https://github.com/denizlab/OAI-KL-Grade-Classification
   ```
   It contains **metadata CSVs + reference code + weights only — no pixels.**

## 6. Build the training-ready data

From the repository root, with the training environment installed (`requirements.txt`):

```bash
python scripts/run_all.py --quick          # 'basic' variant only  (recommended)
# or:  python scripts/run_all.py           # all three contrast variants
python scripts/validate_dataset.py         # 29-check hard gate — must pass
```

This writes:

```
data/
├── raw/            pointers only (README)
├── interim/        pre-resize grayscale copies      (git-ignored, regenerable)
├── processed/
│   └── images/
│       └── basic/  {train,val,test}/{0,1,2,3,4}/*.png   ← used by training + eval
└── splits/
metadata/
├── master_manifest.csv            9,786 rows — full provenance per image
├── processed_manifest_basic.csv   8,260 rows — path + per-image mean/std
├── {train,val,test}.csv
├── class_weights.json             train-split-only balanced weights
└── normalization_stats.json
```

## 7. What each downstream task needs

| Task | Needs |
|---|---|
| **Dashboard / inference** | *nothing* — the model accepts an arbitrary uploaded image |
| **Retrain KD-MobileNetV2** | `data/processed/images/basic/{train,val}` + `metadata/processed_manifest_basic.csv` + `metadata/class_weights.json` + `reports/phase8/teacher_logits/*.npz` |
| **Regenerate teacher-logit bank** | `data/processed/images/basic/` + teacher ONNX/checkpoints (see `README.md` §8) |
| **Reproduce Phases 2–7** | all three contrast variants + `data/interim/` (or regenerate) |
