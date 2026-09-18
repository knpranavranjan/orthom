# data/raw/ — source data pointers

Raw source data is **not copied** into `data/raw/` to avoid duplicating ~212 MB
of images inside a cloud-synced folder. The pipeline reads the sources in place
via `--input-dir` / config `paths.raw_root`. Nothing in these source trees is
ever modified.

| logical name | on-disk location | contents |
|---|---|---|
| `raw/kaggle` | `../../kaggle/` | Kaggle *Knee Osteoarthritis Dataset with Severity Grading* (shashwatwork). `train/ val/ test/ auto_test/` × class `0..4`, 9 786 PNG, 224×224, 8-bit grayscale. |
| `raw/oai`   | `../../OAI-KL-Grade-Classification/` | denizlab OAI-KL-Grade-Classification repo. **Metadata CSVs + reference code + model weights only — no pixel data.** |

## Provenance (verified — see `reports/cross_dataset_overlap.csv`)

- Every one of the 4 130 Kaggle patient IDs exists in `OAI_summary.csv`.
- All 8 260 Kaggle `train+val+test` knees match the OAI **baseline visit (00m)**
  KL grade **exactly** (0 disagreements).
- ⇒ The Kaggle images **are** OAI 00m per-knee ROIs. The two sources are **not
  independent**. They must **never be merged** (that is literally the same scan
  on both sides of a train/test boundary).

To obtain real OAI pixel data you must apply for the OAI Data Use Agreement
(<https://nda.nih.gov/oai/>) and run the detector pipeline in
`OAI-KL-Grade-Classification/oai-knee-detection/`. That is out of scope for the
current preprocessing phase.
