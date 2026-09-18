# Raw data provenance

`oai_matchthem_demographic.rdata` / `.csv` — the `osteoarthritis` dataset bundled
with the R package **MatchThem** (CRAN), built from the publicly available
**Osteoarthritis Initiative (OAI)** cohort. Downloaded 2026-09-18 from:

    https://raw.githubusercontent.com/cran/MatchThem/master/data/osteoarthritis.rdata

No registration/DUA needed for this derived extract — it ships as package data.
The full OAI dataset (with WOMAC scores, KL grades, imaging) is *not* included
here; access to that requires an NIH NDA account and a signed data use
agreement, which this dataset sidesteps by using only demographic + outcome
columns that MatchThem's authors already extracted and republished.

2,585 participants, age 60-79 at OAI enrollment, 7 columns:

| column | meaning | coding |
|---|---|---|
| `AGE` | age in years | int |
| `SEX` | sex | 1 / 2 (mapping not published by MatchThem; treated as an opaque 2-level category) |
| `BMI` | body mass index | float, 1 missing |
| `RAC` | race | 0=other, 1=Caucasian, 2=African American, 3=Asian; 2 missing |
| `SMK` | smoking status | 0=non-smoker, 1=smoker; 28 missing |
| `OSP` | osteoporosis at baseline | 0=negative, 1=positive |
| `KOA` | knee OA diagnosed at follow-up | 0=at risk / not diagnosed, 1=diagnosed; **193 missing (target — dropped, not imputed)** |

**This is a demographic-only proxy dataset.** It has no WOMAC score, no KL
grade, and no functional/sensor measurements — see
`../../README.md` for what this model does and does not cover, and why.
