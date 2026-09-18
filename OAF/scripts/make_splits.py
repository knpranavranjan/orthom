"""STEP 11 - Reproducible PATIENT-LEVEL, KL-STRATIFIED train/val/test split.

* Unit of splitting is the PATIENT (OAI ID parsed from the filename), never the
  image - both knees of a patient always land in the same split.
* Stratification key is the patient's WORST knee KL grade (max of the two
  knees). This keeps the rare high grades (esp. KL4) proportionally represented.
* Ratios 70 / 15 / 15, fixed seed 42. Deterministic given the manifest.

Reads / writes ``metadata/master_manifest.csv`` (fills the ``split`` column) and
writes ``metadata/train.csv`` / ``val.csv`` / ``test.csv``.
The Kaggle ``auto_test`` rows are left as ``split=excluded_autotest``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RANDOM_SEED, get_logger  # noqa: E402

LOG = get_logger("split")


def make_patient_split(patients: pd.DataFrame, ratios=(0.70, 0.15, 0.15),
                       seed: int = RANDOM_SEED) -> pd.Series:
    """Return a Series index=patient_id -> {'train','val','test'} (stratified)."""
    p = patients.sort_values("patient_id").reset_index(drop=True)
    strat = p["stratum"].to_numpy()

    train_p, rest_p, strat_train, strat_rest = train_test_split(
        p["patient_id"], strat, train_size=ratios[0], random_state=seed, stratify=strat
    )
    rel_val = ratios[1] / (ratios[1] + ratios[2])
    val_p, test_p = train_test_split(
        rest_p, train_size=rel_val, random_state=seed, stratify=strat_rest
    )
    out = pd.Series(index=p["patient_id"], dtype=object)
    out.loc[train_p] = "train"
    out.loc[val_p] = "val"
    out.loc[test_p] = "test"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("metadata/master_manifest.csv"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    ap.add_argument("--out-dir", type=Path, default=Path("metadata"))
    args = ap.parse_args()

    mf = pd.read_csv(args.manifest, dtype={"patient_id": str, "kl_grade": "Int64"})
    manual = mf[mf["original_split"].isin(["train", "val", "test"])].copy()
    LOG.info("manual knees: %d  patients: %d", len(manual), manual["patient_id"].nunique())

    # patient-level stratum = worst (max) KL grade across the patient's knees
    pat = (manual.groupby("patient_id")["kl_grade"]
           .agg(["max", "count"]).reset_index()
           .rename(columns={"max": "stratum", "count": "n_knees"}))
    LOG.info("knees per patient: %s", pat["n_knees"].value_counts().to_dict())
    LOG.info("patient stratum (worst-knee KL) dist: %s",
             pat["stratum"].value_counts().sort_index().to_dict())

    assign = make_patient_split(pat, tuple(args.ratios), args.seed)
    manual["split"] = manual["patient_id"].map(assign)

    # write split column back into the full manifest
    mf.loc[manual.index, "split"] = manual["split"].values
    mf.to_csv(args.manifest, index=False)
    LOG.info("updated %s split column", args.manifest)

    # ---- verification ----
    ok = True
    sets = {s: set(manual.loc[manual["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        LOG.info("patient overlap %s/%s: %d", a, b, len(inter))
        ok &= not inter
    if manual["split"].isna().any():
        LOG.error("%d manual knees unassigned", int(manual["split"].isna().sum()))
        ok = False

    counts = manual.groupby(["split", "kl_grade"]).size().unstack(fill_value=0)
    counts["TOTAL"] = counts.sum(axis=1)
    LOG.info("knee counts by split x KL:\n%s", counts.to_string())
    frac = manual.groupby("split").size() / len(manual)
    LOG.info("split fractions (knees): %s", frac.round(4).to_dict())
    perc = (manual.groupby(["split", "kl_grade"]).size()
            / manual.groupby("split").size()).unstack(fill_value=0).round(4)
    LOG.info("per-split KL proportions:\n%s", perc.to_string())

    cols = list(mf.columns)
    for s in ("train", "val", "test"):
        sub = manual[manual["split"] == s][cols].sort_values(["kl_grade", "patient_id", "knee_side"])
        sub.to_csv(args.out_dir / f"{s}.csv", index=False)
        LOG.info("wrote %s (%d rows)", args.out_dir / f"{s}.csv", len(sub))

    if not ok:
        LOG.error("SPLIT VERIFICATION FAILED")
        sys.exit(1)
    LOG.info("split verification PASSED (patient-disjoint, all knees assigned)")


if __name__ == "__main__":
    main()
