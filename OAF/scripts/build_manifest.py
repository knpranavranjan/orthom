"""STEP 10 - Build the master metadata + master manifest.

Inputs
------
    reports/dataset_audit_kaggle.csv
    reports/duplicates.csv
    reports/suspicious_images.csv

Outputs
-------
    metadata/master_metadata.csv   minimal per-sample identity table
    metadata/master_manifest.csv   full per-sample manifest (split filled later)

Scope decision (documented in reports/dataset_report.md):
    * The Kaggle 'train' + 'val' + 'test' folders are the manually KL-graded,
      patient-disjoint set -> these become working manifest rows.
    * The Kaggle 'auto_test' folder is an *auto-detected re-crop* of the SAME
      test knees (1526/1526 knee-identity match, 21 KL-label conflicts). It is
      kept in the manifest but marked duplicate_status=AUTOTEST_REDETECTION and
      split=excluded_autotest so it can never leak into train/val/test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import KL_CLASS_NAMES, get_logger  # noqa: E402

LOG = get_logger("manifest")

MANIFEST_COLUMNS = [
    "sample_id", "dataset_source", "patient_id", "knee_side", "original_path",
    "original_split", "kl_grade", "class_name",
    "width", "height", "channels", "file_format",
    "sha256", "perceptual_hash", "quality_status", "duplicate_status", "split",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=Path("reports/dataset_audit_kaggle.csv"))
    ap.add_argument("--duplicates", type=Path, default=Path("reports/duplicates.csv"))
    ap.add_argument("--suspicious", type=Path, default=Path("reports/suspicious_images.csv"))
    ap.add_argument("--out-metadata", type=Path, default=Path("metadata/master_metadata.csv"))
    ap.add_argument("--out-manifest", type=Path, default=Path("metadata/master_manifest.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.audit, dtype={"class": str, "patient_id": str,
                                        "unique_hash": str, "perceptual_hash": str,
                                        "original_split": str})
    df = df[df["is_readable"].astype(str).str.lower() == "true"].copy()
    LOG.info("readable images: %d", len(df))

    # ---- quality status ----
    sus = pd.read_csv(args.suspicious) if args.suspicious.exists() else pd.DataFrame(columns=["file_path", "reason_codes"])
    sus_map = dict(zip(sus["file_path"], sus["reason_codes"])) if len(sus) else {}

    # ---- duplicate status (from knee-identity + exact passes) ----
    dup = pd.read_csv(args.duplicates) if args.duplicates.exists() else pd.DataFrame(columns=["method", "file_path", "reason_codes"])
    exact_fp = set(dup.loc[dup["method"] == "sha256", "file_path"])
    # knee-identity leakage rows that stay ENTIRELY inside train/val/test would be
    # real intra-manual duplicates; rows whose only twin is in auto_test are not.
    knee_dup = dup[dup["method"] == "knee_identity"].copy()
    intra_manual_fp = set(
        knee_dup.loc[~knee_dup["distinct_splits"].fillna("").str.contains("auto_test")
                     | (knee_dup["distinct_splits"].fillna("").str.count(",") >= 2), "file_path"]
    )
    autotest_twin_fp = set(knee_dup["file_path"]) - intra_manual_fp

    rows = []
    for _, r in df.iterrows():
        fp = r["file_path"]
        knee_side = r["knee_side"]
        split_folder = r["original_split"]
        is_autotest = split_folder == "auto_test"
        sample_id = (f"kaggle_auto_{r['patient_id']}_{knee_side}" if is_autotest
                     else f"kaggle_{r['patient_id']}_{knee_side}")

        if fp in exact_fp:
            dup_status = "EXACT_DUPLICATE"
        elif is_autotest:
            dup_status = "AUTOTEST_REDETECTION"      # same knee as a 'test' sample, auto re-crop
        elif fp in intra_manual_fp:
            dup_status = "INTRA_MANUAL_DUPLICATE"    # would be real train/val/test leakage
        elif fp in autotest_twin_fp:
            dup_status = "HAS_AUTOTEST_TWIN"         # informational; the twin lives in excluded auto_test
        else:
            dup_status = "OK"

        quality_status = f"SUSPECT:{sus_map[fp]}" if fp in sus_map else "OK"

        split = "excluded_autotest" if is_autotest else "unassigned"

        kl = int(r["class"]) if str(r["class"]).isdigit() else ""
        rows.append({
            "sample_id": sample_id,
            "dataset_source": "kaggle_knee_oa_severity",
            "patient_id": r["patient_id"],
            "knee_side": knee_side,
            "original_path": fp,
            "original_split": split_folder,
            "kl_grade": kl,
            "class_name": KL_CLASS_NAMES.get(kl, "") if kl != "" else "",
            "width": int(r["width"]), "height": int(r["height"]),
            "channels": int(r["channels"]), "file_format": r["format"],
            "sha256": r["unique_hash"], "perceptual_hash": r["perceptual_hash"],
            "quality_status": quality_status,
            "duplicate_status": dup_status,
            "split": split,
        })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["original_split", "kl_grade", "patient_id", "knee_side"]).reset_index(drop=True)

    # ---- sanity: manual knees must be unique ----
    manual = manifest[manifest["original_split"].isin(["train", "val", "test"])]
    dup_ids = manual["sample_id"][manual["sample_id"].duplicated()].unique()
    if len(dup_ids):
        LOG.error("non-unique sample_id in manual set: %s", dup_ids[:10])
        sys.exit(1)
    LOG.info("manual (train/val/test) unique knees: %d ; auto_test rows: %d",
             len(manual), int((manifest["original_split"] == "auto_test").sum()))

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_manifest, index=False)
    LOG.info("wrote %s (%d rows)", args.out_manifest, len(manifest))

    meta = manifest[["sample_id", "patient_id", "knee_side", "dataset_source",
                     "original_path", "kl_grade"]].copy()
    meta.to_csv(args.out_metadata, index=False)
    LOG.info("wrote %s (%d rows)", args.out_metadata, len(meta))

    LOG.info("duplicate_status tally: %s", manifest["duplicate_status"].value_counts().to_dict())
    LOG.info("quality_status tally:   %s", manifest["quality_status"].value_counts().to_dict())
    LOG.info("manual KL distribution: %s",
             manual["kl_grade"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
