"""STEP 7 + STEP 8 - Duplicate detection and cross-dataset overlap analysis.

Consumes the per-image audit CSV(s) produced by ``audit_dataset.py`` and writes:

    reports/duplicates.csv            exact + perceptual duplicates, with reason codes
    reports/cross_dataset_overlap.csv Kaggle <-> OAI provenance / overlap findings

Nothing is deleted.  Duplicates that carry different KL grades are flagged
``LABEL_CONFLICT``; duplicates that straddle the original train/val/test folders
are flagged ``SPLIT_LEAKAGE_CANDIDATE``.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import get_logger  # noqa: E402

LOG = get_logger("dups")

PHASH_MAX_HAMMING = 4  # <= this Hamming distance between 64-bit pHashes => "near duplicate"


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def knee_identity_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Same physical knee (patient_id + knee_side) appearing in more than one folder.

    This is the *reliable* leakage signal for this dataset: it does not depend on
    a pixel-similarity threshold.  It cleanly surfaces the auto_test <-> test
    relationship (auto_test is a re-detected crop of the same test knees).
    """
    rows = []
    df = df[df["knee_side"].isin(["L", "R"])].copy()
    df["knee"] = df["patient_id"].astype(str) + "_" + df["knee_side"].astype(str)
    for knee, grp in df.groupby("knee"):
        splits = sorted(set(grp["original_split"].astype(str)) - {""})
        if len(grp) < 2 or len(splits) < 2:
            continue
        classes = sorted(set(grp["class"].astype(str)))
        codes = ["SAME_KNEE_MULTIPLE_FOLDERS", "SPLIT_LEAKAGE_CANDIDATE"]
        if len(classes) > 1:
            codes.append("LABEL_CONFLICT")
        members = list(grp["file_path"])
        for fp, cl, sp in zip(grp["file_path"], grp["class"].astype(str), grp["original_split"].astype(str)):
            rows.append({
                "method": "knee_identity",
                "group_id": f"knee:{knee}",
                "file_path": fp,
                "group_size": len(grp),
                "distinct_classes": ",".join(classes),
                "distinct_splits": ",".join(splits),
                "reason_codes": "|".join(codes),
                "group_members": " ; ".join(f"{m}[{c}/{s}]" for m, c, s in
                                            zip(members, grp["class"].astype(str),
                                                grp["original_split"].astype(str))),
            })
    return pd.DataFrame(rows)


def exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, grp in df.groupby("unique_hash"):
        if not h or len(grp) < 2:
            continue
        classes = sorted(set(grp["class"].astype(str)))
        splits = sorted(set(grp["original_split"].astype(str)) - {""})
        codes = ["EXACT_DUPLICATE"]
        if len(classes) > 1:
            codes.append("LABEL_CONFLICT")
        if len(splits) > 1:
            codes.append("SPLIT_LEAKAGE_CANDIDATE")
        members = list(grp["file_path"])
        for fp in members:
            rows.append({
                "method": "sha256",
                "group_id": f"sha:{h[:12]}",
                "file_path": fp,
                "group_size": len(grp),
                "distinct_classes": ",".join(classes),
                "distinct_splits": ",".join(splits),
                "reason_codes": "|".join(codes),
                "group_members": " ; ".join(members),
            })
    return pd.DataFrame(rows)


def perceptual_duplicates(df: pd.DataFrame, max_hamming: int = PHASH_MAX_HAMMING) -> pd.DataFrame:
    """Union-find over pHashes within <= max_hamming Hamming distance.

    O(n^2) in the number of *distinct* hashes.  For ~10k images (a few thousand
    distinct hashes) this is a couple of seconds - fine for a one-off audit.
    Exact-duplicate pairs are excluded (they are already reported by sha256).
    """
    sub = df[df["perceptual_hash"].astype(str).str.len() > 0].copy()
    by_hash: dict[str, list[str]] = defaultdict(list)
    for _, r in sub.iterrows():
        by_hash[r["perceptual_hash"]].append(r["file_path"])
    hashes = list(by_hash)
    LOG.info("perceptual: %d images, %d distinct pHashes", len(sub), len(hashes))

    parent = {h: h for h in hashes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for a, b in itertools.combinations(hashes, 2):
        if _hamming_hex(a, b) <= max_hamming:
            union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for h in hashes:
        clusters[find(h)].extend(by_hash[h])

    fp_meta = df.set_index("file_path")[["class", "original_split", "unique_hash", "patient_id", "knee_side"]]
    rows = []
    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        meta = fp_meta.loc[members]
        if meta["unique_hash"].nunique() <= 1:
            continue  # identical bytes -> already covered by exact pass
        classes = sorted(set(meta["class"].astype(str)))
        splits = sorted(set(meta["original_split"].astype(str)) - {""})
        knees = sorted(set(meta["patient_id"].astype(str) + meta["knee_side"].astype(str)))
        codes = ["PERCEPTUAL_NEAR_DUPLICATE"]
        if len(classes) > 1:
            codes.append("LABEL_CONFLICT")
        if len(splits) > 1:
            codes.append("SPLIT_LEAKAGE_CANDIDATE")
        if len(knees) == 1:
            codes.append("SAME_KNEE_DIFFERENT_CROP")
        for fp in members:
            rows.append({
                "method": "phash",
                "group_id": f"ph:{cid[:12]}",
                "file_path": fp,
                "group_size": len(members),
                "distinct_classes": ",".join(classes),
                "distinct_splits": ",".join(splits),
                "reason_codes": "|".join(codes),
                "group_members": " ; ".join(members),
            })
    return pd.DataFrame(rows)


def cross_dataset_overlap(kaggle_audit: pd.DataFrame, oai_summary_csv: Path) -> pd.DataFrame:
    """Compare Kaggle knees against the OAI master summary.

    The Kaggle 'Knee Osteoarthritis Dataset with Severity Grading' is organised
    from OAI.  We test that claim rigorously: patient-ID membership and, for the
    OAI baseline visit (00m), exact KL-grade agreement per knee.
    """
    oai = pd.read_csv(oai_summary_csv)
    oai["patient_id"] = oai["ID"].astype(str)
    oai["knee_side"] = oai["SIDE"].map({1: "R", 2: "L"}).astype(str)
    oai["klg"] = pd.to_numeric(oai["KLG"], errors="coerce")

    kag = kaggle_audit.copy()
    kag["patient_id"] = kag["patient_id"].astype(str)
    kag["class_int"] = pd.to_numeric(kag["class"], errors="coerce")

    kag_pids = set(kag["patient_id"])
    oai_pids = set(oai["patient_id"])
    oai_pids_00m = set(oai.loc[oai["Visit"] == "00m", "patient_id"])

    rows = []
    rows.append({"check": "kaggle_unique_patients", "value": len(kag_pids)})
    rows.append({"check": "oai_summary_unique_patients", "value": len(oai_pids)})
    rows.append({"check": "kaggle_patients_present_in_oai", "value": len(kag_pids & oai_pids)})
    rows.append({"check": "kaggle_patients_absent_from_oai", "value": len(kag_pids - oai_pids)})
    rows.append({"check": "kaggle_patients_in_oai_baseline_00m", "value": len(kag_pids & oai_pids_00m)})

    # per-knee KL agreement against OAI 00m (manual folders only: train/val/test)
    oai_00m = (oai[oai["Visit"] == "00m"]
               .dropna(subset=["klg"])
               .assign(klg=lambda d: d["klg"].astype(int))
               .set_index(["patient_id", "knee_side"])["klg"]
               .sort_index())
    manual = kag[kag["original_split"].isin(["train", "val", "test"])]
    agree = disagree = missing = 0
    conflict_examples = []
    for _, r in manual.iterrows():
        key = (r["patient_id"], str(r["knee_side"]))
        if key in oai_00m.index and pd.notna(r["class_int"]):
            oai_klg = int(oai_00m.loc[key]) if not hasattr(oai_00m.loc[key], "__len__") else int(oai_00m.loc[key].iloc[0])
            if int(r["class_int"]) == oai_klg:
                agree += 1
            else:
                disagree += 1
                if len(conflict_examples) < 20:
                    conflict_examples.append(f"{key}: kaggle={int(r['class_int'])} oai00m={oai_klg}")
        else:
            missing += 1
    rows.append({"check": "manual_knees_KL_agree_with_oai_00m", "value": agree})
    rows.append({"check": "manual_knees_KL_disagree_with_oai_00m", "value": disagree})
    rows.append({"check": "manual_knees_not_found_in_oai_00m", "value": missing})
    rows.append({"check": "kl_disagreement_examples", "value": " | ".join(conflict_examples) or "(none)"})

    verdict = ("SAME_SOURCE: Kaggle images are OAI baseline (00m) knee ROIs; "
               "datasets are NOT independent - do NOT merge (identical underlying scans / patients)."
               if disagree == 0 and len(kag_pids - oai_pids) == 0
               else "PARTIAL/UNCLEAR overlap - inspect kl_disagreement_examples")
    rows.append({"check": "VERDICT", "value": verdict})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=Path("reports/dataset_audit_kaggle.csv"))
    ap.add_argument("--oai-summary", type=Path,
                    default=Path("OAI-KL-Grade-Classification/data/OAI_summary.csv"))
    ap.add_argument("--out-duplicates", type=Path, default=Path("reports/duplicates.csv"))
    ap.add_argument("--out-overlap", type=Path, default=Path("reports/cross_dataset_overlap.csv"))
    ap.add_argument("--phash-hamming", type=int, default=PHASH_MAX_HAMMING)
    args = ap.parse_args()

    df = pd.read_csv(args.audit, dtype={"class": str, "patient_id": str, "unique_hash": str,
                                        "perceptual_hash": str, "original_split": str}).fillna(
        {"original_split": "", "perceptual_hash": "", "unique_hash": ""})

    exact = exact_duplicates(df)
    LOG.info("exact-duplicate rows: %d (groups: %d)",
             len(exact), exact["group_id"].nunique() if len(exact) else 0)
    knee = knee_identity_duplicates(df)
    LOG.info("knee-identity leakage rows: %d (distinct knees: %d)",
             len(knee), knee["group_id"].nunique() if len(knee) else 0)
    perc = perceptual_duplicates(df, args.phash_hamming)
    LOG.info("perceptual near-duplicate rows: %d (groups: %d)",
             len(perc), perc["group_id"].nunique() if len(perc) else 0)

    dup = pd.concat([exact, knee, perc], ignore_index=True)
    args.out_duplicates.parent.mkdir(parents=True, exist_ok=True)
    dup.to_csv(args.out_duplicates, index=False)
    LOG.info("wrote %s (%d rows)", args.out_duplicates, len(dup))

    if len(dup):
        LOG.info("reason-code tallies:")
        codes = dup["reason_codes"].str.split("|").explode().value_counts()
        for k, v in codes.items():
            LOG.info("   %-28s %d", k, v)

    overlap = cross_dataset_overlap(df, args.oai_summary)
    overlap.to_csv(args.out_overlap, index=False)
    LOG.info("wrote %s", args.out_overlap)
    for _, r in overlap.iterrows():
        LOG.info("   %-42s %s", r["check"], r["value"])


if __name__ == "__main__":
    main()
