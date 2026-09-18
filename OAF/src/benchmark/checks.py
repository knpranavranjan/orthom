"""Startup safety checks (spec sections 43-44). Fail loudly on any leakage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import PROJECT_ROOT


class LeakageError(RuntimeError):
    pass


def leakage_and_distribution_check(variant: str = "basic", *, logger=None) -> dict:
    log = (logger.info if logger else print)
    manifest = PROJECT_ROOT / "metadata" / f"processed_manifest_{variant}.csv"
    if not manifest.exists():
        raise LeakageError(f"missing processed manifest: {manifest}")
    df = pd.read_csv(manifest, dtype={"patient_id": str})

    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    for s, d in splits.items():
        if len(d) == 0:
            raise LeakageError(f"split '{s}' is empty in {manifest.name}")

    pid = {s: set(d["patient_id"]) for s, d in splits.items()}
    sid = {s: set(d["sample_id"]) for s, d in splits.items()}
    problems = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pov = pid[a] & pid[b]
        sov = sid[a] & sid[b]
        log(f"  patient overlap {a} n {b}: {len(pov)}  |  sample overlap: {len(sov)}")
        if pov:
            problems.append(f"{a}/{b} share {len(pov)} patient IDs (e.g. {sorted(pov)[:5]})")
        if sov:
            problems.append(f"{a}/{b} share {len(sov)} sample IDs")

    # no processed test image path leaking into train/val rows
    test_paths = set(splits["test"]["processed_path"])
    for s in ("train", "val"):
        bad = set(splits[s]["processed_path"]) & test_paths
        if bad:
            problems.append(f"{len(bad)} test image paths present in '{s}' rows")

    dist = {}
    for s, d in splits.items():
        vc = d["kl_grade"].value_counts().to_dict()
        dist[s] = {int(k): int(v) for k, v in sorted(vc.items())}
        present = sorted(dist[s])
        log(f"  {s:5s} n={len(d):5d}  KL dist={dist[s]}")
        if present != [0, 1, 2, 3, 4]:
            problems.append(f"split '{s}' missing classes: has {present}")

    if problems:
        raise LeakageError("DATA LEAKAGE / DISTRIBUTION CHECK FAILED:\n  - " + "\n  - ".join(problems))
    log("  leakage + distribution check PASSED (patient- and sample-disjoint; all 5 classes present)")
    return {"class_distribution": dist,
            "patients": {s: len(v) for s, v in pid.items()},
            "counts": {s: len(v) for s, v in splits.items()}}
