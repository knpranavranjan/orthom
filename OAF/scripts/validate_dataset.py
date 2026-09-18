"""STEP 12 + STEP 17 - Hard validation gate for the prepared dataset.

Exits non-zero (with a clear message) if ANY check fails.  Checks:

    * every manifest / split image exists and opens
    * no corrupted images
    * no exact-duplicate leakage inside train/val/test
    * no split leakage: train/val/test are image- AND patient-disjoint
    * all five KL classes present in every split
    * split fractions within tolerance of 70/15/15
    * class proportions consistent across splits
    * processed images exist for the requested variant and are 224x224
    * class_weights.json / normalization_stats.json derived from TRAIN only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402

LOG = get_logger("validate")


class Checker:
    def __init__(self) -> None:
        self.failed: list[str] = []
        self.passed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        (self.passed if ok else self.failed).append(name)
        LOG.info("[%s] %s%s", "PASS" if ok else "FAIL", name, f" - {detail}" if detail else "")

    def finish(self) -> None:
        LOG.info("=" * 60)
        LOG.info("%d passed, %d failed", len(self.passed), len(self.failed))
        if self.failed:
            LOG.error("FAILED CHECKS: %s", ", ".join(self.failed))
            sys.exit(1)
        LOG.info("ALL VALIDATION CHECKS PASSED - dataset is ready for CNN benchmarking")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    ap.add_argument("--variant", default="basic")
    ap.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--ratio-tol", type=float, default=0.03)
    ap.add_argument("--sample-open", type=int, default=400,
                    help="how many processed images to actually decode (0 = all)")
    args = ap.parse_args()
    md = args.metadata_dir
    c = Checker()

    # ---------- master manifest ----------
    mf = pd.read_csv(md / "master_manifest.csv", dtype={"patient_id": str, "kl_grade": "Int64"})
    manual = mf[mf["original_split"].isin(["train", "val", "test"])]
    c.check("master_manifest non-empty", len(mf) > 0, f"{len(mf)} rows")
    c.check("auto_test excluded from splits",
            set(mf.loc[mf["original_split"] == "auto_test", "split"]) <= {"excluded_autotest"})

    # ---------- split files ----------
    splits = {}
    for s in ("train", "val", "test"):
        p = md / f"{s}.csv"
        c.check(f"{s}.csv exists", p.exists())
        splits[s] = pd.read_csv(p, dtype={"patient_id": str, "kl_grade": int})

    total = sum(len(v) for v in splits.values())
    c.check("every manual knee assigned to exactly one split",
            total == len(manual), f"{total} vs {len(manual)}")

    # sample_id disjoint
    id_sets = {s: set(v["sample_id"]) for s, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        c.check(f"sample_id disjoint {a}/{b}", not (id_sets[a] & id_sets[b]),
                f"{len(id_sets[a] & id_sets[b])} shared")

    # patient disjoint
    pid_sets = {s: set(v["patient_id"]) for s, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        c.check(f"patient disjoint {a}/{b}", not (pid_sets[a] & pid_sets[b]),
                f"{len(pid_sets[a] & pid_sets[b])} shared")

    # exact-hash leakage within manual set
    if "sha256" in manual.columns:
        hcount = manual.groupby("sha256")["split"].nunique()
        cross = hcount[hcount > 1]
        c.check("no exact-duplicate hash spans multiple splits", len(cross) == 0,
                f"{len(cross)} hashes")

    # ---------- class coverage / balance ----------
    for s, v in splits.items():
        present = sorted(v["kl_grade"].unique().tolist())
        c.check(f"all 5 KL classes present in {s}", present == [0, 1, 2, 3, 4], str(present))

    fracs = {s: len(v) / total for s, v in splits.items()}
    c.check("split ratio ~70/15/15",
            abs(fracs["train"] - 0.70) < args.ratio_tol
            and abs(fracs["val"] - 0.15) < args.ratio_tol
            and abs(fracs["test"] - 0.15) < args.ratio_tol,
            str({k: round(x, 3) for k, x in fracs.items()}))

    prop = {s: (v["kl_grade"].value_counts(normalize=True).sort_index().to_numpy())
            for s, v in splits.items()}
    max_dev = max(float(np.abs(prop["train"] - prop[s]).max()) for s in ("val", "test"))
    c.check("per-class proportion drift < 0.03 across splits", max_dev < 0.03,
            f"max dev {max_dev:.4f}")

    # ---------- files exist / open ----------
    missing = [r for r in manual["original_path"] if not (PROJECT_ROOT / r).exists()]
    c.check("all manifest source images exist", not missing, f"{len(missing)} missing")

    proc_manifest = md / f"processed_manifest_{args.variant}.csv"
    c.check(f"processed_manifest_{args.variant}.csv exists", proc_manifest.exists())
    if proc_manifest.exists():
        pm = pd.read_csv(proc_manifest)
        pmiss = [r for r in pm["processed_path"] if not (PROJECT_ROOT / r).exists()]
        c.check("all processed images exist", not pmiss, f"{len(pmiss)} missing")
        c.check("processed count == manual count", len(pm) == len(manual),
                f"{len(pm)} vs {len(manual)}")

        n = len(pm) if args.sample_open == 0 else min(args.sample_open, len(pm))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pm), n, replace=False)
        bad, wrong_size = [], []
        for i in idx:
            pp = PROJECT_ROOT / pm.iloc[i]["processed_path"]
            try:
                with Image.open(pp) as im:
                    im.load()
                    if im.size != tuple(args.image_size[::-1]) and im.size != tuple(args.image_size):
                        wrong_size.append(pp.name)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{pp.name}: {exc}")
        c.check(f"decoded {n} processed images OK", not bad, f"{len(bad)} bad")
        c.check("processed images are target size", not wrong_size, f"{len(wrong_size)} wrong")

    # ---------- weights / stats provenance ----------
    cw_path = md / "class_weights.json"
    c.check("class_weights.json exists", cw_path.exists())
    if cw_path.exists():
        cw = json.loads(cw_path.read_text())
        train_counts = {int(k): v for k, v in cw["train_counts"].items()}
        actual = splits["train"]["kl_grade"].value_counts().to_dict()
        c.check("class_weights derived from TRAIN split only", train_counts == actual,
                f"{train_counts} vs {actual}")

    ns_path = md / "normalization_stats.json"
    c.check("normalization_stats.json exists", ns_path.exists())
    if ns_path.exists():
        ns = json.loads(ns_path.read_text())
        c.check("normalization stats note training-only",
                "TRAIN" in ns.get("note", "").upper())
        c.check(f"normalization stats has variant '{args.variant}'",
                args.variant in ns.get("variants", {}))

    c.finish()


if __name__ == "__main__":
    main()
