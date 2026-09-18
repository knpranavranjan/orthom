"""STEP 13 helper - dataset-specific grayscale normalization stats.

Computed from the TRAIN split ONLY (spec section 18). Runs each training image
through the deterministic preprocessing pipeline (per contrast variant) and
accumulates pixel mean / std on the [0,1] float image.

Writes ``metadata/normalization_stats.json`` and patches the ``mean``/``std``
under ``intensity_normalization.dataset_grayscale`` in the preprocessing YAML.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402
from src.preprocessing.pipeline import DeterministicPreprocessor, load_config, variants_from_config  # noqa: E402

LOG = get_logger("normstats")


def accumulate(train_csv: Path, pre: DeterministicPreprocessor) -> dict:
    df = pd.read_csv(train_csv)
    n = 0
    s = 0.0
    ss = 0.0
    mn, mx = 1.0, 0.0
    for rel in tqdm(df["original_path"], desc="train imgs"):
        arr = pre.process_path(PROJECT_ROOT / rel).astype(np.float32) / 255.0
        s += float(arr.sum())
        ss += float((arr ** 2).sum())
        n += arr.size
        mn = min(mn, float(arr.min()))
        mx = max(mx, float(arr.max()))
    mean = s / n
    var = max(ss / n - mean ** 2, 0.0)
    return {"mean": round(mean, 6), "std": round(var ** 0.5, 6),
            "min": round(mn, 6), "max": round(mx, 6), "n_pixels": n, "n_images": len(df)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    ap.add_argument("--train-csv", type=Path, default=Path("metadata/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("metadata/normalization_stats.json"))
    ap.add_argument("--patch-config", action="store_true", default=False,
                    help="rewrite mean/std in the YAML (strips comments; off by default)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = {"source": str(args.train_csv), "note": "TRAIN split only.", "variants": {}}
    for v in variants_from_config(cfg):
        pre = DeterministicPreprocessor(cfg, contrast_override=v["contrast"])
        LOG.info("computing stats for variant '%s' (contrast=%s)", v["name"], v["contrast"])
        stats = accumulate(args.train_csv, pre)
        result["variants"][v["name"]] = {**stats, "contrast": v["contrast"]}
        LOG.info("  %s -> mean=%.5f std=%.5f range=[%.3f,%.3f]",
                 v["name"], stats["mean"], stats["std"], stats["min"], stats["max"])

    args.out.write_text(json.dumps(result, indent=2))
    LOG.info("wrote %s", args.out)

    if args.patch_config:
        base = result["variants"].get("basic") or next(iter(result["variants"].values()))
        cfg["intensity_normalization"]["dataset_grayscale"]["mean"] = [base["mean"]]
        cfg["intensity_normalization"]["dataset_grayscale"]["std"] = [base["std"]]
        with open(args.config, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        LOG.info("patched %s dataset_grayscale mean/std = %.5f / %.5f (variant 'basic')",
                 args.config, base["mean"], base["std"])


if __name__ == "__main__":
    main()
