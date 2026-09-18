"""STEP 15 - Class weights from the TRAINING split only.

Writes ``metadata/class_weights.json`` with the raw counts and three weighting
schemes.  The pipeline baseline (per spec section 20) is Weighted Cross-Entropy
using the ``balanced`` (inverse-frequency) weights.  Validation / test data are
never touched here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import KL_CLASS_NAMES, get_logger  # noqa: E402

LOG = get_logger("weights")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, default=Path("metadata/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("metadata/class_weights.json"))
    ap.add_argument("--beta", type=float, default=0.9999,
                    help="beta for class-balanced (effective number) weights")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv, dtype={"kl_grade": int})
    classes = list(range(5))
    counts = np.array([int((df["kl_grade"] == k).sum()) for k in classes], dtype=float)
    n = counts.sum()
    k = len(classes)
    if (counts == 0).any():
        LOG.error("class(es) with zero training samples: %s", [c for c, v in zip(classes, counts) if v == 0])
        sys.exit(1)

    # 1) sklearn "balanced": n / (k * count), mean weight ~= 1
    balanced = n / (k * counts)

    # 2) normalised inverse frequency: (1/count) / sum(1/count) * k  -> mean 1
    inv = 1.0 / counts
    inv_norm = inv / inv.sum() * k

    # 3) class-balanced / effective number of samples (Cui et al. 2019)
    eff_num = 1.0 - np.power(args.beta, counts)
    cb = (1.0 - args.beta) / eff_num
    cb = cb / cb.sum() * k

    payload = {
        "source": str(args.train_csv),
        "note": "Computed from TRAINING split only. Baseline loss = Weighted Cross-Entropy with 'balanced'.",
        "classes": classes,
        "class_names": [KL_CLASS_NAMES[c] for c in classes],
        "train_counts": {str(c): int(v) for c, v in zip(classes, counts)},
        "train_total": int(n),
        "weights": {
            "balanced": {str(c): round(float(w), 6) for c, w in zip(classes, balanced)},
            "inverse_frequency_normalised": {str(c): round(float(w), 6) for c, w in zip(classes, inv_norm)},
            "class_balanced_effective_num": {str(c): round(float(w), 6) for c, w in zip(classes, cb)},
        },
        "class_balanced_beta": args.beta,
        "recommended": "balanced",
        "weights_list_balanced": [round(float(w), 6) for w in balanced],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    LOG.info("wrote %s", args.out)
    LOG.info("train counts: %s", payload["train_counts"])
    LOG.info("balanced weights: %s", payload["weights"]["balanced"])


if __name__ == "__main__":
    main()
