"""STEP 16 - Visual QA: per-class sample grids + preprocessing walkthrough.

    reports/samples/kl{0..4}_samples.png     16 random train knees per KL grade
    reports/samples/before_after_preprocessing.png
        original -> grayscale -> ROI -> pad+resize -> CLAHE -> hist-eq -> normalized
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import KL_CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, ensure_dir, get_logger  # noqa: E402
from src.preprocessing import transforms as T  # noqa: E402
from src.preprocessing.pipeline import DeterministicPreprocessor, load_config  # noqa: E402

LOG = get_logger("visual")


def class_grids(train_csv: Path, out_dir: Path, n: int = 16, seed: int = RANDOM_SEED) -> None:
    df = pd.read_csv(train_csv, dtype={"kl_grade": int})
    rng = np.random.default_rng(seed)
    for kl in range(5):
        sub = df[df["kl_grade"] == kl]
        take = sub.sample(min(n, len(sub)), random_state=seed)
        cols = 4
        rows = int(np.ceil(len(take) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
        for ax in np.array(axes).ravel():
            ax.axis("off")
        for ax, (_, r) in zip(np.array(axes).ravel(), take.iterrows()):
            img = T.load_grayscale(PROJECT_ROOT / r["original_path"])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{r['patient_id']}{r['knee_side']}", fontsize=7)
        fig.suptitle(f"KL {kl} - {KL_CLASS_NAMES[kl]}  (train, n shown={len(take)}, total={len(sub)})",
                     fontsize=12)
        fig.tight_layout()
        out = out_dir / f"kl{kl}_samples.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        LOG.info("wrote %s", out)


def before_after(cfg: dict, train_csv: Path, out_path: Path, n_examples: int = 4,
                 seed: int = RANDOM_SEED) -> None:
    df = pd.read_csv(train_csv, dtype={"kl_grade": int})
    picks = pd.concat([df[df["kl_grade"] == kl].sample(1, random_state=seed)
                       for kl in range(min(n_examples, 5))]).reset_index(drop=True)
    pre_basic = DeterministicPreprocessor(cfg, contrast_override="none")
    pre_clahe = DeterministicPreprocessor(cfg, contrast_override="clahe")
    pre_histeq = DeterministicPreprocessor(cfg, contrast_override="hist_eq")

    stages = ["original", "grayscale", "ROI (provided)", "pad+resize 224",
              "CLAHE", "hist-eq", "normalized (z)"]
    ncol = len(stages)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(ncol * 1.9, len(picks) * 2.0))
    ns_path = PROJECT_ROOT / "metadata" / "normalization_stats.json"
    import json
    mean = std = None
    if ns_path.exists():
        v = json.loads(ns_path.read_text())["variants"].get("basic")
        if v:
            mean, std = v["mean"], v["std"]

    for row, (_, r) in enumerate(picks.iterrows()):
        src = PROJECT_ROOT / r["original_path"]
        orig = T.load_grayscale(src)
        gray = orig  # already single channel
        roi = pre_basic._roi(gray)
        padded = T.pad_to_square(roi, 0)
        resized = T.resize(padded, tuple(cfg["image"]["size"]), cfg["image"]["resample"])
        clahe_img = pre_clahe.process_array(gray)
        histeq_img = pre_histeq.process_array(gray)
        z = (resized.astype(np.float32) / 255.0 - (mean or resized.mean() / 255)) / (std or (resized.std() / 255) or 1)

        imgs = [orig, gray, roi, resized, clahe_img, histeq_img, z]
        for col, (ax, im) in enumerate(zip(axes[row], imgs)):
            if col == ncol - 1:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(stages[col], fontsize=8)
        axes[row][0].set_ylabel(f"KL{r['kl_grade']}", fontsize=9)
    fig.suptitle("Deterministic preprocessing walkthrough (train samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    ap.add_argument("--train-csv", type=Path, default=Path("metadata/train.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/samples"))
    args = ap.parse_args()
    ensure_dir(args.out_dir)
    cfg = load_config(args.config)
    class_grids(args.train_csv, args.out_dir)
    before_after(cfg, args.train_csv, args.out_dir / "before_after_preprocessing.png")


if __name__ == "__main__":
    main()
