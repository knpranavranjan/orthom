"""STEP 5 + STEP 6 - Class distribution + image-quality analysis.

Reads the per-image audit CSV and produces:

    reports/dataset_distribution.csv   KL0..KL4 counts / % / imbalance ratios
    reports/class_distribution.png     bar chart of the raw distribution
    reports/suspicious_images.csv      per-image quality flags (nothing deleted)

Quality heuristics (flag, do NOT delete):
    CORRUPT                unreadable / truncated
    VERY_DARK             mean pixel < 15 (0-255)
    VERY_BRIGHT           mean pixel > 240
    LOW_DYNAMIC_RANGE     (max - min) < 30  OR  std < 8
    UNUSUAL_SIZE          min(width,height) < 96  OR  != modal size
    UNUSUAL_ASPECT_RATIO  aspect ratio outside [0.8, 1.25]
    INVALID_FORMAT        not one of PNG/JPEG/BMP/TIFF
    UNEXPECTED_CHANNELS   channels not in {1, 3}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import KL_CLASS_NAMES, get_logger  # noqa: E402

LOG = get_logger("quality")

DARK_MEAN = 15.0
BRIGHT_MEAN = 240.0
LOW_RANGE = 30.0
LOW_STD = 8.0
MIN_DIM = 96
AR_LO, AR_HI = 0.80, 1.25
VALID_FORMATS = {"PNG", "JPEG", "JPG", "BMP", "TIFF", "TIF"}


def flag_row(r: pd.Series, modal_size: tuple[int, int]) -> list[str]:
    codes: list[str] = []
    if r["is_corrupt"] is True or r["is_readable"] is not True:
        return ["CORRUPT"]
    try:
        mean_p, mn, mx, sd = float(r["mean_pixel"]), float(r["min_pixel"]), float(r["max_pixel"]), float(r["std_pixel"])
        w, h, ar = int(r["width"]), int(r["height"]), float(r["aspect_ratio"])
        ch = int(r["channels"])
    except (ValueError, TypeError):
        return ["CORRUPT"]

    if mean_p < DARK_MEAN:
        codes.append("VERY_DARK")
    if mean_p > BRIGHT_MEAN:
        codes.append("VERY_BRIGHT")
    if (mx - mn) < LOW_RANGE or sd < LOW_STD:
        codes.append("LOW_DYNAMIC_RANGE")
    if min(w, h) < MIN_DIM or (w, h) != modal_size:
        codes.append("UNUSUAL_SIZE")
    if not (AR_LO <= ar <= AR_HI):
        codes.append("UNUSUAL_ASPECT_RATIO")
    if str(r["format"]).upper() not in VALID_FORMATS:
        codes.append("INVALID_FORMAT")
    if ch not in (1, 3):
        codes.append("UNEXPECTED_CHANNELS")
    return codes


def distribution_table(df: pd.DataFrame, group_col: str = "dataset_source",
                       extra_groups: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    rows = []
    groups = list(df.groupby(group_col))
    for gname, gdf in (extra_groups or {}).items():
        groups.append((gname, gdf))
    for name, grp in groups + [("ALL", df)]:
        counts = grp["class"].value_counts()
        rec = {"Dataset": name}
        for k in range(5):
            rec[f"KL{k}"] = int(counts.get(str(k), counts.get(k, 0)))
        total = sum(rec[f"KL{k}"] for k in range(5))
        rec["Total"] = total
        nz = [rec[f"KL{k}"] for k in range(5) if rec[f"KL{k}"] > 0]
        rec["class_pct"] = "; ".join(
            f"KL{k}:{100*rec[f'KL{k}']/total:.1f}%" for k in range(5)) if total else ""
        rec["imbalance_ratio_max_min"] = round(max(nz) / min(nz), 2) if nz else ""
        rec["minority_majority_ratio"] = round(min(nz) / max(nz), 4) if nz else ""
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_distribution(df: pd.DataFrame, out_png: Path) -> None:
    splits = [s for s in ["train", "val", "test", "auto_test"] if s in set(df["original_split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 4), sharey=False)
    if len(splits) == 1:
        axes = [axes]
    for ax, s in zip(axes, splits):
        sub = df[df["original_split"] == s]
        counts = [int((sub["class"] == str(k)).sum()) for k in range(5)]
        ax.bar([f"KL{k}" for k in range(5)], counts, color="#3b6fb0")
        ax.set_title(f"{s}  (n={sum(counts)})")
        ax.set_xlabel("KL grade")
        for i, c in enumerate(counts):
            ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("image count")
    fig.suptitle("Kaggle knee-OA X-ray: raw KL-grade distribution by folder", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    LOG.info("wrote %s", out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=Path("reports/dataset_audit_kaggle.csv"))
    ap.add_argument("--out-dist", type=Path, default=Path("reports/dataset_distribution.csv"))
    ap.add_argument("--out-plot", type=Path, default=Path("reports/class_distribution.png"))
    ap.add_argument("--out-suspicious", type=Path, default=Path("reports/suspicious_images.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.audit, dtype={"class": str}).convert_dtypes()
    # normalise bool-ish columns
    for c in ("is_readable", "is_corrupt"):
        df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    modal_size = tuple(df.loc[df["is_readable"], ["width", "height"]].mode().iloc[0].astype(int))
    LOG.info("modal image size: %s", modal_size)

    # ---- distribution ----
    # Only one image source (kaggle) is present, so the primary distribution
    # table is broken out by the dataset's own folders, plus a MANUAL rollup
    # (train+val+test = the patient-disjoint, manually-graded set we will use).
    manual = df[df["original_split"].isin(["train", "val", "test"])]
    dist = distribution_table(df, group_col="original_split",
                              extra_groups={"MANUAL(train+val+test)": manual})
    args.out_dist.parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(args.out_dist, index=False)
    LOG.info("wrote %s\n%s", args.out_dist, dist.to_string(index=False))

    plot_distribution(df, args.out_plot)

    # ---- suspicious images ----
    sus_rows = []
    for _, r in df.iterrows():
        codes = flag_row(r, modal_size)
        if codes:
            sus_rows.append({
                "file_path": r["file_path"], "dataset_source": r["dataset_source"],
                "original_split": r["original_split"], "class": r["class"],
                "width": r["width"], "height": r["height"], "channels": r["channels"],
                "mean_pixel": r["mean_pixel"], "std_pixel": r["std_pixel"],
                "min_pixel": r["min_pixel"], "max_pixel": r["max_pixel"],
                "aspect_ratio": r["aspect_ratio"], "format": r["format"],
                "reason_codes": "|".join(codes),
            })
    sus = pd.DataFrame(sus_rows)
    sus.to_csv(args.out_suspicious, index=False)
    LOG.info("wrote %s (%d flagged of %d images, %.2f%%)",
             args.out_suspicious, len(sus), len(df), 100 * len(sus) / len(df))
    if len(sus):
        tally = sus["reason_codes"].str.split("|").explode().value_counts()
        for k, v in tally.items():
            LOG.info("   %-22s %d", k, v)


if __name__ == "__main__":
    main()
