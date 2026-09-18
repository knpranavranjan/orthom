"""Plotting: confusion matrices, training curves, Pareto trade-off charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]


def confusion_matrix_plots(cm, model_name: str, out_dir: Path) -> dict:
    cm = np.asarray(cm, dtype=float)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    for kind, mat, fmt in (("raw", cm, "{:.0f}"), ("normalized", cmn, "{:.2f}")):
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=(mat.max() or 1))
        ax.set_xticks(range(5), CLASS_NAMES)
        ax.set_yticks(range(5), CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{model_name} - {kind} confusion (val)")
        thr = (mat.max() or 1) / 2
        for i in range(5):
            for j in range(5):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > thr else "black", fontsize=8)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = out_dir / f"{model_name}_confusion_matrix{'' if kind == 'raw' else '_normalized'}.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        paths[kind] = str(p)
    return paths


def training_curve_plot(history: list[dict], model_name: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(ep, [h["train_acc"] for h in history], "o-", label="train acc")
    axes[1].plot(ep, [h["val_acc"] for h in history], "s-", label="val acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(ep, [h["val_macro_f1"] for h in history], "s-", color="tab:green", label="val macro-F1")
    axes[2].plot(ep, [h["val_qwk"] for h in history], "d-", color="tab:purple", label="val QWK")
    best = max(history, key=lambda h: h["val_macro_f1"])
    axes[2].axvline(best["epoch"], color="grey", ls="--", lw=1)
    axes[2].set_title(f"Selection metrics (best F1 @ ep{best['epoch']})")
    axes[2].set_xlabel("epoch"); axes[2].legend()

    fig.suptitle(f"{model_name} - training curves", fontsize=12)
    fig.tight_layout()
    p = out_dir / f"{model_name}_training_curve.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return str(p)


def pareto_plots(rows: list[dict], out_dir: Path) -> dict:
    """rows need: model, macro_f1, mean_latency_ms, parameters, model_size_mb."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if not rows:
        return paths
    names = [r["model"] for r in rows]
    f1 = np.array([r.get("macro_f1") or 0 for r in rows], float)
    lat = np.array([r.get("mean_latency_ms") or np.nan for r in rows], float)
    par = np.array([(r.get("parameters") or 0) / 1e6 for r in rows], float)
    size = np.array([r.get("model_size_mb") or 0 for r in rows], float)

    def _scatter(x, y, s, xlabel, fname, title):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        sizes = 60 + 340 * (s - s.min()) / (np.ptp(s) or 1)
        ax.scatter(x, y, s=sizes, alpha=0.7, edgecolor="black", linewidth=0.5)
        for xi, yi, n in zip(x, y, names):
            ax.annotate(n, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xlabel); ax.set_ylabel("Validation Macro-F1")
        ax.set_title(title); ax.grid(alpha=0.3)
        fig.tight_layout(); p = out_dir / fname
        fig.savefig(p, dpi=130); plt.close(fig)
        return str(p)

    paths["f1_vs_latency"] = _scatter(
        lat, f1, par, "Mean batch-1 inference latency (ms)  [GPU if available]",
        "pareto_macroF1_vs_latency.png",
        "Performance vs. latency (bubble = #params in M)")
    paths["f1_vs_size"] = _scatter(
        size, f1, par, "Checkpoint size (MB)", "pareto_macroF1_vs_size.png",
        "Performance vs. model size (bubble = #params in M)")
    return paths


# --------------------------------------------------------------------------- #
# PHASE 3 - controlled-experiment comparison charts
# --------------------------------------------------------------------------- #
def comparison_bar(df, group_col: str, out_path: Path, title: str,
                   metrics=("val_macro_f1", "val_qwk", "val_balanced_accuracy")):
    """Grouped bars: x = model, one bar per `group_col` value, one panel per metric.

    df needs columns: 'model', group_col, and each metric.
    """
    import pandas as pd  # noqa: PLC0415
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    models = list(dict.fromkeys(df["model"]))
    groups = list(dict.fromkeys(df[group_col].astype(str)))
    metrics = [m for m in metrics if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4), squeeze=False)
    x = np.arange(len(models))
    w = 0.8 / max(len(groups), 1)
    for ax, metric in zip(axes[0], metrics):
        for gi, g in enumerate(groups):
            vals = [df[(df["model"] == m) & (df[group_col].astype(str) == g)][metric].mean()
                    for m in models]
            ax.bar(x + gi * w - 0.4 + w / 2, vals, w, label=str(g))
            for xi, v in zip(x + gi * w - 0.4 + w / 2, vals):
                if pd.notna(v):
                    ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric); ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(title=group_col, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def multiseed_bar(df, out_path: Path, title: str = "Multi-seed robustness (mean +/- std)"):
    """df: rows per (model, seed) with val_macro_f1, val_qwk, val_balanced_accuracy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ("val_macro_f1", "val_qwk", "val_balanced_accuracy") if m in df.columns]
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.2), squeeze=False)
    x = np.arange(len(models))
    for ax, metric in zip(axes[0], metrics):
        means = [df[df["model"] == m][metric].mean() for m in models]
        stds = [df[df["model"] == m][metric].std(ddof=0) for m in models]
        ax.bar(x, means, 0.5, yerr=stds, capsize=5)
        for xi, mn, sd in zip(x, means, stds):
            ax.text(xi, mn, f"{mn:.3f}\n+/-{sd:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_title(metric); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)
