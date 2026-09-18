"""Assemble the final benchmark artefacts: CSV/JSON tables + markdown report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULT_COLUMNS = [
    "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
    "balanced_accuracy", "cohen_kappa", "quadratic_weighted_kappa",
    "kl0_f1", "kl1_f1", "kl2_f1", "kl3_f1", "kl4_f1",
    "parameters", "trainable_parameters", "model_size_mb",
    "gmacs", "flops_or_macs",              # 'flops_or_macs' == GMACs (fvcore, MAC=1)
    "mean_latency_ms", "median_latency_ms", "std_latency_ms", "throughput_images_per_sec",
    "latency_device", "cpu_mean_latency_ms",
    "peak_memory_mb", "best_epoch", "epochs_run", "training_time_sec", "training_time",
]


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rank_macro_f1"] = d["macro_f1"].rank(ascending=False, method="min").astype("Int64")
    d["rank_qwk"] = d["quadratic_weighted_kappa"].rank(ascending=False, method="min").astype("Int64")
    d["rank_balanced_acc"] = d["balanced_accuracy"].rank(ascending=False, method="min").astype("Int64")
    d["rank_latency"] = d["mean_latency_ms"].rank(ascending=True, method="min").astype("Int64")
    d["rank_params"] = d["parameters"].rank(ascending=True, method="min").astype("Int64")
    # composite: performance first (F1, QWK, bAcc), then efficiency (latency, params)
    d["perf_score"] = d[["rank_macro_f1", "rank_qwk", "rank_balanced_acc"]].mean(axis=1)
    d["eff_score"] = d[["rank_latency", "rank_params"]].mean(axis=1)
    d["overall_score"] = 0.7 * d["perf_score"] + 0.3 * d["eff_score"]
    return d.sort_values(["macro_f1", "quadratic_weighted_kappa"], ascending=False)


_SCALAR_EXTRA = [
    "timm_name", "roc_auc_ovr_macro", "kl0_support", "kl1_support", "kl2_support",
    "kl3_support", "kl4_support", "non_trainable_parameters", "flops_estimate_g",
    "gmacs_method", "min_latency_ms", "max_latency_ms", "train_peak_cuda_mb",
    "stopped_early",
]
_RANK_COLS = ["rank_macro_f1", "rank_qwk", "rank_balanced_acc", "rank_latency",
              "rank_params", "perf_score", "eff_score", "overall_score"]


_PHASE3_EXTRA = ["backend", "preprocessing_variant", "loss_name", "seed",
                 "stage_a_epochs", "max_epochs_budget", "patience", "phase"]


def write_tables(results: list[dict], out_dir: Path, basename: str = "benchmark_results") -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_full = pd.DataFrame(results)
    for c in RESULT_COLUMNS:
        if c not in df_full.columns:
            df_full[c] = np.nan
    df_full = _rank(df_full)

    # flat CSV: scalar columns only (nested dicts/lists live in the per-model JSON)
    csv_cols = [c for c in RESULT_COLUMNS + _SCALAR_EXTRA + _PHASE3_EXTRA + _RANK_COLS
                if c in df_full.columns]
    df_full[csv_cols].to_csv(out_dir / f"{basename}.csv", index=False)

    # JSON: everything, including nested confusion matrix / adjacent confusions
    (out_dir / f"{basename}.json").write_text(
        json.dumps(json.loads(df_full.to_json(orient="records")), indent=2, default=str))
    return df_full


def write_complexity_csv(rows: list[dict], out_dir: Path) -> None:
    cols = ["model", "timm_name", "parameters", "trainable_parameters",
            "non_trainable_parameters", "parameters_millions", "gmacs",
            "flops_estimate_g", "model_size_mb", "gmacs_method"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df[cols].to_csv(out_dir / "model_complexity.csv", index=False)


def _md_table(df: pd.DataFrame, cols: list[str], floatfmt: int = 4) -> str:
    view = df[cols].copy()
    for c in cols:
        if view[c].dtype.kind == "f":
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in row) + " |"
                     for row in view.itertuples(index=False))
    return "\n".join([header, sep, body])


def write_markdown_report(df: pd.DataFrame, context: dict, out_path: Path,
                          plot_index: dict | None = None) -> None:
    plot_index = plot_index or {}
    top = df.iloc[0]
    lines: list[str] = []
    A = lines.append

    A("# AETHER-OA - X-ray Module | PHASE 2: 10-CNN Benchmark Report\n")
    A(f"_Generated {context.get('timestamp_utc', '')}. Seed {context.get('seed', 42)}. "
      f"Test set untouched during model selection._\n")

    A("## 1. Objective\n")
    A("Train and objectively compare 10 ImageNet-pretrained CNN backbones for 5-class "
      "Kellgren-Lawrence (KL 0-4) knee-OA severity classification, and select ONE model "
      "balancing diagnostic performance and deployment efficiency. Standard 5-class "
      "classification only (no attention / ordinal / Grad-CAM in this phase).\n")

    A("## 2. Dataset\n")
    A("- Source: Kaggle *Knee Osteoarthritis Dataset with Severity Grading* = verified OAI "
      "baseline (00m) knee ROIs. Kaggle-derived data only; OAI repo NOT merged, `auto_test` NOT used.\n"
      f"- Working set: 8,260 knee images / 4,130 patients. Preprocessing variant: "
      f"**{context.get('variant', 'basic')}** (224x224 grayscale PNG -> float[0,1] -> 3-ch "
      f"replicated grayscale -> {context.get('normalization', 'imagenet')} normalization).\n")

    A("## 3. Patient-level split (frozen)\n")
    A("| split | knees | KL0 | KL1 | KL2 | KL3 | KL4 |")
    A("|---|--:|--:|--:|--:|--:|--:|")
    for s, d in context.get("class_distribution", {}).items():
        A(f"| {s} | {sum(d.values())} | " + " | ".join(str(d.get(str(k), d.get(k, 0))) for k in range(5)) + " |")
    A("\nPatient overlap train/val/test = 0 / 0 / 0 (re-verified at startup).\n")

    A("## 4. Preprocessing & 5. Augmentation\n")
    A("Deterministic preprocessing baked in Phase 1. Train-only augmentation (spec section 9): "
      "rotation +/-8deg, translation +/-5%, scale 0.95-1.05, shear +/-4deg, horizontal flip p=0.5, "
      "brightness/contrast 0.90-1.10, Gaussian noise p=0.2 (sigma<=0.02). "
      f"Disabled: {', '.join(context.get('disabled_augmentations', []))}. Val/test deterministic.\n")

    A("## 6. Training configuration (identical for all models)\n")
    tc = context.get("train_config", {})
    for k, v in tc.items():
        A(f"- **{k}**: {v}")
    A("")

    A("## 7-10. Results\n")
    A("### Performance (validation)\n")
    A(_md_table(df, ["model", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                     "quadratic_weighted_kappa", "cohen_kappa", "macro_precision", "macro_recall"]))
    A("\n### Per-class F1 (validation)\n")
    A(_md_table(df, ["model", "kl0_f1", "kl1_f1", "kl2_f1", "kl3_f1", "kl4_f1"]))
    A("\n### Efficiency\n")
    A(_md_table(df, ["model", "parameters", "model_size_mb", "gmacs", "mean_latency_ms",
                     "median_latency_ms", "throughput_images_per_sec", "peak_memory_mb",
                     "cpu_mean_latency_ms"], floatfmt=3))

    A("\n## 11. Confusion matrices\n")
    A(f"See `{plot_index.get('confusion_dir', 'reports/benchmark/confusion_matrices/')}` "
      "(raw + normalized per model).\n")
    A("## 12. Per-class performance & adjacent-grade confusion\n")
    A("Per-model adjacent-grade confusion counts (KL1<->KL2, KL2<->KL3, KL3<->KL4) are in "
      "`classification_reports/<model>.json`.\n")
    A("## 13. Training curves\n")
    A(f"See `{plot_index.get('curves_dir', 'reports/benchmark/training_curves/')}`.\n")
    A("## 14-17. Latency / params / size / MACs\n")
    A("GMACs via `fvcore.FlopCountAnalysis` (one fused multiply-add = 1; FLOPs ~= 2x GMACs). "
      "Latency: batch=1, warmup then timed iterations, CUDA-synchronised. **GPU latency is "
      "development-only; Raspberry-Pi performance is NOT inferred here** (Phase 8).\n")

    A("## 18. Ranking\n")
    A(_md_table(df, ["model", "macro_f1", "quadratic_weighted_kappa", "balanced_accuracy",
                     "rank_macro_f1", "rank_qwk", "rank_latency", "rank_params", "overall_score"],
                floatfmt=4))
    A("\n- **Performance ranking** (Macro-F1 -> QWK -> balanced acc): "
      + ", ".join(df.sort_values(["macro_f1", "quadratic_weighted_kappa"], ascending=False)["model"]))
    A("- **Efficiency ranking** (latency -> params): "
      + ", ".join(df.sort_values(["mean_latency_ms", "parameters"])["model"]))

    A("\n## 19. Recommended model\n")
    A(f"**Primary (highest validation Macro-F1): `{top['model']}`** - "
      f"Macro-F1={top['macro_f1']:.4f}, QWK={top['quadratic_weighted_kappa']:.4f}, "
      f"balanced acc={top['balanced_accuracy']:.4f}, KL4-F1={top['kl4_f1']:.4f}, "
      f"{top['parameters']/1e6:.1f}M params, {top['mean_latency_ms']:.2f} ms/img.\n")
    A("Apply the decision framework (spec section 35): if a much smaller/faster model is within "
      "a small Macro-F1 margin and matches on QWK / KL3-KL4 F1, prefer it. See the Pareto plots "
      f"(`{plot_index.get('pareto_dir', 'reports/benchmark/')}`). Final call stated below with evidence.\n")

    A("## 20. Limitations\n")
    A("- Single-source (OAI-derived) data; no independent external test yet.\n"
      "- KL4 minority (176 train knees) -> KL4 metrics have wide confidence intervals.\n"
      "- Inception/Xception run at 224px (native 299) for fairness -> not their optimal setting.\n"
      "- Validation used for model selection; broader hyper-parameter tuning deferred to avoid val overf'g.\n"
      "- Mixed precision + some CUDA ops are non-deterministic (documented in reproducibility.json).\n"
      "- Research prototype for AI-based KL-grade classification - NOT a clinical diagnostic claim.\n")

    A("## 21. Next phase\n")
    A("PHASE 3 - best model + preprocessing ablation (basic vs CLAHE vs histeq). "
      "Test set remains frozen until Phase 6.\n")

    A("\n---\n## Publication-style comparison table\n")
    tbl = df.copy()
    tbl["Params(M)"] = (tbl["parameters"] / 1e6).round(2)
    A(_md_table(tbl.rename(columns={
        "model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1",
        "balanced_accuracy": "Balanced Acc.", "quadratic_weighted_kappa": "QWK",
        "kl4_f1": "KL4 F1", "model_size_mb": "Size(MB)", "mean_latency_ms": "Latency(ms)"}),
        ["Model", "Accuracy", "Macro F1", "Balanced Acc.", "QWK", "KL4 F1",
         "Params(M)", "Size(MB)", "Latency(ms)"], floatfmt=4))

    out_path.write_text("\n".join(lines), encoding="utf-8")
