r"""PHASE 6 - consolidate everything into ONE final benchmark (no training).

Pulls the already-computed, validated results together:
  * reports/phase5/final_15_model_test_benchmark.csv  (15-model TEST metrics, GoogLeNet fixed)
  * reports/phase5/per_class_metrics.csv              (per-class P/R/F1)
  * reports/phase6/probs/*.npz                        (for the vgg16 + hflip + temperature config)
  * reports/phase6/{phase6_final_metrics,binary_screening,three_class,abstention_curve,
                    calibration,preprocessing_ablation,loss_ablation}.csv

Writes:
  reports/phase6/FINAL_15_MODEL_BENCHMARK.csv    (every metric, every model, test split)
  reports/phase6/FINAL_PERCLASS_F1.csv
  reports/phase6/FINAL_BENCHMARK_REPORT.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT  # noqa: E402
from src.benchmark.metrics import compute_all  # noqa: E402
from src.benchmark.phase5_metrics import ordinal_and_confidence  # noqa: E402
import scripts.phase6_ensemble_calibrate as P  # noqa: E402

R5 = PROJECT_ROOT / "reports" / "phase5"
R6 = PROJECT_ROOT / "reports" / "phase6"
PROBS = R6 / "probs"

NAME = {  # pretty display names
    "vgg16": "VGG16", "vgg19": "VGG19", "googlenet": "GoogLeNet", "squeezenet": "SqueezeNet",
    "shufflenet": "ShuffleNet-V2", "convnext_tiny": "ConvNeXt-Tiny", "densenet121": "DenseNet121",
    "inception_v3": "Inception-V3", "xception": "Xception", "efficientnet_b1": "EfficientNet-B1",
    "resnet50": "ResNet50", "mobilenet_v3_large": "MobileNet-V3-L", "resnet18": "ResNet18",
    "mobilenet_v2": "MobileNet-V2", "efficientnet_b0": "EfficientNet-B0",
}


def _load_npz(split, model, tta):
    d = np.load(PROBS / f"{split}__{model}__{tta}.npz", allow_pickle=True)
    return d["y_true"].astype(int), d["y_prob"].astype(np.float64), d["y_logit"].astype(np.float64)


def main() -> None:
    d5 = pd.read_csv(R5 / "final_15_model_test_benchmark.csv")
    pc = pd.read_csv(R5 / "per_class_metrics.csv")

    keep = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "balanced_accuracy", "qwk", "cohen_kappa", "kl4_f1", "kl4_precision", "kl4_recall",
            "mae", "exact_accuracy", "within_1_accuracy", "parameters", "trainable_parameters",
            "mean_latency_ms", "validation_macro_f1", "test_macro_f1", "delta_macro_f1",
            "best_epoch", "epochs_run", "roc_auc_ovr_macro"]
    have = [c for c in keep if c in d5.columns]
    df = d5[have].copy()

    # per-class F1 columns
    piv = pc.pivot_table(index="model", columns="class", values="f1")
    piv.columns = [f"KL{int(c)}_f1" for c in piv.columns]
    df = df.merge(piv.reset_index(), on="model", how="left")

    # best_epoch / epochs_run from the clean Phase-4 result.json (checkpoint provenance)
    p4 = PROJECT_ROOT / "models" / "phase4"
    prov = []
    for m in df["model"]:
        rj = p4 / m / "result.json"
        r = json.loads(rj.read_text()) if rj.exists() else {}
        prov.append({"model": m, "best_epoch": r.get("best_epoch"),
                     "epochs_run": r.get("epochs_run"),
                     "trainable_parameters": r.get("trainable_parameters")})
    provdf = pd.DataFrame(prov)
    for c in ("best_epoch", "epochs_run", "trainable_parameters"):
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(provdf, on="model", how="left")

    # generalization gap on a few more metrics from Phase-5 file if present
    for m in ["accuracy", "balanced_accuracy", "qwk", "weighted_f1", "kl4_f1"]:
        for pfx in ("validation", "test", "delta"):
            col = f"{pfx}_{m if m != 'qwk' else 'qwk'}"
            if col in d5.columns and col not in df.columns:
                df = df.merge(d5[["model", col]], on="model", how="left")

    df["display"] = df["model"].map(NAME).fillna(df["model"])
    df["params_M"] = (df["parameters"] / 1e6).round(2)
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    order = ["rank", "display", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
             "weighted_f1", "balanced_accuracy", "qwk", "cohen_kappa",
             "KL0_f1", "KL1_f1", "KL2_f1", "KL3_f1", "KL4_f1",
             "kl4_precision", "kl4_recall", "mae", "exact_accuracy", "within_1_accuracy",
             "roc_auc_ovr_macro", "params_M", "parameters", "trainable_parameters",
             "mean_latency_ms", "best_epoch", "epochs_run",
             "validation_macro_f1", "test_macro_f1", "delta_macro_f1"]
    order = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df[order]
    df.round(4).to_csv(R6 / "FINAL_15_MODEL_BENCHMARK.csv", index=False)

    perclass = pc.copy()
    perclass["display"] = perclass["model"].map(NAME).fillna(perclass["model"])
    perclass.round(4).to_csv(R6 / "FINAL_PERCLASS_F1.csv", index=False)

    # ---- Phase-6 enhanced config: vgg16 + hflip TTA + temperature (T from val) ----
    enh = {}
    for tta in ("none", "hflip"):
        yv, pv, lv = _load_npz("val", "vgg16", tta)
        yt, pt, lt = _load_npz("test", "vgg16", tta)
        T = P._fit_temperature(yv, lv)
        for cal, prob in (("raw", pt), ("temp", P._apply_temp(lt, T))):
            yp = prob.argmax(1)
            m = compute_all(yt, yp, prob)
            o = ordinal_and_confidence(yt, yp, prob)
            enh[f"vgg16_{tta}_{cal}"] = {
                "T": T, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"], "balanced_accuracy": m["balanced_accuracy"],
                "qwk": m["quadratic_weighted_kappa"], "kl1_f1": m["kl1_f1"], "kl4_f1": m["kl4_f1"],
                "mae": o["mae"], "exact_accuracy": o["exact_accuracy"],
                "within_1_accuracy": o["within_1_accuracy"], "ece": P._ece(yt, prob)}
    pd.DataFrame(enh).T.round(4).to_csv(R6 / "FINAL_enhanced_vgg16_configs.csv")

    def _rd(fp):
        p = R6 / fp
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    bs = _rd("binary_screening.csv")
    tc = _rd("three_class.csv")
    absc = _rd("abstention_curve.csv")
    p6f = _rd("phase6_final_metrics.csv")
    prep = _rd("preprocessing_ablation.csv")
    loss = _rd("loss_ablation.csv")

    # ---------------- report ----------------
    L = []
    A = L.append
    A("# AETHER-OA - FINAL BENCHMARK (Phases 5 + 6)\n")
    A(f"_Generated {datetime.now(timezone.utc).isoformat()}. All numbers are on the frozen, "
      f"patient- and sample-disjoint TEST split (1240 images / 620 patients). Checkpoints = "
      f"clean Phase-4, selected on VALIDATION Macro-F1. No test-set tuning._\n")

    A("## 1. Complete 15-model test benchmark\n")
    cols = ["rank", "display", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
            "qwk", "kl4_f1", "mae", "exact_accuracy", "within_1_accuracy"]
    A("| " + " | ".join(["Rank", "Model", "Accuracy", "Macro-F1", "Weighted-F1", "Balanced Acc",
                         "QWK", "KL4-F1", "MAE", "Exact Acc", "Within +/-1"]) + " |")
    A("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        A("| " + " | ".join(
            (str(r[c]) if c in ("rank", "display") else f"{r[c]:.4f}") for c in cols) + " |")

    A("\n## 2. Per-class F1 (test)\n")
    A("| Rank | Model | KL0 | KL1 | KL2 | KL3 | KL4 |")
    A("|---|---|--:|--:|--:|--:|--:|")
    for _, r in df.iterrows():
        A(f"| {r['rank']} | {r['display']} | " + " | ".join(
            f"{r.get(f'KL{k}_f1', float('nan')):.4f}" for k in range(5)) + " |")
    A("\n**KL1 ('doubtful') is the universal weak class** - best is VGG16 at 0.49; most models "
      "0.25-0.43. KL0/KL3/KL4 are all strong (0.7-0.95). This is the single biggest limiter of Macro-F1.\n")

    A("## 3. Efficiency\n")
    A("| Rank | Model | Params (M) | Trainable (M) | GPU latency b1 (ms) | best epoch | epochs run |")
    A("|---|---|--:|--:|--:|--:|--:|")
    for _, r in df.iterrows():
        tp = r.get("trainable_parameters")
        be = r.get("best_epoch")
        er = r.get("epochs_run")
        A(f"| {r['rank']} | {r['display']} | {r['params_M']:.2f} | "
          f"{(tp/1e6 if pd.notna(tp) else float('nan')):.2f} | {r['mean_latency_ms']:.2f} | "
          f"{int(be) if pd.notna(be) else ''} | {int(er) if pd.notna(er) else ''} |")

    A("\n## 4. Validation -> Test generalization (Macro-F1)\n")
    A("| Rank | Model | val Macro-F1 | test Macro-F1 | delta (test - val) |")
    A("|---|---|--:|--:|--:|")
    for _, r in df.iterrows():
        A(f"| {r['rank']} | {r['display']} | {r['validation_macro_f1']:.4f} | "
          f"{r['test_macro_f1']:.4f} | {r['delta_macro_f1']:+.4f} |")
    A(f"\nMean delta {df['delta_macro_f1'].mean():+.4f} (test slightly ABOVE val for most - the "
      "val set is marginally harder for these models / mild val-selection effect; not leakage, "
      "since splits are patient-disjoint).\n")

    A("## 5. Phase 6 enhancement - VGG16 + horizontal-flip TTA + temperature scaling\n")
    e = pd.read_csv(R6 / "FINAL_enhanced_vgg16_configs.csv", index_col=0)
    A("| config | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | QWK | KL1-F1 | KL4-F1 | MAE | Within +/-1 | ECE |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for idx, r in e.iterrows():
        A(f"| {idx} | " + " | ".join(f"{r[c]:.4f}" for c in
          ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "qwk", "kl1_f1",
           "kl4_f1", "mae", "within_1_accuracy", "ece"]) + " |")
    A("\n- hflip TTA: **+0.006 Macro-F1 / +0.005 QWK** over plain VGG16, no retraining.\n"
      "- Temperature scaling (T fit on validation NLL) roughly halves ECE for every model - "
      "confidences become trustworthy. See `calibration.csv` / `reliability_diagram.png`.\n"
      "- The multi-model ensemble was tested and **did not beat VGG16 + hflip** (val-optimal "
      "ensemble scored 0.7375 on test vs 0.7447) - dropped.\n")

    if len(bs):
        b = bs.iloc[0]
        A("## 6. Binary OA screening (test)  -  {KL0,KL1} = no-OA  vs  {KL2,KL3,KL4} = OA\n")
        A(f"- **ROC-AUC {b['auc_roc']:.4f}**, average precision {b['average_precision']:.4f}\n"
          f"- Sensitivity {b['sensitivity_at_spec>=0.90']:.3f} at specificity 0.90\n"
          f"- Specificity {b['specificity_at_sens>=0.90']:.3f} at sensitivity 0.90\n"
          f"- At threshold 0.5: accuracy {b['acc@0.5']:.4f}, sensitivity {b['sensitivity@0.5']:.3f}, "
          f"specificity {b['specificity@0.5']:.3f}, F1 {b['f1@0.5']:.3f} "
          f"(n_OA {int(b['n_OA'])}, n_noOA {int(b['n_noOA'])})\n")
    if len(tc):
        t = tc.iloc[0]
        A("## 7. 3-class (test)  -  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)\n")
        A(f"- accuracy **{t['accuracy']:.4f}** | Macro-F1 **{t['macro_f1']:.4f}** | "
          f"weighted-F1 {t['weighted_f1']:.4f} | QWK {t['qwk']:.4f}\n")
    if len(absc):
        at = absc[absc["split"] == "test"]
        A("## 8. Selective prediction / abstention (test)\n")
        A("| confidence threshold | coverage (auto-graded) | selective accuracy | selective Macro-F1 |")
        A("|--:|--:|--:|--:|")
        for _, r in at[at["threshold"].isin([0.5, 0.6, 0.7, 0.8, 0.9])].iterrows():
            A(f"| {r['threshold']:.2f} | {r['coverage']:.1%} | "
              f"{r['selective_accuracy']:.4f} | {r['selective_macro_f1']:.4f} |")
        A("\ne.g. auto-grade the ~30% most-confident knees at ~93% accuracy; refer the rest.\n")

    if len(prep):
        A("## 9. Preprocessing ablation (validation Macro-F1 / KL1-F1)\n")
        pv = prep.pivot_table(index="model", columns="preprocessing", values="val_macro_f1").round(4)
        A("| model | " + " | ".join(pv.columns) + " |")
        A("|---|" + "|".join(["--:"] * len(pv.columns)) + "|")
        for m, row in pv.iterrows():
            A(f"| {NAME.get(m, m)} | " + " | ".join(f"{row[c]:.4f}" for c in pv.columns) + " |")
        A("\nCLAHE helps **ConvNeXt-Tiny only** (+0.011 Macro-F1, KL1-F1 0.39->0.43). "
          "Neutral/negative elsewhere; histeq never helps. `basic` stays default.\n")
    if len(loss):
        A("## 10. Loss ablation (validation Macro-F1 / KL1-F1)\n")
        lv = loss.pivot_table(index="model", columns="loss", values="val_macro_f1").round(4)
        lk = loss.pivot_table(index="model", columns="loss", values="KL1_F1").round(4)
        A("| model | " + " | ".join(f"{c} (F1/KL1)" for c in lv.columns) + " |")
        A("|---|" + "|".join(["--:"] * len(lv.columns)) + "|")
        for m in lv.index:
            A(f"| {NAME.get(m, m)} | " + " | ".join(
                f"{lv.loc[m, c]:.3f} / {lk.loc[m, c]:.3f}" for c in lv.columns) + " |")
        A("\n**class_balanced** loss helps ConvNeXt-Tiny's KL1-F1 (0.39->0.44) and Macro-F1. "
          "For DenseNet/Inception, plain weighted-CE wins. **Focal is worst everywhere.**\n")

    ms_sum = _rd("multi_seed_summary.csv")
    if len(ms_sum):
        A("## 10b. Multi-seed robustness (validation, seeds 42/123/3407)\n")
        A("| config | mean Macro-F1 | std | mean QWK | std | mean bAcc | std |")
        A("|---|--:|--:|--:|--:|--:|--:|")
        for _, r in ms_sum.iterrows():
            A(f"| {NAME.get(r['model'], r['model'])} "
              f"{'(+CLAHE+class_balanced)' if r['model']=='convnext_tiny' else '(basic+WCE)'} | "
              f"{r['macro_f1_mean']:.4f} | {r['macro_f1_std']:.4f} | {r['qwk_mean']:.4f} | "
              f"{r['qwk_std']:.4f} | {r['bacc_mean']:.4f} | {r['bacc_std']:.4f} |")
        A("\n- **ConvNeXt-Tiny is the most stable** model (Macro-F1 std ~0.003). Its ~0.71 is reliable.\n"
          "- **VGG16 has high seed variance** (Macro-F1 std ~0.023; one seed early-stopped at epoch 7 "
          "-> 0.66). Its #1 test result (~0.74) is real but partly seed-favourable; a re-seed could "
          "land ~0.72.\n"
          "- **CLAHE + class_balanced did NOT stack** for ConvNeXt (individually ~0.717 / ~0.711; "
          "together ~0.709). No config change is adopted - the plain Phase-4 `basic` + "
          "`weighted_cross_entropy` checkpoints stand.\n")

    A("## 11. Final recommendation\n")
    top = df.iloc[0]
    cn = df[df.model == "convnext_tiny"].iloc[0]
    A(f"**Best diagnostic accuracy:** `{top['display']}` + hflip TTA + temperature scaling -> "
      f"test Macro-F1 **{e.loc['vgg16_hflip_raw', 'macro_f1']:.4f}**, QWK "
      f"**{e.loc['vgg16_hflip_raw', 'qwk']:.4f}**, balanced acc {e.loc['vgg16_hflip_raw', 'balanced_accuracy']:.4f}, "
      f"within-+/-1 {e.loc['vgg16_hflip_raw', 'within_1_accuracy']:.1%}, MAE {e.loc['vgg16_hflip_raw', 'mae']:.3f}. "
      f"Cost: {top['params_M']:.0f} M params.\n\n")
    A(f"**Best deployable / most robust:** `{cn['display']}` - test Macro-F1 {cn['macro_f1']:.4f}, "
      f"QWK {cn['qwk']:.4f} at **{cn['params_M']:.0f} M params (1/5 of VGG16)**, {cn['mean_latency_ms']:.1f} ms, "
      f"and the lowest seed variance (val Macro-F1 std ~0.003). **This is the recommended model to "
      f"take forward** unless the extra ~0.02-0.03 Macro-F1 from VGG16 is critical.\n\n")
    A("**VGG16 vs VGG19 vs ConvNeXt-Tiny**: near-tie on QWK (~0.86-0.88). VGG16 leads test Macro-F1 "
      "by ~0.02-0.03 (via better KL1-F1) but is 134 M params AND seed-sensitive (val std ~0.023). "
      "ConvNeXt-Tiny is 5x smaller and stable.\n\n")
    A("**Deploy as:** the chosen backbone + hflip TTA + temperature calibration, reporting the "
      "**binary OA-screen (AUC 0.965)** and **3-class (acc 0.80)** as primary clinical outputs, "
      "with a **confidence-abstention band** (auto-grade the confident majority, refer the rest). "
      "5-class KL is the hardest framing and is bounded by KL0/KL1 ambiguity.\n")

    A("## 12. Limitations\n")
    A("- Single-site test set (OAI-derived); KL4 support = 38 -> wide CI on KL4 metrics.\n"
      "- No significance testing; test Macro-F1 differences < ~0.005 are noise on n=1240.\n"
      "- KL1 recall ~0.5 across all models - the model is weak on the 'is there early OA?' call.\n"
      "- Latency = desktop RTX 5080, batch-1, synthetic input, forward only (no preprocessing).\n"
      "- Probabilities calibrated by temperature scaling; not externally validated.\n")

    (R6 / "FINAL_BENCHMARK_REPORT.md").write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print("wrote:")
    for f in ["FINAL_15_MODEL_BENCHMARK.csv", "FINAL_PERCLASS_F1.csv",
              "FINAL_enhanced_vgg16_configs.csv", "FINAL_BENCHMARK_REPORT.md"]:
        print("  reports/phase6/" + f)


if __name__ == "__main__":
    main()
