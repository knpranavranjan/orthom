"""PHASE 7 - single-shot FINAL TEST evaluation of the frozen Phase-7 winner(s).

Evaluation ONLY. No training, no threshold search on test. Temperature and the
abstain threshold are fitted on VALIDATION; the 1240-image test split is scored
exactly once per model here.

    python scripts/phase7_final_eval.py --experiment-id mobilenet_v2__M8_stack_ema \
        --experiment-id efficientnet_b0__M7_stack --log-file reports/phase7/p7_finaleval.log

For every model reports: 5-class (Acc, Macro/Weighted-F1, Balanced Acc, QWK,
per-class F1, MAE, within-1), binary OA screen (Acc/AUC/PR-AUC/sens/spec),
3-class (Acc/Macro-F1/QWK), calibration (ECE pre/post-T), abstention curve,
hflip-TTA delta, latency/size/params, and the Phase-5 -> Phase-7 generalisation gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402
from src.datasets import build_dataloaders  # noqa: E402
from src.benchmark import complexity as cx, latency as lat  # noqa: E402
from src.benchmark.metrics import compute_all  # noqa: E402
from src.benchmark.phase5_metrics import ordinal_and_confidence  # noqa: E402
from src.benchmark.models import build_model, count_parameters  # noqa: E402
from src.benchmark.plots import confusion_matrix_plots  # noqa: E402

LOG = get_logger("p7eval")
CFG = PROJECT_ROOT / "configs" / "benchmark_phase7.yaml"
OUT = PROJECT_ROOT / "reports" / "phase7"
PHASE5_CSV = PROJECT_ROOT / "reports" / "phase5" / "final_15_model_test_benchmark.csv"
_TAG = ""


# --------------------------------------------------------------------------- #
def _softmax_T(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _fit_T(y, z):
    from scipy.optimize import minimize_scalar
    def nll(logT):
        p = _softmax_T(z, np.exp(logT))
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
    return float(np.exp(minimize_scalar(nll, bounds=(-2.5, 2.5), method="bounded").x))


def _ece(y, p, n_bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def _binary_screen(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    p_oa = p[:, 2:].sum(1)
    yb = (y >= 2).astype(int)
    pred = (p_oa >= 0.5).astype(int)
    tp = int(((pred == 1) & (yb == 1)).sum()); fp = int(((pred == 1) & (yb == 0)).sum())
    tn = int(((pred == 0) & (yb == 0)).sum()); fn = int(((pred == 0) & (yb == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {
        "accuracy": round((pred == yb).mean(), 4),
        "roc_auc": round(float(roc_auc_score(yb, p_oa)), 4),
        "pr_auc": round(float(average_precision_score(yb, p_oa)), 4),
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _three_class(y, p):
    mp = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    g3 = np.stack([p[:, 0], p[:, 1] + p[:, 2], p[:, 3] + p[:, 4]], 1)
    t3 = np.vectorize(mp.get)(y); pr = g3.argmax(1)
    from sklearn.metrics import f1_score, cohen_kappa_score
    return {"accuracy": round((t3 == pr).mean(), 4),
            "macro_f1": round(float(f1_score(t3, pr, average="macro")), 4),
            "qwk": round(float(cohen_kappa_score(t3, pr, weights="quadratic")), 4),
            "confusion": np.bincount(t3 * 3 + pr, minlength=9).reshape(3, 3).tolist()}


def _abstention_curve(y, p):
    conf = p.max(1); pred = p.argmax(1); rows = []
    for t in np.round(np.arange(0.0, 0.96, 0.05), 2):
        keep = conf >= t
        if keep.sum() == 0:
            continue
        rows.append({"threshold": float(t), "coverage": round(float(keep.mean()), 4),
                     "selective_accuracy": round(float((pred[keep] == y[keep]).mean()), 4),
                     "n_kept": int(keep.sum())})
    return rows


def _logits(model, loader, device, *, tta=False, ordinal=None):
    import torch
    ys, zs = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            views = [x, torch.flip(x, dims=[-1])] if tta else [x]
            if ordinal == "corn":
                from src.benchmark.ordinal import corn_logits_to_probs
                pr = np.mean([corn_logits_to_probs(model(v).float().cpu().numpy()) for v in views], 0)
                zs.append(np.log(np.clip(pr, 1e-8, 1)))          # log-prob stand-in for "logits"
            else:
                pr = torch.stack([torch.softmax(model(v).float(), 1) for v in views]).mean(0)
                zs.append(torch.log(pr.clamp_min(1e-8)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys).astype(int), np.concatenate(zs).astype(np.float64)


# --------------------------------------------------------------------------- #
def eval_one(eid: str, base_cfg: dict, device) -> dict:
    import torch
    run_dir = PROJECT_ROOT / base_cfg["paths"]["models_dir"] / eid
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    # arch: checkpoint's own model_name wins (Phase-8 KD dirs are named K6_*, not <arch>__*)
    arch = ck.get("model_name") or eid.split("__", 1)[0]
    ordinal = ck.get("ordinal") or None
    head = 4 if ordinal == "corn" else base_cfg["num_classes"]
    _cfg_json = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    rc = _cfg_json.get("resolved_cfg", base_cfg)
    variant = rc.get("preprocessing_variant", base_cfg.get("preprocessing_variant", "basic"))
    norm = rc.get("data", base_cfg["data"])["normalization"]
    aug = rc.get("data", base_cfg["data"]).get("augmentation_yaml", "configs/augmentation.yaml")

    built = build_model(arch, num_classes=head, in_chans=base_cfg["in_channels"], pretrained=True)
    built.model.load_state_dict(ck["state_dict"], strict=True)
    built.model.to(device).eval()
    pc = count_parameters(built.model)
    LOG.info("[%s] arch=%s ordinal=%s ema=%s variant=%s | %.2fM params", eid, arch, ordinal,
             ck.get("ema"), variant, pc["parameters"] / 1e6)

    dl = {s: build_dataloaders(variant=variant, normalization=norm, batch_size=64, num_workers=0,
                               out_channels=base_cfg["in_channels"], imbalance="none",
                               augmentation_yaml=aug, return_meta=False, pin_memory=False,
                               persistent_workers=False, seed=rc.get("seed", 42))[s]
          for s in ("val", "test")}

    yv, zv = _logits(built.model, dl["val"], device, ordinal=ordinal)
    yt, zt = _logits(built.model, dl["test"], device, ordinal=ordinal)
    yt_tta, zt_tta = _logits(built.model, dl["test"], device, tta=True, ordinal=ordinal)

    T = _fit_T(yv, zv)
    p_raw = _softmax_T(zt, 1.0)
    p_cal = _softmax_T(zt, T)
    p_tta = _softmax_T(zt_tta, T)

    m = compute_all(yt, p_cal.argmax(1), p_cal)
    o = ordinal_and_confidence(yt, p_cal.argmax(1), p_cal)
    m_tta = compute_all(yt_tta, p_tta.argmax(1), p_tta)

    # abstain threshold from VAL @ 90% selective accuracy
    pv = _softmax_T(zv, T); cv = pv.max(1); pdv = pv.argmax(1)
    thr = 0.0
    for t in np.round(np.arange(0.30, 0.981, 0.01), 2):
        k = cv >= t
        if k.sum() >= 50 and (pdv[k] == yv[k]).mean() >= 0.90:
            thr = float(t); break
    keep_t = p_cal.max(1) >= thr
    sel_acc = float((p_cal.argmax(1)[keep_t] == yt[keep_t]).mean()) if keep_t.any() else float("nan")

    cm_dir = OUT / "confusion_matrices"
    confusion_matrix_plots(m["confusion_matrix"], f"phase7_{eid}", cm_dir)

    p5 = pd.read_csv(PHASE5_CSV) if PHASE5_CSV.exists() else pd.DataFrame()
    p5row = p5[p5["model"] == arch]
    p5_macro = float(p5row["macro_f1"].iloc[0]) if len(p5row) else None
    val_macro = float(ck.get("selection_value") or np.nan)

    res = {
        "experiment_id": eid, "arch": arch, "ordinal": ordinal or "", "ema": bool(ck.get("ema")),
        "preprocessing": variant, "temperature": round(T, 4), "abstain_threshold": round(thr, 3),
        "test_accuracy": round(m["accuracy"], 4), "test_macro_f1": round(m["macro_f1"], 4),
        "test_weighted_f1": round(m["weighted_f1"], 4),
        "test_balanced_accuracy": round(m["balanced_accuracy"], 4),
        "test_qwk": round(m["quadratic_weighted_kappa"], 4),
        "test_mae": round(o["mae"], 4), "test_within1": round(o["within_1_accuracy"], 4),
        **{f"test_KL{k}_f1": round(m[f"kl{k}_f1"], 4) for k in range(5)},
        "tta_hflip_macro_f1": round(m_tta["macro_f1"], 4),
        "tta_hflip_qwk": round(m_tta["quadratic_weighted_kappa"], 4),
        "tta_delta_macro_f1": round(m_tta["macro_f1"] - m["macro_f1"], 4),
        "ece_pre_T": round(_ece(yt, p_raw), 4), "ece_post_T": round(_ece(yt, p_cal), 4),
        "abstain_coverage": round(float(keep_t.mean()), 4), "abstain_selective_acc": round(sel_acc, 4),
        "binary_oa": _binary_screen(yt, p_cal),
        "three_class": _three_class(yt, p_cal),
        "parameters": pc["parameters"], "model_size_mb": cx.checkpoint_size_mb(run_dir / "best.pt"),
        "gmacs": cx.gmacs_fvcore(built.model, (1, 3, 224, 224), device="cpu").get("gmacs"),
        "cpu_latency_ms": lat.measure_latency(built.model, torch.device("cpu"), (1, 3, 224, 224),
                                              warmup=5, iters=30)["mean_ms"],
        "val_macro_f1": round(val_macro, 4) if val_macro == val_macro else None,
        "phase5_test_macro_f1": p5_macro,
        "gen_gap_val_minus_test": (round(val_macro - m["macro_f1"], 4) if val_macro == val_macro else None),
        "per_class": {f"KL{k}": {"precision": round(m[f"kl{k}_precision"], 4),
                                 "recall": round(m[f"kl{k}_recall"], 4),
                                 "f1": round(m[f"kl{k}_f1"], 4),
                                 "support": int(m[f"kl{k}_support"])} for k in range(5)},
        "abstention_curve": _abstention_curve(yt, p_cal),
        "confusion_matrix": m["confusion_matrix"],
    }
    (OUT / f"final_eval_{eid}.json").write_text(json.dumps(res, indent=2, default=str))
    LOG.info("[%s] TEST macroF1=%.4f QWK=%.4f | binOA acc=%.4f auc=%.4f | 3cls acc=%.4f | "
             "TTA macroF1=%.4f (%+.4f) | ECE %.3f->%.3f | gap %.4f",
             eid, res["test_macro_f1"], res["test_qwk"], res["binary_oa"]["accuracy"],
             res["binary_oa"]["roc_auc"], res["three_class"]["accuracy"], res["tta_hflip_macro_f1"],
             res["tta_delta_macro_f1"], res["ece_pre_T"], res["ece_post_T"],
             res["gen_gap_val_minus_test"] or 0.0)
    del built.model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", action="append", required=True)
    ap.add_argument("--config", type=Path, default=CFG,
                    help="benchmark config whose paths.models_dir holds the run dirs (default: phase7)")
    ap.add_argument("--tag", default="", help="filename suffix for the FINAL_* outputs")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()
    if args.log_file:
        add_file_logger(args.log_file)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(args.config if args.config.is_absolute() else PROJECT_ROOT / args.config)
    global _TAG, OUT
    _TAG = ("_" + args.tag) if args.tag else ""
    OUT = PROJECT_ROOT / base_cfg["paths"]["reports_dir"]
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for eid in args.experiment_id:
        try:
            results.append(eval_one(eid, base_cfg, device))
        except Exception as e:  # noqa: BLE001
            LOG.exception("FAILED %s: %s", eid, e)

    if not results:
        LOG.error("no models evaluated successfully")
        sys.exit(1)
    flat = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("binary_oa", "three_class", "per_class", "abstention_curve", "confusion_matrix")}
        row["binary_oa_acc"] = r["binary_oa"]["accuracy"]
        row["binary_oa_auc"] = r["binary_oa"]["roc_auc"]
        row["binary_oa_sensitivity"] = r["binary_oa"]["sensitivity"]
        row["binary_oa_specificity"] = r["binary_oa"]["specificity"]
        row["three_class_acc"] = r["three_class"]["accuracy"]
        flat.append(row)
    df = pd.DataFrame(flat).sort_values("test_macro_f1", ascending=False)
    df.to_csv(OUT / f"FINAL_MOBILENETV2_BENCHMARK{_TAG}.csv", index=False)

    pc_rows = []
    for r in results:
        for k in range(5):
            pc_rows.append({"experiment_id": r["experiment_id"], "class": f"KL{k}", **r["per_class"][f"KL{k}"]})
    pd.DataFrame(pc_rows).to_csv(OUT / f"FINAL_PERCLASS_METRICS{_TAG}.csv", index=False)

    ab_rows = []
    for r in results:
        for a in r["abstention_curve"]:
            ab_rows.append({"experiment_id": r["experiment_id"], **a})
    pd.DataFrame(ab_rows).to_csv(OUT / f"ABSTENTION_CURVE{_TAG}.csv", index=False)

    _write_report(results)
    LOG.info("wrote %s + FINAL_PERCLASS_METRICS.csv + ABSTENTION_CURVE.csv + FINAL_PHASE7_REPORT.md",
             OUT / f"FINAL_MOBILENETV2_BENCHMARK{_TAG}.csv")


def _write_report(results: list[dict]):
    L = ["# AETHER-OA X-ray - Phase 7 FINAL TEST evaluation", "",
         "Evaluation only. Temperature + abstain threshold fitted on VALIDATION; the",
         "1240-image test split scored once per model. KL1 remains the intrinsic weak class.",
         ""]
    best = max(results, key=lambda r: r["test_macro_f1"])
    L += [f"**Best by Macro-F1:** `{best['experiment_id']}` - test Macro-F1 "
          f"{best['test_macro_f1']}, QWK {best['test_qwk']}, binary-OA acc "
          f"{best['binary_oa']['accuracy']} (AUC {best['binary_oa']['roc_auc']}), "
          f"3-class acc {best['three_class']['accuracy']}.", ""]
    L.append("## 5-class KL (test)")
    L.append("| run | Acc | MacroF1 | wF1 | bAcc | QWK | MAE | w1 | KL0 | KL1 | KL2 | KL3 | KL4 | +hflip |")
    L.append("|---|--|--|--|--|--|--|--|--|--|--|--|--|--|")
    for r in sorted(results, key=lambda r: r["test_macro_f1"], reverse=True):
        L.append(f"| {r['experiment_id']} | {r['test_accuracy']} | {r['test_macro_f1']} | "
                 f"{r['test_weighted_f1']} | {r['test_balanced_accuracy']} | {r['test_qwk']} | "
                 f"{r['test_mae']} | {r['test_within1']} | {r['test_KL0_f1']} | {r['test_KL1_f1']} | "
                 f"{r['test_KL2_f1']} | {r['test_KL3_f1']} | {r['test_KL4_f1']} | "
                 f"{r['tta_hflip_macro_f1']} ({r['tta_delta_macro_f1']:+}) |")
    L += ["", "## Binary OA screen  {KL0,KL1} vs {KL2,KL3,KL4}"]
    L.append("| run | Acc | ROC-AUC | PR-AUC | Sens | Spec | Prec |")
    L.append("|---|--|--|--|--|--|--|")
    for r in sorted(results, key=lambda r: r["binary_oa"]["roc_auc"], reverse=True):
        b = r["binary_oa"]
        L.append(f"| {r['experiment_id']} | {b['accuracy']} | {b['roc_auc']} | {b['pr_auc']} | "
                 f"{b['sensitivity']} | {b['specificity']} | {b['precision']} |")
    L += ["", "## 3-class  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)"]
    L.append("| run | Acc | MacroF1 | QWK |")
    L.append("|---|--|--|--|")
    for r in sorted(results, key=lambda r: r["three_class"]["accuracy"], reverse=True):
        t = r["three_class"]
        L.append(f"| {r['experiment_id']} | {t['accuracy']} | {t['macro_f1']} | {t['qwk']} |")
    L += ["", "## Calibration / abstention / efficiency / generalisation"]
    L.append("| run | T | ECE pre->post | abstain thr | coverage | selective-acc | params | size MB | CPU ms | val->test gap |")
    L.append("|---|--|--|--|--|--|--|--|--|--|")
    for r in results:
        L.append(f"| {r['experiment_id']} | {r['temperature']} | {r['ece_pre_T']}->{r['ece_post_T']} | "
                 f"{r['abstain_threshold']} | {r['abstain_coverage']} | {r['abstain_selective_acc']} | "
                 f"{r['parameters']:,} | {r['model_size_mb']} | {r['cpu_latency_ms']} | "
                 f"{r['gen_gap_val_minus_test']} |")
    L += ["", "## Honest read on the >90% goal", "",
          "- **5-class accuracy does not reach 90%** and is not expected to for any model on",
          "  OAI KL grades (label noise floor; field SOTA ~0.72-0.75).",
          f"- **Binary OA screening** is the clinically meaningful task: best test accuracy "
          f"{max(r['binary_oa']['accuracy'] for r in results)}, best AUC "
          f"{max(r['binary_oa']['roc_auc'] for r in results)}.",
          "- KL1 F1 stays the ceiling-limited class; compare per-run above."]
    (OUT / f"FINAL_PHASE7_REPORT{_TAG}.md").write_text("\n".join(L))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
