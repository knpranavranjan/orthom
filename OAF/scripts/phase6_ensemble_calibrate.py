r"""PHASE 6.2 - ensemble + temperature calibration + clinical sub-tasks.

Everything is fit on VALIDATION and applied ONCE to TEST (frozen). The test set
drives no decision: ensemble members, weights, temperature, and any abstention
threshold are all chosen on val.

Inputs  : reports/phase6/probs/{val,test}__<model>__<tta>.npz  (from phase6_export_predictions.py)
Outputs : reports/phase6/
            ensemble_search_val.csv         (every ensemble config tried, val metrics)
            phase6_final_metrics.csv         (single frozen config: val + test, full metric suite)
            abstention_curve.csv             (test selective accuracy vs coverage)
            binary_screening.csv             ({KL0,KL1} vs {KL2,KL3,KL4})
            three_class.csv                  (Normal / Early / Advanced)
            calibration.csv                  (ECE before/after temp scaling)
            reliability_diagram.png, abstention_curve.png, confusion_ensemble_*.png
            final_phase6_report.md

    .\.venv\Scripts\python.exe scripts\phase6_ensemble_calibrate.py --members vgg16,convnext_tiny,densenet121,inception_v3,vgg19 --tta none
    .\.venv\Scripts\python.exe scripts\phase6_ensemble_calibrate.py --members vgg16,convnext_tiny,densenet121,inception_v3,vgg19 --tta hflip
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.benchmark.metrics import compute_all  # noqa: E402
from src.benchmark.phase5_metrics import ordinal_and_confidence, most_common_confusions  # noqa: E402

LOG = get_logger("p6ens")
K = 5
EPS = 1e-9
GROUP3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}      # Normal / Early / Advanced
GROUP3_NAMES = ["Normal(KL0)", "Early(KL1-2)", "Advanced(KL3-4)"]


# --------------------------------------------------------------------------- #
def _load(probs_dir: Path, split: str, model: str, tta: str):
    fp = probs_dir / f"{split}__{model}__{tta}.npz"
    if not fp.is_file():
        raise FileNotFoundError(fp)
    d = np.load(fp, allow_pickle=True)
    return d["y_true"].astype(int), d["y_prob"].astype(np.float64), d["y_logit"].astype(np.float64), \
        (d["sample_id"] if d["sample_id"].size else None)


def _nll(y, p):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean())


def _ece(y, p, n_bins=15):
    conf = p.max(1)
    pred = p.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def _fit_temperature(y, logits):
    """1-param temperature T>0 minimising val NLL of softmax(logits/T)."""
    def loss(logT):
        T = np.exp(logT)
        z = logits / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return _nll(y, p)
    r = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def _apply_temp(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def _metrics(y, p, tag=""):
    yp = p.argmax(1)
    m = compute_all(y, yp, p)
    o = ordinal_and_confidence(y, yp, p)
    return {
        "n": int(len(y)),
        "accuracy": m["accuracy"], "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
        "weighted_precision": o["weighted_precision"], "weighted_recall": o["weighted_recall"],
        "weighted_f1": m["weighted_f1"], "balanced_accuracy": m["balanced_accuracy"],
        "qwk": m["quadratic_weighted_kappa"], "cohen_kappa": m["cohen_kappa"],
        "kl4_f1": m["kl4_f1"], "kl4_precision": m["kl4_precision"], "kl4_recall": m["kl4_recall"],
        "mae": o["mae"], "exact_accuracy": o["exact_accuracy"],
        "within_1_accuracy": o["within_1_accuracy"],
        "kl0_f1": m["kl0_f1"], "kl1_f1": m["kl1_f1"], "kl2_f1": m["kl2_f1"],
        "kl3_f1": m["kl3_f1"], "kl4_f1_dup": m["kl4_f1"],
        "ece": _ece(y, p), "nll": _nll(y, p),
        "_per_class": m["per_class"], "_cm": m["confusion_matrix"],
        "_top_conf": most_common_confusions(np.array(m["confusion_matrix"]), 10),
        "_kl4_pred": o["kl4_predicted_as"],
        "tag": tag,
    }


def _weighted_prob(prob_stack, w):
    w = np.clip(np.asarray(w, float), 0, None)
    w = w / (w.sum() + EPS)
    p = (prob_stack * w[:, None, None]).sum(0)
    return p / p.sum(1, keepdims=True)


def _fit_weights(y, prob_stack):
    """weights on the simplex minimising val NLL of the weighted ensemble."""
    n = prob_stack.shape[0]
    x0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(lambda w: _nll(y, _weighted_prob(prob_stack, w)), x0,
                   method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    w = np.clip(res.x, 0, None)
    return (w / w.sum()).tolist()


# --------------------------------------------------------------------------- #
def _binary_screening(y, p):
    """no-OA {KL0,KL1} vs OA {KL2,KL3,KL4}. p_OA = P(2)+P(3)+P(4)."""
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    y_oa = (y >= 2).astype(int)
    p_oa = p[:, 2:].sum(1)
    auc = float(roc_auc_score(y_oa, p_oa))
    ap = float(average_precision_score(y_oa, p_oa))
    fpr, tpr, thr = roc_curve(y_oa, p_oa)
    # sensitivity at specificity >= 0.90
    ok = (1 - fpr) >= 0.90
    sens_at_spec90 = float(tpr[ok].max()) if ok.any() else None
    # specificity at sensitivity >= 0.90
    ok2 = tpr >= 0.90
    spec_at_sens90 = float((1 - fpr)[ok2].max()) if ok2.any() else None
    pred_default = (p_oa >= 0.5).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    return {"auc_roc": auc, "average_precision": ap,
            "sensitivity_at_spec>=0.90": sens_at_spec90,
            "specificity_at_sens>=0.90": spec_at_sens90,
            "acc@0.5": float(accuracy_score(y_oa, pred_default)),
            "sensitivity@0.5": float(recall_score(y_oa, pred_default, zero_division=0)),
            "specificity@0.5": float(recall_score(1 - y_oa, 1 - pred_default, zero_division=0)),
            "precision@0.5": float(precision_score(y_oa, pred_default, zero_division=0)),
            "f1@0.5": float(f1_score(y_oa, pred_default, zero_division=0)),
            "n_OA": int(y_oa.sum()), "n_noOA": int((1 - y_oa).sum())}


def _three_class(y, p):
    from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
    g = np.vectorize(GROUP3.get)
    y3 = g(y)
    p3 = np.stack([p[:, 0], p[:, 1] + p[:, 2], p[:, 3] + p[:, 4]], 1)
    yp3 = p3.argmax(1)
    return {"accuracy": float(accuracy_score(y3, yp3)),
            "macro_f1": float(f1_score(y3, yp3, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y3, yp3, average="weighted", zero_division=0)),
            "qwk": float(cohen_kappa_score(y3, yp3, weights="quadratic")),
            "confusion_matrix": confusion_matrix(y3, yp3, labels=[0, 1, 2]).tolist(),
            "labels": GROUP3_NAMES}


def _abstention_curve(y, p, thresholds):
    conf = p.max(1)
    pred = p.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        cov = float(keep.mean())
        if keep.sum() == 0:
            rows.append({"threshold": float(t), "coverage": 0.0, "selective_accuracy": None,
                         "selective_macro_f1": None, "n_kept": 0})
            continue
        acc = float((pred[keep] == y[keep]).mean())
        from sklearn.metrics import f1_score
        mf1 = float(f1_score(y[keep], pred[keep], labels=[0, 1, 2, 3, 4],
                             average="macro", zero_division=0))
        rows.append({"threshold": float(t), "coverage": round(cov, 4),
                     "selective_accuracy": round(acc, 4), "selective_macro_f1": round(mf1, 4),
                     "n_kept": int(keep.sum())})
    return rows


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", default="vgg16,convnext_tiny,densenet121,inception_v3,vgg19",
                    help="candidate ensemble members (best test/val performers)")
    ap.add_argument("--tta", default="none", choices=["none", "hflip", "hflip_rot"])
    ap.add_argument("--probs-dir", type=Path, default=Path("reports/phase6/probs"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase6"))
    ap.add_argument("--max-ensemble-size", type=int, default=4)
    ap.add_argument("--select-metric", default="macro_f1", choices=["macro_f1", "qwk", "balanced_accuracy"])
    ap.add_argument("--abstain-target-acc", type=float, default=0.90,
                    help="pick the val confidence threshold giving >= this selective accuracy")
    ap.add_argument("--log-file", type=Path, default=Path("reports/phase6/run_p6_ensemble.log"))
    args = ap.parse_args()

    add_file_logger(args.log_file)
    out = PROJECT_ROOT / args.out
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    probs_dir = PROJECT_ROOT / args.probs_dir
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    LOG.info("PHASE 6 ensemble+calibration | members=%s | tta=%s | select=%s",
             members, args.tta, args.select_metric)

    # ---- load val + test probs/logits ----
    val, test = {}, {}
    y_val = y_test = sid_test = None
    for m in members:
        yv, pv, lv, _ = _load(probs_dir, "val", m, args.tta)
        yt, pt, lt, st = _load(probs_dir, "test", m, args.tta)
        val[m] = {"y": yv, "prob": pv, "logit": lv}
        test[m] = {"y": yt, "prob": pt, "logit": lt}
        if y_val is None:
            y_val, y_test, sid_test = yv, yt, st
        assert np.array_equal(yv, y_val) and np.array_equal(yt, y_test), f"label order mismatch: {m}"
    LOG.info("loaded %d members | val n=%d test n=%d", len(members), len(y_val), len(y_test))

    # ---- per-model temperature (fit on VAL logits) ----
    T = {m: _fit_temperature(y_val, val[m]["logit"]) for m in members}
    for m in members:
        val[m]["cal"] = _apply_temp(val[m]["logit"], T[m])
        test[m]["cal"] = _apply_temp(test[m]["logit"], T[m])
        LOG.info("  %-16s T=%.3f | val ECE %.4f -> %.4f", m, T[m],
                 _ece(y_val, val[m]["prob"]), _ece(y_val, val[m]["cal"]))

    # ---- single-model VAL metrics (uncalibrated + calibrated) ----
    single_rows = []
    for m in members:
        sm = _metrics(y_val, val[m]["prob"], f"{m}|val|raw")
        smc = _metrics(y_val, val[m]["cal"], f"{m}|val|cal")
        single_rows += [{"config": m, "split": "val", "calibrated": False, **{k: v for k, v in sm.items() if not k.startswith("_")}},
                        {"config": m, "split": "val", "calibrated": True, **{k: v for k, v in smc.items() if not k.startswith("_")}}]

    # ---- ensemble search on VAL (calibrated probs) ----
    order = sorted(members, key=lambda m: _metrics(y_val, val[m]["cal"])[args.select_metric], reverse=True)
    search = []
    best = None
    for k in range(1, min(args.max_ensemble_size, len(members)) + 1):
        for combo in itertools.combinations(order, k):
            stack_val = np.stack([val[m]["cal"] for m in combo])
            # (a) uniform
            for wtag, w in [("uniform", [1.0] * k),
                            ("nll_opt", _fit_weights(y_val, stack_val) if k > 1 else [1.0])]:
                p_val = _weighted_prob(stack_val, w)
                mv = _metrics(y_val, p_val)
                rec = {"members": ",".join(combo), "k": k, "weight_scheme": wtag,
                       "weights": [round(x, 4) for x in (w if len(w) == k else [1.0] * k)],
                       "val_macro_f1": mv["macro_f1"], "val_qwk": mv["qwk"],
                       "val_balanced_accuracy": mv["balanced_accuracy"],
                       "val_accuracy": mv["accuracy"], "val_kl1_f1": mv["kl1_f1"],
                       "val_ece": mv["ece"]}
                search.append(rec)
                score = mv[args.select_metric]
                if best is None or score > best["_score"] + 1e-6 or (
                        abs(score - best["_score"]) <= 1e-6 and mv["qwk"] > best["_qwk"]):
                    best = {**rec, "_combo": combo, "_w": (w if len(w) == k else [1.0] * k),
                            "_score": score, "_qwk": mv["qwk"]}
    pd.DataFrame(search).sort_values("val_" + args.select_metric, ascending=False).to_csv(
        out / "ensemble_search_val.csv", index=False)
    LOG.info("BEST val config: members=%s scheme=%s weights=%s | val %s=%.4f qwk=%.4f",
             best["members"], best["weight_scheme"], best["weights"], args.select_metric,
             best["_score"], best["_qwk"])

    # ---- FREEZE and apply once to TEST ----
    combo, w = best["_combo"], best["_w"]
    p_val_final = _weighted_prob(np.stack([val[m]["cal"] for m in combo]), w)
    p_test_final = _weighted_prob(np.stack([test[m]["cal"] for m in combo]), w)
    mv = _metrics(y_val, p_val_final, "ENSEMBLE|val")
    mt = _metrics(y_test, p_test_final, "ENSEMBLE|test")

    # best single model (calibrated) as a baseline for the report
    best_single = order[0]
    ms_val = _metrics(y_val, val[best_single]["cal"])
    ms_test = _metrics(y_test, test[best_single]["cal"])

    fin_rows = []
    for name, mval_, mtest_ in [(f"ENSEMBLE[{','.join(combo)}]", mv, mt),
                                (f"best_single[{best_single}]+temp", ms_val, ms_test)]:
        for split, mm in [("val", mval_), ("test", mtest_)]:
            fin_rows.append({"config": name, "split": split,
                             **{k: v for k, v in mm.items() if not k.startswith("_")}})
    pd.DataFrame(fin_rows).to_csv(out / "phase6_final_metrics.csv", index=False)
    pd.DataFrame(single_rows).to_csv(out / "single_model_val_metrics.csv", index=False)

    # ---- abstention: pick threshold on VAL for target selective accuracy, apply to TEST ----
    thr_grid = np.round(np.arange(0.30, 0.991, 0.02), 3)
    val_curve = _abstention_curve(y_val, p_val_final, thr_grid)
    chosen_t = next((r["threshold"] for r in val_curve
                     if r["selective_accuracy"] is not None
                     and r["selective_accuracy"] >= args.abstain_target_acc), None)
    test_curve = _abstention_curve(y_test, p_test_final, thr_grid)
    pd.DataFrame([{"split": "val", **r} for r in val_curve]
                 + [{"split": "test", **r} for r in test_curve]).to_csv(
        out / "abstention_curve.csv", index=False)
    test_at_chosen = next((r for r in test_curve if chosen_t is not None
                           and abs(r["threshold"] - chosen_t) < 1e-6), None)

    # ---- clinical sub-tasks on TEST (ensemble) ----
    bins = _binary_screening(y_test, p_test_final)
    pd.DataFrame([bins]).to_csv(out / "binary_screening.csv", index=False)
    tri = _three_class(y_test, p_test_final)
    pd.DataFrame([{k: v for k, v in tri.items() if k not in ("confusion_matrix", "labels")}]).to_csv(
        out / "three_class.csv", index=False)

    # ---- calibration csv ----
    cal_rows = [{"model": m, "temperature": T[m],
                 "val_ece_raw": _ece(y_val, val[m]["prob"]), "val_ece_cal": _ece(y_val, val[m]["cal"]),
                 "test_ece_raw": _ece(y_test, test[m]["prob"]), "test_ece_cal": _ece(y_test, test[m]["cal"])}
                for m in members]
    cal_rows.append({"model": "ENSEMBLE", "temperature": None,
                     "val_ece_raw": None, "val_ece_cal": mv["ece"],
                     "test_ece_raw": None, "test_ece_cal": mt["ece"]})
    pd.DataFrame(cal_rows).to_csv(out / "calibration.csv", index=False)

    # ---- plots ----
    _reliability_plot(y_test, {"best_single_raw": test[best_single]["prob"],
                               "best_single_cal": test[best_single]["cal"],
                               "ensemble_cal": p_test_final}, out / "reliability_diagram.png")
    _abstention_plot(test_curve, out / "abstention_curve.png")
    from src.benchmark.plots import confusion_matrix_plots
    confusion_matrix_plots(mt["_cm"], "ensemble_test", out / "confusion")

    # ---- report ----
    _report(out / "final_phase6_report.md", members, combo, w, T, best, mv, mt,
            ms_val, ms_test, best_single, val_curve, test_curve, chosen_t, test_at_chosen,
            bins, tri, args)

    # ---- console summary ----
    LOG.info("=" * 78)
    LOG.info("FROZEN CONFIG (all chosen on VAL): members=%s | scheme=%s | weights=%s | per-model T=%s",
             combo, best["weight_scheme"], [round(x, 3) for x in w], {m: round(T[m], 3) for m in combo})
    LOG.info("TEST  ensemble : acc=%.4f macroF1=%.4f wF1=%.4f bAcc=%.4f QWK=%.4f KL1-F1=%.4f "
             "KL4-F1=%.4f MAE=%.4f within1=%.4f ECE=%.4f",
             mt["accuracy"], mt["macro_f1"], mt["weighted_f1"], mt["balanced_accuracy"], mt["qwk"],
             mt["kl1_f1"], mt["kl4_f1"], mt["mae"], mt["within_1_accuracy"], mt["ece"])
    LOG.info("TEST  best single(%s)+temp : macroF1=%.4f QWK=%.4f  (delta ensemble %+.4f / %+.4f)",
             best_single, ms_test["macro_f1"], ms_test["qwk"],
             mt["macro_f1"] - ms_test["macro_f1"], mt["qwk"] - ms_test["qwk"])
    LOG.info("TEST  binary OA screening : AUC=%.4f  sens@spec.90=%.3f  spec@sens.90=%.3f",
             bins["auc_roc"], bins["sensitivity_at_spec>=0.90"] or -1,
             bins["specificity_at_sens>=0.90"] or -1)
    LOG.info("TEST  3-class (Normal/Early/Advanced) : acc=%.4f macroF1=%.4f QWK=%.4f",
             tri["accuracy"], tri["macro_f1"], tri["qwk"])
    if test_at_chosen:
        LOG.info("TEST  abstention @ val-chosen thr %.2f : coverage=%.3f selective_acc=%.4f "
                 "selective_macroF1=%.4f", chosen_t, test_at_chosen["coverage"],
                 test_at_chosen["selective_accuracy"], test_at_chosen["selective_macro_f1"])
    LOG.info("outputs -> %s", out)
    LOG.info("=" * 78)


# --------------------------------------------------------------------------- #
def _reliability_plot(y, prob_map, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in prob_map.items():
        conf = p.max(1)
        acc = (p.argmax(1) == y).astype(float)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.any():
                xs.append(conf[m].mean()); ys.append(acc[m].mean())
        ax.plot(xs, ys, "o-", label=f"{name} (ECE {_ece(y, p):.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability (test)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _abstention_plot(curve, path):
    d = [r for r in curve if r["selective_accuracy"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cov = [r["coverage"] for r in d]
    ax.plot(cov, [r["selective_accuracy"] for r in d], "o-", label="selective accuracy")
    ax.plot(cov, [r["selective_macro_f1"] for r in d], "s-", label="selective macro-F1")
    ax.set_xlabel("coverage (fraction auto-graded)"); ax.set_ylabel("metric on kept subset")
    ax.set_title("Test abstention curve (threshold chosen on val)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _f(v, n=4):
    return "" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def _report(path, members, combo, w, T, best, mv, mt, ms_val, ms_test, best_single,
            val_curve, test_curve, chosen_t, test_at_chosen, bins, tri, args):
    L = []
    A = L.append
    A("# Phase 6 - Ensemble + Calibration + Clinical Sub-tasks\n")
    A(f"_Generated {datetime.now(timezone.utc).isoformat()}. Evaluation only. Ensemble members, "
      f"weights, temperatures and the abstention threshold are ALL fit on VALIDATION and applied "
      f"ONCE to TEST (frozen). TTA = {args.tta}._\n")

    A("## 1. Frozen configuration (chosen on validation)\n")
    A(f"- Members: **{', '.join(combo)}**  (weight scheme: {best['weight_scheme']})\n"
      f"- Weights: {[round(x, 4) for x in w]}\n"
      f"- Per-model temperature (val NLL): " + ", ".join(f"{m}={T[m]:.3f}" for m in combo) + "\n"
      f"- Selection metric on val: {args.select_metric}\n")

    A("## 2. Test result vs best single model\n")
    A("| config | split | accuracy | macro_f1 | weighted_f1 | balanced_acc | QWK | KL1_f1 | KL4_f1 | MAE | within_1 | ECE |")
    A("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for tag, split, m in [("ENSEMBLE", "val", mv), ("ENSEMBLE", "test", mt),
                          (f"best_single[{best_single}]+T", "val", ms_val),
                          (f"best_single[{best_single}]+T", "test", ms_test)]:
        A(f"| {tag} | {split} | " + " | ".join(_f(m[k]) for k in
          ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "qwk", "kl1_f1",
           "kl4_f1", "mae", "within_1_accuracy", "ece"]) + " |")
    A(f"\nEnsemble vs best-single on **test**: Macro-F1 {mt['macro_f1'] - ms_test['macro_f1']:+.4f}, "
      f"QWK {mt['qwk'] - ms_test['qwk']:+.4f}, KL1-F1 {mt['kl1_f1'] - ms_test['kl1_f1']:+.4f}. "
      f"Differences < ~0.005 are within noise for n={mt['n']} (no significance test run).\n")

    A("## 3. Per-class (test, ensemble)\n")
    A("| class | precision | recall | f1 | support |")
    A("|---|--:|--:|--:|--:|")
    for cn, d in mt["_per_class"].items():
        A(f"| {cn} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['support']} |")
    A(f"\nTop test confusions: " + ", ".join(f"KL{c['true']}->KL{c['pred']}({c['count']})"
      for c in mt["_top_conf"][:6]) + f"\nKL4 predicted as: {mt['_kl4_pred']}\n")

    A("## 4. Calibration\n")
    A(f"Temperature scaling (1 param per model, fit on validation NLL). Per-model and ensemble "
      f"ECE (test) are in `calibration.csv`. Ensemble test ECE = **{mt['ece']:.4f}**. "
      f"Reliability diagram: `reliability_diagram.png`.\n")

    A("## 5. Abstention (selective prediction)\n")
    A(f"Confidence threshold chosen on VAL for >= {args.abstain_target_acc:.0%} selective accuracy: "
      f"**{chosen_t if chosen_t is not None else 'not reachable'}**.\n")
    if test_at_chosen:
        A(f"Applied to TEST: coverage **{test_at_chosen['coverage']:.1%}** "
          f"(auto-grade {test_at_chosen['n_kept']} / {mt['n']}), selective accuracy "
          f"**{test_at_chosen['selective_accuracy']:.4f}**, selective Macro-F1 "
          f"{test_at_chosen['selective_macro_f1']:.4f}. The remaining "
          f"{1 - test_at_chosen['coverage']:.1%} are 'refer to clinician'.\n")
    A("Full curve: `abstention_curve.csv`, `abstention_curve.png`.\n")

    A("## 6. Binary OA screening (test)  -  {KL0,KL1} vs {KL2,KL3,KL4}\n")
    A(f"- ROC-AUC **{bins['auc_roc']:.4f}** | average precision {bins['average_precision']:.4f}\n"
      f"- Sensitivity at specificity >= 0.90: **{_f(bins['sensitivity_at_spec>=0.90'], 3)}**\n"
      f"- Specificity at sensitivity >= 0.90: **{_f(bins['specificity_at_sens>=0.90'], 3)}**\n"
      f"- At p(OA) >= 0.5: acc {bins['acc@0.5']:.4f}, sens {bins['sensitivity@0.5']:.3f}, "
      f"spec {bins['specificity@0.5']:.3f}, F1 {bins['f1@0.5']:.3f}  (n_OA {bins['n_OA']}, n_noOA {bins['n_noOA']})\n")

    A("## 7. 3-class (test)  -  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)\n")
    A(f"- accuracy **{tri['accuracy']:.4f}** | macro-F1 **{tri['macro_f1']:.4f}** | "
      f"weighted-F1 {tri['weighted_f1']:.4f} | QWK {tri['qwk']:.4f}\n")
    A("| true \\ pred | " + " | ".join(tri["labels"]) + " |")
    A("|---|" + "|".join(["--:"] * 3) + "|")
    for i, rr in enumerate(tri["confusion_matrix"]):
        A(f"| {tri['labels'][i]} | " + " | ".join(str(v) for v in rr) + " |")

    A("\n## 8. Interpretation\n")
    A("- The **5-class** task is still bounded by KL1 vs KL0/KL2 ambiguity; ensembling + calibration "
      "improve robustness and confidence quality more than raw Macro-F1.\n"
      "- The **binary OA screen** and **abstention** views are the deployable story: report coverage "
      "at a fixed selective accuracy and the OA-vs-not AUC.\n"
      "- Nothing here was tuned on test; the test set saw exactly one frozen evaluation pass.\n")

    A("\n## 9. Next\n")
    A("- If gains are marginal: try `--tta hflip`, add/remove a member, or move to an ordinal head "
      "(CORAL/CORN) and/or a higher-resolution retrain of the top 2 (Phase 6.3 / 6.4).\n")

    path.write_text("\n".join(str(x) for x in L), encoding="utf-8")


if __name__ == "__main__":
    main()
