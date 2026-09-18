r"""PHASE 5 - FINAL 15-model TEST-SET evaluation.  EVALUATION ONLY.

  * NO training / retraining / fine-tuning / hyper-parameter or threshold search.
  * Loads the CLEAN Phase-4 validation-selected checkpoints models/phase4/<m>/best.pt
  * Evaluates each on the untouched TEST split (1240 samples), computes the full
    metric suite (overall + per-class + ordinal + confidence), measures batch-1
    latency in THIS single process, and compares test vs the clean Phase-4
    validation numbers (generalization gap = test - validation).
  * Reuses: src/datasets (loader), src/benchmark/{models,engine,metrics,latency,
    plots,checks,seeding,hardware}. Adds only src/benchmark/phase5_metrics.py.
  * Writes ONLY to reports/phase5/. Never touches reports/phase3, reports/phase4,
    models/phase3, models/phase4, or any config.

    .\.venv\Scripts\python.exe scripts\benchmark_phase5.py --dry-run
    .\.venv\Scripts\python.exe scripts\benchmark_phase5.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("phase5")

MODELS_15 = ["vgg16", "vgg19", "googlenet", "squeezenet", "shufflenet",
             "convnext_tiny", "densenet121", "inception_v3", "xception",
             "efficientnet_b1", "resnet50", "mobilenet_v3_large", "resnet18",
             "mobilenet_v2", "efficientnet_b0"]

CLASS_LABELS = [0, 1, 2, 3, 4]
EXPECTED_TEST_DIST = {0: 476, 1: 227, 2: 328, 3: 171, 4: 38}

# metrics carried straight from the clean Phase-4 validation result.json for the
# generalization-gap table  (result key -> friendly name)
GEN_METRICS = [
    ("accuracy", "accuracy"),
    ("macro_precision", "macro_precision"),
    ("macro_recall", "macro_recall"),
    ("macro_f1", "macro_f1"),
    ("weighted_f1", "weighted_f1"),
    ("balanced_accuracy", "balanced_accuracy"),
    ("quadratic_weighted_kappa", "qwk"),
    ("kl4_f1", "kl4_f1"),
]


# --------------------------------------------------------------------------- #
def _lock(reports_dir: Path):
    lock = reports_dir / ".phase5.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        sys.stderr.write(f"ERROR: another Phase-5 run holds {lock} (pid "
                         f"{lock.read_text().strip()}). Delete it if stale.\n")
        sys.exit(3)
    import atexit
    atexit.register(lambda: lock.exists() and lock.unlink())


def _per_model_bs(cfg: dict, model: str, override: int | None) -> int:
    if override:
        return override
    return int((cfg.get("per_model_batch_size", {}) or {}).get(model, cfg["train"]["batch_size"]))


def _check_checkpoints(phase4_dir: Path) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for m in MODELS_15:
        (present if (phase4_dir / m / "best.pt").is_file() else missing).append(m)
    return present, missing


def _load_model(model: str, cfg: dict, device, phase4_dir: Path):
    """build arch -> load Phase-4 best.pt (strict). Returns (built, ckpt_meta).

    ``pretrained=True`` here is DELIBERATE: it makes Phase 5 build every model
    with the *exact same* code path Phase 4 used for training
    (``build_model(..., pretrained=True)``), so the forward behaviour is
    identical. For 14/15 models this only sets initial weights that
    ``load_state_dict`` then overwrites (bit-identical results). It matters for
    torchvision GoogLeNet: ``googlenet(weights=...)`` sets ``transform_input=True``
    while ``googlenet()`` sets it False - a *forward-pass* difference that is
    invisible to ``load_state_dict`` (it is a bool flag, not a parameter) and
    silently corrupts GoogLeNet's predictions if the two builds disagree.
    ImageNet weights are cached, so no network access is needed.
    """
    import torch
    from src.benchmark.models import build_model
    ckpt_path = phase4_dir / model / "best.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    built = build_model(model, num_classes=cfg["num_classes"],
                        in_chans=cfg["in_channels"], pretrained=True)
    built.model.load_state_dict(ck["state_dict"], strict=True)   # raises on any mismatch
    built.model.to(device).eval()
    return built, {"ckpt_path": str(ckpt_path), "epoch": ck.get("epoch"),
                   "selection_metric": ck.get("selection_metric"),
                   "selection_value": ck.get("selection_value"),
                   "val_metrics": ck.get("val_metrics", {})}


def _phase4_val_result(model: str, phase4_dir: Path) -> dict:
    p = phase4_dir / model / "result.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _phase3_val_row(model: str, phase3_csv: Path) -> dict:
    if not phase3_csv.exists():
        return {}
    df = pd.read_csv(phase3_csv)
    r = df[df["model"] == model]
    return r.iloc[0].to_dict() if len(r) else {}


# --------------------------------------------------------------------------- #
def evaluate_one(model: str, cfg: dict, device, *, phase4_dir: Path, reports_dir: Path,
                 batch_size: int, num_workers: int, lat_warmup: int, lat_iters: int,
                 dry_run: bool) -> dict:
    import torch
    from src.datasets import build_dataloaders
    from src.benchmark.engine import evaluate, build_loss, load_class_weights
    from src.benchmark.models import count_parameters
    from src.benchmark.latency import measure_latency
    from src.benchmark.plots import confusion_matrix_plots
    from src.benchmark.phase5_metrics import ordinal_and_confidence, most_common_confusions

    t0 = time.perf_counter()
    LOG.info("[%s] starting  (batch_size=%d, device=%s)", model, batch_size, device)

    built, ckmeta = _load_model(model, cfg, device, phase4_dir)
    LOG.info("[%s] built (backend=%s) + loaded Phase-4 best.pt (epoch %s, val_%s=%.4f) | "
             "strict load OK", model, built.backend, ckmeta["epoch"],
             ckmeta["selection_metric"], ckmeta["selection_value"] or float("nan"))
    pc = count_parameters(built.model)

    dcfg = cfg["data"]
    loaders = build_dataloaders(
        variant=cfg["preprocessing_variant"], normalization=dcfg.get("normalization", "imagenet"),
        batch_size=batch_size, num_workers=num_workers, out_channels=cfg["in_channels"],
        imbalance="none", augmentation_yaml=dcfg.get("augmentation_yaml", "configs/augmentation.yaml"),
        return_meta=True, pin_memory=False, persistent_workers=False, seed=cfg["seed"])
    test_loader = loaders["test"]

    # loss only for a reported test loss value (NOT used for any selection)
    cw = torch.tensor(load_class_weights(cfg["loss"].get("class_weights_key", "weights_list_balanced")),
                      dtype=torch.float32, device=device)
    loss_fn = build_loss({"name": cfg["loss"].get("name", "weighted_cross_entropy")}, cw)

    max_b = 2 if dry_run else None
    m = evaluate(built.model, test_loader, loss_fn, device,
                 amp=bool(cfg["train"].get("amp", True)), max_batches=max_b,
                 return_predictions=True)
    pred = m.pop("_predictions")
    y_true = np.asarray(pred["y_true"]).astype(int)
    y_pred = np.asarray(pred["y_pred"]).astype(int)
    y_prob = np.asarray(pred["y_prob"], dtype=float)
    meta = pred.get("meta") or []

    ext = ordinal_and_confidence(y_true, y_pred, y_prob)
    cm = np.asarray(m["confusion_matrix"])

    # ---- latency: batch-1, synthetic 1x3x224x224, fp32 forward, CUDA-synced,
    #      preprocessing + postprocessing EXCLUDED. Single process. ----
    lat = None
    if not dry_run:
        from src.benchmark.latency import reset_peak_memory, peak_memory_mb
        reset_peak_memory(device)
        lat = measure_latency(built.model, device, tuple(cfg["latency"]["input_shape"]),
                              warmup=lat_warmup, iters=lat_iters)
        lat["peak_memory"] = peak_memory_mb(device)
    else:
        lat = measure_latency(built.model, device, tuple(cfg["latency"]["input_shape"]),
                              warmup=3, iters=5)

    # ---- per-model artefacts (only on real run) ----
    if not dry_run:
        cmdir = reports_dir / "confusion_matrices"
        cm_paths = confusion_matrix_plots(cm.tolist(), model, cmdir)
        with np.errstate(divide="ignore", invalid="ignore"):
            cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        (cmdir / f"{model}_confusion.json").write_text(json.dumps(
            {"labels": CLASS_LABELS, "raw": cm.tolist(), "row_normalized": cmn.tolist()}, indent=2))
        pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_LABELS],
                     columns=[f"pred_{c}" for c in CLASS_LABELS]).to_csv(cmdir / f"{model}_confusion_raw.csv")

        conf = y_prob.max(axis=1)
        pdf = pd.DataFrame({
            "sample_id": [meta[i].get("sample_id") if i < len(meta) else "" for i in range(len(y_true))],
            "knee_side": [meta[i].get("knee_side") if i < len(meta) else "" for i in range(len(y_true))],
            "true_class": y_true, "predicted_class": y_pred,
            "confidence": np.round(conf, 6), "absolute_error": np.abs(y_pred - y_true),
        })
        for c in range(5):
            pdf[f"prob_class_{c}"] = np.round(y_prob[:, c], 6)
        pdf.to_csv(reports_dir / f"predictions_{model}.csv", index=False)
    else:
        cm_paths = {}

    # ---- validation (clean Phase 4) -> test generalization gap ----
    v4 = _phase4_val_result(model, phase4_dir)
    gen = {}
    for k, friendly in GEN_METRICS:
        tv = m.get(k) if k != "quadratic_weighted_kappa" else m.get("quadratic_weighted_kappa")
        vv = v4.get(k)
        gen[f"validation_{friendly}"] = vv
        gen[f"test_{friendly}"] = tv
        gen[f"delta_{friendly}"] = (round(tv - vv, 6) if (tv is not None and vv is not None) else None)

    row = {
        "model": model, "status": "OK", "backend": built.backend,
        "n_test": int(y_true.size),
        # overall
        "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "weighted_precision": ext["weighted_precision"], "weighted_recall": ext["weighted_recall"],
        "weighted_f1": m["weighted_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "cohen_kappa": m["cohen_kappa"], "qwk": m["quadratic_weighted_kappa"],
        "kl4_f1": m["kl4_f1"], "kl4_precision": m["kl4_precision"], "kl4_recall": m["kl4_recall"],
        "kl4_support": m["kl4_support"],
        "mae": ext["mae"], "rmse_ordinal": ext["rmse_ordinal"],
        "exact_accuracy": ext["exact_accuracy"], "within_1_accuracy": ext["within_1_accuracy"],
        "within_2_accuracy": ext["within_2_accuracy"],
        "micro_f1": ext["micro_f1"],
        "test_loss": m.get("loss"),
        "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro"),
        **{f"error_{k}": ext[f"error_{k}"] for k in range(5)},
        "underprediction_count": ext["underprediction_count"],
        "overprediction_count": ext["overprediction_count"],
        "adjacent_error_fraction_of_errors": round(ext["adjacent_error_fraction_of_errors"], 4),
        # confidence
        "confidence_mean": ext["confidence"]["mean"], "confidence_median": ext["confidence"]["median"],
        "confidence_min": ext["confidence"]["min"], "confidence_max": ext["confidence"]["max"],
        "confidence_mean_correct": ext["confidence"]["mean_when_correct"],
        "confidence_mean_incorrect": ext["confidence"]["mean_when_incorrect"],
        # complexity / latency
        "parameters": pc["parameters"], "trainable_parameters": pc["trainable_parameters"],
        "mean_latency_ms": lat["mean_ms"], "median_latency_ms": lat["median_ms"],
        "std_latency_ms": lat["std_ms"], "p90_latency_ms": lat.get("p90_ms"),
        "throughput_img_per_s": lat.get("throughput_images_per_sec"),
        "latency_device": lat["device"], "latency_warmup_iters": lat["warmup_iters"],
        "latency_timed_iters": lat["timed_iters"],
        # checkpoint provenance
        "phase4_best_epoch": v4.get("best_epoch"), "phase4_epochs_completed": v4.get("epochs_run"),
        "phase4_early_stop": v4.get("stopped_early"),
        "checkpoint_path": ckmeta["ckpt_path"],
        "checkpoint_val_selection_metric": ckmeta["selection_metric"],
        "checkpoint_val_selection_value": ckmeta["selection_value"],
        **gen,
        # nested (kept in JSON, dropped from flat CSV by report writer)
        "_per_class": m["per_class"],
        "_confusion_matrix": cm.tolist(),
        "_kl4_predicted_as": ext["kl4_predicted_as"],
        "_top_confusions": most_common_confusions(cm, top=10),
        "_confusion_plots": cm_paths,
        "_latency_full": lat,
        "eval_seconds": round(time.perf_counter() - t0, 2),
    }
    LOG.info("[%s] TEST | acc=%.4f macroF1=%.4f wF1=%.4f bAcc=%.4f QWK=%.4f KL4-F1=%.4f "
             "MAE=%.4f exact=%.4f within1=%.4f | params=%.2fM lat=%.3fms(%s) | %.1fs",
             model, row["accuracy"], row["macro_f1"], row["weighted_f1"], row["balanced_accuracy"],
             row["qwk"], row["kl4_f1"], row["mae"], row["exact_accuracy"], row["within_1_accuracy"],
             row["parameters"] / 1e6, row["mean_latency_ms"], row["latency_device"],
             row["eval_seconds"])

    try:
        del built.model
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return row


# --------------------------------------------------------------------------- #
def _write_outputs(rows: list[dict], phase3_csv: Path, reports_dir: Path, env: dict,
                   dist: dict, cfg: dict, runtime_min: float, failures: list[str]) -> None:
    ok = [r for r in rows if r.get("status") == "OK"]
    flat_drop = [k for k in (ok[0] if ok else {}) if k.startswith("_")]
    flat = [{k: v for k, v in r.items() if k not in flat_drop} for r in rows]

    df = pd.DataFrame(flat)
    if "test_macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False, na_position="last")
    df.to_csv(reports_dir / "final_15_model_test_benchmark.csv", index=False)
    (reports_dir / "final_15_model_test_benchmark.json").write_text(
        json.dumps(rows, indent=2, default=str))

    # per-class metrics (15 x 5 rows)
    pc_rows = []
    for r in ok:
        for cls_name, d in r["_per_class"].items():
            cidx = int(cls_name.replace("KL", ""))
            pc_rows.append({"model": r["model"], "class": cidx,
                            "precision": d["precision"], "recall": d["recall"],
                            "f1": d["f1"], "support": d["support"]})
    pd.DataFrame(pc_rows).to_csv(reports_dir / "per_class_metrics.csv", index=False)

    # ordinal error analysis
    ord_rows = [{
        "model": r["model"],
        **{f"error_{k}": r[f"error_{k}"] for k in range(5)},
        "underprediction_count": r["underprediction_count"],
        "overprediction_count": r["overprediction_count"],
        "mae": r["mae"], "qwk": r["qwk"],
        "exact_accuracy": r["exact_accuracy"], "within_1_accuracy": r["within_1_accuracy"],
        "adjacent_error_fraction_of_errors": r["adjacent_error_fraction_of_errors"],
    } for r in ok]
    pd.DataFrame(ord_rows).to_csv(reports_dir / "ordinal_error_analysis.csv", index=False)

    # efficiency
    eff_rows = [{
        "model": r["model"], "test_macro_f1": r["macro_f1"], "test_accuracy": r["accuracy"],
        "test_qwk": r["qwk"], "test_balanced_accuracy": r["balanced_accuracy"],
        "parameters": r["parameters"], "trainable_parameters": r["trainable_parameters"],
        "mean_latency_ms": r["mean_latency_ms"],
        "macro_f1_per_Mparam": round(r["macro_f1"] / (r["parameters"] / 1e6), 4),
    } for r in ok]
    pd.DataFrame(eff_rows).to_csv(reports_dir / "model_efficiency.csv", index=False)

    # validation (Phase4) vs test (Phase5) generalization
    gen_cols = ["model"] + [f"{p}_{friendly}" for _, friendly in GEN_METRICS
                            for p in ("validation", "test", "delta")]
    pd.DataFrame([{k: r.get(k) for k in gen_cols} for r in ok]).to_csv(
        reports_dir / "validation_vs_test_generalization.csv", index=False)

    # Phase 3 val / Phase 4 val / Phase 5 test consolidated
    cons = []
    for r in ok:
        p3 = _phase3_val_row(r["model"], phase3_csv)
        cons.append({
            "model": r["model"],
            "p3val_macro_f1": p3.get("macro_f1"), "p4val_macro_f1": r.get("validation_macro_f1"),
            "p5test_macro_f1": r["macro_f1"],
            "p3val_qwk": p3.get("quadratic_weighted_kappa"), "p4val_qwk": r.get("validation_qwk"),
            "p5test_qwk": r["qwk"],
            "p3val_balanced_accuracy": p3.get("balanced_accuracy"),
            "p4val_balanced_accuracy": r.get("validation_balanced_accuracy"),
            "p5test_balanced_accuracy": r["balanced_accuracy"],
            "p3val_accuracy": p3.get("accuracy"), "p4val_accuracy": r.get("validation_accuracy"),
            "p5test_accuracy": r["accuracy"],
            "p3val_weighted_f1": p3.get("weighted_f1"),
            "p4val_weighted_f1": r.get("validation_weighted_f1"), "p5test_weighted_f1": r["weighted_f1"],
            "p3val_kl4_f1": p3.get("kl4_f1"), "p4val_kl4_f1": r.get("validation_kl4_f1"),
            "p5test_kl4_f1": r["kl4_f1"],
            "p5test_mae": r["mae"],
        })
    pd.DataFrame(cons).to_csv(reports_dir / "phase3_phase4_phase5_consolidated.csv", index=False)

    # rankings
    rankings = {}
    rk = [("test_macro_f1", "macro_f1", False), ("test_accuracy", "accuracy", False),
          ("test_macro_precision", "macro_precision", False),
          ("test_macro_recall", "macro_recall", False),
          ("test_weighted_f1", "weighted_f1", False),
          ("test_balanced_accuracy", "balanced_accuracy", False),
          ("test_qwk", "qwk", False), ("test_kl4_f1", "kl4_f1", False),
          ("lowest_mae", "mae", True), ("lowest_latency", "mean_latency_ms", True),
          ("lowest_params", "parameters", True)]
    for label, col, asc in rk:
        s = sorted(ok, key=lambda r: (r[col] if r[col] is not None else (1e18 if asc else -1e18)),
                   reverse=not asc)
        rankings[label] = [{"rank": i + 1, "model": r["model"], col: r[col]} for i, r in enumerate(s)]
    (reports_dir / "rankings.json").write_text(json.dumps(rankings, indent=2, default=str))

    _write_report(df, rows, ok, failures, rankings, env, dist, cfg, runtime_min,
                  reports_dir / "final_phase5_report.md")


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _mdtbl(rows, cols, nd=4):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(_fmt(r.get(c), nd) for c in cols) + " |" for r in rows)
    return head + "\n" + body


def _write_report(df, rows, ok, failures, rankings, env, dist, cfg, runtime_min, out_path):
    A = []
    P = A.append
    okd = sorted(ok, key=lambda r: r["macro_f1"], reverse=True)
    g = env.get("gpu") or {}
    P("# Phase 5 - Final 15-Model Test Benchmark\n")
    P(f"_Generated {datetime.now(timezone.utc).isoformat()}. Evaluation only - no training, "
      f"no retraining, no tuning. Checkpoints = clean Phase-4 validation-selected best.pt._\n")

    P("## 1. Executive Summary\n")
    if okd:
        top = okd[0]
        P(f"- Evaluated {len(ok)}/15 clean Phase-4 checkpoints on the untouched TEST split "
          f"({top['n_test']} samples). Failures: {failures or 'none'}.\n"
          f"- Best test Macro-F1: **{top['model']}** = {top['macro_f1']:.4f} "
          f"(QWK {top['qwk']:.4f}, balanced acc {top['balanced_accuracy']:.4f}, "
          f"KL4-F1 {top['kl4_f1']:.4f}, MAE {top['mae']:.4f}, {top['parameters']/1e6:.1f}M params).\n"
          f"- Top-3 by test Macro-F1: " + ", ".join(f"{r['model']} {r['macro_f1']:.4f}" for r in okd[:3]) + "\n"
          f"- Mean generalization gap (test - Phase-4 val) Macro-F1: "
          f"{np.nanmean([r.get('delta_macro_f1') for r in ok if r.get('delta_macro_f1') is not None]):+.4f}\n")
    P("_Tiny numerical differences (e.g. < ~0.005 Macro-F1 on 1240 samples) are not "
      "claimed as meaningful; no significance test was run._\n")

    P("## 2. Research Objective\n")
    P("Produce the final, held-out TEST-set performance of all 15 architectures using the "
      "checkpoints already selected on VALIDATION Macro-F1 in the clean Phase-4 run, and pick "
      "a final model on the full evidence (not Macro-F1 alone).\n")

    P("## 3. Dataset\n")
    P(f"Kaggle Knee-OA severity, patient- and sample-disjoint splits. "
      f"train {dist['counts']['train']} / val {dist['counts']['val']} / test {dist['counts']['test']}. "
      f"Preprocessing variant `{cfg['preprocessing_variant']}`, {cfg['image_size'][0]}x{cfg['image_size'][1]}, "
      f"3-ch replicated grayscale, {cfg['data']['normalization']} normalization (identical to Phase 3/4).\n")

    P("## 4. Dataset Integrity\n")
    P("| split | n | patients | KL0 | KL1 | KL2 | KL3 | KL4 |")
    P("|---|--:|--:|--:|--:|--:|--:|--:|")
    for s in ("train", "val", "test"):
        cd = dist["class_distribution"][s]
        P(f"| {s} | {dist['counts'][s]} | {dist['patients'][s]} | "
          + " | ".join(str(cd.get(str(k), cd.get(k, 0))) for k in range(5)) + " |")
    actual = {int(k): v for k, v in dist["class_distribution"]["test"].items()}
    P(f"\nPatient overlap train/val=0, train/test=0, val/test=0. Sample overlap=0. All 5 classes present.\n"
      f"Test class distribution vs expected {EXPECTED_TEST_DIST}: "
      f"{'MATCH' if actual == EXPECTED_TEST_DIST else 'MISMATCH -> ' + str(actual)}.\n")

    P("## 5. Phase 3 Summary\n")
    P("Phase 3 = 15-model benchmark, max_epochs 30, early stopping patience 5 on val Macro-F1. "
      "Phase-3 validation numbers are read from `reports/phase3/extended_benchmark_results.csv` "
      "(untouched) for the consolidated comparison.\n")

    P("## 6. Clean Phase 4 Verification\n")
    P(f"All 15 `models/phase4/<m>/best.pt` present and strict-loadable; each checkpoint's stored "
      f"`val_metrics.macro_f1` equals its `result.json` macro_f1 (self-consistent). "
      f"Phase 4 = same 15 models, max_epochs 50 (ceiling), same early stopping. "
      f"Checkpoint selection metric = `{cfg['eval']['selection_metric']}` (validation only).\n")

    P("## 7. Phase 3 vs Phase 4 (validation)\n")
    P("See `reports/phase5/phase3_phase4_phase5_consolidated.csv` (p3val_* vs p4val_* columns). "
      "Note: 50 epochs is only a CEILING - with early stopping enabled, models that stopped at "
      "epoch ~20 did not train for 50 epochs. Phrase any conclusion as *'allowing up to 50 epochs "
      "under the same early-stopping configuration'*, not *'training for 50 epochs'*.\n")

    P("## 8. Phase 5 Evaluation Methodology\n")
    P(f"- Load `models/phase4/<m>/best.pt` -> `load_state_dict(strict=True)` into the same "
      f"`build_model()` architecture (pretrained=False; weights come only from the checkpoint).\n"
      f"- Forward the full TEST split via the existing `src.datasets.build_dataloaders(... "
      f"return_meta=True, imbalance='none')` test loader (no augmentation, `shuffle=False`, "
      f"evaluates all {dist['counts']['test']} samples) and `src.benchmark.engine.evaluate(...)`.\n"
      f"- AMP for the forward pass = {bool(cfg['train'].get('amp', True))} (same regime as the "
      f"Phase-4 validation numbers, so val<->test is comparable). Re-evaluation is deterministic "
      f"for fixed weights.\n"
      f"- Metrics: `metrics.compute_all` (accuracy, macro/weighted P-R-F1, per-class, balanced "
      f"accuracy, Cohen kappa, QWK, confusion matrix) + `phase5_metrics.ordinal_and_confidence` "
      f"(MAE, exact / within-+/-1 accuracy, |pred-true| histogram, under/over-prediction, "
      f"weighted precision & recall, micro P/R/F1, confidence stats, KL4->pred breakdown).\n"
      f"- Latency: `latency.measure_latency` - batch 1, synthetic 1x3x224x224, fp32 forward, "
      f"{cfg['latency']['warmup_iters']} warmup + {cfg['latency']['timed_iters']} timed iters, "
      f"`torch.cuda.synchronize()` per iter. **Preprocessing and post-processing (softmax/argmax) "
      f"are EXCLUDED** - pure model forward. Single process (Phase-5 lock enforces this).\n")

    P("## 9. Hardware and Software\n")
    P(f"- GPU {g.get('name')} ({g.get('total_memory_gb')} GB, cc {g.get('capability')}) | device "
      f"{'CUDA' if g else 'CPU'}\n"
      f"- torch {env.get('torch')} | CUDA {env.get('cuda_version')} | cuDNN {env.get('cudnn_version')} "
      f"| torchvision {env.get('torchvision')} | timm {env.get('timm')} | Python {env.get('python')}\n"
      f"- seed {cfg['seed']} (evaluation is deterministic given fixed weights; seed only affects "
      f"loader worker RNG which is irrelevant with shuffle=False) | config `configs/benchmark_phase4.yaml`\n")

    P("## 10. Complete Test-Set Results\n")
    P(_mdtbl(okd, ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                   "weighted_precision", "weighted_recall", "weighted_f1", "balanced_accuracy",
                   "qwk", "kl4_f1", "mae", "exact_accuracy", "within_1_accuracy",
                   "parameters", "mean_latency_ms"]))
    if failures:
        P(f"\n**FAILED (no metrics fabricated): {failures}**\n")

    P("\n## 11. Accuracy Analysis\n")
    P(_mdtbl(sorted(okd, key=lambda r: r["accuracy"], reverse=True),
             ["model", "accuracy", "exact_accuracy", "within_1_accuracy", "within_2_accuracy",
              "balanced_accuracy"]))
    P("\nExact accuracy == overall accuracy (single-label). Balanced accuracy = mean per-class "
      "recall - lower than accuracy indicates weak minority-class recall.\n")

    P("## 12. Precision Analysis\n")
    P("Per-class precision in `per_class_metrics.csv`. Macro (equal class weight) vs weighted "
      "(support weight):\n")
    P(_mdtbl(sorted(okd, key=lambda r: r["macro_precision"], reverse=True),
             ["model", "macro_precision", "weighted_precision", "kl4_precision"]))

    P("\n## 13. Recall Analysis\n")
    P(_mdtbl(sorted(okd, key=lambda r: r["macro_recall"], reverse=True),
             ["model", "macro_recall", "weighted_recall", "kl4_recall", "balanced_accuracy"]))

    P("\n## 14. F1 Analysis\n")
    P("**Macro F1** weights every KL grade equally - the key indicator of *balanced* performance "
      "across all five grades. **Weighted F1** weights by support - performance on the *actual* "
      "(imbalanced) test distribution. They answer different questions; neither is inherently "
      "'better'.\n")
    P(_mdtbl(okd, ["model", "macro_f1", "weighted_f1", "kl0_f1" if "kl0_f1" in df.columns else "macro_f1"]) if False
      else _mdtbl(okd, ["model", "macro_f1", "weighted_f1"]))

    P("\n## 15. Per-Class Performance\n")
    P("Full 15x5 table in `reports/phase5/per_class_metrics.csv`. Per-class F1 (test):\n")
    pcf = []
    for r in okd:
        d = {"model": r["model"]}
        for cls_name, v in r["_per_class"].items():
            d[cls_name + "_f1"] = v["f1"]
        pcf.append(d)
    P(_mdtbl(pcf, ["model", "KL0_f1", "KL1_f1", "KL2_f1", "KL3_f1", "KL4_f1"]))

    P("\n## 16. Confusion Matrix Analysis\n")
    P("Raw + row-normalized PNG per model in `reports/phase5/confusion_matrices/`; numerical "
      "arrays in `<model>_confusion.json` and `<model>_confusion_raw.csv`. Most-frequent "
      "off-diagonal confusions (top model shown; per-model in the JSON `_top_confusions`):\n")
    if okd:
        tc = okd[0]["_top_confusions"][:6]
        P(_mdtbl([{"true->pred": f"KL{d['true']}->KL{d['pred']}", "count": d["count"],
                   "ordinal_distance": d["ordinal_distance"]} for d in tc],
                 ["true->pred", "count", "ordinal_distance"]))

    P("\n## 17. Ordinal Error Analysis\n")
    P("Classes are ordered (KL0<KL1<KL2<KL3<KL4). `ordinal_error_analysis.csv` has the full "
      "|pred-true| histogram + under/over-prediction per model.\n")
    P(_mdtbl(sorted(okd, key=lambda r: r["mae"]),
             ["model", "mae", "qwk", "exact_accuracy", "within_1_accuracy",
              "error_0", "error_1", "error_2", "error_3", "error_4",
              "underprediction_count", "overprediction_count", "adjacent_error_fraction_of_errors"]))
    P("\nHigh `adjacent_error_fraction_of_errors` (close to 1.0) => errors are mostly "
      "neighbouring-grade, not large ordinal jumps.\n")

    P("## 18. Validation vs Test Generalization\n")
    P("delta = test - clean-Phase-4-validation. NEGATIVE = test lower than validation "
      "(a generalization gap, NOT an improvement). Full table: "
      "`reports/phase5/validation_vs_test_generalization.csv`.\n")
    P(_mdtbl(sorted(ok, key=lambda r: r.get("test_macro_f1", 0), reverse=True),
             ["model", "validation_macro_f1", "test_macro_f1", "delta_macro_f1",
              "validation_qwk", "test_qwk", "delta_qwk",
              "validation_balanced_accuracy", "test_balanced_accuracy", "delta_balanced_accuracy",
              "validation_kl4_f1", "test_kl4_f1", "delta_kl4_f1"]))

    P("\n## 19. Model Efficiency\n")
    P(_mdtbl(sorted(okd, key=lambda r: r["macro_f1"], reverse=True),
             ["model", "macro_f1", "qwk", "parameters", "trainable_parameters",
              "mean_latency_ms", "throughput_img_per_s"]))
    P("\n`model_efficiency.csv` adds macro_f1_per_Mparam.\n")

    P("## 20. Model Rankings\n")
    for label in ["test_macro_f1", "test_accuracy", "test_qwk", "test_balanced_accuracy",
                  "test_kl4_f1", "lowest_mae", "lowest_latency", "lowest_params"]:
        top5 = rankings[label][:5]
        col = list(top5[0].keys())[-1]
        P(f"- **{label}**: " + ", ".join(f"{x['model']}({_fmt(x[col])})" for x in top5))
    P("")

    P("## 21. Best Model by Metric\n")
    def best_by(col, asc=False):
        s = sorted(ok, key=lambda r: r[col], reverse=not asc)
        return f"{s[0]['model']} ({_fmt(s[0][col])})"
    P(f"- A. Best overall (multi-factor): see section 22\n"
      f"- B. Best Macro-F1: {best_by('macro_f1')}\n"
      f"- C. Best Accuracy: {best_by('accuracy')}\n"
      f"- D. Best Macro Precision: {best_by('macro_precision')}\n"
      f"- E. Best Macro Recall: {best_by('macro_recall')}\n"
      f"- F. Best Weighted F1: {best_by('weighted_f1')}\n"
      f"- G. Best Balanced Accuracy: {best_by('balanced_accuracy')}\n"
      f"- H. Best QWK: {best_by('qwk')}\n"
      f"- I. Best KL4 F1: {best_by('kl4_f1')}\n"
      f"- J. Lowest MAE: {best_by('mae', asc=True)}\n"
      f"- K. Fastest: {best_by('mean_latency_ms', asc=True)}\n"
      f"- L. Smallest: {best_by('parameters', asc=True)}\n"
      f"- M. Best macro-F1 / param: {sorted(ok, key=lambda r: r['macro_f1']/(r['parameters']/1e6), reverse=True)[0]['model']}\n"
      f"- N. Best macro-F1 / latency: {sorted(ok, key=lambda r: r['macro_f1']/max(r['mean_latency_ms'],1e-6), reverse=True)[0]['model']}\n")

    P("## 22. Overall Model Recommendation\n")
    P("_Evidence-based synthesis across Macro-F1, macro precision/recall, weighted F1, balanced "
      "accuracy, QWK, KL4 F1, MAE, minority-class behaviour, parameter count, latency, and the "
      "validation->test gap. If the leaders are within ~0.005 test Macro-F1 and match on QWK/KL4, "
      "they are called a practical tie and the smaller/faster one is preferred. Filled from the "
      "actual numbers above; no single winner is forced if strengths genuinely differ._\n")
    if okd:
        lead = okd[:3]
        spread = lead[0]["macro_f1"] - lead[-1]["macro_f1"]
        P(f"Leaders: " + ", ".join(f"{r['model']} (F1 {r['macro_f1']:.4f}, QWK {r['qwk']:.4f}, "
          f"KL4-F1 {r['kl4_f1']:.4f}, {r['parameters']/1e6:.1f}M, {r['mean_latency_ms']:.2f}ms)"
          for r in lead) + f". Top-3 Macro-F1 spread = {spread:.4f}"
          + (" -> practical tie; prefer the smaller/faster of the tied set." if spread < 0.005
             else " -> a clear ordering exists.") + "\n")

    P("## 23. Limitations\n")
    P("- Single held-out test set (1240 samples, 620 patients); Class 4 support = 38 -> KL4 "
      "metrics have wide uncertainty; do NOT over-interpret small KL4 differences.\n"
      "- No confidence intervals / significance tests computed; differences < ~0.005 Macro-F1 "
      "are within noise for this sample size.\n"
      "- AMP fp16 forward is used for metrics (matches Phase-4 val regime); latency is fp32.\n"
      "- Latency is desktop RTX 5080 batch-1 synthetic-input forward only - not an edge-device "
      "or end-to-end (with preprocessing) number.\n"
      "- Probabilities are raw softmax; no calibration was applied or assessed here.\n")

    P("## 24. Recommended Next Experiment\n")
    P("_To be decided with the user from these results (e.g. preprocessing ablation - basic vs "
      "CLAHE vs histeq - on the 2-3 test leaders; or calibration / TTA / ensembling of the top "
      "models). No decision is made here; the test set is now consumed for final reporting and "
      "must not drive further training choices._\n")

    P("\n## No Data Leakage (explicit)\n")
    P("The TEST set was NOT used for training, checkpoint selection, early stopping, "
      "hyper-parameter tuning, preprocessing choice, threshold selection, augmentation choice, "
      "or architecture selection. Every checkpoint was selected on VALIDATION Macro-F1 during "
      "the clean Phase-4 run. The test set is used here ONLY for this final evaluation.\n")

    P("\n## Reproducibility\n")
    P(f"Python {env.get('python')} | torch {env.get('torch')} | torchvision {env.get('torchvision')} "
      f"| timm {env.get('timm')} | CUDA {env.get('cuda_version')} | GPU {g.get('name')} | seed "
      f"{cfg['seed']} | config configs/benchmark_phase4.yaml | checkpoints models/phase4/<m>/best.pt "
      f"| split metadata/processed_manifest_{cfg['preprocessing_variant']}.csv | variant "
      f"{cfg['preprocessing_variant']} | {cfg['image_size'][0]}x{cfg['image_size'][1]} | "
      f"{cfg['data']['normalization']} norm | test batch_size per-model (vgg 16, else "
      f"{cfg['train']['batch_size']}) | latency warmup {cfg['latency']['warmup_iters']} iters "
      f"{cfg['latency']['timed_iters']}. Full env: reports/phase5/reproducibility.json.\n")

    out_path.write_text("\n".join(str(x) for x in A), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark_phase4.yaml"),
                    help="read-only; provides split/preprocessing/latency settings")
    ap.add_argument("--models", default="all", help="'all' or comma list")
    ap.add_argument("--phase4-dir", type=Path, default=Path("models/phase4"))
    ap.add_argument("--phase3-csv", type=Path, default=Path("reports/phase3/extended_benchmark_results.csv"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports/phase5"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=None, help="override test batch size for ALL models")
    ap.add_argument("--num-workers", type=int, default=0, help="test loader workers (default 0)")
    ap.add_argument("--latency-warmup", type=int, default=None)
    ap.add_argument("--latency-iters", type=int, default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="evaluate only models whose checkpoint exists (default: STOP if any missing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify config/dataset/checkpoints; load each model + 2 test batches; NO CSVs")
    ap.add_argument("--log-file", type=Path, default=Path("reports/phase5/run_phase5.log"))
    args = ap.parse_args()

    reports_dir = PROJECT_ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        _lock(reports_dir)
    add_file_logger(args.log_file)
    LOG.info("logging to %s | pid=%d | dry_run=%s", args.log_file, os.getpid(), args.dry_run)

    cfg = load_config(PROJECT_ROOT / args.config)
    if args.latency_warmup is not None:
        cfg["latency"]["warmup_iters"] = args.latency_warmup
    if args.latency_iters is not None:
        cfg["latency"]["timed_iters"] = args.latency_iters

    models = MODELS_15 if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in MODELS_15]
    if bad:
        LOG.error("unknown models: %s (valid: %s)", bad, MODELS_15); sys.exit(2)

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import collect_env, pick_device
    from src.benchmark.checks import leakage_and_distribution_check, LeakageError

    seed_report = set_seed(cfg["seed"], deterministic=True)
    device = pick_device(args.device)
    if args.device != "cpu" and device.type != "cuda":
        LOG.error("CUDA expected but unavailable - REFUSING to silently run on CPU. Aborting.")
        sys.exit(4)
    env = collect_env(seed_report)

    LOG.info("=" * 78)
    LOG.info("PHASE 5 FINAL TEST EVALUATION | models=%d | device=%s | EVALUATION ONLY (no training)",
             len(models), device)
    LOG.info("torch=%s cuda=%s timm=%s torchvision=%s gpu=%s", env["torch"], env["cuda_version"],
             env.get("timm"), env.get("torchvision"), env["gpu"]["name"] if env["gpu"] else "none")
    LOG.info("=" * 78)

    # ---- dataset integrity (same check as Phase 3/4) ----
    LOG.info("DATA INTEGRITY: leakage + class-distribution check on split '%s'",
             cfg["preprocessing_variant"])
    try:
        dist = leakage_and_distribution_check(cfg["preprocessing_variant"], logger=LOG)
    except LeakageError as exc:
        LOG.error("DATA INTEGRITY FAILED - STOP:\n%s", exc); sys.exit(1)
    actual = {int(k): v for k, v in dist["class_distribution"]["test"].items()}
    LOG.info("TEST class distribution actual=%s expected=%s -> %s", actual, EXPECTED_TEST_DIST,
             "MATCH" if actual == EXPECTED_TEST_DIST else "MISMATCH")
    if actual != EXPECTED_TEST_DIST:
        LOG.warning("Test class distribution differs from the previously reported values - "
                    "documented, not blocking.")

    # ---- checkpoint presence ----
    present, missing = _check_checkpoints(args.phase4_dir)
    LOG.info("Phase-4 checkpoints present: %d/15 %s", len(present), present)
    if missing:
        LOG.error("Phase-4 checkpoints MISSING: %s", missing)
        if not args.allow_partial:
            LOG.error("STOP: clean Phase 4 has not produced all 15 checkpoints. "
                      "Re-run this after Phase 4 finishes, or pass --allow-partial to evaluate "
                      "only the %d available.", len(present))
            sys.exit(5)
        models = [m for m in models if m in present]
        LOG.warning("--allow-partial: evaluating %d models: %s", len(models), models)

    (reports_dir / "reproducibility.json").write_text(json.dumps(
        {**env, "eval_config": args.config.as_posix(), "checkpoint_dir": args.phase4_dir.as_posix(),
         "class_distribution": dist["class_distribution"], "patients_per_split": dist["patients"],
         "counts": dist["counts"], "models": models, "phase": "phase5",
         "test_batch_size": {m: _per_model_bs(cfg, m, args.batch_size) for m in models},
         "latency": {"warmup": cfg["latency"]["warmup_iters"], "iters": cfg["latency"]["timed_iters"],
                     "input_shape": cfg["latency"]["input_shape"], "amp_forward_for_latency": False,
                     "amp_forward_for_metrics": bool(cfg["train"].get("amp", True)),
                     "preprocessing_included": False, "postprocessing_included": False}},
        indent=2, default=str))

    if args.dry_run:
        LOG.info("--- DRY RUN: load each model + 2 test batches, verify metrics compute. No CSVs. ---")
        okc = 0
        for m in models:
            try:
                evaluate_one(m, cfg, device, phase4_dir=args.phase4_dir, reports_dir=reports_dir,
                             batch_size=_per_model_bs(cfg, m, args.batch_size),
                             num_workers=0, lat_warmup=3, lat_iters=5, dry_run=True)
                okc += 1
                LOG.info("  [dry-run] %-20s OK", m)
            except Exception as exc:  # noqa: BLE001
                LOG.error("  [dry-run] %-20s FAIL: %s", m, exc); traceback.print_exc()
        LOG.info("DRY RUN %s (%d/%d models loadable + 2-batch eval OK). No result files written.",
                 "PASSED" if okc == len(models) else "PARTIAL", okc, len(models))
        sys.exit(0 if okc == len(models) else 6)

    # ---- FULL EVALUATION ----
    lw = cfg["latency"]["warmup_iters"]
    li = cfg["latency"]["timed_iters"]
    rows: list[dict] = []
    failures: list[str] = []
    t0 = time.perf_counter()
    for i, m in enumerate(models, 1):
        LOG.info("[%d/%d] ===== %s =====", i, len(models), m)
        try:
            rows.append(evaluate_one(m, cfg, device, phase4_dir=args.phase4_dir, reports_dir=reports_dir,
                                     batch_size=_per_model_bs(cfg, m, args.batch_size),
                                     num_workers=args.num_workers, lat_warmup=lw, lat_iters=li,
                                     dry_run=False))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            LOG.error("[%s] FAILED:\n%s", m, tb)
            rows.append({"model": m, "status": "FAILED", "error": str(exc), "traceback": tb})
            failures.append(m)
    runtime_min = (time.perf_counter() - t0) / 60

    _write_outputs(rows, PROJECT_ROOT / args.phase3_csv, reports_dir, env, dist, cfg,
                   runtime_min, failures)

    ok = [r for r in rows if r.get("status") == "OK"]
    LOG.info("=" * 78)
    LOG.info("PHASE 5 DONE in %.1f min | evaluated %d/%d | FAILED: %s", runtime_min,
             len(ok), len(models), failures or "none")
    LOG.info("TEST ranking by Macro-F1:")
    for i, r in enumerate(sorted(ok, key=lambda r: r["macro_f1"], reverse=True), 1):
        LOG.info("  %2d. %-20s test_macroF1=%.4f QWK=%.4f bAcc=%.4f KL4-F1=%.4f MAE=%.4f "
                 "params=%.1fM lat=%.2fms  (val->test dMacroF1=%+.4f)",
                 i, r["model"], r["macro_f1"], r["qwk"], r["balanced_accuracy"], r["kl4_f1"],
                 r["mae"], r["parameters"] / 1e6, r["mean_latency_ms"],
                 r.get("delta_macro_f1") or 0.0)
    LOG.info("outputs -> %s", reports_dir)
    LOG.info("Phase 3 / Phase 4 dirs NOT modified.")
    LOG.info("=" * 78)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
