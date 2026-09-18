"""PHASE 7 - MobileNetV2 edge-optimization runner.

Stages (run in order; each appends to reports/phase7/EXPERIMENT_REGISTRY.csv):

    matrix     M0 baseline + one-dimension-changed runs (+ CORN) per candidate
    combine    stack the dimensions that beat M0 on VAL  ->  M7_stack / M8_stack_ema
    multiseed  seeds 123 & 3407 on the global top-2 configs (seed 42 already run)

Usage:
    python scripts/benchmark_phase7.py --stage matrix    --log-file reports/phase7/p7_matrix.log
    python scripts/benchmark_phase7.py --stage combine   --log-file reports/phase7/p7_combine.log
    python scripts/benchmark_phase7.py --stage multiseed --log-file reports/phase7/p7_seed.log
    python scripts/benchmark_phase7.py --stage matrix --dry-run

The TEST split is never loaded here.  scripts/phase7_final_eval.py does that once,
after the winner is frozen.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402
from src.datasets import build_dataloaders  # noqa: E402
from src.benchmark import complexity as cx  # noqa: E402
from src.benchmark import latency as lat  # noqa: E402
from src.benchmark.engine import build_loss, evaluate, fit, load_class_weights  # noqa: E402
from src.benchmark.models import build_model, count_parameters  # noqa: E402
from src.benchmark import experiments_phase7 as EXP  # noqa: E402

LOG = get_logger("p7")
CFG_PATH = PROJECT_ROOT / "configs" / "benchmark_phase7.yaml"
REG_CSV = PROJECT_ROOT / "reports" / "phase7" / "EXPERIMENT_REGISTRY.csv"
LOCK = PROJECT_ROOT / "reports" / "phase7" / ".phase7.lock"


# --------------------------------------------------------------------------- #
def _acquire_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        LOG.error("lock %s exists - another Phase-7 run is active. Abort.", LOCK)
        sys.exit(2)
    os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
    os.close(fd)
    atexit.register(lambda: LOCK.exists() and LOCK.unlink())


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    p_oa = y_prob[:, 2:].sum(axis=1)
    y_bin = (yt >= 2).astype(int)
    bin_acc = float(((p_oa >= 0.5).astype(int) == y_bin).mean())
    try:
        bin_auc = float(roc_auc_score(y_bin, p_oa)) if len(np.unique(y_bin)) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        bin_auc = float("nan")
    m3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    t3 = np.vectorize(m3.get)(yt)
    p3 = np.vectorize(m3.get)(yp)
    acc3 = float((t3 == p3).mean())
    mae = float(np.abs(yt - yp).mean())
    within1 = float((np.abs(yt - yp) <= 1).mean())
    return {"val_binary_oa_acc": round(bin_acc, 4), "val_binary_oa_auc": round(bin_auc, 4),
            "val_3class_acc": round(acc3, 4), "val_mae": round(mae, 4),
            "val_within1": round(within1, 4)}


def _reg_seen() -> set[str]:
    if not REG_CSV.exists():
        return set()
    with open(REG_CSV, newline="") as fh:
        return {r["experiment_id"] for r in csv.DictReader(fh)}


def _reg_append(row: dict):
    REG_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not REG_CSV.exists()
    with open(REG_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXP.REGISTRY_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in EXP.REGISTRY_COLUMNS})


def _reg_rows() -> list[dict]:
    if not REG_CSV.exists():
        return []
    with open(REG_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
def run_one(spec: dict, base_cfg: dict, device, *, dry: bool) -> dict | None:
    import torch

    eid = spec["experiment_id"]
    cfg = _deep_merge(base_cfg, spec["overrides"])
    model_name = spec["model"]
    variant = cfg["preprocessing_variant"]
    dcfg, tcfg = cfg["data"], cfg["train"]
    is_corn = cfg["loss"].get("name") == "corn"
    head_classes = 4 if is_corn else cfg["num_classes"]
    bs = int(cfg.get("per_model_batch_size", {}).get(model_name, tcfg["batch_size"]))
    imbalance = dcfg.get("imbalance", "weighted_ce")
    aug_yaml = dcfg.get("augmentation_yaml", "configs/augmentation.yaml")
    seed = int(cfg.get("seed", 42))

    LOG.info("=== %s | model=%s seed=%d variant=%s imbalance=%s aug=%s loss=%s ema=%s corn=%s ===",
             eid, model_name, seed, variant, imbalance, Path(aug_yaml).name,
             cfg["loss"].get("name"), bool(tcfg.get("ema")), is_corn)
    LOG.info("    note: %s", spec.get("note", ""))
    if dry:
        LOG.info("    [dry-run] overrides=%s", json.dumps(spec["overrides"]))
        return None

    from src.common import RANDOM_SEED  # noqa: F401
    try:
        from src.benchmark.runner import LOG as _rl  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / eid
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        variant=variant, normalization=dcfg.get("normalization", "imagenet"),
        batch_size=bs, num_workers=dcfg.get("num_workers", 4), out_channels=cfg["in_channels"],
        imbalance=imbalance, augmentation_yaml=aug_yaml, return_meta=False,
        pin_memory=dcfg.get("pin_memory", True),
        persistent_workers=dcfg.get("persistent_workers", True), seed=seed)
    val_meta = build_dataloaders(
        variant=variant, normalization=dcfg.get("normalization", "imagenet"),
        batch_size=bs, num_workers=0, out_channels=cfg["in_channels"],
        imbalance="none", return_meta=True, pin_memory=False,
        persistent_workers=False, seed=seed)

    built = build_model(model_name, num_classes=head_classes,
                        in_chans=cfg["in_channels"], pretrained=True)
    pc_full = count_parameters(built.model)

    (out_dir / "config.json").write_text(json.dumps(
        {"experiment_id": eid, "model": model_name, "seed": seed, "head_classes": head_classes,
         "resolved_cfg": cfg, "overrides": spec["overrides"], "note": spec.get("note")},
        indent=2, default=str))

    lat.reset_peak_memory(device)
    t0 = time.perf_counter()
    _fit_kw = {}
    if spec.get("_smoke"):
        _fit_kw = dict(max_epochs=1, max_train_batches=4, max_val_batches=4)
    fit_summary = fit(built, loaders, device, cfg, out_dir, logger=LOG, **_fit_kw)
    train_secs = round(time.perf_counter() - t0, 1)

    # ---- reload best, evaluate on val with predictions ----
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    built.model.load_state_dict(best["state_dict"])
    _lc = cfg["loss"]
    _k = ("class_balanced_effective_num" if _lc.get("name") == "class_balanced"
          else _lc.get("class_weights_key", "weights_list_balanced"))
    _cw = None if _lc.get("name") in ("cross_entropy", "corn") else torch.tensor(
        load_class_weights(_k), dtype=torch.float32, device=device)
    loss_fn = build_loss(_lc, _cw)
    ordinal = "corn" if is_corn else None
    ve = evaluate(built.model, val_meta["val"], loss_fn, device, amp=tcfg.get("amp", True),
                  return_predictions=True, ordinal=ordinal)
    pred = ve.pop("_predictions")
    clin = _clinical_metrics(pred["y_true"], pred["y_pred"], pred["y_prob"])

    # ---- profile ----
    comp = cx.param_counts(built.model)
    gm = cx.gmacs_fvcore(built.model, tuple(cfg["complexity"]["flop_input_shape"]), device="cpu")
    ckpt_mb = cx.checkpoint_size_mb(out_dir / "best.pt")
    gpu_lat = (lat.measure_latency(built.model, device, tuple(cfg["latency"]["input_shape"]),
                                   warmup=cfg["latency"]["warmup_iters"],
                                   iters=cfg["latency"]["timed_iters"])
               if device.type == "cuda" else None)
    cpu_lat = lat.measure_latency(built.model, torch.device("cpu"),
                                  tuple(cfg["latency"]["input_shape"]),
                                  warmup=cfg["latency"].get("cpu_warmup_iters", 5),
                                  iters=cfg["latency"].get("cpu_timed_iters", 30))

    tl = cfg["transfer_learning"]
    row = {
        "experiment_id": eid, "family": spec["family"], "model": model_name, "seed": seed,
        "ordinal": "corn" if is_corn else "", "ema": bool(tcfg.get("ema")),
        "preprocessing": variant, "loss": _lc.get("name"), "imbalance": imbalance,
        "augmentation": Path(aug_yaml).name,
        "differential_lr": bool(tl.get("differential_lr")),
        "stage_a_epochs": tl.get("stage_a_epochs"),
        "weight_decay": cfg["optimizer"].get("weight_decay"),
        "max_epochs": tcfg["max_epochs"], "patience": tcfg["early_stopping"]["patience"],
        "best_epoch": fit_summary["best_epoch"], "epochs_run": fit_summary["epochs_run"],
        "stopped_early": fit_summary["stopped_early"],
        "val_accuracy": round(ve["accuracy"], 4), "val_macro_f1": round(ve["macro_f1"], 4),
        "val_weighted_f1": round(ve["weighted_f1"], 4),
        "val_balanced_accuracy": round(ve["balanced_accuracy"], 4),
        "val_qwk": round(ve["quadratic_weighted_kappa"], 4),
        **{f"val_KL{k}_F1": round(ve[f"kl{k}_f1"], 4) for k in range(5)},
        **clin,
        "parameters": comp["parameters"], "model_size_mb": ckpt_mb, "gmacs": gm.get("gmacs"),
        "gpu_latency_ms": (gpu_lat or {}).get("mean_ms"), "cpu_latency_ms": cpu_lat["mean_ms"],
        "training_time_sec": train_secs, "checkpoint": str(out_dir / "best.pt"),
        "note": spec.get("note", ""),
    }
    (out_dir / "result.json").write_text(json.dumps({**row, "val_full": ve}, indent=2, default=str))
    _reg_append(row)
    LOG.info("    -> val_macroF1=%.4f QWK=%.4f KL1F1=%.4f | binOA acc=%.4f auc=%.4f | 3cls=%.4f "
             "| params=%s size=%.1fMB cpu_lat=%.1fms  (%.0fs)",
             row["val_macro_f1"], row["val_qwk"], row["val_KL1_F1"], row["val_binary_oa_acc"],
             row["val_binary_oa_auc"], row["val_3class_acc"], f"{row['parameters']:,}",
             row["model_size_mb"], row["cpu_latency_ms"], train_secs)
    try:
        del built.model
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return row


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["matrix", "combine", "multiseed"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run experiment_ids already in the registry")
    ap.add_argument("--only", default="", help="comma-list of experiment_id substrings to include")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch / few batches end-to-end plumbing test (writes to a _smoke registry)")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    if args.log_file:
        add_file_logger(args.log_file)
    global REG_CSV
    if args.smoke:
        REG_CSV = REG_CSV.with_name("EXPERIMENT_REGISTRY_smoke.csv")
    if not args.dry_run:
        _acquire_lock()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(CFG_PATH)
    LOG.info("PHASE 7 stage=%s device=%s  cfg=%s", args.stage, device, CFG_PATH.name)

    if args.stage == "matrix":
        specs = EXP.matrix_specs()
    elif args.stage == "combine":
        specs = EXP.combine_specs(_reg_rows())
    else:  # multiseed
        rows = [r for r in _reg_rows() if r["family"] in ("matrix", "combine")]
        rows = [r for r in rows if r.get("val_macro_f1")]
        rows.sort(key=lambda r: float(r["val_macro_f1"]), reverse=True)
        # global top-2, PLUS the best config for each architecture (so the
        # MobileNetV2 winner is always seed-tested even if ConvNeXt tops the board)
        picked, seen_ids, seen_arch = [], set(), set()
        for r in rows[:2]:
            picked.append(r); seen_ids.add(r["experiment_id"]); seen_arch.add(r["model"])
        for r in rows:
            if r["model"] not in seen_arch:
                picked.append(r); seen_ids.add(r["experiment_id"]); seen_arch.add(r["model"])
        top = picked
        LOG.info("multiseed finalists (by val_macro_f1): %s",
                 [(r["experiment_id"], r["val_macro_f1"]) for r in top])
        finalists = []
        for r in top:
            cfgp = PROJECT_ROOT / base_cfg["paths"]["models_dir"] / r["experiment_id"] / "config.json"
            ov = json.loads(cfgp.read_text())["overrides"] if cfgp.exists() else {}
            finalists.append({"experiment_id": r["experiment_id"], "model": r["model"], "overrides": ov})
        specs = EXP.multiseed_specs(finalists)

    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        specs = [s for s in specs if any(x in s["experiment_id"] for x in subs)]

    seen = _reg_seen()
    todo = [s for s in specs if args.force or s["experiment_id"] not in seen]
    LOG.info("%d spec(s); %d to run, %d already in registry", len(specs), len(todo), len(specs) - len(todo))

    for i, spec in enumerate(todo, 1):
        LOG.info("[%d/%d]", i, len(todo))
        if args.smoke:
            spec = {**spec, "_smoke": True}
        try:
            run_one(spec, base_cfg, device, dry=args.dry_run)
        except Exception as e:  # noqa: BLE001
            LOG.exception("FAILED %s: %s", spec["experiment_id"], e)

    LOG.info("stage %s done. registry -> %s", args.stage, REG_CSV)


if __name__ == "__main__":
    main()
