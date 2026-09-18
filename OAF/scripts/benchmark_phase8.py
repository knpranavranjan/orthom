"""PHASE 8 - MobileNetV2 accuracy push via knowledge distillation (2.23 M params fixed).

Student  : MobileNetV2 (timm mobilenetv2_100), 3-ch, 5-class, ImageNet-pretrained.
Teachers : frozen VGG16 / ConvNeXt-Tiny logits from the Phase-8 teacher bank.
Recipe   : Phase-7 winning base (strong-aug + WCE) + 2-stage transfer + KD.

Experiments (one dimension vs configs/benchmark_phase8.yaml):
    K0_baseline        no KD (alpha=0)         <- reproduces mobilenet_v2__M_strongaug
    K1_kd_cnx_t3a5     KD ConvNeXt   T=3 a=0.5
    K2_kd_cnx_t4a7     KD ConvNeXt   T=4 a=0.7
    K3_kd_vgg_t4a7     KD VGG16      T=4 a=0.7
    K4_kd_ens_t4a7     KD VGG16+CNX  T=4 a=0.7
    K5_kd_ens_ema_ls   K4 + EMA + label_smoothing 0.05
    K6_kd_ens_long     K4 + EMA + max_epochs 70 / patience 8 + alpha 0.8
    K7_kd_ens_mixup    K4 + EMA + mixup(0.2)

    python scripts/benchmark_phase8.py --stage kd        --log-file reports/phase8/p8_kd.log
    python scripts/benchmark_phase8.py --stage multiseed --log-file reports/phase8/p8_seed.log
    python scripts/benchmark_phase8.py --stage kd --dry-run
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
from src.benchmark import complexity as cx, latency as lat  # noqa: E402
from src.benchmark.engine import (build_scheduler, evaluate, load_class_weights,  # noqa: E402
                                  ModelEMA, EarlyStopping, build_optimizer)
from src.benchmark.models import build_model, count_parameters, set_stage, split_param_groups  # noqa: E402
from src.benchmark.distill import TeacherBank, kd_loss, mixup_batch  # noqa: E402

LOG = get_logger("p8")
CFG_PATH = PROJECT_ROOT / "configs" / "benchmark_phase8.yaml"
REG = PROJECT_ROOT / "reports" / "phase8" / "KD_REGISTRY.csv"
LOCK = PROJECT_ROOT / "reports" / "phase8" / ".phase8.lock"
TBANK_DIR = PROJECT_ROOT / "reports" / "phase8" / "teacher_logits"

REG_COLS = ["experiment_id", "family", "seed", "kd", "teachers", "temperature", "alpha",
            "ema", "label_smoothing", "mixup_alpha", "max_epochs", "patience",
            "best_epoch", "epochs_run", "stopped_early",
            "val_accuracy", "val_macro_f1", "val_qwk", "val_balanced_accuracy", "val_weighted_f1",
            "val_mae", "val_within1", "val_KL0_F1", "val_KL1_F1", "val_KL2_F1", "val_KL3_F1", "val_KL4_F1",
            "val_binary_oa_acc", "val_binary_oa_auc", "val_3class_acc",
            "val_macro_f1_tta", "parameters", "model_size_mb", "gmacs",
            "gpu_latency_ms", "cpu_latency_ms", "training_time_sec", "checkpoint", "note"]

# ---- experiment matrix ---------------------------------------------------- #
def _specs():
    E = {
        "K0_baseline":     {"alpha": 0.0, "teachers": ["convnext_tiny"], "note": "no KD (== M_strongaug)"},
        "K1_kd_cnx_t3a5":  {"alpha": 0.5, "temperature": 3.0, "teachers": ["convnext_tiny"]},
        "K2_kd_cnx_t4a7":  {"alpha": 0.7, "temperature": 4.0, "teachers": ["convnext_tiny"]},
        "K3_kd_vgg_t4a7":  {"alpha": 0.7, "temperature": 4.0, "teachers": ["vgg16"]},
        "K4_kd_ens_t4a7":  {"alpha": 0.7, "temperature": 4.0, "teachers": ["vgg16", "convnext_tiny"]},
        "K5_kd_ens_ema_ls": {"alpha": 0.7, "temperature": 4.0, "teachers": ["vgg16", "convnext_tiny"],
                             "ema": True, "label_smoothing": 0.05},
        "K6_kd_ens_long":  {"alpha": 0.8, "temperature": 4.0, "teachers": ["vgg16", "convnext_tiny"],
                            "ema": True, "max_epochs": 70, "patience": 8},
        "K7_kd_ens_mixup": {"alpha": 0.7, "temperature": 4.0, "teachers": ["vgg16", "convnext_tiny"],
                            "ema": True, "mixup_alpha": 0.2},
    }
    return [{"experiment_id": k, "family": "kd", "overrides": v} for k, v in E.items()]


# ------------------------------------------------------------------------- #
def _acquire_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        LOG.error("lock %s exists - another Phase-8 run active. Abort.", LOCK); sys.exit(2)
    os.write(fd, f"{os.getpid()}\n".encode()); os.close(fd)
    atexit.register(lambda: LOCK.exists() and LOCK.unlink())


def _reg_seen():
    if not REG.exists():
        return set()
    with open(REG, newline="") as fh:
        return {r["experiment_id"] for r in csv.DictReader(fh)}


def _reg_append(row):
    REG.parent.mkdir(parents=True, exist_ok=True)
    new = not REG.exists()
    with open(REG, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REG_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in REG_COLS})


def _clin(y, yp, prob):
    from sklearn.metrics import roc_auc_score
    y, yp = np.asarray(y), np.asarray(yp)
    p_oa = prob[:, 2:].sum(1); yb = (y >= 2).astype(int)
    acc = float(((p_oa >= 0.5).astype(int) == yb).mean())
    try:
        auc = float(roc_auc_score(yb, p_oa)) if len(np.unique(yb)) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        auc = float("nan")
    m = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    acc3 = float((np.vectorize(m.get)(y) == np.vectorize(m.get)(yp)).mean())
    return acc, auc, acc3, float(np.abs(y - yp).mean()), float((np.abs(y - yp) <= 1).mean())


def _evaluate_tta(model, loader, device):
    import torch
    model.eval(); ys, ps = [], []
    with torch.inference_mode():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            p = torch.softmax(model(x).float(), 1) + torch.softmax(model(torch.flip(x, [-1])).float(), 1)
            ps.append((p / 2).cpu().numpy()); ys.append(y.numpy())
    from src.benchmark.metrics import compute_all
    yt = np.concatenate(ys); pp = np.concatenate(ps)
    return compute_all(yt, pp.argmax(1), pp)["macro_f1"]


# ------------------------------------------------------------------------- #
def run_one(spec, base_cfg, device, bank_cache, *, dry, smoke):
    import torch
    import torch.nn.functional as F

    eid = spec["experiment_id"]
    cfg = copy.deepcopy(base_cfg)
    d = cfg["distill"]
    ov = spec["overrides"]
    for k in ("alpha", "temperature", "teachers", "mixup_alpha"):
        if k in ov:
            d[k] = ov[k]
    ema_on = bool(ov.get("ema", False))
    ls = float(ov.get("label_smoothing", cfg["loss"].get("label_smoothing", 0.0)))
    max_epochs = int(ov.get("max_epochs", cfg["train"]["max_epochs"]))
    patience = int(ov.get("patience", cfg["train"]["early_stopping"]["patience"]))
    seed = int(ov.get("seed", cfg.get("seed", 42)))
    teachers = list(d["teachers"])
    alpha = float(d["alpha"]); T = float(d["temperature"]); mixup_a = float(d.get("mixup_alpha", 0.0))
    kd_on = alpha > 0.0

    LOG.info("=== %s | kd=%s teachers=%s T=%.1f a=%.2f ema=%s ls=%.2f mixup=%.2f ep=%d seed=%d ===",
             eid, kd_on, teachers if kd_on else "-", T, alpha, ema_on, ls, mixup_a, max_epochs, seed)
    LOG.info("    %s", ov.get("note", ""))
    if dry:
        return None
    if smoke:
        max_epochs, patience = 1, 1

    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # teacher bank (cache across experiments by teacher-tuple)
    bank = None
    if kd_on:
        key = tuple(teachers)
        if key not in bank_cache:
            paths = [TBANK_DIR / f"train__{t}.npz" for t in teachers] + \
                    [TBANK_DIR / f"val__{t}.npz" for t in teachers]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"teacher bank missing: {missing} - run scripts/phase8_teacher_bank.py")
            tw = cfg["distill"].get("teacher_weights") or [1.0] * len(teachers)
            tw = tw[:len(teachers)]
            bank_cache[key] = {
                "train": TeacherBank([TBANK_DIR / f"train__{t}.npz" for t in teachers], tw),
                "val": TeacherBank([TBANK_DIR / f"val__{t}.npz" for t in teachers], tw),
            }
            LOG.info("    teacher bank loaded: train n=%d val n=%d",
                     bank_cache[key]["train"].n, bank_cache[key]["val"].n)
        bank = bank_cache[key]

    dcfg = cfg["data"]
    loaders = build_dataloaders(variant=cfg["preprocessing_variant"], normalization=dcfg["normalization"],
                                batch_size=cfg["train"]["batch_size"], num_workers=dcfg["num_workers"],
                                out_channels=cfg["in_channels"], imbalance=dcfg.get("imbalance", "weighted_ce"),
                                augmentation_yaml=dcfg["augmentation_yaml"], return_meta=True,
                                pin_memory=True, persistent_workers=True, seed=seed)
    val_plain = build_dataloaders(variant=cfg["preprocessing_variant"], normalization=dcfg["normalization"],
                                  batch_size=128, num_workers=0, out_channels=cfg["in_channels"],
                                  imbalance="none", return_meta=True, pin_memory=False,
                                  persistent_workers=False, seed=seed)["val"]

    built = build_model("mobilenet_v2", num_classes=5, in_chans=3, pretrained=True)
    model = built.model.to(device)
    cw = torch.tensor(load_class_weights("weights_list_balanced"), dtype=torch.float32, device=device)

    out_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"experiment_id": eid, "kd": kd_on, "teachers": teachers, "temperature": T, "alpha": alpha,
         "ema": ema_on, "label_smoothing": ls, "mixup_alpha": mixup_a, "max_epochs": max_epochs,
         "patience": patience, "seed": seed, "note": ov.get("note", "")}, indent=2))

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    early = EarlyStopping(patience=patience, mode="max")
    ema = ModelEMA(model, 0.999) if ema_on else None
    stage_a = min(int(cfg["transfer_learning"]["stage_a_epochs"]), max_epochs)

    def make_opt(stage):
        oc = dict(cfg["optimizer"])
        oc["lr"] = float(cfg["transfer_learning"]["stage_a_lr"] if stage == "A"
                         else cfg["transfer_learning"]["stage_b_lr"])
        return build_optimizer([p for p in model.parameters() if p.requires_grad], oc)

    stage = "A" if stage_a > 0 else "B"
    set_stage(built, stage)
    opt = make_opt(stage)
    sched = build_scheduler(opt, cfg["scheduler"], max_epochs)

    history = []
    wall0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if stage == "A" and epoch == stage_a + 1:
            stage = "B"; set_stage(built, "B"); opt = make_opt("B")
            sched = build_scheduler(opt, cfg["scheduler"], max_epochs - stage_a)
            LOG.info("    -> stage B (unfreeze all)")
        model.train(); run_soft = run_hard = n = 0.0
        for bi, (x, y, meta) in enumerate(loaders["train"]):
            if smoke and bi >= 6:
                break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if kd_on:
                sids = [str(s) for s in meta["sample_id"]]
                t_soft = torch.tensor(bank["train"].ensemble_soft(sids, T), dtype=torch.float32, device=device)
            else:
                t_soft = None
            if mixup_a > 0 and kd_on:
                oh = F.one_hot(y, 5).float()
                x, t_soft, oh = mixup_batch(x, t_soft, oh, mixup_a)
                y_for_hard = oh
            else:
                y_for_hard = y
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                if kd_on:
                    if isinstance(y_for_hard, torch.Tensor) and y_for_hard.ndim == 2:
                        hard = -(F.log_softmax(logits, 1) * y_for_hard).sum(1)
                        hard = (hard * cw[y_for_hard.argmax(1)]).mean()
                        s_log_T = F.log_softmax(logits / T, 1)
                        soft = F.kl_div(s_log_T, t_soft.clamp_min(1e-8), reduction="batchmean") * (T * T)
                        loss = alpha * soft + (1 - alpha) * hard
                    else:
                        loss, soft, hard = kd_loss(logits, t_soft, y, T=T, alpha=alpha,
                                                   class_weights=cw, label_smoothing=ls)
                else:
                    soft = torch.zeros((), device=device)
                    hard = loss = F.cross_entropy(logits, y, weight=cw, label_smoothing=ls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if ema:
                ema.update(model)
            bs = y.size(0)
            run_soft += float(soft.detach()) * bs; run_hard += float(hard.detach()) * bs; n += bs
        if sched is not None:
            sched.step()

        if ema:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        va = evaluate(model, val_plain, lambda a, b: torch.tensor(0.0), device, amp=True)
        if ema:
            model.load_state_dict(raw, strict=True)

        sel = va["macro_f1"]
        is_best = early.update(sel, epoch)
        history.append({"epoch": epoch, "stage": stage, "soft": run_soft / n, "hard": run_hard / n,
                        "val_macro_f1": sel, "val_qwk": va["quadratic_weighted_kappa"],
                        "val_acc": va["accuracy"], "is_best": is_best})
        LOG.info("    ep%02d/%d %s soft=%.3f hard=%.3f | val_macroF1=%.4f QWK=%.4f acc=%.4f%s",
                 epoch, max_epochs, stage, run_soft / n, run_hard / n, sel,
                 va["quadratic_weighted_kappa"], va["accuracy"], "  * best" if is_best else "")
        ckpt = {"model_name": "mobilenet_v2", "timm_name": built.timm_name, "epoch": epoch,
                "state_dict": ema.state_dict(model) if ema else model.state_dict(),
                "val_metrics": {k: v for k, v in va.items() if not k.startswith("_")},
                "selection_metric": "val_macro_f1", "selection_value": sel, "ema": bool(ema)}
        torch.save(ckpt, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt, out_dir / "best.pt")
        if early.should_stop:
            LOG.info("    early stop @ %d (best %.4f @ %d)", epoch, early.best, early.best_epoch)
            break
    train_secs = round(time.perf_counter() - wall0, 1)

    # ---- reload best, full val eval + TTA ----
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    ve = evaluate(model, val_plain, lambda a, b: torch.tensor(0.0), device, amp=True,
                  return_predictions=True)
    pred = ve.pop("_predictions")
    b_acc, b_auc, acc3, mae, w1 = _clin(pred["y_true"], pred["y_pred"], pred["y_prob"])
    tta_f1 = _evaluate_tta(model, val_plain, device)

    comp = cx.param_counts(model)
    gm = cx.gmacs_fvcore(model, (1, 3, 224, 224), device="cpu")
    gpu_lat = (lat.measure_latency(model, device, (1, 3, 224, 224), warmup=20, iters=100)
               if device.type == "cuda" else None)
    cpu_lat = lat.measure_latency(model, torch.device("cpu"), (1, 3, 224, 224), warmup=5, iters=30)

    row = {"experiment_id": eid, "family": spec["family"], "seed": seed, "kd": kd_on,
           "teachers": "+".join(teachers) if kd_on else "", "temperature": T, "alpha": alpha,
           "ema": ema_on, "label_smoothing": ls, "mixup_alpha": mixup_a,
           "max_epochs": max_epochs, "patience": patience,
           "best_epoch": early.best_epoch, "epochs_run": len(history), "stopped_early": early.should_stop,
           "val_accuracy": round(ve["accuracy"], 4), "val_macro_f1": round(ve["macro_f1"], 4),
           "val_qwk": round(ve["quadratic_weighted_kappa"], 4),
           "val_balanced_accuracy": round(ve["balanced_accuracy"], 4),
           "val_weighted_f1": round(ve["weighted_f1"], 4),
           "val_mae": round(mae, 4), "val_within1": round(w1, 4),
           **{f"val_KL{k}_F1": round(ve[f"kl{k}_f1"], 4) for k in range(5)},
           "val_binary_oa_acc": round(b_acc, 4), "val_binary_oa_auc": round(b_auc, 4),
           "val_3class_acc": round(acc3, 4), "val_macro_f1_tta": round(tta_f1, 4),
           "parameters": comp["parameters"], "model_size_mb": cx.checkpoint_size_mb(out_dir / "best.pt"),
           "gmacs": gm.get("gmacs"), "gpu_latency_ms": (gpu_lat or {}).get("mean_ms"),
           "cpu_latency_ms": cpu_lat["mean_ms"], "training_time_sec": train_secs,
           "checkpoint": str(out_dir / "best.pt"), "note": ov.get("note", "")}
    (out_dir / "result.json").write_text(json.dumps({**row, "history": history}, indent=2, default=str))
    _reg_append(row)
    LOG.info("    -> val_macroF1=%.4f (TTA %.4f) QWK=%.4f KL1=%.4f | binOA %.4f/%.4f | 3cls %.4f | %.0fs",
             row["val_macro_f1"], row["val_macro_f1_tta"], row["val_qwk"], row["val_KL1_F1"],
             row["val_binary_oa_acc"], row["val_binary_oa_auc"], row["val_3class_acc"], train_secs)
    del model, built.model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["kd", "multiseed"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()
    if args.log_file:
        add_file_logger(args.log_file)
    global REG
    if args.smoke:
        REG = REG.with_name("KD_REGISTRY_smoke.csv")
    if not args.dry_run:
        _acquire_lock()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(CFG_PATH)
    LOG.info("PHASE 8 stage=%s device=%s", args.stage, device)

    if args.stage == "kd":
        specs = _specs()
    else:
        rows = [r for r in csv.DictReader(open(REG)) if r.get("val_macro_f1")]
        rows.sort(key=lambda r: float(r["val_macro_f1"]), reverse=True)
        top = rows[:2]
        LOG.info("multiseed finalists: %s", [(r["experiment_id"], r["val_macro_f1"]) for r in top])
        specs = []
        for r in top:
            ovp = json.loads((PROJECT_ROOT / base_cfg["paths"]["models_dir"] / r["experiment_id"] / "config.json").read_text())
            for s in (123, 3407):
                ov = {k: ovp[k] for k in ("alpha", "temperature", "teachers", "mixup_alpha",
                                          "label_smoothing", "max_epochs", "patience") if k in ovp}
                ov["ema"] = ovp.get("ema", False); ov["seed"] = s; ov["note"] = f"{r['experiment_id']} @ seed {s}"
                specs.append({"experiment_id": f"{r['experiment_id']}__seed{s}", "family": "multiseed",
                              "overrides": ov})

    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        specs = [s for s in specs if any(x in s["experiment_id"] for x in subs)]
    seen = _reg_seen()
    todo = [s for s in specs if args.force or s["experiment_id"] not in seen]
    LOG.info("%d spec(s); %d to run", len(specs), len(todo))

    bank_cache = {}
    for i, s in enumerate(todo, 1):
        LOG.info("[%d/%d]", i, len(todo))
        try:
            run_one(s, base_cfg, device, bank_cache, dry=args.dry_run, smoke=args.smoke)
        except Exception as e:  # noqa: BLE001
            LOG.exception("FAILED %s: %s", s["experiment_id"], e)
    LOG.info("stage %s done -> %s", args.stage, REG)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
