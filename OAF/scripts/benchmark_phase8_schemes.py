"""PHASE 8 - controlled label-scheme sweep for the KD-trained MobileNetV2.

Trains the SAME best KD recipe (auto-picked from reports/phase8/KD_REGISTRY.csv)
on four label schemes and reports val + test for each, so the accuracy / macro-F1
/ QWK / clinical-granularity trade-off is decided on evidence:

    5class       KL0 KL1 KL2 KL3 KL4                      (clinical KL standard)
    4class_kl01  {KL0,KL1}  KL2  KL3  KL4                 (fold the ambiguous Normal/Doubtful boundary)
    3class       Normal(KL0)  Early(KL1-2)  Advanced(KL3-4)
    binary       No-OA(KL0-1)  OA(KL2-4)

Teacher soft targets are collapsed the same way (prob mass summed into merged
columns). Test split scored once per scheme.

    python scripts/benchmark_phase8_schemes.py --log-file reports/phase8/p8_schemes.log
    python scripts/benchmark_phase8_schemes.py --only 4class_kl01,binary
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402
from src.datasets import build_dataloaders  # noqa: E402
from src.benchmark import complexity as cx, latency as lat  # noqa: E402
from src.benchmark.engine import build_scheduler, build_optimizer, EarlyStopping, ModelEMA  # noqa: E402
from src.benchmark.models import build_model, count_parameters, set_stage  # noqa: E402
from src.benchmark.distill import TeacherBank  # noqa: E402

LOG = get_logger("p8s")
CFG_PATH = PROJECT_ROOT / "configs" / "benchmark_phase8.yaml"
KD_REG = PROJECT_ROOT / "reports" / "phase8" / "KD_REGISTRY.csv"
TBANK_DIR = PROJECT_ROOT / "reports" / "phase8" / "teacher_logits"
OUT = PROJECT_ROOT / "reports" / "phase8"
LOCK = OUT / ".phase8_schemes.lock"

SCHEMES = {
    "5class":      {"map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, "names": ["KL0", "KL1", "KL2", "KL3", "KL4"]},
    "4class_kl01": {"map": {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}, "names": ["KL0-1", "KL2", "KL3", "KL4"]},
    "3class":      {"map": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}, "names": ["Normal", "Early", "Advanced"]},
    "binary":      {"map": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}, "names": ["No-OA", "OA"]},
}


def _lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        LOG.error("lock exists - abort"); sys.exit(2)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    atexit.register(lambda: LOCK.exists() and LOCK.unlink())


def _best_kd_cfg() -> dict:
    rows = [r for r in csv.DictReader(open(KD_REG)) if r["family"] == "kd" and r.get("val_macro_f1")]
    rows.sort(key=lambda r: float(r["val_macro_f1"]), reverse=True)
    b = rows[0]
    LOG.info("best KD recipe: %s (val_macroF1=%s teachers=%s T=%s a=%s ema=%s ls=%s mixup=%s)",
             b["experiment_id"], b["val_macro_f1"], b["teachers"], b["temperature"], b["alpha"],
             b["ema"], b["label_smoothing"], b["mixup_alpha"])
    teachers = b["teachers"].split("+") if b["teachers"] else ["convnext_tiny"]
    return {"experiment_id": b["experiment_id"], "teachers": teachers,
            "T": float(b["temperature"]), "alpha": float(b["alpha"]),
            "ema": b["ema"] in ("True", "true", "1"),
            "label_smoothing": float(b["label_smoothing"] or 0.0),
            "mixup_alpha": float(b["mixup_alpha"] or 0.0)}


def _metrics(y, p, names):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                                 balanced_accuracy_score, cohen_kappa_score, roc_auc_score)
    nc = len(names)
    yp = p.argmax(1)
    out = {
        "accuracy": round(float(accuracy_score(y, yp)), 4),
        "macro_f1": round(float(f1_score(y, yp, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y, yp, average="weighted")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, yp)), 4),
        "macro_precision": round(float(precision_score(y, yp, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y, yp, average="macro", zero_division=0)), 4),
    }
    if nc > 2:
        out["qwk"] = round(float(cohen_kappa_score(y, yp, weights="quadratic")), 4)
    else:
        out["qwk"] = None
    if nc == 2:
        out["roc_auc"] = round(float(roc_auc_score(y, p[:, 1])), 4)
        tp = int(((yp == 1) & (y == 1)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
        tn = int(((yp == 0) & (y == 0)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
        out["sensitivity"] = round(tp / max(tp + fn, 1), 4)
        out["specificity"] = round(tn / max(tn + fp, 1), 4)
    per = f1_score(y, yp, average=None, labels=list(range(nc)), zero_division=0)
    sup = np.bincount(y, minlength=nc)
    out["per_class_f1"] = {names[i]: round(float(per[i]), 4) for i in range(nc)}
    out["support"] = {names[i]: int(sup[i]) for i in range(nc)}
    return out


def _infer(model, loader, device, nc):
    import torch
    ys, ps = [], []
    model.eval()
    with torch.inference_mode():
        for b in loader:
            x, y = b[0].to(device), b[1]
            pr = torch.softmax(model(x).float(), 1)
            pr = pr + torch.softmax(model(torch.flip(x, [-1])).float(), 1)   # hflip TTA
            ps.append((pr / 2).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ps).astype(np.float64)


def run_scheme(name: str, kd: dict, base_cfg: dict, device, *, smoke: bool) -> dict:
    import torch
    import torch.nn.functional as F

    lm = SCHEMES[name]["map"]; names = SCHEMES[name]["names"]; nc = len(names)
    seed = 42
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    LOG.info("=== scheme=%s  (%d classes: %s) ===", name, nc, names)

    dcfg = base_cfg["data"]
    dl = build_dataloaders(variant=base_cfg["preprocessing_variant"], normalization=dcfg["normalization"],
                           batch_size=base_cfg["train"]["batch_size"], num_workers=dcfg["num_workers"],
                           out_channels=3, imbalance=dcfg.get("imbalance", "weighted_ce"),
                           augmentation_yaml=dcfg["augmentation_yaml"], return_meta=True,
                           pin_memory=True, persistent_workers=True, seed=seed, label_map=lm)
    ev = build_dataloaders(variant=base_cfg["preprocessing_variant"], normalization=dcfg["normalization"],
                           batch_size=128, num_workers=0, out_channels=3, imbalance="none",
                           return_meta=False, pin_memory=False, persistent_workers=False,
                           seed=seed, label_map=lm)
    cw = dl["class_weights"].to(device)
    assert dl["num_classes"] == nc, (dl["num_classes"], nc)

    banks = {}
    if kd["alpha"] > 0:
        for split in ("train",):
            banks[split] = TeacherBank([TBANK_DIR / f"{split}__{t}.npz" for t in kd["teachers"]],
                                       [1.0] * len(kd["teachers"]))

    built = build_model("mobilenet_v2", num_classes=nc, in_chans=3, pretrained=True)
    model = built.model.to(device)
    out_dir = PROJECT_ROOT / "models" / "phase8_schemes" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    early = EarlyStopping(patience=5, mode="max")
    ema = ModelEMA(model, 0.999) if kd["ema"] else None
    max_ep = 1 if smoke else base_cfg["train"]["max_epochs"]
    stage_a = min(3, max_ep)
    T, alpha, ls = kd["T"], kd["alpha"], kd["label_smoothing"]

    def mk_opt(stage):
        oc = dict(base_cfg["optimizer"])
        oc["lr"] = float(base_cfg["transfer_learning"]["stage_a_lr" if stage == "A" else "stage_b_lr"])
        return build_optimizer([p for p in model.parameters() if p.requires_grad], oc)

    stage = "A" if stage_a > 0 else "B"
    set_stage(built, stage); opt = mk_opt(stage)
    sched = build_scheduler(opt, base_cfg["scheduler"], max_ep)

    hist = []
    t0 = time.perf_counter()
    for ep in range(1, max_ep + 1):
        if stage == "A" and ep == stage_a + 1:
            stage = "B"; set_stage(built, "B"); opt = mk_opt("B")
            sched = build_scheduler(opt, base_cfg["scheduler"], max_ep - stage_a)
        model.train()
        for bi, (x, y, meta) in enumerate(dl["train"]):
            if smoke and bi >= 6:
                break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                hard = F.cross_entropy(logits, y, weight=cw, label_smoothing=ls)
                if alpha > 0:
                    sids = [str(s) for s in meta["sample_id"]]
                    ts = torch.tensor(banks["train"].ensemble_soft(sids, T, collapse=lm),
                                      dtype=torch.float32, device=device)
                    soft = F.kl_div(F.log_softmax(logits / T, 1), ts.clamp_min(1e-8),
                                    reduction="batchmean") * (T * T)
                    loss = alpha * soft + (1 - alpha) * hard
                else:
                    loss = hard
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if ema:
                ema.update(model)
        if sched:
            sched.step()

        if ema:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        yv, pv = _infer(model, ev["val"], device, nc)
        if ema:
            model.load_state_dict(raw, strict=True)
        vm = _metrics(yv, pv, names)
        is_best = early.update(vm["macro_f1"], ep)
        hist.append({"epoch": ep, **{k: vm[k] for k in ("accuracy", "macro_f1", "qwk")}})
        LOG.info("  ep%02d/%d %s val_acc=%.4f val_macroF1=%.4f qwk=%s%s", ep, max_ep, stage,
                 vm["accuracy"], vm["macro_f1"], vm["qwk"], "  *best" if is_best else "")
        sd = ema.state_dict(model) if ema else model.state_dict()
        torch.save({"state_dict": sd, "scheme": name, "names": names, "epoch": ep,
                    "val": vm}, out_dir / "last.pt")
        if is_best:
            torch.save({"state_dict": sd, "scheme": name, "names": names, "epoch": ep,
                        "val": vm}, out_dir / "best.pt")
        if early.should_stop:
            LOG.info("  early stop @ %d (best %.4f @ %d)", ep, early.best, early.best_epoch)
            break
    secs = round(time.perf_counter() - t0, 1)

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    yv, pv = _infer(model, ev["val"], device, nc)
    yt, pt = _infer(model, ev["test"], device, nc)
    val_m = _metrics(yv, pv, names)
    test_m = _metrics(yt, pt, names)
    comp = count_parameters(model)
    cpu_lat = lat.measure_latency(model, torch.device("cpu"), (1, 3, 224, 224), warmup=5, iters=30)["mean_ms"]

    res = {"scheme": name, "n_classes": nc, "class_names": names,
           "best_epoch": early.best_epoch, "epochs_run": len(hist), "train_secs": secs,
           "parameters": comp["parameters"], "model_size_mb": cx.checkpoint_size_mb(out_dir / "best.pt"),
           "cpu_latency_ms": round(cpu_lat, 3),
           "val": val_m, "test": test_m}
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    LOG.info("  -> TEST acc=%.4f macroF1=%.4f qwk=%s | per-class F1 %s",
             test_m["accuracy"], test_m["macro_f1"], test_m["qwk"], test_m["per_class_f1"])
    del model, built.model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of scheme names")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()
    if args.log_file:
        add_file_logger(args.log_file)
    _lock()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(CFG_PATH)
    kd = _best_kd_cfg()
    schemes = [s.strip() for s in args.only.split(",") if s.strip()] or list(SCHEMES)

    results = []
    for name in schemes:
        try:
            results.append(run_scheme(name, kd, base_cfg, device, smoke=args.smoke))
        except Exception as e:  # noqa: BLE001
            LOG.exception("FAILED scheme %s: %s", name, e)
    if not results:
        return

    rows = []
    for r in results:
        rows.append({
            "scheme": r["scheme"], "n_classes": r["n_classes"],
            "val_accuracy": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
            "val_qwk": r["val"]["qwk"],
            "test_accuracy": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
            "test_weighted_f1": r["test"]["weighted_f1"], "test_balanced_accuracy": r["test"]["balanced_accuracy"],
            "test_qwk": r["test"]["qwk"], "test_roc_auc": r["test"].get("roc_auc"),
            "test_sensitivity": r["test"].get("sensitivity"), "test_specificity": r["test"].get("specificity"),
            "test_per_class_f1": json.dumps(r["test"]["per_class_f1"]),
            "parameters": r["parameters"], "model_size_mb": r["model_size_mb"],
            "cpu_latency_ms": r["cpu_latency_ms"], "kd_recipe": kd["experiment_id"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "LABEL_SCHEME_SWEEP.csv", index=False)
    (OUT / "label_scheme_sweep.json").write_text(json.dumps(results, indent=2, default=str))

    L = ["# Phase 8 - label-scheme sweep (KD MobileNetV2, recipe = " + kd["experiment_id"] + ")", "",
         "Same KD recipe, four label schemes. Test scored once per scheme.", "",
         "| scheme | classes | val Acc | val MacroF1 | test Acc | test MacroF1 | test wF1 | test QWK | test AUC | params |",
         "|---|--|--|--|--|--|--|--|--|--|"]
    for r in rows:
        L.append(f"| {r['scheme']} | {r['n_classes']} | {r['val_accuracy']} | {r['val_macro_f1']} | "
                 f"**{r['test_accuracy']}** | **{r['test_macro_f1']}** | {r['test_weighted_f1']} | "
                 f"{r['test_qwk']} | {r['test_roc_auc']} | {r['parameters']:,} |")
    L += ["", "## Per-class test F1", ""]
    for r in results:
        L.append(f"- **{r['scheme']}**: {r['test']['per_class_f1']}  (support {r['test']['support']})")
    L += ["", "## Note", "- 5class is the clinical KL standard. 4class_kl01 folds the noisy KL0/KL1 boundary.",
          "- Higher accuracy on a coarser scheme is expected; weigh it against lost clinical granularity.",
          "- KL4 stays its own class in 5class / 4class because its F1 is already ~0.88 (not the weak point)."]
    (OUT / "LABEL_SCHEME_SWEEP.md").write_text("\n".join(L))
    LOG.info("wrote %s + .md + .json", OUT / "LABEL_SCHEME_SWEEP.csv")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
