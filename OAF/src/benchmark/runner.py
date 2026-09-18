"""End-to-end run of ONE model: train -> select best -> evaluate -> profile.

Shared by scripts/train_model.py and scripts/benchmark_models.py so the exact
same code path is used whether you run one model or all ten.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, get_logger
from src.datasets import build_dataloaders
from . import complexity as cx
from . import latency as lat
from .checks import leakage_and_distribution_check
from .engine import build_loss, evaluate, fit
from .models import build_model, count_parameters
from .plots import confusion_matrix_plots, training_curve_plot

LOG = get_logger("runner")


def _predictions_dataframe(pred: dict) -> pd.DataFrame:
    meta = pred.get("meta") or []
    n = len(pred["y_true"])
    base = {
        "sample_id": [meta[i].get("sample_id") if i < len(meta) else "" for i in range(n)],
        "patient_id": [meta[i].get("patient_id") if i < len(meta) else "" for i in range(n)],
        "side": [meta[i].get("knee_side") if i < len(meta) else "" for i in range(n)],
        "true_label": pred["y_true"],
        "predicted_label": pred["y_pred"],
    }
    for k in range(5):
        base[f"prob_kl{k}"] = pred["y_prob"][:, k]
    return pd.DataFrame(base)


def run_one_model(model_name: str, cfg: dict, device, *,
                  reports_dir: Path, models_dir: Path,
                  max_epochs: int | None = None,
                  max_train_batches: int | None = None,
                  max_val_batches: int | None = None,
                  num_workers: int | None = None,
                  batch_size: int | None = None,
                  env: dict | None = None,
                  extra_config: dict | None = None,
                  out_subdir: str | None = None,
                  artifact_name: str | None = None,
                  logger: logging.Logger | None = None) -> dict:
    log = logger or LOG
    variant = cfg["preprocessing_variant"]
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    bs = int(batch_size or tcfg["batch_size"])
    nw = dcfg["num_workers"] if num_workers is None else int(num_workers)
    # filename tag for report artefacts (unique per experiment run; = model_name for benchmarks)
    tag = artifact_name or out_subdir or model_name
    tag = tag.split("/")[-1]

    out_model_dir = models_dir / (out_subdir or model_name)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    loaders = build_dataloaders(
        variant=variant, normalization=dcfg.get("normalization", "imagenet"),
        batch_size=bs, num_workers=nw, out_channels=cfg["in_channels"],
        imbalance="weighted_ce",
        augmentation_yaml=dcfg.get("augmentation_yaml", "configs/augmentation.yaml"),
        return_meta=False, pin_memory=dcfg.get("pin_memory", True),
        persistent_workers=dcfg.get("persistent_workers", True),
        seed=cfg["seed"])
    # a second, meta-returning val loader for prediction export
    val_meta_loaders = build_dataloaders(
        variant=variant, normalization=dcfg.get("normalization", "imagenet"),
        batch_size=bs, num_workers=0, out_channels=cfg["in_channels"],
        imbalance="none", return_meta=True, pin_memory=False,
        persistent_workers=False, seed=cfg["seed"])

    # ---- model ----
    built = build_model(model_name, num_classes=cfg["num_classes"],
                        in_chans=cfg["in_channels"], pretrained=True)
    log.info("[%s] built | backend=%s | device=%s | variant=%s | norm=%s | bs=%d | workers=%d",
             model_name, built.backend, device, variant,
             dcfg.get("normalization", "imagenet"), bs, nw)
    if built.notes:
        log.info("[%s] NOTE: %s", model_name, built.notes)
    pc_full = count_parameters(built.model)  # before freezing (Stage B trains all)

    # ---- experiment config record (spec section 45) ----
    exp_cfg = {
        "model_name": model_name, "timm_name": built.timm_name,
        "input_size": cfg["image_size"], "in_channels": cfg["in_channels"],
        "native_input_size": built.native_input_size,
        "fairness_note": built.notes or "native 224px; no resize deviation",
        "num_classes": cfg["num_classes"], "preprocessing_variant": variant,
        "normalization": dcfg.get("normalization"),
        "augmentation_yaml": dcfg.get("augmentation_yaml"),
        "loss": cfg["loss"], "optimizer": cfg["optimizer"], "scheduler": cfg.get("scheduler"),
        "transfer_learning": cfg["transfer_learning"],
        "batch_size": bs, "max_epochs": int(max_epochs or tcfg["max_epochs"]),
        "early_stopping": tcfg["early_stopping"], "amp": tcfg.get("amp", True),
        "grad_clip_norm": tcfg.get("grad_clip_norm"), "seed": cfg["seed"],
        "selection_metric": cfg["eval"]["selection_metric"],
        "dataset_manifest": f"metadata/processed_manifest_{variant}.csv",
        "class_weights_key": cfg["loss"].get("class_weights_key"),
        "gradcam_layer_candidates": built.gradcam_layers,
        "classifier_attr": built.classifier_attr,
        "backend": built.backend,
        "env": env or {},
    }
    if extra_config:
        exp_cfg.update(extra_config)
    (out_model_dir / "config.json").write_text(json.dumps(exp_cfg, indent=2, default=str))

    # ---- train ----
    lat.reset_peak_memory(device)
    fit_summary = fit(built, loaders, device, cfg, out_model_dir, logger=log,
                      max_epochs=max_epochs, max_train_batches=max_train_batches,
                      max_val_batches=max_val_batches)
    train_peak_mem = lat.peak_memory_mb(device)

    # ---- load best checkpoint, evaluate on val WITH predictions ----
    import torch
    best = torch.load(out_model_dir / "best.pt", map_location=device, weights_only=False)
    built.model.load_state_dict(best["state_dict"])
    from .engine import load_class_weights
    _lc = cfg["loss"]
    _k = ("class_balanced_effective_num" if _lc.get("name") == "class_balanced"
          else _lc.get("class_weights_key", "weights_list_balanced"))
    _cw = None if _lc.get("name") == "cross_entropy" else torch.tensor(
        load_class_weights(_k), dtype=torch.float32, device=device)
    loss_fn = build_loss(_lc, _cw)
    val_eval = evaluate(built.model, val_meta_loaders["val"], loss_fn, device,
                        amp=tcfg.get("amp", True), max_batches=max_val_batches,
                        return_predictions=True)
    pred = val_eval.pop("_predictions")

    # ---- per-model artefacts ----
    (reports_dir / "classification_reports").mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_predictions").mkdir(parents=True, exist_ok=True)
    clean_metrics = {k: v for k, v in val_eval.items() if not k.startswith("_")}
    (reports_dir / "classification_reports" / f"{tag}.json").write_text(
        json.dumps(clean_metrics, indent=2))
    _predictions_dataframe(pred).to_csv(
        reports_dir / "model_predictions" / f"{tag}_val_predictions.csv", index=False)

    cm_paths = confusion_matrix_plots(val_eval["confusion_matrix"], tag,
                                      reports_dir / "confusion_matrices")
    curve_path = training_curve_plot(fit_summary["history"], tag,
                                     reports_dir / "training_curves")

    # ---- complexity ----
    comp = cx.param_counts(built.model)
    gm = cx.gmacs_fvcore(built.model, tuple(cfg["complexity"]["flop_input_shape"]),
                         device="cpu")
    ckpt_mb = cx.checkpoint_size_mb(out_model_dir / "best.pt")

    # ---- latency (GPU if available, then CPU) ----
    lat.reset_peak_memory(device)
    gpu_lat = None
    if device.type == "cuda":
        gpu_lat = lat.measure_latency(
            built.model, device, tuple(cfg["latency"]["input_shape"]),
            warmup=cfg["latency"]["warmup_iters"], iters=cfg["latency"]["timed_iters"])
        infer_peak_mem = lat.peak_memory_mb(device)
    else:
        infer_peak_mem = {}
    import torch as _t
    cpu_lat = lat.measure_latency(
        built.model, _t.device("cpu"), tuple(cfg["latency"]["input_shape"]),
        warmup=cfg["latency"].get("cpu_warmup_iters", 5),
        iters=cfg["latency"].get("cpu_timed_iters", 30))
    primary_lat = gpu_lat or cpu_lat

    (reports_dir / "latency").mkdir(parents=True, exist_ok=True)
    (reports_dir / "latency" / f"{tag}_latency.json").write_text(json.dumps(
        {"gpu": gpu_lat, "cpu": cpu_lat, "train_peak_memory": train_peak_mem,
         "infer_peak_memory": infer_peak_mem}, indent=2))

    # ---- assemble result row ----
    result = {
        "model": model_name, "timm_name": built.timm_name,
        "accuracy": val_eval["accuracy"],
        "macro_precision": val_eval["macro_precision"],
        "macro_recall": val_eval["macro_recall"],
        "macro_f1": val_eval["macro_f1"],
        "weighted_f1": val_eval["weighted_f1"],
        "balanced_accuracy": val_eval["balanced_accuracy"],
        "cohen_kappa": val_eval["cohen_kappa"],
        "quadratic_weighted_kappa": val_eval["quadratic_weighted_kappa"],
        "roc_auc_ovr_macro": val_eval.get("roc_auc_ovr_macro"),
        **{f"kl{k}_f1": val_eval[f"kl{k}_f1"] for k in range(5)},
        **{f"kl{k}_support": val_eval[f"kl{k}_support"] for k in range(5)},
        "parameters": comp["parameters"],
        "trainable_parameters": pc_full["trainable_parameters"],
        "non_trainable_parameters": comp["non_trainable_parameters"],
        "model_size_mb": ckpt_mb,
        "gmacs": gm.get("gmacs"),
        "flops_or_macs": gm.get("gmacs"),   # spec section 30 column name; value is GMACs (MAC=1)
        "flops_estimate_g": gm.get("flops_estimate_g"),
        "gmacs_method": gm.get("method"),
        "mean_latency_ms": primary_lat["mean_ms"],
        "median_latency_ms": primary_lat["median_ms"],
        "std_latency_ms": primary_lat["std_ms"],
        "min_latency_ms": primary_lat["min_ms"],
        "max_latency_ms": primary_lat["max_ms"],
        "throughput_images_per_sec": primary_lat["throughput_images_per_sec"],
        "latency_device": primary_lat["device"],
        "cpu_mean_latency_ms": cpu_lat["mean_ms"],
        "peak_memory_mb": (infer_peak_mem.get("cuda_peak_alloc_mb")
                           or train_peak_mem.get("cuda_peak_alloc_mb")
                           or train_peak_mem.get("process_rss_mb")),
        "train_peak_cuda_mb": train_peak_mem.get("cuda_peak_alloc_mb"),
        "best_epoch": fit_summary["best_epoch"],
        "epochs_run": fit_summary["epochs_run"],
        "training_time_sec": fit_summary["training_time_sec"],
        "training_time": fit_summary["training_time_sec"],   # spec section 30 alias
        "stopped_early": fit_summary["stopped_early"],
        "adjacent_confusions": val_eval["adjacent_confusions"],
        "confusion_matrix": val_eval["confusion_matrix"],
        "gradcam_layer_candidates": built.gradcam_layers,
        "backend": built.backend,
        "preprocessing_variant": variant,
        "loss_name": cfg["loss"].get("name"),
        "seed": cfg["seed"],
        "stage_a_epochs": cfg["transfer_learning"].get("stage_a_epochs"),
        "max_epochs_budget": int(max_epochs or tcfg["max_epochs"]),
        "patience": tcfg["early_stopping"].get("patience"),
        "differential_lr": cfg["transfer_learning"].get("differential_lr"),
        "artefacts": {"best_ckpt": str(out_model_dir / "best.pt"),
                      "confusion": cm_paths, "training_curve": curve_path,
                      "config": str(out_model_dir / "config.json")},
    }
    if extra_config:
        result.update({k: v for k, v in extra_config.items() if k not in result})
    (out_model_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    # free GPU
    try:
        del built.model
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return result


def dry_run_model(model_name: str, cfg: dict, device, *, logger=None, n_batches: int = 2) -> dict:
    """Build model + a few batches; verify shapes / loss / backward. No training."""
    import torch
    log = logger or LOG
    dcfg = cfg["data"]
    loaders = build_dataloaders(
        variant=cfg["preprocessing_variant"], normalization=dcfg.get("normalization", "imagenet"),
        batch_size=min(8, cfg["train"]["batch_size"]), num_workers=0,
        out_channels=cfg["in_channels"], imbalance="weighted_ce",
        augmentation_yaml=dcfg.get("augmentation_yaml"), return_meta=False,
        pin_memory=False, persistent_workers=False, seed=cfg["seed"])
    built = build_model(model_name, num_classes=cfg["num_classes"],
                        in_chans=cfg["in_channels"], pretrained=True)
    model = built.model.to(device).train()
    loss_fn = build_loss(cfg["loss"], loaders["class_weights"].to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    checks = {"model": model_name, "timm_name": built.timm_name, "ok": True, "issues": []}
    for bi, (x, y) in enumerate(loaders["train"]):
        if bi >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        if tuple(x.shape[1:]) != (cfg["in_channels"], *cfg["image_size"]):
            checks["issues"].append(f"input shape {tuple(x.shape)} != [B,{cfg['in_channels']},"
                                    f"{cfg['image_size'][0]},{cfg['image_size'][1]}]")
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if tuple(out.shape) != (x.shape[0], cfg["num_classes"]):
            checks["issues"].append(f"output shape {tuple(out.shape)} != [{x.shape[0]},{cfg['num_classes']}]")
        loss = loss_fn(out, y)
        if not torch.isfinite(loss):
            checks["issues"].append(f"non-finite loss {loss.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        checks[f"batch{bi}"] = {"in": list(x.shape), "out": list(out.shape),
                                "loss": round(float(loss.detach()), 5)}
    # one val batch
    for x, y in loaders["val"]:
        x, y = x.to(device), y.to(device)
        with torch.inference_mode():
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
        checks["val_batch"] = {"in": list(x.shape), "out": list(out.shape)}
        break
    checks["ok"] = not checks["issues"]
    log.info("[dry-run] %-20s %s %s", model_name, "OK" if checks["ok"] else "FAIL",
             checks.get("batch0", {}))
    try:
        del model, built
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return checks
