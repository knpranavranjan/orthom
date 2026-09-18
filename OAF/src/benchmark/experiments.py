"""PHASE 3 - PART C/D/G/H : controlled-experiment definitions + append-only tracking.

An "experiment" = one training run with ONE dimension changed vs the Phase-3
baseline (configs/benchmark_phase3.yaml). Everything else is held fixed.

Master table (append-only, never overwritten):
    reports/phase3/experiment_results.csv    (columns = EXPERIMENT_COLUMNS)
Per-run artefacts:
    models/phase3/experiments/<experiment_id>/{best.pt,last.pt,config.json,history.json,result.json}
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

EXPERIMENT_COLUMNS = [
    "experiment_id", "family", "model", "preprocessing", "loss", "optimizer",
    "learning_rate", "stage_a_epochs", "max_epochs", "patience", "differential_lr",
    "augmentation", "seed",
    "best_epoch", "epochs_run",
    "val_accuracy", "val_macro_f1", "val_qwk", "val_balanced_accuracy",
    "KL0_F1", "KL1_F1", "KL2_F1", "KL3_F1", "KL4_F1",
    "parameters", "gmacs", "latency_ms_gpu", "latency_ms_cpu", "training_time_sec",
    "stopped_early", "checkpoint",
]


# --------------------------------------------------------------------------- #
# Spec generation - each returns a list of dicts:
#   {"experiment_id", "family", "model", "overrides": <partial cfg>, "batch_size"?}
# --------------------------------------------------------------------------- #
def preprocessing_specs(models, variants=("basic", "clahe", "histeq")):
    out = []
    for m in models:
        for v in variants:
            out.append({"experiment_id": f"{m}__prep__{v}", "family": "preprocessing",
                        "model": m, "overrides": {"preprocessing_variant": v}})
    return out


def schedule_specs(models):
    grid = [("e30p5", 30, 5), ("e50p8", 50, 8)]
    out = []
    for m in models:
        for tag, ep, pat in grid:
            out.append({"experiment_id": f"{m}__sched__{tag}", "family": "schedule",
                        "model": m,
                        "overrides": {"train": {"max_epochs": ep,
                                                "early_stopping": {"patience": pat}}}})
    return out


def finetune_specs(models):
    out = []
    for m in models:
        out.append({"experiment_id": f"{m}__ft__A_head3", "family": "finetune", "model": m,
                    "overrides": {"transfer_learning": {"stage_a_epochs": 3, "differential_lr": None}}})
        out.append({"experiment_id": f"{m}__ft__B_head5", "family": "finetune", "model": m,
                    "overrides": {"transfer_learning": {"stage_a_epochs": 5, "differential_lr": None}}})
        out.append({"experiment_id": f"{m}__ft__C_diffLR", "family": "finetune", "model": m,
                    "overrides": {"transfer_learning": {"stage_a_epochs": 3,
                                                        "differential_lr": {"backbone_lr": 1.0e-5,
                                                                            "head_lr": 1.0e-4}}}})
    return out


def loss_specs(models):
    losses = [("wce", "weighted_cross_entropy"), ("focal", "focal"),
              ("classbal", "class_balanced")]
    out = []
    for m in models:
        for tag, name in losses:
            out.append({"experiment_id": f"{m}__loss__{tag}", "family": "loss", "model": m,
                        "overrides": {"loss": {"name": name}}})
    return out


def attention_specs(models=("resnet18_spatial_attention", "convnext_tiny_spatial_attention")):
    return [{"experiment_id": f"{m}__attn__baseline", "family": "attention",
             "model": m, "overrides": {}} for m in models]


def multiseed_specs(models, seeds=(42, 123, 3407), base_overrides=None):
    base_overrides = base_overrides or {}
    out = []
    for m in models:
        for s in seeds:
            ov = copy.deepcopy(base_overrides)
            ov["seed"] = s
            out.append({"experiment_id": f"{m}__seed__{s}", "family": "multiseed",
                        "model": m, "overrides": ov})
    return out


FAMILIES = {
    "preprocessing": preprocessing_specs,
    "schedule": schedule_specs,
    "finetune": finetune_specs,
    "loss": loss_specs,
    "attention": attention_specs,
    "multiseed": multiseed_specs,
}


# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def result_to_experiment_row(spec: dict, cfg: dict, result: dict) -> dict:
    tl = cfg["transfer_learning"]
    tr = cfg["train"]
    return {
        "experiment_id": spec["experiment_id"], "family": spec["family"],
        "model": spec["model"],
        "preprocessing": cfg["preprocessing_variant"],
        "loss": cfg["loss"]["name"],
        "optimizer": cfg["optimizer"]["name"],
        "learning_rate": (f"bb={tl['differential_lr']['backbone_lr']},head={tl['differential_lr']['head_lr']}"
                          if tl.get("differential_lr") else cfg["optimizer"]["lr"]),
        "stage_a_epochs": tl["stage_a_epochs"],
        "max_epochs": tr["max_epochs"], "patience": tr["early_stopping"]["patience"],
        "differential_lr": bool(tl.get("differential_lr")),
        "augmentation": Path(cfg["data"]["augmentation_yaml"]).name,
        "seed": cfg["seed"],
        "best_epoch": result.get("best_epoch"), "epochs_run": result.get("epochs_run"),
        "val_accuracy": result.get("accuracy"),
        "val_macro_f1": result.get("macro_f1"),
        "val_qwk": result.get("quadratic_weighted_kappa"),
        "val_balanced_accuracy": result.get("balanced_accuracy"),
        "KL0_F1": result.get("kl0_f1"), "KL1_F1": result.get("kl1_f1"),
        "KL2_F1": result.get("kl2_f1"), "KL3_F1": result.get("kl3_f1"),
        "KL4_F1": result.get("kl4_f1"),
        "parameters": result.get("parameters"), "gmacs": result.get("gmacs"),
        "latency_ms_gpu": (result.get("mean_latency_ms")
                           if result.get("latency_device") == "cuda" else None),
        "latency_ms_cpu": result.get("cpu_mean_latency_ms"),
        "training_time_sec": result.get("training_time_sec"),
        "stopped_early": result.get("stopped_early"),
        "checkpoint": result.get("artefacts", {}).get("best_ckpt"),
    }


def append_row(csv_path: Path, row: dict, *, force: bool = False) -> bool:
    """Append one experiment row. Skips (returns False) if experiment_id already
    present and not force. Never rewrites existing rows."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            existing = {r["experiment_id"] for r in csv.DictReader(fh)}
    if row["experiment_id"] in existing and not force:
        return False
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPERIMENT_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in EXPERIMENT_COLUMNS})
    return True
