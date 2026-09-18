"""PHASE 7 - MobileNetV2 edge-optimization: controlled experiment matrix.

Each spec changes exactly ONE dimension vs configs/benchmark_phase7.yaml (== M0),
except the explicitly-labelled "combine" rows (M7/M8) which stack the dimensions
that beat M0 on VALIDATION Macro-F1 for that specific model.

Spec dict shape (consumed by scripts/benchmark_phase7.py):
    {"experiment_id", "family", "model", "overrides": <partial cfg>, "note"}

The test split is never referenced here.
"""
from __future__ import annotations

import copy

# --------------------------------------------------------------------------- #
# Reusable single-dimension override fragments
# --------------------------------------------------------------------------- #
DIM = {
    "difflr": {                      # gentler backbone unfreeze for a tiny net
        "transfer_learning": {"stage_a_epochs": 5,
                              "differential_lr": {"backbone_lr": 1.0e-5, "head_lr": 1.0e-3}},
        "optimizer": {"weight_decay": 5.0e-5},
    },
    "focal":      {"loss": {"name": "focal", "focal_gamma": 2.0}},
    "classbal":   {"loss": {"name": "class_balanced"}},
    "sqrtsampler": {"data": {"imbalance": "sqrt_inv_freq"}},
    "clahe":      {"preprocessing_variant": "clahe"},
    "strongaug":  {"data": {"augmentation_yaml": "configs/augmentation_phase7_strong.yaml"}},
}
EMA_FRAG = {"train": {"ema": {"decay": 0.999}}}
CORN_FRAG = {"loss": {"name": "corn"}}          # runner builds a 4-logit head

# id -> ordered list of single-dimension keys to test for each model
MATRIX = {
    "mobilenet_v2":   ["difflr", "focal", "classbal", "sqrtsampler", "clahe", "strongaug"],
    "efficientnet_b0": ["difflr", "strongaug"],
    "convnext_tiny":  ["difflr", "strongaug"],
}
# models that also get a standalone CORN ordinal run in the matrix stage
CORN_MODELS = ("mobilenet_v2", "efficientnet_b0", "convnext_tiny")

# dimensions eligible to be stacked into the M7 "combine" row (CE-loss family only;
# focal/classbal are mutually exclusive losses so only the better of the two is kept)
STACKABLE = ("difflr", "sqrtsampler", "clahe", "strongaug")
LOSS_DIMS = ("focal", "classbal")

MIN_GAIN = 0.002        # a dimension must beat M0 val Macro-F1 by this to be adopted


def _mid(model: str, key: str) -> str:
    return f"{model}__{key}"


def matrix_specs() -> list[dict]:
    out = []
    for model, keys in MATRIX.items():
        out.append({"experiment_id": _mid(model, "M0_baseline"), "family": "matrix",
                    "model": model, "overrides": {}, "note": "Phase-4 recipe control"})
        for k in keys:
            out.append({"experiment_id": _mid(model, f"M_{k}"), "family": "matrix",
                        "model": model, "overrides": copy.deepcopy(DIM[k]),
                        "note": f"single dimension: {k}"})
    for model in CORN_MODELS:
        out.append({"experiment_id": _mid(model, "M_corn"), "family": "matrix",
                    "model": model, "overrides": copy.deepcopy(CORN_FRAG),
                    "note": "CORN ordinal head (4 logits)"})
    return out


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def combine_specs(registry_rows: list[dict]) -> list[dict]:
    """registry_rows: list of dicts with keys experiment_id, model, val_macro_f1.

    For each model: adopt every STACKABLE dimension that beat its M0 by >= MIN_GAIN,
    plus the better of {focal, classbal} if it beat M0. Emit M7_stack and
    M8_stack_ema. Also emit M_corn_ema (CORN + EMA + any adopted sampler/aug dims)
    when that model ran CORN.
    """
    by_model: dict[str, dict[str, float]] = {}
    for r in registry_rows:
        eid = r["experiment_id"]
        if "__M" not in eid:
            continue
        model = eid.split("__", 1)[0]
        key = eid.split("__", 1)[1]
        by_model.setdefault(model, {})[key] = float(r.get("val_macro_f1") or 0.0)

    out = []
    for model, scores in by_model.items():
        base = scores.get("M0_baseline", 0.0)
        adopted = [k for k in STACKABLE
                   if scores.get(f"M_{k}", 0.0) >= base + MIN_GAIN]
        loss_pick = None
        loss_cands = {k: scores.get(f"M_{k}", 0.0) for k in LOSS_DIMS if f"M_{k}" in scores}
        if loss_cands:
            bk = max(loss_cands, key=loss_cands.get)
            if loss_cands[bk] >= base + MIN_GAIN:
                loss_pick = bk

        stack = {}
        for k in adopted:
            stack = _deep_merge(stack, DIM[k])
        if loss_pick:
            stack = _deep_merge(stack, DIM[loss_pick])
        note = f"stack: {adopted + ([loss_pick] if loss_pick else []) or ['(none beat M0 -> == M0)']}"

        out.append({"experiment_id": _mid(model, "M7_stack"), "family": "combine",
                    "model": model, "overrides": stack, "note": note})
        out.append({"experiment_id": _mid(model, "M8_stack_ema"), "family": "combine",
                    "model": model, "overrides": _deep_merge(stack, EMA_FRAG),
                    "note": note + " + EMA"})

        if f"M_corn" in scores:
            corn_stack = copy.deepcopy(CORN_FRAG)
            for k in adopted:
                if k in ("sqrtsampler", "strongaug", "difflr"):
                    corn_stack = _deep_merge(corn_stack, DIM[k])
            corn_stack = _deep_merge(corn_stack, EMA_FRAG)
            out.append({"experiment_id": _mid(model, "M_corn_ema"), "family": "combine",
                        "model": model, "overrides": corn_stack,
                        "note": "CORN + EMA + adopted structural dims"})
    return out


def multiseed_specs(finalists: list[dict], seeds=(123, 3407)) -> list[dict]:
    """finalists: [{'experiment_id','model','overrides'}, ...] (seed 42 already run)."""
    out = []
    for f in finalists:
        for s in seeds:
            ov = _deep_merge(f["overrides"], {"seed": s})
            out.append({"experiment_id": f"{f['experiment_id']}__seed{s}", "family": "multiseed",
                        "model": f["model"], "overrides": ov,
                        "note": f"{f['experiment_id']} @ seed {s}"})
    return out


REGISTRY_COLUMNS = [
    "experiment_id", "family", "model", "seed", "ordinal", "ema",
    "preprocessing", "loss", "imbalance", "augmentation", "differential_lr",
    "stage_a_epochs", "weight_decay", "max_epochs", "patience",
    "best_epoch", "epochs_run", "stopped_early",
    "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_balanced_accuracy",
    "val_qwk", "val_mae", "val_within1",
    "val_KL0_F1", "val_KL1_F1", "val_KL2_F1", "val_KL3_F1", "val_KL4_F1",
    "val_binary_oa_acc", "val_binary_oa_auc", "val_3class_acc",
    "parameters", "model_size_mb", "gmacs",
    "gpu_latency_ms", "cpu_latency_ms", "training_time_sec", "checkpoint", "note",
]
