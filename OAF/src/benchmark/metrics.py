"""All required classification metrics (spec sections 17-20), one implementation
used for every model.

Inputs are always ``y_true`` (int array), ``y_pred`` (int array) and optionally
``y_prob`` (N x 5 array of softmax probabilities).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_LABELS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]


def compute_all(y_true, y_pred, y_prob=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    out: dict = {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_LABELS,
                                                 average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS,
                                           average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS,
                                      average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=CLASS_LABELS)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=CLASS_LABELS, weights="quadratic")),
    }

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    out["per_class"] = {
        CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(s[i])}
        for i in range(5)
    }
    for i in range(5):
        out[f"kl{i}_f1"] = float(f[i])
        out[f"kl{i}_precision"] = float(p[i])
        out[f"kl{i}_recall"] = float(r[i])
        out[f"kl{i}_support"] = int(s[i])

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    out["confusion_matrix"] = cm.tolist()
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    out["confusion_matrix_normalized"] = np.nan_to_num(cmn).tolist()

    # neighbouring-grade confusion (spec section 20)
    out["adjacent_confusions"] = {
        "KL1_as_KL2": int(cm[1, 2]), "KL2_as_KL1": int(cm[2, 1]),
        "KL2_as_KL3": int(cm[2, 3]), "KL3_as_KL2": int(cm[3, 2]),
        "KL3_as_KL4": int(cm[3, 4]), "KL4_as_KL3": int(cm[4, 3]),
    }
    off = cm.copy()
    np.fill_diagonal(off, 0)
    dist1 = sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) == 1)
    dist_ge2 = sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) >= 2)
    out["errors_off_by_1"] = int(dist1)
    out["errors_off_by_2plus"] = int(dist_ge2)

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        present = np.unique(y_true)
        try:
            if len(present) == 5:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                  labels=CLASS_LABELS))
                out["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted",
                                  labels=CLASS_LABELS))
                pc = {}
                for i in range(5):
                    yi = (y_true == i).astype(int)
                    pc[CLASS_NAMES[i]] = (float(roc_auc_score(yi, y_prob[:, i]))
                                          if yi.sum() and yi.sum() != yi.size else None)
                out["roc_auc_per_class_ovr"] = pc
            else:
                out["roc_auc_ovr_macro"] = None
                out["roc_auc_note"] = f"only classes {present.tolist()} present; macro OVR-AUC undefined"
        except ValueError as exc:
            out["roc_auc_ovr_macro"] = None
            out["roc_auc_note"] = f"undefined: {exc}"
    return out


def flat_row(model_name: str, m: dict) -> dict:
    """Flatten compute_all() output into one CSV row (metric subset)."""
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "balanced_accuracy", "cohen_kappa", "quadratic_weighted_kappa",
            "kl0_f1", "kl1_f1", "kl2_f1", "kl3_f1", "kl4_f1",
            "kl0_support", "kl1_support", "kl2_support", "kl3_support", "kl4_support",
            "roc_auc_ovr_macro", "errors_off_by_1", "errors_off_by_2plus"]
    row = {"model": model_name}
    row.update({k: m.get(k) for k in keys})
    return row
