"""PHASE 5 - extra metrics NOT already produced by ``metrics.compute_all``.

Reuses ``metrics.compute_all`` for accuracy / macro & weighted F1 / per-class
P-R-F1 / balanced accuracy / QWK / confusion matrix, and ADDS only what the
Phase-5 spec requires on top:

  * ordinal    : MAE, exact accuracy, within-+/-1 accuracy,
                 |pred-true| error histogram (0..4), under/over-prediction counts
  * weighted   : weighted precision, weighted recall  (compute_all has weighted_f1 only)
  * micro      : micro precision/recall/f1 (== accuracy for single-label multiclass;
                 reported but flagged non-primary)
  * confidence : mean/median/min/max of max-softmax; split by correct / incorrect
  * class-4    : the KL4 -> {KL0..KL4} prediction breakdown (severe-grade detection)

Nothing here modifies training-infrastructure files.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

CLASS_LABELS = [0, 1, 2, 3, 4]


def ordinal_and_confidence(y_true, y_pred, y_prob=None) -> dict:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    n = int(yt.size)
    err = np.abs(yp - yt)
    signed = yp - yt

    out: dict = {
        "n_samples": n,
        "mae": float(err.mean()),
        "exact_accuracy": float((err == 0).mean()),
        "within_1_accuracy": float((err <= 1).mean()),
        "within_2_accuracy": float((err <= 2).mean()),
        "rmse_ordinal": float(np.sqrt((err.astype(float) ** 2).mean())),
        # |pred - true| histogram
        **{f"error_{k}": int((err == k).sum()) for k in range(5)},
        "underprediction_count": int((signed < 0).sum()),   # predicted a LOWER grade than truth
        "overprediction_count": int((signed > 0).sum()),    # predicted a HIGHER grade than truth
        "correct_count": int((err == 0).sum()),
        # weighted precision / recall (compute_all only has weighted_f1)
        "weighted_precision": float(precision_score(yt, yp, labels=CLASS_LABELS,
                                                    average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(yt, yp, labels=CLASS_LABELS,
                                              average="weighted", zero_division=0)),
        # micro (== accuracy for single-label multiclass) - reported, NOT primary
        "micro_precision": float(precision_score(yt, yp, labels=CLASS_LABELS,
                                                 average="micro", zero_division=0)),
        "micro_recall": float(recall_score(yt, yp, labels=CLASS_LABELS,
                                           average="micro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, labels=CLASS_LABELS,
                                   average="micro", zero_division=0)),
    }

    # adjacent vs large ordinal jump split
    out["adjacent_errors"] = int(((err == 1)).sum())
    out["large_jump_errors"] = int((err >= 2).sum())
    out["adjacent_error_fraction_of_errors"] = (
        float((err == 1).sum() / max((err > 0).sum(), 1)))

    # KL4 -> predicted breakdown (severe-grade detection)
    kl4_mask = yt == 4
    out["kl4_support"] = int(kl4_mask.sum())
    out["kl4_predicted_as"] = {f"KL{c}": int(((yt == 4) & (yp == c)).sum()) for c in range(5)}

    if y_prob is not None:
        pr = np.asarray(y_prob, dtype=float)
        conf = pr.max(axis=1)                       # max-softmax = model confidence
        correct = err == 0
        out["confidence"] = {
            "mean": float(conf.mean()),
            "median": float(np.median(conf)),
            "min": float(conf.min()),
            "max": float(conf.max()),
            "std": float(conf.std()),
            "mean_when_correct": float(conf[correct].mean()) if correct.any() else None,
            "mean_when_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
            "median_when_correct": float(np.median(conf[correct])) if correct.any() else None,
            "median_when_incorrect": float(np.median(conf[~correct])) if (~correct).any() else None,
        }
    return out


def most_common_confusions(cm, top: int = 8) -> list[dict]:
    """Off-diagonal cells of a 5x5 raw confusion matrix, largest first."""
    cm = np.asarray(cm)
    pairs = []
    for i in range(5):
        for j in range(5):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": i, "pred": j, "count": int(cm[i, j]),
                              "ordinal_distance": abs(i - j)})
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top]
