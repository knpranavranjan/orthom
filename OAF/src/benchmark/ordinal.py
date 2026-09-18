"""CORN ordinal-regression head/loss for KL-grade (Phase 7, opt-in).

CORN = Conditional Ordinal Regression for Neural networks (Shi, Cao, Raschka 2021).
The classifier head emits K-1 = 4 logits instead of K = 5. Logit k models the
*conditional* probability  P(y > k | y >= k).  Training uses only the subset of a
mini-batch that satisfies  y >= k  for task k, which removes the rank-inconsistency
of plain CORAL and keeps calibrated conditional probabilities.

Nothing here is imported by Phases 1-6. `engine.fit` only touches this module when
`cfg["loss"]["name"] == "corn"`; `engine.evaluate` only when `ordinal="corn"` is
passed explicitly. Default code paths are unchanged.

Build the backbone with `build_model(..., num_classes=4)` when using CORN.
"""
from __future__ import annotations

import numpy as np

NUM_TASKS = 4          # K-1 for a 5-class ordinal target
NUM_CLASSES = 5


def corn_loss(logits, targets):
    """CORN conditional loss.

    logits  : (N, 4) float  -- raw, pre-sigmoid
    targets : (N,)   long   -- KL grade in {0,1,2,3,4}
    Returns a scalar tensor (mean over the non-empty conditional tasks).
    """
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    n = logits.size(0)
    total = logits.new_zeros(())
    n_tasks = 0
    for k in range(NUM_TASKS):
        mask = targets >= k                      # rows still "in play" for task k
        if mask.sum() == 0:
            continue
        # binary label for task k on the conditioned subset: 1 if y > k
        bin_label = (targets[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[mask, k], bin_label, reduction="mean")
        n_tasks += 1
    return total / max(n_tasks, 1)


def _conditional_probs(logits: np.ndarray) -> np.ndarray:
    """(N,4) logits -> (N,4) UNconditional P(y > k), enforced non-increasing."""
    cond = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))   # P(y>k | y>=k)
    p_gt = np.cumprod(cond, axis=1)                            # P(y>k)
    return p_gt


def corn_logits_to_probs(logits) -> np.ndarray:
    """(N,4) CORN logits -> (N,5) proper class-probability matrix (rows sum to 1).

    P(y=0)   = 1 - P(y>0)
    P(y=k)   = P(y>k-1) - P(y>k)      for 1 <= k <= 3
    P(y=4)   = P(y>3)
    """
    logits = np.asarray(logits, dtype=np.float64)
    p_gt = _conditional_probs(logits)                          # (N,4)
    n = logits.shape[0]
    probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for k in range(1, NUM_TASKS):
        probs[:, k] = p_gt[:, k - 1] - p_gt[:, k]
    probs[:, NUM_TASKS] = p_gt[:, NUM_TASKS - 1]
    probs = np.clip(probs, 1e-8, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def corn_logits_to_label(logits) -> np.ndarray:
    """(N,4) CORN logits -> (N,) integer KL grade via the standard rank rule:
    y_hat = number of thresholds with P(y>k) > 0.5 (contiguous from the left)."""
    logits = np.asarray(logits, dtype=np.float64)
    p_gt = _conditional_probs(logits)
    passed = p_gt > 0.5
    # count the leading run of True so the prediction stays rank-consistent
    y = np.zeros(logits.shape[0], dtype=np.int64)
    for i in range(logits.shape[0]):
        c = 0
        for k in range(NUM_TASKS):
            if passed[i, k]:
                c += 1
            else:
                break
        y[i] = c
    return y
