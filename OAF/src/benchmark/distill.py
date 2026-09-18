"""Knowledge-distillation utilities for Phase 8 (MobileNetV2 accuracy push).

Offline KD: teacher LOGITS are pre-computed once per processed image (Phase-8
teacher bank) and looked up by sample_id during student training. The student is
MobileNetV2 (unchanged 2.23 M params); teachers are the frozen Phase-3/4/7
VGG16 / ConvNeXt-Tiny / ResNet18 ONNX graphs.

Hinton KD:  L = alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
              + (1 - alpha) * CE(student, hard_label; class_weights, label_smoothing)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class TeacherBank:
    """sample_id -> stacked teacher logits (n_teachers, 5), loaded from .npz files."""

    def __init__(self, npz_paths: list[str | Path], weights: list[float] | None = None):
        self.paths = [Path(p) for p in npz_paths]
        self.names = [p.stem.split("__")[-1] for p in self.paths]
        self.weights = np.asarray(weights if weights is not None else [1.0] * len(self.paths),
                                  dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self._logits: dict[str, np.ndarray] = {}
        per_teacher = []
        ids0 = None
        for p in self.paths:
            d = np.load(p, allow_pickle=True)
            ids = [str(s) for s in d["sample_id"]]
            lg = d["logits"].astype(np.float64)
            per_teacher.append(dict(zip(ids, lg)))
            ids0 = set(ids) if ids0 is None else (ids0 & set(ids))
        for sid in ids0:
            self._logits[sid] = np.stack([pt[sid] for pt in per_teacher], axis=0)  # (K,5)
        self.n = len(self._logits)

    def has(self, sid: str) -> bool:
        return sid in self._logits

    def ensemble_soft(self, sids: list[str], T: float, collapse: dict | None = None) -> np.ndarray:
        """Return (B,C) weighted-mean of per-teacher softmax(logits / T).

        collapse: optional {orig_class -> merged_class} (Phase-8 label-scheme sweep).
        The 5-class teacher probability mass is *summed* into the merged columns,
        then renormalised, which is the correct soft target for a merged scheme.
        """
        out = np.zeros((len(sids), 5), dtype=np.float64)
        for i, sid in enumerate(sids):
            L = self._logits[sid] / T                       # (K,5)
            L = L - L.max(axis=1, keepdims=True)
            e = np.exp(L)
            sm = e / e.sum(axis=1, keepdims=True)           # (K,5)
            out[i] = (self.weights[:, None] * sm).sum(axis=0)
        if collapse is not None:
            cmap = {int(k): int(v) for k, v in collapse.items()}
            nc = max(cmap.values()) + 1
            merged = np.zeros((len(sids), nc), dtype=np.float64)
            for oc in range(5):
                merged[:, cmap.get(oc, oc)] += out[:, oc]
            out = merged
        out = np.clip(out, 1e-8, None)
        return out / out.sum(axis=1, keepdims=True)


def kd_loss(student_logits, teacher_soft, hard_labels, *, T: float, alpha: float,
            class_weights=None, label_smoothing: float = 0.0):
    """student_logits (B,5) raw; teacher_soft (B,5) already softmax@T; hard_labels (B,)."""
    import torch
    import torch.nn.functional as F

    s_log_T = F.log_softmax(student_logits / T, dim=1)
    t = teacher_soft.clamp_min(1e-8)
    t = t / t.sum(dim=1, keepdim=True)
    soft = F.kl_div(s_log_T, t, reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, hard_labels, weight=class_weights,
                           label_smoothing=label_smoothing)
    return alpha * soft + (1.0 - alpha) * hard, soft.detach(), hard.detach()


def mixup_batch(x, teacher_soft, hard_onehot, alpha: float, generator=None):
    """Mixup on inputs + soft targets (hard label becomes soft one-hot mix)."""
    import torch

    if alpha <= 0:
        return x, teacher_soft, hard_onehot
    lam = float(np.random.default_rng().beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device, generator=generator)
    x = lam * x + (1 - lam) * x[perm]
    teacher_soft = lam * teacher_soft + (1 - lam) * teacher_soft[perm]
    hard_onehot = lam * hard_onehot + (1 - lam) * hard_onehot[perm]
    return x, teacher_soft, hard_onehot
