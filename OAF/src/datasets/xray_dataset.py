"""Model-agnostic knee-X-ray dataset (PHASE 2).

Contract: every candidate CNN receives ``(image_tensor, label)`` and returns
``logits[5]``.  This dataset yields:

    image : float32 tensor, shape (C, H, W)   C in {1, 3} per config
    label : int64 scalar in [0, 4]   (KL grade)
    meta  : dict(sample_id, patient_id, knee_side, split)   [when return_meta=True]

* Deterministic preprocessing is baked into the on-disk processed PNGs by
  ``scripts/prepare_dataset.py`` (Phase 1). This loader adds ONLY: float scaling
  -> [0,1], (train-only) clinically constrained augmentation, intensity
  normalization, and 1->3 channel replication.
* All classes are module-level and picklable so ``num_workers > 0`` works under
  the Windows 'spawn' start method.

This module imports torch at import time; only Phase 2+ code imports it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.common import PROJECT_ROOT
from src.preprocessing.pipeline import load_config

try:  # keep the module importable even if the benchmark package layout changes
    from src.benchmark.seeding import seed_worker
except Exception:  # noqa: BLE001
    def seed_worker(worker_id: int) -> None:  # fallback: seed numpy/random per worker
        import random as _r
        s = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(s)
        _r.seed(s)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Augmentation (train only) - a picklable callable, NOT a closure.
# --------------------------------------------------------------------------- #
class TrainAugmentor:
    """Clinically-constrained augmentation from configs/augmentation.yaml.

    Operates on a float tensor (C, H, W) in [0, 1] BEFORE normalization.
    """

    def __init__(self, aug_cfg: dict):
        from torchvision.transforms import v2

        p = aug_cfg.get("pipeline", {})
        self.enabled = bool(aug_cfg.get("enabled", False))
        aff: Dict[str, Any] = {}
        if p.get("random_rotation", {}).get("enabled"):
            aff["degrees"] = tuple(p["random_rotation"]["degrees"])
        if p.get("random_translation", {}).get("enabled"):
            aff["translate"] = tuple(p["random_translation"]["max_fraction"])
        if p.get("random_scale", {}).get("enabled"):
            aff["scale"] = tuple(p["random_scale"]["range"])
        if p.get("random_affine_shear", {}).get("enabled"):
            aff["shear"] = tuple(p["random_affine_shear"]["degrees"])
        ops = []
        if aff:
            aff.setdefault("degrees", 0)
            ops.append(v2.RandomAffine(**aff))
        if p.get("horizontal_flip", {}).get("enabled"):
            ops.append(v2.RandomHorizontalFlip(p=float(p["horizontal_flip"].get("p", 0.5))))
        bc = p.get("brightness_contrast_jitter", {})
        if bc.get("enabled"):
            b = bc.get("brightness", [0.9, 1.1])
            c = bc.get("contrast", [0.9, 1.1])
            ops.append(v2.ColorJitter(brightness=(float(b[0]), float(b[1])),
                                      contrast=(float(c[0]), float(c[1]))))
        self._core = v2.Compose(ops) if ops else None
        gn = p.get("gaussian_noise", {})
        self._gn_p = float(gn.get("p", 0.0)) if gn.get("enabled") else 0.0
        self._gn_std = float(gn.get("std_range", [0.0, 0.02])[1])
        self.disabled = sorted(k for k, v in p.items()
                               if isinstance(v, dict) and v.get("enabled") is False)

    def __call__(self, t: "torch.Tensor") -> "torch.Tensor":
        if not self.enabled:
            return t
        if self._core is not None:
            t = self._core(t)
        if self._gn_p and torch.rand(1).item() < self._gn_p:
            t = torch.clamp(t + torch.randn_like(t) * self._gn_std, 0.0, 1.0)
        return t


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class KneeXrayDataset(Dataset):
    """One split of a processed_manifest_<variant>.csv."""

    def __init__(self, manifest_csv, split: str, normalization: str = "imagenet",
                 out_channels: int = 3, float_range=(0.0, 1.0),
                 norm_stats: Optional[dict] = None,
                 augmentor: Optional[TrainAugmentor] = None,
                 return_meta: bool = False,
                 label_map: Optional[dict] = None):
        df = pd.read_csv(manifest_csv, dtype={"patient_id": str})
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no rows for split={split!r} in {manifest_csv}")
        self.split = split
        # optional KL-grade -> merged-label remap (Phase-8 label-scheme sweep).
        # e.g. {0:0,1:0,2:1,3:2,4:3} folds KL0+KL1. None -> raw 0..4.
        self.label_map = {int(k): int(v) for k, v in label_map.items()} if label_map else None
        self.out_ch = int(out_channels)
        self.lo, self.hi = float(float_range[0]), float(float_range[1])
        self.return_meta = bool(return_meta)
        self.augmentor = augmentor if split == "train" else None
        self.normalization = normalization

        if normalization == "imagenet":
            mean = np.array(_IMAGENET_MEAN, np.float32)
            std = np.array(_IMAGENET_STD, np.float32)
        elif normalization == "dataset_grayscale":
            s = norm_stats or {"mean": 0.5, "std": 0.25}
            mean = np.repeat(np.float32(s["mean"]), self.out_ch)
            std = np.repeat(np.float32(s["std"]), self.out_ch)
        elif normalization == "minmax":
            mean = np.zeros(self.out_ch, np.float32)
            std = np.ones(self.out_ch, np.float32)
        else:
            raise ValueError(f"unknown normalization {normalization!r}")
        self._mean = torch.from_numpy(mean.reshape(-1, 1, 1))
        self._std = torch.from_numpy(std.reshape(-1, 1, 1))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        path = PROJECT_ROOT / r["processed_path"]
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(path)
        x = arr.astype(np.float32) / 255.0 * (self.hi - self.lo) + self.lo   # (H,W)
        x = np.repeat(x[None], self.out_ch, axis=0) if self.out_ch > 1 else x[None]
        t = torch.from_numpy(np.ascontiguousarray(x))                        # (C,H,W)
        if self.augmentor is not None:
            t = self.augmentor(t)
        t = (t - self._mean) / self._std
        _lbl = int(r["kl_grade"])
        if self.label_map is not None:
            _lbl = self.label_map.get(_lbl, _lbl)
        label = torch.tensor(_lbl, dtype=torch.long)
        if self.return_meta:
            return t.float(), label, {
                "sample_id": str(r["sample_id"]), "patient_id": str(r["patient_id"]),
                "knee_side": str(r["knee_side"]), "split": str(r["split"])}
        return t.float(), label


# --------------------------------------------------------------------------- #
# Loader factory
# --------------------------------------------------------------------------- #
def _load_norm_stats(variant: str) -> Optional[dict]:
    p = PROJECT_ROOT / "metadata" / "normalization_stats.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("variants", {}).get(variant)
    return {"mean": v["mean"], "std": v["std"]} if v else None


def build_dataloaders(variant: str = "basic",
                      normalization: str = "imagenet",
                      batch_size: int = 64,
                      num_workers: int = 4,
                      out_channels: int = 3,
                      imbalance: str = "weighted_ce",
                      augmentation_yaml: str | Path = "configs/augmentation.yaml",
                      return_meta: bool = False,
                      pin_memory: bool = True,
                      persistent_workers: bool = True,
                      drop_last_train: bool = True,
                      seed: int = 42,
                      label_map: Optional[dict] = None) -> dict:
    """Return {'train','val','test'} DataLoaders + class weights + dataset handles.

    label_map: optional {kl_grade -> merged_label} for the Phase-8 label-scheme
    sweep. When given, class weights + num_classes are recomputed for the merged
    scheme from the TRAIN split only (still leakage-free).
    """
    KX = KneeXrayDataset
    manifest = PROJECT_ROOT / "metadata" / f"processed_manifest_{variant}.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    aug_cfg = load_config(PROJECT_ROOT / augmentation_yaml)
    augmentor = TrainAugmentor(aug_cfg)
    norm_stats = _load_norm_stats(variant) if normalization == "dataset_grayscale" else None

    ds = {s: KX(manifest, s, normalization=normalization, out_channels=out_channels,
                norm_stats=norm_stats, augmentor=augmentor if s == "train" else None,
                return_meta=return_meta, label_map=label_map)
          for s in ("train", "val", "test")}

    if label_map is not None:
        lm = {int(k): int(v) for k, v in label_map.items()}
        n_classes = max(lm.values()) + 1
        tr = ds["train"].df["kl_grade"].map(lambda k: lm.get(int(k), int(k))).to_numpy()
        counts = np.bincount(tr, minlength=n_classes).astype(np.float64)
        # sklearn 'balanced': n_samples / (n_classes * count_c)
        bal = counts.sum() / (n_classes * np.maximum(counts, 1))
        class_weights = torch.tensor(bal, dtype=torch.float32)
    else:
        n_classes = 5
        cw = json.loads((PROJECT_ROOT / "metadata" / "class_weights.json").read_text())
        class_weights = torch.tensor(cw["weights_list_balanced"], dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=persistent_workers and num_workers > 0,
                  worker_init_fn=(seed_worker if num_workers > 0 else None))

    if imbalance in ("weighted_sampler", "sqrt_inv_freq"):
        if label_map is not None:
            _lm = {int(k): int(v) for k, v in label_map.items()}
            labels = ds["train"].df["kl_grade"].map(lambda k: _lm.get(int(k), int(k))).to_numpy()
        else:
            labels = ds["train"].df["kl_grade"].to_numpy()
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        if imbalance == "sqrt_inv_freq":
            # milder than full inverse-frequency: reweight by 1/sqrt(count) so KL1/KL4
            # are lifted toward the mean without letting the model memorise them.
            per_class = 1.0 / np.sqrt(counts)
        else:
            per_class = 1.0 / counts
        sampler = WeightedRandomSampler(per_class[labels], num_samples=len(labels),
                                        replacement=True, generator=g)
        train_loader = DataLoader(ds["train"], batch_size=batch_size, sampler=sampler,
                                  drop_last=drop_last_train, **common)
    else:
        train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                                  drop_last=drop_last_train, generator=g, **common)

    return {
        "train": train_loader,
        "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False, **common),
        "datasets": ds,
        "class_weights": class_weights,
        "num_classes": n_classes,
        "normalization": normalization,
        "variant": variant,
        "disabled_augmentations": augmentor.disabled,
    }
