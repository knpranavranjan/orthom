"""Compose the deterministic preprocessing pipeline from a YAML config.

Used for train / val / test alike (it is deterministic - no randomness).
Augmentation is a separate train-only stage applied at load time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from . import transforms as T


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class DeterministicPreprocessor:
    """Read -> grayscale -> validate -> ROI -> (pad) -> resize -> contrast.

    Output is a uint8 ndarray of shape (H, W) - contrast-enhanced, resized.
    Float scaling, intensity normalization and channel replication are applied
    later at tensor-build time (kept out of the on-disk copy so it stays lossless).
    """

    cfg: Dict[str, Any]
    contrast_override: str | None = None
    _img_cfg: dict = field(init=False)
    _roi_cfg: dict = field(init=False)
    _contrast_cfg: dict = field(init=False)

    def __post_init__(self) -> None:
        self._img_cfg = self.cfg["image"]
        self._roi_cfg = self.cfg["roi"]
        self._contrast_cfg = self.cfg["contrast"]

    # -- individual stages ------------------------------------------------- #
    def _roi(self, arr: np.ndarray) -> np.ndarray:
        mode = self._roi_cfg.get("mode", "provided")
        if mode == "center_crop":
            return T.center_crop(arr, float(self._roi_cfg.get("center_crop_frac", 0.9)))
        # "full" and "provided" are both no-ops for already-cropped knee ROIs
        return arr

    def _contrast(self, arr: np.ndarray) -> np.ndarray:
        method = self.contrast_override or self._contrast_cfg.get("method", "none")
        if method in ("none", None):
            return arr
        if method == "clahe":
            c = self._contrast_cfg.get("clahe", {})
            return T.clahe(arr, float(c.get("clip_limit", 2.0)),
                           tuple(c.get("tile_grid_size", (8, 8))))
        if method in ("hist_eq", "histeq", "hist_equalize"):
            return T.hist_equalize(arr)
        raise ValueError(f"unknown contrast method: {method}")

    # -- full pipeline --------------------------------------------------- #
    def process_array(self, arr: np.ndarray) -> np.ndarray:
        arr = T.validate_image(arr)
        arr = self._roi(arr)
        if self._img_cfg.get("pad_to_square_before_resize", True):
            arr = T.pad_to_square(arr, fill=0)
        arr = T.resize(arr, tuple(self._img_cfg["size"]),
                       self._img_cfg.get("resample", "bilinear"))
        arr = self._contrast(arr)
        return arr

    def process_path(self, path: str | Path) -> np.ndarray:
        return self.process_array(T.load_grayscale(path))

    # -- optional: produce a model-ready float tensor-like ndarray ------- #
    def to_model_array(self, arr_uint8: np.ndarray,
                       norm_stats: Dict[str, Any] | None = None) -> np.ndarray:
        """(H,W) uint8 -> (C,H,W) float32, normalized per config.

        norm_stats overrides cfg['intensity_normalization']['dataset_grayscale'].
        """
        ncfg = self.cfg["intensity_normalization"]
        rng = tuple(ncfg.get("to_float_range", (0.0, 1.0)))
        x = T.to_float(arr_uint8, rng)                      # (H,W) in [0,1]
        out_ch = int(self.cfg["channels"].get("output_channels", 3))
        x = T.replicate_channels(x, out_ch)                 # (H,W,C)

        mode = ncfg.get("mode", "dataset_grayscale")
        if mode == "imagenet":
            mean = np.array(ncfg["imagenet"]["mean"], dtype=np.float32)
            std = np.array(ncfg["imagenet"]["std"], dtype=np.float32)
        elif mode == "dataset_grayscale":
            ds = norm_stats or ncfg["dataset_grayscale"]
            m = np.array(ds["mean"], dtype=np.float32)
            s = np.array(ds["std"], dtype=np.float32)
            mean = np.repeat(m, out_ch) if m.size == 1 else m
            std = np.repeat(s, out_ch) if s.size == 1 else s
        elif mode == "minmax":
            mean = np.zeros(out_ch, np.float32); std = np.ones(out_ch, np.float32)
        elif mode == "zscore_per_image":
            mean = np.full(out_ch, float(x.mean()), np.float32)
            std = np.full(out_ch, float(x.std()) or 1.0, np.float32)
        else:
            raise ValueError(f"unknown normalization mode: {mode}")

        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1)).astype(np.float32)  # (C,H,W)


def variants_from_config(cfg: Dict[str, Any]) -> list[dict]:
    """Expand cfg['variants'] into concrete per-variant contrast settings."""
    out = []
    for v in cfg.get("variants", [{"name": "basic", "contrast": "none"}]):
        out.append({"name": v["name"], "contrast": v.get("contrast", "none")})
    return out
