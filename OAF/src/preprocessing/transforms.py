"""Pure image transforms for the X-ray preprocessing pipeline.

Every function takes and returns a ``numpy.ndarray`` (uint8 unless noted) and has
no torch dependency, so the audit / prepare stages run without a DL stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_RESAMPLE = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


class ImageValidationError(ValueError):
    """Raised when an image fails a hard validation check."""


def load_grayscale(path: str | Path) -> np.ndarray:
    """Read an image from disk as single-channel uint8 (H, W)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageValidationError(f"unreadable image: {path}")
    if arr.ndim == 3:
        # OpenCV loads BGR; for a replicated-gray X-ray any channel is equal,
        # but do a proper luminance conversion to be safe.
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint16:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def validate_image(arr: np.ndarray, min_dim: int = 32,
                   min_dynamic_range: int = 5) -> np.ndarray:
    """Hard checks - raise on clearly unusable data. Soft flags live in the audit."""
    if arr.ndim != 2:
        raise ImageValidationError(f"expected 2-D grayscale, got shape {arr.shape}")
    h, w = arr.shape
    if min(h, w) < min_dim:
        raise ImageValidationError(f"image too small: {w}x{h}")
    if int(arr.max()) - int(arr.min()) < min_dynamic_range:
        raise ImageValidationError("degenerate image (near-constant intensity)")
    return arr


def center_crop(arr: np.ndarray, frac: float) -> np.ndarray:
    h, w = arr.shape[:2]
    ch, cw = int(round(h * frac)), int(round(w * frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def pad_to_square(arr: np.ndarray, fill: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(arr, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=fill)


def resize(arr: np.ndarray, size: Tuple[int, int], resample: str = "bilinear") -> np.ndarray:
    """size = (H, W). Uses INTER_AREA automatically when downscaling for quality."""
    h, w = arr.shape[:2]
    target_h, target_w = size
    interp = _RESAMPLE.get(resample, cv2.INTER_LINEAR)
    if target_h < h or target_w < w:
        interp = cv2.INTER_AREA
    return cv2.resize(arr, (target_w, target_h), interpolation=interp)


def clahe(arr: np.ndarray, clip_limit: float = 2.0,
          tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
    return op.apply(arr)


def hist_equalize(arr: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(arr)


def hist_match(arr: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the intensity histogram of ``arr`` to ``reference`` (both uint8)."""
    src_hist, _ = np.histogram(arr.ravel(), 256, (0, 256))
    ref_hist, _ = np.histogram(reference.ravel(), 256, (0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64); src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist).astype(np.float64); ref_cdf /= ref_cdf[-1]
    lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return lut[arr]


def to_float(arr: np.ndarray, out_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    lo, hi = out_range
    return arr.astype(np.float32) / 255.0 * (hi - lo) + lo


def replicate_channels(arr: np.ndarray, n: int = 3) -> np.ndarray:
    """(H, W) -> (H, W, n) by replication. NEVER colourises."""
    if n == 1:
        return arr[..., None] if arr.ndim == 2 else arr
    if arr.ndim == 2:
        return np.repeat(arr[..., None], n, axis=2)
    return arr
