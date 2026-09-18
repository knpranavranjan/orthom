"""Deployment preprocessing - identical to Phase-1 `basic` + the training-time
load transform, but torch-free (numpy + OpenCV only) so it runs on a Raspberry Pi.

Pipeline for an arbitrary input image:
    read -> grayscale -> letterbox to square (fill 0) -> resize 224x224 (INTER_AREA
    on downscale, matching scripts/prepare_dataset.py) -> float32/255 -> replicate
    to 3 channels -> ImageNet normalisation -> (1,3,224,224) float32
"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


def _to_gray_uint8(img) -> np.ndarray:
    """Accept a path, a PIL.Image, or an HxW / HxWxC numpy array -> HxW uint8."""
    import cv2
    if isinstance(img, str):
        a = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"cannot read image: {img}")
    elif hasattr(img, "convert"):                       # PIL.Image
        a = np.array(img)
    else:
        a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)     # RGB/BGR both -> luminance
    if a.dtype == np.uint16:
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(a, 0, 255).astype(np.uint8)


def letterbox_square(a: np.ndarray, fill: int = 0) -> np.ndarray:
    import cv2
    h, w = a.shape[:2]
    if h == w:
        return a
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(a, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=fill)


def resize_224(a: np.ndarray) -> np.ndarray:
    import cv2
    interp = cv2.INTER_AREA if (a.shape[0] > SIZE or a.shape[1] > SIZE) else cv2.INTER_LINEAR
    return cv2.resize(a, (SIZE, SIZE), interpolation=interp)


def preprocess(img, *, already_224: bool = False) -> np.ndarray:
    """Return a (1, 3, 224, 224) float32 tensor ready for the ONNX model.

    already_224=True skips letterbox+resize (use for the pre-processed
    data/processed/images/basic/*.png test files, for exact benchmark parity).
    """
    g = _to_gray_uint8(img)
    if not already_224:
        g = resize_224(letterbox_square(g))
    elif g.shape[:2] != (SIZE, SIZE):
        g = resize_224(letterbox_square(g))
    x = g.astype(np.float32) / 255.0                     # (H,W) in [0,1]
    x = np.repeat(x[None], 3, axis=0)                    # (3,H,W) replicated grayscale
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return x[None].astype(np.float32)                    # (1,3,H,W)
