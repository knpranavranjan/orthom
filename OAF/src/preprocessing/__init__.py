"""Modular, config-driven X-ray preprocessing.

Pipeline order (spec section 14):
    read -> grayscale -> validate -> ROI -> resize -> intensity-normalize
    -> optional contrast enhancement -> channel conversion -> tensor

``transforms`` holds pure numpy/OpenCV ops (no torch).  ``pipeline`` composes
them from ``configs/preprocessing.yaml``.  Tensor conversion + normalization to
float happen at load time in ``src.datasets.xray_dataset`` so the on-disk
processed images stay lossless uint8.
"""
from .transforms import (
    center_crop,
    clahe,
    hist_equalize,
    load_grayscale,
    pad_to_square,
    replicate_channels,
    resize,
    to_float,
    validate_image,
)
from .pipeline import DeterministicPreprocessor, load_config

__all__ = [
    "center_crop", "clahe", "hist_equalize", "load_grayscale", "pad_to_square",
    "replicate_channels", "resize", "to_float", "validate_image",
    "DeterministicPreprocessor", "load_config",
]
