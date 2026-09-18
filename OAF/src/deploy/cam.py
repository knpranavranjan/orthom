"""Grad-free Class Activation Map for the deployed MobileNetV2 (torch-free).

MobileNetV2's head is  features -> global-average-pool -> Linear(1280, 5)  with
nothing in between, so Grad-CAM's per-channel weight collapses to the classifier
weight and the map is exactly the original CAM (Zhou et al. 2016):

    CAM_c(x, y)  =  ReLU( sum_k  W[c, k] * feat_k(x, y) )

`feat` is the second output of models/deploy/mobilenet_v2_oa_cam.onnx (1x1280x7x7);
`W` is models/deploy/mobilenet_v2_oa_cam.npz["W"] (5x1280). No autograd, one forward.
"""
from __future__ import annotations

import numpy as np

_SIZE = 224


def class_activation_map(feat: np.ndarray, W: np.ndarray, grade: int) -> np.ndarray:
    """feat: (1,1280,7,7) or (1280,7,7).  W: (5,1280).  -> (7,7) float in [0,1]."""
    f = feat[0] if feat.ndim == 4 else feat
    cam = np.einsum("kij,k->ij", f.astype(np.float64), W[int(grade)].astype(np.float64))
    cam = np.maximum(cam, 0.0)
    m = float(cam.max())
    return (cam / m) if m > 0 else cam


def _resize(cam: np.ndarray, size: int = _SIZE) -> np.ndarray:
    import cv2
    return cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)


def render_overlay(gray224: np.ndarray, cam: np.ndarray, *, alpha: float = 0.42,
                   size: int = _SIZE) -> np.ndarray:
    """gray224: HxW uint8 (the letterboxed 224 X-ray).  cam: 7x7 in [0,1].
    Returns an (H,W,3) uint8 RGB overlay (JET heat over the film)."""
    import cv2
    g = gray224
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    if g.shape[:2] != (size, size):
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    heat = np.clip(_resize(cam, size), 0.0, 1.0)
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1]        # BGR->RGB
    base_rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    a = (heat[..., None] * alpha)                                             # fade where cam is cold
    out = (base_rgb * (1 - a) + heat_rgb * a).clip(0, 255).astype(np.uint8)
    return out


def png_bytes(rgb: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])                            # RGB->BGR for cv2
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def focus_region(cam: np.ndarray, *, hot: float = 0.5) -> dict:
    """Describe where the model is looking from the 7x7 map.

    Returns {region, side_hint, coverage, centroid:[cx,cy], spread}. `region` is
    conservative: without the knee's L/R laterality we cannot call medial vs
    lateral definitively, so an off-centre focus is reported as 'one compartment'.
    """
    c = cam / (cam.max() + 1e-8)
    ys, xs = np.mgrid[0:c.shape[0], 0:c.shape[1]].astype(np.float64)
    w = c.sum()
    cx = float((xs * c).sum() / (w + 1e-8)) / (c.shape[1] - 1)               # 0..1 left->right
    cy = float((ys * c).sum() / (w + 1e-8)) / (c.shape[0] - 1)               # 0..1 top->bottom
    coverage = float((c >= hot).mean())
    # spread = mean distance of hot cells from the centroid (normalised)
    hotmask = c >= hot
    if hotmask.any():
        d = np.sqrt(((xs[hotmask] / (c.shape[1] - 1) - cx)) ** 2 +
                    ((ys[hotmask] / (c.shape[0] - 1) - cy)) ** 2)
        spread = float(d.mean())
    else:
        spread = 0.0

    if coverage >= 0.5 or spread >= 0.33:
        region, side = "diffuse / bicompartmental", "both sides of the joint"
    elif abs(cx - 0.5) < 0.14:
        region, side = "central joint line", "centre of the joint space"
    else:
        region = "one compartment"
        side = "the lateral third of the image" if cx > 0.5 else "the medial third of the image"
    return {"region": region, "side_hint": side, "coverage": round(coverage, 3),
            "centroid": [round(cx, 3), round(cy, 3)], "spread": round(spread, 3)}
