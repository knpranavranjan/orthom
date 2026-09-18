"""Regression test for the X-ray input-validation gate (`src/deploy/validate.py`).

ID  = real knee-X-ray tiles sliced from reports/samples/kl*_samples.png
      (+ a few realistic variants: upscaled, JPEG round-trip, black border,
       burnt-in marker, faint colour cast) — ALL must be ACCEPTED.
OOD = colour noise / colour gradient / colour-map, solid fill, grayscale
      gradient, screenshot-with-text, scanned document, QR-like, face proxy —
      ALL must be REJECTED.

Prints a confusion summary and exits non-zero on any regression.
Run from the OAF repo root:  .venv/bin/python scripts/test_xray_gate.py
"""
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deploy.preprocess import preprocess          # noqa: E402
from src.deploy import validate as V                  # noqa: E402
import onnxruntime as ort                             # noqa: E402

CAM = ROOT / "models" / "deploy" / "mobilenet_v2_oa_cam.onnx"
_sess = ort.InferenceSession(str(CAM), providers=["CPUExecutionProvider"]) if CAM.exists() else None
_IN = _sess.get_inputs()[0].name if _sess else None


def _metrics(img):
    m = V.preflight_metrics(img)
    if _sess is not None:
        lg, ft = _sess.run(["logits", "features"], {_IN: preprocess(img)})
        m = {**m, **V.model_ood_metrics(lg[0], ft[0])}
    return m


def _decide(img):
    return V.decide(_metrics(img))


def _tiles(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    g = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY) if im.ndim == 3 else im
    mk = cv2.morphologyEx(((g < 245).astype(np.uint8)) * 255,
                          cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mk, 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w > 120 and h > 120 and 0.5 < w / h < 2.0 and a > 0.4 * w * h:
            yield g[y:y + h, x:x + w].copy()


def id_cases():
    for p in sorted((ROOT / "reports" / "samples").glob("kl*_samples.png")):
        for j, t in enumerate(_tiles(p)):
            yield f"{p.stem}#{j}", t
            if j % 4 == 0:                              # a few variants, not all
                yield f"{p.stem}#{j}~up", cv2.resize(t, (880, 880), interpolation=cv2.INTER_CUBIC)
                ok, e = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield f"{p.stem}#{j}~jpg", cv2.imdecode(e, cv2.IMREAD_GRAYSCALE)
                yield f"{p.stem}#{j}~border", cv2.copyMakeBorder(t, 55, 55, 80, 80, cv2.BORDER_CONSTANT, value=0)
                tb = t.copy()
                cv2.putText(tb, "R", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2)
                yield f"{p.stem}#{j}~mark", tb


def ood_cases():
    rng = np.random.default_rng(7)
    yield "rgb_noise", rng.integers(0, 256, (300, 300, 3), np.uint8)
    yield "solid", np.full((256, 256), 120, np.uint8)
    grad = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    yield "gray_gradient", grad
    yield "colour_gradient", cv2.merge([grad, grad[::-1], np.full_like(grad, 40)])
    yield "colourmap", cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    ss = np.full((320, 460, 3), 248, np.uint8)
    for r in range(24, 300, 26):
        ss[r:r + 12, 20:20 + int(rng.integers(140, 420))] = 30
    yield "screenshot", ss
    doc = np.full((420, 560), 250, np.uint8)
    for y in range(28, 400, 22):
        cv2.putText(doc, "lorem ipsum dolor sit amet " * 2, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 25, 1)
    yield "document", doc
    yield "qr_like", cv2.resize((rng.integers(0, 2, (23, 23)) * 255).astype(np.uint8),
                                (280, 280), interpolation=cv2.INTER_NEAREST)
    face = np.full((256, 256), 95, np.uint8)
    cv2.ellipse(face, (128, 140), (70, 95), 0, 0, 360, 175, -1)
    cv2.circle(face, (105, 120), 9, 40, -1); cv2.circle(face, (151, 120), 9, 40, -1)
    yield "face_proxy", cv2.GaussianBlur(face, (0, 0), 2)


tp = fp = tn = fn = 0
fails = []
for nm, im in id_cases():
    if _decide(im)["accept"]:
        tp += 1
    else:
        fn += 1; fails.append(("ID rejected", nm, _decide(im)["reason"]))
for nm, im in ood_cases():
    d = _decide(im)
    if d["accept"]:
        fp += 1; fails.append(("OOD accepted", nm, "-"))
    else:
        tn += 1

n_id, n_ood = tp + fn, tn + fp
print(f"ID  (real knee X-rays)   accepted {tp}/{n_id}   ({100*tp/max(n_id,1):.1f}%)")
print(f"OOD (non-X-ray uploads)  rejected {tn}/{n_ood}   ({100*tn/max(n_ood,1):.1f}%)")
for kind, nm, why in fails:
    print(f"  ! {kind:13s} {nm:22s} {why}")

# Gate policy: NEVER reject a real knee X-ray; reject the realistic wrong-file set.
ok = (fn == 0) and (fp == 0)
print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
