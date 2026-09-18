"""Input validation / out-of-distribution gate for the AETHER-OA X-ray API.

Problem: the KL grader is a 5-class softmax head — it *always* returns a grade,
even for a selfie, a screenshot or a document. This module decides, BEFORE a
grade or a heatmap is shown, whether the upload is plausibly a knee radiograph
and, if not, returns a "please upload a knee X-ray" warning instead.

Two stages, both torch-free (NumPy + OpenCV):

  1. `preflight_metrics(img)` — cheap statistics on the raw upload: colour
     content, tonal continuity, contrast, detail, edge/sharpness. ~1 ms,
     no model. Catches the realistic wrong-file cases (colour photos,
     screenshots, memes, scanned documents, solid / gradient fills).

  2. `model_ood_metrics(logits, feat)` — signals from the network itself:
     free-energy of the logits, L2 norm of the penultimate GAP-1280 feature
     vector, dead-unit fraction, max softmax. A second line of defence for
     grayscale-but-not-a-knee inputs.

`decide(metrics)` combines them: a "hard" failure (colour) rejects outright;
otherwise it rejects when two or more soft/model checks fail.

Thresholds live in `GATE`, calibrated against 80 real knee-X-ray tiles from
`reports/samples/kl*_samples.png` plus synthetic non-X-ray images, and can be
overridden per deployment via `mobilenet_v2_oa_deploy.json` -> `"input_gate"`.

    ID  (n=80) : chroma 0.000  uniq 0.31-0.98  p99_p01 0.27-0.97  std 0.06-0.30
                 entropy 0.58-0.97  edge <0.17  lap 6-130  energy <-2.3
                 feat_norm 3.4-9.7  dead_frac 0.39-0.67
"""
import numpy as np

GATE = {
    # ── stage 1 — HARD: a knee radiograph is monochrome ──
    "chroma_max": 0.050,        # mean |max(RGB)-min(RGB)| / 255              ID 0.000
    # ── stage 1 — SOFT (reject when >= 2 soft/model checks fail) ──
    "sat_max": 0.200,           # mean HSV saturation over non-dark px / 255
    "uniq_frac_min": 0.150,     # fraction of 8-bit levels used (posterisation)  ID min 0.31
    "p99_p01_min": 0.180,       # (P99-P01)/255                (contrast)         ID min 0.27
    "std_min": 0.030,           # intensity std / 255          (flatness)         ID min 0.06
    "entropy_min": 0.420,       # 64-bin hist entropy / log2 64 (detail)   ID>=0.58, bordered X-rays dip
    "edge_frac_max": 0.400,     # fraction of strong-gradient px (text / UI)      ID max 0.17
    "lap_var_min": 3.0,         # var(Laplacian)         (near-uniform)    ID>=6.4, upscaled X-rays dip
    "lap_var_max": 6000.0,      # var(Laplacian)              (sharp glyph edges)
    # ── stage 2 — model space ──
    "energy_max": -1.0,         # -logsumexp(logits)   higher => more OOD         ID max -2.3
    "feat_norm_lo": 2.8,        # ||GAP feature||_2                               ID min 3.4
    "feat_norm_hi": 18.0,       # ||GAP feature||_2                               ID max 9.7
    "dead_frac_max": 0.78,      # fraction of ~zero GAP units      ID max 0.67
    # ── decision ──
    "max_soft_fail": 1,         # accept only if (soft + model) failures <= this
}


# ── stage 1 : raw-image statistics ───────────────────────────────────

def _prep_gray(img):
    """-> (gray_u8 HxW, chroma 0..1, sat_mean 0..1). Downscales huge inputs."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        a = np.clip(a, 0, 255) if a.max() > 1.5 else a * 255.0
        a = a.astype(np.uint8)
    if a.ndim == 2:
        gray, chroma, sat = a, 0.0, 0.0
    else:
        if a.shape[2] == 4:
            a = a[..., :3]
        c = a.astype(np.int16)
        chroma = float(np.mean(c.max(axis=2) - c.min(axis=2)) / 255.0)
        try:
            import cv2
            hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            v = hsv[..., 2]
            s = hsv[..., 1][v > 25]
            sat = float(s.mean() / 255.0) if s.size else chroma
            gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        except Exception:  # noqa: BLE001
            gray = a.mean(axis=2).astype(np.uint8)
            sat = chroma
    h, w = gray.shape[:2]
    if max(h, w) > 512:
        try:
            import cv2
            k = 512 / max(h, w)
            gray = cv2.resize(gray, (max(1, int(w * k)), max(1, int(h * k))),
                              interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001
            step = int(np.ceil(max(h, w) / 512))
            gray = gray[::step, ::step]
    return gray, chroma, sat


def preflight_metrics(img) -> dict:
    gray, chroma, sat = _prep_gray(img)
    h, w = gray.shape[:2]
    g = gray.astype(np.float64)
    n = g.size

    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    uniq_frac = float((hist > (0.0005 * n)).sum() / 256.0)

    p01, p99 = np.percentile(gray, [1, 99])
    p99_p01 = float((p99 - p01) / 255.0)
    std = float(g.std() / 255.0)

    h64 = hist.reshape(64, 4).sum(axis=1)
    pr = h64 / max(h64.sum(), 1.0)
    pr = pr[pr > 0]
    entropy = float(-(pr * np.log2(pr)).sum() / np.log2(64)) if pr.size else 0.0

    try:
        import cv2
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        edge_frac = float((mag > 48.0).mean())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:  # noqa: BLE001
        dx = np.abs(np.diff(g, axis=1))
        dy = np.abs(np.diff(g, axis=0))
        edge_frac = float(((dx[:-1, :] + dy[:, :-1]) > 48.0).mean())
        lap_var = float(np.var(np.diff(g, n=2, axis=0)))

    # sharpness normalised by contrast: real X-rays carry fine trabecular
    # texture everywhere (high), a blurred / low-pass image does not (low).
    hf_ratio = float(lap_var / ((g.std() ** 2) + 1.0))

    return {
        "chroma": round(chroma, 4), "sat_mean": round(sat, 4),
        "uniq_frac": round(uniq_frac, 4), "p99_p01": round(p99_p01, 4),
        "std": round(std, 4), "entropy": round(entropy, 4),
        "edge_frac": round(edge_frac, 4), "lap_var": round(lap_var, 2),
        "hf_ratio": round(hf_ratio, 5),
        "h": int(h), "w": int(w),
    }


# ── stage 2 : model-space signals ───────────────────────────────────

def _logsumexp(z) -> float:
    m = float(np.max(z))
    return m + float(np.log(np.exp(z - m).sum()))


def model_ood_metrics(logits, feat=None) -> dict:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    out = {"energy": round(-_logsumexp(logits), 4), "max_prob": round(float(probs.max()), 4)}

    if feat is not None:
        f = np.asarray(feat, dtype=np.float64)
        gap = f.mean(axis=(1, 2)) if f.ndim == 3 else f.reshape(-1)
        if gap.size >= 32:                       # a real GAP-1280 vector, not a stub
            out["feat_norm"] = round(float(np.linalg.norm(gap)), 3)
            out["dead_frac"] = round(float(np.mean(np.abs(gap) < 1e-3)), 4)
    return out


# ── decision ────────────────────────────────────────────────────────

_MESSAGES = {
    "colour": "This looks like a colour photo or screenshot, not an X-ray. "
              "Please upload a knee radiograph.",
    "posterised": "This image doesn't have the continuous tone of an X-ray. "
                  "Please upload a knee radiograph.",
    "low_contrast": "This image is too flat to be an X-ray. Please upload a knee radiograph.",
    "flat": "This image is nearly blank. Please upload a knee radiograph.",
    "low_detail": "This image lacks radiographic detail. Please upload a knee radiograph.",
    "too_much_edge": "This looks like text or a screenshot, not an X-ray. "
                     "Please upload a knee radiograph.",
    "sharpness": "This image doesn't look like an X-ray. Please upload a knee radiograph.",
    "not_knee": "This doesn't look like a knee X-ray. Please upload a knee radiograph.",
    "tiny": "This image is too small to assess. Please upload a clearer knee X-ray.",
}
_DEFAULT_MSG = "This doesn't look like a knee X-ray. Please upload a knee radiograph."


def decide(m: dict, gate=None) -> dict:
    G = {**GATE, **(gate or {})}

    if m.get("h", 999) < 64 or m.get("w", 999) < 64:
        return _reject("tiny", m, hard=["tiny"])

    hard = []
    if m["chroma"] > G["chroma_max"]:
        hard.append("colour")

    soft = []
    if m.get("sat_mean", 0.0) > G["sat_max"]:
        soft.append("colour")
    if m["uniq_frac"] < G["uniq_frac_min"]:
        soft.append("posterised")
    if m["p99_p01"] < G["p99_p01_min"]:
        soft.append("low_contrast")
    if m["std"] < G["std_min"]:
        soft.append("flat")
    if m["entropy"] < G["entropy_min"]:
        soft.append("low_detail")
    if m["edge_frac"] > G["edge_frac_max"]:
        soft.append("too_much_edge")
    if not (G["lap_var_min"] <= m["lap_var"] <= G["lap_var_max"]):
        soft.append("sharpness")

    model = []
    if "energy" in m and m["energy"] > G["energy_max"]:
        model.append("not_knee")
    if "feat_norm" in m:
        if not (G["feat_norm_lo"] <= m["feat_norm"] <= G["feat_norm_hi"]):
            model.append("not_knee")
        if m["dead_frac"] > G["dead_frac_max"]:
            model.append("not_knee")

    n_fail = len(soft) + len(model)
    accept = (not hard) and n_fail <= G["max_soft_fail"]
    if accept:
        return {"accept": True, "reason": "ok", "message": None,
                "hard_fail": [], "soft_fail": soft, "model_fail": model, "metrics": m}
    return _reject((hard or soft or model or ["not_knee"])[0], m,
                   hard=hard, soft=soft, model=model)


def _reject(reason, m, hard=None, soft=None, model=None):
    return {
        "accept": False, "reason": reason,
        "message": _MESSAGES.get(reason, _DEFAULT_MSG),
        "hard_fail": hard or [], "soft_fail": soft or [], "model_fail": model or [],
        "metrics": m,
    }
