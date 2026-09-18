"""AETHER-OA X-ray inference REST API (torch-free).

Replaces the Gradio kiosk dashboard with a headless JSON service so an external
frontend (the OA-screening kiosk) can own the UI:

    knee radiograph (multipart upload)
      -> preprocess (NumPy + OpenCV)
      -> MobileNetV2 ONNX (distilled, 2.23 M params, ONNX Runtime CPU)
      -> temperature scale -> softmax -> KL grade / OA screen / 3-class / abstain
      -> grad-free Class Activation Map -> PNG heatmap overlay (data URL)
      -> templated findings + recommendation

Endpoints
    GET  /health              - model card, calibration, whether the CAM is loaded
    GET  /api/xray/meta       - same payload as /health (frontend-friendly alias)
    POST /api/xray/grade      - form-data: image=<file>  [explain=true|false]

No PyTorch, no CUDA, no timm at inference. Runs on a Raspberry Pi.
"""
# NOTE: no `from __future__ import annotations` here - FastAPI/pydantic must see
# the real `UploadFile` type on the route signature, not a stringised ForwardRef.
import io
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .predictor import KneeOAPredictor, KL_NAMES, DISCLAIMER

SEVERITY = {0: "No radiographic OA", 1: "Doubtful / very early",
            2: "Early OA", 3: "Moderate OA", 4: "Severe OA"}


def _decode_image(raw: bytes) -> np.ndarray:
    """bytes -> HxW or HxWxC uint8 ndarray. OpenCV first, Pillow as fallback."""
    try:
        import cv2
        buf = np.frombuffer(raw, np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            return arr
    except Exception:  # noqa: BLE001 - fall through to PIL
        pass
    from PIL import Image
    return np.array(Image.open(io.BytesIO(raw)))


def _card(pred: KneeOAPredictor) -> dict:
    return {
        "model": pred.model_name,
        "has_cam": bool(pred.has_cam),
        "temperature": pred.temperature,
        "abstain_threshold": pred.abstain_threshold,
        "class_names": {i: KL_NAMES[i] for i in range(5)},
        "severity": SEVERITY,
        "test_metrics": pred.meta.get("test_metrics", {}),
        "disclaimer": DISCLAIMER,
    }


def _shape_response(r: dict) -> dict:
    """KneeOAPredictor.predict(...) output -> the JSON the frontend consumes.

    Adds `dist` (plain 5-vector, the frontend's KLDistribution shape) and folds
    the base64 CAM into a ready-to-render `heatmap` data URL.
    """
    g = r["kl_grade"]
    dist = [r["probabilities"][f"KL{i}"] for i in range(5)]
    dist_raw = [r["probabilities_uncalibrated"][f"KL{i}"] for i in range(5)]

    heatmap = None
    interpretation = None
    findings: list[str] = []
    attention_note = None
    focus_region = None
    ex = r.get("explain") or {}
    if ex:
        interpretation = ex.get("interpretation")
        findings = list(ex.get("findings", []))
        attention_note = ex.get("attention_note")
        focus_region = ex.get("focus_region")
        b64 = ex.get("heatmap_png_b64")
        if b64:
            heatmap = f"data:image/png;base64,{b64}"

    return {
        "grade": g,
        "kl_name": KL_NAMES[g],
        "severity": SEVERITY[g],
        "dist": dist,
        "dist_uncalibrated": dist_raw,
        "confidence": r["confidence"],
        "expected_grade": r["expected_grade"],
        "oa_screen": r["oa_screen"],
        "three_class": r["three_class"],
        "abstain": r["abstain"],
        "abstain_threshold": r["abstain_threshold"],
        "recommendation": r["recommendation"],
        "findings": findings,
        "interpretation": interpretation,
        "attention_note": attention_note,
        "focus_region": focus_region,
        "heatmap": heatmap,
        "latency_ms": r["latency_ms"],
        "model": r["model"],
        "disclaimer": r["disclaimer"],
    }


def create_app(onnx_path: Union[str, Path], *, intra_threads: int = 4,
               allow_origins: Optional[List[str]] = None):
    """Build the FastAPI app around a single deployed ONNX model."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise SystemExit(f"model not found: {onnx_path}")

    predictor = KneeOAPredictor(onnx_path, intra_threads=intra_threads)

    app = FastAPI(
        title="AETHER-OA X-ray inference API",
        version="1.0.0",
        description="Headless KL grading + grad-free CAM for the OA-screening kiosk.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins if allow_origins is not None else ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.predictor = predictor

    @app.get("/health")
    @app.get("/api/xray/meta")
    def meta() -> dict:
        return {"status": "ok", **_card(predictor)}

    @app.post("/api/xray/grade")
    async def grade(image: UploadFile = File(...),
                    explain: str = Form("true")) -> JSONResponse:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            arr = _decode_image(raw)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=415,
                                detail=f"cannot decode image: {exc}") from exc
        want_explain = str(explain).strip().lower() not in ("0", "false", "no", "")
        try:
            result = predictor.predict(arr, explain=want_explain)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                detail=f"inference failed: {exc}") from exc
        return JSONResponse(_shape_response(result))

    return app
