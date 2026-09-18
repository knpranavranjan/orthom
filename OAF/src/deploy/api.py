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

from . import validate as V
from .predictor import KneeOAPredictor, KL_NAMES, DISCLAIMER
from .risk_predictor import DemographicRiskPredictor

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
        "input_gate": {"enabled": True, "overrides": pred.meta.get("input_gate", {})},
        "disclaimer": DISCLAIMER,
    }


def _rejection(check: dict) -> dict:
    """Shape a gate rejection into the response the frontend renders as a warning."""
    return {
        "accepted": False,
        "reason": check["reason"],
        "message": check["message"],
        "input_check": {
            "hard_fail": check["hard_fail"],
            "soft_fail": check["soft_fail"],
            "model_fail": check["model_fail"],
            "metrics": check["metrics"],
        },
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
        "accepted": True,
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
               allow_origins: Optional[List[str]] = None,
               risk_onnx_path: Optional[Union[str, Path]] = None):
    """Build the FastAPI app around the X-ray ONNX model, plus the demographic
    risk model's routes when `risk_onnx_path` is given and exists (optional:
    an X-ray-only deployment should not fail to start without it)."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise SystemExit(f"model not found: {onnx_path}")

    predictor = KneeOAPredictor(onnx_path, intra_threads=intra_threads)

    risk_predictor: Optional[DemographicRiskPredictor] = None
    if risk_onnx_path is not None and Path(risk_onnx_path).exists():
        risk_predictor = DemographicRiskPredictor(
            Path(risk_onnx_path), intra_threads=intra_threads)

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

    gate_cfg = predictor.meta.get("input_gate") or None

    @app.post("/api/xray/grade")
    async def grade(image: UploadFile = File(...),
                    explain: str = Form("true"),
                    validate: str = Form("true")) -> JSONResponse:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            arr = _decode_image(raw)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=415,
                                detail=f"cannot decode image: {exc}") from exc

        want_explain = str(explain).strip().lower() not in ("0", "false", "no", "")
        do_gate = str(validate).strip().lower() not in ("0", "false", "no", "")

        # ── input validation gate ──────────────────────────────────────
        # Stage 1: cheap image statistics. A hard failure (colour photo,
        # screenshot, document) is rejected here WITHOUT running the model.
        if do_gate:
            pf = V.preflight_metrics(arr)
            d1 = V.decide(pf, gate_cfg)
            if not d1["accept"] and d1["hard_fail"]:
                return JSONResponse(_rejection(d1))

        # forward pass (also yields logits + GAP features for stage 2)
        try:
            result = predictor.predict(arr, explain=want_explain, return_raw=do_gate)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                detail=f"inference failed: {exc}") from exc

        # Stage 2: model-space OOD signals on the logits + penultimate features.
        if do_gate:
            fg = result.get("feat_gap")
            mo = V.model_ood_metrics(result["logits"],
                                     np.asarray(fg, dtype=np.float32) if fg else None)
            d2 = V.decide({**pf, **mo}, gate_cfg)
            if not d2["accept"]:
                return JSONResponse(_rejection(d2))
            result_check = {"reason": "ok", **{k: pf[k] for k in
                            ("chroma", "uniq_frac", "p99_p01", "entropy",
                             "edge_frac", "lap_var")}, **mo}
        else:
            result_check = {"reason": "skipped"}

        for k in ("logits", "feat_gap"):
            result.pop(k, None)
        payload = _shape_response(result)
        payload["input_check"] = result_check
        return JSONResponse(payload)

    if risk_predictor is not None:
        class DemographicRiskRequest(BaseModel):
            age: float
            bmi: float
            sex: int

        @app.get("/api/risk/meta")
        def risk_meta() -> dict:
            return {
                "status": "ok",
                "model": risk_predictor.model_name,
                "feature_order": risk_predictor.feature_order,
                "scope_warning": risk_predictor.meta.get("scope_warning"),
                "test_metrics": risk_predictor.meta.get("test_metrics", {}),
                "disclaimer": DISCLAIMER,
            }

        @app.post("/api/risk/demographic")
        def risk_demographic(body: DemographicRiskRequest) -> dict:
            result = risk_predictor.predict(age=body.age, bmi=body.bmi, sex=body.sex)
            return {"accepted": True, **result, "disclaimer": DISCLAIMER}

    return app
