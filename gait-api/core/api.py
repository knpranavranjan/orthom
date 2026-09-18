"""AETHER-OA gait analysis REST API.

Exposes the same scoring engine app.py's Streamlit UI uses
(`gait_engine.analyze_session` + `oa_risk_engine.evaluate_oa_preventive_risk`)
as a headless JSON service, so the OSTRIVA kiosk (or anything else) can send a
recorded knee-angle time series and get the composite OA risk report back —
without needing Streamlit, webrtc, or a live camera in this process.

This is a SEPARATE service from OAF's X-ray API by design — kept as its own
deployable unit rather than folded into OAF. Its own runtime footprint is
actually light (numpy + pandas + scipy + fastapi, confirmed by tracing this
file's real import chain — no MediaPipe, no OpenCV, despite the source
project it's copied out of depending on both for other things). Pose
extraction (turning video into the angle time series this API scores)
happens in-browser (oa copy/src/services/poseCapture.ts) — this API only
does the signal-processing + risk-scoring half, never touches a video frame.

    GET  /health                model info + clinical norms
    POST /api/gait/analyze      body: GaitAnalyzeRequest -> the oa_risk report
"""
# NOTE: no `from __future__ import annotations` here — GaitAnalyzeRequest is a
# locally-scoped class inside create_app(); FastAPI must see the real class on
# the route signature to resolve it as a request body, not a stringised
# ForwardRef it can't look up (same reason OAF/src/deploy/api.py avoids it).
from typing import List, Optional

import numpy as np
import pandas as pd

from .gait_engine import analyze_session
from .oa_risk_engine import CLINICAL_NORMS, SCIENTIFIC_SOURCES

DISCLAIMER = ("AI-assisted movement screening — not a diagnosis. Composite risk "
              "score is a preventive/triage signal; clinical correlation is required.")


def _clean(obj):
    """Recursively cast numpy scalars to plain Python types so FastAPI/JSON
    encoding never trips on np.float64/np.bool_ (analyze_session already
    casts most fields, but cycle dicts and edge cases can still carry them)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def create_app(*, allow_origins: Optional[List[str]] = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    class Frame(BaseModel):
        time_sec: float
        left_knee_angle: Optional[float] = None
        right_knee_angle: Optional[float] = None

    class GaitAnalyzeRequest(BaseModel):
        frames: List[Frame]
        fps: float = 30.0
        # Other domains, same as analyze_session's optional kwargs — all
        # optional because a patient may only have completed the gait walk.
        seated_left_flexion: Optional[float] = None
        seated_right_flexion: Optional[float] = None
        seated_left_extension: Optional[float] = None
        seated_right_extension: Optional[float] = None
        sts_completed_reps: Optional[float] = None
        sts_mean_ascent_sec: Optional[float] = None
        sts_peak_trunk_lean_deg: Optional[float] = None
        standing_knee_ankle_ratio: Optional[float] = None
        standing_baseline_ext: Optional[float] = None

    app = FastAPI(
        title="AETHER-OA gait analysis API",
        version="1.0.0",
        description="Headless knee-kinematics + OA preventive-risk scoring for recorded gait sessions.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins if allow_origins is not None else ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    @app.get("/api/gait/meta")
    def meta() -> dict:
        return {
            "status": "ok",
            "clinical_norms": CLINICAL_NORMS,
            "scientific_sources": SCIENTIFIC_SOURCES,
            "disclaimer": DISCLAIMER,
        }

    @app.post("/api/gait/analyze")
    def analyze(body: GaitAnalyzeRequest) -> dict:
        if len(body.frames) < 10:
            raise HTTPException(status_code=400,
                                detail=f"need at least 10 frames, got {len(body.frames)}")

        df = pd.DataFrame([f.model_dump() for f in body.frames])

        try:
            result = analyze_session(
                df,
                fps=body.fps,
                seated_left_flexion=body.seated_left_flexion,
                seated_right_flexion=body.seated_right_flexion,
                seated_left_extension=body.seated_left_extension,
                seated_right_extension=body.seated_right_extension,
                sts_completed_reps=body.sts_completed_reps,
                sts_mean_ascent_sec=body.sts_mean_ascent_sec,
                sts_peak_trunk_lean_deg=body.sts_peak_trunk_lean_deg,
                standing_knee_ankle_ratio=body.standing_knee_ankle_ratio,
                standing_baseline_ext=body.standing_baseline_ext,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

        result.pop("dataframe", None)  # not JSON-serializable, and the caller sent this data already
        result["disclaimer"] = DISCLAIMER
        return _clean(result)

    return app
