"""AETHER-OA gait analysis API server.

    python serve_gait_api.py --port 8001

Headless JSON service around gait_engine.analyze_session +
oa_risk_engine.evaluate_oa_preventive_risk — see core/api.py for the routes
and why this runs as its own process rather than living inside OAF.

This is a deliberately minimal DEPLOYMENT COPY of the full
OA_Computer_Vision project (a separate working directory, not part of this
repo) — just this API's dependency chain (core/api.py, gait_engine.py,
oa_risk_engine.py; numpy+pandas+scipy+fastapi, no MediaPipe/OpenCV/Streamlit
at all, since pose detection now runs in-browser — see
oa copy/src/services/poseCapture.ts). Deliberately excludes the source
project's pose_detector.py, the Streamlit desktop app (app.py,
aether_complete_gait.py), and the real recorded patient/subject session
data (AETHER_OA_PATIENT_DATA/, data/raw_sessions/) — none of that belongs in
a public deployment repo. Keep this folder minimal; don't copy more of the
source project into it without a reason.

    GET  /health                POST /api/gait/analyze
"""
from __future__ import annotations

import argparse
import os

from core.api import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # 0.0.0.0 (not 127.0.0.1) and PORT-env-var-aware, same reasoning as
    # OAF/scripts/serve_api.py: a container platform injects $PORT and
    # expects every interface bound, not just loopback.
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8001)))
    ap.add_argument("--cors", default="*",
                    help="comma-separated allowed origins, or '*' (default)")
    args = ap.parse_args()

    origins = ["*"] if args.cors.strip() == "*" else [
        o.strip() for o in args.cors.split(",") if o.strip()]

    app = create_app(allow_origins=origins)
    print(f"serving gait analysis API on http://{args.host}:{args.port}  (CORS: {origins})")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
