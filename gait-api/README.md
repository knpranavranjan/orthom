# gait-api

Deployable subset of the `OA_Computer_Vision` project (a separate working
directory on the author's machine, not part of this repo) — just the
headless scoring service the OSTRIVA kiosk's gait-walk screen calls.

## What's here, and why only this

- `core/gait_engine.py` — smoothing, gait-cycle detection, cycle metrics
- `core/oa_risk_engine.py` — 4-domain OA risk scoring against literature-cited clinical norms
- `core/api.py` — FastAPI wrapper (`POST /api/gait/analyze`, `GET /health`)
- `serve_gait_api.py` — entry point

**Deliberately not copied here:** the source project's MediaPipe pose
detector (`pose_detector.py`), its Streamlit desktop app (`app.py`,
`aether_complete_gait.py`), and the real recorded patient/subject session
data (`AETHER_OA_PATIENT_DATA/`, `data/raw_sessions/`). None of that is
needed to run this API, and the session data in particular has no business
in a public deployment repo. Pose extraction happens in-browser
(`oa copy/src/services/poseCapture.ts`) — this service receives an
already-extracted knee-angle time series, never a video frame.

## Run it

    pip install -r requirements.txt
    python serve_gait_api.py --port 8001

## Deploy

Part of the repo-root `render.yaml` blueprint (`rootDir: gait-api`) — pushes
to `orthom`'s main branch redeploy it the same way as the `OAF` service.
