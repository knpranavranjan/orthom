"""AETHER-OA X-ray inference API server (Phase 9 - headless, no Gradio).

    python scripts/serve_api.py --host 0.0.0.0 --port 8000

Serves the deployed MobileNetV2 ONNX as a JSON REST API so an external frontend
(the OA-screening kiosk) can upload a knee radiograph and receive the KL grade,
calibrated probabilities, binary OA screen, 3-class band, abstain flag and a
grad-free Class Activation Map overlay. Torch-free (ONNX Runtime + NumPy + OpenCV).

    GET  /health              GET  /api/xray/meta        POST /api/xray/grade
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402
from src.deploy.api import create_app  # noqa: E402

LOG = get_logger("api")


def _resolve_model(model_dir: Path, name: str) -> Path:
    direct = model_dir / (name if name.endswith(".onnx") else f"{name}.onnx")
    if direct.exists():
        return direct
    cands = [f for f in sorted(model_dir.glob("*.onnx"))
             if not f.name.endswith(("_cam.onnx", ".int8.onnx"))
             and (model_dir / f"{f.name[:-5]}_deploy.json").exists()]
    if not cands:
        raise SystemExit(f"no deployable ONNX (+_deploy.json) in {model_dir}")
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path,
                    default=PROJECT_ROOT / "models" / "deploy")
    ap.add_argument("--model", default="mobilenet_v2_oa",
                    help="ONNX stem inside --model-dir (default: mobilenet_v2_oa)")
    ap.add_argument("--risk-model", type=Path,
                    default=PROJECT_ROOT / "models" / "risk" / "demographic_risk.onnx",
                    help="demographic risk ONNX model; routes are skipped if not found")
    # 0.0.0.0 by default (not 127.0.0.1) and PORT-env-var-aware: a container
    # platform (Render, Railway, Fly.io, ...) injects $PORT and expects the
    # process to bind every interface, not just loopback -- localhost dev
    # still reaches a 0.0.0.0-bound server fine, so this is safe both ways.
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--threads", type=int, default=4,
                    help="ONNX Runtime intra-op threads")
    ap.add_argument("--cors", default="*",
                    help="comma-separated allowed origins, or '*' (default)")
    args = ap.parse_args()

    mdir = args.model_dir if args.model_dir.is_absolute() else PROJECT_ROOT / args.model_dir
    onnx_path = _resolve_model(mdir, args.model)
    origins = ["*"] if args.cors.strip() == "*" else [
        o.strip() for o in args.cors.split(",") if o.strip()]

    risk_path = args.risk_model if args.risk_model.is_absolute() else PROJECT_ROOT / args.risk_model
    app = create_app(onnx_path, intra_threads=args.threads, allow_origins=origins,
                     risk_onnx_path=risk_path)
    risk_note = risk_path.name if risk_path.exists() else "not found, /api/risk/* disabled"
    LOG.info("serving %s on http://%s:%d  (CORS: %s, risk model: %s)",
             onnx_path.name, args.host, args.port, origins, risk_note)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
