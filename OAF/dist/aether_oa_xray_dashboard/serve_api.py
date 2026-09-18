"""AETHER-OA X-ray inference API - edge bundle entrypoint (headless, no Gradio).

    python serve_api.py --host 0.0.0.0 --port 8000

Serves the bundled MobileNetV2 ONNX as a JSON REST API. The frontend is external
(the OA-screening kiosk); this process only does inference. Torch-free.

    GET  /health              GET  /api/xray/meta        POST /api/xray/grade
"""
import argparse
from pathlib import Path

from deploy.api import create_app

HERE = Path(__file__).resolve().parent


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
    ap.add_argument("--model-dir", type=Path, default=HERE)
    ap.add_argument("--model", default="mobilenet_v2_oa")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cors", default="*")
    args = ap.parse_args()

    onnx_path = _resolve_model(args.model_dir, args.model)
    origins = ["*"] if args.cors.strip() == "*" else [
        o.strip() for o in args.cors.split(",") if o.strip()]
    app = create_app(onnx_path, intra_threads=args.threads, allow_origins=origins)

    import uvicorn
    print(f"serving {onnx_path.name} on http://{args.host}:{args.port}  (CORS: {origins})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
