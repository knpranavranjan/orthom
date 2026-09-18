"""PHASE 7 - build the lightweight edge/dashboard deliverable and (optionally)
reclaim space from the 15-model training checkpoints.

    # 1. bundle the deployable model(s) + inference code + dashboard into dist/
    python scripts/phase7_package.py --model mobilenet_v2_oa --model convnext_tiny_oa

    # 2. (optional) zip the big Phase 3/4/7 training checkpoints out of the way
    python scripts/phase7_package.py --archive-heavy
    python scripts/phase7_package.py --archive-heavy --prune       # + delete originals after zipping

Nothing is deleted unless --prune is given. Phase 3/4 result JSON / reports are
never touched (only the .pt weight files).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402

LOG = get_logger("pkg")
DEPLOY = PROJECT_ROOT / "models" / "deploy"
DIST = PROJECT_ROOT / "dist" / "aether_oa_xray_dashboard"

EDGE_REQS = """# AETHER-OA X-ray inference API - Raspberry Pi / edge runtime (no torch)
onnxruntime>=1.19
numpy>=1.26
opencv-python-headless>=4.9
pillow>=10.0
psutil>=5.9
# REST API (serve_api.py) - the frontend is external
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
"""

README = """# AETHER-OA - Knee X-ray Screening (edge bundle)

AI-assisted Kellgren-Lawrence (KL 0-4) grading from a knee radiograph.
**Research prototype - not a diagnostic device. Clinical correlation required.**

## Contents
- `*.onnx` + `*_deploy.json`  - model(s) and their calibration / metric cards
- `deploy/`                   - torch-free preprocessing + predictor + REST API (`api.py`)
- `serve_api.py`              - headless JSON inference server (no UI; the frontend is external)
- `requirements-edge.txt`     - runtime deps (no PyTorch)

## Run (Raspberry Pi or any CPU box)
```
python -m pip install -r requirements-edge.txt
python serve_api.py --host 0.0.0.0 --port 8000 --default {default}
```
Point the OA-screening kiosk frontend at `http://<device-ip>:8000`
(`VITE_XRAY_API` in the kiosk's `.env`).

### Endpoints
- `GET  /health` · `/api/xray/meta`  - model card, calibration, CAM availability
- `POST /api/xray/grade`  - form-data `image=<file>`, optional `explain=true|false`
  -> KL grade, calibrated probabilities, binary OA screen, 3-class band, abstain
  flag, grad-free CAM overlay (PNG data URL), findings + recommendation, latency

## Latency benchmark on the actual Pi
```
python benchmark_edge.py --real-images 128
```
"""


def bundle(models: list[str], default: str):
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "deploy").mkdir(parents=True)
    picked = []
    for m in models:
        onnx = DEPLOY / f"{m}.onnx"
        card = DEPLOY / f"{m}_deploy.json"
        if not onnx.exists() or not card.exists():
            LOG.warning("skip %s (missing %s or its _deploy.json)", m, onnx.name)
            continue
        shutil.copy2(onnx, DIST / onnx.name)
        shutil.copy2(card, DIST / card.name)
        # optional grad-free CAM artefacts (heatmap explainability)
        for ext in (f"{m}_cam.onnx", f"{m}_cam.npz"):
            if (DEPLOY / ext).exists():
                shutil.copy2(DEPLOY / ext, DIST / ext)
                LOG.info("bundled %s", ext)
        picked.append(m)
        LOG.info("bundled %s (%.1f MB)", m, onnx.stat().st_size / 1e6)
    if not picked:
        raise SystemExit("no valid models to bundle - run scripts/export_onnx.py first")

    for f in ("__init__.py", "preprocess.py", "predictor.py", "cam.py", "api.py", "validate.py"):
        shutil.copy2(PROJECT_ROOT / "src" / "deploy" / f, DIST / "deploy" / f)
    # predictor imports `from .preprocess import preprocess`; keep it a package
    for f in ("serve_api.py", "benchmark_edge.py"):
        src = (PROJECT_ROOT / "scripts" / f).read_text(encoding="utf-8")
        src = src.replace("from src.deploy.api", "from deploy.api")
        src = src.replace("from src.deploy.predictor", "from deploy.predictor")
        src = src.replace("from src.deploy.preprocess", "from deploy.preprocess")
        src = src.replace("from src.common import PROJECT_ROOT, get_logger",
                          "import logging, pathlib\nPROJECT_ROOT = pathlib.Path(__file__).resolve().parent\n"
                          "def get_logger(n): logging.basicConfig(level=logging.INFO); return logging.getLogger(n)")
        # in the bundle the ONNX files sit next to the script, not under models/deploy/
        src = src.replace('PROJECT_ROOT / "models" / "deploy"', "PROJECT_ROOT")
        src = src.replace("PROJECT_ROOT / 'models' / 'deploy'", "PROJECT_ROOT")
        (DIST / f).write_text(src, encoding="utf-8")

    dflt = default if default in picked else picked[0]
    (DIST / "requirements-edge.txt").write_text(EDGE_REQS, encoding="utf-8")
    (DIST / "README.md").write_text(README.format(default=dflt), encoding="utf-8")
    (DIST / "bundle_manifest.json").write_text(json.dumps(
        {"models": picked, "default": dflt,
         "cards": {m: json.loads((DEPLOY / f"{m}_deploy.json").read_text(encoding="utf-8")) for m in picked}},
        indent=2), encoding="utf-8")
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 1e6
    LOG.info("bundle -> %s  (%d model(s), %.1f MB total)", DIST, len(picked), total)


def archive_heavy(prune: bool):
    arch_dir = PROJECT_ROOT / "models" / "archive"
    arch_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for phase in ("phase3", "phase4", "phase7"):
        root = PROJECT_ROOT / "models" / phase
        if root.exists():
            targets += sorted(root.rglob("*.pt"))
    if not targets:
        LOG.info("no .pt checkpoints found under models/phase{3,4,7}")
        return
    zpath = arch_dir / "training_checkpoints.zip"
    total = sum(f.stat().st_size for f in targets) / 1e6
    LOG.info("zipping %d checkpoints (%.0f MB) -> %s", len(targets), total, zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in targets:
            z.write(f, f.relative_to(PROJECT_ROOT).as_posix())
    LOG.info("archive written (%.0f MB compressed)", zpath.stat().st_size / 1e6)
    if prune:
        for f in targets:
            f.unlink()
        LOG.info("pruned %d original .pt files (result.json / config.json / reports kept)", len(targets))
    else:
        LOG.info("originals kept. re-run with --prune to delete them now that they are zipped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="deploy basename to bundle (repeat)")
    ap.add_argument("--default", default="mobilenet_v2_oa")
    ap.add_argument("--archive-heavy", action="store_true",
                    help="zip models/phase{3,4,7}/**/*.pt into models/archive/")
    ap.add_argument("--prune", action="store_true", help="with --archive-heavy: delete originals after zipping")
    args = ap.parse_args()

    if args.model:
        bundle(args.model, args.default)
    if args.archive_heavy:
        archive_heavy(args.prune)
    if not args.model and not args.archive_heavy:
        ap.error("nothing to do: pass --model ... and/or --archive-heavy")


if __name__ == "__main__":
    main()
