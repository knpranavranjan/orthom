# AETHER-OA - Knee X-ray Screening (edge bundle)

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
python serve_api.py --host 0.0.0.0 --port 8000 --default mobilenet_v2_oa
```
Then point the OA-screening kiosk frontend at `http://<device-ip>:8000`
(`VITE_XRAY_API` in the kiosk's `.env`).

### Endpoints
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` · `/api/xray/meta` | - | model card, calibration, CAM availability |
| POST | `/api/xray/grade` | form-data `image=<file>`, optional `explain=true\|false` | KL grade, calibrated probabilities, binary OA screen, 3-class band, abstain flag, grad-free CAM overlay (PNG data URL), templated findings + recommendation, inference time |

## Latency benchmark on the actual Pi
```
python benchmark_edge.py --real-images 128
```
