# `models/`

Only the **final deployment artifacts** and **small JSON provenance** are tracked
in Git. Training checkpoints (`*.pt`), the 7.4 GB `training_checkpoints.zip`, and
every historical/benchmark weight file were removed before the first commit and
are **not** distributed — they are re-creatable from `configs/` + the data.

| Path | Tracked? | What it is |
|---|:--:|---|
| `models/deploy/mobilenet_v2_oa.onnx` | ✅ | final served model (logits) |
| `models/deploy/mobilenet_v2_oa_cam.onnx` | ✅ | final model + `1×1280×7×7` feature map (for grad-free CAM) |
| `models/deploy/mobilenet_v2_oa_cam.npz` | ✅ | classifier weights `W (5×1280)`, `b (5)` for the CAM |
| `models/deploy/mobilenet_v2_oa_deploy.json` | ✅ | temperature, abstain threshold, class names, frozen test metrics |
| `models/mobilenet_final/best.pt` | ✅ | canonical trainable checkpoint (EMA weights) that produced the ONNX |
| `models/mobilenet_final/{config,result}.json` | ✅ | the exact KD recipe + validation/test metrics of the deployed model |
| `models/mobilenet_final/teacher_logits/*.npz` | ✅ | cached VGG16 + ConvNeXt-Tiny soft targets — enough to re-run distillation |
| `models/phase3/`, `models/phase4/`, `models/phase7/` | ✅ (JSON only) | per-model `config/history/result.json` for every benchmarked architecture |
| `models/benchmark/`, `models/phase6/`, `models/phase8*/`, `models/archive/` | ❌ removed | historical training checkpoints — provenance lives in `reports/` |

**Teacher checkpoints** (`VGG16`, `ConvNeXt-Tiny`) are *training-time only* and are
not required for inference, the dashboard, ONNX deployment, prediction, or CAM.
See `README.md` §8 for their provenance and how to regenerate them.
