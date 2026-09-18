"""PHASE 6 - shared inference helpers (no training, no core-file changes).

* load_phase4_model : build the architecture EXACTLY as Phase 4 did
  (build_model(pretrained=True) -> load_state_dict(strict=True) from the clean
  Phase-4 checkpoint). pretrained=True guarantees identical forward behaviour
  (e.g. torchvision GoogLeNet transform_input=True). Weights are then overwritten
  by the checkpoint.
* predict_logits_probs : forward a split, optionally with deterministic
  test-time augmentation (TTA), return per-sample logits + softmax probs + labels
  + sample ids.

TTA policy (clinically justified, matches the training-aug constraints):
  - hflip : original + horizontal flip, mean of softmax probs. Horizontal flip is
            exact under per-channel normalization (flip commutes with (x-mean)/std)
            and KL grade is side-agnostic (L/R already normalised per file).
  - hflip_rot : + rotations of +/- 7 deg (fill = 0 in normalised space ~= mid-grey
            background; a small approximation). Off by default.
Vertical flip / large rotations are never used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common import PROJECT_ROOT

TTA_VIEWS = {
    "none": ["id"],
    "hflip": ["id", "hflip"],
    "hflip_rot": ["id", "hflip", "rot+7", "rot-7", "hflip_rot+7", "hflip_rot-7"],
}


def load_phase4_model(name: str, cfg: dict, device, phase4_dir: Path | None = None):
    from src.benchmark.models import build_model
    phase4_dir = Path(phase4_dir or (PROJECT_ROOT / "models" / "phase4"))
    ckpt = phase4_dir / name / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    built = build_model(name, num_classes=cfg["num_classes"],
                        in_chans=cfg["in_channels"], pretrained=True)
    built.model.load_state_dict(ck["state_dict"], strict=True)
    built.model.to(device).eval()
    return built, ck


def _apply_view(x: torch.Tensor, view: str) -> torch.Tensor:
    """x: (B,C,H,W) already normalised."""
    if view == "id":
        return x
    if view == "hflip":
        return torch.flip(x, dims=[-1])
    from torchvision.transforms.v2 import functional as F
    if view == "rot+7":
        return F.rotate(x, 7, fill=0.0)
    if view == "rot-7":
        return F.rotate(x, -7, fill=0.0)
    if view == "hflip_rot+7":
        return F.rotate(torch.flip(x, dims=[-1]), 7, fill=0.0)
    if view == "hflip_rot-7":
        return F.rotate(torch.flip(x, dims=[-1]), -7, fill=0.0)
    raise ValueError(view)


@torch.inference_mode()
def predict_logits_probs(model, loader, device, *, tta: str = "none",
                         amp: bool = True) -> dict:
    views = TTA_VIEWS[tta]
    all_true, all_prob, all_logit, all_id, all_side = [], [], [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        x = x.to(device, non_blocking=True)
        prob_sum = None
        logit_sum = None
        for v in views:
            xv = _apply_view(x, v)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(xv)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            out = out.float()
            p = torch.softmax(out, dim=1)
            prob_sum = p if prob_sum is None else prob_sum + p
            logit_sum = out if logit_sum is None else logit_sum + out
        prob = (prob_sum / len(views)).cpu().numpy()
        logit = (logit_sum / len(views)).cpu().numpy()
        all_true.append(y.numpy())
        all_prob.append(prob)
        all_logit.append(logit)
        if meta is not None:
            n = len(y)
            all_id += [meta["sample_id"][i] for i in range(n)]
            all_side += [meta["knee_side"][i] for i in range(n)]
    return {
        "y_true": np.concatenate(all_true).astype(int),
        "y_prob": np.concatenate(all_prob).astype(np.float32),
        "y_logit": np.concatenate(all_logit).astype(np.float32),
        "sample_id": np.array(all_id, dtype=object) if all_id else None,
        "knee_side": np.array(all_side, dtype=object) if all_side else None,
        "tta": tta, "views": views,
    }
