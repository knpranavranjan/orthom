"""Model complexity: parameter counts, GMACs (fvcore), checkpoint size.

fvcore's ``FlopCountAnalysis`` counts one fused multiply-add as 1 -> we report
this as **GMACs** (not FLOPs). FLOPs ~= 2 x GMACs. The column is labelled
``gmacs`` everywhere; conversions are never invented (spec section 25).
"""
from __future__ import annotations

import os
import warnings


def param_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
        "parameters_millions": round(total / 1e6, 3),
    }


def gmacs_fvcore(model, input_shape=(1, 3, 224, 224), device="cpu") -> dict:
    """Return {'gmacs', 'method', 'unsupported_ops'} or an error note."""
    import torch
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:  # noqa: BLE001
        return {"gmacs": None, "method": "fvcore-missing", "error": str(exc)}

    was_training = model.training
    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            total = fca.total()
            unsupported = dict(fca.unsupported_ops())
        out = {"gmacs": round(total / 1e9, 4), "method": "fvcore.FlopCountAnalysis (MAC=1)",
               "flops_estimate_g": round(2 * total / 1e9, 4),
               "unsupported_ops": {k: int(v) for k, v in unsupported.items()} or None}
    except Exception as exc:  # noqa: BLE001
        out = {"gmacs": None, "method": "fvcore-failed", "error": str(exc)}
    finally:
        if was_training:
            model.train()
    return out


def checkpoint_size_mb(path) -> float | None:
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 3)
    except OSError:
        return None
