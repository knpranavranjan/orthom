"""Global seeding + determinism control (spec section 15)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> dict:
    """Seed Python / NumPy / torch / CUDA. Returns a small report dict.

    ``deterministic=True`` enables cuDNN deterministic mode and
    ``torch.use_deterministic_algorithms`` in warn-only mode so training does not
    crash on ops without a deterministic kernel (those are logged instead).
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    report = {"seed": seed, "deterministic_requested": deterministic}
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report["use_deterministic_algorithms"] = "warn_only"
        except Exception as exc:  # noqa: BLE001
            report["use_deterministic_algorithms"] = f"failed: {exc}"
    else:
        torch.backends.cudnn.benchmark = True
        report["use_deterministic_algorithms"] = False
    return report


def seed_worker(worker_id: int) -> None:  # DataLoader worker_init_fn
    import torch
    wseed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(wseed)
    random.seed(wseed)
