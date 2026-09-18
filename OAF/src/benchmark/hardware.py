"""Hardware / software fingerprint for the reproducibility metadata file."""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def collect_env(seed_report: dict | None = None) -> dict:
    import torch
    import torchvision

    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or _safe(lambda: platform.uname().processor),
        "cpu_count_logical": _safe(lambda: __import__("os").cpu_count()),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": _safe(lambda: torch.backends.cudnn.version()),
        "cudnn_deterministic": _safe(lambda: torch.backends.cudnn.deterministic),
        "cudnn_benchmark": _safe(lambda: torch.backends.cudnn.benchmark),
    }
    info["timm"] = _safe(lambda: __import__("timm").__version__)
    info["numpy"] = _safe(lambda: __import__("numpy").__version__)
    info["fvcore"] = _safe(lambda: __import__("fvcore").__version__, "not-installed")
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 1)
        info["psutil"] = psutil.__version__
    except Exception:  # noqa: BLE001
        info["psutil"] = "not-installed"

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu"] = {
            "name": torch.cuda.get_device_name(idx),
            "index": idx,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "multi_processor_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
            "device_count": torch.cuda.device_count(),
        }
    else:
        info["gpu"] = None
    if seed_report:
        info["seeding"] = seed_report
    return info


def pick_device(prefer: str = "auto"):
    import torch
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
