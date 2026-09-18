"""Single-image (batch=1) inference latency + peak memory (spec sections 26-27).

GPU latency is for development only. Raspberry-Pi / edge numbers are NOT inferred
from this (spec section 26) - a separate on-device benchmark comes in Phase 8.
"""
from __future__ import annotations

import statistics
import time


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(model, device, input_shape=(1, 3, 224, 224),
                    warmup: int = 20, iters: int = 100) -> dict:
    import torch

    model = model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            model(x)
        _sync(device)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1000.0)  # ms

    samples.sort()
    mean = statistics.fmean(samples)
    return {
        "device": device.type,
        "input_shape": list(input_shape),
        "warmup_iters": warmup,
        "timed_iters": iters,
        "mean_ms": round(mean, 4),
        "median_ms": round(statistics.median(samples), 4),
        "std_ms": round(statistics.pstdev(samples), 4),
        "min_ms": round(samples[0], 4),
        "max_ms": round(samples[-1], 4),
        "p90_ms": round(samples[min(len(samples) - 1, int(0.9 * len(samples)))], 4),
        "throughput_images_per_sec": round(1000.0 / mean, 2) if mean > 0 else None,
    }


def peak_memory_mb(device) -> dict:
    """Peak CUDA memory since last reset + current process RSS."""
    import torch
    out: dict = {}
    if device.type == "cuda":
        out["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
        out["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / 1e6, 2)
    try:
        import psutil
        out["process_rss_mb"] = round(psutil.Process().memory_info().rss / 1e6, 2)
    except Exception:  # noqa: BLE001
        out["process_rss_mb"] = None
    return out


def reset_peak_memory(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
