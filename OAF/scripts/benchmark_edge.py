"""EDGE / Raspberry-Pi latency benchmark for the deployable ONNX models.

Runs each models/deploy/*.onnx (and *.int8.onnx if present) through onnxruntime
on CPU at batch 1, single-thread and 4-thread, and reports median / mean / p90
latency, throughput, model size and process RSS. On an actual Raspberry Pi run
this same script there; on an x86 dev box the numbers are a (faster) proxy.

    python scripts/benchmark_edge.py                       # every models/deploy/*.onnx
    python scripts/benchmark_edge.py --model mobilenet_v2_oa convnext_tiny
    python scripts/benchmark_edge.py --real-images 128     # also time the full preprocess+predict path
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402

LOG = get_logger("edge")
DEPLOY = PROJECT_ROOT / "models" / "deploy"
OUT = PROJECT_ROOT / "reports" / "phase7"


def _rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:  # noqa: BLE001
        return float("nan")


def _bench_session(fp: Path, threads: int, n: int, warm: int) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(fp), sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(warm):
        sess.run(None, {inp: x})
    startup_ms = (time.perf_counter() - t0) * 1000 / max(warm, 1)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {inp: x})
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    rss = _rss_mb()
    del sess
    gc.collect()
    return {"threads": threads, "median_ms": round(ts[len(ts) // 2], 2),
            "mean_ms": round(sum(ts) / len(ts), 2), "p90_ms": round(ts[int(0.9 * len(ts))], 2),
            "min_ms": round(ts[0], 2), "fps": round(1000.0 / (sum(ts) / len(ts)), 1),
            "warm_first_infer_ms": round(startup_ms, 2), "rss_mb": rss}


def _bench_real(fp: Path, n: int) -> dict | None:
    """Full preprocess + calibrated predict path on real processed test PNGs."""
    try:
        import pandas as pd
        from src.deploy.predictor import KneeOAPredictor
    except Exception as e:  # noqa: BLE001
        LOG.warning("real-image path skipped: %s", e)
        return None
    man = pd.read_csv(PROJECT_ROOT / "metadata" / "processed_manifest_basic.csv")
    files = [PROJECT_ROOT / p for p in man[man["split"] == "test"]["processed_path"].head(n)]
    pred = KneeOAPredictor(fp, intra_threads=4)
    for f in files[:8]:
        pred.predict(str(f), already_224=True)
    ts = []
    for f in files:
        t = time.perf_counter()
        pred.predict(str(f), already_224=True)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"n": len(ts), "median_ms": round(ts[len(ts) // 2], 2),
            "mean_ms": round(sum(ts) / len(ts), 2), "p90_ms": round(ts[int(0.9 * len(ts))], 2),
            "fps": round(1000.0 / (sum(ts) / len(ts)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=[], help="deploy basenames (default: all)")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--real-images", type=int, default=0)
    ap.add_argument("--threads", nargs="*", type=int, default=[1, 4])
    args = ap.parse_args()

    fps = []
    for f in sorted(DEPLOY.glob("*.onnx")):
        base = f.name.replace(".int8.onnx", "").replace(".onnx", "")
        if args.model and base not in args.model:
            continue
        fps.append(f)
    if not fps:
        LOG.error("no ONNX files in %s (run scripts/export_onnx.py first)", DEPLOY)
        return

    rows = []
    for f in fps:
        size_mb = round(f.stat().st_size / 1e6, 2)
        quant = "int8" if f.name.endswith(".int8.onnx") else "fp32"
        base = f.name.replace(".int8.onnx", "").replace(".onnx", "")
        for th in args.threads:
            b = _bench_session(f, th, args.iters, args.warmup)
            row = {"model": base, "precision": quant, "size_mb": size_mb, **b}
            if args.real_images and quant == "fp32" and th == max(args.threads):
                rb = _bench_real(f, args.real_images)
                if rb:
                    row["real_pipeline_median_ms"] = rb["median_ms"]
                    row["real_pipeline_fps"] = rb["fps"]
            rows.append(row)
            LOG.info("%-22s %-4s %2dthr  median=%6.2fms  mean=%6.2fms  p90=%6.2fms  fps=%6.1f  size=%.1fMB  rss=%.0fMB",
                     base, quant, th, b["median_ms"], b["mean_ms"], b["p90_ms"], b["fps"], size_mb, b["rss_mb"])

    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "EDGE_BENCHMARK.csv", index=False)
    (OUT / "EDGE_BENCHMARK.json").write_text(json.dumps(rows, indent=2))
    LOG.info("wrote %s", OUT / "EDGE_BENCHMARK.csv")
    LOG.info("NOTE: run this same script ON the Raspberry Pi for real hardware numbers; "
             "x86 figures are a lower bound on Pi latency (~4-10x slower on a Pi 4 CPU).")


if __name__ == "__main__":
    main()
