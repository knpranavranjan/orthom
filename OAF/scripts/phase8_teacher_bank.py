"""PHASE 8 - build the teacher-logit bank for MobileNetV2 knowledge distillation.

Runs each frozen teacher ONNX (VGG16, ConvNeXt-Tiny, ResNet18 by default) over
every processed TRAIN + VAL image (basic variant, no augmentation) and caches raw
logits keyed by sample_id:

    reports/phase8/teacher_logits/<split>__<teacher>.npz   { sample_id, logits }

TEST images are never touched. Teachers are pure feature extractors here.

    python scripts/phase8_teacher_bank.py
    python scripts/phase8_teacher_bank.py --teacher vgg16 --teacher convnext_tiny
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402
from src.deploy.preprocess import preprocess  # noqa: E402

LOG = get_logger("tbank")
DEPLOY = PROJECT_ROOT / "models" / "deploy"
OUT = PROJECT_ROOT / "reports" / "phase8" / "teacher_logits"
DEFAULT_TEACHERS = ["vgg16", "convnext_tiny", "resnet18"]


def run_teacher(onnx_fp: Path, rows: pd.DataFrame, batch: int = 64) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_fp), sess_options=so,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    logits = np.zeros((len(rows), 5), dtype=np.float32)
    buf, idx = [], []
    t0 = time.perf_counter()
    for i, (_, r) in enumerate(rows.iterrows()):
        x = preprocess(str(PROJECT_ROOT / r["processed_path"]), already_224=True)[0]
        buf.append(x); idx.append(i)
        if len(buf) == batch or i == len(rows) - 1:
            z = sess.run([out], {inp: np.stack(buf).astype(np.float32)})[0]
            logits[idx] = z[:, :5]
            buf, idx = [], []
        if (i + 1) % 1000 == 0:
            LOG.info("    %d/%d (%.0fs)", i + 1, len(rows), time.perf_counter() - t0)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", default=[], help="deploy ONNX basename (repeat)")
    ap.add_argument("--variant", default="basic")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    args = ap.parse_args()
    teachers = args.teacher or DEFAULT_TEACHERS
    OUT.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(PROJECT_ROOT / "metadata" / f"processed_manifest_{args.variant}.csv")
    LOG.info("teacher bank: teachers=%s splits=%s variant=%s", teachers, args.splits, args.variant)

    summary = []
    for t in teachers:
        fp = DEPLOY / f"{t}.onnx"
        if not fp.exists():
            LOG.warning("SKIP %s - %s not found", t, fp)
            continue
        for split in args.splits:
            rows = man[man["split"] == split].reset_index(drop=True)
            LOG.info("[%s / %s] %d images", t, split, len(rows))
            lg = run_teacher(fp, rows)
            npz = OUT / f"{split}__{t}.npz"
            np.savez_compressed(npz, sample_id=rows["sample_id"].astype(str).to_numpy(),
                                logits=lg.astype(np.float32))
            acc = float((lg.argmax(1) == rows["kl_grade"].to_numpy()).mean())
            LOG.info("  -> %s  (self-acc on %s = %.4f)", npz.name, split, acc)
            summary.append({"teacher": t, "split": split, "n": len(rows), "acc": round(acc, 4)})

    pd.DataFrame(summary).to_csv(OUT.parent / "teacher_bank_summary.csv", index=False)
    LOG.info("done -> %s", OUT)


if __name__ == "__main__":
    main()
