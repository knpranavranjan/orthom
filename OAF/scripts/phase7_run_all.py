"""PHASE 7 orchestrator - run every remaining stage unattended and stop.

    combine  ->  multiseed  ->  freeze winners  ->  final TEST eval  ->
    ONNX export  ->  edge latency benchmark  ->  lightweight dashboard bundle

Assumes the `matrix` stage is already in reports/phase7/EXPERIMENT_REGISTRY.csv.
Safe to re-run: each stage skips experiment_ids already present; export/edge/package
just overwrite their own outputs.

    python scripts/phase7_run_all.py --log-file reports/phase7/p7_runall.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402

LOG = get_logger("p7all")
PY = sys.executable
REG = PROJECT_ROOT / "reports" / "phase7" / "EXPERIMENT_REGISTRY.csv"
ARCHES = ("mobilenet_v2", "efficientnet_b0", "convnext_tiny")


def sh(cmd: list[str], tag: str):
    LOG.info(">>> %s\n    %s", tag, " ".join(str(c) for c in cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    LOG.info("<<< %s  exit=%d  (%.0fs)", tag, r.returncode, time.perf_counter() - t0)
    if r.returncode != 0:
        LOG.warning("stage %s returned non-zero; continuing", tag)
    return r.returncode


def best_per_arch(df: pd.DataFrame, exportable_only: bool = False) -> dict[str, str]:
    out = {}
    for a in ARCHES:
        sub = df[df["experiment_id"].str.startswith(a + "__")].copy()
        if exportable_only:
            sub = sub[sub["ordinal"].fillna("") != "corn"]
        sub = sub.dropna(subset=["val_macro_f1"])
        if len(sub):
            out[a] = sub.sort_values("val_macro_f1", ascending=False).iloc[0]["experiment_id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default="reports/phase7/p7_runall.log")
    ap.add_argument("--skip-multiseed", action="store_true")
    args = ap.parse_args()
    add_file_logger(args.log_file)
    LOG.info("=== PHASE 7 run-all start ===")

    bp7 = ["scripts/benchmark_phase7.py"]
    sh([PY, *bp7, "--stage", "combine", "--log-file", "reports/phase7/p7_combine.log"], "combine")
    if not args.skip_multiseed:
        sh([PY, *bp7, "--stage", "multiseed", "--log-file", "reports/phase7/p7_seed.log"], "multiseed")

    df = pd.read_csv(REG)
    if "ordinal" not in df.columns:
        df["ordinal"] = ""
    # de-dupe multi-seed rows down to their base id for winner selection: keep the
    # single best row per experiment_id (seeds share an id prefix but are distinct rows)
    true_best = best_per_arch(df)                       # may be a CORN config
    exp_best = best_per_arch(df, exportable_only=True)  # softmax head -> ONNX-able
    LOG.info("frozen winners (val Macro-F1):")
    for a in ARCHES:
        LOG.info("  %-16s eval=%s  export=%s", a, true_best.get(a), exp_best.get(a))
    (PROJECT_ROOT / "reports" / "phase7" / "frozen_winners.json").write_text(
        json.dumps({"eval": true_best, "export": exp_best}, indent=2))

    # ---- final TEST eval on the true best per arch ----
    eval_ids = sorted(set(true_best.values()) | set(exp_best.values()))
    cmd = [PY, "scripts/phase7_final_eval.py", "--log-file", "reports/phase7/p7_finaleval.log"]
    for e in eval_ids:
        cmd += ["--experiment-id", e]
    sh(cmd, "final_eval")

    # ---- ONNX export (exportable winner per arch) ----
    cmd = [PY, "scripts/export_onnx.py", "--config", "configs/benchmark_phase7.yaml",
           "--out-dir", "models/deploy"]
    for a in ARCHES:
        eid = exp_best.get(a)
        if not eid:
            continue
        cmd += ["--model", a, "--checkpoint", f"models/phase7/{eid}/best.pt", "--name", f"{a}_oa"]
    sh(cmd, "export_onnx")

    # ---- edge latency benchmark ----
    sh([PY, "scripts/benchmark_edge.py", "--real-images", "128"], "benchmark_edge")

    # ---- package lightweight bundle ----
    sh([PY, "scripts/phase7_package.py",
        "--model", "mobilenet_v2_oa", "--model", "efficientnet_b0_oa", "--model", "convnext_tiny_oa",
        "--default", "mobilenet_v2_oa"], "package")

    LOG.info("=== PHASE 7 run-all done ===  see reports/phase7/FINAL_MOBILENETV2_BENCHMARK.csv "
             "+ FINAL_PHASE7_REPORT.md + EDGE_BENCHMARK.csv + dist/aether_oa_xray_dashboard/")


if __name__ == "__main__":
    main()
