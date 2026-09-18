"""PHASE 8 orchestrator - finish the KD + label-scheme work unattended.

    KD multiseed (top-2)  ->  KD final TEST eval  ->  label-scheme sweep
      (5class / 4class_kl01 / 3class / binary, same best KD recipe)

Assumes the KD matrix (K0..K7) is already in reports/phase8/KD_REGISTRY.csv.

    python scripts/phase8_run_all.py --log-file reports/phase8/p8_runall.log
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402

LOG = get_logger("p8all")
PY = sys.executable
KD_REG = PROJECT_ROOT / "reports" / "phase8" / "KD_REGISTRY.csv"


def sh(cmd, tag):
    LOG.info(">>> %s\n    %s", tag, " ".join(str(c) for c in cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    LOG.info("<<< %s exit=%d (%.0fs)", tag, r.returncode, time.perf_counter() - t0)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default="reports/phase8/p8_runall.log")
    ap.add_argument("--skip-multiseed", action="store_true")
    args = ap.parse_args()
    add_file_logger(args.log_file)
    LOG.info("=== PHASE 8 run-all ===")

    if not args.skip_multiseed:
        sh([PY, "scripts/benchmark_phase8.py", "--stage", "multiseed",
            "--log-file", "reports/phase8/p8_seed.log"], "kd_multiseed")

    # best KD run overall (matrix + multiseed) for the final TEST eval
    rows = [r for r in csv.DictReader(open(KD_REG)) if r.get("val_macro_f1")]
    rows.sort(key=lambda r: float(r["val_macro_f1"]), reverse=True)
    top = [r["experiment_id"] for r in rows[:3]]
    LOG.info("KD finalists for test eval: %s", top)
    cmd = [PY, "scripts/phase7_final_eval.py", "--config", "configs/benchmark_phase8.yaml",
           "--tag", "phase8_kd", "--log-file", "reports/phase8/p8_finaleval.log"]
    for e in top:
        cmd += ["--experiment-id", e]
    sh(cmd, "kd_final_eval")

    # label-scheme sweep (auto-picks best KD recipe from the registry)
    sh([PY, "scripts/benchmark_phase8_schemes.py", "--log-file", "reports/phase8/p8_schemes.log"],
       "label_scheme_sweep")

    LOG.info("=== PHASE 8 run-all done ===  see reports/phase8/FINAL_MOBILENETV2_BENCHMARK_phase8_kd.csv "
             "+ LABEL_SCHEME_SWEEP.csv/.md")


if __name__ == "__main__":
    main()
