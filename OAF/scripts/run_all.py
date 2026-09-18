"""Run the full preprocessing pipeline end to end (STEP 4 -> STEP 18).

    python scripts/run_all.py            # full run, all 3 contrast variants
    python scripts/run_all.py --quick    # single 'basic' variant

Stops BEFORE any CNN training, by design.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*cmd: str) -> None:
    print(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}", flush=True)
    subprocess.run([PY, *cmd], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="only the 'basic' variant")
    args = ap.parse_args()
    variants = "basic" if args.quick else "all"

    run("scripts/audit_dataset.py", "--source", "kaggle",
        "--input-dir", "kaggle", "--out", "reports/dataset_audit_kaggle.csv")
    run("scripts/audit_dataset.py", "--source", "oai",
        "--oai-summary", "OAI-KL-Grade-Classification/data/OAI_summary.csv",
        "--out", "reports/dataset_audit_oai.csv")
    run("scripts/analyze_quality.py")
    run("scripts/analyze_duplicates.py")
    run("scripts/build_manifest.py")
    run("scripts/make_splits.py")
    run("scripts/compute_class_weights.py")
    run("scripts/compute_norm_stats.py")
    run("scripts/prepare_dataset.py", "--config", "configs/preprocessing.yaml",
        "--variants", variants)
    run("scripts/visual_report.py")
    run("scripts/validate_dataset.py")
    print("\nPipeline complete. CNN training is intentionally NOT started.")


if __name__ == "__main__":
    main()
