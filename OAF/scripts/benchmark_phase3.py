"""PHASE 3 - PART A + B : extended 15-CNN benchmark.

    python scripts/benchmark_phase3.py --dry-run          # build + few batches for the 5 NEW models
    python scripts/benchmark_phase3.py                     # train the 5 NEW models, then merge -> 15-model report
    python scripts/benchmark_phase3.py --models vgg16,vgg19
    python scripts/benchmark_phase3.py --include-attention # also train resnet18/convnext_tiny spatial-attention

The 10 Phase-2 models are NOT retrained; their frozen results
(models/benchmark/<m>/result.json) are merged into the extended report.
The test split is never loaded. Selection metric = validation Macro-F1.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("phase3")


def _resolve(arg: str, cfg: dict, include_attn: bool) -> list[str]:
    zoo = list(cfg["models"].keys())
    if include_attn:
        zoo = zoo + list(cfg.get("attention_models", []))
    if arg in ("all", "", None):
        return zoo
    picked = [m.strip() for m in arg.split(",") if m.strip()]
    known = set(zoo) | set(cfg.get("attention_models", []))
    bad = [m for m in picked if m not in known]
    if bad:
        raise SystemExit(f"unknown models: {bad}\navailable: {sorted(known)}")
    return picked


def _load_phase2_results(cfg: dict) -> list[dict]:
    d = PROJECT_ROOT / cfg.get("phase2_results_dir", "models/benchmark")
    out = []
    for m in cfg.get("phase2_models", []):
        rj = d / m / "result.json"
        if rj.exists():
            r = json.loads(rj.read_text())
            r.setdefault("phase", "phase2")
            r.setdefault("preprocessing_variant", "basic")
            r.setdefault("loss_name", "weighted_cross_entropy")
            r.setdefault("seed", 42)
            out.append(r)
        else:
            LOG.warning("Phase-2 result missing: %s (skipped from merged table)", rj)
    return out


def _copy_phase2_plots(cfg: dict, reports_dir: Path) -> None:
    src = PROJECT_ROOT / "reports" / "benchmark"
    for sub in ("confusion_matrices", "training_curves", "classification_reports",
                "model_predictions", "latency"):
        s = src / sub
        if not s.exists():
            continue
        dst = reports_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in s.iterdir():
            if f.is_file() and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark_phase3.yaml"))
    ap.add_argument("--models", default="all", help="'all' (5 new) or comma list")
    ap.add_argument("--include-attention", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stage-a-epochs", type=int, default=None)
    ap.add_argument("--fast-dev", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None, help="override for ALL models")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="train only; skip 15-model report merge")
    ap.add_argument("--log-file", type=Path, default=Path("reports/phase3/run_extended.log"),
                    help="also write the full run log here (no shell piping needed)")
    args = ap.parse_args()

    if args.log_file:
        add_file_logger(args.log_file)
        LOG.info("logging to %s", args.log_file)

    cfg = load_config(args.config)
    if args.stage_a_epochs is not None:
        cfg["transfer_learning"]["stage_a_epochs"] = args.stage_a_epochs
    models = _resolve(args.models, cfg, args.include_attention)

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import collect_env, pick_device
    from src.benchmark.checks import leakage_and_distribution_check
    from src.benchmark.runner import run_one_model, dry_run_model

    seed_report = set_seed(cfg["seed"], deterministic=True)
    device = pick_device(args.device)
    env = collect_env(seed_report)

    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 74)
    LOG.info("PHASE 3 EXTENDED BENCHMARK | new models=%s | dry_run=%s | device=%s",
             models, args.dry_run, device)
    LOG.info("torch=%s cuda=%s timm=%s torchvision=%s gpu=%s vram=%.1fGB",
             env["torch"], env["cuda_version"], env.get("timm"), env.get("torchvision"),
             env["gpu"]["name"] if env["gpu"] else "none",
             env["gpu"]["total_memory_gb"] if env["gpu"] else 0.0)
    LOG.info("=" * 74)

    LOG.info("leakage + class-distribution check")
    dist = leakage_and_distribution_check(cfg["preprocessing_variant"], logger=LOG)
    (reports_dir / "reproducibility.json").write_text(json.dumps(
        {**env, "benchmark_config": cfg, "class_distribution": dist["class_distribution"],
         "patients_per_split": dist["patients"], "models_requested": models,
         "phase": "phase3"}, indent=2, default=str))

    # ---------------- DRY RUN ----------------
    if args.dry_run:
        LOG.info("--- DRY RUN (no training) ---")
        res = []
        for m in models:
            try:
                res.append(dry_run_model(m, cfg, device, logger=LOG))
            except Exception as exc:  # noqa: BLE001
                LOG.error("[dry-run] %s FAILED: %s", m, exc)
                traceback.print_exc()
                res.append({"model": m, "ok": False, "issues": [str(exc)]})
        (reports_dir / "dry_run_report.json").write_text(json.dumps(res, indent=2, default=str))
        ok = all(r.get("ok") for r in res)
        for r in res:
            LOG.info("  %-32s %s %s", r["model"], "OK" if r.get("ok") else "FAIL",
                     r.get("issues") or "")
        LOG.info("DRY RUN %s -> %s", "PASSED" if ok else "FAILED",
                 reports_dir / "dry_run_report.json")
        sys.exit(0 if ok else 1)

    # ---------------- TRAIN THE NEW MODELS ----------------
    pmbs = cfg.get("per_model_batch_size", {}) or {}
    fast = dict(max_train_batches=6, max_val_batches=4) if args.fast_dev else {}
    new_results = []
    t0 = time.perf_counter()
    for i, m in enumerate(models, 1):
        rj = models_dir / m / "result.json"
        if args.skip_existing and rj.exists():
            LOG.info("[%d/%d] skip %s (exists)", i, len(models), m)
            new_results.append(json.loads(rj.read_text()))
            continue
        bs = args.batch_size or pmbs.get(m) or cfg["train"]["batch_size"]
        LOG.info("[%d/%d] ==== %s ====  (batch_size=%d)", i, len(models), m, bs)
        try:
            r = run_one_model(m, cfg, device, reports_dir=reports_dir, models_dir=models_dir,
                              max_epochs=args.epochs, num_workers=args.num_workers, batch_size=bs,
                              env={"torch": env["torch"], "cuda": env["cuda_version"],
                                   "gpu": env["gpu"]["name"] if env["gpu"] else None,
                                   "timm": env.get("timm"), "torchvision": env.get("torchvision")},
                              extra_config={"phase": "phase3"}, logger=LOG, **fast)
            r["phase"] = "phase3"
            new_results.append(r)
        except Exception as exc:  # noqa: BLE001
            LOG.error("MODEL %s FAILED: %s", m, exc)
            traceback.print_exc()
            new_results.append({"model": m, "error": str(exc)})
    LOG.info("new models trained in %.1f min", (time.perf_counter() - t0) / 60)

    if args.no_merge:
        return

    # ---------------- MERGE 10 (phase 2) + new -> 15-model extended report ----------------
    from src.benchmark.report import write_tables, write_complexity_csv, write_markdown_report
    from src.benchmark.plots import pareto_plots

    phase2 = _load_phase2_results(cfg)
    ok_new = [r for r in new_results if "error" not in r and "macro_f1" in r]
    combined = phase2 + ok_new
    if not combined:
        LOG.error("no results to merge"); sys.exit(1)

    _copy_phase2_plots(cfg, reports_dir)

    df = write_tables(combined, reports_dir, basename="extended_benchmark_results")
    comp_rows = [{"model": r["model"], "timm_name": r.get("timm_name"),
                  "parameters": r.get("parameters"),
                  "trainable_parameters": r.get("trainable_parameters"),
                  "non_trainable_parameters": r.get("non_trainable_parameters"),
                  "parameters_millions": round((r.get("parameters") or 0) / 1e6, 3),
                  "gmacs": r.get("gmacs"), "flops_estimate_g": r.get("flops_estimate_g"),
                  "model_size_mb": r.get("model_size_mb"), "gmacs_method": r.get("gmacs_method")}
                 for r in combined]
    write_complexity_csv(comp_rows, reports_dir)
    pareto_plots(combined, reports_dir)

    context = {
        "timestamp_utc": env["timestamp_utc"], "seed": cfg["seed"],
        "variant": cfg["preprocessing_variant"], "normalization": cfg["data"]["normalization"],
        "class_distribution": dist["class_distribution"],
        "disabled_augmentations": ["vertical_flip", "arbitrary_rotation", "heavy_color_jitter",
                                   "elastic_warp", "random_erasing", "large_random_crop"],
        "train_config": {
            "loss": "weighted cross-entropy (train-only class weights)",
            "optimizer": f"adamw lr(stageB)={cfg['optimizer']['lr']} wd={cfg['optimizer']['weight_decay']}",
            "scheduler": cfg["scheduler"]["name"],
            "transfer_learning": (f"Stage A {cfg['transfer_learning']['stage_a_epochs']} ep head-only "
                                  f"@lr {cfg['transfer_learning']['stage_a_lr']}, then Stage B full "
                                  f"fine-tune @lr {cfg['transfer_learning']['stage_b_lr']}"),
            "batch_size": f"{cfg['train']['batch_size']} (vgg16/vgg19: "
                          f"{cfg.get('per_model_batch_size', {}).get('vgg16', 64)})",
            "max_epochs": cfg["train"]["max_epochs"],
            "early_stopping": f"monitor val_macro_f1, patience {cfg['train']['early_stopping']['patience']}",
            "amp": cfg["train"]["amp"], "seed": cfg["seed"],
            "input": "3x224x224 (3-ch replicated grayscale, ImageNet norm)",
        },
    }
    write_markdown_report(df, context, reports_dir / "final_phase3_report.md",
                          plot_index={"confusion_dir": "reports/phase3/confusion_matrices/",
                                      "curves_dir": "reports/phase3/training_curves/",
                                      "pareto_dir": "reports/phase3/"})
    LOG.info("wrote %s", reports_dir / "extended_benchmark_results.csv")
    LOG.info("wrote %s", reports_dir / "final_phase3_report.md")
    LOG.info("15-MODEL TABLE (val, by Macro-F1):\n%s",
             df[["model", "macro_f1", "quadratic_weighted_kappa", "balanced_accuracy",
                 "kl4_f1", "parameters", "mean_latency_ms", "phase"]].to_string(index=False))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
