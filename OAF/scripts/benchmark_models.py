"""PHASE 2 - run the full 10-CNN benchmark (or a subset), then build the report.

    python scripts/benchmark_models.py --dry-run                 # verify plumbing only
    python scripts/benchmark_models.py --models resnet18 --epochs 2   # sanity
    python scripts/benchmark_models.py                            # all 10, full run
    python scripts/benchmark_models.py --models resnet18,resnet50 # subset

The test split is never loaded for selection. Selection metric = validation Macro-F1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("benchmark")


def _resolve_models(arg: str, cfg: dict) -> list[str]:
    zoo = list(cfg["models"].keys())
    if arg in ("all", "", None):
        return zoo
    picked = [m.strip() for m in arg.split(",") if m.strip()]
    unknown = [m for m in picked if m not in zoo]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}\navailable: {zoo}")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    ap.add_argument("--models", default="all", help="'all' or comma list")
    ap.add_argument("--dry-run", action="store_true", help="build + few batches, no training")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (sanity)")
    ap.add_argument("--stage-a-epochs", type=int, default=None,
                    help="override head-only warmup epochs (default 3)")
    ap.add_argument("--fast-dev", action="store_true", help="limit batches/epoch")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip models that already have models/benchmark/<m>/result.json")
    ap.add_argument("--no-report", action="store_true", help="skip final report assembly")
    ap.add_argument("--log-file", type=Path, default=Path("reports/benchmark/run.log"),
                    help="also write the full run log here (no shell piping needed)")
    args = ap.parse_args()
    if args.log_file:
        add_file_logger(args.log_file)

    cfg = load_config(args.config)
    if args.variant:
        cfg["preprocessing_variant"] = args.variant
    if args.stage_a_epochs is not None:
        cfg["transfer_learning"]["stage_a_epochs"] = args.stage_a_epochs
    models = _resolve_models(args.models, cfg)

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

    LOG.info("=" * 72)
    LOG.info("PHASE 2 BENCHMARK | models=%s | dry_run=%s | device=%s", models, args.dry_run, device)
    LOG.info("torch=%s cuda=%s timm=%s gpu=%s vram=%.1fGB", env["torch"], env["cuda_version"],
             env.get("timm"), env["gpu"]["name"] if env["gpu"] else "none",
             env["gpu"]["total_memory_gb"] if env["gpu"] else 0.0)
    LOG.info("=" * 72)

    LOG.info("STEP 43-44: leakage + class-distribution check")
    dist = leakage_and_distribution_check(cfg["preprocessing_variant"], logger=LOG)

    # reproducibility metadata (spec section 15)
    repro = {**env, "benchmark_config": cfg, "class_distribution": dist["class_distribution"],
             "patients_per_split": dist["patients"], "models_requested": models}
    (reports_dir / "reproducibility.json").write_text(json.dumps(repro, indent=2, default=str))
    LOG.info("wrote %s", reports_dir / "reproducibility.json")

    # ---------------- DRY RUN ----------------
    if args.dry_run:
        LOG.info("--- DRY RUN (no training) ---")
        results = []
        for m in models:
            try:
                results.append(dry_run_model(m, cfg, device, logger=LOG))
            except Exception as exc:  # noqa: BLE001
                LOG.error("[dry-run] %s FAILED: %s", m, exc)
                traceback.print_exc()
                results.append({"model": m, "ok": False, "issues": [str(exc)]})
        (reports_dir / "dry_run_report.json").write_text(json.dumps(results, indent=2, default=str))
        ok = all(r.get("ok") for r in results)
        LOG.info("DRY RUN %s -> %s", "PASSED" if ok else "FAILED", reports_dir / "dry_run_report.json")
        for r in results:
            LOG.info("  %-20s %s %s", r["model"], "OK" if r.get("ok") else "FAIL",
                     r.get("issues") or "")
        sys.exit(0 if ok else 1)

    # ---------------- FULL / SANITY RUN ----------------
    fast = dict(max_train_batches=6, max_val_batches=4) if args.fast_dev else {}
    all_results, history_rows, complexity_rows = [], [], []
    t_start = time.perf_counter()
    for i, m in enumerate(models, 1):
        rj = models_dir / m / "result.json"
        if args.skip_existing and rj.exists():
            LOG.info("[%d/%d] skip %s (result.json exists)", i, len(models), m)
            all_results.append(json.loads(rj.read_text()))
            continue
        LOG.info("[%d/%d] ==== %s ====", i, len(models), m)
        try:
            res = run_one_model(
                m, cfg, device, reports_dir=reports_dir, models_dir=models_dir,
                max_epochs=args.epochs, num_workers=args.num_workers,
                batch_size=args.batch_size,
                env={"torch": env["torch"], "cuda": env["cuda_version"],
                     "gpu": env["gpu"]["name"] if env["gpu"] else None, "timm": env.get("timm")},
                logger=LOG, **fast)
            all_results.append(res)
            for h in json.loads((models_dir / m / "history.json").read_text()):
                history_rows.append(h)
            comp = json.loads((models_dir / m / "result.json").read_text())
            complexity_rows.append({
                "model": m, "timm_name": comp["timm_name"],
                "parameters": comp["parameters"], "trainable_parameters": comp["trainable_parameters"],
                "non_trainable_parameters": comp["non_trainable_parameters"],
                "parameters_millions": round(comp["parameters"] / 1e6, 3),
                "gmacs": comp["gmacs"], "flops_estimate_g": comp.get("flops_estimate_g"),
                "model_size_mb": comp["model_size_mb"], "gmacs_method": comp.get("gmacs_method")})
        except Exception as exc:  # noqa: BLE001
            LOG.error("MODEL %s FAILED: %s", m, exc)
            traceback.print_exc()
            all_results.append({"model": m, "error": str(exc)})

    total_min = (time.perf_counter() - t_start) / 60
    LOG.info("all models done in %.1f min", total_min)

    if history_rows:
        pd.DataFrame(history_rows).to_csv(reports_dir / "training_history.csv", index=False)

    ok_results = [r for r in all_results if "error" not in r and "macro_f1" in r]
    if not ok_results:
        LOG.error("no successful models - skipping report")
        sys.exit(1)

    from src.benchmark.report import write_tables, write_complexity_csv, write_markdown_report
    from src.benchmark.plots import pareto_plots

    df = write_tables(ok_results, reports_dir)
    if complexity_rows:
        write_complexity_csv(complexity_rows, reports_dir)
    pareto_plots(ok_results, reports_dir)

    if not args.no_report:
        context = {
            "timestamp_utc": env["timestamp_utc"], "seed": cfg["seed"],
            "variant": cfg["preprocessing_variant"],
            "normalization": cfg["data"].get("normalization"),
            "class_distribution": dist["class_distribution"],
            "disabled_augmentations": ["vertical_flip", "arbitrary_rotation", "heavy_color_jitter",
                                      "elastic_warp", "random_erasing", "large_random_crop"],
            "train_config": {
                "loss": "weighted cross-entropy (train-only class weights)",
                "optimizer": f"{cfg['optimizer']['name']} lr(stageB)={cfg['optimizer']['lr']} "
                             f"wd={cfg['optimizer']['weight_decay']}",
                "scheduler": cfg.get("scheduler", {}).get("name"),
                "transfer_learning": f"Stage A {cfg['transfer_learning']['stage_a_epochs']} ep "
                                     f"head-only @lr {cfg['transfer_learning']['stage_a_lr']}, "
                                     f"then Stage B full fine-tune @lr {cfg['transfer_learning']['stage_b_lr']}",
                "batch_size": cfg["train"]["batch_size"],
                "max_epochs": args.epochs or cfg["train"]["max_epochs"],
                "early_stopping": f"monitor {cfg['train']['early_stopping']['monitor']}, "
                                  f"patience {cfg['train']['early_stopping']['patience']}",
                "amp": cfg["train"].get("amp"), "seed": cfg["seed"],
                "input": f"{cfg['in_channels']}x{cfg['image_size'][0]}x{cfg['image_size'][1]} "
                         f"(3-ch replicated grayscale, ImageNet norm)",
            },
        }
        write_markdown_report(df, context, reports_dir / "final_benchmark_report.md",
                              plot_index={"confusion_dir": "reports/benchmark/confusion_matrices/",
                                          "curves_dir": "reports/benchmark/training_curves/",
                                          "pareto_dir": "reports/benchmark/"})
        LOG.info("wrote %s", reports_dir / "final_benchmark_report.md")

    LOG.info("TOP by validation Macro-F1:\n%s",
             df[["model", "macro_f1", "quadratic_weighted_kappa", "balanced_accuracy",
                 "kl4_f1", "parameters", "mean_latency_ms"]].head(10).to_string(index=False))
    LOG.info("Recommended (primary, highest val Macro-F1): %s", df.iloc[0]["model"])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
