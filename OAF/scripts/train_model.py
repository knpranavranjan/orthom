"""PHASE 2 - train + evaluate + profile ONE CNN.

    python scripts/train_model.py --model resnet18
    python scripts/train_model.py --model resnet50 --epochs 2 --fast-dev     # sanity
    python scripts/train_model.py --model convnext_tiny --batch-size 32

Never touches the test split. Selection = validation Macro-F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="benchmark model name (see configs/benchmark.yaml)")
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    ap.add_argument("--variant", default=None, help="override preprocessing variant (default: basic)")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--stage-a-epochs", type=int, default=None,
                    help="override head-only warmup epochs (default 3)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--fast-dev", action="store_true",
                    help="limit to a few train/val batches per epoch (smoke test)")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--reports-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.variant:
        cfg["preprocessing_variant"] = args.variant
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_amp:
        cfg["train"]["amp"] = False
    if args.stage_a_epochs is not None:
        cfg["transfer_learning"]["stage_a_epochs"] = args.stage_a_epochs

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import collect_env, pick_device
    from src.benchmark.checks import leakage_and_distribution_check
    from src.benchmark.runner import run_one_model

    seed_report = set_seed(cfg["seed"], deterministic=True)
    device = pick_device(args.device)
    env = collect_env(seed_report)
    LOG.info("device=%s | torch=%s | cuda=%s | gpu=%s",
             device, env["torch"], env["cuda_version"],
             env["gpu"]["name"] if env["gpu"] else "none")

    LOG.info("running leakage + class-distribution check ...")
    dist = leakage_and_distribution_check(cfg["preprocessing_variant"], logger=LOG)

    models_dir = args.models_dir or (PROJECT_ROOT / cfg["paths"]["models_dir"])
    reports_dir = args.reports_dir or (PROJECT_ROOT / cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    fast = dict(max_train_batches=6, max_val_batches=4) if args.fast_dev else {}
    result = run_one_model(
        args.model, cfg, device, reports_dir=reports_dir, models_dir=models_dir,
        max_epochs=args.epochs, num_workers=args.num_workers, batch_size=args.batch_size,
        env={"torch": env["torch"], "cuda": env["cuda_version"],
             "gpu": env["gpu"]["name"] if env["gpu"] else None,
             "timm": env.get("timm")},
        logger=LOG, **fast)

    LOG.info("DONE %s | val macroF1=%.4f QWK=%.4f bAcc=%.4f KL4-F1=%.4f | "
             "params=%.2fM size=%.1fMB lat=%.2fms(%s) best_epoch=%d time=%.0fs",
             args.model, result["macro_f1"], result["quadratic_weighted_kappa"],
             result["balanced_accuracy"], result["kl4_f1"],
             result["parameters"] / 1e6, result["model_size_mb"] or 0,
             result["mean_latency_ms"], result["latency_device"],
             result["best_epoch"], result["training_time_sec"])
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("confusion_matrix", "artefacts", "adjacent_confusions",
                                   "gradcam_layer_candidates")}, indent=2, default=str))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
