r"""PHASE 6.1 - export per-model logits + softmax probs on VAL and TEST.

Evaluation only. Loads clean Phase-4 checkpoints (models/phase4/<m>/best.pt),
forwards the val and test splits (optionally with deterministic TTA), and writes
    reports/phase6/probs/<split>__<model>__<tta>.npz   (y_true, y_prob, y_logit, sample_id, knee_side)

These arrays feed phase6_ensemble_calibrate.py, which fits everything (ensemble
weights, temperature) on VAL and applies it ONCE to TEST.

    .\.venv\Scripts\python.exe scripts\phase6_export_predictions.py --dry-run
    .\.venv\Scripts\python.exe scripts\phase6_export_predictions.py --models vgg16,convnext_tiny,densenet121,inception_v3,vgg19 --tta none
    .\.venv\Scripts\python.exe scripts\phase6_export_predictions.py --models vgg16,convnext_tiny,densenet121,inception_v3,vgg19 --tta hflip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("p6export")

ALL15 = ["vgg16", "vgg19", "googlenet", "squeezenet", "shufflenet", "convnext_tiny",
         "densenet121", "inception_v3", "xception", "efficientnet_b1", "resnet50",
         "mobilenet_v3_large", "resnet18", "mobilenet_v2", "efficientnet_b0"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark_phase4.yaml"))
    ap.add_argument("--models", default="vgg16,convnext_tiny,vgg19,densenet121,inception_v3")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tta", default="none", choices=["none", "hflip", "hflip_rot"])
    ap.add_argument("--phase4-dir", type=Path, default=Path("models/phase4"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase6/probs"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-file", type=Path, default=Path("reports/phase6/run_p6_export.log"))
    args = ap.parse_args()

    add_file_logger(args.log_file)
    cfg = load_config(PROJECT_ROOT / args.config)
    models = ALL15 if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL15]
    if bad:
        LOG.error("unknown models %s", bad); sys.exit(2)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import collect_env, pick_device
    from src.benchmark.checks import leakage_and_distribution_check, LeakageError
    from src.datasets import build_dataloaders
    from src.benchmark.phase6_infer import load_phase4_model, predict_logits_probs
    from src.benchmark.metrics import compute_all

    set_seed(cfg["seed"], deterministic=True)
    device = pick_device(args.device)
    if args.device != "cpu" and device.type != "cuda":
        LOG.error("CUDA expected but unavailable - aborting (no silent CPU)."); sys.exit(4)
    env = collect_env()
    LOG.info("PHASE 6 export | models=%s | splits=%s | tta=%s | device=%s",
             models, splits, args.tta, device)

    try:
        dist = leakage_and_distribution_check(cfg["preprocessing_variant"], logger=LOG)
    except LeakageError as exc:
        LOG.error("DATA INTEGRITY FAILED:\n%s", exc); sys.exit(1)

    missing = [m for m in models if not (args.phase4_dir / m / "best.pt").is_file()]
    if missing:
        LOG.error("missing Phase-4 checkpoints: %s", missing); sys.exit(5)

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pm_bs = cfg.get("per_model_batch_size", {}) or {}
    summary = []
    t0 = time.perf_counter()

    for mi, m in enumerate(models, 1):
        bs = args.batch_size or pm_bs.get(m) or cfg["train"]["batch_size"]
        built, ck = load_phase4_model(m, cfg, device, args.phase4_dir)
        LOG.info("[%d/%d] %s loaded (backend=%s, ckpt val_%s=%.4f) bs=%d",
                 mi, len(models), m, built.backend, ck.get("selection_metric"),
                 ck.get("selection_value") or float("nan"), bs)
        for split in splits:
            loaders = build_dataloaders(
                variant=cfg["preprocessing_variant"], normalization=cfg["data"]["normalization"],
                batch_size=bs, num_workers=0, out_channels=cfg["in_channels"],
                imbalance="none", augmentation_yaml=cfg["data"]["augmentation_yaml"],
                return_meta=True, pin_memory=False, persistent_workers=False, seed=cfg["seed"])
            loader = loaders[split]
            r = predict_logits_probs(built.model, loader, device, tta=args.tta,
                                     amp=bool(cfg["train"].get("amp", True)))
            y_pred = r["y_prob"].argmax(1)
            mm = compute_all(r["y_true"], y_pred, r["y_prob"])
            LOG.info("   %-5s n=%d  acc=%.4f macroF1=%.4f QWK=%.4f",
                     split, r["y_true"].size, mm["accuracy"], mm["macro_f1"],
                     mm["quadratic_weighted_kappa"])
            if args.dry_run:
                summary.append({"model": m, "split": split, "n": int(r["y_true"].size),
                                "macro_f1": mm["macro_f1"]})
                break  # dry-run: one split, don't write
            fp = out_dir / f"{split}__{m}__{args.tta}.npz"
            np.savez_compressed(fp, y_true=r["y_true"], y_prob=r["y_prob"], y_logit=r["y_logit"],
                                sample_id=r["sample_id"] if r["sample_id"] is not None else np.array([]),
                                knee_side=r["knee_side"] if r["knee_side"] is not None else np.array([]),
                                tta=args.tta,
                                model=m, split=split,
                                val_macro_f1_ckpt=ck.get("selection_value"))
            summary.append({"model": m, "split": split, "n": int(r["y_true"].size),
                            "macro_f1": mm["macro_f1"], "qwk": mm["quadratic_weighted_kappa"],
                            "file": str(fp.relative_to(PROJECT_ROOT))})
        del built.model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    (out_dir / f"_export_summary_{args.tta}.json").write_text(json.dumps(
        {"env": env, "tta": args.tta, "splits": splits, "models": models,
         "results": summary}, indent=2, default=str))
    LOG.info("done in %.1f min | wrote %d npz to %s", (time.perf_counter() - t0) / 60,
             sum(1 for s in summary if "file" in s), out_dir)
    if args.dry_run:
        LOG.info("DRY RUN ok - no npz written.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
