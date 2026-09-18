"""PHASE 3 - PART C/D/G : controlled optimization experiments.

One dimension changed per run vs the Phase-3 baseline (configs/benchmark_phase3.yaml).
Every run appends ONE row to reports/phase3/experiment_results.csv (append-only).

    # after the 15-model benchmark, pick the top models and run:
    python scripts/run_experiments.py --experiment preprocessing --models convnext_tiny,densenet121,inception_v3
    python scripts/run_experiments.py --experiment schedule      --models convnext_tiny,densenet121
    python scripts/run_experiments.py --experiment finetune      --models convnext_tiny,densenet121
    python scripts/run_experiments.py --experiment loss          --models convnext_tiny,densenet121
    python scripts/run_experiments.py --experiment attention     # resnet18/convnext_tiny spatial-attention
    python scripts/run_experiments.py --experiment multiseed     --models convnext_tiny --variant clahe --loss focal
    python scripts/run_experiments.py --experiment all           --models convnext_tiny,densenet121

    python scripts/run_experiments.py --experiment preprocessing --models convnext_tiny --dry-run

Test split is never loaded. Selection metric = validation Macro-F1.
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
from src.benchmark import experiments as EXP  # noqa: E402

LOG = get_logger("exp")

PER_FAMILY_CSV = {
    "preprocessing": "preprocessing_ablation.csv",
    "schedule": "training_schedule_ablation.csv",
    "finetune": "finetune_ablation.csv",
    "loss": "loss_ablation.csv",
    "attention": "attention_ablation.csv",
    "multiseed": "multi_seed_results.csv",
}


def _build_specs(family, models, args):
    if family == "preprocessing":
        return EXP.preprocessing_specs(models)
    if family == "schedule":
        return EXP.schedule_specs(models)
    if family == "finetune":
        return EXP.finetune_specs(models)
    if family == "loss":
        return EXP.loss_specs(models)
    if family == "attention":
        m = models if models else None
        return EXP.attention_specs(m) if m else EXP.attention_specs()
    if family == "multiseed":
        base = {}
        if args.variant:
            base["preprocessing_variant"] = args.variant
        if args.loss:
            base["loss"] = {"name": args.loss}
        return EXP.multiseed_specs(models, base_overrides=base)
    raise SystemExit(f"unknown experiment family {family!r}")


def _write_family_views(master_csv: Path, out_dir: Path):
    if not master_csv.exists():
        return
    df = pd.read_csv(master_csv)
    for fam, fname in PER_FAMILY_CSV.items():
        sub = df[df["family"] == fam]
        if len(sub):
            sub.to_csv(out_dir / fname, index=False)


def _make_plots(master_csv: Path, out_dir: Path):
    from src.benchmark.plots import comparison_bar, multiseed_bar
    if not master_csv.exists():
        return
    df = pd.read_csv(master_csv)
    plots = out_dir
    if (df["family"] == "preprocessing").any():
        comparison_bar(df[df.family == "preprocessing"], "preprocessing",
                       plots / "preprocessing_comparison.png",
                       "Experiment 1 - preprocessing (basic vs CLAHE vs histeq)")
    if (df["family"] == "schedule").any():
        d = df[df.family == "schedule"].copy()
        d["schedule"] = "e" + d["max_epochs"].astype(str) + "p" + d["patience"].astype(str)
        comparison_bar(d, "schedule", plots / "epoch_comparison.png",
                       "Experiment 2 - training schedule (30/5 vs 50/8)")
    if (df["family"] == "loss").any():
        comparison_bar(df[df.family == "loss"], "loss", plots / "loss_comparison.png",
                       "Experiment 4 - loss function (WCE vs focal vs class-balanced)")
    if (df["family"] == "finetune").any():
        d = df[df.family == "finetune"].copy()
        d["ft"] = d["experiment_id"].str.split("__ft__").str[-1]
        comparison_bar(d, "ft", plots / "finetune_comparison.png",
                       "Experiment 3 - fine-tuning strategy (A/B/C)")
    if (df["family"] == "multiseed").any():
        multiseed_bar(df[df.family == "multiseed"], plots / "multiseed_comparison.png")
    if (df["family"] == "attention").any():
        # attention vs plain baseline (pull plain from Phase-2 extended results)
        ext = PROJECT_ROOT / "reports/phase3/extended_benchmark_results.csv"
        rows = df[df.family == "attention"][["model", "val_macro_f1", "val_qwk", "val_balanced_accuracy"]].copy()
        if ext.exists():
            e = pd.read_csv(ext)
            for base in ("resnet18", "convnext_tiny"):
                r = e[e.model == base]
                if len(r):
                    rows = pd.concat([rows, pd.DataFrame([{
                        "model": base, "val_macro_f1": r.iloc[0]["macro_f1"],
                        "val_qwk": r.iloc[0]["quadratic_weighted_kappa"],
                        "val_balanced_accuracy": r.iloc[0]["balanced_accuracy"]}])], ignore_index=True)
        rows["variant"] = rows["model"].apply(lambda m: "attention" if "attention" in m else "plain")
        rows["base"] = rows["model"].str.replace("_spatial_attention", "", regex=False)
        comparison_bar(rows.rename(columns={"base": "model"}), "variant",
                       plots / "attention_comparison.png",
                       "Part D - spatial attention vs plain baseline")


def _multiseed_summary(master_csv: Path, out_dir: Path):
    if not master_csv.exists():
        return
    df = pd.read_csv(master_csv)
    d = df[df.family == "multiseed"]
    if not len(d):
        return
    g = d.groupby("model").agg(
        n_seeds=("seed", "count"),
        macro_f1_mean=("val_macro_f1", "mean"), macro_f1_std=("val_macro_f1", lambda s: s.std(ddof=0)),
        qwk_mean=("val_qwk", "mean"), qwk_std=("val_qwk", lambda s: s.std(ddof=0)),
        bacc_mean=("val_balanced_accuracy", "mean"),
        bacc_std=("val_balanced_accuracy", lambda s: s.std(ddof=0)),
    ).reset_index()
    g.to_csv(out_dir / "multi_seed_summary.csv", index=False)
    LOG.info("multi-seed summary:\n%s", g.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True,
                    choices=["preprocessing", "schedule", "finetune", "loss", "attention",
                             "multiseed", "all"])
    ap.add_argument("--models", default="", help="comma list of benchmark model names (top-N)")
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark_phase3.yaml"))
    ap.add_argument("--variant", default=None, help="multiseed: fix preprocessing variant")
    ap.add_argument("--loss", default=None, help="multiseed: fix loss name")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (quick test)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run experiment ids already in the master CSV")
    ap.add_argument("--fast-dev", action="store_true")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="also write the full run log here (default reports/phase3/run_exp_<family>.log)")
    args = ap.parse_args()

    lf = args.log_file or Path(f"reports/phase3/run_exp_{args.experiment}.log")
    add_file_logger(lf)
    LOG.info("logging to %s", lf)

    cfg0 = load_config(args.config)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    families = list(EXP.FAMILIES) if args.experiment == "all" else [args.experiment]
    if args.experiment != "attention" and not models and "attention" not in families:
        raise SystemExit("--models is required (except for --experiment attention)")

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import collect_env, pick_device
    from src.benchmark.checks import leakage_and_distribution_check
    from src.benchmark.runner import run_one_model, dry_run_model

    device = pick_device(args.device)
    reports_dir = PROJECT_ROOT / cfg0["paths"]["reports_dir"]
    exp_models_dir = PROJECT_ROOT / cfg0["paths"]["models_dir"] / "experiments"
    reports_dir.mkdir(parents=True, exist_ok=True)
    master_csv = reports_dir / "experiment_results.csv"

    set_seed(cfg0["seed"], deterministic=True)
    env = collect_env()
    LOG.info("PHASE 3 EXPERIMENTS | families=%s | models=%s | device=%s | dry_run=%s",
             families, models or "(attention default)", device, args.dry_run)
    dist = leakage_and_distribution_check(cfg0["preprocessing_variant"], logger=LOG)

    all_specs = []
    for fam in families:
        all_specs += _build_specs(fam, models, args)
    LOG.info("%d experiment runs queued", len(all_specs))

    if args.dry_run:
        seen = set()
        for sp in all_specs:
            m = sp["model"]
            if m in seen:
                continue
            seen.add(m)
            try:
                dry_run_model(m, EXP.deep_merge(cfg0, sp["overrides"]), device, logger=LOG)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[dry-run] %s FAILED: %s", m, exc); traceback.print_exc()
        LOG.info("DRY RUN done (%d distinct models). No experiment rows written.", len(seen))
        return

    fast = dict(max_train_batches=6, max_val_batches=4) if args.fast_dev else {}
    t0 = time.perf_counter()
    for i, sp in enumerate(all_specs, 1):
        eid = sp["experiment_id"]
        cfg = EXP.deep_merge(cfg0, sp["overrides"])
        if args.epochs:
            cfg["train"]["max_epochs"] = args.epochs
        out_sub = f"experiments/{eid}"
        rj = exp_models_dir / eid / "result.json"
        if rj.exists() and not args.force:
            LOG.info("[%d/%d] %s : result.json exists -> reuse (append if new to master)", i, len(all_specs), eid)
            result = json.loads(rj.read_text())
        else:
            LOG.info("[%d/%d] ==== %s ====  model=%s  overrides=%s  seed=%s",
                     i, len(all_specs), eid, sp["model"], sp["overrides"], cfg.get("seed"))
            # re-seed per run so (a) ablation runs are order-independent and
            # (b) --experiment multiseed actually varies the RNG (incl. head init),
            # not just the dataloader shuffle order.
            set_seed(int(cfg.get("seed", 42)), deterministic=True)
            try:
                bs = (cfg.get("per_model_batch_size", {}) or {}).get(sp["model"]) or cfg["train"]["batch_size"]
                result = run_one_model(
                    sp["model"], cfg, device,
                    reports_dir=reports_dir / "experiments",
                    models_dir=PROJECT_ROOT / cfg["paths"]["models_dir"],
                    out_subdir=out_sub, batch_size=bs,
                    env={"torch": env["torch"], "gpu": env["gpu"]["name"] if env["gpu"] else None},
                    extra_config={"experiment_id": eid, "family": sp["family"]},
                    logger=LOG, **fast)
            except Exception as exc:  # noqa: BLE001
                LOG.error("EXPERIMENT %s FAILED: %s", eid, exc); traceback.print_exc()
                continue
        row = EXP.result_to_experiment_row(sp, cfg, result)
        wrote = EXP.append_row(master_csv, row, force=args.force)
        LOG.info("   %s macro_f1=%.4f qwk=%.4f bAcc=%.4f best_ep=%s  -> master %s",
                 eid, row["val_macro_f1"] or 0, row["val_qwk"] or 0,
                 row["val_balanced_accuracy"] or 0, row["best_epoch"],
                 "APPENDED" if wrote else "already present")

    LOG.info("experiments done in %.1f min", (time.perf_counter() - t0) / 60)
    _write_family_views(master_csv, reports_dir)
    _make_plots(master_csv, reports_dir)
    _multiseed_summary(master_csv, reports_dir)
    LOG.info("master table: %s", master_csv)
    LOG.info("per-family CSVs + comparison plots written under %s", reports_dir)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
