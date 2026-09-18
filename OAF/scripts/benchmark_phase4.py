"""PHASE 4 - 15-model EXTENDED TRAINING BUDGET benchmark (max_epochs 30 -> 50).

Controlled experiment: the ONLY intended change vs Phase 3 is the maximum epoch
budget (30 -> 50). Early stopping stays enabled (patience 5 on val Macro-F1).
All 15 models are retrained from scratch (the 10 Phase-2 checkpoints were trained
at the 30-epoch budget and cannot be reused for a 50-epoch comparison).

Outputs are fully isolated in models/phase4/ and reports/phase4/. Phase 3 files
are never read-modified or overwritten - reports/phase3/... is opened read-only
for the comparison table.

    python scripts/benchmark_phase4.py --dry-run          # build all 15, few batches, no training
    python scripts/benchmark_phase4.py                     # full run (all 15 @ up to 50 epochs)
    python scripts/benchmark_phase4.py --models vgg16,convnext_tiny
    python scripts/benchmark_phase4.py --skip-existing     # resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, add_file_logger, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("phase4")

CMP_METRICS = [
    ("macro_f1", "macro_f1"),
    ("quadratic_weighted_kappa", "qwk"),
    ("balanced_accuracy", "balanced_accuracy"),
    ("kl4_f1", "kl4_f1"),
    ("mean_latency_ms", "latency"),
]


def _resolve(arg: str, cfg: dict) -> list[str]:
    zoo = list(cfg["models"].keys())
    if arg in ("all", "", None):
        return zoo
    picked = [m.strip() for m in arg.split(",") if m.strip()]
    bad = [m for m in picked if m not in zoo]
    if bad:
        raise SystemExit(f"unknown models: {bad}\navailable: {zoo}")
    return picked


def _early_stop_reason(r: dict, max_epochs: int, patience: int) -> str:
    if r.get("error"):
        return "FAILED"
    if r.get("stopped_early"):
        return (f"early stop: no val_macro_f1 improvement for {patience} "
                f"epochs (patience exhausted)")
    if int(r.get("epochs_run", 0)) >= int(max_epochs):
        return f"reached max_epochs budget ({max_epochs})"
    return "completed"


def _build_comparison(p4_rows: list[dict], phase3_csv: Path, phase3_label: str,
                      out_csv: Path, max_epochs: int, patience: int) -> pd.DataFrame:
    if not phase3_csv.exists():
        LOG.warning("Phase-3 CSV not found at %s - skipping comparison table", phase3_csv)
        return pd.DataFrame()
    p3 = pd.read_csv(phase3_csv).set_index("model")
    rows = []
    for r in p4_rows:
        m = r["model"]
        row = {"model": m,
               "best_epoch": r.get("best_epoch"),
               "epochs_completed": r.get("epochs_run"),
               "max_epochs": max_epochs,
               "early_stop": bool(r.get("stopped_early")),
               "early_stop_reason": _early_stop_reason(r, max_epochs, patience),
               "phase4_status": "FAILED" if r.get("error") else "OK"}
        for p3col, short in CMP_METRICS:
            v4 = r.get(p3col)
            v3 = float(p3.loc[m, p3col]) if (m in p3.index and pd.notna(p3.loc[m].get(p3col))) else None
            row[f"{phase3_label}_{short}"] = v3
            row[f"phase4_{short}"] = v4
            row[f"delta_{short}"] = (round(v4 - v3, 6) if (v4 is not None and v3 is not None) else None)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("phase4_macro_f1", ascending=False, na_position="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def _fmt(v, nd=4):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else (
        f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _md_rank(df: pd.DataFrame, by: str, cols: list[str], asc: bool = False) -> str:
    d = df.dropna(subset=[by]).sort_values(by, ascending=asc)
    head = "| rank | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    body = "\n".join(f"| {i} | " + " | ".join(_fmt(row[c]) for c in cols) + " |"
                     for i, (_, row) in enumerate(d.iterrows(), 1))
    return "\n".join([head, sep, body])


def _write_phase4_report(p4_df: pd.DataFrame, cmp_df: pd.DataFrame, env: dict, dist: dict,
                         cfg: dict, runtime_min: float, failures: list[str], out_path: Path,
                         phase3_label: str) -> None:
    max_ep = cfg["train"]["max_epochs"]
    pat = cfg["train"]["early_stopping"]["patience"]
    A = []
    P = A.append
    ok = p4_df[p4_df.get("phase4_status", "OK") != "FAILED"] if "phase4_status" in p4_df else p4_df

    # ---- helpers on comparison frame ----
    has_cmp = len(cmp_df) > 0
    improved = degraded = flat = []
    if has_cmp:
        d = cmp_df.dropna(subset=["delta_macro_f1"])
        improved = d[d.delta_macro_f1 > 0.002]["model"].tolist()
        degraded = d[d.delta_macro_f1 < -0.002]["model"].tolist()
        flat = d[(d.delta_macro_f1 >= -0.002) & (d.delta_macro_f1 <= 0.002)]["model"].tolist()
        used_budget = cmp_df[cmp_df.epochs_completed > 30]["model"].tolist()
        stopped_early = cmp_df[cmp_df.early_stop == True]["model"].tolist()  # noqa: E712
        best_gain = d.sort_values("delta_macro_f1", ascending=False).head(1)

    P(f"# AETHER-OA - X-ray Module | PHASE 4: 15-Model Extended-Budget Benchmark (max_epochs = {max_ep})\n")
    P(f"_Generated {datetime.now(timezone.utc).isoformat()}. Seed {cfg['seed']}. "
      f"Controlled experiment vs Phase 3 (max_epochs = 30). Test set NEVER used._\n")

    P("## 1. Executive summary\n")
    if has_cmp:
        mean_df1 = cmp_df["delta_macro_f1"].dropna().mean()
        med_df1 = cmp_df["delta_macro_f1"].dropna().median()
        P(f"- Retrained all **15 models** identically to Phase 3 except **max_epochs 30 -> {max_ep}**; "
          f"early stopping (patience {pat} on val Macro-F1) left enabled.\n"
          f"- Macro-F1 delta (Phase4 - {phase3_label}): mean **{mean_df1:+.4f}**, median **{med_df1:+.4f}** "
          f"across {cmp_df['delta_macro_f1'].notna().sum()} comparable models.\n"
          f"- Improved (> +0.002 Macro-F1): **{len(improved)}** {improved}\n"
          f"- Practically unchanged (+/-0.002): **{len(flat)}** {flat}\n"
          f"- Degraded (< -0.002): **{len(degraded)}** {degraded}\n"
          f"- Models that used > 30 epochs: **{len(used_budget)}** {used_budget}\n"
          f"- Best validation Macro-F1 (Phase 4): **{ok.iloc[0]['model']}** = {ok.iloc[0]['macro_f1']:.4f}\n")
        if len(best_gain):
            bg = best_gain.iloc[0]
            P(f"- Largest Macro-F1 gain from the extra budget: **{bg['model']}** "
              f"({bg[f'{phase3_label}_macro_f1']:.4f} -> {bg['phase4_macro_f1']:.4f}, "
              f"{bg['delta_macro_f1']:+.4f}).\n")
    if failures:
        P(f"- **FAILED models: {failures}** (see section 6; no metrics fabricated).\n")

    P("## 2. Experimental objective\n")
    P("Answer, with all other variables held constant: *does raising the maximum training "
      f"budget from 30 to {max_ep} epochs improve the 15-model benchmark?* "
      "30 epochs = baseline (Phase 3); 50 epochs = experimental condition (Phase 4). "
      "No architecture / loss / augmentation / class-weight / hyper-parameter changes.\n")

    P("## 3. Hardware / software environment\n")
    g = env.get("gpu") or {}
    P(f"- GPU: {g.get('name')} ({g.get('total_memory_gb')} GB, capability {g.get('capability')})\n"
      f"- PyTorch {env.get('torch')} | CUDA {env.get('cuda_version')} | cuDNN {env.get('cudnn_version')} "
      f"| torchvision {env.get('torchvision')} | timm {env.get('timm')}\n"
      f"- OS {env.get('os')} | Python {env.get('python')} | device = {'CUDA' if g else 'CPU'}\n"
      f"- cuDNN deterministic={env.get('cudnn_deterministic')}, benchmark={env.get('cudnn_benchmark')}; AMP on.\n")

    P("## 4. Dataset integrity verification\n")
    P("Repeated the Phase-3 leakage + class-distribution checks (all PASSED before training):\n")
    P("| split | n | KL0 | KL1 | KL2 | KL3 | KL4 |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    for s in ("train", "val", "test"):
        cd = dist["class_distribution"][s]
        P(f"| {s} | {sum(cd.values())} | " + " | ".join(str(cd.get(str(k), cd.get(k, 0))) for k in range(5)) + " |")
    P("\npatient overlap train/val = 0, train/test = 0, val/test = 0 ; sample overlap = 0 ; all 5 classes present.\n")

    P("## 5. Training configuration (identical to Phase 3 except max_epochs)\n")
    tl = cfg["transfer_learning"]
    P(f"- preprocessing variant `{cfg['preprocessing_variant']}`, {cfg['image_size'][0]}x{cfg['image_size'][1]}, "
      f"3-ch replicated grayscale, {cfg['data']['normalization']} normalization\n"
      f"- augmentation `{Path(cfg['data']['augmentation_yaml']).name}` (train only); val/test deterministic\n"
      f"- loss `{cfg['loss']['name']}` with TRAIN-ONLY class weights (`{cfg['loss']['class_weights_key']}`)\n"
      f"- optimizer {cfg['optimizer']['name']} lr(StageB)={cfg['optimizer']['lr']} wd={cfg['optimizer']['weight_decay']}; "
      f"cosine schedule, warmup {cfg['scheduler']['warmup_epochs']} ep\n"
      f"- transfer learning: Stage A {tl['stage_a_epochs']} ep head-only @ {tl['stage_a_lr']}, "
      f"then Stage B full fine-tune @ {tl['stage_b_lr']}\n"
      f"- batch size {cfg['train']['batch_size']} (vgg16/vgg19: "
      f"{cfg.get('per_model_batch_size', {}).get('vgg16')} - VRAM necessity, same as Phase 3), "
      f"num_workers {cfg['data']['num_workers']}, AMP {cfg['train']['amp']}\n"
      f"- **max_epochs {max_ep}** (Phase 3 = 30), early stopping monitor `{cfg['train']['early_stopping']['monitor']}` "
      f"mode {cfg['train']['early_stopping']['mode']} patience {pat} min_delta {cfg['train']['early_stopping']['min_delta']}\n"
      f"- selection metric `{cfg['eval']['selection_metric']}`, seed {cfg['seed']}\n")

    P("## 6. 15-model results (Phase 4, validation)\n")
    cols = ["model", "macro_f1", "quadratic_weighted_kappa", "balanced_accuracy", "weighted_f1",
            "cohen_kappa", "macro_precision", "macro_recall",
            "kl0_f1", "kl1_f1", "kl2_f1", "kl3_f1", "kl4_f1",
            "parameters", "model_size_mb", "gmacs", "mean_latency_ms", "cpu_mean_latency_ms",
            "peak_memory_mb", "best_epoch", "epochs_run", "training_time_sec"]
    cols = [c for c in cols if c in p4_df.columns]
    P(_md_rank(p4_df, "macro_f1", cols))
    if failures:
        P(f"\n**FAILED (no metrics):** {failures}\n")

    P(f"\n## 7. Phase 3 vs Phase 4 comparison (delta = Phase4 - {phase3_label})\n")
    if has_cmp:
        ccols = ["model",
                 f"{phase3_label}_macro_f1", "phase4_macro_f1", "delta_macro_f1",
                 f"{phase3_label}_qwk", "phase4_qwk", "delta_qwk",
                 f"{phase3_label}_balanced_accuracy", "phase4_balanced_accuracy", "delta_balanced_accuracy",
                 f"{phase3_label}_kl4_f1", "phase4_kl4_f1", "delta_kl4_f1",
                 f"{phase3_label}_latency", "phase4_latency",
                 "best_epoch", "epochs_completed", "early_stop"]
        ccols = [c for c in ccols if c in cmp_df.columns]
        P(_md_rank(cmp_df, "phase4_macro_f1", ccols))
        P(f"\nFull table: `reports/phase4/phase3_vs_phase4_comparison.csv`\n")
    else:
        P("_Phase-3 CSV unavailable - comparison skipped._\n")

    P("## 8. Ranking by Macro-F1\n"); P(_md_rank(p4_df, "macro_f1",
        ["model", "macro_f1", "quadratic_weighted_kappa", "balanced_accuracy", "kl4_f1", "parameters"]))
    P("\n## 9. Ranking by Quadratic Weighted Kappa\n"); P(_md_rank(p4_df, "quadratic_weighted_kappa",
        ["model", "quadratic_weighted_kappa", "macro_f1", "balanced_accuracy", "kl4_f1"]))
    P("\n## 10. Ranking by Balanced Accuracy\n"); P(_md_rank(p4_df, "balanced_accuracy",
        ["model", "balanced_accuracy", "macro_f1", "quadratic_weighted_kappa", "kl4_f1"]))
    P("\n## 11. Ranking by KL4 F1\n"); P(_md_rank(p4_df, "kl4_f1",
        ["model", "kl4_f1", "macro_f1", "quadratic_weighted_kappa", "balanced_accuracy"]))

    P("\n## 12. Training-time comparison\n")
    if has_cmp and "training_time_sec" in p4_df.columns:
        tt = p4_df[["model", "training_time_sec", "epochs_run", "best_epoch"]].copy()
        tt["sec_per_epoch"] = (tt.training_time_sec / tt.epochs_run.replace(0, pd.NA)).round(1)
        P(_md_rank(tt, "training_time_sec", ["model", "training_time_sec", "epochs_run",
                                             "best_epoch", "sec_per_epoch"], asc=False))
        P(f"\nTotal Phase-4 wall time: **{runtime_min:.1f} min**.\n")

    P("## 13. Parameter / latency analysis\n")
    if {"parameters", "mean_latency_ms", "macro_f1"}.issubset(p4_df.columns):
        pa = p4_df[["model", "parameters", "gmacs", "model_size_mb", "mean_latency_ms",
                    "cpu_mean_latency_ms", "macro_f1"]].copy()
        pa["params_M"] = (pa.parameters / 1e6).round(2)
        pa["f1_per_Mparam"] = (pa.macro_f1 / pa.params_M).round(4)
        P(_md_rank(pa, "macro_f1", ["model", "params_M", "gmacs", "model_size_mb",
                                    "mean_latency_ms", "cpu_mean_latency_ms", "macro_f1",
                                    "f1_per_Mparam"]))
        light = pa.sort_values("params_M").head(5)["model"].tolist()
        P(f"\nLightest 5: {light}. Pareto plots: `reports/phase4/pareto_macroF1_vs_latency.png`, "
          f"`reports/phase4/pareto_macroF1_vs_size.png`.\n")

    P("## 14. Early-stopping analysis\n")
    if has_cmp:
        es = cmp_df[["model", "best_epoch", "epochs_completed", "early_stop", "early_stop_reason"]]
        P(_md_rank(es, "epochs_completed", ["model", "best_epoch", "epochs_completed",
                                            "early_stop", "early_stop_reason"], asc=False))
        P(f"\n- Stopped early (patience {pat}): {stopped_early}\n"
          f"- Ran past 30 epochs (i.e. genuinely used the extra budget): {used_budget}\n"
          f"- Best epoch <= 30 for {int((cmp_df.best_epoch <= 30).sum())}/{len(cmp_df)} models "
          f"=> the extra 20-epoch budget was mostly unused for those.\n")

    P("## 15. Overfitting observations\n")
    P("Signals inspected per model (from `models/phase4/<m>/history.json`): (a) val Macro-F1 peaks "
      "then declines while train loss keeps dropping; (b) best_epoch far below epochs_completed with a "
      "long no-improvement tail; (c) Phase4 Macro-F1 below Phase3 despite more epochs.\n")
    if has_cmp:
        od = cmp_df[(cmp_df.delta_macro_f1 < -0.002) & (cmp_df.epochs_completed > 30)]["model"].tolist()
        P(f"- Models that trained longer AND scored lower than Phase 3 (overfitting-consistent): "
          f"{od if od else 'none'}\n"
          f"- Models whose best_epoch <= 30 but kept training to satisfy patience: "
          f"{cmp_df[(cmp_df.best_epoch <= 30) & (cmp_df.epochs_completed > cmp_df.best_epoch + 1)]['model'].tolist()}\n")

    P("## 16. Best-performing models (Phase 4, top 5 by Macro-F1)\n")
    top5 = p4_df.sort_values("macro_f1", ascending=False).head(5)
    P(_md_rank(top5, "macro_f1", ["model", "macro_f1", "quadratic_weighted_kappa",
                                  "balanced_accuracy", "kl4_f1", "parameters", "mean_latency_ms"]))

    P("\n## 17. Most efficient models\n")
    if "parameters" in p4_df.columns:
        eff = p4_df.assign(params_M=(p4_df.parameters / 1e6).round(2)).sort_values("params_M").head(6)
        P(_md_rank(eff, "macro_f1", ["model", "params_M", "mean_latency_ms", "cpu_mean_latency_ms",
                                     "macro_f1", "quadratic_weighted_kappa"]))

    P("\n## 18. Models benefiting from 50 epochs\n")
    if has_cmp:
        ben = cmp_df[cmp_df.delta_macro_f1 > 0.002].sort_values("delta_macro_f1", ascending=False)
        P(_md_rank(ben, "delta_macro_f1", ["model", f"{phase3_label}_macro_f1", "phase4_macro_f1",
                                           "delta_macro_f1", "delta_qwk", "delta_balanced_accuracy",
                                           "best_epoch", "epochs_completed"]) if len(ben)
          else "_None improved by more than +0.002 Macro-F1._")

    P("\n## 19. Models NOT benefiting from 50 epochs\n")
    if has_cmp:
        nob = cmp_df[cmp_df.delta_macro_f1 <= 0.002].sort_values("delta_macro_f1")
        P(_md_rank(nob, "delta_macro_f1", ["model", f"{phase3_label}_macro_f1", "phase4_macro_f1",
                                           "delta_macro_f1", "best_epoch", "epochs_completed",
                                           "early_stop"], asc=True) if len(nob)
          else "_All models improved._")

    P("\n## 20. Final recommendation for the next phase\n")
    P("_See the terminal summary for the evidence-based read on whether 50 epochs was worth it. "
      "No production model is selected in this phase - Phase 4 only isolates the epoch-budget effect. "
      "Recommended next experiments are listed in the run summary (preprocessing ablation on the top "
      "models, and/or fine-tuning-strategy ablation), keeping the test set frozen._\n")

    out_path.write_text("\n".join(str(x) for x in A), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark_phase4.yaml"))
    ap.add_argument("--models", default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fast-dev", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs (debug only)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--log-file", type=Path, default=Path("reports/phase4/run_phase4.log"))
    args = ap.parse_args()

    # ---- single-instance lock: a concurrent Phase-4 run would race on the output
    # dirs and contaminate timing/latency. Refuse the second invocation. ----
    lock = PROJECT_ROOT / "reports" / "phase4" / ".phase4.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        try:
            fd = __import__("os").open(str(lock), __import__("os").O_CREAT | __import__("os").O_EXCL | __import__("os").O_WRONLY)
            __import__("os").write(fd, str(__import__("os").getpid()).encode())
            __import__("os").close(fd)
        except FileExistsError:
            sys.stderr.write(f"ERROR: another Phase-4 run holds {lock} (pid {lock.read_text().strip()}). "
                             f"Refusing to start a second concurrent run. Delete the lock if stale.\n")
            sys.exit(3)
        import atexit
        atexit.register(lambda: lock.exists() and lock.unlink())

    add_file_logger(args.log_file)
    LOG.info("logging to %s | pid=%d", args.log_file, __import__("os").getpid())

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["max_epochs"] = args.epochs
    models = _resolve(args.models, cfg)

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import collect_env, pick_device
    from src.benchmark.checks import leakage_and_distribution_check, LeakageError
    from src.benchmark.runner import run_one_model, dry_run_model

    seed_report = set_seed(cfg["seed"], deterministic=True)
    device = pick_device(args.device)
    if args.device != "cpu" and device.type != "cuda":
        LOG.error("CUDA requested/expected but not available - REFUSING to silently run on CPU. Aborting.")
        sys.exit(2)
    env = collect_env(seed_report)

    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    max_ep = cfg["train"]["max_epochs"]
    pat = cfg["train"]["early_stopping"]["patience"]
    LOG.info("=" * 78)
    LOG.info("PHASE 4 EXTENDED-BUDGET BENCHMARK | models=%d | max_epochs=%d | patience=%d | device=%s",
             len(models), max_ep, pat, device)
    LOG.info("torch=%s cuda=%s timm=%s torchvision=%s gpu=%s vram=%.1fGB",
             env["torch"], env["cuda_version"], env.get("timm"), env.get("torchvision"),
             env["gpu"]["name"] if env["gpu"] else "none",
             env["gpu"]["total_memory_gb"] if env["gpu"] else 0.0)
    LOG.info("output isolation: models_dir=%s reports_dir=%s (Phase 3 dirs untouched)",
             cfg["paths"]["models_dir"], cfg["paths"]["reports_dir"])
    LOG.info("=" * 78)

    LOG.info("DATA INTEGRITY: leakage + class-distribution check (same as Phase 3)")
    try:
        dist = leakage_and_distribution_check(cfg["preprocessing_variant"], logger=LOG)
    except LeakageError as exc:
        LOG.error("DATA INTEGRITY CHECK FAILED - STOPPING before training:\n%s", exc)
        sys.exit(1)

    (reports_dir / "reproducibility.json").write_text(json.dumps(
        {**env, "benchmark_config": cfg, "class_distribution": dist["class_distribution"],
         "patients_per_split": dist["patients"], "models_requested": models,
         "phase": "phase4", "max_epochs": max_ep}, indent=2, default=str))

    if args.dry_run:
        LOG.info("--- DRY RUN (build all 15, few batches, NO training) ---")
        res = []
        for m in models:
            try:
                res.append(dry_run_model(m, cfg, device, logger=LOG))
            except Exception as exc:  # noqa: BLE001
                LOG.error("[dry-run] %s FAILED: %s", m, exc); traceback.print_exc()
                res.append({"model": m, "ok": False, "issues": [str(exc)]})
        (reports_dir / "dry_run_report.json").write_text(json.dumps(res, indent=2, default=str))
        ok = all(r.get("ok") for r in res)
        for r in res:
            LOG.info("  %-20s %s %s", r["model"], "OK" if r.get("ok") else "FAIL", r.get("issues") or "")
        LOG.info("DRY RUN %s", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)

    # ---------------- FULL RUN ----------------
    pmbs = cfg.get("per_model_batch_size", {}) or {}
    fast = dict(max_train_batches=6, max_val_batches=4) if args.fast_dev else {}
    results: list[dict] = []
    failures: list[str] = []
    t0 = time.perf_counter()
    for i, m in enumerate(models, 1):
        rj = models_dir / m / "result.json"
        if args.skip_existing and rj.exists():
            LOG.info("[%d/%d] SKIP %s (result.json exists)", i, len(models), m)
            results.append(json.loads(rj.read_text()))
            continue
        bs = pmbs.get(m) or cfg["train"]["batch_size"]
        LOG.info("[%d/%d] [%s] starting  (batch_size=%d, max_epochs=%d)", i, len(models), m, bs, max_ep)
        try:
            r = run_one_model(m, cfg, device, reports_dir=reports_dir, models_dir=models_dir,
                              max_epochs=None, num_workers=args.num_workers, batch_size=bs,
                              env={"torch": env["torch"], "cuda": env["cuda_version"],
                                   "gpu": env["gpu"]["name"] if env["gpu"] else None,
                                   "timm": env.get("timm"), "torchvision": env.get("torchvision")},
                              extra_config={"phase": "phase4", "max_epochs_config": max_ep},
                              logger=LOG, **fast)
            r["phase"] = "phase4"
            results.append(r)
            LOG.info("[%s] final evaluation | macro_f1=%.4f qwk=%.4f bAcc=%.4f kl4_f1=%.4f | "
                     "best_epoch=%d epochs_completed=%d early_stop=%s | reason: %s | train_time=%.1fs",
                     m, r["macro_f1"], r["quadratic_weighted_kappa"], r["balanced_accuracy"],
                     r["kl4_f1"], r["best_epoch"], r["epochs_run"], r["stopped_early"],
                     _early_stop_reason(r, max_ep, pat), r["training_time_sec"])
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            LOG.error("[%s] FAILED (%s): %s", m, "CUDA OOM" if oom else "RuntimeError", exc)
            traceback.print_exc()
            results.append({"model": m, "phase": "phase4", "error": f"{'OOM: ' if oom else ''}{exc}"})
            failures.append(m)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            LOG.error("[%s] FAILED: %s", m, exc)
            traceback.print_exc()
            results.append({"model": m, "phase": "phase4", "error": str(exc)})
            failures.append(m)

    runtime_min = (time.perf_counter() - t0) / 60
    LOG.info("PHASE 4 training complete in %.1f min | ok=%d failed=%d %s",
             runtime_min, len(results) - len(failures), len(failures), failures or "")

    ok_results = [r for r in results if "error" not in r and "macro_f1" in r]
    (reports_dir / "phase4_raw_results.json").write_text(json.dumps(results, indent=2, default=str))
    if not ok_results:
        LOG.error("no successful models - cannot build report"); sys.exit(1)

    from src.benchmark.report import write_tables, write_complexity_csv
    from src.benchmark.plots import pareto_plots

    p4_df = write_tables(ok_results, reports_dir, basename="extended_benchmark_results")
    write_complexity_csv([{"model": r["model"], "timm_name": r.get("timm_name"),
                           "parameters": r.get("parameters"),
                           "trainable_parameters": r.get("trainable_parameters"),
                           "non_trainable_parameters": r.get("non_trainable_parameters"),
                           "parameters_millions": round((r.get("parameters") or 0) / 1e6, 3),
                           "gmacs": r.get("gmacs"), "flops_estimate_g": r.get("flops_estimate_g"),
                           "model_size_mb": r.get("model_size_mb"),
                           "gmacs_method": r.get("gmacs_method")} for r in ok_results], reports_dir)
    pareto_plots(ok_results, reports_dir)

    cmp_df = _build_comparison(ok_results, PROJECT_ROOT / cfg["compare_to"],
                               cfg.get("compare_phase_label", "phase3"),
                               reports_dir / "phase3_vs_phase4_comparison.csv", max_ep, pat)

    if not args.no_report:
        _write_phase4_report(p4_df, cmp_df, env, dist, cfg, runtime_min, failures,
                             reports_dir / "final_phase4_report.md",
                             cfg.get("compare_phase_label", "phase3"))
        LOG.info("wrote %s", reports_dir / "final_phase4_report.md")

    # ---- terminal summary ----
    LOG.info("=" * 78)
    LOG.info("PHASE 4 RANKING (validation Macro-F1):")
    for i, (_, r) in enumerate(p4_df.iterrows(), 1):
        LOG.info("  %2d. %-20s F1=%.4f QWK=%.4f bAcc=%.4f KL4=%.4f best_ep=%d/%d ep_done=%d%s",
                 i, r["model"], r["macro_f1"], r["quadratic_weighted_kappa"], r["balanced_accuracy"],
                 r["kl4_f1"], int(r["best_epoch"]), max_ep, int(r["epochs_run"]),
                 "  EARLY-STOP" if r.get("stopped_early") else "")
    if len(cmp_df):
        LOG.info("-" * 78)
        LOG.info("PHASE 3 -> PHASE 4 Macro-F1 delta:")
        for _, r in cmp_df.sort_values("delta_macro_f1", ascending=False, na_position="last").iterrows():
            d = r["delta_macro_f1"]
            LOG.info("  %-20s %s%.4f -> %.4f  (%+.4f)  best_ep=%s ep_done=%s%s",
                     r["model"],
                     "" if pd.notna(r[f"{cfg.get('compare_phase_label','phase3')}_macro_f1"]) else "?",
                     r[f"{cfg.get('compare_phase_label','phase3')}_macro_f1"] or 0,
                     r["phase4_macro_f1"] or 0, d if pd.notna(d) else 0,
                     r["best_epoch"], r["epochs_completed"], "  EARLY" if r["early_stop"] else "")
        dd = cmp_df["delta_macro_f1"].dropna()
        LOG.info("-" * 78)
        LOG.info("delta Macro-F1: mean %+.4f | median %+.4f | improved(>+0.002) %d | flat %d | degraded(<-0.002) %d",
                 dd.mean(), dd.median(), int((dd > 0.002).sum()),
                 int(((dd >= -0.002) & (dd <= 0.002)).sum()), int((dd < -0.002).sum()))
        beneficial = dd.mean() > 0.002 and (dd > 0.002).sum() > (dd < -0.002).sum()
        LOG.info("VERDICT: 50 epochs %s a meaningful benefit over 30 across the 15-model benchmark.",
                 "PROVIDES" if beneficial else "does NOT provide")
    LOG.info("Phase 3 dirs untouched: models/phase3/ and reports/phase3/ unchanged.")
    LOG.info("=" * 78)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
