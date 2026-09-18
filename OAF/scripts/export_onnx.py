r"""Export a clean Phase-4 checkpoint to ONNX for the Raspberry-Pi dashboard.

  * build the arch exactly as Phase 4 (build_model pretrained=True) + load best.pt
  * torch.onnx.export  ->  models/deploy/<model>.onnx        (fp32, dynamic batch)
  * onnxruntime weight-only int8 quantization  ->  <model>.int8.onnx   (no calib data)
  * fit temperature T on VALIDATION logits ; pick an abstain threshold on VALIDATION
    for a target selective accuracy
  * parity check: ONNX vs PyTorch on the first N processed TEST images
  * write models/deploy/<model>_deploy.json  (T, threshold, test metrics, meta)

    .\.venv\Scripts\python.exe scripts\export_onnx.py --model mobilenet_v2
    .\.venv\Scripts\python.exe scripts\export_onnx.py --model convnext_tiny --model mobilenet_v2 --model vgg16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402
from src.preprocessing.pipeline import load_config  # noqa: E402

LOG = get_logger("onnx")
CLASS_LABELS = [0, 1, 2, 3, 4]


def _softmax_T(logits, T):
    z = np.asarray(logits, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _fit_temperature(y, logits):
    from scipy.optimize import minimize_scalar
    def nll(logT):
        p = _softmax_T(logits, np.exp(logT))
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
    return float(np.exp(minimize_scalar(nll, bounds=(-2.5, 2.5), method="bounded").x))


def _pick_threshold(y, p, target_acc):
    conf = p.max(1); pred = p.argmax(1)
    for t in np.round(np.arange(0.30, 0.981, 0.01), 2):
        keep = conf >= t
        if keep.sum() >= 50 and (pred[keep] == y[keep]).mean() >= target_acc:
            return float(t), float(keep.mean()), float((pred[keep] == y[keep]).mean())
    return 0.0, 1.0, float((pred == y).mean())


def _torch_val_test_logits(built, cfg, device):
    import torch
    from src.datasets import build_dataloaders
    bs = (cfg.get("per_model_batch_size", {}) or {}).get(built.name) or cfg["train"]["batch_size"]
    out = {}
    for split in ("val", "test"):
        dl = build_dataloaders(variant=cfg["preprocessing_variant"],
                               normalization=cfg["data"]["normalization"], batch_size=bs,
                               num_workers=0, out_channels=cfg["in_channels"], imbalance="none",
                               augmentation_yaml=cfg["data"]["augmentation_yaml"],
                               return_meta=False, pin_memory=False, persistent_workers=False,
                               seed=cfg["seed"])[split]
        ys, zs = [], []
        built.model.eval()
        with torch.inference_mode():
            for x, y in dl:
                o = built.model(x.to(device))
                o = o[0] if isinstance(o, (tuple, list)) else o
                zs.append(o.float().cpu().numpy()); ys.append(y.numpy())
        out[split] = (np.concatenate(ys).astype(int), np.concatenate(zs).astype(np.float64))
    return out


def export_one(model: str, cfg: dict, device, out_dir: Path, *, opset: int, n_parity: int,
               target_acc: float, quantize: bool, checkpoint: str | None = None,
               out_name: str | None = None) -> dict:
    import torch
    from src.benchmark.models import build_model, count_parameters
    from src.benchmark.metrics import compute_all
    from src.benchmark.phase5_metrics import ordinal_and_confidence
    from src.deploy.preprocess import preprocess

    name = out_name or model
    ck_path = Path(checkpoint) if checkpoint else PROJECT_ROOT / "models" / "phase4" / model / "best.pt"
    if not ck_path.is_absolute():
        ck_path = PROJECT_ROOT / ck_path
    if not ck_path.is_file():
        raise FileNotFoundError(ck_path)
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    if ck.get("ordinal") == "corn":
        raise NotImplementedError(
            f"{ck_path} is a CORN (4-logit ordinal) checkpoint; export_onnx currently "
            "handles softmax heads only. Re-run the winning config without CORN, or extend "
            "export_one() with the corn_logits_to_probs decode before exporting.")
    built = build_model(model, num_classes=cfg["num_classes"], in_chans=cfg["in_channels"],
                        pretrained=True)
    built.model.load_state_dict(ck["state_dict"], strict=True)
    built.model.to(device).eval()
    pc = count_parameters(built.model)
    LOG.info("[%s] loaded Phase-4 best.pt (epoch %s, val_%s=%.4f) | %.2fM params | backend=%s",
             model, ck.get("epoch"), ck.get("selection_metric"),
             ck.get("selection_value") or float("nan"), pc["parameters"] / 1e6, built.backend)

    # ---- temperature + threshold from VAL, test metrics for the card ----
    lg = _torch_val_test_logits(built, cfg, device)
    yv, zv = lg["val"]; yt, zt = lg["test"]
    T = _fit_temperature(yv, zv)
    thr, cov, sel_acc = _pick_threshold(yv, _softmax_T(zv, T), target_acc)
    pt = _softmax_T(zt, T)
    m = compute_all(yt, pt.argmax(1), pt)
    o = ordinal_and_confidence(yt, pt.argmax(1), pt)
    LOG.info("[%s] temperature T=%.3f | abstain thr=%.2f (val coverage %.1f%% @ %.1f%% acc) | "
             "test macroF1=%.4f QWK=%.4f", model, T, thr, cov * 100, sel_acc * 100,
             m["macro_f1"], m["quadratic_weighted_kappa"])

    # ---- ONNX export (fp32, dynamic batch) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp = out_dir / f"{name}.onnx"
    dummy = torch.randn(1, cfg["in_channels"], 224, 224, device="cpu")
    built.model.to("cpu").eval()                       # export on CPU -> clean, portable graph
    # dynamo=False -> legacy TorchScript exporter: single self-contained .onnx (weights inlined),
    # tighter fp32 numerics, no onnxscript/external-data file to ship to the Pi.
    torch.onnx.export(
        built.model, dummy, str(onnx_fp), input_names=["input"], output_names=["logits"],
        opset_version=opset, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True, dynamo=False)
    import onnx
    m_onnx = onnx.load(str(onnx_fp), load_external_data=True)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in m_onnx.graph.initializer):
        onnx.save(m_onnx, str(onnx_fp), save_as_external_data=False)   # consolidate to one file
        for extra in out_dir.glob(f"{name}.onnx.data*"):
            extra.unlink()
    onnx.checker.check_model(str(onnx_fp))
    built.model.to(device)
    LOG.info("[%s] wrote %s (%.2f MB, single file)", model, onnx_fp, onnx_fp.stat().st_size / 1e6)

    int8_fp = None
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8_fp = out_dir / f"{name}.int8.onnx"
            quantize_dynamic(str(onnx_fp), str(int8_fp), weight_type=QuantType.QInt8)
            LOG.info("[%s] wrote %s (%.2f MB, int8 weights)", model, int8_fp,
                     int8_fp.stat().st_size / 1e6)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[%s] int8 quantization failed: %s", model, exc)
            int8_fp = None

    # ---- parity: ONNX(fp32) vs PyTorch on first N processed TEST pngs ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_fp), providers=["CPUExecutionProvider"])
    man = __import__("pandas").read_csv(PROJECT_ROOT / "metadata" / "processed_manifest_basic.csv")
    tst = man[man["split"] == "test"].head(n_parity)
    built_cpu = built.model.to("cpu").eval()
    max_abs = 0.0
    agree = 0
    for _, r in tst.iterrows():
        x = preprocess(str(PROJECT_ROOT / r["processed_path"]), already_224=True)
        z_onnx = sess.run(["logits"], {"input": x})[0][0]
        with torch.inference_mode():
            z_pt = built_cpu(torch.from_numpy(x))
            z_pt = (z_pt[0] if isinstance(z_pt, (tuple, list)) else z_pt).float().numpy()[0]
        max_abs = max(max_abs, float(np.abs(z_onnx - z_pt).max()))
        agree += int(np.argmax(z_onnx) == np.argmax(z_pt))
    built.model.to(device)
    LOG.info("[%s] ONNX<->PyTorch(CPU) parity on %d test imgs: max|dlogit|=%.4g, argmax agree %d/%d",
             model, len(tst), max_abs, agree, len(tst))
    parity_ok = max_abs < 5e-3 and agree == len(tst)

    # ---- int8 vs fp32 test agreement (informational) ----
    int8_agree = None
    if int8_fp:
        s8 = ort.InferenceSession(str(int8_fp), providers=["CPUExecutionProvider"])
        a = 0
        for _, r in tst.iterrows():
            x = preprocess(str(PROJECT_ROOT / r["processed_path"]), already_224=True)
            a += int(np.argmax(s8.run(["logits"], {"input": x})[0][0]) ==
                     np.argmax(sess.run(["logits"], {"input": x})[0][0]))
        int8_agree = f"{a}/{len(tst)}"
        LOG.info("[%s] int8 vs fp32 argmax agreement on %d test imgs: %s", model, len(tst), int8_agree)

    deploy = {
        "model": name, "arch": model, "backbone": ck.get("timm_name"), "backend": built.backend,
        "onnx_fp32": onnx_fp.name, "onnx_int8": int8_fp.name if int8_fp else None,
        "opset": opset, "input": "1x3x224x224 float32 (grayscale->3ch replicate, ImageNet norm)",
        "temperature": round(T, 4),
        "abstain_threshold": round(thr, 3),
        "abstain_note": f"confidence < {thr:.2f} => REFER (val: {cov*100:.0f}% coverage @ {sel_acc*100:.0f}% selective accuracy)",
        "parameters": pc["parameters"],
        "parameters_millions": round(pc["parameters"] / 1e6, 3),
        "checkpoint": str(ck_path.relative_to(PROJECT_ROOT).as_posix()),
        "checkpoint_val_macro_f1": ck.get("selection_value"),
        "onnx_pytorch_parity_max_abs_logit_diff": round(max_abs, 6),
        "onnx_pytorch_argmax_agreement": f"{agree}/{len(tst)}",
        "int8_vs_fp32_argmax_agreement": int8_agree,
        "test_metrics": {
            "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
            "weighted_f1": round(m["weighted_f1"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "qwk": round(m["quadratic_weighted_kappa"], 4),
            "kl4_f1": round(m["kl4_f1"], 4), "mae": round(o["mae"], 4),
            "within_1_accuracy": round(o["within_1_accuracy"], 4),
            "per_class_f1": {f"KL{i}": round(m[f"kl{i}_f1"], 4) for i in range(5)},
        },
        "class_names": {str(i): n for i, n in {
            0: "KL0 Normal", 1: "KL1 Doubtful", 2: "KL2 Minimal OA",
            3: "KL3 Moderate OA", 4: "KL4 Severe OA"}.items()},
    }
    (out_dir / f"{name}_deploy.json").write_text(json.dumps(deploy, indent=2))
    deploy["_parity_ok"] = parity_ok
    return deploy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="arch name for build_model (repeat for several)")
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="explicit best.pt path per --model (default models/phase4/<model>/best.pt)")
    ap.add_argument("--name", action="append", default=[],
                    help="output basename per --model (default = arch name)")
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark_phase4.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("models/deploy"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--parity-images", type=int, default=64)
    ap.add_argument("--target-selective-acc", type=float, default=0.90)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    from src.benchmark.seeding import set_seed
    from src.benchmark.hardware import pick_device
    cfg = load_config(PROJECT_ROOT / args.config)
    set_seed(cfg["seed"], deterministic=True)
    device = pick_device(args.device)
    out_dir = PROJECT_ROOT / args.out_dir
    LOG.info("export -> %s | models=%s | device=%s | opset=%d", out_dir, args.model, device, args.opset)

    ckpts = args.checkpoint + [None] * (len(args.model) - len(args.checkpoint))
    names = args.name + [None] * (len(args.model) - len(args.name))
    summary = []
    for m, ck, nm in zip(args.model, ckpts, names):
        try:
            d = export_one(m, cfg, device, out_dir, opset=args.opset,
                           n_parity=args.parity_images, target_acc=args.target_selective_acc,
                           quantize=not args.no_quantize, checkpoint=ck, out_name=nm)
            summary.append(d)
        except Exception as exc:  # noqa: BLE001
            LOG.error("[%s] EXPORT FAILED: %s", m, exc)
            import traceback; traceback.print_exc()
            summary.append({"model": m, "error": str(exc)})

    (out_dir / "_export_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("=" * 70)
    for d in summary:
        if "error" in d:
            LOG.info("  %-16s FAILED: %s", d["model"], d["error"]); continue
        LOG.info("  %-16s fp32 ok | int8 %s | parity %s | T=%.2f thr=%.2f | test macroF1=%.4f QWK=%.4f",
                 d["model"], "ok" if d["onnx_int8"] else "-",
                 "OK" if d["_parity_ok"] else "!!CHECK!!", d["temperature"],
                 d["abstain_threshold"], d["test_metrics"]["macro_f1"], d["test_metrics"]["qwk"])
    LOG.info("=" * 70)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
