r"""Export the frozen MobileNetV2 with a SECOND output = the 1280-channel feature
map (input to global-average-pool), so a grad-free Class Activation Map can be
computed at inference by ONNX Runtime alone.

Why grad-free is exact here: MobileNetV2's head is  features -> GAP -> Linear(1280,5)
with nothing between the pool and the linear layer. Grad-CAM's channel weight
alpha_k^c therefore collapses to  W[c,k] / (H*W)  (a constant), so

    CAM_c(x,y)  proportional to  ReLU( sum_k  W[c,k] * feat_k(x,y) )

which is the original Class Activation Mapping (Zhou et al. 2016) and needs no
autograd. This script emits:

    models/deploy/mobilenet_v2_oa_cam.onnx     outputs: logits (1x5), features (1x1280x7x7)
    models/deploy/mobilenet_v2_oa_cam.npz      W (5x1280), b (5,)   classifier weights

    .\.venv\Scripts\python.exe scripts\export_onnx_cam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, get_logger  # noqa: E402

LOG = get_logger("onnxcam")
CKPT = PROJECT_ROOT / "models" / "mobilenet_final" / "best.pt"
OUT_ONNX = PROJECT_ROOT / "models" / "deploy" / "mobilenet_v2_oa_cam.onnx"
OUT_NPZ = PROJECT_ROOT / "models" / "deploy" / "mobilenet_v2_oa_cam.npz"


def main() -> None:
    import torch
    import torch.nn as nn
    import onnx
    import onnxruntime as ort
    from src.benchmark.models import build_model
    from src.deploy.preprocess import preprocess

    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    built = build_model("mobilenet_v2", num_classes=5, in_chans=3, pretrained=True)
    built.model.load_state_dict(ck["state_dict"], strict=True)
    m = built.model.cpu().eval()
    LOG.info("loaded %s (epoch %s, val_%s=%.4f, ema=%s)", CKPT.name, ck.get("epoch"),
             ck.get("selection_metric"), ck.get("selection_value") or float("nan"), ck.get("ema"))

    W = m.classifier.weight.detach().cpu().numpy().astype(np.float32)   # (5, 1280)
    b = m.classifier.bias.detach().cpu().numpy().astype(np.float32)     # (5,)
    np.savez(OUT_NPZ, W=W, b=b)
    LOG.info("wrote %s  W%s b%s", OUT_NPZ.name, W.shape, b.shape)

    class MNetCAM(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            feat = self.net.forward_features(x)          # (B, 1280, 7, 7) - after conv_head+bn2+ReLU6
            logits = self.net.forward_head(feat)         # (B, 5) - GAP + classifier
            return logits, feat

    wrap = MNetCAM(m).eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        lg_ref, ft_ref = wrap(dummy)
        lg_plain = m(dummy)
    assert torch.allclose(lg_ref, lg_plain, atol=1e-5), "wrapper logits != plain model logits"
    LOG.info("wrapper parity vs plain model: max|dlogit|=%.2e  feat shape=%s",
             float((lg_ref - lg_plain).abs().max()), tuple(ft_ref.shape))

    torch.onnx.export(
        wrap, dummy, str(OUT_ONNX),
        input_names=["input"], output_names=["logits", "features"],
        opset_version=17, do_constant_folding=True, dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}})
    mo = onnx.load(str(OUT_ONNX), load_external_data=True)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in mo.graph.initializer):
        onnx.save(mo, str(OUT_ONNX), save_as_external_data=False)
        for extra in OUT_ONNX.parent.glob(f"{OUT_ONNX.name}.data*"):
            extra.unlink()
    onnx.checker.check_model(str(OUT_ONNX))
    LOG.info("wrote %s (%.2f MB)", OUT_ONNX.name, OUT_ONNX.stat().st_size / 1e6)

    # ---- parity + CAM sanity on real processed test images ----
    sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    base = ort.InferenceSession(str(PROJECT_ROOT / "models" / "deploy" / "mobilenet_v2_oa.onnx"),
                                providers=["CPUExecutionProvider"])
    import pandas as pd
    man = pd.read_csv(PROJECT_ROOT / "metadata" / "processed_manifest_basic.csv")
    tst = man[man["split"] == "test"].sample(24, random_state=0)
    max_dl, agree = 0.0, 0
    cam_on_joint = []
    for _, r in tst.iterrows():
        x = preprocess(str(PROJECT_ROOT / r["processed_path"]), already_224=True).astype(np.float32)
        lg, ft = sess.run(["logits", "features"], {"input": x})
        lg0 = base.run(["logits"], {"input": x})[0]
        max_dl = max(max_dl, float(np.abs(lg - lg0).max()))
        agree += int(lg.argmax() == lg0.argmax())
        c = int(lg.argmax())
        cam = np.maximum(np.einsum("kij,k->ij", ft[0], W[c]), 0.0)       # (7,7)
        if cam.max() > 0:
            cam /= cam.max()
        # crude "is the hot spot near the joint line (vertical centre band)?" check
        rr = np.argmax(cam.sum(1))
        cam_on_joint.append(2 <= rr <= 4)
    LOG.info("CAM-ONNX vs base-ONNX: max|dlogit|=%.2e  argmax agree %d/%d", max_dl, agree, len(tst))
    LOG.info("CAM hot-row in central band on %d/%d sampled test images", sum(cam_on_joint), len(cam_on_joint))
    LOG.info("OK - grad-free CAM export ready.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
