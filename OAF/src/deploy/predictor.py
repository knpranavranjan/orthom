"""KneeOAPredictor - ONNX Runtime inference for the AETHER-OA X-ray dashboard.

Torch-free at inference (onnxruntime + numpy + OpenCV) -> runs on a Raspberry Pi.
Loads:  models/deploy/<model>.onnx   (or *.int8.onnx)
        models/deploy/<model>_deploy.json  (temperature, abstain threshold, test metrics, meta)

Outputs per image:
    predicted KL grade (0-4) + name, 5 class probabilities, confidence (max prob),
    calibrated probabilities (temperature-scaled), ordinal expected grade,
    binary OA screen  (P_OA = P2+P3+P4),
    3-class  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4),
    abstain flag  (confidence < deploy.json threshold  ->  "refer to clinician"),
    inference latency (ms).
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

from .preprocess import preprocess, _to_gray_uint8, letterbox_square, resize_224

KL_NAMES = {0: "KL0 - Normal", 1: "KL1 - Doubtful", 2: "KL2 - Minimal OA",
            3: "KL3 - Moderate OA", 4: "KL4 - Severe OA"}
G3_NAMES = ["Normal (KL0)", "Early (KL1-2)", "Advanced (KL3-4)"]
DISCLAIMER = ("AI-assisted assessment - not a diagnosis. Kellgren-Lawrence grade is a "
              "radiographic scale; clinical correlation is required.")

# templated radiographic findings per KL grade (generic wording; not patient-specific)
_FINDINGS = {
    0: ["No definite osteophytes", "Joint space preserved", "Normal subchondral bone"],
    1: ["Doubtful joint-space narrowing", "Possible early osteophytic lipping",
        "Findings are equivocal"],
    2: ["Definite osteophytes present", "Possible joint-space narrowing",
        "Minimal / early osteoarthritis"],
    3: ["Moderate joint-space narrowing", "Multiple osteophytes",
        "Some subchondral sclerosis"],
    4: ["Marked joint-space narrowing", "Large osteophytes",
        "Severe sclerosis; possible bony deformity"],
}
_RECO = {
    0: "No radiographic OA. Clinical correlation if symptomatic.",
    1: "Equivocal findings. Clinical correlation; consider follow-up imaging if symptoms progress.",
    2: "Early OA. Clinical correlation, symptom-based management, routine follow-up.",
    3: "Established OA. Orthopaedic review; manage symptoms and function.",
    4: "Advanced OA. Orthopaedic referral; assess joint-preserving vs arthroplasty options.",
}


def _softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / float(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class KneeOAPredictor:
    def __init__(self, onnx_path, *, providers=None, intra_threads: int | None = None):
        import onnxruntime as ort
        self.onnx_path = str(onnx_path)
        so = ort.SessionOptions()
        if intra_threads:
            so.intra_op_num_threads = int(intra_threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            self.onnx_path, sess_options=so,
            providers=providers or ["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        meta_path = Path(self.onnx_path).with_name(
            Path(self.onnx_path).name.split(".onnx")[0].replace(".int8", "") + "_deploy.json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.abstain_threshold = float(self.meta.get("abstain_threshold", 0.0))
        self.model_name = self.meta.get("model", Path(self.onnx_path).stem)

        # optional grad-free CAM model (logits + 1280x7x7 features) + classifier weights
        self._so = so
        self._providers = providers or ["CPUExecutionProvider"]
        stem = Path(self.onnx_path).name.split(".onnx")[0].replace(".int8", "")
        self._cam_onnx = Path(self.onnx_path).with_name(stem + "_cam.onnx")
        self._cam_npz = Path(self.onnx_path).with_name(stem + "_cam.npz")
        self.has_cam = self._cam_onnx.exists() and self._cam_npz.exists()
        self._cam_sess = None
        self._cam_W = None

    def _ensure_cam(self):
        import onnxruntime as ort
        if self._cam_sess is None:
            self._cam_sess = ort.InferenceSession(str(self._cam_onnx), sess_options=self._so,
                                                  providers=self._providers)
            d = np.load(self._cam_npz)
            self._cam_W = d["W"].astype(np.float32)               # (5, 1280)

    # ------------------------------------------------------------------ #
    def _forward_logits(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out_name], {self.in_name: x})[0]

    def predict(self, image, *, already_224: bool = False, explain: bool = False,
                return_raw: bool = False) -> dict:
        t0 = time.perf_counter()
        x = preprocess(image, already_224=already_224)
        # the CAM graph also yields the 1x1280x7x7 feature map, which the input
        # validator needs — take that path when explaining OR when raw features
        # are requested and the CAM model is present.
        want_cam = (explain or return_raw) and self.has_cam
        if want_cam:
            self._ensure_cam()
            logits, feat = self._cam_sess.run(["logits", "features"], {self.in_name: x})
            logits = logits[0]
        else:
            logits = self._forward_logits(x)[0]                   # (5,)
            feat = None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        p_raw = _softmax(logits, T=1.0)
        p_cal = _softmax(logits, T=self.temperature)
        grade = int(np.argmax(p_cal))
        conf = float(p_cal[grade])
        exp_grade = float(np.dot(np.arange(5), p_cal))            # ordinal expectation

        p_oa = float(p_cal[2] + p_cal[3] + p_cal[4])              # KL>=2 => radiographic OA
        g3 = np.array([p_cal[0], p_cal[1] + p_cal[2], p_cal[3] + p_cal[4]])
        g3_idx = int(np.argmax(g3))

        abstain = conf < self.abstain_threshold
        out = {
            "model": self.model_name,
            "kl_grade": grade,
            "kl_name": KL_NAMES[grade],
            "confidence": round(conf, 4),
            "probabilities": {f"KL{i}": round(float(p_cal[i]), 4) for i in range(5)},
            "probabilities_uncalibrated": {f"KL{i}": round(float(p_raw[i]), 4) for i in range(5)},
            "expected_grade": round(exp_grade, 3),
            "oa_screen": {
                "label": "OA likely" if p_oa >= 0.5 else "No / doubtful OA",
                "p_oa": round(p_oa, 4),
                "p_no_oa": round(1 - p_oa, 4),
            },
            "three_class": {"label": G3_NAMES[g3_idx],
                            "probs": {G3_NAMES[i]: round(float(g3[i]), 4) for i in range(3)}},
            "recommendation": ("REFER - low confidence, clinician review recommended"
                               if abstain else _RECO[grade]),
            "abstain": bool(abstain),
            "abstain_threshold": self.abstain_threshold,
            "latency_ms": round(latency_ms, 2),
            "disclaimer": DISCLAIMER,
            "backbone_test_metrics": self.meta.get("test_metrics", {}),
        }

        if explain:
            out["explain"] = self._explain(image, already_224, grade, feat, conf)

        if return_raw:
            out["logits"] = [float(v) for v in np.asarray(logits).reshape(-1)]
            out["feat_gap"] = (
                [float(v) for v in np.asarray(feat)[0].mean(axis=(1, 2)).reshape(-1)]
                if feat is not None else None
            )
        return out

    # ------------------------------------------------------------------ #
    def _explain(self, image, already_224, grade, feat, conf) -> dict:
        from . import cam as C
        block = {
            "available": bool(self.has_cam),
            "method": "Class Activation Mapping (grad-free; exact for a GAP+Linear head)",
            "findings": list(_FINDINGS[grade]),
        }
        if feat is None:
            block["note"] = "CAM model not found next to the deployed ONNX; heatmap unavailable."
            block["interpretation"] = _FINDINGS[grade][0] + "."
            return block
        gray = _to_gray_uint8(image)
        gray = gray if already_224 and gray.shape[:2] == (224, 224) else resize_224(letterbox_square(gray))
        cmap = C.class_activation_map(feat, self._cam_W, grade)
        overlay = C.render_overlay(gray, cmap)
        fr = C.focus_region(cmap)
        block.update({
            "heatmap_png_b64": base64.b64encode(C.png_bytes(overlay)).decode("ascii"),
            "focus_region": fr,
            "attention_note": (f"Model attention is {fr['region']}, concentrated on {fr['side_hint']} "
                               f"({int(fr['coverage']*100)}% of the field above half-intensity)."),
            "interpretation": (f"Predicted {KL_NAMES[grade]} at {conf*100:.0f}% confidence. "
                               + "; ".join(_FINDINGS[grade]) + ". "
                               + f"Salient region: {fr['region']} ({fr['side_hint']})."),
        })
        return block

    def benchmark(self, n: int = 100, warmup: int = 10) -> dict:
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        for _ in range(warmup):
            self._forward_logits(x)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            self._forward_logits(x)
            ts.append((time.perf_counter() - t) * 1000.0)
        ts.sort()
        return {"onnx": self.onnx_path, "n": n,
                "mean_ms": round(sum(ts) / n, 3), "median_ms": round(ts[n // 2], 3),
                "p90_ms": round(ts[int(0.9 * n)], 3), "min_ms": round(ts[0], 3),
                "max_ms": round(ts[-1], 3), "throughput_ips": round(1000.0 / (sum(ts) / n), 1),
                "model_size_mb": round(Path(self.onnx_path).stat().st_size / 1e6, 2)}
