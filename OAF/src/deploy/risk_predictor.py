"""DemographicRiskPredictor - ONNX Runtime inference for the OSTRIVA demographic
OA-risk model (see risk-model/README.md at the repo root for training + scope).

Torch-free, sklearn-free at inference (onnxruntime + numpy only) -> runs on a
Raspberry Pi, matching KneeOAPredictor's deployment shape.

Loads:  models/risk/demographic_risk.onnx
        models/risk/demographic_risk_deploy.json  (feature order/meaning, metrics, scope warning)

SCOPE: demographic-only (age, sex, BMI - the fields registration already
collects; race/smoking/osteoporosis were tested and dropped, see
risk-model/reports/feature_ablation.json). Does not see WOMAC, KL grade, or
any functional/sensor feature - it is one input to the overall risk picture,
not a replacement for the existing clinical/functional heuristics in the
frontend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DemographicRiskPredictor:
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

        meta_path = Path(self.onnx_path).with_name("demographic_risk_deploy.json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.model_name = self.meta.get("model", Path(self.onnx_path).stem)
        self.feature_order = self.meta.get("feature_order", ["AGE", "BMI", "SEX"])

        out_names = [o.name for o in self.sess.get_outputs()]
        # to_onnx(..., options={id(pipe): {"zipmap": False}}) yields [label, probabilities]
        self.proba_out = out_names[1] if len(out_names) > 1 else out_names[0]

    def predict(self, *, age: float, bmi: float, sex: int) -> dict:
        feeds = {
            "AGE": np.array([[age]], dtype=np.float32),
            "BMI": np.array([[bmi]], dtype=np.float32),
            "SEX": np.array([[sex]], dtype=np.int64),
        }
        out = self.sess.run([self.proba_out], feeds)[0]
        p_oa = float(out[0][1])

        band = "high" if p_oa >= 0.65 else "moderate" if p_oa >= 0.4 else "low"
        return {
            "p_oa": p_oa,
            "band": band,
            "model": self.model_name,
            "scope_warning": self.meta.get("scope_warning"),
            "test_metrics": self.meta.get("test_metrics", {}),
        }
