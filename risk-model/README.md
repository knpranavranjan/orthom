# OSTRIVA demographic risk model

First real (trained, not hand-tuned) model in the OSTRIVA clinical-risk
pipeline. It predicts P(knee OA) from six demographic fields — age, sex, BMI,
race, smoking status, baseline osteoporosis — trained on 2,392 OAI
participants (see `data/raw/README.md` for provenance).

## What this is not

This is **not** the multimodal clinical+functional risk model the OSTRIVA
design calls for. It does not see WOMAC, symptom duration, injury history,
knee ROM, gait, sit-to-stand, or pressure data — none of that exists as a
labeled training set yet (the kiosk that collects it is still being piloted).
Held-out ROC-AUC is 0.65: real signal, well short of what WOMAC + imaging
would add. Treat its output as one more piece of evidence alongside the
existing heuristic scores in `oa copy/src/services/clinicalScore.ts` and
`computeRisk()`, not a replacement for them.

Retrain against real WOMAC/functional data the moment a labeled pilot dataset
exists — the training script, preprocessing, and ONNX export path here are
built to be re-run, not one-off.

## Layout

    data/raw/          the OAI-derived source CSV + provenance note
    data/processed/     cleaned CSV (target-missing rows dropped)
    scripts/train_demographic_risk.py   compares 3 classifiers, exports ONNX
    models/             demographic_risk.onnx + demographic_risk_deploy.json
    reports/            cross-val + held-out test metrics

## Retrain

    pip install scikit-learn skl2onnx onnxruntime pandas numpy
    python scripts/train_demographic_risk.py

Exports the *entire fitted pipeline* (imputation + scaling/one-hot +
classifier) to ONNX, so serving needs only `onnxruntime` — no
scikit-learn/pandas at inference, matching how AETHER-OA's X-ray model is
deployed (`OAF/scripts/serve_api.py`). The script asserts the ONNX output
matches sklearn's `predict_proba` before saving, so a bad conversion fails
loudly instead of shipping silently.

## Serving

`models/demographic_risk.onnx` + `demographic_risk_deploy.json` are copied
into `OAF/models/risk/` and served by `OAF/src/deploy/risk_predictor.py` at
`POST /api/risk/demographic` — see that project for the running API.
