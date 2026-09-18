"""Train the OSTRIVA demographic OA-risk model and export it to ONNX.

Data: data/raw/oai_matchthem_demographic.csv (see data/raw/README.md for
provenance). This is a DEMOGRAPHIC-ONLY proxy model. It does NOT see WOMAC,
KL grade, or any functional/sensor feature: those evidence streams stay on
the existing heuristic scoring (`oa copy/src/services/clinicalScore.ts`,
`computeRisk()` in `inference.ts`) until a dataset that actually carries them
is available.

Features are AGE, BMI, SEX only — deliberately dropping the raw dataset's
RAC (race), SMK (smoking) and OSP (baseline osteoporosis) columns. They were
tested (see reports/feature_ablation.json) and added only +0.007 CV ROC-AUC
over age+bmi+sex alone (0.649 vs 0.642), which does not justify adding three
new patient-facing intake questions (with Hindi translation) to the kiosk
just to feed this proxy model — age, sex and BMI are the only fields the
registration screen already collects.

Compares three candidate classifiers inside one preprocessing pipeline
(median imputation + one-hot for sex) via stratified cross-validation, picks
the best by ROC-AUC, refits on the full training split, evaluates once on a
held-out test split, then exports the *whole fitted pipeline* (preprocessing
included) to ONNX with skl2onnx so the serving side only needs onnxruntime —
no scikit-learn/pandas at inference, matching how the X-ray model is
deployed (OAF/scripts/serve_api.py).

    python scripts/train_demographic_risk.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, roc_auc_score,
                              precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "oai_matchthem_demographic.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

NUMERIC = ["AGE", "BMI"]
CATEGORICAL = ["SEX"]
TARGET = "KOA"
SEED = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "demographic_risk_clean.csv", index=False)
    print(f"loaded {before} rows, {len(df)} after dropping rows with missing target")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=10, random_state=SEED),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED),
}


def main() -> None:
    df = load_data()
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}
        print(f"{name:>24s}: CV ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
    print(f"\nbest by CV ROC-AUC: {best_name}")

    best_pipe = Pipeline([("prep", build_preprocessor()), ("clf", CANDIDATES[best_name])])
    best_pipe.fit(X_train, y_train)

    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }
    print("held-out test metrics:", json.dumps(test_metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import onnx
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType

    initial_types = [
        ("AGE", FloatTensorType([None, 1])),
        ("BMI", FloatTensorType([None, 1])),
        ("SEX", Int64TensorType([None, 1])),
    ]
    onnx_model = to_onnx(
        best_pipe, initial_types=initial_types,
        options={id(best_pipe): {"zipmap": False}},
        target_opset=17,
    )
    onnx_path = MODELS_DIR / "demographic_risk.onnx"
    onnx.save(onnx_model, str(onnx_path))
    print(f"saved ONNX model -> {onnx_path}")

    # sanity-check the exported graph reproduces sklearn's probabilities
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = X_test.iloc[:5]
    feeds = {
        "AGE": sample[["AGE"]].to_numpy(dtype=np.float32),
        "BMI": sample[["BMI"]].fillna(sample["BMI"].median()).to_numpy(dtype=np.float32),
        "SEX": sample[["SEX"]].to_numpy(dtype=np.int64),
    }
    onnx_out = sess.run(None, feeds)
    onnx_proba = onnx_out[1][:, 1]
    sklearn_proba = best_pipe.predict_proba(sample)[:, 1]
    max_diff = float(np.max(np.abs(onnx_proba - sklearn_proba)))
    print(f"onnx vs sklearn max prob diff on 5 samples: {max_diff:.6f}")
    if max_diff > 1e-4:
        raise SystemExit("ONNX export does not match sklearn predictions closely enough")

    deploy_json = {
        "model": "demographic_risk_v1",
        "algorithm": best_name,
        "feature_order": ["AGE", "BMI", "SEX"],
        "feature_meaning": {
            "AGE": "years",
            "BMI": "kg/m^2",
            "SEX": "opaque 2-level code from source data (1 or 2)",
        },
        "output": "P(knee OA diagnosed at follow-up), class index 1 of a 2-class softmax",
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": SEED,
        "data_provenance": "data/raw/README.md — OAI cohort via MatchThem CRAN package, demographic-only, no WOMAC/KL",
        "dropped_features": {
            "RAC": "race", "SMK": "smoking status", "OSP": "baseline osteoporosis",
            "reason": "tested via reports/feature_ablation.json — added only +0.007 CV "
                      "ROC-AUC over age+bmi+sex; not worth 3 new kiosk intake questions",
        },
        "scope_warning": (
            "Demographic-only proxy model (age, sex, BMI). Does not see WOMAC, KL "
            "grade, or any functional/sensor feature. Not validated for clinical "
            "use — screening support signal only, to be superseded once real "
            "WOMAC+functional training data exists."
        ),
    }
    (MODELS_DIR / "demographic_risk_deploy.json").write_text(
        json.dumps(deploy_json, indent=2), encoding="utf-8")
    (REPORTS_DIR / "demographic_risk_report.json").write_text(
        json.dumps({"cv_results": cv_results, "test_metrics": test_metrics,
                    "chosen_model": best_name}, indent=2), encoding="utf-8")
    print(f"saved deploy metadata -> {MODELS_DIR / 'demographic_risk_deploy.json'}")


if __name__ == "__main__":
    main()
