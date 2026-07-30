"""
train_models.py
----------------
Trains all machine learning models used by the app and saves them as .pkl
files inside models/. Also writes evaluation metrics to
reports/model_metrics.json so the Streamlit "Model Performance" page can
display them without retraining at runtime.

Models trained:
    1. Crop Recommendation      -> RandomForest, LightGBM, XGBoost (best kept)
    2. Disease Risk Level       -> RandomForest classifier
    3. Fertilizer Recommendation-> RandomForest classifier
    4. Irrigation Need          -> RandomForest classifier

Run once:  python utils/train_models.py
"""

from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42
metrics_report: dict = {}


def evaluate(name: str, model, X_test, y_test, le: LabelEncoder | None = None) -> dict:
    preds = model.predict(X_test)
    result = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "precision_macro": round(float(precision_score(y_test, preds, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, preds, average="macro", zero_division=0)), 4),
    }
    try:
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            result["roc_auc"] = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
        else:
            result["roc_auc"] = round(float(roc_auc_score(y_test, proba, multi_class="ovr")), 4)
    except Exception:
        result["roc_auc"] = None
    print(f"  [{name}] acc={result['accuracy']}  f1_macro={result['f1_macro']}")
    return result


# ----------------------------------------------------------------------
# 1. CROP RECOMMENDATION MODEL (real Kaggle dataset)
# ----------------------------------------------------------------------
def train_crop_model():
    print("\n=== Training Crop Recommendation Models ===")
    df = pd.read_csv(os.path.join(DATASET_DIR, "Crop_recommendation.csv"))
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y_raw = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    candidates["RandomForest"] = (rf, X_test)

    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=RANDOM_STATE, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    candidates["LightGBM"] = (lgbm, X_test)

    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
    )
    xgb.fit(X_train, y_train)
    candidates["XGBoost"] = (xgb, X_test)

    model_metrics = {}
    best_name, best_model, best_acc = None, None, -1
    for name, (model, Xte) in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        m = evaluate(name, model, Xte, y_test)
        m["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
        m["cv_accuracy_std"] = round(float(cv_scores.std()), 4)
        model_metrics[name] = m
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_name = name
            best_model = model

    cm = confusion_matrix(y_test, best_model.predict(X_test)).tolist()
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(feature_cols, best_model.feature_importances_.round(4).tolist()))

    joblib.dump(best_model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "crop_label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "crop_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "crop_feature_cols.pkl"))

    metrics_report["crop_recommendation"] = {
        "best_model": best_name,
        "models_compared": model_metrics,
        "confusion_matrix": cm,
        "classes": le.classes_.tolist(),
        "feature_importance": feature_importance,
    }
    print(f"  Best model: {best_name} (accuracy={best_acc})")


# ----------------------------------------------------------------------
# 2. DISEASE RISK MODEL (synthetic)
# ----------------------------------------------------------------------
def train_disease_model():
    print("\n=== Training Disease Risk Model ===")
    df = pd.read_csv(os.path.join(DATASET_DIR, "disease_risk.csv"))
    feature_cols = ["temperature", "humidity", "rainfall", "leaf_wetness_hours"]

    crop_le = LabelEncoder()
    df["crop_enc"] = crop_le.fit_transform(df["crop"])
    feature_cols_full = feature_cols + ["crop_enc"]

    risk_le = LabelEncoder()
    y = risk_le.fit_transform(df["risk_level"])
    X = df[feature_cols_full]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    m = evaluate("DiseaseRisk-RF", model, X_test, y_test)
    cm = confusion_matrix(y_test, model.predict(X_test)).tolist()

    joblib.dump(model, os.path.join(MODELS_DIR, "disease_risk_model.pkl"))
    joblib.dump(crop_le, os.path.join(MODELS_DIR, "disease_crop_encoder.pkl"))
    joblib.dump(risk_le, os.path.join(MODELS_DIR, "disease_risk_label_encoder.pkl"))
    joblib.dump(feature_cols_full, os.path.join(MODELS_DIR, "disease_feature_cols.pkl"))

    metrics_report["disease_risk"] = {
        "model": "RandomForest",
        "metrics": m,
        "confusion_matrix": cm,
        "classes": risk_le.classes_.tolist(),
    }


# ----------------------------------------------------------------------
# 3. FERTILIZER RECOMMENDATION MODEL (synthetic)
# ----------------------------------------------------------------------
def train_fertilizer_model():
    print("\n=== Training Fertilizer Recommendation Model ===")
    df = pd.read_csv(os.path.join(DATASET_DIR, "fertilizer_recommendation.csv"))
    feature_cols = ["N", "P", "K", "ph"]

    crop_le = LabelEncoder()
    df["crop_enc"] = crop_le.fit_transform(df["crop"])
    feature_cols_full = feature_cols + ["crop_enc"]

    fert_le = LabelEncoder()
    y = fert_le.fit_transform(df["recommended_fertilizer"])
    X = df[feature_cols_full]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    m = evaluate("Fertilizer-RF", model, X_test, y_test)

    joblib.dump(model, os.path.join(MODELS_DIR, "fertilizer_model.pkl"))
    joblib.dump(crop_le, os.path.join(MODELS_DIR, "fertilizer_crop_encoder.pkl"))
    joblib.dump(fert_le, os.path.join(MODELS_DIR, "fertilizer_label_encoder.pkl"))
    joblib.dump(feature_cols_full, os.path.join(MODELS_DIR, "fertilizer_feature_cols.pkl"))

    metrics_report["fertilizer"] = {"model": "RandomForest", "metrics": m}


# ----------------------------------------------------------------------
# 4. IRRIGATION MODEL (synthetic)
# ----------------------------------------------------------------------
def train_irrigation_model():
    print("\n=== Training Irrigation Need Model ===")
    df = pd.read_csv(os.path.join(DATASET_DIR, "irrigation.csv"))
    feature_cols = ["soil_moisture_pct", "temperature", "rainfall_forecast_mm"]

    crop_le = LabelEncoder()
    df["crop_enc"] = crop_le.fit_transform(df["crop"])
    season_le = LabelEncoder()
    df["season_enc"] = season_le.fit_transform(df["season"])
    feature_cols_full = feature_cols + ["crop_enc", "season_enc"]

    need_le = LabelEncoder()
    y = need_le.fit_transform(df["irrigation_needed"])
    X = df[feature_cols_full]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    m = evaluate("Irrigation-RF", model, X_test, y_test)

    joblib.dump(model, os.path.join(MODELS_DIR, "irrigation_model.pkl"))
    joblib.dump(crop_le, os.path.join(MODELS_DIR, "irrigation_crop_encoder.pkl"))
    joblib.dump(season_le, os.path.join(MODELS_DIR, "irrigation_season_encoder.pkl"))
    joblib.dump(need_le, os.path.join(MODELS_DIR, "irrigation_label_encoder.pkl"))
    joblib.dump(feature_cols_full, os.path.join(MODELS_DIR, "irrigation_feature_cols.pkl"))

    metrics_report["irrigation"] = {"model": "RandomForest", "metrics": m}


def main():
    train_crop_model()
    train_disease_model()
    train_fertilizer_model()
    train_irrigation_model()

    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)
    print("\nAll models trained and saved to /models. Metrics written to /reports/model_metrics.json")


if __name__ == "__main__":
    main()
