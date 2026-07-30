"""
model_utils.py
---------------
Reusable helpers to load trained models (cached) and run predictions for
crop recommendation, disease risk, fertilizer recommendation, and irrigation.
"""

from __future__ import annotations
import os
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load(path: str) -> Any:
    full_path = os.path.join(MODELS_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Model file not found: {full_path}. Run `python utils/train_models.py` first."
        )
    return joblib.load(full_path)


@st.cache_resource(show_spinner=False)
def load_crop_artifacts() -> dict:
    """Load crop recommendation model + supporting encoders."""
    return {
        "model": _load("crop_model.pkl"),
        "label_encoder": _load("crop_label_encoder.pkl"),
        "scaler": _load("crop_scaler.pkl"),
        "feature_cols": _load("crop_feature_cols.pkl"),
    }


@st.cache_resource(show_spinner=False)
def load_disease_artifacts() -> dict:
    return {
        "model": _load("disease_risk_model.pkl"),
        "crop_encoder": _load("disease_crop_encoder.pkl"),
        "risk_encoder": _load("disease_risk_label_encoder.pkl"),
        "feature_cols": _load("disease_feature_cols.pkl"),
    }


@st.cache_resource(show_spinner=False)
def load_fertilizer_artifacts() -> dict:
    return {
        "model": _load("fertilizer_model.pkl"),
        "crop_encoder": _load("fertilizer_crop_encoder.pkl"),
        "label_encoder": _load("fertilizer_label_encoder.pkl"),
        "feature_cols": _load("fertilizer_feature_cols.pkl"),
    }


@st.cache_resource(show_spinner=False)
def load_irrigation_artifacts() -> dict:
    return {
        "model": _load("irrigation_model.pkl"),
        "crop_encoder": _load("irrigation_crop_encoder.pkl"),
        "season_encoder": _load("irrigation_season_encoder.pkl"),
        "label_encoder": _load("irrigation_label_encoder.pkl"),
        "feature_cols": _load("irrigation_feature_cols.pkl"),
    }


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def predict_crop(N: float, P: float, K: float, temperature: float,
                  humidity: float, ph: float, rainfall: float, top_k: int = 5) -> dict:
    """Returns top-k recommended crops with confidence scores."""
    artifacts = load_crop_artifacts()
    model = artifacts["model"]
    le = artifacts["label_encoder"]
    feature_cols = artifacts["feature_cols"]

    row = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_cols)

    try:
        proba = model.predict_proba(row)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise RuntimeError(f"Crop prediction failed: {exc}") from exc

    top_idx = np.argsort(proba)[::-1][:top_k]
    top_crops = [(le.inverse_transform([i])[0], float(proba[i])) for i in top_idx]
    return {
        "top_crop": top_crops[0][0],
        "confidence": top_crops[0][1],
        "top_k": top_crops,
    }


def predict_disease_risk(crop: str, temperature: float, humidity: float,
                          rainfall: float, leaf_wetness_hours: float) -> dict:
    artifacts = load_disease_artifacts()
    model = artifacts["model"]
    crop_le = artifacts["crop_encoder"]
    risk_le = artifacts["risk_encoder"]
    feature_cols = artifacts["feature_cols"]

    if crop not in crop_le.classes_:
        crop_enc = 0
    else:
        crop_enc = int(crop_le.transform([crop])[0])

    row = pd.DataFrame(
        [[temperature, humidity, rainfall, leaf_wetness_hours, crop_enc]],
        columns=feature_cols,
    )
    proba = model.predict_proba(row)[0]
    pred_idx = int(np.argmax(proba))
    risk_level = risk_le.inverse_transform([pred_idx])[0]
    return {
        "risk_level": risk_level,
        "confidence": float(proba[pred_idx]),
        "probabilities": dict(zip(risk_le.classes_, proba.round(4).tolist())),
    }


def predict_fertilizer(crop: str, N: float, P: float, K: float, ph: float) -> dict:
    artifacts = load_fertilizer_artifacts()
    model = artifacts["model"]
    crop_le = artifacts["crop_encoder"]
    label_le = artifacts["label_encoder"]
    feature_cols = artifacts["feature_cols"]

    crop_enc = int(crop_le.transform([crop])[0]) if crop in crop_le.classes_ else 0
    row = pd.DataFrame([[N, P, K, ph, crop_enc]], columns=feature_cols)
    proba = model.predict_proba(row)[0]
    pred_idx = int(np.argmax(proba))
    fertilizer = label_le.inverse_transform([pred_idx])[0]
    return {"fertilizer": fertilizer, "confidence": float(proba[pred_idx])}


def predict_irrigation(crop: str, season: str, soil_moisture_pct: float,
                        temperature: float, rainfall_forecast_mm: float) -> dict:
    artifacts = load_irrigation_artifacts()
    model = artifacts["model"]
    crop_le = artifacts["crop_encoder"]
    season_le = artifacts["season_encoder"]
    label_le = artifacts["label_encoder"]
    feature_cols = artifacts["feature_cols"]

    crop_enc = int(crop_le.transform([crop])[0]) if crop in crop_le.classes_ else 0
    season_enc = int(season_le.transform([season])[0]) if season in season_le.classes_ else 0
    row = pd.DataFrame(
        [[soil_moisture_pct, temperature, rainfall_forecast_mm, crop_enc, season_enc]],
        columns=feature_cols,
    )
    proba = model.predict_proba(row)[0]
    pred_idx = int(np.argmax(proba))
    needed = label_le.inverse_transform([pred_idx])[0]
    return {"irrigation_needed": needed, "confidence": float(proba[pred_idx])}
