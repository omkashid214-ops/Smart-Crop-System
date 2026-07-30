"""
config.py
---------
Centralised configuration for the Smart Crop Recommendation & Disease Risk
Prediction System. Reads secrets from environment variables / Streamlit
secrets so no API keys are ever hard-coded.
"""

from __future__ import annotations
import os

try:
    import streamlit as st
    _HAS_STREAMLIT_SECRETS = True
except Exception:  # pragma: no cover
    _HAS_STREAMLIT_SECRETS = False


def get_secret(key: str, default: str = "") -> str:
    """Fetch a secret from st.secrets first, then environment variables."""
    if _HAS_STREAMLIT_SECRETS:
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    return os.environ.get(key, default)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_DIR = os.path.join(BASE_DIR, "style")

# ---------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------
OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ---------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------
APP_NAME = "Smart Crop Recommendation & Disease Risk Prediction System"
APP_ICON = "🌾"
APP_VERSION = "1.0.0"

CROPS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee",
]

STATES = [
    "Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka", "Tamil Nadu",
    "Gujarat", "Madhya Pradesh", "Rajasthan", "West Bengal", "Bihar",
    "Andhra Pradesh", "Haryana",
]

SEASONS = ["Kharif", "Rabi", "Zaid"]
