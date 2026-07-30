"""Batch Prediction page: upload a CSV, get crop recommendations for every row."""
from __future__ import annotations
import os, sys
import io
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header
from utils.model_utils import load_crop_artifacts

st.set_page_config(page_title=f"Batch Prediction | {APP_NAME}", page_icon="📁", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("📁 Dataset Upload & Batch Prediction", "Score hundreds of rows at once")

REQUIRED_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

st.markdown(
    f"Upload a CSV with the following columns: `{'`, `'.join(REQUIRED_COLS)}`. "
    "Extra columns are kept and passed through untouched."
)

sample_csv = pd.DataFrame(
    [[90, 42, 43, 20.9, 82.0, 6.5, 202.9], [85, 58, 41, 21.8, 80.3, 7.0, 226.7]],
    columns=REQUIRED_COLS,
)
st.download_button(
    "⬇️ Download Sample CSV Template",
    data=sample_csv.to_csv(index=False).encode("utf-8"),
    file_name="batch_prediction_template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        st.stop()

    if df[REQUIRED_COLS].isnull().any().any():
        st.warning("Some rows contain missing values in required columns; those rows will be skipped.")
        df = df.dropna(subset=REQUIRED_COLS)

    with st.spinner(f"Scoring {len(df)} rows..."):
        artifacts = load_crop_artifacts()
        model = artifacts["model"]
        le = artifacts["label_encoder"]
        X = df[REQUIRED_COLS]
        try:
            preds = model.predict(X)
            probas = model.predict_proba(X)
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
            st.stop()

        df["recommended_crop"] = le.inverse_transform(preds)
        df["confidence"] = probas.max(axis=1).round(4)

    st.success(f"Scored {len(df)} rows successfully.")
    st.dataframe(df, use_container_width=True, height=420)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Results CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="batch_crop_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        summary = df["recommended_crop"].value_counts().reset_index()
        summary.columns = ["crop", "count"]
        st.bar_chart(summary.set_index("crop"))
