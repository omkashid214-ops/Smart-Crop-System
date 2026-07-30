"""About page."""
from __future__ import annotations
import os, sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, APP_VERSION
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header

st.set_page_config(page_title=f"About | {APP_NAME}", page_icon="ℹ️", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("ℹ️ About This Project", f"Version {APP_VERSION}")

st.markdown(
    """
    <div class="glass-card fade-in">
    <h3>🌾 Smart Crop Recommendation & Disease Risk Prediction System</h3>
    <p>
    An end-to-end data science portfolio project combining a real agricultural dataset with
    domain-realistic synthetic data to demonstrate a full ML product lifecycle: data generation,
    EDA, feature engineering, multi-model training &amp; comparison, an interactive Streamlit UI,
    live weather integration, and downloadable PDF reporting.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="glass-card">
        <h4>📊 Data Sources</h4>
        <ul>
            <li><b>Crop Recommendation</b> — real Kaggle dataset (2,200 rows, 22 crops, N-P-K-temp-humidity-pH-rainfall)</li>
            <li><b>Disease Risk</b> — synthetic, weather-driven (5,500 rows)</li>
            <li><b>Fertilizer Recommendation</b> — synthetic, NPK-deficit driven (5,060 rows)</li>
            <li><b>Irrigation</b> — synthetic, soil-moisture &amp; forecast driven (5,060 rows)</li>
            <li><b>Farmer Records</b> — synthetic, state/crop/economics (5,200 rows)</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="glass-card">
        <h4>🧠 Machine Learning</h4>
        <ul>
            <li>RandomForest, LightGBM, and XGBoost compared on crop recommendation</li>
            <li>Stratified train/test split + 5-fold cross-validation</li>
            <li>RandomForest classifiers for disease risk, fertilizer, and irrigation modules</li>
            <li>All metrics (accuracy, F1, precision, recall, ROC-AUC, confusion matrix) tracked</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="glass-card">
    <h4>🛠️ Tech Stack</h4>
    <p>Python 3.11 • Streamlit • Pandas • NumPy • Scikit-learn • LightGBM • XGBoost •
    Plotly • Matplotlib • Joblib • ReportLab • Requests • OpenWeatherMap API</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="glass-card">
    <h4>🚀 Future Scope</h4>
    <ul>
        <li>Real leaf-image disease classification with a CNN (currently weather-based only)</li>
        <li>Integration with government mandi price APIs for real-time market data</li>
        <li>Multi-language regional support for farmers</li>
        <li>Mobile-first PWA build</li>
        <li>SHAP-based explainability for every prediction</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-footer">
    Built for educational &amp; portfolio purposes. Not a substitute for professional
    agricultural, agronomic, or financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
