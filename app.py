"""
app.py
------
Landing page for the Smart Crop Recommendation & Disease Risk Prediction
System. Run with:  streamlit run app.py
"""

from __future__ import annotations
import streamlit as st

from config import APP_NAME, APP_ICON, APP_VERSION
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

dark_mode = init_theme_state()
with st.sidebar:
    st.markdown(f"## {APP_ICON} Smart Crop System")
    st.caption(f"v{APP_VERSION}")
    theme_toggle_sidebar()
    st.markdown("---")
    st.markdown(
        "Use the **pages menu above** (or the links below) to navigate:\n\n"
        "- 🌱 Crop Recommendation\n"
        "- 🦠 Disease Risk Prediction\n"
        "- 🧪 Fertilizer Recommendation\n"
        "- 🌦️ Live Weather\n"
        "- 📊 Dashboard\n"
        "- 📁 Batch Prediction\n"
        "- 📈 Model Performance\n"
        "- ℹ️ About"
    )

load_css(dark_mode=st.session_state.dark_mode)

# ---------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner fade-in">
        <h1>{APP_ICON} Smart Crop Recommendation &amp; Disease Risk Prediction System</h1>
        <p>AI-powered decisions for healthier crops, smarter fertilization, and better yields</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Quick KPIs
# ---------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
kpis = [
    ("🌾", "Crops Supported", "22"),
    ("🧬", "ML Models Trained", "4"),
    ("📊", "Synthetic + Real Records", "20,000+"),
    ("🎯", "Best Model Accuracy", "99.3%"),
]
for col, (icon, label, value) in zip([col1, col2, col3, col4], kpis):
    with col:
        st.markdown(
            f"""
            <div class="glass-card metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
section_header("What can you do here?", "Explore every module of the platform")

features = [
    ("🌱", "Crop Recommendation", "Get the best crop for your soil & climate with confidence scores and top-5 alternatives."),
    ("🦠", "Disease Risk Prediction", "Estimate disease risk from weather conditions, with prevention & treatment guidance."),
    ("🧪", "Fertilizer Recommendation", "Find the right fertilizer, dosage, cost estimate, and organic alternative."),
    ("🌦️", "Live Weather", "Pull real-time weather via OpenWeatherMap to feed straight into your predictions."),
    ("📊", "Interactive Dashboard", "Explore KPIs, correlations, and feature importance across all datasets."),
    ("📁", "Batch Prediction", "Upload a CSV and get crop recommendations for hundreds of rows at once."),
    ("📈", "Model Performance", "Compare RandomForest, LightGBM & XGBoost — accuracy, F1, ROC, confusion matrix."),
    ("📄", "PDF Reports", "Download a polished PDF report of any prediction to share or archive."),
]

rows = [features[i:i + 4] for i in range(0, len(features), 4)]
for row in rows:
    cols = st.columns(len(row))
    for col, (icon, title, desc) in zip(cols, row):
        with col:
            st.markdown(
                f"""
                <div class="feature-card fade-in">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")
st.info(
    "👈 Use the sidebar page navigation to jump into any module. "
    "Start with **Crop Recommendation** to see the core ML pipeline in action."
)

st.markdown(
    """
    <div class="app-footer">
        Built with Streamlit • scikit-learn • LightGBM • XGBoost 🌾 Smart Crop System
    </div>
    """,
    unsafe_allow_html=True,
)
