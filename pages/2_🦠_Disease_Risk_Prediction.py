"""Disease Risk Prediction page."""
from __future__ import annotations
import os, sys
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, CROPS
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header, risk_badge
from utils.model_utils import predict_disease_risk
from utils.crop_info import get_disease_guidance
from utils.pdf_utils import build_disease_report

st.set_page_config(page_title=f"Disease Risk | {APP_NAME}", page_icon="🦠", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("🦠 Disease Risk Prediction", "Estimate crop disease risk from weather conditions")

with st.form("disease_form"):
    c1, c2 = st.columns(2)
    with c1:
        crop = st.selectbox("Crop", CROPS)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=78.0, step=1.0)
    with c2:
        rainfall = st.number_input("Recent Rainfall (mm)", min_value=0.0, max_value=400.0, value=90.0, step=5.0)
        leaf_wetness_hours = st.slider("Leaf Wetness Duration (hours/day)", 0.0, 24.0, 8.0, 0.5)
        leaf_image = st.file_uploader("Optional: Upload Leaf Image", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("🔬 Predict Disease Risk", use_container_width=True)

if leaf_image is not None:
    st.caption(
        "Note: leaf-image upload is captured for your records here; this demo's risk model "
        "uses weather-based features only (no computer-vision leaf diagnosis model is bundled)."
    )
    st.image(leaf_image, caption="Uploaded leaf image", width=240)

if submitted:
    result = predict_disease_risk(crop, temperature, humidity, rainfall, leaf_wetness_hours)
    # The trained model predicts a risk LEVEL (Low/Medium/High) from weather features.
    # Guidance is generic prevention/treatment reference content for that risk level.
    guidance = get_disease_guidance("Healthy" if result["risk_level"] == "Low" else "default")
    st.session_state["last_disease_result"] = result
    st.session_state["last_disease_inputs"] = {
        "Crop": crop, "Temperature (°C)": temperature, "Humidity (%)": humidity,
        "Rainfall (mm)": rainfall, "Leaf Wetness (hrs)": leaf_wetness_hours,
    }
    st.session_state["last_disease_guidance"] = guidance

if "last_disease_result" in st.session_state:
    result = st.session_state["last_disease_result"]
    inputs = st.session_state["last_disease_inputs"]
    guidance = st.session_state["last_disease_guidance"]
    risk_level = result["risk_level"]

    colA, colB = st.columns([1, 1.2])
    with colA:
        st.markdown(
            f"""
            <div class="glass-card fade-in">
                <h3>Risk Assessment</h3>
                {risk_badge(risk_level)}
                <p style="margin-top:10px;"><b>Confidence:</b> {result['confidence']*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pdf_bytes = build_disease_report(inputs, result, guidance)
        st.download_button(
            "📄 Download PDF Report", data=pdf_bytes,
            file_name="disease_risk_report.pdf", mime="application/pdf",
            use_container_width=True,
        )

    with colB:
        gauge_color = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}.get(risk_level, "#616161")
        gauge_value = {"Low": 20, "Medium": 55, "High": 90}.get(risk_level, 50)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={"text": "Disease Risk Gauge"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 35], "color": "#E8F5E9"},
                    {"range": [35, 65], "color": "#FFF3CD"},
                    {"range": [65, 100], "color": "#FADBD8"},
                ],
            },
        ))
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🛡️ Prevention", "🌿 Organic Treatment", "🧪 Chemical Treatment"])
    with t1:
        for tip in guidance["prevention"]:
            st.markdown(f"- {tip}")
    with t2:
        for tip in guidance["organic_treatment"]:
            st.markdown(f"- {tip}")
    with t3:
        for tip in guidance["chemical_treatment"]:
            st.markdown(f"- {tip}")
