"""Crop Recommendation page: N,P,K,temp,humidity,ph,rainfall,state,season -> crop."""
from __future__ import annotations
import os, sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, APP_ICON, STATES, SEASONS
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header
from utils.model_utils import predict_crop
from utils.crop_info import get_crop_details
from utils.pdf_utils import build_crop_report

st.set_page_config(page_title=f"Crop Recommendation | {APP_NAME}", page_icon="🌱", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("🌱 Crop Recommendation", "Enter soil & climate parameters to get the best crop suggestion")

with st.form("crop_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen - N (kg/ha)", min_value=0.0, max_value=150.0, value=90.0, step=1.0)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
        state = st.selectbox("State", STATES)
    with c2:
        P = st.number_input("Phosphorus - P (kg/ha)", min_value=0.0, max_value=150.0, value=42.0, step=1.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        season = st.selectbox("Season", SEASONS)
    with c3:
        K = st.number_input("Potassium - K (kg/ha)", min_value=0.0, max_value=210.0, value=43.0, step=1.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=200.0, step=5.0)

    submitted = st.form_submit_button("🔍 Recommend Crop", use_container_width=True)

if submitted:
    try:
        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall, top_k=5)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.session_state["last_crop_result"] = result
    st.session_state["last_crop_inputs"] = {
        "N": N, "P": P, "K": K, "Temperature (°C)": temperature,
        "Humidity (%)": humidity, "Soil pH": ph, "Rainfall (mm)": rainfall,
        "State": state, "Season": season,
    }

if "last_crop_result" in st.session_state:
    result = st.session_state["last_crop_result"]
    inputs = st.session_state["last_crop_inputs"]
    top_crop = result["top_crop"]
    details = get_crop_details(top_crop)

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 1.4])

    with colA:
        st.markdown(
            f"""
            <div class="glass-card fade-in">
                <h3>✅ Recommended Crop</h3>
                <h1 style="color:#2E7D32; text-transform:capitalize;">{top_crop}</h1>
                <p><b>Confidence:</b> {result['confidence']*100:.1f}%</p>
                <p><b>Category:</b> {details['category']}</p>
                <p><b>Expected Yield:</b> {details['expected_yield_range']}</p>
                <p><b>Growing Season:</b> {details['growing_season']}</p>
                <p><b>Water Need:</b> {details['water_need']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pdf_bytes = build_crop_report(inputs, result)
        st.download_button(
            "📄 Download PDF Report", data=pdf_bytes,
            file_name=f"crop_recommendation_{top_crop}.pdf", mime="application/pdf",
            use_container_width=True,
        )

    with colB:
        crops = [c.title() for c, _ in result["top_k"]]
        probs = [p * 100 for _, p in result["top_k"]]
        fig = px.bar(
            x=probs, y=crops, orientation="h",
            labels={"x": "Confidence (%)", "y": "Crop"},
            title="Top 5 Crop Probabilities", color=probs,
            color_continuous_scale=["#D7CCC8", "#66BB6A", "#2E7D32"],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=380,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Top 5 Candidate Crops")
    st.table(
        [{"Rank": i + 1, "Crop": c.title(), "Confidence": f"{p*100:.1f}%"} for i, (c, p) in enumerate(result["top_k"])]
    )
