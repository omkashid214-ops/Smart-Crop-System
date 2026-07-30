"""Fertilizer Recommendation page."""
from __future__ import annotations
import os, sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, CROPS, DATASET_DIR
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header
from utils.model_utils import predict_fertilizer

st.set_page_config(page_title=f"Fertilizer Recommendation | {APP_NAME}", page_icon="🧪", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("🧪 Fertilizer Recommendation", "Get the right fertilizer, dosage & cost estimate for your crop")


@st.cache_data(show_spinner=False)
def load_fert_lookup():
    return pd.read_csv(os.path.join(DATASET_DIR, "fertilizer_recommendation.csv"))


fert_df = load_fert_lookup()

with st.form("fert_form"):
    c1, c2 = st.columns(2)
    with c1:
        crop = st.selectbox("Crop", CROPS)
        N = st.number_input("Nitrogen - N (kg/ha)", min_value=0.0, max_value=150.0, value=55.0)
        P = st.number_input("Phosphorus - P (kg/ha)", min_value=0.0, max_value=150.0, value=40.0)
    with c2:
        K = st.number_input("Potassium - K (kg/ha)", min_value=0.0, max_value=210.0, value=45.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    submitted = st.form_submit_button("🧪 Recommend Fertilizer", use_container_width=True)

if submitted:
    result = predict_fertilizer(crop, N, P, K, ph)
    fert_name = result["fertilizer"]

    # Look up representative cost/quantity/organic-alternative context for this fertilizer + crop
    subset = fert_df[(fert_df["crop"] == crop) & (fert_df["recommended_fertilizer"] == fert_name)]
    if subset.empty:
        subset = fert_df[fert_df["recommended_fertilizer"] == fert_name]
    avg_qty = subset["quantity_kg_per_acre"].mean() if not subset.empty else None
    avg_cost = subset["estimated_cost_inr"].mean() if not subset.empty else None
    organic_alt = subset["organic_alternative"].mode()[0] if not subset.empty else "Vermicompost"

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(
            f"""<div class="glass-card fade-in"><h4>🧪 Recommended Fertilizer</h4>
            <h2 style="color:#2E7D32;">{fert_name}</h2>
            <p><b>Confidence:</b> {result['confidence']*100:.1f}%</p></div>""",
            unsafe_allow_html=True,
        )
    with colB:
        qty_txt = f"{avg_qty:.1f} kg/acre" if avg_qty is not None else "N/A"
        cost_txt = f"₹{avg_cost:,.0f} / acre" if avg_cost is not None else "N/A"
        st.markdown(
            f"""<div class="glass-card fade-in"><h4>📦 Suggested Dosage</h4>
            <h2 style="color:#6D4C41;">{qty_txt}</h2>
            <p><b>Estimated Cost:</b> {cost_txt}</p></div>""",
            unsafe_allow_html=True,
        )
    with colC:
        st.markdown(
            f"""<div class="glass-card fade-in"><h4>🌿 Organic Alternative</h4>
            <h2 style="color:#2E7D32;">{organic_alt}</h2>
            <p>Consider blending with chemical fertilizer to improve long-term soil health.</p></div>""",
            unsafe_allow_html=True,
        )

    st.caption(
        "Dosage & cost figures are representative estimates derived from the project's synthetic "
        "fertilizer dataset — always confirm with a local soil-testing lab before field application."
    )
