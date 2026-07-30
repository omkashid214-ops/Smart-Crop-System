"""Interactive Dashboard: KPIs, correlations, feature importance, prediction stats."""
from __future__ import annotations
import os, sys, json
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, DATASET_DIR, REPORTS_DIR
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header

st.set_page_config(page_title=f"Dashboard | {APP_NAME}", page_icon="📊", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("📊 Interactive Dashboard", "Explore the data behind every prediction")


@st.cache_data(show_spinner=False)
def load_datasets():
    crop_df = pd.read_csv(os.path.join(DATASET_DIR, "Crop_recommendation.csv"))
    disease_df = pd.read_csv(os.path.join(DATASET_DIR, "disease_risk.csv"))
    fert_df = pd.read_csv(os.path.join(DATASET_DIR, "fertilizer_recommendation.csv"))
    irrigation_df = pd.read_csv(os.path.join(DATASET_DIR, "irrigation.csv"))
    farmer_df = pd.read_csv(os.path.join(DATASET_DIR, "farmer_records.csv"))
    return crop_df, disease_df, fert_df, irrigation_df, farmer_df


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(REPORTS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


crop_df, disease_df, fert_df, irrigation_df, farmer_df = load_datasets()
metrics = load_metrics()

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("🌾", "Crop Records", f"{len(crop_df):,}"),
    ("🦠", "Disease Risk Records", f"{len(disease_df):,}"),
    ("🧪", "Fertilizer Records", f"{len(fert_df):,}"),
    ("👨‍🌾", "Farmer Records", f"{len(farmer_df):,}"),
]
for col, (icon, label, value) in zip([k1, k2, k3, k4], kpis):
    with col:
        st.markdown(
            f"""<div class="glass-card metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(
    ["🌾 Crop Data", "🦠 Disease Risk", "🧪 Fertilizer & Irrigation", "👨‍🌾 Farmer Records"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        counts = crop_df["label"].value_counts().reset_index()
        counts.columns = ["crop", "count"]
        fig = px.bar(counts, x="count", y="crop", orientation="h",
                     title="Records per Crop", color="count",
                     color_continuous_scale=["#D7CCC8", "#66BB6A", "#2E7D32"])
        fig.update_layout(height=560, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        corr = crop_df[num_cols].corr()
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="Greens",
                          title="Feature Correlation Heatmap")
        fig2.update_layout(height=560)
        st.plotly_chart(fig2, use_container_width=True)

    if metrics.get("crop_recommendation", {}).get("feature_importance"):
        fi = metrics["crop_recommendation"]["feature_importance"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance")
        fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                      title="Crop Model Feature Importance", color="Importance",
                      color_continuous_scale=["#D7CCC8", "#6D4C41"])
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        risk_counts = disease_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig = px.pie(risk_counts, names="risk_level", values="count", hole=0.45,
                     title="Disease Risk Level Distribution",
                     color="risk_level",
                     color_discrete_map={"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(disease_df.sample(min(1500, len(disease_df)), random_state=42),
                          x="humidity", y="leaf_wetness_hours", color="risk_level",
                          color_discrete_map={"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"},
                          title="Humidity vs Leaf Wetness (colored by risk)")
        st.plotly_chart(fig, use_container_width=True)

    top_diseases = disease_df[disease_df["disease_name"] != "Healthy"]["disease_name"].value_counts().head(10).reset_index()
    top_diseases.columns = ["disease", "count"]
    fig = px.bar(top_diseases, x="count", y="disease", orientation="h", title="Top 10 Predicted Diseases (synthetic)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fert_counts = fert_df["recommended_fertilizer"].value_counts().reset_index()
        fert_counts.columns = ["fertilizer", "count"]
        fig = px.bar(fert_counts, x="count", y="fertilizer", orientation="h", title="Fertilizer Recommendation Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        irrig_counts = irrigation_df["irrigation_needed"].value_counts().reset_index()
        irrig_counts.columns = ["irrigation_needed", "count"]
        fig = px.pie(irrig_counts, names="irrigation_needed", values="count", hole=0.45,
                     title="Irrigation Needed Distribution")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(fert_df, x="crop", y="estimated_cost_inr", title="Estimated Fertilizer Cost by Crop (₹/acre)")
    fig.update_layout(xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        state_rev = farmer_df.groupby("state")["estimated_revenue_inr"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(state_rev, x="estimated_revenue_inr", y="state", orientation="h",
                     title="Avg. Estimated Revenue by State (synthetic)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(farmer_df.sample(min(1500, len(farmer_df)), random_state=42),
                          x="land_size_acres", y="yield_quintal_per_acre", color="crop",
                          title="Land Size vs Yield")
        st.plotly_chart(fig, use_container_width=True)

    irrigation_type_counts = farmer_df["irrigation_type"].value_counts().reset_index()
    irrigation_type_counts.columns = ["irrigation_type", "count"]
    fig = px.pie(irrigation_type_counts, names="irrigation_type", values="count", hole=0.45,
                 title="Irrigation Type Usage Across Farmer Records")
    st.plotly_chart(fig, use_container_width=True)
