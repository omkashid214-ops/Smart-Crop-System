"""Model Performance page: RandomForest vs LightGBM vs XGBoost comparison."""
from __future__ import annotations
import os, sys, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, REPORTS_DIR
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header

st.set_page_config(page_title=f"Model Performance | {APP_NAME}", page_icon="📈", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)

section_header("📈 Model Performance", "Compare RandomForest, LightGBM & XGBoost on the crop dataset")

metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
if not os.path.exists(metrics_path):
    st.error("No metrics report found. Run `python utils/train_models.py` first.")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

crop_metrics = metrics.get("crop_recommendation", {})
models_compared = crop_metrics.get("models_compared", {})
best_model = crop_metrics.get("best_model", "N/A")

st.info(f"🏆 **Best performing model on the Crop Recommendation task:** {best_model}")

if models_compared:
    rows = []
    for name, m in models_compared.items():
        rows.append({
            "Model": name, "Accuracy": m["accuracy"], "F1 (macro)": m["f1_macro"],
            "Precision (macro)": m["precision_macro"], "Recall (macro)": m["recall_macro"],
            "CV Accuracy (mean)": m.get("cv_accuracy_mean"),
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    melt_df = comp_df.melt(id_vars="Model", value_vars=["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"],
                            var_name="Metric", value_name="Score")
    fig = px.bar(melt_df, x="Model", y="Score", color="Metric", barmode="group",
                 title="Model Comparison — Crop Recommendation",
                 color_discrete_sequence=["#2E7D32", "#66BB6A", "#6D4C41", "#D7CCC8"])
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

if crop_metrics.get("confusion_matrix") and crop_metrics.get("classes"):
    st.markdown("### Confusion Matrix — Best Model")
    cm = crop_metrics["confusion_matrix"]
    classes = crop_metrics["classes"]
    fig = px.imshow(cm, x=classes, y=classes, text_auto=True, color_continuous_scale="Greens",
                     labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

if crop_metrics.get("feature_importance"):
    st.markdown("### Feature Importance — Best Model")
    fi = crop_metrics["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance")
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#D7CCC8", "#2E7D32"])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### Other Models — Quick Metrics")
c1, c2, c3 = st.columns(3)
other_models = [
    ("🦠 Disease Risk (RandomForest)", metrics.get("disease_risk", {}).get("metrics", {})),
    ("🧪 Fertilizer (RandomForest)", metrics.get("fertilizer", {}).get("metrics", {})),
    ("💧 Irrigation (RandomForest)", metrics.get("irrigation", {}).get("metrics", {})),
]
for col, (title, m) in zip([c1, c2, c3], other_models):
    with col:
        acc = m.get("accuracy", "N/A")
        f1 = m.get("f1_macro", "N/A")
        st.markdown(
            f"""<div class="glass-card">
            <h4>{title}</h4>
            <p><b>Accuracy:</b> {acc}</p>
            <p><b>F1 (macro):</b> {f1}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.caption(
    "All models were trained with an 80/20 stratified train-test split and evaluated with "
    "5-fold cross-validation on the training set (see utils/train_models.py)."
)
