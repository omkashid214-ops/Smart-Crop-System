"""
ui_utils.py
-----------
Reusable UI helpers: CSS injection, themed metric cards, and small
formatting helpers shared across pages.
"""

from __future__ import annotations
import os
import base64
import streamlit as st

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STYLE_DIR


def load_css(dark_mode: bool = False) -> None:
    """Inject the shared stylesheet, plus a dark-mode override block."""
    css_path = os.path.join(STYLE_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    if dark_mode:
        dark_css_path = os.path.join(STYLE_DIR, "dark_mode.css")
        if os.path.exists(dark_css_path):
            with open(dark_css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_theme_state() -> bool:
    """Ensures a dark_mode flag lives in session_state; returns current value."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    return st.session_state.dark_mode


def theme_toggle_sidebar() -> None:
    st.session_state.dark_mode = st.sidebar.toggle(
        "🌗 Dark Mode", value=st.session_state.get("dark_mode", False)
    )


def metric_card(label: str, value: str, delta: str = "", icon: str = "🌱") -> str:
    """Returns HTML for a glassmorphism metric card."""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="glass-card metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    color_map = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}
    color = color_map.get(risk_level, "#616161")
    return f'<span class="risk-badge" style="background:{color}">{risk_level} Risk</span>'


def img_to_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
