"""Live Weather page using OpenWeatherMap."""
from __future__ import annotations
import os, sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_NAME, OPENWEATHER_API_KEY
from utils.ui_utils import load_css, init_theme_state, theme_toggle_sidebar, section_header
from utils.weather_utils import get_live_weather

st.set_page_config(page_title=f"Live Weather | {APP_NAME}", page_icon="🌦️", layout="wide")
dark_mode = init_theme_state()
with st.sidebar:
    theme_toggle_sidebar()
load_css(dark_mode=st.session_state.dark_mode)
from config import APP_NAME, OPENWEATHER_API_KEY

st.write("API Loaded:", bool(OPENWEATHER_API_KEY))
st.write("Key Length:", len(OPENWEATHER_API_KEY))
st.write("First 5 characters:", OPENWEATHER_API_KEY[:5] if OPENWEATHER_API_KEY else "None")
section_header("🌦️ Live Weather", "Pull real-time conditions to feed straight into your predictions")

if not OPENWEATHER_API_KEY:
    st.warning(
        "No OpenWeatherMap API key detected. Add `OPENWEATHER_API_KEY` to your "
        "`.streamlit/secrets.toml` file (see README) to enable live weather lookups. "
        "You can still use manual weather inputs on the other pages."
    )

c1, c2 = st.columns([2, 1])
with c1:
    city = st.text_input("City", value="Pune")
with c2:
    country = st.text_input("Country Code", value="IN", max_chars=2)

if st.button("🔄 Fetch Live Weather", use_container_width=True):
    with st.spinner("Contacting OpenWeatherMap..."):
        data, error = get_live_weather(city, country)

    if error:
        st.error(error)
    else:
        st.session_state["live_weather"] = data

if "live_weather" in st.session_state:
    data = st.session_state["live_weather"]
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🌡️", "Temperature", f"{data['temperature']:.1f} °C"),
        ("💧", "Humidity", f"{data['humidity']}%"),
        ("🌬️", "Wind Speed", f"{data['wind_speed']} m/s"),
        ("🌧️", "Rain (1h)", f"{data['rainfall_1h']} mm"),
    ]
    for col, (icon, label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"""<div class="glass-card metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    st.info(f"📍 **{data['city']}** — {data['description']}")
    st.caption(
        "Tip: copy these temperature/humidity/rainfall values into the Crop Recommendation "
        "or Disease Risk pages for location-aware predictions."
    )
