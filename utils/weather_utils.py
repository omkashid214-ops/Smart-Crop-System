"""
weather_utils.py
-----------------
Thin wrapper around the OpenWeatherMap "Current Weather" API.
Requires OPENWEATHER_API_KEY to be set via Streamlit secrets or an
environment variable. Fails gracefully (returns None + error message)
when the key is missing or the request fails, so the rest of the app
keeps working with manual weather input.
"""

from __future__ import annotations
import logging
from typing import Optional

import requests

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL

logger = logging.getLogger(__name__)


def get_live_weather(city: str, country_code: str = "IN") -> tuple[Optional[dict], Optional[str]]:
    """
    Fetch current weather for a city.

    Returns:
        (data, error) tuple. `data` is None if the call failed, in which
        case `error` contains a human-readable message.
    """
    if not OPENWEATHER_API_KEY:
        return None, "No OpenWeatherMap API key configured. Add OPENWEATHER_API_KEY to secrets."

    params = {
        "q": f"{city},{country_code}",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=8)
        if resp.status_code != 200:
            return None, f"Weather API returned status {resp.status_code}: {resp.text[:120]}"
        payload = resp.json()
        data = {
            "city": payload.get("name", city),
            "temperature": payload["main"]["temp"],
            "humidity": payload["main"]["humidity"],
            "pressure": payload["main"]["pressure"],
            "wind_speed": payload["wind"]["speed"],
            "description": payload["weather"][0]["description"].title(),
            "icon": payload["weather"][0]["icon"],
            "rainfall_1h": payload.get("rain", {}).get("1h", 0.0),
        }
        return data, None
    except requests.exceptions.RequestException as exc:
        logger.warning("Weather API request failed: %s", exc)
        return None, f"Could not reach weather service: {exc}"
    except (KeyError, ValueError) as exc:
        logger.warning("Weather API response parsing failed: %s", exc)
        return None, f"Unexpected response from weather service: {exc}"
