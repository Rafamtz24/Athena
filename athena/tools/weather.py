"""
Weather tool.

Fetches current weather conditions from wttr.in — a free, keyless weather
service — so weather queries can be answered with real values instead of the
title/snippet text a generic web search returns.

Network-only and best-effort: any failure (no connection, timeout, bad
response, unexpected shape) returns None so the caller can degrade gracefully
without crashing Athena or fabricating data.
"""

import json
import urllib.parse
import urllib.request
from typing import Optional


_WTTR_URL = "https://wttr.in/{location}?format=j1"


def fetch_weather(
    location: str,
    timeout: float = 10.0,
    user_agent: str = "Athena/1.0",
) -> Optional[dict]:
    """Fetch current weather for a location (or IP-based when empty).

    Returns a dict with normalized fields, or None on any failure.
    """
    encoded = urllib.parse.quote((location or "").strip())
    url = _WTTR_URL.format(location=encoded)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    try:
        current = payload["current_condition"][0]
    except (KeyError, IndexError, TypeError):
        return None

    # Resolve a human-readable location name from nearest_area (best effort).
    resolved = location
    try:
        area = payload["nearest_area"][0]
        parts = [
            area.get("areaName", [{}])[0].get("value", ""),
            area.get("region", [{}])[0].get("value", ""),
            area.get("country", [{}])[0].get("value", ""),
        ]
        joined = ", ".join(p for p in parts if p)
        if joined:
            resolved = joined
    except (KeyError, IndexError, TypeError):
        pass

    def _first_value(field: str) -> str:
        try:
            return current.get(field, [{}])[0].get("value", "")
        except (IndexError, AttributeError, TypeError):
            return ""

    return {
        "location": resolved or "your location",
        "description": _first_value("weatherDesc"),
        "temp_c": current.get("temp_C"),
        "feels_like_c": current.get("FeelsLikeC"),
        "humidity": current.get("humidity"),
        "wind_kmph": current.get("windspeedKmph"),
    }


def format_weather(data: dict) -> str:
    """Render weather data into a compact block for the reasoning prompt."""
    lines = [f"Current weather for {data.get('location', 'your location')}:"]
    if data.get("description"):
        lines.append(f"- Conditions: {data['description']}")
    if data.get("temp_c") is not None:
        feels = data.get("feels_like_c")
        temp_line = f"- Temperature: {data['temp_c']}°C"
        if feels is not None and feels != data.get("temp_c"):
            temp_line += f" (feels like {feels}°C)"
        lines.append(temp_line)
    if data.get("humidity") is not None:
        lines.append(f"- Humidity: {data['humidity']}%")
    if data.get("wind_kmph") is not None:
        lines.append(f"- Wind: {data['wind_kmph']} km/h")
    return "\n".join(lines)
