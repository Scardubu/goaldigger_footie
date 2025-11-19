#!/usr/bin/env python3
"""Shared utilities for normalizing and scoring weather metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def score_weather_impact(weather_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return consistent weather impact metrics for downstream features."""

    if not weather_data:
        return {
            "impact": 0.0,
            "condition_index": 0.0,
            "temperature_c": None,
            "wind_kph": None,
            "humidity_pct": None,
            "source_live": 0.0,
        }

    temperature_c = _coerce_float(weather_data.get("temperature_c"))
    humidity_pct = _coerce_float(
        weather_data.get("humidity_pct")
        or weather_data.get("humidity_percent")
        or weather_data.get("humidity")
    )
    wind_kph = _coerce_float(
        weather_data.get("wind_kph")
        or weather_data.get("wind_speed_kmh")
        or weather_data.get("wind_speed")
    )
    precipitation_mm = _coerce_float(
        weather_data.get("precipitation_mm")
        or weather_data.get("precipitation")
    )
    condition_raw = str(
        weather_data.get("conditions")
        or weather_data.get("condition")
        or ""
    ).lower().strip()

    impact = 0.0

    if temperature_c is not None:
        if temperature_c < -5:
            impact -= 0.30
        elif temperature_c < 2:
            impact -= 0.22
        elif temperature_c < 7:
            impact -= 0.12
        elif temperature_c > 38:
            impact -= 0.30
        elif temperature_c > 32:
            impact -= 0.22
        elif temperature_c > 27:
            impact -= 0.12
        else:
            impact += 0.04

    if wind_kph is not None:
        if wind_kph > 45:
            impact -= 0.20
        elif wind_kph > 35:
            impact -= 0.16
        elif wind_kph > 25:
            impact -= 0.10
        elif wind_kph > 15:
            impact -= 0.05
        elif wind_kph < 5:
            impact += 0.02

    if precipitation_mm is not None:
        if precipitation_mm > 12:
            impact -= 0.18
        elif precipitation_mm > 6:
            impact -= 0.12
        elif precipitation_mm > 2:
            impact -= 0.06

    condition_map = {
        "clear": 0.05,
        "sunny": 0.05,
        "partly_cloudy": 0.02,
        "few clouds": 0.02,
        "cloudy": -0.02,
        "overcast": -0.04,
        "light_rain": -0.06,
        "rain": -0.12,
        "heavy_rain": -0.18,
        "snow": -0.22,
        "sleet": -0.18,
        "mist": -0.05,
        "fog": -0.12,
        "thunderstorm": -0.28,
        "storm": -0.28,
    }

    condition_index = 0.0
    for token, score in condition_map.items():
        if token in condition_raw:
            condition_index = score
            impact += score
            break

    if humidity_pct is not None:
        if humidity_pct > 85:
            impact -= 0.06
        elif humidity_pct < 30:
            impact -= 0.03

    impact = max(-0.45, min(impact, 0.15))

    source = str(weather_data.get("source") or "").lower()
    source_live = 1.0 if "api" in source or "openweather" in source else 0.0

    return {
        "impact": impact,
        "condition_index": condition_index,
        "temperature_c": temperature_c,
        "wind_kph": wind_kph,
        "humidity_pct": humidity_pct,
        "source_live": source_live,
    }
