"""Weather data integration via Open-Meteo (free, no API key needed).

Weather affects tennis significantly:
- Temperature: ball bounce, player fatigue, ball speed
- Humidity: ball heaviness, grip, player endurance
- Wind: serve toss disruption, ball trajectory
- Altitude: ball flight (Denver, Bogota, etc.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tennis_predictor.config import CACHE_DIR, WEATHER_CONFIG

# Mapping of major tournaments to coordinates
# This gets built/extended from data, but we seed with known venues
VENUE_COORDS: dict[str, tuple[float, float, float]] = {
    # Grand Slams
    "Australian Open": (-37.8218, 144.9785, 17),
    "Roland Garros": (48.8469, 2.2531, 75),
    "Wimbledon": (51.4341, -0.2143, 40),
    "Us Open": (40.7499, -73.8459, 10),
    # Masters 1000
    "Indian Wells": (33.7238, -116.3052, -21),
    "Miami Open": (25.9526, -80.1411, 3),
    "Monte Carlo": (43.7519, 7.4406, 200),
    "Madrid": (40.3707, -3.6872, 650),  # High altitude!
    "Rome": (41.9242, 12.4536, 21),
    "Canada": (45.5019, -73.5674, 233),  # Montreal default
    "Cincinnati": (39.2571, -84.2868, 265),
    "Shanghai": (31.0406, 121.3552, 4),
    "Paris": (48.8951, 2.2364, 69),  # Paris Masters (Bercy)
    # ATP 500
    "Rotterdam": (51.8797, 4.4863, -1),
    "Dubai": (25.2048, 55.2708, 5),
    "Barcelona": (41.3928, 2.1140, 90),
    "Hamburg": (53.5763, 9.9854, 6),
    "Washington": (38.9497, -77.0458, 15),
    "Beijing": (39.9830, 116.4087, 50),
    "Tokyo": (35.6956, 139.7529, 40),
    "Vienna": (48.2117, 16.3633, 171),
    "Basel": (47.5498, 7.6197, 260),
    "Halle": (52.0690, 8.3528, 80),
    "Queens Club": (51.4884, -0.2130, 10),
    # ATP 250 (selected)
    "Brisbane": (-27.4710, 153.0234, 10),
    "Adelaide": (-34.9285, 138.6007, 50),
    "Doha": (25.2637, 51.4467, 10),
    "Santiago": (-33.4489, -70.6693, 520),
    "Marrakech": (31.6348, -7.9999, 466),
    "Buenos Aires": (-34.5824, -58.4368, 25),
    "Acapulco": (16.8531, -99.8237, 3),
    "Dallas": (32.8318, -96.7954, 131),
    "Bogota": (4.6097, -74.0818, 2640),  # Very high altitude!
    "Quito": (-0.1807, -78.4678, 2850),  # Very high altitude!
    "Gstaad": (46.4749, 7.2847, 1050),
    "Kitzbuhel": (47.4497, 6.3653, 760),
    "Umag": (45.4319, 13.5230, 10),
    "Atlanta": (33.8489, -84.3737, 320),
    "Winston Salem": (36.0999, -80.2442, 280),
    "Stockholm": (59.3293, 18.0686, 28),
    "Antwerp": (51.1964, 4.4109, 7),
    "Metz": (49.1196, 6.1764, 180),
}


def get_venue_coords(tourney_name: str) -> tuple[float, float, float] | None:
    """Look up venue coordinates. Returns (lat, lon, altitude_m) or None."""
    # Direct lookup
    for key, coords in VENUE_COORDS.items():
        if key.lower() in tourney_name.lower() or tourney_name.lower() in key.lower():
            return coords
    return None


def geocode_tournament(tourney_name: str) -> tuple[float, float, float] | None:
    """Geocode a tournament name using Open-Meteo's free geocoding API."""
    cache_file = CACHE_DIR / "geocode_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Load cache
    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())

    if tourney_name in cache:
        c = cache[tourney_name]
        return (c["lat"], c["lon"], c.get("elevation", 0)) if c else None

    # Try Open-Meteo geocoding
    url = WEATHER_CONFIG["geocoding_base"]
    try:
        resp = requests.get(url, params={"name": tourney_name, "count": 1}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                r = results[0]
                coords = {
                    "lat": r["latitude"],
                    "lon": r["longitude"],
                    "elevation": r.get("elevation", 0),
                }
                cache[tourney_name] = coords
                cache_file.write_text(json.dumps(cache, indent=2))
                return (coords["lat"], coords["lon"], coords["elevation"])
    except requests.RequestException:
        pass

    cache[tourney_name] = None
    cache_file.write_text(json.dumps(cache, indent=2))
    return None


def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> dict | None:
    """Fetch historical weather data from Open-Meteo.

    Args:
        lat: Latitude.
        lon: Longitude.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        Dict with daily weather data, or None on failure.
    """
    # Check cache first
    cache_key = f"{lat:.2f}_{lon:.2f}_{start_date}_{end_date}"
    cache_file = CACHE_DIR / "weather" / f"{cache_key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = WEATHER_CONFIG["api_base"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(WEATHER_CONFIG["variables"]),
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            cache_file.write_text(json.dumps(data, indent=2))
            return data
        else:
            return None
    except requests.RequestException:
        return None


def get_match_weather(
    tourney_name: str,
    match_date: pd.Timestamp,
) -> dict:
    """Get weather features for a specific match.

    Returns a dict of weather features or NaN values if unavailable.
    """
    default = {
        "weather_temp_max": np.nan,
        "weather_temp_min": np.nan,
        "weather_precipitation": np.nan,
        "weather_wind_max": np.nan,
        "weather_wind_gust_max": np.nan,
        "weather_altitude": np.nan,
        "weather_is_indoor": np.nan,
    }

    if pd.isna(match_date):
        return default

    # Known indoor tournaments
    indoor_tournaments = {
        "paris", "rotterdam", "vienna", "basel", "stockholm", "antwerp",
        "metz", "marseille", "sofia", "montpellier", "dallas", "atp finals",
    }
    is_indoor = any(t in tourney_name.lower() for t in indoor_tournaments)

    if is_indoor:
        return {
            "weather_temp_max": 22.0,  # Controlled indoor temp
            "weather_temp_min": 20.0,
            "weather_precipitation": 0.0,
            "weather_wind_max": 0.0,
            "weather_wind_gust_max": 0.0,
            "weather_altitude": 0.0,
            "weather_is_indoor": 1.0,
        }

    coords = get_venue_coords(tourney_name)
    if coords is None:
        coords = geocode_tournament(tourney_name)
    if coords is None:
        return default

    lat, lon, altitude = coords
    date_str = match_date.strftime("%Y-%m-%d")

    weather = fetch_weather(lat, lon, date_str, date_str)
    if weather is None or "daily" not in weather:
        return default

    daily = weather["daily"]
    idx = 0  # Single day

    return {
        "weather_temp_max": _safe_get(daily, "temperature_2m_max", idx),
        "weather_temp_min": _safe_get(daily, "temperature_2m_min", idx),
        "weather_precipitation": _safe_get(daily, "precipitation_sum", idx),
        "weather_wind_max": _safe_get(daily, "windspeed_10m_max", idx),
        "weather_wind_gust_max": _safe_get(daily, "windgusts_10m_max", idx),
        "weather_altitude": float(altitude),
        "weather_is_indoor": 0.0,
    }


def _safe_get(daily: dict, key: str, idx: int) -> float:
    vals = daily.get(key, [])
    if idx < len(vals) and vals[idx] is not None:
        return float(vals[idx])
    return np.nan
