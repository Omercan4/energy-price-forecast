"""Weather forecasts from Open-Meteo (data from DWD and others, CC BY 4.0, non-commercial use).

For training we use the *historical forecast* archive, not measured weather: on the day before delivery
only forecasts are known, so measured weather would leak information into the model.
Note: the archive stitches together the first hours of each past model run, so the lead time is a bit
shorter than a true day-ahead forecast (see MODEL_CARD.md).
"""

from datetime import date, timedelta

import pandas as pd

from epf import config
from epf.data.http import cached_json

HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo.com (CC BY 4.0)"
CHUNK_DAYS = 365
FRESH_HOURS = 3


def parse_hourly(data) -> pd.DataFrame:
    """Open-Meteo's {"hourly": {"time": [...], "x": [...]}} -> DataFrame with a UTC index."""
    hourly = data["hourly"]
    frame = pd.DataFrame({name: hourly[name] for name in config.WEATHER_VARIABLES})
    frame.index = pd.to_datetime(hourly["time"], utc=True)  # we ask for timezone=UTC
    return frame


def combine_places(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per hour for the whole country: three columns."""
    cities = [frames[place] for place in ("berlin", "cologne", "munich")]
    return pd.DataFrame(
        {
            "temp": sum(f["temperature_2m"] for f in cities) / 3,
            "rad": sum(f["shortwave_radiation"] for f in cities) / 3,
            "wind_ns": frames["north_sea"]["wind_speed_100m"],
            "wind_inland": sum(f["wind_speed_100m"] for f in cities) / 3,
        }
    )


def fetch_place(place: str, start: date, end: date, forecast: bool) -> pd.DataFrame:
    latitude, longitude = config.WEATHER_PLACES[place]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(config.WEATHER_VARIABLES),
        "timezone": "UTC",
    }
    url = FORECAST_URL if forecast else HISTORICAL_URL
    path = config.RAW_DIR / "weather" / f"{'fc' if forecast else 'hist'}_{place}_{start}_{end}.json"
    recent = end >= date.today() - timedelta(days=3)
    return parse_hourly(cached_json(url, path, params=params, max_age_hours=FRESH_HOURS if recent else None))


def fetch_weather(start: date, end: date, forecast: bool = False) -> pd.DataFrame:
    """Hourly country-wide weather features from start to end (both days included)."""
    chunks, current = [], start
    while current <= end:
        stop = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        frames = {place: fetch_place(place, current, stop, forecast) for place in config.WEATHER_PLACES}
        chunks.append(combine_places(frames))
        current = stop + timedelta(days=1)
    return pd.concat(chunks).sort_index()
