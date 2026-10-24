"""Build the hourly dataset: price, TSO forecasts for load/wind/solar, weather forecasts."""

from datetime import date, timedelta

import pandas as pd

from epf import config
from epf.data import smard, weather


def build_dataset(start: date | str, end: date | str, include_weather: bool = True) -> pd.DataFrame:
    """One row per hour (UTC index). Columns: the SMARD series plus temp, rad, wind_ns."""
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    columns = [smard.fetch_series(name, start, end) for name in config.SMARD_SERIES]
    frame = pd.concat(columns, axis=1)
    if include_weather:
        frame = frame.join(weather.fetch_weather(start, end, forecast=False), how="left")
    frame = frame.sort_index()
    frame.index.name = "time"
    return frame


def save_dataset(path=None, start=config.HISTORY_START, end=None, include_weather=True) -> pd.DataFrame:
    """Download everything up to `end` (default: tomorrow) and save it as Parquet."""
    end = end or (date.today() + timedelta(days=1))
    frame = build_dataset(start, end, include_weather)
    path = path or config.PROCESSED_DIR / "dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def load_dataset(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or config.PROCESSED_DIR / "dataset.parquet")
