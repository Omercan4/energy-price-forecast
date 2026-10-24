"""Hourly electricity market data from SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0).

SMARD stores every series in weekly files. An index file lists the start time of each file
(milliseconds since 1970). We download the files we need in parallel and cache them on disk.
Old weeks never change, so they are cached forever; the newest files are refreshed.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from epf import config
from epf.data.http import cached_json

BASE_URL = "https://www.smard.de/app/chart_data"
SOURCE = "Bundesnetzagentur | SMARD.de (CC BY 4.0)"
DAY_MS = 24 * 3600 * 1000
FRESH_FILES = 2  # the newest weekly files may still change
FRESH_HOURS = 2


def to_millis(day: date | str) -> int:
    """Midnight UTC of a day, in milliseconds since 1970."""
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def parse_week(series, start_ms, end_ms):
    """[[timestamp_ms, value], ...] -> pandas Series with a UTC index. Missing values are dropped."""
    rows = [(t, v) for t, v in series if v is not None and start_ms <= t < end_ms]
    index = pd.to_datetime([t for t, _ in rows], unit="ms", utc=True)
    return pd.Series([v for _, v in rows], index=index, dtype="float64")


def fetch_series(name: str, start: date | str, end: date | str) -> pd.Series:
    """Hourly values of one series from the start day to the end day (both included, UTC days)."""
    if name not in config.SMARD_SERIES:
        raise ValueError(f"Unknown series '{name}'. Choose from: {', '.join(config.SMARD_SERIES)}")
    filter_id, region = config.SMARD_SERIES[name]
    start_ms, end_ms = to_millis(start), to_millis(end) + DAY_MS

    base = f"{BASE_URL}/{filter_id}/{region}"
    cache = config.RAW_DIR / "smard"
    index = cached_json(f"{base}/index_hour.json", cache / f"{filter_id}_{region}_index.json", max_age_hours=FRESH_HOURS)
    stamps = index["timestamps"]
    newest = set(stamps[-FRESH_FILES:])
    wanted = [t for t in stamps if t < end_ms and t + 8 * DAY_MS > start_ms]

    def load(stamp):
        path = cache / f"{filter_id}_{region}_{stamp}.json"
        age = FRESH_HOURS if stamp in newest else None
        data = cached_json(f"{base}/{filter_id}_{region}_hour_{stamp}.json", path, max_age_hours=age)
        return parse_week(data["series"], start_ms, end_ms)

    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(load, wanted))
    if not parts:
        return pd.Series(dtype="float64", name=name)
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = name
    return series
