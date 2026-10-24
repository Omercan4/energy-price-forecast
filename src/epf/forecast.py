"""Forecast tomorrow's 24 hourly prices and write the result as JSON for the API and the web page."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from epf import config
from epf.data import smard, weather
from epf.features import add_features

HISTORY_DAYS = 14  # how much price history the lag features need (the longest lag is a week)
WEATHER_COLUMNS = ["temp", "rad", "wind_ns", "wind_inland"]


class InputsNotReady(RuntimeError):
    """The inputs for the target day are not complete yet (for example the weather forecast)."""


def load_model(path: Path | None = None):
    return joblib.load(path or config.MODEL_DIR / "champion.joblib")


def day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start (inclusive) and end (exclusive) of a local day, in UTC."""
    start = pd.Timestamp(day, tz=config.TIMEZONE).tz_convert("UTC")
    end = pd.Timestamp(day + timedelta(days=1), tz=config.TIMEZONE).tz_convert("UTC")  # not start + 24 h: DST days differ
    return start, end


def collect_inputs(target: date) -> pd.DataFrame:
    """Everything known before the day-ahead auction of `target`: past prices, grid-operator forecasts, weather forecast."""
    first = target - timedelta(days=HISTORY_DAYS)
    # SMARD days are UTC days; ask for one day more on each side to cover the local day completely.
    series = [smard.fetch_series(name, first, target + timedelta(days=1)) for name in config.SMARD_SERIES]
    frame = pd.concat(series, axis=1).sort_index()
    start, end = day_bounds(target)

    # The price of the target day must not be used, even if it has been published already.
    frame.loc[frame.index >= start, "price"] = float("nan")
    frame = frame[frame.index < end]

    weather_fc = weather.fetch_weather(target - timedelta(days=1), target + timedelta(days=1), forecast=True)
    frame = frame.join(weather_fc, how="left")
    target_rows = frame[frame.index >= start]
    if len(target_rows) < 23:
        raise InputsNotReady(f"Hourly rows for {target} are not complete yet")
    missing = [c for c in WEATHER_COLUMNS if target_rows[c].notna().sum() < 23]
    if missing:
        raise InputsNotReady(f"Weather forecast for {target} is not complete yet: {missing}")
    return frame


def forecast_day(target: date, model=None, inputs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The q10, q50 and q90 forecast for each hour of `target`, with the main inputs next to it."""
    model = model or load_model()
    frame = collect_inputs(target) if inputs is None else inputs
    featured = add_features(frame)
    start, end = day_bounds(target)
    rows = featured[(featured.index >= start) & (featured.index < end)]
    rows = rows.dropna(subset=model.features)
    if len(rows) < 23:
        raise InputsNotReady(f"Only {len(rows)} complete rows for {target}")
    result = model.predict(rows)
    return result.join(rows[["temp", "rad", "wind_ns", "wind_inland"]])


def to_payload(target: date, forecast: pd.DataFrame, metadata: dict) -> dict:
    local = forecast.index.tz_convert(config.TIMEZONE)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_day": target.isoformat(),
        "unit": "EUR/MWh",
        "model": {"trained_until": metadata.get("trained_until"), "backtest_mae": metadata.get("backtest_mae")},
        "note": "Forecast made with information known at 12:00 the day before. Not trading advice.",
        "hours": [
            {
                "time": t.isoformat(),
                "hour": int(t.hour),
                "q10": round(float(r["q10"]), 2),
                "q50": round(float(r["q50"]), 2),
                "q90": round(float(r["q90"]), 2),
                "temp": round(float(r["temp"]), 1),
                "wind_ns": round(float(r["wind_ns"]), 1),
                "rad": round(float(r["rad"]), 0),
            }
            for t, (_, r) in zip(local, forecast.iterrows())
        ],
        "summary": {
            "mean_q50": round(float(forecast["q50"].mean()), 2),
            "min_q50": round(float(forecast["q50"].min()), 2),
            "max_q50": round(float(forecast["q50"].max()), 2),
            "cheapest_hour": int(local[int(forecast["q50"].to_numpy().argmin())].hour),
            "most_expensive_hour": int(local[int(forecast["q50"].to_numpy().argmax())].hour),
        },
    }


def write_payload(payload: dict, docs_dir: Path | None = None) -> Path:
    docs_dir = docs_dir or config.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "forecasts").mkdir(exist_ok=True)
    text = json.dumps(payload, indent=1)
    (docs_dir / "forecast.json").write_text(text)  # the newest forecast, for the page
    (docs_dir / "forecasts" / f"{payload['target_day']}.json").write_text(text)  # kept for the daily monitoring
    return docs_dir / "forecast.json"
