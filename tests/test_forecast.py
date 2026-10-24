from datetime import date

import pandas as pd
import pytest

from epf import config, forecast
from epf.features import training_table
from epf.models.gbm import QuantileForecaster
from tests.conftest import make_frame

FAST = {"n_estimators": 40, "num_leaves": 8, "min_child_samples": 20}


def test_day_bounds_handle_daylight_saving():
    start, end = forecast.day_bounds(date(2026, 10, 25))  # clocks go back: this local day has 25 hours

    assert (end - start) == pd.Timedelta(hours=25)
    start, end = forecast.day_bounds(date(2026, 3, 29))  # clocks go forward: 23 hours
    assert (end - start) == pd.Timedelta(hours=23)


def test_forecast_for_a_day_whose_price_is_unknown():
    frame = make_frame(days=420, seed=3)
    model = QuantileForecaster(params=FAST, calibration_days=30).fit(training_table(frame.iloc[: -24 * 3]))
    target = date(2026, 2, 20)
    local = frame.index.tz_convert(config.TIMEZONE)
    frame = frame[local.date <= target]
    frame.loc[frame.index.tz_convert(config.TIMEZONE).date == target, "price"] = float("nan")  # as on the morning of the day before

    result = forecast.forecast_day(target, model=model, inputs=frame)

    assert len(result) == 24
    assert (result["q10"] <= result["q50"]).all()


def test_the_payload_has_24_hours_and_a_summary():
    index = pd.date_range("2026-10-01", periods=24, freq="h", tz="UTC")
    result = pd.DataFrame({"q10": 10.0, "q50": 20.0, "q90": 30.0, "temp": 5.0, "rad": 0.0, "wind_ns": 3.0, "wind_inland": 2.0}, index=index)
    result.iloc[5, result.columns.get_loc("q50")] = -4.0

    payload = forecast.to_payload(date(2026, 10, 1), result, {"trained_until": "x", "backtest_mae": 19.0})

    assert len(payload["hours"]) == 24
    assert payload["summary"]["min_q50"] == -4.0
    assert payload["note"].endswith("Not trading advice.")


def test_a_missing_weather_forecast_stops_the_job_with_a_clear_error(monkeypatch):
    from epf.data import weather

    frame = make_frame(days=30, seed=4)
    monkeypatch.setattr(forecast.smard, "fetch_series", lambda name, a, b: frame[name] if name in frame else frame["price"])
    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: pd.DataFrame(columns=["temp", "rad", "wind_ns", "wind_inland"]))
    monkeypatch.setattr(forecast.weather, "fetch_weather", weather.fetch_weather)

    with pytest.raises(forecast.InputsNotReady):
        forecast.collect_inputs(frame.index[-1].date())
