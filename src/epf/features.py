"""Turn the hourly dataset into a table for a model: one row per delivery hour.

The forecasting task (the sentence that keeps us honest):
    Predict the 24 hourly day-ahead prices of day D, using only what is known at 12:00 on day D-1.
At that time, the prices of day D-1 are known (the auction for D-1 ran a day earlier), the prices of D are not.
So price features are lags of 24 hours or more. The load, wind and solar *forecasts* of the grid operators
and the weather *forecasts* for day D are known, so they are allowed. Measured values of day D are not.
"""

import holidays
import numpy as np
import pandas as pd

from epf import config

DE_HOLIDAYS = holidays.Germany()
PRICE_LAGS = (24, 48, 72, 168)  # the same local hour 1, 2, 3 and 7 days earlier (the names say hours)
CALENDAR = ["hour", "weekday", "month", "is_holiday"]
WEATHER = ["temp", "rad", "wind_ns", "wind_inland"]  # weather forecasts for day D
PRICE_HISTORY = [
    *[f"price_lag_{lag}" for lag in PRICE_LAGS],
    "price_mean_prev_day",
    "price_min_prev_day",
    "price_max_prev_day",
    "price_mean_prev_7d",
]
# Forecasts that the grid operators publish for day D. They are the strongest inputs, but they come
# out late (wind and solar forecasts for tomorrow appear in the evening, after the 12:00 auction),
# so a model that uses them is a reference, not something you can run before the auction.
GRID_FORECASTS = ["load_fc", "wind_onshore_fc", "wind_offshore_fc", "solar_fc", "residual_load_fc"]

SAFE_FEATURES = [*CALENDAR, *WEATHER, *PRICE_HISTORY]  # all known before the auction: the model we release
FULL_FEATURES = [*CALENDAR, *GRID_FORECASTS, *WEATHER, *PRICE_HISTORY]  # needs the grid operators' forecasts
FEATURES = FULL_FEATURES  # every column that training_table makes sure is present


def local_hour_lag(price: pd.Series, local: pd.DatetimeIndex, days: int) -> np.ndarray:
    """The price at the same local clock hour `days` days earlier.

    Local time is used, not "24 hours ago": after a daylight-saving change the two differ by one hour, and on a
    25-hour day the 24-hours-ago price of the last hour would lie inside the day we are forecasting.
    The repeated hour (02:00 twice) is matched by its occurrence. If the hour does not exist
    (02:00 on a 23-hour day), the hour before is used.
    """
    dates, hours = local.date, local.hour
    occurrence = pd.Series(0, index=price.index).groupby([dates, hours]).cumcount().to_numpy()
    values = {(d, h, o): v for d, h, o, v in zip(dates, hours, occurrence, price.to_numpy())}
    out = np.full(len(price), np.nan)
    for i, (d, h, o) in enumerate(zip(dates, hours, occurrence)):
        day = d - pd.Timedelta(days=days).to_pytimedelta()
        for key in ((day, h, o), (day, h, 0), (day, h - 1, 0)):
            if key in values:
                out[i] = values[key]
                break
    return out


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the feature columns to a copy of the dataset. Works on any index with hourly UTC rows."""
    out = frame.copy()
    price = out["price"]
    local = out.index.tz_convert(config.TIMEZONE)

    out["hour"] = local.hour
    out["weekday"] = local.day_of_week
    out["month"] = local.month
    out["is_holiday"] = [int(day in DE_HOLIDAYS) for day in local.date]

    out["residual_load_fc"] = out["load_fc"] - out["wind_onshore_fc"] - out["wind_offshore_fc"] - out["solar_fc"]

    for lag in PRICE_LAGS:
        out[f"price_lag_{lag}"] = local_hour_lag(price, local, lag // 24)

    # Statistics of the local day before the delivery day. Computed per local day, then joined to every hour.
    local_day = pd.Series(local.date, index=out.index)
    daily = price.groupby(local_day).agg(["mean", "min", "max"])
    previous = daily.shift(1)  # the day before: this is what is known at 12:00 on D-1
    out["price_mean_prev_day"] = local_day.map(previous["mean"]).to_numpy()
    out["price_min_prev_day"] = local_day.map(previous["min"]).to_numpy()
    out["price_max_prev_day"] = local_day.map(previous["max"]).to_numpy()
    out["price_mean_prev_7d"] = local_day.map(daily["mean"].rolling(7, min_periods=5).mean().shift(1)).to_numpy()
    return out


def training_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a price and all features of both feature sets. Used for training and for the backtest."""
    featured = add_features(frame)
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured.dropna(subset=["price", *FEATURES])
