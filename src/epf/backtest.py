"""Rolling-origin backtest: for each test month, train on everything before it, then forecast the month.

This mimics real use (the model is retrained regularly and only knows the past) and avoids the
random train/test split that leaks the future into time series models.
"""

import pandas as pd

from epf import config
from epf.features import training_table
from epf.models.gbm import QuantileForecaster

CALIBRATION_DAYS = 60  # the last 60 days before each test month calibrate the prediction interval


def month_starts(index: pd.DatetimeIndex, months: int) -> list[pd.Timestamp]:
    """The first day (local time) of each of the last `months` months in the data."""
    local = index.tz_convert(config.TIMEZONE)
    firsts = pd.date_range(local.min().normalize(), local.max().normalize(), freq="MS", tz=config.TIMEZONE)
    complete = [m for m in firsts if m + pd.offsets.MonthBegin(1) <= local.max() + pd.Timedelta(hours=1)]
    return complete[-months:]


def run_backtest(frame: pd.DataFrame, months: int = 12, min_train_days: int = 365, make_model=None, extra_models=None):
    """Forecast each of the last `months` months with a model trained only on earlier data.

    Returns a DataFrame with the actual price, the naive baselines and the model's q10/q50/q90.
    extra_models: {name: make_model} for more point forecasts (their q50 becomes a column with that name).
    """
    make_model = make_model or (lambda: QuantileForecaster(calibration_days=CALIBRATION_DAYS))
    table = training_table(frame)
    pieces = []
    for start in month_starts(table.index, months):
        end = start + pd.offsets.MonthBegin(1)
        train = table[table.index < start.tz_convert("UTC")]
        test = table[(table.index >= start.tz_convert("UTC")) & (table.index < end.tz_convert("UTC"))]
        if len(train) < min_train_days * 24 or test.empty:
            continue
        forecast = make_model().fit(train).predict(test)
        part = pd.DataFrame({"price": test["price"], "naive_24": test["price_lag_24"], "naive_168": test["price_lag_168"]})
        for name, make in (extra_models or {}).items():
            part[name] = make().fit(train).predict(test)["q50"]
        pieces.append(part.join(forecast).assign(train_end=start.tz_convert("UTC")))
    return pd.concat(pieces)
