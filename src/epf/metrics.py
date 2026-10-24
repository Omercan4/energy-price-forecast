"""Error measures. MAPE is not used: prices are often near zero or negative, which breaks it."""

import numpy as np
import pandas as pd


def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def smape(actual, forecast) -> float:
    """Symmetric MAPE in percent. Hours where both values are 0 are left out."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denominator = (np.abs(a) + np.abs(f)) / 2
    keep = denominator > 0
    return float(np.mean(np.abs(a - f)[keep] / denominator[keep]) * 100)


def pinball(actual, forecast, q: float) -> float:
    """Pinball loss of a quantile forecast: the standard score for quantile models (lower is better)."""
    diff = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(actual, lower, upper) -> float:
    """Share of actual values inside [lower, upper]. For an 80 % interval, about 0.80 is the goal."""
    a = np.asarray(actual)
    return float(np.mean((a >= np.asarray(lower)) & (a <= np.asarray(upper))))


def score_table(actual: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """MAE, RMSE and sMAPE for several forecasts of the same actual values."""
    rows = {name: {"MAE": mae(actual, f), "RMSE": rmse(actual, f), "sMAPE %": smape(actual, f)} for name, f in forecasts.items()}
    return pd.DataFrame(rows).T.round(2)
