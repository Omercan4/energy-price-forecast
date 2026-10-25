"""Classical statistical baselines, named as in the forecasting literature (Hyndman and Athanasopoulos).

Every forecast for day D uses only the prices before D (what is known at 12:00 on D-1), refitted for each day on the
last 28 days. That is the same information the machine learning model has, so the comparison is fair.

  seasonal naive, m      the value one season ago: m = 24 is "same hour yesterday", m = 168 "same hour last week"
  seasonal average       the mean of the same hour over the last k weeks (a seasonal moving average)
  Holt-Winters           exponential smoothing with additive seasonality, ETS(A,N,A), m = 24
  SARIMA(1,0,0)(1,1,0)24 an autoregressive model: AR(1), plus a seasonal AR(1) on the 24-hour seasonal difference
  SARIMA(1,0,1)(0,1,1)24 ARMA(1,1) plus a seasonal MA(1) on the 24-hour seasonal difference ("airline-type" model)
  SARIMAX                the same SARIMA(1,0,1)(0,1,1)24 plus exogenous regressors: the four weather forecasts of the day
                         (temperature, solar radiation, wind at the North Sea and inland). Same information as ridge and LightGBM.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from epf import config
from epf.features import local_hour_lag

HISTORY_HOURS = 28 * 24
SARIMA_AR = ((1, 0, 0), (1, 1, 0, 24))
SARIMA_ARMA = ((1, 0, 1), (0, 1, 1, 24))
EXOG = ["temp", "rad", "wind_ns", "wind_inland"]
NAMES = ["sma_4w", "ets", "sarima_ar", "sarima_arma", "sarimax"]


def seasonal_average(price: pd.Series, weeks: int = 4) -> pd.Series:
    """Mean of the price at the same local hour 1..weeks weeks earlier (the seasonal moving average)."""
    local = price.index.tz_convert(config.TIMEZONE)
    lags = np.column_stack([local_hour_lag(price, local, 7 * k) for k in range(1, weeks + 1)])
    return pd.Series(np.nanmean(lags, axis=1), index=price.index)


def forecast_one_day(history: np.ndarray, horizon: int, exog_history: np.ndarray | None = None, exog_future: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """ETS, the two SARIMA models and (with exogenous inputs) SARIMAX for the next `horizon` hours.
    A model that fails returns NaN."""
    # imported here, not at the top: the daily forecast job does not install statsmodels, only the research code needs it
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(history, trend=None, seasonal="add", seasonal_periods=24, initialization_method="estimated").fit()
            out["ets"] = fit.forecast(horizon)
        except Exception:
            out["ets"] = np.full(horizon, np.nan)
        for name, (order, seasonal) in {"sarima_ar": SARIMA_AR, "sarima_arma": SARIMA_ARMA}.items():
            try:
                fit = SARIMAX(history, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(
                    disp=False, maxiter=60
                )
                out[name] = fit.forecast(horizon)
            except Exception:
                out[name] = np.full(horizon, np.nan)
        if exog_history is not None and exog_future is not None:
            try:
                mean, std = exog_history.mean(axis=0), exog_history.std(axis=0) + 1e-9  # standardised, so the optimiser converges
                fit = SARIMAX(history, exog=(exog_history - mean) / std, order=SARIMA_ARMA[0], seasonal_order=SARIMA_ARMA[1],
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
                out["sarimax"] = fit.forecast(horizon, exog=(exog_future - mean) / std)
            except Exception:
                out["sarimax"] = np.full(horizon, np.nan)
    return out


def classical_backtest(price: pd.Series, test_index: pd.DatetimeIndex, n_jobs: int = -1, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecasts of the classical models for every hour in `test_index`, one fit per local day.
    exog: the weather forecasts (columns EXOG), needed for SARIMAX."""
    price = price.dropna()
    local_days = pd.Series(test_index.tz_convert(config.TIMEZONE).date, index=test_index)
    jobs = []
    for _, hours in local_days.groupby(local_days):
        start = hours.index.min()
        past = price[price.index < start].iloc[-HISTORY_HOURS:]
        exog_history = exog_future = None
        if exog is not None:
            x = exog[EXOG]
            exog_history = x.reindex(past.index).to_numpy(dtype=float)
            exog_future = x.reindex(hours.index).to_numpy(dtype=float)
            if np.isnan(exog_history).any() or np.isnan(exog_future).any():
                exog_history = exog_future = None  # an incomplete weather forecast: SARIMAX falls back below
        jobs.append((hours.index, past.to_numpy(), len(hours), exog_history, exog_future))
    results = Parallel(n_jobs=n_jobs)(delayed(forecast_one_day)(h, n, eh, ef) for _, h, n, eh, ef in jobs)
    frame = pd.DataFrame(index=test_index, columns=["ets", "sarima_ar", "sarima_arma", "sarimax"], dtype=float)
    for (idx, *_), res in zip(jobs, results):
        for name, values in res.items():
            frame.loc[idx, name] = values
    frame["sma_4w"] = seasonal_average(price).reindex(test_index).to_numpy()
    return frame[NAMES]
