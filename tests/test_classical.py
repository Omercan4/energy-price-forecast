import numpy as np
import pandas as pd

from epf.models import classical


def periodic_prices(days=60, seed=0, start="2025-05-01"):
    """A clean daily pattern plus a little noise, so any sensible seasonal model should find it."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")  # May to June: no clock change in the range
    pattern = 50 + 30 * np.sin(np.arange(len(index)) / 24 * 2 * np.pi)
    return pd.Series(pattern + rng.normal(0, 1, len(index)), index=index)


def test_seasonal_average_is_the_mean_of_the_same_hour_in_the_previous_weeks():
    price = periodic_prices(days=50)
    out = classical.seasonal_average(price, weeks=4)
    t = price.index[-1]
    expected = np.mean([price[t - pd.Timedelta(days=7 * k)] for k in range(1, 5)])

    assert abs(out[t] - expected) < 1e-9


def test_the_seasonal_average_has_no_value_without_history():
    price = periodic_prices(days=10)

    assert np.isnan(classical.seasonal_average(price, weeks=4).iloc[0])


def test_each_classical_model_finds_a_clean_daily_pattern():
    price = periodic_prices(days=40)
    history, truth = price.iloc[: -24].to_numpy()[-classical.HISTORY_HOURS:], price.iloc[-24:].to_numpy()

    out = classical.forecast_one_day(history, 24)

    for name in ("ets", "sarima_ar", "sarima_arma"):
        assert len(out[name]) == 24
        assert np.abs(out[name] - truth).mean() < 5, name  # the pattern has an amplitude of 30


def test_the_backtest_forecasts_only_from_prices_before_each_day():
    price = periodic_prices(days=45)
    test_index = price.index[-48:]
    changed = price.copy()
    changed.loc[test_index] += 1000  # the prices of the test days must not influence their own forecasts

    a = classical.classical_backtest(price, test_index, n_jobs=1)
    b = classical.classical_backtest(changed, test_index, n_jobs=1)

    local_dates = test_index.tz_convert("Europe/Berlin").date
    first_day = test_index[local_dates == local_dates[0]]  # the first LOCAL day is forecast from history only, so it cannot change
    pd.testing.assert_frame_equal(a.loc[first_day, ["ets", "sarima_ar", "sarima_arma"]], b.loc[first_day, ["ets", "sarima_ar", "sarima_arma"]])
    assert list(a.columns) == classical.NAMES
