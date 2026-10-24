import numpy as np
import pandas as pd

from epf import anomaly, monitor


def residual_series(n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-03-01", periods=n_days * 24, freq="h", tz="UTC")
    return pd.Series(rng.normal(0, 5, len(index)), index=index)


def test_an_obvious_outlier_is_flagged_and_normal_hours_are_not():
    residual = residual_series()
    stats = anomaly.residual_stats(residual)
    residual.iloc[200] = 120.0

    flags = anomaly.flag_by_residual(residual, stats, threshold=4.0)

    assert flags.iloc[200]
    assert flags.mean() < 0.01


def test_robust_stats_are_not_pulled_by_outliers():
    residual = residual_series()
    clean = anomaly.residual_stats(residual)
    residual.iloc[::50] = 500.0  # many extreme values
    dirty = anomaly.residual_stats(residual)

    assert abs(clean[12]["mad"] - dirty[12]["mad"]) < 2


def test_precision_and_recall():
    flags = pd.Series([True, True, False, False])
    events = pd.Series([True, False, True, False])

    assert anomaly.precision_recall(flags, events) == {"flagged": 2, "events": 2, "precision": 0.5, "recall": 0.5}


def test_reference_events_are_negative_prices_and_spikes():
    price = pd.Series([50.0] * 98 + [-5.0, 900.0])

    assert anomaly.reference_events(price).sum() == 2


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shifted_one():
    rng = np.random.default_rng(0)
    expected = pd.Series(rng.normal(0, 1, 5000))

    assert monitor.psi(expected, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert monitor.psi(expected, pd.Series(rng.normal(2, 1, 5000))) > 0.25


def test_rolling_health_raises_an_alert_when_the_error_doubles():
    records = [{"day": f"2026-09-{d:02d}", "mae": 40.0, "mae_naive_168": 45.0, "coverage_80": 0.8} for d in range(1, 11)]

    assert monitor.rolling_health(records, backtest_mae=19.0)["status"] == "alert"
    assert monitor.rolling_health(records, backtest_mae=30.0)["status"] == "ok"
    assert monitor.rolling_health([], backtest_mae=19.0)["status"] == "no data yet"


def test_day_record_scores_model_and_baseline():
    index = pd.date_range("2026-09-01", periods=24, freq="h", tz="UTC")
    forecast = pd.DataFrame({"q10": 40.0, "q50": 50.0, "q90": 60.0}, index=index)
    actual = pd.Series(52.0, index=index)
    naive = pd.Series(70.0, index=index)

    record = monitor.day_record("2026-09-01", forecast, actual, naive)

    assert record["mae"] == 2.0 and record["mae_naive_168"] == 18.0 and record["coverage_80"] == 1.0
