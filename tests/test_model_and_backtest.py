import numpy as np
import pandas as pd

from epf import features
from epf.backtest import month_starts, run_backtest
from epf.models.gbm import QuantileForecaster
from tests.conftest import make_frame

FAST = {"n_estimators": 40, "num_leaves": 8, "min_child_samples": 20}


def table_for(days=420):
    return features.training_table(make_frame(days=days, seed=1))


def test_quantiles_do_not_cross():
    table = table_for(120)
    out = QuantileForecaster(params=FAST).fit(table).predict(table)

    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()


def test_the_model_learns_the_price_pattern_better_than_last_week():
    table = table_for(200)
    train, test = table.iloc[: -24 * 20], table.iloc[-24 * 20 :]
    out = QuantileForecaster(params=FAST).fit(train).predict(test)

    model_error = (test["price"] - out["q50"]).abs().mean()
    naive_error = (test["price"] - test["price_lag_168"]).abs().mean()
    assert model_error < naive_error


def test_calibration_makes_the_interval_wider_when_it_was_too_narrow():
    table = table_for(200)
    calibrated = QuantileForecaster(params={**FAST, "n_estimators": 5}, calibration_days=40).fit(table)  # a weak model: wide errors

    assert calibrated.offset > 0
    raw_width = (calibrated.predict(table)["q90_raw"] - calibrated.predict(table)["q10_raw"]).mean()
    new_width = (calibrated.predict(table)["q90"] - calibrated.predict(table)["q10"]).mean()
    assert new_width > raw_width


def test_calibrated_coverage_is_close_to_the_target():
    table = table_for(260)
    train, test = table.iloc[: -24 * 40], table.iloc[-24 * 40 :]
    model = QuantileForecaster(params=FAST, calibration_days=50).fit(train)
    out = model.predict(test)
    covered = ((test["price"] >= out["q10"]) & (test["price"] <= out["q90"])).mean()

    assert 0.7 <= covered <= 0.95


def test_backtest_trains_only_on_the_past():
    frame = make_frame(days=500, seed=2)
    result = run_backtest(
        frame, months=3, min_train_days=100,
        make_model=lambda: QuantileForecaster(params=FAST, calibration_days=30),
    )

    assert not result.empty
    assert (result.index >= result["train_end"]).all()  # every forecast is after the end of its training data
    assert result["train_end"].nunique() == 3
    assert {"price", "naive_24", "naive_168", "q10", "q50", "q90"} <= set(result.columns)


def test_month_starts_returns_only_complete_months():
    index = pd.date_range("2025-01-01", "2025-06-15", freq="h", tz="UTC")
    starts = month_starts(index, 3)

    assert [s.month for s in starts] == [3, 4, 5]


def test_predictions_use_only_the_chosen_features():
    table = table_for(120)
    model = QuantileForecaster(params=FAST).fit(table)
    changed = table.copy()
    changed["load_fc"] = changed["load_fc"] * 5  # a grid forecast is not part of the safe feature set

    np.testing.assert_allclose(model.predict(table)["q50"], model.predict(changed)["q50"])
