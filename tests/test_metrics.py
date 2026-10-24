import numpy as np

from epf import metrics


def test_basic_errors():
    actual, forecast = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 30.0])

    assert metrics.mae(actual, forecast) == (2 + 2 + 0) / 3
    assert round(metrics.rmse(actual, forecast), 4) == round(np.sqrt((4 + 4 + 0) / 3), 4)


def test_smape_survives_zero_and_negative_prices():
    assert metrics.smape([0.0, -10.0], [0.0, -12.0]) > 0  # the (0, 0) hour is skipped, no division by zero


def test_pinball_loss_is_asymmetric():
    # forecasting too low costs q per unit, too high costs (1 - q) per unit
    assert metrics.pinball([10.0], [8.0], 0.9) == 0.9 * 2
    assert round(metrics.pinball([10.0], [12.0], 0.9), 4) == round(0.1 * 2, 4)


def test_coverage():
    assert metrics.coverage([1, 5, 9], [0, 0, 0], [6, 6, 6]) == 2 / 3
