import numpy as np
import pandas as pd
import pytest


def make_frame(days=60, seed=0, start="2025-01-01"):
    """A made-up hourly dataset (UTC) shaped like the real one: a daily price pattern driven by residual load."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    local = index.tz_convert("Europe/Berlin")
    hour = local.hour.to_numpy()
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 20000 * rng.uniform(0.6, 1.0, len(index))
    wind_on = rng.uniform(2000, 30000, len(index))
    wind_off = rng.uniform(0, 5000, len(index))
    load = 45000 + 8000 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 800, len(index))
    residual = load - solar - wind_on - wind_off
    price = 60 + residual / 400 + rng.normal(0, 4, len(index))
    return pd.DataFrame(
        {
            "price": price,
            "load_fc": load,
            "wind_onshore_fc": wind_on,
            "wind_offshore_fc": wind_off,
            "solar_fc": solar,
            "temp": 10 + rng.normal(0, 3, len(index)),
            "rad": solar / 30,
            "wind_ns": wind_on / 1500,
            "wind_inland": wind_on / 2000,
        },
        index=pd.DatetimeIndex(index, name="time"),
    )


@pytest.fixture
def frame():
    return make_frame()
