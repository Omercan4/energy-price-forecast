import numpy as np
import pandas as pd

from epf import features


def test_price_lags_use_only_the_past(frame):
    featured = features.add_features(frame)
    row = featured.iloc[300]

    assert row["price_lag_24"] == frame["price"].iloc[300 - 24]
    assert row["price_lag_168"] == frame["price"].iloc[300 - 168]


def test_lags_follow_the_local_clock_across_daylight_saving_changes():
    """After the clock change on 2026-10-25 (a 25-hour day), "the same hour yesterday" is 25 hours back, not 24."""
    index = pd.date_range("2026-10-20", "2026-10-28", freq="h", tz="UTC")
    frame = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    frame["load_fc"] = frame["wind_onshore_fc"] = frame["wind_offshore_fc"] = frame["solar_fc"] = 1.0
    frame["temp"] = frame["rad"] = frame["wind_ns"] = frame["wind_inland"] = 1.0

    featured = features.add_features(frame)
    local = featured.index.tz_convert("Europe/Berlin")
    # 2026-10-26 03:00 local (after the change): yesterday 03:00 local was 25 hours earlier
    row = featured[(local.day == 26) & (local.hour == 3)].iloc[0]
    yesterday = featured[(local.day == 25) & (local.hour == 3)].iloc[0]

    assert row["price_lag_24"] == yesterday["price"]


def test_the_last_hour_of_a_25_hour_day_does_not_look_into_its_own_day():
    index = pd.date_range("2026-10-20", "2026-10-26", freq="h", tz="UTC")
    frame = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    local = frame.index.tz_convert("Europe/Berlin")
    last_hour = frame.index[(local.day == 25) & (local.hour == 23)][0]
    yesterday_23 = frame.index[(local.day == 24) & (local.hour == 23)][0]

    lag = features.local_hour_lag(frame["price"], local, 1)[frame.index.get_loc(last_hour)]

    assert lag == frame.loc[yesterday_23, "price"]


def test_changing_the_price_of_the_target_day_does_not_change_its_features(frame):
    """The core no-leakage test: features of day D must not depend on the prices of day D."""
    target_day = pd.Timestamp("2025-02-10", tz="Europe/Berlin")
    start, end = target_day.tz_convert("UTC"), (target_day + pd.Timedelta(days=1)).tz_convert("UTC")
    changed = frame.copy()
    changed.loc[(changed.index >= start) & (changed.index < end), "price"] += 1000

    before = features.add_features(frame)
    after = features.add_features(changed)
    day = (before.index >= start) & (before.index < end)

    pd.testing.assert_frame_equal(before.loc[day, features.SAFE_FEATURES], after.loc[day, features.SAFE_FEATURES])
    pd.testing.assert_frame_equal(before.loc[day, features.FULL_FEATURES], after.loc[day, features.FULL_FEATURES])


def test_previous_day_statistics_are_from_the_local_day_before(frame):
    featured = features.add_features(frame)
    local = frame.index.tz_convert("Europe/Berlin")
    day = pd.Timestamp("2025-02-10").date()
    previous = pd.Timestamp("2025-02-09").date()
    expected = frame["price"][local.date == previous].mean()

    assert np.allclose(featured.loc[local.date == day, "price_mean_prev_day"], expected)


def test_the_safe_set_has_no_grid_forecasts():
    assert not set(features.SAFE_FEATURES) & set(features.GRID_FORECASTS)
    assert set(features.GRID_FORECASTS) <= set(features.FULL_FEATURES)


def test_holidays_are_flagged(frame):
    frame = frame.loc["2025-01-01":"2025-01-03"]
    featured = features.add_features(frame)
    local = featured.index.tz_convert("Europe/Berlin")

    assert featured.loc[local.day == 1, "is_holiday"].eq(1).all()  # New Year's Day
    assert featured.loc[local.day == 2, "is_holiday"].eq(0).all()


def test_training_table_drops_rows_without_history(frame):
    table = features.training_table(frame)

    assert table.index.min() >= frame.index.min() + pd.Timedelta(hours=168)
    assert table[features.FEATURES].notna().all().all()
