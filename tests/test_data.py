import pandas as pd

from epf.data import quality, smard, weather


def test_parse_week_drops_missing_values_and_other_days():
    series = [[1_700_000_000_000, 10.0], [1_700_003_600_000, None], [1_700_007_200_000, 12.0], [1_800_000_000_000, 99.0]]

    out = smard.parse_week(series, 1_700_000_000_000, 1_700_100_000_000)

    assert out.tolist() == [10.0, 12.0]
    assert str(out.index.tz) == "UTC"


def test_weather_parsing_and_country_wide_mean():
    def place(temp):
        return pd.DataFrame(
            {"temperature_2m": temp, "wind_speed_100m": 10.0, "shortwave_radiation": 100.0},
            index=pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        )

    frames = {"north_sea": place(0.0), "berlin": place(3.0), "cologne": place(6.0), "munich": place(9.0)}
    combined = weather.combine_places(frames)

    assert combined["temp"].tolist() == [6.0, 6.0]  # the mean of the three cities
    assert list(combined.columns) == ["temp", "rad", "wind_ns", "wind_inland"]


def test_quality_report_counts_gaps_and_dst_days(frame):
    gappy = frame.drop(frame.index[10:13])

    report = quality.quality_report(gappy)

    assert report["missing_hours"] == 3
    assert report["duplicate_hours"] == 0
    assert report["negative_price_hours"] >= 0


def test_unknown_smard_series_is_rejected():
    try:
        smard.fetch_series("nonsense", "2025-01-01", "2025-01-02")
        assert False, "should have raised"
    except ValueError as error:
        assert "price" in str(error)
