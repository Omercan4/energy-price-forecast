"""Data quality checks. They return numbers, so the README and the tests can both use them."""

import pandas as pd

from epf import config


def quality_report(frame: pd.DataFrame) -> dict:
    """Simple checks on the hourly dataset (UTC index)."""
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h", tz="UTC")
    local = frame.index.tz_convert(config.TIMEZONE)
    hours_per_day = pd.Series(1, index=local).groupby(local.date).sum()
    price = frame["price"].dropna()
    return {
        "rows": len(frame),
        "first": frame.index.min().isoformat(),
        "last": frame.index.max().isoformat(),
        "missing_hours": int(len(expected.difference(frame.index))),
        "duplicate_hours": int(frame.index.duplicated().sum()),
        "missing_share": {column: round(float(frame[column].isna().mean()), 4) for column in frame.columns},
        "dst_short_days": int((hours_per_day == 23).sum()),
        "dst_long_days": int((hours_per_day == 25).sum()),
        "negative_price_hours": int((price < 0).sum()),
        "price_min": float(price.min()),
        "price_max": float(price.max()),
        "price_mean": round(float(price.mean()), 2),
    }
