"""Simple models every serious model has to beat."""

import pandas as pd


def seasonal_naive(table: pd.DataFrame, lag: int) -> pd.Series:
    """The price `lag` hours ago. lag=24: same hour yesterday. lag=168: same hour last week."""
    return table[f"price_lag_{lag}"]
