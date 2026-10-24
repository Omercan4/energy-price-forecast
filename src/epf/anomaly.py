"""Find unusual price hours.

Idea: a price is unusual if it is far from what the model expected *given the grid situation*.
That is better than looking at the raw price, because a price of 150 EUR/MWh is normal at 19:00 in winter
and strange at 13:00 in a sunny week. We use the forecast residual (actual minus median forecast) and
compare it with the typical residual for the same hour of the day, using robust statistics
(median and MAD), so the outliers themselves do not distort the thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from epf import config

MAD_SCALE = 1.4826  # makes the MAD comparable to a standard deviation for normal data


def local_hour(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_convert(config.TIMEZONE).hour.to_numpy()


def residual_stats(residual: pd.Series) -> dict[int, dict[str, float]]:
    """Median and scaled MAD of the residual for each hour of the day."""
    hours = pd.Series(local_hour(residual.index), index=residual.index)
    stats = {}
    for hour, values in residual.groupby(hours):
        median = float(values.median())
        mad = float((values - median).abs().median()) * MAD_SCALE
        stats[int(hour)] = {"median": median, "mad": max(mad, 1e-6)}
    return stats


def robust_z(residual: pd.Series, stats: dict[int, dict[str, float]]) -> pd.Series:
    """How many robust standard deviations each residual is away from the typical residual of its hour."""
    hours = local_hour(residual.index)
    median = np.array([stats[h]["median"] for h in hours])
    mad = np.array([stats[h]["mad"] for h in hours])
    return pd.Series((residual.to_numpy() - median) / mad, index=residual.index)


def flag_by_residual(residual: pd.Series, stats, threshold: float = 4.0) -> pd.Series:
    return robust_z(residual, stats).abs() >= threshold


def flag_by_isolation_forest(features: pd.DataFrame, contamination: float = 0.02, seed: int = 42) -> pd.Series:
    """The classic alternative: an Isolation Forest on the raw features (no forecast model involved)."""
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    return pd.Series(model.fit_predict(features) == -1, index=features.index)


def reference_events(price: pd.Series, spike_quantile: float = 0.99) -> pd.Series:
    """Hours that a market analyst would call remarkable: negative prices and the top 1 % of prices."""
    return (price < 0) | (price >= price.quantile(spike_quantile))


def precision_recall(flags: pd.Series, events: pd.Series) -> dict[str, float]:
    flags, events = flags.astype(bool), events.astype(bool)
    true_positive = int((flags & events).sum())
    return {
        "flagged": int(flags.sum()),
        "events": int(events.sum()),
        "precision": round(true_positive / flags.sum(), 3) if flags.sum() else 0.0,
        "recall": round(true_positive / events.sum(), 3) if events.sum() else 0.0,
    }


def explain(row: pd.Series, typical: pd.DataFrame) -> str:
    """One plain sentence about why an hour is unusual, from the grid situation of that hour.

    `typical` holds the median and quartiles of the main features over the whole dataset.
    """
    parts = []
    direction = "higher" if row["price"] > row["q50"] else "lower"
    parts.append(f"price {row['price']:.0f} EUR/MWh, {abs(row['price'] - row['q50']):.0f} {direction} than forecast ({row['q50']:.0f})")
    residual_load = row["residual_load_fc"]
    if residual_load <= typical.loc["q25", "residual_load_fc"]:
        parts.append("residual load forecast is low (much wind or sun)")
    elif residual_load >= typical.loc["q75", "residual_load_fc"]:
        parts.append("residual load forecast is high (little wind or sun)")
    if row["wind_onshore_fc"] + row["wind_offshore_fc"] >= typical.loc["q75", "wind_total_fc"]:
        parts.append("wind forecast is strong")
    if row["solar_fc"] >= typical.loc["q75", "solar_fc"] and row["solar_fc"] > 0:
        parts.append("solar forecast is high")
    if row["is_holiday"] or row["weekday"] >= 5:
        parts.append("weekend or holiday (low demand)")
    return "; ".join(parts)


def typical_levels(table: pd.DataFrame) -> pd.DataFrame:
    """Quartiles of the features used in explanations."""
    data = pd.DataFrame(
        {
            "residual_load_fc": table["residual_load_fc"],
            "wind_total_fc": table["wind_onshore_fc"] + table["wind_offshore_fc"],
            "solar_fc": table["solar_fc"],
        }
    )
    return data.quantile([0.25, 0.5, 0.75]).rename(index={0.25: "q25", 0.5: "q50", 0.75: "q75"})
