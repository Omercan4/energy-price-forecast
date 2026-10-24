"""Watch the model after it is released: is it still as good as in the backtest?

Every day we compare yesterday's forecast with the prices that were published afterwards, keep a rolling error,
and check whether the inputs still look like the training data (PSI, the population stability index).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from epf import metrics

ALERT_FACTOR = 1.5  # alert when the rolling MAE is 1.5 times the backtest MAE
ROLLING_DAYS = 30


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index: how much has the distribution of a feature moved?
    Rule of thumb: below 0.1 stable, 0.1 to 0.25 some drift, above 0.25 large drift."""
    expected, actual = expected.dropna(), actual.dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def day_record(day: str, forecast: pd.DataFrame, actual: pd.Series, naive: pd.Series) -> dict:
    """The record of one finished day: errors of the model and of the weekly-naive baseline."""
    joined = forecast.join(actual.rename("price"), how="inner").join(naive.rename("naive_168"), how="left")
    return {
        "day": day,
        "hours": len(joined),
        "mae": round(metrics.mae(joined["price"], joined["q50"]), 2),
        "mae_naive_168": round(metrics.mae(joined["price"], joined["naive_168"]), 2) if joined["naive_168"].notna().all() else None,
        "coverage_80": round(metrics.coverage(joined["price"], joined["q10"], joined["q90"]), 3),
        "hourly": [
            {"time": t.isoformat(), "price": round(float(r["price"]), 2), "q10": round(float(r["q10"]), 2),
             "q50": round(float(r["q50"]), 2), "q90": round(float(r["q90"]), 2)}
            for t, r in joined.iterrows()
        ],
    }


def rolling_health(records: list[dict], backtest_mae: float, days: int = ROLLING_DAYS) -> dict:
    """The model-health box on the page."""
    recent = sorted(records, key=lambda r: r["day"])[-days:]
    if not recent:
        return {"status": "no data yet", "days": 0}
    mae = float(np.mean([r["mae"] for r in recent]))
    naive = [r["mae_naive_168"] for r in recent if r["mae_naive_168"] is not None]
    return {
        "status": "alert" if mae > ALERT_FACTOR * backtest_mae else "ok",
        "days": len(recent),
        "rolling_mae": round(mae, 2),
        "rolling_mae_naive_168": round(float(np.mean(naive)), 2) if naive else None,
        "backtest_mae": round(backtest_mae, 2),
        "coverage_80": round(float(np.mean([r["coverage_80"] for r in recent])), 3),
        "alert_threshold": round(ALERT_FACTOR * backtest_mae, 2),
    }


def load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def save_history(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(records, key=lambda r: r["day"]), indent=1))
