"""LightGBM quantile models: one model per quantile gives a forecast with an interval.

The raw quantile models are usually too confident (their 80 % interval contains fewer than 80 % of the
actual prices). We fix this with conformalized quantile regression (Romano et al., 2019): the last
`calibration_days` of the training data are held back, and we measure by how much the interval would have
had to be wider to contain 80 % of those prices. That amount is added to the interval of every forecast.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from epf import config
from epf.features import SAFE_FEATURES

PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}


class QuantileForecaster:
    """Fits one LightGBM model per quantile (default 10 %, 50 %, 90 %).

    calibration_days > 0 turns on the interval correction described above (needs the quantiles 0.1 and 0.9).
    """

    def __init__(self, quantiles=config.QUANTILES, params=None, features=None, calibration_days=0, target_coverage=0.8):
        self.features = list(features or SAFE_FEATURES)
        self.quantiles = tuple(quantiles)
        self.params = {**PARAMS, **(params or {})}
        self.calibration_days = calibration_days
        self.target_coverage = target_coverage
        self.offset = 0.0
        self.models = {}

    def fit(self, table: pd.DataFrame):
        table = table.sort_index()
        if self.calibration_days:
            cut = table.index.max() - pd.Timedelta(days=self.calibration_days)
            train, calibration = table[table.index <= cut], table[table.index > cut]
        else:
            train, calibration = table, None

        for q in self.quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            model.fit(train[self.features], train["price"])
            self.models[q] = model

        if calibration is not None and len(calibration):
            raw = self._raw(calibration)
            # how far the actual price is outside the interval (negative if it is inside)
            score = np.maximum(raw["q10"] - calibration["price"], calibration["price"] - raw["q90"]).to_numpy()
            n = len(score)
            level = min(1.0, self.target_coverage * (1 + 1 / n))  # the finite-sample correction of conformal prediction
            self.offset = float(np.quantile(score, level))
        return self

    def _raw(self, table: pd.DataFrame) -> pd.DataFrame:
        raw = np.column_stack([self.models[q].predict(table[self.features]) for q in self.quantiles])
        raw.sort(axis=1)  # the quantile models never cross
        names = [f"q{int(q * 100)}" for q in self.quantiles]
        return pd.DataFrame(raw, index=table.index, columns=names)

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """Columns q10, q50, q90. With calibration the interval is widened (or narrowed) by the learned offset.
        The raw interval stays available as q10_raw and q90_raw."""
        out = self._raw(table)
        if self.offset and {"q10", "q90"} <= set(out.columns):
            out["q10_raw"], out["q90_raw"] = out["q10"], out["q90"]
            out["q10"], out["q90"] = out["q10"] - self.offset, out["q90"] + self.offset
        return out
