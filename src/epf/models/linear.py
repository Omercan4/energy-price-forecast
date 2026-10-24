"""A plain linear model (ridge regression) on the same features: the 'is the fancy model needed?' check."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from epf.features import SAFE_FEATURES


class RidgeForecaster:
    def __init__(self, alpha: float = 10.0, features=None):
        self.features = list(features or SAFE_FEATURES)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, table: pd.DataFrame):
        self.model.fit(table[self.features], table["price"])
        return self

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"q50": self.model.predict(table[self.features])}, index=table.index)
