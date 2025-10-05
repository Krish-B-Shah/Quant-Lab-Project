from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class ReturnForecaster:
	alpha: float = 1.0
	model: Optional[Ridge] = None

	def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReturnForecaster":
		self.model = Ridge(alpha=self.alpha)
		self.model.fit(X.values, y.values)
		return self

	def predict(self, X: pd.DataFrame) -> pd.Series:
		if self.model is None:
			raise RuntimeError("Model not fit.")
		pred = self.model.predict(X.values)
		return pd.Series(pred, index=X.index)
