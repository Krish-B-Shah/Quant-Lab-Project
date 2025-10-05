from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
	return returns.cov()


def ewma_covariance(returns: pd.DataFrame, lambda_: float = 0.94) -> pd.DataFrame:
	# RiskMetrics-style EWMA covariance
	ret = returns - returns.mean()
	weights = np.array([lambda_ ** i for i in range(len(ret))][::-1])
	weights = weights / weights.sum()
	X = ret.values
	w_diag = np.diag(weights)
	cov = X.T @ w_diag @ X
	return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
