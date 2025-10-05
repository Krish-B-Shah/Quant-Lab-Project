from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.linalg import inv


def mean_variance_optimize(
	returns: pd.DataFrame,
	risk_aversion: float = 1.0,
	long_only: bool = True,
	weights_sum_to_one: bool = True,
	epsilon: float = 1e-8,
) -> pd.Series:
	mu = returns.mean()
	cov = returns.cov() + np.eye(len(mu)) * epsilon
	inv_cov = pd.DataFrame(inv(cov.values), index=cov.index, columns=cov.columns)
	raw = inv_cov @ mu
	w = raw / raw.sum()
	if long_only:
		w = w.clip(lower=0.0)
		if w.sum() == 0:
			w = pd.Series(np.full_like(w, 1.0 / len(w)), index=w.index)
		else:
			w = w / w.sum()
	if weights_sum_to_one:
		w = w / w.sum()
	return w
