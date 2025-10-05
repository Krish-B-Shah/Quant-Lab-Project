from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity: pd.Series) -> plt.Axes:
	ax = equity.plot(title="Equity Curve")
	ax.set_xlabel("Date")
	ax.set_ylabel("Equity")
	return ax


def plot_drawdown(equity: pd.Series) -> plt.Axes:
	roll_max = equity.cummax()
	drawdown = equity / roll_max - 1.0
	ax = drawdown.plot(title="Drawdown")
	ax.set_xlabel("Date")
	ax.set_ylabel("Drawdown")
	return ax
