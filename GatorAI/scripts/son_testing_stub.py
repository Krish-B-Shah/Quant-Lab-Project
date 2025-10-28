"""Simple tester for Strategy classes.

This script is intentionally minimal: it creates a tiny synthetic price
DataFrame, instantiates each strategy, calls generate_signals() and
allocate(), and prints concise summaries. Drop this into
`GatorAI/scripts/son_testing_stub.py` and run it from the repo root with
the project's venv active.
"""

from pathlib import Path
import sys
import pandas as pd

# Ensure local package sources are importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtesting.strategy import (
    EqualWeightStrategy,
    MomentumStrategy,
    VolatilityWeightedStrategy,
    MeanReversionStrategy,
)
from backtesting.strategy import StrategyConfig


def make_sample_prices():
    # 6 business days of simple synthetic prices for 3 tickers
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    data = {
        "SPY": [100.0, 101.0, 100.5, 101.5, 102.0, 101.8],
        "QQQ": [200.0, 202.0, 201.0, 203.0, 204.0, 203.5],
        "IWM": [50.0, 50.5, 50.3, 50.8, 51.0, 50.9],
    }
    return pd.DataFrame(data, index=idx)


def summarize(df: pd.DataFrame, name: str, n: int = 5):
    print(f"\n--- {name} (shape={df.shape}) ---")
    print(df.head(n).round(6))
    print()


def main():
    prices = make_sample_prices()

    strategies = [
        EqualWeightStrategy(),
        MomentumStrategy(config=StrategyConfig(params={"lookback": 1, "long_only": True})),
        VolatilityWeightedStrategy(config=StrategyConfig(params={"vol_window": 2, "long_only": True})),
        MeanReversionStrategy(config=StrategyConfig(params={"lookback": 2, "long_only": True})),
    ]

    for s in strategies:
        print(f"\n=== Strategy: {s.name} ===")
        sig = s.generate_signals(prices)
        summarize(sig, "signals")
        w = s.allocate(sig, prices)
        summarize(w, "weights")
        # quick checks
        print("weight row sums:", w.sum(axis=1).round(6).tolist())


if __name__ == "__main__":
    main()
