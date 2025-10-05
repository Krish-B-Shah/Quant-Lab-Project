from pathlib import Path
from backtesting.son_stub import son_backtest_stub

"""The testing file is in the data/processed directory"""
def main():
    # Pick one of your processed CSVs
    csv_path = Path("data/processed/SPY_sontest.csv")

    # Run the stub
    stats = son_backtest_stub(csv_path, price_col="Adj Close", cost_bps=1.0)

    # Print results
    print("=== Backtest Stub Results ===")
    for k, v in stats.items():
        print(f"{k:15s}: {v}")

if __name__ == "__main__":
    main()
