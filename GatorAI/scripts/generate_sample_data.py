from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "examples"
EXAMPLES.mkdir(parents=True, exist_ok=True)

idx = pd.date_range("2020-01-01", periods=30, freq="B")
np.random.seed(42)

for t in ["SPY", "QQQ", "IWM"]:
	ret = pd.Series(np.random.normal(0.0005, 0.01, size=len(idx)), index=idx, name=t)
	ret.to_csv(EXAMPLES / f"{t}_returns.csv", header=True)

print(f"Wrote sample returns to {EXAMPLES}")
