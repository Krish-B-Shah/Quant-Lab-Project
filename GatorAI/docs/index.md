# Welcome to GatorAI

[![CI](https://github.com/your-org/gatorai/workflows/CI/badge.svg)](https://github.com/your-org/gatorai/actions)
[![Coverage](https://codecov.io/gh/your-org/gatorai/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/gatorai)
[![PyPI](https://img.shields.io/pypi/v/gatorai)](https://pypi.org/project/gatorai/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**GatorAI** is a professional-grade quantitative trading platform built with Python, featuring modular architecture for research and production use.

## 🚀 Key Features

- **Modular Data Pipeline**: Async data fetching from multiple sources (Yahoo Finance, Alpha Vantage, Polygon) with SQLite storage
- **Research-Grade Backtesting**: Vectorized backtesting engine with strategy abstraction and performance analytics
- **Advanced Optimization**: Multiple portfolio optimization methods (Mean-Variance, Black-Litterman, Risk Parity, CVaR)
- **Interactive Dashboard**: Streamlit-based UI for real-time analysis and visualization
- **Production Ready**: Docker containerization, CI/CD pipelines, and comprehensive testing

## 📦 Quick Installation

```bash
pip install gatorai
```

Or for development:

```bash
git clone https://github.com/your-org/gatorai.git
cd gatorai
pip install -e ".[dev]"
```

## 🏃 Quick Start

```python
from gatorai.data import DataManager
from gatorai.backtesting import run_backtest_strategy
from gatorai.optimization import mean_variance_optimize

# Fetch data
dm = DataManager()
await dm.fetch(["SPY", "QQQ", "IWM"])

# Run backtest
result = run_backtest_strategy(prices_df, strategy, rebalance="monthly")

# Optimize portfolio
weights = mean_variance_optimize(returns_df)
```

## 📊 Dashboard

Launch the interactive dashboard:

```bash
gatorai-dashboard
```

Or run with Docker:

```bash
docker-compose up gatorai
```

## 📚 Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](user-guide/data-pipeline.md)
- [API Reference](api/data.md)
- [Developer Guide](developer/architecture.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](developer/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/your-org/gatorai/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ by the GatorAI team. Special thanks to the quantitative finance community for inspiration and open-source tools.
