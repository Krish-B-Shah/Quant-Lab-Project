# 🐊 GatorAI

**Quant Lab**

A structured pipeline for market data acquisition, cleaning, feature engineering, backtesting strategies, and portfolio optimization with an interactive Streamlit dashboard.

**Focus:** SPY, QQQ, and IWM

---

## 📋 Table of Contents

- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Targets and Tickers](#-targets-and-tickers)
- [Testing](#-testing)
- [Tech Stack](#-tech-stack)
- [Project Goals](#-project-goals)
- [Development Roadmap](#-development-roadmap)
- [Team](#-team)
- [Contributions](#-contributions)
- [License](#-license)
- [Contact](#-contact)

---

## 🧱 Project Structure

```
GatorAI/
├── data/                 # Raw and processed datasets, test samples
├── src/                  # Core Python modules
│   ├── data/            # Data acquisition and processing
│   ├── backtesting/     # Backtesting engine
│   ├── optimization/    # Portfolio optimization
│   └── dashboard/       # Streamlit application
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit and integration tests
├── docs/                # Documentation and guides
└── scripts/             # Utility scripts
```

---

## ⚙️ Quickstart

### 1. Set up environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Unix/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate sample data (optional)

```bash
python scripts/generate_sample_data.py
```

### 4. Launch dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## 🎯 Targets and Tickers

**Initial Focus:** SPY, QQQ, IWM

**Extend ticker universe via:**
- `src/data/utils.py` — Configuration utilities
- `src/data/fetch_data.py` — Data acquisition functions

---

## 🧪 Testing

Run all tests:

```bash
pytest -q
```

---

## 🧩 Tech Stack

**Languages:** Python

**Libraries:** pandas, numpy, matplotlib, yfinance, PyPortfolioOpt, scikit-learn, streamlit, plotly

**Tools:** Git, Jupyter, Streamlit, pytest

---

## 📈 Project Goals

- Build a modular, reproducible research environment for quantitative portfolio optimization
- Implement AI-assisted risk and return modeling
- Develop an interactive dashboard for real-time portfolio analysis and backtesting
- Deliver a polished, production-ready prototype by end of semester

---

## 🗓️ Development Roadmap

| Phase | Weeks | Focus |
|-------|-------|-------|
| **Phase 1** | 1–3 | Data collection, cleaning, and pipeline setup |
| **Phase 2** | 4–6 | Backtesting engine and performance metrics |
| **Phase 3** | 7–8 | Portfolio optimization and AI modeling |
| **Phase 4** | 9–10 | Dashboard integration, testing, and documentation |

---

## 🧠 Team

**University of Florida — Fall 2025**

- **Krish Shah** — Team Lead / Integration & Architecture
- Neerav Gandhi
- Sparsh Mogha
- Son Tran
- Navaj Sivkumar
- Mahdi Haque
- Muhammad Ismael
- Sidhharth Radhakrishnan

---

## 🤝 Contributions

We follow a simple Git workflow:

1. Create a new branch for your feature
2. Commit with clear, descriptive messages
3. Open a Pull Request
4. Merge after review and testing

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📫 Contact

**Team Lead:** Krish Shah  
**Institution:** University of Florida  
**Semester:** Fall 2025
