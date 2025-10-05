# 🐊 GatorAI  
**Quant Research & Portfolio Optimization Lab** for SPY, QQQ, and IWM  

GatorAI provides a structured pipeline for **market data acquisition, cleaning, feature engineering, backtesting strategies, and portfolio optimization**, with a **Streamlit dashboard prototype** for visualization.  

---

## 🧱 Project Structure
data/ → Raw and processed datasets, plus small examples for testing
src/ → Core Python source code organized by module (data, backtesting, optimization, dashboard)
notebooks/ → Jupyter notebooks for exploration, backtesting, optimization, and dashboard prototyping
tests/ → Unit and integration tests
docs/ → Setup guide, API reference, and project plan
scripts/ → Utility scripts for environment setup and data generation

yaml
Copy code

---

## ⚙️ Quickstart  

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Windows: venv\Scripts\activate)
Install requirements

bash
Copy code
pip install -r requirements.txt
(Optional) Generate example data

bash
Copy code
python scripts/generate_sample_data.py
Run the Streamlit dashboard prototype

bash
Copy code
streamlit run src/dashboard/app.py
🎯 Targets and Tickers
Initial focus: SPY, QQQ, and IWM.
The ticker universe can be extended via:

src/data/utils.py — configuration utilities

src/data/fetch_data.py — data acquisition functions

🧪 Testing
Run all tests with:

bash
Copy code
pytest -q
🧠 Team (University of Florida, Fall 2025)
Krish Shah — Team Lead / Integration & Architecture

Neerav Gandhi

Sparsh Mogha

Son Tran

Navaj Sivkumar

Mahdi Haque

Muhammad Ismael

Sidhharth Radhakrishnan

🧩 Tech Stack
Languages: Python
Libraries: pandas, numpy, matplotlib, yfinance, PyPortfolioOpt, scikit-learn, streamlit, plotly
Tools: Git, Jupyter, Streamlit, pytest

📜 License
This project is licensed under the MIT License.
See LICENSE for details.

📈 Project Goals
Build a modular, reproducible research environment for quantitative portfolio optimization.

Implement AI-assisted risk and return modeling.

Develop an interactive dashboard for real-time portfolio analysis and backtesting.

Deliver a polished, production-ready prototype by end of semester.

🗓️ Development Roadmap
Phase	Weeks	Focus
Phase 1	1–3	Data collection, cleaning, and pipeline setup
Phase 2	4–6	Backtesting engine and performance metrics
Phase 3	7–8	Portfolio optimization and AI modeling
Phase 4	9–10	Dashboard integration, testing, and documentation

🤝 Contributions
We follow a simple Git workflow:

Create a new branch for your feature

Commit with clear messages

Open a Pull Request

Merge after review and testing

📫 Contact
Team Lead: Krish Shah
University of Florida — Fall 2025
