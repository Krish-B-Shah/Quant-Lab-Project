GatorAI
========

Quant research and portfolio optimization lab for SPY, QQQ, and IWM. The project provides a structured pipeline for market data acquisition, cleaning, feature engineering, backtesting strategies, and portfolio optimization, with a Streamlit dashboard prototype.

Project Structure
-----------------

- `data/`: Raw and processed datasets plus small examples for tests.
- `src/`: Core Python source code organized by module (data, backtesting, optimization, dashboard).
- `notebooks/`: Jupyter notebooks for exploration, backtesting, optimization, and dashboard prototyping.
- `tests/`: Unit and integration tests.
- `docs/`: Setup guide, API reference, and project plan.
- `scripts/`: Utility scripts for environment setup and data generation.

Quickstart
----------

1. Create and activate a virtual environment.
2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Generate example data:

   ```bash
   python scripts/generate_sample_data.py
   ```

4. Run the Streamlit dashboard prototype:

   ```bash
   streamlit run src/dashboard/app.py
   ```

Targets and Tickers
-------------------

Initial focus is on SPY, QQQ, and IWM. You can extend tickers and universes via utilities in `src/data/utils.py` and data fetchers in `src/data/fetch_data.py`.

Testing
-------

Run tests with:

```bash
pytest -q
```

License
-------

MIT


