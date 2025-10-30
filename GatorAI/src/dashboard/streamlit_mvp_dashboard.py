import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
root = Path(__file__).resolve().parents[2]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data.manager import DataManager
from data.storage.sqlite_adapter import SQLiteAdapter
from backtesting.backtest_engine import run_backtest_strategy
from backtesting.strategy import EqualWeightStrategy, MomentumStrategy, VolatilityWeightedStrategy, MeanReversionStrategy, StrategyConfig
from optimization.optimizer import mean_variance_optimize, black_litterman_optimize, risk_parity_optimize, cvar_optimize

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard MVP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Sidebar Configuration
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

# Data Management Section
st.sidebar.subheader("📊 Data Management")

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", value="SPY,QQQ,IWM")
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

if st.sidebar.button("🔄 Fetch Latest Data"):
    with st.spinner("Fetching data..."):
        try:
            storage = SQLiteAdapter()
            dm = DataManager(storage)
            asyncio.run(dm.fetch(tickers, interval="1d", refresh=True))
            st.session_state.data_manager = dm
            st.session_state.data_loaded = True
            st.sidebar.success("✅ Data updated successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching data: {e}")

# Data Source Selection (legacy - keep for compatibility)
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Upload CSV", "Sample Data", "Live Feed"]
)

# File uploader (placeholder)
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        st.session_state.data_loaded = True
        st.sidebar.success("✅ File uploaded successfully!")
elif data_source == "Sample Data":
    st.session_state.data_loaded = True
elif data_source == "Live Feed":
    if st.session_state.data_manager is None:
        st.sidebar.warning("⚠️ Please fetch data first using the button above")
    else:
        st.session_state.data_loaded = True

# Strategy Selection
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Configuration")
strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ["Equal Weight", "Momentum", "Volatility Weighted", "Mean Reversion"]
)

# Timeframe Selection
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)

# Dynamic Parameter inputs based on strategy
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")

if strategy_type == "Equal Weight":
    long_only = st.sidebar.checkbox("Long Only", value=True)
    params = {"long_only": long_only}
elif strategy_type == "Momentum":
    lookback = st.sidebar.slider("Lookback Period", min_value=5, max_value=252, value=90, step=5)
    top_k = st.sidebar.number_input("Top K Assets (0 for all)", min_value=0, max_value=20, value=0)
    long_only = st.sidebar.checkbox("Long Only", value=True)
    params = {"lookback": lookback, "top_k": top_k if top_k > 0 else None, "long_only": long_only}
elif strategy_type == "Volatility Weighted":
    vol_window = st.sidebar.slider("Volatility Window", min_value=5, max_value=252, value=21, step=5)
    long_only = st.sidebar.checkbox("Long Only", value=True)
    params = {"vol_window": vol_window, "long_only": long_only}
elif strategy_type == "Mean Reversion":
    lookback = st.sidebar.slider("Lookback Period", min_value=5, max_value=252, value=20, step=5)
    top_k = st.sidebar.number_input("Top K Assets (0 for all)", min_value=0, max_value=20, value=0)
    long_only = st.sidebar.checkbox("Long Only", value=True)
    params = {"lookback": lookback, "top_k": top_k if top_k > 0 else None, "long_only": long_only}

# Action buttons
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    run_backtest = st.button("▶️ Run Backtest", width='stretch')
    if run_backtest:
        if st.session_state.data_manager is None:
            st.sidebar.error("Please fetch data first!")
        else:
            with st.spinner("Running backtest..."):
                try:
                    storage = SQLiteAdapter()
                    prices_df = pd.DataFrame()

                    for ticker in tickers:
                        price_data = storage.read_price_data(ticker)
                        if not price_data.empty:
                            prices_df[ticker] = price_data.set_index("datetime")["adj_close"]

                    if prices_df.empty:
                        st.sidebar.error("No price data available!")
                    else:
                        strategy_class = {
                            "Equal Weight": EqualWeightStrategy,
                            "Momentum": MomentumStrategy,
                            "Volatility Weighted": VolatilityWeightedStrategy,
                            "Mean Reversion": MeanReversionStrategy
                        }[strategy_type]

                        config = StrategyConfig(params=params)
                        strategy = strategy_class(config=config)

                        result = run_backtest_strategy(
                            prices_df,
                            strategy,
                            rebalance="monthly",
                            cost_bps=5,
                            slippage=0.1
                        )

                        st.session_state.backtest_results = {
                            "result": result,
                            "strategy": strategy_type,
                            "params": params
                        }
                        st.sidebar.success("✅ Backtest completed!")

                except Exception as e:
                    st.sidebar.error(f"❌ Backtest failed: {e}")

with col2:
    optimize = st.button("🔧 Optimize", width='stretch')
    if optimize:
        if st.session_state.data_manager is None:
            st.sidebar.error("Please fetch data first!")
        else:
            with st.spinner("Running optimization..."):
                try:
                    storage = SQLiteAdapter()
                    returns_df = pd.DataFrame()

                    for ticker in tickers:
                        price_data = storage.read_price_data(ticker)
                        if not price_data.empty:
                            prices = price_data.set_index("datetime")["adj_close"]
                            returns = prices.pct_change().dropna()
                            returns_df[ticker] = returns

                    if returns_df.empty:
                        st.sidebar.error("No returns data available!")
                    else:
                        opt_func = {
                            "Mean-Variance": mean_variance_optimize,
                            "Black-Litterman": black_litterman_optimize,
                            "Risk Parity": risk_parity_optimize,
                            "CVaR": cvar_optimize
                        }["Mean-Variance"]  # Default to Mean-Variance for MVP

                        weights = opt_func(returns_df)

                        st.session_state.optimization_results = {
                            "weights": weights,
                            "method": "Mean-Variance",
                            "tickers": tickers
                        }
                        st.sidebar.success("✅ Optimization completed!")

                except Exception as e:
                    st.sidebar.error(f"❌ Optimization failed: {e}")

# Main Dashboard Area
st.title("📊 Trading Dashboard MVP")
st.markdown("*Minimum Viable Product - Skeleton Version*")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Status", "Ready" if st.session_state.data_loaded else "No Data", "")
with col2:
    st.metric("Strategy", strategy_type, "")
with col3:
    st.metric("Timeframe", timeframe, "")
with col4:
    st.metric("Status", "Idle", "")

st.markdown("---")

# Tab layout for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Backtest Results", "🔧 Optimizer", "📁 Data Preview"])

# Tab 1: Overview
with tab1:
    st.subheader("Market Overview")

    if st.session_state.data_manager is not None:
        storage = SQLiteAdapter()
        latest_prices = {}

        for ticker in tickers:
            price_data = storage.read_price_data(ticker)
            if not price_data.empty:
                latest = price_data.iloc[-1]
                prev = price_data.iloc[-2] if len(price_data) > 1 else latest
                change = (latest["adj_close"] - prev["adj_close"]) / prev["adj_close"] * 100
                latest_prices[ticker] = {
                    "price": latest["adj_close"],
                    "change": change,
                    "volume": latest.get("volume", 0)
                }

        if latest_prices:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("##### Portfolio Price Chart")
                # Show combined portfolio chart
                portfolio_prices = pd.DataFrame()
                for ticker in tickers:
                    price_data = storage.read_price_data(ticker)
                    if not price_data.empty:
                        portfolio_prices[ticker] = price_data.set_index("datetime")["adj_close"]

                if not portfolio_prices.empty:
                    # Normalize to first value for comparison
                    normalized_prices = portfolio_prices.div(portfolio_prices.iloc[0])
                    st.line_chart(normalized_prices)
                else:
                    st.line_chart(pd.DataFrame(np.random.randn(100, 1) * 10 + 100, columns=['Price']))

            with col2:
                st.markdown("##### Key Metrics")
                for ticker, data in latest_prices.items():
                    st.metric(
                        f"{ticker} Price",
                        f"${data['price']:.2f}",
                        f"{data['change']:+.2f}%",
                        delta_color="normal" if data['change'] >= 0 else "inverse"
                    )
                    if data['volume'] > 0:
                        st.caption(f"Volume: {data['volume']:,.0f}")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("##### Price Chart (Placeholder)")
                chart_data = pd.DataFrame(
                    np.random.randn(100, 1) * 10 + 100,
                    columns=['Price']
                )
                st.line_chart(chart_data)

            with col2:
                st.markdown("##### Key Metrics")
                st.info("📌 **Latest Price:** $XXX.XX")
                st.info("📊 **Volume:** X,XXX,XXX")
                st.info("📈 **Change (24h):** +X.XX%")
                st.info("🎯 **Signal:** PENDING")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("##### Price Chart (Placeholder)")
            chart_data = pd.DataFrame(
                np.random.randn(100, 1) * 10 + 100,
                columns=['Price']
            )
            st.line_chart(chart_data)

        with col2:
            st.markdown("##### Key Metrics")
            st.info("📌 **Latest Price:** $XXX.XX")
            st.info("📊 **Volume:** X,XXX,XXX")
            st.info("📈 **Change (24h):** +X.XX%")
            st.info("🎯 **Signal:** PENDING")

# Tab 2: Backtest Results
with tab2:
    st.subheader("Backtest Results")

    if st.session_state.backtest_results:
        result = st.session_state.backtest_results["result"]
        strategy_name = st.session_state.backtest_results["strategy"]

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Equity", f"{result.equity_curve.iloc[-1]:.4f}")
        with col2:
            st.metric("CAGR", f"{result.stats['cagr']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{result.stats['sharpe']:.2f}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"##### {strategy_name} Strategy - Equity Curve")
            st.area_chart(result.equity_curve)

        with col2:
            st.markdown("##### Returns Distribution")
            returns_data = pd.DataFrame({
                'Returns': result.returns
            })
            st.bar_chart(returns_data)

        st.markdown("---")
        st.markdown("##### Performance Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Value': [
                f"{result.stats['total_return']:.2%}",
                f"{result.stats['cagr']:.2%}",
                f"{result.stats['vol']:.2%}",
                f"{result.stats['sharpe']:.2f}",
                f"{result.stats['max_drawdown']:.2%}",
                f"{result.stats.get('win_rate', 'N/A')}"
            ]
        })
        st.dataframe(stats_df, width='stretch')

    else:
        st.info("Run a backtest from the sidebar to see results here.")

        # Placeholder content
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", "TBD", "TBD%")
        with col2:
            st.metric("Sharpe Ratio", "TBD", "")
        with col3:
            st.metric("Max Drawdown", "TBD", "TBD%")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Equity Curve (Placeholder)")
            equity_data = pd.DataFrame(
                np.random.randn(100, 1).cumsum() + 100,
                columns=['Equity']
            )
            st.area_chart(equity_data)

        with col2:
            st.markdown("##### Trade Distribution (Placeholder)")
            trade_data = pd.DataFrame({
                'Wins': [np.random.randint(40, 60)],
                'Losses': [np.random.randint(20, 40)],
                'Breakeven': [np.random.randint(5, 15)]
            })
            st.bar_chart(trade_data.T)

# Tab 3: Optimizer
with tab3:
    st.subheader("Strategy Optimizer")

    if st.session_state.optimization_results:
        weights = st.session_state.optimization_results["weights"]
        method = st.session_state.optimization_results["method"]
        tickers = st.session_state.optimization_results["tickers"]

        st.markdown(f"##### {method} Optimization Results")

        # Display weights
        weights_df = pd.DataFrame({
            'Asset': tickers,
            'Weight': [weights.get(ticker, 0) for ticker in tickers]
        }).round(4)

        st.dataframe(weights_df, width='stretch')

        # Pie chart of weights
        if weights_df['Weight'].sum() > 0:
            st.markdown("##### Portfolio Allocation")
            st.bar_chart(weights_df.set_index('Asset')['Weight'])

        # Optimization metrics (placeholder for now)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Return", "TBD", "")
        with col2:
            st.metric("Expected Volatility", "TBD", "")
        with col3:
            st.metric("Sharpe Ratio", "TBD", "")

    else:
        st.info("Run optimization from the sidebar to see results here.")

        # Placeholder content
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Parameter Ranges")
            st.text_input("Parameter 1 Range", value="10-50")
            st.text_input("Parameter 2 Range", value="20-60")
            st.text_input("Parameter 3 Range", value="5-15")

        with col2:
            st.markdown("##### Optimization Settings")
            st.selectbox("Optimization Method", ["Grid Search", "Random Search", "Genetic Algorithm"])
            st.number_input("Max Iterations", value=100, min_value=10, max_value=1000)
            st.selectbox("Target Metric", ["Sharpe Ratio", "Total Return", "Win Rate"])

        st.markdown("---")
        st.markdown("##### Optimization Results (Placeholder)")

        optimization_results = pd.DataFrame({
            'Param1': np.random.randint(10, 50, 10),
            'Param2': np.random.randint(20, 60, 10),
            'Param3': np.random.randint(5, 15, 10),
            'Sharpe': np.random.uniform(0.5, 2.5, 10).round(2),
            'Return': np.random.uniform(-10, 30, 10).round(2),
            'Win Rate': np.random.uniform(40, 70, 10).round(1)
        })
        st.dataframe(optimization_results, width='stretch')

# Tab 4: Data Preview
with tab4:
    st.subheader("Data Preview")

    if st.session_state.data_manager is not None:
        storage = SQLiteAdapter()

        # Show data for each ticker
        for ticker in tickers:
            price_data = storage.read_price_data(ticker)
            if not price_data.empty:
                st.markdown(f"##### {ticker} Price Data")

                # Display recent data (last 20 rows)
                recent_data = price_data.tail(20).copy()
                recent_data['datetime'] = recent_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(recent_data, width='stretch')

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"##### {ticker} Statistics")
                    numeric_cols = price_data.select_dtypes(include=[np.number]).columns
                    if not numeric_cols.empty:
                        st.dataframe(price_data[numeric_cols].describe(), width='stretch')

                with col2:
                    st.markdown(f"##### {ticker} Info")
                    st.info(f"📊 **Rows:** {len(price_data)}")
                    st.info(f"📅 **Date Range:** {price_data['datetime'].min().strftime('%Y-%m-%d')} to {price_data['datetime'].max().strftime('%Y-%m-%d')}")
                    st.info(f"💾 **Memory Usage:** ~{price_data.memory_usage(deep=True).sum() / 1024:.2f} KB")

                st.markdown("---")
            else:
                st.warning(f"⚠️ No data available for {ticker}")

        if not any(storage.read_price_data(ticker).empty for ticker in tickers):
            st.success("✅ All data loaded successfully")
    else:
        st.warning("⚠️ No data loaded. Please fetch data from the sidebar.")
        st.info("💡 **Tip:** Click 'Fetch Latest Data' in the sidebar to load real market data")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚧 <strong>MVP Version</strong> - This is a skeleton dashboard with placeholder functionality</p>
        <p>Future modules will be integrated: Backtesting Engine • Strategy Optimizer • Live Trading • Risk Management</p>
    </div>
    """,
    unsafe_allow_html=True
)