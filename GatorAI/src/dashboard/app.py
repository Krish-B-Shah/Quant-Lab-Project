import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

st.set_page_config(page_title="GatorAI Quant Dashboard", layout="wide", page_icon="📊")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Basic Authentication
def check_password():
    def password_entered():
        if st.session_state["password"] == "gatorai2024":
            st.session_state.authenticated = True
            del st.session_state["password"]
        else:
            st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

st.title("🚀 GatorAI Professional Quant Dashboard")
st.markdown("*Research-grade quantitative analysis platform*")

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

# Data Management Section
st.sidebar.markdown("---")
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
            st.sidebar.success("✅ Data updated successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching data: {e}")

# Strategy Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Configuration")

strategy_options = {
    "Equal Weight": EqualWeightStrategy,
    "Momentum": MomentumStrategy,
    "Volatility Weighted": VolatilityWeightedStrategy,
    "Mean Reversion": MeanReversionStrategy
}

selected_strategy = st.sidebar.selectbox("Strategy", list(strategy_options.keys()))

# Strategy parameters
params = {}
if selected_strategy == "Momentum":
    params["lookback"] = st.sidebar.slider("Lookback Period", 5, 252, 90)
elif selected_strategy == "Volatility Weighted":
    params["vol_window"] = st.sidebar.slider("Volatility Window", 5, 252, 21)
elif selected_strategy == "Mean Reversion":
    params["lookback"] = st.sidebar.slider("Lookback Period", 5, 252, 20)

rebalance_freq = st.sidebar.selectbox("Rebalancing Frequency", ["daily", "weekly", "monthly", "quarterly"])
cost_bps = st.sidebar.slider("Transaction Costs (bps)", 0, 50, 5)
slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.1, 0.1) / 100

# Optimization Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Optimization")

opt_methods = {
    "Mean-Variance": mean_variance_optimize,
    "Black-Litterman": black_litterman_optimize,
    "Risk Parity": risk_parity_optimize,
    "CVaR": cvar_optimize
}

selected_opt = st.sidebar.selectbox("Optimization Method", list(opt_methods.keys()))

# Action Buttons
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.sidebar.button("▶️ Run Backtest", use_container_width=True):
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
                        strategy_class = strategy_options[selected_strategy]
                        config = StrategyConfig(params=params)
                        strategy = strategy_class(config=config)

                        result = run_backtest_strategy(
                            prices_df,
                            strategy,
                            rebalance=rebalance_freq,
                            cost_bps=cost_bps,
                            slippage=slippage
                        )

                        st.session_state.backtest_results = {
                            "result": result,
                            "strategy": selected_strategy,
                            "params": params
                        }
                        st.sidebar.success("✅ Backtest completed!")

                except Exception as e:
                    st.sidebar.error(f"❌ Backtest failed: {e}")

with col2:
    if st.sidebar.button("🔧 Optimize", use_container_width=True):
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
                        opt_func = opt_methods[selected_opt]
                        weights = opt_func(returns_df)

                        st.session_state.optimization_results = {
                            "weights": weights,
                            "method": selected_opt,
                            "tickers": tickers
                        }
                        st.sidebar.success("✅ Optimization completed!")

                except Exception as e:
                    st.sidebar.error(f"❌ Optimization failed: {e}")

# Main Dashboard Content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Backtesting", "🎛️ Optimization", "📁 Data", "📤 Export"])

# Tab 1: Overview
with tab1:
    st.header("Market Overview & Key Metrics")

    if st.session_state.data_manager is not None:
        storage = SQLiteAdapter()
        latest_prices = {}

        for ticker in tickers:
            price_data = storage.read_price_data(ticker)
            if not price_data.empty:
                latest = price_data.iloc[-1]
                latest_prices[ticker] = {
                    "price": latest["adj_close"],
                    "change": (latest["adj_close"] / latest["close"] - 1) * 100 if latest["close"] != 0 else 0
                }

        if latest_prices:
            cols = st.columns(len(latest_prices))
            for i, (ticker, data) in enumerate(latest_prices.items()):
                with cols[i]:
                    st.metric(
                        f"{ticker} Price",
                        f"${data['price']:.2f}",
                        f"{data['change']:+.2f}%"
                    )

    # Sample performance chart
    if st.session_state.backtest_results:
        result = st.session_state.backtest_results["result"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode='lines',
            name='Portfolio Equity',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Backtesting Results
with tab2:
    st.header("Backtesting Results")

    if st.session_state.backtest_results:
        result = st.session_state.backtest_results["result"]
        strategy_name = st.session_state.backtest_results["strategy"]

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Equity", f"{result.equity_curve.iloc[-1]:.4f}")
        with col2:
            st.metric("CAGR", f"{result.stats['cagr']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{result.stats['sharpe']:.2f}")
        with col4:
            st.metric("Volatility", f"{result.stats['vol']:.2%}")

        # Equity curve
        fig = px.line(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            title=f"{strategy_name} Strategy - Equity Curve",
            labels={"x": "Date", "y": "Equity"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Returns distribution
        fig2 = px.histogram(
            result.returns,
            title="Returns Distribution",
            labels={"value": "Daily Return"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Run a backtest from the sidebar to see results here.")

# Tab 3: Optimization Results
with tab3:
    st.header("Portfolio Optimization")

    if st.session_state.optimization_results:
        weights = st.session_state.optimization_results["weights"]
        method = st.session_state.optimization_results["method"]

        st.subheader(f"Optimal Weights - {method}")

        # Weights table
        weights_df = pd.DataFrame({
            "Ticker": weights.index,
            "Weight": weights.values
        })
        st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

        # Pie chart
        fig = px.pie(
            weights_df,
            values="Weight",
            names="Ticker",
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run optimization from the sidebar to see results here.")

# Tab 4: Data Preview
with tab4:
    st.header("Data Preview")

    if st.session_state.data_manager is not None:
        storage = SQLiteAdapter()

        for ticker in tickers:
            price_data = storage.read_price_data(ticker)
            if not price_data.empty:
                st.subheader(f"{ticker} Price Data")
                st.dataframe(price_data.tail(10))

                # Price chart
                fig = px.line(
                    price_data,
                    x="datetime",
                    y="adj_close",
                    title=f"{ticker} Price History"
                )
                st.plotly_chart(fig, use_container_width=True)

# Tab 5: Export
with tab5:
    st.header("Export Results")

    if st.session_state.backtest_results:
        result = st.session_state.backtest_results["result"]

        # Export backtest results
        export_data = {
            "equity_curve": result.equity_curve,
            "returns": result.returns,
            "stats": result.stats
        }

        if st.button("📄 Export Backtest Results (JSON)"):
            import json
            json_str = json.dumps({
                "equity_curve": result.equity_curve.to_dict(),
                "returns": result.returns.to_dict(),
                "stats": result.stats
            }, indent=2, default=str)

            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="backtest_results.json",
                mime="application/json"
            )

    if st.session_state.optimization_results:
        weights = st.session_state.optimization_results["weights"]

        if st.button("📊 Export Optimization Weights (CSV)"):
            csv_data = weights.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="optimization_weights.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 <strong>GatorAI Professional Quant Platform</strong> - Week 4-5 Implementation</p>
        <p>Built with Streamlit, integrated with modular quant backend</p>
    </div>
    """,
    unsafe_allow_html=True
)
