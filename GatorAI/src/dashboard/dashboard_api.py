"""
Enhanced Streamlit Dashboard with FastAPI Integration

This dashboard connects to the FastAPI backend to provide:
- Real-time data fetching and visualization
- ML predictions with feature engineering pipeline
- Backtesting with comprehensive metrics
- Portfolio optimization with allocation visualization
- Full data flow visualization: raw data → features → ML → signals → portfolio
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional, Tuple
import time
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="GatorAI Quant Lab Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ml_predictions' not in st.session_state:
    st.session_state.ml_predictions = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'features_data' not in st.session_state:
    st.session_state.features_data = None
if 'prices_data' not in st.session_state:
    st.session_state.prices_data = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None


# ==================== API Client Functions ====================

def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"API connection error: {e}")
        return False


def fetch_data_api(tickers: List[str], interval: str = "1d", refresh: bool = False) -> Dict:
    """Fetch data via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/fetch",
            json={
                "tickers": tickers,
                "interval": interval,
                "refresh": refresh
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {"status": "error", "message": str(e)}


def get_prices_api(tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[Dict]:
    """Get price data via API."""
    try:
        tickers_str = ",".join(tickers)
        params = {"tickers": tickers_str}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        response = requests.get(f"{API_BASE_URL}/data/prices", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting prices: {e}")
        return None


def get_features_api(tickers: List[str], features: Optional[List[str]] = None) -> Optional[Dict]:
    """Get features via API."""
    try:
        tickers_str = ",".join(tickers)
        params = {"tickers": tickers_str}
        if features:
            params["features"] = ",".join(features)
        
        response = requests.get(f"{API_BASE_URL}/data/features", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting features: {e}")
        return None


def predict_api(tickers: List[str], features: Optional[List[str]] = None, retrain: bool = False) -> Optional[Dict]:
    """Get ML predictions via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "tickers": tickers,
                "features": features,
                "retrain": retrain,
                "blend": 0.5
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None


def backtest_api(request: Dict) -> Optional[Dict]:
    """Run backtest via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/backtest",
            json=request,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return None


def optimize_api(request: Dict) -> Optional[Dict]:
    """Run optimization via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize",
            json=request,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error optimizing portfolio: {e}")
        return None


# ==================== Visualization Functions ====================

def plot_price_data(prices_data: Dict) -> go.Figure:
    """Plot price data."""
    fig = go.Figure()
    
    if "data" in prices_data and "prices" in prices_data["data"]:
        dates = pd.to_datetime(prices_data["data"]["datetime"])
        for ticker, prices in prices_data["data"]["prices"].items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Price History",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        hovermode='x unified'
    )
    return fig


def plot_features(features_data: Dict, ticker: str, feature_names: List[str]) -> go.Figure:
    """Plot technical features."""
    if "data" not in features_data or ticker not in features_data["data"]:
        return go.Figure()
    
    ticker_data = features_data["data"][ticker]
    dates = pd.to_datetime(ticker_data["datetime"])
    features = ticker_data["features"]
    
    fig = make_subplots(
        rows=len(feature_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=feature_names
    )
    
    for i, feature_name in enumerate(feature_names):
        if feature_name in features:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=features[feature_name],
                    mode='lines',
                    name=feature_name,
                    line=dict(width=1.5)
                ),
                row=i+1,
                col=1
            )
    
    fig.update_layout(
        title=f"Technical Indicators - {ticker}",
        height=300 * len(feature_names),
        showlegend=False
    )
    fig.update_xaxes(title_text="Date", row=len(feature_names), col=1)
    
    return fig


def plot_ml_predictions(predictions: Dict, tickers: List[str]) -> go.Figure:
    """Plot ML predictions."""
    fig = go.Figure()
    
    if "predictions" in predictions:
        preds = predictions["predictions"]
        tickers_list = list(preds.keys())
        values = list(preds.values())
        
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig.add_trace(go.Bar(
            x=tickers_list,
            y=values,
            marker_color=colors,
            text=[f"{v:.2%}" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="ML Predicted Returns",
        xaxis_title="Ticker",
        yaxis_title="Predicted Return",
        height=400
    )
    return fig


def plot_equity_curve(equity_curve: Dict) -> go.Figure:
    """Plot equity curve."""
    fig = go.Figure()
    
    dates = pd.to_datetime(list(equity_curve.keys()))
    values = list(equity_curve.values())
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title="Backtest Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=400,
        hovermode='x unified'
    )
    return fig


def plot_portfolio_allocation(weights: Dict) -> go.Figure:
    """Plot portfolio allocation."""
    fig = go.Figure()
    
    tickers = list(weights.keys())
    values = list(weights.values())
    
    fig.add_trace(go.Pie(
        labels=tickers,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=400
    )
    return fig


def plot_data_flow(prices_data: Dict, features_data: Dict, predictions: Dict, 
                   backtest_results: Optional[Dict], optimization_results: Optional[Dict]) -> go.Figure:
    """Plot the complete data flow visualization."""
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=("Raw Prices", "Features (RSI)", "ML Predictions", 
                       "Equity Curve", "Returns Distribution", "Portfolio Allocation"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "histogram"}, {"type": "pie"}]]
    )
    
    # Row 1, Col 1: Raw Prices
    if prices_data and "data" in prices_data and "prices" in prices_data["data"]:
        dates = pd.to_datetime(prices_data["data"]["datetime"])
        for ticker, prices in list(prices_data["data"]["prices"].items())[:3]:  # Limit to 3 tickers
            fig.add_trace(
                go.Scatter(x=dates, y=prices, mode='lines', name=ticker, showlegend=False),
                row=1, col=1
            )
    
    # Row 1, Col 2: Features (RSI)
    if features_data and "data" in features_data:
        for ticker in list(features_data["data"].keys())[:1]:  # First ticker only
            ticker_data = features_data["data"][ticker]
            if "rsi" in ticker_data.get("features", {}):
                dates = pd.to_datetime(ticker_data["datetime"])
                rsi_values = ticker_data["features"]["rsi"]
                fig.add_trace(
                    go.Scatter(x=dates, y=rsi_values, mode='lines', name="RSI", 
                             line=dict(color='purple'), showlegend=False),
                    row=1, col=2
                )
    
    # Row 1, Col 3: ML Predictions
    if predictions and "predictions" in predictions:
        preds = predictions["predictions"]
        fig.add_trace(
            go.Bar(x=list(preds.keys()), y=list(preds.values()), showlegend=False),
            row=1, col=3
        )
    
    # Row 2, Col 1: Equity Curve
    if backtest_results and "results" in backtest_results and "equity_curve" in backtest_results["results"]:
        equity = backtest_results["results"]["equity_curve"]
        dates = pd.to_datetime(list(equity.keys()))
        values = list(equity.values())
        fig.add_trace(
            go.Scatter(x=dates, y=values, mode='lines', name="Equity", 
                     line=dict(color='green'), showlegend=False),
            row=2, col=1
        )
    
    # Row 2, Col 2: Returns Distribution
    if backtest_results and "results" in backtest_results and "returns" in backtest_results["results"]:
        returns = list(backtest_results["results"]["returns"].values())
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=30, showlegend=False),
            row=2, col=2
        )
    
    # Row 2, Col 3: Portfolio Allocation
    if optimization_results and "weights" in optimization_results:
        weights = optimization_results["weights"]
        fig.add_trace(
            go.Pie(labels=list(weights.keys()), values=list(weights.values()), showlegend=False),
            row=2, col=3
        )
    
    fig.update_layout(
        title="Complete Data Flow: Raw Data → Features → ML → Signals → Portfolio",
        height=600,
        showlegend=False
    )
    
    return fig


# ==================== Main Dashboard ====================

def main():
    """Main dashboard function."""
    st.markdown('<div class="main-header">🚀 GatorAI Quant Lab Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Real-time quantitative analysis platform with ML integration*")
    
    # Check API connection
    if not check_api_health():
        st.error("⚠️ API server is not available. Please start the FastAPI server first.")
        st.info("To start the API server, run: `uvicorn src.api.main:app --reload`")
        st.stop()
    else:
        st.session_state.api_connected = True
        st.sidebar.success("✅ API Connected")
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Data Management
    st.sidebar.subheader("📊 Data Management")
    tickers_input = st.sidebar.text_input("Tickers (comma-separated)", value="SPY,QQQ,IWM")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fetch_button = st.button("🔄 Fetch Data", use_container_width=True)
    with col2:
        refresh_button = st.button("🔄 Refresh", use_container_width=True)
    
    if fetch_button or refresh_button:
        with st.spinner("Fetching data from API..."):
            result = fetch_data_api(tickers, refresh=refresh_button)
            if result.get("status") == "success":
                st.sidebar.success("✅ Data fetch initiated!")
                st.session_state.data_fetched = True
                st.session_state.last_refresh = datetime.now()
                # Wait a bit for data to be processed
                time.sleep(2)
                # Load prices and features
                prices_response = get_prices_api(tickers)
                if prices_response:
                    st.session_state.prices_data = prices_response
                features_response = get_features_api(tickers)
                if features_response:
                    st.session_state.features_data = features_response
            else:
                st.sidebar.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    # ML Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 ML Predictions")
    use_ml = st.sidebar.checkbox("Use ML Predictions", value=False)
    retrain_model = st.sidebar.checkbox("Retrain Model", value=False)
    ml_blend = st.sidebar.slider("ML Blend Factor", 0.0, 1.0, 0.5, 0.1)
    
    if st.sidebar.button("🔮 Generate Predictions", use_container_width=True):
        with st.spinner("Generating ML predictions..."):
            predictions = predict_api(tickers, retrain=retrain_model)
            if predictions and predictions.get("status") == "success":
                st.session_state.ml_predictions = predictions
                st.sidebar.success("✅ Predictions generated!")
            else:
                st.sidebar.error("❌ Failed to generate predictions")
    
    # Backtest Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Backtesting")
    strategy = st.sidebar.selectbox("Strategy", ["equal_weight", "momentum", "volatility_weighted", "mean_reversion"])
    
    strategy_params = {}
    if strategy == "momentum":
        strategy_params["lookback"] = st.sidebar.slider("Lookback Period", 5, 252, 90)
    elif strategy == "volatility_weighted":
        strategy_params["vol_window"] = st.sidebar.slider("Volatility Window", 5, 252, 21)
    elif strategy == "mean_reversion":
        strategy_params["lookback"] = st.sidebar.slider("Lookback Period", 5, 252, 20)
    
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["daily", "weekly", "monthly", "quarterly"])
    cost_bps = st.sidebar.slider("Cost (bps)", 0, 50, 5)
    slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.1, 0.01) / 100
    
    if st.sidebar.button("▶️ Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            backtest_request = {
                "tickers": tickers,
                "strategy": strategy,
                "strategy_params": strategy_params,
                "rebalance": rebalance_freq,
                "cost_bps": cost_bps,
                "slippage": slippage,
                "use_ml_predictions": use_ml,
                "ml_blend": ml_blend
            }
            result = backtest_api(backtest_request)
            if result and result.get("status") == "success":
                st.session_state.backtest_results = result
                st.sidebar.success("✅ Backtest completed!")
            else:
                st.sidebar.error("❌ Backtest failed")
    
    # Optimization Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Optimization")
    opt_method = st.sidebar.selectbox("Method", ["mean_variance", "black_litterman", "risk_parity", "cvar"])
    risk_aversion = st.sidebar.slider("Risk Aversion", 0.1, 5.0, 1.0, 0.1)
    long_only = st.sidebar.checkbox("Long Only", value=True)
    
    if st.sidebar.button("🔧 Optimize Portfolio", use_container_width=True):
        with st.spinner("Optimizing portfolio..."):
            opt_request = {
                "tickers": tickers,
                "method": opt_method,
                "risk_aversion": risk_aversion,
                "long_only": long_only,
                "use_ml_predictions": use_ml,
                "ml_blend": ml_blend
            }
            result = optimize_api(opt_request)
            if result and result.get("status") == "success":
                st.session_state.optimization_results = result
                st.sidebar.success("✅ Optimization completed!")
            else:
                st.sidebar.error("❌ Optimization failed")
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Data Pipeline", "🤖 ML Predictions", 
        "🎯 Backtesting", "🎛️ Optimization", "🔄 Data Flow"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Market Overview")
        
        if st.session_state.prices_data:
            # Latest prices
            st.subheader("Latest Prices")
            if "data" in st.session_state.prices_data and "prices" in st.session_state.prices_data["data"]:
                prices = st.session_state.prices_data["data"]["prices"]
                dates = st.session_state.prices_data["data"]["datetime"]
                if dates and prices:
                    latest_idx = -1
                    cols = st.columns(len(tickers))
                    for i, ticker in enumerate(tickers):
                        if ticker in prices and len(prices[ticker]) > 0:
                            latest_price = prices[ticker][latest_idx]
                            prev_price = prices[ticker][latest_idx-1] if len(prices[ticker]) > 1 else latest_price
                            change_pct = ((latest_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                            with cols[i]:
                                st.metric(
                                    ticker,
                                    f"${latest_price:.2f}",
                                    f"{change_pct:+.2f}%"
                                )
            
            # Price chart
            st.subheader("Price History")
            fig = plot_price_data(st.session_state.prices_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tickers", len(tickers))
        with col2:
            st.metric("Data Fetched", "✅" if st.session_state.data_fetched else "❌")
        with col3:
            st.metric("ML Predictions", "✅" if st.session_state.ml_predictions else "❌")
        with col4:
            st.metric("Last Refresh", st.session_state.last_refresh.strftime("%H:%M:%S") if st.session_state.last_refresh else "N/A")
    
    # Tab 2: Data Pipeline
    with tab2:
        st.header("Data Pipeline: Raw Data → Features")
        
        if st.session_state.prices_data:
            st.subheader("Raw Price Data")
            fig = plot_price_data(st.session_state.prices_data)
            st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.features_data:
            st.subheader("Technical Features")
            selected_ticker = st.selectbox("Select Ticker", tickers, key="features_ticker")
            feature_names = st.multiselect(
                "Select Features",
                ["rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_mid", "rolling_vol", "rolling_sharpe"],
                default=["rsi", "macd", "bb_upper", "bb_lower"],
                key="features_select"
            )
            
            if feature_names:
                fig = plot_features(st.session_state.features_data, selected_ticker, feature_names)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fetch data to see features")
    
    # Tab 3: ML Predictions
    with tab3:
        st.header("Machine Learning Predictions")
        
        if st.session_state.ml_predictions:
            predictions = st.session_state.ml_predictions
            st.subheader("Predicted Returns")
            fig = plot_ml_predictions(predictions, tickers)
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction table
            if "predictions" in predictions:
                pred_df = pd.DataFrame({
                    "Ticker": list(predictions["predictions"].keys()),
                    "Predicted Return": [f"{v:.2%}" for v in predictions["predictions"].values()]
                })
                st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("Generate ML predictions to see results")
    
    # Tab 4: Backtesting
    with tab4:
        st.header("Backtesting Results")
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results["results"]
            stats = results.get("stats", {})
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CAGR", f"{stats.get('cagr', 0):.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{stats.get('sharpe', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{stats.get('max_drawdown', 0):.2%}")
            with col4:
                st.metric("Volatility", f"{stats.get('volatility', 0):.2%}")
            
            # Equity curve
            st.subheader("Equity Curve")
            if "equity_curve" in results:
                fig = plot_equity_curve(results["equity_curve"])
                st.plotly_chart(fig, use_container_width=True)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            if "returns" in results:
                returns = list(results["returns"].values())
                fig = px.histogram(x=returns, nbins=30, title="Daily Returns Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a backtest to see results")
    
    # Tab 5: Optimization
    with tab5:
        st.header("Portfolio Optimization")
        
        if st.session_state.optimization_results:
            results = st.session_state.optimization_results
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Return", f"{results.get('expected_return', 0):.2%}")
            with col2:
                st.metric("Expected Volatility", f"{results.get('expected_volatility', 0):.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            
            # Allocation
            st.subheader("Portfolio Allocation")
            if "weights" in results:
                fig = plot_portfolio_allocation(results["weights"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Weights table
                weights_df = pd.DataFrame({
                    "Ticker": list(results["weights"].keys()),
                    "Weight": [f"{v:.2%}" for v in results["weights"].values()]
                })
                st.dataframe(weights_df, use_container_width=True)
        else:
            st.info("Run optimization to see results")
    
    # Tab 6: Data Flow
    with tab6:
        st.header("Complete Data Flow Visualization")
        st.markdown("""
        This visualization shows the complete pipeline:
        **Raw Data → Features → ML Predictions → Trading Signals → Portfolio Allocation**
        """)
        
        fig = plot_data_flow(
            st.session_state.prices_data,
            st.session_state.features_data,
            st.session_state.ml_predictions,
            st.session_state.backtest_results,
            st.session_state.optimization_results
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pipeline status
        st.subheader("Pipeline Status")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Raw Data", "✅" if st.session_state.prices_data else "❌")
        with col2:
            st.metric("Features", "✅" if st.session_state.features_data else "❌")
        with col3:
            st.metric("ML Predictions", "✅" if st.session_state.ml_predictions else "❌")
        with col4:
            st.metric("Backtest", "✅" if st.session_state.backtest_results else "❌")
        with col5:
            st.metric("Optimization", "✅" if st.session_state.optimization_results else "❌")


if __name__ == "__main__":
    main()

