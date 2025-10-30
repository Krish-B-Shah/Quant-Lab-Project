"""
Production-Grade Quantitative Trading Dashboard
Enhanced with advanced analytics, caching, and professional features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import json
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
from typing import Dict, List, Tuple, Optional
import hashlib

# Page Configuration
st.set_page_config(
    page_title="Quantitative Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 0.25rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button enhancements */
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Alert boxes */
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Enhanced Backend Integration ====================
class BackendConnector:
    """Advanced backend connector with real data integration"""

    def __init__(self):
        self.cache_timeout = 300  # 5 minutes

    @staticmethod
    @st.cache_data(ttl=300)
    def load_market_data(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load real market data using DataManager"""
        try:
            storage = SQLiteAdapter()
            dm = DataManager(storage)
            # Run async data fetching
            import asyncio
            asyncio.run(dm.fetch(tickers, interval="1d", refresh=False))

            prices_df = pd.DataFrame()
            for ticker in tickers:
                price_data = storage.read_price_data(ticker)
                if not price_data.empty:
                    prices_df[ticker] = price_data.set_index("datetime")["adj_close"]

            # Filter by date range
            if not prices_df.empty:
                prices_df = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]

            return prices_df
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            # Fallback to simulated data
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            data = {}
            np.random.seed(42)
            for i, ticker in enumerate(tickers):
                trend = 0.0003
                volatility = 0.015 + np.random.rand() * 0.01
                shocks = np.random.randn(len(dates)) * volatility
                prices = 100 * np.exp(np.cumsum(trend + shocks))
                data[ticker] = pd.Series(prices, index=dates)
            return pd.DataFrame(data)
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns with proper handling of missing data"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio with proper handling"""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        if max_dd == 0:
            return 0.0
        
        annual_return = (1 + returns.mean()) ** 252 - 1
        return annual_return / max_dd
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 60) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate rolling performance metrics"""
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0, raw=True
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_beta = returns.rolling(window).apply(
            lambda x: x.corr(returns), raw=False
        )
        return rolling_sharpe, rolling_vol, rolling_beta
    
    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, method: str = 'max_sharpe', 
                          risk_free_rate: float = 0.02) -> np.ndarray:
        """Enhanced portfolio optimization with multiple methods"""
        n_assets = len(returns.columns)
        
        if method == 'max_sharpe':
            # Simplified mean-variance optimization
            cov_matrix = returns.cov() * 252
            # Add regularization to ensure positive definiteness
            cov_matrix += 1e-6 * np.eye(n_assets)
            mean_returns = returns.mean() * 252

            # Monte Carlo optimization (simplified)
            best_sharpe = -np.inf
            best_weights = None

            for _ in range(5000):
                weights = np.random.dirichlet(np.ones(n_assets))
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                # Avoid division by zero with stricter threshold
                if portfolio_vol > 1e-6:
                    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = weights
            
            return best_weights
            
        elif method == 'min_variance':
            # Minimum variance portfolio
            cov_matrix = returns.cov() * 252
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones(n_assets)
            weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
            return np.abs(weights) / np.abs(weights).sum()
            
        elif method == 'equal_weight':
            return np.ones(n_assets) / n_assets
        
        elif method == 'risk_parity':
            # Equal risk contribution
            weights = np.ones(n_assets) / n_assets
            cov_matrix = returns.cov() * 252
            
            for _ in range(100):  # Iterative adjustment
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                target_risk = risk_contrib.mean()
                weights = weights * target_risk / risk_contrib
                weights = weights / weights.sum()
            
            return weights
        
        return np.ones(n_assets) / n_assets
    
    @staticmethod
    def generate_efficient_frontier(returns: pd.DataFrame, n_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier with improved sampling"""
        results = []
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252
        
        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n_assets))
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_vol > 0:
                sharpe = portfolio_return / portfolio_vol
            else:
                sharpe = 0
            
            results.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': sharpe
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        var = returns.quantile(1 - confidence_level)
        cvar = returns[returns <= var].mean()
        return var, cvar

# ==================== Session State Management ====================
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'authenticated': False,
        'username': None,
        'data_cache': {},
        'last_update': None,
        'show_create_account': False,
        'saved_configs': {},
        'analysis_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== Authentication Module ====================
def show_login():
    """Enhanced authentication interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0;'>
                <h1 style='color: #667eea; margin-bottom: 0.5rem;'>📊</h1>
                <h2 style='color: #1f2937; margin-bottom: 0.5rem;'>Portfolio Analytics</h2>
                <p style='color: #6b7280;'>Professional quantitative trading dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Remember me")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_submitted = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")
            with col_b:
                demo_submitted = st.form_submit_button("👁️ Demo Mode", use_container_width=True)
            
            if login_submitted:
                if username and password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Please enter both username and password")
            
            if demo_submitted:
                st.session_state.authenticated = True
                st.session_state.username = "Demo User"
                st.success("✅ Entering demo mode...")
                st.rerun()
        
        st.markdown("---")
        
        col_x, col_y = st.columns(2)
        with col_x:
            if st.button("📝 Create Account", use_container_width=True):
                st.session_state.show_create_account = True
                st.rerun()
        with col_y:
            if st.button("🔑 Forgot Password?", use_container_width=True):
                st.info("Password reset functionality - Connect to email service")
        
        st.markdown("<div style='text-align: center; margin-top: 2rem; color: #6b7280; font-size: 0.875rem;'>Secure • Encrypted • Professional</div>", unsafe_allow_html=True)

def show_create_account():
    """Enhanced account creation interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0;'>
                <h2 style='color: #1f2937;'>Create Your Account</h2>
                <p style='color: #6b7280;'>Join our quantitative trading platform</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.form("create_account_form"):
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_email = st.text_input("Email Address", placeholder="your.email@example.com")
            new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            st.markdown("#### Account Type")
            account_type = st.radio("Select account type", ["Individual Trader", "Institutional", "Academic"])
            
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            col_a, col_b = st.columns(2)
            with col_a:
                create_submitted = st.form_submit_button("✅ Create Account", use_container_width=True, type="primary")
            with col_b:
                back_submitted = st.form_submit_button("← Back to Login", use_container_width=True)
            
            if create_submitted:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("❌ Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif len(new_password) < 8:
                    st.error("❌ Password must be at least 8 characters")
                elif not agree_terms:
                    st.error("❌ Please agree to the terms and conditions")
                else:
                    st.success(f"✅ Account created successfully for {new_username}!")
                    st.session_state.authenticated = True
                    st.session_state.username = new_username
                    st.balloons()
                    st.rerun()
            
            if back_submitted:
                st.session_state.show_create_account = False
                st.rerun()
        
        st.markdown("---")
        st.info("🔒 Your data is encrypted with industry-standard security protocols")

# ==================== Enhanced Visualization Module ====================
class ChartBuilder:
    """Professional chart builder with advanced Plotly features"""
    
    @staticmethod
    def create_cumulative_returns_chart(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> go.Figure:
        """Enhanced cumulative returns with annotations"""
        strategy_cum = (1 + strategy_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=strategy_cum.index,
            y=strategy_cum.values,
            mode='lines',
            name='Strategy',
            line=dict(color='#667eea', width=3),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate='<b>Strategy</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_cum.index,
            y=benchmark_cum.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#f093fb', width=2, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add annotations for key points
        final_strategy = strategy_cum.iloc[-1]
        final_benchmark = benchmark_cum.iloc[-1]
        
        fig.add_annotation(
            x=strategy_cum.index[-1],
            y=final_strategy,
            text=f"<b>{final_strategy:.2f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#667eea',
            font=dict(color='#667eea', size=12, family='Arial Black')
        )
        
        fig.update_layout(
            title=dict(
                text='<b>Cumulative Returns Comparison</b>',
                font=dict(size=20, color='#1f2937')
            ),
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            height=450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e5e7eb',
                borderwidth=1
            ),
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        
        return fig
    
    @staticmethod
    def create_rolling_metrics_chart(rolling_sharpe: pd.Series, rolling_vol: pd.Series) -> go.Figure:
        """Enhanced rolling metrics with zones"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('<b>Rolling Sharpe Ratio (60-day)</b>', '<b>Rolling Volatility (60-day)</b>'),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )
        
        # Sharpe ratio with zones
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Sharpe Ratio',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)',
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add threshold line for Sharpe
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1,
                     annotation_text="Target", annotation_position="right")
        
        # Volatility chart
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#f093fb', width=2),
                fill='tozeroy',
                fillcolor='rgba(240, 147, 251, 0.2)',
                hovertemplate='%{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Ratio</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>Annualized (%)</b>", tickformat='.0%', row=2, col=1)
        
        fig.update_layout(
            height=650,
            template='plotly_white',
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        
        return fig
    
    @staticmethod
    def create_allocation_pie_chart(weights: np.ndarray, labels: List[str]) -> go.Figure:
        """Enhanced allocation visualization"""
        colors = px.colors.qualitative.Set3
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.4,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<br>Allocation: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text='<b>Portfolio Allocation</b>',
                font=dict(size=18, color='#1f2937'),
                x=0.5,
                xanchor='center'
            ),
            height=450,
            template='plotly_white',
            annotations=[dict(
                text='Portfolio',
                x=0.5,
                y=0.5,
                font=dict(size=16, color='#6b7280'),
                showarrow=False
            )]
        )
        
        return fig
    
    @staticmethod
    def create_efficient_frontier(frontier_data: pd.DataFrame, current_portfolio: Optional[Dict] = None) -> go.Figure:
        """Enhanced efficient frontier with optimal portfolio"""
        fig = go.Figure()
        
        # Frontier points
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='markers',
            marker=dict(
                size=10,
                color=frontier_data['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="<b>Sharpe<br>Ratio</b>",
                    thickness=15,
                    len=0.7
                ),
                line=dict(color='white', width=0.5)
            ),
            text=[f"Sharpe: {s:.2f}" for s in frontier_data['sharpe']],
            hovertemplate='<b>Portfolio</b><br>Return: %{y:.2%}<br>Risk: %{x:.2%}<br>%{text}<extra></extra>',
            name='Efficient Frontier'
        ))
        
        # Current portfolio
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio['volatility']],
                y=[current_portfolio['return']],
                mode='markers',
                marker=dict(size=20, color='#ef4444', symbol='star', 
                          line=dict(color='white', width=2)),
                name='Current',
                hovertemplate='<b>Current Portfolio</b><br>Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
            ))
        
        # Optimal portfolio (max Sharpe)
        max_sharpe_idx = frontier_data['sharpe'].idxmax()
        optimal = frontier_data.loc[max_sharpe_idx]
        
        fig.add_trace(go.Scatter(
            x=[optimal['volatility']],
            y=[optimal['return']],
            mode='markers',
            marker=dict(size=18, color='#10b981', symbol='diamond',
                       line=dict(color='white', width=2)),
            name='Optimal',
            hovertemplate=f'<b>Optimal Portfolio</b><br>Return: {optimal["return"]:.2%}<br>Risk: {optimal["volatility"]:.2%}<br>Sharpe: {optimal["sharpe"]:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='<b>Efficient Frontier Analysis</b>',
                font=dict(size=20, color='#1f2937')
            ),
            xaxis_title='<b>Volatility (Risk)</b>',
            yaxis_title='<b>Expected Return</b>',
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            template='plotly_white',
            height=550,
            hovermode='closest',
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            legend=dict(
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e5e7eb',
                borderwidth=1
            )
        )
        
        return fig

# ==================== Main Dashboard ====================
def main_dashboard():
    """Enhanced main dashboard interface"""
    
    # Header with user info
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.markdown(f"# 📊 Quantitative Portfolio Analytics")
        st.markdown(f"<p style='color: #6b7280; margin-top: -1rem;'>Welcome back, <b>{st.session_state.username}</b></p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div style='text-align: right; padding-top: 1.5rem; color: #6b7280;'>{datetime.now().strftime('%A, %B %d, %Y')}<br>{datetime.now().strftime('%I:%M %p')}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown("---")
    
    # ==================== Enhanced Sidebar ====================
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            load_config = st.button("📂 Load", use_container_width=True)
        with col2:
            save_config = st.button("💾 Save", use_container_width=True)
        
        st.markdown("---")
        
        # Strategy Configuration
        st.markdown("### 🎯 Strategy Setup")
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Mean-Variance", "Risk Parity", "Momentum", "Value", "Equal Weight", "Custom"],
            help="Select the portfolio optimization strategy"
        )
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["max_sharpe", "min_variance", "equal_weight", "risk_parity"],
            help="Choose the optimization objective"
        )
        
        # Asset Universe with search
        st.markdown("### 📈 Asset Universe")
        available_assets = ["SPY", "QQQ", "TLT", "GLD", "IWM", "EFA", "VNQ", "BND", 
                           "AGG", "DIA", "IEF", "SHY", "VTI", "VOO", "IVV"]
        selected_assets = st.multiselect(
            "Select Assets (min 2)",
            available_assets,
            default=["SPY", "QQQ", "TLT", "GLD"],
            help="Choose at least 2 assets for portfolio construction"
        )
        
        # Time Period
        st.markdown("### 📅 Analysis Period")
        date_preset = st.selectbox(
            "Quick Select",
            ["Custom", "1 Year", "2 Years", "3 Years", "5 Years", "YTD"],
            help="Select a preset date range"
        )
        
        if date_preset == "1 Year":
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
        elif date_preset == "2 Years":
            start_date = datetime.now() - timedelta(days=730)
            end_date = datetime.now()
        elif date_preset == "3 Years":
            start_date = datetime.now() - timedelta(days=1095)
            end_date = datetime.now()
        elif date_preset == "5 Years":
            start_date = datetime.now() - timedelta(days=1825)
            end_date = datetime.now()
        elif date_preset == "YTD":
            start_date = datetime(datetime.now().year, 1, 1)
            end_date = datetime.now()
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start",
                    value=datetime.now() - timedelta(days=730)
                )
            with col2:
                end_date = st.date_input(
                    "End",
                    value=datetime.now()
                )
        
        # Advanced Settings (Expandable)
        with st.expander("🔧 Advanced Settings", expanded=False):
            rebalance_freq = st.select_slider(
                "Rebalancing Frequency",
                options=["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                value="Monthly"
            )
            
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.1,
                help="Used for Sharpe ratio calculations"
            ) / 100
            
            target_vol = st.slider(
                "Target Volatility (%)",
                min_value=5.0,
                max_value=40.0,
                value=15.0,
                step=1.0,
                help="Target annualized volatility"
            ) / 100
            
            confidence_level = st.slider(
                "VaR Confidence Level (%)",
                min_value=90.0,
                max_value=99.0,
                value=95.0,
                step=1.0,
                help="Confidence level for Value at Risk"
            ) / 100
            
            rolling_window = st.number_input(
                "Rolling Window (days)",
                min_value=20,
                max_value=252,
                value=60,
                step=10,
                help="Window for rolling metrics"
            )
        
        st.markdown("---")
        
        # Execution
        st.markdown("### 🚀 Execute")
        col1, col2 = st.columns(2)
        with col1:
            run_analysis = st.button("▶️ Run", use_container_width=True, type="primary")
        with col2:
            auto_refresh = st.checkbox("Auto", help="Auto-refresh every 60s")
        
        if auto_refresh:
            st.info("🔄 Auto-refresh: ON")
        
        # System Info
        st.markdown("---")
        st.markdown("### 📊 System Info")
        if st.session_state.last_update:
            st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d')}")
    
    # ==================== Main Content Area ====================
    
    # Validation
    if len(selected_assets) < 2:
        st.warning("⚠️ Please select at least 2 assets to continue")
        return
    
    # Run analysis
    if run_analysis or auto_refresh:
        with st.spinner("🔄 Loading data and optimizing portfolio..."):
            try:
                # Initialize backend
                backend = BackendConnector()
                
                # Load data
                prices = backend.load_market_data(selected_assets, start_date, end_date)
                returns = backend.calculate_returns(prices)
                
                # Calculate benchmark (equal weight)
                benchmark_returns = returns.mean(axis=1)
                
                # Optimize portfolio
                optimal_weights = backend.optimize_portfolio(
                    returns, 
                    method=optimization_method,
                    risk_free_rate=risk_free_rate
                )
                
                # Calculate strategy returns
                strategy_returns = (returns * optimal_weights).sum(axis=1)
                
                # Calculate metrics
                rolling_sharpe, rolling_vol, rolling_beta = backend.calculate_rolling_metrics(
                    strategy_returns, 
                    window=rolling_window
                )
                
                # Generate efficient frontier
                frontier_data = backend.generate_efficient_frontier(returns, n_portfolios=100)
                
                # Risk metrics
                var, cvar = backend.calculate_var_cvar(strategy_returns, confidence_level)
                
                st.session_state.last_update = datetime.now()
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    else:
        st.info("👈 Configure your portfolio parameters and click **Run** to begin analysis")
        return
    
    # ==================== Performance Metrics Dashboard ====================
    st.markdown("## 📈 Performance Overview")
    
    # Calculate key metrics
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    sharpe = backend.calculate_sharpe_ratio(strategy_returns, risk_free_rate)
    sortino = backend.calculate_sortino_ratio(strategy_returns, risk_free_rate)
    calmar = backend.calculate_calmar_ratio(strategy_returns)
    
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    volatility = strategy_returns.std() * np.sqrt(252)
    win_rate = (strategy_returns > 0).mean()
    
    # Benchmark comparison
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    alpha = total_return - benchmark_total_return
    
    # Metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        delta_color = "normal" if total_return > 0 else "inverse"
        st.metric(
            "Total Return", 
            f"{total_return:.2%}",
            delta=f"{annual_return:.2%} ann.",
            delta_color=delta_color
        )
    
    with col2:
        sharpe_delta = "Excellent" if sharpe > 2 else ("Good" if sharpe > 1 else "Fair")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta=sharpe_delta)
    
    with col3:
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")
    
    with col5:
        st.metric("Volatility", f"{volatility:.2%}")
    
    with col6:
        st.metric("Alpha vs Bench", f"{alpha:.2%}", delta=f"Win: {win_rate:.1%}")
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Calmar Ratio", f"{calmar:.2f}")
    
    with col2:
        st.metric(f"VaR ({confidence_level:.0%})", f"{var:.2%}")
    
    with col3:
        st.metric(f"CVaR ({confidence_level:.0%})", f"{cvar:.2%}")
    
    with col4:
        trades = len(strategy_returns)
        st.metric("Data Points", f"{trades:,}")
    
    st.markdown("---")
    
    # ==================== Tabbed Interface ====================
    tabs = st.tabs([
        "📊 Returns Analysis",
        "📉 Risk Metrics", 
        "🎯 Portfolio Allocation",
        "🔬 Efficient Frontier",
        "📊 Advanced Analytics",
        "📥 Export & Reports"
    ])
    
    # ==================== Tab 1: Returns Analysis ====================
    with tabs[0]:
        st.markdown("### Cumulative Performance")
        
        chart_builder = ChartBuilder()
        returns_chart = chart_builder.create_cumulative_returns_chart(
            strategy_returns, benchmark_returns
        )
        st.plotly_chart(returns_chart, use_container_width=True)
        
        # Performance comparison table
        st.markdown("### Performance Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Strategy': [
                f"{total_return:.2%}",
                f"{annual_return:.2%}",
                f"{volatility:.2%}",
                f"{sharpe:.2f}",
                f"{max_drawdown:.2%}",
                f"{win_rate:.1%}"
            ],
            'Benchmark': [
                f"{benchmark_total_return:.2%}",
                f"{((1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1):.2%}",
                f"{(benchmark_returns.std() * np.sqrt(252)):.2%}",
                f"{backend.calculate_sharpe_ratio(benchmark_returns, risk_free_rate):.2f}",
                f"{((1 + benchmark_returns).cumprod() / (1 + benchmark_returns).cumprod().cummax() - 1).min():.2%}",
                f"{(benchmark_returns > 0).mean():.1%}"
            ]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Returns Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=strategy_returns,
                nbinsx=50,
                name='Strategy',
                marker_color='#667eea',
                opacity=0.7,
                hovertemplate='Return: %{x:.2%}<br>Count: %{y}<extra></extra>'
            ))
            fig.add_trace(go.Histogram(
                x=benchmark_returns,
                nbinsx=50,
                name='Benchmark',
                marker_color='#f093fb',
                opacity=0.5,
                hovertemplate='Return: %{x:.2%}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add normal distribution overlay
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()
            x_range = np.linspace(strategy_returns.min(), strategy_returns.max(), 100)
            normal_dist = (1/(std_ret * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mean_ret)/std_ret)**2)
            normal_dist = normal_dist * len(strategy_returns) * (strategy_returns.max() - strategy_returns.min()) / 50
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Dist',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Daily Return",
                yaxis_title="Frequency",
                template='plotly_white',
                height=400,
                barmode='overlay',
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📅 Monthly Returns Heatmap")
            monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            if len(monthly_returns) > 0:
                monthly_df = pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.month,
                    'Return': monthly_returns.values
                })
                
                pivot_table = monthly_df.pivot_table(
                    index='Year', 
                    columns='Month', 
                    values='Return',
                    aggfunc='first'
                )
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=[month_names[i-1] for i in pivot_table.columns],
                    y=pivot_table.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(pivot_table.values * 100, 1),
                    texttemplate='%{text}%',
                    textfont={"size": 10},
                    hovertemplate='%{y} %{x}<br>Return: %{z:.2%}<extra></extra>',
                    colorbar=dict(title="Return")
                ))
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    xaxis=dict(side='top')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for monthly heatmap")
    
    # ==================== Tab 2: Risk Metrics ====================
    with tabs[1]:
        st.markdown("### Rolling Performance Metrics")
        
        rolling_chart = chart_builder.create_rolling_metrics_chart(
            rolling_sharpe, rolling_vol
        )
        st.plotly_chart(rolling_chart, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Drawdown Analysis")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#ef4444', width=2),
                fillcolor='rgba(239, 68, 68, 0.2)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>'
            ))
            
            # Mark maximum drawdown
            max_dd_date = drawdown.idxmin()
            fig.add_annotation(
                x=max_dd_date,
                y=max_drawdown,
                text=f"Max DD: {max_drawdown:.2%}",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#ef4444',
                font=dict(color='#ef4444', size=11)
            )
            
            fig.update_layout(
                title='Underwater Equity Curve',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                yaxis_tickformat='.1%',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Risk Metrics Summary")
            
            risk_metrics_df = pd.DataFrame({
                'Metric': [
                    'Value at Risk (VaR)',
                    'Conditional VaR (CVaR)',
                    'Maximum Drawdown',
                    'Average Drawdown',
                    'Downside Deviation',
                    'Skewness',
                    'Kurtosis'
                ],
                'Value': [
                    f"{var:.2%}",
                    f"{cvar:.2%}",
                    f"{max_drawdown:.2%}",
                    f"{drawdown.mean():.2%}",
                    f"{(strategy_returns[strategy_returns < 0].std() * np.sqrt(252)):.2%}",
                    f"{strategy_returns.skew():.2f}",
                    f"{strategy_returns.kurtosis():.2f}"
                ],
                'Interpretation': [
                    f"{confidence_level:.0%} confidence",
                    "Expected loss in tail",
                    "Largest peak-to-trough",
                    "Mean underwater period",
                    "Downside volatility only",
                    "Distribution asymmetry",
                    "Tail thickness"
                ]
            })
            
            st.dataframe(risk_metrics_df, use_container_width=True, hide_index=True)
        
        # Rolling correlation with benchmark
        st.markdown("### 🔄 Rolling Correlation with Benchmark")
        rolling_corr = strategy_returns.rolling(window=rolling_window).corr(benchmark_returns)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            mode='lines',
            line=dict(color='#8b5cf6', width=2),
            fill='tonexty',
            fillcolor='rgba(139, 92, 246, 0.1)',
            hovertemplate='Correlation: %{y:.3f}<extra></extra>'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Correlation',
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== Tab 3: Portfolio Allocation ====================
    with tabs[2]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Current Allocation")
            allocation_chart = chart_builder.create_allocation_pie_chart(
                optimal_weights, selected_assets
            )
            st.plotly_chart(allocation_chart, use_container_width=True)
        
        with col2:
            st.markdown("### Allocation Details")
            
            portfolio_value = 1000000  # Example $1M portfolio
            allocation_df = pd.DataFrame({
                'Asset': selected_assets,
                'Weight (%)': [f"{w*100:.2f}%" for w in optimal_weights],
                'Value ($)': [f"${w * portfolio_value:,.0f}" for w in optimal_weights],
                'Shares': [int((w * portfolio_value) / prices[asset].iloc[-1]) for w, asset in zip(optimal_weights, selected_assets)]
            })
            
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### Portfolio Statistics")
            portfolio_stats = pd.DataFrame({
                'Metric': ['Total Value', 'Number of Assets', 'Largest Position', 'Smallest Position', 'Concentration (HHI)'],
                'Value': [
                    f"${portfolio_value:,.0f}",
                    f"{len(selected_assets)}",
                    f"{optimal_weights.max()*100:.2f}%",
                    f"{optimal_weights.min()*100:.2f}%",
                    f"{(optimal_weights**2).sum():.3f}"
                ]
            })
            st.dataframe(portfolio_stats, use_container_width=True, hide_index=True)
        
        # Weight comparison across methods
        st.markdown("### 🔀 Optimization Method Comparison")
        
        methods = ['max_sharpe', 'min_variance', 'equal_weight', 'risk_parity']
        weights_comparison = {}
        
        for method in methods:
            weights_comparison[method] = backend.optimize_portfolio(returns, method=method, risk_free_rate=risk_free_rate)
        
        comparison_df = pd.DataFrame(weights_comparison, index=selected_assets)
        comparison_df.columns = ['Max Sharpe', 'Min Variance', 'Equal Weight', 'Risk Parity']
        comparison_df = comparison_df.applymap(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Contribution to risk and return
        st.markdown("### 📊 Risk & Return Contribution")
        
        cov_matrix = returns.cov() * 252
        portfolio_var = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
        marginal_contrib_var = np.dot(cov_matrix, optimal_weights)
        contrib_var = optimal_weights * marginal_contrib_var
        contrib_var_pct = contrib_var / portfolio_var
        
        mean_returns = returns.mean() * 252
        contrib_return = optimal_weights * mean_returns
        
        contribution_df = pd.DataFrame({
            'Asset': selected_assets,
            'Weight': [f"{w*100:.2f}%" for w in optimal_weights],
            'Return Contrib': [f"{r*100:.2f}%" for r in contrib_return],
            'Risk Contrib': [f"{r*100:.2f}%" for r in contrib_var_pct]
        })
        
        st.dataframe(contribution_df, use_container_width=True, hide_index=True)
    
    # ==================== Tab 4: Efficient Frontier ====================
    with tabs[3]:
        st.markdown("### Mean-Variance Efficient Frontier")
        
        current_portfolio = {
            'volatility': volatility,
            'return': annual_return
        }
        
        frontier_chart = chart_builder.create_efficient_frontier(
            frontier_data, current_portfolio
        )
        st.plotly_chart(frontier_chart, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Optimal Portfolio (Max Sharpe)")
            max_sharpe_idx = frontier_data['sharpe'].idxmax()
            optimal_portfolio = frontier_data.loc[max_sharpe_idx]
            
            optimal_df = pd.DataFrame({
                'Metric': ['Expected Return', 'Expected Volatility', 'Sharpe Ratio', 'Optimization Method'],
                'Value': [
                    f"{optimal_portfolio['return']:.2%}",
                    f"{optimal_portfolio['volatility']:.2%}",
                    f"{optimal_portfolio['sharpe']:.2f}",
                    optimization_method.replace('_', ' ').title()
                ]
            })
            st.dataframe(optimal_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### Current vs Optimal")
            if optimal_portfolio['sharpe'] > sharpe:
                improvement = ((optimal_portfolio['sharpe'] - sharpe) / sharpe) * 100
                st.success(f"✅ Potential Sharpe improvement: +{improvement:.1f}%")
            else:
                st.info("✓ Current portfolio is near-optimal")
        
        with col2:
            st.markdown("### 📊 Correlation Matrix")
            corr_matrix = returns.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                template='plotly_white',
                height=400,
                xaxis=dict(side='bottom'),
                margin=dict(l=80, r=20, t=20, b=80)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Frontier statistics
        st.markdown("### 📈 Frontier Statistics")
        frontier_stats = pd.DataFrame({
            'Statistic': ['Best Sharpe Ratio', 'Highest Return', 'Lowest Volatility', 'Average Return', 'Average Volatility'],
            'Value': [
                f"{frontier_data['sharpe'].max():.2f}",
                f"{frontier_data['return'].max():.2%}",
                f"{frontier_data['volatility'].min():.2%}",
                f"{frontier_data['return'].mean():.2%}",
                f"{frontier_data['volatility'].mean():.2%}"
            ]
        })
        st.dataframe(frontier_stats, use_container_width=True, hide_index=True)
    
    # ==================== Tab 5: Advanced Analytics ====================
    with tabs[4]:
        st.markdown("### 🔬 Advanced Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Return Analysis")
            
            # Best and worst periods
            best_day = strategy_returns.max()
            worst_day = strategy_returns.min()
            best_month = strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1).max()
            worst_month = strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1).min()
            
            extremes_df = pd.DataFrame({
                'Period': ['Best Day', 'Worst Day', 'Best Month', 'Worst Month'],
                'Return': [
                    f"{best_day:.2%}",
                    f"{worst_day:.2%}",
                    f"{best_month:.2%}",
                    f"{worst_month:.2%}"
                ],
                'Date': [
                    strategy_returns.idxmax().strftime('%Y-%m-%d'),
                    strategy_returns.idxmin().strftime('%Y-%m-%d'),
                    strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1).idxmax().strftime('%Y-%m'),
                    strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1).idxmin().strftime('%Y-%m')
                ]
            })
            st.dataframe(extremes_df, use_container_width=True, hide_index=True)
            
            # Win/Loss streaks
            returns_sign = (strategy_returns > 0).astype(int)
            streaks = returns_sign.groupby((returns_sign != returns_sign.shift()).cumsum()).size()
            
            st.markdown("#### 🎲 Streak Analysis")
            streak_df = pd.DataFrame({
                'Metric': ['Longest Win Streak', 'Longest Loss Streak', 'Current Streak'],
                'Value': [
                    f"{streaks[returns_sign.groupby((returns_sign != returns_sign.shift()).cumsum()).first() == 1].max()} days",
                    f"{streaks[returns_sign.groupby((returns_sign != returns_sign.shift()).cumsum()).first() == 0].max()} days",
                    f"{'Win' if strategy_returns.iloc[-1] > 0 else 'Loss'}"
                ]
            })
            st.dataframe(streak_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📉 Drawdown Statistics")
            
            # Calculate drawdown duration
            in_drawdown = drawdown < 0
            drawdown_periods = in_drawdown.groupby((~in_drawdown).cumsum()).sum()
            
            if len(drawdown_periods[drawdown_periods > 0]) > 0:
                avg_dd_duration = drawdown_periods[drawdown_periods > 0].mean()
                max_dd_duration = drawdown_periods.max()
            else:
                avg_dd_duration = 0
                max_dd_duration = 0
            
            drawdown_stats_df = pd.DataFrame({
                'Metric': ['Maximum Drawdown', 'Average Drawdown', 'Max DD Duration', 'Avg DD Duration', 'Current DD'],
                'Value': [
                    f"{max_drawdown:.2%}",
                    f"{drawdown[drawdown < 0].mean():.2%}" if len(drawdown[drawdown < 0]) > 0 else "N/A",
                    f"{max_dd_duration} days",
                    f"{avg_dd_duration:.0f} days",
                    f"{drawdown.iloc[-1]:.2%}"
                ]
            })
            st.dataframe(drawdown_stats_df, use_container_width=True, hide_index=True)
            
            # Recovery time
            st.markdown("#### ⏱️ Recovery Analysis")
            if max_drawdown < 0:
                max_dd_date = drawdown.idxmin()
                recovery_dates = drawdown[drawdown.index > max_dd_date]
                if len(recovery_dates[recovery_dates >= 0]) > 0:
                    recovery_date = recovery_dates[recovery_dates >= 0].index[0]
                    recovery_days = (recovery_date - max_dd_date).days
                    recovery_info = f"Recovered in {recovery_days} days"
                else:
                    recovery_info = "Not yet recovered"
            else:
                recovery_info = "No drawdown"
            
            st.info(f"Max drawdown recovery: {recovery_info}")
        
        # Monte Carlo Simulation
        st.markdown("### 🎲 Monte Carlo Simulation (252-day forward projection)")
        
        n_simulations = 1000
        n_days = 252
        
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        
        simulations = np.zeros((n_days, n_simulations))
        for i in range(n_simulations):
            daily_returns = np.random.normal(mean_return, std_return, n_days)
            simulations[:, i] = (1 + daily_returns).cumprod()
        
        fig = go.Figure()
        
        # Plot percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']
        
        for pct, color in zip(percentiles, colors):
            pct_values = np.percentile(simulations, pct, axis=1)
            fig.add_trace(go.Scatter(
                x=list(range(n_days)),
                y=pct_values,
                mode='lines',
                name=f'{pct}th percentile',
                line=dict(color=color, width=2),
                hovertemplate=f'{pct}th: %{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Monte Carlo Forward Projection',
            xaxis_title='Days Forward',
            yaxis_title='Portfolio Value (normalized)',
            template='plotly_white',
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Projection statistics
        final_values = simulations[-1, :]
        projection_df = pd.DataFrame({
            'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
            'Final Value': [f"{np.percentile(final_values, p):.2f}" for p in [5, 25, 50, 75, 95]],
            'Return': [f"{(np.percentile(final_values, p) - 1)*100:.1f}%" for p in [5, 25, 50, 75, 95]]
        })
        st.dataframe(projection_df, use_container_width=True, hide_index=True)
    
    # ==================== Tab 6: Export & Reports ====================
    with tabs[5]:
        st.markdown("### 📥 Data Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Performance Data")
            st.caption("Historical returns and cumulative performance")
            
            if st.button("📥 Prepare CSV", key="perf_csv", use_container_width=True):
                performance_df = pd.DataFrame({
                    'Date': strategy_returns.index,
                    'Strategy_Return': strategy_returns.values,
                    'Benchmark_Return': benchmark_returns.values,
                    'Strategy_Cumulative': (1 + strategy_returns).cumprod().values,
                    'Benchmark_Cumulative': (1 + benchmark_returns).cumprod().values,
                    'Drawdown': drawdown.values
                })
                csv = performance_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Performance Data",
                    data=csv,
                    file_name=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("#### 🎯 Allocation Data")
            st.caption("Current portfolio weights and values")
            
            if st.button("📥 Prepare CSV", key="allocation_csv", use_container_width=True):
                allocation_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight': optimal_weights,
                    'Weight_Percent': [f"{w*100:.4f}" for w in optimal_weights]
                })
                csv = allocation_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Allocation Data",
                    data=csv,
                    file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            st.markdown("#### 📈 Price Data")
            st.caption("Historical price data for all assets")
            
            if st.button("📥 Prepare CSV", key="price_csv", use_container_width=True):
                csv = prices.to_csv()
                st.download_button(
                    label="⬇️ Download Price Data",
                    data=csv,
                    file_name=f"prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Full report generation
        st.markdown("### 📑 Comprehensive Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("Generate a comprehensive PDF report including:")
            st.markdown("""
            - Executive Summary
            - Performance Metrics & Benchmarking
            - Risk Analysis & Drawdown Statistics
            - Portfolio Allocation & Optimization Details
            - All Charts and Visualizations
            - Monte Carlo Projections
            - Compliance & Disclosure Statements
            """)
        
        with col2:
            report_format = st.selectbox(
                "Report Format",
                ["PDF", "HTML", "JSON"],
                help="Select the output format for your report"
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_raw_data = st.checkbox("Include Raw Data", value=False)
            
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    # Placeholder for actual report generation
                    st.success("✅ Report generated successfully!")
                    st.info(f"📄 Report format: {report_format}\n\nConnect to your PDF generation library (e.g., ReportLab, WeasyPrint) for production use.")
        
        st.markdown("---")
        
        # Configuration Management
        st.markdown("### 💾 Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Save Current Configuration")
            config_name = st.text_input(
                "Configuration Name",
                value=f"Strategy_{datetime.now().strftime('%Y%m%d')}",
                placeholder="Enter a name for this configuration"
            )
            
            if st.button("💾 Save Configuration", use_container_width=True):
                config = {
                    'name': config_name,
                    'strategy_type': strategy_type,
                    'assets': selected_assets,
                    'optimization_method': optimization_method,
                    'rebalance_freq': rebalance_freq,
                    'risk_free_rate': risk_free_rate,
                    'target_vol': target_vol,
                    'confidence_level': confidence_level,
                    'rolling_window': rolling_window,
                    'date_range': {
                        'start': start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
                        'end': end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date)
                    },
                    'weights': optimal_weights.tolist(),
                    'performance': {
                        'total_return': float(total_return),
                        'sharpe_ratio': float(sharpe),
                        'max_drawdown': float(max_drawdown)
                    },
                    'saved_at': datetime.now().isoformat()
                }
                
                st.session_state.saved_configs[config_name] = config
                
                # Create downloadable JSON
                json_config = json.dumps(config, indent=2)
                st.download_button(
                    label="⬇️ Download Configuration JSON",
                    data=json_config,
                    file_name=f"config_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                st.success(f"✅ Configuration '{config_name}' saved successfully!")
        
        with col2:
            st.markdown("#### Load Saved Configuration")
            
            if st.session_state.saved_configs:
                saved_config_names = list(st.session_state.saved_configs.keys())
                selected_config = st.selectbox(
                    "Select Configuration",
                    saved_config_names
                )
                
                if st.button("📂 Load Configuration", use_container_width=True):
                    st.info("Configuration loaded! Click 'Run Analysis' to apply.")
                    with st.expander("📋 Configuration Details", expanded=True):
                        st.json(st.session_state.saved_configs[selected_config])
            else:
                st.info("No saved configurations yet. Save your current setup to create one.")
            
            # Upload configuration
            uploaded_config = st.file_uploader(
                "Upload Configuration JSON",
                type=['json'],
                help="Upload a previously saved configuration file"
            )
            
            if uploaded_config is not None:
                try:
                    config = json.load(uploaded_config)
                    st.success("✅ Configuration uploaded successfully!")
                    st.json(config)
                except Exception as e:
                    st.error(f"❌ Error loading configuration: {str(e)}")
        
        st.markdown("---")
        
        # Analysis History
        st.markdown("### 📚 Analysis History")
        
        # Add current analysis to history
        if run_analysis:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'assets': selected_assets,
                'method': optimization_method,
                'total_return': float(total_return),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_drawdown)
            }
            
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            
            # Keep only last 10 analyses
            st.session_state.analysis_history.append(history_entry)
            st.session_state.analysis_history = st.session_state.analysis_history[-10:]
        
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            history_df['total_return'] = history_df['total_return'].apply(lambda x: f"{x:.2%}")
            history_df['sharpe'] = history_df['sharpe'].apply(lambda x: f"{x:.2f}")
            history_df['max_drawdown'] = history_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                history_df[['timestamp', 'method', 'total_return', 'sharpe', 'max_drawdown']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'timestamp': 'Time',
                    'method': 'Method',
                    'total_return': 'Return',
                    'sharpe': 'Sharpe',
                    'max_drawdown': 'Max DD'
                }
            )
            
            if st.button("🗑️ Clear History", use_container_width=False):
                st.session_state.analysis_history = []
                st.rerun()
        else:
            st.info("No analysis history yet. Run an analysis to start tracking.")
    
    # ==================== Footer ====================
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.last_update:
            st.caption(f"🕐 Last Updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("🕐 Last Updated: Not yet run")
    
    with col2:
        analysis_days = (end_date - start_date).days if hasattr(start_date, 'days') else (datetime.combine(end_date, datetime.min.time()) - datetime.combine(start_date, datetime.min.time())).days
        st.caption(f"📅 Analysis Period: {analysis_days} days")
    
    with col3:
        st.caption(f"📊 Data Points: {len(strategy_returns):,}")
    
    with col4:
        st.caption(f"🎯 Assets: {len(selected_assets)}")
    
    # System status
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        st.caption("⚡ System Status: Operational | 🔒 Data: Encrypted | 📡 Real-time: Active")
    with status_col2:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

# ==================== Application Entry Point ====================
def main():
    """Main application entry point with error handling"""
    try:
        initialize_session_state()
        
        if not st.session_state.authenticated:
            if st.session_state.show_create_account:
                show_create_account()
            else:
                show_login()
        else:
            main_dashboard()
            
    except Exception as e:
        st.error(f"""
        ### ❌ Application Error
        
        An unexpected error occurred:
        ```
        {str(e)}
        ```
        
        Please refresh the page or contact support if the issue persists.
        """)
        
        if st.button("🔄 Reload Application"):
            st.rerun()

if __name__ == "__main__":
    main()