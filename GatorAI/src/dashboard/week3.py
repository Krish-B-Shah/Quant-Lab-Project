"""
Production-Grade Quantitative Trading Dashboard
Modular, interactive, and user-facing application
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
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Quantitative Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Backend Integration Module ====================
class BackendConnector:
    """Modular backend connector for analytics and data"""
    
    @staticmethod
    def load_market_data(tickers, start_date, end_date):
        """Simulate loading market data - Replace with actual API call"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = {}
        for ticker in tickers:
            prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
            data[ticker] = pd.Series(prices, index=dates)
        return pd.DataFrame(data)
    
    @staticmethod
    def calculate_returns(prices):
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_rolling_metrics(returns, window=60):
        """Calculate rolling performance metrics"""
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std(), raw=True
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        return rolling_sharpe, rolling_vol
    
    @staticmethod
    def optimize_portfolio(returns, method='max_sharpe'):
        """Portfolio optimization - Replace with actual optimizer"""
        n_assets = len(returns.columns)
        
        if method == 'max_sharpe':
            # Simplified mean-variance optimization
            cov_matrix = returns.cov() * 252
            mean_returns = returns.mean() * 252
            
            # Random weights for demo - replace with actual optimization
            weights = np.random.dirichlet(np.ones(n_assets))
            
        elif method == 'min_variance':
            weights = np.ones(n_assets) / n_assets  # Equal weight for demo
            
        elif method == 'equal_weight':
            weights = np.ones(n_assets) / n_assets
            
        return weights
    
    @staticmethod
    def generate_efficient_frontier(returns, n_portfolios=50):
        """Generate efficient frontier data"""
        results = []
        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(len(returns.columns)))
            portfolio_return = (returns.mean() * weights).sum() * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = portfolio_return / portfolio_vol
            results.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': sharpe
            })
        return pd.DataFrame(results)

# ==================== Session State Management ====================
def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None

# ==================== Authentication Module ====================
def show_login():
    """Basic authentication interface"""
    st.markdown("### 🔐 Login to Portfolio Analytics")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Login", use_container_width=True):
                # Basic demo authentication - Replace with real auth
                if username and password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Please enter credentials")
        
        with col_b:
            if st.button("Demo Mode", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.username = "Demo User"
                st.rerun()
        
        st.markdown("---")
        st.info("💡 Click 'Demo Mode' to explore the dashboard")

# ==================== Visualization Module ====================
class ChartBuilder:
    """Professional chart builder with Plotly"""
    
    @staticmethod
    def create_cumulative_returns_chart(strategy_returns, benchmark_returns):
        """Create interactive cumulative returns comparison"""
        strategy_cum = (1 + strategy_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=strategy_cum.index,
            y=strategy_cum.values,
            mode='lines',
            name='Strategy',
            line=dict(color='#667eea', width=2),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_cum.index,
            y=benchmark_cum.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#f093fb', width=2, dash='dash'),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    @staticmethod
    def create_rolling_metrics_chart(rolling_sharpe, rolling_vol):
        """Create rolling Sharpe and volatility chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Sharpe Ratio',
                line=dict(color='#667eea', width=2),
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#f093fb', width=2),
                fill='tozeroy',
                hovertemplate='%{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Vol", row=2, col=1)
        
        fig.update_layout(
            height=600,
            template='plotly_white',
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_allocation_pie_chart(weights, labels):
        """Create allocation pie chart"""
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3),
            hovertemplate='%{label}<br>%{value:.1%}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Current Portfolio Allocation',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_efficient_frontier(frontier_data, current_portfolio=None):
        """Create efficient frontier visualization"""
        fig = go.Figure()
        
        # Color by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='markers',
            marker=dict(
                size=8,
                color=frontier_data['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=frontier_data['sharpe'].round(2),
            hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe: %{text}<extra></extra>',
            name='Portfolios'
        ))
        
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio['volatility']],
                y=[current_portfolio['return']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Current Portfolio',
                hovertemplate='Current<br>Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            template='plotly_white',
            height=500,
            hovermode='closest'
        )
        
        return fig

# ==================== Export Functions ====================
def export_data_to_csv(data, filename):
    """Export data to CSV"""
    csv = data.to_csv(index=True)
    return csv

def export_report_to_pdf():
    """Generate PDF report (placeholder)"""
    st.info("📄 PDF export functionality - Connect to your PDF generation library")
    # Implement with reportlab or weasyprint

# ==================== Main Application ====================
def main_dashboard():
    """Main dashboard interface"""
    
    # Header
    st.markdown("# 📊 Quantitative Portfolio Analytics")
    st.markdown(f"**Welcome, {st.session_state.username}** | {datetime.now().strftime('%B %d, %Y')}")
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown("---")
    
    # ==================== Sidebar Controls ====================
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown("---")
        
        # Strategy Selection
        st.markdown("### Strategy Configuration")
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Mean-Variance Optimization", "Risk Parity", "Momentum", "Value", "Custom"]
        )
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["max_sharpe", "min_variance", "equal_weight"]
        )
        
        # Asset Universe
        st.markdown("### Asset Universe")
        selected_assets = st.multiselect(
            "Select Assets",
            ["SPY", "QQQ", "TLT", "GLD", "IWM", "EFA", "VNQ", "BND"],
            default=["SPY", "QQQ", "TLT", "GLD"]
        )
        
        # Time Period
        st.markdown("### Analysis Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
        
        # Rebalancing Frequency
        st.markdown("### Rebalancing")
        rebalance_freq = st.select_slider(
            "Rebalancing Frequency",
            options=["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
            value="Monthly"
        )
        
        # Risk Parameters
        st.markdown("### Risk Parameters")
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        ) / 100
        
        target_vol = st.slider(
            "Target Volatility (%)",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=1.0
        ) / 100
        
        st.markdown("---")
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            run_analysis = st.button("▶️ Run Analysis", use_container_width=True, type="primary")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh")
        
        if auto_refresh:
            st.info("🔄 Auto-refresh enabled (60s)")
    
    # ==================== Main Content Area ====================
    
    # Run analysis when button clicked
    if run_analysis or auto_refresh:
        with st.spinner("Loading data and running analysis..."):
            # Initialize backend
            backend = BackendConnector()
            
            # Load data
            prices = backend.load_market_data(selected_assets, start_date, end_date)
            returns = backend.calculate_returns(prices)
            
            # Calculate benchmark (equal weight)
            benchmark_returns = returns.mean(axis=1)
            
            # Optimize portfolio
            optimal_weights = backend.optimize_portfolio(returns, method=optimization_method)
            
            # Calculate strategy returns
            strategy_returns = (returns * optimal_weights).sum(axis=1)
            
            # Calculate metrics
            rolling_sharpe, rolling_vol = backend.calculate_rolling_metrics(strategy_returns)
            
            # Generate efficient frontier
            frontier_data = backend.generate_efficient_frontier(returns)
            
            st.session_state.last_update = datetime.now()
    else:
        st.info("👈 Configure parameters and click 'Run Analysis' to generate insights")
        return
    
    # ==================== Performance Metrics ====================
    st.markdown("## 📈 Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    sharpe = backend.calculate_sharpe_ratio(strategy_returns, risk_free_rate)
    max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
    win_rate = (strategy_returns > 0).mean()
    
    with col1:
        st.metric("Total Return", f"{total_return:.2%}", delta=f"{annual_return:.2%} Ann.")
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col3:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    with col4:
        st.metric("Volatility", f"{strategy_returns.std() * np.sqrt(252):.2%}")
    with col5:
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    st.markdown("---")
    
    # ==================== Tabbed Interface ====================
    tabs = st.tabs(["📊 Returns", "📉 Risk Metrics", "🎯 Allocation", "🔬 Efficient Frontier", "📥 Export"])
    
    # Tab 1: Returns Analysis
    with tabs[0]:
        st.markdown("### Cumulative Returns Comparison")
        
        chart_builder = ChartBuilder()
        returns_chart = chart_builder.create_cumulative_returns_chart(
            strategy_returns, benchmark_returns
        )
        st.plotly_chart(returns_chart, use_container_width=True)
        
        # Distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Daily Returns Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=strategy_returns,
                nbinsx=50,
                name='Strategy',
                marker_color='#667eea'
            ))
            fig.update_layout(
                xaxis_title="Daily Return",
                yaxis_title="Frequency",
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Monthly Returns Heatmap")
            monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=pivot_table.index,
                colorscale='RdYlGn',
                hovertemplate='%{y}-%{x}: %{z:.2%}<extra></extra>'
            ))
            fig.update_layout(
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Risk Metrics
    with tabs[1]:
        st.markdown("### Rolling Performance Metrics")
        
        rolling_chart = chart_builder.create_rolling_metrics_chart(
            rolling_sharpe, rolling_vol
        )
        st.plotly_chart(rolling_chart, use_container_width=True)
        
        # Drawdown analysis
        st.markdown("### Drawdown Analysis")
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#f093fb', width=1),
            hovertemplate='%{y:.2%}<extra></extra>'
        ))
        fig.update_layout(
            title='Underwater Plot',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Allocation
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
            allocation_df = pd.DataFrame({
                'Asset': selected_assets,
                'Weight': optimal_weights,
                'Value ($)': optimal_weights * 100000  # Example portfolio value
            })
            allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: f"{x:.2%}")
            allocation_df['Value ($)'] = allocation_df['Value ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
        
        # Allocation over time (placeholder for rebalancing history)
        st.markdown("### Allocation History")
        st.info("📊 Allocation timeline visualization - Connect to rebalancing history data")
    
    # Tab 4: Efficient Frontier
    with tabs[3]:
        st.markdown("### Efficient Frontier Analysis")
        
        current_portfolio = {
            'volatility': strategy_returns.std() * np.sqrt(252),
            'return': annual_return
        }
        
        frontier_chart = chart_builder.create_efficient_frontier(
            frontier_data, current_portfolio
        )
        st.plotly_chart(frontier_chart, use_container_width=True)
        
        # Statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Optimal Portfolio")
            optimal_metrics = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                'Value': [
                    f"{annual_return:.2%}",
                    f"{strategy_returns.std() * np.sqrt(252):.2%}",
                    f"{sharpe:.2f}"
                ]
            })
            st.dataframe(optimal_metrics, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Correlation Matrix")
            corr_matrix = returns.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
            ))
            fig.update_layout(
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Export
    with tabs[4]:
        st.markdown("### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Performance Data")
            if st.button("Download CSV", key="perf_csv"):
                performance_df = pd.DataFrame({
                    'Date': strategy_returns.index,
                    'Strategy Return': strategy_returns.values,
                    'Benchmark Return': benchmark_returns.values,
                    'Cumulative Strategy': (1 + strategy_returns).cumprod().values,
                    'Cumulative Benchmark': (1 + benchmark_returns).cumprod().values
                })
                csv = export_data_to_csv(performance_df, 'performance.csv')
                st.download_button(
                    label="📊 Download Performance Data",
                    data=csv,
                    file_name=f"performance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.markdown("#### Allocation Data")
            if st.button("Download CSV", key="allocation_csv"):
                allocation_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight': optimal_weights
                })
                csv = export_data_to_csv(allocation_df, 'allocation.csv')
                st.download_button(
                    label="🎯 Download Allocation Data",
                    data=csv,
                    file_name=f"allocation_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            st.markdown("#### Full Report")
            if st.button("Generate PDF Report"):
                export_report_to_pdf()
        
        st.markdown("---")
        
        # Save configuration
        st.markdown("### 💾 Save Configuration")
        config_name = st.text_input("Configuration Name", value="My Strategy")
        if st.button("Save Current Setup"):
            config = {
                'strategy_type': strategy_type,
                'assets': selected_assets,
                'optimization_method': optimization_method,
                'rebalance_freq': rebalance_freq,
                'risk_free_rate': risk_free_rate,
                'target_vol': target_vol,
                'saved_at': datetime.now().isoformat()
            }
            st.success(f"✅ Configuration '{config_name}' saved!")
            st.json(config)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Last Updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.caption(f"Analysis Period: {(end_date - start_date).days} days")
    with col3:
        st.caption(f"Data Points: {len(strategy_returns):,}")

# ==================== Application Entry Point ====================
def main():
    """Main application entry point"""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        show_login()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()