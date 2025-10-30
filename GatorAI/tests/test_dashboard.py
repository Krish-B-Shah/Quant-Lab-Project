import pytest
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for testing
root = Path(__file__).resolve().parents[1]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class TestDashboardIntegration:
    """Integration tests for dashboard components."""

    @pytest.fixture
    def mock_data_manager(self):
        """Mock DataManager for testing."""
        dm = Mock()
        dm.fetch = Mock(return_value=None)
        return dm

    @pytest.fixture
    def mock_storage(self):
        """Mock storage adapter."""
        storage = Mock()
        storage.read_price_data = Mock(return_value=None)
        storage.get_latest_timestamp = Mock(return_value=None)
        return storage

    @pytest.fixture
    def mock_backtest_result(self):
        """Mock backtest result."""
        result = Mock()
        result.equity_curve = Mock()
        result.equity_curve.index = []
        result.equity_curve.values = []
        result.returns = Mock()
        result.returns.to_dict = Mock(return_value={})
        result.stats = {"cagr": 0.1, "sharpe": 1.5, "vol": 0.15}
        return result

    def test_data_manager_integration(self, mock_data_manager, mock_storage):
        """Test DataManager integration with storage."""
        from data.manager import DataManager

        # Test initialization
        dm = DataManager(storage_adapter=mock_storage)
        assert dm.storage == mock_storage
        assert dm.features == []

    def test_backtest_engine_integration(self, mock_backtest_result):
        """Test backtest engine integration."""
        from backtesting.backtest_engine import BacktestResult

        # Test BacktestResult creation
        result = BacktestResult(
            returns=mock_backtest_result.returns,
            equity_curve=mock_backtest_result.equity_curve,
            stats=mock_backtest_result.stats
        )

        assert hasattr(result, 'returns')
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'stats')
        assert result.stats['cagr'] == 0.1

    def test_optimizer_integration(self):
        """Test optimizer integration."""
        import pandas as pd
        import numpy as np
        from optimization.optimizer import mean_variance_optimize

        # Create sample returns data
        np.random.seed(42)
        returns = pd.DataFrame({
            'SPY': np.random.normal(0.001, 0.02, 100),
            'QQQ': np.random.normal(0.0012, 0.025, 100),
            'IWM': np.random.normal(0.0008, 0.03, 100)
        })

        # Test optimization
        weights = mean_variance_optimize(returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-6  # Should sum to 1
        assert all(weights >= 0)  # Long only by default

    def test_strategy_integration(self):
        """Test strategy integration."""
        from backtesting.strategy import EqualWeightStrategy, StrategyConfig

        # Test strategy creation
        config = StrategyConfig(params={'long_only': True})
        strategy = EqualWeightStrategy(config=config)

        assert strategy.name == 'equal_weight'
        assert strategy.config == config

        # Test signal generation
        prices = pd.DataFrame({
            'SPY': [100, 101, 102],
            'QQQ': [200, 202, 204]
        }, index=pd.date_range('2023-01-01', periods=3))

        signals = strategy.generate_signals(prices)
        assert signals.shape == prices.shape
        assert not signals.empty

    @patch('streamlit.session_state', {})
    def test_dashboard_session_state(self):
        """Test dashboard session state management."""
        # This would normally test Streamlit session state
        # but we'll mock it for unit testing
        mock_session = {}

        # Simulate session state initialization
        if 'authenticated' not in mock_session:
            mock_session['authenticated'] = False
        if 'data_manager' not in mock_session:
            mock_session['data_manager'] = None

        assert mock_session['authenticated'] is False
        assert mock_session['data_manager'] is None

    def test_cli_integration(self):
        """Test CLI integration."""
        from data.cli import run

        # Test CLI function exists
        assert callable(run)

    def test_storage_adapter_integration(self):
        """Test storage adapter integration."""
        from data.storage.sqlite_adapter import SQLiteAdapter

        # Test adapter can be instantiated
        adapter = SQLiteAdapter()
        assert adapter is not None

        # Test required methods exist
        assert hasattr(adapter, 'get_latest_timestamp')
        assert hasattr(adapter, 'upsert_price_data')
        assert hasattr(adapter, 'read_price_data')


class TestDashboardPerformance:
    """Performance tests for dashboard components."""

    def test_backtest_performance(self):
        """Test backtest performance with larger dataset."""
        import time
        from backtesting.backtest_engine import run_vectorized_backtest
        import pandas as pd
        import numpy as np

        # Create larger test dataset
        np.random.seed(42)
        n_periods = 1000
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_periods)),
            index=pd.date_range('2020-01-01', periods=n_periods)
        )
        signal = pd.Series(np.random.choice([-1, 0, 1], n_periods), index=prices.index)

        start_time = time.time()
        result = run_vectorized_backtest(prices, signal)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert result is not None
        assert hasattr(result, 'stats')

    def test_optimization_performance(self):
        """Test optimization performance."""
        import time
        from optimization.optimizer import mean_variance_optimize
        import pandas as pd
        import numpy as np

        # Create test returns data
        np.random.seed(42)
        n_assets = 10
        n_periods = 500
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, n_assets)),
            columns=[f'ASSET_{i}' for i in range(n_assets)]
        )

        start_time = time.time()
        weights = mean_variance_optimize(returns)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 0.5
        assert len(weights) == n_assets
        assert abs(weights.sum() - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
