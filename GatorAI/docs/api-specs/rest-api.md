# REST API Specifications

This document defines the REST API endpoints for the GatorAI platform.

## Base URL

```
https://api.gatorai.com/v1
```

## Authentication

All API requests require authentication via Bearer token:

```
Authorization: Bearer <your-api-key>
```

## Data Endpoints

### GET /data/prices

Retrieve historical price data for specified tickers.

**Parameters:**
- `tickers` (required): Comma-separated list of ticker symbols
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `interval` (optional): Data interval (1d, 1h, 1m) - default: 1d

**Response:**
```json
{
  "status": "success",
  "data": {
    "SPY": [
      {
        "datetime": "2023-01-01T00:00:00Z",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 103.0,
        "volume": 1000000
      }
    ]
  }
}
```

### POST /data/fetch

Trigger data fetching for specified tickers.

**Request Body:**
```json
{
  "tickers": ["SPY", "QQQ", "IWM"],
  "interval": "1d",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Data fetch initiated",
  "job_id": "fetch_12345"
}
```

### GET /data/features

Retrieve computed technical features for tickers.

**Parameters:**
- `tickers` (required): Comma-separated list of ticker symbols
- `features` (required): Comma-separated list of features (rsi, macd, bollinger)
- `period` (optional): Lookback period for features

## Backtesting Endpoints

### POST /backtest/run

Execute a backtest with specified parameters.

**Request Body:**
```json
{
  "strategy": {
    "name": "momentum",
    "config": {
      "lookback": 90,
      "long_only": true
    }
  },
  "tickers": ["SPY", "QQQ", "IWM"],
  "rebalance": "monthly",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "cost_bps": 5,
  "slippage": 0.1
}
```

**Response:**
```json
{
  "status": "success",
  "backtest_id": "bt_12345",
  "results": {
    "equity_curve": {
      "2020-01-01": 1.0,
      "2020-02-01": 1.05
    },
    "stats": {
      "cagr": 0.12,
      "sharpe": 1.8,
      "max_drawdown": -0.15,
      "total_return": 0.25
    }
  }
}
```

### GET /backtest/{backtest_id}

Retrieve backtest results by ID.

### GET /backtest/{backtest_id}/performance

Get detailed performance metrics for a backtest.

## Optimization Endpoints

### POST /optimize/portfolio

Run portfolio optimization with specified method.

**Request Body:**
```json
{
  "method": "mean_variance",
  "tickers": ["SPY", "QQQ", "IWM"],
  "constraints": {
    "long_only": true,
    "max_weight": 0.4,
    "min_weight": 0.0
  },
  "risk_aversion": 1.0
}
```

**Response:**
```json
{
  "status": "success",
  "optimization_id": "opt_12345",
  "weights": {
    "SPY": 0.5,
    "QQQ": 0.3,
    "IWM": 0.2
  },
  "expected_return": 0.08,
  "expected_volatility": 0.12,
  "sharpe_ratio": 0.67
}
```

### GET /optimize/{optimization_id}

Retrieve optimization results by ID.

## Strategy Endpoints

### GET /strategies

List available trading strategies.

**Response:**
```json
{
  "status": "success",
  "strategies": [
    {
      "name": "equal_weight",
      "description": "Equal weight allocation across all assets",
      "parameters": []
    },
    {
      "name": "momentum",
      "description": "Momentum-based strategy",
      "parameters": [
        {
          "name": "lookback",
          "type": "integer",
          "default": 90,
          "min": 5,
          "max": 252
        }
      ]
    }
  ]
}
```

### POST /strategies/{strategy_name}/signals

Generate signals for a strategy.

**Request Body:**
```json
{
  "tickers": ["SPY", "QQQ"],
  "parameters": {
    "lookback": 60
  },
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

## Error Responses

All endpoints return standardized error responses:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid ticker symbol",
    "details": {
      "ticker": "INVALID"
    }
  }
}
```

## Rate Limits

- 1000 requests per hour for data endpoints
- 100 requests per hour for backtesting endpoints
- 50 requests per hour for optimization endpoints

## WebSocket Streams (Future)

Real-time data streaming will be available via WebSocket:

```
ws://api.gatorai.com/v1/stream
```

Supported streams:
- `prices.{ticker}`: Real-time price updates
- `backtest.{backtest_id}`: Backtest progress updates
- `optimization.{optimization_id}`: Optimization progress updates
