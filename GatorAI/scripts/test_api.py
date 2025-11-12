#!/usr/bin/env python3
"""
Simple test script to verify the API is working.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_fetch_data():
    """Test data fetch endpoint."""
    print("\nTesting data fetch endpoint...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/fetch",
            json={
                "tickers": ["SPY", "QQQ"],
                "interval": "1d",
                "refresh": False
            },
            timeout=30
        )
        if response.status_code == 200:
            print("✅ Data fetch initiated")
            return True
        else:
            print(f"❌ Data fetch failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return False

def test_get_prices():
    """Test get prices endpoint."""
    print("\nTesting get prices endpoint...")
    try:
        # Wait a bit for data to be fetched
        time.sleep(3)
        response = requests.get(
            f"{API_BASE_URL}/data/prices",
            params={"tickers": "SPY,QQQ"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Get prices successful")
                print(f"   Found data for {len(data.get('data', {}).get('prices', {}))} tickers")
                return True
            else:
                print(f"❌ Get prices failed: {data}")
                return False
        else:
            print(f"❌ Get prices failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Get prices error: {e}")
        return False

def test_get_features():
    """Test get features endpoint."""
    print("\nTesting get features endpoint...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/data/features",
            params={"tickers": "SPY"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Get features successful")
                return True
            else:
                print(f"❌ Get features failed: {data}")
                return False
        else:
            print(f"❌ Get features failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Get features error: {e}")
        return False

def test_predict():
    """Test predict endpoint."""
    print("\nTesting predict endpoint...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "tickers": ["SPY", "QQQ"],
                "retrain": False,
                "blend": 0.5
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Predict successful")
                predictions = data.get("predictions", {})
                print(f"   Generated predictions for {len(predictions)} tickers")
                return True
            else:
                print(f"❌ Predict failed: {data}")
                return False
        else:
            print(f"❌ Predict failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Predict error: {e}")
        return False

def test_backtest():
    """Test backtest endpoint."""
    print("\nTesting backtest endpoint...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/backtest",
            json={
                "tickers": ["SPY", "QQQ"],
                "strategy": "equal_weight",
                "strategy_params": {},
                "rebalance": "monthly",
                "cost_bps": 5.0,
                "slippage": 0.001,
                "use_ml_predictions": False
            },
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Backtest successful")
                stats = data.get("results", {}).get("stats", {})
                print(f"   CAGR: {stats.get('cagr', 0):.2%}")
                print(f"   Sharpe: {stats.get('sharpe', 0):.2f}")
                return True
            else:
                print(f"❌ Backtest failed: {data}")
                return False
        else:
            print(f"❌ Backtest failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return False

def test_optimize():
    """Test optimize endpoint."""
    print("\nTesting optimize endpoint...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize",
            json={
                "tickers": ["SPY", "QQQ"],
                "method": "mean_variance",
                "risk_aversion": 1.0,
                "long_only": True,
                "use_ml_predictions": False
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Optimize successful")
                weights = data.get("weights", {})
                print(f"   Optimized weights: {weights}")
                return True
            else:
                print(f"❌ Optimize failed: {data}")
                return False
        else:
            print(f"❌ Optimize failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Optimize error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("GatorAI API Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_fetch_data,
        test_get_prices,
        test_get_features,
        test_predict,
        test_backtest,
        test_optimize
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test {test.__name__} raised exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total - passed} test(s) failed")

if __name__ == "__main__":
    main()

