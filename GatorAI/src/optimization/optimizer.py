from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.linalg import inv
import cvxpy as cp
from scipy.optimize import minimize

'''
Mean–Variance Optimization (Markowitz) – maximize return per unit of risk.
Black–Litterman Model – adjust market equilibrium returns with investor views.
Risk Parity – equalize each asset’s contribution to portfolio risk.
CVaR Optimization – minimize expected loss in the worst scenarios.
Equal Risk Contribution – a simplified version of risk parity.
Monte Carlo Simulation – test robustness under simulated future returns.
'''


# Mean-Variance (Markowitz)
def mean_variance_optimize(
        returns: pd.DataFrame,
        risk_aversion: float = 1.0,
        long_only: bool = True,
        weights_sum_to_one: bool = True,
        epsilon: float = 1e-8,
) -> pd.Series:
    mu = returns.mean()
    cov = returns.cov() + np.eye(len(mu)) * epsilon
    inv_cov = pd.DataFrame(inv(cov.values), index=cov.index, columns=cov.columns)
    raw = inv_cov @ mu
    w = raw / raw.sum()
    if long_only:
        w = w.clip(lower=0.0)
        if w.sum() == 0:
            w = pd.Series(np.full_like(w, 1.0 / len(w)), index=w.index)
        else:
            w = w / w.sum()
    if weights_sum_to_one:
        w = w / w.sum()
    return w


def placeholder_weights():
    portfolio_weights = {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}
    """Generate dummy portfolio weights for testing.

     	Returns:
        Series with portfolio weights: SPY=50%, QQQ=30%, IWM=20%
    """
    return pd.Series(portfolio_weights)


# Black–Litterman Model
def black_litterman_optimize(
        returns: pd.DataFrame,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        tau: float = 0.05,
        omega: Optional[np.ndarray] = None,
        market_weights: Optional[pd.Series] = None,
        risk_aversion: float = 2.5,
) -> pd.Series:
    """
    Black–Litterman implementation with proper μ_bl integration.
    Σ	Covariance matrix
    τ	Scalar controlling uncertainty of market equilibrium
    π	Market equilibrium returns (implied by market weights)
    P	View matrix (which assets/views you have opinions on)
    Q	Expected returns for those views
    Ω	Uncertainty in your views
    """
    Σ = returns.cov().values
    n = Σ.shape[0]
    assets = returns.columns

    # 1: Market equilibrium (implied) returns
    if market_weights is None:
        market_weights = pd.Series(np.ones(n) / n, index=assets)
    π = risk_aversion * Σ @ market_weights.values

    # 2: Combine with investor views, (if there are any)
    if P is None or Q is None:
        μ_bl = π
    else:
        if omega is None:
            omega = np.diag(np.diag(P @ (tau * Σ) @ P.T))
        μ_bl = np.linalg.inv(np.linalg.inv(tau * Σ) + P.T @ np.linalg.inv(omega) @ P) @ (
                np.linalg.inv(tau * Σ) @ π + P.T @ np.linalg.inv(omega) @ Q
        )

    μ_bl = pd.Series(μ_bl, index=assets)

    # 3: Compute weights directly using μ_bl
    cov = pd.DataFrame(Σ, index=assets, columns=assets)
    inv_cov = pd.DataFrame(np.linalg.inv(cov.values), index=assets, columns=assets)
    raw = inv_cov @ μ_bl
    w = raw / raw.sum()

    # 4: Optional constraints
    w = w.clip(lower=0)
    w = w / w.sum()
    return w


# Risk Parity Optimization
def risk_parity_optimize(returns: pd.DataFrame, long_only: bool = True) -> pd.Series:
    Σ = returns.cov().values
    assets = returns.columns
    n = len(assets)

    def risk_contribution(weights):
        port_var = weights @ Σ @ weights
        mrc = Σ @ weights
        rc = weights * mrc
        return rc / port_var

    def objective(weights):
        rc = risk_contribution(weights)
        return np.sum((rc - rc.mean()) ** 2)

    x0 = np.ones(n) / n
    bounds = [(0, 1) if long_only else (None, None)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(objective, x0, bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=assets)


# Conditional Value at Risk (CVaR)
def cvar_optimize(
        returns: pd.DataFrame,
        alpha: float = 0.95,
        target_return: Optional[float] = None,
        long_only: bool = True,
) -> pd.Series:
    """
    Minimize CVaR subject to constraints.
    """
    T, n = returns.shape
    R = returns.values
    w = cp.Variable(n)
    z = cp.Variable()
    u = cp.Variable(T)

    constraints = [
        u >= -R @ w - z,
        u >= 0,
        cp.sum(w) == 1,
    ]
    if long_only:
        constraints.append(w >= 0)
    if target_return is not None:
        μ = returns.mean().values
        constraints.append(μ @ w >= target_return)

    obj = cp.Minimize(z + (1 / ((1 - alpha) * T)) * cp.sum(u))
    cp.Problem(obj, constraints).solve(solver=cp.SCS, verbose=False)
    return pd.Series(np.array(w.value).flatten(), index=returns.columns)


# Equal-Risk-Contribution (wrapper for risk parity)
def equal_risk_contribution(returns: pd.DataFrame) -> pd.Series:
    return risk_parity_optimize(returns)


# Monte Carlo Robustness Simulations
def monte_carlo_simulation(
        optimizer_fn,
        returns: pd.DataFrame,
        n_sims: int = 100,
        horizon: int = 252,
        random_seed: Optional[int] = None,
        **optimizer_kwargs,
) -> pd.DataFrame:
    """
    Simulates many random return paths and re-optimizes each time
    Returns a DataFrame of simulated optimal weights.
    """
    rng = np.random.default_rng(random_seed)
    μ = returns.mean().values
    Σ = returns.cov().values
    assets = returns.columns
    sims = []

    for _ in range(n_sims):
        sim_data = pd.DataFrame(
            rng.multivariate_normal(μ, Σ, size=horizon),
            columns=assets,
        )
        w = optimizer_fn(sim_data, **optimizer_kwargs)
        sims.append(w)

    return pd.DataFrame(sims)


# Visualization Tools
import matplotlib.pyplot as plt
import seaborn as sns


# Plots return vs. risk tradeoff
def plot_efficient_frontier(returns: pd.DataFrame, n_points: int = 25):
    risk_levels = np.logspace(-2, 2, n_points)
    results = []
    for λ in risk_levels:
        w = mean_variance_optimize(returns, risk_aversion=λ)
        μp = float(w @ returns.mean())
        σp = float(np.sqrt(w.T @ returns.cov() @ w))
        results.append((σp, μp))
    x, y = zip(*results)
    plt.plot(x, y, "-o", label="Efficient Frontier")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier (Mean-Variance)")
    plt.legend()
    plt.grid(True)


# Shows how assets are correlated
def plot_correlation_heatmap(returns: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()


# Visualizes many simulations
def plot_risk_return_scatter(weights_df: pd.DataFrame, returns: pd.DataFrame):
    exp_returns = []
    risks = []
    for _, w in weights_df.iterrows():
        μp = float(w @ returns.mean())
        σp = float(np.sqrt(w.T @ returns.cov() @ w))
        exp_returns.append(μp)
        risks.append(σp)
    plt.scatter(risks, exp_returns, alpha=0.6)
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Monte Carlo Portfolio Risk/Return Scatter")
    plt.grid(True)