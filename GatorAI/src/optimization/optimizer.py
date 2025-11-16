from __future__ import annotations

from typing import Optional, Dict

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


# ---------------------------------------------------------
# Helpers: ML blending and constraint enforcement
# ---------------------------------------------------------
def get_effective_mu(
        returns: pd.DataFrame,
        predicted_returns: Optional[pd.Series] = None,
        blend: float = 0.5,
) -> pd.Series:
    """
    Combine historical mean returns and ML-predicted expected returns.

    blend = 0.0 -> purely historical
    blend = 1.0 -> purely ML predictions

    predicted_returns may be a pd.Series indexed by asset names; missing
    values are filled with historical means.
    """
    mu_hist = returns.mean()
    if predicted_returns is None:
        return mu_hist
    pred = predicted_returns.reindex(mu_hist.index).fillna(mu_hist)
    mu_eff = (1.0 - blend) * mu_hist + blend * pred
    return mu_eff


def _sector_index_map(assets: pd.Index, sector_map: Optional[Dict[str, str]]) -> Dict[str, np.ndarray]:
    """
    Build mapping sector -> numpy array of indices for assets in that sector.
    """
    if sector_map is None:
        return {}
    mapping: Dict[str, list] = {}
    for i, a in enumerate(assets):
        s = sector_map.get(a)
        if s is None:
            continue
        mapping.setdefault(s, []).append(i)
    return {k: np.array(v, dtype=int) for k, v in mapping.items()}


def _enforce_constraints_posthoc(
        w: pd.Series,
        *,
        long_only: bool = True,
        position_limit: Optional[float] = None,
        sector_caps: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        previous_weights: Optional[pd.Series] = None,
        max_turnover: Optional[float] = None,
        weights_sum_to_one: bool = True,
) -> pd.Series:
    """
    Apply practical constraints after an analytical or numerical solve.
    This includes:
     - enforce long_only (clip negatives)
     - apply per-position upper bound (position_limit)
     - apply sector caps by proportional scaling within sector if sum > cap
     - enforce turnover (simple proportional shrink towards previous weights to meet L1 cap)
     - renormalize to sum-to-one if requested
    """

    w = w.copy().astype(float).reindex(w.index).fillna(0.0)

    # long-only
    if long_only:
        w = w.clip(lower=0.0)

    # position limit
    if position_limit is not None:
        w = w.clip(upper=position_limit)

    # sector caps: proportional scaling within sector if over cap
    if sector_caps and sector_map:
        sectors = _sector_index_map(w.index, sector_map)
        for sector, idxs in sectors.items():
            cap = sector_caps.get(sector)
            if cap is None:
                continue
            names = w.index[idxs]
            total = float(w.loc[names].sum())
            if total > cap and total > 0:
                scale = cap / total
                w.loc[names] = w.loc[names] * scale

    # renormalize to 1 (if requested)
    if weights_sum_to_one:
        s = w.sum()
        if s == 0:
            # fallback to equal weight
            n = len(w)
            w = pd.Series(np.ones(n) / n, index=w.index)
        else:
            w = w / s

    # turnover enforcement (L1): if prev provided and max_turnover specified
    if max_turnover is not None and previous_weights is not None:
        prev = previous_weights.reindex(w.index).fillna(0.0)
        l1 = float((w - prev).abs().sum())
        if l1 > max_turnover and l1 > 0:
            # scale the difference to meet max_turnover
            factor = max_turnover / l1
            w = prev + (w - prev) * factor
            if weights_sum_to_one:
                s = w.sum()
                if s == 0:
                    w = pd.Series(np.ones(len(w)) / len(w), index=w.index)
                else:
                    w = w / s

    return w


# ---------------------------------------------------------
# Mean-Variance (Markowitz)
# ---------------------------------------------------------
def mean_variance_optimize(
        returns: pd.DataFrame,
        predicted_returns: Optional[pd.Series] = None,
        blend: float = 0.5,
        risk_aversion: float = 1.0,
        # regularization
        l2_reg: float = 0.0,
        l1_reg: float = 0.0,
        # risk model
        risk_model: str = "sample",
        factor_returns: Optional[pd.DataFrame] = None,
        long_only: bool = True,
        weights_sum_to_one: bool = True,
        epsilon: float = 1e-8,
        # real-world constraints:
        sector_caps: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        position_limit: Optional[float] = None,
        previous_weights: Optional[pd.Series] = None,
        max_turnover: Optional[float] = None,
) -> pd.Series:
    """
    Mean-Variance optimization, slightly modified to accept ML predicted returns
    and allow post-hoc enforcement of common production constraints for integration
    with dashboard/backtester.
    """
    assets = returns.columns
    n = len(assets)
    mu = get_effective_mu(returns, predicted_returns, blend).reindex(assets)
    cov = _get_covariance(returns, risk_model=risk_model, factor_returns=factor_returns, epsilon=epsilon)

    # cvxpy Markowitz with optional L1/L2 regularization
    w = cp.Variable(n)
    objective = (
        risk_aversion * cp.quad_form(w, cov.values)
        - mu.values @ w
        + l2_reg * cp.sum_squares(w)
        + l1_reg * cp.norm1(w)
    )
    constraints = []
    if weights_sum_to_one:
        constraints.append(cp.sum(w) == 1)
    if long_only:
        constraints.append(w >= 0)
    if position_limit is not None:
        constraints.append(w <= position_limit)
    if sector_caps and sector_map:
        sectors = _sector_index_map(assets, sector_map)
        for _, idxs in sectors.items():
            cap = sector_caps.get(_)
            if cap is None or len(idxs) == 0:
                continue
            constraints.append(cp.sum(w[idxs]) <= cap)
    if max_turnover is not None and previous_weights is not None:
        prev = previous_weights.reindex(assets).fillna(0.0).values
        constraints.append(cp.norm1(w - prev) <= max_turnover)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        w_out = pd.Series(np.ones(n) / n, index=assets)
    else:
        w_out = pd.Series(np.array(w.value).flatten(), index=assets)

    # Enforce remaining constraints and normalization cleanly
    w_out = _enforce_constraints_posthoc(
        w_out,
        long_only=long_only,
        position_limit=position_limit,
        sector_caps=sector_caps,
        sector_map=sector_map,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        weights_sum_to_one=weights_sum_to_one,
    )

    return w_out


def placeholder_weights():
    portfolio_weights = {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}
    """Generate dummy portfolio weights for testing.

     	Returns:
        Series with portfolio weights: SPY=50%, QQQ=30%, IWM=20%
    """
    return pd.Series(portfolio_weights)


# ---------------------------------------------------------
# Black–Litterman Model
# ---------------------------------------------------------
def black_litterman_optimize(
        returns: pd.DataFrame,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        tau: float = 0.05,
        omega: Optional[np.ndarray] = None,
        market_weights: Optional[pd.Series] = None,
        risk_aversion: float = 2.5,
        # ML integration
        predicted_returns: Optional[pd.Series] = None,
        blend: float = 0.5,
        # regularization
        l2_reg: float = 0.0,
        l1_reg: float = 0.0,
        # risk model
        risk_model: str = "sample",
        factor_returns: Optional[pd.DataFrame] = None,
        # constraints
        position_limit: Optional[float] = None,
        sector_caps: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        previous_weights: Optional[pd.Series] = None,
        max_turnover: Optional[float] = None,
        epsilon: float = 1e-8,
) -> pd.Series:
    """
    Black–Litterman implementation with proper μ_bl integration.
    Σ	Covariance matrix
    τ	Scalar controlling uncertainty of market equilibrium
    π	Market equilibrium returns (implied by market weights)
    P	View matrix (which assets/views you have opinions on)
    Q	Expected returns for those views
    Ω	Uncertainty in your views

    This version blends the Black-Litterman implied returns with ML predictions
    (if provided) via `blend` and applies simple production constraints post-hoc.
    """
    Σ = _get_covariance(returns, risk_model=risk_model, factor_returns=factor_returns, epsilon=epsilon).values
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

    # Blend BL returns with ML predictions (treat μ_bl as historical baseline if pred provided)
    if predicted_returns is None:
        mu = μ_bl
    else:
        # Blend predicted_returns with μ_bl: predicted has weight `blend`
        mu = (1.0 - blend) * μ_bl + blend * predicted_returns.reindex(assets).fillna(μ_bl)

    # 3: Optimize via cvxpy with regularization and constraints
    cov = pd.DataFrame(Σ, index=assets, columns=assets)
    w = cp.Variable(n)
    objective = (
        risk_aversion * cp.quad_form(w, cov.values)
        - mu.values @ w
        + l2_reg * cp.sum_squares(w)
        + l1_reg * cp.norm1(w)
    )
    constraints = [cp.sum(w) == 1, w >= 0]
    if position_limit is not None:
        constraints.append(w <= position_limit)
    if sector_caps and sector_map:
        sectors = _sector_index_map(assets, sector_map)
        for sector, idxs in sectors.items():
            cap = sector_caps.get(sector)
            if cap is None or len(idxs) == 0:
                continue
            constraints.append(cp.sum(w[idxs]) <= cap)
    if max_turnover is not None and previous_weights is not None:
        prev = previous_weights.reindex(assets).fillna(0.0).values
        constraints.append(cp.norm1(w - prev) <= max_turnover)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        w_out = pd.Series(np.ones(n) / n, index=assets)
    else:
        w_out = pd.Series(np.array(w.value).flatten(), index=assets)

    # Cleanup enforcement
    w_out = _enforce_constraints_posthoc(
        w_out,
        long_only=True,
        position_limit=position_limit,
        sector_caps=sector_caps,
        sector_map=sector_map,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        weights_sum_to_one=True,
    )

    return w_out


# ---------------------------------------------------------
# Risk Parity Optimization
# ---------------------------------------------------------
def risk_parity_optimize(
        returns: pd.DataFrame,
        long_only: bool = True,
        predicted_returns: Optional[pd.Series] = None,
        blend: float = 0.5,
        # risk model
        risk_model: str = "sample",
        factor_returns: Optional[pd.DataFrame] = None,
        # constraints
        position_limit: Optional[float] = None,
        sector_caps: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        previous_weights: Optional[pd.Series] = None,
        max_turnover: Optional[float] = None,
) -> pd.Series:
    Σ = _get_covariance(returns, risk_model=risk_model, factor_returns=factor_returns).values
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
    # Add turnover constraint if requested (scipy-form)
    if max_turnover is not None and previous_weights is not None:
        prev_arr = previous_weights.reindex(assets).fillna(0.0).values
        cons.append({'type': 'ineq', 'fun': lambda w, prev=prev_arr, cap=max_turnover: cap - np.sum(np.abs(w - prev))})
    # Add sector caps as constraints
    if sector_caps and sector_map:
        sectors = _sector_index_map(assets, sector_map)
        for sector, idxs in sectors.items():
            cap = sector_caps.get(sector)
            if cap is None:
                continue
            cons.append({'type': 'ineq', 'fun': lambda w, idxs=idxs, cap=cap: cap - np.sum(w[idxs])})

    res = minimize(objective, x0, bounds=bounds, constraints=cons)
    if not res.success:
        w = pd.Series(np.ones(n) / n, index=assets)
    else:
        w = pd.Series(res.x, index=assets)

    # enforce position limit & long-only & normalization
    w = _enforce_constraints_posthoc(
        w,
        long_only=long_only,
        position_limit=position_limit,
        sector_caps=sector_caps,
        sector_map=sector_map,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        weights_sum_to_one=True,
    )
    return w


# ---------------------------------------------------------
# Conditional Value at Risk (CVaR)
# ---------------------------------------------------------
def cvar_optimize(
        returns: pd.DataFrame,
        predicted_returns: Optional[pd.Series] = None,
        blend: float = 0.5,
        alpha: float = 0.95,
        target_return: Optional[float] = None,
        # regularization
        l2_reg: float = 0.0,
        l1_reg: float = 0.0,
        long_only: bool = True,
        # constraints
        position_limit: Optional[float] = None,
        sector_caps: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        previous_weights: Optional[pd.Series] = None,
        max_turnover: Optional[float] = None,
) -> pd.Series:
    """
    Minimize CVaR subject to constraints. Uses historical return scenarios in `returns`.
    Now accepts ML-predicted returns + blending to set target_return or to be used
    in additional constraints if desired (we use blended mean for target_return tests).
    """
    T, n = returns.shape
    R = returns.values
    assets = returns.columns

    w = cp.Variable(n)
    z = cp.Variable()
    u = cp.Variable(T)

    # blended expected returns used for optional target constraint
    mu = get_effective_mu(returns, predicted_returns, blend).reindex(assets).fillna(0).values

    constraints = [
        u >= -R @ w - z,
        u >= 0,
        cp.sum(w) == 1,
    ]
    if long_only:
        constraints.append(w >= 0)
    if position_limit is not None:
        constraints.append(w <= position_limit)

    # sector caps via cvxpy constraints
    if sector_caps and sector_map:
        sectors = _sector_index_map(assets, sector_map)
        for sector, idxs in sectors.items():
            cap = sector_caps.get(sector)
            if cap is None or len(idxs) == 0:
                continue
            constraints.append(cp.sum(w[idxs]) <= cap)

    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    if max_turnover is not None and previous_weights is not None:
        prev = previous_weights.reindex(assets).fillna(0.0).values
        constraints.append(cp.norm1(w - prev) <= max_turnover)

    obj = cp.Minimize(z + (1 / ((1 - alpha) * T)) * cp.sum(u) + l2_reg * cp.sum_squares(w) + l1_reg * cp.norm1(w))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        # fallback equal weights
        w_out = pd.Series(np.ones(n) / n, index=assets)
    else:
        w_out = pd.Series(np.array(w.value).flatten(), index=assets)

    # final enforcement (cleanup)
    w_out = _enforce_constraints_posthoc(
        w_out,
        long_only=long_only,
        position_limit=position_limit,
        sector_caps=sector_caps,
        sector_map=sector_map,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        weights_sum_to_one=True,
    )

    return w_out


# ---------------------------------------------------------
# Equal-Risk-Contribution (wrapper for risk parity)
# ---------------------------------------------------------
def equal_risk_contribution(returns: pd.DataFrame, **kwargs) -> pd.Series:
    return risk_parity_optimize(returns, **kwargs)


# ---------------------------------------------------------
# Factor-risk model utilities (Fama–French style)
# ---------------------------------------------------------
def _estimate_factor_risk(returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate asset covariance using a linear factor model:
    R_it = B_i F_t + e_it
    Cov(R) = B Cov(F) B^T + diag(var(e))
    """
    # align by date index
    aligned = returns.join(factor_returns, how="inner")
    X = aligned[factor_returns.columns].values  # T x k
    Σ_f = np.cov(X, rowvar=False)

    assets = returns.columns
    T = X.shape[0]
    XTX_inv = np.linalg.pinv(X.T @ X)
    B = []
    idio_vars = []
    for a in assets:
        y = aligned[a].values.reshape(T, 1)
        beta = XTX_inv @ (X.T @ y)  # k x 1
        resid = y - X @ beta
        B.append(beta.flatten())
        idio_vars.append(float(np.var(resid, ddof=1)))
    B = np.array(B)  # n x k
    D = np.diag(idio_vars)
    Σ = B @ Σ_f @ B.T + D
    return pd.DataFrame(Σ, index=assets, columns=assets)


def _get_covariance(
        returns: pd.DataFrame,
        *,
        risk_model: str = "sample",
        factor_returns: Optional[pd.DataFrame] = None,
        epsilon: float = 1e-8,
) -> pd.DataFrame:
    if risk_model == "factor":
        if factor_returns is None or factor_returns.empty:
            raise ValueError("factor_returns must be provided when risk_model='factor'")
        cov = _estimate_factor_risk(returns, factor_returns)
    else:
        cov = returns.cov()
    cov = cov + np.eye(cov.shape[0]) * epsilon
    cov.index = returns.columns
    cov.columns = returns.columns
    return cov


# ---------------------------------------------------------
# Monte Carlo Robustness Simulations
# ---------------------------------------------------------
def monte_carlo_simulation(
        optimizer_fn,
        returns: pd.DataFrame,
        n_sims: int = 100,
        horizon: int = 252,
        random_seed: Optional[int] = None,
        predicted_returns: Optional[pd.Series] = None,
        blend: float = 0.5,
        **optimizer_kwargs,
) -> pd.DataFrame:
    """
    Simulates many random return paths and re-optimizes each time
    Returns a DataFrame of simulated optimal weights.
    """
    rng = np.random.default_rng(random_seed)
    μ = get_effective_mu(returns, predicted_returns, blend).values
    Σ = returns.cov().values
    assets = returns.columns
    sims = []

    for _ in range(n_sims):
        sim_data = pd.DataFrame(
            rng.multivariate_normal(μ, Σ, size=horizon),
            columns=assets,
        )
        try:
            w = optimizer_fn(sim_data, predicted_returns=predicted_returns, blend=blend, **optimizer_kwargs)
        except TypeError:
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
