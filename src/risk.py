"""Shared simple-return portfolio calculations. See docs/methodology.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import BENCHMARKS, DEFAULT_WEIGHTS, THEMES, TICKERS


def normalize_weights(weights=None, available=None) -> pd.Series:
    """Normalize finite long-only capital weights; never silently drop a holding."""
    w = pd.Series(DEFAULT_WEIGHTS if weights is None else weights, dtype=float)
    if w.empty or not w.index.is_unique or not np.isfinite(w).all() or (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, unique and have positive total")
    w = w[w > 0]
    if available is not None and not w.index.isin(available).all():
        raise ValueError(f"Missing symbols: {w.index[~w.index.isin(available)].tolist()}")
    return w / w.sum()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing or not prices.columns.is_unique:
        raise ValueError("Prices require unique columns and unique increasing dates")
    if prices.empty or ((prices <= 0) & prices.notna()).any().any() or np.isinf(prices.to_numpy()).any():
        raise ValueError("Observed prices must be finite and positive")
    return prices.pct_change(fill_method=None).iloc[1:]


def aligned_returns(prices, weights=None, missing_policy="common_start"):
    """Trim leading incomplete histories, but reject gaps/trailing missing prices."""
    if missing_policy not in ("common_start", "strict"):
        raise ValueError("missing_policy must be common_start or strict")
    w = normalize_weights(weights, prices.columns)
    simple_returns(prices)  # validate before slicing
    selected = prices[w.index]
    complete = selected.notna().all(axis=1)
    if not complete.any():
        raise ValueError("No common price history")
    first = int(np.flatnonzero(complete.to_numpy())[0])
    if missing_policy == "strict" and not complete.all():
        raise ValueError("Strict history policy rejects missing prices")
    selected = selected.iloc[first:]
    if selected.isna().any().any():
        raise ValueError("Internal or trailing missing prices; no filling or gap bridging")
    r = simple_returns(selected)
    if len(r) < 2:
        raise ValueError("At least three common prices (two returns) are required")
    return r, w, {"missing_policy": missing_policy, "discarded_leading_prices": first,
                  "price_start": str(selected.index[0]), "price_end": str(selected.index[-1]),
                  "observations": len(r)}


def _validate_returns(returns):
    if len(returns) < 2 or not np.isfinite(np.asarray(returns)).all() or (np.asarray(returns) < -1).any():
        raise ValueError("Need at least two finite simple returns, each >= -1")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and increasing")


def portfolio_pnl(returns, weights=None, mode="constant_weight"):
    w = normalize_weights(weights, returns.columns)
    r = returns[w.index]
    _validate_returns(r)
    if mode == "constant_weight":
        return r.mul(w, axis=1)
    if mode != "buy_and_hold":
        raise ValueError("mode must be constant_weight or buy_and_hold")
    asset_wealth = (1 + r).cumprod().mul(w, axis=1)
    previous = asset_wealth.shift(1)
    previous.iloc[0] = w
    total = previous.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("Portfolio exhausted before the end of history")
    return r * previous.div(total, axis=0)


def portfolio_returns(returns, weights=None, mode="constant_weight"):
    return portfolio_pnl(returns, weights, mode).sum(axis=1).rename("portfolio_return")


def drawdown_series(return_series):
    if len(return_series) == 0 or not np.isfinite(return_series).all() or (return_series < -1).any():
        raise ValueError("Invalid simple returns")
    equity = (1 + return_series).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1


def max_drawdown(return_series):
    return float(drawdown_series(return_series).min())


def beta_to_benchmark(asset_returns, benchmark_returns):
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2 or not np.isfinite(aligned.to_numpy()).all():
        return np.nan
    variance = aligned.iloc[:, 1].var(ddof=1) if aligned.iloc[:, 1].nunique() > 1 else 0.0
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance) if variance > 0 else np.nan


def component_risk(returns, weights=None, mode="constant_weight", periods=252):
    """Annual volatility components; sum equals realized portfolio volatility."""
    if not np.isfinite(periods) or periods <= 0:
        raise ValueError("periods must be positive")
    pnl = portfolio_pnl(returns, weights, mode)
    port = pnl.sum(axis=1)
    vol = float(port.std(ddof=1) * np.sqrt(periods)) if port.nunique() > 1 else 0.0
    if vol == 0:
        return pd.Series(0.0, index=pnl.columns)
    return pnl.apply(lambda x: x.cov(port) * periods / vol)


def risk_contribution(returns, weights=None):
    components = component_risk(returns, weights)
    total = components.sum()
    return (components / total if total > 0 else components * np.nan).sort_values(ascending=False)


def compute_summary(prices, weights=None, *, mode="constant_weight", missing_policy="common_start",
                    periods=252, risk_free_rate=0.0):
    if not np.isfinite(periods) or periods <= 0 or not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("Invalid annualization or effective annual risk-free rate")
    returns, w, history = aligned_returns(prices, weights, missing_policy)
    port = portfolio_returns(returns, w, mode)
    vol = float(port.std(ddof=1) * np.sqrt(periods)) if port.nunique() > 1 else 0.0
    daily_rf = np.expm1(np.log1p(risk_free_rate) / periods)
    components = component_risk(returns, w, mode, periods)
    shares = components / vol if vol > 0 else components * np.nan
    benchmark_returns = simple_returns(prices)
    betas = {b: beta_to_benchmark(port, benchmark_returns[b].reindex(port.index))
             for b in BENCHMARKS if b in prices.columns}
    summary = {"total_return": float((1 + port).prod() - 1),
               "annual_return": float(port.mean() * periods),
               "annual_vol": vol,
               "sharpe": float((port.mean() - daily_rf) * periods / vol) if vol > 0 else np.nan,
               "max_drawdown": max_drawdown(port), "betas": betas,
               "component_volatility": components.to_dict(), "weights": w.to_dict(),
               "mode": mode, "periods": periods, "risk_free_rate": risk_free_rate,
               "history": history}
    return summary, shares.sort_values(ascending=False), port


def correlation_matrix(prices, lookback=60):
    # Complete rows give all pairs the same observation window.
    return simple_returns(prices).dropna().tail(lookback).corr()


def concentration_table(prices, weights=None):
    returns, w, _ = aligned_returns(prices, weights)
    rc = risk_contribution(returns, w)
    return pd.DataFrame([{"ticker": t, "theme": THEMES.get(t, "Unclassified"),
                          "weight": w[t], "annual_vol": returns[t].std() * np.sqrt(252),
                          "risk_contribution": rc[t]} for t in w.index]).sort_values("risk_contribution", ascending=False)


def print_risk_report(prices, weights=None):
    summary, _, _ = compute_summary(prices, weights)
    print("\nHISTORICAL RISK REPORT — daily constant weights; not account performance")
    for key in ("total_return", "annual_return", "annual_vol", "max_drawdown"):
        print(f"  {key}: {summary[key]:.2%}")
    print(f"  Sharpe (rf={summary['risk_free_rate']:.1%}): {summary['sharpe']:.3f}")
    print(f"  History: {summary['history']}")
    print(f"  Betas: {summary['betas']}")
    print(concentration_table(prices, weights).to_string(index=False))
