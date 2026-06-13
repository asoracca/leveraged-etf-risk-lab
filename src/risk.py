"""
risk.py
-------
Portfolio risk metrics for leveraged ETF exposure.
"""

import numpy as np
import pandas as pd

from src.config import BENCHMARKS, DEFAULT_WEIGHTS, THEMES, TICKERS


def normalize_weights(weights: dict[str, float] | None = None, available: list[str] | None = None) -> pd.Series:
    weights = weights or DEFAULT_WEIGHTS
    available = available or list(weights.keys())
    filtered = {ticker: weights[ticker] for ticker in weights if ticker in available}
    series = pd.Series(filtered, dtype=float)
    return series / series.sum()


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    available = [ticker for ticker in TICKERS if ticker in returns.columns]
    w = normalize_weights(weights, available)
    return returns[w.index].mul(w, axis=1).sum(axis=1)


def max_drawdown(return_series: pd.Series) -> float:
    equity = (1 + return_series).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def beta_to_benchmark(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0:
        return np.nan
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())


def risk_contribution(returns: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    available = [ticker for ticker in TICKERS if ticker in returns.columns]
    w = normalize_weights(weights, available)
    cov = returns[w.index].cov() * 252
    portfolio_vol = float(np.sqrt(w.T @ cov @ w))
    if portfolio_vol == 0:
        return pd.Series(index=w.index, dtype=float)
    marginal_contrib = cov @ w / portfolio_vol
    contrib = w * marginal_contrib / portfolio_vol
    return contrib.sort_values(ascending=False)


def compute_summary(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> tuple[dict, pd.Series, pd.Series]:
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    port = portfolio_returns(returns, weights)
    equity = (1 + port).cumprod()

    annual_return = port.mean() * 252
    annual_vol = port.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    betas = {}
    for benchmark in BENCHMARKS:
        if benchmark in returns.columns:
            betas[benchmark] = beta_to_benchmark(port, returns[benchmark])

    summary = {
        "total_return": equity.iloc[-1] - 1,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(port),
        "betas": betas,
    }

    return summary, risk_contribution(returns, weights), port


def correlation_matrix(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    columns = [ticker for ticker in TICKERS + BENCHMARKS if ticker in returns.columns]
    return returns[columns].tail(lookback).corr()


def concentration_table(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    available = [ticker for ticker in TICKERS if ticker in returns.columns]
    w = normalize_weights(weights, available)
    rc = risk_contribution(returns, weights)

    rows = []
    for ticker in w.index:
        rows.append({
            "ticker": ticker,
            "theme": THEMES.get(ticker, "Unclassified"),
            "weight": w[ticker],
            "annual_vol": returns[ticker].std() * np.sqrt(252),
            "risk_contribution": rc.get(ticker, np.nan),
        })

    return pd.DataFrame(rows).sort_values("risk_contribution", ascending=False)


def print_risk_report(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> None:
    summary, rc, _ = compute_summary(prices, weights)
    table = concentration_table(prices, weights)

    print("\n" + "=" * 70)
    print("  LEVERAGED ETF RISK REPORT")
    print("=" * 70)
    print(f"  Total return:       {summary['total_return']:.1%}")
    print(f"  Annual return:      {summary['annual_return']:.1%}")
    print(f"  Annual volatility:  {summary['annual_vol']:.1%}")
    print(f"  Sharpe ratio:       {summary['sharpe']:.2f}")
    print(f"  Max drawdown:       {summary['max_drawdown']:.1%}")

    print("\n  Portfolio beta:")
    for benchmark, beta in summary["betas"].items():
        print(f"    vs {benchmark}: {beta:.2f}")

    print("\n  Top risk contributors:")
    for ticker, value in rc.head(5).items():
        print(f"    {ticker:<5} {value:>7.1%}  {THEMES.get(ticker, '')}")

    print("\n  Full concentration table:")
    printable = table.copy()
    for col in ["weight", "annual_vol", "risk_contribution"]:
        printable[col] = printable[col].map(lambda x: f"{x:.1%}")
    print(printable.to_string(index=False))
    print("=" * 70)
