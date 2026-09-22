"""
rolling_sharpe.py  —  add to leveraged-etf-risk-lab/src/
---------------------------------------------------------
Rolling Sharpe and Sortino ratio time series for your real portfolio.

Sharpe  = (portfolio return - risk-free rate) / volatility
Sortino = (portfolio return - risk-free rate) / downside volatility only
          (doesn't penalise you for upside swings — more realistic)

Rolling means: computed over a sliding 60-day window, plotted over time.
This tells you whether your portfolio's risk-adjusted performance is getting
better or worse as market conditions change.

Run standalone:
    python src/rolling_sharpe.py

Or import:
    from src.rolling_sharpe import compute_rolling_sharpe, plot_rolling_sharpe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path


RISK_FREE_RATE = 0.0   # explicit effective annual assumption; not a fetched rate


def load_portfolio(csv_path="portfolio_values.csv"):
    """
    Load portfolio from CSV (ticker, market_value).
    USD is an ETF symbol here, not a cash marker.
    """
    df = pd.read_csv(csv_path)
    df["weight"] = df["market_value"] / df["market_value"].sum()
    return df


def fetch_portfolio_history(portfolio_df, period="1y"):
    """
    Download daily close prices for all tickers.
    Returns a DataFrame of daily returns, weighted by portfolio allocation.
    """
    tickers = portfolio_df["ticker"].tolist()
    weights = dict(zip(portfolio_df["ticker"], portfolio_df["weight"]))

    print(f"Fetching 1y history for {len(tickers)} tickers: {tickers}")
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(c).strip("_") for c in raw.columns]

    close_cols = [c for c in raw.columns if c.startswith("Close")]
    prices = raw[close_cols].copy()
    prices.columns = [c.replace("Close_", "") for c in prices.columns]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    from src.risk import aligned_returns, portfolio_returns
    daily_returns, norm_weights, history = aligned_returns(prices, weights)
    portfolio = portfolio_returns(daily_returns, norm_weights)
    print(f"  History policy: {history}")
    return portfolio, prices[norm_weights.index], norm_weights.to_dict()


def compute_rolling_sharpe(returns, window=60, rf_daily=(1+RISK_FREE_RATE)**(1/252)-1):
    """
    Rolling Sharpe ratio over `window` trading days.
    Annualised: multiply daily Sharpe by sqrt(252).
    """
    excess = returns - rf_daily
    rolling_mean = excess.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
    sharpe.name = "rolling_sharpe"
    return sharpe


def compute_rolling_sortino(returns, window=60, rf_daily=(1+RISK_FREE_RATE)**(1/252)-1):
    """
    Rolling Sortino: RMS negative excess returns over all window observations.
    """
    excess = returns - rf_daily

    def downside_std(x):
        downside = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
        return downside if downside > 0 else np.nan

    rolling_mean     = excess.rolling(window).mean()
    rolling_downside = excess.rolling(window).apply(downside_std, raw=True)
    sortino = (rolling_mean / rolling_downside) * np.sqrt(252)
    sortino.name = "rolling_sortino"
    return sortino


def compute_rolling_var(returns, window=60, confidence=0.95):
    """
    Rolling Value at Risk (VaR) at given confidence level.
    Historical lower return quantile; not a calibrated future loss bound.
    """
    var = returns.rolling(window).quantile(1 - confidence)
    var.name = f"var_{int(confidence*100)}"
    return var


def print_current_stats(returns, sharpe, sortino, var95):
    """Print today's snapshot of all risk metrics."""
    ann_return = returns.mean() * 252
    ann_vol    = returns.std()  * np.sqrt(252)
    total_ret  = (1 + returns).prod() - 1
    from src.risk import max_drawdown
    max_dd = max_drawdown(returns)
    win_rate   = (returns > 0).mean()

    current_sharpe  = float(sharpe.iloc[-1])  if not sharpe.isna().all()  else np.nan
    current_sortino = float(sortino.iloc[-1]) if not sortino.isna().all() else np.nan
    current_var     = float(var95.iloc[-1])   if not var95.isna().all()   else np.nan

    print("\n" + "="*57)
    print("  PORTFOLIO RISK DASHBOARD")
    print("  (daily rebalanced at supplied capital weights, 1Y history)")
    print("="*57)
    print(f"  Annual return (approx):   {ann_return:>+8.1%}")
    print(f"  Annual volatility:        {ann_vol:>8.1%}")
    print(f"  Total return (1Y):        {total_ret:>+8.1%}")
    print(f"  Win rate (daily):         {win_rate:>8.1%}")
    print(f"  Max drawdown:             {max_dd:>8.1%}")
    print(f"─"*57)
    print(f"  Rolling Sharpe  (60d):    {current_sharpe:>8.2f}")
    print(f"  Rolling Sortino (60d):    {current_sortino:>8.2f}")
    print(f"  VaR 95% (daily):          {current_var:>8.1%}  (historical 5th percentile, not a loss bound)")
    print("="*57)

    print("  Historical rolling estimates; no recommendation or calibrated tail forecast.")


def plot_rolling_sharpe(returns, sharpe, sortino, var95, prices, weights):
    """4-panel dashboard: Sharpe, Sortino, VaR, and cumulative return."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("Portfolio Risk Dashboard — Rolling 60-Day Metrics",
                 fontsize=14, fontweight="bold", y=0.99)

    cum_return = (1 + returns).cumprod()

    # Panel 1: Cumulative return
    axes[0].plot(cum_return.index, cum_return.values, color="#1D9E75", linewidth=1.8)
    axes[0].axhline(1, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
    axes[0].fill_between(cum_return.index, cum_return.values, 1,
                          where=(cum_return.values >= 1), alpha=0.15, color="#1D9E75")
    axes[0].fill_between(cum_return.index, cum_return.values, 1,
                          where=(cum_return.values < 1), alpha=0.15, color="#A32D2D")
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Portfolio Cumulative Return (1Y)", fontsize=11)
    axes[0].grid(True, alpha=0.25)

    # Panel 2: Rolling Sharpe
    axes[1].plot(sharpe.index, sharpe.values, color="#378ADD", linewidth=1.5,
                 label="Sharpe")
    axes[1].plot(sortino.index, sortino.values, color="#8A4FD8", linewidth=1.5,
                 linestyle="--", label="Sortino", alpha=0.8)
    axes[1].axhline(1.0,  color="green",  linestyle=":", alpha=0.5, label="Good (1.0)")
    axes[1].axhline(0.5,  color="orange", linestyle=":", alpha=0.5, label="Ok (0.5)")
    axes[1].axhline(0,    color="black",  linewidth=0.8)
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Rolling Sharpe & Sortino Ratio (60-Day Window)", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_ylim(-3, 5)
    axes[1].grid(True, alpha=0.25)

    # Panel 3: Rolling VaR
    axes[2].plot(var95.index, var95.values * 100, color="#A32D2D", linewidth=1.5)
    axes[2].fill_between(var95.index, var95.values * 100, 0, alpha=0.2, color="#A32D2D")
    axes[2].axhline(-5, color="orange", linestyle="--", alpha=0.5, label="-5% threshold")
    axes[2].set_ylabel("VaR (%)")
    axes[2].set_title("Rolling 95% VaR — Historical 5th Percentile (not a loss bound)", fontsize=11)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    # Panel 4: Drawdown
    from src.risk import drawdown_series
    dd = drawdown_series(returns)
    axes[3].fill_between(dd.index, dd.values * 100, 0, color="#A32D2D", alpha=0.4)
    axes[3].set_ylabel("Drawdown (%)")
    axes[3].set_title("Underwater Curve (% from Peak)", fontsize=11)
    axes[3].grid(True, alpha=0.25)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/rolling_sharpe.png", dpi=150, bbox_inches="tight")
    print("\nSaved: data/rolling_sharpe.png")


if __name__ == "__main__":
    portfolio = load_portfolio("portfolio_values.csv")

    print("\nPortfolio weights:")
    for _, row in portfolio.iterrows():
        print(f"  {row['ticker']:6s}  ${row['market_value']:>8,.2f}  ({row['weight']:.1%})")

    returns, prices, weights = fetch_portfolio_history(portfolio, period="1y")

    sharpe  = compute_rolling_sharpe(returns,  window=60)
    sortino = compute_rolling_sortino(returns, window=60)
    var95   = compute_rolling_var(returns,     window=60, confidence=0.95)

    print_current_stats(returns, sharpe, sortino, var95)
    plot_rolling_sharpe(returns, sharpe, sortino, var95, prices, weights)

    print("\nDone. Check data/rolling_sharpe.png")
