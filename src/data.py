"""
data.py
-------
Market data loading helpers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BENCHMARKS, TICKERS


def fetch_price_history(period: str = "2y") -> pd.DataFrame:
    """Fetch adjusted close prices for portfolio tickers and benchmarks."""
    import yfinance as yf

    symbols = TICKERS + BENCHMARKS
    print(f"Fetching prices for {symbols}...")

    raw = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if "Close" in raw:
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices = prices.dropna(how="all")
    Path("data").mkdir(exist_ok=True)
    prices.to_csv("data/prices.csv")

    print(f"Loaded {len(prices)} trading days.")
    print(f"Available symbols: {', '.join(prices.columns)}")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns with no implicit filling."""
    from src.risk import simple_returns
    return simple_returns(prices)
