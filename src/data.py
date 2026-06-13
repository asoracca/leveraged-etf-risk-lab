"""
data.py
-------
Market data loading helpers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import BENCHMARKS, TICKERS


def fetch_price_history(period: str = "2y") -> pd.DataFrame:
    """Fetch adjusted close prices for portfolio tickers and benchmarks."""
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

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    Path("data").mkdir(exist_ok=True)
    prices.to_csv("data/prices.csv")

    print(f"Loaded {len(prices)} trading days.")
    print(f"Available symbols: {', '.join(prices.columns)}")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
