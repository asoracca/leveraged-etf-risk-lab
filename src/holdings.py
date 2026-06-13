"""
holdings.py
-----------
Load real portfolio weights from a CSV when available.
"""

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_WEIGHTS


HOLDINGS_FILE = Path("portfolio_values.csv")


def load_weights() -> dict[str, float]:
    """
    Load weights from portfolio_values.csv if it exists.

    Expected columns:
      ticker,market_value

    If the file does not exist, fall back to DEFAULT_WEIGHTS.
    """
    if not HOLDINGS_FILE.exists():
        return DEFAULT_WEIGHTS

    df = pd.read_csv(HOLDINGS_FILE)
    required = {"ticker", "market_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"portfolio_values.csv is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["ticker", "market_value"]).copy()
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    df = df.dropna(subset=["market_value"])
    df = df[df["market_value"] > 0]

    grouped = df.groupby("ticker")["market_value"].sum()
    weights = grouped / grouped.sum()
    return weights.to_dict()
