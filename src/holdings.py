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

    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Holdings must have nonempty tickers and finite nonnegative values")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    if df["ticker"].eq("").any():
        raise ValueError("Empty holding ticker")
    df["market_value"] = pd.to_numeric(df["market_value"], errors="raise")
    from src.risk import normalize_weights
    # Validate every row before grouping so a negative row cannot be hidden.
    normalize_weights(dict(enumerate(df["market_value"])))
    grouped = df.groupby("ticker")["market_value"].sum()
    return normalize_weights(grouped).to_dict()
