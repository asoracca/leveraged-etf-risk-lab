"""
config.py
---------
Editable universe and example portfolio weights.

Defaults are illustrative equal weights. Keep personal values only in the
ignored portfolio_values.csv file.
"""

TICKERS = [
    "QLD",
    "USD",
    "RKLX",
    "SMCL",
    "APLX",
    "ASTX",
    "KORU",
    "LABX",
    "MRVU",
    "NBIG",
    "LITX",
]

BENCHMARKS = ["SPY", "QQQ", "SOXX"]

THEMES = {
    "QLD": "Nasdaq leverage",
    "USD": "Semiconductor leverage",
    "RKLX": "Space / Rocket Lab leverage",
    "SMCL": "AI infrastructure leverage",
    "APLX": "Mega-cap tech leverage",
    "ASTX": "Space / ASTS leverage",
    "KORU": "International leverage",
    "LABX": "Biotech leverage",
    "MRVU": "Magnificent 7 leverage",
    "NBIG": "AI / cloud leverage",
    "LITX": "Lithium leverage",
}

# Synthetic equal-capital example, not personal allocations.
DEFAULT_WEIGHTS = {ticker: 1 / len(TICKERS) for ticker in TICKERS}
