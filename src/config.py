"""
config.py
---------
Editable universe and example portfolio weights.

Replace DEFAULT_WEIGHTS with your real portfolio weights when you want a more
accurate dashboard. Keep weights as decimals that sum to 1.0.
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

DEFAULT_WEIGHTS = {
    "QLD": 0.24,
    "USD": 0.43,
    "RKLX": 0.04,
    "SMCL": 0.03,
    "APLX": 0.01,
    "ASTX": 0.13,
    "KORU": 0.01,
    "LABX": 0.01,
    "MRVU": 0.01,
    "NBIG": 0.10,
    "LITX": 0.01,
}
