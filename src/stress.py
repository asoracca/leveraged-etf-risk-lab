"""
stress.py
---------
Simple scenario stress tests for leveraged ETF portfolios.
"""

import numpy as np
import pandas as pd

from src.config import DEFAULT_WEIGHTS, THEMES
from src.risk import normalize_weights


SCENARIOS = {
    "SPY -5% risk-off day": {
        "QLD": -0.10,
        "USD": -0.12,
        "RKLX": -0.14,
        "SMCL": -0.16,
        "APLX": -0.08,
        "ASTX": -0.14,
        "KORU": -0.10,
        "LABX": -0.08,
        "MRVU": -0.10,
        "NBIG": -0.14,
        "LITX": -0.10,
    },
    "QQQ -8% tech selloff": {
        "QLD": -0.16,
        "USD": -0.18,
        "RKLX": -0.12,
        "SMCL": -0.22,
        "APLX": -0.14,
        "ASTX": -0.10,
        "KORU": -0.08,
        "LABX": -0.06,
        "MRVU": -0.16,
        "NBIG": -0.20,
        "LITX": -0.08,
    },
    "Semis -10%": {
        "QLD": -0.08,
        "USD": -0.20,
        "RKLX": -0.06,
        "SMCL": -0.20,
        "APLX": -0.04,
        "ASTX": -0.05,
        "KORU": -0.06,
        "LABX": -0.03,
        "MRVU": -0.08,
        "NBIG": -0.10,
        "LITX": -0.04,
    },
    "Space/AI high-beta unwind": {
        "QLD": -0.08,
        "USD": -0.10,
        "RKLX": -0.25,
        "SMCL": -0.18,
        "APLX": -0.05,
        "ASTX": -0.25,
        "KORU": -0.05,
        "LABX": -0.05,
        "MRVU": -0.10,
        "NBIG": -0.22,
        "LITX": -0.08,
    },
}


def run_stress_tests(weights=None, scenarios=None):
    """Deterministic one-session shocks; every held symbol needs an assumption."""
    w = normalize_weights(weights)
    rows = []
    for name, moves in (SCENARIOS if scenarios is None else scenarios).items():
        if not set(w.index).issubset(moves):
            raise ValueError(f"Missing scenario shocks: {set(w.index) - set(moves)}")
        shocks = pd.Series(moves).reindex(w.index)
        if not np.isfinite(shocks).all() or (shocks < -1).any():
            raise ValueError("Shocks must be finite simple returns >= -1")
        contributions = w * shocks
        worst = contributions.idxmin()
        rows.append({"scenario": name, "kind": "deterministic assumed one-day stress",
                     "portfolio_return": float(contributions.sum()),
                     "weights": w.to_dict(), "assumed_shocks": shocks.to_dict(),
                     "contributions": contributions.to_dict(),
                     "formula": "sum(weight[i] * assumed_shock[i])",
                     "worst_contributor": worst,
                     "worst_contributor_theme": THEMES.get(worst, "Unclassified"),
                     "worst_contribution": float(contributions[worst])})
    return pd.DataFrame(rows).sort_values("portfolio_return") if rows else pd.DataFrame()


def print_stress_report(weights: dict[str, float] | None = None) -> None:
    results = run_stress_tests(weights)

    print("\n" + "=" * 70)
    print("  DETERMINISTIC STRESS ASSUMPTIONS — NOT CALIBRATED FORECASTS")
    print("=" * 70)
    printable = results.copy()
    for col in ["portfolio_return", "worst_contribution"]:
        printable[col] = printable[col].map(lambda x: f"{x:.1%}")
    print(printable.to_string(index=False))
    print("=" * 70)
