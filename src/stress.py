"""
stress.py
---------
Simple scenario stress tests for leveraged ETF portfolios.
"""

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


def run_stress_tests(weights: dict[str, float] | None = None) -> pd.DataFrame:
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    rows = []

    for scenario_name, moves in SCENARIOS.items():
        weighted_loss = 0.0
        worst_ticker = None
        worst_contribution = 0.0

        for ticker, weight in w.items():
            move = moves.get(ticker, 0.0)
            contribution = weight * move
            weighted_loss += contribution
            if contribution < worst_contribution:
                worst_ticker = ticker
                worst_contribution = contribution

        rows.append({
            "scenario": scenario_name,
            "portfolio_return": weighted_loss,
            "worst_contributor": worst_ticker,
            "worst_contributor_theme": THEMES.get(worst_ticker, ""),
            "worst_contribution": worst_contribution,
        })

    return pd.DataFrame(rows).sort_values("portfolio_return")


def print_stress_report(weights: dict[str, float] | None = None) -> None:
    results = run_stress_tests(weights)

    print("\n" + "=" * 70)
    print("  STRESS TESTS")
    print("=" * 70)
    printable = results.copy()
    for col in ["portfolio_return", "worst_contribution"]:
        printable[col] = printable[col].map(lambda x: f"{x:.1%}")
    print(printable.to_string(index=False))
    print("=" * 70)
