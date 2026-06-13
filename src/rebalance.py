"""
rebalance.py
------------
Decision rules for a leveraged ETF portfolio.

This module does not place trades. It gives risk-based recommendations:
  HOLD        risk is acceptable
  TRIM        one position contributes too much portfolio risk
  DELEVERAGE  portfolio beta/volatility is too high
  RISK_OFF    drawdown is beyond the allowed threshold
"""

from pathlib import Path
import pandas as pd

from src.data import fetch_price_history
from src.holdings import load_weights
from src.risk import compute_summary, concentration_table


RULES = {
    "max_single_risk_contribution": 0.35,
    "max_beta_qqq": 2.50,
    "max_beta_spy": 2.25,
    "max_annual_vol": 0.75,
    "max_drawdown": -0.20,
}


def generate_rebalance_recommendations(prices, weights=None):
    weights = weights or load_weights()
    summary, risk_contrib, _ = compute_summary(prices, weights)
    table = concentration_table(prices, weights)

    recommendations = []

    max_dd = summary["max_drawdown"]
    annual_vol = summary["annual_vol"]
    beta_spy = summary["betas"].get("SPY")
    beta_qqq = summary["betas"].get("QQQ")

    portfolio_flags = []

    if max_dd <= RULES["max_drawdown"]:
        portfolio_flags.append(
            f"Drawdown is {max_dd:.1%}, below risk limit of {RULES['max_drawdown']:.0%}"
        )

    if annual_vol >= RULES["max_annual_vol"]:
        portfolio_flags.append(
            f"Annual volatility is {annual_vol:.1%}, above limit of {RULES['max_annual_vol']:.0%}"
        )

    if beta_spy is not None and beta_spy >= RULES["max_beta_spy"]:
        portfolio_flags.append(
            f"Beta vs SPY is {beta_spy:.2f}, above limit of {RULES['max_beta_spy']:.2f}"
        )

    if beta_qqq is not None and beta_qqq >= RULES["max_beta_qqq"]:
        portfolio_flags.append(
            f"Beta vs QQQ is {beta_qqq:.2f}, above limit of {RULES['max_beta_qqq']:.2f}"
        )

    for _, row in table.iterrows():
        ticker = row["ticker"]
        weight = row["weight"]
        risk = row["risk_contribution"]
        theme = row["theme"]

        action = "HOLD"
        reason = "Risk contribution is within limits."

        if risk >= RULES["max_single_risk_contribution"]:
            action = "TRIM"
            reason = (
                f"{ticker} contributes {risk:.1%} of portfolio risk, "
                f"above limit of {RULES['max_single_risk_contribution']:.0%}."
            )

        if portfolio_flags and action == "HOLD":
            action = "DELEVERAGE"
            reason = "Portfolio-level risk is elevated: " + "; ".join(portfolio_flags)

        if max_dd <= RULES["max_drawdown"]:
            action = "RISK_OFF"
            reason = (
                f"Portfolio drawdown is {max_dd:.1%}. "
                "Risk-off mode triggered before adding exposure."
            )

        recommendations.append({
            "ticker": ticker,
            "theme": theme,
            "weight": round(weight, 4),
            "risk_contribution": round(risk, 4),
            "action": action,
            "reason": reason,
        })

    df = pd.DataFrame(recommendations)

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/rebalance_recommendations.csv", index=False)

    return df, summary


def print_rebalance_report(prices, weights=None):
    recommendations, summary = generate_rebalance_recommendations(prices, weights)

    print("\n" + "=" * 70)
    print("  REBALANCE DECISION ENGINE")
    print("=" * 70)

    print(f"  Annual volatility: {summary['annual_vol']:.1%}")
    print(f"  Max drawdown:      {summary['max_drawdown']:.1%}")

    for benchmark, beta in summary["betas"].items():
        print(f"  Beta vs {benchmark}:     {beta:.2f}")

    print("\n  Recommendations:")
    print(
        recommendations[
            ["ticker", "weight", "risk_contribution", "action", "reason"]
        ].to_string(index=False)
    )

    print("\nSaved: data/rebalance_recommendations.csv")
    print("=" * 70)


if __name__ == "__main__":
    prices = fetch_price_history(period="2y")
    weights = load_weights()
    print_rebalance_report(prices, weights)
