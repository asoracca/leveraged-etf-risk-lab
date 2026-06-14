"""
exposure.py
-----------
Theme-level exposure analysis for a leveraged ETF portfolio.

This answers:
  Am I diversified, or am I stacking the same trade across different tickers?
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import fetch_price_history
from src.holdings import load_weights
from src.risk import normalize_weights, risk_contribution


THEME_BUCKETS = {
    "QLD": "Nasdaq / mega-cap tech",
    "USD": "Semiconductors",
    "RKLX": "Space",
    "SMCL": "AI infrastructure",
    "APLX": "Mega-cap tech",
    "ASTX": "Space",
    "KORU": "International",
    "LABX": "Biotech",
    "MRVU": "Mega-cap tech",
    "NBIG": "AI infrastructure",
    "LITX": "Lithium / materials",
}


def compute_theme_exposure(prices, weights=None):
    if weights is None:
        weights = load_weights()
    weights = normalize_weights(weights)
    returns = prices.pct_change().dropna(how="all")
    ticker_risk = risk_contribution(returns, weights)

    rows = []

    for ticker, weight in weights.items():
        theme = THEME_BUCKETS.get(ticker, "Other")
        rows.append({
            "ticker": ticker,
            "theme": theme,
            "weight": weight,
            "risk_contribution": ticker_risk.get(ticker, 0.0),
        })

    ticker_df = pd.DataFrame(rows)

    theme_df = (
        ticker_df
        .groupby("theme", as_index=False)
        .agg(
            weight=("weight", "sum"),
            risk_contribution=("risk_contribution", "sum"),
            tickers=("ticker", lambda x: ", ".join(sorted(x))),
        )
        .sort_values("risk_contribution", ascending=False)
    )

    Path("data").mkdir(exist_ok=True)
    ticker_df.to_csv("data/ticker_exposure.csv", index=False)
    theme_df.to_csv("data/theme_exposure.csv", index=False)

    return ticker_df, theme_df


def print_theme_exposure(prices, weights=None):
    _, theme_df = compute_theme_exposure(prices, weights)

    print("\n" + "=" * 70)
    print("  THEME EXPOSURE")
    print("=" * 70)

    printable = theme_df.copy()
    printable["weight"] = printable["weight"].map(lambda x: f"{x:.1%}")
    printable["risk_contribution"] = printable["risk_contribution"].map(lambda x: f"{x:.1%}")

    print(printable.to_string(index=False))
    print("\nSaved: data/theme_exposure.csv")
    print("=" * 70)


def plot_theme_exposure(prices, weights=None):
    _, theme_df = compute_theme_exposure(prices, weights)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Theme Exposure vs Theme Risk Contribution", fontweight="bold")

    axes[0].barh(theme_df["theme"], theme_df["weight"], color="#378ADD")
    axes[0].set_title("Portfolio Weight by Theme")
    axes[0].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(theme_df["theme"], theme_df["risk_contribution"], color="#A32D2D")
    axes[1].set_title("Risk Contribution by Theme")
    axes[1].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(True, axis="x", alpha=0.25)

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/theme_exposure.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: data/theme_exposure.png")


if __name__ == "__main__":
    prices = fetch_price_history(period="2y")
    weights = load_weights()
    print_theme_exposure(prices, weights)
    plot_theme_exposure(prices, weights)
