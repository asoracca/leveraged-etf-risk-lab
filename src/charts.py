"""
charts.py
---------
Visualization helpers for risk, correlation, drawdown, and stress tests.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.risk import compute_summary, concentration_table, correlation_matrix
from src.stress import run_stress_tests


def _save(filename: str) -> None:
    Path("data").mkdir(exist_ok=True)
    path = f"data/{filename}"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_equity_and_drawdown(prices, weights=None):
    _, _, port = compute_summary(prices, weights)
    equity = (1 + port).cumprod()
    drawdown = equity / equity.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Leveraged ETF Portfolio Equity and Drawdown", fontweight="bold")

    axes[0].plot(equity.index, equity, linewidth=2, color="#1D9E75")
    axes[0].set_title("Compounded Equity")
    axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(drawdown.index, drawdown, 0, color="#A32D2D", alpha=0.35)
    axes[1].plot(drawdown.index, drawdown, color="#A32D2D", linewidth=1.5)
    axes[1].set_title("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(True, alpha=0.25)

    _save("equity_drawdown.png")


def plot_risk_contribution(prices, weights=None):
    table = concentration_table(prices, weights)

    plt.figure(figsize=(12, 6))
    colors = ["#A32D2D" if value > 0.20 else "#378ADD" for value in table["risk_contribution"]]
    plt.bar(table["ticker"], table["risk_contribution"], color=colors, alpha=0.85)
    plt.axhline(0.20, color="black", linestyle="--", alpha=0.45, label="20% risk concentration")
    plt.title("Portfolio Risk Contribution by Holding")
    plt.ylabel("Share of Portfolio Volatility")
    plt.gca().yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    _save("risk_contribution.png")


def plot_correlation_heatmap(prices):
    corr = correlation_matrix(prices)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("60-Day Correlation Matrix")
    _save("correlation_heatmap.png")


def plot_stress_tests(weights=None):
    results = run_stress_tests(weights).sort_values("portfolio_return")

    plt.figure(figsize=(12, 5))
    plt.barh(results["scenario"], results["portfolio_return"], color="#A32D2D", alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", alpha=0.35)
    plt.title("Scenario Stress Tests")
    plt.xlabel("Estimated Portfolio Return")
    plt.gca().xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.grid(True, axis="x", alpha=0.25)
    _save("stress_tests.png")


def plot_all(prices, weights=None):
    plot_equity_and_drawdown(prices, weights)
    plot_risk_contribution(prices, weights)
    plot_correlation_heatmap(prices)
    plot_stress_tests(weights)
