"""
main.py
-------
Run the leveraged ETF risk dashboard.

Usage:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from src.charts import plot_all
from src.data import fetch_price_history
from src.holdings import load_weights
from src.risk import print_risk_report
from src.stress import print_stress_report


def main():
    Path("data").mkdir(exist_ok=True)
    weights = load_weights()

    print("\n-- 1. Load Market Data --------------------------------")
    prices = fetch_price_history(period="2y")

    print("\n-- 2. Portfolio Risk Report ---------------------------")
    print_risk_report(prices, weights)

    print("\n-- 3. Stress Tests ------------------------------------")
    print_stress_report(weights)

    print("\n-- 4. Charts ------------------------------------------")
    plot_all(prices, weights)

    print("\nDone. Outputs saved to data/.")


if __name__ == "__main__":
    main()
