"""
optimal_leverage.py — leveraged-etf-risk-lab

Reproduces the "CAGR vs Daily Leverage" curve (the parabola you uploaded) for any
underlying index, and finds the leverage that HISTORICALLY maximized growth.

THE MATH
--------
Long-run log-growth of a daily-rebalanced L-times position is approximately
        g(L) ≈ L·μ − ½·(L·σ)²            (μ, σ = daily mean & stdev of the 1x asset)
The first term (return) is linear in L; the second — VOLATILITY DRAG — grows with L².
So growth peaks at the Kelly-optimal leverage
        L* = μ / σ²
Higher-volatility assets (NASDAQ vs S&P) get a LOWER optimal leverage, because σ is squared.

This module also SIMULATES the real daily-compounded path (no approximation), which
captures the actual decay you see in KORU (3x) and your 2x single-stock ETFs.

CAVEATS (put these in your journal):
  - In-sample / backward-looking. The historical optimum is NOT knowable in advance.
  - Assumes iid returns; real markets have autocorrelation, fat tails, sequence risk.
  - Costs matter: LETFs charge ~0.9-1.0% expense + financing on the borrowed portion.

USAGE
-----
    python optimal_leverage.py                 # runs S&P 500 and NASDAQ (mirrors the picture)
    from optimal_leverage import analyze
    analyze("Korea (KORU underlying)", "EWY")  # your own holding
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"

# Map YOUR leveraged holdings to their 1x underlying so you can analyze the real bet:
#   KORU (3x) -> EWY (MSCI South Korea)      USD (2x) -> SOXX (semiconductors)
#   QLD  (2x) -> QQQ (Nasdaq-100)            others are single stocks (use the stock ticker)
HOLDINGS_UNDERLYING = {
    "S&P 500":  "^GSPC",
    "NASDAQ":   "^IXIC",
    "Korea (KORU)": "EWY",
    "Semis (USD)":  "SOXX",
    "Nasdaq (QLD)": "QQQ",
}


# ----------------------------------------------------------------------
def fetch_returns(ticker: str, fetch_fn=None, period: str = "max") -> pd.Series:
    """Daily simple returns of the 1x underlying. Reuses your fetcher if given."""
    if fetch_fn is not None:
        try:
            s = fetch_fn(ticker, period=period)["Close"].dropna()
            if len(s) > 50:
                return s.pct_change().dropna()
        except Exception:
            pass
    s = yf.Ticker(ticker).history(period=period)["Close"].dropna()
    return s.pct_change().dropna()


def lev_cagr(r: pd.Series, L: float, ann_fee: float = 0.0095,
             ann_fin: float = 0.04) -> float:
    """CAGR of a daily-rebalanced L-times position, net of fees + financing.
    Simulates the actual compounded path (captures decay exactly)."""
    daily = L * r - L * ann_fee / 252 - max(L - 1.0, 0.0) * ann_fin / 252
    growth = float((1.0 + daily).prod())
    yrs = len(r) / 252.0
    return growth ** (1.0 / yrs) - 1.0 if growth > 0 else -1.0


def vol_drag(L: float, sigma_daily: float) -> float:
    """Annualized volatility drag at leverage L (the term that bends the curve down)."""
    return 0.5 * (L * sigma_daily) ** 2 * 252.0


def kelly_optimal(r: pd.Series) -> float:
    """Closed-form growth-optimal leverage L* = mean / variance (daily)."""
    return float(r.mean() / r.var())


def leverage_curve(r: pd.Series, Lmax: float = 4.0, n: int = 41, **costs):
    Ls = np.linspace(0.0, Lmax, n)
    cagr = np.array([lev_cagr(r, L, **costs) for L in Ls])
    peak_L = float(Ls[int(np.argmax(cagr))])
    return Ls, cagr, peak_L


# ----------------------------------------------------------------------
def analyze(name: str, ticker: str, fetch_fn=None, save: bool = True, **costs) -> dict:
    r = fetch_returns(ticker, fetch_fn)
    sigma_d = float(r.std())
    Ls, cagr, peak_L = leverage_curve(r, **costs)
    Lstar = kelly_optimal(r)

    print("=" * 60)
    print(f"  OPTIMAL LEVERAGE — {name}  ({ticker})")
    print(f"  History: {len(r)} days (~{len(r)/252:.0f}y) | daily vol {sigma_d*100:.2f}%")
    print("=" * 60)
    print(f"  {'L':>4} | {'CAGR':>8} | {'Vol drag (ann)':>14}")
    print("  " + "-" * 36)
    for L in (1, 2, 3, 4):
        print(f"  {L:>4} | {lev_cagr(r, L, **costs)*100:7.2f}% | {vol_drag(L, sigma_d)*100:12.1f}%")
    print("  " + "-" * 36)
    print(f"  Empirical peak (simulated, net of cost): {peak_L:.2f}x")
    print(f"  Kelly closed-form  L* = mean/var       : {Lstar:.2f}x")
    print(f"  -> Above this, more leverage LOWERS long-run growth (decay wins).")
    print("=" * 60)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(Ls, cagr * 100, lw=2.5, color="navy")
        plt.axvline(peak_L, ls="--", color="darkorange",
                    label=f"peak ≈ {peak_L:.1f}x")
        plt.scatter([1, 2, 3], [lev_cagr(r, L, **costs) * 100 for L in (1, 2, 3)],
                    color="darkorange", zorder=5)
        plt.xlabel("Daily Leverage"); plt.ylabel("CAGR %")
        plt.title(f"Optimal Leverage — {name}")
        plt.grid(alpha=0.3); plt.legend()
        fname = os.path.join(DATA_DIR, f"optimal_leverage_{ticker.strip('^')}.png")
        plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
        print(f"  Saved: {fname}")

    return {"name": name, "ticker": ticker, "daily_vol": sigma_d,
            "empirical_peak": peak_L, "kelly_Lstar": Lstar,
            "cagr_by_L": {L: lev_cagr(r, L, **costs) for L in (1, 2, 3, 4)}}


def main():
    for name, ticker in HOLDINGS_UNDERLYING.items():
        try:
            analyze(name, ticker)
        except Exception as e:
            print(f"  [warn] {name} ({ticker}) failed: {e}")


if __name__ == "__main__":
    main()