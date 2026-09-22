"""Daily-reset leverage model and historical leverage curves.

Synthetic model, not fund tracking or a recommendation. See docs/methodology.md.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

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
    import yfinance as yf

    if fetch_fn is not None:
        try:
            s = fetch_fn(ticker, period=period)["Close"].dropna()
            if len(s) > 50:
                return s.pct_change(fill_method=None).dropna()
        except Exception:
            pass
    s = yf.Ticker(ticker).history(period=period)["Close"].dropna()
    return s.pct_change(fill_method=None).dropna()


def simulate_daily_reset(returns, leverage=3.0, annual_fee=0.0095,
                         annual_financing=0.04, periods=252, invalid_policy="raise"):
    """Model NAV reset daily; fee on NAV, financing on max(L-1,0).

    A modeled loss >=100% either raises or liquidates irreversibly at zero.
    Nonfinite inputs and underlying returns below -100% always raise.
    """
    r = np.asarray(returns, dtype=float)
    parameters = [leverage, annual_fee, annual_financing, periods]
    if (r.ndim != 1 or not len(r) or not np.isfinite(r).all() or (r < -1).any()
            or not np.isfinite(parameters).all() or min(parameters[:3]) < 0 or periods <= 0):
        raise ValueError("Invalid returns, leverage, costs or annualization")
    if invalid_policy not in ("raise", "liquidate"):
        raise ValueError("invalid_policy must be raise or liquidate")
    cost = (annual_fee + max(leverage - 1, 0) * annual_financing) / periods
    raw = leverage * r - cost
    wealth = [1.0]
    realized = []
    exhausted_at = None
    for day, value in enumerate(raw):
        if exhausted_at is not None:
            realized.append(0.0)  # no exposure after liquidation
            wealth.append(0.0)
            continue
        if value <= -1:
            if invalid_policy == "raise":
                raise ValueError(f"Modeled loss reaches 100% at observation {day}")
            exhausted_at = day
            value = -1.0
        realized.append(float(value))
        wealth.append(wealth[-1] * (1 + value))
    if not np.isfinite(wealth).all():
        raise ValueError("Modeled wealth overflow")
    return {"model": "synthetic daily-reset NAV", "leverage": leverage,
            "annual_fee": annual_fee, "annual_financing": annual_financing,
            "periods": periods, "daily_cost": cost, "invalid_policy": invalid_policy,
            "underlying_returns": r.tolist(), "fund_returns": realized,
            "wealth": wealth, "total_return": wealth[-1] - 1,
            "exhausted_at": exhausted_at}


def lev_cagr(r: pd.Series, L: float, ann_fee: float = 0.0095,
             ann_fin: float = 0.04) -> float:
    """Historical model CAGR; shares the simulator's cost and liquidation rules."""
    result = simulate_daily_reset(r, L, ann_fee, ann_fin, invalid_policy="liquidate")
    return result["wealth"][-1] ** (252 / len(r)) - 1


def vol_drag(L: float, sigma_daily: float) -> float:
    """Annualized volatility drag at leverage L (the term that bends the curve down)."""
    return 0.5 * (L * sigma_daily) ** 2 * 252.0


def kelly_optimal(r: pd.Series) -> float:
    """Closed-form growth-optimal leverage L* = mean / variance (daily)."""
    return float(r.mean() / r.var()) if len(r) >= 2 and r.var() > 0 else np.nan


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
    print(f"  -> In-sample model peak only; not a future optimum.")
    print("=" * 60)

    if save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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