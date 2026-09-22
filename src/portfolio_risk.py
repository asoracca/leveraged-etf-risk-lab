"""
portfolio_risk.py — leveraged-etf-risk-lab

Reads your Schwab "Positions" CSV export(s) and computes the risk numbers that
actually matter for a book of leveraged ETFs:
  - look-through leverage  (because a 2x ETF is really 2x its underlying)
  - concentration          (how much is ONE bet in disguise)
  - margin status          (borrowing on top of leverage)
  - flags vs YOUR OWN rules (>-30% drawdown, theme > 50%, position size)

It describes exposure and risk ONLY. It never tells you to buy or sell.

USAGE
-----
    python portfolio_risk.py "Individual-Positions-*.csv" "Roth*.csv"
    python portfolio_risk.py            # auto-globs *Positions*.csv in this folder
"""

from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

# ── Leverage factor of each ETF you hold (UPDATE when you add a new ticker) ──
LEVERAGE = {
    "APLX": 2, "ASTX": 2, "KORU": 3, "LABX": 2, "MRVU": 2, "NBIG": 2,
    "QLD": 2, "RKLX": 2, "SMCL": 2, "USD": 2, "LITX": 2,
}
# ── Theme so we can see the REAL bet behind many tickers ──
THEME = {
    "USD": "Semis/AI", "SMCL": "Semis/AI", "LABX": "Semis/AI", "MRVU": "Semis/AI",
    "APLX": "Semis/AI", "NBIG": "Semis/AI", "LITX": "Semis/AI", "QLD": "Semis/AI",
    "ASTX": "Space", "RKLX": "Space", "KORU": "Korea",
}
GAIN_LOSS_THRESHOLD = -30.0  # cost-basis gain %, not peak-to-trough drawdown


# ----------------------------------------------------------------------
def _num(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("--", "", "N/A"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _col(df: pd.DataFrame, key: str):
    for c in df.columns:
        if key.lower() in str(c).lower():
            return c
    return None


def load_positions(path: str) -> tuple[str, pd.DataFrame, float]:
    """Return (account_name, holdings_df, cash)."""
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    acct = (first.split("account")[-1].split("...")[0].strip().strip('"')
            if "account" in first.lower() else os.path.basename(path))

    df = pd.read_csv(path, skiprows=2)
    df.columns = [str(c).strip() for c in df.columns]
    c_sym, c_mv = _col(df, "Symbol"), _col(df, "Mkt Val")
    c_cb, c_gp = _col(df, "Cost Basis"), _col(df, "Gain %")
    if c_sym is None or c_mv is None:
        raise ValueError(
            f"'{os.path.basename(path)}' doesn't look like a Schwab Positions export "
            f"(missing Symbol/Mkt Val columns). Columns found: {list(df.columns)}")

    cash = 0.0
    rows = []
    for _, r in df.iterrows():
        sym = str(r[c_sym]).strip()
        mv = _num(r[c_mv])
        if sym.lower().startswith("cash"):
            cash = mv if not np.isnan(mv) else 0.0
            continue
        if sym.lower().startswith("positions total") or sym in ("", "nan"):
            continue
        rows.append({
            "symbol": sym.upper(),
            "mkt_val": mv,
            "cost": _num(r[c_cb]) if c_cb else np.nan,
            "gain_pct": _num(r[c_gp]) if c_gp else np.nan,
            "leverage": LEVERAGE.get(sym.upper(), np.nan),
            "theme": THEME.get(sym.upper(), "Other"),
        })
    return acct, pd.DataFrame(rows), cash


# ----------------------------------------------------------------------
def analyze(files: list[str]) -> None:
    frames = []
    for path in files:
        acct, h, cash = load_positions(path)
        h["account"] = acct
        h["cash"] = cash
        frames.append(h)
        report_account(acct, h, cash)

    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        total_cash = sum(f["cash"].iloc[0] for f in frames)
        report_account("COMBINED (all accounts)", combined, total_cash)


def report_account(name: str, h: pd.DataFrame, cash: float) -> None:
    h = h.dropna(subset=["mkt_val"]).copy()
    unknown = h[h["leverage"].isna()]["symbol"].unique().tolist()
    h["leverage"] = h["leverage"].fillna(1.0)
    # combine the same ticker held across multiple accounts into one row
    h = (h.groupby(["symbol", "theme", "leverage"], as_index=False)
           .agg(mkt_val=("mkt_val", "sum"), cost=("cost", "sum")))
    h["gain_pct"] = np.where(h["cost"] > 0,
                             (h["mkt_val"] - h["cost"]) / h["cost"] * 100.0, np.nan)

    gross = h["mkt_val"].sum()                       # long ETF market value
    equity = gross + cash                            # cash may be negative (margin)
    lookthrough = (h["mkt_val"] * h["leverage"]).sum()
    eff_lev = lookthrough / equity if equity else np.nan

    print("\n" + "=" * 62)
    print(f"  {name}")
    print("=" * 62)
    print(f"  Equity (incl. cash)      : ${equity:,.0f}")
    print(f"  Gross ETF market value   : ${gross:,.0f}")
    print(f"  Cash                     : ${cash:,.0f}"
          + ("   ⚠ MARGIN (borrowing)" if cash < -1 else ""))
    print(f"  Look-through exposure    : ${lookthrough:,.0f}   (ETF value × its 2x/3x)")
    print(f"  EFFECTIVE LEVERAGE       : {eff_lev:0.2f}x"
          + ("   ⚠ above 2x" if eff_lev and eff_lev > 2 else ""))
    if unknown:
        print(f"  [warn] no leverage set for {unknown} — assumed 1x. Add them to LEVERAGE.")

    # Concentration by holding
    h = h.sort_values("mkt_val", ascending=False)
    top = h.iloc[0]
    print("-" * 62)
    print(f"  {'Symbol':<7}{'Theme':<11}{'Value':>11}{'% gross':>9}{'Gain%':>8}")
    for _, r in h.iterrows():
        flag = "  ⚠<-30%" if (not np.isnan(r["gain_pct"]) and r["gain_pct"] < GAIN_LOSS_THRESHOLD) else ""
        print(f"  {r['symbol']:<7}{r['theme']:<11}${r['mkt_val']:>9,.0f}"
              f"{r['mkt_val']/gross*100:>8.1f}%{r['gain_pct']:>7.0f}%{flag}")

    # Concentration by theme
    theme = h.groupby("theme")["mkt_val"].sum().sort_values(ascending=False)
    print("-" * 62)
    print("  Theme concentration (share of gross exposure):")
    for t, v in theme.items():
        bar = "█" * int(round(v / gross * 20))
        print(f"    {t:<10} {v/gross*100:5.1f}%  {bar}")

    # Flags vs her own rules
    print("-" * 62)
    print("  Risk check (context only — no buy/sell advice):")
    print(f"    • Largest single position: {top['symbol']} = "
          f"{top['mkt_val']/equity*100:.0f}% of equity "
          f"(capital concentration; not a maximum loss estimate).")
    biggest_theme = theme.index[0]
    if theme.iloc[0] / gross > 0.50:
        print(f"    • {biggest_theme} is {theme.iloc[0]/gross*100:.0f}% of the book "
              f"→ this is largely ONE bet, not diversified.")
    dd = h[h["gain_pct"] < GAIN_LOSS_THRESHOLD]["symbol"].tolist()
    if dd:
        print(f"    • Cost-basis gain < -30% threshold (not drawdown): {', '.join(dd)}.")
    if cash < -1:
        print(f"    • Margin debit of ${-cash:,.0f} — leverage stacked on leverage.")
    print("    Thresholds are illustrative assumptions; no trade action is implied.")
    print("=" * 62)


# ----------------------------------------------------------------------
def main():
    # expand any globs passed as arguments
    expanded = []
    for f in sys.argv[1:]:
        expanded.extend(sorted(glob.glob(f)) or [f])
    # if none passed, auto-find ONLY Schwab position exports (never other CSVs)
    if not expanded:
        expanded = sorted(glob.glob("*Positions*.csv")) + sorted(glob.glob("*-Positions-*.csv"))
        expanded = sorted(set(expanded))
    if not expanded:
        msg = (
            "No position CSVs found in this folder.\n"
            "Put your Schwab exports here (names containing 'Positions'), then rerun,\n"
            "or pass the paths explicitly, e.g.:\n"
            '  python portfolio_risk.py "Individual-Positions-*.csv" "Roth*.csv"'
        )
        print(msg)
        return
    print(f"Loading: {expanded}")
    analyze(expanded)


if __name__ == "__main__":
    main()