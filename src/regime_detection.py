"""
regime_detection.py  —  add to leveraged-etf-risk-lab/src/
-----------------------------------------------------------
Market regime detection using a 2-state Hidden Markov Model (HMM).

WHY REGIMES MATTER
------------------
Markets don't behave the same way all the time. There are two persistent states:

  State 0 — Bull / Low-Vol regime
    • Low daily volatility (~12% annualised)
    • Positive drift (markets trend up)
    • VIX typically below 20
    • Strategy: hold your leveraged ETFs, be aggressive

  State 1 — Bear / High-Vol regime
    • High daily volatility (~35%+ annualised)
    • Negative or flat drift (markets chop or crash)
    • VIX typically above 25
    • Strategy: reduce leverage, raise cash, hedge

A 3x leveraged ETF in a high-vol bear regime doesn't just underperform —
it gets destroyed by volatility decay. SOXL in 2022: -79%.

HOW HMM WORKS
-------------
You can't directly observe which "regime" the market is in — hence "hidden."
But you CAN observe returns and volatility. The HMM learns:

  1. Two hidden states (regimes) with different return/vol characteristics
  2. Transition probabilities — how likely is the regime to switch today?
  3. Given today's returns, what's the probability we're in each state?

The math:
  - Each state emits returns from a Gaussian: N(μ_k, σ_k²)
  - Transition matrix: P(state tomorrow | state today)
  - Viterbi algorithm: find the most likely sequence of hidden states

This is exactly what hedge funds use for regime-conditional allocation.
Standard in systematic macro and vol-targeting strategies.

WHAT YOU LEARN
--------------
1. Hidden Markov Models — probabilistic sequence models used in finance, NLP, genomics
2. Expectation-Maximization (EM) algorithm — how the model trains itself
3. Viterbi algorithm — dynamic programming to find the optimal state sequence
4. Regime-conditional backtesting — different strategies for different regimes
5. Vol-targeting — adjusting leverage to maintain constant portfolio vol

Install:
    pip install hmmlearn

Run:
    python src/regime_detection.py

Or import:
    from src.regime_detection import fit_regime_model, get_current_regime
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
import yfinance as yf

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn not installed. Run: pip install hmmlearn --break-system-packages")


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(ticker="SPY", period="5y"):
    """Fetch daily prices and compute features for regime detection."""
    print(f"Fetching {ticker} ({period})...")
    data = yf.Ticker(ticker).history(period=period)
    data.index = pd.to_datetime(data.index).tz_localize(None)

    # Features for regime detection
    data["log_ret"]  = np.log(data["Close"] / data["Close"].shift(1))
    data["rv5"]      = data["log_ret"].rolling(5).std()  * np.sqrt(252)
    data["rv20"]     = data["log_ret"].rolling(20).std() * np.sqrt(252)
    data["abs_ret"]  = data["log_ret"].abs()

    return data.dropna()


def fetch_vix(period="5y"):
    """Fetch VIX for cross-validation of regime labels."""
    vix = yf.Ticker("^VIX").history(period=period)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix[["Close"]].rename(columns={"Close": "VIX"}).dropna()


# ── HMM Model ────────────────────────────────────────────────────────────────

def fit_regime_model(data, n_states=2, n_iter=200):
    """
    Fit a Gaussian HMM to returns and volatility.

    The model uses two features:
      1. Log return (the direction/magnitude of today's move)
      2. 5-day rolling vol (how volatile the recent environment is)

    n_states=2: Bull (low-vol) vs Bear (high-vol)
    n_iter:     EM algorithm iterations (more = better fit)

    Returns:
        model    — fitted HMM
        states   — Series of regime labels (0 or 1) per day
        probs    — DataFrame of state probabilities per day
    """
    if not HMM_AVAILABLE:
        print("⚠️  Using fallback: vol-threshold regime (VIX proxy)")
        return _fallback_regime(data)

    # Feature matrix: [log_return, 5d_realized_vol]
    X = data[["log_ret", "rv5"]].values

    # Fit Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
    )
    model.fit(X)

    # Predict states (Viterbi)
    raw_states = model.predict(X)

    # Relabel: state with LOWER vol = State 0 (bull), HIGHER vol = State 1 (bear)
    # Find which state has lower mean vol
    state_vols = {}
    for s in range(n_states):
        mask = raw_states == s
        state_vols[s] = data["rv5"].values[mask].mean()

    low_vol_state  = min(state_vols, key=state_vols.get)
    high_vol_state = max(state_vols, key=state_vols.get)

    # Remap so state 0 = bull, state 1 = bear
    remap = {low_vol_state: 0, high_vol_state: 1}
    states = pd.Series([remap[s] for s in raw_states], index=data.index, name="regime")

    # State probabilities
    probs = pd.DataFrame(
        model.predict_proba(X),
        index=data.index,
        columns=[f"p_state_{i}" for i in range(n_states)]
    )
    # Remap prob columns to match relabeled states
    probs = probs.rename(columns={
        f"p_state_{low_vol_state}":  "p_bull",
        f"p_state_{high_vol_state}": "p_bear",
    })

    # Print transition matrix and state stats
    _print_model_summary(model, states, data, low_vol_state, high_vol_state)

    return model, states, probs


def _fallback_regime(data):
    """
    Fallback when hmmlearn isn't installed.
    Simple threshold: high vol (rv20 > 25%) = bear regime.
    """
    states = pd.Series(
        (data["rv20"] > 0.25).astype(int),
        index=data.index,
        name="regime"
    )
    probs = pd.DataFrame({
        "p_bull": (states == 0).astype(float),
        "p_bear": (states == 1).astype(float),
    }, index=data.index)
    print("  Using vol-threshold fallback (rv20 > 25% = bear)")
    return None, states, probs


def _print_model_summary(model, states, data, low_state, high_state):
    """Print transition probabilities and regime characteristics."""
    print("\n── HMM Regime Summary ─────────────────────────────────")

    n_bull = (states == 0).sum()
    n_bear = (states == 1).sum()
    n_total = len(states)

    print(f"  State 0 — BULL (low-vol):  {n_bull:>4} days ({n_bull/n_total:.0%})")
    print(f"  State 1 — BEAR (high-vol): {n_bear:>4} days ({n_bear/n_total:.0%})")

    # Regime statistics
    for label, state_id in [("BULL", 0), ("BEAR", 1)]:
        mask = states == state_id
        ret_ann  = data.loc[mask, "log_ret"].mean() * 252
        vol_ann  = data.loc[mask, "log_ret"].std()  * np.sqrt(252)
        sharpe   = ret_ann / vol_ann if vol_ann > 0 else 0
        print(f"\n  {label} regime:")
        print(f"    Ann. return:  {ret_ann:+.1%}")
        print(f"    Ann. vol:     {vol_ann:.1%}")
        print(f"    Sharpe:       {sharpe:.2f}")

    # Transition matrix
    T = model.transmat_
    print(f"\n  Transition matrix (rows = current state, cols = next state):")
    print(f"                  → Bull    → Bear")
    print(f"  From Bull:       {T[low_state, low_state]:.2f}      {T[low_state, high_state]:.2f}")
    print(f"  From Bear:       {T[high_state, low_state]:.2f}      {T[high_state, high_state]:.2f}")

    # Persistence = how many days the regime tends to last
    bull_persist = 1 / (1 - T[low_state, low_state])
    bear_persist = 1 / (1 - T[high_state, high_state])
    print(f"\n  Expected duration: Bull = {bull_persist:.0f} days | Bear = {bear_persist:.0f} days")
    print(f"{'─'*55}")


# ── Current regime ────────────────────────────────────────────────────────────

def get_current_regime(model, states, probs, data):
    """
    What regime are we in TODAY? What should you do?
    """
    today_state = int(states.iloc[-1])
    today_p_bear = float(probs["p_bear"].iloc[-1])
    today_ret    = float(data["log_ret"].iloc[-1]) * 252
    today_vol    = float(data["rv20"].iloc[-1])

    print("\n" + "="*55)
    print(f"  CURRENT MARKET REGIME — {states.index[-1].date()}")
    print("="*55)
    print(f"  Regime:           {'BULL (low-vol)' if today_state == 0 else 'BEAR (high-vol)'}")
    print(f"  Bear probability: {today_p_bear:.1%}")
    print(f"  20d Realized Vol: {today_vol:.1%}")
    print(f"  Current drift:    {today_ret:+.1%} ann.")
    print(f"{'─'*55}")

    if today_state == 0:
        if today_p_bear < 0.2:
            print(f"  Signal:  ✅ HOLD / RISK ON — low-vol bull regime")
            print(f"           Leveraged ETFs in their element. SOXL/QLD favourable.")
        else:
            print(f"  Signal:  ⚠️  WATCH — bull regime but bear probability rising")
            print(f"           Regime may be transitioning. Tighten stops.")
    else:
        if today_p_bear > 0.8:
            print(f"  Signal:  ❌ RISK OFF — high-conviction bear regime")
            print(f"           Reduce leveraged ETF exposure. Volatility decay accelerates.")
        else:
            print(f"  Signal:  ⚠️  CAUTION — bear regime, moderate conviction")
            print(f"           Monitor for regime flip. Don't add to leveraged positions.")

    print("="*55)

    return {
        "regime":   "bull" if today_state == 0 else "bear",
        "p_bear":   today_p_bear,
        "rv20":     today_vol,
        "signal":   "risk_on" if today_state == 0 else "risk_off",
    }


# ── Regime-conditional backtest ──────────────────────────────────────────────

def regime_conditional_backtest(data, states):
    """
    Compare strategy performance in bull vs bear regimes.

    Key question: does SOXL earn its risk premium only in bull regimes?
    If so, a regime filter (reduce exposure in bear) should improve Sharpe.

    This is the vol-targeting idea: instead of constant 3x leverage,
    target constant portfolio volatility — reduce leverage when vol is high.
    """
    combined = data.join(states, how="inner")
    returns  = combined["log_ret"]

    print("\n── Regime-Conditional Performance ─────────────────────")

    results = {}
    for label, state_id in [("BULL (State 0)", 0), ("BEAR (State 1)", 1)]:
        mask = combined["regime"] == state_id
        r    = returns[mask]

        if len(r) < 10:
            continue

        ann_ret = r.mean() * 252
        ann_vol = r.std()  * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
        win_rt  = (r > 0).mean()
        n_days  = len(r)

        print(f"\n  {label} ({n_days} days):")
        print(f"    Return (ann):  {ann_ret:+.1%}")
        print(f"    Vol (ann):     {ann_vol:.1%}")
        print(f"    Sharpe:        {sharpe:.2f}")
        print(f"    Win rate:      {win_rt:.0%}")

        results[label] = {
            "ann_return": ann_ret,
            "ann_vol":    ann_vol,
            "sharpe":     sharpe,
        }

    # Regime-switching strategy: invest only in bull, cash in bear
    bull_mask = combined["regime"] == 0
    strategy_ret = returns.copy()
    strategy_ret[~bull_mask] = 0.0  # cash out in bear regime

    full_ann_ret  = returns.mean() * 252
    full_ann_vol  = returns.std()  * np.sqrt(252)
    full_sharpe   = full_ann_ret / full_ann_vol

    strat_ann_ret = strategy_ret.mean() * 252
    strat_ann_vol = strategy_ret[strategy_ret != 0].std() * np.sqrt(252) if bull_mask.any() else 0
    strat_sharpe  = strat_ann_ret / full_ann_vol  # penalise by full vol

    print(f"\n  COMPARISON (Buy & Hold vs Regime-Filtered):")
    print(f"  {'Strategy':35}  {'Sharpe':>8}  {'Ann Ret':>9}")
    print(f"  {'─'*55}")
    print(f"  {'Buy & Hold (always invested)':35}  {full_sharpe:>8.2f}  {full_ann_ret:>+8.1%}")
    print(f"  {'Regime-filtered (bull only)':35}  {strat_sharpe:>8.2f}  {strat_ann_ret:>+8.1%}")

    improvement = strat_sharpe - full_sharpe
    if improvement > 0.1:
        print(f"\n  ✅ Regime filter adds {improvement:.2f} Sharpe — worth implementing")
    elif improvement > 0:
        print(f"\n  ⚠️  Small improvement ({improvement:.2f} Sharpe) — regime filter marginal")
    else:
        print(f"\n  ❌ Regime filter hurts ({improvement:.2f} Sharpe) — buy-and-hold better")

    return results, strategy_ret


# ── Vol targeting ─────────────────────────────────────────────────────────────

def compute_vol_targeted_returns(returns, target_vol=0.15, lookback=20):
    """
    Vol-targeting: scale position size so portfolio vol ≈ target_vol.

    In a high-vol regime, reduce leverage. In low-vol, increase.
    The leverage at each time t:
        leverage_t = target_vol / realized_vol_t

    Cap leverage at 2.0 (don't over-leverage), floor at 0 (no short).

    This is what systematic macro funds do. They don't hold fixed leverage —
    they target a volatility budget and dynamically size positions.
    """
    rv = returns.rolling(lookback).std() * np.sqrt(252)
    leverage = (target_vol / rv).clip(0, 2.0)   # cap at 2x
    vt_returns = returns * leverage.shift(1)     # use yesterday's leverage

    ann_ret_bh  = returns.mean() * 252
    ann_vol_bh  = returns.std()  * np.sqrt(252)
    sharpe_bh   = ann_ret_bh / ann_vol_bh

    ann_ret_vt  = vt_returns.mean() * 252
    ann_vol_vt  = vt_returns.std()  * np.sqrt(252)
    sharpe_vt   = ann_ret_vt / ann_vol_vt

    avg_lev = leverage.mean()

    print(f"\n── Vol-Targeting (target={target_vol:.0%} ann vol) ─────────────")
    print(f"  Avg leverage:      {avg_lev:.2f}x")
    print(f"  {'Strategy':30}  {'Sharpe':>8}  {'Ann Ret':>9}  {'Ann Vol':>9}")
    print(f"  {'─'*58}")
    print(f"  {'Buy & Hold':30}  {sharpe_bh:>8.2f}  {ann_ret_bh:>+8.1%}  {ann_vol_bh:>8.1%}")
    print(f"  {'Vol-Targeted':30}  {sharpe_vt:>8.2f}  {ann_ret_vt:>+8.1%}  {ann_vol_vt:>8.1%}")

    if sharpe_vt > sharpe_bh:
        print(f"\n  ✅ Vol-targeting improves Sharpe by {sharpe_vt-sharpe_bh:.2f}")
    else:
        print(f"\n  ❌ Vol-targeting hurts by {sharpe_vt-sharpe_bh:.2f} (regime not volatile enough to matter)")

    return vt_returns, leverage


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_regimes(data, states, probs, vix_data=None, ticker="SPY"):
    """
    4-panel dashboard:
      1. Price coloured by regime (green = bull, red = bear)
      2. Regime probability (bear state)
      3. Realized vol + VIX for validation
      4. Cumulative return by regime
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    fig.suptitle(f"Market Regime Detection — {ticker} (2-State HMM)",
                 fontsize=14, fontweight="bold")

    bull_color = "#1D9E75"
    bear_color = "#A32D2D"

    # Panel 1: Price coloured by regime
    ax = axes[0]
    for i, date in enumerate(data.index[:-1]):
        next_date = data.index[i + 1]
        color = bull_color if states.iloc[i] == 0 else bear_color
        ax.fill_betweenx(
            [data["Close"].min() * 0.99, data["Close"].max() * 1.01],
            date, next_date, color=color, alpha=0.15
        )
    ax.plot(data.index, data["Close"], color="black", linewidth=1.2, zorder=5)
    ax.set_ylabel(f"{ticker} Price ($)")
    ax.set_title(f"{ticker} Price — Green = Bull Regime | Red = Bear Regime")
    bull_patch = mpatches.Patch(color=bull_color, alpha=0.5, label="Bull / Low-Vol")
    bear_patch = mpatches.Patch(color=bear_color, alpha=0.5, label="Bear / High-Vol")
    ax.legend(handles=[bull_patch, bear_patch], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)

    # Panel 2: Bear probability
    ax2 = axes[1]
    ax2.fill_between(probs.index, probs["p_bear"], 0,
                     color=bear_color, alpha=0.4, label="P(Bear regime)")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="50% threshold")
    ax2.axhline(0.8, color=bear_color, linestyle="--", linewidth=0.8, alpha=0.5, label="High conviction")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(Bear)")
    ax2.set_title("Probability of Being in Bear Regime")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Panel 3: Realized vol + VIX
    ax3 = axes[2]
    rv20 = data["rv20"] * 100
    ax3.plot(data.index, rv20, color="#378ADD", linewidth=1.3, label="20d Realized Vol (%)")
    if vix_data is not None:
        vix_aligned = vix_data.reindex(data.index, method="ffill")
        ax3.plot(vix_aligned.index, vix_aligned["VIX"], color="grey",
                 linewidth=1.0, alpha=0.7, label="VIX")
    ax3.axhline(20, color=bull_color, linestyle="--", alpha=0.5, linewidth=0.8, label="Bull threshold (~20%)")
    ax3.axhline(30, color=bear_color, linestyle="--", alpha=0.5, linewidth=0.8, label="Bear threshold (~30%)")
    ax3.set_ylabel("Volatility (%)")
    ax3.set_title("Realized Vol + VIX — Regime Validation")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.2)

    # Panel 4: Cumulative returns by regime (vs buy-and-hold)
    ax4 = axes[3]
    combined = data.join(states, how="inner")

    # Regime-filtered: only invest in bull
    r_bh  = combined["log_ret"]
    r_filt = r_bh.copy()
    r_filt[combined["regime"] == 1] = 0.0

    cum_bh   = np.exp(r_bh.cumsum())
    cum_filt = np.exp(r_filt.cumsum())

    ax4.plot(cum_bh.index, cum_bh.values, color="grey",     linewidth=1.3, label="Buy & Hold")
    ax4.plot(cum_filt.index, cum_filt.values, color=bull_color,
             linewidth=1.8, linestyle="--", label="Regime-Filtered (cash in bear)")
    ax4.axhline(1, color="black", linewidth=0.6, alpha=0.4)
    ax4.set_ylabel("Growth of $1")
    ax4.set_title("Cumulative Return: Buy & Hold vs Regime-Filtered")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/regime_detection.png", dpi=150, bbox_inches="tight")
    print("\nSaved: data/regime_detection.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys

    # Auto-install hmmlearn if missing
    if not HMM_AVAILABLE:
        print("Installing hmmlearn...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn",
                               "--break-system-packages", "-q"])
        from hmmlearn import hmm
        HMM_AVAILABLE = True

    print("\n" + "="*55)
    print("  MARKET REGIME DETECTION")
    print("  Model: 2-State Gaussian HMM")
    print("  Features: log returns + 5d realized vol")
    print("="*55)

    # Use SPY — more data, cleaner regime signal than SOXL
    # (SOXL's own regimes are just levered SPY with amplified noise)
    data = fetch_data(ticker="SPY", period="5y")
    vix  = fetch_vix(period="5y")

    # Fit HMM
    model, states, probs = fit_regime_model(data, n_states=2)

    # Current regime signal
    result = get_current_regime(model, states, probs, data)

    # Regime-conditional backtest on SPY
    regime_results, strategy_returns = regime_conditional_backtest(data, states)

    # Vol-targeting
    vt_returns, leverage = compute_vol_targeted_returns(
        data["log_ret"], target_vol=0.15, lookback=20
    )

    # Plot
    plot_regimes(data, states, probs, vix_data=vix, ticker="SPY")

    print("\n── What to do with this ───────────────────────────────")
    print("  1. Add regime signal to daily iv_rank.py output")
    print("  2. Only sell SOXL puts when BOTH IV Rank > 50 AND regime = BULL")
    print("  3. Scale position size by leverage (vol-targeting)")
    print("  4. Check regime at market open — if flip to BEAR, hedge or reduce")
    print("\nDone. Check data/regime_detection.png")
