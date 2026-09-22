# Leveraged ETF Risk Lab

A Python workbench for portfolio return conventions, concentrated exposure and
path-dependent daily leverage. The reproducible demo uses only **synthetic data**.
Historical analyses are educational measurements, not trade recommendations.

## Offline demo

Python 3.11 or 3.12. Install once (requires package access), then run offline:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m src.demo --seed 42 --output reports/demo.json
python -m unittest discover -s tests -v
```

The demo writes strict JSON and also prints it. It never loads personal holdings,
requests credentials, or contacts a data service. Optional chart/HMM tests skip
when their dependencies are absent; the CI matrix installs and tests both.

| Synthetic example | Result |
|---|---:|
| 50/50 daily rebalanced fixture | +4.7375% |
| Same initial weights, buy-and-hold | +4.4500% |
| Rebalanced max drawdown / benchmark beta | −5.0% / 1.0 |
| Absolute risk components sum | 0.916515 annual volatility |
| Same-terminal flat / volatile path, gross 3x | 0% / −5.4545% |
| Two labels sharing one growth factor | 2.5x initial capital exposure |
| Missing first B price, common-start policy | −0.25% over the shorter window |
| Same missing history, strict policy | Explicit rejection |

Change assumptions and compare the exported report:

```bash
python -m src.demo --seed 42 --weight-a 0.25 \
  --scenario mild_growth_shock --mode buy_and_hold --output reports/alternative.json
```

The report contains prices, weights, daily returns, sample window, costs, factor
maps, scenario shocks and rule thresholds. Undefined statistics are JSON `null`.
The seed controls a separate illustrative 60-session draw; the hand fixture is fixed.

## Architecture and methods

- `src/risk.py`: shared validation, history policies, simple returns, daily
  rebalancing, buy-and-hold drift, beta, drawdown and volatility attribution.
- `src/optimal_leverage.py`: daily-reset model with fees, financing and explicit
  liquidation policy; the historical leverage curve reuses it.
- `src/rebalance.py`, `stress.py`, `exposure.py`: auditable threshold evaluations,
  deterministic shocks and explicit factor assumptions.
- `src/demo.py`: offline orchestration and JSON output. It uses the same Python
  functions as historical reports; there is no duplicate browser calculation.
- `tests/`: hand-calculated fixture, edge cases, CLI integration, synthetic chart
  smoke test and optional HMM fitting/evaluation boundary tests.

[Methodology](docs/methodology.md) defines every convention and derives the fixture.
The daily model is not an exact fund replica. Short samples, serial dependence,
tracking error, transaction costs and uncertain exposure maps limit interpretation.
An optional TypeScript dashboard is not included in this version.

## Historical data and figures

```bash
python -m pip install -r requirements.txt -r requirements-regimes.txt
python main.py
```

`main.py` downloads historical adjusted closes from Yahoo Finance, reads ignored
`portfolio_values.csv` if present (otherwise illustrative equal weights), and runs
risk, theme exposure, rule checks, stress and charts. Missing histories follow
common-start alignment; gaps and missing held symbols fail explicitly. These are
hypothetical historical allocations, not reconstructed account returns.

The optional regime, leverage-curve, positions-CSV and rolling-analysis modules
are **separate entry points**, not automatically invoked by `main.py`.
For a local historical `date,price` CSV, inferred HMM states can be inspected with:

```bash
python -m src.regime_detection private/prices.csv --train-end 2024-12-31 --seed 42
```

Evaluation must follow the training window. Inferred volatility states are not
future predictions. The old hindsight timing backtest is no longer exposed.

Existing `assets/` figures and `docs/archive/` are retained historical artifacts;
they are not validation results for the corrected math. New report outputs go in
ignored `reports/` or `data/`. `portfolio_values.example.csv` contains uniform
synthetic amounts. Keep account exports in ignored `private/`, never in fixtures,
public screenshots or commits. Existing repository history is not rewritten.

## Validation

```bash
python -m unittest discover -s tests -v
python -m src.demo --seed 42 --output reports/demo.json
```

CI runs these offline computations on Python 3.11 and 3.12 after dependency
installation and retains the synthetic JSON artifact. Optional integration tests
replace the market-data fetcher and block sockets. No live data accuracy or trading
performance claim is made by passing tests.
