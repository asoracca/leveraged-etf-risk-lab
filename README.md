# Leveraged ETF Risk Lab

A quantitative risk dashboard for a concentrated portfolio of leveraged ETFs and high-volatility thematic exposure.

## Research Question

Am I being paid for taking aggressive leveraged ETF risk, or am I accidentally stacking the same trade across semiconductors, AI, space, biotech, and high-beta growth?

This project was built from a real personal investing problem: I hold a portfolio tilted toward leveraged ETFs and high-volatility themes. The goal is not to make the portfolio look safer than it is. The goal is to measure the risk clearly enough to make better decisions.

## Why This Project Exists

Leveraged ETFs can be useful tactical instruments, but they are path-dependent. A 2x or 3x ETF does not simply deliver two or three times the long-term return of the underlying index. Daily reset, volatility decay, drawdowns, and correlated exposure can dominate the result.

This project turns a risky personal portfolio into a quant research lab:

- measure portfolio beta vs SPY, QQQ, and SOXX
- estimate annualized volatility and max drawdown
- identify which holdings contribute the most risk
- detect crowded/correlated exposure
- run simple stress tests
- keep a weekly decision journal

## Example Universe

The default ticker list is based on my current leveraged ETF themes:

| Ticker | Theme |
|---|---|
| QLD | 2x Nasdaq-100 |
| USD | 2x Semiconductors |
| RKLX | 2x Rocket Lab exposure |
| SMCL | 2x Super Micro exposure |
| APLX | 2x Apple exposure |
| ASTX | 2x AST SpaceMobile exposure |
| KORU | 3x South Korea equities |
| LABX | 2x biotech exposure |
| MRVU | 2x Magnificent 7 exposure |
| NBIG | 2x Nebius exposure |
| LITX | 2x lithium exposure |

You can edit tickers in `src/config.py`. For real weights, copy `portfolio_values.example.csv` to `portfolio_values.csv` and update the market values from your broker.

## Methodology

The dashboard uses daily adjusted close data from Yahoo Finance.

Core metrics:

- log returns
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- beta vs SPY, QQQ, and SOXX
- correlation matrix
- risk contribution by holding
- scenario stress tests

## Stress Tests

The project asks practical questions:

- What if SPY drops 5%?
- What if QQQ drops 8%?
- What if semiconductors drop 10%?
- What if every leveraged position has a bad day at once?
- Which ticker contributes the most portfolio damage?

These are not predictions. They are risk rehearsals.

## Quickstart

```bash
git clone https://github.com/asoracca/leveraged-etf-risk-lab.git
cd leveraged-etf-risk-lab
pip install -r requirements.txt
python main.py
```

Outputs are saved to `data/`.

To use real portfolio weights:

```bash
cp portfolio_values.example.csv portfolio_values.csv
# edit portfolio_values.csv with current broker market values
python main.py
```

## Journal

The `journal/risk_journal.csv` file is for weekly observations.

The goal is to track:

- current exposure
- largest risk contributor
- portfolio beta
- portfolio drawdown
- decision made
- what happened 1-4 weeks later

That turns this from a dashboard into a live research process.

## Personal Context

I am building a quant portfolio around markets I actually touch: leveraged ETFs, options premium, space stocks, semiconductors, AI, and high-volatility thematic exposure.

This project is the risk layer. It connects:

- SOXL Vol Surface: options premium and volatility
- New Space Radar: catalyst-driven equity signals
- Leveraged ETF Risk Lab: portfolio-level risk, leverage, and drawdown control

The common theme is learning to turn aggressive investing instincts into measurable, testable, risk-aware systems.

## Not Financial Advice

This project is for research and education. Leveraged ETFs can lose money quickly, especially over multi-day holding periods in volatile markets.
