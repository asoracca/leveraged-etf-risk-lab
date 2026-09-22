# Financial conventions

All demo inputs are synthetic, normalized prices and capital weights. Reports from
Yahoo adjusted closes are historical estimates downloaded from a live service,
not account performance. Current weights applied to past data describe a
hypothetical portfolio, not the owner's past holdings.

## Returns and allocations

Prices must be positive, on a unique increasing trading-session index. A row is
one session; the caller must supply an appropriate trading calendar. Simple return
is `r[t] = P[t]/P[t-1] - 1`. Log return is `log(1+r[t])`; weighted asset log returns
are **not** exact portfolio simple returns. The portfolio core uses simple returns.

Weights are nonnegative capital amounts normalized once to sum to one. Negative,
nonfinite, empty and all-zero weights are errors. Zero-weight assets are excluded.
No holding may disappear silently. Leverage inside a fund belongs in its modeled
return or exposure map, not in negative capital weights.

Daily constant weights: `r_p[t] = sum(w[i] * r[i,t])`, with costless rebalancing each
session. Buy-and-hold: initial quantities are fixed and weights drift with asset
wealth; each day's P&L is divided by the previous portfolio wealth. No flows,
taxes, commissions, spreads or account-level borrowing are modeled.

## Missing histories and alignment

`common_start` (default) starts at the first complete held-asset price row and
reports how many leading rows were removed. `strict` requires every held price
on the requested window. Both reject internal/trailing gaps and absent symbols.
There is no forward fill, backfill, zero-return substitution, pairwise covariance,
or renormalization into surviving assets. At least three common prices are needed.
Benchmark returns are computed on original adjacent rows, then matched to portfolio
return dates; beta drops unavailable pairs and needs at least two pairs. Benchmarks
do not shorten the portfolio's own history. Calendar synchronization is the
caller's responsibility; these policies cannot identify an omitted whole session.

## Risk and compounding

Wealth starts at 1 and compounds `product(1+r_p)`. Drawdown is wealth divided by
its running peak **including initial wealth 1**, minus 1. Total return is final
wealth minus 1. `annual_return` means arithmetic mean daily return times 252,
not CAGR or a prediction. Volatility is sample standard deviation (`ddof=1`)
times `sqrt(252)`. The period count is configurable. These scaling assumptions
ignore serial dependence; very short samples are pedagogical, not estimates.

Risk-free rate defaults to effective annual 0, converted as
`(1+rf)^(1/252)-1`. Sharpe uses annualized mean daily excess return divided by
annualized volatility. Zero volatility gives undefined Sharpe. Beta uses sample
covariance divided by benchmark sample variance; a flat benchmark gives undefined
beta. Undefined values appear as `null` in JSON, not a claim of zero risk.

For constant weights and annual covariance `C`, absolute volatility components are
`w[i]*(Cw)[i]/sigma_p`; their sum is `sigma_p`. Risk shares divide components by
`sigma_p` and sum to 1. For drift, attribution uses realized daily constituent P&L
contributions `x[i,t]`: `252*Cov(x[i],r_p)/sigma_p`. This is historical attribution,
not the derivative with respect to today's weights. A zero-volatility portfolio
has zero absolute components and undefined shares. Negative components are
possible from covariance even with long-only holdings.

## Hand calculation

The checked-in fixture has A returns `[.1,-.1,.1]`, flat B, and a benchmark with
`[.05,-.05,.05]`. At 50/50 daily weights, portfolio returns are `[.05,-.05,.05]`:
wealth `[1,1.05,.9975,1.047375]`, total `4.7375%`, drawdown `-5%`, beta `1`.
Sample daily variance is `1/300`, annual variance `.84`, volatility `sqrt(.84)`.
A contributes all volatility and B zero. Buy-and-hold final wealth is
`.5*1.089 + .5 = 1.0445`. The example deliberately magnifies daily movements.

## Implementation references

The pinned core uses [pandas 2.2.3 fractional changes](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.pct_change.html)
with `fill_method=None`, and sample covariance consistent with the
[NumPy 2.2 covariance convention](https://numpy.org/doc/2.2/reference/generated/numpy.cov.html).
