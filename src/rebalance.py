"""Educational threshold evaluations; no action labels or trade recommendations."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.risk import compute_summary

RULES = {"max_single_risk_contribution": .35, "max_beta_qqq": 2.50,
         "max_beta_spy": 2.25, "max_annual_vol": .75, "max_drawdown": -.20}


def evaluate_rules(summary, risk_shares, rules=None):
    limits = dict(RULES)
    if rules is not None:
        if set(rules) - set(limits):
            raise ValueError("Unknown rule threshold")
        limits.update(rules)
    if not np.isfinite(list(limits.values())).all():
        raise ValueError("Thresholds must be finite")
    inputs = [("max_drawdown", summary['max_drawdown'], '<=', limits['max_drawdown']),
              ("annual_vol", summary['annual_vol'], '>=', limits['max_annual_vol']),
              ("beta_SPY", summary['betas'].get('SPY'), '>=', limits['max_beta_spy']),
              ("beta_QQQ", summary['betas'].get('QQQ'), '>=', limits['max_beta_qqq'])]
    inputs += [(f"risk_share_{ticker}", value, '>=', limits['max_single_risk_contribution'])
               for ticker, value in risk_shares.items()]
    rows = []
    for metric, value, op, threshold in inputs:
        known = value is not None and np.isfinite(value)
        triggered = bool(value <= threshold if op == '<=' else value >= threshold) if known else None
        rows.append({'metric': metric, 'value': float(value) if known else None,
                     'operator': op, 'threshold': threshold, 'triggered': triggered,
                     'status': 'unavailable' if triggered is None else ('threshold_met' if triggered else 'threshold_not_met'),
                     'trigger': f'{metric} {op} {threshold}',
                     'purpose': 'educational threshold, not a forecast or trade instruction'})
    return rows


def generate_rebalance_recommendations(prices, weights=None):
    """Compatibility entry point; returns rule evaluations, not recommendations."""
    if weights is None:
        from src.holdings import load_weights
        weights = load_weights()
    summary, shares, _ = compute_summary(prices, weights)
    return pd.DataFrame(evaluate_rules(summary, shares)), summary


def print_rebalance_report(prices, weights=None):
    rows, _ = generate_rebalance_recommendations(prices, weights)
    print('\nEDUCATIONAL RULE EVALUATIONS — thresholds are assumptions')
    print(rows[['metric', 'value', 'operator', 'threshold', 'status']].to_string(index=False))
    Path('data').mkdir(exist_ok=True)
    rows.to_csv('data/rule_evaluations.csv', index=False)


if __name__ == '__main__':
    from src.data import fetch_price_history
    print_rebalance_report(fetch_price_history())
