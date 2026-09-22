"""Credential-free synthetic workbench: python -m src.demo."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from src.risk import compute_summary
from src.optimal_leverage import simulate_daily_reset
from src.exposure import look_through_exposure
from src.rebalance import evaluate_rules
from src.stress import run_stress_tests


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_report(seed=42, weight_a=.5, scenario="shared_growth_shock", mode="constant_weight"):
    if not 0 <= weight_a <= 1:
        raise ValueError("weight_a must be between 0 and 1")
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/portfolio.json'
    fixture = json.loads(path.read_text())
    prices = pd.DataFrame(fixture['prices'], index=pd.to_datetime(fixture['dates']))
    reports = {}
    for fixture_mode in ('constant_weight', 'buy_and_hold'):
        summary, shares, returns = compute_summary(prices, fixture['weights'], mode=fixture_mode)
        reports[fixture_mode] = {**summary, 'risk_shares': shares.to_dict(), 'returns': returns.tolist()}
    incomplete = prices.copy()
    incomplete.loc[incomplete.index[0], 'B'] = np.nan
    trimmed, _, _ = compute_summary(incomplete, fixture['weights'])
    try:
        compute_summary(incomplete, fixture['weights'], missing_policy='strict')
    except ValueError as exc:
        strict = {'status': 'rejected', 'reason': str(exc)}
    # Two fund labels with the same explicitly assumed underlying growth factor.
    rng = np.random.default_rng(seed)
    underlying = rng.normal(.0003, .012, 60)
    fund_a = simulate_daily_reset(underlying, leverage=2)
    fund_b = simulate_daily_reset(underlying, leverage=3)
    synthetic = pd.DataFrame({'Growth_2x': fund_a['wealth'], 'Innovation_3x': fund_b['wealth'],
                              'SPY': np.r_[1, np.cumprod(1 + underlying)]},
                             index=pd.bdate_range('2024-01-02', periods=61))
    weights = {'Growth_2x': weight_a, 'Innovation_3x': 1 - weight_a}
    summary, shares, returns = compute_summary(synthetic, weights, mode=mode)
    exposures = {'Growth_2x': {'shared_growth': 2}, 'Innovation_3x': {'shared_growth': 3}}
    scenarios = {'shared_growth_shock': {'Growth_2x': -.10, 'Innovation_3x': -.15},
                 'mild_growth_shock': {'Growth_2x': -.02, 'Innovation_3x': -.03}}
    if scenario not in scenarios:
        raise ValueError('Unknown scenario')
    paths = {}
    for name, path in {'flat': [0., 0.], 'volatile': [.1, -1/11]}.items():
        paths[name] = {'underlying_total_return': float(np.prod(1 + np.array(path)) - 1),
                       'gross_model': simulate_daily_reset(path, annual_fee=0, annual_financing=0),
                       'net_model': simulate_daily_reset(path)}
    return clean_json({'schema_version': 1, 'input_type': 'synthetic', 'seed': seed,
                       'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                       'fixture': fixture, 'portfolio': reports, 'path_dependence': paths,
                       'missing_history': {'common_start': trimmed, 'strict': strict,
                                           'assumption_changed': 'B first price removed; weights unchanged'},
                       'workbench': {'input_type': 'synthetic illustrative draw, not a calibrated forecast',
                                     'generator': 'numpy default_rng, normal(mean=.0003, std=.012), 60 observations',
                                     'weights': weights, 'summary': summary, 'returns': returns.tolist(),
                                     'assumptions_changed_from_default': {'weight_a': weight_a - .5,
                                                                        'scenario': scenario, 'mode': mode},
                                     'factor_map': exposures,
                                     'factor_exposure_per_initial_capital': look_through_exposure(weights, exposures),
                                     'rules': evaluate_rules(summary, shares),
                                     'stress': run_stress_tests(weights, {scenario: scenarios[scenario]}).to_dict('records')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--weight-a', type=float, default=.5)
    parser.add_argument('--scenario', choices=['shared_growth_shock', 'mild_growth_shock'], default='shared_growth_shock')
    parser.add_argument('--mode', choices=['constant_weight', 'buy_and_hold'], default='constant_weight')
    args = parser.parse_args()
    try:
        report = build_report(args.seed, args.weight_a, args.scenario, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
