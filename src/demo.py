"""Credential-free synthetic workbench: python -m src.demo."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from src.risk import compute_summary


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


def build_report(seed=42):
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/portfolio.json'
    fixture = json.loads(path.read_text())
    prices = pd.DataFrame(fixture['prices'], index=pd.to_datetime(fixture['dates']))
    reports = {}
    for mode in ('constant_weight', 'buy_and_hold'):
        summary, shares, returns = compute_summary(prices, fixture['weights'], mode=mode)
        reports[mode] = {**summary, 'risk_shares': shares.to_dict(), 'returns': returns.tolist()}
    return clean_json({'schema_version': 1, 'input_type': 'synthetic', 'seed': seed,
                       'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                       'fixture': fixture, 'portfolio': reports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = build_report(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
