import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.demo import build_report
from src.exposure import look_through_exposure
from src.holdings import load_weights


class IntegrationTests(unittest.TestCase):
    def test_seed_and_network_free_report(self):
        with patch('socket.socket', side_effect=AssertionError('Network forbidden')):
            a = build_report(42)
            b = build_report(42)
            c = build_report(43)
        self.assertEqual(a, b)
        self.assertNotEqual(a['workbench']['returns'], c['workbench']['returns'])
        json.dumps(a, allow_nan=False)
        self.assertEqual(a['workbench']['factor_exposure_per_initial_capital'], {'shared_growth': 2.5})
        self.assertEqual(a['missing_history']['strict']['status'], 'rejected')
        self.assertAlmostEqual(a['missing_history']['common_start']['total_return'], -.0025)

    def test_cli_and_changed_assumptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            process = subprocess.run([sys.executable, '-m', 'src.demo', '--seed', '42', '--weight-a', '.25',
                                      '--scenario', 'mild_growth_shock', '--mode', 'buy_and_hold',
                                      '--output', str(path)], check=True, text=True, capture_output=True)
            report = json.loads(path.read_text())
            self.assertEqual(report, json.loads(process.stdout))
            work = report['workbench']
            self.assertEqual(work['factor_exposure_per_initial_capital']['shared_growth'], 2.75)
            self.assertEqual(work['summary']['mode'], 'buy_and_hold')
            self.assertAlmostEqual(work['stress'][0]['portfolio_return'], -.0275)

    def test_exposure_and_holdings_validation(self):
        with self.assertRaises(ValueError):
            look_through_exposure({'A': 1}, {})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.csv'
            with patch('src.holdings.HOLDINGS_FILE', path):
                path.write_text('ticker,market_value\nA,100\nA,100\nB,200\n')
                self.assertEqual(load_weights(), {'A': .5, 'B': .5})
                for invalid in ('A,-10\nB,100', 'A,NaN', 'A,0', 'A,inf'):
                    path.write_text('ticker,market_value\n' + invalid)
                    with self.assertRaises(ValueError):
                        load_weights()

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') and importlib.util.find_spec('yfinance'), 'Optional rolling dependencies')
    def test_rolling_risk_uses_excess_downside_and_keeps_usd_etf(self):
        from src.rolling_sharpe import compute_rolling_sortino, load_portfolio
        r = pd.Series([-.1, .1, 0.])
        expected = -.01 / np.sqrt((.11 ** 2 + .01 ** 2) / 3) * np.sqrt(252)
        self.assertAlmostEqual(compute_rolling_sortino(r, window=3, rf_daily=.01).iloc[-1], expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.csv'
            path.write_text('ticker,market_value\nUSD,100\nQLD,100\n')
            self.assertEqual(set(load_portfolio(path)['ticker']), {'USD', 'QLD'})

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') and importlib.util.find_spec('seaborn'), 'Optional chart dependencies')
    def test_default_runner_offline_with_synthetic_inputs(self):
        import main
        from src.config import TICKERS, BENCHMARKS
        rng = np.random.default_rng(42)
        symbols = TICKERS + BENCHMARKS
        prices = pd.DataFrame(100 * np.exp(rng.normal(0, .01, (65, len(symbols))).cumsum(axis=0)),
                              columns=symbols, index=pd.bdate_range('2024-01-01', periods=65))
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with patch('main.fetch_price_history', return_value=prices), patch('main.load_weights', return_value={t: 1 for t in TICKERS}), patch('socket.socket', side_effect=AssertionError('Network forbidden')), patch('sys.stdout', new_callable=io.StringIO):
                    main.main()
                for name in ('equity_drawdown.png', 'risk_contribution.png', 'theme_exposure.png', 'correlation_heatmap.png', 'stress_tests.png', 'rule_evaluations.csv'):
                    self.assertGreater((Path('data') / name).stat().st_size, 0)
            finally:
                os.chdir(cwd)
