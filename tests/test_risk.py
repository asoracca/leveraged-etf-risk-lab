import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.risk import (compute_summary, normalize_weights, max_drawdown, beta_to_benchmark,
                      risk_contribution, component_risk, portfolio_returns, simple_returns)

FIXTURE = Path(__file__).parent / 'fixtures' / 'portfolio.json'


def fixture():
    f = json.loads(FIXTURE.read_text())
    return f, pd.DataFrame(f['prices'], index=pd.to_datetime(f['dates']))


class RiskTests(unittest.TestCase):
    def test_hand_calculated_fixture(self):
        f, p = fixture()
        s, rc, r = compute_summary(p, f['weights'])
        e = f['expected']
        np.testing.assert_allclose(r, e['returns'])
        for key in ('total_return', 'max_drawdown'):
            self.assertAlmostEqual(s[key], e[key])
        self.assertAlmostEqual(s['betas']['SPY'], e['beta'])
        self.assertAlmostEqual(s['annual_vol'] ** 2, e['annual_variance'])
        self.assertAlmostEqual(sum(s['component_volatility'].values()), s['annual_vol'])
        for t, expected in e['risk_shares'].items():
            self.assertAlmostEqual(rc[t], expected)
        bh, _, _ = compute_summary(p, f['weights'], mode='buy_and_hold')
        self.assertAlmostEqual(bh['total_return'], e['buy_and_hold_total_return'])
        self.assertAlmostEqual(sum(bh['component_volatility'].values()), bh['annual_vol'])

    def test_log_returns_are_not_simple(self):
        f, p = fixture()
        _, _, actual = compute_summary(p, f['weights'])
        wrong = np.log(p[['A', 'B']] / p[['A', 'B']].shift()).iloc[1:].mean(axis=1)
        self.assertFalse(np.allclose(actual, wrong))

    def test_initial_loss_counts_in_drawdown(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-.2, .1])), -.2)

    def test_invalid_weights(self):
        for weights in ({}, {'A': -1, 'B': 2}, {'A': np.nan}, {'A': np.inf}, {'A': 0}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                normalize_weights(weights)
        with self.assertRaisesRegex(ValueError, 'Missing symbols'):
            normalize_weights({'MISSING': 1}, ['A'])

    def test_zero_variance_and_flat_benchmark(self):
        p = pd.DataFrame({'A': [100.] * 4, 'SPY': [100.] * 4})
        s, rc, _ = compute_summary(p, {'A': 1})
        self.assertEqual(s['annual_vol'], 0)
        self.assertEqual(s['component_volatility']['A'], 0)
        self.assertTrue(np.isnan(s['sharpe']) and np.isnan(rc['A']))
        self.assertTrue(np.isnan(s['betas']['SPY']))

    def test_identical_assets(self):
        r = pd.DataFrame({'A': [.1, -.1, .1], 'B': [.1, -.1, .1]})
        w = {'A': .25, 'B': .75}
        np.testing.assert_allclose(risk_contribution(r, w).reindex(['A', 'B']), [.25, .75])
        self.assertAlmostEqual(component_risk(r, w).sum(), portfolio_returns(r, w).std() * np.sqrt(252))

    def test_missing_history_policies(self):
        f, p = fixture()
        p.loc[p.index[0], 'B'] = np.nan
        s, _, _ = compute_summary(p, f['weights'])
        self.assertEqual(s['history']['discarded_leading_prices'], 1)
        self.assertAlmostEqual(s['total_return'], -.0025)
        with self.assertRaisesRegex(ValueError, 'Strict'):
            compute_summary(p, f['weights'], missing_policy='strict')
        p.loc[p.index[2], 'B'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Internal'):
            compute_summary(p, f['weights'])

    def test_insufficient_and_invalid_prices(self):
        _, p = fixture()
        for bad in (p.iloc[:2], p.iloc[::-1], p.rename(columns={'B': 'A'}), p * 0, p * np.inf):
            with self.subTest(), self.assertRaises(ValueError):
                compute_summary(bad, {'A': 1})
        with self.assertRaises(ValueError):
            compute_summary(p, {'C': 1})
        self.assertTrue(np.isnan(beta_to_benchmark(pd.Series([.1]), pd.Series([.2]))))

    def test_rf_and_reordered_columns(self):
        f, p = fixture()
        s, _, r = compute_summary(p[['SPY', 'B', 'A']], f['weights'], risk_free_rate=.05)
        expected = (r.mean() - (1.05 ** (1 / 252) - 1)) * 252 / s['annual_vol']
        self.assertAlmostEqual(s['sharpe'], expected)
        self.assertTrue(simple_returns(p.assign(B=[100, np.nan, 100, 100]))['B'].iloc[:2].isna().all())


if __name__ == '__main__':
    unittest.main()
