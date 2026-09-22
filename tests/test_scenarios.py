import unittest
import numpy as np
from src.optimal_leverage import simulate_daily_reset, lev_cagr
from src.rebalance import evaluate_rules
from src.stress import run_stress_tests


class ScenarioTests(unittest.TestCase):
    def test_equal_terminal_different_path(self):
        flat, volatile = [0, 0], [.1, -1/11]
        self.assertAlmostEqual(np.prod(1 + np.array(volatile)), 1)
        a = simulate_daily_reset(flat, annual_fee=0, annual_financing=0)
        b = simulate_daily_reset(volatile, annual_fee=0, annual_financing=0)
        self.assertAlmostEqual(a['total_return'], 0)
        self.assertAlmostEqual(b['total_return'], -3/55)

    def test_costs_and_existing_curve_parity(self):
        result = simulate_daily_reset([0, 0], leverage=3, annual_fee=.01, annual_financing=.04)
        self.assertAlmostEqual(result['daily_cost'], .09 / 252)
        self.assertAlmostEqual(result['wealth'][-1], (1 - .09 / 252) ** 2)
        self.assertAlmostEqual(lev_cagr([0, 0], 3, .01, .04), result['wealth'][-1] ** 126 - 1)

    def test_invalid_returns_and_liquidation(self):
        for r in ([], [np.nan], [np.inf], [-1.1], [[.1]]):
            with self.subTest(r=r), self.assertRaises(ValueError):
                simulate_daily_reset(r)
        with self.assertRaises(ValueError):
            simulate_daily_reset([-.5, 1])
        result = simulate_daily_reset([-.5, 1], invalid_policy='liquidate')
        self.assertEqual(result['wealth'], [1, 0, 0])
        self.assertEqual(result['exhausted_at'], 0)
        for kwargs in ({'leverage': -1}, {'annual_fee': -1}, {'periods': 0}, {'invalid_policy': 'ignore'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_daily_reset([.1], **kwargs)

    def test_rules_boundary_and_unavailable(self):
        summary = {'max_drawdown': -.2, 'annual_vol': .75, 'betas': {'SPY': np.nan}}
        rows = evaluate_rules(summary, {'A': .35})
        self.assertTrue(rows[0]['triggered'] and rows[1]['triggered'] and rows[-1]['triggered'])
        self.assertIsNone(rows[2]['triggered'])
        self.assertEqual(rows[3]['status'], 'unavailable')
        self.assertEqual(rows[-1]['trigger'], 'risk_share_A >= 0.35')

    def test_stress_components_and_missing_shocks(self):
        row = run_stress_tests({'A': .5, 'B': .5}, {'test': {'A': -.1, 'B': -.2}}).iloc[0]
        self.assertAlmostEqual(row['portfolio_return'], -.15)
        self.assertAlmostEqual(sum(row['contributions'].values()), row['portfolio_return'])
        for shocks in ({'A': -.1}, {'A': np.nan, 'B': 0}, {'A': -2, 'B': 0}):
            with self.assertRaises(ValueError):
                run_stress_tests({'A': .5, 'B': .5}, {'bad': shocks})
