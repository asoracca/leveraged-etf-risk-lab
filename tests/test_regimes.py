import importlib.util
import unittest
import numpy as np
import pandas as pd
from src.regime_detection import make_features, fit_regime_model, evaluate_regimes, get_current_regime


@unittest.skipUnless(importlib.util.find_spec('hmmlearn'), 'Optional hmmlearn dependency not installed')
class RegimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        r = np.r_[rng.normal(.0002, .005, 80), rng.normal(-.0002, .03, 80)]
        prices = pd.Series(100 * np.exp(np.r_[0, r].cumsum()), index=pd.bdate_range('2024-01-01', periods=161))
        cls.features = make_features(prices)
        cls.fit, _, _ = fit_regime_model(cls.features.iloc[:130], seed=42)

    def test_future_changes_do_not_change_past_inferences(self):
        evaluation = self.features.iloc[130:].copy()
        before = self.fit.model.means_.copy()
        _, full = evaluate_regimes(self.fit, evaluation)
        _, prefix = evaluate_regimes(self.fit, evaluation.iloc[:10])
        np.testing.assert_allclose(full.iloc[:10], prefix)
        evaluation.iloc[10:, 0] += .5
        _, altered = evaluate_regimes(self.fit, evaluation)
        np.testing.assert_allclose(full.iloc[:10], altered.iloc[:10])
        np.testing.assert_array_equal(before, self.fit.model.means_)
        np.testing.assert_allclose(full.sum(axis=1), 1)

    def test_boundary_and_invalid_training(self):
        with self.assertRaisesRegex(ValueError, 'strictly after'):
            evaluate_regimes(self.fit, self.features.iloc[129:])
        for data in (self.features.iloc[:2], self.features.iloc[::-1], self.features * 0):
            with self.assertRaises(ValueError):
                fit_regime_model(data)
        with self.assertRaises(ValueError):
            fit_regime_model(self.features, n_states=3)

    def test_inferred_labels_and_training_only_scaling(self):
        train = self.features.iloc[:130]
        np.testing.assert_allclose(self.fit.center, train[['log_ret', 'rv5']].mean())
        states, probs = evaluate_regimes(self.fit, self.features.iloc[130:])
        report = get_current_regime(self.fit, states, probs)
        self.assertIn(report['inferred_regime'], ['low_vol', 'high_vol'])
        self.assertIn('not a forecast', report['interpretation'])
        self.assertNotIn('signal', report)
