"""Optional two-state HMM: inferred volatility states, never future predictions.

Fit on a declared training window. Evaluation uses frozen parameters and only
prefixes ending at each evaluation date. No trading backtest is implied.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES = ['log_ret', 'rv5']


@dataclass
class RegimeFit:
    model: object
    center: np.ndarray
    scale: np.ndarray
    training_features: np.ndarray
    training_end: object
    order: np.ndarray


def make_features(prices):
    from src.risk import simple_returns
    r = simple_returns(prices.to_frame('price'))['price']
    logs = np.log1p(r)
    return pd.DataFrame({'log_ret': logs, 'rv5': logs.rolling(5).std() * np.sqrt(252)}).dropna()


def _features(data, minimum):
    if (len(data) < minimum or not data.index.is_unique or not data.index.is_monotonic_increasing
            or not set(FEATURES).issubset(data.columns)):
        raise ValueError(f'Need {minimum} feature rows with unique increasing dates')
    values = data[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 1] < 0).any():
        raise ValueError('Features must be finite with nonnegative volatility')
    return values


def fit_regime_model(data, n_states=2, n_iter=200, seed=42):
    """Return fitted model wrapper and retrospective training labels/probabilities.

    Training probabilities are smoothed in-sample. They are not a valid trading
    signal. State labels are ordered by fitted mean volatility, not return sign.
    """
    if n_states != 2:
        raise ValueError('Only two volatility states are supported')
    x = _features(data, 30)
    from hmmlearn.hmm import GaussianHMM
    center, scale = x.mean(axis=0), x.std(axis=0)
    if (scale <= 0).any():
        raise ValueError('Training features must have nonzero variance')
    z = (x - center) / scale
    model = GaussianHMM(n_components=2, covariance_type='full', n_iter=n_iter, random_state=seed)
    model.fit(z)
    if not np.isfinite(model.means_).all() or not np.isfinite(model.transmat_).all():
        raise ValueError('HMM fit produced nonfinite parameters')
    order = np.argsort(model.means_[:, 1], kind='stable')
    fitted = RegimeFit(model, center, scale, z, data.index[-1], order)
    probs = pd.DataFrame(model.predict_proba(z)[:, order], index=data.index,
                         columns=['p_low_vol', 'p_high_vol'])
    states = probs.idxmax(axis=1).map({'p_low_vol': 0, 'p_high_vol': 1}).rename('inferred_regime')
    probs.attrs['interpretation'] = 'retrospective smoothed training inference; not a forecast'
    return fitted, states, probs


def evaluate_regimes(fitted, data):
    """Frozen fit, causal filtered endpoint at each date; no evaluation refitting."""
    x = _features(data, 1)
    if data.index[0] <= fitted.training_end:
        raise ValueError('Evaluation must start strictly after the training window')
    z = (x - fitted.center) / fitted.scale
    probabilities = []
    # Prefix evaluation prevents later observations from smoothing earlier labels.
    # Deliberately simple O(n²) reference implementation for small research windows.
    for end in range(1, len(z) + 1):
        history = np.vstack([fitted.training_features, z[:end]])
        probabilities.append(fitted.model.predict_proba(history)[-1, fitted.order])
    probs = pd.DataFrame(probabilities, index=data.index, columns=['p_low_vol', 'p_high_vol'])
    probs.attrs['interpretation'] = 'filtered observed-date inference; not a forecast'
    states = probs.idxmax(axis=1).map({'p_low_vol': 0, 'p_high_vol': 1}).rename('inferred_regime')
    return states, probs


def get_current_regime(model, states, probs, data=None):
    """Describe the last observation without an action recommendation."""
    return {'as_of': str(states.index[-1]),
            'inferred_regime': 'low_vol' if states.iloc[-1] == 0 else 'high_vol',
            'p_high_vol': float(probs['p_high_vol'].iloc[-1]),
            'interpretation': probs.attrs.get('interpretation', 'inferred state, not a forecast'),
            'fit_converged': bool(model.model.monitor_.converged)}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prices_csv', help='Local date,price CSV; no automatic downloads')
    parser.add_argument('--train-end', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prices = pd.read_csv(args.prices_csv, index_col='date', parse_dates=True)['price']
    features = make_features(prices)
    cutoff = pd.Timestamp(args.train_end)
    fit, _, _ = fit_regime_model(features.loc[features.index <= cutoff], seed=args.seed)
    states, probs = evaluate_regimes(fit, features.loc[features.index > cutoff])
    print(json.dumps(get_current_regime(fit, states, probs), indent=2))


if __name__ == '__main__':
    main()
