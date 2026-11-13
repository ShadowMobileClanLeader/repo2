import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from intraday import metrics


def _r(mu, sd=0.01, n=2000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n))


def test_psr_matches_normal_formula():
    r = _r(0.0005)
    sr = r.mean() / r.std(ddof=1)
    assert metrics.probabilistic_sharpe(r) == pytest.approx(norm.cdf(sr * np.sqrt(len(r) - 1)), abs=0.02)


def test_expected_max_sharpe_grows_with_trials_and_dispersion():
    a = metrics.expected_max_sharpe(10, 0.001)
    b = metrics.expected_max_sharpe(100, 0.001)
    c = metrics.expected_max_sharpe(100, 0.004)
    assert 0 < a < b < c
    assert metrics.expected_max_sharpe(1, 0.001) == 0.0


def test_dsr_falls_as_trials_increase():
    r = _r(0.0006)
    d1 = metrics.deflated_sharpe(r, 1, 0.0004)["dsr"]
    d50 = metrics.deflated_sharpe(r, 50, 0.0004)["dsr"]
    d500 = metrics.deflated_sharpe(r, 500, 0.0004)["dsr"]
    assert d1 > d50 > d500


def test_best_of_many_noise_strategies_is_not_significant():
    rng = np.random.default_rng(3)
    trials = [pd.Series(rng.normal(0, 0.01, 1000)) for _ in range(200)]
    srs = np.array([t.mean() / t.std() for t in trials])
    best = trials[int(srs.argmax())]
    assert metrics.probabilistic_sharpe(best) > 0.95            # naive test is fooled
    assert metrics.deflated_sharpe(best, 200, srs.var())["dsr"] < 0.95   # deflated test is not
