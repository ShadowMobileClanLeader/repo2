import numpy as np
import pandas as pd
import pytest

from intraday.portfolio import PortfolioConfig, contributions, inverse_vol_weights, run_portfolio, strategy_weights


def _R(n=400, vols=(0.01, 0.02), rho=0.0, seed=1, mu=0.0005):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, len(vols)))
    if rho and len(vols) >= 2:
        z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
    idx = pd.bdate_range("2020-01-01", periods=n).date
    return pd.DataFrame(mu + z * np.array(vols), index=idx, columns=[f"s{i}" for i in range(len(vols))])


def test_inverse_vol_without_correlation():
    w = inverse_vol_weights(_R(2000, rho=0.0), PortfolioConfig(corr_penalty=False))
    assert w["s0"] == pytest.approx(2 / 3, abs=0.03) and w.sum() == pytest.approx(1.0)


def test_correlation_penalty_downweights_correlated_pair():
    R = _R(2000, vols=(0.01, 0.01, 0.01), rho=0.9)
    w = inverse_vol_weights(R, PortfolioConfig())
    assert w["s2"] > w["s0"] and w["s2"] > w["s1"]
    assert w["s2"] == pytest.approx(0.48, abs=0.05)            # pair penalised by 1 + 0.9


def test_weights_use_only_past_data():
    R = _R(300)
    W1 = strategy_weights(R, PortfolioConfig())
    R2 = R.copy()
    R2.iloc[200:] *= 10                                        # change the future
    W2 = strategy_weights(R2, PortfolioConfig())
    cut = R.index[200]
    pd.testing.assert_frame_equal(W1[W1.index <= cut], W2[W2.index <= cut])


def test_sleeve_without_history_gets_zero_weight():
    R = _R(200)
    R.iloc[:150, 1] = np.nan
    W = strategy_weights(R, PortfolioConfig(min_history=60))
    assert (W.loc[R.index[100], "s1"] == 0) and W.loc[R.index[100], "s0"] == pytest.approx(1.0)


def test_vol_target_and_gross_cap():
    R = _R(600, vols=(0.002, 0.002))
    cfg = PortfolioConfig(target_vol=0.10, max_scale=10, max_gross=4.0, sleeve_max_gross={"s0": 4.0, "s1": 4.0},
                          max_stock_weight=10)
    res = run_portfolio(R, cfg)
    assert res.scale.max() <= 1.0 + 1e-9                       # gross cap: sum(a_i * 4) <= 4
    cfg2 = PortfolioConfig(target_vol=0.10, max_scale=10, max_gross=100, max_stock_weight=100)
    res2 = run_portfolio(R, cfg2)
    realised = res2.returns.iloc[200:].std() * np.sqrt(252)
    assert realised == pytest.approx(0.10, rel=0.25)


def test_per_stock_cap():
    R = _R(300)
    cfg = PortfolioConfig(max_scale=10, max_gross=100, max_stock_weight=0.10,
                          sleeve_max_stock_weight={"s0": 0.2, "s1": 0.2})
    res = run_portfolio(R, cfg)
    assert res.scale.max() <= 0.5 + 1e-9


def test_kill_switch_stops_trading_after_drawdown():
    R = _R(300, mu=0.0)
    R.iloc[150:170] = -0.03
    res = run_portfolio(R, PortfolioConfig(kill_drawdown=0.10, max_gross=100, max_stock_weight=100))
    assert res.killed_on is not None
    assert (res.returns[res.returns.index > res.killed_on] == 0).all()
    assert (res.allocations.loc[res.allocations.index > res.killed_on] == 0).all().all()


def test_contributions_table():
    R = _R(500, vols=(0.01, 0.01, 0.01))
    t = contributions(R, PortfolioConfig(max_gross=100, max_stock_weight=100))
    assert set(t.columns) >= {"standalone_sharpe", "marginal_sharpe", "risk_share"}
    assert t["risk_share"].sum() == pytest.approx(1.0, abs=0.05)
