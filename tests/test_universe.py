import datetime as dt

import numpy as np
import pandas as pd
import pytest

from intraday import universe


def _daily(symbol, n, price=10.0, vol=2_000_000, start="2020-01-01", split_at=None):
    dates = pd.bdate_range(start, periods=n).date
    rng = np.random.default_rng(0)
    close = price * np.cumprod(1 + rng.normal(0, 0.01, n))
    raw = pd.DataFrame({"symbol": symbol, "date": dates, "open": close, "high": close * 1.01,
                        "low": close * 0.99, "close": close, "volume": float(vol)})
    adj = raw.copy()
    if split_at is not None:   # 2:1 split on day split_at: raw price halves, adjusted series is smooth
        raw.loc[split_at:, ["open", "high", "low", "close"]] /= 2
        raw.loc[split_at:, "volume"] *= 2
        adj[["open", "high", "low", "close"]] /= 2
        adj["volume"] *= 2
    return raw, adj


def test_features_are_point_in_time():
    raw, adj = _daily("AAA", 80)
    f1 = universe.compute_daily_features(raw, adj)
    raw2, adj2 = raw.copy(), adj.copy()
    raw2.loc[60:, "close"] *= 3          # change the "future"
    adj2.loc[60:, "close"] *= 3
    f2 = universe.compute_daily_features(raw2, adj2)
    cols = ["close", "adv14", "atr14", "dollar_vol20", "ret1"]
    pd.testing.assert_frame_equal(f1.loc[:59, cols], f2.loc[:59, cols])


def test_atr_is_in_raw_dollars_across_split():
    raw, adj = _daily("AAA", 60, price=100, split_at=40)
    f = universe.compute_daily_features(raw, adj).set_index("date")
    d_before, d_after = f.index[35], f.index[55]
    # before the split the raw price is ~2x the adjusted price, so raw ATR ~ 2x adjusted ATR
    ratio = f.loc[d_before, "atr14"] / f.loc[d_after, "atr14"]
    assert 1.6 < ratio < 2.5


def test_orb_eligible_filters():
    fp = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "close": [10, 4, 10, 10],
                       "adv14": [2e6, 2e6, 5e5, 2e6], "atr14": [1.0, 1.0, 1.0, 0.3]})
    assert universe.orb_eligible(fp) == ["A"]


def test_liquid_top_n_requires_history_and_price():
    fp = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "close": [10, 10, 3, 50],
                       "dollar_vol20": [5e7, 9e7, 9e9, 1e7], "n_hist": [100, 30, 100, 100]})
    assert universe.liquid_top_n(fp, 2) == ["A", "D"]


@pytest.mark.parametrize("sym,name,ok", [
    ("AAPL", "Apple Inc. Common Stock", True),
    ("BRK.B", "Berkshire Hathaway", True),
    ("SPY", "SPDR S&P 500 ETF Trust", False),
    ("TQQQ", "ProShares UltraPro QQQ", False),
    ("ABCDW", "Acme Acquisition Corp Warrants", False),
    ("VFIAX", None, False),
    ("451CVR013", None, False),
    ("TWTR", None, True),
])
def test_common_stock_filter(sym, name, ok):
    assert universe.is_common_stock_like(sym, name) is ok
