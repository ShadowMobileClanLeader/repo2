import numpy as np
import pandas as pd
import pytest

from intraday.backtest import HistoricalMarketData, LookAheadError
from intraday.engine import Context, Portfolio
from intraday.orb_data import compute_relvol, select_stocks_in_play
from intraday.strategies.orb import ORBStocksInPlay

from helpers import SESSION, bar


def test_relvol_uses_previous_14_days_only():
    vol = pd.DataFrame({"A": [100.0] * 14 + [500.0, 50.0], "B": [np.nan] + [10.0] * 15})
    avg, rel = compute_relvol(vol)
    assert rel["A"].iloc[14] == pytest.approx(5.0)          # today's 500 vs prior mean 100
    assert rel["A"].iloc[15] == pytest.approx(50 / ((13 * 100 + 500) / 14))
    assert np.isnan(rel["B"].iloc[14])                       # a missing day in the window -> unrankable
    assert rel["B"].iloc[15] == pytest.approx(1.0)


def _snap():
    return pd.DataFrame({
        "symbol": ["UP", "DN", "DOJI", "LOWRV", "CHEAP", "UP2"],
        "open":   [10.0, 50.0, 20.0, 30.0, 4.0, 40.0],
        "high":   [10.6, 50.2, 20.5, 31.0, 4.5, 41.0],
        "low":    [9.9, 48.0, 19.5, 29.5, 3.9, 39.8],
        "close":  [10.5, 48.5, 20.0, 30.8, 4.4, 40.9],
        "relvol": [8.0, 6.0, 9.0, 0.9, 20.0, 3.0],
        "atr14":  [0.8, 2.0, 1.0, 1.0, 0.6, 1.5],
    })


def test_selection_filters_and_ranks():
    sel = select_stocks_in_play(_snap(), top_n=2)
    assert list(sel["symbol"]) == ["UP", "DN"]                # DOJI, LOWRV (<100%), CHEAP (<$5) excluded
    assert list(sel["direction"]) == [1, -1]


class _MD:
    def __init__(self, session):
        self.session = session
        self.asked = []

    def orb_symbols_to_load(self, feed, top_n, **kw):
        return ["UP", "DN"]

    def orb_snapshot(self, feed, now, delay):
        self.asked.append(now)
        return _snap()


def _ctx():
    return Context(SESSION, Portfolio(cash=100_000), 100_000, [])


def test_orb_targets_sizing_and_timing():
    s = ORBStocksInPlay(top_n=20)
    md = _MD(SESSION)
    s.prepare_session(SESSION, md)
    b = {"UP": bar("UP", SESSION, 3, 10.5)}
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=4), b, _ctx()) is None    # before 09:35
    t = s.on_bar(SESSION.open + pd.Timedelta(minutes=5), b, _ctx())
    cap = 100_000 / 20
    up = t["UP"]
    shares = min(0.01 * cap / (0.1 * 0.8), 4 * cap / 10.6)                          # = 625 vs 1886.8
    assert up.entry_stop == 10.6 and up.stop_offset == pytest.approx(0.08)
    assert up.weight == pytest.approx(shares * 10.6 / 100_000)
    dn = t["DN"]
    shares_dn = min(0.01 * cap / 0.2, 4 * cap / 48.0)                                # leverage cap binds
    assert dn.entry_stop == 48.0 and dn.weight == pytest.approx(-shares_dn * 48.0 / 100_000)
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=6), b, _ctx()) is None      # decides once
    assert sum(abs(x.weight) for x in t.values()) <= 4.0 * 2 / 20 + 1e-12


def test_delayed_variant_waits_until_0950():
    s = ORBStocksInPlay(delay_minutes=15)
    md = _MD(SESSION)
    s.prepare_session(SESSION, md)
    b = {"UP": bar("UP", SESSION, 10, 10.5)}
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=19), b, _ctx()) is None
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=20), b, _ctx()) is not None
    assert md.asked == [SESSION.open + pd.Timedelta(minutes=20)]


def test_snapshot_guard_blocks_early_access():
    md = HistoricalMarketData(features=pd.DataFrame({"date": [], "symbol": []}))
    md.set_session(SESSION, None)
    with pytest.raises(LookAheadError):
        md.orb_snapshot("sip", SESSION.open + pd.Timedelta(minutes=4))
    with pytest.raises(LookAheadError):
        md.orb_snapshot("sip", SESSION.open + pd.Timedelta(minutes=19), delay_minutes=15)
