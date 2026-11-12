import numpy as np
import pandas as pd
import pytest

from intraday.engine import Context, Portfolio, Target
from intraday.strategies.common import decile_weights
from intraday.strategies.vwap_filter import VWAPFilter
from intraday.strategies.xs import CrossSectional, daily_scores

from helpers import SESSION, bar


def _hist(n_syms=20, n_days=25, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_syms):
        out[f"S{i:02d}"] = pd.DataFrame({"close": 50.0, "ret1": rng.normal(0, 0.01, n_days),
                                         "ret_intraday": rng.normal(0, 0.01, n_days),
                                         "ret_overnight": rng.normal(0, 0.01, n_days)})
    return out


def test_decile_weights_dollar_neutral():
    score = pd.Series(np.arange(20, dtype=float), index=[f"S{i}" for i in range(20)])
    w = decile_weights(score, 0.1, 1.0)
    assert w == {"S0": -0.25, "S1": -0.25, "S18": 0.25, "S19": 0.25}
    assert sum(w.values()) == pytest.approx(0) and sum(map(abs, w.values())) == pytest.approx(1.0)


def test_signal_definitions():
    h = _hist()
    r1 = daily_scores(h, "rev1")
    assert r1["S03"] == pytest.approx(-h["S03"]["ret1"].iloc[-1])
    r5 = daily_scores(h, "rev5")
    assert r5["S03"] == pytest.approx(-(np.prod(1 + h["S03"]["ret1"].iloc[-5:]) - 1))
    li = daily_scores(h, "lps_intraday", 21)
    assert li["S03"] == pytest.approx(np.prod(1 + h["S03"]["ret_intraday"].iloc[-21:]) - 1)
    lo = daily_scores(h, "lps_overnight", 21)
    assert lo["S03"] == pytest.approx(-(np.prod(1 + h["S03"]["ret_overnight"].iloc[-21:]) - 1))
    combo = daily_scores(h, "lps_combo", 21)
    assert combo.corr(li) > 0 and combo.corr(lo) > 0


def test_signal_requires_full_lookback():
    h = _hist(n_days=10)
    assert daily_scores(h, "lps_intraday", 21).empty


class _MD:
    def __init__(self, hist):
        self.hist = hist

    def xs_universe(self):
        return list(self.hist)

    def daily_history(self, s, n):
        return self.hist[s].iloc[-n:]


def _ctx(pos=None):
    return Context(SESSION, Portfolio(cash=1e6, positions=pos or {}), 1e6, [])


def test_xs_trades_once_at_0935_and_skips_bad_ticks():
    h = _hist()
    s = CrossSectional(signal="rev1")
    s.prepare_session(SESSION, _MD(h))
    bars = {k: bar(k, SESSION, 0, 50.0) for k in h}
    worst = daily_scores(h, "rev1").idxmax()               # would be a long
    bars[worst] = bar(worst, SESSION, 0, 80.0)              # +60% vs prior close -> dropped
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=4), bars, _ctx()) is None
    t = s.on_bar(SESSION.open + pd.Timedelta(minutes=5), bars, _ctx())
    assert worst not in t and len(t) == 2 * int(19 * 0.1)
    assert sum(x.weight for x in t.values()) == pytest.approx(0)
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=10), bars, _ctx()) is None


def test_eod_reversal_timing_and_direction():
    h = _hist()
    s = CrossSectional(signal="eod_reversal")
    s.prepare_session(SESSION, _MD(h))
    px = {k: 50.0 * (1 + 0.01 * i) for i, k in enumerate(h)}   # S19 biggest winner by 15:00
    bars = {k: bar(k, SESSION, 325, v) for k, v in px.items()}
    assert s.on_bar(SESSION.open + pd.Timedelta(minutes=330), bars, _ctx()) is None
    t = s.on_bar(SESSION.open + pd.Timedelta(minutes=360), bars, _ctx())
    assert t["S19"].weight < 0 and t["S00"].weight > 0


class _Inner:
    name = "inner"

    def __init__(self, targets):
        self.targets = targets

    def prepare_session(self, session, md):
        return ["A"]

    def on_bar(self, now, bars, ctx):
        return self.targets


def test_vwap_filter_blocks_entries_on_wrong_side_only():
    f = VWAPFilter(_Inner({"A": Target(1.0), "B": Target(-1.0)}))
    f.prepare_session(SESSION, None)
    # first bar: price == VWAP for both, which is not a confirmation -> both blocked
    assert f.on_bar(SESSION.open, {"A": bar("A", SESSION, 0, 100, c=100), "B": bar("B", SESSION, 0, 100, c=100)}, _ctx()) is None
    f.blocked = 0
    # A falls below its VWAP (long blocked); B falls below its VWAP (short allowed)
    out = f.on_bar(SESSION.open, {"A": bar("A", SESSION, 1, 99, c=99), "B": bar("B", SESSION, 1, 99, c=99)}, _ctx())
    assert set(out) == {"B"} and f.blocked == 1
    # an exit always passes, even on the "wrong" side
    f.inner.targets = {"A": Target(0.0)}
    assert set(f.on_bar(SESSION.open, {"A": bar("A", SESSION, 2, 98, c=98)}, _ctx({"A": 10}))) == {"A"}
