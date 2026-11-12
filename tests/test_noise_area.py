import datetime as dt

import numpy as np
import pandas as pd
import pytest

from intraday.calendar import make_session
from intraday.engine import Context, Portfolio
from intraday.strategies.noise_area import NoiseAreaMomentum

from helpers import FakeMD, SESSION, bar


def _history_day(d, move=0.01):
    """A prior session where every minute closes exactly `move` above the open."""
    s = make_session(d)
    ts = [(s.open + pd.Timedelta(minutes=m)).tz_convert("UTC") for m in range(390)]
    return pd.DataFrame({"symbol": "SPY", "ts": ts, "open": 100.0, "close": 100.0 * (1 + move)})


@pytest.fixture
def strat():
    days = pd.bdate_range("2024-02-12", "2024-03-01").date[-14:]
    daily_ret = np.tile([0.01, -0.01], 8)[:15]                 # daily sd ~ 1.035%
    daily = pd.DataFrame({"symbol": "SPY", "date": pd.bdate_range("2024-02-09", "2024-03-01").date[-15:],
                          "close": 99.0, "ret1": daily_ret})
    md = FakeMD(daily, {d: _history_day(d) for d in days}, SESSION)
    s = NoiseAreaMomentum()
    s.prepare_session(SESSION, md)
    return s


def ctx(pos=0):
    pf = Portfolio(cash=100_000, positions={"SPY": pos} if pos else {}, last_price={"SPY": 100.0})
    return Context(SESSION, pf, 100_000, [])


def feed(s, closes_by_minute, vol=1000.0, pos=0):
    """Feed bars for minutes 0..max; returns the output at the last bar."""
    out = None
    last = max(closes_by_minute)
    for m in range(last + 1):
        c = closes_by_minute.get(m, 100.0)
        o = 100.0 if m == 0 else c
        b = bar("SPY", SESSION, m, o, max(o, c), min(o, c), c, vol)
        b.vwap = c
        out = s.on_bar(SESSION.open + pd.Timedelta(minutes=m + 1), {"SPY": b}, ctx(pos))
    return out


def test_sigma_and_bands(strat):
    assert strat.sigma.loc[30] == pytest.approx(0.01)
    strat.open_px = 100.0
    hi, lo = strat.bands(30)
    assert hi == pytest.approx(101.0)             # max(open 100, prev close 99) * 1.01
    assert lo == pytest.approx(99.0 * 0.99)       # min(...) * 0.99


def test_leverage_is_vol_targeted_and_capped(strat):
    sd = np.std(np.tile([0.01, -0.01], 8)[:15][-14:], ddof=1)
    assert strat.leverage == pytest.approx(min(4.0, 0.02 / sd))


def test_only_decides_on_30_minute_grid(strat):
    # far above the band from minute 10 on; first decision is at the 10:00 bar close (minute 29 bar)
    closes = {m: 102.0 for m in range(10, 29)}
    assert feed(strat, closes) is None
    t = feed(strat, {29: 102.0})
    assert t["SPY"].weight == pytest.approx(strat.leverage)


def test_vwap_filter_blocks_entry(strat):
    closes = {m: 110.0 for m in range(1, 29)}     # VWAP pulled far above
    closes[29] = 101.5                             # above UB (101) but below VWAP
    assert feed(strat, closes) is None


def test_short_entry_and_trailing_exit(strat):
    out = feed(strat, {29: 97.5})
    assert out["SPY"].weight == pytest.approx(-strat.leverage)


def test_long_trailing_stop_exit(strat):
    closes = {29: 102.0}
    feed(strat, closes)
    strat.vwap.pv, strat.vwap.v = 0.0, 0.0
    # next decision at minute 59: price back inside the band -> exit long
    out = None
    for m in range(30, 60):
        b = bar("SPY", SESSION, m, 100.5, 100.5, 100.5, 100.5)
        b.vwap = 100.5
        out = strat.on_bar(SESSION.open + pd.Timedelta(minutes=m + 1), {"SPY": b}, ctx(pos=100))
    assert out["SPY"].weight == 0.0


def test_prev_close_is_dividend_adjusted():
    days = pd.bdate_range("2024-02-12", "2024-03-01").date[-14:]
    daily = pd.DataFrame({"symbol": "SPY", "date": pd.bdate_range("2024-02-09", "2024-03-01").date[-15:],
                          "close": 99.0, "ret1": np.tile([0.01, -0.01], 8)[:15]})
    md = FakeMD(daily, {d: _history_day(d) for d in days}, SESSION)
    md.dividend = 1.0
    s = NoiseAreaMomentum()
    s.prepare_session(SESSION, md)
    assert s.prev_close == pytest.approx(98.0)
