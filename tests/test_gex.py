import datetime as dt

import numpy as np
import pandas as pd
import pytest

from intraday import gex
from intraday.strategies.gex_filter import GammaRegimeFilter

NOW = pd.Timestamp("2026-11-19 09:00", tz="America/New_York")


def test_implied_vol_round_trip():
    for call in (True, False):
        for K in (90.0, 100.0, 110.0):
            p = float(gex.black_price(100.0, K, 0.1, 0.25, call))
            assert gex.implied_vol(p, 100.0, K, 0.1, call) == pytest.approx(0.25, abs=1e-4)
    assert np.isnan(gex.implied_vol(0.0, 100.0, 110.0, 0.1, True))         # below intrinsic/no value


def test_gamma_matches_finite_difference_of_delta():
    S, K, T, v = 100.0, 102.0, 0.05, 0.2

    def delta(s):
        st = v * np.sqrt(T)
        return gex._ncdf((np.log(s / K) + 0.5 * st ** 2) / st)
    h = 0.01
    fd = (delta(S + h) - delta(S - h)) / (2 * h)
    assert float(gex.bs_gamma(S, K, T, v)) == pytest.approx(float(fd), rel=1e-3)


def _chain(F=100.0, oi_call=1000.0, oi_put=1000.0, expiry=dt.date(2026, 11, 27), vol=0.2):
    T = gex.years_to_expiry(expiry, NOW, False)
    rows = []
    for K in np.arange(80, 121, 2.5):
        for typ, oi in (("call", oi_call), ("put", oi_put)):
            p = float(gex.black_price(F, K, T, vol, typ == "call"))
            rows.append({"expiry": expiry, "strike": K, "type": typ, "oi": oi, "bid": p * 0.99, "ask": p * 1.01,
                         "multiplier": 100.0, "am_settled": False, "oi_date": "2026-11-18"})
    return pd.DataFrame(rows)


def test_forward_from_parity_and_vol_recovery():
    tbl, spot = gex.chain_gamma(_chain(F=101.3), NOW)
    assert spot == pytest.approx(101.3, rel=1e-3)
    assert tbl.iv.median() == pytest.approx(0.2, abs=0.01)


def test_gex_sign_convention():
    t_calls, s = gex.chain_gamma(_chain(oi_call=1000, oi_put=0), NOW)
    t_puts, _ = gex.chain_gamma(_chain(oi_call=0, oi_put=1000), NOW)
    assert gex.gex_at(t_calls, s) > 0 > gex.gex_at(t_puts, s)
    # units: $ per 1% move = gamma * OI * 100 * S^2 * 0.01
    atm = t_calls[(t_calls.type == "call") & (t_calls.strike == 100.0)].iloc[0]
    one = float(gex.bs_gamma(atm.forward, 100.0, atm["T"], atm.iv)) * 1000 * 100 * atm.forward ** 2 * 0.01
    assert one > 0


def test_gamma_flip_between_put_and_call_walls():
    ch = _chain(oi_call=0, oi_put=0)
    ch.loc[(ch.type == "put") & (ch.strike == 95.0), "oi"] = 5000      # put wall below
    ch.loc[(ch.type == "call") & (ch.strike == 105.0), "oi"] = 5000    # call wall above
    tbl, s = gex.chain_gamma(ch, NOW)
    flip = gex.gamma_flip(tbl, s)
    assert flip is not None and 95.0 < flip < 105.0
    assert gex.gex_at(tbl, 96.0) < 0 < gex.gex_at(tbl, 104.0)


class _Inner:
    name = "inner"

    def __init__(self):
        self.prepared = False

    def prepare_session(self, session, md):
        self.prepared = True
        return ["SPY"]

    def on_bar(self, now, bars, ctx):
        return {"SPY": "target"}


class _MD:
    def __init__(self, g):
        self.g = g

    def gamma_exposure(self):
        return self.g


@pytest.mark.parametrize("g,active", [({"total_gex_usd_per_1pct": -5e9}, True),
                                      ({"total_gex_usd_per_1pct": 3e9}, False),
                                      (None, False)])
def test_gamma_regime_filter(g, active):
    f = GammaRegimeFilter(_Inner())
    syms = f.prepare_session(None, _MD(g))
    assert f.active is active
    assert syms == (["SPY"] if active else [])
    assert (f.on_bar(None, {}, None) is not None) is active
