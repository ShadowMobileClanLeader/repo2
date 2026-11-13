import itertools

import numpy as np
import pandas as pd
import pytest

from intraday.engine import Engine, Order, RiskLimits, Target
from intraday.live import AlpacaPaperBroker, LiveFeed, vol_target_scale
from intraday.portfolio import PortfolioConfig, run_portfolio

from helpers import SESSION, ScriptedStrategy, bar, flat_day, ts_at


class FakeClient:
    """Fills market orders at a fixed price on the next status poll; records everything."""

    def __init__(self, fill_price=100.0):
        self.orders, self.fill_price, self.polls = {}, fill_price, 0
        self._ids = itertools.count(1)
        self.deleted = []

    def trading_post(self, path, payload):
        assert path == "/v2/orders"
        oid = f"b{next(self._ids)}"
        self.orders[oid] = {"payload": payload, "status": "new", "filled_qty": "0"}
        return {"id": oid}

    def trading_get(self, path, **kw):
        oid = path.rsplit("/", 1)[-1]
        o = self.orders[oid]
        self.polls += 1
        if o["status"] == "new" and o["payload"]["type"] == "market":
            o.update(status="filled", filled_qty=o["payload"]["qty"], filled_avg_price=str(self.fill_price),
                     filled_at="2024-03-04T15:00:00Z")
        return {"id": oid, **{k: v for k, v in o.items() if k != "payload"}}

    def trading_delete(self, path):
        oid = path.rsplit("/", 1)[-1]
        self.deleted.append(oid)
        if self.orders[oid]["status"] == "new":
            self.orders[oid]["status"] = "canceled"


def test_broker_maps_orders_and_reports_fills_with_ref_price():
    c = FakeClient(fill_price=100.05)
    b = AlpacaPaperBroker(c, sleep=lambda s: None)
    b.process_bar(ts_at(SESSION, 0), {"SPY": bar("SPY", SESSION, 0, 100.0)})
    b.submit(Order("SPY", -10, "market", reason="test"))
    p = c.orders["b1"]["payload"]
    assert p["side"] == "sell" and p["qty"] == "10" and p["type"] == "market" and p["time_in_force"] == "day"
    f = b.process_bar(ts_at(SESSION, 1), {"SPY": bar("SPY", SESSION, 1, 100.0)})
    assert len(f) == 1 and f[0].qty == -10 and f[0].price == pytest.approx(100.05) and f[0].ref_price == 100.0
    assert f[0].fees > 0                              # SEC/TAF on a sell, no commission
    assert b.positions == {"SPY": -10} and b.open_orders() == []


def test_stop_order_payload_and_cancel():
    c = FakeClient()
    b = AlpacaPaperBroker(c, sleep=lambda s: None)
    b.submit(Order("SPY", 5, "stop", 101.234))
    assert c.orders["b1"]["payload"]["stop_price"] == "101.23" and c.orders["b1"]["payload"]["type"] == "stop"
    b.cancel(b.open_orders()[0].id)
    assert c.deleted == ["b1"]
    b.process_bar(ts_at(SESSION, 1), {})
    assert b.open_orders() == []


def test_end_session_only_closes_our_positions():
    c = FakeClient()
    b = AlpacaPaperBroker(c, sleep=lambda s: None)
    b.submit(Order("SPY", 7, "market"))
    b.process_bar(ts_at(SESSION, 1), {})
    fills = b.end_session({"SPY": 100.0}, ts_at(SESSION, 389))
    assert [f.qty for f in fills] == [-7] and b.positions == {}
    assert all(o["payload"]["symbol"] == "SPY" for o in c.orders.values())


def test_engine_runs_unchanged_on_live_broker():
    """The same Engine + strategy drive the live broker: flat by 15:54, fills reconciled."""
    c = FakeClient(fill_price=100.0)
    strat = ScriptedStrategy({1: {"AAA": Target(0.5)}})
    eng = Engine(strat, AlpacaPaperBroker(c, sleep=lambda s: None), 100_000, RiskLimits())
    strat.prepare_session(SESSION, None)
    res = eng.run_session(SESSION, flat_day(SESSION, price=100.0))
    assert [f.qty for f in eng.fills] == [500, -500]
    assert eng.fills[-1].purpose == "flatten" and not eng.portfolio.positions
    assert res.eod_equity == pytest.approx(100_000 - sum(f.fees for f in eng.fills))


class FakeBarsClient:
    def __init__(self, df):
        self.df, self.calls = df, []

    def bars(self, symbols, tf, start, end, feed, adj):
        self.calls.append((start, end, feed))
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self.df[(self.df.ts >= s) & (self.df.ts <= e)]


def _bars_df(n):
    ts = [ts_at(SESSION, m) for m in range(n)]
    return pd.DataFrame({"symbol": "SPY", "ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 1.0, "trade_count": 1.0, "vwap": 100.0})


def test_feed_waits_for_bar_close_and_yields_in_order():
    df = _bars_df(390)
    clock = {"t": ts_at(SESSION, 0) + pd.Timedelta(seconds=5)}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock["t"] += pd.Timedelta(seconds=s)

    feed = LiveFeed(FakeBarsClient(df), ["SPY"], SESSION, now=lambda: clock["t"], sleep=sleep)
    ev = list(itertools.islice(iter(feed), 3))
    assert [t for t, _ in ev] == [ts_at(SESSION, m) for m in range(3)]
    assert clock["t"] >= ts_at(SESSION, 3) + pd.Timedelta(seconds=2)     # never yields a bar before it closes


def test_feed_catch_up_returns_only_closed_bars():
    df = _bars_df(390)
    now = ts_at(SESSION, 120) + pd.Timedelta(seconds=30)
    feed = LiveFeed(FakeBarsClient(df), ["SPY"], SESSION, now=lambda: now, sleep=lambda s: None)
    ev = feed.catch_up()
    assert len(ev) == 120 and ev[-1][0] == ts_at(SESSION, 119)


def test_vol_target_scale_matches_portfolio_rule():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 200), index=pd.bdate_range("2024-01-01", periods=200).date)
    cfg = PortfolioConfig(target_vol=0.05, vol_lookback=60, max_scale=3.0, max_gross=4.0,
                          sleeve_max_gross={"s": 4.0}, max_stock_weight=0.1, sleeve_max_stock_weight={"s": 0.0})
    res = run_portfolio(pd.DataFrame({"s": r}), cfg)
    nxt = vol_target_scale(r, 0.05, 60, 3.0, 4.0, 4.0)
    # the live rule applied to data up to day t equals the portfolio's scale on day t+1
    r2 = pd.concat([r, pd.Series([0.0], index=[pd.Timestamp("2024-10-21").date()])])
    res2 = run_portfolio(pd.DataFrame({"s": r2}), cfg)
    assert res2.scale.iloc[-1] == pytest.approx(nxt)
