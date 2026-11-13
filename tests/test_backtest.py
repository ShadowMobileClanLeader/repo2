import pandas as pd
import pytest

from intraday import data
from intraday.backtest import HistoricalMarketData, LookAheadError, bar_events, session_bars
from intraday.calendar import make_session

from helpers import SESSION, ts_at


def test_bar_events_group_by_timestamp_in_order():
    df = pd.DataFrame({"symbol": ["B", "A", "A"], "ts": [ts_at(SESSION, 1), ts_at(SESSION, 1), ts_at(SESSION, 0)],
                       "open": [1.0, 2.0, 3.0], "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "vwap": 1.0})
    ev = list(bar_events(df))
    assert [t for t, _ in ev] == [ts_at(SESSION, 0), ts_at(SESSION, 1)]
    assert set(ev[1][1]) == {"A", "B"} and ev[0][1]["A"].open == 3.0


def test_session_bars_excludes_extended_hours():
    s = SESSION
    df = pd.DataFrame({"symbol": "A", "ts": [s.open.tz_convert("UTC") - pd.Timedelta(minutes=1), ts_at(s, 0),
                                             ts_at(s, 389), s.close.tz_convert("UTC")],
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "vwap": 1.0})
    assert len(session_bars(df, s)) == 2


def test_market_data_refuses_lookahead():
    md = HistoricalMarketData(features=pd.DataFrame({"date": []}))
    md.set_session(SESSION, None)
    with pytest.raises(LookAheadError):
        md.intraday_history(data.SPY_1MIN, [SESSION.date])


def test_spec_bounds_end_before_close():
    start, end = data.SPY_1MIN.bounds(SESSION)
    assert start.startswith("2024-03-04T14:30") and end.startswith("2024-03-04T20:59:59")
    _, end5 = data.OPEN5.bounds(SESSION)
    assert end5.startswith("2024-03-04T14:34:59")


def test_daily_history_excludes_current_session():
    import datetime as dt
    feats = pd.DataFrame({"symbol": "SPY", "date": [dt.date(2024, 2, 28), dt.date(2024, 2, 29),
                                                    dt.date(2024, 3, 1), SESSION.date, dt.date(2024, 3, 5)],
                          "ret_intraday": [1, 2, 3, 4, 5]})
    md = HistoricalMarketData(features=feats)
    md.set_session(SESSION, dt.date(2024, 3, 1))
    h = md.daily_history("SPY", 2)
    assert list(h["ret_intraday"]) == [2, 3]


def test_ex_dividend_from_adjustment_ratio():
    import datetime as dt
    feats = pd.DataFrame({"symbol": "SPY", "date": [dt.date(2024, 3, 1), SESSION.date, dt.date(2024, 3, 5)],
                          "close": [500.0, 999.0, 999.0], "adj_ratio": [0.99, 0.99 / (1 - 1.5 / 500), 2.0]})
    md = HistoricalMarketData(features=feats)
    md.set_session(SESSION, dt.date(2024, 3, 1))
    assert md.ex_dividend("SPY") == pytest.approx(1.5)
    md.set_session(make_session("2024-03-05"), SESSION.date)
    assert md.ex_dividend("SPY") == 0.0          # ratio jump > 5% (split-like) -> ignored


def test_reset_capital_daily(monkeypatch):
    import datetime as dt
    from intraday import backtest as bt
    from intraday.calendar import make_session
    from intraday.costs import CostModel
    from intraday.engine import Target
    from helpers import flat_day
    days = [make_session("2024-03-04"), make_session("2024-03-05")]
    monkeypatch.setattr(bt.calendar, "sessions", lambda a, b: days)

    def fake_load(spec, d, symbols=None):
        s = days[0] if d == days[0].date else days[1]
        ev = flat_day(s, price=100.0, overrides={m: (110.0,) * 4 for m in range(100, 390)})
        rows = [{"symbol": "AAA", "ts": t, "open": b["AAA"].open, "high": b["AAA"].high, "low": b["AAA"].low,
                 "close": b["AAA"].close, "volume": 1.0, "vwap": b["AAA"].close} for t, b in ev]
        return pd.DataFrame(rows)
    monkeypatch.setattr(bt.data, "load_intraday", fake_load)

    class Long:
        name = "long"
        def prepare_session(self, s, md):
            self.done = False
            return ["AAA"]
        def on_bar(self, now, bars, ctx):
            if not self.done:
                self.done = True
                return {"AAA": Target(1.0)}

    free = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0, min_commission=0,
                     sec_fee_rate=0, taf_per_share=0)
    md = HistoricalMarketData(features=pd.DataFrame({"date": [], "symbol": []}))
    r = bt.run_backtest(Long(), days[0].date, days[1].date, data.SPY_1MIN, costs=free, md=md, reset_capital_daily=True)
    assert list(r.daily["sod_equity"]) == [100_000.0, 100_000.0]
    assert r.returns.iloc[0] == pytest.approx(r.returns.iloc[1])
    r2 = bt.run_backtest(Long(), days[0].date, days[1].date, data.SPY_1MIN, costs=free, md=md)
    assert r2.daily["sod_equity"].iloc[1] == pytest.approx(110_000.0)
