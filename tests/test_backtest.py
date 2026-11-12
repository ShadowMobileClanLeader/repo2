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
