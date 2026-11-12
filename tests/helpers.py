from __future__ import annotations

import pandas as pd

from intraday.calendar import make_session
from intraday.engine import Bar

SESSION = make_session("2024-03-04")              # normal day, 09:30-16:00 ET
HALF_DAY = make_session("2024-11-29", close_hhmm="13:00")


def ts_at(session, minute: int) -> pd.Timestamp:
    """UTC start time of the bar `minute` minutes after the open."""
    return (session.open + pd.Timedelta(minutes=minute)).tz_convert("UTC")


def bar(symbol, session, minute, o, h=None, l=None, c=None, v=1000.0) -> Bar:
    h = max(o, c if c is not None else o) if h is None else h
    l = min(o, c if c is not None else o) if l is None else l
    c = o if c is None else c
    return Bar(symbol, ts_at(session, minute), o, h, l, c, v)


def flat_day(session, symbol="AAA", price=100.0, overrides=None):
    """One bar per minute for the full session at a constant price, with overrides
    {minute: (o, h, l, c)}."""
    overrides = overrides or {}
    n = int((session.close - session.open) / pd.Timedelta(minutes=1))
    events = []
    for m in range(n):
        o, h, l, c = overrides.get(m, (price, price, price, price))
        events.append((ts_at(session, m), {symbol: Bar(symbol, ts_at(session, m), o, h, l, c, 1000.0)}))
    return events


class ScriptedStrategy:
    """Returns pre-scripted targets at given bar-close minutes; records every call time."""

    name = "scripted"

    def __init__(self, script: dict[int, dict]):
        self.script = script
        self.calls: list[pd.Timestamp] = []
        self.session = None

    def prepare_session(self, session, md):
        self.session = session
        return ["AAA"]

    def on_bar(self, now, bars, ctx):
        self.calls.append(now)
        minute = int((now - ctx.session.open) / pd.Timedelta(minutes=1))  # minutes since open at bar close
        return self.script.get(minute)


class FakeMD:
    """Minimal market-data stand-in: daily feature rows + per-day intraday frames."""

    def __init__(self, daily: "pd.DataFrame", intraday: dict | None = None, session=None):
        self.daily = daily
        self.intraday = intraday or {}
        self.session = session

    def daily_history(self, symbol, n):
        d = self.daily[(self.daily["symbol"] == symbol) & (self.daily["date"] < self.session.date)]
        return d.iloc[-n:]

    def ex_dividend(self, symbol):
        return getattr(self, "dividend", 0.0)

    def previous_session_dates(self, n):
        return sorted(d for d in self.intraday if d < self.session.date)[-n:]

    def intraday_history(self, spec, days, symbols=None):
        assert all(d < self.session.date for d in days)
        frames = [self.intraday[d] for d in days if d in self.intraday]
        return pd.concat(frames) if frames else pd.DataFrame(columns=["symbol", "ts", "open", "close"])
