"""Backtest driver: replays cached bars through the shared Engine with a SimBroker."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import calendar, data
from .calendar import Session
from .costs import CostModel
from .engine import Bar, Engine, RiskLimits, Strategy
from .sim_broker import SimBroker
from .splits import check_range


class LookAheadError(RuntimeError):
    pass


class HistoricalMarketData:
    """Serves strategies ONLY data from sessions strictly before the current one."""

    def __init__(self, features: pd.DataFrame | None = None):
        self._features = features
        self.session: Session | None = None
        self.prev_session: dt.date | None = None

    @cached_property
    def features(self) -> pd.DataFrame:
        if self._features is not None:
            return self._features
        from .universe import FEATURES_FILE
        return pd.read_parquet(FEATURES_FILE)

    @cached_property
    def _features_by_date(self) -> dict:
        return {d: g for d, g in self.features.groupby("date")}

    @cached_property
    def _by_symbol(self) -> dict:
        return {s: g.reset_index(drop=True) for s, g in self.features.groupby("symbol")}

    def daily_history(self, symbol: str, n: int) -> pd.DataFrame:
        """Last n daily-feature rows for `symbol`, dated strictly before the current session."""
        g = self._by_symbol.get(symbol)
        if g is None:
            return self.features.iloc[0:0]
        i = int(np.searchsorted(g["date"].to_numpy(), self.session.date, side="left"))
        return g.iloc[max(0, i - n):i]

    def ex_dividend(self, symbol: str) -> float:
        """Cash dividend going ex on the current session (a scheduled, pre-announced corporate
        action; live trading gets it from Alpaca's corporate-actions API). Derived from the change
        in adjusted/raw close ratio: D = C_prev * (1 - ratio_prev / ratio_today). Only the ratio of
        today's row is read, never today's prices. Splits (ratio jumps > 5%) return 0."""
        g = self._by_symbol.get(symbol)
        if g is None or "adj_ratio" not in g:
            return 0.0
        dates = g["date"].to_numpy()
        i = int(np.searchsorted(dates, self.session.date, side="left"))
        if i == 0 or i >= len(g) or dates[i] != self.session.date:
            return 0.0
        r_prev, r_now = g["adj_ratio"].iloc[i - 1], g["adj_ratio"].iloc[i]
        f = r_prev / r_now
        if not (0.95 < f < 1.0):
            return 0.0
        return float(g["close"].iloc[i - 1] * (1 - f))

    # ---- opening-range data (Stocks in Play) ------------------------------------
    def _open5(self, feed: str) -> dict:
        cache = self.__dict__.setdefault("_open5_cache", {})
        if feed not in cache:
            from .orb_data import table_path
            t = pd.read_parquet(table_path(feed))
            cache[feed] = {d: g for d, g in t.groupby("date")}
        return cache[feed]

    def orb_snapshot(self, feed: str, now: pd.Timestamp, delay_minutes: int = 0) -> pd.DataFrame:
        """Today's 09:30-09:35 bars + relative volume for symbols that pass the prior-day filters,
        joined with prior-day ATR. Refuses to answer before the data would exist live:
        09:35 for real-time feeds, 09:35 + delay for delayed SIP."""
        available = self.session.open + pd.Timedelta(minutes=5 + delay_minutes)
        if now < available:
            raise LookAheadError(f"opening-range data for {feed} not available until {available:%H:%M}, asked at {now:%H:%M}")
        from .universe import orb_eligible
        prev = self.daily_features()
        elig = set(orb_eligible(prev))
        day = self._open5(feed).get(self.session.date)
        if day is None:
            return pd.DataFrame(columns=["symbol", "open", "high", "low", "close", "volume", "relvol", "atr14"])
        day = day[day["symbol"].isin(elig)]
        return day.merge(prev[["symbol", "atr14"]], on="symbol", how="left")

    def orb_symbols_to_load(self, feed: str, top_n: int, **kw) -> list[str]:
        """Data plumbing only: which symbols' 1-minute bars to load for today. The strategy cannot
        see rankings before orb_snapshot() allows it; this just avoids loading ~1000 symbols."""
        from .orb_data import select_stocks_in_play
        snap = self.orb_snapshot(feed, self.session.close, 0)
        return list(select_stocks_in_play(snap, top_n, **kw)["symbol"])

    @cached_property
    def _xs_monthly(self) -> dict:
        from .config import DATA_START, HOLDOUT_END
        from .universe import monthly_top_n
        dates = [s.date for s in calendar.sessions(DATA_START, HOLDOUT_END)]
        return monthly_top_n(self.features, dates, self.xs_n)

    xs_n = 200

    def xs_universe(self) -> list[str]:
        """Point-in-time top-200 liquid universe in force today (formed at the month start
        from the previous session's features)."""
        from .universe import universe_on
        return universe_on(self._xs_monthly, self.session.date)

    def set_session(self, session: Session, prev: dt.date | None) -> None:
        self.session, self.prev_session = session, prev

    def daily_features(self) -> pd.DataFrame:
        """Daily features computed at the close of the previous session."""
        f = self._features_by_date.get(self.prev_session)
        return f if f is not None else self.features.iloc[0:0]

    def intraday_history(self, spec: data.IntradaySpec, days: Iterable[dt.date],
                         symbols: Iterable[str] | None = None) -> pd.DataFrame:
        days = list(days)
        if any(d >= self.session.date for d in days):
            raise LookAheadError(f"history request for {max(days)} on/after {self.session.date}")
        frames = [data.load_intraday(spec, d, symbols) for d in days]
        return pd.concat(frames, ignore_index=True) if frames else data._empty()

    def previous_session_dates(self, n: int) -> list[dt.date]:
        return [s.date for s in calendar.previous_sessions(self.session.date, n)]


def bar_events(df: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, dict[str, Bar]]]:
    """Group a bars frame into time-ordered events {symbol: Bar}."""
    if df.empty:
        return
    df = df.sort_values(["ts", "symbol"])
    ts = df["ts"].to_numpy()
    cols = [df[c].to_numpy() for c in ("symbol", "open", "high", "low", "close", "volume", "vwap")]
    bounds = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(df)]])
    for a, b in zip(starts, ends):
        t = pd.Timestamp(ts[a]).tz_localize("UTC") if pd.Timestamp(ts[a]).tzinfo is None else pd.Timestamp(ts[a])
        yield t, {cols[0][i]: Bar(cols[0][i], t, cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i], cols[6][i])
                  for i in range(a, b)}


def session_bars(df: pd.DataFrame, s: Session) -> pd.DataFrame:
    """Restrict to regular-session bars [open, close)."""
    o, c = s.open.tz_convert("UTC"), s.close.tz_convert("UTC")
    return df[(df["ts"] >= o) & (df["ts"] < c)]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame

    @property
    def returns(self) -> pd.Series:
        return self.daily["ret"]


def run_backtest(strategy: Strategy, start: dt.date, end: dt.date, spec: data.IntradaySpec,
                 costs: CostModel = CostModel(), capital: float = 100_000.0,
                 risk: RiskLimits = RiskLimits(), md: HistoricalMarketData | None = None,
                 bar_minutes: int = 1) -> BacktestResult:
    check_range(start, end)
    md = md or HistoricalMarketData()
    engine = Engine(strategy, SimBroker(costs), capital, risk, bar_minutes)
    rows = []
    allsess = calendar.sessions(start - dt.timedelta(days=10), end)
    for i, s in enumerate(allsess):
        if s.date < start:
            continue
        md.set_session(s, allsess[i - 1].date if i > 0 else None)
        symbols = strategy.prepare_session(s, md)
        df = session_bars(data.load_intraday(spec, s.date, symbols), s) if symbols else data._empty()
        r = engine.run_session(s, bar_events(df))
        rows.append(r.__dict__)
    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily["pnl"] = daily["eod_equity"] - daily["sod_equity"]
        daily["ret"] = daily["pnl"] / daily["sod_equity"]
        daily = daily.set_index("date")
    fills = pd.DataFrame([f.__dict__ for f in engine.fills])
    orders = pd.DataFrame([o.__dict__ for o in engine.orders])
    return BacktestResult(daily, fills, orders)
