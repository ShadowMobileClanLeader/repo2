"""Live (PAPER) plumbing for the shared engine: broker, bar feed and market data.

The Engine and strategies are the exact same objects used in backtests; only these three
adapters differ. Everything goes through AlpacaClient, which refuses any non-paper endpoint.

Account hygiene: the paper account may hold positions that are not ours (e.g. options). The
broker tracks ONLY the positions created by its own fills and only ever closes those.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .alpaca_client import AlpacaClient
from .calendar import Session, make_session
from .config import UTC
from .costs import CostModel
from .engine import Bar, Fill, Order
from .universe import compute_daily_features

TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}

# Alpaca charges no commission; regulatory fees on sells are passed through.
ALPACA_FEES = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0.0,
                        min_commission=0.0)


class AlpacaPaperBroker:
    """Broker protocol on Alpaca paper. Market/stop orders only (protective stops are submitted by
    the engine after fills, so no OTO is needed). Fills are discovered by polling order status."""

    def __init__(self, client: AlpacaClient, tag: str = "fwd", fee_model: CostModel = ALPACA_FEES,
                 log: Callable[..., None] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.client, self.tag, self.fees = client, tag, fee_model
        self.log = log or (lambda **kw: None)
        self.sleep = sleep
        self._open: dict[int, Order] = {}
        self._seen: dict[int, tuple[int, float]] = {}      # engine order id -> (filled qty, avg price)
        self._ref: dict[int, float] = {}
        self.positions: dict[str, int] = {}
        self.last_price: dict[str, float] = {}

    # -- protocol -----------------------------------------------------------------
    def submit(self, o: Order) -> None:
        payload = {"symbol": o.symbol, "qty": str(abs(int(o.qty))), "side": "buy" if o.qty > 0 else "sell",
                   "type": "market" if o.type == "market" else "stop", "time_in_force": "day",
                   "client_order_id": f"{self.tag}-{dt.date.today():%Y%m%d}-{o.id}-{int(time.time())}"}
        if o.type == "stop":
            payload["stop_price"] = f"{o.stop_price:.2f}"
        resp = self.client.trading_post("/v2/orders", payload)
        o.broker_id = resp["id"]
        self._open[o.id] = o
        self._seen[o.id] = (0, 0.0)
        self._ref[o.id] = self.last_price.get(o.symbol, math.nan)
        self.log(event="broker_submit", order_id=o.id, broker_id=o.broker_id, payload=payload,
                 ref_price=self._ref[o.id], reason=o.reason)

    def cancel(self, order_id: int) -> None:
        o = self._open.get(order_id)
        if o is None:
            return
        try:
            self.client.trading_delete(f"/v2/orders/{o.broker_id}")
        except Exception as e:                       # already filled / not cancelable
            self.log(event="cancel_failed", order_id=order_id, error=str(e)[:200])
        self.log(event="broker_cancel", order_id=order_id)
        # keep tracking until the broker reports a terminal state (a fill may still arrive)

    def open_orders(self) -> list[Order]:
        return list(self._open.values())

    def process_bar(self, ts: pd.Timestamp, bars: dict[str, Bar]) -> list[Fill]:
        for s, b in bars.items():
            self.last_price[s] = b.close
        return self._poll()

    def end_session(self, last_prices: dict[str, float], ts: pd.Timestamp) -> list[Fill]:
        """Cancel our open orders and close any position WE hold; never touches other positions."""
        for oid in list(self._open):
            self.cancel(oid)
        fills = self._poll()
        for sym, q in list(self.positions.items()):
            if q != 0:
                self.submit(Order(sym, -q, "market", None, "flatten", "end of session safety flatten", created=ts))
        for _ in range(30):
            fills += self._poll()
            if not any(self.positions.values()) and not self._open:
                break
            self.sleep(2)
        return fills

    # -- internals ---------------------------------------------------------------------
    def _poll(self) -> list[Fill]:
        fills = []
        for oid, o in list(self._open.items()):
            st = self.client.trading_get(f"/v2/orders/{o.broker_id}")
            cum = int(float(st.get("filled_qty") or 0))
            avg = float(st.get("filled_avg_price") or 0.0)
            prev_q, prev_avg = self._seen[oid]
            if cum > prev_q:
                dq = cum - prev_q
                px = (cum * avg - prev_q * prev_avg) / dq
                sq = int(math.copysign(dq, o.qty))
                t = pd.Timestamp(st.get("filled_at") or st.get("updated_at") or dt.datetime.now(UTC))
                ref = self._ref.get(oid, math.nan)
                f = Fill(o.id, o.symbol, sq, px, ref if ref == ref else px, self.fees.fees(sq, px), t,
                         o.purpose, o.reason, note=f"alpaca {st.get('status')}")
                fills.append(f)
                self.positions[o.symbol] = self.positions.get(o.symbol, 0) + sq
                if self.positions[o.symbol] == 0:
                    del self.positions[o.symbol]
                self._seen[oid] = (cum, avg)
                self.log(event="broker_fill", order_id=oid, broker_id=o.broker_id, qty=sq, price=px,
                         ref_price=ref, status=st.get("status"))
            if st.get("status") in TERMINAL:
                self._open.pop(oid, None)
                if st.get("status") != "filled":
                    self.log(event="broker_terminal", order_id=oid, status=st.get("status"))
        return fills


class LiveFeed:
    """Yields (bar_start_utc, {symbol: Bar}) as each 1-minute bar closes, from the IEX feed
    (the only real-time feed on the free plan). A bar that has not appeared `max_wait` seconds
    after its minute ends is skipped (no event), exactly like a missing bar in a backtest."""

    def __init__(self, client: AlpacaClient, symbols: list[str], session: Session, feed: str = "iex",
                 poll_after: float = 2.0, max_wait: float = 25.0, now: Callable[[], pd.Timestamp] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.client, self.symbols, self.session, self.feed = client, symbols, session, feed
        self.poll_after, self.max_wait = poll_after, max_wait
        self.now = now or (lambda: pd.Timestamp.now(tz=UTC))
        self.sleep = sleep

    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self.client.bars(self.symbols, "1Min", start.isoformat(), end.isoformat(), self.feed, "raw")

    @staticmethod
    def _events(df: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, dict[str, Bar]]]:
        for ts, g in df.sort_values("ts").groupby("ts"):
            yield ts, {r.symbol: Bar(r.symbol, ts, r.open, r.high, r.low, r.close, r.volume,
                                     r.vwap if r.vwap == r.vwap else math.nan) for r in g.itertuples()}

    def catch_up(self) -> list[tuple[pd.Timestamp, dict[str, Bar]]]:
        """Bars that already closed today (for warming strategy state when starting mid-session)."""
        o = self.session.open.tz_convert(UTC)
        last_closed = self.now().floor("min") - pd.Timedelta(minutes=1)
        if last_closed < o:
            return []
        return list(self._events(self._fetch(o, last_closed + pd.Timedelta(seconds=59))))

    def __iter__(self):
        o, c = self.session.open.tz_convert(UTC), self.session.close.tz_convert(UTC)
        m = max(o, self.now().floor("min"))
        while m < c:
            ready = m + pd.Timedelta(minutes=1, seconds=self.poll_after)
            wait = (ready - self.now()).total_seconds()
            if wait > 0:
                self.sleep(wait)
            deadline = m + pd.Timedelta(minutes=1, seconds=self.max_wait)
            while True:
                df = self._fetch(m, m + pd.Timedelta(seconds=59))
                if len(df) or self.now() > deadline:
                    break
                self.sleep(2)
            for ev in self._events(df):
                yield ev
            m += pd.Timedelta(minutes=1)


class LiveMarketData:
    """Same interface as HistoricalMarketData, served from the Alpaca API (no holdout lock: live
    trading is not research). Only data from BEFORE the session is exposed."""

    def __init__(self, client: AlpacaClient, symbol: str = "SPY", intraday_feed: str = "iex"):
        self.client, self.symbol, self.intraday_feed = client, symbol, intraday_feed
        self._intraday: dict[dt.date, pd.DataFrame] = {}
        self._cal = None
        self.session: Session | None = None

    def _calendar(self) -> list[Session]:
        if self._cal is None:
            today = dt.date.today()
            rows = self.client.calendar((today - dt.timedelta(days=200)).isoformat(), (today + dt.timedelta(days=10)).isoformat())
            self._cal = [make_session(r["date"], r["open"], r["close"]) for r in rows]
        return self._cal

    def session_for(self, d: dt.date) -> Session | None:
        return next((s for s in self._calendar() if s.date == d), None)

    def set_session(self, session: Session) -> None:
        self.session = session
        start = (session.date - dt.timedelta(days=120)).isoformat()
        end = (session.date - dt.timedelta(days=1)).isoformat()
        raw = self.client.bars([self.symbol], "1Day", start, end, "sip", "raw")
        adj = self.client.bars([self.symbol], "1Day", start, end, "sip", "all")
        for df in (raw, adj):
            df["date"] = df["ts"].dt.tz_convert("America/New_York").dt.date
        self.features = compute_daily_features(raw, adj)
        self.features = self.features[self.features["date"] < session.date]

    def daily_history(self, symbol: str, n: int) -> pd.DataFrame:
        f = self.features[(self.features["symbol"] == symbol) & (self.features["date"] < self.session.date)]
        return f.iloc[-n:]

    def previous_session_dates(self, n: int) -> list[dt.date]:
        return [s.date for s in self._calendar() if s.date < self.session.date][-n:]

    def intraday_history(self, spec, days, symbols=None) -> pd.DataFrame:
        frames = []
        for d in days:
            if d >= self.session.date:
                raise ValueError("live market data only serves past sessions")
            if d not in self._intraday:
                s = self.session_for(d)
                o, c = s.open.tz_convert(UTC), s.close.tz_convert(UTC) - pd.Timedelta(seconds=1)
                self._intraday[d] = self.client.bars(symbols or [self.symbol], "1Min", o.isoformat(), c.isoformat(),
                                                     self.intraday_feed, "raw")
            frames.append(self._intraday[d])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def ex_dividend(self, symbol: str) -> float:
        d = self.session.date.isoformat()
        rows = self.client.corporate_actions("cash_dividend", d, d)
        return float(sum(float(r.get("rate") or 0) for r in rows
                         if r.get("symbol") == symbol and str(r.get("ex_date")) == d))


def vol_target_scale(returns: pd.Series, target_vol: float, lookback: int, max_scale: float,
                     max_gross: float, sleeve_max_gross: float) -> float:
    """Same rule as portfolio.run_portfolio for a single sleeve (weight 1)."""
    r = returns.dropna().iloc[-lookback:]
    if len(r) < 20:
        return 0.0
    vol = float(r.std(ddof=1) * np.sqrt(252))
    if vol <= 0:
        return 0.0
    return min(target_vol / vol, max_scale, max_gross / sleeve_max_gross)
