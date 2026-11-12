"""Event-driven intraday engine shared by the backtester and the live (paper) runner.

Flow for every bar event (bars keyed by symbol, all with the same start timestamp):
  1. broker.process_bar(bars)   -> fills for orders submitted after the PREVIOUS bar
                                   (sim: next-bar open / stop levels; live: poll Alpaca)
  2. portfolio.apply(fills), mark-to-market at bar closes
  3. if bar close >= flatten deadline: flatten everything, no new entries
     else: targets = strategy.on_bar(...)  ->  risk checks  ->  orders
Strategies never see a bar before it has closed, and every order fills no earlier than
the next bar, so a signal can never trade at the price that generated it.

Strategies output TARGET POSITIONS (signed weight of strategy equity), optionally with a
stop-entry level (enter only if price trades through it) and a protective stop-loss.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from .calendar import Session
from .config import ET

# ----------------------------------------------------------------------------- types


@dataclass(slots=True)
class Bar:
    symbol: str
    ts: pd.Timestamp          # bar START (UTC), as delivered by Alpaca
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = math.nan


@dataclass(frozen=True)
class Target:
    weight: float                       # signed fraction of strategy equity; 0 = flat
    entry_stop: float | None = None     # enter via stop order at this level (only when flat)
    stop_loss: float | None = None      # protective stop for the resulting position (fixed level)
    reason: str = ""
    stop_offset: float | None = None    # protective stop at (executed entry price -/+ offset), for entries


_order_ids = itertools.count(1)


@dataclass
class Order:
    symbol: str
    qty: int                            # signed shares (+ buy, - sell)
    type: str = "market"                # "market" | "stop"
    stop_price: float | None = None
    purpose: str = "rebalance"          # entry | exit | rebalance | stop_loss | flatten | kill
    reason: str = ""
    attached_stop_loss: float | None = None   # OTO child stop created on fill (fixed level)
    attached_stop_offset: float | None = None # OTO child stop at fill price -/+ offset
    created: pd.Timestamp | None = None
    id: int = field(default_factory=lambda: next(_order_ids))
    broker_id: str | None = None


@dataclass(frozen=True)
class Fill:
    order_id: int
    symbol: str
    qty: int
    price: float              # executed price incl. slippage
    ref_price: float          # price before slippage (bar open / stop level)
    fees: float
    ts: pd.Timestamp
    purpose: str
    reason: str
    note: str = ""


def child_stop_level(entry_qty: int, entry_price: float, offset: float) -> float:
    """Protective stop `offset` dollars on the losing side of an executed entry."""
    return entry_price - offset if entry_qty > 0 else entry_price + offset


class Broker(Protocol):
    def submit(self, order: Order) -> None: ...
    def cancel(self, order_id: int) -> None: ...
    def open_orders(self) -> list[Order]: ...
    def process_bar(self, ts: pd.Timestamp, bars: dict[str, Bar]) -> list[Fill]: ...
    def end_session(self, last_prices: dict[str, float], ts: pd.Timestamp) -> list[Fill]: ...


class Strategy(Protocol):
    name: str

    def prepare_session(self, session: Session, md) -> list[str]:
        """Called before the open. Return the symbols whose bars the strategy needs today.
        `md` only serves data from before `session.date`."""

    def on_bar(self, now: pd.Timestamp, bars: dict[str, Bar], ctx: "Context") -> dict[str, Target] | None:
        """Called after each bar closes (`now` = close time, ET). Return targets to change
        (symbols omitted keep their current target), or None for no change."""


# ----------------------------------------------------------------------------- portfolio


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    last_price: dict[str, float] = field(default_factory=dict)
    fees_paid: float = 0.0
    slippage_paid: float = 0.0
    traded_notional: float = 0.0

    def apply(self, f: Fill) -> None:
        self.cash -= f.qty * f.price + f.fees
        q = self.positions.get(f.symbol, 0) + f.qty
        if q == 0:
            self.positions.pop(f.symbol, None)
        else:
            self.positions[f.symbol] = q
        self.fees_paid += f.fees
        self.slippage_paid += abs(f.price - f.ref_price) * abs(f.qty)
        self.traded_notional += abs(f.qty) * f.price
        self.last_price.setdefault(f.symbol, f.price)

    def mark(self, bars: dict[str, Bar]) -> None:
        for s, b in bars.items():
            self.last_price[s] = b.close

    def equity(self) -> float:
        return self.cash + sum(q * self.last_price[s] for s, q in self.positions.items())

    def gross(self) -> float:
        return sum(abs(q) * self.last_price[s] for s, q in self.positions.items())

    def net(self) -> float:
        return sum(q * self.last_price[s] for s, q in self.positions.items())


@dataclass
class Context:
    session: Session
    portfolio: Portfolio
    sod_equity: float              # start-of-day equity, used for sizing
    open_orders: list[Order]

    def position(self, symbol: str) -> int:
        return self.portfolio.positions.get(symbol, 0)


# ----------------------------------------------------------------------------- risk


@dataclass(frozen=True)
class RiskLimits:
    max_gross_leverage: float = 4.0      # gross notional / equity
    max_symbol_weight: float = 4.0       # |notional| / equity per symbol
    max_daily_loss: float = 0.05         # fraction of start-of-day equity -> flatten + halt for the day
    kill_switch_file: Path | None = None # if this file exists: flatten everything, stop trading
    min_trade_notional: float = 100.0    # ignore smaller rebalances


# ----------------------------------------------------------------------------- engine


@dataclass
class SessionResult:
    date: dt.date
    sod_equity: float
    eod_equity: float
    fees: float
    slippage: float
    traded_notional: float
    max_gross: float
    n_fills: int
    halted: str = ""


class Engine:
    def __init__(self, strategy: Strategy, broker: Broker, capital: float,
                 risk: RiskLimits = RiskLimits(), bar_minutes: int = 1, logger=None):
        self.strategy, self.broker, self.risk = strategy, broker, risk
        self.portfolio = Portfolio(cash=capital)
        self.bar = pd.Timedelta(minutes=bar_minutes)
        self.fills: list[Fill] = []
        self.orders: list[Order] = []
        self.stops: dict[str, float] = {}   # desired protective stop level per symbol
        self.log = logger or (lambda **kw: None)

    # -- public ------------------------------------------------------------------
    def run_session(self, session: Session, events: Iterable[tuple[pd.Timestamp, dict[str, Bar]]]) -> SessionResult:
        pf = self.portfolio
        sod = pf.equity()
        f0, s0, n0, fills0 = pf.fees_paid, pf.slippage_paid, pf.traded_notional, len(self.fills)
        deadline = session.flatten_time - self.bar   # last decision so fills land by flatten_time
        halted, flattened, max_gross = "", False, 0.0
        last_ts = session.open
        for ts, bars in events:
            last_ts = ts
            self._on_fills(self.broker.process_bar(ts, bars))
            pf.mark(bars)
            max_gross = max(max_gross, pf.gross())
            now = (ts + self.bar).tz_convert(ET)
            if not halted:
                halted = self._risk_halt(sod)
                if halted:
                    self._flatten(now, "kill" if "kill" in halted else "flatten", halted)
                    flattened = True
            if halted:
                continue
            if now >= deadline:
                if not flattened:
                    self._flatten(now, "flatten", f"session flatten deadline {session.flatten_time:%H:%M}")
                    flattened = True
                continue
            ctx = Context(session, pf, sod, self.broker.open_orders())
            targets = self.strategy.on_bar(now, bars, ctx)
            if targets:
                self._reconcile(targets, now, sod)
        # anything still open (e.g. no bar after the flatten order): close at last price
        self._on_fills(self.broker.end_session(dict(pf.last_price), last_ts))
        assert not pf.positions, f"positions left overnight: {pf.positions}"
        self.stops.clear()
        return SessionResult(session.date, sod, pf.equity(), pf.fees_paid - f0, pf.slippage_paid - s0,
                             pf.traded_notional - n0, max_gross, len(self.fills) - fills0, halted)

    # -- internals -------------------------------------------------------------
    def _on_fills(self, fills: list[Fill]) -> None:
        """Apply all fills of one bar first, then bring protective stops in line."""
        touched = []
        for f in fills:
            self.portfolio.apply(f)
            self.fills.append(f)
            self.log(event="fill", **f.__dict__)
            order = next((o for o in reversed(self.orders) if o.id == f.order_id), None)
            if order is not None and order.attached_stop_loss is not None:
                self.stops[f.symbol] = order.attached_stop_loss
            elif order is not None and order.attached_stop_offset is not None:
                self.stops[f.symbol] = child_stop_level(f.qty, f.price, order.attached_stop_offset)
            if f.symbol not in touched:
                touched.append(f.symbol)
        for sym in touched:
            if self.portfolio.positions.get(sym, 0) == 0:
                self.stops.pop(sym, None)
            self._sync_stop(sym, fills[-1].ts)

    def _submit(self, o: Order) -> None:
        self.orders.append(o)
        self.log(event="order", **{k: v for k, v in o.__dict__.items()})
        self.broker.submit(o)

    def _cancel_symbol(self, sym: str, purposes: tuple[str, ...] | None = None) -> None:
        for o in self.broker.open_orders():
            if o.symbol == sym and (purposes is None or o.purpose in purposes):
                self.broker.cancel(o.id)

    def _pending_market_qty(self, sym: str) -> int:
        return sum(o.qty for o in self.broker.open_orders() if o.symbol == sym and o.type == "market")

    def _sync_stop(self, sym: str, now: pd.Timestamp) -> None:
        pos = self.portfolio.positions.get(sym, 0)
        level = self.stops.get(sym)
        existing = [o for o in self.broker.open_orders() if o.symbol == sym and o.purpose == "stop_loss"]
        if pos == 0 or level is None:
            for o in existing:
                self.broker.cancel(o.id)
            return
        if len(existing) == 1 and existing[0].qty == -pos and existing[0].stop_price == level:
            return
        for o in existing:
            self.broker.cancel(o.id)
        self._submit(Order(sym, -pos, "stop", level, "stop_loss", f"protective stop @ {level:.4f}", created=now))

    def _flatten(self, now: pd.Timestamp, purpose: str, reason: str) -> None:
        for o in self.broker.open_orders():
            self.broker.cancel(o.id)
        self.stops.clear()
        for sym, q in list(self.portfolio.positions.items()):
            self._submit(Order(sym, -q, "market", None, purpose, reason, created=now))

    def _risk_halt(self, sod: float) -> str:
        r = self.risk
        if r.kill_switch_file is not None and Path(r.kill_switch_file).exists():
            return f"kill switch file {r.kill_switch_file} present"
        if sod > 0 and self.portfolio.equity() < sod * (1 - r.max_daily_loss):
            return f"daily loss limit {r.max_daily_loss:.1%} breached"
        return ""

    def _reconcile(self, targets: dict[str, Target], now: pd.Timestamp, sod: float) -> None:
        pf, r = self.portfolio, self.risk
        # gross-leverage cap: scale all non-zero target weights proportionally
        merged = {s: q * pf.last_price[s] / sod for s, q in pf.positions.items()} if sod > 0 else {}
        for s, t in targets.items():
            merged[s] = max(-r.max_symbol_weight, min(r.max_symbol_weight, t.weight))
        gross = sum(abs(w) for w in merged.values())
        scale = min(1.0, r.max_gross_leverage / gross) if gross > 0 else 1.0
        for sym, t in targets.items():
            w = max(-r.max_symbol_weight, min(r.max_symbol_weight, t.weight)) * scale
            cur = pf.positions.get(sym, 0)
            price = pf.last_price.get(sym)
            if w == 0:
                self._cancel_symbol(sym)
                self.stops.pop(sym, None)
                pending = self._pending_market_qty(sym)
                if cur + pending != 0:
                    self._submit(Order(sym, -(cur + pending), "market", None, "exit", t.reason, created=now))
                continue
            if t.entry_stop is not None and cur == 0:
                qty = int(w * sod / t.entry_stop)
                self._cancel_symbol(sym, ("entry",))
                if qty != 0:
                    self._submit(Order(sym, qty, "stop", t.entry_stop, "entry", t.reason,
                                       attached_stop_loss=t.stop_loss, attached_stop_offset=t.stop_offset,
                                       created=now))
                continue
            if price is None:
                continue
            qty = int(w * sod / price)
            delta = qty - cur - self._pending_market_qty(sym)
            if delta != 0 and abs(delta) * price >= r.min_trade_notional:
                purpose = "entry" if cur == 0 else "rebalance"
                self._submit(Order(sym, delta, "market", None, purpose, t.reason,
                                   attached_stop_loss=t.stop_loss if cur == 0 else None,
                                   attached_stop_offset=t.stop_offset if cur == 0 else None, created=now))
            if t.stop_loss is not None:
                self.stops[sym] = t.stop_loss
                if cur != 0:
                    self._sync_stop(sym, now)
