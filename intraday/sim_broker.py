"""Simulated broker for backtests.

Fill rules (documented, unit-tested):
  * Market orders submitted after bar t fill at the OPEN of that symbol's next bar.
  * Stop orders (buy stop above / sell stop below):
        gap through the level at the open -> fill at the open (worse than the stop)
        level touched inside the bar        -> fill at the stop level
  * Slippage (bps, adverse) and fees from CostModel are applied to every fill.
  * Orders with `attached_stop_loss` create a child stop on fill (like an Alpaca OTO order).
    If that child's level is also inside the SAME bar's range, we assume the worst case:
    the stop was hit in that bar (we cannot know the intrabar order of high and low).
  * Orders for a symbol without a bar stay pending. At session end, any remaining
    position is closed at the last known price (with slippage) and flagged.
"""
from __future__ import annotations

import pandas as pd

from .costs import CostModel
from .engine import Bar, Fill, Order, child_stop_level


class SimBroker:
    def __init__(self, costs: CostModel):
        self.costs = costs
        self._open: dict[int, Order] = {}
        self._positions: dict[str, int] = {}   # broker's own view, to close leftovers at EOD

    def submit(self, order: Order) -> None:
        self._open[order.id] = order

    def cancel(self, order_id: int) -> None:
        self._open.pop(order_id, None)

    def open_orders(self) -> list[Order]:
        return list(self._open.values())

    # --------------------------------------------------------------------------
    def _fill(self, o: Order, ref: float, ts: pd.Timestamp, note: str = "") -> Fill:
        px = self.costs.fill_price(o.symbol, ref, o.qty)
        f = Fill(o.id, o.symbol, o.qty, px, ref, self.costs.fees(o.qty, px), ts, o.purpose, o.reason, note)
        self._positions[o.symbol] = self._positions.get(o.symbol, 0) + o.qty
        if self._positions[o.symbol] == 0:
            del self._positions[o.symbol]
        return f

    @staticmethod
    def _stop_ref(o: Order, b: Bar) -> float | None:
        s = o.stop_price
        if o.qty > 0:      # buy stop: triggers when price rises to s
            if b.open >= s:
                return b.open
            if b.high >= s:
                return s
        else:              # sell stop: triggers when price falls to s
            if b.open <= s:
                return b.open
            if b.low <= s:
                return s
        return None

    def process_bar(self, ts: pd.Timestamp, bars: dict[str, Bar]) -> list[Fill]:
        fills: list[Fill] = []
        # market orders first (they execute at the open), then stops in submission order
        orders = sorted(self._open.values(), key=lambda o: (o.type != "market", o.id))
        for o in orders:
            if o.id not in self._open:
                continue
            b = bars.get(o.symbol)
            if b is None:
                continue
            if o.type == "market":
                ref, note = b.open, ""
            else:
                ref = self._stop_ref(o, b)
                if ref is None:
                    continue
                note = "gap_through_stop" if ref == b.open else ""
            del self._open[o.id]
            fill = self._fill(o, ref, ts, note)
            fills.append(fill)
            level = o.attached_stop_loss
            if level is None and o.attached_stop_offset is not None:
                level = child_stop_level(o.qty, fill.price, o.attached_stop_offset)
            if level is not None:
                child = Order(o.symbol, -o.qty, "stop", level, "stop_loss",
                              f"OTO stop for order {o.id}", created=ts)
                # worst case: if the child level is inside this bar's range, assume it was hit
                if (o.qty > 0 and b.low <= child.stop_price) or (o.qty < 0 and b.high >= child.stop_price):
                    fills.append(self._fill(child, min(child.stop_price, ref) if o.qty > 0 else max(child.stop_price, ref),
                                            ts, "same_bar_stop_assumed"))
                else:
                    self._open[child.id] = child
        return fills

    def end_session(self, last_prices: dict[str, float], ts: pd.Timestamp) -> list[Fill]:
        fills = []
        self._open.clear()
        for sym, q in list(self._positions.items()):
            o = Order(sym, -q, "market", None, "flatten", "end of session: no bar after flatten order", created=ts)
            fills.append(self._fill(o, last_prices[sym], ts, "eod_forced_at_last_price"))
        return fills
