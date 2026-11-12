"""Strategy 1: 5-minute Opening Range Breakout on Stocks in Play
(Zarattini, Barbon, Aziz 2024, "A Profitable Day Trading Strategy for the U.S. Equity Market").

Paper rules (explicit):
  * filters: open > $5, 14-day average volume >= 1M shares, 14-day ATR > $0.50
  * relative volume of the first 5 minutes vs its 14-day average >= 100%; trade the top 20
  * bullish first candle -> buy stop at the opening-range high; bearish -> sell stop at the low;
    doji -> no trade
  * stop loss 10% of ATR from the EXECUTED entry price; no target; exit at the close
  * risk 1% of the capital deployed per position, leverage capped at 4x; $0.0035/share commission

Our interpretations (flagged, not in the paper):
  * capital per position = start-of-day equity / top_n, so the whole book is capped at 4x
    (shares = min(risk * C / stop_dist, max_lev * C / entry_level))
  * ATR = simple 14-day mean of true range (Wilder vs simple not stated)
  * one attempt per stock per day; the stop-entry stays live until filled or the flatten deadline
  * a stop level already passed when the order is placed triggers at the next bar's open
  * flat by 15:55 (engine rule) rather than the paper's 16:00 close
  * ETFs are excluded from our universe (the paper appears to include them)

Variants: selection_feed "sip" (paper replication) or "iex" (what a free-plan live account sees);
delay_minutes=15 decides at 09:50 from 15-minute-delayed SIP opening bars.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..engine import Target
from ..orb_data import select_stocks_in_play
from .common import minutes_since_open


@dataclass
class ORBStocksInPlay:
    top_n: int = 20
    min_relvol: float = 1.0
    stop_atr_frac: float = 0.10
    risk_frac: float = 0.01
    max_leverage: float = 4.0
    min_open: float = 5.0
    selection_feed: str = "sip"
    delay_minutes: int = 0
    name: str = "orb_sip"

    def prepare_session(self, session, md):
        self.session, self.md = session, md
        self.decided = False
        self.selection = None
        return md.orb_symbols_to_load(self.selection_feed, self.top_n,
                                      min_relvol=self.min_relvol, min_open=self.min_open)

    def on_bar(self, now, bars, ctx):
        if self.decided or minutes_since_open(now, self.session) < 5 + self.delay_minutes:
            return None
        self.decided = True
        snap = self.md.orb_snapshot(self.selection_feed, now, self.delay_minutes)
        sel = select_stocks_in_play(snap, self.top_n, self.min_relvol, self.min_open)
        sel = sel[sel["atr14"] > 0]
        self.selection = sel
        cap = ctx.sod_equity / self.top_n
        targets = {}
        for r in sel.itertuples(index=False):
            entry = r.high if r.direction > 0 else r.low
            stop_dist = self.stop_atr_frac * r.atr14
            shares = min(self.risk_frac * cap / stop_dist, self.max_leverage * cap / entry)
            weight = r.direction * shares * entry / ctx.sod_equity
            side = "long above OR high" if r.direction > 0 else "short below OR low"
            targets[r.symbol] = Target(
                weight, entry_stop=float(entry), stop_offset=float(stop_dist),
                reason=f"in-play[{self.selection_feed}] relvol {r.relvol:.1f}x: {side} {entry:.2f}, "
                       f"stop {self.stop_atr_frac:.0%} ATR ({stop_dist:.2f})")
        return targets or None
