"""Strategy 4: VWAP mean-reversion on SPY — a deliberately simple CONTROL.

There is no canonical published "VWAP mean reversion" rule set, so this definition is ours
(one tunable parameter, k):
  * sigma = std of SPY's daily open-to-close returns over the prior 14 sessions
  * every `step` minutes between 10:00 and 15:30 ET:
      flat and close < VWAP * (1 - k*sigma)  -> long  1x
      flat and close > VWAP * (1 + k*sigma)  -> short 1x
  * exit when price crosses back through VWAP (the reversion target) or at the flatten deadline
No leverage, no stop: it is a baseline, not a candidate we expect to win.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engine import Target
from .common import SessionVWAP, minutes_since_open, realized_daily_vol


@dataclass
class VWAPMeanReversion:
    k: float = 0.5
    step: int = 5
    lookback: int = 14
    symbol: str = "SPY"
    first_minute: int = 30      # 10:00
    last_minute: int = 360      # 15:30
    name: str = "vwap_mr"

    def prepare_session(self, session, md):
        self.session = session
        self.vwap = SessionVWAP()
        hist = md.daily_history(self.symbol, self.lookback)
        self.sigma = realized_daily_vol(hist["ret_intraday"], self.lookback)
        return [self.symbol]

    def on_bar(self, now, bars, ctx):
        b = bars.get(self.symbol)
        if b is None:
            return None
        v = self.vwap.update(b)
        if self.sigma != self.sigma:
            return None
        pos = ctx.position(self.symbol)
        px = b.close
        if pos > 0 and px >= v:
            return {self.symbol: Target(0.0, reason=f"long reverted to VWAP {v:.2f}")}
        if pos < 0 and px <= v:
            return {self.symbol: Target(0.0, reason=f"short reverted to VWAP {v:.2f}")}
        m = minutes_since_open(now, self.session)
        if pos != 0 or m % self.step or not (self.first_minute <= m <= self.last_minute):
            return None
        band = self.k * self.sigma
        if px < v * (1 - band):
            return {self.symbol: Target(1.0, reason=f"close {px:.2f} < VWAP {v:.2f} - {band:.2%}")}
        if px > v * (1 + band):
            return {self.symbol: Target(-1.0, reason=f"close {px:.2f} > VWAP {v:.2f} + {band:.2%}")}
        return None
