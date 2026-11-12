"""Strategy 2: SPY intraday momentum — "noise area" (Zarattini, Aziz, Barbon 2024, "Beat the Market").

Rules as implemented (see research/phase2_strategy_specs.md for sources / ambiguities):
  * move(t)  = |close_t / open_day - 1| for each minute-of-day t
  * sigma(t) = mean of move(t) over the previous `lookback` (14) sessions
  * prev_close is reduced by the cash dividend on ex-dates (as in the authors' code)
  * upper(t) = max(open_day, prev_close) * (1 + band_mult * sigma(t))
    lower(t) = min(open_day, prev_close) * (1 - band_mult * sigma(t))
  * decisions only at bar closes on the `step` grid (every 30 min from 10:00)
      long  if close > upper(t) and close > VWAP
      short if close < lower(t) and close < VWAP
  * trailing stop, checked at the same decision times:
      long exits if close < max(upper(t), VWAP); short exits if close > min(lower(t), VWAP)
  * size: leverage = min(max_leverage, target_vol / sigma_spy_daily(14)), set once per day
  * flat at the engine's flatten deadline (15:54 fill; the paper exits at the 16:00 close)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import data
from ..engine import Target
from .common import SessionVWAP, minutes_since_open, realized_daily_vol


@dataclass
class NoiseAreaMomentum:
    lookback: int = 14
    step: int = 30
    first_decision: int = 30          # minutes after the open (10:00)
    band_mult: float = 1.0
    target_vol: float = 0.02          # daily
    max_leverage: float = 4.0
    vol_lookback: int = 14
    use_vwap: bool = True
    adjust_dividends: bool = True     # prev close minus ex-dividend amount (authors' code)
    symbol: str = "SPY"
    spec: data.IntradaySpec = field(default_factory=lambda: data.SPY_1MIN)
    name: str = "noise_area"

    def prepare_session(self, session, md):
        self.session = session
        self.vwap = SessionVWAP()
        self.open_px = None
        prev = md.daily_history(self.symbol, max(self.vol_lookback, 1) + 1)
        self.dividend = md.ex_dividend(self.symbol) if self.adjust_dividends else 0.0
        self.prev_close = float(prev["close"].iloc[-1]) - self.dividend if len(prev) else np.nan
        sd = realized_daily_vol(prev["ret1"], self.vol_lookback)
        self.leverage = min(self.max_leverage, self.target_vol / sd) if sd == sd and sd > 0 else 0.0
        self.sigma = self._sigma_profile(md)
        return [self.symbol]

    def _sigma_profile(self, md) -> pd.Series:
        """Mean absolute move from the open, by minute-of-day, over the prior `lookback` sessions."""
        days = md.previous_session_dates(self.lookback)
        cache = self.__dict__.setdefault("_moves", {})
        moves = []
        for d in days:
            if d not in cache:
                df = md.intraday_history(self.spec, [d], [self.symbol])
                if df.empty:
                    cache[d] = None
                else:
                    ts_et = df["ts"].dt.tz_convert("America/New_York")
                    minute = (ts_et.dt.hour * 60 + ts_et.dt.minute) - (9 * 60 + 30) + 1   # bar close, minutes after open
                    first_open = df["open"].iloc[0]
                    cache[d] = pd.Series((df["close"] / first_open - 1).abs().to_numpy(), index=minute.to_numpy())
            if cache[d] is not None:
                moves.append(cache[d])
        for d in [k for k in cache if days and k < days[0]]:
            del cache[d]
        if len(moves) < max(5, self.lookback // 2):
            return pd.Series(dtype=float)
        return pd.concat(moves, axis=1).mean(axis=1)

    def bands(self, minute: int) -> tuple[float, float]:
        s = self.sigma.get(minute, np.nan) * self.band_mult
        hi = max(self.open_px, self.prev_close) * (1 + s)
        lo = min(self.open_px, self.prev_close) * (1 - s)
        return hi, lo

    def on_bar(self, now, bars, ctx):
        b = bars.get(self.symbol)
        if b is None:
            return None
        if self.open_px is None:
            self.open_px = b.open
        v = self.vwap.update(b) if self.use_vwap else np.nan
        m = minutes_since_open(now, self.session)
        if m < self.first_decision or (m - self.first_decision) % self.step or self.sigma.empty:
            return None
        if self.leverage <= 0 or self.prev_close != self.prev_close:
            return None
        hi, lo = self.bands(m)
        if hi != hi:
            return None
        px, pos = b.close, ctx.position(self.symbol)
        up_stop = max(hi, v) if self.use_vwap else hi
        dn_stop = min(lo, v) if self.use_vwap else lo
        if pos > 0 and px < up_stop:
            return {self.symbol: Target(0.0, reason=f"long trail: {px:.2f} < max(UB {hi:.2f}, VWAP {v:.2f})")}
        if pos < 0 and px > dn_stop:
            return {self.symbol: Target(0.0, reason=f"short trail: {px:.2f} > min(LB {lo:.2f}, VWAP {v:.2f})")}
        if pos <= 0 and px > hi and (not self.use_vwap or px > v):
            return {self.symbol: Target(self.leverage, reason=f"break up: {px:.2f} > UB {hi:.2f} & VWAP {v:.2f}, lev {self.leverage:.2f}")}
        if pos >= 0 and px < lo and (not self.use_vwap or px < v):
            return {self.symbol: Target(-self.leverage, reason=f"break down: {px:.2f} < LB {lo:.2f} & VWAP {v:.2f}, lev {self.leverage:.2f}")}
        return None
