"""Strategy 3: cross-sectional long/short on a point-in-time top-200 liquid universe.

Long the top decile, short the bottom decile, equal weight, dollar-neutral, gross 1x
(volatility targeting is a Phase-3 portfolio decision). Each signal is a separate, pre-specified
variant — every one is recorded as a trial.

Daily signals (scores known at the prior close; hold open -> flatten deadline):
  rev1          -ret(prior day)                          short-term reversal
  rev5          -cum ret(prior 5 days)                   1-week reversal (de Groot, Huij, Zhou 2012)
  lps_intraday  +cum intraday ret(prior 21 days)         Lou, Polk, Skouras (2019) intraday persistence
  lps_overnight -cum overnight ret(prior 21 days)        LPS: past overnight winners lose intraday
  lps_combo     z(lps_intraday) + z(lps_overnight)       pre-specified equal weights (not fitted)
Intraday signal:
  eod_reversal  -ret(prior close -> 15:00), enter 15:30, hold to the flatten deadline
                (Baltussen, Da, Soebhag 2025 hold 15:30-16:00; we must be flat by 15:55)

Deviations: LPS measure "open" as the 09:30-10:00 VWAP and rebalance monthly with value
weights; we use the official open, daily rebalancing, equal weights. HKS same-half-hour
continuation is not implemented: its large-cap effect (~2 bps) is below the half-spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..engine import Target
from .common import decile_weights, minutes_since_open, zscore_cross_section

DAILY_SIGNALS = ("rev1", "rev5", "lps_intraday", "lps_overnight", "lps_combo")


def daily_scores(hist: dict[str, pd.DataFrame], signal: str, lookback: int = 21) -> pd.Series:
    """hist: symbol -> daily feature rows strictly before today (oldest first)."""
    vals = {}
    for sym, h in hist.items():
        if signal == "rev1":
            if len(h) >= 1:
                vals[sym] = -h["ret1"].iloc[-1]
        elif signal == "rev5":
            if len(h) >= 5:
                vals[sym] = -(np.prod(1 + h["ret1"].iloc[-5:].to_numpy()) - 1)
        elif signal in ("lps_intraday", "lps_overnight", "lps_combo"):
            if len(h) >= lookback:
                w = h.iloc[-lookback:]
                vals[sym] = (np.prod(1 + w["ret_intraday"].to_numpy()) - 1,
                             np.prod(1 + w["ret_overnight"].to_numpy()) - 1)
        else:
            raise ValueError(signal)
    if signal.startswith("lps"):
        df = pd.DataFrame(vals, index=["intra", "over"]).T.dropna()
        if signal == "lps_intraday":
            return df["intra"]
        if signal == "lps_overnight":
            return -df["over"]
        return zscore_cross_section(df["intra"]) - zscore_cross_section(df["over"])
    return pd.Series(vals, dtype=float).dropna()


@dataclass
class CrossSectional:
    signal: str = "lps_combo"
    frac: float = 0.10
    gross: float = 1.0
    lookback: int = 21
    entry_minute: int = 5           # decide at the 09:35 bar close (5-minute bars)
    eod_signal_minute: int = 330    # 15:00
    eod_entry_minute: int = 360     # 15:30
    max_abs_move: float = 0.30      # drop symbols whose move vs prior close suggests a split/bad tick
    name: str = "xs"

    def prepare_session(self, session, md):
        self.session, self.md = session, md
        self.universe = md.xs_universe()
        self.done = False
        self.hist = {s: md.daily_history(s, max(self.lookback, 5) + 1) for s in self.universe}
        self.prev_close = {s: float(h["close"].iloc[-1]) for s, h in self.hist.items() if len(h)}
        self.eod_score = None
        return self.universe

    def on_bar(self, now, bars, ctx):
        m = minutes_since_open(now, self.session)
        if self.signal == "eod_reversal":
            return self._eod(m, bars)
        if self.done or m < self.entry_minute:
            return None
        self.done = True
        score = daily_scores(self.hist, self.signal, self.lookback)
        score = score[[s for s in score.index if s in bars and self._sane(s, bars[s].close)]]
        w = decile_weights(score, self.frac, self.gross)
        return {s: Target(x, reason=f"{self.signal} score {score[s]:+.4f} ({'long' if x > 0 else 'short'} decile)")
                for s, x in w.items()} or None

    def _sane(self, sym, px) -> bool:
        pc = self.prev_close.get(sym)
        return pc is not None and pc > 0 and abs(px / pc - 1) < self.max_abs_move

    def _eod(self, m, bars):
        if m == self.eod_signal_minute:
            r = {s: b.close / self.prev_close[s] - 1 for s, b in bars.items()
                 if s in self.prev_close and self._sane(s, b.close)}
            self.eod_score = -pd.Series(r, dtype=float)
        if m == self.eod_entry_minute and self.eod_score is not None and not self.done:
            self.done = True
            w = decile_weights(self.eod_score[[s for s in self.eod_score.index if s in bars]], self.frac, self.gross)
            return {s: Target(x, reason=f"EOD reversal: ret to 15:00 {-self.eod_score[s]:+.2%}") for s, x in w.items()} or None
        return None
