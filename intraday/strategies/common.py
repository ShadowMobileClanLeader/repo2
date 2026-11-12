"""Small building blocks shared by strategies (and identical in backtest and live)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..engine import Bar


class SessionVWAP:
    """Cumulative session VWAP from bar-level VWAPs (falls back to typical price)."""

    def __init__(self):
        self.pv = 0.0
        self.v = 0.0

    def update(self, b: Bar) -> float:
        px = b.vwap if b.vwap == b.vwap and b.vwap > 0 else (b.high + b.low + b.close) / 3
        self.pv += px * b.volume
        self.v += b.volume
        return self.value

    @property
    def value(self) -> float:
        return self.pv / self.v if self.v > 0 else math.nan


def minutes_since_open(now: pd.Timestamp, session) -> int:
    return int(round((now - session.open) / pd.Timedelta(minutes=1)))


def realized_daily_vol(returns: pd.Series, lookback: int) -> float:
    r = returns.dropna().iloc[-lookback:]
    return float(r.std(ddof=1)) if len(r) >= max(5, lookback // 2) else math.nan


def zscore_cross_section(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def decile_weights(score: pd.Series, frac: float = 0.1, gross: float = 1.0) -> dict[str, float]:
    """Long the top `frac`, short the bottom `frac`, equal-weighted, dollar-neutral, total gross `gross`."""
    s = score.dropna().sort_values()
    k = int(np.floor(len(s) * frac))
    if k == 0:
        return {}
    w = gross / 2 / k
    out = {sym: -w for sym in s.index[:k]}
    out.update({sym: w for sym in s.index[-k:]})
    return out
