"""Gamma-regime filter (UNTESTED — no historical open interest exists to backtest it).

Wraps a strategy and lets it trade only on days when estimated dealer gamma exposure (SPX + SPY,
from the previous session's open interest) is BELOW `threshold` (default 0: negative gamma).
Motivation: Baltussen, Da, Lammers & Martens (2021, JFE) find intraday momentum in index futures
is strongest when dealers are short gamma, because their hedging trades with the move.
If GEX cannot be computed, the filter does not trade (fail closed).
"""
from __future__ import annotations


class GammaRegimeFilter:
    def __init__(self, inner, threshold: float = 0.0):
        self.inner, self.threshold = inner, threshold
        self.name = f"{inner.name}+gex_filter"
        self.active = False
        self.gex = None

    def prepare_session(self, session, md):
        self.gex = md.gamma_exposure()
        total = None if self.gex is None else self.gex.get("total_gex_usd_per_1pct")
        self.active = total is not None and total == total and total < self.threshold
        if not self.active:
            return []
        return self.inner.prepare_session(session, md)

    def on_bar(self, now, bars, ctx):
        if not self.active:
            return None
        return self.inner.on_bar(now, bars, ctx)

    def __getattr__(self, item):              # expose inner state (decisions, leverage, ...)
        return getattr(self.__dict__["inner"], item)
