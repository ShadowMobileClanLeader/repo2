"""Transaction cost model: slippage (bps, per symbol) + commissions + regulatory fees.

Defaults are deliberately conservative-ish for an IBKR-Pro-like account. Alpaca itself charges
no commission, but we do not want results that only work at zero cost. All values are
configurable; use `stressed(2)` / `stressed(3)` for slippage stress tests.
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class CostModel:
    default_slippage_bps: float = 5.0            # single stocks
    symbol_slippage_bps: Mapping[str, float] = field(
        default_factory=lambda: {"SPY": 1.5, "QQQ": 1.5, "IWM": 2.0})
    symbol_slippage_per_share: Mapping[str, float] = field(default_factory=dict)  # $/share, overrides bps
    # time-of-day profile for the default (non-symbol-specific) slippage: ((minutes after 09:30, bps), ...)
    # linearly interpolated, flat outside the first/last point. Built from measured quoted half-spreads.
    tod_profile: tuple = ()
    slippage_multiplier: float = 1.0              # 2.0 / 3.0 for stress runs
    commission_per_share: float = 0.0035
    min_commission: float = 0.35                  # per order
    max_commission_pct: float = 0.01              # cap as fraction of trade value
    sec_fee_rate: float = 27.80e-6                # $ per $ of sale proceeds (sells & short sells)
    taf_per_share: float = 0.000166               # FINRA TAF on sells
    taf_max: float = 8.30

    def slippage_bps(self, symbol: str, ts=None) -> float:
        if symbol in self.symbol_slippage_bps:
            base = self.symbol_slippage_bps[symbol]
        elif self.tod_profile and ts is not None:
            et = ts.tz_convert("America/New_York")
            m = (et.hour * 60 + et.minute) - (9 * 60 + 30)
            xs, ys = zip(*self.tod_profile)
            base = float(np.interp(m, xs, ys))
        else:
            base = self.default_slippage_bps
        return base * self.slippage_multiplier

    def fill_price(self, symbol: str, ref_price: float, qty: float, ts=None) -> float:
        """Adverse slippage: buys pay up, sells receive less."""
        side = 1.0 if qty > 0 else -1.0
        per_share = self.symbol_slippage_per_share.get(symbol)
        if per_share is not None:
            return ref_price + side * per_share * self.slippage_multiplier
        return ref_price * (1.0 + side * self.slippage_bps(symbol, ts) / 1e4)

    def fees(self, qty: float, price: float) -> float:
        """Commission + regulatory fees in $ for one fill (qty signed, + buy / - sell)."""
        shares = abs(qty)
        if shares == 0:
            return 0.0
        notional = shares * price
        comm = max(self.min_commission, self.commission_per_share * shares)
        comm = min(comm, self.max_commission_pct * notional) if notional > 0 else comm
        reg = 0.0
        if qty < 0:
            reg = self.sec_fee_rate * notional + min(self.taf_max, self.taf_per_share * shares)
        return comm + reg

    def stressed(self, multiplier: float) -> "CostModel":
        return replace(self, slippage_multiplier=self.slippage_multiplier * multiplier)

    def round_trip_bps(self, symbol: str, price: float, shares: float) -> float:
        """Approximate total round-trip cost in bps of notional (for capacity / reporting)."""
        notional = price * shares
        fees = self.fees(shares, price) + self.fees(-shares, price)
        return 2 * self.slippage_bps(symbol) + (fees / notional * 1e4 if notional else math.inf)
