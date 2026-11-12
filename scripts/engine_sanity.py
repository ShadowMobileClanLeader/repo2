"""End-to-end check of the engine on real cached SPY bars (development period only).

Strategy: 100% long SPY after the first bar, flattened by the engine at 15:54.
Expected daily gross return = open(09:31 bar) -> open(15:54 bar), which we compute
independently from the raw bars and compare. Also reports cost drag and runtime.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from intraday import data, metrics
from intraday.backtest import run_backtest
from intraday.costs import CostModel
from intraday.engine import Target

FREE = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0, min_commission=0,
                 sec_fee_rate=0, taf_per_share=0)


class LongSPYIntraday:
    name = "long_spy_intraday"

    def prepare_session(self, session, md):
        self.sent = False
        return ["SPY"]

    def on_bar(self, now, bars, ctx):
        if not self.sent and "SPY" in bars:
            self.sent = True
            return {"SPY": Target(1.0, reason="baseline: long at open")}
        return None


def independent(start, end) -> pd.Series:
    from intraday import calendar
    out = {}
    for s in calendar.sessions(start, end):
        df = data.load_intraday(data.SPY_1MIN, s.date).set_index("ts")
        t_in = (s.open + pd.Timedelta(minutes=1)).tz_convert("UTC")
        t_out = (s.flatten_time - pd.Timedelta(minutes=1)).tz_convert("UTC")
        out[s.date] = df.loc[t_out, "open"] / df.loc[t_in, "open"] - 1
    return pd.Series(out)


def main():
    start, end = dt.date(2023, 1, 3), dt.date(2023, 12, 29)
    t = time.time()
    res = run_backtest(LongSPYIntraday(), start, end, data.SPY_1MIN, costs=FREE, capital=1_000_000)
    secs = time.time() - t
    ref = independent(start, end)
    diff = (res.returns - ref).abs()
    print(f"{len(res.daily)} sessions in {secs:.1f}s; max |engine - independent| daily return = {diff.max():.2e}")
    print("(differences come only from whole-share rounding)")
    print("engine, zero cost :", {k: round(v, 4) for k, v in metrics.summary(res.returns).items()})
    for mult in (1, 2, 3):
        r = run_backtest(LongSPYIntraday(), start, end, data.SPY_1MIN, costs=CostModel().stressed(mult), capital=1_000_000)
        drag = (res.returns - r.returns).mean() * 1e4
        print(f"costs x{mult}: mean daily drag {drag:.2f} bps, sharpe {metrics.sharpe(r.returns):.2f}, "
              f"fills {len(r.fills)}, flatten fill times {sorted(set(r.fills[r.fills.purpose=='flatten'].ts.dt.tz_convert('America/New_York').dt.strftime('%H:%M')))}")


if __name__ == "__main__":
    main()
