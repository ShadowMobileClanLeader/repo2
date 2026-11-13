"""ORB diagnostics: does the gross edge replicate, and what do costs / fill assumptions do to it?

Runs the paper-default ORB (SIP and IEX selection) under
  zero cost | paper cost level (commission only, no slippage) | agreed costs (10 bps)
crossed with the two same-bar stop bounds (worst / next_bar), and measures the average
R-multiple per trade (R = 1% of the capital slice, the paper's risk unit).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from intraday import data, metrics, research
from intraday.backtest import HistoricalMarketData, run_backtest
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.costs import CostModel
from intraday.strategies.orb import ORBStocksInPlay
from intraday.universe import FEATURES_FILE

COSTS = {
    "zero": CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0, min_commission=0,
                      sec_fee_rate=0, taf_per_share=0),
    "paper": CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0.0035,
                       min_commission=0, sec_fee_rate=0, taf_per_share=0),
    "agreed": CostModel(default_slippage_bps=10.0),
}
IEX_START = dt.date(2020, 8, 25)


def trade_r(fills: pd.DataFrame, daily: pd.DataFrame, top_n: int = 20, risk_frac: float = 0.01) -> pd.Series:
    """Per-trade P&L in R units (R = risk_frac * start-of-day equity / top_n), costs included."""
    if fills.empty:
        return pd.Series(dtype=float)
    f = fills.copy()
    f["date"] = f["ts"].dt.tz_convert("America/New_York").dt.date
    f["cash"] = -f["qty"] * f["price"] - f["fees"]
    pnl = f.groupby(["date", "symbol"])["cash"].sum()          # every trade is round-tripped intraday
    r_unit = daily["sod_equity"].reindex(pnl.index.get_level_values(0)).to_numpy() * risk_frac / top_n
    return pnl / r_unit


def one(args) -> dict:
    feed, cost_name, same_bar = args
    f = pd.read_parquet(FEATURES_FILE)
    start = DEV_START if feed == "sip" else IEX_START
    res = run_backtest(ORBStocksInPlay(selection_feed=feed, name=f"orb_{feed}"), start, DEV_END, data.SPY_1MIN,
                       costs=COSTS[cost_name], capital=100_000.0, md=HistoricalMarketData(features=f),
                       same_bar_stop=same_bar)
    r = res.returns
    spy = f[f.symbol == "SPY"].set_index("date")["ret1"]
    research.record_trial("orb", {"selection_feed": feed, "variant": feed, "diagnostic": f"{cost_name}_{same_bar}"},
                          (start, DEV_END), f"diag_{cost_name}", r, spy=spy)
    R = trade_r(res.fills, res.daily)
    paper_end = dt.date(2023, 12, 31)
    idx = pd.Index(r.index)
    return {"feed": feed, "costs": cost_name, "same_bar": same_bar, **metrics.summary(r, spy),
            "trades": int(len(R)), "avg_R": float(R.mean()), "median_R": float(R.median()),
            "win_rate": float((R > 0).mean()),
            "sharpe_in_paper_sample": metrics.sharpe(r[idx <= paper_end]),
            "sharpe_post_pub": metrics.sharpe(r[idx > paper_end]),
            "by_year": {int(y): round(float((1 + g).prod() - 1), 3) for y, g in r.groupby(pd.Index([d.year for d in r.index]))}}


def main(workers: int = 3):
    jobs = [(feed, c, sb) for feed in ("sip", "iex") for c in ("zero", "paper") for sb in ("worst", "next_bar")]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    (ROOT / "research" / "phase2" / "orb_diagnostics.json").write_text(json.dumps(rows, indent=1, default=str))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df[["feed", "costs", "same_bar", "cagr", "vol", "sharpe", "max_dd", "beta", "trades", "avg_R", "win_rate",
              "sharpe_in_paper_sample", "sharpe_post_pub"]].round(3).to_string(index=False))
    for r in rows:
        print(r["feed"], r["costs"], r["same_bar"], r["by_year"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
