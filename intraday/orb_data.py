"""Opening-range volume tables for the Stocks-in-Play ORB (built from cached 5-minute open bars).

Relative volume (Zarattini, Barbon, Aziz 2024, §3):
    RelVol(t, j) = ORVolume(t, j) / mean(ORVolume(t-i, j), i = 1..14)
where ORVolume is the volume of the 09:30-09:35 bar. A symbol that was requested but has no
bar on a day counts as zero volume (common on IEX); a day where it was never requested makes
the 14-day average undefined (NaN), so the symbol is not rankable that day.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import calendar, data
from .config import CACHE_DIR, DATA_START, HOLDOUT_END

LOOKBACK = 14


def table_path(feed: str):
    return CACHE_DIR / "features" / f"open5_{feed}.parquet"


def open5_spec(feed: str) -> data.IntradaySpec:
    return data.IntradaySpec(feed, "5Min", "split", "open5")


def compute_relvol(vol: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """vol: dates x symbols opening volume (NaN = not requested). Returns trailing average of the
    PREVIOUS `lookback` days (all must be present) and today's relative volume."""
    avg = vol.shift(1).rolling(lookback, min_periods=lookback).mean()
    rel = vol / avg.where(avg > 0)
    return avg, rel


def build_table(feed: str) -> pd.DataFrame:
    spec = open5_spec(feed)
    rows, requested = [], {}
    for s in calendar.sessions(DATA_START, HOLDOUT_END):
        pq, man = data._intraday_paths(spec, s.date)
        if not man.exists():
            continue
        requested[s.date] = json.loads(man.read_text())
        if pq.exists():
            df = pd.read_parquet(pq)
            df["date"] = s.date
            rows.append(df[["date", "symbol", "open", "high", "low", "close", "volume"]])
    bars = pd.concat(rows, ignore_index=True)
    bars["symbol"] = bars["symbol"].astype(str)
    dates = sorted(requested)
    syms = sorted(set().union(*map(set, requested.values())))
    vol = pd.DataFrame(np.nan, index=dates, columns=syms)
    for d, ss in requested.items():
        vol.loc[d, ss] = 0.0
    piv = bars.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")
    vol.update(piv)
    avg, rel = compute_relvol(vol)
    out = bars.merge(
        pd.DataFrame({"avg14": avg.stack(), "relvol": rel.stack()}).rename_axis(["date", "symbol"]).reset_index(),
        on=["date", "symbol"], how="left")
    out.to_parquet(table_path(feed), index=False)
    return out


def select_stocks_in_play(snap: pd.DataFrame, top_n: int = 20, min_relvol: float = 1.0,
                          min_open: float = 5.0) -> pd.DataFrame:
    """snap: today's opening bars with relvol, already restricted to prior-day-eligible symbols.
    Returns the top_n by relative volume with a direction (+1 bullish candle, -1 bearish, doji dropped)."""
    s = snap[(snap["relvol"] >= min_relvol) & (snap["open"] > min_open)].copy()
    s["direction"] = np.sign(s["close"] - s["open"])
    s = s[s["direction"] != 0]
    return s.sort_values(["relvol", "symbol"], ascending=[False, True]).head(top_n)
