"""Populate the local parquet cache.

  python -m scripts.download_data calendar candidates daily features spy open5

Downloading holdout-period data into the cache is allowed (it is not *read* by
research code until the holdout lock is lifted).
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from collections import defaultdict

import pandas as pd

from intraday import calendar, data, universe
from intraday.alpaca_client import AlpacaClient
from intraday.config import DATA_START, HOLDOUT_END


def main(stages: list[str]) -> None:
    client = AlpacaClient()
    t0 = time.time()
    if "calendar" in stages:
        calendar.download_calendar(client)
        print("calendar ok", flush=True)
    if "candidates" in stages:
        c = universe.build_candidates(client)
        print("candidates", len(c), c["source"].str.split(",").explode().value_counts().to_dict(), flush=True)
    if "daily" in stages:
        syms = list(universe.load_candidates()["symbol"]) + ["SPY", "QQQ", "IWM"]
        for adj in ("raw", "all"):
            for y in range(DATA_START.year, HOLDOUT_END.year + 1):
                data.fetch_daily_year(client, y, syms, adj, end=HOLDOUT_END)
                print(f"daily {adj} {y} done ({time.time() - t0:.0f}s)", flush=True)
    if "features" in stages:
        raw = data.load_daily("raw", DATA_START, HOLDOUT_END, guard=False)
        adj = data.load_daily("all", DATA_START, HOLDOUT_END, guard=False)
        f = universe.compute_daily_features(raw, adj)
        universe.FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(universe.FEATURES_FILE, index=False)
        print("features", f.shape, flush=True)
    if "spy" in stages:
        sess = calendar.sessions(DATA_START, HOLDOUT_END)
        data.fetch_intraday(client, data.SPY_1MIN, {s.date: ["SPY"] for s in sess}, workers=6)
        print(f"spy 1min done ({time.time() - t0:.0f}s)", flush=True)
    if "open5" in stages:
        fetch_open5(client)
        print(f"open5 done ({time.time() - t0:.0f}s)", flush=True)
    if "iex_open5" in stages:   # IEX opening volumes, for live-realistic ORB selection
        need = {d: s for d, s in open5_symbols_by_day().items() if d >= IEX_START}
        data.fetch_intraday(client, IEX_OPEN5, need, batch=500, workers=10)
        print(f"iex_open5 done ({time.time() - t0:.0f}s)", flush=True)
    if "spy_iex" in stages:
        sess = calendar.sessions(IEX_START, HOLDOUT_END)
        data.fetch_intraday(client, IEX_1MIN, {s.date: ["SPY"] for s in sess}, workers=4)
        print(f"spy_iex done ({time.time() - t0:.0f}s)", flush=True)
    if "xs5" in stages:         # 5-minute bars for the point-in-time top-200 universe
        need = xs_symbols_by_day()
        data.fetch_intraday(client, XS_5MIN, need, batch=100, workers=6)
        print(f"xs5 done ({time.time() - t0:.0f}s)", flush=True)


IEX_START = dt.date(2020, 7, 27)   # first day of Alpaca IEX history
IEX_OPEN5 = data.IntradaySpec("iex", "5Min", "split", "open5")
IEX_1MIN = data.IntradaySpec("iex", "1Min", "raw", "rth")
XS_5MIN = data.IntradaySpec("sip", "5Min", "raw", "rth")


def xs_symbols_by_day(n: int = 200) -> dict:
    feats = pd.read_parquet(universe.FEATURES_FILE)
    dates = [s.date for s in calendar.sessions(DATA_START, HOLDOUT_END)]
    monthly = universe.monthly_top_n(feats, dates, n)
    return {d: universe.universe_on(monthly, d) for d in dates if universe.universe_on(monthly, d)}


def open5_symbols_by_day(lookback: int = 14) -> dict:
    """For each session D: every symbol ORB-eligible on D or on any of the next `lookback` sessions,
    so both today's opening volume and the trailing 14-day average are cached."""
    feats = pd.read_parquet(universe.FEATURES_FILE)
    sess = calendar.sessions(DATA_START, HOLDOUT_END)
    dates = [s.date for s in sess]
    by_prev = {d: g for d, g in feats.groupby("date")}
    eligible = {}
    for i in range(1, len(dates)):
        fp = by_prev.get(dates[i - 1])
        eligible[dates[i]] = set(universe.orb_eligible(fp)) if fp is not None else set()
    need = defaultdict(set)
    for i, d in enumerate(dates):
        for j in range(i, min(len(dates), i + lookback + 1)):
            need[d] |= eligible.get(dates[j], set())
    return {d: sorted(s) for d, s in need.items() if s}


def fetch_open5(client) -> None:
    need = open5_symbols_by_day()
    sizes = pd.Series({d: len(s) for d, s in need.items()})
    print("open5 symbols/day: median", int(sizes.median()), "max", int(sizes.max()), flush=True)
    data.fetch_intraday(client, data.OPEN5, need, batch=500, workers=10)


if __name__ == "__main__":
    main(sys.argv[1:])
