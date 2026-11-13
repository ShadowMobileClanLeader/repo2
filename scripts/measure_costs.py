"""Measure trading costs from historical SIP quotes (development period only).

For a stratified sample of sessions (6 per year, fixed seed) and the times each strategy actually
trades, take all NBBO quotes in a short window right after that time (when our next-bar-open
order would execute) and record, per (symbol, time):
  half_spread_bps   median (ask - bid) / (2 * mid) in the window      -> cost of crossing
  touch_shares      median displayed size on the side we would hit     -> does our order fit?
Writes research/phase3/measured_costs.json and .parquet.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from intraday import calendar, orb_data, universe
from intraday.alpaca_client import AlpacaClient
from intraday.config import DEV_END, DEV_START, ROOT

OUT = ROOT / "research" / "phase3"
SPY_TIMES = ["10:00", "11:00", "12:30", "14:00", "15:30", "15:54"]
XS_TIMES = ["09:35", "15:30", "15:50"]
ORB_TIMES = ["09:35", "09:40", "10:00", "11:00", "15:54"]


def sample_sessions(per_year: int = 6, seed: int = 11) -> list:
    rng = np.random.default_rng(seed)
    sess = [s for s in calendar.sessions(DEV_START, DEV_END) if not s.is_early_close]
    out = []
    for y in sorted({s.date.year for s in sess}):
        ys = [s for s in sess if s.date.year == y]
        out += list(rng.choice(ys, min(per_year, len(ys)), replace=False))
    return sorted(out, key=lambda s: s.date)


def quotes(client: AlpacaClient, symbols: list[str], start: pd.Timestamp, seconds: int) -> pd.DataFrame:
    rows, token = [], None
    end = start + pd.Timedelta(seconds=seconds)
    while True:
        p = {"symbols": ",".join(sorted(set(symbols))), "start": start.isoformat(), "end": end.isoformat(),
             "feed": "sip", "limit": 10000}
        if token:
            p["page_token"] = token
        j = client.data_get("/v2/stocks/quotes", **p)
        for sym, qs in (j.get("quotes") or {}).items():
            for q in qs:
                rows.append((sym, q["bp"], q["ap"], q["bs"], q["as"]))
        token = j.get("next_page_token")
        if not token:
            break
    return pd.DataFrame(rows, columns=["symbol", "bp", "ap", "bs", "as"])


def summarise(q: pd.DataFrame) -> pd.DataFrame:
    q = q[(q.bp > 0) & (q.ap > q.bp)]
    mid = (q.ap + q.bp) / 2
    q = q.assign(hs=(q.ap - q.bp) / (2 * mid) * 1e4, mid=mid)
    q = q[q.hs < 500]                                   # drop obviously broken quotes (>5% half-spread)
    return q.groupby("symbol").agg(half_spread_bps=("hs", "median"), mid=("mid", "median"),
                                   bid_size=("bs", "median"), ask_size=("as", "median"), n_quotes=("hs", "size"))


def measure_one(client, category, session, hhmm, symbols, seconds) -> pd.DataFrame:
    t = pd.Timestamp(f"{session.date} {hhmm}").tz_localize("America/New_York").tz_convert("UTC")
    s = summarise(quotes(client, symbols, t, seconds))
    return s.reset_index().assign(category=category, date=session.date, time=hhmm)


def main():
    client = AlpacaClient()
    feats = pd.read_parquet(universe.FEATURES_FILE, columns=["symbol", "date", "close", "adv14", "atr14",
                                                              "dollar_vol20", "n_hist"])
    alld = sorted(feats["date"].unique())
    prev_of = {alld[i]: alld[i - 1] for i in range(1, len(alld))}
    by_date = {d: g for d, g in feats.groupby("date")}
    monthly = universe.monthly_top_n(feats, alld, 200)
    tables = {f: pd.read_parquet(orb_data.table_path(f)) for f in ("sip", "iex")}
    tables = {f: {d: g for d, g in t.groupby("date")} for f, t in tables.items()}
    jobs = []
    for s in sample_sessions():
        for hh in SPY_TIMES:
            jobs.append(("spy", s, hh, ["SPY"], 2))
        xs = universe.universe_on(monthly, s.date)
        for hh in XS_TIMES:
            jobs.append(("xs_top200", s, hh, xs, 1))
        fp = by_date.get(prev_of.get(s.date))
        elig = set(universe.orb_eligible(fp)) if fp is not None else set()
        for feed in ("sip", "iex"):
            day = tables[feed].get(s.date)
            if day is None or not elig:
                continue
            sel = list(orb_data.select_stocks_in_play(day[day.symbol.isin(elig)], 20)["symbol"])
            for hh in ORB_TIMES:
                jobs.append((f"orb_{feed}", s, hh, sel, 2))
    with ThreadPoolExecutor(6) as ex:
        frames = list(ex.map(lambda j: measure_one(client, *j), jobs))
    df = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "measured_spreads.parquet", index=False)

    df["year"] = [d.year for d in df["date"]]
    summary = {}
    for cat, g in df.groupby("category"):
        summary[cat] = {
            "observations": int(len(g)), "days": int(g["date"].nunique()),
            "half_spread_bps_mean": round(float(g.half_spread_bps.mean()), 3),
            "half_spread_bps_median": round(float(g.half_spread_bps.median()), 3),
            "half_spread_bps_p90": round(float(g.half_spread_bps.quantile(0.9)), 3),
            "half_spread_usd_mean": round(float((g.half_spread_bps * g["mid"] / 1e4).mean()), 5),
            "by_time_mean_bps": g.groupby("time").half_spread_bps.mean().round(3).to_dict(),
            "by_year_mean_bps": {int(k): round(float(v), 3) for k, v in g.groupby("year").half_spread_bps.mean().items()},
            "median_touch_shares": float(np.median(np.r_[g.bid_size, g.ask_size])),
        }
    (OUT / "measured_costs.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
