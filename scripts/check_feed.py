"""Phase 1: which feed does this account have, and how does IEX vs SIP affect volume signals?

Uses development-period days only (never the holdout). Writes research/phase1_feed_report.json.
"""
from __future__ import annotations

import datetime as dt
import json
import warnings

import numpy as np
import pandas as pd
import requests

from intraday import calendar, data, universe
from intraday.alpaca_client import AlpacaClient
from intraday.config import ROOT

IEX_1MIN = data.IntradaySpec("iex", "1Min", "raw", "rth")
IEX_OPEN5 = data.IntradaySpec("iex", "5Min", "split", "open5")


def entitlement(client: AlpacaClient) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    out = {}
    for feed in ("iex", "sip"):
        for label, start in (("realtime_last_10min", now - dt.timedelta(minutes=10)),
                             ("historical_2024", dt.datetime(2024, 3, 4, 14, 30, tzinfo=dt.timezone.utc))):
            try:
                client.bars(["SPY"], "1Min", start.isoformat(), (start + dt.timedelta(minutes=9)).isoformat(), feed)
                out[f"{feed}_{label}"] = "allowed"
            except requests.HTTPError as e:
                out[f"{feed}_{label}"] = f"denied: {str(e)[:90]}"
    return out


def spy_comparison(client, days: list[dt.date]) -> dict:
    sess = {s.date: s for s in calendar.sessions(min(days), max(days))}
    data.fetch_intraday(client, IEX_1MIN, {d: ["SPY"] for d in days}, workers=8, progress=False)
    data.fetch_intraday(client, data.SPY_1MIN, {d: ["SPY"] for d in days}, workers=8, progress=False)
    rows = []
    for d in days:
        s = sess[d]
        sip = data.load_intraday(data.SPY_1MIN, d).set_index("ts")
        iex = data.load_intraday(IEX_1MIN, d).set_index("ts")
        j = sip.join(iex, rsuffix="_iex", how="left")
        r = {"date": d, "iex_vol_share": iex["volume"].sum() / sip["volume"].sum(),
             "iex_missing_minutes": float(j["close_iex"].isna().mean()),
             "minute_vol_corr_log": float(np.corrcoef(np.log1p(j["volume"]), np.log1p(j["volume_iex"].fillna(0)))[0, 1]),
             "close_diff_bps_median": float((j["close_iex"] / j["close"] - 1).abs().median() * 1e4)}
        for hhmm in ("10:00", "12:00", "15:30"):
            cut = pd.Timestamp(f"{d} {hhmm}").tz_localize("America/New_York").tz_convert("UTC")
            a, b = sip[sip.index < cut], iex[iex.index < cut]
            vs = (a["vwap"] * a["volume"]).sum() / a["volume"].sum()
            vi = (b["vwap"] * b["volume"]).sum() / b["volume"].sum()
            r[f"vwap_diff_bps_{hhmm}"] = abs(vi / vs - 1) * 1e4
        rows.append(r)
    df = pd.DataFrame(rows)
    return {"days": len(df), **{k: float(df[k].median()) for k in df.columns if k != "date"},
            "vwap_diff_bps_15:30_p95": float(df["vwap_diff_bps_15:30"].quantile(0.95))}


def relvol_comparison(client, blocks: list[tuple[dt.date, dt.date]], top_n: int = 20) -> dict:
    feats = pd.read_parquet(universe.FEATURES_FILE)
    by_date = {d: g for d, g in feats.groupby("date")}
    results = []
    for start, end in blocks:
        sess = calendar.sessions(start, end)
        dates = [s.date for s in sess]
        prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
        eval_days = dates[15:]
        elig = {d: universe.orb_eligible(by_date[prev[d]]) for d in eval_days}
        allsyms = sorted(set().union(*elig.values()))
        need = {d: allsyms for d in dates[1:]}
        for spec in (IEX_OPEN5, data.OPEN5):
            data.fetch_intraday(client, spec, need, batch=1000, workers=12, progress=False)
        vol = {}
        for spec in (IEX_OPEN5, data.OPEN5):
            frames = [data.load_intraday(spec, d, allsyms).assign(date=d) for d in dates[1:]]
            v = pd.concat(frames).pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")
            vol[spec.feed] = v.reindex(index=dates[1:], columns=allsyms).fillna(0.0)
        for d in eval_days:
            i = dates.index(d)
            hist = dates[i - 14:i]
            syms = elig[d]
            rv = {}
            for feed, v in vol.items():
                avg = v.loc[hist, syms].mean()
                rv[feed] = (v.loc[d, syms] / avg.replace(0, np.nan)).dropna()
            both = pd.concat(rv, axis=1).dropna()
            top_sip = set(both["sip"].nlargest(top_n).index)
            top_iex = set(both["iex"].nlargest(top_n).index)
            share = (vol["iex"].loc[d, syms].sum() / vol["sip"].loc[d, syms].sum())
            results.append({"date": d, "n_eligible": len(syms), "n_compared": len(both),
                            "spearman": both["sip"].rank().corr(both["iex"].rank()),
                            f"top{top_n}_overlap": len(top_sip & top_iex) / top_n,
                            "iex_open5_vol_share": share,
                            "iex_zero_open_volume_frac": float((vol["iex"].loc[d, syms] == 0).mean())})
    df = pd.DataFrame(results)
    return {"eval_days": len(df), "median_eligible": float(df["n_eligible"].median()),
            "spearman_median": float(df["spearman"].median()),
            f"top{top_n}_overlap_median": float(df[f"top{top_n}_overlap"].median()),
            f"top{top_n}_overlap_min": float(df[f"top{top_n}_overlap"].min()),
            "iex_open5_vol_share_median": float(df["iex_open5_vol_share"].median()),
            "iex_zero_open_volume_frac_median": float(df["iex_zero_open_volume_frac"].median())}


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    client = AlpacaClient()
    acct = client.account()
    report = {"account": {k: acct.get(k) for k in ("status", "multiplier", "shorting_enabled", "pattern_day_trader")},
              "entitlement": entitlement(client)}
    rng = np.random.default_rng(7)
    # Alpaca IEX history starts 2020-07-27, so samples come from 2020-09 onward (still dev period)
    dev_days = [s.date for s in calendar.sessions(dt.date(2020, 9, 1), dt.date(2025, 6, 30))]
    report["spy_1min_iex_vs_sip"] = spy_comparison(client, sorted(rng.choice(dev_days, 40, replace=False).tolist()))
    print(json.dumps(report, indent=1, default=str), flush=True)
    report["orb_relative_volume_iex_vs_sip"] = relvol_comparison(
        client, [(dt.date(2021, 9, 1), dt.date(2021, 10, 19)), (dt.date(2023, 3, 1), dt.date(2023, 4, 18)),
         (dt.date(2025, 2, 3), dt.date(2025, 3, 21))])
    out = ROOT / "research" / "phase1_feed_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
