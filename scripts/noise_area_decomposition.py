"""Where do the published noise-area returns go? Decomposition over the development period.

  A  independent vectorized implementation of the authors' code, exit 16:00, zero cost
  B  same, but flat by 15:54 (our hard 15:55 rule)
  C  our event-driven engine, zero cost (adds next-bar-open fills)
  D  engine with the paper's cost level (~$0.0045/share all-in -> ~0.25 bps slippage + $0.0035 commission)
  E  engine with our agreed costs (1.5 bps SPY slippage + $0.0035/share + SEC/TAF)
Diagnostic only: the rules are the paper defaults; nothing is tuned.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from intraday import calendar, data, metrics, research
from intraday.backtest import HistoricalMarketData, run_backtest
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.costs import CostModel
from intraday.strategies.noise_area import NoiseAreaMomentum
from intraday.universe import FEATURES_FILE

FREE = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0, min_commission=0,
                 sec_fee_rate=0, taf_per_share=0)
PAPER = CostModel(symbol_slippage_bps={"SPY": 0.25}, commission_per_share=0.0035, min_commission=0.35,
                  sec_fee_rate=0, taf_per_share=0)


def vectorized(start: dt.date, end: dt.date) -> pd.DataFrame:
    f = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]).set_index("date")
    sess = calendar.sessions(start - dt.timedelta(days=40), end)
    days = {}
    for s in sess:
        df = data.load_intraday(data.SPY_1MIN, s.date)
        o, c = s.open.tz_convert("UTC"), s.close.tz_convert("UTC")
        df = df[(df.ts >= o) & (df.ts < c)].reset_index(drop=True)
        df["m"] = ((df.ts - o) / pd.Timedelta(minutes=1)).astype(int) + 1
        days[s.date] = df.set_index("m")
    dates = [s.date for s in sess]
    rows = {}
    for i, s in enumerate(sess):
        if s.date < start or i < 14:
            continue
        df = days[s.date]
        mv = pd.concat([(days[d]["close"] / days[d]["open"].iloc[0] - 1).abs() for d in dates[i - 14:i]], axis=1).mean(axis=1)
        fp = f.loc[:dates[i - 1]]
        pc = fp["close"].iloc[-1]
        lev = min(4, 0.02 / fp["ret1"].iloc[-14:].std(ddof=1))
        op = df["open"].iloc[0]
        vw = (df["vwap"] * df["volume"]).cumsum() / df["volume"].cumsum()
        sig_mv = mv.reindex(df.index)
        ub, lb = max(op, pc) * (1 + sig_mv), min(op, pc) * (1 - sig_mv)
        sig = pd.Series(np.nan, index=df.index)
        for m in df.index[(df.index >= 30) & ((df.index - 30) % 30 == 0)]:
            cl = df.at[m, "close"]
            sig[m] = 1 if (cl > ub[m] and cl > vw[m]) else (-1 if (cl < lb[m] and cl < vw[m]) else 0)
        pos = sig.ffill().fillna(0)
        dp = df["close"].diff().shift(-1).fillna(0)
        shares = lev / op
        last_m = int((s.flatten_time - s.open) / pd.Timedelta(minutes=1)) - 1   # 15:54 on normal days
        rows[s.date] = {"A_vec_1600_free": float((pos * shares * dp).sum()),
                        "B_vec_1554_free": float((pos[pos.index < last_m] * shares * dp[dp.index < last_m]).sum())}
    return pd.DataFrame(rows).T


def engine(costs: CostModel, start, end) -> pd.Series:
    f = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")])
    return run_backtest(NoiseAreaMomentum(), start, end, data.SPY_1MIN, costs=costs,
                        md=HistoricalMarketData(features=f)).returns


def main():
    with ProcessPoolExecutor(3) as ex:
        fv = ex.submit(vectorized, DEV_START, DEV_END)
        fc = ex.submit(engine, FREE, DEV_START, DEV_END)
        fd = ex.submit(engine, PAPER, DEV_START, DEV_END)
        vec, c, d = fv.result(), fc.result(), fd.result()
    e = pd.read_parquet(ROOT / "research/phase2/noise_area__paper__x1__daily.parquet")["ret"]
    df = vec.join(pd.DataFrame({"C_engine_free": c, "D_engine_paper_costs": d, "E_engine_our_costs": e}), how="inner")
    spy = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]).set_index("date")["ret1"]
    research.record_trial("noise_area", {"variant": "paper", "diagnostic": "zero_cost"}, (DEV_START, DEV_END), "x0", c, spy=spy)
    research.record_trial("noise_area", {"variant": "paper", "diagnostic": "paper_cost_level"}, (DEV_START, DEV_END), "paper_costs", d, spy=spy)
    summ = {k: {m: round(v, 4) for m, v in metrics.summary(df[k]).items()} for k in df}
    years = df.groupby(pd.Index([x.year for x in df.index])).apply(lambda g: (1 + g).prod() - 1).round(3)
    out = ROOT / "research" / "phase2" / "noise_area_decomposition.json"
    out.write_text(json.dumps({"summary": summ, "by_year": years.to_dict()}, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summ).T[["cagr", "vol", "sharpe", "max_dd"]].to_string())
    print(years.to_string())


if __name__ == "__main__":
    main()
