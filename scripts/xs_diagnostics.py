"""Cross-sectional diagnostics: gross (zero-cost) edge per signal and break-even cost.

Net results for daily open-to-close long/short books are dominated by costs (two full turnovers
per day), and compounding a losing book shrinks positions into min-commission territory. So we
report the gross edge in bps/day and the one-way slippage that would make it break even, plus an
independent vectorised cross-check of the rev1 decile spread from daily open->close returns.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from intraday import metrics, research
from intraday.backtest import HistoricalMarketData, run_backtest
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.costs import CostModel
from intraday.strategies.xs import CrossSectional
from intraday.universe import FEATURES_FILE, monthly_top_n, universe_on
from scripts.phase2_run import XS_5MIN

FREE = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0, min_commission=0,
                 sec_fee_rate=0, taf_per_share=0)
SIGNALS = ("rev1", "rev5", "lps_intraday", "lps_overnight", "lps_combo", "eod_reversal")


def gross(sig: str) -> dict:
    f = pd.read_parquet(FEATURES_FILE)
    res = run_backtest(CrossSectional(signal=sig, name=f"xs_{sig}"), DEV_START, DEV_END, XS_5MIN, costs=FREE,
                       capital=1e7, md=HistoricalMarketData(features=f), bar_minutes=5)
    r = res.returns
    spy = f[f.symbol == "SPY"].set_index("date")["ret1"]
    research.record_trial("xs", {"signal": sig, "variant": sig, "diagnostic": "zero_cost"}, (DEV_START, DEV_END), "x0", r, spy=spy)
    turnover = (res.daily["traded_notional"] / res.daily["sod_equity"]).mean()      # both legs, in+out
    mean_bps = r.mean() * 1e4
    return {"signal": sig, "gross_bps_day": mean_bps, "gross_sharpe": metrics.sharpe(r),
            "t_stat": r.mean() / r.std(ddof=1) * np.sqrt(len(r)), "turnover_x_equity": turnover,
            "breakeven_one_way_bps": mean_bps / turnover if turnover > 0 else np.nan,
            "by_year_bps": {int(y): round(float(g.mean() * 1e4), 2) for y, g in r.groupby(pd.Index([d.year for d in r.index]))}}


def vectorized_rev1() -> float:
    """Independent check: decile spread of next-day open->close return sorted on prior-day return."""
    f = pd.read_parquet(FEATURES_FILE, columns=["symbol", "date", "close", "dollar_vol20", "n_hist", "ret1", "ret_intraday"])
    dates = sorted(d for d in f["date"].unique() if DEV_START <= d <= DEV_END)
    alld = sorted(f["date"].unique())
    monthly = monthly_top_n(f, alld, 200)
    piv_r1 = f.pivot(index="date", columns="symbol", values="ret1")
    piv_oc = f.pivot(index="date", columns="symbol", values="ret_intraday")
    idx = {d: i for i, d in enumerate(alld)}
    spreads = []
    for d in dates:
        prev = alld[idx[d] - 1]
        u = [s for s in universe_on(monthly, d) if s in piv_r1.columns]
        sig = -piv_r1.loc[prev, u].dropna()
        oc = piv_oc.loc[d, u]
        sig = sig[oc.reindex(sig.index).notna()]
        k = int(len(sig) * 0.1)
        if k == 0:
            continue
        s = sig.sort_values()
        spreads.append(0.5 * (oc[s.index[-k:]].mean() - oc[s.index[:k]].mean()))   # gross 1x: 0.5 long, 0.5 short
    return float(np.mean(spreads) * 1e4)


def main():
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(gross, SIGNALS))
        vec = ex.submit(vectorized_rev1).result()
    df = pd.DataFrame(rows).set_index("signal")
    out = {"gross": df.drop(columns="by_year_bps").round(3).to_dict("index"),
           "by_year_bps": df["by_year_bps"].to_dict(),
           "vectorized_rev1_open_to_close_bps_day": vec}
    (ROOT / "research" / "phase2" / "xs_diagnostics.json").write_text(json.dumps(out, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(df.drop(columns="by_year_bps").round(3).to_string())
    print("vectorized rev1 (official open->close, gross 1x):", round(vec, 2), "bps/day")
    print(pd.DataFrame(df["by_year_bps"].to_dict()).round(1).to_string())


if __name__ == "__main__":
    main()
