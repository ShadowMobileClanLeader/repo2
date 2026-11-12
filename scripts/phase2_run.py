"""Phase 2: run every PRE-REGISTERED variant on the development period at 1x/2x/3x slippage.

  python -m scripts.phase2_run noise_area vwap_mr orb xs

Every run is appended to research/trials.jsonl (the multiple-testing count for Phase 4).
No parameter is tuned here: variants are the paper defaults plus the ablations agreed up front.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from intraday import calendar, data, metrics, research
from intraday.backtest import HistoricalMarketData, run_backtest
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.costs import CostModel
from intraday.engine import RiskLimits
from intraday.universe import FEATURES_FILE

OUT = ROOT / "research" / "phase2"
IEX_START = dt.date(2020, 7, 27)
IEX_1MIN = data.IntradaySpec("iex", "1Min", "raw", "rth")
XS_5MIN = data.IntradaySpec("sip", "5Min", "raw", "rth")

# Papers' own samples: results inside them are replication, not out-of-sample evidence.
PAPER_SAMPLE_END = {"noise_area": dt.date(2024, 4, 30), "orb": dt.date(2023, 12, 31)}

SPY_COSTS = CostModel()                                    # SPY 1.5 bps
ORB_COSTS = CostModel(default_slippage_bps=10.0)           # stocks in play: 10 bps (agreed)
XS_COSTS = CostModel(default_slippage_bps=5.0)             # liquid large caps: 5 bps


@dataclass
class Variant:
    strategy: str
    label: str
    params: dict
    spec: data.IntradaySpec
    costs: CostModel
    start: dt.date = DEV_START
    bar_minutes: int = 1
    vwap_filter: bool = False
    features: str = "spy"          # "spy" (SPY rows only) or "all"
    same_bar_stop: str = "worst"   # fill-model bound for stops inside the entry bar
    risk: RiskLimits = field(default_factory=RiskLimits)


def variants() -> dict[str, list[Variant]]:
    # IEX-based variants need 14 sessions of IEX history before the first decision
    iex_ok = calendar.sessions(IEX_START, IEX_START + dt.timedelta(days=40))[20].date
    return {
        "noise_area": [
            Variant("noise_area", "paper", {}, data.SPY_1MIN, SPY_COSTS),
            Variant("noise_area", "no_vwap", {"use_vwap": False}, data.SPY_1MIN, SPY_COSTS),
            Variant("noise_area", "iex_signals", {"spec": IEX_1MIN}, IEX_1MIN, SPY_COSTS, start=iex_ok),
        ],
        "vwap_mr": [Variant("vwap_mr", "k0.5", {"k": 0.5}, data.SPY_1MIN, SPY_COSTS)],
        "orb": [
            Variant("orb", "sip", {"selection_feed": "sip"}, data.SPY_1MIN, ORB_COSTS, features="all"),
            Variant("orb", "iex", {"selection_feed": "iex", "name": "orb_iex"}, data.SPY_1MIN, ORB_COSTS,
                    start=iex_ok, features="all"),
            Variant("orb", "sip_delay15", {"delay_minutes": 15, "name": "orb_sip_0950"}, data.SPY_1MIN, ORB_COSTS,
                    features="all"),
            Variant("orb", "sip+vwap_filter", {"selection_feed": "sip"}, data.SPY_1MIN, ORB_COSTS,
                    features="all", vwap_filter=True),
            # optimistic fill bound: a stop inside the entry bar is only checked from the next bar
            Variant("orb", "sip_nextbar_fill", {"selection_feed": "sip"}, data.SPY_1MIN, ORB_COSTS,
                    features="all", same_bar_stop="next_bar"),
            Variant("orb", "iex_nextbar_fill", {"selection_feed": "iex", "name": "orb_iex"}, data.SPY_1MIN,
                    ORB_COSTS, start=iex_ok, features="all", same_bar_stop="next_bar"),
        ],
        "xs": [Variant("xs", sig, {"signal": sig, "name": f"xs_{sig}"}, XS_5MIN, XS_COSTS, bar_minutes=5,
                       features="all")
               for sig in ("rev1", "rev5", "lps_intraday", "lps_overnight", "lps_combo", "eod_reversal")]
              + [Variant("xs", "lps_combo+vwap_filter", {"signal": "lps_combo", "name": "xs_lps_combo"}, XS_5MIN,
                         XS_COSTS, bar_minutes=5, features="all", vwap_filter=True)],
    }


def make_strategy(v: Variant):
    from intraday.strategies.noise_area import NoiseAreaMomentum
    from intraday.strategies.orb import ORBStocksInPlay
    from intraday.strategies.vwap_filter import VWAPFilter
    from intraday.strategies.vwap_mr import VWAPMeanReversion
    from intraday.strategies.xs import CrossSectional
    cls = {"noise_area": NoiseAreaMomentum, "vwap_mr": VWAPMeanReversion, "orb": ORBStocksInPlay,
           "xs": CrossSectional}[v.strategy]
    s = cls(**v.params)
    return VWAPFilter(s) if v.vwap_filter else s


def spy_returns() -> pd.Series:
    f = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")])
    return f.set_index("date")["ret1"]


def run_one(args) -> dict:
    v, mult = args
    feats = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]) if v.features == "spy" \
        else pd.read_parquet(FEATURES_FILE)
    md = HistoricalMarketData(features=feats)
    t = time.time()
    res = run_backtest(make_strategy(v), v.start, DEV_END, v.spec, costs=v.costs.stressed(mult),
                       capital=100_000.0, risk=v.risk, md=md, bar_minutes=v.bar_minutes,
                       same_bar_stop=v.same_bar_stop)
    r = res.returns
    spy = spy_returns()
    costs_label = f"x{mult}"
    params = {**{k: (str(x) if isinstance(x, data.IntradaySpec) else x) for k, x in v.params.items()},
              "variant": v.label, "vwap_filter": v.vwap_filter}
    if v.same_bar_stop != "worst":
        params["same_bar_stop"] = v.same_bar_stop
    extra = {"trades": int((res.fills["purpose"] == "entry").sum()) if len(res.fills) else 0,
             "avg_gross": float((res.daily["max_gross"] / res.daily["sod_equity"]).mean()),
             "cost_drag_bps_day": float(((res.daily["fees"] + res.daily["slippage"]) / res.daily["sod_equity"]).mean() * 1e4),
             "runtime_s": round(time.time() - t, 1)}
    row = research.record_trial(v.strategy, params, (v.start, DEV_END), costs_label, r, extra, spy)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{v.strategy}__{v.label}__{costs_label}".replace("+", "_")
    res.daily.to_parquet(OUT / f"{stem}__daily.parquet")
    if mult == 1 and len(res.fills):
        res.fills.to_parquet(OUT / f"{stem}__fills.parquet")
    split = PAPER_SAMPLE_END.get(v.strategy)
    if split:
        idx = pd.Index(r.index)
        row["in_paper_sample"] = metrics.summary(r[idx <= split], spy)
        row["post_publication"] = metrics.summary(r[idx > split], spy)
    row["by_year"] = {int(y): round(float((1 + g).prod() - 1), 4) for y, g in r.groupby(pd.Index([d.year for d in r.index]))}
    (OUT / f"{stem}__summary.json").write_text(json.dumps(row, indent=1, default=str))
    return {"strategy": v.strategy, "variant": v.label, "costs": costs_label,
            **{k: row[k] for k in ("sharpe", "sortino", "max_dd", "cagr", "beta", "alpha_ann")}, **extra}


def main(names: list[str]) -> None:
    allv = variants()
    jobs = []
    for n in names:                       # "orb" or "orb:sip,iex" to pick variants
        strat, _, labels = n.partition(":")
        wanted = set(labels.split(",")) if labels else None
        jobs += [(v, m) for v in allv[strat] if wanted is None or v.label in wanted for m in (1, 2, 3)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("P2_WORKERS", "3"))) as ex:
        rows = list(ex.map(run_one, jobs))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
