"""Phase 4 parameter grids (development period, measured costs, fixed capital per day).

  python -m scripts.phase4_grid noise     # band_mult x lookback, band_mult x decision step
  python -m scripts.phase4_grid orb xs    # small grids so the rejected strategies get a fair test

Every grid point is a recorded trial. Daily returns go to research/phase4/grid/<name>.parquet.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from intraday import data, research
from intraday.backtest import HistoricalMarketData, run_backtest
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.universe import FEATURES_FILE

os.environ.setdefault("COST_PROFILE", "measured")
from scripts import phase2_run as P2  # noqa: E402  (builds the measured cost models)

OUT = ROOT / "research" / "phase4" / "grid"

NOISE_BAND = [0.5, 0.75, 1.0, 1.25, 1.5]
NOISE_LOOKBACK = [7, 14, 30, 60, 90]
NOISE_STEP = [15, 30, 60]


def grid(name: str) -> list[dict]:
    if name == "noise":
        pts = [{"band_mult": b, "lookback": lb, "step": 30} for b in NOISE_BAND for lb in NOISE_LOOKBACK]
        pts += [{"band_mult": b, "lookback": 14, "step": st} for b in NOISE_BAND for st in NOISE_STEP if st != 30]
        return [{"strategy": "noise_area", **p} for p in pts]
    if name == "orb":
        return [{"strategy": "orb", "stop_atr_frac": f, "top_n": n, "selection_feed": "iex", "same_bar": "next_bar"}
                for f in (0.05, 0.10, 0.25, 0.50) for n in (10, 20)]
    if name == "xs":
        return [{"strategy": "xs", "signal": "rev1", "frac": fr, "xs_n": n} for fr in (0.1, 0.2) for n in (100, 200)]
    raise ValueError(name)


def key(p: dict) -> str:
    return "__".join(f"{k}={v}" for k, v in p.items())


def run_point(p: dict) -> dict:
    out = OUT / f"{key(p)}.parquet"
    if out.exists():
        return {**p, "cached": True}
    from intraday.strategies.noise_area import NoiseAreaMomentum
    from intraday.strategies.orb import ORBStocksInPlay
    from intraday.strategies.xs import CrossSectional
    s = p["strategy"]
    if s == "noise_area":
        feats = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")])
        strat = NoiseAreaMomentum(band_mult=p["band_mult"], lookback=p["lookback"], step=p["step"])
        kw = dict(spec=data.SPY_1MIN, costs=P2.SPY_COSTS, start=DEV_START, bar_minutes=1, same_bar="worst")
    elif s == "orb":
        feats = pd.read_parquet(FEATURES_FILE)
        strat = ORBStocksInPlay(stop_atr_frac=p["stop_atr_frac"], top_n=p["top_n"], selection_feed=p["selection_feed"],
                                name="orb_iex")
        start = [x for x in P2.variants()["orb"] if x.label == "iex"][0].start
        kw = dict(spec=data.SPY_1MIN, costs=P2.ORB_IEX_COSTS, start=start, bar_minutes=1, same_bar=p["same_bar"])
    else:
        feats = pd.read_parquet(FEATURES_FILE)
        strat = CrossSectional(signal=p["signal"], frac=p["frac"], name=f"xs_{p['signal']}")
        kw = dict(spec=P2.XS_5MIN, costs=P2.XS_COSTS, start=DEV_START, bar_minutes=5, same_bar="worst")
    md = HistoricalMarketData(features=feats)
    if s == "xs":
        md.xs_n = p["xs_n"]
    res = run_backtest(strat, kw["start"], DEV_END, kw["spec"], costs=kw["costs"], capital=100_000.0, md=md,
                       bar_minutes=kw["bar_minutes"], same_bar_stop=kw["same_bar"], reset_capital_daily=True)
    r = res.returns
    OUT.mkdir(parents=True, exist_ok=True)
    r.rename("ret").to_frame().to_parquet(out)
    spy = feats[feats.symbol == "SPY"].set_index("date")["ret1"]
    research.record_trial(s, {k: v for k, v in p.items() if k != "strategy"} | {"phase": "4_grid"},
                          (kw["start"], DEV_END), "measured_x1", r, spy=spy)
    return {**p, "cached": False}


def load(name: str) -> dict[str, pd.Series]:
    out = {}
    for p in grid(name):
        f = OUT / f"{key(p)}.parquet"
        if f.exists():
            out[key(p)] = pd.read_parquet(f)["ret"]
    return out


def main(names: list[str]):
    pts = [p for n in names for p in grid(n)]
    workers = int(os.environ.get("P4_WORKERS", "3"))
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_point, pts):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
