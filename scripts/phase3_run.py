"""Phase 3: portfolio + risk layer on the sleeves' net daily returns (measured costs).

Sleeves (fixed before seeing measured-cost results; the VWAP strategy is a control, not a sleeve):
  noise_area   SPY noise-area momentum, paper rules (SIP signals; IEX signals in the live-realistic case)
  orb_iex      ORB stocks-in-play with IEX selection (judged on IEX as agreed); both fill bounds
  xs_rev1      cross-sectional 1-day reversal, top-200 point-in-time universe
Scenarios: all candidates (worst / optimistic ORB fills), live-realistic, and "eligible only"
(rule stated in advance: dev net Sharpe > 0 at measured costs AND >= 0 at 2x measured costs).
Every scenario is also run on the 2x and 3x cost sleeves.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from intraday import metrics
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.portfolio import PortfolioConfig, contributions, run_portfolio
from intraday.universe import FEATURES_FILE

P2 = ROOT / "research" / "phase2"
OUT = ROOT / "research" / "phase3"

SLEEVE_FILES = {
    "noise_area": "noise_area__paper",
    "noise_area_iex": "noise_area__iex_signals",
    "orb_iex": "orb__iex",
    "orb_iex_optimistic": "orb__iex_nextbar_fill",
    "xs_rev1": "xs__rev1",
}
SCENARIOS = {
    "all_candidates": ["noise_area", "orb_iex", "xs_rev1"],
    "all_candidates_orb_optimistic": ["noise_area", "orb_iex_optimistic", "xs_rev1"],
    "live_realistic": ["noise_area_iex", "orb_iex", "xs_rev1"],
}
CFG = PortfolioConfig(
    weight_lookback=120, min_history=60, corr_penalty=True, target_vol=0.10, vol_lookback=60, max_scale=3.0,
    max_gross=4.0, max_stock_weight=0.10, daily_loss_limit=0.03, kill_drawdown=0.12,
    # worst-case exposure per unit of sleeve capital (from each sleeve's own limits)
    sleeve_max_gross={"noise_area": 4.0, "noise_area_iex": 4.0, "orb_iex": 4.0, "orb_iex_optimistic": 4.0,
                      "xs_rev1": 1.0},
    # single-stock exposure per unit of sleeve capital; SPY is an index ETF (bounded by the gross cap)
    sleeve_max_stock_weight={"noise_area": 0.0, "noise_area_iex": 0.0, "orb_iex": 0.20,
                             "orb_iex_optimistic": 0.20, "xs_rev1": 0.03},
)


def sleeve_returns(cost: str) -> pd.DataFrame:
    cols = {}
    for name, stem in SLEEVE_FILES.items():
        p = P2 / f"{stem}__measured_{cost}__daily.parquet"
        cols[name] = pd.read_parquet(p)["ret"]
    R = pd.DataFrame(cols)
    R.index = pd.Index(R.index)
    return R.sort_index()


def eligible(cost_x1: pd.DataFrame, cost_x2: pd.DataFrame) -> list[str]:
    base = ["noise_area", "orb_iex", "xs_rev1"]
    return [s for s in base if metrics.sharpe(cost_x1[s].dropna()) > 0 and metrics.sharpe(cost_x2[s].dropna()) >= 0]


def evaluate(R: pd.DataFrame, spy: pd.Series) -> dict:
    res = run_portfolio(R, CFG)
    r = res.returns
    first = res.scale[res.scale > 0].index.min()
    r = r[r.index >= first] if first is not None else r
    s = metrics.summary(r, spy)
    bh = metrics.summary(spy[(spy.index >= r.index.min()) & (spy.index <= r.index.max())])
    gross_used = sum(res.allocations[c] * CFG.sleeve_max_gross.get(c, 1.0) for c in R.columns)
    return {"start": str(r.index.min()), "end": str(r.index.max()), **{k: round(v, 4) for k, v in s.items()},
            "killed_on": str(res.killed_on) if res.killed_on is not None else None,
            "days_loss_gt_limit": int((r < -CFG.daily_loss_limit).sum()),
            "avg_scale": round(float(res.scale[res.scale > 0].mean()), 3),
            "max_worstcase_gross": round(float(gross_used.max()), 3),
            "spy_buy_hold_same_window": {k: round(v, 4) for k, v in bh.items()},
            "by_year": {int(y): round(float((1 + g).prod() - 1), 4) for y, g in r.groupby(pd.Index([d.year for d in r.index]))}}


def main():
    spy = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]).set_index("date")["ret1"]
    R = {c: sleeve_returns(c) for c in ("x1", "x2", "x3")}
    scenarios = dict(SCENARIOS)
    elig = eligible(R["x1"], R["x2"])
    scenarios["eligible_only"] = elig
    out = {"config": {k: v for k, v in CFG.__dict__.items()}, "eligible_rule": "dev net Sharpe > 0 at measured x1 and >= 0 at x2",
           "eligible": elig, "sleeves": {}, "correlation_x1": R["x1"][["noise_area", "orb_iex", "xs_rev1"]].corr().round(3).to_dict(),
           "scenarios": {}}
    for name in SLEEVE_FILES:
        out["sleeves"][name] = {c: {k: round(v, 4) for k, v in metrics.summary(R[c][name].dropna(), spy).items()}
                                for c in ("x1", "x2", "x3")}
    for sc, cols in scenarios.items():
        if not cols:
            out["scenarios"][sc] = {"note": "no sleeve passes the eligibility rule"}
            continue
        out["scenarios"][sc] = {c: evaluate(R[c][cols], spy) for c in ("x1", "x2", "x3")}
        contrib = contributions(R["x1"][cols], CFG, spy)
        out["scenarios"][sc]["contributions_x1"] = contrib.round(4).to_dict("index")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase3_results.json").write_text(json.dumps(out, indent=1, default=str))

    pd.set_option("display.width", 220)
    print("eligible:", elig)
    print(pd.DataFrame({n: v["x1"] for n, v in out["sleeves"].items()}).T[["sharpe", "sortino", "max_dd", "cagr", "beta", "alpha_ann"]].round(3).to_string())
    for sc, v in out["scenarios"].items():
        if "x1" not in v:
            print(sc, v)
            continue
        print(f"\n== {sc} ({', '.join(scenarios[sc])})")
        print(pd.DataFrame({c: {k: v[c][k] for k in ("start", "cagr", "vol", "sharpe", "sortino", "max_dd", "beta",
                                                      "alpha_ann", "killed_on", "days_loss_gt_limit", "avg_scale")}
                            for c in ("x1", "x2", "x3")}).to_string())
        print(pd.DataFrame(v["contributions_x1"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
