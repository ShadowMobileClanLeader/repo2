"""Compile research/phase2/*__summary.json into research/phase2_results.md."""
from __future__ import annotations

import json

import pandas as pd

from intraday import metrics
from intraday.config import DEV_END, DEV_START, ROOT
from intraday.universe import FEATURES_FILE

P2 = ROOT / "research" / "phase2"


def spy_bh(start, end) -> dict:
    f = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]).set_index("date")
    r = f["ret1"]
    r = r[(r.index >= start) & (r.index <= end)]
    return metrics.summary(r)


def main():
    rows = []
    for p in sorted(P2.glob("*__summary.json")):
        j = json.loads(p.read_text())
        strat, variant, costs = p.name.split("__")[:3]
        rows.append({"strategy": strat, "variant": variant, "costs": costs, "start": j["start"],
                     "cagr": j["cagr"], "vol": j["vol"], "sharpe": j["sharpe"], "sortino": j["sortino"],
                     "max_dd": j["max_dd"], "beta": j.get("beta"), "alpha": j.get("alpha_ann"),
                     "trades": j.get("trades"), "cost_bps_day": j.get("cost_drag_bps_day"),
                     "post_pub_sharpe": (j.get("post_publication") or {}).get("sharpe"),
                     "in_paper_sharpe": (j.get("in_paper_sample") or {}).get("sharpe")})
    df = pd.DataFrame(rows).sort_values(["strategy", "variant", "costs"])
    spy = spy_bh(DEV_START, DEV_END)
    lines = ["# Phase 2 results (development period, net of costs)", "",
             f"SPY buy-and-hold {DEV_START} → {DEV_END}: CAGR {spy['cagr']:.1%}, vol {spy['vol']:.1%}, "
             f"Sharpe {spy['sharpe']:.2f}, Sortino {spy['sortino']:.2f}, max DD {spy['max_dd']:.1%}", "",
             df.to_markdown(index=False, floatfmt=".3f")]
    (ROOT / "research" / "phase2_results.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    print("SPY B&H:", {k: round(v, 3) for k, v in spy.items()})


if __name__ == "__main__":
    main()
