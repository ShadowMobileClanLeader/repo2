"""ONE-SHOT holdout evaluation of the frozen rules (research/FROZEN_RULES.json).

Refuses to run without the frozen-rules file; unlocking is logged to research/holdout_access.log.
Evaluates exactly the frozen configuration: primary = IEX signals, secondary = SIP signals, at
measured costs x1/x2/x3, sized by the frozen portfolio rules (vol target estimated with
pre-holdout data where the window reaches back). Nothing here may be used to change the rules.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from intraday import data, metrics, splits
from intraday.backtest import HistoricalMarketData, run_backtest
from intraday.config import FROZEN_RULES_FILE, HOLDOUT_END, HOLDOUT_START, ROOT
from intraday.portfolio import PortfolioConfig, run_portfolio
from intraday.strategies.noise_area import NoiseAreaMomentum
from intraday.universe import FEATURES_FILE

OUT = ROOT / "research" / "phase4"
P2 = ROOT / "research" / "phase2"
IEX_1MIN = data.IntradaySpec("iex", "1Min", "raw", "rth")


def main():
    rules = json.loads(FROZEN_RULES_FILE.read_text())
    splits.unlock_holdout("one-shot evaluation of frozen rules " + rules["frozen_at"])
    import os
    os.environ["COST_PROFILE"] = "measured"
    from scripts import phase2_run as P2R
    feats = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")])
    spy = feats.set_index("date")["ret1"]
    sp = rules["strategy_params"]
    pcfg = PortfolioConfig(**{k: v for k, v in rules["portfolio"].items()})
    out = {"rules": rules, "results": {}}
    for feed, spec in (("iex", IEX_1MIN), ("sip", data.SPY_1MIN)):
        for mult in (1, 2, 3):
            strat = NoiseAreaMomentum(band_mult=sp["band_mult"], lookback=sp["lookback"], step=sp["step"],
                                      target_vol=sp["sleeve_target_vol_daily"], max_leverage=sp["sleeve_max_leverage"],
                                      spec=spec)
            res = run_backtest(strat, HOLDOUT_START, HOLDOUT_END, spec, costs=P2R.SPY_COSTS.stressed(mult),
                               capital=100_000.0, md=HistoricalMarketData(features=feats), reset_capital_daily=True)
            hold = res.returns
            # pre-holdout history of the same sleeve (dev run) so the vol-target window is warm on day 1
            label = "iex_signals" if feed == "iex" else "paper"
            pre = pd.read_parquet(P2 / f"noise_area__{label}__measured_x{mult}__daily.parquet")["ret"]
            R = pd.DataFrame({"noise_area": pd.concat([pre, hold])})
            pr = run_portfolio(R, replace(pcfg, kill_drawdown=float("inf")))
            p = pr.returns[pd.Index(pr.returns.index) >= HOLDOUT_START]
            kill = run_portfolio(R[R.index >= HOLDOUT_START], pcfg).killed_on   # live behaviour from day 1
            out["results"][f"{feed}_x{mult}"] = {
                "sleeve": metrics.summary(hold, spy), "portfolio": metrics.summary(p, spy),
                "portfolio_kill_switch_tripped_on": str(kill) if kill is not None else None,
                "monthly": {str(k): round(float(v), 4) for k, v in
                            p.groupby(pd.Index([f"{d.year}-{d.month:02d}" for d in p.index])).apply(lambda g: (1 + g).prod() - 1).items()},
                "trades": int((res.fills["purpose"] == "entry").sum()) if len(res.fills) else 0}
    sb = spy[(spy.index >= HOLDOUT_START) & (spy.index <= HOLDOUT_END)]
    out["spy_buy_hold_holdout"] = metrics.summary(sb)
    (OUT / "holdout_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: {"sleeve": {m: round(v["sleeve"][m], 4) for m in ("cagr", "sharpe", "max_dd")},
                          "portfolio": {m: round(v["portfolio"][m], 4) for m in ("cagr", "vol", "sharpe", "sortino", "max_dd", "beta", "alpha_ann")},
                          "kill": v["portfolio_kill_switch_tripped_on"], "trades": v["trades"]}
                      for k, v in out["results"].items()}, indent=1))
    print("SPY buy&hold holdout:", {k: round(v, 4) for k, v in out["spy_buy_hold_holdout"].items()})


if __name__ == "__main__":
    main()
