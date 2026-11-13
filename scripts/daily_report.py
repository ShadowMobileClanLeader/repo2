"""Daily report for the paper forward test.  python -m scripts.daily_report [YYYY-MM-DD]

PnL, trades (with reasons), drawdown, alpha/beta vs SPY, and backtest-vs-live differences:
  * per fill: live fill vs the backtest's model price (SIP open of the bar after the decision,
    plus the measured slippage the backtest charges)
  * per day: a shadow backtest (same strategy, same start time, SimBroker, measured costs, IEX bars)
Run after 16:15 ET (SIP bars are only available 15 minutes after the fact on the free plan).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("COST_PROFILE", "measured")
from intraday import metrics  # noqa: E402
from intraday.alpaca_client import AlpacaClient  # noqa: E402
from intraday.config import ET, UTC  # noqa: E402
from intraday.engine import Context, Engine, Portfolio, RiskLimits  # noqa: E402
from intraday.live import LiveMarketData  # noqa: E402
from intraday.backtest import bar_events  # noqa: E402
from intraday.sim_broker import SimBroker  # noqa: E402
from scripts import phase2_run as P2R  # noqa: E402
from scripts.paper_trade import LABEL, LIVE, RETURNS, make_strategy, rules  # noqa: E402

WARNINGS = """> **Read before trusting these numbers**
> - **Forward test only:** this strategy FAILED the one-shot holdout (research/phase4_report.md). It is paper-traded to collect new out-of-sample data, not because it passed.
> - **Alpaca paper fills are optimistic:** simulated fills at/near the quote with no queue position, no partial-fill realism and no market impact; real fills would be worse.
> - **Short selling is simplified:** paper shorting assumes borrow is always available and free; real borrow availability, locate requirements and fees are not modelled (SPY is easy to borrow, but not free).
> - Paper results will therefore **overstate** real performance."""


def load_events(day: str) -> list[dict]:
    p = LIVE / day / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def fill_comparison(client: AlpacaClient, ev: list[dict], day: str) -> pd.DataFrame:
    subs = {e["order_id"]: e for e in ev if e["event"] == "broker_submit"}
    fills = [e for e in ev if e["event"] == "broker_fill"]
    if not fills:
        return pd.DataFrame()
    s = pd.Timestamp(f"{day} 09:30", tz=ET).tz_convert(UTC)
    sip = client.bars(["SPY"], "1Min", s.isoformat(), (s + pd.Timedelta(hours=6, minutes=30)).isoformat(), "sip", "raw")
    sip = sip.set_index("ts")
    per_share = P2R.SPY_COSTS.symbol_slippage_per_share.get("SPY", 0.0)
    rows = []
    for f in fills:
        sub = subs.get(f["order_id"], {})
        t = pd.Timestamp(sub.get("logged_at", f["logged_at"])).floor("min")
        model_open = float(sip["open"].get(t, np.nan))
        side = 1 if f["qty"] > 0 else -1
        model = model_open + side * per_share
        rows.append({"submitted": str(pd.Timestamp(sub.get("logged_at", "")).tz_convert(ET).strftime("%H:%M:%S")) if sub else "",
                     "qty": f["qty"], "live_price": f["price"], "decision_price_iex": f.get("ref_price"),
                     "backtest_model_price": model,
                     "live_vs_model_bps": side * (f["price"] - model) / model * 1e4 if model == model else np.nan,
                     "live_vs_decision_bps": side * (f["price"] - f["ref_price"]) / f["ref_price"] * 1e4
                     if f.get("ref_price") == f.get("ref_price") and f.get("ref_price") else np.nan,
                     "reason": sub.get("reason", "")})
    return pd.DataFrame(rows)


def shadow_backtest(client: AlpacaClient, day: str, start_after: pd.Timestamp | None, capital: float) -> dict:
    """Same strategy, same capital, same start time, simulated fills on IEX bars with measured costs."""
    r = rules()
    md = LiveMarketData(client)
    session = md.session_for(pd.Timestamp(day).date())
    md.set_session(session)
    strat = make_strategy(r)
    syms = strat.prepare_session(session, md)
    o, c = session.open.tz_convert(UTC), session.close.tz_convert(UTC) - pd.Timedelta(seconds=1)
    bars = client.bars(syms, "1Min", o.isoformat(), c.isoformat(), "iex", "raw")
    events = list(bar_events(bars))
    warm = [e for e in events if start_after is not None and e[0] < start_after]
    live = [e for e in events if start_after is None or e[0] >= start_after]
    dummy = Context(session, Portfolio(cash=capital), capital, [])
    for ts, b in warm:
        strat.on_bar((ts + pd.Timedelta(minutes=1)).tz_convert(ET), b, dummy)
    eng = Engine(strat, SimBroker(P2R.SPY_COSTS), capital, RiskLimits(max_gross_leverage=4.0, max_symbol_weight=4.0))
    if warm:
        eng.portfolio.last_price.update({s: x.close for s, x in warm[-1][1].items()})
    res = eng.run_session(session, live)
    return {"pnl": res.eod_equity - res.sod_equity, "fills": [
        {"time": str(f.ts.tz_convert(ET).strftime("%H:%M")), "qty": f.qty, "price": round(f.price, 4), "reason": f.reason}
        for f in eng.fills]}


def main(day: str | None = None):
    day = day or str(pd.Timestamp.now(tz=ET).date())
    client = AlpacaClient()
    ev = load_events(day)
    summ = json.loads((LIVE / day / "summary.json").read_text())
    start_ev = next((e for e in ev if e["event"] == "catch_up"), None)
    start_after = pd.Timestamp(start_ev["logged_at"]).floor("min") if start_ev and summ.get("started_mid_session") else None
    fills = fill_comparison(client, ev, day)
    shadow = shadow_backtest(client, day, start_after, summ["sleeve_capital"])

    hist = pd.read_csv(RETURNS)
    live = hist[hist["source"] == "live"].copy()
    live["port"] = live["ret"].astype(float) * summ["scale"]
    eq = (1 + live["port"]).cumprod()
    dd_now, dd_max = float(eq.iloc[-1] / eq.max() - 1), float((eq / eq.cummax() - 1).min())
    spy = client.bars(["SPY"], "1Day", live["date"].min(), day, "sip", "all")
    spy["date"] = spy["ts"].dt.tz_convert("America/New_York").dt.date.astype(str)
    spy_r = spy.set_index("date")["close"].pct_change()
    lr = live.set_index("date")["port"]
    beta, alpha = metrics.beta_alpha(lr, spy_r) if len(lr) >= 20 else (np.nan, np.nan)

    decisions = [e for e in ev if e["event"] == "decision" and e.get("phase") == "live"]
    lines = [f"# Paper forward test — {day}", "", f"**{LABEL}**", "", WARNINGS, "",
             "## Today",
             f"- Sleeve capital ${summ['sleeve_capital']:,.0f} (base ${summ['base_capital']:,.0f} × vol-target scale {summ['scale']:.2f})"
             + (" — started mid-session (state warmed from earlier bars, no orders for them)" if summ.get("started_mid_session") else ""),
             f"- PnL **${summ['sleeve_pnl']:,.2f}** = {summ['sleeve_return']:+.3%} of sleeve, {summ['portfolio_return']:+.3%} of base capital; fees ${summ['fees']:.2f}",
             f"- Fills {summ['fills']}; max gross ${summ['max_gross']:,.0f}; halted: {summ['halted'] or 'no'}",
             f"- Shadow backtest (same start, SimBroker, measured costs, IEX bars): PnL ${shadow['pnl']:,.2f} "
             f"→ live − backtest = ${summ['sleeve_pnl'] - shadow['pnl']:,.2f}", "",
             "## Decisions (every 30 minutes)", "| time | close | UB | LB | VWAP | pos | action | reason |", "|---|---|---|---|---|---|---|---|"]
    for d in decisions:
        lines.append(f"| {pd.Timestamp(d['time']).strftime('%H:%M')} | {d['close']:.2f} | {d['ub']:.2f} | {d['lb']:.2f} | "
                     f"{d['vwap']:.2f} | {d['position']} | {d['action']} | {d['reason']} |")
    lines += ["", "## Fills: live vs backtest model"]
    if fills.empty:
        lines.append("No fills today.")
    else:
        lines.append(fills.round(4).to_markdown(index=False))
        lines.append(f"\nMean live − model: {fills['live_vs_model_bps'].mean():+.2f} bps (positive = live worse than the backtest assumes).")
    lines += ["", "## Shadow backtest fills", pd.DataFrame(shadow["fills"]).to_markdown(index=False) if shadow["fills"] else "None.",
              "", "## Since start of forward test",
              f"- Live days: {len(live)}; cumulative return on base capital {float(eq.iloc[-1] - 1):+.3%}",
              f"- Drawdown now {dd_now:.2%}, max {dd_max:.2%} (kill switch at −{rules()['portfolio']['kill_drawdown']:.0%})",
              f"- Beta to SPY {beta:.2f}, alpha {alpha:+.2%}/yr" if len(lr) >= 20 else
              f"- Alpha/beta vs SPY: needs ≥ 20 live days (have {len(lr)})"]
    out = LIVE / "reports" / f"{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
