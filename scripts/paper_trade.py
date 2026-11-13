"""Paper-trade the frozen noise-area rules on Alpaca PAPER (forward test; the strategy FAILED the
holdout — this run collects genuinely new out-of-sample data and exercises the live stack).

  python -m scripts.paper_trade --bootstrap    # once: simulated sleeve history for the vol target
  python -m scripts.paper_trade                # trade today's session (blocks until the close)
  python -m scripts.paper_trade --flatten      # emergency: cancel our orders, close OUR SPY stock position

Same Engine + NoiseAreaMomentum as the backtests; AlpacaPaperBroker / LiveFeed / LiveMarketData
are the only live-specific parts. Everything is logged to live/<date>/events.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from intraday import data
from intraday.alpaca_client import AlpacaClient
from intraday.config import ET, FROZEN_RULES_FILE, ROOT
from intraday.engine import Context, Engine, Portfolio, RiskLimits
from intraday.live import AlpacaPaperBroker, LiveFeed, LiveMarketData, vol_target_scale
from intraday.strategies.noise_area import NoiseAreaMomentum

LIVE = ROOT / "live"
KILL = LIVE / "KILL"
RETURNS = LIVE / "sleeve_returns.csv"
IEX_1MIN = data.IntradaySpec("iex", "1Min", "raw", "rth")
LABEL = "FORWARD TEST ONLY - strategy failed the one-shot holdout (research/phase4_report.md)"


class JsonlLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, **kw):
        kw.setdefault("logged_at", pd.Timestamp.now(tz="UTC").isoformat())
        self.f.write(json.dumps(kw, default=str) + "\n")
        self.f.flush()


def rules() -> dict:
    return json.loads(FROZEN_RULES_FILE.read_text())


def make_strategy(r: dict) -> NoiseAreaMomentum:
    p = r["strategy_params"]
    return NoiseAreaMomentum(band_mult=p["band_mult"], lookback=p["lookback"], step=p["step"],
                             first_decision=p["first_decision_minute"], use_vwap=p["use_vwap"],
                             adjust_dividends=p["adjust_dividends"], target_vol=p["sleeve_target_vol_daily"],
                             max_leverage=p["sleeve_max_leverage"], spec=IEX_1MIN)


def bootstrap(r: dict, sessions: int = 90) -> None:
    """Simulated sleeve returns (frozen rules, IEX signals, measured costs) for the sessions before
    the forward test, so the 60-day vol-target window is warm on day one. Reads holdout-period data
    only to estimate risk for live trading (logged unlock); nothing is evaluated or tuned."""
    import os
    os.environ["COST_PROFILE"] = "measured"
    from intraday import calendar, splits
    from intraday.backtest import HistoricalMarketData, run_backtest
    from intraday.config import HOLDOUT_END
    from intraday.universe import FEATURES_FILE
    from scripts import phase2_run as P2R
    splits.unlock_holdout("live warm-up: vol-target estimate for frozen rules (no evaluation)")
    sess = calendar.sessions(HOLDOUT_END - dt.timedelta(days=200), HOLDOUT_END)[-sessions:]
    feats = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")])
    res = run_backtest(make_strategy(r), sess[0].date, sess[-1].date, IEX_1MIN, costs=P2R.SPY_COSTS,
                       md=HistoricalMarketData(features=feats), reset_capital_daily=True)
    df = pd.DataFrame({"date": res.returns.index, "ret": res.returns.values, "source": "bootstrap_sim"})
    LIVE.mkdir(parents=True, exist_ok=True)
    df.to_csv(RETURNS, index=False)
    print(f"bootstrap: {len(df)} simulated sessions -> {RETURNS}")


def flatten(client: AlpacaClient) -> None:
    """Cancel orders WE placed (client_order_id prefix) and close OUR SPY stock position."""
    for o in client.trading_get("/v2/orders", status="open", limit=500):
        if str(o.get("client_order_id", "")).startswith("noisefwd-"):
            client.trading_delete(f"/v2/orders/{o['id']}")
            print("canceled", o["id"])
    for p in client.trading_get("/v2/positions"):
        if p["symbol"] == "SPY" and p.get("asset_class") == "us_equity":
            q = int(float(p["qty"]))
            client.trading_post("/v2/orders", {"symbol": "SPY", "qty": str(abs(q)), "side": "sell" if q > 0 else "buy",
                                               "type": "market", "time_in_force": "day",
                                               "client_order_id": f"noisefwd-flatten-{int(pd.Timestamp.now().timestamp())}"})
            print("flatten order sent for SPY", q)


def trade_today(client: AlpacaClient, r: dict, base_capital: float) -> None:
    today = pd.Timestamp.now(tz=ET).date()
    md = LiveMarketData(client)
    session = md.session_for(today)
    if session is None:
        print("no session today"); return
    if pd.Timestamp.now(tz=ET) >= session.flatten_time - pd.Timedelta(minutes=2):
        print("too late to trade today"); return
    day_dir = LIVE / str(today)
    log = JsonlLog(day_dir / "events.jsonl")
    log(event="start", label=LABEL, rules_frozen_at=r["frozen_at"], base_capital=base_capital)
    if KILL.exists():
        log(event="halt", reason=f"kill switch file present: {KILL.read_text()[:200]}")
        print("KILL file present - not trading"); return

    hist = pd.read_csv(RETURNS) if RETURNS.exists() else pd.DataFrame(columns=["date", "ret", "source"])
    pc = r["portfolio"]
    scale = vol_target_scale(hist["ret"].astype(float), pc["target_vol"], pc["vol_lookback"], pc["max_scale"],
                             pc["max_gross"], pc["sleeve_max_gross"]["noise_area"])
    alloc = base_capital * scale
    log(event="allocation", vol_target=pc["target_vol"], scale=scale, sleeve_capital=alloc,
        history_days=int(len(hist)))
    if alloc <= 0:
        log(event="halt", reason="no allocation (insufficient history)"); return

    md.set_session(session)
    strat = make_strategy(r)
    symbols = strat.prepare_session(session, md)
    log(event="prepared", symbols=symbols, prev_close=strat.prev_close, dividend=getattr(strat, "dividend", 0.0),
        leverage=strat.leverage, sigma_points=int(len(strat.sigma)))

    broker = AlpacaPaperBroker(client, tag="noisefwd", log=log)
    feed = LiveFeed(client, symbols, session)
    # warm strategy state (open price, VWAP) if starting after the open; no orders from these bars
    warm = feed.catch_up()
    dummy = Context(session, Portfolio(cash=alloc), alloc, [])
    for ts, bars in warm:
        strat.on_bar((ts + pd.Timedelta(minutes=1)).tz_convert(ET), bars, dummy)
        broker.last_price.update({s: b.close for s, b in bars.items()})
    for d in strat.decisions:
        log(event="decision", phase="catch_up_not_traded", **d)
    n_warm = len(strat.decisions)
    log(event="catch_up", bars=len(warm))

    risk = RiskLimits(max_gross_leverage=pc["max_gross"], max_symbol_weight=pc["max_gross"],
                      max_daily_loss=min(1.0, pc["daily_loss_limit"] / scale), kill_switch_file=KILL)
    engine = Engine(strat, broker, alloc, risk, logger=log)
    engine.portfolio.last_price.update(broker.last_price)

    class Logged:      # log each live decision as it happens
        def __iter__(self_inner):
            seen = n_warm
            for ev in feed:
                yield ev
                for d in strat.decisions[seen:]:
                    log(event="decision", phase="live", **d)
                seen = len(strat.decisions)

    res = engine.run_session(session, Logged())
    pnl = res.eod_equity - res.sod_equity
    summary = {"date": str(today), "label": LABEL, "base_capital": base_capital, "scale": scale,
               "sleeve_capital": alloc, "sleeve_pnl": pnl, "sleeve_return": pnl / alloc,
               "portfolio_return": pnl / base_capital, "fees": res.fees, "fills": res.n_fills,
               "max_gross": res.max_gross, "halted": res.halted,
               "started_mid_session": bool(warm), "decisions_live": len(strat.decisions) - n_warm}
    (day_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    hist = pd.concat([hist, pd.DataFrame([{"date": str(today), "ret": pnl / alloc, "source": "live"}])])
    hist.to_csv(RETURNS, index=False)
    live = hist[hist["source"] == "live"]
    if len(live):
        port = live["ret"].astype(float) * scale
        eq = (1 + port).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        if dd < -pc["kill_drawdown"]:
            KILL.write_text(f"drawdown {dd:.1%} exceeded {pc['kill_drawdown']:.0%} on {today}; manual reset required")
            log(event="kill_switch", drawdown=dd)
    log(event="end", **summary)
    print(json.dumps(summary, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--flatten", action="store_true")
    ap.add_argument("--capital", type=float, default=100_000.0)
    a = ap.parse_args()
    client = AlpacaClient()                     # refuses any non-paper endpoint
    r = rules()
    if a.bootstrap:
        bootstrap(r)
    elif a.flatten:
        flatten(client)
    else:
        trade_today(client, r, a.capital)


if __name__ == "__main__":
    sys.exit(main())
