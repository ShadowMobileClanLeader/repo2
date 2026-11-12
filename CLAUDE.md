# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Intraday US-equity strategy research with an honest-evaluation pipeline, plus a runner for
**Alpaca PAPER trading only**. Goal: find the strategy (or combination) with the best
risk-adjusted, out-of-sample, net-of-cost performance (max DD < 15%) and paper-trade it.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                   # all tests (pytest.ini sets pythonpath = . tests)
python -m pytest tests/test_engine.py::test_flat_by_1555_and_no_decisions_after_deadline -q
python -m scripts.download_data calendar candidates daily features spy open5   # fill parquet cache (stages are independent args)
python -m scripts.check_feed                          # IEX vs SIP report -> research/phase1_feed_report.json
python -m scripts.engine_sanity                       # engine vs independent SPY calc + cost drag
```

Needs `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in the environment. The cache (`data/cache/`,
git-ignored) is lost when a cloud container is reclaimed; rebuild with `scripts/download_data.py`.
The `open5`/`iex_open5` stages are slow (hours): Alpaca bar-request latency scales with the
number of symbols (~16 ms/symbol), so downloads parallelise across sessions, not symbols.
Download gotchas: the 200 req/min limit is per account — when running two download processes
set `ALPACA_REQ_PER_MIN` so they sum below 200; never SIGSTOP a downloader (its open requests
keep counting against Alpaca and throttle everything else) — kill and restart it, cached days
are skipped; a process stuck in long 429 back-offs recovers faster if restarted.

## Non-negotiable research rules (from the project owner)

* Paper endpoint only. `alpaca_client.assert_paper_url` is checked on every trading request;
  never add a live-endpoint setting.
* Work in phases (1 data/engine, 2 strategies, 3 portfolio/risk, 4 evaluation, 5 paper trading)
  and **stop after each phase** to show results.
* **Never tune on the holdout** (2025-11-12 → 2026-11-11, frozen in `config.py` — do not move it).
  Say so explicitly if tempted. Holdout reads raise `HoldoutLockedError` until
  `research/FROZEN_RULES.json` exists and `splits.unlock_holdout(reason)` is called (logged).
* Implement strategies from the cited papers; flag ambiguities instead of guessing. If published
  results don't reproduce, report it — don't adjust parameters until they do.
* Count every strategy/parameter variant tried (needed for the deflated Sharpe in Phase 4).
* Decisions already made: commission $0.0035/share (min $0.35); slippage 1.5 bps SPY,
  **10 bps for ORB stocks-in-play**, 5 bps liquid large caps; ORB is evaluated with SIP-based
  selection (paper replication) AND IEX-based selection (2020-07-27+, what live can actually see),
  judged on IEX; plus a labelled variant selecting at 9:50 from 15-min-delayed SIP.

## Architecture (the parts that span files)

* **One engine for backtest and live** (`intraday/engine.py`). `Engine.run_session` consumes an
  iterable of `(bar_start_ts_utc, {symbol: Bar})` events and talks to a `Broker` protocol:
  `SimBroker` (`sim_broker.py`) in backtests, an Alpaca paper broker in live (Phase 5). Never fork
  strategy or order logic between the two paths.
* **Strategy protocol**: `prepare_session(session, md) -> symbols` (md serves only data from
  before the session; `HistoricalMarketData` raises `LookAheadError` otherwise) and
  `on_bar(now_et_bar_close, bars, ctx) -> {symbol: Target} | None`. `Target(weight, entry_stop,
  stop_loss, reason)` — weight is a signed fraction of **start-of-day** equity; omitted symbols
  keep their current state; weight 0 cancels that symbol's orders and exits.
* **Timing/fill semantics** (tested; don't weaken): orders fill no earlier than the next bar —
  market at next open, stops at level or gap-open. Orders with `attached_stop_loss` become
  broker-side OTO child stops; if the child level is inside the entry bar's range, the sim assumes
  it was hit. The engine makes its last decision at `flatten_time - bar_length` so everything is
  flat by close−5 min (15:55; 12:55 on half days, from Alpaca's calendar); `end_session` asserts
  no overnight positions. Risk halts (daily loss, kill-switch file) flatten and stop the day.
* **Data** (`data.py`): intraday cache is one parquet per (dataset, session) plus a
  `.symbols.json` manifest of *requested* symbols, so "no bars" ≠ "not fetched". `IntradaySpec`
  names a dataset (`feed_timeframe_adjustment_window`, e.g. `sip_1Min_raw_rth`,
  `sip_5Min_split_open5`). Intraday prices are raw; multi-day features are dimensionless or
  converted to raw dollars (e.g. ATR = adjusted ATR × raw/adjusted close).
* **Point-in-time universe** (`universe.py`): candidate pool = Alpaca active+inactive stocks +
  S&P 500 history (`data/reference/`, fja05680/sp500) + corporate-action delistings, because
  Alpaca's asset list omits many delisted names whose bars still exist. `compute_daily_features`
  values are known at that date's close and may only be used from the next session.
* **Feeds**: account is on the free plan — real-time IEX only; SIP for history (not the last
  15 min); IEX history starts 2020-07-27. IEX carries ~2% of volume, so volume-ranked selections
  differ a lot from SIP (stocks-in-play top-20 overlap ~45%); prices/VWAP are close (1–2 bps).
