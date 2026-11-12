# Intraday research + Alpaca paper trading

Research framework for intraday US-equity strategies, tested honestly net of costs, with
a paper-trading runner that uses the **same engine** as the backtester.

**Paper trading only.** The trading endpoint is hard-wired to
`https://paper-api.alpaca.markets`; the client refuses anything else (including an
`APCA_API_BASE_URL` pointing elsewhere). Keys come from `ALPACA_API_KEY` /
`ALPACA_SECRET_KEY`.

## Layout

| module | role |
|---|---|
| `intraday/config.py` | endpoints, paths, **frozen** dev/holdout dates |
| `intraday/alpaca_client.py` | REST client: paper-only guard, rate limiter (190 req/min), retries |
| `intraday/calendar.py` | trading sessions incl. early closes; flatten deadline = close − 5 min |
| `intraday/data.py` | parquet cache for daily + intraday bars; research loaders enforce holdout lock |
| `intraday/universe.py` | survivorship-aware candidate pool + point-in-time daily features & filters |
| `intraday/engine.py` | event-driven engine, `Strategy` protocol, `Target`, `Portfolio`, risk limits |
| `intraday/sim_broker.py` | backtest fills (next-bar open, stop logic, OTO stops, costs) |
| `intraday/costs.py` | slippage (bps per symbol, stress multiplier) + commission + SEC/TAF fees |
| `intraday/splits.py` | holdout lock (`FROZEN_RULES.json` + logged unlock), walk-forward folds |
| `intraday/backtest.py` | replays cached bars through the engine; look-ahead-guarded market data |
| `intraday/metrics.py` | Sharpe, Sortino, max DD, CAGR, beta/alpha |

## Engine contract

* A strategy implements `prepare_session(session, md) -> symbols` (md only serves data from
  before the session) and `on_bar(now, bars, ctx) -> {symbol: Target} | None`.
* `Target(weight, entry_stop=None, stop_loss=None, reason="")`: signed fraction of
  start-of-day equity; optional stop-entry level and protective stop.
* The bar that generates a signal can never fill it: orders execute at the next bar
  (open for market orders; stop level or gap-open for stops). If an OTO stop level is inside
  the entry bar's range we assume it was hit (worst case).
* Every position is closed by the flatten deadline (15:55 ET; 12:55 on half days).
  No decisions are taken after it.
* Risk limits: gross leverage, per-symbol weight, daily loss (flatten + halt), kill-switch file.

## Research periods (frozen)

* Data: 2016-01-04 →. Development: 2016-04-01 → 2025-11-11. Holdout: 2025-11-12 → 2026-11-11.
* Holdout reads raise `HoldoutLockedError` until `research/FROZEN_RULES.json` exists and
  `unlock_holdout(reason)` is called; every unlock is appended to `research/holdout_access.log`.
* Walk-forward: 36-month train / 12-month test, rolling, inside development only.

## Usage

```bash
pip install -r requirements.txt
python -m scripts.download_data calendar candidates daily features spy open5
python -m scripts.check_feed        # IEX vs SIP report -> research/phase1_feed_report.json
python -m scripts.engine_sanity     # end-to-end engine check on SPY
python -m pytest -q
```

Third-party data: `data/reference/sp500_ticker_start_end.csv` from
[fja05680/sp500](https://github.com/fja05680/sp500) (MIT licence), used for point-in-time
S&P 500 membership.
