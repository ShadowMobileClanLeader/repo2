# Phase 2 — interim findings (ORB pending data)

All results: development period 2016-04-01 → 2025-11-11, holdout untouched. Paper-default
parameters only; nothing tuned. 42 trials recorded so far (research/trials.jsonl).
SPY buy-and-hold over the same period: CAGR 15.1%, Sharpe 0.88, max DD −33.8%.

## Noise-area SPY momentum (Beat the Market)
| Step | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Authors' logic, exit 16:00, zero cost (independent vectorised) | 20.3% | 1.33 | −22% |
| … flat by 15:54 | 19.2% | 1.31 | −21% |
| Our engine (next-bar fills), zero cost | 18.2% | 1.25 | −20% |
| … paper cost level (~0.25 bps/side) | 13.2% | 0.94 | −26% |
| … agreed costs (1.5 bps SPY) | −4.4% | −0.24 | −56% |
| … agreed costs, IEX signals (2020-08 →) | 1.0% | 0.14 | −22% |
Gross edge reproduces the paper (Sharpe 1.33). Net result is decided by the cost assumption:
cost drag at 1.5 bps is ~8.4 bps/day because the book runs ~1.6× gross and trades daily.
Max DD exceeds 15% even at paper costs at the paper's 2%-daily vol target (Phase 3 sizing).

## VWAP mean reversion (control)
Gross Sharpe −0.47, net −1.54: no edge, as expected for SPY (intraday momentum dominates).

## Cross-sectional, top-200 point-in-time universe, open→close, gross 1×
| Signal | Gross bps/day | Gross Sharpe | t | Break-even one-way slippage |
|---|---|---|---|---|
| rev1 | 5.3 | 1.12 | 3.5 | 2.7 bps |
| rev5 | 3.9 | 0.77 | 2.4 | 2.0 bps |
| lps_intraday | −1.5 | −0.32 | −1.0 | — |
| lps_overnight | −0.1 | −0.03 | −0.1 | — |
| lps_combo | −1.5 | −0.39 | −1.2 | — |
| eod_reversal | 0.5 | 0.64 | 2.0 | 0.2 bps |
At the agreed 5 bps every variant loses ~13–17 bps/day net; with compounding the books shrink
into min-commission territory (−96% DD). rev1 is the only signal with a robust gross edge
(positive in 9/10 years); entering at 09:35 rather than the official open adds ~2 bps/day
(vectorised check: 5.5 bps/day at 09:35→15:50 vs 3.4 bps/day at open→close).
Lou-Polk-Skouras shows no edge in this large-cap, daily-rebalanced form (2016-2025).
The VWAP filter cuts the lps_combo loss (fewer trades) but does not create an edge.
