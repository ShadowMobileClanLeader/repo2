# Phase 2 report — strategy replication on the development period

Period 2016-04-01 → 2025-11-11 (holdout untouched). Paper-default parameters only, nothing tuned.
**68 trials recorded** (research/trials.jsonl), including cost/fill diagnostics — all count toward
the Phase-4 deflated Sharpe. Specs and every interpretation: research/phase2_strategy_specs.md.
Full numeric table: research/phase2_results.md.

SPY buy-and-hold, same period: CAGR 15.1%, Sharpe 0.88, Sortino 1.24, max DD −33.8%.

## Headline
**No strategy is profitable net of the agreed costs.** Two have a real gross edge that replicates
the published result, and both live or die on the cost assumption:

| Strategy | Gross edge (zero cost) | Net, agreed costs | Break-even one-way cost |
|---|---|---|---|
| SPY noise-area momentum | Sharpe 1.25 (paper 1.33), CAGR 18% | Sharpe −0.24 | ≈ 1.2 bps all-in (agreed: ~1.9) |
| ORB stocks-in-play (SIP) | Sharpe 2.2–3.0 *if* optimistic fills; −5 if pessimistic | wiped out | ≈ 2.5 bps (agreed: 10) |
| ORB stocks-in-play (IEX) | Sharpe 1.3–2.0 optimistic; −2.7 pessimistic | wiped out | ≈ 1.3 bps |
| Cross-sectional rev1 | Sharpe 1.12, 5.3 bps/day, t = 3.5 | wiped out | ≈ 2.7 bps (agreed: 5) |
| Cross-sectional rev5 | Sharpe 0.77 | wiped out | ≈ 2.0 bps |
| Lou-Polk-Skouras (3 forms) | none (Sharpe −0.4 … 0) | — | — |
| EOD reversal | Sharpe 0.64, 0.5 bps/day | — | ≈ 0.2 bps |
| VWAP mean reversion (control) | Sharpe −0.47 | Sharpe −1.54 | — |

## 1. ORB on Stocks in Play
- **Replication (paper cost level = commission only, as in the paper):** with stops inside the entry
  minute checked from the next bar, SIP selection gives CAGR 37.6%, Sharpe 2.22, max DD −19%
  (paper: 41.6%, 2.81, −12%); 2024–2025 post-publication Sharpe 2.13. With the worst-case
  assumption (stop hit inside the entry minute), Sharpe −6.0. 1-minute bars cannot tell which:
  ~60% of stop-outs have the stop inside the entry bar (stop = 10% ATR ≈ 0.3% of price).
  Average trade: +0.16R (optimistic) vs −0.27R (pessimistic).
- **Why costs kill it:** risk-based sizing puts ~3.3× the capital slice into each trade, so every
  1 bp of one-way slippage costs ~0.066R. At 10 bps the round trip costs ~0.66R against an edge of
  ≤ 0.16R. Every variant (SIP, IEX, 09:50 delayed, VWAP filter, both fill bounds) goes to ~zero.
- **IEX selection** (what live sees) is weaker than SIP: optimistic Sharpe 1.26 at paper costs,
  post-publication 0.66. The 09:50 delayed-SIP variant and the VWAP filter do not help.
- SIP vs IEX top-20 overlap over the whole dev period: 50% (median).

## 2. SPY noise-area momentum (Beat the Market)
- Gross edge replicates exactly (independent vectorised: Sharpe 1.33 = paper). Your 15:55 rule
  costs little (1.31); next-bar fills 1.25; paper cost level 0.94 (CAGR 13.2%, max DD −26%);
  agreed costs −0.24. Paper's own years 2016/2017 were negative too (−12.8%, −6.9%).
- Post-publication (2024-05 → 2025-11) at agreed costs: Sharpe −0.47.
- IEX-based signals (from 2020-08): Sharpe 0.14 vs 0.53 for SIP signals on the same window.
- Max DD exceeds 15% even at paper costs at the paper's 2%-daily vol target → needs lower risk
  (Phase 3).

## 3. Cross-sectional (top-200 point-in-time, open→close, decile L/S, gross 1×)
- Only short-term reversal has a gross edge (rev1 positive in 9/10 years). It turns over 2× equity
  per day, so 5 bps per side costs ~10 bps/day vs a 5.3 bps/day edge.
- Entering at 09:35 rather than the official open is worth ~2 bps/day (verified independently).
- Lou-Polk-Skouras intraday/overnight persistence does not appear in this large-cap, daily form.

## 4. VWAP mean reversion and VWAP filter
- Control loses before costs (consistent with SPY intraday momentum). As a filter it reduces
  trading but creates no edge (ORB: Sharpe unchanged; XS lps_combo: smaller loss only).

## Engine checks done in this phase
- Noise area: engine vs independent vectorised implementation, daily corr 0.99.
- rev1: engine 5.33 vs vectorised 5.48 bps/day on the same window.
- ORB: entry fills match raw opening-range levels to < 0.001%; a split-adjusted price-level bug was
  found and fixed before any ORB result was produced.

## What this means for Phase 3/4
1. The decision is the cost model, not the signal. 1.5 bps SPY and 5/10 bps single-stock slippage
   were agreed as starting points; nothing survives them. These are *assumptions*, and they can
   be measured: Alpaca serves historical NBBO quotes, so we can measure actual quoted spreads for
   SPY, the top-200 names and stocks-in-play at the times we trade.
2. ORB's gross result can be pinned down with tick data (Alpaca historical trades) for the ~60% of
   entries whose stop lies inside the entry minute (~2 h of downloads).
3. Do not lower the cost assumption *because* it makes something work — that would be tuning to
   the result. Measuring it is legitimate; picking it is not.
