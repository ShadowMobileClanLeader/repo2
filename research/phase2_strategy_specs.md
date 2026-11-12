# Phase 2 — strategy specifications, sources and deviations

Every rule below is tagged **[paper]** (stated in the source), **[code]** (authors' published
code, not the paper text), or **[ours]** (our interpretation where the source is silent or where
our hard constraints differ). Nothing here was tuned on results.

## 1. ORB on Stocks in Play — Zarattini, Barbon, Aziz (2024), SSRN 4729284 (v. Feb 16 2024)

| Rule | Implementation | Tag |
|---|---|---|
| Universe | Point-in-time US common stocks (NYSE/Nasdaq/AMEX) incl. delisted; ETFs excluded | paper: all US equities (appears to include ETFs) → **deviation** |
| Filters | open > $5; 14-day avg daily volume ≥ 1M shares; 14-day ATR > $0.50 | paper |
| ATR | simple 14-day mean of true range, from split/dividend-adjusted bars converted to raw $ | ours (method not stated) |
| Relative volume | first-5-min volume / mean of the prior 14 days' first-5-min volume; ≥ 100% | paper |
| Selection | top 20 by relative volume | paper |
| Direction | bullish first candle → buy stop at OR high; bearish → sell stop at OR low; doji → none | paper |
| Stop | 10% of ATR from the **executed** entry price (OTO child order) | paper |
| Exit | no target; flat by 15:55 (engine) | paper says 16:00 → **deviation (your rule)** |
| Sizing | risk 1% of capital per position, cap 4× leverage; capital per position = equity / 20 | paper; split across positions **ours** (ambiguous) |
| Re-entry | one attempt per stock per day; the stop-entry rests until filled or the deadline | ours (implied) |
| Gap through level | stop triggers at the next bar's open (worse than the level) | ours |
| Costs | $0.0035/share (min $0.35) + SEC/TAF + **10 bps slippage per side** | paper: commission only, **no slippage** |

Variants (pre-registered): `sip` (paper replication, SIP selection and OR levels); `iex`
(selection and OR levels from IEX, from 2020-07-27; what a free-plan live account can see);
`sip_delay15` (decide at 09:50 from 15-min-delayed SIP opening bars — stop levels already
passed trigger immediately); `sip+vwap_filter`.

Published (2016–2023, net of commission only): 41.6%/yr, Sharpe 2.81, max DD 12%, beta 0.00.

## 2. SPY noise-area momentum — Zarattini, Aziz, Barbon (2024), "Beat the Market", SSRN 4824172

| Rule | Implementation | Tag |
|---|---|---|
| σ(t) | 14-day mean of \|close(t)/open(9:30) − 1\| per minute of day | paper |
| Bands | UB = max(open, prev close)·(1+σ); LB = min(open, prev close)·(1−σ) | paper |
| Prev close | minus cash dividend on ex-dates | code |
| Decisions | at bar closes 10:00, 10:30, …; stops only checked then too | paper |
| Entry | long if close > UB **and** > VWAP; short mirrored | code (paper text has no VWAP entry condition; economically identical because the VWAP stop would exit at the same check) |
| Trailing stop | long exits below max(UB, VWAP); short above min(LB, VWAP) | paper |
| VWAP | session VWAP from Alpaca bar VWAPs | ours (code uses typical price) |
| Sizing | equity × min(4, 2% / σ_SPY,14d) / price, set once per day | paper |
| Exit | flat by 15:55 | paper says 16:00 → **deviation (your rule)** |
| Costs | 1.5 bps slippage + $0.0035/share + SEC/TAF (~1.9 bps per side) | paper: $0.0035 + $0.001/share (~0.17 bps per side) |

Variants: `paper`, `no_vwap` (band-only entry/stop — the VWAP-filter ablation), `iex_signals`
(signals from IEX bars from 2020-08, what live sees).

Published (2007-05 → 2024-04): 19.6%/yr, Sharpe 1.33, max DD 25%, beta −0.07.
No rigorous independent out-of-sample test was found (re-optimisations only).

## 3. Cross-sectional long/short (top-200 point-in-time liquid universe)

Long top decile / short bottom decile, equal weight, dollar-neutral, gross 1×; 5-minute bars;
enter at 09:35, flat by 15:55; 5 bps slippage per side.

| Signal | Definition | Source | Evidence for large caps |
|---|---|---|---|
| rev1 | −prior-day return | classic reversal | weak / decayed post-2010 |
| rev5 | −prior 5-day return | de Groot, Huij, Zhou (2012) | ~0.6%/wk net for top-100 (1990–2009); weak since |
| lps_intraday | + past-21-day cumulative intraday return | Lou, Polk, Skouras (2019) | strong in 1993–2013, monthly VW; costs "much less attractive" |
| lps_overnight | − past-21-day cumulative overnight return | Lou, Polk, Skouras (2019) | as above |
| lps_combo | z(intraday) + z(−overnight), fixed equal weights | ours | — |
| eod_reversal | −return prior close→15:00; enter 15:30, flat 15:50 | Baltussen, Da, Soebhag (2025) | 3.2 bps/day alpha in top quintile; authors doubt it survives costs |

Deviations: LPS define "open" as the 09:30–10:00 VWAP and rebalance monthly with value
weights; we trade the open daily with equal weights. Heston–Korajczyk–Sadka same-half-hour
continuation is **not implemented**: its large-cap effect (~2 bps per half hour) is below the
half-spread, per the paper itself.

## 4. VWAP mean reversion (control) and VWAP filter

No canonical published definition exists (Zarattini & Aziz's VWAP paper is trend-following).
Ours: σ = 14-day std of SPY open→close returns; every 5 min from 10:00 to 15:30, go long 1× if
close < VWAP·(1 − 0.5σ), short if close > VWAP·(1 + 0.5σ); exit on VWAP touch or flatten.
As a filter: new positions only on the VWAP-confirming side (long above / short below session
VWAP) at the time the order is placed.
