# Phase 3 report — measured costs, portfolio and risk layer

Development period only (2016-04-01 → 2025-11-11); holdout untouched. Full numbers:
research/phase3/phase3_results.json, research/phase3/measured_costs.json.
Trials recorded so far: **119** (17 strategy variants × agreed and measured costs at 1×/2×/3×,
plus zero-cost and paper-cost diagnostics). Portfolio configurations evaluated: 4 scenarios ×
4 volatility targets.

## 1. Measured costs (SIP quotes, 60 sessions, 6 per year)
Quoted half-spread (per side) at the times each strategy actually trades, mean across observations:

| Where we trade | 09:35 | 09:40 | 10:00 | 11:00 | 15:30 | 15:50/54 | Agreed before |
|---|---|---|---|---|---|---|---|
| SPY (one tick ≈ $0.0064/share) | — | — | 0.22 | 0.17 | 0.16 | 0.16 bps | 1.5 bps |
| Top-200 liquid stocks | 6.2 | — | — | — | 1.9 | 2.7 bps | 5 bps |
| Stocks in play, SIP top-20 | 14.8 | 10.6 | 8.1 | 5.4 | — | 2.6 bps | 10 bps |
| Stocks in play, IEX top-20 | 16.9 | 12.9 | 9.1 | 6.1 | — | 3.0 bps | 10 bps |

The cost model now uses these as a time-of-day profile (linear interpolation; SPY per share);
commissions/fees unchanged; 2×/3× stress multiplies the measured slippage. **Not captured**:
walking the book (median displayed size at the touch is ~200 shares for single stocks, while ORB
orders are ~400–500 shares) and stop-order overshoot — both make single-stock costs higher, so
the 2×/3× stress results matter for ORB/XS. The measurement is not a choice made to rescue a
strategy: it lowered SPY costs and raised costs at the open.

## 2. Sleeves at measured costs (net, fixed capital per day)
| Sleeve | Sharpe ×1 | ×2 | ×3 | CAGR ×1 | Max DD ×1 | Beta | Alpha/yr |
|---|---|---|---|---|---|---|---|
| Noise area, SIP signals (2016→) | **0.87** | 0.70 | 0.53 | 12.0% | −30% | −0.06 | 13.2% |
| Noise area, IEX signals (2020-08→) | **1.18** | 1.07 | 0.97 | 17.0% | −15% | −0.03 | 17.2% |
| ORB IEX, worst-case fills | −26.7 | | | | −100% | | |
| ORB IEX, optimistic fills | −10.7 | | | | −100% | | |
| XS rev1 | −1.46 | −3.3 | −5.2 | −16% | −83% | 0.06 | |
| VWAP MR (control) | −0.74 | | | | | | |

ORB entries happen in the first minutes, when spreads are 15–17 bps: costs run 37–74 bps/day of
sleeve capital, far above any gross edge. All cross-sectional signals lose net (open spreads 6 bps).
Noise area is positive after publication (2024-05 →): Sharpe 0.65 (SIP signals), 0.52 (IEX).

## 3. Portfolio construction
Implemented in `intraday/portfolio.py` (unit-tested), all decisions from past data only:
inverse-volatility weights / (1 + Σ positive correlations), 120-day window, monthly rebalance;
volatility target (default 10%/yr, 60-day window); caps on worst-case gross (≤ 4× equity) and on
single-stock exposure (≤ 10% of equity); drawdown kill switch at 12% (flatten, manual reset);
per-sleeve intraday daily-loss limit in the engine. Sleeve correlations are ~0 (−0.05…0.06), so
the correlation penalty barely matters here.

| Scenario (common start) | Sharpe | CAGR | Max DD | Marginal Sharpe of each sleeve |
|---|---|---|---|---|
| All candidates (2020-12→) | −15.1 | −63% | −99% | noise +2.5, ORB −5.8…−14, rev1 +0.1 |
| Live-realistic, IEX signals (2020-12→) | −15.3 | −64% | −99% | noise_iex +4.4, ORB −14.1, rev1 +2.0 |
| **Eligible only = noise area (2016-07→)** | **0.92** | **10.1%** | **−12.7%** | — |
| SPY buy-and-hold, same window | 0.88 | 15.3% | −33.8% | |

**Inverse-volatility weighting ignores expected returns**: steadily-losing, low-volatility sleeves
(ORB, rev1) get *large* weights and sink the combination. The leave-one-out marginal Sharpe makes
this explicit. Combining is only sensible over sleeves with a demonstrated net edge; the
eligibility rule (fixed before these runs: dev net Sharpe > 0 at measured costs and ≥ 0 at 2×)
admits only the noise-area sleeve, so in practice the "portfolio" is one sleeve plus the risk layer.

### Eligible-only portfolio (noise area, 10% vol target)
| Costs | CAGR | Vol | Sharpe | Sortino | Max DD | Beta | Alpha/yr |
|---|---|---|---|---|---|---|---|
| measured ×1 | 10.1% | 11.1% | 0.92 | 1.70 | −12.7% | −0.05 | 11.1% |
| ×2 | 8.2% | 11.1% | 0.76 | 1.38 | −17.3% | −0.05 | 9.3% |
| ×3 | 6.3% | 11.1% | 0.60 | 1.08 | −22.0% | −0.05 | 7.5% |

By year (×1): 2016 −4.1%, 2017 −9.0%, 2018 +20.2%, 2019 +0.6%, 2020 +11.1%, 2021 +16.5%,
2022 +16.7%, 2023 +24.7%, 2024 +21.3%, 2025 (to Nov 11) +2.0%.
Returns are almost entirely alpha (beta −0.05); it made money in 2022 when SPY fell.
Kill switch (12%) would have tripped twice: 2017-06 (recovered 2018-11) and **2025-09-26 — i.e. the
strategy enters the holdout period in a drawdown**.
2 days had a portfolio loss above the 3% daily limit (the intraday limit would have stopped them earlier).

### Volatility target sensitivity (measured ×1; reported, not selected)
| Target | CAGR | Vol | Sharpe | Max DD | Kill trips |
|---|---|---|---|---|---|
| 5% | 5.3% | 5.8% | 0.92 | −7.2% | 0 |
| 7.5% | 7.9% | 8.6% | 0.92 | −10.4% | 0 |
| 10% | 10.1% | 11.1% | 0.92 | −12.7% | 2 |
| 12.5% | 11.6% | 12.8% | 0.92 | −16.3% | 3 |
Realised vol runs ~10–15% above target. Under 2× costs, the 10% target breaches the 15% DD rule.

## 4. What this means for Phase 4
- One candidate remains: SPY noise-area momentum (paper rules), with the vol target as the main
  risk decision. Phase 4 must test it honestly: parameter heatmaps (band multiplier, lookback,
  decision interval, vol target), walk-forward, deflated Sharpe over all 119+ trials,
  capacity, and IEX-signal robustness — then, once rules are frozen, the holdout.
- Temptation to flag: the 2025-09 drawdown is right at the holdout boundary. I will not look at
  holdout data or adjust anything because of it.
