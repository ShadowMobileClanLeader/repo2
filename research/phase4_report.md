# Phase 4 report — evaluation (development period; holdout section added after the one-shot run)

Development period 2016-04-01 → 2025-11-11, measured costs, fixed capital per day. Numbers:
research/phase4/phase4_results.json. Heatmaps: research/phase4/heatmap_*.png.
Trials recorded: **166 rows / 81 distinct configurations** (correlation-adjusted ≈ 38 independent).

## 1. Ranking (net, measured costs ×1 unless stated)
| Strategy | Sharpe | Sortino | Max DD | ×2 costs | ×3 costs | Verdict |
|---|---|---|---|---|---|---|
| Noise area (SPY), paper defaults, SIP signals | 0.87 | — | −30% (sleeve, 4× cap) | 0.70 | 0.53 | **candidate** |
| Noise area, IEX signals (2020-08→) | 1.18 | — | −15% | 1.07 | 0.97 | candidate (live signal) |
| ORB stocks-in-play, any variant/params | ≤ −3.0 | | −100% | | | rejected |
| Cross-sectional (all signals/params) | ≤ −1.06 | | −74…−97% | | | rejected |
| VWAP mean reversion (control) | −0.74 | | −51% | | | rejected |
| **Portfolio: noise area @ 5% vol target** | 0.92 | 1.70 | −7.2% | 0.76 (DD −8.6%) | 0.60 (DD −11.3%) | |
| SPY buy-and-hold | 0.88 | 1.24 | −33.8% | | | benchmark |

## 2. Parameter sensitivity
- **Noise area — robust.** All 35 grid points (band multiplier 0.5–1.5 × lookback 7–90 days ×
  decision interval 15/30/60 min) are net-positive; 86% above Sharpe 0.5; median 0.80; the paper
  default (0.86) sits inside a broad plateau. Wider bands (1.25–1.5) are a bit better *and* have
  much smaller drawdowns (−10…−14% vs −30%), consistent with the paper's own robustness check.
  15-minute decisions are clearly worse (more trading).
- **ORB — rejected at every point** (stop 5–50% ATR × top 10/20, IEX selection, optimistic fills):
  best −3.0. Larger stops lose less (smaller positions → smaller cost per R) but never break even.
- **Cross-sectional rev1 — rejected at every point** (decile 10/20% × universe 100/200): best −1.06.

## 3. Walk-forward (36-month train / 12-month test, re-select from the 35-point grid by train Sharpe)
| | Sharpe | CAGR | Max DD |
|---|---|---|---|
| Walk-forward re-selected (OOS, 2019-04 → 2025-11) | 0.93 | 11.8% | −16.6% |
| Paper defaults, same windows | **1.25** | 17.9% | −13.7% |
Re-optimising parameters *hurt* out of sample (in-sample winners like band 1.5/lookback 90 in 2019
failed the next year) → **keep the paper defaults; do not tune.** Last fold (2025-04 → 2025-11):
re-selected −1.02; the strategy is weak in 2025 (+2.0% to Nov 11).

## 4. Multiple testing — deflated Sharpe ratio (Bailey & López de Prado)
Null variance of a Sharpe estimate = 1/(T−1) (the noise-area grid's dispersion, 0.000404, matches
it; the cross-trial variance is meaningless here because cost-killed strategies are not "unskilled
draws" — it gives SR0 = 15 and DSR = 0, reported for completeness).

| Number of trials assumed | Expected max Sharpe under no skill | DSR sleeve | DSR portfolio |
|---|---|---|---|
| 1 (pre-registered published hypothesis, tested first) | 0 | 0.998 | 0.999 |
| 38 (correlation-adjusted effective trials) | 0.70 | **0.70** | 0.76 |
| 81 distinct configurations | 0.79 | 0.59 | 0.66 |
| 166 recorded rows | 0.87 | 0.49 | 0.56 |
**Honest reading:** as a single test of a published idea the edge is highly significant; after
charging for everything we tried it does **not** clear the 95% bar on development data alone.
The holdout is the decisive test.

## 5. Capacity (noise area)
Square-root impact (Y = 1, on top of the measured spread — conservative, partly double-counting for
small orders) using SPY's 20-day dollar volume and the strategy's actual leveraged fills:
| Capital | Median share of the fill minute's volume | Impact/side | Est. Sharpe |
|---|---|---|---|
| $100k | 0.3% | 0.25 bps | 0.63 |
| $1M | 3% | 0.8 bps | 0.14 |
| $10M | 32% | 2.5 bps | < 0 |
Every extra 1 bp per side costs ~0.93 Sharpe (daily, leveraged turnover). Capacity is small —
fine for a $100k paper account, unlikely to survive at ≥ $10M.

## 6. Out-of-sample evidence so far
- Post-publication (2024-05 → 2025-11), measured costs: Sharpe 0.65 (SIP signals), 0.52 (IEX).
- Walk-forward OOS of the *fixed* defaults: 1.25 (2019-04 → 2025-11).
- Development period overlaps the paper's own sample (to 2024-04), so most of it is replication,
  not out-of-sample.

## 7. Proposed frozen rules (research/FROZEN_RULES.proposed.json)
Noise area, paper defaults; live signals from IEX; measured costs; single sleeve at a **5% vol
target** (pre-stated rule: highest of 5/7.5/10% with dev max DD at 2× costs ≤ 12% → 5%; 7.5% gave
−12.7%); gross ≤ 4×; flat by 15:55; daily loss limit 3%; kill switch at 12% drawdown (manual reset).
Holdout primary metric: IEX signals at measured costs; SIP and 2×/3× costs reported alongside.

## 8. HOLDOUT — one-shot evaluation of the frozen rules (2025-11-12 → 2026-11-11, 251 sessions)
Rules frozen and committed (9e0cde1, research/FROZEN_RULES.json) before the holdout was unlocked;
one unlock, logged in research/holdout_access.log. Results: research/phase4/holdout_results.json.

| Holdout | CAGR | Vol | Sharpe | Max DD | Beta | Alpha/yr |
|---|---|---|---|---|---|---|
| **Primary: IEX signals, measured costs, 5% vol target** | **−1.6%** | 5.4% | **−0.26** | −4.4% | −0.07 | −0.5% |
| IEX, 2× costs | −2.0% | 5.4% | −0.34 | −4.7% | −0.07 | −0.9% |
| IEX, 3× costs | −2.4% | 5.4% | −0.42 | −5.1% | −0.07 | −1.4% |
| SIP signals, measured costs | −0.7% | 5.4% | −0.10 | −3.7% | −0.07 | +0.4% |
| Sleeve at full strategy leverage (IEX, ×1) | −7.5% | | −0.47 | −14.5% | | |
| SPY buy-and-hold | +12.5% | 13.7% | 0.93 | −8.9% | 1 | |

Monthly (primary): 9 of 13 months negative; largest month −1.5% (Sep 2026). Kill switch never tripped.

## 9. Verdict
**No strategy survives out of sample. Nothing passes Phase 4.**
- The only development-period survivor (SPY noise-area momentum) lost money in the holdout under
  every cost level and both signal feeds, and underperformed SPY buy-and-hold by ~14 points.
- One year is a short test (a true Sharpe-0.9 strategy has roughly a 1-in-6 chance of a negative
  year), but it is consistent with everything else: the deflated Sharpe was only ~0.7 after
  charging for our trials, the strategy was flat in 2025 (+2.0%) and in a drawdown entering the
  holdout, and capacity is small. The evidence does not support claiming an edge.
- ORB stocks-in-play, the cross-sectional strategies and the VWAP control were already rejected on
  the development period (net negative at measured costs at every parameter tested).
- Temptation flagged and refused: the wider band (1.25) looked better in development; I have not
  run it (or anything else) on the holdout. The holdout is now used up — any re-test of a modified
  rule on 2025-11 → 2026-11 would be tuning on the holdout.
