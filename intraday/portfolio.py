"""Phase 3: combine strategy sleeves into one risk-managed portfolio.

Each strategy runs as a SLEEVE with its own engine and capital (exactly how the live runner will
run them). Because every sleeve is flat overnight, the portfolio's daily return is
    r_p(t) = sum_i a_i(t) * r_i(t),     a_i = scale(t) * w_i(t)  (allocation as a fraction of equity)
with everything decided from data strictly before day t:
  * w_i  inverse-volatility weights, divided by a correlation penalty (1 + sum_j max(rho_ij, 0)),
         estimated on the trailing `weight_lookback` days, rebalanced on the first day of each month
  * scale  volatility targeting: target_vol / trailing realised vol of today's weight vector applied
         to past sleeve returns, capped by
           - max_scale
           - gross exposure:  sum_i a_i * sleeve_max_gross_i       <= max_gross
           - per-stock weight: sum_i a_i * sleeve_max_stock_weight_i <= max_stock_weight
             (worst case: every sleeve at its per-symbol limit in the same stock)
  * kill switch: once the portfolio drawdown from its peak exceeds kill_drawdown, all
         allocations go to zero for the rest of the backtest (live: manual reset required)
The daily-loss limit is enforced INTRADAY by each sleeve's engine and by the live runner; here we
report how often a day's portfolio loss exceeded it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics


@dataclass(frozen=True)
class PortfolioConfig:
    weight_lookback: int = 120
    min_history: int = 60
    corr_penalty: bool = True
    target_vol: float = 0.10                # annualised
    vol_lookback: int = 60
    max_scale: float = 3.0
    max_gross: float = 4.0                  # Reg-T intraday buying power
    max_stock_weight: float = 0.25          # |notional| / equity in any one stock, across sleeves
    sleeve_max_gross: dict = field(default_factory=dict)
    sleeve_max_stock_weight: dict = field(default_factory=dict)
    daily_loss_limit: float = 0.03
    kill_drawdown: float = 0.15


def month_starts(index: pd.Index) -> list:
    out, last = [], None
    for d in index:
        if last is None or (d.year, d.month) != (last.year, last.month):
            out.append(d)
        last = d
    return out


def inverse_vol_weights(hist: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    """Weights from a window of past sleeve returns. Sleeves with too little history get 0."""
    ok = [c for c in hist if hist[c].notna().sum() >= cfg.min_history and hist[c].std() > 0]
    w = pd.Series(0.0, index=hist.columns)
    if not ok:
        return w
    h = hist[ok]
    inv = 1.0 / h.std()
    if cfg.corr_penalty and len(ok) > 1:
        c = h.corr().clip(lower=0).fillna(0)
        pen = 1.0 + (c.sum(axis=1) - 1.0)            # exclude self-correlation
        inv = inv / pen
    w[ok] = inv / inv.sum()
    return w


def strategy_weights(R: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    rows = {}
    for d in month_starts(R.index):
        hist = R[R.index < d].iloc[-cfg.weight_lookback:]
        rows[d] = inverse_vol_weights(hist, cfg)
    W = pd.DataFrame(rows).T.reindex(R.index).ffill().fillna(0.0)
    return W[R.columns]


def _caps(w: pd.Series, cfg: PortfolioConfig) -> float:
    g = sum(w[s] * cfg.sleeve_max_gross.get(s, 1.0) for s in w.index)          # unknown sleeve: 1x
    k = sum(w[s] * cfg.sleeve_max_stock_weight.get(s, 1.0) for s in w.index)
    cap = cfg.max_scale
    if g > 0:
        cap = min(cap, cfg.max_gross / g)
    if k > 0:
        cap = min(cap, cfg.max_stock_weight / k)
    return cap


@dataclass
class PortfolioResult:
    returns: pd.Series
    allocations: pd.DataFrame      # a_i(t)
    weights: pd.DataFrame          # w_i(t)
    scale: pd.Series
    killed_on: object = None


def run_portfolio(R: pd.DataFrame, cfg: PortfolioConfig) -> PortfolioResult:
    """R: daily NET returns per sleeve (per unit of sleeve capital); NaN = sleeve not running yet."""
    W = strategy_weights(R, cfg)
    Rz = R.fillna(0.0)
    scale = pd.Series(0.0, index=R.index)
    port = pd.Series(0.0, index=R.index)
    eq, peak, killed = 1.0, 1.0, None
    for i, d in enumerate(R.index):
        w = W.loc[d]
        if killed is None and w.sum() > 0:
            past = Rz.iloc[max(0, i - cfg.vol_lookback):i]
            if len(past) >= 20:
                vol = float((past @ w).std(ddof=1) * np.sqrt(metrics.ANN))
                if vol > 0:
                    scale[d] = min(cfg.target_vol / vol, _caps(w, cfg))
        port[d] = float((Rz.loc[d] * w).sum() * scale[d])
        eq *= 1 + port[d]
        peak = max(peak, eq)
        if killed is None and eq / peak - 1 < -cfg.kill_drawdown:
            killed = d
    A = W.mul(scale, axis=0)
    if killed is not None:                     # stop from the NEXT day on
        after = R.index > killed
        A.loc[after] = 0.0
        port.loc[after] = 0.0
        scale.loc[after] = 0.0
    return PortfolioResult(port, A, W, scale, killed)


def contributions(R: pd.DataFrame, cfg: PortfolioConfig, spy: pd.Series | None = None) -> pd.DataFrame:
    """Standalone Sharpe, leave-one-out marginal Sharpe contribution, and share of portfolio risk."""
    full = run_portfolio(R, cfg)
    rp = full.returns
    var_p = rp.var()
    rows = {}
    for s in R.columns:
        loo = run_portfolio(R.drop(columns=s), cfg).returns if R.shape[1] > 1 else rp * 0
        contrib = full.allocations[s] * R[s].fillna(0)
        rows[s] = {"standalone_sharpe": metrics.sharpe(R[s].dropna()),
                   "standalone_max_dd": metrics.max_drawdown(R[s].dropna()),
                   "avg_weight": float(full.weights[s].mean()),
                   "avg_allocation": float(full.allocations[s].mean()),
                   "portfolio_sharpe_without": metrics.sharpe(loo),
                   "marginal_sharpe": metrics.sharpe(rp) - metrics.sharpe(loo),
                   "risk_share": float(contrib.cov(rp) / var_p) if var_p > 0 else np.nan,
                   "return_share": float(contrib.sum() / rp.sum()) if rp.sum() != 0 else np.nan}
        if spy is not None:
            rows[s]["beta"], rows[s]["alpha_ann"] = metrics.beta_alpha(R[s].dropna(), spy)
    return pd.DataFrame(rows).T
