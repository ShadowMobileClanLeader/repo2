"""Performance metrics on daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(ANN)) if downside > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (ANN / len(r)) - 1)


def beta_alpha(r: pd.Series, mkt: pd.Series) -> tuple[float, float]:
    """OLS beta to the market and annualised alpha (intercept * 252)."""
    df = pd.concat([r, mkt], axis=1, join="inner").dropna()
    if len(df) < 20:
        return float("nan"), float("nan")
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    b = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    return float(b), float(a * ANN)


def summary(r: pd.Series, mkt: pd.Series | None = None) -> dict:
    out = {"days": int(r.notna().sum()), "cagr": cagr(r), "vol": float(r.std() * np.sqrt(ANN)),
           "sharpe": sharpe(r), "sortino": sortino(r), "max_dd": max_drawdown(r),
           "hit_rate": float((r > 0).mean())}
    if mkt is not None:
        out["beta"], out["alpha_ann"] = beta_alpha(r, mkt)
    return out


# --------------------------------------------------------------- multiple-testing adjustment
EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P(true per-period Sharpe > sr_benchmark), allowing for
    skewness and fat tails. Both Sharpes are PER PERIOD (daily), not annualised."""
    from scipy.stats import kurtosis, norm, skew
    r = r.dropna()
    t = len(r)
    sd = r.std(ddof=1)
    if t < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / denom))


def expected_max_sharpe(n_trials: int, var_trial_sharpes: float) -> float:
    """SR0: expected maximum of n_trials per-period Sharpe estimates under the null of no skill."""
    from scipy.stats import norm
    if n_trials <= 1 or var_trial_sharpes <= 0:
        return 0.0
    g = EULER_GAMMA
    return float(np.sqrt(var_trial_sharpes) * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                                               + g * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r: pd.Series, n_trials: int, var_trial_sharpes: float) -> dict:
    """DSR (Bailey & Lopez de Prado 2014) = PSR against the Sharpe you would expect from the best of
    n_trials unskilled strategies. var_trial_sharpes is the variance of the trials' PER-PERIOD Sharpes."""
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpes)
    return {"dsr": probabilistic_sharpe(r, sr0), "sr0_daily": sr0, "sr0_annual": sr0 * np.sqrt(ANN),
            "sharpe_daily": float(r.mean() / r.std(ddof=1)), "n_trials": n_trials}
