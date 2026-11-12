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
