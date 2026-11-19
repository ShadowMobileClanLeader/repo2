"""Dealer gamma exposure (GEX) from Alpaca option chains (SPX + SPY).

UNTESTED SIGNAL: Alpaca only serves the latest open interest, so this has never been backtested.

Method (computed before the open from the PREVIOUS session's open interest and closing quotes):
  * forward F per expiry from put-call parity on near-the-money strikes (no SPX index feed needed)
  * implied vol per strike from the out-of-the-money mid quote (Black-76 on the forward)
  * gamma per contract (Black-Scholes, r = q = 0 approximation, spot ~ forward)
  * GEX = sum over calls of  gamma * OI * multiplier * S^2 * 1%
        - sum over puts  of  gamma * OI * multiplier * S^2 * 1%
    i.e. dollars of underlying dealers must trade per 1% move, under the common convention that
    customers buy puts and sell calls (dealers long calls, short puts). The convention is an
    ASSUMPTION — open interest does not reveal who is long.
  * gamma flip: the spot level (within +/-6%) where GEX changes sign, holding vols fixed.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from .config import ET, UTC

SQRT2PI = math.sqrt(2 * math.pi)
YEAR_MIN = 365.0 * 24 * 60


def _ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def black_price(F, K, T, sigma, call: bool):
    F, K, T, sigma = map(lambda a: np.asarray(a, dtype=float), (F, K, T, sigma))
    st = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * st ** 2) / st
    d2 = d1 - st
    if call:
        return F * _ncdf(d1) - K * _ncdf(d2)
    return K * _ncdf(-d2) - F * _ncdf(-d1)


def implied_vol(price: float, F: float, K: float, T: float, call: bool, lo=0.01, hi=5.0, tol=1e-6) -> float:
    """Bisection on Black-76 (undiscounted). NaN if the price is outside no-arbitrage bounds."""
    intrinsic = max(0.0, F - K) if call else max(0.0, K - F)
    if not (price > intrinsic and price < (F if call else K)) or T <= 0:
        return float("nan")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(black_price(F, K, T, mid, call)) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def bs_gamma(S, K, T, sigma):
    S, K, T, sigma = map(lambda a: np.asarray(a, dtype=float), (S, K, T, sigma))
    st = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * st ** 2) / st
    return np.exp(-0.5 * d1 ** 2) / (SQRT2PI * S * st)


def years_to_expiry(expiry: dt.date, now: pd.Timestamp, am_settled: bool) -> float:
    close = pd.Timestamp(f"{expiry} {'09:30' if am_settled else '16:00'}").tz_localize(ET)
    minutes = (close - now.tz_convert(ET)).total_seconds() / 60
    return max(minutes, 60.0) / YEAR_MIN          # floor at one hour (0DTE at the open)


def implied_forward(chain: pd.DataFrame) -> float:
    """Put-call parity on the strikes where |C - P| is smallest: F = K + (C - P)."""
    c = chain[chain.type == "call"].set_index("strike")["mid"]
    p = chain[chain.type == "put"].set_index("strike")["mid"]
    both = pd.concat({"c": c, "p": p}, axis=1).dropna()
    both = both[(both.c > 0) & (both.p > 0)]
    if both.empty:
        return float("nan")
    near = both.assign(d=(both.c - both.p).abs()).nsmallest(3, "d")
    return float(np.median(near.index.to_numpy() + (near.c - near.p).to_numpy()))


def chain_gamma(chain: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """chain columns: expiry, strike, type, oi, bid, ask, multiplier, am_settled.
    Returns per-contract gamma table and the estimated spot (front-expiry forward)."""
    ch = chain.copy()
    ch["mid"] = np.where((ch.bid > 0) & (ch.ask > 0), (ch.bid + ch.ask) / 2, np.nan)
    ch["oi"] = ch["oi"].fillna(0.0)
    rows = []
    spot = float("nan")
    for exp, g in ch.groupby("expiry"):
        T = years_to_expiry(exp, now, bool(g.am_settled.iloc[0]))
        F = implied_forward(g)
        if not np.isfinite(F):
            continue
        if not np.isfinite(spot):
            spot = F                                                # nearest usable expiry
        # one vol per strike from the out-of-the-money side
        otm = g[((g.type == "call") & (g.strike >= F)) | ((g.type == "put") & (g.strike < F))]
        vols = {r.strike: implied_vol(r.mid, F, r.strike, T, r.type == "call")
                for r in otm.itertuples() if np.isfinite(r.mid)}
        g = g.assign(T=T, forward=F, iv=g.strike.map(vols))
        rows.append(g)
    if not rows:
        return ch.iloc[0:0].assign(T=[], forward=[], iv=[], gamma=[]), spot
    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(out.iv) & (out.iv > 0)]
    out["gamma"] = bs_gamma(out.forward, out.strike, out["T"], out.iv)
    return out, spot


def gex_at(table: pd.DataFrame, S: float) -> float:
    """Dealer GEX ($ per 1% move) if the underlying were at S (vols held fixed)."""
    if table.empty:
        return 0.0
    scale = S / table.forward                                      # move every expiry's forward with spot
    g = bs_gamma(table.forward * scale, table.strike, table["T"], table.iv)
    sign = np.where(table.type == "call", 1.0, -1.0)
    return float(np.sum(sign * g * table.oi * table.multiplier * (table.forward * scale) ** 2 * 0.01))


def gamma_flip(table: pd.DataFrame, spot: float, width: float = 0.06, n: int = 121) -> float | None:
    grid = spot * np.linspace(1 - width, 1 + width, n)
    vals = np.array([gex_at(table, s) for s in grid])
    sgn = np.sign(vals)
    idx = np.flatnonzero(sgn[1:] != sgn[:-1])
    if len(idx) == 0:
        return None
    i = idx[np.argmin(np.abs(grid[idx] - spot))]                    # crossing nearest to spot
    x0, x1, y0, y1 = grid[i], grid[i + 1], vals[i], vals[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


# ---------------------------------------------------------------- Alpaca fetch
def fetch_chain(client, underlying: str, asof: dt.date, max_days: int = 60) -> pd.DataFrame:
    """Contracts (with open interest as of the previous session) merged with indicative quotes."""
    end = (asof + dt.timedelta(days=max_days)).isoformat()
    contracts, tok = [], None
    while True:
        p = dict(underlying_symbols=underlying, limit=10000, expiration_date_gte=asof.isoformat(),
                 expiration_date_lte=end)
        if tok:
            p["page_token"] = tok
        j = client.trading_get("/v2/options/contracts", **p)
        contracts += j.get("option_contracts", [])
        tok = j.get("next_page_token")
        if not tok:
            break
    quotes, tok = {}, None
    while True:
        p = dict(feed="indicative", limit=1000, expiration_date_gte=asof.isoformat(), expiration_date_lte=end)
        if tok:
            p["page_token"] = tok
        j = client.data_get(f"/v1beta1/options/snapshots/{underlying}", **p)
        for sym, v in (j.get("snapshots") or {}).items():
            q = v.get("latestQuote") or {}
            quotes[sym] = (q.get("bp"), q.get("ap"))
        tok = j.get("next_page_token")
        if not tok:
            break
    rows = []
    for c in contracts:
        bp, ap = quotes.get(c["symbol"], (None, None))
        rows.append({"symbol": c["symbol"], "underlying": underlying,
                     "expiry": pd.Timestamp(c["expiration_date"]).date(), "strike": float(c["strike_price"]),
                     "type": c["type"], "oi": float(c["open_interest"]) if c.get("open_interest") else 0.0,
                     "oi_date": c.get("open_interest_date"), "bid": float(bp or 0), "ask": float(ap or 0),
                     "multiplier": float(c.get("size") or 100),
                     "am_settled": c.get("root_symbol") == "SPX" and underlying == "SPX"})
    return pd.DataFrame(rows)


def snapshot(client, asof: dt.date, now: pd.Timestamp | None = None,
             underlyings=("SPX", "SPY")) -> tuple[dict, pd.DataFrame]:
    """Compute today's pre-open GEX. Returns (summary, per-contract table)."""
    now = now or pd.Timestamp.now(tz=UTC)
    tables, summary = [], {"asof": str(asof), "computed_at": str(now), "convention": "dealer long calls, short puts",
                           "by_underlying": {}}
    for u in underlyings:
        chain = fetch_chain(client, u, asof)
        tbl, spot = chain_gamma(chain, now)
        g = gex_at(tbl, spot) if np.isfinite(spot) else float("nan")
        summary["by_underlying"][u] = {"spot_est": spot, "gex_usd_per_1pct": g, "contracts": int(len(chain)),
                                       "contracts_with_iv": int(len(tbl)), "total_oi": float(chain.oi.sum()),
                                       "oi_date": str(chain.oi_date.dropna().max()) if len(chain) else None,
                                       "gamma_flip": gamma_flip(tbl, spot) if np.isfinite(spot) and len(tbl) else None}
        tables.append(tbl)
    summary["total_gex_usd_per_1pct"] = float(np.nansum([v["gex_usd_per_1pct"] for v in summary["by_underlying"].values()]))
    summary["regime"] = "negative" if summary["total_gex_usd_per_1pct"] < 0 else "positive"
    return summary, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
