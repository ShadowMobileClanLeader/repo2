"""Point-in-time universes (survivorship-bias control).

Candidate pool = union of
  * Alpaca assets, active AND inactive, on NYSE/NASDAQ/AMEX (ETF-like names removed)
  * every S&P 500 member since 2016 (fja05680/sp500, MIT licence, vendored in data/reference)
  * symbols removed via mergers / worthless removal in Alpaca corporate actions (good from ~2020)
Delisted names keep their price history in Alpaca's data API, so including them
means a stock that later died is still eligible on the dates it was liquid.

Every eligibility decision for date D uses only daily bars dated strictly before D.

Known residual bias (reported, not fixed): small/mid caps delisted before ~2020 that were
never in the S&P 500 and are absent from Alpaca's asset list are missing.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_START, REFERENCE_DIR

CANDIDATES_FILE = CACHE_DIR / "reference" / "candidates.parquet"
FEATURES_FILE = CACHE_DIR / "features" / "daily_features.parquet"

_NON_COMMON = re.compile(
    r"\b(ETF|ETN|FUND|TRUST UNITS?|ISHARES|SPDR|PROSHARES|DIREXION|INVESCO|VANGUARD|WISDOMTREE|"
    r"GLOBAL X|VANECK|FIRST TRUST|LEVERAGED|INVERSE|2X|3X|WARRANTS?|RIGHTS?|UNITS?|"
    r"PREFERRED|DEPOSITARY SHARES|NOTES DUE|SENIOR NOTES|ACQUISITION CORP)\b",
    re.I,
)
_SYMBOL_OK = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def is_common_stock_like(symbol: str, name: str | None) -> bool:
    if not _SYMBOL_OK.match(symbol):
        return False
    if len(symbol) == 5 and symbol.endswith("X"):  # mutual-fund convention
        return False
    if isinstance(name, str) and _NON_COMMON.search(name):
        return False
    return True


def build_candidates(client) -> pd.DataFrame:
    rows = []
    for status in ("active", "inactive"):
        for a in client.assets(status):
            if a["exchange"] in ("NYSE", "NASDAQ", "AMEX"):
                rows.append({"symbol": a["symbol"], "name": a["name"], "source": f"alpaca_{status}"})
    sp = pd.read_csv(REFERENCE_DIR / "sp500_ticker_start_end.csv")
    sp = sp[sp["end_date"].isna() | (pd.to_datetime(sp["end_date"]) >= pd.Timestamp(DATA_START))]
    rows += [{"symbol": s, "name": None, "source": "sp500_history"} for s in sp["ticker"]]
    ca = client.corporate_actions("cash_merger,stock_merger,stock_and_cash_merger,worthless_removal",
                                  DATA_START.isoformat(), dt.date.today().isoformat())
    for c in ca:
        s = c.get("acquiree_symbol") or c.get("symbol")
        if s:
            rows.append({"symbol": s, "name": None, "source": "corp_action_removed"})
    df = pd.DataFrame(rows)
    df = df[[is_common_stock_like(s, n) for s, n in zip(df["symbol"], df["name"])]]
    df = df.groupby("symbol", as_index=False).agg(name=("name", "first"), source=("source", lambda x: ",".join(sorted(set(x)))))
    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CANDIDATES_FILE, index=False)
    return df


def load_candidates() -> pd.DataFrame:
    return pd.read_parquet(CANDIDATES_FILE)


# ------------------------------------------------------------------ daily features
def compute_daily_features(raw: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Per symbol-date features known AT THE CLOSE of `date` (to be used from the next session).

    raw: unadjusted daily bars; adj: split+dividend adjusted daily bars (same symbols/dates).
    Returns columns: symbol, date, close (raw), adv14 (split-adj shares), atr14 (in raw $),
    dollar_vol20 (median raw $ volume), n_hist, ret1 (adj close-to-close).
    """
    r = raw[["symbol", "date", "open", "close", "volume"]].rename(
        columns={"open": "open_raw", "close": "close_raw", "volume": "volume_raw"})
    a = adj[["symbol", "date", "open", "high", "low", "close", "volume"]]
    df = a.merge(r, on=["symbol", "date"], how="inner").sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    prev_close = g["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    tr = tr.fillna(df["high"] - df["low"])
    atr_adj = tr.groupby(df["symbol"], sort=False).transform(lambda s: s.rolling(14, min_periods=14).mean())
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "date": df["date"],
        "close": df["close_raw"],
        "adv14": g["volume"].transform(lambda s: s.rolling(14, min_periods=14).mean()),
        "atr14": atr_adj * (df["close_raw"] / df["close"]),
        "dollar_vol20": (df["close_raw"] * df["volume_raw"]).groupby(df["symbol"], sort=False)
        .transform(lambda s: s.rolling(20, min_periods=20).median()),
        "n_hist": g.cumcount() + 1,
        "ret1": df["close"] / prev_close - 1.0,
        "ret_overnight": df["open"] / prev_close - 1.0,
        "ret_intraday": df["close"] / df["open"] - 1.0,
    })
    return out


def features_asof(features: pd.DataFrame, date: dt.date, prev_session: dt.date) -> pd.DataFrame:
    """Rows usable for trading on `date`: computed at the close of the previous session.

    A symbol whose last row is older than `prev_session` (halted / not trading) is excluded.
    """
    f = features[features["date"] == prev_session]
    assert (f["date"] < date).all()
    return f


def orb_eligible(feat_prev: pd.DataFrame, min_price: float = 5.0, min_adv: float = 1_000_000,
                 min_atr: float = 0.50) -> list[str]:
    """Zarattini-Barbon-Aziz stocks-in-play pre-filters (applied to prior-day data)."""
    f = feat_prev
    m = (f["close"] > min_price) & (f["adv14"] >= min_adv) & (f["atr14"] > min_atr)
    return sorted(f.loc[m, "symbol"])


def liquid_top_n(feat_prev: pd.DataFrame, n: int, min_price: float = 5.0, min_hist: int = 60) -> list[str]:
    """Top-n by trailing 20-day median dollar volume, as known before the formation date."""
    f = feat_prev[(feat_prev["close"] >= min_price) & (feat_prev["n_hist"] >= min_hist)]
    f = f.dropna(subset=["dollar_vol20"]).sort_values("dollar_vol20", ascending=False)
    return list(f["symbol"].head(n))
