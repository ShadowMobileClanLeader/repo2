"""Thin REST client for Alpaca PAPER trading + market data.

Safety: the trading base URL is hard-wired to the paper endpoint and checked
on every request. If APCA_API_BASE_URL is set to anything else, we refuse to
start. Keys come only from environment variables.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Iterable

import pandas as pd
import requests

from .config import DATA_URL, PAPER_TRADING_URL


class LiveEndpointError(RuntimeError):
    """Raised whenever anything other than the paper endpoint is configured."""


def assert_paper_url(url: str) -> None:
    if not url.rstrip("/").startswith(PAPER_TRADING_URL):
        raise LiveEndpointError(
            f"Refusing to use trading endpoint {url!r}: only {PAPER_TRADING_URL} is allowed."
        )


def _keys_from_env() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    return key, secret


class RateLimiter:
    """Sliding-window limiter shared across threads (Alpaca free plan: 200 req/min)."""

    def __init__(self, max_calls: int = 190, period: float = 60.0):
        self.max_calls, self.period = max_calls, period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0]) + 0.01
            time.sleep(wait)


class AlpacaClient:
    def __init__(self, trading_url: str = PAPER_TRADING_URL, data_url: str = DATA_URL,
                 rate_limiter: RateLimiter | None = None):
        env_url = os.environ.get("APCA_API_BASE_URL")
        if env_url:
            assert_paper_url(env_url)
        assert_paper_url(trading_url)
        self.trading_url = trading_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        key, secret = _keys_from_env()
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._local = threading.local()
        self.limiter = rate_limiter or RateLimiter()

    # -- low level ----------------------------------------------------------
    @property
    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self._headers)
            self._local.session = s
        return s

    def _request(self, method: str, base: str, path: str, **kw: Any) -> Any:
        if base == "trading":
            assert_paper_url(self.trading_url)
            url = self.trading_url + path
        else:
            url = self.data_url + path
        for attempt in range(6):
            self.limiter.acquire()
            try:
                r = self._session.request(method, url, timeout=60, **kw)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if not r.ok:
                raise requests.HTTPError(f"{r.status_code} {url}: {r.text[:300]}", response=r)
            return r.json() if r.content else None
        raise RuntimeError(f"Request failed after retries: {method} {url}")

    def trading_get(self, path: str, **params: Any) -> Any:
        return self._request("GET", "trading", path, params=params)

    def trading_post(self, path: str, payload: dict) -> Any:
        return self._request("POST", "trading", path, json=payload)

    def trading_delete(self, path: str) -> Any:
        return self._request("DELETE", "trading", path)

    def data_get(self, path: str, **params: Any) -> Any:
        return self._request("GET", "data", path, params=params)

    # -- account / reference -----------------------------------------------
    def account(self) -> dict:
        return self.trading_get("/v2/account")

    def calendar(self, start: str, end: str) -> list[dict]:
        return self.trading_get("/v2/calendar", start=start, end=end)

    def assets(self, status: str = "active") -> list[dict]:
        return self.trading_get("/v2/assets", status=status, asset_class="us_equity")

    def corporate_actions(self, types: str, start: str, end: str) -> list[dict]:
        out: list[dict] = []
        token = None
        while True:
            p = {"types": types, "start": start, "end": end, "limit": 1000}
            if token:
                p["page_token"] = token
            j = self.data_get("/v1/corporate-actions", **p)
            for kind, rows in (j.get("corporate_actions") or {}).items():
                for row in rows:
                    out.append({"type": kind, **row})
            token = j.get("next_page_token")
            if not token:
                return out

    # -- bars ------------------------------------------------------------------
    def bars(self, symbols: Iterable[str], timeframe: str, start: str, end: str,
             feed: str = "sip", adjustment: str = "raw") -> pd.DataFrame:
        """Fetch bars for many symbols (paginated). Timestamps are bar START, UTC."""
        syms = ",".join(sorted(set(symbols)))
        rows: list[dict] = []
        token = None
        while True:
            p = {"symbols": syms, "timeframe": timeframe, "start": start, "end": end,
                 "feed": feed, "adjustment": adjustment, "limit": 10000}
            if token:
                p["page_token"] = token
            j = self.data_get("/v2/stocks/bars", **p)
            for sym, bars in (j.get("bars") or {}).items():
                for b in bars:
                    rows.append({"symbol": sym, "ts": b["t"], "open": b["o"], "high": b["h"],
                                 "low": b["l"], "close": b["c"], "volume": b["v"],
                                 "trade_count": b.get("n"), "vwap": b.get("vw")})
            token = j.get("next_page_token")
            if not token:
                break
        return bars_frame(rows)


BAR_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def bars_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=BAR_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ["open", "high", "low", "close", "vwap"]:
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["trade_count"] = df["trade_count"].astype("float64")
    df["symbol"] = df["symbol"].astype("string")
    return df
