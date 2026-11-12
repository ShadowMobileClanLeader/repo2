"""Bar download + local parquet cache.

Layout
  cache/daily/{adjustment}/{year}.parquet            all candidate symbols, one row per symbol-day
  cache/daily/{adjustment}/{year}.symbols.json       symbols requested for that year (incl. ones with no data)
  cache/intraday/{dataset}/{year}/{date}.parquet      bars for one session, many symbols
  cache/intraday/{dataset}/{year}/{date}.symbols.json symbols requested for that session

`dataset` = "{feed}_{timeframe}_{adjustment}_{window}", e.g. "sip_1Min_raw_rth" or
"sip_5Min_split_open5" (only the 09:30 5-minute bar).

Research loaders (load_*) enforce the holdout lock; fetch_* (download) functions do not
return data to research code and are allowed to fill the cache for the holdout period.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .alpaca_client import BAR_COLUMNS, AlpacaClient, bars_frame
from .calendar import Session, sessions
from .config import CACHE_DIR, UTC
from .splits import check_range

WINDOWS = {
    "rth": None,          # full regular session [open, close)
    "open5": 5,           # first 5 minutes only
}


@dataclass(frozen=True)
class IntradaySpec:
    feed: str = "sip"
    timeframe: str = "1Min"
    adjustment: str = "raw"
    window: str = "rth"

    @property
    def name(self) -> str:
        return f"{self.feed}_{self.timeframe}_{self.adjustment}_{self.window}"

    def bounds(self, s: Session) -> tuple[str, str]:
        start = s.open.tz_convert(UTC)
        minutes = WINDOWS[self.window]
        end = s.close if minutes is None else s.open + pd.Timedelta(minutes=minutes)
        # Alpaca's `end` is inclusive of bars starting at `end`: stop one second before.
        end = end.tz_convert(UTC) - pd.Timedelta(seconds=1)
        return start.isoformat(), end.isoformat()


SPY_1MIN = IntradaySpec("sip", "1Min", "raw", "rth")
OPEN5 = IntradaySpec("sip", "5Min", "split", "open5")


def _intraday_paths(spec: IntradaySpec, day: dt.date) -> tuple[Path, Path]:
    d = CACHE_DIR / "intraday" / spec.name / str(day.year)
    return d / f"{day}.parquet", d / f"{day}.symbols.json"


def _read_manifest(p: Path) -> set[str]:
    return set(json.loads(p.read_text())) if p.exists() else set()


def _empty() -> pd.DataFrame:
    return bars_frame([])


def cached_symbols(spec: IntradaySpec, day: dt.date) -> set[str]:
    return _read_manifest(_intraday_paths(spec, day)[1])


def fetch_intraday_day(client: AlpacaClient, spec: IntradaySpec, session: Session,
                       symbols: Iterable[str], batch: int = 25) -> int:
    """Download any not-yet-cached symbols for one session. Returns rows added."""
    pq, man = _intraday_paths(spec, session.date)
    have = _read_manifest(man)
    need = sorted(set(symbols) - have)
    if not need:
        return 0
    start, end = spec.bounds(session)
    frames = [client.bars(need[i:i + batch], spec.timeframe, start, end, spec.feed, spec.adjustment)
              for i in range(0, len(need), batch)]
    new = pd.concat(frames, ignore_index=True) if frames else _empty()
    old = pd.read_parquet(pq) if pq.exists() else _empty()
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    pq.parent.mkdir(parents=True, exist_ok=True)
    out = out.drop_duplicates(["symbol", "ts"], keep="last")   # safe if a previous run died mid-write
    out.sort_values(["symbol", "ts"]).to_parquet(pq, index=False)
    man.write_text(json.dumps(sorted(have | set(need))))
    return len(new)


def fetch_intraday(client: AlpacaClient, spec: IntradaySpec, symbols_by_day: dict[dt.date, Iterable[str]],
                   batch: int = 25, workers: int = 8, progress: bool = True) -> None:
    """Parallel over sessions (Alpaca latency scales with symbols, so we parallelise days)."""
    sess = {s.date: s for s in sessions(min(symbols_by_day), max(symbols_by_day))}
    todo = [(sess[d], syms) for d, syms in symbols_by_day.items() if d in sess]
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_intraday_day, client, spec, s, syms, batch) for s, syms in todo]
        for f in as_completed(futs):
            f.result()
            done += 1
            if progress and done % 50 == 0:
                print(f"[{spec.name}] {done}/{len(todo)} sessions", flush=True)


def load_intraday(spec: IntradaySpec, day: dt.date, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Read one cached session (holdout-guarded). Raises if requested symbols were never fetched."""
    check_range(day, day)
    pq, man = _intraday_paths(spec, day)
    if symbols is not None:
        missing = set(symbols) - _read_manifest(man)
        if missing:
            raise KeyError(f"{spec.name} {day}: not cached for {sorted(missing)[:10]} (+{max(0, len(missing) - 10)})")
    if not pq.exists():
        return _empty()
    df = pd.read_parquet(pq)
    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------- daily
def _daily_paths(adjustment: str, year: int) -> tuple[Path, Path]:
    d = CACHE_DIR / "daily" / adjustment
    return d / f"{year}.parquet", d / f"{year}.symbols.json"


def fetch_daily_year(client: AlpacaClient, year: int, symbols: Iterable[str], adjustment: str,
                     batch: int = 200, workers: int = 8, end: dt.date | None = None) -> None:
    pq, man = _daily_paths(adjustment, year)
    have = _read_manifest(man)
    need = sorted(set(symbols) - have)
    if not need:
        return
    start = f"{year}-01-01"
    stop = min(dt.date(year, 12, 31), end or dt.date(year, 12, 31)).isoformat()
    chunks = [need[i:i + batch] for i in range(0, len(need), batch)]
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(lambda c: client.bars(c, "1Day", start, stop, "sip", adjustment), chunks))
    new = pd.concat(frames, ignore_index=True)
    new["date"] = new["ts"].dt.tz_convert("America/New_York").dt.date
    old = pd.read_parquet(pq) if pq.exists() else None
    out = pd.concat([old, new], ignore_index=True) if old is not None else new
    pq.parent.mkdir(parents=True, exist_ok=True)
    out.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).to_parquet(pq, index=False)
    man.write_text(json.dumps(sorted(have | set(need))))


def load_daily(adjustment: str, start: dt.date, end: dt.date, symbols: Iterable[str] | None = None,
               guard: bool = True) -> pd.DataFrame:
    """Daily bars in [start, end]. `guard=False` is only for universe construction, which
    uses data strictly before each formation date (tested) and never returns holdout prices."""
    if guard:
        check_range(start, end)
    frames = []
    for y in range(start.year, end.year + 1):
        pq, _ = _daily_paths(adjustment, y)
        if pq.exists():
            frames.append(pd.read_parquet(pq))
    if not frames:
        return pd.DataFrame(columns=BAR_COLUMNS + ["date"])
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]
    return df.reset_index(drop=True)
