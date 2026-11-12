"""Trading sessions (with early closes) from Alpaca's calendar, cached to parquet."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .config import CACHE_DIR, DATA_START, ET, FLATTEN_MINUTES_BEFORE_CLOSE, HOLDOUT_END

CAL_FILE = CACHE_DIR / "calendar.parquet"


@dataclass(frozen=True)
class Session:
    date: dt.date
    open: pd.Timestamp   # tz-aware ET
    close: pd.Timestamp  # tz-aware ET

    @property
    def flatten_time(self) -> pd.Timestamp:
        """All positions must be flat by this time (15:55 normally, 12:55 on half days)."""
        return self.close - pd.Timedelta(minutes=FLATTEN_MINUTES_BEFORE_CLOSE)

    @property
    def is_early_close(self) -> bool:
        return self.close.hour < 16


def download_calendar(client, start: dt.date = DATA_START, end: dt.date | None = None) -> pd.DataFrame:
    end = end or (HOLDOUT_END + dt.timedelta(days=400))
    rows = client.calendar(start.isoformat(), end.isoformat())
    df = pd.DataFrame(rows)[["date", "open", "close"]]
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CAL_FILE, index=False)
    _calendar_frame.cache_clear()
    return df


@lru_cache(maxsize=1)
def _calendar_frame() -> pd.DataFrame:
    if not CAL_FILE.exists():
        from .alpaca_client import AlpacaClient
        download_calendar(AlpacaClient())
    return pd.read_parquet(CAL_FILE)


def make_session(date: str | dt.date, open_hhmm: str = "09:30", close_hhmm: str = "16:00") -> Session:
    d = pd.Timestamp(date).date()
    o = pd.Timestamp(f"{d} {open_hhmm}").tz_localize(ET)
    c = pd.Timestamp(f"{d} {close_hhmm}").tz_localize(ET)
    return Session(d, o, c)


def sessions(start: dt.date, end: dt.date) -> list[Session]:
    df = _calendar_frame()
    out = []
    for r in df.itertuples(index=False):
        d = pd.Timestamp(r.date).date()
        if start <= d <= end:
            out.append(make_session(d, r.open, r.close))
    return out


def previous_sessions(date: dt.date, n: int) -> list[Session]:
    """The n sessions strictly before `date` (oldest first)."""
    allsess = sessions(date - dt.timedelta(days=n * 2 + 15), date - dt.timedelta(days=1))
    return allsess[-n:]
