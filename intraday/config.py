"""Central configuration: paths, endpoints, research periods.

The research periods are FROZEN here on purpose. The holdout must not move
once research starts, even though "the most recent 12 months" keeps rolling.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# --- Endpoints -------------------------------------------------------------
# Only the PAPER trading endpoint is allowed. There is deliberately no setting
# for the live endpoint anywhere in this code base.
PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("INTRADAY_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
REFERENCE_DIR = ROOT / "data" / "reference"
LOG_DIR = Path(os.environ.get("INTRADAY_LOG_DIR", ROOT / "logs"))

# --- Research periods (frozen 2026-11-12) ---------------------------------
# Alpaca SIP history starts 2016-01-04; the first ~3 months are warm-up for
# 60-day liquidity filters and 14-day lookbacks.
DATA_START = dt.date(2016, 1, 4)
DEV_START = dt.date(2016, 4, 1)
DEV_END = dt.date(2025, 11, 11)
HOLDOUT_START = dt.date(2025, 11, 12)
HOLDOUT_END = dt.date(2026, 11, 11)

# Rules must be frozen (this file must exist) before holdout data can be read.
FROZEN_RULES_FILE = ROOT / "research" / "FROZEN_RULES.json"
HOLDOUT_ACCESS_LOG = ROOT / "research" / "holdout_access.log"

# --- Session timing ---------------------------------------------------------
FLATTEN_MINUTES_BEFORE_CLOSE = 5  # flat by 15:55 ET on normal days
