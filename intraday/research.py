"""Research bookkeeping: every backtest variant is recorded so Phase 4 can apply a
multiple-testing adjustment (deflated Sharpe) over ALL trials, not just the survivors."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Callable

import pandas as pd

from . import metrics
from .config import ROOT
from .splits import Fold, walk_forward_folds

TRIALS_FILE = ROOT / "research" / "trials.jsonl"
RETURNS_DIR = ROOT / "research" / "returns"


def trial_id(strategy: str, params: dict, period: tuple[dt.date, dt.date], costs: str) -> str:
    key = json.dumps({"s": strategy, "p": params, "per": [str(period[0]), str(period[1])], "c": costs},
                     sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def record_trial(strategy: str, params: dict, period: tuple[dt.date, dt.date], costs: str,
                 returns: pd.Series, extra: dict | None = None, spy: pd.Series | None = None) -> dict:
    """Append one trial (idempotent per id) and save its daily returns to parquet."""
    tid = trial_id(strategy, params, period, costs)
    row = {"id": tid, "strategy": strategy, "params": params, "start": str(period[0]), "end": str(period[1]),
           "costs": costs, "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           **metrics.summary(returns, spy), **(extra or {})}
    RETURNS_DIR.mkdir(parents=True, exist_ok=True)
    returns.rename("ret").to_frame().to_parquet(RETURNS_DIR / f"{tid}.parquet")
    existing = {r["id"] for r in load_trials_raw()}
    if tid not in existing:
        TRIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRIALS_FILE, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    return row


def load_trials_raw() -> list[dict]:
    if not TRIALS_FILE.exists():
        return []
    return [json.loads(line) for line in TRIALS_FILE.read_text().splitlines() if line.strip()]


def load_trials() -> pd.DataFrame:
    return pd.DataFrame(load_trials_raw())


def walk_forward(run: Callable[[dict, dt.date, dt.date], pd.Series], grid: list[dict],
                 select: Callable[[pd.Series], float] = metrics.sharpe,
                 folds: list[Fold] | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """For each fold: evaluate every param set on the TRAIN window, pick the best by `select`,
    then run it on the unseen TEST window. Returns the stitched out-of-sample daily returns and
    a table of choices per fold. `run(params, start, end)` must return daily returns."""
    folds = folds or walk_forward_folds()
    cache: dict[tuple, pd.Series] = {}
    oos, rows = [], []
    for f in folds:
        scores = []
        for p in grid:
            k = (json.dumps(p, sort_keys=True), f.train_start, f.train_end)
            if k not in cache:
                cache[k] = run(p, f.train_start, f.train_end)
            scores.append(select(cache[k]))
        best = grid[max(range(len(grid)), key=lambda i: (scores[i] if scores[i] == scores[i] else -1e9))]
        r = run(best, f.test_start, f.test_end)
        oos.append(r)
        rows.append({"test_start": f.test_start, "test_end": f.test_end, "chosen": best,
                     "train_score": max(s for s in scores if s == s) if any(s == s for s in scores) else float("nan"),
                     "test_sharpe": metrics.sharpe(r)})
    return pd.concat(oos).sort_index(), pd.DataFrame(rows)
