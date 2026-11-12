"""Development / holdout split and walk-forward folds.

The holdout is locked: research code that loads data overlapping the holdout
raises HoldoutLockedError unless the rules have been frozen (FROZEN_RULES_FILE
exists) and unlock_holdout() was called with a reason, which is logged.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from . import config


class HoldoutLockedError(RuntimeError):
    pass


_state = {"unlocked": False}


def holdout_unlocked() -> bool:
    return _state["unlocked"]


def unlock_holdout(reason: str) -> None:
    """Unlock holdout reads for this process. Requires frozen rules on disk."""
    if not config.FROZEN_RULES_FILE.exists():
        raise HoldoutLockedError(
            f"Rules are not frozen ({config.FROZEN_RULES_FILE} missing). Freeze rules before touching the holdout."
        )
    config.HOLDOUT_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.HOLDOUT_ACCESS_LOG, "a") as f:
        f.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "reason": reason,
                            "frozen_rules": config.FROZEN_RULES_FILE.read_text()}) + "\n")
    _state["unlocked"] = True


def lock_holdout() -> None:
    _state["unlocked"] = False


def check_range(start: dt.date, end: dt.date) -> None:
    """Raise if [start, end] touches the holdout while it is locked."""
    if end >= config.HOLDOUT_START and not holdout_unlocked():
        raise HoldoutLockedError(
            f"Requested data through {end} overlaps holdout (from {config.HOLDOUT_START}); holdout is locked."
        )


@dataclass(frozen=True)
class Fold:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date


def _add_months(d: dt.date, months: int) -> dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    return dt.date(d.year + y, m + 1, 1)


def walk_forward_folds(start: dt.date = config.DEV_START, end: dt.date = config.DEV_END,
                       train_months: int = 36, test_months: int = 12,
                       anchored: bool = False) -> list[Fold]:
    """Rolling (or anchored/expanding) train windows, each followed by an unseen test window.

    Test windows are contiguous and non-overlapping; the last one is truncated at `end`.
    Fold boundaries fall on the first of a month.
    """
    if end >= config.HOLDOUT_START:
        raise HoldoutLockedError("Walk-forward folds must stay inside the development period.")
    folds = []
    first = dt.date(start.year, start.month, 1)
    test_start = _add_months(first, train_months)
    while test_start <= end:
        test_end = min(_add_months(test_start, test_months) - dt.timedelta(days=1), end)
        train_start = start if anchored else max(start, _add_months(test_start, -train_months))
        folds.append(Fold(train_start, test_start - dt.timedelta(days=1), test_start, test_end))
        test_start = _add_months(test_start, test_months)
    return folds
