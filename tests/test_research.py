import datetime as dt

import numpy as np
import pandas as pd

from intraday import research
from intraday.splits import Fold


def _returns(start, end, mu):
    idx = pd.bdate_range(start, end).date
    rng = np.random.default_rng(abs(hash((mu, start))) % 2**32)
    return pd.Series(mu + rng.normal(0, 0.01, len(idx)), index=idx)


def test_walk_forward_selects_on_train_and_reports_test_only():
    folds = [Fold(dt.date(2020, 1, 1), dt.date(2020, 12, 31), dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
             Fold(dt.date(2021, 1, 1), dt.date(2021, 12, 31), dt.date(2022, 1, 1), dt.date(2022, 6, 30))]
    calls = []

    def run(p, start, end):
        calls.append((p["mu"], start, end))
        # param "good" is only good in 2020 (train of fold 1); afterwards it is bad
        mu = p["mu"] if start.year == 2020 else -p["mu"]
        return _returns(start, end, mu)

    oos, table = research.walk_forward(run, [{"mu": 0.002}, {"mu": -0.002}], folds=folds)
    assert table.loc[0, "chosen"] == {"mu": 0.002}         # picked on 2020 train data
    assert table.loc[0, "test_sharpe"] < 0                  # ...and it fails out of sample
    assert oos.index.min() >= dt.date(2021, 1, 1)           # only test windows are stitched
    assert oos.index.is_monotonic_increasing


def test_record_trial_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "TRIALS_FILE", tmp_path / "t.jsonl")
    monkeypatch.setattr(research, "RETURNS_DIR", tmp_path / "r")
    r = _returns("2021-01-01", "2021-03-01", 0.001)
    period = (dt.date(2021, 1, 1), dt.date(2021, 3, 1))
    research.record_trial("s", {"a": 1}, period, "base", r)
    research.record_trial("s", {"a": 1}, period, "base", r)
    research.record_trial("s", {"a": 2}, period, "base", r)
    assert len(research.load_trials()) == 2
