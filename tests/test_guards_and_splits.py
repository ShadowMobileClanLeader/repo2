import datetime as dt

import pytest

from intraday import config, splits
from intraday.alpaca_client import AlpacaClient, LiveEndpointError, assert_paper_url


def test_live_url_rejected():
    with pytest.raises(LiveEndpointError):
        assert_paper_url("https://api.alpaca.markets")
    assert_paper_url("https://paper-api.alpaca.markets")


def test_client_refuses_live_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(LiveEndpointError):
        AlpacaClient()


def test_client_refuses_live_arg(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    with pytest.raises(LiveEndpointError):
        AlpacaClient(trading_url="https://api.alpaca.markets")


def test_holdout_locked_by_default():
    splits.lock_holdout()
    splits.check_range(config.DEV_START, config.DEV_END)
    with pytest.raises(splits.HoldoutLockedError):
        splits.check_range(config.DEV_START, config.HOLDOUT_START)


def test_unlock_requires_frozen_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FROZEN_RULES_FILE", tmp_path / "FROZEN.json")
    monkeypatch.setattr(config, "HOLDOUT_ACCESS_LOG", tmp_path / "access.log")
    with pytest.raises(splits.HoldoutLockedError):
        splits.unlock_holdout("peek")
    (tmp_path / "FROZEN.json").write_text('{"strategy": "x"}')
    splits.unlock_holdout("final evaluation")
    try:
        splits.check_range(config.HOLDOUT_START, config.HOLDOUT_END)
        assert "final evaluation" in (tmp_path / "access.log").read_text()
    finally:
        splits.lock_holdout()


def test_walk_forward_folds_are_sequential_and_inside_dev():
    folds = splits.walk_forward_folds(train_months=36, test_months=12)
    assert folds[0].test_start == dt.date(2019, 4, 1)
    for a, b in zip(folds, folds[1:]):
        assert b.test_start == a.test_end + dt.timedelta(days=1)
    for f in folds:
        assert f.train_end < f.test_start and f.train_start >= config.DEV_START
        assert f.test_end <= config.DEV_END
    assert folds[-1].test_end == config.DEV_END


def test_anchored_folds_start_at_dev_start():
    folds = splits.walk_forward_folds(anchored=True)
    assert all(f.train_start == config.DEV_START for f in folds)


def test_folds_cannot_reach_holdout():
    with pytest.raises(splits.HoldoutLockedError):
        splits.walk_forward_folds(end=config.HOLDOUT_END)
