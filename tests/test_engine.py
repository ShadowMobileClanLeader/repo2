import pandas as pd
import pytest

from intraday.costs import CostModel
from intraday.engine import Engine, RiskLimits, Target
from intraday.sim_broker import SimBroker

from helpers import HALF_DAY, SESSION, ScriptedStrategy, flat_day

FREE = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0,
                 min_commission=0, sec_fee_rate=0, taf_per_share=0)


def run(script, events, session=SESSION, costs=FREE, risk=RiskLimits(), capital=100_000.0):
    strat = ScriptedStrategy(script)
    strat.prepare_session(session, None)
    eng = Engine(strat, SimBroker(costs), capital, risk)
    res = eng.run_session(session, events)
    return eng, strat, res


def test_signal_fills_on_next_bar_open_not_signal_bar():
    # bar 0 closes at 9:31 with close 100; bar 1 opens at 101
    ev = flat_day(SESSION, overrides={0: (100, 100, 100, 100), 1: (101, 101, 101, 101)}, price=101)
    eng, _, _ = run({1: {"AAA": Target(0.5, reason="test")}}, ev)
    entry = eng.fills[0]
    assert entry.ref_price == 101 and entry.ts == ev[1][0]
    assert entry.qty == int(0.5 * 100_000 / 100)   # sized on the signal price, filled next open


def test_flat_by_1555_and_no_decisions_after_deadline():
    ev = flat_day(SESSION, price=100)
    eng, strat, res = run({1: {"AAA": Target(1.0)}}, ev)
    exit_fill = eng.fills[-1]
    assert exit_fill.purpose == "flatten"
    exit_time = exit_fill.ts.tz_convert("America/New_York")
    assert exit_time.strftime("%H:%M") == "15:54"          # fills at 15:54 open -> flat by 15:55
    assert max(strat.calls).strftime("%H:%M") < "15:54"
    assert not eng.portfolio.positions


def test_half_day_flattens_before_1255():
    ev = flat_day(HALF_DAY, price=100)
    eng, strat, _ = run({1: {"AAA": Target(1.0)}}, ev, session=HALF_DAY)
    assert eng.fills[-1].ts.tz_convert("America/New_York").strftime("%H:%M") == "12:54"


def test_pnl_accounting_long_and_short_with_costs():
    costs = CostModel(default_slippage_bps=10, symbol_slippage_bps={}, commission_per_share=0.01,
                      min_commission=0, sec_fee_rate=0, taf_per_share=0)
    ov = {m: (110, 110, 110, 110) for m in range(100, 390)}
    ev = flat_day(SESSION, price=100, overrides=ov)
    eng, _, res = run({1: {"AAA": Target(0.5)}}, ev, costs=costs)
    q = 500
    expected = q * (110 * (1 - 0.001) - 100 * (1 + 0.001)) - 0.01 * q * 2
    assert res.eod_equity - res.sod_equity == pytest.approx(expected)
    assert res.fees == pytest.approx(0.01 * q * 2)

    eng, _, res = run({1: {"AAA": Target(-0.5)}}, ev, costs=costs)
    expected_short = -q * (110 * (1 + 0.001) - 100 * (1 - 0.001)) - 0.01 * q * 2
    assert res.eod_equity - res.sod_equity == pytest.approx(expected_short)


def test_protective_stop_from_target():
    ov = {1: (100, 100, 100, 100), 50: (99, 99, 94, 95)}
    ev = flat_day(SESSION, price=100, overrides={**ov, **{m: (95, 95, 95, 95) for m in range(51, 390)}})
    eng, _, _ = run({1: {"AAA": Target(0.5, stop_loss=96.0)}}, ev)
    stop = [f for f in eng.fills if f.purpose == "stop_loss"]
    assert len(stop) == 1 and stop[0].ref_price == 96.0 and stop[0].qty == -500
    assert [f for f in eng.fills if f.purpose == "flatten"] == []


def test_entry_stop_only_fills_when_crossed():
    ov = {10: (100, 100.5, 99.5, 100), 20: (100.5, 102, 100.4, 101.5)}
    ev = flat_day(SESSION, price=100, overrides={**ov, **{m: (101.5,) * 4 for m in range(21, 390)}})
    eng, _, _ = run({1: {"AAA": Target(0.5, entry_stop=101.0, stop_loss=99.0)}}, ev)
    entry = eng.fills[0]
    assert entry.purpose == "entry" and entry.ref_price == 101.0 and entry.ts == ev[20][0]
    assert entry.qty == int(0.5 * 100_000 / 101.0)
    assert eng.fills[-1].purpose == "flatten"


def test_target_zero_cancels_pending_entry():
    ev = flat_day(SESSION, price=100, overrides={30: (100, 105, 100, 104)})
    eng, _, _ = run({1: {"AAA": Target(0.5, entry_stop=101.0)}, 10: {"AAA": Target(0.0)}}, ev)
    assert eng.fills == []


def test_gross_leverage_cap_scales_targets():
    ev = flat_day(SESSION, price=100)
    eng, _, _ = run({1: {"AAA": Target(3.0)}}, ev, risk=RiskLimits(max_gross_leverage=1.0, max_symbol_weight=5))
    assert eng.fills[0].qty == 1000


def test_daily_loss_limit_flattens_and_halts():
    ov = {m: (90, 90, 90, 90) for m in range(30, 390)}
    ev = flat_day(SESSION, price=100, overrides=ov)
    eng, strat, res = run({1: {"AAA": Target(1.0)}, 100: {"AAA": Target(1.0)}}, ev,
                          risk=RiskLimits(max_daily_loss=0.05))
    assert "daily loss" in res.halted
    assert eng.fills[1].purpose == "flatten" and eng.fills[1].ts == ev[31][0]
    assert all(c.strftime("%H:%M") <= "10:00" for c in strat.calls)


def test_kill_switch_file(tmp_path):
    ks = tmp_path / "KILL"
    ev = flat_day(SESSION, price=100)

    strat = ScriptedStrategy({1: {"AAA": Target(1.0)}})
    eng = Engine(strat, SimBroker(FREE), 100_000, RiskLimits(kill_switch_file=ks))

    def events():
        for i, e in enumerate(ev):
            if i == 60:
                ks.write_text("stop")
            yield e

    res = eng.run_session(SESSION, events())
    assert "kill switch" in res.halted
    assert eng.fills[-1].purpose == "kill"
    assert eng.fills[-1].ts == ev[61][0]


def test_rebalance_does_not_double_count_pending_orders():
    # AAA has no bars at minutes 1-4, so the first order is still pending when the
    # strategy repeats/changes its target; only the net difference may be ordered.
    from intraday.engine import Bar
    ev = []
    for t, bars in flat_day(SESSION, price=100):
        m = int((t - SESSION.open.tz_convert("UTC")) / pd.Timedelta(minutes=1))
        bars = dict(bars) if not 1 <= m <= 4 else {}
        bars["BBB"] = Bar("BBB", t, 10, 10, 10, 10, 1)
        ev.append((t, bars))
    eng, _, _ = run({1: {"AAA": Target(0.5)}, 2: {"AAA": Target(0.5)}, 3: {"AAA": Target(0.25)}}, ev)
    assert sorted(o.qty for o in eng.orders if o.symbol == "AAA" and o.purpose != "flatten") == [-250, 500]
    assert eng.fills[0].qty + eng.fills[1].qty == 250 and eng.fills[0].ts == ev[5][0]


def test_no_fill_after_last_bar_is_closed_at_last_price():
    ev = flat_day(SESSION, price=100)[:200]     # data stops mid-day (e.g. halt)
    eng, _, res = run({1: {"AAA": Target(0.5)}}, ev)
    assert eng.fills[-1].note == "eod_forced_at_last_price"
    assert not eng.portfolio.positions


def test_engine_tracks_offset_stop_without_duplicating_orders():
    ov = {10: (100, 101.5, 99.9, 101.2), **{m: (101.2,) * 4 for m in range(11, 40)},
          40: (101.0, 101.0, 100.0, 100.2), **{m: (100.2,) * 4 for m in range(41, 390)}}
    ev = flat_day(SESSION, price=100.0, overrides=ov)
    eng, _, _ = run({1: {"AAA": Target(0.5, entry_stop=101.0, stop_offset=0.5)}}, ev)
    entry, stop = eng.fills[0], eng.fills[1]
    assert entry.ref_price == 101.0 and stop.purpose == "stop_loss" and stop.ref_price == pytest.approx(100.5)
    assert len(eng.fills) == 2
    assert [o for o in eng.orders if o.purpose == "stop_loss"] == []   # the broker's OTO child was reused
