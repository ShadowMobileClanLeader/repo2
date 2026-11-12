import pytest

from intraday.costs import CostModel
from intraday.engine import Order
from intraday.sim_broker import SimBroker

from helpers import SESSION, bar, ts_at

FREE = CostModel(default_slippage_bps=0, symbol_slippage_bps={}, commission_per_share=0,
                 min_commission=0, sec_fee_rate=0, taf_per_share=0)


def test_market_order_fills_at_next_bar_open_with_slippage():
    b = SimBroker(CostModel(default_slippage_bps=10, symbol_slippage_bps={}))
    b.submit(Order("AAA", 100, "market"))
    fills = b.process_bar(ts_at(SESSION, 1), {"AAA": bar("AAA", SESSION, 1, 50.0, 51, 49, 50.5)})
    assert len(fills) == 1
    assert fills[0].ref_price == 50.0
    assert fills[0].price == pytest.approx(50.0 * 1.001)
    assert not b.open_orders()


def test_order_waits_when_symbol_has_no_bar():
    b = SimBroker(FREE)
    b.submit(Order("AAA", 100, "market"))
    assert b.process_bar(ts_at(SESSION, 1), {"BBB": bar("BBB", SESSION, 1, 10.0)}) == []
    assert len(b.open_orders()) == 1


@pytest.mark.parametrize("o,h,l,expected_ref,note", [
    (99.0, 101.0, 98.0, 100.0, ""),                    # touched intrabar -> stop level
    (102.0, 103.0, 101.0, 102.0, "gap_through_stop"),   # gapped above -> open
])
def test_buy_stop(o, h, l, expected_ref, note):
    b = SimBroker(FREE)
    b.submit(Order("AAA", 10, "stop", 100.0))
    f = b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, o, h, l, o)})
    assert f[0].ref_price == expected_ref and f[0].note == note


def test_buy_stop_not_reached_stays_open():
    b = SimBroker(FREE)
    b.submit(Order("AAA", 10, "stop", 100.0))
    assert b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, 98, 99.99, 97, 99)}) == []
    assert len(b.open_orders()) == 1


@pytest.mark.parametrize("o,h,l,expected_ref", [(101, 102, 99, 100.0), (97, 98, 96, 97.0)])
def test_sell_stop(o, h, l, expected_ref):
    b = SimBroker(FREE)
    b.submit(Order("AAA", -10, "stop", 100.0))
    f = b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, o, h, l, o)})
    assert f[0].ref_price == expected_ref and f[0].qty == -10


def test_attached_stop_same_bar_assumes_worst_case():
    b = SimBroker(FREE)
    b.submit(Order("AAA", 10, "stop", 100.0, attached_stop_loss=99.0))
    # bar trades through both entry (100) and stop (99): assume we got stopped out
    f = b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, 99.5, 100.5, 98.5, 99.0)})
    assert [x.qty for x in f] == [10, -10]
    assert f[1].ref_price == 99.0 and f[1].note == "same_bar_stop_assumed"
    assert not b.open_orders()


def test_attached_stop_rests_when_not_hit():
    b = SimBroker(FREE)
    b.submit(Order("AAA", 10, "market", attached_stop_loss=95.0))
    f = b.process_bar(ts_at(SESSION, 1), {"AAA": bar("AAA", SESSION, 1, 100, 101, 99, 100)})
    assert len(f) == 1
    (child,) = b.open_orders()
    assert child.type == "stop" and child.qty == -10 and child.stop_price == 95.0 and child.purpose == "stop_loss"


def test_short_entry_attached_stop_is_buy_stop_above():
    b = SimBroker(FREE)
    b.submit(Order("AAA", -10, "stop", 100.0, attached_stop_loss=101.0))
    f = b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, 100.5, 100.8, 99.5, 99.8)})
    assert [x.qty for x in f] == [-10]
    (child,) = b.open_orders()
    assert child.qty == 10 and child.stop_price == 101.0
    f2 = b.process_bar(ts_at(SESSION, 6), {"AAA": bar("AAA", SESSION, 6, 100.0, 101.2, 99.9, 101.1)})
    assert f2[0].qty == 10 and f2[0].ref_price == 101.0


def test_end_session_closes_leftover_at_last_price():
    b = SimBroker(FREE)
    b.submit(Order("AAA", 10, "market"))
    b.process_bar(ts_at(SESSION, 1), {"AAA": bar("AAA", SESSION, 1, 100)})
    f = b.end_session({"AAA": 103.0}, ts_at(SESSION, 389))
    assert f[0].qty == -10 and f[0].ref_price == 103.0 and f[0].note == "eod_forced_at_last_price"


def test_offset_stop_is_relative_to_executed_price_after_gap():
    b = SimBroker(FREE)
    b.submit(Order("AAA", 10, "stop", 100.0, attached_stop_offset=0.5))
    # gaps through the entry level: fills at the 101 open, so the stop is 100.5 (not 99.5)
    f = b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, 101.0, 101.5, 100.8, 101.2)})
    assert len(f) == 1 and f[0].ref_price == 101.0
    (child,) = b.open_orders()
    assert child.stop_price == pytest.approx(100.5) and child.qty == -10


def test_offset_stop_for_short_uses_slipped_price():
    b = SimBroker(CostModel(default_slippage_bps=10, symbol_slippage_bps={}, commission_per_share=0,
                            min_commission=0, sec_fee_rate=0, taf_per_share=0))
    b.submit(Order("AAA", -10, "stop", 100.0, attached_stop_offset=1.0))
    f = b.process_bar(ts_at(SESSION, 5), {"AAA": bar("AAA", SESSION, 5, 100.5, 100.6, 99.5, 99.8)})
    (child,) = b.open_orders()
    assert child.stop_price == pytest.approx(f[0].price + 1.0) and f[0].price == pytest.approx(99.9)
