import pytest

from intraday.costs import CostModel


def test_slippage_is_adverse_and_symbol_specific():
    c = CostModel(default_slippage_bps=5.0, symbol_slippage_bps={"SPY": 1.5})
    assert c.fill_price("SPY", 100.0, +10) == pytest.approx(100.015)
    assert c.fill_price("SPY", 100.0, -10) == pytest.approx(99.985)
    assert c.fill_price("XYZ", 100.0, +10) == pytest.approx(100.05)
    assert c.fill_price("XYZ", 100.0, -10) == pytest.approx(99.95)


def test_stress_multiplies_slippage_only():
    c = CostModel(default_slippage_bps=4.0, symbol_slippage_bps={})
    s2, s3 = c.stressed(2), c.stressed(3)
    assert s2.slippage_bps("A") == pytest.approx(8.0)
    assert s3.slippage_bps("A") == pytest.approx(12.0)
    assert s3.fees(100, 50.0) == c.fees(100, 50.0)


def test_commission_minimum_and_per_share():
    c = CostModel(commission_per_share=0.0035, min_commission=0.35, sec_fee_rate=0, taf_per_share=0)
    assert c.fees(10, 100.0) == pytest.approx(0.35)          # minimum applies
    assert c.fees(1000, 100.0) == pytest.approx(3.5)


def test_commission_cap_pct_of_value():
    c = CostModel(commission_per_share=0.0035, min_commission=0.35, max_commission_pct=0.01,
                  sec_fee_rate=0, taf_per_share=0)
    assert c.fees(10, 1.0) == pytest.approx(0.10)             # 1% of $10 < $0.35 minimum


def test_regulatory_fees_only_on_sells():
    c = CostModel(commission_per_share=0.0, min_commission=0.0, sec_fee_rate=27.8e-6,
                  taf_per_share=0.000166, taf_max=8.30)
    assert c.fees(1000, 50.0) == 0.0
    assert c.fees(-1000, 50.0) == pytest.approx(27.8e-6 * 50_000 + 0.166)
    assert c.fees(-10_000_000, 1.0) == pytest.approx(27.8e-6 * 10_000_000 + 8.30)  # TAF capped


def test_zero_qty_has_no_cost():
    assert CostModel().fees(0, 100.0) == 0.0


def test_per_share_slippage_overrides_bps_and_stresses():
    c = CostModel(symbol_slippage_bps={"SPY": 1.5}, symbol_slippage_per_share={"SPY": 0.005})
    assert c.fill_price("SPY", 400.0, +10) == pytest.approx(400.005)
    assert c.fill_price("SPY", 400.0, -10) == pytest.approx(399.995)
    assert c.stressed(3).fill_price("SPY", 400.0, +10) == pytest.approx(400.015)
    assert c.fill_price("XYZ", 100.0, +1) == pytest.approx(100.05)


def test_time_of_day_profile_interpolates():
    import pandas as pd
    c = CostModel(symbol_slippage_bps={}, tod_profile=((5, 15.0), (30, 8.0), (384, 2.0)))
    t = lambda hhmm: pd.Timestamp(f"2024-03-04 {hhmm}").tz_localize("America/New_York").tz_convert("UTC")
    assert c.slippage_bps("X", t("09:35")) == pytest.approx(15.0)
    assert c.slippage_bps("X", t("09:31")) == pytest.approx(15.0)             # flat before first point
    assert c.slippage_bps("X", t("09:47:30")) == pytest.approx(15 - 7 * (12 / 25), abs=0.3)
    assert c.slippage_bps("X", t("15:54")) == pytest.approx(2.0)
    assert c.stressed(2).slippage_bps("X", t("15:54")) == pytest.approx(4.0)
    assert c.slippage_bps("X") == pytest.approx(5.0)                          # no timestamp -> default
    assert c.fill_price("X", 100.0, 1, t("09:35")) == pytest.approx(100.15)
