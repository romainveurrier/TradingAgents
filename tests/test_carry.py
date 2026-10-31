"""The carry engine: portfolio-margin math, book accounting, liquidation, funding model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tradingagents.hyperliquid.carry import funding_model as fm, pm
from tradingagents.hyperliquid.carry.engine import Book, CarryCosts, simulate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("interp", ["borrow", "balance"])
def test_closed_forms_match_the_account(interp):
    p = pm.PMParams(interpretation=interp)
    E, S = 100_000.0, 80_000.0
    for L in (1.5, 2.0, 2.5, 3.0):
        Q = L * E / S
        usdc = E - Q * S
        assert pm.pm_ratio(usdc, Q, S, -Q, S, S, p) == pytest.approx(pm.entry_ratio(L, p) + 20 / (0.75 * L * E), rel=1e-9)
        k = pm.rally_to_liquidation(L, p)
        exact = replace(p, offset=0.0)          # the closed forms ignore the 20 USDC offset
        assert pm.pm_ratio(usdc, Q, S * k, -Q, S * k, S, exact) == pytest.approx(p.trigger, rel=1e-12)
        assert pm.pm_ratio(usdc, Q, S * k * 1.001, -Q, S * k * 1.001, S, exact) > p.trigger
    assert pm.max_entry_leverage(p) == pytest.approx(1 / 0.3)
    assert pm.borrow_rate(0.9) == pytest.approx(0.05 + 4.75 * 0.1)


def _panel(prices, funding=1.25e-5, hours=1):
    idx = pd.date_range("2026-01-01", periods=len(prices) - 1, freq=f"{hours}h", tz="UTC")
    o, c = np.array(prices[:-1], float), np.array(prices[1:], float)
    return pd.DataFrame({"p_open": o, "p_close": c, "p_high": np.maximum(o, c), "p_low": np.minimum(o, c),
                         "s_open": o, "s_close": c, "s_high": np.maximum(o, c), "s_low": np.minimum(o, c),
                         "funding": funding, "known_at": idx + pd.Timedelta(hours=hours)}, index=idx)


def test_each_bar_changes_equity_by_exactly_its_flows():
    panel = _panel([80_000, 80_400, 79_900, 81_000, 80_500])
    costs = CarryCosts(borrow_apy=0.10)
    res = simulate(panel, lambda t, b, x: 2.0, "t", costs=costs, rebalance_frac=None, band=10)
    b = res.book
    q = b.btc
    # Delta-neutral with spot = perp prices: price P&L cancels; equity = start - fees + funding - interest.
    assert res.equity.iloc[-1] == pytest.approx(100_000 - b.fees + b.funding - b.interest, abs=1e-6)
    assert b.funding > 0 and b.interest > 0
    assert q * 80_000 == pytest.approx(2 * 100_000)          # sized on pre-trade equity


def test_a_rally_beyond_the_closed_form_liquidates_at_3x_but_not_at_2x():
    rally = _panel([80_000, 80_000, 80_000 * 1.12], hours=24)
    for L, expect in ((3.0, 1), (2.0, 0)):
        res = simulate(rally, lambda t, b, x, L=L: L, f"L{L}", rebalance_frac=None, band=10, decide_every="10D")
        assert res.book.liquidations == expect, L


def test_open_and_close_costs_only_fees():
    b = Book(usdc=100_000)
    c = CarryCosts()
    b.trade_to(2.5, 80_000, 80_000, c)
    b.trade_to(0.0, 80_000, 80_000, c)
    assert b.equity(80_000, 80_000) == pytest.approx(100_000 - b.fees)
    assert b.btc == 0 and b.perp == 0


def test_the_funding_model_matches_the_venue_and_monte_carlo():
    rng = np.random.default_rng(3)
    z = rng.normal(6e-4, 4e-4, 400_000)
    mc = np.mean(np.sign(z) * np.maximum(np.abs(z) - 5e-4, 0))
    assert float(fm.expected_soft(6e-4, 4e-4)) == pytest.approx(mc, abs=2e-6)
    assert float(fm.funding_from_premium_8h(-3e-4)) == pytest.approx(1e-4)       # inside the band: the floor
    assert float(fm.funding_from_premium_8h(9e-4)) == pytest.approx(4e-4)        # above: P - c


def test_forecast_features_never_read_the_hour_they_forecast():
    idx = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    f = pd.DataFrame({"rate": np.full(400, 1.25e-5), "premium": np.zeros(400)}, index=idx)
    base = fm.features(f)
    g = f.copy()
    g.iloc[300:, :] = 9.0                       # poison from hour 300 on
    poisoned = fm.features(g)
    pd.testing.assert_frame_equal(base.iloc[:301], poisoned.iloc[:301])


def test_a_junk_spot_open_is_not_a_price():
    from tradingagents.hyperliquid.carry.data import _junk
    df = pd.DataFrame({"p_low": [96_000.0, 96_000.0], "p_high": [98_000.0, 98_000.0],
                       "s_open": [240_000.0, 97_000.0], "s_high": [240_000.0, 97_500.0],
                       "s_low": [96_000.0, 96_500.0], "s_close": [97_578.0, 97_400.0],
                       "s_volume": [145.0, 150.0]})
    assert _junk(df).tolist() == [True, False]
