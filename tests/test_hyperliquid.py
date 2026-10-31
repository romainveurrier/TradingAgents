"""The Hyperliquid extension: timing contract, venue math, accounting, policy,
point-in-time data, statistics and the walk-forward loop. No network: the
venue's /info endpoint is served from a synthetic tape."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tradingagents.dataflows import config as df_config
from tradingagents.dataflows.vendors import crypto_pit, hyperliquid as hl
from tradingagents.hyperliquid import clock, metrics
from tradingagents.hyperliquid.account import PerpAccount
from tradingagents.hyperliquid.policy import ExecutionPolicy
from tradingagents.hyperliquid.simulate import MarketTape, replay, run_period
from tradingagents.hyperliquid.venue import (
    FUNDING_INTEREST_1H,
    CostModel,
    funding_rate_1h,
    maintenance_margin_rate,
)

pytestmark = pytest.mark.unit

DAY = "2026-10-14"
CUT = clock.cutoff(DAY)
POISON = 987_654_321.0


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def _price(ms: int) -> float:
    """A smooth, known path: 80k plus a slow sine."""
    return 80_000 + 2_000 * math.sin(ms / 3.6e6 / 37)


class FakeVenue:
    """Serves candleSnapshot and fundingHistory; anything at or after ``poison_from`` is absurd."""

    def __init__(self, poison_from: datetime | None = None, poison=POISON, rate=FUNDING_INTEREST_1H):
        self.poison_ms = _ms(poison_from) if poison_from else None
        self.poison = poison
        self.rate = rate
        self.calls = 0

    def __call__(self, payload, retries=6):
        self.calls += 1
        if payload["type"] == "candleSnapshot":
            req = payload["req"]
            step = hl.INTERVAL_MS[req["interval"]]
            t = req["startTime"] - req["startTime"] % step
            out = []
            while t <= req["endTime"] and len(out) < 5000:
                bad = self.poison_ms is not None and t + step > self.poison_ms
                o = self.poison if bad else _price(t)
                c = self.poison if bad else _price(t + step)
                out.append({"t": t, "T": t + step - 1, "o": str(o), "c": str(c), "h": str(max(o, c) * 1.001),
                            "l": str(min(o, c) * 0.999), "v": "10.0", "n": 100})
                t += step
            return out
        if payload["type"] == "fundingHistory":
            t = payload["startTime"] - payload["startTime"] % 3_600_000 + 3_600_000 + 19
            if payload["startTime"] % 3_600_000 == 19:
                t = payload["startTime"]
            out = []
            while t < payload["endTime"] and len(out) < 500:
                bad = self.poison_ms is not None and t >= self.poison_ms
                out.append({"coin": "BTC", "fundingRate": str(0.5 if bad else self.rate),
                            "premium": str(0.9 if bad else -0.0001), "time": t})
                t += 3_600_000
            return out
        raise AssertionError(payload)


@pytest.fixture
def venue(tmp_path, monkeypatch):
    df_config.set_config({"data_cache_dir": str(tmp_path / "cache"), "hyperliquid_offline": False,
                          "crypto_venue": "hyperliquid"})
    hl.clear_memory()
    fake = FakeVenue()
    monkeypatch.setattr(hl, "_post", fake)
    yield fake
    hl.clear_memory()
    df_config.set_config({"crypto_venue": None})


# --------------------------------------------------------------------------- clock

def test_the_timing_contract():
    assert clock.cutoff("2026-10-14") == datetime(2026, 10, 15, tzinfo=UTC)
    assert clock.execution_time("2026-10-14") == datetime(2026, 10, 15, 0, 15, tzinfo=UTC)
    # The outcome of D is known at the next execution: D+1 must not see it, D+2 may.
    assert not clock.outcome_visible_to("2026-10-14", "2026-10-15")
    assert clock.outcome_visible_to("2026-10-14", "2026-10-16")
    assert clock.resolution_date("2026-10-14") == "2026-10-16"


# --------------------------------------------------------------------------- venue

def test_funding_sits_on_the_interest_rate_until_the_clamp_binds():
    assert funding_rate_1h(0.0) == pytest.approx(FUNDING_INTEREST_1H)
    assert funding_rate_1h(-0.0003) == pytest.approx(FUNDING_INTEREST_1H)   # |I - P| = 4 bp < 5 bp
    assert funding_rate_1h(0.001) == pytest.approx((0.001 - 0.0005) / 8)    # clamp binds
    assert funding_rate_1h(-0.001) == pytest.approx((-0.001 + 0.0005) / 8)
    assert funding_rate_1h(10.0) == 0.04                                     # 4%/h cap


def test_fees_and_margin():
    assert CostModel().fee_rate() == pytest.approx(0.00045)
    assert CostModel(staking="gold").fee_rate() == pytest.approx(0.00045 * 0.8)
    assert CostModel(maker=True).fee_rate() == pytest.approx(0.00015)
    assert CostModel(maker=True).slippage_rate() == 0.0
    assert maintenance_margin_rate(1e6) == pytest.approx(0.0125)
    assert maintenance_margin_rate(2e8) == pytest.approx(0.025)


# --------------------------------------------------------------------------- account

def test_a_fill_costs_exactly_its_fee_and_slippage():
    cost = CostModel(slippage_bps=2.0)
    acct = PerpAccount(collateral=100_000)
    before = acct.equity(80_000)
    fill = acct.trade(1.0, 80_000, cost, "t", "open")
    assert fill.fill_px == pytest.approx(80_000 * 1.0002)
    assert acct.equity(80_000) == pytest.approx(before - fill.fee - fill.slippage_cost, abs=1e-9)


def test_flipping_realizes_the_closed_leg_and_opens_the_rest_at_the_fill():
    cost = CostModel(slippage_bps=0.0)
    acct = PerpAccount(collateral=100_000)
    acct.trade(1.0, 80_000, cost, "t0", "open")
    fill = acct.trade(-2.0, 82_000, cost, "t1", "flip")
    assert fill.realized_pnl == pytest.approx(2_000)
    assert acct.qty == pytest.approx(-1.0) and acct.entry_px == pytest.approx(82_000)
    fees = 0.00045 * (80_000 + 2 * 82_000)
    assert acct.equity(82_000) == pytest.approx(100_000 + 2_000 - fees)


def test_longs_pay_positive_funding_and_shorts_receive_it():
    long_, short = PerpAccount(collateral=1e5, qty=1.0, entry_px=8e4), PerpAccount(collateral=1e5, qty=-1.0, entry_px=8e4)
    assert long_.pay_funding(0.0001, 80_000) == pytest.approx(8.0)
    assert short.pay_funding(0.0001, 80_000) == pytest.approx(-8.0)
    assert long_.collateral == pytest.approx(1e5 - 8) and short.collateral == pytest.approx(1e5 + 8)


def test_a_5x_long_is_liquidated_near_a_19_percent_drop():
    cost = CostModel(slippage_bps=0.0)
    acct = PerpAccount(collateral=100_000)
    acct.trade(500_000 / 80_000, 80_000, cost, "t", "open")
    assert acct.check_liquidation(80_000 * 0.83, cost, "t1") is None
    assert acct.check_liquidation(80_000 * 0.80, cost, "t2") is not None
    assert acct.liquidated and acct.qty == 0


# --------------------------------------------------------------------------- policy

def test_ratings_map_to_signed_exposure_and_hold_keeps_the_position():
    p = ExecutionPolicy()
    assert [p.target(r, 0.3) for r in ("Buy", "Overweight", "Hold", "Underweight", "Sell", "REVIEW")] == \
        [1.0, 0.5, 0.3, -0.5, -1.0, 0.3]
    flat = ExecutionPolicy(mode="long_flat")
    assert [flat.target(r, 0.8) for r in ("Buy", "Underweight", "Sell")] == [1.0, 0.4, 0.0]


def test_leverage_scales_the_target_within_the_cap_only_when_enabled():
    lev = ExecutionPolicy(max_gross=5.0, use_leverage=True)
    assert lev.target("Buy", 0, leverage=3) == 3.0
    assert lev.target("Sell", 0, leverage=12) == -5.0
    assert lev.target("Overweight", 0, leverage=0.2) == 0.5          # never below 1x
    assert lev.target("Hold", 2.0, leverage=5) == 2.0
    assert ExecutionPolicy(max_gross=5.0).target("Buy", 0, leverage=3) == 1.0


def test_the_band_ignores_drift_but_never_a_flip_or_a_close():
    p = ExecutionPolicy(band=0.05)
    assert not p.should_trade(1.0, 1.03)
    assert p.should_trade(1.0, 1.06)
    assert p.should_trade(-0.5, 0.02) and p.should_trade(0.0, 0.01)


# --------------------------------------------------------------------------- point in time

def _tool_outputs():
    return [
        hl.get_hl_ohlcv("BTC-USD", "2026-10-01", DAY),
        hl.get_hl_indicator("BTC-USD", "close_10_ema", DAY, 10),
        hl.get_hl_indicator("BTC-USD", "rsi", DAY, 10),
        hl.build_hl_snapshot("BTC-USD", DAY, 10),
        hl.get_hl_funding_report("BTC-USD", DAY, 7),
        hl.get_hl_market_structure("BTC-USD", DAY, 30),
    ]


@pytest.mark.parametrize("poison", [POISON, 1.0])
def test_no_tool_output_depends_on_anything_after_the_cutoff(tmp_path, monkeypatch, poison):
    """The canary: poison every bar and print from the cutoff on; the tools must not notice."""
    outputs = []
    for i, fake in enumerate((FakeVenue(), FakeVenue(poison_from=CUT, poison=poison))):
        df_config.set_config({"data_cache_dir": str(tmp_path / f"c{i}")})
        hl.clear_memory()
        monkeypatch.setattr(hl, "_post", fake)
        outputs.append(_tool_outputs())
    hl.clear_memory()
    assert outputs[0] == outputs[1]
    assert not any("987,654,321" in o or "987654321" in o for o in outputs[1])


def test_the_last_bar_served_is_the_one_closing_at_the_cutoff(venue):
    daily = hl.admissible_candles("BTC-USD", "1d", DAY, 5)
    hourly = hl.admissible_candles("BTC-USD", "1h", DAY, 2)
    funding = hl.admissible_funding("BTC-USD", DAY, 2)
    assert daily["close_time"].iloc[-1] == CUT
    assert hourly["close_time"].iloc[-1] == CUT
    assert funding["time"].iloc[-1] < CUT


def test_a_date_only_news_stamp_counts_from_a_day_later():
    pdt = datetime(2026, 10, 14, 7, 0, tzinfo=UTC)           # midnight Pacific (PDT)
    pst = datetime(2026, 11, 14, 8, 0, tzinfo=UTC)           # midnight Pacific (PST)
    precise = datetime(2026, 10, 14, 18, 28, tzinfo=UTC)
    assert crypto_pit.news_known_at(pdt) == pdt + timedelta(days=1)
    assert crypto_pit.news_known_at(pst) == pst + timedelta(days=1)
    assert crypto_pit.news_known_at(precise) == precise
    # Daylight saving ends on 2026-11-01: that Pacific day lasts 25 hours.
    dst_day = datetime(2026, 11, 1, 7, 0, tzinfo=UTC)
    assert crypto_pit.news_known_at(dst_day) == datetime(2026, 11, 2, 8, 0, tzinfo=UTC)


def test_news_of_the_analysis_day_itself_is_withheld_unless_precisely_stamped(tmp_path, monkeypatch):
    df_config.set_config({"data_cache_dir": str(tmp_path)})
    items = {
        "2026-10-13": [{"title": "old news", "source": "S", "stamp": "2026-10-13T07:00:00+00:00"}],
        "2026-10-14": [{"title": "same-day dated", "source": "S", "stamp": "2026-10-14T07:00:00+00:00"},
                       {"title": "same-day precise", "source": "S", "stamp": "2026-10-14T18:28:00+00:00"},
                       {"title": "after the cutoff", "source": "S", "stamp": "2026-10-15T00:30:00+00:00"}],
    }
    monkeypatch.setattr(crypto_pit, "_google_day", lambda q, day: items.get(day, []))
    text = crypto_pit.get_crypto_news_pit("BTC-USD", "2026-10-10", DAY)
    assert "old news" in text and "same-day precise" in text
    assert "same-day dated" not in text and "after the cutoff" not in text


def test_fear_greed_values_are_served_only_once_known(tmp_path, monkeypatch):
    df_config.set_config({"data_cache_dir": str(tmp_path)})
    rows = [{"stamp": f"2026-10-{d:02d}T00:00:00+00:00", "value": d, "label": "x"} for d in range(1, 20)]
    monkeypatch.setattr(crypto_pit, "fear_greed_history", lambda: rows)
    text = crypto_pit.get_fear_greed_report(DAY, 30)
    assert "Latest: 14" in text and "| 2026-10-15 |" not in text


# --------------------------------------------------------------------------- metrics

def test_psr_and_min_trl_match_their_formulas():
    rng = np.random.default_rng(0)
    r = rng.normal(0.002, 0.02, 200)
    m = metrics.moments(r)
    sr = m["mean"] / m["std"]
    term = 1 - m["skew"] * sr + (m["kurt"] - 1) / 4 * sr ** 2
    assert metrics.psr(r) == pytest.approx(metrics.N01.cdf(sr * math.sqrt(199) / math.sqrt(term)))
    assert metrics.min_trl(r) == pytest.approx(1 + term * (metrics.N01.inv_cdf(0.95) / sr) ** 2)
    # For normal returns Mertens reduces to Lo.
    assert metrics.sharpe_se_mertens(0.1, 30, 0.0, 3.0) == pytest.approx(metrics.sharpe_se_lo(0.1, 30))


def test_directional_tests_and_bootstrap():
    actual = np.array([1, -1] * 15, float)
    pt = metrics.pesaran_timmermann(actual, actual)
    assert pt["hit"] == 1.0 and pt["p"] < 0.001
    assert metrics.binomial_p_one_sided(20, 30) == pytest.approx(0.0494, abs=1e-4)
    a = metrics.stationary_bootstrap(np.arange(30.0), np.mean, B=200, seed=1)
    b = metrics.stationary_bootstrap(np.arange(30.0), np.mean, B=200, seed=1)
    assert np.array_equal(a, b) and abs(a.mean() - 14.5) < 1.5


# --------------------------------------------------------------------------- engine

def test_a_held_period_charges_24_funding_prints_and_marks_every_bar(venue):
    tape = MarketTape("BTC-USD", clock.execution_time("2026-10-13") - timedelta(hours=1),
                      clock.execution_time("2026-10-15") + timedelta(hours=1))
    acct = PerpAccount(collateral=100_000)
    res = run_period(acct, tape, CostModel(), ExecutionPolicy(), "2026-10-13", 1.0)
    assert res.funding_prints == 24 and len(res.marks) == 96
    assert res.traded and res.fee == pytest.approx(0.00045 * 100_000 * 1.0001, rel=1e-6)
    pnl = acct.qty * (res.end_px - acct.entry_px)
    assert res.equity_post == pytest.approx(100_000 - res.fee - res.funding + pnl, abs=1e-6)
    assert res.cutoff_snapshot["time"] == clock.cutoff("2026-10-14").isoformat()


def test_replay_is_deterministic(venue):
    tape = MarketTape("BTC-USD", clock.execution_time("2026-10-10"), clock.execution_time("2026-10-14"))
    seq = {"2026-10-10": "Buy", "2026-10-11": "Hold", "2026-10-12": "Sell", "2026-10-13": "Underweight"}
    run = lambda: replay(list(seq), tape, CostModel(), ExecutionPolicy(),  # noqa: E731
                         lambda d, x, a: (seq[d], None, None), 100_000, "BTC-USD")
    a, b = run(), run()
    assert [r.equity_post for r in a] == [r.equity_post for r in b]
    assert a[0].target == 1.0 and a[2].target == -1.0 and a[3].target == -0.5
    assert not a[1].traded                                  # Hold: no trade, drift within the band


# --------------------------------------------------------------------------- runner

class _Log:
    def __init__(self):
        self.entries = {}

    def load_entries(self):
        return [{"date": d, "pending": e["pending"]} for d, e in self.entries.items()]

    def update_with_outcome(self, ticker, date, raw, alpha, days, reflection, resolution_date=None):
        self.entries[date].update(pending=False, resolved=resolution_date)
        return True


class _Reflector:
    def reflect_on_final_decision(self, **kw):
        return "lesson"


class FakeGraph:
    """Records what each decision could see; rates by a fixed script."""

    def __init__(self, script):
        self.script = script
        self.memory_log = _Log()
        self.reflector = _Reflector()
        self.seen = {}

    def propagate(self, ticker, d, asset_type, portfolio=None):
        visible = sorted(k for k, e in self.memory_log.entries.items() if not e["pending"])
        self.seen[d] = {"settled": visible, "book": portfolio.render(ticker)}
        self.memory_log.entries[d] = {"pending": True}
        rating = self.script[d]
        lev = " \n\n**Leverage**: 2x" if rating == "Buy" else ""
        return {"final_trade_decision": f"**Rating**: {rating}{lev}", "trader_investment_plan": "",
                "investment_plan": "", "sentiment_report": ""}, rating


def test_the_walk_forward_loop_settles_only_what_each_decision_may_know(venue, tmp_path, monkeypatch):
    from tradingagents.hyperliquid import runner

    monkeypatch.setattr(crypto_pit, "prefetch_news", lambda *a, **k: 0)
    dates = runner.decision_dates("2026-10-08", "2026-10-12")
    script = dict(zip(dates, ["Buy", "Hold", "Sell", "Overweight", "Hold"], strict=True))
    spec = runner.WalkForwardSpec(out_dir=str(tmp_path / "run"), start=dates[0], end=dates[-1])
    graph = FakeGraph(script)
    rows = runner.run_walk_forward(spec, config=runner.hyperliquid_config(tmp_path / "run"),
                                   graph=graph, log=lambda s: None)

    assert [r["rating"] for r in rows] == list(script.values())
    assert rows[0]["target"] == 2.0                          # Buy at the PM's 2x
    for d in dates:   # decision D sees exactly the outcomes of D-2 and earlier
        assert graph.seen[d]["settled"] == [d0 for d0 in dates if clock.outcome_visible_to(d0, d)]
    assert "LONG" in graph.seen[dates[1]]["book"]
    # Equity chains from one period to the next.
    for prev, nxt in zip(rows, rows[1:], strict=False):
        assert nxt["equity_pre"] == pytest.approx(prev["equity_post"])

    # Resuming replays the ledger without asking the graph again.
    resumed = FakeGraph({})
    resumed.memory_log = graph.memory_log                  # the log persists on disk between runs
    again = runner.run_walk_forward(spec, config=runner.hyperliquid_config(tmp_path / "run"),
                                    graph=resumed, log=lambda s: None)
    assert resumed.seen == {}
    assert [r["equity_post"] for r in again] == [r["equity_post"] for r in rows]
    ledger = [json.loads(x) for x in (tmp_path / "run" / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == len(dates)


def test_leverage_and_stop_are_read_from_the_decision_text():
    from tradingagents.hyperliquid import runner
    assert runner.parse_leverage("**Rating**: Buy\n\n**Leverage**: 3.5x") == 3.5
    assert runner.parse_leverage("**Rating**: Buy") is None
    assert runner.parse_stop("**Stop Loss**: 79,250.5\n") == 79250.5
    assert runner.liquidation_price(1.0, 80_000, 20_000, 0.0125) == pytest.approx(60_000 / 0.9875)
    assert re.search("Leverage is available", __import__(
        "tradingagents.hyperliquid.venue", fromlist=["x"]).venue_contract_text(max_gross_leverage=5))


# --------------------------------------------------------------------------- the graph

def test_the_graph_runs_on_hyperliquid_with_every_venue_tool(tmp_path, monkeypatch):
    """The real graph, scripted model, offline vendors: the Hyperliquid wiring end to end."""
    import tests.test_graph_end_to_end as e2e
    from tradingagents.agents import tools as agent_tools
    from tradingagents.agents.analysts import sentiment_analyst
    from tradingagents.dataflows import router
    from tradingagents.graph import trading_graph
    from tradingagents.hyperliquid import runner

    called: set[str] = set()
    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(vendors, vendor, lambda *a, _m=method, **k: called.add(_m) or f"{_m} data")
    monkeypatch.setattr(agent_tools, "build_hl_snapshot", lambda *a, **k: called.add("hl_snapshot") or "snap")
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages", lambda *a, **k: pytest.fail("live feed"))
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", lambda *a, **k: pytest.fail("live feed"))
    monkeypatch.setattr("tradingagents.agents.context.resolve_instrument_identity", lambda t: {})
    prompts: list[str] = []
    model = e2e.ScriptedModel(structured=True)
    original = e2e.ScriptedModel._generate

    def spy(self, messages, *a, **k):
        prompts.append("\n".join(str(m.content) for m in messages))
        return original(self, messages, *a, **k)

    monkeypatch.setattr(e2e.ScriptedModel, "_generate", spy)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: e2e._Client(model))
    for key, value in (("curr_date", "2026-10-14"), ("end_date", "2026-10-14"), ("start_date", "2026-10-07")):
        monkeypatch.setitem(e2e.ARGS, key, value)

    cfg = runner.hyperliquid_config(tmp_path, hyperliquid={"max_gross_leverage": 5.0})
    graph = trading_graph.TradingAgentsGraph(["market", "social", "news", "derivatives"], config=cfg)
    book = runner.PerpBook({"equity": 101_000.0, "collateral": 100_500.0, "qty": 0.5, "entry_px": 80_000.0,
                            "exposure": 0.4, "mark_px": 81_000.0, "fees_paid": 20.0, "funding_paid": 3.0,
                            "time": "2026-10-15T00:00:00+00:00"}, 100_000.0)
    state, rating = graph.propagate("BTC-USD", "2026-10-14", "crypto", portfolio=book)

    assert rating == "Overweight"
    assert {"get_stock_data", "get_indicators", "hl_snapshot", "get_news", "get_global_news",
            "get_perp_funding", "get_perp_market_structure", "get_fear_greed"} <= called
    assert not {"get_fundamentals", "get_balance_sheet"} & called
    assert "Hyperliquid BTC-USDC perpetual" in state["instrument_context"]
    assert "Leverage is available" in state["instrument_context"]
    assert any("LONG 0.50000 BTC" in p for p in prompts)
    assert any("Derivatives report" in p for p in prompts)
    assert state["fundamentals_report"].strip()
