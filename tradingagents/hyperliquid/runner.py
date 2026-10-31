"""Walk-forward simulation of TradingAgents on a Hyperliquid perpetual.

For every decision date ``D``, in order:

1. Settle into the memory log every past decision whose outcome is known by
   the cutoff of ``D`` (``clock.outcome_visible_to``), from the simulator's own
   net P&L.
2. Describe the book as it stood at the cutoff (``PerpBook``) and run the
   graph for ``D``. Every tool it calls serves data as of ``D``'s cutoff.
3. Map the Portfolio Manager's rating and leverage to a target exposure
   (``ExecutionPolicy``), trade at the execution instant and hold to the next
   one (``simulate.run_period``).
4. Append everything to ``ledger.jsonl``. A rerun resumes after the last row,
   replaying earlier periods from the ledger instead of asking the model again.

Nothing in step 3 or 4 is visible to the graph of ``D`` or of ``D+1``: the
book that ``D+1`` reads is snapshotted at ``D+1``'s cutoff.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.hyperliquid import clock
from tradingagents.hyperliquid.account import PerpAccount
from tradingagents.hyperliquid.policy import ExecutionPolicy
from tradingagents.hyperliquid.simulate import MarketTape, parse_date, realized_vol_at, run_period
from tradingagents.hyperliquid.venue import CostModel, maintenance_margin_rate

logger = logging.getLogger(__name__)

SYMBOL = "BTC-USD"


def decision_dates(start: str, end: str) -> list[str]:
    d, last, out = parse_date(start), parse_date(end), []
    while d <= last:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def hyperliquid_config(out_dir: str | Path, **overrides) -> dict:
    """A run config for the Hyperliquid pipeline, with every file kept under ``out_dir``."""
    out = Path(out_dir)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update({
        "results_dir": str(out / "results"),
        "data_cache_dir": str(out / "cache"),
        "memory_log_path": str(out / "trading_memory.md"),
        "crypto_venue": "hyperliquid",
        "settlement_mode": "external",
        "holding_period_days": 1,
        "llm_provider": "chatgpt",
        "deep_think_llm": "gpt-6-sol",
        "quick_think_llm": "gpt-6-luna",
        "llm_max_retries": 8,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "checkpoint_enabled": False,
    })
    cfg["data_vendors"] = {**cfg["data_vendors"], "core_stock_apis": "hyperliquid",
                           "technical_indicators": "hyperliquid", "crypto_derivatives": "hyperliquid"}
    cfg["tool_vendors"] = {**cfg.get("tool_vendors", {}), "get_news": "crypto_pit",
                           "get_global_news": "crypto_pit"}
    cfg["hyperliquid"] = {**cfg["hyperliquid"]}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


@dataclass
class WalkForwardSpec:
    out_dir: str
    start: str = "2026-09-30"
    end: str = "2026-10-29"
    symbol: str = SYMBOL
    equity0: float = 100_000.0
    analysts: tuple = ("market", "social", "news", "derivatives")
    policy: ExecutionPolicy = field(default_factory=lambda: ExecutionPolicy(max_gross=5.0, use_leverage=True))
    cost: CostModel = field(default_factory=CostModel)
    latency_minutes: int = 15
    attempts: int = 3

    @property
    def latency(self) -> timedelta:
        return timedelta(minutes=self.latency_minutes)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["policy"] = asdict(self.policy)
        d["cost"] = asdict(self.cost)
        return d


# ---------------------------------------------------------------------------
# The book the agents read
# ---------------------------------------------------------------------------

def liquidation_price(qty: float, entry: float, collateral: float, mmr: float) -> float | None:
    """The mark at which equity meets maintenance margin, for a single cross position."""
    if not qty:
        return None
    px = (qty * entry - collateral) / (qty * (1 - mmr * (1 if qty > 0 else -1)))
    return px if px > 0 else None


class PerpBook:
    """A perpetual book as the decision agents see it (duck-types ``PortfolioContext``)."""

    def __init__(self, snapshot: dict | None, equity0: float, last_period: dict | None = None):
        self.s = snapshot or {"equity": equity0, "collateral": equity0, "qty": 0.0, "entry_px": 0.0,
                              "exposure": 0.0, "mark_px": None, "fees_paid": 0.0, "funding_paid": 0.0,
                              "time": None}
        self.equity0 = equity0
        self.last = last_period

    def render(self, ticker: str) -> str:
        s = self.s
        lines = [f"Book at the cutoff ({s.get('time') or 'start of the simulation'}), Hyperliquid cross margin, USDC:"]
        lines.append(f"- Account equity: {s['equity']:,.2f} USDC (started at {self.equity0:,.2f}; "
                     f"{s['equity'] / self.equity0 - 1:+.2%} since inception)")
        if s["qty"]:
            side = "LONG" if s["qty"] > 0 else "SHORT"
            px = s["mark_px"]
            mmr = maintenance_margin_rate(abs(s["qty"]) * px)
            liq = liquidation_price(s["qty"], s["entry_px"], s["collateral"], mmr)
            lines.append(f"- Current position in {ticker} perpetual: {side} {abs(s['qty']):,.5f} BTC, "
                         f"notional {abs(s['qty']) * px:,.0f} USDC at mark {px:,.1f}")
            lines.append(f"- Net exposure: {s['exposure']:+.2f}x equity; average entry {s['entry_px']:,.1f}; "
                         f"unrealized P&L {s['qty'] * (px - s['entry_px']):+,.2f} USDC")
            lines.append(f"- Estimated liquidation price: {liq:,.0f}" if liq else
                         "- Estimated liquidation price: none (position fully collateralized)")
        else:
            lines.append(f"- No current position in {ticker} (flat)")
        lines.append(f"- Fees paid to date: {s['fees_paid']:,.2f} USDC; funding paid to date: "
                     f"{s['funding_paid']:+,.2f} USDC (negative = received)")
        if self.last:
            lines.append(f"- Previous decision ({self.last['trade_date']}): {self.last['rating']}"
                         + (f" at {self.last['leverage']:g}x" if self.last.get("leverage") else "")
                         + f", executed at {self.last['ref_px']:,.1f}; its 24 h outcome is not known yet")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.s, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Reading the decision
# ---------------------------------------------------------------------------

_NUM = r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"


def parse_leverage(decision: str) -> float | None:
    m = re.search(r"\*\*Leverage\*\*:\s*([0-9]+(?:\.[0-9]+)?)\s*x", decision or "", re.I)
    return float(m.group(1)) if m else None


def parse_stop(trader_plan: str) -> float | None:
    m = re.search(r"\*\*Stop Loss\*\*:\s*" + _NUM, trader_plan or "")
    return float(m.group(1).replace(",", "")) if m else None


def parse_field(text: str, label: str) -> str | None:
    m = re.search(rf"\*\*{label}\*\*:?\s*\**\s*([A-Za-z][A-Za-z ]*)", text or "")
    return m.group(1).strip() if m else None


def _sentiment_band(report: str) -> str | None:
    m = re.search(r"\*\*Overall Sentiment:\*\*\s*\*\*([A-Za-z ]+)\*\*", report or "")
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def settle(graph, symbol: str, row: dict, log: Callable[[str], None]) -> bool:
    """Write a decision's realized outcome and reflection into the memory log."""
    raw = row["ret"]
    alpha = raw - row["btc_return"]
    reflection = graph.reflector.reflect_on_final_decision(
        final_decision=row["final_trade_decision"], raw_return=raw, alpha_return=alpha,
        benchmark_name="BTC buy-and-hold (1x long)", holding_days=1)
    ok = graph.memory_log.update_with_outcome(symbol, row["trade_date"], raw, alpha, 1, reflection,
                                              resolution_date=clock.resolution_date(row["trade_date"]))
    log(f"  settled {row['trade_date']}: net {raw:+.2%} (BTC {row['btc_return']:+.2%}), "
        f"visible from {clock.resolution_date(row['trade_date'])}")
    return ok


def run_walk_forward(spec: WalkForwardSpec, config: dict | None = None, graph=None,
                     log: Callable[[str], None] = print) -> list[dict]:
    """Run (or resume) the walk-forward simulation; returns the ledger rows."""
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.vendors import crypto_pit, hyperliquid as hl
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = config or hyperliquid_config(out)
    config["hyperliquid"] = {**config.get("hyperliquid", {}), "fee_tier": spec.cost.fee_tier,
                             "staking": spec.cost.staking, "slippage_bps": spec.cost.slippage_bps,
                             "latency_minutes": spec.latency_minutes,
                             "max_gross_leverage": spec.policy.max_gross if spec.policy.use_leverage else 1.0}
    set_config(config)
    (out / "spec.json").write_text(json.dumps({"spec": spec.as_dict(), "config": {
        k: v for k, v in config.items() if k not in ("results_dir", "data_cache_dir", "memory_log_path")}},
        indent=2, default=str))

    dates = decision_dates(spec.start, spec.end)
    first_exec = clock.execution_time(dates[0], spec.latency)
    last_exec = clock.execution_time(dates[-1], spec.latency) + timedelta(days=1)
    log(f"prefetching venue data and news for {dates[0]}..{dates[-1]}")
    log(f"  hyperliquid rows: {hl.prefetch(spec.symbol, first_exec, last_exec)}")
    log(f"  news items: {crypto_pit.prefetch_news(spec.symbol, parse_date(dates[0]), parse_date(dates[-1]))}")
    tape = MarketTape(spec.symbol, first_exec - timedelta(hours=1), last_exec)
    if tape.missing_bars:
        log(f"  warning: {tape.missing_bars} missing 15-minute bars in the window")

    ledger_path = out / "ledger.jsonl"
    rows = {r["trade_date"]: r for r in _read_ledger(ledger_path)}
    graph = graph or TradingAgentsGraph(list(spec.analysts), config=config)
    settled = {e["date"] for e in graph.memory_log.load_entries() if not e.get("pending")}
    # A day whose graph never finished left nothing in the memory log to settle.
    settled |= {d0 for d0, r in rows.items() if r.get("error")}

    account = PerpAccount(collateral=spec.equity0)
    snapshot, last = None, None
    for d in dates:
        if d in rows:   # resume: replay the recorded decision through the engine
            row = rows[d]
            res = run_period(account, tape, spec.cost, spec.policy, d, row["target"], row.get("stop_px"),
                             spec.latency, keep_marks=False)
            if abs(res.equity_post - row["equity_post"]) > 1e-6 * max(1.0, row["equity_post"]):
                raise RuntimeError(f"replay of {d} diverged from the ledger: the tape or engine changed")
            snapshot, last = res.cutoff_snapshot, row
            continue

        for d0 in sorted(rows):
            visible = d0 not in settled and clock.outcome_visible_to(d0, d, spec.latency)
            if visible and settle(graph, spec.symbol, rows[d0], log):
                settled.add(d0)

        book = PerpBook(snapshot, spec.equity0, last)
        log(f"[{d}] deciding (cutoff {clock.cutoff(d):%Y-%m-%d %H:%M}Z, equity {book.s['equity']:,.0f}, "
            f"exposure {book.s['exposure']:+.2f}x)")
        t0 = time.time()
        final_state, rating, error = None, "Hold", None
        for attempt in range(1, spec.attempts + 1):
            try:
                final_state, rating = graph.propagate(spec.symbol, d, "crypto", portfolio=book)
                error = None
                break
            except Exception as exc:  # a provider outage must not end a month-long run
                error = f"{type(exc).__name__}: {exc}"
                log(f"  attempt {attempt} failed: {error}")
                time.sleep(30 * attempt)
        decision = (final_state or {}).get("final_trade_decision", "")
        trader_plan = (final_state or {}).get("trader_investment_plan", "")
        leverage = parse_leverage(decision)
        stop = parse_stop(trader_plan)
        ref = tape.price_at(clock.execution_time(d, spec.latency))
        x_now = account.exposure(ref)
        sigma = realized_vol_at(spec.symbol, d) if spec.policy.vol_target else None
        target = spec.policy.target(rating if not error else "Hold", x_now, sigma, leverage)
        res = run_period(account, tape, spec.cost, spec.policy, d, target, stop, spec.latency)
        row = {
            "trade_date": d, "rating": rating if not error else "ERROR", "leverage": leverage,
            "stop_px": stop, "error": error, "wall_seconds": round(time.time() - t0, 1),
            "trader_action": parse_field(trader_plan, "Action"),
            "research_recommendation": parse_field((final_state or {}).get("investment_plan", ""), "Recommendation"),
            "sentiment_band": _sentiment_band((final_state or {}).get("sentiment_report", "")),
            "final_trade_decision": decision, "trader_plan": trader_plan,
            "book_seen": book.render(spec.symbol),
            **res.as_dict(marks=True),
        }
        _append(ledger_path, row)
        rows[d] = row
        if error:
            settled.add(d)
        snapshot, last = res.cutoff_snapshot, row
        log(f"  -> {row['rating']}{f' {leverage:g}x' if leverage else ''}: target {target:+.2f}x at "
            f"{res.ref_px:,.1f}; period net {res.ret:+.2%} (BTC {res.btc_return:+.2%}), "
            f"fees {res.fee:,.2f}, funding {res.funding:+,.2f}, equity {res.equity_post:,.0f} "
            f"[{row['wall_seconds']:.0f}s]")

    # Outcomes known by now (the last ones settle after the window) close the log.
    for d0 in sorted(rows):
        known = d0 not in settled and clock.outcome_known_at(d0, spec.latency) <= clock.effective_cutoff(None)
        if known and settle(graph, spec.symbol, rows[d0], log):
            settled.add(d0)
    return [rows[d] for d in dates if d in rows]
