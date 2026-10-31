# TradingAgents on Hyperliquid (BTC perpetual)

This fork lets TradingAgents trade the **BTC-USDC perpetual on Hyperliquid**. Fees, hourly funding,
slippage, margin and liquidation are all modeled. It also evaluates the agents walk-forward, with no
look-ahead. Everything is opt-in behind `crypto_venue="hyperliquid"`; without it, upstream behavior is
unchanged.

- Research plan, pre-registered before any run: [`PLAN.md`](PLAN.md)
- Papers: [`papers/methods`](../../papers/methods) (design, math, protocol) and
  [`papers/empirical`](../../papers/empirical) (the October 2026 run)

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]" matplotlib
.venv/bin/tradingagents login chatgpt          # or set an API key and pass --provider/--quick/--deep

# Walk-forward backtest (resumable; rerun the same command to continue)
.venv/bin/python scripts/hl_backtest.py --out experiments/my_run --start 2026-09-30 --end 2026-10-29

# Every number, table and figure the papers use, rebuilt offline from the run directory
.venv/bin/python scripts/hl_report.py --run experiments/my_run --papers papers
make -C papers papers                          # compile both papers

# Pretraining-leakage probe (clock, memorization, event)
.venv/bin/python scripts/hl_leakage_probe.py --run experiments/my_run --out papers/generated --mode forced
```

Programmatic use for a single live decision:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.hyperliquid.runner import PerpBook, hyperliquid_config

cfg = hyperliquid_config("runs/live", hyperliquid={"max_gross_leverage": 5.0})
graph = TradingAgentsGraph(["market", "social", "news", "derivatives"], config=cfg)
state, rating = graph.propagate("BTC-USD", "2026-10-31", "crypto", portfolio=PerpBook(None, 100_000))
```

## What changed

| Piece | Where |
|---|---|
| Hyperliquid `/info` vendor: candles and funding, served point-in-time by bar close | `tradingagents/dataflows/vendors/hyperliquid.py` |
| Point-in-time news (known-at rule) and Fear & Greed | `tradingagents/dataflows/vendors/crypto_pit.py` |
| Derivatives analyst (funding, premium, volatility, carry), in the fundamentals slot | `tradingagents/agents/analysts/derivatives_analyst.py` |
| Archived-only sentiment sources on Hyperliquid | `tradingagents/agents/analysts/sentiment_analyst.py` |
| Venue contract in every agent's instrument context; optional PM `leverage` field | `agents/context.py`, `agents/schemas.py`, `hyperliquid/venue.py` |
| Fees, funding and margin rules (one source of truth) | `tradingagents/hyperliquid/venue.py` |
| Timing contract: cutoff, execution, outcome visibility | `tradingagents/hyperliquid/clock.py` |
| Cross-margin account: fills, funding, liquidation | `tradingagents/hyperliquid/account.py` |
| Rating → target exposure (long/short, long/flat, leverage, vol-target, stops) | `tradingagents/hyperliquid/policy.py` |
| Holding-period engine on 15-minute bars | `tradingagents/hyperliquid/simulate.py` |
| Walk-forward runner, memory settled from net P&L | `tradingagents/hyperliquid/runner.py` |
| Baselines, statistics, report | `hyperliquid/{baselines,metrics,report}.py` |
| Tests, including the point-in-time canary | `tests/test_hyperliquid.py` |

## Rating → position

| Rating | Target exposure (× equity) |
|---|---|
| Buy | +1 × L |
| Overweight | +0.5 × L |
| Hold | keep current position |
| Underweight | −0.5 × L |
| Sell | −1 × L |

`L ∈ [1, 5]` is the Portfolio Manager's leverage multiple. The venue contract tells the agents to raise
it only when the evidence is decisive. Drift below 5% of equity is not rebalanced.

## Timing (no look-ahead)

A decision dated `D` follows three rules:

- It sees only data known by `(D+1) 00:00 UTC`: closed candles, prior funding prints, and news whose
  known-at time has passed.
- It trades at `00:15 UTC` at the next 15-minute bar's open, and holds 24 h.
- Its outcome is written to memory so that decision `D+2` (not `D+1`) can read it.
