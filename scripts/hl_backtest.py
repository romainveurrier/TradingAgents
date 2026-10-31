"""Walk-forward TradingAgents backtest on the Hyperliquid BTC perpetual.

    python scripts/hl_backtest.py --out experiments/hl_btc_2026-10            # run / resume
    python scripts/hl_backtest.py --out experiments/hl_btc_2026-10 --prefetch-only

Defaults are the pre-registered primary configuration (docs/hyperliquid/PLAN.md):
decisions 2026-09-30..2026-10-29, $100k, long/short, PM leverage up to 5x,
tier-0 taker fees, 1 bp slippage, 15 min latency, ChatGPT plan (gpt-6-luna /
gpt-6-sol). A rerun resumes after the last decision in the ledger.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows.config import set_config  # noqa: E402
from tradingagents.dataflows.vendors import crypto_pit, hyperliquid as hl  # noqa: E402
from tradingagents.hyperliquid import clock  # noqa: E402
from tradingagents.hyperliquid.policy import ExecutionPolicy  # noqa: E402
from tradingagents.hyperliquid.runner import (  # noqa: E402
    WalkForwardSpec,
    decision_dates,
    hyperliquid_config,
    run_walk_forward,
)
from tradingagents.hyperliquid.simulate import parse_date  # noqa: E402
from tradingagents.hyperliquid.venue import CostModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-09-30")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--max-leverage", type=float, default=5.0)
    ap.add_argument("--no-leverage", action="store_true")
    ap.add_argument("--fee-tier", type=int, default=0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--provider", default="chatgpt")
    ap.add_argument("--quick", default="gpt-6-luna")
    ap.add_argument("--deep", default="gpt-6-sol")
    ap.add_argument("--prefetch-only", action="store_true")
    ap.add_argument("--offline", action="store_true", help="serve the stored data only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    use_lev = not args.no_leverage
    spec = WalkForwardSpec(
        out_dir=args.out, start=args.start, end=args.end, equity0=args.equity,
        policy=ExecutionPolicy(max_gross=args.max_leverage if use_lev else 1.0, use_leverage=use_lev),
        cost=CostModel(fee_tier=args.fee_tier, slippage_bps=args.slippage_bps),
    )
    config = hyperliquid_config(args.out, llm_provider=args.provider, quick_think_llm=args.quick,
                                deep_think_llm=args.deep, hyperliquid_offline=args.offline)
    if args.prefetch_only:
        set_config(config)
        dates = decision_dates(args.start, args.end)
        first = clock.execution_time(dates[0])
        last = clock.execution_time(dates[-1]) + timedelta(days=1)
        print("hyperliquid rows:", hl.prefetch(spec.symbol, first, last), flush=True)
        print("news items:", crypto_pit.prefetch_news(spec.symbol, parse_date(dates[0]), parse_date(dates[-1])),
              flush=True)
        return
    rows = run_walk_forward(spec, config, log=lambda s: print(s, flush=True))
    print(f"done: {len(rows)} decisions; ledger at {Path(args.out) / 'ledger.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
