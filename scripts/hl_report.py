"""Rebuild every number, table and figure the papers quote, from a run directory alone.

    python scripts/hl_report.py --run experiments/hl_btc_2026-10 --papers papers

Writes papers/generated/{summary.json,numbers.tex,table_*.tex} and papers/figures/*.pdf.
Offline: the venue data comes from the run's own cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.hyperliquid.report import analyze  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--papers", default="papers")
    ap.add_argument("--mc", type=int, default=2000, help="random-rating Monte Carlo draws")
    ap.add_argument("--boot", type=int, default=10000, help="stationary-bootstrap resamples")
    args = ap.parse_args()
    s = analyze(args.run, args.papers, mc_draws=args.mc, boot=args.boot)
    a = s["stats"]["agent_primary"]
    print(f"agents: {a['total_return']:+.2%} (Sharpe {a['sharpe']:.2f}, MDD {a['mdd']:.2%}); "
          f"buy & hold {s['stats']['buy_hold']['total_return']:+.2%}; replay drift {s['replay_max_abs_drift']:.1e}")


if __name__ == "__main__":
    main()
