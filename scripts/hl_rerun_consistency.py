"""Re-run past decisions with identical inputs to measure LLM sampling noise.

For each chosen date the graph sees what the original run saw: the same
cutoff, data, the book snapshot from the ledger and the memory log filtered to
lessons resolved by that date. Only the model's sampling differs. Results go to
a separate directory; the experiment is not modified.

    python scripts/hl_rerun_consistency.py --run experiments/hl_btc_2026-10 --every 6 --out papers/generated
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.hyperliquid.policy import RATING_SCORE  # noqa: E402
from tradingagents.hyperliquid.runner import PerpBook, hyperliquid_config, parse_leverage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=6)
    args = ap.parse_args()
    run = Path(args.run)
    spec = json.loads((run / "spec.json").read_text())["spec"]
    rows = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines() if x.strip()]
    rerun_dir = run / "consistency"
    rerun_dir.mkdir(exist_ok=True)
    shutil.copy(run / "trading_memory.md", rerun_dir / "trading_memory.md")
    original = (run / "trading_memory.md").read_text()
    cfg = hyperliquid_config(run, hyperliquid_offline=True, results_dir=str(rerun_dir / "results"),
                             memory_log_path=str(rerun_dir / "trading_memory.md"),
                             hyperliquid={"max_gross_leverage": spec["policy"]["max_gross"]})
    graph = TradingAgentsGraph(list(spec["analysts"]), config=cfg)
    out = []
    picks = list(range(args.every - 1, len(rows), args.every))
    for i in picks:
        row = rows[i]
        prev = rows[i - 1] if i else None
        book = PerpBook(prev["cutoff_snapshot"] if prev else None, spec["equity0"], prev)
        # Start from the original log every time, minus this date's own entry: the rerun then
        # reads exactly the lessons the original did (the date filter hides later ones) and
        # records its own decision instead of skipping it as already present.
        log = graph.memory_log
        entries = original.split(log._SEPARATOR)
        log._log_path.write_text(log._SEPARATOR.join(e for e in entries
                                                     if not e.lstrip().startswith(f"[{row['trade_date']} |")))
        state, rating = graph.propagate(spec["symbol"], row["trade_date"], "crypto", portfolio=book)
        rec = {"date": row["trade_date"], "original": row["rating"], "rerun": rating,
               "original_leverage": row.get("leverage"), "rerun_leverage": parse_leverage(state["final_trade_decision"])}
        out.append(rec)
        print(rec, flush=True)
    same = sum(r["original"] == r["rerun"] for r in out)
    same_dir = sum((RATING_SCORE.get(r["original"], 0) > 0) == (RATING_SCORE.get(r["rerun"], 0) > 0)
                   and (RATING_SCORE.get(r["original"], 0) < 0) == (RATING_SCORE.get(r["rerun"], 0) < 0) for r in out)
    dist = [abs(RATING_SCORE.get(r["original"], 0) - RATING_SCORE.get(r["rerun"], 0)) for r in out]
    summary = {"pairs": out, "n": len(out), "same_rating": same, "same_direction": same_dir,
               "mean_abs_notch_change": sum(dist) / len(dist) if dist else float("nan")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "consistency.json").write_text(json.dumps(summary, indent=2))
    macros = {"ConsN": str(len(out)), "ConsSame": str(same), "ConsDir": str(same_dir),
              "ConsNotch": f"{summary['mean_abs_notch_change']:.1f}"}
    (Path(args.out) / "consistency.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    lines = ["\\begin{tabular}{lllll}", "\\toprule", "Date & Original & Lev. & Re-run & Lev. \\\\", "\\midrule"]
    lines += [f"{r['date']} & {r['original']} & {r['original_leverage'] or '--'} & {r['rerun']} & "
              f"{r['rerun_leverage'] or '--'} \\\\" for r in out]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (Path(args.out) / "table_consistency.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
