"""Probe the models for look-ahead through pretraining (leaks L6 and L7 of the plan).

For each model tier:

- **Clock probe**: ask for today's date with no date given. A provider that
  injects the real date would reveal that the "future" exists (not what it holds).
- **Memorization probe**: ask for Bitcoin's daily close on dates inside the
  test window and on control dates before the model's training cutoff, and
  compare with the venue's own closes. Recall on the controls and none in the
  window is what a clean cutoff looks like (cf. Lopez-Lira, Tang & Zhu 2025).
- **Event probe**: ask what Bitcoin did in October 2026.

    python scripts/hl_leakage_probe.py --run experiments/hl_btc_2026-10 --out papers/generated
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows.config import set_config  # noqa: E402
from tradingagents.dataflows.vendors.hyperliquid import _daily_frame  # noqa: E402
from tradingagents.hyperliquid.runner import hyperliquid_config  # noqa: E402
from tradingagents.llm_clients import create_tier_client  # noqa: E402

WINDOW = ["2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26", "2026-10-29"]
CONTROL = ["2025-09-15", "2025-11-15", "2026-01-15", "2026-03-02", "2026-04-15"]

PRICE_Q = ("What was the closing price of Bitcoin in US dollars on {d} (daily close, UTC)? "
           "Reply with a single number and nothing else, or the word UNKNOWN if you do not know it.")
DATE_Q = ("Without guessing, what is today's date? If nothing available to you states it, "
          "reply with the single word UNKNOWN.")
EVENT_Q = ("In one sentence: what did the price of Bitcoin do during October 2026? "
           "If you have no knowledge of that month, reply with the single word UNKNOWN.")


def _number(text: str) -> float | None:
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", text.replace("$", ""))
    if not m or "UNKNOWN" in text.upper():
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = hyperliquid_config(args.run, hyperliquid_offline=True)
    set_config(cfg)
    closes = _daily_frame("BTC-USD", "2026-10-30", history_days=470).set_index("Date")["Close"]
    out = {"models": {}}
    for tier in ("quick", "deep"):
        llm = create_tier_client(cfg, tier).get_llm()
        model = cfg[f"{tier}_think_llm"]
        rec = {"clock": llm.invoke(DATE_Q).content.strip(), "event": llm.invoke(EVENT_Q).content.strip(),
               "prices": []}
        for group, dates in (("window", WINDOW), ("control", CONTROL)):
            for d in dates:
                answer = llm.invoke(PRICE_Q.format(d=d)).content.strip()
                guess, truth = _number(answer), float(closes.get(d, float("nan")))
                rec["prices"].append({"group": group, "date": d, "answer": answer[:80], "guess": guess,
                                      "truth": truth,
                                      "abs_err": abs(guess / truth - 1) if guess and truth == truth else None})
        out["models"][model] = rec
        print(model, json.dumps(rec, indent=1)[:1500], flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "leakage_probe.json").write_text(json.dumps(out, indent=2))

    lines = ["\\begin{tabular}{llrrr}", "\\toprule",
             "Model & Dates & Answered & Unknown & Median abs. error \\\\", "\\midrule"]
    for model, rec in out["models"].items():
        for group, label in (("control", "Before cutoff (controls)"), ("window", "Test window (Oct 2026)")):
            ps = [p for p in rec["prices"] if p["group"] == group]
            errs = sorted(p["abs_err"] for p in ps if p["abs_err"] is not None)
            med = f"{errs[len(errs) // 2] * 100:.1f}\\%" if errs else "--"
            lines.append(f"\\texttt{{{model}}} & {label} & {len(errs)} & {len(ps) - len(errs)} & {med} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (Path(args.out) / "table_leakage.tex").write_text("\n".join(lines))
    clock_lines = [f"\\item \\texttt{{{m}}}: clock probe ``{_tex(r['clock'][:120])}''; "
                   f"event probe ``{_tex(r['event'][:200])}''" for m, r in out["models"].items()]
    (Path(args.out) / "leakage_answers.tex").write_text("\\begin{itemize}\n" + "\n".join(clock_lines)
                                                         + "\n\\end{itemize}\n")


def _tex(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s.replace("\n", " ")


if __name__ == "__main__":
    main()
