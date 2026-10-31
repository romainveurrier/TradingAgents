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
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist

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
FORCED_Q = ("Give your best point estimate of the closing price of Bitcoin in US dollars on {d} "
            "(daily close, UTC). Even if you are unsure, reply with a single number and nothing else.")
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
    ap.add_argument("--mode", choices=("abstain", "forced"), default="abstain",
                    help="abstain: UNKNOWN allowed; forced: a number is required")
    ap.add_argument("--tabulate", action="store_true", help="rebuild the forced table from saved answers")
    ap.add_argument("--dense", action="store_true",
                    help="every day of the window and the 15th of each month before the cutoff")
    args = ap.parse_args()
    if args.dense:
        WINDOW[:] = [f"2026-10-{d:02d}" for d in range(1, 31)]
        CONTROL[:] = [f"{y}-{m:02d}-15" for y, m in ((2025, 8), (2025, 9), (2025, 10), (2025, 11), (2025, 12),
                                                    (2026, 1), (2026, 2), (2026, 3), (2026, 4))]
    if args.mode == "forced":
        saved = Path(args.out) / "leakage_probe_forced.json"
        if args.tabulate and saved.exists():
            return forced_table(json.loads(saved.read_text()), Path(args.out))
        return forced(args)
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


def forced(args) -> None:
    """Forced point estimates: memorization shows as a much smaller error inside the window."""
    cfg = hyperliquid_config(args.run, hyperliquid_offline=True)
    set_config(cfg)
    closes = _daily_frame("BTC-USD", "2026-10-30", history_days=470).set_index("Date")["Close"]
    naive = float(closes["2026-04-15"])   # what a model frozen at its cutoff would say for any later date
    out = {"naive_anchor_date": "2026-04-15", "naive_anchor": naive, "models": {}}
    for tier in ("quick", "deep"):
        llm = create_tier_client(cfg, tier).get_llm()
        model = cfg[f"{tier}_think_llm"]
        rows = []
        for group, dates in (("window", WINDOW), ("control", CONTROL)):
            for d in dates:
                answer = llm.invoke(FORCED_Q.format(d=d)).content.strip()
                guess, truth = _number(answer), float(closes[d])
                rows.append({"group": group, "date": d, "answer": answer[:60], "guess": guess, "truth": truth,
                             "abs_err": abs(guess / truth - 1) if guess else None,
                             "naive_err": abs(naive / truth - 1)})
        out["models"][model] = rows
        print(model, json.dumps(rows, indent=1)[:1600], flush=True)
    (Path(args.out) / "leakage_probe_forced.json").write_text(json.dumps(out, indent=2))
    forced_table(out, Path(args.out))


def forced_table(out: dict, out_dir: Path) -> None:
    """Errors on controls vs the window, and whether the window guesses move with the price.

    Memorization would show as window guesses that track the realized path; a
    model that does not know the window answers with a level that does not move with it.
    """
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Model & Control error & Window error & Guessed change & Realized change & Corr. ($p$) \\\\",
             "\\midrule"]
    for model, rows in out["models"].items():
        def med(group, rows=rows):
            e = sorted(r["abs_err"] for r in rows if r["group"] == group and r["abs_err"] is not None)
            return e[len(e) // 2] if e else float("nan")
        win = [r for r in rows if r["group"] == "window" and r["guess"]]
        g = [r["guess"] for r in win]
        t = [r["truth"] for r in win]
        mg, mt = sum(g) / len(g), sum(t) / len(t)
        cov = sum((a - mg) * (b - mt) for a, b in zip(g, t, strict=True))
        sg = sum((a - mg) ** 2 for a in g) ** 0.5
        st = sum((b - mt) ** 2 for b in t) ** 0.5
        corr = cov / (sg * st) if sg > 0 and st > 0 else float("nan")
        n = len(g)
        # Fisher z: two-sided p for the correlation being zero.
        pval = (2 * (1 - NormalDist().cdf(abs(math.atanh(max(min(corr, 0.999999), -0.999999)) * math.sqrt(n - 3))))
                if n > 3 and corr == corr else float("nan"))
        lines.append(f"\\texttt{{{model}}} & {med('control') * 100:.1f}\\% & {med('window') * 100:.1f}\\% & "
                     f"{(g[-1] / g[0] - 1) * 100:+.1f}\\% & {(t[-1] / t[0] - 1) * 100:+.1f}\\% & "
                     f"{corr:+.2f} ({pval:.2f}) \\\\")
        out.setdefault("comovement", {})[model] = {"corr": corr, "p": pval, "n": n,
                                                   "guessed_change": g[-1] / g[0] - 1,
                                                   "realized_change": t[-1] / t[0] - 1}
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table_leakage_forced.tex").write_text("\n".join(lines))
    (out_dir / "leakage_probe_forced.json").write_text(json.dumps(out, indent=2))


def _tex(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s.replace("\n", " ")


if __name__ == "__main__":
    main()
