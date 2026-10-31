"""Phase 1: the portfolio-margin funding-carry engine on Hyperliquid BTC.

    python scripts/hl_carry_research.py --out experiments/carry_v1

Runs every trial pre-registered in ``<out>/registry.json`` on the daily panel
(2024-07-01..), the primary on the hourly panel, the cost and interpretation
sensitivities, the benchmarks, the deflated Sharpe ratio across the trials,
and writes ``results.json``, tables and figures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows.config import set_config  # noqa: E402
from tradingagents.hyperliquid import metrics  # noqa: E402
from tradingagents.hyperliquid.carry import data, funding_model as fm, pm  # noqa: E402
from tradingagents.hyperliquid.carry.engine import CarryCosts, simulate  # noqa: E402

START, END = "2024-07-01", "2026-10-31"
H_START = "2026-04-07"


def gated(forecast: pd.Series, L_hi: float, borrow: float):
    """The registered gating rule (registry.json), reading the forecast made at the decision hour."""
    def policy(t, book, cur):
        F = float(forecast.asof(t)) * 8760
        if not math.isfinite(F):
            return 0.0
        if borrow + 0.02 <= F:
            return L_hi
        if F >= 0.02:
            return 1.0
        if F >= 0 and book.btc > 0:
            return min(cur, 1.0)
        return 0.0
    return policy


def fixed(L: float):
    return lambda t, book, cur: L


def daily_returns(res) -> np.ndarray:
    return res.daily_returns().to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    out = Path(args.out)
    reg = json.loads((out / "registry.json").read_text())
    set_config({"data_cache_dir": str(out / "cache")})

    f = data.funding_hourly("2024-01-01", END)
    fc168 = fm.walk_forward(f, horizon=168, start=START)
    fc24 = fm.walk_forward(f, horizon=24, start=START)
    forecast_scores = {"24h": fm.score(fc24).to_dict("index"), "168h": fm.score(fc168).to_dict("index")}

    daily = data.daily_panel("2024-01-01", END)
    daily = daily[daily.index >= pd.Timestamp(START, tz="UTC")]
    hourly = data.hourly_panel(H_START, END)
    basis = {"daily": data.basis_stats(daily), "hourly": data.basis_stats(hourly)}
    base = CarryCosts()
    borrow = base.borrow_apy
    models = {"persist168": fc168["persist168"], "ridge (168 h)": fc168["ridge"]}

    def policy_for(trial):
        if trial["policy"] == "always on":
            return fixed(trial["L"])
        return gated(models[trial["model"]], trial["L_hi"], borrow)

    results, series = {}, {}
    for trial in reg["trials"]:
        res = simulate(daily, policy_for(trial), trial["id"], costs=base)
        results[trial["id"]] = res.summary() | {"events": len(res.events)}
        series[trial["id"]] = res
        print(f"{trial['id']:24s} ann {res.summary()['ann']:+7.2%} SR {res.summary()['sharpe']:6.2f} "
              f"MDD {res.summary()['mdd']:6.2%} liq {res.book.liquidations} maxratio {res.ratio_max.max():.3f}",
              flush=True)

    prim_id = reg["primary"]
    prim_trial = next(t for t in reg["trials"] if t["id"] == prim_id)
    prim = series[prim_id]

    # Sensitivities of the primary (not trials)
    sens = {}
    for name, c, p in (
        ("borrow 10%", replace(base, borrow_apy=0.10), pm.DEFAULT_PM),
        ("borrow 20%", replace(base, borrow_apy=0.20), pm.DEFAULT_PM),
        ("maker fills", replace(base, maker=True), pm.DEFAULT_PM),
        ("fee tier 3", replace(base, fee_tier=3), pm.DEFAULT_PM),
        ("PM interpretation: balance", base, pm.PMParams(interpretation="balance")),
    ):
        pol = gated(models[prim_trial["model"]], prim_trial["L_hi"], c.borrow_apy)
        sens[name] = simulate(daily, pol, name, costs=c, pm=p).summary()

    # Hourly-resolution checks on the window with hourly spot and perp
    hourly_runs = {}
    for name, pol in (("primary", gated(models[prim_trial["model"]], prim_trial["L_hi"], borrow)),
                      ("fixed_L2.5", fixed(2.5)), ("fixed_L3.0", fixed(3.0))):
        r = simulate(hourly, pol, name, costs=base)
        hourly_runs[name] = r.summary() | {"deleverages": sum(e["type"] == "deleverage" for e in r.events)}

    # Benchmarks
    bh = (daily["p_close"] / daily["p_open"] - 1 - daily["funding"]).to_numpy()
    bench = {"BTC buy & hold (perp, 1x)": metrics.summarize(bh)}

    # Inference on the primary: DSR across the registered trials, bootstrap CIs, per-year
    r = daily_returns(prim)
    trial_srs = [metrics.sharpe(daily_returns(series[t["id"]])) / math.sqrt(365) for t in reg["trials"]]
    sr_boot = metrics.stationary_bootstrap(r, metrics.sharpe, B=args.boot, mean_block=5)
    ann_boot = metrics.stationary_bootstrap(r, lambda x: float(np.mean(x) * 365), B=args.boot, mean_block=5)
    inference = {"psr0": metrics.psr(r), "dsr": metrics.dsr(r, trial_srs), "n_trials": len(trial_srs),
                 "sharpe_ci": metrics.ci(sr_boot), "ann_mean_ci": metrics.ci(ann_boot),
                 "min_trl_days": metrics.min_trl(r)}
    # Every trial gets the same inference, so a post-hoc favourite is judged against all nine.
    per_trial = {}
    for t in reg["trials"]:
        rt = daily_returns(series[t["id"]])
        boot = metrics.stationary_bootstrap(rt, metrics.sharpe, B=min(args.boot, 2000), mean_block=5)
        eqt = series[t["id"]].equity
        per_trial[t["id"]] = {
            "psr0": metrics.psr(rt), "dsr": metrics.dsr(rt, trial_srs), "sharpe_ci": metrics.ci(boot),
            "by_year": {int(y): float(eqt[eqt.index.year == y].iloc[-1] / eqt[eqt.index.year == y].iloc[0] - 1)
                        for y in sorted({i.year for i in eqt.index}) if (eqt.index.year == y).sum() > 30},
            "liquidation_dates": [str(e["t"].date()) for e in series[t["id"]].events if e["type"] == "liquidation"],
        }
    eq = prim.equity
    years = {}
    for y in sorted({t.year for t in eq.index}):
        e = eq[eq.index.year == y]
        if len(e) > 30:
            years[y] = {"return": float(e.iloc[-1] / e.iloc[0] - 1), "days": len(e)}
    rallies = (daily["p_high"] / daily["p_open"] - 1)
    stress = {"max_intraday_rally": float(rallies.max()), "rally_date": str(rallies.idxmax().date()),
              "days_rally_gt_10pct": int((rallies > 0.10).sum()),
              "rally_to_liq": {L: pm.rally_to_liquidation(L) - 1 for L in (1.5, 2.0, 2.5, 3.0)},
              "max_entry_leverage": pm.max_entry_leverage()}

    res_all = {"registry_primary": prim_id, "forecast_scores": forecast_scores, "basis": basis,
               "trials": results, "sensitivities": sens, "hourly": hourly_runs, "benchmarks": bench,
               "inference": inference, "per_trial": per_trial, "primary_by_year": years, "stress": stress,
               "primary_events": [{**e, "t": str(e["t"])} for e in prim.events]}
    (out / "results.json").write_text(json.dumps(res_all, indent=2, default=str))
    figures(series, prim, daily, f, out, prim_id)
    write_report(res_all, reg, out)
    print(json.dumps({"inference": inference, "years": years, "stress": stress, "sens": {k: (round(v['ann'], 4), round(v['sharpe'], 2), v['liquidations']) for k, v in sens.items()}, "hourly": {k: (round(v['ann'], 4), round(v['sharpe'], 2), v['liquidations'], v['deleverages']) for k, v in hourly_runs.items()}, "basis": basis, "bench": {k: (round(v['ann_return'], 4), round(v['sharpe'], 2)) for k, v in bench.items()}}, indent=1, default=str))


def write_report(r: dict, reg: dict, out: Path) -> None:
    """The results tables, generated from results.json so no number is typed by hand."""
    pct = lambda x: f"{x:+.1%}"  # noqa: E731
    lines = ["| Trial | Ann. return | Vol | Sharpe [95% CI] | Max DD | Liquidations | Mean L | "
             "Funding | Borrow | Fees | PSR | DSR |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in reg["trials"]:
        v, q = r["trials"][t["id"]], r["per_trial"][t["id"]]
        star = " **(primary)**" if t["id"] == reg["primary"] else ""
        lines.append(f"| {t['id']}{star} | {pct(v['ann'])} | {v['vol']:.1%} | {v['sharpe']:.2f} "
                     f"[{q['sharpe_ci'][0]:.1f}, {q['sharpe_ci'][1]:.1f}] | {v['mdd']:.1%} | {v['liquidations']} | "
                     f"{v['mean_leverage']:.2f} | {v['funding']:,.0f} | {v['interest']:,.0f} | {v['fees']:,.0f} | "
                     f"{q['psr0']:.3f} | {q['dsr']:.3f} |")
    years = sorted({y for q in r["per_trial"].values() for y in q["by_year"]})
    ylines = ["| Trial | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
    for t in reg["trials"]:
        by = r["per_trial"][t["id"]]["by_year"]
        ylines.append(f"| {t['id']} | " + " | ".join(pct(by[y]) if y in by else "–" for y in years) + " |")
    slines = ["| Primary under | Ann. return | Sharpe | Max DD | Liquidations |", "|---|---|---|---|---|"]
    for k, v in r["sensitivities"].items():
        slines.append(f"| {k} | {pct(v['ann'])} | {v['sharpe']:.2f} | {v['mdd']:.1%} | {v['liquidations']} |")
    hlines = ["| Hourly run (2026-04-07..) | Ann. return | Sharpe | Max DD | Max PM ratio | Deleverages |",
              "|---|---|---|---|---|---|"]
    for k, v in r["hourly"].items():
        hlines.append(f"| {k} | {pct(v['ann'])} | {v['sharpe']:.2f} | {v['mdd']:.1%} | {v['max_ratio']:.3f} | "
                      f"{v['deleverages']} |")
    fl = ["| Model | 24 h R² vs 7-day persistence | 24 h decision hit | 7 d R² | 7 d decision hit |",
          "|---|---|---|---|---|"]
    for m in r["forecast_scores"]["24h"]:
        a, b = r["forecast_scores"]["24h"][m], r["forecast_scores"]["168h"][m]
        fl.append(f"| {m} | {a['r2_vs_persist168']:+.3f} | {a['decision_hit']:.1%} | {b['r2_vs_persist168']:+.3f} | "
                  f"{b['decision_hit']:.1%} |")
    tables = {"TRIALS": "\n".join(lines), "YEARS": "\n".join(ylines), "SENS": "\n".join(slines),
              "HOURLY": "\n".join(hlines), "FORECAST": "\n".join(fl)}
    (out / "tables.json").write_text(json.dumps(tables, indent=2))
    for k, v in tables.items():
        (out / f"table_{k.lower()}.md").write_text(v + "\n")
    template = Path(__file__).resolve().parents[1] / "docs" / "hyperliquid" / "CARRY_REPORT.template.md"
    if template.exists():
        text = template.read_text()
        for k, v in tables.items():
            text = text.replace("{{" + k + "}}", v)
        template.with_name("CARRY_REPORT.md").write_text(text)


def figures(series, prim, daily, f, out: Path, prim_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from tradingagents.hyperliquid.report import _style
    _style()
    from tradingagents.hyperliquid.report import _dodged_end_labels
    colors = {"fixed_L1.0": "#eda100", "fixed_L2.0": "#1baf7a", "fixed_L2.5": "#e87ba4",
              "fixed_L3.0": "#eb6834", prim_id: "#2a78d6"}
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ends = []
    for k, col in colors.items():
        e = series[k].equity / series[k].equity.iloc[0] * 100
        ax.plot(e.index, e.values, color=col, lw=1.8 if k == prim_id else 1.1, label=k)
        ends.append((e.index[-1], float(e.iloc[-1]), f"{k} {e.iloc[-1] - 100:+.0f}%", col))
    _dodged_end_labels(ax, ends, gap_frac=0.06)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Equity (start = 100)")
    ax.set_title("Portfolio-margin BTC funding carry, net of fees, slippage and borrow (daily)", loc="left")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    for h in leg.legend_handles:
        h.set_linewidth(2.2)
    fig.savefig(figdir / "carry_equity.pdf")
    fig.savefig(figdir / "carry_equity.png", dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    fr = f["rate"][f.index >= pd.Timestamp(START, tz="UTC")].rolling(168).mean() * 8760 * 100
    a1.plot(fr.index, fr.values, color="#0b0b0b", lw=0.8)
    a1.axhline(5, color="#e34948", lw=1, ls=(0, (4, 2)))
    a1.annotate("borrow 5%", (fr.index[-1], 5), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                va="center", annotation_clip=False)
    a1.set_ylabel("Funding, 7-day mean (% a year)")
    lev = prim.leverage.resample("1D").last()
    a2.step(lev.index, lev.values, where="post", color="#2a78d6", lw=1.2)
    a2.set_ylabel("Primary leverage")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.savefig(figdir / "carry_funding_leverage.pdf")
    fig.savefig(figdir / "carry_funding_leverage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
