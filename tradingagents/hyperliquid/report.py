"""Everything the papers report, rebuilt from a run directory alone.

``analyze(run_dir, out_dir)`` reads ``spec.json`` and ``ledger.jsonl``, reloads
the venue tape from the run's own cache (offline), and:

1. replays the agents' decisions through the engine and checks the replay
   reproduces the ledger (determinism);
2. replays the *same* decisions under counterfactual policies and cost models,
   and runs the rule-based baselines;
3. computes performance, Sharpe inference (Lo/Mertens SE, PSR, MinTRL, DSR),
   stationary-bootstrap intervals, directional and timing tests, rating
   diagnostics, a random-rating Monte Carlo and a power analysis;
4. writes ``summary.json``, LaTeX tables, a LaTeX macro file of every number
   quoted in the text, and the figures.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.config import set_config
from tradingagents.hyperliquid import clock, metrics
from tradingagents.hyperliquid.baselines import LABELS as BASELINE_LABELS, run_baselines
from tradingagents.hyperliquid.policy import RATING_SCORE, ExecutionPolicy
from tradingagents.hyperliquid.simulate import MarketTape, PeriodResult, equity_path, replay
from tradingagents.hyperliquid.venue import FUNDING_INTEREST_1H, CostModel, funding_rate_1h

# Entity -> color, fixed across every figure (validated categorical order, light mode).
COLORS = {
    "agent": "#2a78d6", "agent_1x": "#eb6834", "buy_hold": "#1baf7a",
    "tsmom30": "#eda100", "macd": "#e87ba4",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LONG, SHORT, NEUTRAL = "#2a78d6", "#e34948", "#b9b8b2"

RATINGS = ("Sell", "Underweight", "Hold", "Overweight", "Buy")


def _load(run_dir: Path):
    spec = json.loads((run_dir / "spec.json").read_text())
    rows = [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines() if x.strip()]
    return spec, rows


def _policy(d: dict) -> ExecutionPolicy:
    return ExecutionPolicy(**{k: v for k, v in d.items() if k in ExecutionPolicy.__dataclass_fields__})


def _cost(d: dict) -> CostModel:
    return CostModel(**{k: v for k, v in d.items() if k in CostModel.__dataclass_fields__})


def _stats(results: list[PeriodResult], equity0: float) -> dict:
    r = np.array([p.ret for p in results])
    eq = equity_path(results, equity0)
    s = metrics.summarize(r, eq)
    s.update({
        "turnover": float(sum(abs(p.exposure_post_trade - p.exposure_pre) for p in results)),
        "trades": int(sum(p.traded for p in results)),
        "fees": float(sum(p.fee for p in results)),
        "slippage": float(sum(p.slippage for p in results)),
        "funding": float(sum(p.funding for p in results)),
        "mean_abs_exposure": float(np.mean([abs(p.exposure_post_trade) for p in results])),
        "max_abs_exposure": float(max(abs(p.exposure_post_trade) for p in results)),
        "liquidated": bool(any(p.liquidated for p in results)),
        "stops": int(sum(p.stop_hit for p in results)),
        "equity_end": float(results[-1].equity_post),
    })
    return s


def _decisions(rows):
    return {r["trade_date"]: (r["rating"] if r["rating"] in RATING_SCORE else "Hold", r.get("stop_px"),
                              r.get("leverage")) for r in rows}


def _replay_decisions(dates, tape, cost, policy, decisions, equity0, symbol):
    return replay(dates, tape, cost, policy, lambda d, x, a: decisions[d], equity0, symbol)


def analyze(run_dir: str | Path, out_dir: str | Path, mc_draws: int = 2000, boot: int = 10000,
            figures: bool = True) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)
    spec, rows = _load(run_dir)
    s = spec["spec"]
    symbol, equity0 = s["symbol"], float(s["equity0"])
    set_config({"data_cache_dir": str(run_dir / "cache"), "hyperliquid_offline": True,
                "crypto_venue": "hyperliquid"})
    policy, cost = _policy(s["policy"]), _cost(s["cost"])
    dates = [r["trade_date"] for r in rows]
    t0 = clock.execution_time(dates[0])
    t1 = clock.execution_time(dates[-1]) + timedelta(days=1)
    tape = MarketTape(symbol, t0 - timedelta(hours=1), t1)
    decisions = _decisions(rows)

    # 1. determinism
    primary = _replay_decisions(dates, tape, cost, policy, decisions, equity0, symbol)
    drift = max(abs(p.equity_post - r["equity_post"]) for p, r in zip(primary, rows, strict=True))

    # 2. counterfactual policies (same decisions) and baselines
    variants = {
        "primary": policy,
        "unlevered": replace(policy, use_leverage=False, max_gross=1.0),
        "long_flat": replace(policy, mode="long_flat", use_leverage=False, max_gross=1.0),
        "vol_target": replace(policy, use_leverage=False, max_gross=1.0, vol_target=0.40, vol_cap=2.0),
        "primary_stops": replace(policy, use_stops=True),
    }
    var_res = {k: (primary if k == "primary" else
                   _replay_decisions(dates, tape, cost, p, decisions, equity0, symbol))
               for k, p in variants.items()}
    cost_models = {
        "base": cost,
        "no_slippage": replace(cost, slippage_bps=0.0),
        "slippage_2bp": replace(cost, slippage_bps=2.0),
        "slippage_5bp": replace(cost, slippage_bps=5.0),
        "tier_3": replace(cost, fee_tier=3),
        "tier_6_diamond": replace(cost, fee_tier=6, staking="diamond"),
        "maker_fills": replace(cost, maker=True, slippage_bps=0.0),
        "no_costs": CostModel(fee_tier=4, maker=True, slippage_bps=0.0),
    }
    sens = {k: _stats(_replay_decisions(dates, tape, c, policy, decisions, equity0, symbol), equity0)
            for k, c in cost_models.items()}
    base = run_baselines(dates, tape, cost, equity0, symbol)

    stats = {f"agent_{k}": _stats(v, equity0) for k, v in var_res.items()}
    stats.update({k: _stats(v, equity0) for k, v in base.items()})

    # 3. inference on the primary (and unlevered) series
    def inference(res):
        r = np.array([p.ret for p in res])
        sr_boot = metrics.stationary_bootstrap(r, lambda x: metrics.sharpe(x), B=boot)
        tot_boot = metrics.stationary_bootstrap(r, lambda x: float(np.prod(1 + x) - 1), B=boot)
        bh = np.array([p.ret for p in base["buy_hold"]])
        diff_boot = metrics.stationary_bootstrap(r - bh, lambda x: float(x.mean()), B=boot)
        x = np.array([p.exposure_post_trade for p in res])
        btc = np.array([p.btc_return for p in res])
        fsum = np.array([p.funding_rate_sum for p in res])
        direction = np.sign(x)
        hits = int(((direction * btc) > 0).sum())
        n_dir = int((direction != 0).sum())
        return {
            "sharpe_ci": metrics.ci(sr_boot), "total_ci": metrics.ci(tot_boot),
            "excess_vs_bh_mean_daily": float((r - bh).mean()), "excess_vs_bh_ci": metrics.ci(diff_boot),
            "p_excess_le_0": float((diff_boot <= 0).mean()),
            "pt": metrics.pesaran_timmermann(direction, btc),
            "direction_hits": hits, "direction_n": n_dir,
            "binomial_p": metrics.binomial_p_one_sided(hits, n_dir) if n_dir else float("nan"),
            "perm": metrics.permutation_timing(x, btc, fsum, cost.all_in_rate()),
            "psr0": metrics.psr(r), "min_trl": metrics.min_trl(r),
        }

    inf = {"primary": inference(primary), "unlevered": inference(var_res["unlevered"])}
    trial_srs = [stats[f"agent_{k}"]["sharpe"] / math.sqrt(365) for k in variants]
    for k in inf:
        inf[k]["dsr"] = metrics.dsr(np.array([p.ret for p in var_res[k]]), trial_srs)
        inf[k]["dsr_trials"] = len(trial_srs)

    # rating diagnostics: what each rating was followed by (BTC, not the account)
    btc = {p.trade_date: p.btc_return for p in primary}
    by_rating = {}
    for rating in (*RATINGS, "ERROR", "REVIEW"):
        ds = [r["trade_date"] for r in rows if r["rating"] == rating]
        if ds:
            v = np.array([btc[d] for d in ds])
            by_rating[rating] = {"n": len(ds), "mean_btc": float(v.mean()), "up_share": float((v > 0).mean())}
    scored = [(RATING_SCORE[r["rating"]], btc[r["trade_date"]]) for r in rows if r["rating"] in RATING_SCORE]
    ic = metrics.spearman([a for a, _ in scored], [b for _, b in scored]) if len(scored) > 3 else float("nan")
    ic_p = metrics.spearman_perm_p([a for a, _ in scored], [b for _, b in scored]) if len(scored) > 3 else float("nan")
    lev = [r.get("leverage") for r in rows]
    lev_used = [x for x in lev if x]

    # random-rating Monte Carlo: the agent's own (rating, leverage) propensities, order randomized
    rng = np.random.default_rng(2026)
    pool = [decisions[d] for d in dates]
    mc_tot, mc_sr = [], []
    for _ in range(mc_draws):
        draw = {d: pool[rng.integers(len(pool))] for d in dates}
        res = replay(dates, tape, cost, policy, lambda d, x, a, dr=draw: dr[d], equity0, symbol)
        rr = np.array([p.ret for p in res])
        mc_tot.append(float(np.prod(1 + rr) - 1))
        mc_sr.append(metrics.sharpe(rr))
    mc_tot, mc_sr = np.array(mc_tot), np.array(mc_sr)
    agent_tot = stats["agent_primary"]["total_return"]

    # power analysis and break-even, from the window's own BTC returns
    btc_r = np.array([p.btc_return for p in primary])
    T = len(btc_r)
    abs_r, sig = float(np.abs(btc_r).mean()), float(btc_r.std(ddof=1))
    c = cost.all_in_rate()
    se_ann = math.sqrt(1 / T) * math.sqrt(365)
    sr_needed_daily = 1.96 / math.sqrt(T)
    # A sign forecaster at accuracy p earns (2p-1) E|r| a day; its daily Sharpe is about that over sigma.
    p_needed = 0.5 + sr_needed_daily * sig / (2 * abs_r)
    power = {
        "T": T, "btc_mean_abs": abs_r, "btc_sigma": sig, "cost_per_side": c,
        "se_sharpe_ann": se_ann, "min_sig_sharpe_ann": 1.96 * se_ann,
        "p_breakeven_turnover1": 0.5 + c / (2 * abs_r), "p_needed_5pct": p_needed,
        "hits_needed_binomial": next(k for k in range(T + 1) if metrics.binomial_p_one_sided(k, T) <= 0.05),
        "kelly_55": (0.10 * abs_r) / sig ** 2, "kelly_60": (0.20 * abs_r) / sig ** 2,
    }

    # venue facts measured in the window
    fr = tape.funding[(tape.funding["time"] >= t0) & (tape.funding["time"] < t1)]
    pred = np.array([funding_rate_1h(p) for p in fr["premium"]])
    venue = {
        "funding_prints": len(fr),
        "funding_formula_match": float(np.mean(np.abs(pred - fr["rate"].to_numpy()) < 1e-10)),
        "funding_at_baseline": float(np.isclose(fr["rate"], FUNDING_INTEREST_1H, atol=1e-12).mean()),
        "funding_mean_ann": float(fr["rate"].mean() * 8760),
        "funding_negative_share": float((fr["rate"] < 0).mean()),
        "premium_mean_bp": float(fr["premium"].mean() * 1e4),
        "btc_window_return": float(tape.price_at(t1) / tape.price_at(t0) - 1),
        "btc_start": tape.price_at(t0), "btc_end": tape.price_at(t1), "missing_bars": tape.missing_bars,
    }
    news = _news_lag_stats(run_dir, dates)

    pipeline = _pipeline_stats(rows)

    summary = {
        "generated": datetime.now(UTC).isoformat(), "spec": s, "pipeline": pipeline,
        "replay_max_abs_drift": drift, "stats": stats, "sensitivity": sens, "inference": inf,
        "by_rating": by_rating, "rating_counts": dict(Counter(r["rating"] for r in rows)),
        "ic": ic, "ic_p": ic_p,
        "leverage": {"used_days": len(lev_used), "mean_when_used": float(np.mean(lev_used)) if lev_used else 1.0,
                     "max": float(max(lev_used)) if lev_used else 1.0, "values": lev},
        "mc": {"draws": mc_draws, "agent_total_pct": float((mc_tot < agent_tot).mean()),
               "mean": float(mc_tot.mean()), "q05": float(np.quantile(mc_tot, 0.05)),
               "q95": float(np.quantile(mc_tot, 0.95)),
               "agent_sharpe_pct": float((mc_sr < stats["agent_primary"]["sharpe"]).mean())},
        "power": power, "venue": venue, "news": news,
        "errors": [r["trade_date"] for r in rows if r.get("error")],
        "wall_seconds_mean": float(np.mean([r.get("wall_seconds", 0) for r in rows])),
    }
    (out_dir / "generated" / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    _write_tables(summary, rows, primary, out_dir / "generated")
    if figures:
        _figures(primary, var_res, base, rows, tape, mc_tot, summary, out_dir / "figures", equity0)
    return summary


def _direction(label: str | None) -> int:
    if not label:
        return 0
    label = label.strip().title()
    if label in RATING_SCORE:
        return int(np.sign(RATING_SCORE[label]))
    return {"Buy": 1, "Sell": -1}.get(label, 0)


def _pipeline_stats(rows) -> dict:
    """How the desk's stages agreed: research manager -> trader -> portfolio manager."""
    rm = [_direction(r.get("research_recommendation")) for r in rows]
    tr = [_direction(r.get("trader_action")) for r in rows]
    pm = [_direction(r["rating"]) for r in rows]
    n = len(rows)
    rm_scores = [RATING_SCORE.get((r.get("research_recommendation") or "").title()) for r in rows]
    pm_scores = [RATING_SCORE.get(r["rating"]) for r in rows]
    softened = sum(1 for a, b in zip(rm_scores, pm_scores, strict=True)
                   if a is not None and b is not None and abs(b) < abs(a))
    sharpened = sum(1 for a, b in zip(rm_scores, pm_scores, strict=True)
                    if a is not None and b is not None and abs(b) > abs(a))
    return {
        "n": n,
        "pm_eq_rm_direction": sum(a == b for a, b in zip(pm, rm, strict=True)),
        "pm_eq_trader_direction": sum(a == b for a, b in zip(pm, tr, strict=True)),
        "pm_softened_rm": softened, "pm_sharpened_rm": sharpened,
        "rm_counts": dict(Counter((r.get("research_recommendation") or "n/a").title() for r in rows)),
        "trader_counts": dict(Counter((r.get("trader_action") or "n/a").title() for r in rows)),
        "sentiment_counts": dict(Counter(r.get("sentiment_band") or "n/a" for r in rows)),
    }


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _news_lag_stats(run_dir: Path, dates) -> dict:
    """Age at the cutoff of the newest headline each decision could read."""
    from tradingagents.dataflows.vendors import crypto_pit
    ages = []
    for d in dates:
        items = crypto_pit._collect(crypto_pit.ASSET_QUERIES["BTC"], clock.cutoff(d) - timedelta(days=7),
                                    clock.cutoff(d))
        if items:
            newest = max(datetime.fromisoformat(i["known_at"]) for i in items)
            ages.append((clock.cutoff(d) - newest).total_seconds() / 3600)
    date_only = 0
    total = 0
    for f in (run_dir / "cache" / "crypto_pit").glob("gn_*.json"):
        for it in json.loads(f.read_text()):
            total += 1
            date_only += crypto_pit.is_date_only_stamp(datetime.fromisoformat(it["stamp"]))
    return {"newest_headline_age_h_median": float(np.median(ages)) if ages else float("nan"),
            "newest_headline_age_h_max": float(max(ages)) if ages else float("nan"),
            "items_cached": total, "date_only_share": date_only / total if total else float("nan")}


# ---------------------------------------------------------------------------
# Tables and macros
# ---------------------------------------------------------------------------

def _pct(x, d=2):
    return "--" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}\\%"


def _sci(p: float) -> str:
    if p < 1e-12:
        return "$<10^{-12}$"
    if p < 1e-3:
        e = math.floor(math.log10(p))
        return f"${p / 10 ** e:.1f}\\times 10^{{{e}}}$"
    return f"{p * 100:.1f}\\%"


def _num(x, d=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


STRATEGY_LABELS = {
    "agent_primary": "TradingAgents-HL (primary, PM leverage $\\le$5x)",
    "agent_unlevered": "\\quad same decisions, 1x",
    "agent_long_flat": "\\quad same decisions, long/flat 1x",
    "agent_vol_target": "\\quad same decisions, vol-target 40\\%",
    "agent_primary_stops": "\\quad primary + trader stop-losses",
    **{k: v.replace("&", "\\&") for k, v in BASELINE_LABELS.items()},
}


def _write_tables(summary, rows, primary, gen: Path):
    st = summary["stats"]
    lines = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "Strategy & Return & Vol (ann.) & Sharpe & Sortino & Max DD & Turnover & Fees & Funding \\\\",
             "\\midrule"]
    for k in ("agent_primary", "agent_unlevered", "agent_long_flat", "agent_vol_target",
              "agent_primary_stops", "buy_hold", "tsmom30", "sma20", "macd", "rsi_rev", "cash"):
        v = st[k]
        if k == "buy_hold":
            lines.append("\\midrule")
        lines.append(f"{STRATEGY_LABELS[k]} & {_pct(v['total_return'])} & {_pct(v['vol_ann'], 1)} & "
                     f"{_num(v['sharpe'])} & {_num(v['sortino'])} & {_pct(-v['mdd'])} & "
                     f"{_num(v['turnover'], 1)} & {v['fees']:,.0f} & {v['funding']:+,.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "table_performance.tex").write_text("\n".join(lines))

    sens = summary["sensitivity"]
    labels = {"base": "Base: tier 0 taker 4.5 bp + 1 bp slippage", "no_slippage": "Taker, no slippage",
              "slippage_2bp": "Slippage 2 bp", "slippage_5bp": "Slippage 5 bp",
              "tier_3": "Fee tier 3 (3.0 bp taker)", "tier_6_diamond": "Tier 6 + Diamond staking (1.44 bp)",
              "maker_fills": "All fills as maker (1.5 bp)", "no_costs": "Frictionless (no fee, no slippage)"}
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Cost model & Return & Sharpe & Fees & Slippage \\\\",
             "\\midrule"]
    for k, lab in labels.items():
        v = sens[k]
        lines.append(f"{lab} & {_pct(v['total_return'])} & {_num(v['sharpe'])} & {v['fees']:,.0f} & "
                     f"{v['slippage']:,.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "table_costs.tex").write_text("\n".join(lines))

    lines = ["\\begin{longtable}{llrrrrrrr}", "\\toprule",
             "Date & Rating & Lev. & Target $x$ & Fill & BTC & Net & Fees & Funding \\\\", "\\midrule",
             "\\endhead"]
    for r, p in zip(rows, primary, strict=True):
        lines.append(f"{r['trade_date']} & {r['rating']} & {r['leverage'] or '--'} & {p.target:+.2f} & "
                     f"{p.ref_px:,.0f} & {_pct(p.btc_return)} & {_pct(p.ret)} & {p.fee:,.0f} & "
                     f"{p.funding:+,.0f} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    (gen / "table_ledger.tex").write_text("\n".join(lines))

    pw = summary["power"]
    m = 0.0125
    sig, abs_r = pw["btc_sigma"], pw["btc_mean_abs"]
    mu55 = 0.10 * abs_r
    lines = ["\\begin{tabular}{rrrrr}", "\\toprule",
             "$L$ & Adverse move to liquidation & $P$(liquidated within 1 day) & $P$(within 30 days) & "
             "Growth at $p=0.55$ (ann.) \\\\", "\\midrule"]
    for lev in (1, 2, 3, 5, 10, 20, 40):
        d = (1 / lev - m) / (1 - m)
        p1 = 2 * metrics.N01.cdf(math.log(1 - d) / sig) if d < 1 else 0.0
        p30 = 1 - (1 - p1) ** 30
        g = (lev * mu55 - 0.5 * lev ** 2 * sig ** 2) * 365
        lines.append(f"{lev}x & {d * 100:.2f}\\% & {_sci(p1)} & {_sci(p30)} & {g * 100:+.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "table_leverage.tex").write_text("\n".join(lines))

    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "PM rating & Days & Mean next-period BTC return & Share up \\\\",
             "\\midrule"]
    for k, v in summary["by_rating"].items():
        lines.append(f"{k} & {v['n']} & {_pct(v['mean_btc'])} & {v['up_share'] * 100:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen / "table_ratings.tex").write_text("\n".join(lines))

    # every number the text quotes, as macros
    a, u = st["agent_primary"], st["agent_unlevered"]
    ip, iu = summary["inference"]["primary"], summary["inference"]["unlevered"]
    pw, ve, mc, nw = summary["power"], summary["venue"], summary["mc"], summary["news"]
    macros = {
        "AgentRet": _pct(a["total_return"]), "AgentSharpe": _num(a["sharpe"]), "AgentMDD": _pct(a["mdd"], 2).lstrip("+"),
        "AgentVol": _pct(a["vol_ann"], 1).lstrip("+"), "AgentFees": f"{a['fees']:,.0f}",
        "AgentFunding": f"{a['funding']:+,.0f}", "AgentTurnover": _num(a["turnover"], 1),
        "AgentTrades": str(a["trades"]), "AgentEnd": f"{a['equity_end']:,.0f}",
        "AgentHit": f"{a['hit_rate'] * 100:.0f}\\%", "AgentMaxExp": _num(a["max_abs_exposure"], 2),
        "AgentMeanExp": _num(a["mean_abs_exposure"], 2),
        "UnlevRet": _pct(u["total_return"]), "UnlevSharpe": _num(u["sharpe"]), "UnlevMDD": _pct(u["mdd"]).lstrip("+"),
        "BHRet": _pct(st["buy_hold"]["total_return"]), "BHSharpe": _num(st["buy_hold"]["sharpe"]),
        "BHFunding": f"{st['buy_hold']['funding']:+,.0f}", "BHMDD": _pct(st["buy_hold"]["mdd"]).lstrip("+"),
        "TSMOMRet": _pct(st["tsmom30"]["total_return"]), "MACDRet": _pct(st["macd"]["total_return"]),
        "SharpeLo": _num(ip["sharpe_ci"][0]), "SharpeHi": _num(ip["sharpe_ci"][1]),
        "RetLo": _pct(ip["total_ci"][0]), "RetHi": _pct(ip["total_ci"][1]),
        "ExcessDaily": _pct(ip["excess_vs_bh_mean_daily"], 3), "ExcessLo": _pct(ip["excess_vs_bh_ci"][0], 3),
        "ExcessHi": _pct(ip["excess_vs_bh_ci"][1], 3), "PSR": _num(ip["psr0"], 3), "DSR": _num(ip["dsr"], 3),
        "MinTRL": "$\\infty$" if not np.isfinite(ip["min_trl"]) else f"{ip['min_trl']:.0f}",
        "PTstat": _num(ip["pt"]["stat"]), "PTp": _num(ip["pt"]["p"], 3), "DirHits": str(ip["direction_hits"]),
        "DirN": str(ip["direction_n"]), "BinomP": _num(ip["binomial_p"], 3),
        "PermGross": _num(ip["perm"]["p_gross"], 3), "PermNet": _num(ip["perm"]["p_net"], 3),
        "UnlevPSR": _num(iu["psr0"], 3), "UnlevPermNet": _num(iu["perm"]["p_net"], 3),
        "IC": _num(summary["ic"]), "ICp": _num(summary["ic_p"], 3),
        "MCpct": f"{mc['agent_total_pct'] * 100:.0f}", "MCqLow": _pct(mc["q05"]), "MCqHigh": _pct(mc["q95"]),
        "MCDraws": f"{mc['draws']:,}",
        "LevDays": str(summary["leverage"]["used_days"]), "LevMean": _num(summary["leverage"]["mean_when_used"], 1),
        "LevMax": _num(summary["leverage"]["max"], 1),
        "PowT": str(pw["T"]), "PowSE": _num(pw["se_sharpe_ann"]), "PowMinSR": _num(pw["min_sig_sharpe_ann"], 1),
        "PowP": f"{pw['p_needed_5pct'] * 100:.0f}\\%", "PowBreakeven": f"{pw['p_breakeven_turnover1'] * 100:.1f}\\%",
        "PowHits": str(pw["hits_needed_binomial"]), "BTCabs": _pct(pw["btc_mean_abs"]).lstrip("+"),
        "BTCsigma": _pct(pw["btc_sigma"]).lstrip("+"), "KellyFive": _num(pw["kelly_55"], 1),
        "KellySix": _num(pw["kelly_60"], 1), "CostSide": f"{pw['cost_per_side'] * 1e4:.1f}",
        "FundPrints": str(ve["funding_prints"]), "FundMatch": f"{ve['funding_formula_match'] * 100:.1f}\\%",
        "FundBase": f"{ve['funding_at_baseline'] * 100:.0f}\\%", "FundAnn": _pct(ve["funding_mean_ann"]),
        "FundNeg": f"{ve['funding_negative_share'] * 100:.0f}\\%", "PremBp": f"{ve['premium_mean_bp']:+.2f}",
        "BTCWindow": _pct(ve["btc_window_return"]), "BTCStart": f"{ve['btc_start']:,.0f}",
        "BTCEnd": f"{ve['btc_end']:,.0f}",
        "NewsAgeMed": f"{nw['newest_headline_age_h_median']:.0f}", "NewsAgeMax": f"{nw['newest_headline_age_h_max']:.0f}",
        "NewsItems": f"{nw['items_cached']:,}", "NewsDateOnly": f"{nw['date_only_share'] * 100:.0f}\\%",
        "Decisions": str(len(rows)), "Errors": str(len(summary["errors"])),
        "WallMin": f"{summary['wall_seconds_mean'] / 60:.1f}",
        "ReplayDrift": f"{summary['replay_max_abs_drift']:.1e}",
    }
    for rating in RATINGS:
        macros[f"N{rating}"] = str(summary["rating_counts"].get(rating, 0))
    pl = summary["pipeline"]
    macros.update({"PMeqRM": str(pl["pm_eq_rm_direction"]), "PMeqTrader": str(pl["pm_eq_trader_direction"]),
                   "PMsoftened": str(pl["pm_softened_rm"]), "PMsharpened": str(pl["pm_sharpened_rm"])})
    bad = [k for k in macros if not k.isalpha()]
    if bad:   # a TeX control sequence is letters only: \\MCq05 would define \\MCq and print "05"
        raise ValueError(f"macro names must be letters only: {bad}")
    text = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items())
    (gen / "numbers.tex").write_text(text + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7,
        "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 200, "figure.dpi": 120,
        "axes.facecolor": "#fcfcfb", "figure.facecolor": "white",
    })
    return plt


def _dodged_end_labels(ax, ends, gap_frac: float = 0.065):
    """Direct labels at the line ends, pushed apart so none overlap, with a leader to each end."""
    lo, hi = ax.get_ylim()
    gap = gap_frac * (hi - lo)
    ordered = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ordered]
    for _ in range(50):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                push = (gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= push
                ys[i] += push
                moved = True
        if not moved:
            break
    import matplotlib.dates as mdates
    for (x, y, text, color), ly in zip(ordered, ys, strict=True):
        ax.plot([x], [y], marker="o", ms=3, color=color, clip_on=False, zorder=5)
        xt = mdates.date2num(x) + 0.8
        ax.plot([mdates.date2num(x), xt], [y, ly], color=GRID if abs(ly - y) < 1e-9 else INK2, lw=0.5,
                clip_on=False)
        ax.annotate(text, (xt, ly), xytext=(2, 0), textcoords="offset points", va="center", fontsize=6.5,
                    color=INK, annotation_clip=False)
    ax.set_ylim(lo, hi)


def _figures(primary, var_res, base, rows, tape, mc_tot, summary, figdir: Path, equity0: float):
    plt = _style()
    import matplotlib.dates as mdates

    curves = [("agent", "TradingAgents-HL", primary, "-"),
              ("agent_1x", "Same decisions, 1x", var_res["unlevered"], (0, (4, 2))),
              ("buy_hold", "Buy & hold", base["buy_hold"], "-"),
              ("tsmom30", "TSMOM(30)", base["tsmom30"], (0, (1, 1.5))),
              ("macd", "MACD", base["macd"], (0, (6, 2, 1, 2)))]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ends = []
    for key, label, res, ls in curves:
        e = equity_path(res, equity0) / equity0 * 100
        ax.plot(e.index, e.values, color=COLORS[key], ls=ls, lw=1.8 if key == "agent" else 1.2, label=label)
        ends.append((e.index[-1], float(e.values[-1]), f"{label} {e.values[-1] - 100:+.1f}%", COLORS[key]))
    ax.axhline(100, color=INK2, lw=0.6)
    _dodged_end_labels(ax, ends)
    ax.set_ylabel("Equity (start = 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, handlelength=3.2)
    for handle in leg.legend_handles:
        handle.set_linewidth(2.2)
    ax.set_title("Net equity, marked every 15 minutes, 2026-10-01 00:15 to 2026-10-31 00:15 UTC", loc="left")
    fig.savefig(figdir / "equity.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    for key, label, res, ls in (curves[0], curves[1], curves[2]):
        e = equity_path(res, equity0)
        dd = (e / e.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, color=COLORS[key], ls=ls, lw=1.2, label=label)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="lower left", ncol=3)
    fig.savefig(figdir / "drawdown.pdf")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.4), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    bars = tape.bars[(tape.bars["open_time"] >= pd.Timestamp(primary[0].exec_time)) &
                     (tape.bars["close_time"] <= pd.Timestamp(primary[-1].next_exec_time))]
    a1.plot(bars["close_time"], bars["close"], color=INK, lw=0.8)
    a1.set_ylabel("BTC perp (USDC)")
    a1.set_title("BTC price and the agents' net exposure (positive = long, negative = short)", loc="left")
    t = [pd.Timestamp(p.exec_time) for p in primary]
    x = [p.exposure_post_trade for p in primary]
    a2.bar(t, x, width=0.92, align="edge", color=[LONG if v > 0 else SHORT if v < 0 else NEUTRAL for v in x],
           edgecolor="#fcfcfb", linewidth=1.0)
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_ylabel("Exposure (x equity)")
    for p, r in zip(primary, rows, strict=True):
        if r.get("leverage") and r["leverage"] > 1 and abs(p.exposure_post_trade) > 1:
            a2.annotate(f"{r['leverage']:g}x", (pd.Timestamp(p.exec_time) + pd.Timedelta(hours=11),
                        p.exposure_post_trade), ha="center", va="bottom" if p.exposure_post_trade > 0 else "top",
                        fontsize=5.5, color=INK2)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.savefig(figdir / "position.pdf")
    plt.close(fig)

    a = summary["stats"]["agent_primary"]
    price_pnl = a["equity_end"] - equity0 + a["fees"] + a["slippage"] + a["funding"]
    steps = [("Price P&L", price_pnl), ("Fees", -a["fees"]), ("Slippage", -a["slippage"]),
             ("Funding", -a["funding"])]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    level = 0.0
    for i, (_, v) in enumerate(steps):
        ax.bar(i, v, bottom=level, color=LONG if v >= 0 else SHORT, width=0.6, edgecolor="#fcfcfb", linewidth=1)
        ax.annotate(f"{v:+,.0f}", (i, level + v), xytext=(0, 3 if v >= 0 else -9), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK)
        level += v
    ax.bar(len(steps), level, color=INK2, width=0.6)
    ax.annotate(f"{level:+,.0f}", (len(steps), level), xytext=(0, 3 if level >= 0 else -9),
                textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(steps) + 1), [s for s, _ in steps] + ["Net"])
    ax.axhline(0, color=INK2, lw=0.6)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.set_ylabel("USDC")
    ax.set_title("Where the primary run's P&L came from", loc="left")
    fig.savefig(figdir / "costs.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.hist(mc_tot * 100, bins=40, color="#86b6ef", edgecolor="#fcfcfb", linewidth=0.6)
    top = ax.get_ylim()[1]
    for (val, lab, col), h in zip(((a["total_return"], "TradingAgents-HL", COLORS["agent"]),
                                   (summary["stats"]["buy_hold"]["total_return"], "Buy & hold",
                                    COLORS["buy_hold"])), (0.94, 0.80), strict=True):
        ax.axvline(val * 100, color=col, lw=1.6)
        ax.annotate(f"{lab} {val * 100:+.1f}%", (val * 100, top * h), xytext=(3, 0), textcoords="offset points",
                    fontsize=6.5, color=INK)
    ax.set_ylim(0, top * 1.02)
    ax.set_xlabel("Total return over the window (%)")
    ax.set_ylabel("Random draws")
    ax.set_title(f"Same rating propensities, random order ({summary['mc']['draws']:,} draws)", loc="left")
    fig.savefig(figdir / "montecarlo.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    rng = np.random.default_rng(1)
    order = [r for r in RATINGS if any(row["rating"] == r for row in rows)]
    for i, rating in enumerate(order):
        v = np.array([p.btc_return for p, row in zip(primary, rows, strict=True) if row["rating"] == rating]) * 100
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COLORS["agent"], edgecolor="#fcfcfb",
                   linewidth=0.8, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean(), v.mean()], color=INK, lw=1.2, zorder=4)
    ax.axhline(0, color=INK2, lw=0.6)
    counts = Counter(row["rating"] for row in rows)
    ax.set_xticks(range(len(order)), [f"{r}\nn={counts[r]}" for r in order])
    ax.set_ylabel("BTC return over the next period (%)")
    ax.set_title("What followed each Portfolio Manager rating (bar = mean)", loc="left")
    fig.savefig(figdir / "ratings.pdf")
    plt.close(fig)


def as_dict(obj) -> dict:
    return asdict(obj)
