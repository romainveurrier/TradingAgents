"""Performance and inference for short, daily, 24/7 return series.

Conventions: ``r`` is the series of per-decision (daily) net returns on equity.
Annualization uses 365 periods (the market never closes). Skewness is the
sample ``g1``; kurtosis is the *raw* (non-excess) sample ``g2 + 3``, as in the
formulas below.

Sharpe-ratio inference, with ``SR`` the per-period ratio and ``T`` observations:

- Lo (2002), IID normal:   Var(SR̂) ≈ (1 + SR²/2) / T
- Mertens (2002), IID non-normal: Var(SR̂) ≈ (1 − γ₃ SR + (γ₄ − 1)/4 · SR²) / T
- Probabilistic Sharpe ratio (Bailey & López de Prado 2012):
  PSR(SR*) = Φ( (SR̂ − SR*) √(T − 1) / √(1 − γ₃ SR̂ + (γ₄ − 1)/4 · SR̂²) )
- Minimum track record length at confidence 1 − α:
  MinTRL = 1 + (1 − γ₃ SR̂ + (γ₄ − 1)/4 · SR̂²) · (z₁₋α / (SR̂ − SR*))²
- Deflated Sharpe ratio (Bailey & López de Prado 2014): PSR with
  SR* = √V[SR_n] · ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e))), γ the
  Euler–Mascheroni constant, over the N variants tried.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

N01 = NormalDist()
EULER_GAMMA = 0.5772156649015329
PERIODS = 365


def moments(r) -> dict:
    r = np.asarray(r, dtype=float)
    T = len(r)
    mu = float(r.mean()) if T else float("nan")
    sd = float(r.std(ddof=1)) if T > 1 else float("nan")
    c = r - mu
    m2 = float((c ** 2).mean()) if T else float("nan")
    skew = float((c ** 3).mean() / m2 ** 1.5) if T and m2 > 0 else 0.0
    kurt = float((c ** 4).mean() / m2 ** 2) if T and m2 > 0 else 3.0
    return {"T": T, "mean": mu, "std": sd, "skew": skew, "kurt": kurt}


def sharpe(r, periods: int = PERIODS) -> float:
    m = moments(r)
    return m["mean"] / m["std"] * math.sqrt(periods) if m["std"] and m["std"] > 0 else float("nan")


def sortino(r, periods: int = PERIODS) -> float:
    r = np.asarray(r, dtype=float)
    dd = math.sqrt(float((np.minimum(r, 0) ** 2).mean())) if len(r) else float("nan")
    return float(r.mean()) / dd * math.sqrt(periods) if dd > 0 else float("nan")


def _sr_var_term(sr: float, skew: float, kurt: float) -> float:
    return 1 - skew * sr + (kurt - 1) / 4 * sr ** 2


def sharpe_se_lo(sr_period: float, T: int) -> float:
    return math.sqrt((1 + 0.5 * sr_period ** 2) / T)


def sharpe_se_mertens(sr_period: float, T: int, skew: float, kurt: float) -> float:
    return math.sqrt(max(_sr_var_term(sr_period, skew, kurt), 1e-12) / T)


def psr(r, sr_star_period: float = 0.0) -> float:
    m = moments(r)
    if not (m["std"] and m["std"] > 0) or m["T"] < 3:
        return float("nan")
    sr = m["mean"] / m["std"]
    denom = math.sqrt(max(_sr_var_term(sr, m["skew"], m["kurt"]), 1e-12))
    return N01.cdf((sr - sr_star_period) * math.sqrt(m["T"] - 1) / denom)


def min_trl(r, sr_star_period: float = 0.0, alpha: float = 0.05) -> float:
    m = moments(r)
    if not (m["std"] and m["std"] > 0):
        return float("nan")
    sr = m["mean"] / m["std"]
    if sr <= sr_star_period:
        return float("inf")
    z = N01.inv_cdf(1 - alpha)
    return 1 + _sr_var_term(sr, m["skew"], m["kurt"]) * (z / (sr - sr_star_period)) ** 2


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    a = N01.inv_cdf(1 - 1 / n_trials)
    b = N01.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def dsr(r, trial_srs_period) -> float:
    trial = np.asarray(list(trial_srs_period), dtype=float)
    trial = trial[np.isfinite(trial)]
    if len(trial) < 2:
        return psr(r)
    return psr(r, expected_max_sharpe(len(trial), float(trial.var(ddof=1))))


def max_drawdown(equity: pd.Series) -> dict:
    e = pd.Series(equity).astype(float)
    peak = e.cummax()
    dd = e / peak - 1
    trough = dd.idxmin() if len(dd) else None
    start = e.loc[:trough].idxmax() if trough is not None else None
    return {"mdd": float(-dd.min()) if len(dd) else 0.0, "peak": start, "trough": trough}


def stationary_bootstrap(r, stat, B: int = 10000, mean_block: float = 3.0, seed: int = 7) -> np.ndarray:
    """Politis & Romano (1994): resample blocks of geometric length, wrapping around."""
    r = np.asarray(r, dtype=float)
    T = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(B)
    for b in range(B):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for t in range(T):
            i = rng.integers(T) if t == 0 or rng.random() < p else (i + 1) % T
            idx[t] = i
        out[b] = stat(r[idx])
    return out


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if not len(s):
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def pesaran_timmermann(pred, actual) -> dict:
    """Directional accuracy test (Pesaran & Timmermann 1992) on the non-zero predictions."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    keep = (pred != 0) & (actual != 0)
    x, y = pred[keep] > 0, actual[keep] > 0
    n = int(keep.sum())
    if n < 5:
        return {"n": n, "hit": float("nan"), "stat": float("nan"), "p": float("nan")}
    P = float((x == y).mean())
    py, px = float(y.mean()), float(x.mean())
    pstar = py * px + (1 - py) * (1 - px)
    v_p = pstar * (1 - pstar) / n
    v_pstar = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
               + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = v_p - v_pstar
    stat = (P - pstar) / math.sqrt(denom) if denom > 0 else float("nan")
    return {"n": n, "hit": P, "expected_hit": pstar, "stat": stat,
            "p": 1 - N01.cdf(stat) if math.isfinite(stat) else float("nan")}


def binomial_p_one_sided(k: int, n: int, p: float = 0.5) -> float:
    """P[X >= k] for X ~ Bin(n, p)."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def linear_returns(x, r_btc, funding_sum, cost_rate: float, x0: float = 0.0) -> np.ndarray:
    """First-order net return of holding exposures ``x`` (set at each period start).

    r_t ≈ x_t r_t^BTC − x_t F_t − c |x_t − x_{t−1}|; used for the permutation
    tests, where re-running the engine for every shuffle adds nothing at this order.
    """
    x = np.asarray(x, float)
    prev = np.concatenate([[x0], x[:-1]])
    return x * np.asarray(r_btc, float) - x * np.asarray(funding_sum, float) - cost_rate * np.abs(x - prev)


def permutation_timing(x, r_btc, funding_sum, cost_rate: float, B: int = 20000, seed: int = 11) -> dict:
    """Does the *order* of the exposures matter? Shuffle them across periods.

    Keeps the exposure distribution (how often long, short, flat) and destroys
    timing. Reports the share of shuffles whose mean return is at least the
    observed one, gross of costs and net of them (turnover changes when shuffled).
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    obs_gross = float((x * np.asarray(r_btc)).mean())
    obs_net = float(linear_returns(x, r_btc, funding_sum, cost_rate).mean())
    g = n = 0
    for _ in range(B):
        xs = rng.permutation(x)
        g += (xs * np.asarray(r_btc)).mean() >= obs_gross - 1e-15
        n += linear_returns(xs, r_btc, funding_sum, cost_rate).mean() >= obs_net - 1e-15
    return {"obs_gross": obs_gross, "obs_net": obs_net, "p_gross": (g + 1) / (B + 1), "p_net": (n + 1) / (B + 1)}


def spearman(a, b) -> float:
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    return float(a.rank().corr(b.rank()))


def spearman_perm_p(a, b, B: int = 20000, seed: int = 5) -> float:
    rho = spearman(a, b)
    if not math.isfinite(rho):
        return float("nan")
    rng = np.random.default_rng(seed)
    ra = pd.Series(a, dtype=float).rank().to_numpy()
    rb = pd.Series(b, dtype=float).rank().to_numpy()
    hits = sum(abs(np.corrcoef(rng.permutation(ra), rb)[0, 1]) >= abs(rho) - 1e-12 for _ in range(B))
    return (hits + 1) / (B + 1)


def summarize(r, equity: pd.Series | None = None, periods: int = PERIODS) -> dict:
    """Headline performance of a return series (and its intraperiod equity path)."""
    r = np.asarray(r, float)
    m = moments(r)
    total = float(np.prod(1 + r) - 1) if len(r) else 0.0
    sr = sharpe(r, periods)
    sr_p = sr / math.sqrt(periods) if math.isfinite(sr) else float("nan")
    eq = equity if equity is not None else pd.Series(np.cumprod(np.concatenate([[1.0], 1 + r])))
    dd = max_drawdown(eq)
    ann = (1 + total) ** (periods / len(r)) - 1 if len(r) and total > -1 else float("nan")
    return {
        "periods": m["T"], "total_return": total, "mean_daily": m["mean"], "vol_daily": m["std"],
        "vol_ann": m["std"] * math.sqrt(periods) if math.isfinite(m["std"]) else float("nan"),
        "ann_return": ann, "sharpe": sr, "sortino": sortino(r, periods),
        "skew": m["skew"], "kurt": m["kurt"], "mdd": dd["mdd"],
        "calmar": ann / dd["mdd"] if dd["mdd"] > 0 and math.isfinite(ann) else float("nan"),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "se_sharpe_lo": sharpe_se_lo(sr_p, m["T"]) * math.sqrt(periods) if m["T"] and math.isfinite(sr_p) else float("nan"),
        "se_sharpe_mertens": (sharpe_se_mertens(sr_p, m["T"], m["skew"], m["kurt"]) * math.sqrt(periods)
                              if m["T"] and math.isfinite(sr_p) else float("nan")),
        "psr0": psr(r), "min_trl": min_trl(r),
    }
