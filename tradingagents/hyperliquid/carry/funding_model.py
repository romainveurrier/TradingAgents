"""Forecasting Hyperliquid funding: the clamp-band model and its baselines.

The venue charges ``f = F_8h / 8`` with ``F_8h = I + soft(P − I, c)``, where
``soft(x, c) = sign(x)·max(|x| − c, 0)``, ``I = 1e-4`` and ``c = 5e-4``
(methods paper, Prop. 2). Funding therefore sits at ``I/8`` whenever the hourly
premium ``P`` stays inside the band ``[I − c, I + c]``.

Model (clamp-band AR):

- The premium follows a Gaussian AR(1), the discrete OU process
  ``P_{t+1} = θ + φ (P_t − θ) + σ ε``.
- Its h-step forecast is Gaussian, with mean ``θ + φ^h (P_t − θ)`` and
  variance ``σ² (1 − φ^{2h}) / (1 − φ²)``.
- Under a Gaussian ``Z ~ N(μ, s²)`` the expected soft threshold is closed-form:
  ``E[(Z − c)⁺] − E[(−Z − c)⁺]``, with
  ``E[(Z − c)⁺] = (μ − c)·Φ((μ − c)/s) + s·φ((μ − c)/s)``.

The forecast of mean funding over the next H hours averages that expectation
over ``h = 1..H``. Baselines are persistence (the last 24 h mean, the last
7 d mean) and a ridge regression on lagged funding and premium features.

Walk-forward: parameters are refit on an expanding window at the start of
each month, from data strictly before it, and forecasts are scored
out-of-sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents.hyperliquid.venue import FUNDING_CLAMP_8H, FUNDING_INTEREST_8H

HOURS_PER_YEAR = 8760
_SQRT2PI = math.sqrt(2 * math.pi)


def _phi(x):
    return np.exp(-0.5 * x * x) / _SQRT2PI


_erf = np.vectorize(math.erf, otypes=[float])


def _norm_cdf(x):
    return 0.5 * (1 + _erf(np.asarray(x, dtype=float) / math.sqrt(2)))


def expected_soft(mu, s, c: float = FUNDING_CLAMP_8H):
    """E[soft(Z, c)] for Z ~ N(mu, s^2), elementwise."""
    mu, s = np.asarray(mu, float), np.maximum(np.asarray(s, float), 1e-12)
    a, b = (mu - c) / s, (-mu - c) / s
    pos = (mu - c) * _norm_cdf(a) + s * _phi(a)
    neg = (-mu - c) * _norm_cdf(b) + s * _phi(b)
    return pos - neg


def funding_from_premium_8h(premium_8h, interest: float = FUNDING_INTEREST_8H, c: float = FUNDING_CLAMP_8H):
    x = np.asarray(premium_8h, float) - interest
    return interest + np.sign(x) * np.maximum(np.abs(x) - c, 0.0)


@dataclass
class ARFit:
    theta: float
    phi: float
    sigma: float

    @classmethod
    def fit(cls, p: np.ndarray) -> ARFit:
        x, y = p[:-1], p[1:]
        xm, ym = x.mean(), y.mean()
        phi = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
        phi = min(max(phi, -0.999), 0.9999)
        theta = float((ym - phi * xm) / (1 - phi))
        resid = y - (theta + phi * (x - theta))
        return cls(theta, phi, float(resid.std(ddof=2)))

    def mean_funding(self, p_now: float, horizon: int) -> float:
        """Expected mean hourly funding over the next ``horizon`` hours."""
        h = np.arange(1, horizon + 1)
        mu = self.theta + self.phi ** h * (p_now - self.theta)
        var = self.sigma ** 2 * (1 - self.phi ** (2 * h)) / max(1 - self.phi ** 2, 1e-12)
        f8 = FUNDING_INTEREST_8H + expected_soft(mu - FUNDING_INTEREST_8H, np.sqrt(var))
        return float(np.mean(f8) / 8)


def features(f: pd.DataFrame) -> pd.DataFrame:
    """Lagged features at each hour t, from prints strictly before t."""
    r, p = f["rate"], f["premium"]
    out = pd.DataFrame(index=f.index)
    for w in (1, 24, 72, 168):
        out[f"f{w}"] = r.rolling(w).mean().shift(1)
        out[f"p{w}"] = p.rolling(w).mean().shift(1)
    out["p_last"] = p.shift(1)
    at_floor = pd.Series(np.isclose(r, FUNDING_INTEREST_8H / 8, atol=1e-12), index=r.index, dtype=float)
    out["base_share_168"] = at_floor.rolling(168).mean().shift(1)
    return out


def target(f: pd.DataFrame, horizon: int) -> pd.Series:
    """Mean hourly funding over [t, t + horizon): what a position opened at t would collect."""
    return f["rate"][::-1].rolling(horizon).mean()[::-1]


def walk_forward(f: pd.DataFrame, horizon: int = 168, start: str = "2024-01-01",
                 ridge: float = 1e-2) -> pd.DataFrame:
    """Out-of-sample forecasts of mean funding over the next ``horizon`` hours, refit monthly."""
    X = features(f)
    y = target(f, horizon)
    rows = []
    months = pd.date_range(pd.Timestamp(start, tz="UTC"), f.index[-1], freq="MS")
    cols = list(X.columns)
    for m0, m1 in zip(months, list(months[1:]) + [f.index[-1] + pd.Timedelta(hours=1)], strict=True):
        # Train on rows whose target window ended before the month starts (no overlap leak).
        train_end = m0 - pd.Timedelta(hours=horizon)
        tr = X.index < train_end
        Xtr, ytr = X[tr].dropna(), y[tr]
        ytr = ytr.loc[Xtr.index].dropna()
        Xtr = Xtr.loc[ytr.index]
        ar = ARFit.fit(f["premium"][f.index < m0].to_numpy())
        dev = (f["premium"] - f["premium"].rolling(168).mean().shift(1))[f.index < m0].dropna().to_numpy()
        ar_local = ARFit.fit(dev)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        Z = ((Xtr - mu) / sd).to_numpy()
        A = np.c_[np.ones(len(Z)), Z]
        beta = np.linalg.solve(A.T @ A + ridge * len(Z) * np.eye(A.shape[1]) * np.r_[0, np.ones(Z.shape[1])],
                               A.T @ ytr.to_numpy())
        te = (X.index >= m0) & (X.index < m1)
        Xte = X[te]
        for t, row in Xte.iterrows():
            if row.isna().any():
                continue
            z = ((row[cols] - mu) / sd).to_numpy()
            rows.append({
                "t": t, "y": y.get(t, np.nan),
                "persist24": row["f24"], "persist168": row["f168"],
                "clamp_ar": ar.mean_funding(row["p_last"], horizon),
                # Local level: the premium reverts toward its recent (168 h) mean, not a global one.
                "clamp_ar_local": ARFit(row["p168"], ar_local.phi, ar_local.sigma).mean_funding(row["p_last"], horizon),
                "ridge": float(beta[0] + z @ beta[1:]),
            })
    out = pd.DataFrame(rows).set_index("t")
    return out


def score(fc: pd.DataFrame, threshold_ann: float = 0.05) -> pd.DataFrame:
    """RMSE (annualized units), out-of-sample R² against persistence, and the hit rate of
    the carry decision: whether funding beats ``threshold_ann`` (the borrow rate)."""
    d = fc.dropna()
    y = d["y"] * HOURS_PER_YEAR
    base = ((d["persist168"] * HOURS_PER_YEAR - y) ** 2).sum()
    rows = {}
    for m in ("persist24", "persist168", "clamp_ar", "clamp_ar_local", "ridge"):
        yhat = d[m] * HOURS_PER_YEAR
        sse = ((yhat - y) ** 2).sum()
        rows[m] = {"rmse_ann": float(np.sqrt(sse / len(d))), "r2_vs_persist168": float(1 - sse / base),
                   "decision_hit": float(((yhat > threshold_ann) == (y > threshold_ann)).mean()),
                   "n": len(d)}
    return pd.DataFrame(rows).T
