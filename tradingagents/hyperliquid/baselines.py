"""Rule-based baselines, decided at the same cutoffs and priced by the same engine.

Each signal reads only the daily candles closed by the decision's cutoff
(``hyperliquid._daily_frame``), so a baseline is held to the same
point-in-time standard as the agents. The set mirrors the comparisons in the
TradingAgents paper (buy-and-hold, MACD, RSI/mean reversion, SMA) plus
time-series momentum (Moskowitz, Ooi & Pedersen 2012).
"""

from __future__ import annotations

import numpy as np
from stockstats import wrap

from tradingagents.dataflows.vendors.hyperliquid import _daily_frame
from tradingagents.hyperliquid.policy import ExecutionPolicy
from tradingagents.hyperliquid.simulate import MarketTape, PeriodResult, replay
from tradingagents.hyperliquid.venue import CostModel


def _closes(symbol: str, d: str):
    return _daily_frame(symbol, d, history_days=120)


def _sign(x: float) -> float:
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


def signal(name: str, symbol: str, d: str) -> float:
    df = _closes(symbol, d)
    c = df["Close"].to_numpy(float)
    if name == "cash":
        return 0.0
    if name == "buy_hold":
        return 1.0
    if name == "tsmom30":
        return _sign(c[-1] / c[-31] - 1)
    if name == "sma20":
        return 1.0 if c[-1] > c[-20:].mean() else -1.0
    stock = wrap(df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy())
    if name == "macd":
        return _sign(float(stock["macdh"].iloc[-1]))
    if name == "rsi_rev":
        rsi = float(stock["rsi"].iloc[-1])
        return 1.0 if rsi < 35 else -1.0 if rsi > 65 else 0.0
    raise ValueError(f"unknown baseline {name!r}")


BASELINES = ("buy_hold", "tsmom30", "sma20", "macd", "rsi_rev", "cash")
LABELS = {
    "buy_hold": "Buy & hold (1x long)", "tsmom30": "Time-series momentum (30 d)",
    "sma20": "SMA(20) trend", "macd": "MACD histogram", "rsi_rev": "RSI(14) reversion",
    "cash": "Cash",
}


def run_baselines(dates, tape: MarketTape, cost: CostModel, equity0: float, symbol: str,
                  names=BASELINES) -> dict[str, list[PeriodResult]]:
    out = {}
    for name in names:
        # Buy & hold never rebalances its drift; the others trade daily signals.
        policy = ExecutionPolicy(band=np.inf) if name == "buy_hold" else ExecutionPolicy(band=0.05)
        out[name] = replay(dates, tape, cost, policy,
                           lambda d, x, acct, n=name: (signal(n, symbol, d), None, None),
                           equity0, symbol)
    return out
