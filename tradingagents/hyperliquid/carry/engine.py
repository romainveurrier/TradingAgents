"""The portfolio-margin carry book, bar by bar.

The book holds ``Q`` BTC spot (UBTC), is short ``Q`` BTC perp, and borrows USDC
for any spot beyond its equity. On each bar:

1. **At the open, on a decision bar,** the policy sets a target leverage ``L``
   (notional per side over equity) from information known before the open.
   The book trades both legs to it, paying spot and perp fees and slippage.
2. **During the bar** it receives (or pays) funding on the short, at the bar
   open as the oracle proxy, and pays interest on borrowed USDC.
3. **Risk check.** The portfolio-margin ratio is checked at the bar's adverse
   extreme: the perp at its high, with spot co-moving. Above the trigger, the
   book is liquidated: closed at that price with taker fees and a
   ``liq_penalty`` on notional.
4. **At the close,** on an hourly panel, a ratio above ``rebalance_ratio``
   deleverages both legs back to a safe ``L`` without waiting for the next
   decision.

Equity ``E = USDC + Q·S + q_perp·(P − P̄)``. The tests check that each bar's
equity change equals funding − interest − fees − slippage + the price P&L of
the two legs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from tradingagents.hyperliquid.carry.pm import (
    DEFAULT_PM,
    PMParams,
    entry_ratio,
    max_entry_leverage,
    pm_ratio,
)
from tradingagents.hyperliquid.venue import PERP_FEE_TIERS

SPOT_FEE_TIERS = {0: (0.0007, 0.0004), 1: (0.0006, 0.0003), 2: (0.0005, 0.0002), 3: (0.0004, 0.0001)}


@dataclass(frozen=True)
class CarryCosts:
    fee_tier: int = 0
    maker: bool = False
    spot_slippage_bps: float = 2.0      # thinner book than the perp
    perp_slippage_bps: float = 1.0
    borrow_apy: float = 0.05            # base rate at utilization <= 80%
    liq_penalty: float = 0.01           # of notional, on a liquidation (assumption)

    def spot_rate(self) -> float:
        taker, maker = SPOT_FEE_TIERS[self.fee_tier]
        return (maker if self.maker else taker) + (0 if self.maker else self.spot_slippage_bps / 1e4)

    def perp_rate(self) -> float:
        taker, maker = PERP_FEE_TIERS[self.fee_tier]
        return (maker if self.maker else taker) + (0 if self.maker else self.perp_slippage_bps / 1e4)


DEFAULT_COSTS = CarryCosts()


@dataclass
class Book:
    usdc: float
    btc: float = 0.0
    perp: float = 0.0          # signed; short < 0
    entry: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    interest: float = 0.0
    liquidations: int = 0
    rebalances: int = 0
    traded_notional: float = 0.0

    def equity(self, s: float, p: float) -> float:
        return self.usdc + self.btc * s + self.perp * (p - self.entry)

    def leverage(self, s: float, p: float) -> float:
        e = self.equity(s, p)
        return self.btc * s / e if e > 0 else 0.0

    def borrowed(self, p: float) -> float:
        return max(0.0, -(self.usdc + self.perp * (p - self.entry)))

    def ratio(self, s: float, p: float, pm: PMParams) -> float:
        if self.btc <= 0 and self.perp == 0 and self.usdc >= 0:
            return 0.0
        return pm_ratio(self.usdc, self.btc, s, self.perp, p, self.entry, pm)

    def trade_to(self, q_target: float, s: float, p: float, costs: CarryCosts, *, penalty: float = 0.0) -> None:
        """Move both legs to ``q_target`` BTC (spot long, perp short of equal size)."""
        dq = q_target - self.btc
        if abs(dq) * s < 1.0:
            return
        spot_cost = abs(dq) * s * (costs.spot_rate() + penalty)
        perp_cost = abs(dq) * p * (costs.perp_rate() + penalty)
        self.usdc -= dq * s + spot_cost + perp_cost
        self.btc += dq
        # Perp short of size a moves to b = q_target: adding averages the entry,
        # reducing realizes (entry − p) per BTC bought back.
        a, b = -self.perp, q_target
        if b > a:
            self.entry = (a * self.entry + (b - a) * p) / b
        elif b < a:
            self.usdc += (a - b) * (self.entry - p)
            if b == 0:
                self.entry = 0.0
        self.perp = -b
        self.fees += spot_cost + perp_cost
        self.traded_notional += abs(dq) * (s + p)
        self.rebalances += 1


@dataclass
class CarryResult:
    name: str
    equity: pd.Series
    leverage: pd.Series
    ratio_max: pd.Series
    book: Book
    events: list = field(default_factory=list)

    def daily_returns(self) -> pd.Series:
        e = self.equity.resample("1D").last().dropna()
        return e.pct_change().dropna()

    def summary(self) -> dict:
        r = self.daily_returns()
        e = self.equity
        years = (e.index[-1] - e.index[0]).total_seconds() / (365 * 86400)
        total = e.iloc[-1] / e.iloc[0] - 1
        ann = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
        vol = r.std() * np.sqrt(365)
        mdd = float((1 - e / e.cummax()).max())
        b = self.book
        return {"name": self.name, "total": float(total), "ann": float(ann), "vol": float(vol),
                "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else float("nan"),
                "mdd": mdd, "max_ratio": float(self.ratio_max.max()), "liquidations": b.liquidations,
                "rebalances": b.rebalances, "fees": b.fees, "funding": b.funding, "interest": b.interest,
                "mean_leverage": float(self.leverage.mean()), "years": years}


def simulate(panel: pd.DataFrame, policy: Callable[[pd.Timestamp, Book, float], float], name: str,
             costs: CarryCosts = DEFAULT_COSTS, pm: PMParams = DEFAULT_PM, equity0: float = 100_000.0,
             decide_every: str = "D", rebalance_frac: float | None = 0.5, band: float = 0.10) -> CarryResult:
    """Run the carry book over ``panel`` (rows indexed by bar open, columns p_*/s_*/funding).

    ``policy(t, book, current_leverage) -> target leverage`` is called at bar
    opens on the decision schedule with information strictly before ``t``.
    Between decisions, a rally that has used up ``rebalance_frac`` of the
    distance from the target's entry ratio to the liquidation trigger trims
    both legs back to the target at the bar close.
    """
    book = Book(usdc=equity0)
    hours = (panel.index[1] - panel.index[0]).total_seconds() / 3600 if len(panel) > 1 else 24
    dt_years = hours / 8760
    eq, lev, rmax, events = [], [], [], []
    last_decision = None
    target = 0.0
    for t, row in panel.iterrows():
        s_o, p_o = row["s_open"], row["p_open"]
        key = t.floor(decide_every)
        if book.equity(s_o, p_o) <= 0:          # a wiped-out account stops trading
            if book.btc > 0:
                book.trade_to(0.0, s_o, p_o, costs)
            eq.append((row["known_at"], book.equity(row["s_close"], row["p_close"])))
            lev.append((row["known_at"], 0.0))
            rmax.append((row["known_at"], float("inf")))
            continue
        if key != last_decision:
            last_decision = key
            cur = book.leverage(s_o, p_o)
            tgt = min(max(0.0, float(policy(t, book, cur))), max_entry_leverage(pm) * 0.99)
            target = tgt
            if (tgt == 0 and book.btc > 0) or (tgt > 0 and abs(cur - tgt) > band) or (tgt > 0 and book.btc == 0):
                book.trade_to(tgt * book.equity(s_o, p_o) / s_o, s_o, p_o, costs)
        # Funding on the short (short receives when positive) and interest on borrowed USDC.
        pay = book.perp * p_o * row["funding"]
        book.usdc -= pay
        book.funding -= pay
        interest = book.borrowed(p_o) * costs.borrow_apy * dt_years
        book.usdc -= interest
        book.interest += interest
        # Adverse extreme: a rally (perp at its high, spot co-moving).
        p_hi = row["p_high"]
        s_hi = s_o * p_hi / p_o
        r_hi = book.ratio(s_hi, p_hi, pm)
        if r_hi > pm.trigger and book.btc > 0:
            book.liquidations += 1
            events.append({"t": t, "type": "liquidation", "ratio": r_hi, "leverage": book.leverage(s_o, p_o)})
            book.trade_to(0.0, s_hi, p_hi, costs, penalty=costs.liq_penalty)
            r_hi = 0.0
        s_c, p_c = row["s_close"], row["p_close"]
        r_c = book.ratio(s_c, p_c, pm)
        if rebalance_frac is not None and book.btc > 0 and target > 0:
            r0 = entry_ratio(target, pm)
            if r_c > r0 + rebalance_frac * (pm.trigger - r0):
                events.append({"t": t, "type": "deleverage", "ratio": r_c, "to": target})
                book.trade_to(target * book.equity(s_c, p_c) / s_c, s_c, p_c, costs)
        eq.append((row["known_at"], book.equity(s_c, p_c)))
        lev.append((row["known_at"], book.leverage(s_c, p_c)))
        rmax.append((row["known_at"], max(r_hi, r_c)))
    idx = pd.DatetimeIndex([a for a, _ in eq])
    return CarryResult(name, pd.Series([b for _, b in eq], index=idx),
                       pd.Series([b for _, b in lev], index=idx), pd.Series([b for _, b in rmax], index=idx),
                       book, events)


def as_row(result: CarryResult, **extra) -> dict:
    return {**result.summary(), **extra}


def costs_dict(c: CarryCosts) -> dict:
    return asdict(c)
