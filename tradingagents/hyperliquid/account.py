"""A cross-margin perpetual account: the accounting the simulator stands on.

State: collateral ``C`` (USDC: deposits plus realized P&L, minus fees, minus
funding paid), signed size ``q`` (coin units, long > 0) and average entry
``P̄``. At a mark price ``P`` the equity is ``E = C + q (P − P̄)`` and the
exposure is ``x = qP / E``.

A fill of ``dq`` at reference ``P⁰`` executes at ``Pᶠ = P⁰ (1 + sgn(dq)·s)``
(``s`` the slippage rate) and pays the fee ``φ |dq| Pᶠ``. Marked at ``P⁰``
the equity then falls by exactly the fee plus ``|dq| |Pᶠ − P⁰|``: trading
moves value to the venue and the counterparty, never to the account. A funding
print at rate ``f`` and oracle ``Pᵒ`` moves ``q Pᵒ f`` out of collateral.
These identities are what the tests check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from tradingagents.hyperliquid.venue import CostModel, maintenance_margin_rate


@dataclass
class Fill:
    time: str
    dq: float
    ref_px: float
    fill_px: float
    fee: float
    slippage_cost: float
    realized_pnl: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerpAccount:
    collateral: float
    qty: float = 0.0
    entry_px: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    slippage_paid: float = 0.0
    realized_pnl: float = 0.0
    traded_notional: float = 0.0
    liquidated: bool = False
    fills: list[Fill] = field(default_factory=list)

    # -- valuation ---------------------------------------------------------
    def unrealized(self, px: float) -> float:
        return self.qty * (px - self.entry_px) if self.qty else 0.0

    def equity(self, px: float) -> float:
        return self.collateral + self.unrealized(px)

    def exposure(self, px: float) -> float:
        eq = self.equity(px)
        return self.qty * px / eq if eq > 0 else 0.0

    def maintenance_margin(self, px: float) -> float:
        notional = abs(self.qty) * px
        return maintenance_margin_rate(notional) * notional

    # -- events ------------------------------------------------------------
    def trade(self, dq: float, ref_px: float, cost: CostModel, time: str, reason: str,
              adv_notional: float | None = None, daily_vol: float | None = None) -> Fill | None:
        """Fill ``dq`` units against a reference price; returns the fill (None for no trade)."""
        if dq == 0 or not math.isfinite(dq):
            return None
        side = 1.0 if dq > 0 else -1.0
        slip = cost.slippage_rate(abs(dq) * ref_px, adv_notional, daily_vol)
        fill_px = ref_px * (1 + side * slip)
        fee = cost.fee_rate() * abs(dq) * fill_px
        realized = 0.0
        q0, q1 = self.qty, self.qty + dq
        if q0 == 0 or (q0 > 0) == (dq > 0):
            # Opening or adding: the entry is the size-weighted average.
            self.entry_px = (q0 * self.entry_px + dq * fill_px) / q1
        else:
            closed = min(abs(dq), abs(q0)) * (1 if q0 > 0 else -1)
            realized = closed * (fill_px - self.entry_px)
            if abs(dq) > abs(q0):          # flipped: the remainder opens at the fill
                self.entry_px = fill_px
            elif q1 == 0:
                self.entry_px = 0.0
        self.qty = q1 if abs(q1) > 1e-12 else 0.0
        if self.qty == 0:
            self.entry_px = 0.0
        self.collateral += realized - fee
        self.realized_pnl += realized
        self.fees_paid += fee
        slip_cost = abs(dq) * abs(fill_px - ref_px)
        self.slippage_paid += slip_cost
        self.traded_notional += abs(dq) * fill_px
        fill = Fill(time, dq, ref_px, fill_px, fee, slip_cost, realized, reason)
        self.fills.append(fill)
        return fill

    def pay_funding(self, rate_1h: float, oracle_px: float) -> float:
        """Apply one hourly print; returns the cash paid (negative when received)."""
        paid = self.qty * oracle_px * rate_1h
        self.collateral -= paid
        self.funding_paid += paid
        return paid

    def check_liquidation(self, adverse_px: float, cost: CostModel, time: str) -> Fill | None:
        """Liquidate if equity at the bar's adverse extreme falls below maintenance margin.

        The venue closes the position at market once account value crosses
        maintenance; the fill is taken at the adverse extreme, a conservative
        bound on where the book was walked.
        """
        if self.qty == 0 or self.liquidated:
            return None
        if self.equity(adverse_px) >= self.maintenance_margin(adverse_px):
            return None
        self.liquidated = True
        return self.trade(-self.qty, adverse_px, cost, time, "liquidation")

    def snapshot(self, px: float) -> dict:
        return {
            "equity": self.equity(px), "collateral": self.collateral, "qty": self.qty,
            "entry_px": self.entry_px, "exposure": self.exposure(px), "mark_px": px,
            "fees_paid": self.fees_paid, "funding_paid": self.funding_paid,
            "slippage_paid": self.slippage_paid, "realized_pnl": self.realized_pnl,
            "traded_notional": self.traded_notional, "liquidated": self.liquidated,
        }
