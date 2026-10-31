"""Hyperliquid venue rules: fees, funding, margin. One source of truth.

Every number here is taken from the Hyperliquid documentation (checked
2026-10-31: Trading > Fees, Funding, Margining) or from the public ``/info``
endpoint (``meta.universe`` for BTC). The simulator, the prompts the agents
read and the papers all quote this module, so a change of venue rules is a
change in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

# Perpetuals fee schedule by 14-day weighted-volume tier: (taker, maker), as fractions.
# Tier 0 is the base tier (< $5M), tier 6 is > $7B.
PERP_FEE_TIERS: dict[int, tuple[float, float]] = {
    0: (0.00045, 0.00015),
    1: (0.00040, 0.00012),
    2: (0.00035, 0.00008),
    3: (0.00030, 0.00004),
    4: (0.00028, 0.00000),
    5: (0.00026, 0.00000),
    6: (0.00024, 0.00000),
}
PERP_TIER_VOLUME_USD = {0: 0, 1: 5e6, 2: 25e6, 3: 100e6, 4: 500e6, 5: 2e9, 6: 7e9}

# HYPE staking discount on trading fees.
STAKING_DISCOUNTS: dict[str, float] = {
    "none": 0.0, "wood": 0.05, "bronze": 0.10, "silver": 0.15,
    "gold": 0.20, "platinum": 0.30, "diamond": 0.40,
}

# Funding: paid every hour at 1/8 of an 8-hour rate F = P + clamp(I - P, -c, c).
FUNDING_INTEREST_8H = 0.0001          # I = 0.01% per 8 hours
FUNDING_INTEREST_1H = FUNDING_INTEREST_8H / 8  # 0.00125% per hour
FUNDING_CLAMP_8H = 0.0005             # c = 0.05%
FUNDING_CAP_1H = 0.04                 # 4% per hour, all assets
HOURS_PER_YEAR = 24 * 365

# BTC perp (meta.universe, marginTableId 56 "tiered 40x"): 40x below $150M notional, 20x above.
BTC_MAX_LEVERAGE = 40
BTC_MARGIN_TIERS = ((0.0, 40), (150e6, 20))


def max_leverage_for(notional_usd: float, tiers=BTC_MARGIN_TIERS) -> int:
    """The venue's max leverage at a position size: the tier whose lower bound it reaches."""
    lev = tiers[0][1]
    for lower, tier_lev in tiers:
        if abs(notional_usd) >= lower:
            lev = tier_lev
    return lev


def maintenance_margin_rate(notional_usd: float = 0.0, tiers=BTC_MARGIN_TIERS) -> float:
    """Maintenance margin as a fraction of notional: half the initial margin at max leverage."""
    return 0.5 / max_leverage_for(notional_usd, tiers)


def funding_rate_1h(premium_8h: float) -> float:
    """The hourly rate the venue charges for an hour whose average premium is ``premium_8h``.

    ``premium_8h`` is the premium index on the 8-hour basis that the docs'
    formula uses. Whenever |I - P| <= 5 bp the clamp binds to I - P and the
    rate is exactly the interest rate, 0.00125%/h: that is why funding prints
    sit on that value most of the time.
    """
    clamp = max(-FUNDING_CLAMP_8H, min(FUNDING_CLAMP_8H, FUNDING_INTEREST_8H - premium_8h))
    rate = (premium_8h + clamp) / 8
    return max(-FUNDING_CAP_1H, min(FUNDING_CAP_1H, rate))


def funding_payment(size: float, oracle_px: float, rate_1h: float) -> float:
    """Cash a position *pays* for one hourly print (negative means it receives).

    ``size`` is signed (long > 0). A positive rate makes longs pay shorts.
    """
    return size * oracle_px * rate_1h


@dataclass(frozen=True)
class CostModel:
    """What one fill costs: the venue fee plus modeled slippage.

    ``slippage_bps`` is per side and covers half the spread plus the price
    walked through the book. Hyperliquid's BTC spread is about 0.13 bp and the
    touch holds millions of dollars, so a $100k market order walks well under
    1 bp; the 1 bp default is deliberately conservative because the historical
    book cannot be observed. ``impact_coef`` optionally adds a square-root
    impact term Y * sigma_daily * sqrt(Q / ADV) (Almgren et al. 2005; Toth et
    al. 2011) for sizes where the book walk is no longer negligible.
    """

    fee_tier: int = 0
    staking: str = "none"
    referral_discount: float = 0.0
    maker: bool = False
    slippage_bps: float = 1.0
    impact_coef: float = 0.0

    def fee_rate(self) -> float:
        taker, maker = PERP_FEE_TIERS[self.fee_tier]
        base = maker if self.maker else taker
        if base <= 0:
            return base  # rebates and zero maker fees are not discounted further
        return base * (1 - STAKING_DISCOUNTS[self.staking]) * (1 - self.referral_discount)

    def slippage_rate(self, order_notional: float = 0.0, adv_notional: float | None = None,
                      daily_vol: float | None = None) -> float:
        """Adverse price move per unit notional for one fill, as a fraction."""
        rate = self.slippage_bps / 1e4
        if self.maker:
            rate = 0.0  # a resting order fills at its own price (adverse selection not modeled)
        if self.impact_coef and adv_notional and daily_vol and order_notional:
            rate += self.impact_coef * daily_vol * (abs(order_notional) / adv_notional) ** 0.5
        return rate

    def all_in_rate(self, **kwargs) -> float:
        return self.fee_rate() + self.slippage_rate(**kwargs)

    def describe(self) -> str:
        side = "maker" if self.maker else "taker"
        return (f"tier {self.fee_tier} {side} fee {self.fee_rate() * 1e4:.2f} bp + "
                f"slippage {self.slippage_bps:.2f} bp per side")


def venue_contract_text(cost: CostModel | None = None, max_gross_leverage: float = 1.0,
                        latency_minutes: int = 15) -> str:
    """The venue rules as the agents read them, quoted from this module's numbers."""
    cost = cost or CostModel()
    per_side = cost.all_in_rate() * 1e4
    annual_long_carry = FUNDING_INTEREST_1H * HOURS_PER_YEAR * 100
    return (
        "Venue: Hyperliquid BTC-USDC perpetual future (on-chain order book, USDC collateral, "
        "cross margin). The ticker above stands for this perpetual. Shorting is allowed. "
        "Your rating is executed as a target net exposure, as a fraction of account equity, "
        f"about {latency_minutes} minutes after the end of the analysis date (00:{latency_minutes:02d} UTC "
        "the next day), and the position is re-decided every 24 hours: "
        "Buy = +100% (long), Overweight = +50% (long), Hold = keep the current position unchanged, "
        "Underweight = -50% (net short), Sell = -100% (net short). "
        + _leverage_text(max_gross_leverage) +
        f"Costs: about {per_side:.1f} bp of traded notional per trade "
        f"({cost.describe()}); going from fully long to fully short trades 200% of equity, "
        f"about {2 * per_side:.0f} bp of equity. Funding is exchanged every hour between longs and "
        "shorts at the oracle price: when the rate is positive longs pay shorts. At the venue's "
        f"baseline rate (0.00125%/hour, which applies whenever the perp premium is small) a long pays about "
        f"{annual_long_carry:.1f}% a year and a short earns it."
    )


def _leverage_text(max_gross_leverage: float) -> str:
    if max_gross_leverage <= 1:
        return f"Maximum gross exposure is {max_gross_leverage:g}x equity. "
    # Liquidation when equity falls to maintenance: an adverse move of about (1/L - m)
    # wipes the account at exposure L (m = maintenance margin rate, 1.25% for BTC).
    m = maintenance_margin_rate()
    dist = ", ".join(f"{lev:g}x about {max(1 / lev - m, 0):.0%}"
                     for lev in (2, 3, max_gross_leverage) if lev <= max_gross_leverage)
    return (
        f"Leverage is available: the Portfolio Manager may set a leverage multiple from 1 to "
        f"{max_gross_leverage:g} that scales the rating's exposure (Buy at 3x = +300% of equity, capped at "
        f"{max_gross_leverage:g}x gross). When you are highly confident in the direction, use leverage "
        "decisively and size it to your conviction; when the evidence is mixed, stay at 1x. "
        "Leverage multiplies losses as much as gains, and the account is liquidated once equity falls to "
        f"the maintenance margin ({m:.2%} of notional): the adverse move that liquidates is, at {dist}. "
        "Funding and fees scale with the leveraged notional. "
    )
