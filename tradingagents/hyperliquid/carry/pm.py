"""Hyperliquid portfolio margin, for a spot-long / perp-short carry book.

Rules (Hyperliquid docs, Trading > Portfolio margin, read 2026-10-31):

- BTC is eligible collateral at LTV 0.5. Its liquidation threshold is
  ``0.5 + 0.5 * LTV = 0.75``.
- The account is liquidatable when ``ratio > 0.95``, where for the borrowable
  token (USDC)::

      ratio = (min_borrow_offset + cross_maintenance + borrowed_USDC)
              / (USDC_balance + min(cap, BTC_balance * price * liq_threshold))

  with ``min_borrow_offset = 20`` USDC.
- USDC borrow rate: ``0.05 + 4.75 * max(0, utilization - 0.8)`` APY, compounded
  continuously.

A carry book holds ``Q`` BTC spot and is short ``Q`` BTC perp. It finances
the spot beyond its equity ``E`` by borrowing USDC. With ``L = Q·S / E`` the
notional per side over equity, and price multiplied by ``k`` since the
position was set:

- Perp loss settles in USDC. **Interpretation "borrow"** (default,
  conservative): a negative USDC balance is a borrow, so losses raise the
  numerator. **Interpretation "balance"**: losses lower the denominator.
- The book is delta-neutral, but its *margin* is not. The perp loss counts in
  full while the spot gain counts at 75%, so a rally drives the ratio toward
  liquidation and a fall makes it safer.

Closed forms, ignoring the 20 USDC offset, with ``m`` the perp maintenance
rate, ``θ`` the liquidation threshold and ``R`` the trigger:

- entry ratio:          ``(m·L + L − 1) / (θ·L)``
- largest L at entry:   ``1 / (1 − R + m + R·(1 − θ))``  (3.33x for BTC)
- rally to liquidation, "borrow":  ``k* = 1 / (L·(1 + m − R·θ))``
- rally to liquidation, "balance": ``k* = (1 − L·(1 − R)) / (L·(m + R·(1 − θ)))``

These are verified against the step-by-step account in the tests. The
interpretation, the caps and whether the feature is live on mainnet must be
checked on testnet before real capital is used.
"""

from __future__ import annotations

from dataclasses import dataclass

BTC_LTV = 0.5
LIQ_THRESHOLD = 0.5 + 0.5 * BTC_LTV          # 0.75
RATIO_TRIGGER = 0.95
MIN_BORROW_OFFSET = 20.0
PERP_MMR = 0.0125
BTC_USER_SUPPLY_CAP = 200.0                  # BTC
USDC_USER_BORROW_CAP = 50e6
MIN_ACCOUNT_VALUE = 10_000.0


def borrow_rate(utilization: float) -> float:
    """USDC borrow APY at a pool utilization (continuous compounding)."""
    return 0.05 + 4.75 * max(0.0, utilization - 0.8)


@dataclass(frozen=True)
class PMParams:
    theta: float = LIQ_THRESHOLD
    trigger: float = RATIO_TRIGGER
    mmr: float = PERP_MMR
    offset: float = MIN_BORROW_OFFSET
    interpretation: str = "borrow"            # or "balance"


DEFAULT_PM = PMParams()


def pm_ratio(usdc: float, btc: float, spot_px: float, perp_size: float, perp_px: float,
             perp_entry: float, p: PMParams = DEFAULT_PM) -> float:
    """The portfolio margin ratio of a book (> trigger means liquidatable).

    ``usdc`` is the settled USDC balance (negative = borrowed), ``btc`` the spot
    balance, ``perp_size`` signed (short < 0).
    """
    upnl = perp_size * (perp_px - perp_entry)
    maint_perp = p.mmr * abs(perp_size) * perp_px
    collateral = min(max(btc, 0.0), BTC_USER_SUPPLY_CAP) * spot_px * p.theta
    if p.interpretation == "borrow":
        cash = usdc + upnl
        num = p.offset + maint_perp + max(0.0, -cash)
        den = max(0.0, cash) + collateral
    elif p.interpretation == "balance":
        num = p.offset + maint_perp + max(0.0, -usdc)
        den = max(0.0, usdc) + upnl + collateral
    else:
        raise ValueError(p.interpretation)
    return num / den if den > 0 else float("inf")


def entry_ratio(L: float, p: PMParams = DEFAULT_PM) -> float:
    return (p.mmr * L + L - 1) / (p.theta * L)


def max_entry_leverage(p: PMParams = DEFAULT_PM) -> float:
    return 1.0 / (1 - p.trigger + p.mmr + p.trigger * (1 - p.theta))


def rally_to_liquidation(L: float, p: PMParams = DEFAULT_PM) -> float:
    """The price multiple ``k*`` at which a book set at leverage ``L`` becomes liquidatable."""
    if p.interpretation == "borrow":
        return 1.0 / (L * (1 + p.mmr - p.trigger * p.theta))
    return (1 - L * (1 - p.trigger)) / (L * (p.mmr + p.trigger * (1 - p.theta)))


def leverage_for_rally(rally: float, p: PMParams = DEFAULT_PM) -> float:
    """The largest L that survives a rally of ``rally`` (e.g. 0.15) without rebalancing."""
    k = 1 + rally
    if p.interpretation == "borrow":
        return 1.0 / (k * (1 + p.mmr - p.trigger * p.theta))
    return 1.0 / (1 - p.trigger + k * (p.mmr + p.trigger * (1 - p.theta)))
