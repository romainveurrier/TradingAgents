"""From a rating to a position: the deterministic layer between the agents and the venue.

The Portfolio Manager speaks the five-tier rating scale. On a perpetual the
rating is executed as a *target net exposure* ``x`` (position notional over
equity). The map is stated to the agents in the venue contract, so what they
say and what is traded agree:

=============  ==========  ===========================================
rating         long/short  long/flat (counterfactual)
=============  ==========  ===========================================
Buy            +1.0        +1.0
Overweight     +0.5        +0.5
Hold           keep x      keep x
Underweight    -0.5        x / 2 if long, else 0
Sell           -1.0        0
REVIEW         keep x      keep x   (no parseable rating: no action)
=============  ==========  ===========================================

Options layered on top, each reported separately in the papers:

- ``vol_target``: scale the non-Hold targets by ``min(cap, σ* / σ̂)``, with
  ``σ̂`` the realized volatility known at the cutoff (Moreira & Muir 2017;
  Harvey et al. 2018).
- ``band``: do not trade when ``|x* − x| < band`` (drift from price moves).
- ``use_stops``: honor the Trader's stop-loss inside the holding period.
- ``use_leverage``: multiply the target by the Portfolio Manager's leverage
  (1 to ``max_gross``), which it is told to raise only when the evidence is
  decisive. Ignoring the multiple replays the same decisions at 1x.
"""

from __future__ import annotations

from dataclasses import dataclass

RATING_SCORE = {"Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2}

LONG_SHORT = {"Buy": 1.0, "Overweight": 0.5, "Underweight": -0.5, "Sell": -1.0}
LONG_FLAT = {"Buy": 1.0, "Overweight": 0.5, "Sell": 0.0}


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: str = "long_short"           # or "long_flat"
    max_gross: float = 1.0
    band: float = 0.05
    vol_target: float | None = None    # annualized, e.g. 0.40
    vol_cap: float = 2.0
    use_stops: bool = False
    use_leverage: bool = False          # honor the PM's leverage multiple (up to max_gross)

    @property
    def name(self) -> str:
        parts = [self.mode]
        if self.vol_target:
            parts.append(f"vt{int(self.vol_target * 100)}")
        if self.use_leverage:
            parts.append(f"lev{self.max_gross:g}")
        if self.use_stops:
            parts.append("stops")
        return "+".join(parts)

    def base_target(self, rating: str, current_x: float) -> float | None:
        """The rating's target before scaling; None means keep the position as it is."""
        if rating not in RATING_SCORE or rating == "Hold":
            return None
        if self.mode == "long_short":
            return LONG_SHORT[rating]
        if self.mode == "long_flat":
            if rating == "Underweight":
                return current_x / 2 if current_x > 0 else 0.0
            return LONG_FLAT[rating]
        raise ValueError(f"unknown policy mode {self.mode!r}")

    def target(self, rating: str, current_x: float, sigma_ann: float | None = None,
               leverage: float | None = None) -> float:
        """The exposure to hold after this decision.

        ``leverage`` is the Portfolio Manager's multiple of the rating's
        exposure; it applies only when ``use_leverage`` is set, clipped to
        ``[1, max_gross]``. A Hold keeps the position whatever the multiple.
        """
        base = self.base_target(rating, current_x)
        if base is None:
            return current_x
        scale = 1.0
        if self.use_leverage and leverage is not None and leverage == leverage:
            scale = max(1.0, min(self.max_gross, float(leverage)))
        if self.vol_target and sigma_ann and sigma_ann > 0:
            scale *= min(self.vol_cap, self.vol_target / sigma_ann)
        cap = self.max_gross * (self.vol_cap if self.vol_target and not self.use_leverage else 1.0)
        return max(-cap, min(cap, base * scale))

    def should_trade(self, target_x: float, current_x: float) -> bool:
        if target_x == current_x:
            return False
        # Closing or flipping is always traded; only drift within the band is ignored.
        if target_x == 0 or (target_x > 0) != (current_x > 0):
            return True
        return abs(target_x - current_x) >= self.band
