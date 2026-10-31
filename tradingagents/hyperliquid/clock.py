"""The timing contract of a decision: when it may look, when it trades, when it is scored.

A decision dated ``D`` is taken after the UTC day ``D`` ends. It may use only
what was known before its **cutoff** ``(D+1)T00:00Z``; it trades at the
**execution** instant, the cutoff plus the pipeline's latency; it holds until
the next decision executes, and its outcome is known only then. Every guard in
the Hyperliquid data path and the simulator reads these functions, so the
contract is stated once.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

DEFAULT_LATENCY = timedelta(minutes=15)


def _day(trade_date: str) -> datetime:
    return datetime.strptime(str(trade_date), "%Y-%m-%d").replace(tzinfo=UTC)


def cutoff(trade_date: str) -> datetime:
    """The instant after which nothing may reach a decision dated ``trade_date``."""
    return _day(trade_date) + timedelta(days=1)


def effective_cutoff(trade_date: str | None, now: datetime | None = None) -> datetime:
    """The cutoff, or now for a run dated today (whose day has not ended)."""
    now = now or datetime.now(UTC)
    if not trade_date:
        return now
    return min(cutoff(trade_date), now)


def cutoff_ms(trade_date: str | None, now: datetime | None = None) -> int:
    return int(effective_cutoff(trade_date, now).timestamp() * 1000)


def execution_time(trade_date: str, latency: timedelta = DEFAULT_LATENCY) -> datetime:
    """When the decision dated ``trade_date`` trades."""
    return cutoff(trade_date) + latency


def outcome_known_at(trade_date: str, latency: timedelta = DEFAULT_LATENCY) -> datetime:
    """When the outcome of the decision dated ``trade_date`` is known: the next execution."""
    return execution_time(trade_date, latency) + timedelta(days=1)


def outcome_visible_to(trade_date: str, later_trade_date: str,
                       latency: timedelta = DEFAULT_LATENCY) -> bool:
    """Whether a later decision may read the outcome of ``trade_date``."""
    return outcome_known_at(trade_date, latency) <= cutoff(later_trade_date)


def resolution_date(trade_date: str, latency: timedelta = DEFAULT_LATENCY) -> str:
    """The memory log's resolution date for ``trade_date``: the first decision date that may read it.

    The memory log shows a lesson to runs dated on or after its resolution
    date, so this is the earliest date whose cutoff follows the outcome.
    """
    known = outcome_known_at(trade_date, latency)
    day = _day(known.strftime("%Y-%m-%d"))
    # The decision dated ``day`` has cutoff day+1 00:00, which follows ``known`` (same day).
    while cutoff(day.strftime("%Y-%m-%d")) < known:
        day += timedelta(days=1)
    return day.strftime("%Y-%m-%d")
