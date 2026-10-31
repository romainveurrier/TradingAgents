"""The holding-period engine: execute a target, carry the position, charge funding.

One function, :func:`run_period`, moves an account through one decision's
life: it trades to the target at the execution instant ``e_D``, then walks
every closed 15-minute bar until the next execution ``e_{D+1}``, applying each
hourly funding print (at the bar open as the oracle proxy), the Trader's stop
if the policy honors it, and the venue's liquidation rule; it marks equity at
every bar close. The walk-forward runner, the counterfactual policy replays
and every baseline go through this same function, so they are priced
identically.

The tape is loaded once for the whole window and only ever *read* forward of
the execution instant: nothing here feeds a decision. What a decision may see
is decided by ``tradingagents.dataflows.vendors.hyperliquid`` and the clock.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from tradingagents.dataflows.vendors.hyperliquid import (
    admissible_candles,
    load_candles,
    load_funding,
)
from tradingagents.hyperliquid.account import PerpAccount
from tradingagents.hyperliquid.clock import DEFAULT_LATENCY, cutoff, execution_time
from tradingagents.hyperliquid.policy import ExecutionPolicy
from tradingagents.hyperliquid.venue import HOURS_PER_YEAR, CostModel

_15M = timedelta(minutes=15)


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


class MarketTape:
    """15-minute bars and funding prints over a simulation window."""

    def __init__(self, symbol: str, start: datetime, end: datetime):
        self.symbol = symbol
        bars = load_candles(symbol, "15m", _ms(start - timedelta(hours=1)), _ms(end + timedelta(hours=1)))
        if bars.empty:
            raise ValueError("no 15-minute bars for the simulation window")
        self.bars = bars.set_index("open_ms", drop=False)
        fr = load_funding(symbol, _ms(start - timedelta(hours=1)), _ms(end + timedelta(hours=1)))
        self.funding = fr
        self._fund_ms = fr["time_ms"].to_numpy()
        self._fund_rate = fr["rate"].to_numpy(float)
        self._open_ms = self.bars["open_ms"].to_numpy()
        self._close_ms = self.bars["close_ms"].to_numpy()
        self._o = self.bars["open"].to_numpy(float)
        self._h = self.bars["high"].to_numpy(float)
        self._l = self.bars["low"].to_numpy(float)
        self._c = self.bars["close"].to_numpy(float)
        # Funding prints falling inside each bar [open, close).
        self._f0 = np.searchsorted(self._fund_ms, self._open_ms, "left")
        self._f1 = np.searchsorted(self._fund_ms, self._close_ms, "left")
        self._open_index = {int(ms): i for i, ms in enumerate(self._open_ms)}
        expected = int((self._open_ms[-1] - self._open_ms[0]) / 900_000) + 1
        self.missing_bars = expected - len(self._open_ms)

    def price_at(self, t: datetime) -> float:
        """The tradeable price at ``t``: the open of the 15 m bar starting then (else the last close)."""
        ms = _ms(t)
        i = self._open_index.get(ms)
        if i is not None:
            return float(self._o[i])
        i = np.searchsorted(self._open_ms, ms, side="right") - 1
        if i < 0:
            raise ValueError(f"no bar before {t}")
        return float(self._c[i])

    def bar_range(self, t0: datetime, t1: datetime) -> range:
        """Indices of the bars opening at or after ``t0`` and closing by ``t1``."""
        i0 = int(np.searchsorted(self._open_ms, _ms(t0), "left"))
        i1 = int(np.searchsorted(self._close_ms, _ms(t1), "right"))
        return range(i0, max(i0, i1))

    def bars_in(self, t0: datetime, t1: datetime) -> pd.DataFrame:
        a, b = _ms(t0), _ms(t1)
        return self.bars[(self.bars["open_ms"] >= a) & (self.bars["close_ms"] <= b)]

    def funding_in(self, a_ms: int, b_ms: int) -> pd.DataFrame:
        i, j = np.searchsorted(self._fund_ms, a_ms, "left"), np.searchsorted(self._fund_ms, b_ms, "left")
        return self.funding.iloc[i:j]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


@dataclass
class PeriodResult:
    trade_date: str
    exec_time: str
    next_exec_time: str
    ref_px: float
    end_px: float
    equity_pre: float
    equity_post: float
    exposure_pre: float
    target: float
    exposure_post_trade: float
    traded: bool
    fee: float
    slippage: float
    funding: float
    funding_prints: int
    stop_hit: bool
    liquidated: bool
    btc_return: float
    funding_rate_sum: float
    marks: list[tuple[str, float]] = field(default_factory=list)
    cutoff_snapshot: dict | None = None

    @property
    def ret(self) -> float:
        return self.equity_post / self.equity_pre - 1

    def as_dict(self, marks: bool = False) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "marks"}
        d["ret"] = self.ret
        if marks:
            d["marks"] = self.marks
        return d


def realized_vol_at(symbol: str, trade_date: str, days: int = 7) -> float | None:
    """Annualized realized volatility from hourly closes known at the decision's cutoff."""
    h = admissible_candles(symbol, "1h", trade_date, history_days=days + 1)
    lr = np.log(h["close"]).diff().dropna().tail(24 * days)
    if len(lr) < 24:
        return None
    return float(np.sqrt((lr ** 2).mean() * HOURS_PER_YEAR))


def run_period(account: PerpAccount, tape: MarketTape, cost: CostModel, policy: ExecutionPolicy,
               trade_date: str, target_x: float, stop_px: float | None = None,
               latency: timedelta = DEFAULT_LATENCY, reason: str = "rebalance",
               keep_marks: bool = True) -> PeriodResult:
    """Trade to ``target_x`` at the decision's execution instant and hold to the next one."""
    t_exec = execution_time(trade_date, latency)
    t_next = t_exec + timedelta(days=1)
    c_next = cutoff(trade_date) + timedelta(days=1)
    ref = tape.price_at(t_exec)
    eq_pre = account.equity(ref)
    x_pre = account.exposure(ref)
    fee0, slip0, fund0 = account.fees_paid, account.slippage_paid, account.funding_paid
    traded = False
    if not account.liquidated and policy.should_trade(target_x, x_pre):
        dq = (target_x * eq_pre - account.qty * ref) / ref
        traded = account.trade(dq, ref, cost, t_exec.isoformat(), reason) is not None
    x_post_trade = account.exposure(ref)
    side = 1 if account.qty > 0 else -1 if account.qty < 0 else 0
    use_stop = policy.use_stops and stop_px and side and (
        (side > 0 and stop_px < ref) or (side < 0 and stop_px > ref))
    stop_hit = False
    prints = 0
    rate_sum = 0.0
    marks: list[tuple[str, float]] = []
    snapshot = None
    c_next_ms = _ms(c_next)
    o, h, lo, c, f0, f1, rates = tape._o, tape._h, tape._l, tape._c, tape._f0, tape._f1, tape._fund_rate
    for i in tape.bar_range(t_exec, t_next):
        # Funding printed during this bar, charged at its open (the oracle proxy).
        for j in range(f0[i], f1[i]):
            account.pay_funding(rates[j], o[i])
            prints += 1
            rate_sum += rates[j]
        if account.qty and use_stop and not stop_hit:
            when = _iso(int(tape._open_ms[i]))
            if side > 0 and lo[i] <= stop_px:
                account.trade(-account.qty, min(o[i], stop_px), cost, when, "stop")
                stop_hit = True
            elif side < 0 and h[i] >= stop_px:
                account.trade(-account.qty, max(o[i], stop_px), cost, when, "stop")
                stop_hit = True
        if account.qty:
            adverse = lo[i] if account.qty > 0 else h[i]
            account.check_liquidation(adverse, cost, _iso(int(tape._open_ms[i])))
        if keep_marks:
            marks.append((_iso(int(tape._close_ms[i])), account.equity(c[i])))
        if tape._close_ms[i] == c_next_ms:
            # What the next decision is told about the book at its cutoff.
            snapshot = account.snapshot(float(c[i]))
            snapshot["time"] = _iso(int(tape._close_ms[i]))
    end_px = tape.price_at(t_next)
    if snapshot is None:
        snapshot = account.snapshot(end_px)
    return PeriodResult(
        trade_date=trade_date, exec_time=t_exec.isoformat(), next_exec_time=t_next.isoformat(),
        ref_px=ref, end_px=end_px, equity_pre=eq_pre, equity_post=account.equity(end_px),
        exposure_pre=x_pre, target=target_x, exposure_post_trade=x_post_trade, traded=traded,
        fee=account.fees_paid - fee0, slippage=account.slippage_paid - slip0,
        funding=account.funding_paid - fund0, funding_prints=prints, stop_hit=stop_hit,
        liquidated=account.liquidated, btc_return=end_px / ref - 1, funding_rate_sum=rate_sum,
        marks=marks, cutoff_snapshot=snapshot,
    )


def replay(dates: Iterable[str], tape: MarketTape, cost: CostModel, policy: ExecutionPolicy,
           decide: Callable[[str, float, PerpAccount], tuple],
           equity0: float, symbol: str, latency: timedelta = DEFAULT_LATENCY) -> list[PeriodResult]:
    """Run a whole decision sequence through the engine.

    ``decide(trade_date, current_exposure, account)`` returns ``(rating_or_target,
    stop_px, leverage)``: a rating string goes through ``policy`` (with the PM's
    leverage multiple, which the policy may ignore); a float is a target
    exposure as is (baselines).
    """
    account = PerpAccount(collateral=equity0)
    out = []
    for d in dates:
        ref = tape.price_at(execution_time(d, latency))
        x_now = account.exposure(ref)
        what, stop, leverage = decide(d, x_now, copy.deepcopy(account))
        if isinstance(what, str):
            sigma = realized_vol_at(symbol, d) if policy.vol_target else None
            target = policy.target(what, x_now, sigma, leverage)
        else:
            target = float(what)
        out.append(run_period(account, tape, cost, policy, d, target, stop, latency))
    return out


def daily_returns(results: list[PeriodResult]) -> np.ndarray:
    return np.array([r.ret for r in results], dtype=float)


def equity_path(results: list[PeriodResult], equity0: float) -> pd.Series:
    pts = [(results[0].exec_time, results[0].equity_pre)] if results else []
    for r in results:
        pts.extend(r.marks)
    s = pd.Series({pd.Timestamp(t): v for t, v in pts}).sort_index()
    return s if not s.empty else pd.Series([equity0])


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC)


def is_finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
