"""Hyperliquid market data, served point-in-time.

Public ``POST https://api.hyperliquid.xyz/info`` endpoints, no key:

- ``candleSnapshot``: OHLCV candles. The venue keeps the most recent 5000
  candles per interval, so 15 m bars reach back about 52 days, 1 h bars about
  208 days and daily bars for years.
- ``fundingHistory``: every hourly funding print (rate and premium), 500 rows
  per page.

Point-in-time rule (see ``tradingagents.hyperliquid.clock``): a run dated ``D``
has the cutoff ``(D+1)T00:00Z``. A candle is served only when it has **closed**
by then, and a funding print only when it was printed before it. The slicing
happens before any indicator is computed, so no statistic can draw on a later
bar. Fetched history is stored under ``data_cache_dir/hyperliquid`` and only
closed candles are stored, so a frozen cache replays a backtest byte for byte.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from stockstats import wrap

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.hyperliquid.clock import cutoff_ms
from tradingagents.hyperliquid.venue import FUNDING_INTEREST_1H, HOURS_PER_YEAR

logger = logging.getLogger(__name__)

INFO_URL = "https://api.hyperliquid.xyz/info"
INTERVAL_MS = {"15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}
_CANDLE_PAGE = 5000
_FUNDING_PAGE = 500
_DAY_MS = 86_400_000

# Vendors are called from the analysts' parallel tool nodes. Each store's rows
# are kept in process, keyed by store name, behind one lock.
_ROWS: dict[str, dict[int, dict]] = {}
# Ranges already asked for in this process: the venue keeps only its latest
# 5000 candles, so a window older than that is asked for once, not on every call.
_ATTEMPTED: set[tuple[str, int, int]] = set()
_LOCK = threading.RLock()


def hl_coin(symbol: str) -> str:
    """The venue's coin name for a pipeline symbol: ``BTC-USD`` -> ``BTC``."""
    return (crypto_base(symbol) or str(symbol).split("-")[0]).upper()


def _cache_dir() -> Path:
    try:
        root = get_config().get("data_cache_dir")
    except Exception:  # noqa: BLE001 — a bare call outside a run still works
        root = None
    path = Path(root or os.path.join(os.path.expanduser("~"), ".tradingagents", "cache")) / "hyperliquid"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _offline() -> bool:
    """A frozen replay: serve the stored history only, never the network."""
    try:
        return bool(get_config().get("hyperliquid_offline"))
    except Exception:  # noqa: BLE001
        return False


def _post(payload: dict, retries: int = 6) -> object:
    delay = 1.0
    last: Exception | None = None
    for _ in range(retries):
        try:
            resp = requests.post(INFO_URL, json=payload, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise VendorUnavailableError(f"Hyperliquid /info unreachable: {last}")


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Stores: closed history on disk, merged by key
# ---------------------------------------------------------------------------

def _store_path(name: str) -> Path:
    return _cache_dir() / f"{name}.json"


def _read_store(name: str) -> list[dict]:
    path = _store_path(name)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _write_store(name: str, rows: list[dict]) -> None:
    path = _store_path(name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)


def _candle_frame(rows: list[dict], interval: str) -> pd.DataFrame:
    cols = ["open_ms", "close_ms", "open_time", "close_time", "open", "high", "low", "close",
            "volume", "trades"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "open_ms": df["t"].astype("int64"),
        "open": df["o"].astype(float), "high": df["h"].astype(float),
        "low": df["l"].astype(float), "close": df["c"].astype(float),
        "volume": df["v"].astype(float), "trades": df["n"].astype("int64"),
    })
    out["close_ms"] = out["open_ms"] + INTERVAL_MS[interval]
    out["open_time"] = pd.to_datetime(out["open_ms"], unit="ms", utc=True)
    out["close_time"] = pd.to_datetime(out["close_ms"], unit="ms", utc=True)
    return out.sort_values("open_ms").drop_duplicates("open_ms").reset_index(drop=True)[cols]


def _rows(name: str, key: str) -> dict[int, dict]:
    if name not in _ROWS:
        _ROWS[name] = {int(r[key]): r for r in _read_store(name)}
    return _ROWS[name]


def _gaps(have: list[int], start_ms: int, end_ms: int, step: int) -> list[tuple[int, int]]:
    if not have:
        return [(start_ms, end_ms)] if start_ms < end_ms else []
    gaps = []
    if start_ms < have[0] - step:
        gaps.append((start_ms, have[0]))
    if have[-1] + step < end_ms:
        gaps.append((have[-1] + 1, end_ms))
    return gaps


def load_candles(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Closed candles opening in ``[start_ms, end_ms)``, fetching what the store lacks."""
    coin = hl_coin(symbol)
    name = f"{coin}_{interval}"
    step = INTERVAL_MS[interval]
    now = _now_ms()
    end_ms = min(end_ms, now)
    with _LOCK:
        rows = _rows(name, "t")
        fetched = False
        for g_start, g_end in _gaps(sorted(rows), start_ms, end_ms, step):
            if _offline() or (name, g_start // step, g_end // step) in _ATTEMPTED:
                continue
            _ATTEMPTED.add((name, g_start // step, g_end // step))
            cursor = g_start
            while cursor < g_end:
                page_end = min(g_end, cursor + _CANDLE_PAGE * step)
                page = _post({"type": "candleSnapshot",
                              "req": {"coin": coin, "interval": interval,
                                      "startTime": int(cursor), "endTime": int(page_end)}})
                if not isinstance(page, list):
                    raise VendorUnavailableError(f"unexpected candleSnapshot reply: {str(page)[:200]}")
                for r in page:
                    # Only closed candles are kept: an open one would still change.
                    if int(r["t"]) + step <= now:
                        rows[int(r["t"])] = {k: r[k] for k in ("t", "o", "h", "l", "c", "v", "n")}
                        fetched = True
                cursor = page_end
        if fetched:
            _write_store(name, [rows[t] for t in sorted(rows)])
        frame = _candle_frame([rows[t] for t in sorted(rows)], interval)
    return frame[(frame["open_ms"] >= start_ms) & (frame["open_ms"] < end_ms)].reset_index(drop=True)


def load_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Hourly funding prints with ``start_ms <= time < end_ms``: rate (per hour) and premium."""
    coin = hl_coin(symbol)
    name = f"{coin}_funding"
    step = 3_600_000
    now = _now_ms()
    end_ms = min(end_ms, now)
    with _LOCK:
        rows = _rows(name, "time")
        fetched = False
        for g_start, g_end in _gaps(sorted(rows), start_ms, end_ms, step):
            if _offline() or (name, g_start // step, g_end // step) in _ATTEMPTED:
                continue
            _ATTEMPTED.add((name, g_start // step, g_end // step))
            cursor = g_start
            while cursor < g_end:
                page = _post({"type": "fundingHistory", "coin": coin,
                              "startTime": int(cursor), "endTime": int(g_end)})
                if not isinstance(page, list):
                    raise VendorUnavailableError(f"unexpected fundingHistory reply: {str(page)[:200]}")
                for r in page:
                    rows[int(r["time"])] = {"time": int(r["time"]), "fundingRate": r["fundingRate"],
                                            "premium": r["premium"]}
                    fetched = True
                if len(page) < _FUNDING_PAGE:
                    break
                cursor = int(page[-1]["time"]) + 1
        if fetched:
            _write_store(name, [rows[t] for t in sorted(rows)])
        ordered = [rows[t] for t in sorted(rows)]
    frame = pd.DataFrame({
        "time_ms": pd.Series([int(r["time"]) for r in ordered], dtype="int64"),
        "rate": pd.Series([float(r["fundingRate"]) for r in ordered], dtype=float),
        "premium": pd.Series([float(r["premium"]) for r in ordered], dtype=float),
    })
    frame["time"] = pd.to_datetime(frame["time_ms"], unit="ms", utc=True)
    return frame[(frame["time_ms"] >= start_ms) & (frame["time_ms"] < end_ms)].reset_index(drop=True)


def clear_memory() -> None:
    """Forget the in-process copies (tests, or a cache directory switch)."""
    with _LOCK:
        _ROWS.clear()
        _ATTEMPTED.clear()


# ---------------------------------------------------------------------------
# Point-in-time views
# ---------------------------------------------------------------------------

def _date_ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def admissible_candles(symbol: str, interval: str, as_of_date: str | None,
                       history_days: int) -> pd.DataFrame:
    """Candles closed by the cutoff of ``as_of_date``, the last ``history_days`` days of them."""
    cut = cutoff_ms(as_of_date)
    frame = load_candles(symbol, interval, cut - history_days * _DAY_MS, cut)
    return frame[frame["close_ms"] <= cut].reset_index(drop=True)


def admissible_funding(symbol: str, as_of_date: str | None, history_days: int) -> pd.DataFrame:
    """Funding printed strictly before the cutoff of ``as_of_date``."""
    cut = cutoff_ms(as_of_date)
    frame = load_funding(symbol, cut - history_days * _DAY_MS, cut)
    return frame[frame["time_ms"] < cut].reset_index(drop=True)


def _daily_frame(symbol: str, as_of_date: str | None, history_days: int = 400) -> pd.DataFrame:
    df = admissible_candles(symbol, "1d", as_of_date, history_days)
    if df.empty:
        raise NoMarketDataError(symbol, hl_coin(symbol), f"no closed Hyperliquid daily candles by {as_of_date}")
    out = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                             "volume": "Volume"})
    out["Date"] = out["open_time"].dt.strftime("%Y-%m-%d")
    out["Notional"] = out["Volume"] * out["Close"]
    return out[["Date", "Open", "High", "Low", "Close", "Volume", "Notional", "trades"]]


def _px(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.6g}"


# ---------------------------------------------------------------------------
# Vendor functions behind the analysts' tools
# ---------------------------------------------------------------------------

def get_hl_ohlcv(symbol: str, start_date: str, end_date: str) -> str:
    """Daily OHLCV for the perp from ``start_date`` to ``end_date`` (dates of the UTC days)."""
    days = max(1, (_date_ms(end_date) - _date_ms(start_date)) // _DAY_MS + 2)
    df = _daily_frame(symbol, end_date, history_days=int(days) + 1)
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    if df.empty:
        raise NoMarketDataError(symbol, hl_coin(symbol), f"no closed daily candles in {start_date}..{end_date}")
    header = (
        f"# Hyperliquid {hl_coin(symbol)}-USDC perpetual, daily candles (UTC days), {start_date} to {end_date}\n"
        f"# A row dated D covers D 00:00 to D+1 00:00 UTC and is listed only once closed.\n"
        f"# Volume is in {hl_coin(symbol)}; Notional is Volume x Close in USD (approximate).\n"
        f"# Rows: {len(df)}\n\n"
    )
    body = df.rename(columns={"trades": "Trades"}).to_csv(index=False, float_format="%.6g")
    return header + body


_INDICATOR_HELP = {
    "close_50_sma": "50-day simple moving average of the close (medium-term trend).",
    "close_200_sma": "200-day simple moving average of the close (long-term trend).",
    "close_10_ema": "10-day exponential moving average (short-term momentum).",
    "macd": "MACD line: EMA12 - EMA26 of the close.",
    "macds": "MACD signal: EMA9 of the MACD line.",
    "macdh": "MACD histogram: MACD - signal.",
    "rsi": "RSI(14): >70 overbought, <30 oversold; can stay extreme in trends.",
    "boll": "Bollinger middle band: 20-day SMA.",
    "boll_ub": "Bollinger upper band: 20-day SMA + 2 standard deviations.",
    "boll_lb": "Bollinger lower band: 20-day SMA - 2 standard deviations.",
    "atr": "ATR(14): average true range in USD, a volatility scale for stops and sizing.",
    "vwma": "Volume-weighted moving average (20 days).",
    "mfi": "Money flow index (14): volume-weighted RSI.",
}


def _indicator_series(df: pd.DataFrame, indicator: str) -> pd.Series:
    stock = wrap(df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy())
    values = stock[indicator]
    values.index = df["Date"].to_numpy()
    return values


def get_hl_indicator(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    """One technical indicator on the perp's closed daily candles, as of ``curr_date``."""
    indicator = indicator.strip().lower()
    if indicator not in _INDICATOR_HELP:
        raise ValueError(f"Indicator {indicator} is not supported. Choose from: {list(_INDICATOR_HELP)}")
    df = _daily_frame(symbol, curr_date, history_days=420)
    values = _indicator_series(df, indicator)
    look_back_days = max(1, min(int(look_back_days), 365))
    tail = values.tail(look_back_days)
    lines = [f"## {indicator} for Hyperliquid {hl_coin(symbol)} perp, daily, last {len(tail)} closed days "
             f"up to {curr_date}:", ""]
    lines += [f"{d}: {_px(float(v)) if pd.notna(v) else 'N/A (not enough history)'}" for d, v in tail.items()]
    lines += ["", _INDICATOR_HELP[indicator]]
    return "\n".join(lines)


def build_hl_snapshot(symbol: str, as_of_date: str, look_back_days: int = 30) -> str:
    """Ground-truth snapshot of the perp as of ``as_of_date``: latest closed day, indicators, closes."""
    df = _daily_frame(symbol, as_of_date, history_days=420)
    latest = df.iloc[-1]
    indicator_rows = []
    for name in ("close_10_ema", "close_50_sma", "close_200_sma", "rsi", "boll", "boll_ub",
                 "boll_lb", "macd", "macds", "macdh", "atr"):
        try:
            value = _indicator_series(df, name).iloc[-1]
            indicator_rows.append(f"| {name} | {_px(float(value))} |")
        except Exception as exc:  # noqa: BLE001 — one indicator must not sink the snapshot
            indicator_rows.append(f"| {name} | N/A ({type(exc).__name__}) |")
    hourly = admissible_candles(symbol, "1h", as_of_date, history_days=3)
    last_hour = hourly.iloc[-1] if not hourly.empty else None
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)
    lines = [
        f"## Verified market data snapshot for Hyperliquid {hl_coin(symbol)}-USDC perpetual",
        "",
        f"- Requested analysis date: {as_of_date}",
        f"- Latest closed daily candle used: {latest['Date']} (closed {latest['Date']} 24:00 UTC)",
        "- Candles that had not closed by the end of the analysis date are excluded.",
    ]
    if last_hour is not None:
        lines.append(f"- Last closed hourly candle: {last_hour['open_time']:%Y-%m-%d %H:%M} UTC, "
                     f"close {_px(last_hour['close'])}")
    lines += ["", "### Latest closed daily candle", "", "| Field | Value |", "|---|---:|"]
    for field in ("Open", "High", "Low", "Close"):
        lines.append(f"| {field} | {_px(float(latest[field]))} |")
    lines.append(f"| Volume ({hl_coin(symbol)}) | {latest['Volume']:,.2f} |")
    lines.append(f"| Notional (USD, approx.) | {latest['Notional']:,.0f} |")
    lines += ["", "### Technical indicators (latest closed day)", "", "| Indicator | Value |",
              "|---|---:|", *indicator_rows]
    lines += ["", f"### Recent closes (last {len(recent)} days)", "", "| Date | Close | Day return |",
              "|---|---:|---:|"]
    rets = df["Close"].pct_change()
    for idx, row in recent.iterrows():
        r = rets.loc[idx]
        lines.append(f"| {row['Date']} | {_px(float(row['Close']))} | "
                     f"{'N/A' if pd.isna(r) else f'{r:+.2%}'} |")
    lines += ["", "Use this snapshot as the source of truth for exact price and indicator claims."]
    return "\n".join(lines)


def get_hl_funding_report(symbol: str, as_of_date: str, look_back_days: int = 14) -> str:
    """Funding and premium history, with the carry it implies for a long or a short."""
    look_back_days = max(2, min(int(look_back_days), 60))
    fr = admissible_funding(symbol, as_of_date, history_days=max(look_back_days, 30) + 1)
    if fr.empty:
        raise NoMarketDataError(symbol, hl_coin(symbol), f"no funding prints before the end of {as_of_date}")
    fr = fr.copy()
    fr["day"] = fr["time"].dt.strftime("%Y-%m-%d")
    baseline = np.isclose(fr["rate"], FUNDING_INTEREST_1H, rtol=0, atol=1e-10)
    last = fr.iloc[-1]
    last24 = fr.tail(24)
    window = fr[fr["time_ms"] >= fr["time_ms"].iloc[-1] - look_back_days * _DAY_MS]
    mean30 = fr["rate"].mean()
    std30 = fr["rate"].std(ddof=1) if len(fr) > 2 else float("nan")
    z = (last24["rate"].mean() - mean30) / std30 if std30 and std30 > 0 else float("nan")
    lines = [
        f"## Hyperliquid {hl_coin(symbol)} perp funding, prints before the end of {as_of_date} (UTC)",
        "",
        "Funding is paid every hour; a positive rate means longs pay shorts. Rates are per hour.",
        f"The venue's baseline is {FUNDING_INTEREST_1H:.7f}/h (0.00125%/h, about "
        f"{FUNDING_INTEREST_1H * HOURS_PER_YEAR:.1%}/yr), charged whenever the premium is within 5 bp of 0.01%/8h.",
        "",
        "### Summary",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Last print ({last['time']:%Y-%m-%d %H:%M} UTC) | {last['rate'] * 1e4:+.4f} bp/h |",
        f"| Mean, last 24 prints | {last24['rate'].mean() * 1e4:+.4f} bp/h "
        f"({last24['rate'].mean() * HOURS_PER_YEAR:+.1%}/yr) |",
        f"| Mean, last {look_back_days} days | {window['rate'].mean() * 1e4:+.4f} bp/h "
        f"({window['rate'].mean() * HOURS_PER_YEAR:+.1%}/yr) |",
        f"| Mean, last ~30 days | {mean30 * 1e4:+.4f} bp/h ({mean30 * HOURS_PER_YEAR:+.1%}/yr) |",
        f"| z-score of last-24h mean vs 30 days | {z:+.2f} |",
        f"| Share of prints at the baseline rate (30 d) | {baseline.mean():.0%} |",
        f"| Share of prints below zero (30 d) | {(fr['rate'] < 0).mean():.0%} |",
        f"| Mean premium, last 24 prints | {last24['premium'].mean() * 1e4:+.2f} bp |",
        f"| Cost of holding a 1x long for 24 h at the last-24h rate | {last24['rate'].sum() * 1e4:+.2f} bp of notional |",
        "",
        f"### Daily funding, last {look_back_days} days",
        "",
        "| Day (UTC) | Sum of hourly rates (bp) | Mean premium (bp) | Prints |",
        "|---|---:|---:|---:|",
    ]
    for day, g in window.groupby("day"):
        lines.append(f"| {day} | {g['rate'].sum() * 1e4:+.3f} | {g['premium'].mean() * 1e4:+.2f} | {len(g)} |")
    lines += ["", "A premium persistently below zero means the perp trades under the oracle (sellers "
              "dominate); above zero, buyers pay up. Funding away from the baseline signals crowded positioning."]
    return "\n".join(lines)


def get_hl_market_structure(symbol: str, as_of_date: str, look_back_days: int = 30) -> str:
    """Realized volatility, returns, ranges and activity from closed hourly and daily candles."""
    look_back_days = max(7, min(int(look_back_days), 90))
    hourly = admissible_candles(symbol, "1h", as_of_date, history_days=look_back_days + 1)
    daily = _daily_frame(symbol, as_of_date, history_days=look_back_days + 40)
    if hourly.empty:
        raise NoMarketDataError(symbol, hl_coin(symbol), f"no closed hourly candles by {as_of_date}")
    h_ret = np.log(hourly["close"]).diff().dropna()
    ann = math.sqrt(HOURS_PER_YEAR)

    def rv(hours):
        x = h_ret.tail(hours)
        return float(np.sqrt((x ** 2).mean()) * ann) if len(x) > 2 else float("nan")

    close = daily["Close"].reset_index(drop=True)

    def ret(days):
        return float(close.iloc[-1] / close.iloc[-1 - days] - 1) if len(close) > days else float("nan")

    last = float(hourly["close"].iloc[-1])
    hi30 = float(daily["High"].tail(30).max())
    lo30 = float(daily["Low"].tail(30).min())
    notional = daily["Notional"]
    lines = [
        f"## Hyperliquid {hl_coin(symbol)} perp market structure as of the end of {as_of_date} (UTC)",
        "",
        f"Last closed hourly close: {_px(last)} ({hourly['open_time'].iloc[-1]:%Y-%m-%d %H:%M} UTC bar)",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Realized vol, last 24 h (annualized) | {rv(24):.1%} |",
        f"| Realized vol, last 7 d (annualized) | {rv(24 * 7):.1%} |",
        f"| Realized vol, last {look_back_days} d (annualized) | {rv(24 * look_back_days):.1%} |",
        f"| Implied daily 1-sigma move at 7 d vol | {rv(24 * 7) / math.sqrt(365):.2%} |",
        f"| Return 1 d / 3 d / 7 d / 30 d | {ret(1):+.2%} / {ret(3):+.2%} / {ret(7):+.2%} / {ret(30):+.2%} |",
        f"| 30-day high / low | {_px(hi30)} / {_px(lo30)} |",
        f"| Distance from 30-day high / low | {last / hi30 - 1:+.2%} / {last / lo30 - 1:+.2%} |",
        f"| Daily notional, last day vs 30-day mean | {notional.iloc[-1] / notional.tail(30).mean():.2f}x |",
        f"| Daily notional, 30-day mean (USD, approx.) | {notional.tail(30).mean():,.0f} |",
        "",
        "### Last 7 days, hourly-derived",
        "",
        "| Day (UTC) | Close | Return | Realized vol (ann.) | High-low range | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    hourly = hourly.copy()
    hourly["day"] = hourly["open_time"].dt.strftime("%Y-%m-%d")
    hourly["lr"] = np.log(hourly["close"]).diff()
    for day, g in list(hourly.groupby("day"))[-7:]:
        day_ret = g["close"].iloc[-1] / g["open"].iloc[0] - 1
        vol = float(np.sqrt((g["lr"].dropna() ** 2).mean()) * ann) if len(g) > 2 else float("nan")
        rng = g["high"].max() / g["low"].min() - 1
        lines.append(f"| {day} | {_px(float(g['close'].iloc[-1]))} | {day_ret:+.2%} | {vol:.1%} | "
                     f"{rng:.2%} | {int(g['trades'].sum()):,} |")
    lines += ["", "Hours are UTC; a day with fewer than 24 rows was still trading at the cutoff."]
    return "\n".join(lines)


def prefetch(symbol: str, start: datetime, end: datetime) -> dict:
    """Load every series a backtest over ``[start, end]`` reads, so runs replay from the store."""
    pad = timedelta(days=430)
    s_ms = lambda d: int(d.timestamp() * 1000)  # noqa: E731
    out = {
        "1d": len(load_candles(symbol, "1d", s_ms(start - pad), s_ms(end))),
        "1h": len(load_candles(symbol, "1h", s_ms(start - timedelta(days=95)), s_ms(end))),
        "15m": len(load_candles(symbol, "15m", s_ms(start - timedelta(days=2)), s_ms(end))),
        "funding": len(load_funding(symbol, s_ms(start - timedelta(days=95)), s_ms(end))),
    }
    return out
