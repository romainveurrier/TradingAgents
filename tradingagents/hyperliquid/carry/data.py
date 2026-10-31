"""Bar panels for the carry research: perp, spot and funding on one clock.

Two resolutions:

- ``daily_panel`` covers 2023-05 to now. Funding is summed from hourly prints.
  The spot leg is UBTC/USDC on Hyperliquid (``@142``) once it trades. Before
  that, and on its first illiquid days (junk prints 2025-02-03..13), it is the
  oracle implied by the perp and its premium, ``S ≈ P / (1 + premium)``. The
  proxy is flagged per row.
- ``hourly_panel`` covers the window where the venue serves both hourly
  series, about the last 208 days.

Every row carries the instant its values became known (``known_at`` = bar
close). Nothing in a row is used before that instant: the engine decides at
bar open from earlier rows only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from tradingagents.dataflows.vendors.hyperliquid import load_candles, load_funding

SPOT_COIN = "@142"     # UBTC/USDC (spot pairs are addressed by "@" + universe index)
UBTC_START = "2025-02-03"   # listing day; the first liquid prints start 2025-02-14
MAX_SPOT_DEVIATION = 0.02   # a spot row further than this from the perp, or with no volume, is not a price


def _junk(df: pd.DataFrame) -> pd.Series:
    """Spot rows that are not prices: no volume, or any of O/H/L/C outside the perp's range ±2%.

    The first UBTC days printed 6,969,696 and 100,002,060, and 2025-02-14 opened
    at 240,000 before closing at 97,578. Checking the close alone let that open through.
    """
    lo, hi = df["p_low"] * (1 - MAX_SPOT_DEVIATION), df["p_high"] * (1 + MAX_SPOT_DEVIATION)
    bad = df["s_volume"].fillna(0) <= 0
    for col in ("s_open", "s_high", "s_low", "s_close"):
        bad |= (df[col] < lo) | (df[col] > hi)
    return bad & df["s_close"].notna()


def _ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def funding_hourly(start: str, end: str) -> pd.DataFrame:
    f = load_funding("BTC-USD", _ms(start), _ms(end))
    f = f.assign(hour=f["time"].dt.floor("h")).drop_duplicates("hour", keep="last").set_index("hour")
    return f[["rate", "premium"]]


def _candles(coin: str, interval: str, start: str, end: str) -> pd.DataFrame:
    c = load_candles(coin, interval, _ms(start), _ms(end))
    return c.set_index("open_time")[["open", "high", "low", "close", "volume", "close_time"]]


def daily_panel(start: str = "2023-05-13", end: str = "2026-10-31") -> pd.DataFrame:
    perp = _candles("BTC-USD", "1d", start, end).add_prefix("p_")
    spot = _candles(SPOT_COIN, "1d", UBTC_START, end).add_prefix("s_")
    f = funding_hourly(start, end)
    fd = f.groupby(f.index.floor("D")).agg(funding=("rate", "sum"), prints=("rate", "size"),
                                            premium=("premium", "mean"), premium_last=("premium", "last"))
    df = perp.join(fd, how="inner").join(spot, how="left")
    proxy = df["s_close"].isna() | _junk(df)
    implied = df["p_close"] / (1 + df["premium_last"])
    for col in ("open", "high", "low", "close"):
        df[f"s_{col}"] = df[f"s_{col}"].where(~proxy, df[f"p_{col}"] / (1 + df["premium"]))
    df["s_close"] = df["s_close"].where(~proxy, implied)
    df["spot_proxy"] = proxy
    df["known_at"] = df["p_close_time"]
    return df.drop(columns=["p_close_time", "s_close_time"], errors="ignore")


def hourly_panel(start: str, end: str) -> pd.DataFrame:
    perp = _candles("BTC-USD", "1h", start, end).add_prefix("p_")
    spot = _candles(SPOT_COIN, "1h", start, end).add_prefix("s_")
    f = funding_hourly(start, end)
    # A print stamped hh:00:00.0x pays for the hour that just ended; it lands in the bar opening at hh.
    df = perp.join(spot, how="inner").join(f.rename(columns={"rate": "funding"}), how="left")
    df = df[~_junk(df)]
    df["funding"] = df["funding"].fillna(0.0)
    df["prints"] = (df["funding"] != 0).astype(int)
    df["spot_proxy"] = False
    df["known_at"] = df["p_close_time"]
    return df.drop(columns=["p_close_time", "s_close_time"], errors="ignore").dropna(subset=["s_close"])


def basis_stats(df: pd.DataFrame) -> dict:
    """Spot-perp basis (perp/spot − 1), on rows with a real spot leg."""
    real = df[~df["spot_proxy"]]
    b = real["p_close"] / real["s_close"] - 1
    return {"rows": len(real), "mean_bp": float(b.mean() * 1e4), "std_bp": float(b.std() * 1e4),
            "p01_bp": float(np.quantile(b, 0.01) * 1e4), "p99_bp": float(np.quantile(b, 0.99) * 1e4)}
