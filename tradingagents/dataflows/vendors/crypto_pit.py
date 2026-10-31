"""Point-in-time crypto context: news headlines and the Fear & Greed index.

Every item carries a **known-at** instant, the earliest moment it can be
proven public, and is served to a run only when its known-at is at or before
the run's cutoff (``tradingagents.hyperliquid.clock``).

News comes from Google News RSS search, which is keyless and reaches back
months. It stamps most items with their publication *date* only, at midnight
Pacific time (07:00 or 08:00 GMT): such an item may have appeared at any time
in the following 24 hours. Its known-at is therefore its stamp plus 24 hours;
an item with a precise timestamp keeps it. Without this rule a query for
"October 5" returns articles about October 6 prices (verified 2026-10-31).
The cost is a news lag of 17 to 41 hours; prices and funding stay current to
the cutoff. The ranking of search results is computed on the day of the fetch,
not on the decision date: a residual selection effect reported in the papers.

Fear & Greed (alternative.me) publishes one value per UTC day, stamped at
00:00. A value is treated as known 24 hours after its stamp, so a run never
reads a value published at or after its cutoff.

Responses are stored under ``data_cache_dir/crypto_pit`` so a backtest replays
from the store.
"""

from __future__ import annotations

import email.utils
import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime, time as dt_time, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.hyperliquid.clock import effective_cutoff

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
_PACIFIC = ZoneInfo("America/Los_Angeles")
_LOCK = threading.RLock()
_LAST_CALL = [0.0]
_MIN_SPACING_S = 1.5

# Asset queries, and the macro queries a crypto desk reads (Google News syntax).
ASSET_QUERIES = {
    "BTC": ("bitcoin", "bitcoin ETF OR bitcoin price", "crypto market"),
}
GLOBAL_QUERIES = (
    "Federal Reserve OR inflation OR interest rates",
    "stock market OR Nasdaq OR S&P 500",
    "crypto regulation OR SEC crypto OR stablecoin",
)


def _cache_dir() -> Path:
    try:
        root = get_config().get("data_cache_dir")
    except Exception:  # noqa: BLE001
        root = None
    path = Path(root or os.path.join(os.path.expanduser("~"), ".tradingagents", "cache")) / "crypto_pit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _offline() -> bool:
    try:
        return bool(get_config().get("hyperliquid_offline"))
    except Exception:  # noqa: BLE001
        return False


def _get(url: str, retries: int = 5) -> str:
    delay = 2.0
    last = None
    for _ in range(retries):
        with _LOCK:
            wait = _MIN_SPACING_S - (time.time() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.time()
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research backtest)"})
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise VendorUnavailableError(f"{url.split('?')[0]} unreachable: {last}")


# ---------------------------------------------------------------------------
# Known-at rules
# ---------------------------------------------------------------------------

def is_date_only_stamp(stamp: datetime) -> bool:
    """Whether a stamp is a bare date: midnight in Pacific time, as Google News writes dates."""
    local = stamp.astimezone(_PACIFIC)
    return (local.hour, local.minute, local.second) == (0, 0, 0)


def news_known_at(stamp: datetime) -> datetime:
    """The earliest instant a news item stamped ``stamp`` is proven public.

    A date-only stamp is known from the next Pacific midnight: 24 hours later on
    most days, 23 or 25 across a daylight-saving change.
    """
    stamp = stamp.astimezone(UTC)
    if not is_date_only_stamp(stamp):
        return stamp
    local_day = stamp.astimezone(_PACIFIC).date() + timedelta(days=1)
    return datetime.combine(local_day, dt_time(0), tzinfo=_PACIFIC).astimezone(UTC)


def fear_greed_known_at(stamp: datetime) -> datetime:
    return stamp.astimezone(UTC) + timedelta(days=1)


# ---------------------------------------------------------------------------
# Google News, one cached query per Pacific day
# ---------------------------------------------------------------------------

def _parse_rss(xml: str) -> list[dict]:
    items = []
    for raw in re.findall(r"<item>(.*?)</item>", xml, re.S):
        title = re.search(r"<title>(.*?)</title>", raw, re.S)
        pub = re.search(r"<pubDate>(.*?)</pubDate>", raw, re.S)
        src = re.search(r"<source[^>]*>(.*?)</source>", raw, re.S)
        if not title or not pub:
            continue
        try:
            stamp = email.utils.parsedate_to_datetime(pub.group(1).strip()).astimezone(UTC)
        except (TypeError, ValueError):
            continue
        text = html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", title.group(1))).strip()
        source = html.unescape(src.group(1)).strip() if src else ""
        if source and text.endswith(f" - {source}"):
            text = text[: -len(source) - 3].strip()
        items.append({"title": text, "source": source, "stamp": stamp.isoformat()})
    return items


def _google_day(query: str, day: str) -> list[dict]:
    """Items Google News returns for ``query`` over the Pacific day ``day`` (cached)."""
    key = hashlib.sha256(f"gn|{query}|{day}".encode()).hexdigest()[:24]
    path = _cache_dir() / f"gn_{day}_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    if _offline():
        return []
    nxt = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    url = GOOGLE_NEWS_RSS.format(q=quote_plus(f"{query} after:{day} before:{nxt}"))
    items = _parse_rss(_get(url))
    # A day still in progress may gain items; only a finished day is stored.
    if datetime.strptime(nxt, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=2) < datetime.now(UTC):
        path.write_text(json.dumps(items))
    return items


def _collect(queries, start: datetime, cutoff: datetime) -> list[dict]:
    """Every item from ``queries`` stamped from ``start`` on and known by ``cutoff``, newest first."""
    seen, out = set(), []
    day = (start - timedelta(days=1)).date()
    last = cutoff.date()
    while day <= last:
        for q in queries:
            for item in _google_day(q, day.isoformat()):
                stamp = datetime.fromisoformat(item["stamp"])
                known = news_known_at(stamp)
                if stamp < start or known > cutoff:
                    continue
                norm = re.sub(r"\W+", " ", item["title"].lower()).strip()
                if norm in seen:
                    continue
                seen.add(norm)
                out.append({**item, "known_at": known.isoformat(), "query": q})
        day += timedelta(days=1)
    out.sort(key=lambda i: (i["known_at"], i["stamp"]), reverse=True)
    return out


def _render(items: list[dict], heading: str, cutoff: datetime, limit: int) -> str:
    lines = [heading,
             f"Point-in-time: only headlines provably public before {cutoff:%Y-%m-%d %H:%M} UTC. "
             "Google News dates most items by day only, so an item counts from 24 h after its date: "
             "the newest headlines here can be up to about 41 hours old. Headlines only, no article bodies.",
             ""]
    if not items:
        lines.append("No headlines found in this window (the search may simply have returned none).")
        return "\n".join(lines)
    for item in items[:limit]:
        stamp = datetime.fromisoformat(item["stamp"])
        when = (f"{stamp.astimezone(_PACIFIC):%Y-%m-%d} (date only)" if is_date_only_stamp(stamp)
                else f"{stamp:%Y-%m-%d %H:%M} UTC")
        lines.append(f"- [{when}] {item['title']}" + (f" ({item['source']})" if item["source"] else ""))
    if len(items) > limit:
        lines.append(f"... {len(items) - limit} more headlines not shown.")
    return "\n".join(lines)


def get_crypto_news_pit(ticker: str, start_date: str, end_date: str, limit: int | None = None) -> str:
    """Headlines about the coin from ``start_date`` to the end of ``end_date``, point-in-time."""
    base = (crypto_base(ticker) or ticker.split("-")[0]).upper()
    queries = ASSET_QUERIES.get(base, (base.lower(),))
    cutoff = effective_cutoff(end_date)
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    limit = limit or int(_cfg("news_article_limit", 20)) * 2
    items = _collect(queries, start, cutoff)
    return _render(items, f"## {base} news headlines, {start_date} to {end_date}", cutoff, limit)


def get_global_news_pit(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Macro and market headlines over the look-back window ending at ``curr_date``, point-in-time."""
    look_back_days = int(look_back_days or _cfg("global_news_lookback_days", 7))
    limit = int(limit or _cfg("global_news_article_limit", 10)) * 2
    cutoff = effective_cutoff(curr_date)
    start = datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=UTC) - timedelta(days=look_back_days)
    items = _collect(GLOBAL_QUERIES, start, cutoff)
    return _render(items, f"## Global macro and market headlines, {look_back_days} days to {curr_date}",
                   cutoff, limit)


def _cfg(key, default):
    try:
        return get_config().get(key, default) or default
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------

def fear_greed_history() -> list[dict]:
    """Every published value, oldest first, as {stamp, value, label}."""
    path = _cache_dir() / "fear_greed.json"
    data = json.loads(path.read_text()) if path.exists() else []
    fresh = data and datetime.fromisoformat(data[-1]["stamp"]) >= datetime.now(UTC) - timedelta(days=1)
    if not fresh and not _offline():
        try:
            raw = json.loads(_get(FEAR_GREED_URL))["data"]
            data = sorted(({"stamp": datetime.fromtimestamp(int(r["timestamp"]), UTC).isoformat(),
                            "value": int(r["value"]), "label": r["value_classification"]} for r in raw),
                          key=lambda r: r["stamp"])
            path.write_text(json.dumps(data))
        except (VendorUnavailableError, KeyError, ValueError) as exc:
            if not data:
                raise VendorUnavailableError(f"Fear & Greed index unavailable: {exc}") from exc
            logger.warning("Fear & Greed refresh failed, using the stored history: %s", exc)
    return data


def get_fear_greed_report(as_of_date: str, look_back_days: int = 30) -> str:
    """The Crypto Fear & Greed index over the window ending at ``as_of_date``, point-in-time."""
    cutoff = effective_cutoff(as_of_date)
    rows = [r for r in fear_greed_history()
            if fear_greed_known_at(datetime.fromisoformat(r["stamp"])) <= cutoff]
    if not rows:
        return "<Fear & Greed index unavailable before this date>"
    rows = rows[-max(2, int(look_back_days)):]
    values = [r["value"] for r in rows]
    last = rows[-1]
    week = values[-7:]
    lines = [
        f"## Crypto Fear & Greed index (alternative.me), values known by {cutoff:%Y-%m-%d %H:%M} UTC",
        "0 = extreme fear, 100 = extreme greed. One value per UTC day.",
        "",
        f"- Latest: {last['value']} ({last['label']}), stamped {last['stamp'][:10]}",
        f"- 7-day mean {sum(week) / len(week):.1f}; {len(values)}-day mean {sum(values) / len(values):.1f}; "
        f"{len(values)}-day range {min(values)} to {max(values)}",
        f"- Change over 7 days: {values[-1] - values[-min(8, len(values))]:+d} points",
        "",
        "| Day | Value | Label |",
        "|---|---:|---|",
    ]
    lines += [f"| {r['stamp'][:10]} | {r['value']} | {r['label']} |" for r in rows[-14:]]
    return "\n".join(lines)


def prefetch_news(ticker: str, start: datetime, end: datetime, look_back_days: int = 8) -> int:
    """Fetch every Pacific day a backtest over ``[start, end]`` will read, for all queries."""
    base = (crypto_base(ticker) or ticker.split("-")[0]).upper()
    queries = (*ASSET_QUERIES.get(base, (base.lower(),)), *GLOBAL_QUERIES)
    day = (start - timedelta(days=look_back_days + 1)).date()
    n = 0
    while day <= end.date():
        for q in queries:
            n += len(_google_day(q, day.isoformat()))
        day += timedelta(days=1)
    fear_greed_history()
    return n
