"""Finnhub data helpers for MFA Bear — OPTIONAL accuracy layer.

Verified against the user's free-tier key (2026-06-29):
  - calendar/earnings  → 200  (reliable FORWARD earnings date; fixes the V1 earnings veto)
  - quote              → 200  (real-time c/pc; used as an integrity-gate cross-check only)
  - stock/candle       → 403  PREMIUM — NOT available free, so we NEVER use Finnhub for price
                              history; yfinance remains the sole OHLCV/technical source.
  - rate limit         → 60 calls/min (x-ratelimit-limit header)

Design contract (matches "numbers belong to code" + negative-skew safety):
  * The key is read from the FINNHUB_API_KEY environment variable (a GitHub secret in CI).
  * With NO key, every function returns None and the whole pipeline runs exactly as before
    (keyless yfinance/FRED). Finnhub is strictly additive — never required.
  * Every call has a timeout and a single retry, honors the rate-limit header with a short
    sleep when low, and on ANY failure returns None. It must never raise and never block a run.
  * Finnhub originates only an earnings DATE and a cross-check QUOTE — never a number that
    feeds technicals/scoring. yfinance stays authoritative for the price series.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 8           # seconds per request — fail fast, never hang a CI run
_MIN_REMAINING = 3     # if x-ratelimit-remaining drops to/below this, pause briefly


def api_key():
    """The Finnhub key from the environment, or None. Centralized so callers can cheaply
    check `if finnhub_data.api_key()` to decide whether to attempt Finnhub at all."""
    k = os.environ.get("FINNHUB_API_KEY", "").strip()
    return k or None


def _get(path, params):
    """GET {_BASE}/{path}?{params}&token=KEY → parsed JSON, or None on any problem.
    Never raises. Honors the rate-limit header by sleeping briefly when remaining is low."""
    key = api_key()
    if not key:
        return None
    params = dict(params)
    params["token"] = key
    url = f"{_BASE}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mfa-bear/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                # be a good citizen on the 60/min budget
                try:
                    remaining = int(resp.headers.get("x-ratelimit-remaining", "60"))
                    if remaining <= _MIN_REMAINING:
                        reset = resp.headers.get("x-ratelimit-reset")
                        time.sleep(1.5)   # short, bounded — never block the run for long
                except (TypeError, ValueError):
                    pass
                body = resp.read().decode("utf-8", "replace")
            return json.loads(body)
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.5)
    return None


def next_earnings(symbol, horizon_days=120):
    """Next FORWARD earnings date (ISO 'YYYY-MM-DD') for `symbol`, or None.

    Queries calendar/earnings from today out to `horizon_days` and returns the earliest
    date >= today. This is materially more reliable than yfinance's tk.calendar (which is
    frequently stale/empty), and it is what makes the V1 earnings-over-full-trade-life veto
    trustworthy. Returns None with no key / on any failure → caller falls back to yfinance."""
    if not api_key():
        return None
    today = datetime.now(timezone.utc).date()
    frm = today.isoformat()
    to = date.fromordinal(today.toordinal() + max(1, horizon_days)).isoformat()
    data = _get("calendar/earnings", {"from": frm, "to": to, "symbol": symbol})
    if not data:
        return None
    cal = data.get("earningsCalendar") or []
    future = []
    for e in cal:
        d = e.get("date")
        if not d:
            continue
        try:
            if date.fromisoformat(d) >= today:
                future.append(d)
        except (ValueError, TypeError):
            continue
    return min(future) if future else None


def quote(symbol):
    """Real-time-ish quote dict {current, prev_close} for `symbol`, or None.
    Used ONLY as an integrity cross-check against the yfinance snapshot — never to
    override the computed price series. Returns None with no key / on any failure."""
    if not api_key():
        return None
    data = _get("quote", {"symbol": symbol})
    if not data:
        return None
    c = data.get("c")
    pc = data.get("pc")
    if not isinstance(c, (int, float)) or c <= 0:
        return None
    return {"current": float(c), "prev_close": float(pc) if isinstance(pc, (int, float)) else None}
