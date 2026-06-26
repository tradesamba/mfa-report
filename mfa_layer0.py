#!/usr/bin/env python3
"""
MFA Layer 0 — Deterministic Market-Data Layer
==============================================

The fix for the MFA pipeline's #1 failure mode: LLMs hallucinating live prices.

This module is the deterministic numeric spine. It NEVER asks an LLM for a number.
It fetches OHLCV / splits / earnings from a market-data API (yfinance), computes
every technical locally from the price series, and runs the data-integrity gate
(the re-homed Step 1.5 + Veto #8) IN CODE — as bounds checks, not a model debate.

Output is a signed, timestamped table of clean rows that downstream Grok-sentiment
and Claude-scoring prompts consume. LLMs may read these numbers; they may never
originate them.

Usage:
    python mfa_layer0.py                      # default ticker set
    python mfa_layer0.py AAPL MSFT NVDA       # explicit tickers
    python mfa_layer0.py --json out.json      # also write JSON

Indicators are computed by hand (pandas/numpy) for transparency and so a
dependency bump can't silently change a number. Every formula is auditable below.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── MFA V5 thresholds (single source of truth — matches the guide) ────────────
ADV_FLOOR_SHARES = 2_000_000      # Veto #6 liquidity floor
RVOL_GATE = 1.50                  # mandatory RVOL > 150% breakout gate
ATH_TOLERANCE = 1.02              # price > ATH * 1.02 = physically impossible
MAX_DAY_JUMP = 0.40               # >40% single-day move w/o split = suspect
BETA_FLAG = 1.5                   # Veto #7 high-beta flag
STALE_DAYS = 5                    # quote older than this many calendar days = stale
HOLD_WINDOW_DAYS = 11             # Veto #1: max swing hold (10 trading days) + 1 buffer
FRED_TIMEOUT = 8                  # seconds; FRED is optional — fail fast, never block the run

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META",
                   "GOOGL", "TSLA", "PLTR", "CRWD", "HOOD"]
BENCHMARK = "SPY"


# ── Indicator math (computed locally; never from an LLM) ──────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    return pd.concat([(high - low),
                      (high - prev_close).abs(),
                      (low - prev_close).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def beta_vs_benchmark(stock_close: pd.Series, bench_close: pd.Series) -> float:
    sr = stock_close.pct_change().dropna()
    br = bench_close.pct_change().dropna()
    joined = pd.concat([sr, br], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return float("nan")
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else float("nan")


def _mins_since_open(ts):
    """Minutes elapsed since the 9:30 ET open for a tz-aware timestamp."""
    return (ts.hour - 9) * 60 + (ts.minute - 30)


def compute_rvol_intraday(tk):
    """Time-of-day-normalized RVOL: today's cumulative volume up to the CURRENT
    point in the session vs. the average cumulative volume up to the SAME point
    across the prior ~20 sessions.

    Self-detects how far into the day we are from the latest 5-minute bar — no
    hard-coded clock time, so it is correct whenever it is run.

    Returns (rvol, status):
      status 'intraday'  → today's session is in progress / just closed; normalized.
      status 'full_day'  → only one (today's) intraday session available; partial-day
                           cumulative vs prior full days would mislead, so caller should
                           fall back to the daily-bar RVOL instead.
      Returns (nan, 'no_intraday') if 5-minute data is unavailable.
    """
    try:
        h = yf.Ticker(tk).history(period="30d", interval="5m")
    except Exception:
        return float("nan"), "no_intraday"
    if h is None or h.empty:
        return float("nan"), "no_intraday"
    try:
        h = h.tz_convert("America/New_York")
    except Exception:
        pass

    mins = np.array([_mins_since_open(t) for t in h.index])
    day = h.index.normalize()
    vol = h["Volume"].to_numpy()
    uniq_days = sorted(pd.unique(day))
    if len(uniq_days) < 2:
        return float("nan"), "full_day"          # not enough history to normalize

    today = uniq_days[-1]
    prior_days = uniq_days[:-1][-20:]            # up to 20 prior sessions
    today_mask = (day == today)
    cutoff = int(mins[today_mask].max())          # how far into the day we are, auto-detected

    vol_today = float(vol[today_mask & (mins <= cutoff)].sum())
    cum_prior = []
    for d in prior_days:
        dmask = (day == d)
        cum_prior.append(float(vol[dmask & (mins <= cutoff)].sum()))
    avg_to_cutoff = float(np.mean(cum_prior)) if cum_prior else float("nan")
    if not avg_to_cutoff or math.isnan(avg_to_cutoff):
        return float("nan"), "full_day"
    return round(vol_today / avg_to_cutoff, 2), "intraday"


# ── Phase 0 regime (deterministic metrics; others flagged for feed/estimate) ──
def _interp(x, lo, hi):
    """Linear map x in [lo, hi] -> [-5, +5], clamped. lo may exceed hi (inverted)."""
    if hi == lo:
        return 0.0
    t = (x - lo) / (hi - lo)
    return round(max(-5.0, min(5.0, -5 + 10 * t)), 1)


def _parse_fred_csv(text):
    """Parse a FRED fredgraph CSV into (latest_value, prior_5obs_value, n_obs).

    FRED CSV format: header line 'observation_date,SERIESID' then 'YYYY-MM-DD,value'
    rows; missing values are '.'. Pure-function so it is testable without network.
    Returns (None, None, 0) if nothing usable is found.
    """
    rows = []
    for line in text.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        val = parts[-1].strip()
        if val in (".", "", "value") or parts[0].strip().lower().startswith("observation"):
            continue
        try:
            rows.append(float(val))
        except ValueError:
            continue
    if not rows:
        return None, None, 0
    latest = rows[-1]
    prior = rows[-6] if len(rows) >= 6 else rows[0]
    return latest, prior, len(rows)


def fetch_fred(series_id):
    """Fetch a FRED series CSV (no API key). Returns (latest, prior5, n) or (None,None,0).

    Network-optional: any failure (timeout, blocked, 4xx) returns the empty tuple so
    the regime simply falls back to 'NEEDS FEED/ESTIMATE' rather than crashing.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mfa-layer0)"})
        with urllib.request.urlopen(req, timeout=FRED_TIMEOUT) as resp:
            text = resp.read().decode("utf-8", "replace")
        return _parse_fred_csv(text)
    except Exception:
        return None, None, 0


def compute_regime(use_fred=True):
    """Score the regime metrics that have a deterministic free source (yfinance).

    Returns (metrics: list[dict], note). V6 requires N>=9 of 12 valid metrics for a
    non-Neutral regime; the metrics we cannot source for free are returned with
    score=None and must be supplied by the operator or (bounded, labeled) LLM
    estimate per V6 §0B. We NEVER fabricate them here.
    """
    def hist(sym, period="6mo"):
        try:
            h = yf.Ticker(sym).history(period=period)
            return h["Close"].dropna() if len(h) else None
        except Exception:
            return None

    metrics = []

    # 1. SPY vs 50-day SMA  (-5: >5% below ... +5: >3% above)
    spy = hist("^GSPC", "6mo")
    if spy is not None and len(spy) >= 50:
        sma50 = spy.tail(50).mean()
        pct = (spy.iloc[-1] - sma50) / sma50 * 100
        metrics.append({"n": "SPY vs 50DMA", "v": f"{pct:+.1f}%", "s": _interp(pct, -5, 3)})
    else:
        metrics.append({"n": "SPY vs 50DMA", "v": "no data", "s": None})

    # 2. VIX level (-5: >30 ... +5: <16). Direction noted.
    vix = hist("^VIX", "1mo")
    if vix is not None and len(vix):
        lvl = float(vix.iloc[-1])
        rising = len(vix) >= 5 and lvl > float(vix.iloc[-5])
        metrics.append({"n": "VIX level", "v": f"{lvl:.1f} {'rising' if rising else 'falling'}",
                        "s": _interp(lvl, 30, 16)})
    else:
        metrics.append({"n": "VIX level", "v": "no data", "s": None})

    # 8. Credit spreads — FRED HY OAS (BAMLH0A0HYM2), in %  (-5: >6.5% or +0.75 spike ... +5: <3.25 & tightening)
    if use_fred:
        hy, hy_prior, hy_n = fetch_fred("BAMLH0A0HYM2")
    else:
        hy, hy_prior, hy_n = None, None, 0
    if hy is not None:
        spike = (hy - hy_prior) if hy_prior is not None else 0.0
        # base score on level; if a sharp 5-obs spike, pull toward bear
        s = _interp(hy, 6.5, 3.25)
        if spike >= 0.75:
            s = min(s, -4.0)
        metrics.append({"n": "Credit spreads (HY OAS)", "v": f"{hy:.2f}% ({spike:+.2f})", "s": s})
    else:
        metrics.append({"n": "Credit spreads (HY OAS)", "v": "NEEDS FEED/ESTIMATE", "s": None})

    # 9. Yield curve — prefer FRED 10Y-2Y (T10Y2Y); fall back to yfinance 10Y-13wk proxy
    tnx, irx = hist("^TNX", "1mo"), hist("^IRX", "1mo")
    fred_curve = fetch_fred("T10Y2Y") if use_fred else (None, None, 0)
    if fred_curve[0] is not None:
        spread = fred_curve[0]
        metrics.append({"n": "Yield curve (10Y-2Y, FRED)", "v": f"{spread:+.2f}pp",
                        "s": _interp(spread, -0.5, 0.75)})
    elif tnx is not None and irx is not None and len(tnx) and len(irx):
        spread = float(tnx.iloc[-1]) / 10 - float(irx.iloc[-1]) / 10
        metrics.append({"n": "Yield curve (10Y-13wk proxy)", "v": f"{spread:+.2f}pp",
                        "s": _interp(spread, -0.5, 0.75)})
    else:
        metrics.append({"n": "Yield curve", "v": "NEEDS FEED/ESTIMATE", "s": None})

    # 10. US Dollar (DXY) 20d momentum  (-5: +4% shock ... +5: falling)
    dxy = hist("DX-Y.NYB", "3mo")
    if dxy is not None and len(dxy) >= 20:
        mom = (dxy.iloc[-1] - dxy.iloc[-20]) / dxy.iloc[-20] * 100
        metrics.append({"n": "DXY 20d momentum", "v": f"{mom:+.1f}%", "s": _interp(mom, 4, -4)})
    else:
        metrics.append({"n": "DXY 20d momentum", "v": "no data", "s": None})

    # 12. VIX term structure VIX/VIX3M  (-5: >=1.05 backwardation ... +5: <=0.90 contango)
    vix3m = hist("^VIX3M", "1mo")
    if vix is not None and vix3m is not None and len(vix) and len(vix3m):
        ratio = float(vix.iloc[-1]) / float(vix3m.iloc[-1])
        metrics.append({"n": "VIX term (VIX/VIX3M)", "v": f"{ratio:.3f}", "s": _interp(ratio, 1.05, 0.90)})
    else:
        metrics.append({"n": "VIX term (VIX/VIX3M)", "v": "no data", "s": None})

    # Metrics with NO free feed — must be supplied (operator or bounded LLM estimate per V6 §0B)
    for name in ["Market breadth (%>200DMA)", "NYSE advance/decline", "Sector rotation",
                 "Fed / macro backdrop", "Economic Surprise Index",
                 "Equity put/call (contrarian)"]:
        metrics.append({"n": name, "v": "NEEDS FEED/ESTIMATE", "s": None})

    return metrics


def regime_summary(metrics):
    """Return the one-line regime verdict string (used by both console + HTML)."""
    valid = [m for m in metrics if m["s"] is not None]
    n = len(valid)
    if n >= 9:
        score = round(2 * sum(m["s"] for m in valid) / n)
        band = ("Bull" if score >= 6 else "Mild Bull" if score >= 2 else
                "Neutral" if score > -2 else "Mild Bear" if score > -6 else "Bear")
        return f"RegimeScore = {score} → band: {band} (N={n} ≥9 ✔)"
    partial = round(2 * sum(m["s"] for m in valid) / n) if n else 0
    return (f"Only N={n} of 12 metrics sourced (need ≥9). Partial avg {partial} → "
            f"FORCED NEUTRAL (±15) per V6 §0A until the remaining {9-n}+ metrics are supplied "
            f"(breadth / A-D / sector / Fed / econ-surprise / credit / put-call).")


def print_regime(metrics):
    print("\n" + "=" * 78)
    print("PHASE 0 — REGIME (deterministic metrics; others need feed/estimate per V6 §0B)")
    print("=" * 78)
    for m in metrics:
        s = f"{m['s']:+.1f}" if m["s"] is not None else "  —"
        print(f"  {m['n']:<30}{m['v']:>22}   score {s}")
    print("\n  " + regime_summary(metrics))


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class TickerRow:
    ticker: str
    ok: bool = True
    conflicts: list = field(default_factory=list)   # integrity failures (Veto #8)
    flags: list = field(default_factory=list)        # non-fatal notes
    # manifest
    last_split: str = ""
    ath: float = float("nan")
    low_52w: float = float("nan")
    high_52w: float = float("nan")
    next_earnings: str = ""
    earnings_in_window: bool = False     # Veto #1: earnings inside the hold window
    adv: float = float("nan")
    # live numbers
    price: float = float("nan")
    as_of: str = ""
    # technicals
    ema_ribbon: str = ""
    ema_spread_pct: float = float("nan")
    macd_hist: float = float("nan")
    rsi14: float = float("nan")
    atr_pct: float = float("nan")
    adx14: float = float("nan")
    rvol: float = float("nan")
    rvol_basis: str = "daily"        # 'intraday' (time-of-day normalized) or 'daily' (full-day)
    beta: float = float("nan")
    # screens
    passes_rvol_gate: bool = False
    passes_adv_floor: bool = False
    # alt-data (Section B) — only real fields; unavailable ones stay N/A, never estimated
    short_pct_float: float = float("nan")
    short_days_to_cover: float = float("nan")
    short_trend: str = ""            # rising / falling vs prior month
    insider_net_shares: float = float("nan")   # buys − sells, last ~6mo
    insider_note: str = ""
    putcall_oi: float = float("nan")  # put OI / call OI, nearest expiry
    dark_pool: str = "N/A (no free feed)"
    gex: str = "N/A (no free feed)"


def _scalar(x):
    """yfinance can hand back 1-element Series; coerce to float safely."""
    if isinstance(x, pd.Series):
        x = x.iloc[-1] if len(x) else float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fetch_alt_data(tk, row):
    """Populate Section B fields that have a REAL free source (yfinance).

    Honest scope: short interest, insider net activity, and put/call OI ratio are
    genuinely available. Dark-pool prints and true dealer-gamma GEX have no free
    feed, so they stay N/A — we never estimate them (a fabricated GEX is worse
    than no GEX). Any failure leaves the field as NaN/N-A; it is never guessed.
    """
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    spf = info.get("shortPercentOfFloat")
    if isinstance(spf, (int, float)):
        row.short_pct_float = round(spf * 100, 2)
    sr = info.get("shortRatio")
    if isinstance(sr, (int, float)):
        row.short_days_to_cover = round(sr, 2)
    cur, prior = info.get("sharesShort"), info.get("sharesShortPriorMonth")
    if isinstance(cur, (int, float)) and isinstance(prior, (int, float)) and prior:
        row.short_trend = "rising" if cur > prior * 1.02 else "falling" if cur < prior * 0.98 else "flat"

    # insider net (buys − sells) from the last ~6 months of Form 4 rows
    try:
        it = tk.insider_transactions
        if it is not None and len(it) and "Text" in it.columns:
            buys = sells = 0.0
            for _, r in it.head(40).iterrows():
                txt = str(r.get("Text", "")).lower()
                sh = r.get("Shares", 0) or 0
                try:
                    sh = float(sh)
                except (TypeError, ValueError):
                    continue
                if "buy" in txt or "purchase" in txt:
                    buys += sh
                elif "sale" in txt or "sell" in txt:
                    sells += sh
            row.insider_net_shares = round(buys - sells, 0)
            row.insider_note = ("net buying" if buys > sells else
                                "net selling" if sells > buys else "flat")
    except Exception:
        pass

    # put/call open-interest ratio from the nearest expiry (contrarian/positioning read)
    try:
        exps = tk.options
        if exps:
            ch = tk.option_chain(exps[0])
            call_oi = float(ch.calls["openInterest"].fillna(0).sum())
            put_oi = float(ch.puts["openInterest"].fillna(0).sum())
            if call_oi > 0:
                row.putcall_oi = round(put_oi / call_oi, 2)
    except Exception:
        pass


def analyze(ticker: str, bench_hist: pd.DataFrame, alt: bool = False,
            intraday: bool = False) -> TickerRow:
    row = TickerRow(ticker=ticker)
    tk = yf.Ticker(ticker)

    hist = tk.history(period="1y", auto_adjust=False)
    if hist.empty or len(hist) < 60:
        row.ok = False
        row.conflicts.append("insufficient history from feed")
        return row

    close = hist["Close"]
    price = _scalar(close.iloc[-1])
    as_of = hist.index[-1].to_pydatetime()
    row.price = round(price, 2)
    row.as_of = as_of.strftime("%Y-%m-%d")

    # ── manifest (from the feed, NOT an LLM) ─────────────────────────────────
    # True all-time high from full history. period="max" is one cheap call, and
    # yfinance always split-adjusts highs/lows (auto_adjust only governs dividends),
    # so old pre-split peaks are already correct and won't cause false conflicts.
    # Fall back to the 1y high only if the full-history pull fails.
    try:
        full = tk.history(period="max", auto_adjust=False)
        row.ath = round(_scalar(full["High"].max()), 2) if not full.empty \
            else round(_scalar(hist["High"].max()), 2)
    except Exception:
        row.ath = round(_scalar(hist["High"].max()), 2)
    row.low_52w = round(_scalar(close.tail(252).min()), 2)
    row.high_52w = round(_scalar(close.tail(252).max()), 2)
    row.adv = round(_scalar(hist["Volume"].tail(20).mean()), 0)

    splits = tk.splits
    if splits is not None and len(splits):
        s_date = splits.index[-1]
        row.last_split = f"{_scalar(splits.iloc[-1]):g}:1 ({s_date.date()})"
    else:
        row.last_split = "none"

    earn_date = None
    try:
        cal = tk.calendar
        ed = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, (list, tuple)) and ed:
                ed = ed[0]
        if ed is not None:
            earn_date = getattr(ed, "date", lambda: ed)()
            row.next_earnings = str(earn_date)
        else:
            row.next_earnings = "unknown"
    except Exception:
        row.next_earnings = "unknown"

    # Veto #1 — earnings inside the hold window [today, today + HOLD_WINDOW_DAYS]
    if isinstance(earn_date, date):
        today = datetime.now(timezone.utc).date()
        if today <= earn_date <= today + timedelta(days=HOLD_WINDOW_DAYS):
            row.earnings_in_window = True

    # ── technicals (computed locally) ────────────────────────────────────────
    e = {span: _scalar(ema(close, span).iloc[-1]) for span in (8, 13, 21, 34, 55, 89)}
    ribbon_up = e[8] > e[13] > e[21] > e[34] > e[55] > e[89]
    ribbon_dn = e[8] < e[13] < e[21] < e[34] < e[55] < e[89]
    row.ema_spread_pct = round((e[8] - e[89]) / e[89] * 100, 2) if e[89] else float("nan")
    row.ema_ribbon = "bullish" if ribbon_up else "bearish" if ribbon_dn else "mixed"

    _, _, hist_macd = macd(close)
    row.macd_hist = round(_scalar(hist_macd.iloc[-1]), 3)
    row.rsi14 = round(_scalar(rsi(close).iloc[-1]), 1)
    row.atr_pct = round(_scalar(atr(hist).iloc[-1]) / price * 100, 2) if price else float("nan")
    row.adx14 = round(_scalar(adx(hist).iloc[-1]), 1)

    # RVOL — full-day baseline (volume so far ÷ 20d avg full-day volume).
    # NOTE: intraday this UNDER-reads (partial day vs full-day avg). When --intraday
    # is set we replace it with a time-of-day-normalized value that is comparable to
    # the >150% gate at any point in the session.
    vol_today = _scalar(hist["Volume"].iloc[-1])
    vol_avg20 = _scalar(hist["Volume"].tail(20).mean())
    row.rvol = round(vol_today / vol_avg20, 2) if vol_avg20 else float("nan")
    row.rvol_basis = "daily"
    if intraday:
        iv, status = compute_rvol_intraday(ticker)
        if status == "intraday" and not math.isnan(iv):
            row.rvol = iv
            row.rvol_basis = "intraday"
        # else: keep daily fallback (market closed / no intraday history)
    row.beta = round(beta_vs_benchmark(close, bench_hist["Close"]), 2)

    # ── DATA-INTEGRITY GATE (re-homed Step 1.5 / Veto #8, all in code) ───────
    age_days = (datetime.now(timezone.utc) - as_of.replace(tzinfo=timezone.utc)).days
    if as_of.year != datetime.now(timezone.utc).year:
        row.conflicts.append(f"stale: quote year {as_of.year} != current")
    elif age_days > STALE_DAYS:
        row.flags.append(f"quote {age_days}d old (feed may be EOD/closed market)")

    if not (row.low_52w * 0.98 <= price <= row.high_52w * 1.02):
        row.conflicts.append(
            f"price {price:.2f} outside 52w [{row.low_52w:.2f}, {row.high_52w:.2f}]")

    if price > row.ath * ATH_TOLERANCE:
        row.conflicts.append(f"price {price:.2f} > ATH*{ATH_TOLERANCE} ({row.ath:.2f}) — impossible")

    day_moves = close.pct_change().abs().tail(252)
    split_dates = set(splits.index.date) if (splits is not None and len(splits)) else set()
    for dt, mv in day_moves.items():
        if mv > MAX_DAY_JUMP and dt.date() not in split_dates:
            row.flags.append(f">{MAX_DAY_JUMP:.0%} 1-day move {dt.date()} ({mv:.0%}) w/o split")
            break

    # ── screens (MFA V5 hard gates, as code filters) ─────────────────────────
    row.passes_rvol_gate = row.rvol >= RVOL_GATE if not math.isnan(row.rvol) else False
    row.passes_adv_floor = row.adv >= ADV_FLOOR_SHARES if not math.isnan(row.adv) else False
    if not row.passes_adv_floor:
        row.flags.append(f"ADV {row.adv:,.0f} < {ADV_FLOOR_SHARES:,} floor (Veto #6)")
    if row.beta > BETA_FLAG:
        row.flags.append(f"beta {row.beta} > {BETA_FLAG} (Veto #7 flag)")
    if row.earnings_in_window:
        row.flags.append(f"earnings {row.next_earnings} in {HOLD_WINDOW_DAYS}d hold window (Veto #1)")

    row.ok = len(row.conflicts) == 0
    if alt and row.ok:
        fetch_alt_data(tk, row)
    return row


def emit_mfa_sections(rows, run_ts):
    """Emit Gemini-replacement Section M / C / E in the exact MFA cheatsheet format.

    These are paste-ready for the Claude scoring step. Because the numbers come
    from code, the Grok Step-1.5 audit is no longer needed — integrity already
    passed in Layer 0. Only CLEARED rows are emitted as survivors.
    """
    cleared = [r for r in rows if r.ok]
    today = datetime.now(timezone.utc).date()
    is_russell = (today.month == 6 and today.weekday() == 4 and today.day >= 25)  # last-Fri-June heuristic

    out = []
    out.append(f"# MFA LAYER 0 OUTPUT (deterministic feed · {run_ts}) — replaces Gemini Sections M/C/E\n")

    out.append("SECTION M — PRE-HOOK MANIFEST")
    for r in cleared:
        out.append(
            f"{r.ticker:<6}| last split: {r.last_split} | ATH ≈ ${r.ath:.2f} "
            f"| 52w range: ${r.low_52w:.2f}–${r.high_52w:.2f} "
            f"| next earnings: {r.next_earnings} | ADV ≈ {r.adv/1e6:.1f}M")

    out.append("\nSECTION B — ALT DATA")
    has_alt = any(not math.isnan(r.short_pct_float) or not math.isnan(r.putcall_oi)
                  or not math.isnan(r.insider_net_shares) for r in cleared)
    if not has_alt:
        out.append("(run with --alt to populate; all fields N/A — score Alt category as N/A per V6 §0B)")
    else:
        out.append("TICKER | Dark Pool | Options Flow (P/C OI) | GEX | SI% float + DTC | Insider (6mo)")
        for r in cleared:
            sif = (f"{r.short_pct_float:.2f}% / {r.short_days_to_cover:.1f}d ({r.short_trend})"
                   if not math.isnan(r.short_pct_float) else "N/A")
            pc = f"P/C OI {r.putcall_oi:.2f}" if not math.isnan(r.putcall_oi) else "N/A"
            ins = (f"{r.insider_net_shares:+,.0f} sh ({r.insider_note})"
                   if not math.isnan(r.insider_net_shares) else "N/A")
            out.append(f"{r.ticker} | {r.dark_pool} | {pc} | {r.gex} | {sif} | {ins}")
        out.append("NOTE: Dark Pool + GEX = N/A (no free feed) — NOT estimated. Score Alt category on "
                   "available fields only, or mark Alt N/A and renormalize Tech/Sent per V6 §0B.")

    out.append("\nSECTION C — TECHNICALS")
    out.append("TICKER | Price + Timestamp | split/ATH/52w reconciliation | EMA Ribbon | "
               "MACD | RSI | ATR% | RVOL% (calendar?) | ADX | Earnings | Beta")
    for r in cleared:
        recon = (f"price ${r.price:.2f} (as of {r.as_of}); split {r.last_split}; "
                 f"ATH ${r.ath:.2f}; within 52w ✔")
        macd_txt = "bull (hist+)" if r.macd_hist > 0 else "bear (hist−)"
        ribbon = {"bullish": "+ribbon up", "bearish": "−ribbon down", "mixed": "mixed"}[r.ema_ribbon]
        rvol_txt = f"{r.rvol*100:.0f}%"
        rvol_txt += " (calendar-driven)" if (is_russell and r.rvol >= RVOL_GATE) else ""
        rvol_txt += " [intraday]" if r.rvol_basis == "intraday" else " [EOD]"
        beta_txt = f"{r.beta:.2f}" + (" ⚠(V7)" if r.beta > BETA_FLAG else "")
        earn_txt = r.next_earnings + (" ⚠(V1 in-window)" if r.earnings_in_window else "")
        out.append(
            f"{r.ticker} | ${r.price:.2f} ({r.as_of}) | {recon} | {ribbon} ({r.ema_spread_pct:+.1f}%) "
            f"| {macd_txt} | {r.rsi14:.0f} | {r.atr_pct:.1f}% | {rvol_txt} | {r.adx14:.0f} "
            f"| {earn_txt} | {beta_txt}")

    out.append("\nSECTION E — SURVIVOR LIST")
    out.append("SURVIVORS: " + ", ".join(r.ticker for r in cleared))
    out.append("\nNOTE: numbers are code-sourced (yfinance) and passed the in-code integrity gate. "
               "Skip Grok Step 1.5 (model-vs-model audit) — it is superseded by Layer 0.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="MFA Layer 0 — deterministic market-data layer")
    ap.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS,
                    help="tickers to analyze (default: MFA test set)")
    ap.add_argument("--json", metavar="PATH", help="also write full result as JSON")
    ap.add_argument("--sections", action="store_true",
                    help="emit paste-ready MFA Section M/C/E (Gemini-step replacement)")
    ap.add_argument("--alt", action="store_true",
                    help="fetch real Section B alt-data (short interest, insider, put/call OI)")
    ap.add_argument("--regime", action="store_true",
                    help="compute deterministic Phase 0 regime metrics (VIX/DXY/curve/term/SPY + FRED)")
    ap.add_argument("--no-fred", action="store_true",
                    help="skip FRED credit-spread/yield-curve fetch (offline / blocked network)")
    ap.add_argument("--intraday", action="store_true",
                    help="use time-of-day-normalized RVOL (run this when the market is open)")
    ap.add_argument("--html", metavar="PATH",
                    help="write a self-contained mobile HTML report with copy-to-Grok/Claude wizard")
    args = ap.parse_args()
    tickers = args.tickers or DEFAULT_TICKERS

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"MFA LAYER 0  ·  {run_ts}  ·  source: yfinance (deterministic feed)\n")

    regime_metrics = None
    regime_sum = "Regime not computed (run with --regime)."
    if args.regime or args.html:
        regime_metrics = compute_regime(use_fred=not args.no_fred)
        regime_sum = regime_summary(regime_metrics)
        if args.regime:
            print_regime(regime_metrics)

    bench = yf.Ticker(BENCHMARK).history(period="1y", auto_adjust=False)
    rows = []
    for t in tickers:
        try:
            rows.append(analyze(t, bench, alt=args.alt, intraday=args.intraday))
        except Exception as ex:
            r = TickerRow(ticker=t, ok=False)
            r.conflicts.append(f"fetch/compute error: {repr(ex)[:120]}")
            rows.append(r)

    # ── data-integrity report (Step 1.5, but deterministic) ──────────────────
    print("=" * 78)
    print("DATA-INTEGRITY GATE  (CONFLICT = dropped before scoring · Veto #8)")
    print("=" * 78)
    print(f"{'TICKER':<7}{'PRICE':>10}{'ATH':>10}{'AS-OF':>13}  VERDICT")
    cleared, dropped = [], []
    for r in rows:
        verdict = "CLEARED" if r.ok else "CONFLICT — " + "; ".join(r.conflicts)
        (cleared if r.ok else dropped).append(r.ticker)
        price = f"{r.price:.2f}" if not math.isnan(r.price) else "—"
        ath = f"{r.ath:.2f}" if not math.isnan(r.ath) else "—"
        print(f"{r.ticker:<7}{price:>10}{ath:>10}{r.as_of:>13}  {verdict}")
        for f in r.flags:
            print(f"{'':<30}  ⚑ {f}")

    # ── technicals table (only on cleared rows; computed locally) ────────────
    print("\n" + "=" * 78)
    print("TECHNICALS  (CLEARED tickers only · computed locally from price series)")
    print("=" * 78)
    hdr = f"{'TICK':<6}{'PRICE':>9}{'RSI':>6}{'MACDh':>8}{'EMA':>9}{'ATR%':>7}{'ADX':>6}{'RVOL':>7}{'BETA':>6}  GATES"
    print(hdr)
    for r in rows:
        if not r.ok:
            continue
        gates = []
        gates.append("RVOL✔" if r.passes_rvol_gate else "RVOL✘")
        gates.append("ADV✔" if r.passes_adv_floor else "ADV✘")
        gates.append("[ID]" if r.rvol_basis == "intraday" else "[EOD]")
        print(f"{r.ticker:<6}{r.price:>9.2f}{r.rsi14:>6.1f}{r.macd_hist:>8.3f}"
              f"{r.ema_ribbon:>9}{r.atr_pct:>7.2f}{r.adx14:>6.1f}{r.rvol:>7.2f}"
              f"{r.beta:>6.2f}  {' '.join(gates)}")

    # ── alt-data table (Section B) — only when --alt; real fields only ───────
    if args.alt:
        print("\n" + "=" * 78)
        print("ALT DATA / SECTION B  (CLEARED only · real fields; unavailable = N/A, never estimated)")
        print("=" * 78)
        print(f"{'TICK':<6}{'SI%float':>9}{'DTC':>6}{'SI trend':>10}{'P/C OI':>8}  "
              f"{'INSIDER (6mo)':<22}{'DARKPOOL':<18}{'GEX'}")
        for r in rows:
            if not r.ok:
                continue
            sif = f"{r.short_pct_float:.2f}" if not math.isnan(r.short_pct_float) else "N/A"
            dtc = f"{r.short_days_to_cover:.1f}" if not math.isnan(r.short_days_to_cover) else "N/A"
            pc = f"{r.putcall_oi:.2f}" if not math.isnan(r.putcall_oi) else "N/A"
            ins = (f"{r.insider_net_shares:+,.0f} ({r.insider_note})"
                   if not math.isnan(r.insider_net_shares) else "N/A")
            print(f"{r.ticker:<6}{sif:>9}{dtc:>6}{r.short_trend or 'N/A':>10}{pc:>8}  "
                  f"{ins:<26}{r.dark_pool:<20}{r.gex}")

    print("\n" + "-" * 78)
    print(f"CLEARED ({len(cleared)}): {', '.join(cleared) or '—'}")
    print(f"DROPPED ({len(dropped)}): {', '.join(dropped) or '—'}")
    print("\nCleared rows are ready for Grok sentiment + Claude scoring.")
    print("These numbers came from code, not an LLM. LLMs may consume — never originate — them.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"run_ts": run_ts,
                       "cleared": cleared, "dropped": dropped,
                       "rows": [asdict(r) for r in rows]}, fh, indent=2, default=str)
        print(f"\nJSON written to {args.json}")

    if args.sections:
        print("\n" + "=" * 78)
        print(emit_mfa_sections(rows, run_ts))

    if args.html:
        from report_html import build_html
        html_doc = build_html(rows, regime_metrics, regime_sum, run_ts)
        with open(args.html, "w") as fh:
            fh.write(html_doc)
        print(f"\nHTML report written to {args.html}")


if __name__ == "__main__":
    main()
