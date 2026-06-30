"""Bullish-momentum universe: ~262 optionable, liquid US equities.

This module owns the full ticker list and the cheap pre-screen that reduces it
to ~25 survivors before the expensive 1y-history analyze() loop runs.

Pre-screen scoring is tuned for BULLISH MOMENTUM:
  score = (price / 52w_high) × log(ADV)
Names closest to their 52w high with highest volume score highest — the
opposite of the bearcall pre-screen which favours names far below their highs.

Flow:
  load_universe() → ~262 tickers
  cheap_prescreen(tickers, max_keep=25) → 25 survivors
      Stage A: single yf.download(all, period='5d') — batch, fast
               reject: price ≤ 0, notional volume < $50M
               score: (price / 52w_high_proxy) × log(ADV)
               keep top 75
      Stage B: per-ticker fast_info on top 75 only
               reject: beta > 3.5, earnings within 10 days
               upgrade score using real 52w high from fast_info
               return top max_keep by score

Override hook (future LLM feed — no code change needed):
  Write a JSON array of tickers to cloud_universe_override.json.
  load_universe() picks it up automatically on the next run.
  Example:
    claude -p "List 80 optionable US equities showing bullish momentum ..." \\
      > cloud/cloud_universe_override.json
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

import yfinance as yf

# ── Universe file paths ────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(_DIR, "cloud_universe.json")
UNIVERSE_OVERRIDE_FILE = os.path.join(_DIR, "cloud_universe_override.json")

# ── Pre-screen thresholds ──────────────────────────────────────────────────────
NOTIONAL_VOL_MIN = 50_000_000   # price × volume; below this = illiquid
BETA_MAX = 3.5                  # too erratic even for long momentum plays
EARNINGS_DAYS = 10              # reject if earnings within this many days (hold window)
STAGE_A_KEEP = 75               # top N by volume score into Stage B
PRESCREEN_DEFAULT_KEEP = 25     # final survivors passed to analyze()

# ── Master universe list (~262 tickers) ────────────────────────────────────────
# Criteria: optionable US equity, ADV typically > 2M shares, beta > 0.7,
# historically elevated IV and momentum character. Biased toward growth/tech
# names that move on sentiment, earnings beats, product cycles.
# Update by editing cloud_universe.json and committing, or by writing
# cloud_universe_override.json from a CLI tool.
BULLISH_UNIVERSE = [
    # ── Tech / high-IV NDX-100 ────────────────────────────────────────────────
    "NVDA", "AMD", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG",
    "NFLX", "ADBE", "CRM", "ABNB", "MSTR", "COIN", "HOOD", "PLTR", "CRWD",
    "SNOW", "DDOG", "ZS", "PANW", "OKTA", "FTNT", "NET", "MDB", "SHOP",
    "SQ", "PYPL", "UBER", "LYFT", "SPOT", "RBLX", "RIVN", "SOFI", "UPST",
    "AFRM", "BILL", "DOCN", "GTLB", "CFLT", "SMCI", "IONQ", "ARM", "AVGO",
    "QCOM", "INTC", "MU", "LRCX", "AMAT", "KLAC", "ON", "MRVL", "ORCL",
    "DELL", "SNAP", "PINS", "TTD", "DKNG", "PENN", "IBKR", "SCHW",
    "SOUN", "ACHR", "JOBY", "RGTI", "LUNR", "APP", "APLT",
    # ── Semiconductors / hardware ─────────────────────────────────────────────
    "TSM", "ASML", "TXN", "ADI", "MCHP", "SWKS", "QRVO", "MPWR", "ENTG",
    "ONTO", "ACLS", "WOLF", "AMBA", "CRUS", "SLAB", "DIOD",
    # ── Software / cloud ─────────────────────────────────────────────────────
    "NOW", "WDAY", "TEAM", "ZM", "DOCU", "HUBS", "DOMO", "APPF", "PCOR",
    "AZPN", "MDLA", "AI", "PATH", "BBAI",
    # ── S&P mid / high-beta ───────────────────────────────────────────────────
    "MELI", "SE", "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI",
    "LCID", "F", "GM", "STLA", "CELH", "DUOL", "CAVA", "LULU", "NKE",
    "DECK", "CROX", "UAA", "TPR", "RL", "PVH", "CPRI", "GPS", "ANF",
    "URBN", "FIVE", "OLLI", "BBWI", "ULTA", "ELF", "KVYO", "HIMS",
    "RXRX", "GLBE", "PCVX", "SPHR", "HCP",
    # ── Consumer / media / travel ─────────────────────────────────────────────
    "DIS", "PARA", "WBD", "CMCSA", "CHTR", "FUBO", "ROKU", "SIRI",
    "BKNG", "EXPE", "MAR", "HLT", "MGM", "LVS", "WYNN", "CZR",
    # ── Financials / high-beta ────────────────────────────────────────────────
    "JPM", "GS", "MS", "BAC", "C", "WFC", "BX", "KKR", "APO", "BN",
    "CG", "ARES", "OWL", "NU", "LC", "OPEN", "COOP", "RKT",
    # ── Energy / commodities ─────────────────────────────────────────────────
    "XOM", "CVX", "COP", "OXY", "HAL", "MPC", "VLO", "DVN", "FANG",
    "EOG", "SLB", "BKR", "HES", "APA", "CTRA", "SM", "RRC", "AR", "EQT",
    "ARCH", "CEIX", "BTU", "METC",
    # ── Biotech / healthcare ─────────────────────────────────────────────────
    "MRNA", "BNTX", "VRTX", "REGN", "GILD", "BIIB", "ALNY", "ARWR",
    "BEAM", "CRSP", "EDIT", "NTLA", "FATE", "SAGE", "ACAD", "INVA",
    "ITCI", "AXSM", "SRTX", "PRGO", "JAZZ", "EXEL", "IOVA", "TGTX",
    "RCUS", "KYMR", "KURA", "FOLD", "PTCT", "RARE",
    # ── Sector ETFs (optionable, liquid) ─────────────────────────────────────
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "ARKK", "ARKG",
    "SMH", "SOXX", "KWEB", "GDX", "GDXJ", "USO", "UNG",
    "TLT", "HYG", "EMB", "EEM", "EWZ", "FXI",
    # ── Misc high-vol / meme-adjacent ────────────────────────────────────────
    "GME", "AMC", "SPCE", "NKLA", "WKHS", "RIDE", "HYLN",
    "PTRA", "REI", "BLNK", "CHPT", "EVGO", "FFIE",
]

# De-duplicate while preserving order
_seen: set = set()
_deduped = []
for _t in BULLISH_UNIVERSE:
    if _t not in _seen:
        _seen.add(_t)
        _deduped.append(_t)
BULLISH_UNIVERSE = _deduped


def load_universe(path: str = UNIVERSE_FILE,
                  override_path: str = UNIVERSE_OVERRIDE_FILE) -> list:
    """Return the active ticker universe.

    Priority:
      1. override_path (cloud_universe_override.json) — LLM-generated list
      2. path (cloud_universe.json) — user-edited list from HTML editor
      3. BULLISH_UNIVERSE constant — embedded fallback
    """
    for p, label in [(override_path, "LLM override"), (path, "JSON file")]:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    tickers = json.load(f)
                if isinstance(tickers, list) and tickers:
                    print(f"[UNIVERSE] loaded {len(tickers)} tickers from {label} ({p})")
                    return [str(t).upper().strip() for t in tickers if t]
            except Exception as e:
                print(f"[UNIVERSE] warning: could not load {p}: {e}")
    print(f"[UNIVERSE] using embedded BULLISH_UNIVERSE ({len(BULLISH_UNIVERSE)} tickers)")
    return list(BULLISH_UNIVERSE)


def cheap_prescreen(tickers: list, max_keep: int = PRESCREEN_DEFAULT_KEEP) -> list:
    """Reduce universe to top max_keep momentum candidates without 1y history.

    Stage A — batch yf.download(period='5d'):
      Gets Close + Volume for all tickers in one call.
      Rejects: price ≤ 0, notional volume (price × volume) < NOTIONAL_VOL_MIN.
      Scores: (price / 5d_high) × log(ADV) — favours names near recent highs
      with strong volume. This is a momentum signal: names running hard.
      Keeps top STAGE_A_KEEP by score.

    Stage B — per-ticker fast_info on Stage A survivors:
      Upgrades score using real 52w high: (price / 52w_high) × log(ADV).
      Rejects: beta > BETA_MAX, next earnings within EARNINGS_DAYS days.
      Returns top max_keep by updated score.

    All rejects printed as [PRESCREEN] lines for diagnostic visibility.
    """
    if not tickers:
        return []

    print(f"[PRESCREEN] stage A: batch fetch for {len(tickers)} tickers ...")
    try:
        raw = yf.download(
            tickers, period="5d", auto_adjust=True,
            progress=False, threads=True
        )
    except Exception as e:
        print(f"[PRESCREEN] batch download failed: {e} — falling back to DEFAULT_TICKERS")
        return tickers[:max_keep]

    if raw is None or raw.empty:
        print("[PRESCREEN] empty download result — skipping pre-screen")
        return tickers[:max_keep]

    scores = {}
    stage_a_rejects = {}

    for t in tickers:
        try:
            try:
                close_series = raw["Close"][t].dropna()
                vol_series = raw["Volume"][t].dropna()
            except (KeyError, TypeError):
                stage_a_rejects[t] = "not in batch download"
                continue

            if close_series.empty or vol_series.empty:
                stage_a_rejects[t] = "no price data"
                continue

            price = float(close_series.iloc[-1])
            avg_vol = float(vol_series.mean())
            recent_high = float(close_series.max())

            if price <= 0:
                stage_a_rejects[t] = f"price {price:.2f} ≤ 0"
                continue

            notional = price * avg_vol
            if notional < NOTIONAL_VOL_MIN:
                stage_a_rejects[t] = (f"notional vol ${notional/1e6:.1f}M "
                                      f"< ${NOTIONAL_VOL_MIN/1e6:.0f}M floor")
                continue

            # Momentum score: how close to recent high × log(ADV)
            # Near 1.0 = at/near high = strong momentum
            proximity = price / recent_high if recent_high > 0 else 0
            adv_score = math.log(max(avg_vol, 1))
            scores[t] = proximity * adv_score

        except Exception as e:
            stage_a_rejects[t] = f"error: {e}"

    for t, reason in stage_a_rejects.items():
        print(f"[PRESCREEN] stage A reject: {t} — {reason}")

    stage_a_survivors = sorted(scores, key=lambda x: scores[x], reverse=True)[:STAGE_A_KEEP]
    print(f"[PRESCREEN] stage A: {len(stage_a_survivors)} survivors from {len(tickers)} "
          f"({len(stage_a_rejects)} rejected)")

    # ── Stage B: per-ticker fast_info on top survivors ─────────────────────────
    print(f"[PRESCREEN] stage B: checking earnings + beta on {len(stage_a_survivors)} tickers ...")
    today = datetime.now(timezone.utc).date()
    earnings_cutoff = today + timedelta(days=EARNINGS_DAYS)
    stage_b_rejects = {}
    stage_b_ok = []

    for t in stage_a_survivors:
        try:
            info = yf.Ticker(t).fast_info

            # Beta check
            beta = getattr(info, "beta", None)
            if beta is None:
                try:
                    beta = yf.Ticker(t).info.get("beta", None)
                except Exception:
                    beta = None
            if beta is not None and float(beta) > BETA_MAX:
                stage_b_rejects[t] = f"beta {beta:.2f} > {BETA_MAX}"
                continue

            # Earnings proximity check
            earn_date = None
            try:
                cal = yf.Ticker(t).calendar
                if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                    if hasattr(cal, 'get'):
                        ed = cal.get("Earnings Date")
                    elif hasattr(cal, 'iloc'):
                        ed = cal.get("Earnings Date", [None])
                        if hasattr(ed, 'iloc'):
                            ed = ed.iloc[0] if len(ed) > 0 else None
                    else:
                        ed = None
                    if ed is not None:
                        if hasattr(ed, 'date'):
                            earn_date = ed.date()
                        elif hasattr(ed, '__iter__') and not isinstance(ed, str):
                            items = list(ed)
                            if items:
                                first = items[0]
                                earn_date = first.date() if hasattr(first, 'date') else None
            except Exception:
                earn_date = None

            if earn_date is not None and today <= earn_date <= earnings_cutoff:
                stage_b_rejects[t] = (f"earnings {earn_date} within "
                                      f"{EARNINGS_DAYS}d ({earnings_cutoff})")
                continue

            # Upgrade to real 52w high from fast_info for final ranking
            try:
                high52 = getattr(info, "year_high", None)
                price_now = getattr(info, "last_price", None)
                if high52 and price_now and high52 > 0:
                    proximity_52w = price_now / high52
                    adv_score = math.log(max(scores.get(t, 1), 1))
                    scores[t] = proximity_52w * adv_score
            except Exception:
                pass  # keep Stage A score

            stage_b_ok.append(t)

        except Exception as e:
            print(f"[PRESCREEN] stage B error for {t}: {e} — keeping")
            stage_b_ok.append(t)

    for t, reason in stage_b_rejects.items():
        print(f"[PRESCREEN] stage B reject: {t} — {reason}")

    result = sorted(stage_b_ok, key=lambda x: scores.get(x, 0), reverse=True)[:max_keep]

    print(f"[PRESCREEN] stage B: {len(result)} final survivors "
          f"(from {len(stage_a_survivors)}, {len(stage_b_rejects)} rejected)")
    print(f"[PRESCREEN] survivors: {result}")
    return result
