"""HTML report builder for MFA Layer 0 (cloud edition).

Produces a single self-contained, mobile-friendly report.html with NO external
dependencies (all CSS/JS inline) so it renders offline in mobile Safari and works
when served as a static file from GitHub Pages.

The page is a 3-stage wizard:
  Stage 1  — one-click copy of the Grok SENTIMENT prompt (survivors + instructions)
  Stage 2  — paste Grok's response into a textarea
  Stage 3  — auto-assembles the full CLAUDE SCORING prompt (Layer 0 data + Grok reply)
             with one-click copy.

All data is baked into the page at generation time; the JS only assembles text and
copies to clipboard — it never fetches anything.
"""

import html
import json
import math


def _fmt(x, nd=2, pct=False, dollar=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:.{nd}f}"
    if dollar:
        s = "$" + s
    if pct:
        s = s + "%"
    return s


def _rows_payload(rows):
    """Reduce TickerRow objects to a JSON-safe dict for embedding in the page."""
    out = []
    for r in rows:
        out.append({
            "ticker": r.ticker, "ok": r.ok,
            "conflicts": r.conflicts, "flags": r.flags,
            "last_split": r.last_split, "ath": r.ath,
            "low_52w": r.low_52w, "high_52w": r.high_52w,
            "next_earnings": r.next_earnings, "earnings_in_window": r.earnings_in_window,
            "adv": r.adv, "price": r.price, "as_of": r.as_of,
            "ema_ribbon": r.ema_ribbon, "ema_spread_pct": r.ema_spread_pct,
            "macd_hist": r.macd_hist, "rsi14": r.rsi14, "atr_pct": r.atr_pct,
            "adx14": r.adx14, "rvol": r.rvol, "rvol_basis": r.rvol_basis, "beta": r.beta,
            "passes_rvol_gate": r.passes_rvol_gate, "passes_adv_floor": r.passes_adv_floor,
            "short_pct_float": r.short_pct_float, "short_days_to_cover": r.short_days_to_cover,
            "short_trend": r.short_trend, "insider_net_shares": r.insider_net_shares,
            "insider_note": r.insider_note, "putcall_oi": r.putcall_oi,
            "dark_pool": r.dark_pool, "gex": r.gex,
        })
    return out


def _section_m(cleared):
    lines = ["SECTION M — PRE-HOOK MANIFEST"]
    for r in cleared:
        lines.append(
            f"{r.ticker:<6}| last split: {r.last_split} | ATH ≈ {_fmt(r.ath, dollar=True)} "
            f"| 52w range: {_fmt(r.low_52w, dollar=True)}–{_fmt(r.high_52w, dollar=True)} "
            f"| next earnings: {r.next_earnings} | ADV ≈ {r.adv/1e6:.1f}M")
    return "\n".join(lines)


def _section_b(cleared):
    has_alt = any(not math.isnan(r.short_pct_float) or not math.isnan(r.putcall_oi)
                  or not math.isnan(r.insider_net_shares) for r in cleared)
    lines = ["SECTION B — ALT DATA"]
    if not has_alt:
        lines.append("(not fetched — score Alt category as N/A per V6 §0B)")
        return "\n".join(lines)
    lines.append("TICKER | Dark Pool | Options Flow (P/C OI) | GEX | SI% float + DTC | Insider (6mo)")
    for r in cleared:
        sif = (f"{r.short_pct_float:.2f}% / {r.short_days_to_cover:.1f}d ({r.short_trend})"
               if not math.isnan(r.short_pct_float) else "N/A")
        pc = f"P/C OI {r.putcall_oi:.2f}" if not math.isnan(r.putcall_oi) else "N/A"
        ins = (f"{r.insider_net_shares:+,.0f} sh ({r.insider_note})"
               if not math.isnan(r.insider_net_shares) else "N/A")
        lines.append(f"{r.ticker} | {r.dark_pool} | {pc} | {r.gex} | {sif} | {ins}")
    lines.append("NOTE: Dark Pool + GEX = N/A (no free feed) — NOT estimated.")
    return "\n".join(lines)


def _section_c(cleared):
    lines = ["SECTION C — TECHNICALS",
             "TICKER | Price + Timestamp | split/ATH/52w reconciliation | EMA Ribbon | "
             "MACD | RSI | ATR% | RVOL% (basis) | ADX | Earnings | Beta"]
    for r in cleared:
        recon = (f"price {_fmt(r.price, dollar=True)} (as of {r.as_of}); split {r.last_split}; "
                 f"ATH {_fmt(r.ath, dollar=True)}; within 52w ✔")
        macd_txt = "bull (hist+)" if r.macd_hist > 0 else "bear (hist−)"
        ribbon = {"bullish": "+ribbon up", "bearish": "−ribbon down", "mixed": "mixed"}.get(r.ema_ribbon, r.ema_ribbon)
        rvol_txt = f"{r.rvol*100:.0f}% [{r.rvol_basis}]"
        beta_txt = f"{r.beta:.2f}" + (" ⚠V7" if r.beta > 1.5 else "")
        earn_txt = r.next_earnings + (" ⚠V1" if r.earnings_in_window else "")
        lines.append(
            f"{r.ticker} | {_fmt(r.price, dollar=True)} ({r.as_of}) | {recon} | {ribbon} "
            f"({r.ema_spread_pct:+.1f}%) | {macd_txt} | {r.rsi14:.0f} | {r.atr_pct:.1f}% "
            f"| {rvol_txt} | {r.adx14:.0f} | {earn_txt} | {beta_txt}")
    return "\n".join(lines)


def _section_opt(cleared):
    """Call chain snapshot block for the Claude prompt (chain-verified, ATM + 1st OTM).
    Only rows with a call_snapshot dict are included — omitted rows get a no-chain note.
    Claude must use ONLY these strikes; it must NOT invent or estimate option prices."""
    lines = ["SECTION OPT — CALL CHAIN SNAPSHOT (chain-verified, 21-45 DTE, ATM + 1st OTM)",
             "TICKER | expiry | DTE | ATM strike | ATM ask | OTM strike | OTM ask | IV%"]
    any_data = False
    for r in cleared:
        snap = getattr(r, "call_snapshot", None)
        if snap:
            any_data = True
            otm_s = f"${snap['otm_strike']:.0f}" if snap.get("otm_strike") else "—"
            otm_a = f"${snap['otm_ask']:.2f}" if snap.get("otm_ask") else "—"
            iv = f"{snap['atm_iv']:.0f}%" if snap.get("atm_iv") else "—"
            lines.append(
                f"{r.ticker} | {snap['expiry']} | {snap['dte']}d "
                f"| ${snap['atm_strike']:.0f} | ${snap['atm_ask']:.2f} "
                f"| {otm_s} | {otm_a} | {iv}")
        else:
            lines.append(f"{r.ticker} — no chain data (omit from options play)")
    if not any_data:
        lines.append("(no call snapshots available — options plays cannot be recommended this run)")
    return "\n".join(lines)


def build_html(rows, regime_metrics, regime_summary, run_ts):
    cleared = [r for r in rows if r.ok]
    dropped = [r for r in rows if not r.ok]
    survivors = [r.ticker for r in cleared]

    # Finnhub data-source badge: prove (or disprove) that Finnhub was reached this run.
    # earnings_source=='finnhub' only when Finnhub returned a forward date; price_xcheck is
    # non-empty only when the Finnhub quote endpoint answered. Summarize across cleared rows.
    fh_earn = sum(1 for r in cleared if getattr(r, "earnings_source", "") == "finnhub")
    fh_quote = sum(1 for r in cleared if getattr(r, "price_xcheck", ""))
    n_clear = len(cleared)
    if fh_earn or fh_quote:
        finnhub_badge = {"on": True,
                         "text": f"✓ Finnhub: {fh_earn}/{n_clear} earnings dates · "
                                 f"{fh_quote}/{n_clear} price cross-checks"}
    else:
        finnhub_badge = {"on": False,
                         "text": ("⚠ Finnhub not used this run (no key / unreachable) — "
                                  "earnings from yfinance, no price cross-check")}

    # FRED data-source badge — mirrors the Finnhub one so both feeds are verifiable at a glance.
    import mfa_layer0
    fred_badge = mfa_layer0.fred_badge_status(regime_metrics)

    sec_m = _section_m(cleared)
    sec_b = _section_b(cleared)
    sec_c = _section_c(cleared)
    sec_opt = _section_opt(cleared)

    grok_prompt = (
        "I have attached the MFA V6 Comprehensive Guide. Run Phase 1 Social Sentiment Analysis.\n"
        "You are given verified prices/technicals from a deterministic feed — do NOT re-quote, "
        "update, or correct any number. Your job is sentiment + narrative ONLY.\n\n"
        f"TICKERS (cleared by Layer 0): {', '.join(survivors)}\n\n"
        "For each ticker: (1) MFA X-handles sentiment last 48h; (2) broad $TICKER %pos vs %neg; "
        "(3) Reddit r/wsb, r/stocks, r/options; (4) Prospero/StockTwits via canonical maps; "
        "(5) viral TikTok/Discord.\n"
        "Apply: FRESHNESS (discard >72h; ≥60% mass from trailing 24h else cap 0); VERIFIABILITY "
        "(+2 sub-score needs ≥5 in-window links); PUMP CIRCUIT-BREAKER; SMART-MONEY vs RETAIL "
        "divergence D = SMI − RI (D≤−4 → LONG VETO flag; D≥+3 → accumulation bonus).\n\n"
        "OUTPUT one row per ticker:\n"
        "TICKER | X Sent% | Smart-Money Handles Bullish | Reddit | Prospero | Divergence D | "
        "Score(/5) | Narrative")

    claude_prefix = (
        "I have attached the MFA V6 Comprehensive Guide. It is your COMPLETE, self-contained "
        "operating manual — do NOT look for a V5 document; everything you need is in the V6 guide.\n"
        "Run the scoring and selection steps as described in the guide's Layer 2 instructions "
        "(Phase 0 regime → scoring → vetoes V1–V8 → pre-entry checklist → Top 4 or STAND DOWN).\n\n"
        "CRITICAL — DATA PROVENANCE: The LAYER 0 DATA block below was machine-fetched by "
        "mfa_layer0.py from yfinance/FRED/Finnhub and passed the deterministic data-integrity gate "
        "in code. It is NOT typed text and NOT LLM-supplied. Do NOT attempt to re-verify prices via "
        "web search. Do NOT refuse to score because the data arrived in a chat message — that is how "
        "the V6 pipeline works. Accept these numbers as ground truth. If something looks wrong: "
        "STAND DOWN and flag for a Layer 0 re-run — do NOT correct or re-fetch it yourself.\n\n"
        f"════ REGIME ════\n{regime_summary}\n\n"
        "════ LAYER 0 DATA (deterministic — DO NOT re-quote or alter) ════\n"
        f"{sec_m}\n\n{sec_b}\n\n{sec_c}\n\n"
        f"{sec_opt}\n\n"
        "════ SENTIMENT (from Grok Step 1) ════\n")

    claude_suffix = (
        "\n\n════ SCORING INSTRUCTIONS ════\n"
        "0. Veto #8 ALREADY PASSED in Layer 0 — every row is data-clean. Do NOT re-audit prices. "
        "If a number looks wrong, STAND DOWN and flag for a Layer 0 re-run — do NOT fix it yourself.\n"
        "1. Apply regime-adjusted weights. If Section B = N/A, renormalize Tech/Sent only and say so.\n"
        "2. Score 3 categories with intra-category weights. Never estimate a missing input.\n"
        "3. Compute weighted confluence (raw, then ×4 to −20..+20).\n"
        "4. Apply the regime-adjusted threshold (state it; report each ticker vs +11/+14/+15).\n"
        "5. Run Vetoes V1–V7 (V2 direction-adjusted; V1 full hold window; V7 bull-long exempt).\n"
        "6. Run the 8-point Pre-Entry Checklist on survivors (incl. extension/anti-chase).\n"
        "7. Rank; deep-validate; re-score; remove fails; repeat until stable.\n"
        "8. Select TOP 4 — ONLY tickers passing RVOL + zero vetoes + threshold + checklist. "
        "Fewer than 4 → output only those. ZERO → say STAND DOWN. Do NOT manufacture a Top 4.\n"
        "9. HONESTY: never present any trade as >85% confidence. Confluence ≠ win probability.\n\n"
        "OPTIONS PLAY RULE: For each top-4 pick, recommend a specific call option using ONLY the "
        "strikes and asks from SECTION OPT above — do NOT invent or estimate strikes. "
        "If SECTION OPT has no data for a ticker, give the stock play only and note 'no chain data'.\n\n"
        "OUTPUT: Top 4 briefing for each pick:\n"
        "  (a) STOCK PLAY: entry level, stop, target, rationale\n"
        "  (b) OPTIONS PLAY: strike → expiry from SECTION OPT, current ask ~$X, why this strike\n"
        "  (c) Confluence breakdown + honest win-rate band\n"
        "+ trade summary table + benchmark confluence table. Or STAND DOWN.")

    payload = {
        "run_ts": run_ts,
        "survivors": survivors,
        "finnhub_badge": finnhub_badge,
        "fred_badge": fred_badge,
        "dropped": [{"t": r.ticker, "why": "; ".join(r.conflicts)} for r in dropped],
        "regime": regime_summary,
        "grok_prompt": grok_prompt,
        "claude_prefix": claude_prefix,
        "claude_suffix": claude_suffix,
        "rows": _rows_payload(rows),
        "regime_metrics": regime_metrics or [],
    }

    data_json = json.dumps(payload)
    stand_down = len(survivors) == 0

    return _TEMPLATE.replace("/*__DATA__*/", data_json) \
                    .replace("__RUN_TS__", html.escape(run_ts)) \
                    .replace("__STAND_DOWN__", "true" if stand_down else "false")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MFA Layer 0 Report</title>
<style>
  :root { --bg:#0f1115; --card:#1a1d24; --fg:#e6e8eb; --mut:#9aa0aa; --acc:#4f8cff;
          --ok:#2ecc71; --bad:#ff5c5c; --warn:#ffb84d; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:14px; }
  h1 { font-size:18px; margin:0 0 2px; }
  h2 { font-size:15px; margin:18px 0 8px; color:var(--acc); }
  .mut { color:var(--mut); font-size:12px; }
  .srcbadge { font-size:11px; margin:4px 0 2px; padding:4px 8px; border-radius:7px; display:inline-block; }
  .srcbadge.on { background:#10331f; color:var(--ok); border:1px solid #1f5c38; }
  .srcbadge.off { background:#33270f; color:var(--warn); border:1px solid #5c451f; }
  .card { background:var(--card); border-radius:12px; padding:14px; margin:12px 0; }
  .banner { padding:12px 14px; border-radius:12px; font-weight:600; margin:12px 0; }
  .standdown { background:#3a1414; color:var(--bad); border:1px solid var(--bad); }
  .go { background:#10331f; color:var(--ok); border:1px solid var(--ok); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { text-align:left; padding:5px 6px; border-bottom:1px solid #262a33; white-space:nowrap; }
  th { color:var(--mut); font-weight:600; }
  .scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .pill { display:inline-block; padding:1px 6px; border-radius:6px; font-size:11px; }
  .p-ok{background:#10331f;color:var(--ok);} .p-bad{background:#3a1414;color:var(--bad);}
  .p-warn{background:#33270f;color:var(--warn);}
  textarea { width:100%; min-height:120px; background:#0c0e12; color:var(--fg);
             border:1px solid #2a2f3a; border-radius:10px; padding:10px; font:12px/1.4 ui-monospace,Menlo,monospace; resize:vertical; }
  button { background:var(--acc); color:#fff; border:0; border-radius:10px; padding:11px 14px;
           font-size:15px; font-weight:600; width:100%; margin-top:8px; cursor:pointer; }
  button.sec { background:#2a2f3a; }
  .hdr { display:flex; align-items:flex-start; justify-content:space-between; gap:10px; }
  .dbgbtn { width:auto; margin:0; padding:8px 12px; font-size:13px; background:#2a2f3a; flex:0 0 auto; }
  #dbgPanel { border:1px solid var(--acc); }
  #dbgPanel input[type=text]{ width:100%; background:#0c0e12; color:var(--fg); border:1px solid #2a2f3a;
             border-radius:10px; padding:10px; font-size:13px; margin-bottom:6px; }
  .dbgrow { display:flex; gap:8px; }
  .dbgrow button { flex:1; }
  .step { font-size:12px; color:var(--mut); margin-bottom:6px; }
  .hidden { display:none; }
  .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>MFA Layer 0 — Daily Report</h1>
    <div class="mut" id="runts"></div>
  </div>
  <button class="dbgbtn" id="dbgToggle" onclick="toggleDbg()">📥 Debug</button>
</div>
<div class="srcbadge" id="finnhubBadge"></div>
<div class="srcbadge" id="fredBadge"></div>

<div class="card hidden" id="dbgPanel">
  <div class="step">DEBUG / VALIDATE · download everything (Layer 0 data + prompts + your LLM replies) as one
    Markdown file to hand to Claude here — to validate results, explain unexpected output, or fine-tune.</div>
  <div class="step" style="margin-top:8px">Paste Claude's final output (optional):</div>
  <textarea id="claudeOut" placeholder="Paste Claude's Top-4 / STAND DOWN reply here (optional)..."></textarea>
  <input type="text" id="debugNote" placeholder='What should Claude explain? e.g. "why STAND DOWN?" (optional)'>
  <div class="dbgrow">
    <button onclick="downloadDebug(this)">📥 Download report</button>
    <button class="sec" onclick="copyDebug(this)">📋 Copy report</button>
  </div>
</div>

<div id="banner"></div>

<div class="card">
  <h2>Phase 0 — Regime</h2>
  <div id="regime" class="mut"></div>
  <div class="scroll"><table id="regimeTbl"></table></div>
</div>

<div class="card">
  <h2>Integrity Gate</h2>
  <div class="scroll"><table id="gateTbl"></table></div>
</div>

<div class="card">
  <h2>Technicals (cleared)</h2>
  <div class="scroll"><table id="techTbl"></table></div>
</div>

<div class="card" id="altCard">
  <h2>Alt Data — Section B</h2>
  <div class="scroll"><table id="altTbl"></table></div>
</div>

<div class="card" id="step1">
  <div class="step">STEP 1 · Copy this into the Grok app (attach the V6 guide first)</div>
  <textarea id="grokBox" readonly></textarea>
  <button onclick="copyEl('grokBox', this)">📋 Copy Grok prompt</button>
</div>

<div class="card" id="step2">
  <div class="step">STEP 2 · Paste Grok's full sentiment reply here</div>
  <textarea id="grokReply" placeholder="Paste Grok's response..."></textarea>
  <button onclick="genClaude()">⚙️ Generate Claude prompt</button>
</div>

<div class="card hidden" id="step3">
  <div class="step">STEP 3 · Copy this into the Claude app (attach the V6 guide first)</div>
  <textarea id="claudeBox" readonly></textarea>
  <button onclick="copyEl('claudeBox', this)">📋 Copy Claude prompt</button>
</div>

<div class="mut" style="margin-top:18px">Numbers are code-sourced (yfinance/FRED). LLMs consume — never originate — them.
Prices are ~15 min delayed; confirm live at your broker before entry.</div>

<script>
const D = /*__DATA__*/;
const STAND_DOWN = __STAND_DOWN__;

function copyEl(id, btn){
  const t = document.getElementById(id);
  t.select(); t.setSelectionRange(0, 999999);
  navigator.clipboard.writeText(t.value).then(()=>{
    const o = btn.textContent; btn.textContent='✅ Copied'; setTimeout(()=>btn.textContent=o,1200);
  }).catch(()=>{ document.execCommand('copy'); });
}

function genClaude(){
  const reply = document.getElementById('grokReply').value.trim();
  const body = D.claude_prefix + (reply || '[PASTE GROK SENTIMENT TABLE HERE]') + D.claude_suffix;
  document.getElementById('claudeBox').value = body;
  document.getElementById('step3').classList.remove('hidden');
  document.getElementById('step3').scrollIntoView({behavior:'smooth'});
}

// ── Debug / validate report ────────────────────────────────────────────────
const DEBUG_SCHEMA_VERSION = 1;
function toggleDbg(){
  const p = document.getElementById('dbgPanel');
  p.classList.toggle('hidden');
  if(!p.classList.contains('hidden')) p.scrollIntoView({behavior:'smooth'});
}
function _slug(s){ return (s||'').replace(/[^0-9A-Za-z]+/g,'-').replace(/^-|-$/g,''); }
function buildDebugReport(){
  // Reconstruct the EXACT Claude prompt the wizard produced (incl. the Grok reply the user pasted).
  const grokReply = (document.getElementById('grokReply')||{}).value || '';
  const claudeOut = (document.getElementById('claudeOut')||{}).value || '';
  const note = (document.getElementById('debugNote')||{}).value || '';
  const claudePrompt = D.claude_prefix + (grokReply.trim() || '[no Grok reply pasted]') + D.claude_suffix;
  const isBear = Array.isArray(D.bcs);
  const kind = isBear ? ('bearcall · profile=' + (D.profile||'?') + (D.slot?(' · slot='+D.slot):'')) : 'cloud (long screen)';
  const L = [];
  L.push('# MFA Debug Report');
  L.push('');
  L.push('> Generated for handing to Claude to **validate results / explain unexpected output / fine-tune**.');
  L.push('> Numbers are deterministic Layer 0 (code-sourced). LLMs consume them — never re-derive or alter.');
  L.push('');
  L.push('## Validation checklist (what to verify)');
  L.push('- Every value Claude used matches the Layer 0 numbers below (no hallucinated prices/technicals).');
  L.push('- The verdict respects the integrity gate (only CLEARED tickers are tradeable).');
  if(isBear){
    L.push('- Each Top-4 bear-call has an EMPTY veto list, `basis=="chain"`, POP ≤ 85, and NO earnings before expiry+2d.');
    L.push('- STAND DOWN is justified by the per-name veto reasons (distinguish a real veto from an after-hours stale-option-liquidity artifact: short_delta≈0 / iv_rv≈0 / V6 on every name ⇒ run was off-hours).');
  } else {
    L.push('- Top 4 = only tickers passing RVOL + zero vetoes + threshold + checklist; STAND DOWN if none.');
  }
  L.push('');
  L.push('## Run metadata');
  L.push('| field | value |');
  L.push('|---|---|');
  L.push('| run_ts | ' + (D.run_ts||'') + ' |');
  L.push('| report kind | ' + kind + ' |');
  L.push('| page URL | ' + location.href + ' |');
  L.push('| downloaded at | ' + new Date().toISOString() + ' |');
  L.push('| user agent | ' + navigator.userAgent + ' |');
  L.push('| debug_schema_version | ' + DEBUG_SCHEMA_VERSION + ' |');
  L.push('');
  L.push('## Data provenance');
  L.push('- FRED: ' + ((D.fred_badge&&D.fred_badge.text)||'(n/a)'));
  L.push('- Finnhub: ' + ((D.finnhub_badge&&D.finnhub_badge.text)||'(n/a)'));
  L.push('');
  L.push('## Regime');
  L.push('`' + (D.regime||'') + '`');
  L.push('');
  L.push('| metric | value | score |');
  L.push('|---|---|---|');
  (D.regime_metrics||[]).forEach(m=>{ const s=(m.s===null||m.s===undefined)?'—':m.s; L.push('| '+m.n+' | '+m.v+' | '+s+' |'); });
  L.push('');
  L.push('## Integrity gate');
  L.push('- Survivors (cleared): ' + ((D.survivors||[]).join(', ')||'(none)'));
  L.push('- Dropped: ' + ((D.dropped||[]).map(d=>d.t+' ('+d.why+')').join('; ')||'(none)'));
  L.push('');
  L.push('## Layer 0 — per ticker');
  L.push('| Tk | ok | price | as_of | RSI | MACDh | EMA | ATR% | ADX | RVOL | RVOLgate | ADVfloor | beta | next_earn | conflicts/flags |');
  L.push('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
  (D.rows||[]).forEach(r=>{
    const cf=[].concat(r.conflicts||[],r.flags||[]).join('; ');
    L.push('| '+[r.ticker, r.ok, r.price, r.as_of, r.rsi14, r.macd_hist, r.ema_ribbon, r.atr_pct,
      r.adx14, r.rvol, r.passes_rvol_gate, r.passes_adv_floor, r.beta, r.next_earnings, cf].join(' | ')+' |');
  });
  L.push('');
  if(isBear){
    L.push('## Bear-call decisions (the STAND-DOWN explainer)');
    L.push('| Tk | score | suitable | tradeable | basis | kind/DTE | short→long | credit/width | CWR | POP | shortΔ | vetoes |');
    L.push('|---|---|---|---|---|---|---|---|---|---|---|---|');
    (D.bcs||[]).forEach(r=>{
      const sl=(r.basis==='chain')?(r.short+'→'+r.long):'—';
      const cw=(r.basis==='chain')?(r.credit+'/'+r.width):'—';
      L.push('| '+[r.t, r.score, r.suitable, r.tradeable, r.basis, (r.kind||'')+' '+(r.dte||'')+'DTE',
        sl, cw, r.cwr, r.pop, r.delta, (r.vetoes||[]).join(' ; ')].join(' | ')+' |');
    });
    L.push('');
  }
  L.push('## Exact Grok prompt (step 1)');
  L.push('```\n' + (D.grok_prompt||'') + '\n```');
  L.push('');
  L.push('## Grok reply (as used)');
  L.push('```\n' + (grokReply.trim() || '[not pasted]') + '\n```');
  L.push('');
  L.push('## Exact Claude prompt (step 3)');
  L.push('```\n' + claudePrompt + '\n```');
  L.push('');
  L.push('## Claude output (the result to validate)');
  L.push('```\n' + (claudeOut.trim() || '[not pasted — pre-Claude debug report]') + '\n```');
  L.push('');
  L.push('## User question');
  L.push(note.trim() || '_(none — general validation requested)_');
  L.push('');
  L.push('## Raw data (lossless — the page\'s embedded D + captured fields)');
  const bundle = {debug_schema_version: DEBUG_SCHEMA_VERSION, downloaded_at: new Date().toISOString(),
    page_url: location.href, user_agent: navigator.userAgent,
    grok_reply: grokReply, claude_prompt: claudePrompt, claude_output: claudeOut, user_note: note, D: D};
  L.push('```json\n' + JSON.stringify(bundle, null, 2) + '\n```');
  return L.join('\n');
}
function _dbgFilename(){
  const isBear = Array.isArray(D.bcs);
  const tag = isBear ? ('bear-' + (D.profile||'p')) : 'cloud';
  return 'mfa-debug_' + tag + '_' + _slug(D.run_ts) + '.md';
}
function downloadDebug(btn){
  const md = buildDebugReport();
  try {
    const blob = new Blob([md], {type:'text/markdown'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = _dbgFilename();
    document.body.appendChild(a); a.click();
    setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 1500);
    const o=btn.textContent; btn.textContent='✅ Downloaded'; setTimeout(()=>btn.textContent=o,1400);
  } catch(e){ copyDebug(btn); }   // fallback for browsers that block Blob download
}
function copyDebug(btn){
  const md = buildDebugReport();
  navigator.clipboard.writeText(md).then(()=>{
    const o=btn.textContent; btn.textContent='✅ Copied'; setTimeout(()=>btn.textContent=o,1400);
  }).catch(()=>{
    const ta=document.createElement('textarea'); ta.value=md; document.body.appendChild(ta);
    ta.select(); document.execCommand('copy'); ta.remove();
    const o=btn.textContent; btn.textContent='✅ Copied'; setTimeout(()=>btn.textContent=o,1400);
  });
}

function el(tag, txt, cls){ const e=document.createElement(tag); if(txt!=null)e.textContent=txt; if(cls)e.className=cls; return e; }
function row(cells){ const tr=document.createElement('tr'); cells.forEach(c=>{ const td=document.createElement(typeof c==='object'&&c.th?'th':'td'); if(typeof c==='object'){td.innerHTML=c.html||'';}else{td.textContent=c;} tr.appendChild(td);}); return tr; }

// run timestamp + banner
document.getElementById('runts').textContent = 'Generated ' + D.run_ts;

// Finnhub data-source badge — visible proof of whether the Action reached Finnhub this run
const fb = D.finnhub_badge || {on:false, text:''};
const fbEl = document.getElementById('finnhubBadge');
if(fbEl){ fbEl.textContent = fb.text || ''; fbEl.className = 'srcbadge ' + (fb.on ? 'on' : 'off'); }

const fr = D.fred_badge || {on:false, text:''};
const frEl = document.getElementById('fredBadge');
if(frEl){ frEl.textContent = fr.text || ''; frEl.className = 'srcbadge ' + (fr.on ? 'on' : 'off'); }
const b = document.getElementById('banner');
if(STAND_DOWN){ b.className='banner standdown'; b.textContent='⛔ STAND DOWN — 0 survivors cleared the integrity gate. No trades today.'; }
else { b.className='banner go'; b.textContent='✅ '+D.survivors.length+' survivors cleared: '+D.survivors.join(', '); }

// regime
document.getElementById('regime').textContent = D.regime;
const rt = document.getElementById('regimeTbl');
rt.appendChild(row([{th:1,html:'Metric'},{th:1,html:'Value'},{th:1,html:'Score'}]));
(D.regime_metrics||[]).forEach(m=>{
  const s = (m.s===null||m.s===undefined)?'—':(m.s>0?'+':'')+m.s;
  rt.appendChild(row([m.n, m.v, s]));
});

// integrity gate
const gt = document.getElementById('gateTbl');
gt.appendChild(row([{th:1,html:'Ticker'},{th:1,html:'Price'},{th:1,html:'ATH'},{th:1,html:'As-of'},{th:1,html:'Verdict'}]));
D.rows.forEach(r=>{
  const v = r.ok ? {html:'<span class="pill p-ok">CLEARED</span>'} : {html:'<span class="pill p-bad">CONFLICT</span> '+(r.conflicts.join('; '))};
  gt.appendChild(row([r.ticker, r.price==null?'—':('$'+r.price.toFixed(2)), r.ath==null?'—':('$'+r.ath.toFixed(2)), r.as_of, v]));
});

// technicals (cleared only)
const tt = document.getElementById('techTbl');
tt.appendChild(row([{th:1,html:'Tk'},{th:1,html:'Price'},{th:1,html:'RSI'},{th:1,html:'MACDh'},{th:1,html:'EMA'},{th:1,html:'ATR%'},{th:1,html:'ADX'},{th:1,html:'RVOL'},{th:1,html:'Beta'},{th:1,html:'Gates'}]));
D.rows.filter(r=>r.ok).forEach(r=>{
  const rv = (r.rvol*100).toFixed(0)+'% '+(r.passes_rvol_gate?'<span class="ok">✔</span>':'<span class="bad">✘</span>')+' <span class="mut">['+r.rvol_basis+']</span>';
  const gates = (r.passes_adv_floor?'ADV✔':'<span class="bad">ADV✘</span>')+(r.beta>1.5?' <span class="warn">β'+r.beta+'</span>':'');
  tt.appendChild(row([r.ticker, '$'+r.price.toFixed(2), r.rsi14.toFixed(0), r.macd_hist.toFixed(2), r.ema_ribbon, r.atr_pct.toFixed(1), r.adx14.toFixed(0), {html:rv}, r.beta.toFixed(2), {html:gates}]));
});

// alt data
const at = document.getElementById('altTbl');
const hasAlt = D.rows.some(r=>r.ok && (r.short_pct_float===r.short_pct_float || r.putcall_oi===r.putcall_oi));
if(!hasAlt){ document.getElementById('altCard').querySelector('.scroll').innerHTML='<div class="mut">Not fetched (run with --alt). Score Alt as N/A per V6 §0B.</div>'; }
else {
  at.appendChild(row([{th:1,html:'Tk'},{th:1,html:'SI%float'},{th:1,html:'DTC'},{th:1,html:'Trend'},{th:1,html:'P/C OI'},{th:1,html:'Insider 6mo'},{th:1,html:'DarkPool'},{th:1,html:'GEX'}]));
  D.rows.filter(r=>r.ok).forEach(r=>{
    const f=v=>v===v?v:null; // NaN check
    at.appendChild(row([r.ticker,
      f(r.short_pct_float)==null?'N/A':r.short_pct_float.toFixed(2)+'%',
      f(r.short_days_to_cover)==null?'N/A':r.short_days_to_cover.toFixed(1),
      r.short_trend||'N/A',
      f(r.putcall_oi)==null?'N/A':r.putcall_oi.toFixed(2),
      f(r.insider_net_shares)==null?'N/A':(r.insider_net_shares>0?'+':'')+Math.round(r.insider_net_shares).toLocaleString()+' ('+r.insider_note+')',
      r.dark_pool, r.gex]));
  });
}

// fill prompts
document.getElementById('grokBox').value = D.grok_prompt;
if(STAND_DOWN){ ['step1','step2'].forEach(id=>document.getElementById(id).classList.add('hidden')); }
</script>
</body>
</html>"""


def universe_editor_html(tickers: list) -> str:
    """Self-contained universe editor page served from GitHub Pages.

    Lets the user view, edit, validate, and download the bullish-universe
    JSON file without touching Python. The downloaded file is committed to
    cloud/cloud_universe.json and picked up automatically on the next run.
    """
    import json as _json
    count = len(set(tickers))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MFA Cloud — Universe Editor</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f1115;color:#e6e8eb;font:15px/1.6 -apple-system,sans-serif;padding:18px}}
h1{{font-size:1.2rem;margin-bottom:4px;color:#fff}}
.sub{{color:#8b949e;font-size:.85rem;margin-bottom:16px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:14px}}
textarea{{width:100%;height:340px;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;
  border-radius:6px;padding:10px;font:13px/1.5 'Menlo','Courier New',monospace;resize:vertical}}
.row{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}}
button{{padding:9px 18px;border:none;border-radius:6px;cursor:pointer;font-size:.9rem;font-weight:600}}
.btn-val{{background:#388bfd;color:#fff}}
.btn-dl{{background:#2ea043;color:#fff}}
.btn-val:hover{{background:#58a6ff}}
.btn-dl:hover{{background:#3fb950}}
#status{{margin-top:10px;padding:8px 12px;border-radius:6px;font-size:.85rem;display:none}}
.ok{{background:#0d2119;color:#3fb950;border:1px solid #238636}}
.err{{background:#2d0f0f;color:#f85149;border:1px solid #da3633}}
.info{{background:#0d1b2e;color:#58a6ff;border:1px solid #388bfd}}
.note{{color:#8b949e;font-size:.82rem;line-height:1.6;margin-top:10px}}
a{{color:#58a6ff}}
</style>
</head>
<body>
<h1>MFA Cloud — Universe Editor</h1>
<p class="sub">Edit the ~{count}-ticker bullish-momentum universe. Download → commit → push to update.</p>

<div class="card">
  <p style="color:#8b949e;font-size:.85rem;margin-bottom:8px">
    One ticker per line, or comma-separated. Uppercase. Focus on optionable US equities with
    ADV&nbsp;&gt;&nbsp;2M, beta&nbsp;0.7–3.5, and momentum character (growth, tech, high-IV names).
    Remove delisted / illiquid / low-IV defensives freely.
  </p>
  <textarea id="ta" spellcheck="false">{chr(10).join(sorted(set(tickers)))}</textarea>
  <div class="row">
    <button class="btn-val" onclick="validate()">Validate</button>
    <button class="btn-dl" onclick="download()">Download universe JSON</button>
  </div>
  <div id="status"></div>
</div>

<div class="card">
  <p style="font-weight:600;margin-bottom:6px">How to update the live universe</p>
  <ol class="note" style="padding-left:1.2rem">
    <li>Edit the textarea above (add/remove tickers).</li>
    <li>Click <b>Validate</b> to check for errors.</li>
    <li>Click <b>Download universe JSON</b> — save the file as
        <code>cloud/cloud_universe.json</code> in your local repo clone.</li>
    <li>Commit and push. The next GitHub Actions workflow run picks it up automatically.</li>
  </ol>
  <p class="note" style="margin-top:8px">
    <b>Future LLM hook:</b> write a JSON array of tickers to
    <code>cloud/cloud_universe_override.json</code> from a Claude or Grok CLI call —
    it takes priority over this file with no code change needed.
  </p>
</div>

<script>
function parseTickers(raw) {{
  return [...new Set(
    raw.split(/[\\n,]+/)
       .map(t => t.trim().toUpperCase().replace(/[^A-Z0-9.]/g,''))
       .filter(t => t.length >= 1 && t.length <= 6)
  )].sort();
}}
function showStatus(msg, cls) {{
  const el = document.getElementById('status');
  el.textContent = msg; el.className = cls; el.style.display = 'block';
}}
function validate() {{
  const raw = document.getElementById('ta').value;
  const tickers = parseTickers(raw);
  const bad = raw.split(/[\\n,]+/)
    .map(t => t.trim().toUpperCase().replace(/[^A-Z0-9.]/g,''))
    .filter(t => t.length > 0 && (t.length > 6 || /^[0-9]+$/.test(t)));
  if (bad.length) {{
    showStatus('⚠ Suspicious entries (length > 6 or digits-only): ' + bad.join(', '), 'err');
    return;
  }}
  document.getElementById('ta').value = tickers.join('\\n');
  showStatus(`✓ Valid — ${{tickers.length}} unique tickers`, 'ok');
}}
function download() {{
  const raw = document.getElementById('ta').value;
  const tickers = parseTickers(raw);
  if (!tickers.length) {{ showStatus('Nothing to download — textarea is empty.', 'err'); return; }}
  const blob = new Blob([JSON.stringify(tickers, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cloud_universe.json';
  a.click();
  showStatus(`Downloaded cloud_universe.json (${{tickers.length}} tickers). Commit it to cloud/ and push.`, 'info');
}}
</script>
</body>
</html>"""
