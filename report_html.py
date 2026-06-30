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

    sec_m = _section_m(cleared)
    sec_b = _section_b(cleared)
    sec_c = _section_c(cleared)

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
        "I have attached the MFA V6 Comprehensive Guide. It is your operating manual.\n"
        "Run Phase 4 and Phase 5 using the data below.\n\n"
        f"════ REGIME ════\n{regime_summary}\n\n"
        "════ LAYER 0 DATA (deterministic — DO NOT re-quote or alter) ════\n"
        f"{sec_m}\n\n{sec_b}\n\n{sec_c}\n\n"
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
        "OUTPUT: Top 4 briefing (options + stock play, confluence breakdown, honest win-rate band) "
        "+ trade summary table + benchmark confluence table. Or STAND DOWN.")

    payload = {
        "run_ts": run_ts,
        "survivors": survivors,
        "finnhub_badge": finnhub_badge,
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
  .step { font-size:12px; color:var(--mut); margin-bottom:6px; }
  .hidden { display:none; }
  .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
</style>
</head>
<body>
<h1>MFA Layer 0 — Daily Report</h1>
<div class="mut" id="runts"></div>
<div class="srcbadge" id="finnhubBadge"></div>

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

function el(tag, txt, cls){ const e=document.createElement(tag); if(txt!=null)e.textContent=txt; if(cls)e.className=cls; return e; }
function row(cells){ const tr=document.createElement('tr'); cells.forEach(c=>{ const td=document.createElement(typeof c==='object'&&c.th?'th':'td'); if(typeof c==='object'){td.innerHTML=c.html||'';}else{td.textContent=c;} tr.appendChild(td);}); return tr; }

// run timestamp + banner
document.getElementById('runts').textContent = 'Generated ' + D.run_ts;

// Finnhub data-source badge — visible proof of whether the Action reached Finnhub this run
const fb = D.finnhub_badge || {on:false, text:''};
const fbEl = document.getElementById('finnhubBadge');
if(fbEl){ fbEl.textContent = fb.text || ''; fbEl.className = 'srcbadge ' + (fb.on ? 'on' : 'off'); }
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
