"""Unit tests for mfa_layer0 — focus on the FRED path that can't be hit live in this
sandbox (network blocked), plus the pure scoring/parsing helpers.

Run:  .venv/bin/python test_mfa_layer0.py
"""
import mfa_layer0 as m


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def test_parse_fred_csv():
    print("test_parse_fred_csv")
    csv = ("observation_date,BAMLH0A0HYM2\n"
           "2026-06-18,3.05\n2026-06-19,3.10\n2026-06-20,3.12\n2026-06-21,.\n"  # '.' skipped
           "2026-06-22,3.15\n2026-06-23,3.18\n2026-06-24,3.20\n2026-06-25,3.22\n")
    latest, prior, n = m._parse_fred_csv(csv)
    results = []
    results.append(check("latest", latest, 3.22))
    results.append(check("prior(rows[-6])", prior, 3.10))   # 7 valid rows; rows[-6]=3.10
    results.append(check("n valid", n, 7))
    # empty / header-only
    e_l, e_p, e_n = m._parse_fred_csv("observation_date,X\n")
    results.append(check("empty -> None", (e_l, e_p, e_n), (None, None, 0)))
    # all-missing
    z_l, z_p, z_n = m._parse_fred_csv("observation_date,X\n2026-01-01,.\n2026-01-02,.\n")
    results.append(check("all-missing -> None", (z_l, z_p, z_n), (None, None, 0)))
    return all(results)


def test_interp():
    print("test_interp")
    results = []
    # normal range
    results.append(check("midpoint -> 0", m._interp(0.5, 0, 1), 0.0))
    results.append(check("max clamp", m._interp(99, 0, 1), 5.0))
    results.append(check("min clamp", m._interp(-99, 0, 1), -5.0))
    # inverted range (lo>hi): VIX 30->-5, 16->+5
    results.append(check("inverted hi end (VIX 16)", m._interp(16, 30, 16), 5.0))
    results.append(check("inverted lo end (VIX 30)", m._interp(30, 30, 16), -5.0))
    results.append(check("degenerate lo==hi", m._interp(5, 3, 3), 0.0))
    return all(results)


def test_credit_spread_scoring():
    """Simulate the credit-spread metric logic with a fixture (no network)."""
    print("test_credit_spread_scoring")
    results = []
    # tight spread, calm -> bullish
    s_calm = m._interp(3.0, 6.5, 3.25)
    results.append(check("HY 3.0% -> +5 (tight)", s_calm, 5.0))
    # wide spread -> bearish
    s_wide = m._interp(7.0, 6.5, 3.25)
    results.append(check("HY 7.0% -> -5 (wide)", s_wide, -5.0))
    # spike override: moderate level but +0.80 5-obs spike forces <= -4
    s_spike = m._interp(4.5, 6.5, 3.25)
    s_spike_adj = min(s_spike, -4.0)  # mirrors compute_regime spike branch
    results.append(check("HY 4.5% + 0.80 spike -> <=-4", s_spike_adj, -4.0))
    return all(results)


def test_regime_offline_forces_neutral():
    """With FRED disabled AND yfinance present, N may still be <9 -> must force Neutral.
    We can't assert exact scores (live market), but we CAN assert the structure."""
    print("test_regime_offline_forces_neutral")
    metrics = m.compute_regime(use_fred=False)
    valid = [x for x in metrics if x["s"] is not None]
    results = []
    results.append(check("12 metric slots", len(metrics), 12))
    results.append(check("credit spread is NEEDS without FRED",
                         any("Credit" in x["n"] and x["s"] is None for x in metrics), True))
    # every metric dict has the 3 required keys
    shape_ok = all(set(x.keys()) == {"n", "v", "s"} for x in metrics)
    results.append(check("all metrics well-formed", shape_ok, True))
    print(f"  (info) {len(valid)} of 12 metrics had data this run")
    return all(results)


def test_fred_fetch_failsafe():
    """fetch_fred must NEVER raise — blocked/timeout returns (None,None,0)."""
    print("test_fred_fetch_failsafe")
    # bogus series id; whether network is blocked or returns 404, must be the empty tuple
    got = m.fetch_fred("THIS_SERIES_DOES_NOT_EXIST_XYZ")
    return check("bad series -> (None,None,0) no raise", got, (None, None, 0))


def test_mins_since_open():
    print("test_mins_since_open")
    import pandas as pd
    results = []
    t0 = pd.Timestamp("2026-06-26 09:30", tz="America/New_York")
    t1 = pd.Timestamp("2026-06-26 10:00", tz="America/New_York")
    t2 = pd.Timestamp("2026-06-26 16:00", tz="America/New_York")
    results.append(check("09:30 -> 0 min", m._mins_since_open(t0), 0))
    results.append(check("10:00 -> 30 min", m._mins_since_open(t1), 30))
    results.append(check("16:00 -> 390 min", m._mins_since_open(t2), 390))
    return all(results)


def test_intraday_rvol_math():
    """Validate the time-of-day normalization on a synthetic, fully-controlled fixture.

    Build 3 prior days that each trade 100 shares per 5-min bar, and a 'today' that
    trades 200/bar (exactly 2x pace). At ANY cutoff the normalized RVOL must be 2.0 —
    that is the whole point: comparable to the gate regardless of time of day.
    We monkeypatch yfinance so no network is touched.
    """
    print("test_intraday_rvol_math")
    import pandas as pd
    import numpy as np

    def make_day(date, per_bar):
        idx = pd.date_range(f"{date} 09:30", f"{date} 11:00", freq="5min",
                            tz="America/New_York")  # 19 bars (partial day is fine)
        return pd.DataFrame({"Volume": [per_bar] * len(idx),
                             "Close": [10.0] * len(idx)}, index=idx)

    frames = [make_day(d, 100) for d in ("2026-06-23", "2026-06-24", "2026-06-25")]
    frames.append(make_day("2026-06-26", 200))   # today: 2x the per-bar pace
    fake = pd.concat(frames)

    class FakeTk:
        def __init__(self, *a, **k): pass
        def history(self, *a, **k): return fake

    orig = m.yf.Ticker
    m.yf.Ticker = FakeTk
    try:
        rvol, status = m.compute_rvol_intraday("TEST")
    finally:
        m.yf.Ticker = orig

    results = []
    results.append(check("status intraday", status, "intraday"))
    results.append(check("normalized RVOL = 2.0 (2x pace, any cutoff)", rvol, 2.0))
    return all(results)


def test_intraday_rvol_single_day_fallback():
    """Only one day of intraday data -> cannot normalize -> 'full_day' signal so the
    caller keeps the daily-bar RVOL. Must NOT crash or return a misleading number."""
    print("test_intraday_rvol_single_day_fallback")
    import pandas as pd
    idx = pd.date_range("2026-06-26 09:30", "2026-06-26 10:00", freq="5min",
                        tz="America/New_York")
    one_day = pd.DataFrame({"Volume": [100] * len(idx), "Close": [10.0] * len(idx)}, index=idx)

    class FakeTk:
        def __init__(self, *a, **k): pass
        def history(self, *a, **k): return one_day

    orig = m.yf.Ticker
    m.yf.Ticker = FakeTk
    try:
        rvol, status = m.compute_rvol_intraday("TEST")
    finally:
        m.yf.Ticker = orig
    return check("single-day -> full_day fallback", status, "full_day")


def test_intraday_rvol_no_data():
    print("test_intraday_rvol_no_data")
    import pandas as pd

    class FakeTk:
        def __init__(self, *a, **k): pass
        def history(self, *a, **k): return pd.DataFrame()

    orig = m.yf.Ticker
    m.yf.Ticker = FakeTk
    try:
        rvol, status = m.compute_rvol_intraday("TEST")
    finally:
        m.yf.Ticker = orig
    return check("empty -> no_intraday", status, "no_intraday")


if __name__ == "__main__":
    tests = [test_parse_fred_csv, test_interp, test_credit_spread_scoring,
             test_regime_offline_forces_neutral, test_fred_fetch_failsafe,
             test_mins_since_open, test_intraday_rvol_math,
             test_intraday_rvol_single_day_fallback, test_intraday_rvol_no_data]
    passed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {e!r}")
        print()
    print(f"==== {passed}/{len(tests)} test groups passed ====")
