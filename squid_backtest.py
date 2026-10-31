#!/usr/bin/env python3
"""
squid_backtest.py
=================
Backtest of the Jungle Rock "Squid Programs" (VIX term-structure dislocation
allocation between ES futures and a short-UX1 / long-UX3 VIX futures spread).

Data (all free, downloaded and cached under --data-dir on first run):
  * VIX futures, every monthly contract   -> Cboe CFE historical CSVs (2004+)
  * VIX spot index                        -> Cboe VIX_History.csv
  * S&P 500 Total Return (^SP500TR)       -> Yahoo Finance via yfinance
  * 3-month T-bill (DTB3)                 -> FRED API (needs a free API key)
  ES excess return is proxied as SP500TR return minus the T-bill rate. You can
  instead pass a ratio-adjusted continuous ES series with --es-csv.

Interpretation choices (the paper is ambiguous; see README notes in chat):
  * ES weight w = 1 - D/7, where D = number of UX1..UX7 out of contango order.
  * VA daily return = -0.5 * r(UX1 held) + 1.0 * r(UX3 held); legs roll
    ROLL_DAYS trading days before VIX expiry.
  * Refined programs: if mc < m < 0 the vol weight is halved; if m > 0 and
    VIX > Vc the vol weight is doubled (ES can go short). m = 1 - UX1/UX7.
  * Weekly programs: weekly mean of D, week-end values of m and VIX, weights
    set at the last trading day of the week and held the following week.
  * Portfolio is fully collateralized: earns the T-bill rate on capital
    (disable with --no-cash). Costs: --cost-bps per unit of notional traded,
    including VIX rolls and quarterly ES rolls.

Requirements: pip install pandas numpy requests matplotlib yfinance
FRED API key: get one free at https://fred.stlouisfed.org/docs/api/api_key.html,
              then either  export FRED_API_KEY=your_key   (recommended)
              or pass      --fred-api-key your_key
              The T-bill series is cached in data/DTB3.csv; use --refresh to update it.
Usage:        python squid_backtest.py               # full run
              python squid_backtest.py --sweep       # threshold robustness grid
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CBOE_NEW = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{date}.csv"
CBOE_ARCHIVE = "https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{code}{yy}_VX.csv"
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
MONTH_CODES = "FGHJKMNQUVXZ"
TODAY = pd.Timestamp.today().normalize()

PROGRAMS = {
    "Cuttlefish":     dict(freq="D", refined=False, short=True),
    "Squid":          dict(freq="W", refined=False, short=True),
    "Colossal Squid": dict(freq="D", refined=True,  short=True),
    "Giant Squid":    dict(freq="W", refined=True,  short=True),
    "Jumbo Squid":    dict(freq="W", refined=True,  short=False),
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7 + 14)


def vx_expiry(y, m):
    """Rule-based monthly VX expiry: Wednesday 30 days before the 3rd Friday of
    the following month (holiday shifts handled by download fallback)."""
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return third_friday(ny, nm) - pd.Timedelta(days=30)


def http_get(session, url):
    try:
        r = session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "Trade Date" in r.text[:2000]:
            return r.text
    except requests.RequestException:
        pass
    return None


def download_contract(session, y, m, exp):
    urls = []
    if y >= 2013:  # current site; try holiday-shifted dates too
        urls += [CBOE_NEW.format(date=(exp - pd.Timedelta(days=k)).strftime("%Y-%m-%d")) for k in (0, 1, 2)]
    if y <= 2014:
        urls.append(CBOE_ARCHIVE.format(code=MONTH_CODES[m - 1], yy=f"{y % 100:02d}"))
    for url in urls:
        text = http_get(session, url)
        if text:
            return text
    return None


def parse_cfe_csv(text):
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("trade date"))
    except StopIteration:
        return None
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    df["settle"] = pd.to_numeric(df["Settle"], errors="coerce")
    df = df.dropna(subset=["date", "settle"])
    df = df[df["settle"] > 0].copy()
    # Before 2007-03-26 VX futures were quoted at 10x the VIX level.
    old = (df["date"] < "2007-03-26") & (df["settle"] > 50)
    df.loc[old, "settle"] /= 10.0
    return df[["date", "settle"]]


def load_vx_futures(data_dir, start, end, refresh=False):
    cache = data_dir / "vx"
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    frames, missing = [], []
    first_year = start.year
    last_year = min(end.year, TODAY.year) + 1
    for y in range(first_year, last_year + 1):
        for m in range(1, 13):
            exp = vx_expiry(y, m)
            if exp < start or exp > end + pd.DateOffset(months=10):
                continue
            f = cache / f"VX_{y}{m:02d}.csv"
            marker = cache / f"VX_{y}{m:02d}.missing"
            stale = exp >= TODAY - pd.Timedelta(days=5)  # live contract: refresh
            if marker.exists() and not refresh:
                continue
            if not f.exists() or refresh or stale:
                text = download_contract(session, y, m, exp)
                time.sleep(0.15)
                if text is None:
                    missing.append(f"{y}-{m:02d}")
                    print(f"[warn] VX missing {y}-{m:02d}")
                    if exp < TODAY:
                        marker.touch()
                    continue
                f.write_text(text)
            df = parse_cfe_csv(f.read_text())
            if df is None or df.empty:
                continue
            if 2013 <= y <= 2014:
                # Overlap years: the current-site file can start months after the contract
                # listed, so merge in the archive copy (current-site rows win on overlap).
                fa = cache / f"VX_{y}{m:02d}_archive.csv"
                marker_a = cache / f"VX_{y}{m:02d}_archive.missing"
                if refresh or (not fa.exists() and not marker_a.exists()):
                    text = http_get(session, CBOE_ARCHIVE.format(code=MONTH_CODES[m - 1], yy=f"{y % 100:02d}"))
                    time.sleep(0.15)
                    if text:
                        fa.write_text(text)
                    else:
                        marker_a.touch()
                if fa.exists():
                    dfa = parse_cfe_csv(fa.read_text())
                    if dfa is not None and not dfa.empty:
                        added = (~dfa["date"].isin(df["date"])).sum()
                        if added:
                            print(f"[info] VX {y}-{m:02d}: added {added} earlier/missing days from the Cboe archive file")
                        df = (pd.concat([df, dfa]).drop_duplicates("date", keep="first")
                              .sort_values("date").reset_index(drop=True))
            last = df["date"].max()
            expiry = last if (exp < TODAY and 0 <= (exp - last).days <= 3) else exp
            df["expiry"] = expiry
            frames.append(df)
    if missing:
        print(f"[warn] could not download {len(missing)} VX contracts: {missing[:10]}...")
    if not frames:
        sys.exit("No VIX futures data loaded.")
    long = pd.concat(frames)
    prices = long.pivot_table(index="date", columns="expiry", values="settle", aggfunc="last")
    return prices.sort_index().sort_index(axis=1)


def load_csv_cached(path, url, refresh=False):
    if refresh or not path.exists():
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        path.write_text(r.text)
    return pd.read_csv(path)


def load_vix_spot(data_dir, refresh=False):
    df = load_csv_cached(data_dir / "VIX_History.csv", CBOE_VIX, refresh)
    df.columns = [c.strip().upper() for c in df.columns]
    s = pd.Series(df["CLOSE"].values, index=pd.to_datetime(df["DATE"]), name="vix")
    return s.sort_index()


def fred_observations(series_id, api_key, start=None, end=None):
    """Download a FRED series through the official API (fred/series/observations).

    Returns a DataFrame with columns date, value. FRED marks missing days with ".",
    which become NaN. The API key is sent only to api.stlouisfed.org.
    """
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start is not None:
        params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    if end is not None:
        params["observation_end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    for attempt in range(3):
        try:
            r = requests.get(FRED_API, params=params, timeout=60)
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"FRED API request failed: {e.__class__.__name__}") from None
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 and attempt < 2:  # rate limited (120 requests/minute)
            time.sleep(5 * (attempt + 1))
            continue
        try:
            payload = r.json()
        except ValueError:
            payload = {}
        if r.status_code != 200 or "observations" not in payload:
            msg = payload.get("error_message") or r.reason or "unknown error"
            # Never echo the request URL: it contains the API key.
            raise RuntimeError(f"FRED API error for {series_id} (HTTP {r.status_code}): {msg}")
        obs = payload["observations"]
        df = pd.DataFrame(obs, columns=["date", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df
    raise RuntimeError(f"FRED API error for {series_id}: rate limited")


def load_rf(data_dir, start=None, end=None, api_key=None, refresh=False, series_id="DTB3"):
    """3-month T-bill rate (annual, decimal) from FRED, cached as CSV in data_dir."""
    path = data_dir / f"{series_id}.csv"
    if refresh or not path.exists():
        if not api_key:
            sys.exit("A FRED API key is required to download the T-bill series. Set the "
                     "FRED_API_KEY environment variable or pass --fred-api-key "
                     "(free key: https://fred.stlouisfed.org/docs/api/api_key.html), "
                     "or use --no-cash to skip interest on collateral.")
        # Pull a little earlier than the backtest start so the first days can forward-fill.
        obs_start = None if start is None else pd.Timestamp(start) - pd.Timedelta(days=30)
        df = fred_observations(series_id, api_key, obs_start, end)
        if df.empty:
            sys.exit(f"FRED returned no observations for {series_id}.")
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return (s.sort_index().ffill() / 100.0).rename("rf_annual")


def load_equity(data_dir, start, refresh=False, es_csv=None):
    """Returns (price series, is_excess). is_excess=True for a futures series."""
    if es_csv:
        df = pd.read_csv(es_csv)
        s = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))
        return s.sort_index(), True
    path = data_dir / "spxtr.csv"
    if refresh or not path.exists():
        import yfinance as yf
        px = yf.download("^SP500TR", start=(start - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                         auto_adjust=False, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px.rename("close").to_csv(path, index_label="date")
    df = pd.read_csv(path)
    s = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))
    return s.sort_index(), False


# --------------------------------------------------------------------------- #
# Signal and leg construction
# --------------------------------------------------------------------------- #
def dislocation_count(v):
    """Eq. (4): count j where |{i: UXi < UXj}| + 1 != j."""
    ranks = (v[None, :] < v[:, None]).sum(axis=1) + 1
    return int((ranks != np.arange(1, len(v) + 1)).sum())


def build_signal_and_legs(prices, roll_days):
    dates = prices.index
    exps = np.array(prices.columns, dtype="datetime64[ns]")
    P = prices.values
    n = len(dates)
    D = np.full(n, np.nan)
    ux1 = np.full(n, np.nan)
    ux7 = np.full(n, np.nan)
    held1 = np.full(n, -1)
    held3 = np.full(n, -1)
    dvals = dates.values

    # first/last date with a price for each contract. A contract only counts as
    # "listed" between these dates, so a month that had not started trading yet
    # (e.g. 2006, when Cboe listed fewer months) is skipped instead of held.
    has_px = ~np.isnan(P)
    first_px = [dates[has_px[:, j]].min() if has_px[:, j].any() else pd.NaT for j in range(P.shape[1])]
    last_px = [dates[has_px[:, j]].max() if has_px[:, j].any() else pd.NaT for j in range(P.shape[1])]
    far = np.datetime64("2262-01-01")
    first_np = np.array([np.datetime64(t) if not pd.isna(t) else far for t in first_px], dtype="datetime64[ns]")
    last_np = np.array([np.datetime64(t) if not pd.isna(t) else np.datetime64("1970-01-01") for t in last_px],
                       dtype="datetime64[ns]")

    curve_issue = np.full(n, "", dtype=object)
    for i in range(n):
        listed = np.where((exps > dvals[i]) & (first_np <= dvals[i]) & (last_np >= dvals[i]))[0]
        sig = listed[:7]
        if len(sig) == 7:
            gaps = np.diff(exps[sig]).astype("timedelta64[D]").astype(int)
            v = P[i, sig]
            if not np.isnan(v).any() and gaps.max() < 40:
                D[i] = dislocation_count(v)
                ux1[i], ux7[i] = v[0], v[6]
        if np.isnan(D[i]):
            curve_issue[i] = describe_curve_gap(dates[i], exps, listed, P[i])
        # tradable contracts: skip the front if within roll window
        if len(listed) >= 4:
            pos_exp = np.searchsorted(dvals, exps[listed[0]], side="right") - 1
            k = 1 if (pos_exp - i) <= roll_days else 0
            held1[i], held3[i] = listed[k], listed[k + 2]

    # VA leg daily return using contracts chosen at the previous close
    r_va = np.zeros(n)
    roll = np.zeros(n, dtype=bool)
    gap_rows = []
    for i in range(1, n):
        c1, c3 = held1[i - 1], held3[i - 1]
        if c1 < 0:
            continue
        r1 = P[i, c1] / P[i - 1, c1] - 1
        r3 = P[i, c3] / P[i - 1, c3] - 1
        for leg, c, r in (("UX1 (short)", c1, r1), ("UX3 (long)", c3, r3)):
            if not np.isnan(r):
                continue
            exp = pd.Timestamp(exps[c])
            missing_on = [d.date() for d, ok in ((dates[i - 1], has_px[i - 1, c]), (dates[i], has_px[i, c])) if not ok]
            if pd.isna(first_px[c]) or dates[i] <= first_px[c]:
                reason = "before contract's first price (file starts late)"
            elif dates[i - 1] >= last_px[c]:
                reason = "after contract's last price (file ends early)"
            else:
                reason = "gap inside contract's price history"
            gap_rows.append({"date": dates[i].date(), "leg": leg, "contract_expiry": exp.date(),
                             "cache_file": f"VX_{exp.year}{exp.month:02d}.csv",
                             "price_missing_on": " & ".join(map(str, missing_on)), "reason": reason})
        r1 = 0.0 if np.isnan(r1) else r1
        r3 = 0.0 if np.isnan(r3) else r3
        r_va[i] = -0.5 * r1 + 1.0 * r3
        roll[i] = held1[i] != held1[i - 1] and held1[i] >= 0
    gaps = pd.DataFrame(gap_rows, columns=["date", "leg", "contract_expiry", "cache_file",
                                           "price_missing_on", "reason"])
    if not gaps.empty:
        n_days = gaps["date"].nunique()
        print(f"[warn] {n_days} days with missing VX prices treated as zero return "
              f"(details: vx_price_gaps.csv in the results folder)")
        by_file = (gaps.groupby(["cache_file", "reason"]).agg(days=("date", "nunique"),
                   first=("date", "min"), last=("date", "max")).sort_values("days", ascending=False))
        print(by_file.head(15).to_string())
    out = pd.DataFrame({"D": D, "ux1": ux1, "ux7": ux7, "r_va": r_va, "va_roll": roll,
                        "curve_issue": curve_issue}, index=dates)
    out["m"] = 1 - out["ux1"] / out["ux7"]
    return out, gaps


def describe_curve_gap(date, exps, listed, prices_today):
    """Why there is no complete UX1..UX7 curve on `date`: which of the next seven
    monthly contracts is unlisted (file missing or starts late) or has no price."""
    d = pd.Timestamp(date)
    listed_months = {(pd.Timestamp(exps[j]).year, pd.Timestamp(exps[j]).month): j for j in listed}
    expected, y, m = [], d.year, d.month
    while len(expected) < 7:
        if vx_expiry(y, m) > d:
            expected.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    problems = []
    for ym in expected:
        tag = f"{ym[0]}-{ym[1]:02d}"
        if ym not in listed_months:
            problems.append(f"{tag} not listed")
        elif np.isnan(prices_today[listed_months[ym]]):
            problems.append(f"{tag} no price")
    return "; ".join(problems) if problems else "other"


def es_roll_days(index):
    flags = pd.Series(False, index=index)
    for y in range(index[0].year, index[-1].year + 1):
        for m in (3, 6, 9, 12):
            target = third_friday(y, m) - pd.Timedelta(days=8)
            pos = index.searchsorted(target)
            if 0 <= pos < len(index):
                flags.iloc[pos] = True
    return flags


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
def target_weights(D, m, vix, refined, short, mc, vc, jumbo_mode):
    w = 1.0 - D / 7.0
    es, va = w.copy(), 1.0 - w
    if refined:
        damp = (m < 0) & (m > mc)
        boost = (m > 0) & (vix > vc)
        es = np.where(damp, 0.5 * w + 0.5, np.where(boost, 2 * w - 1, w))
        va = np.where(damp, 0.5 * (1 - w), np.where(boost, 2 * (1 - w), 1 - w))
        es, va = pd.Series(es, index=D.index), pd.Series(va, index=D.index)
    if not short:
        es = es.clip(lower=0)
        if jumbo_mode == "cap":
            va = va.clip(upper=1)
    return es, va


def run_program(df, cfg, args):
    if cfg["freq"] == "D":
        D, m, vix = df["D"], df["m"], df["vix"]
        es, va = target_weights(D, m, vix, cfg["refined"], cfg["short"], args.mc, args.vc, args.jumbo_mode)
    else:
        week = pd.Series(df.index.to_period("W-FRI"), index=df.index)
        is_end = week != week.shift(-1)
        D_avg = df["D"].groupby(week.values).transform(lambda s: s.expanding().mean())
        es, va = target_weights(D_avg, df["m"], df["vix"], cfg["refined"], cfg["short"],
                                args.mc, args.vc, args.jumbo_mode)
        es = es.where(is_end).ffill()
        va = va.where(is_end).ffill()
    # positions held from close t (after exec lag) into t+1
    pos_es = es.shift(args.exec_lag - 1).fillna(0) if args.exec_lag > 1 else es.fillna(0)
    pos_va = va.shift(args.exec_lag - 1).fillna(0) if args.exec_lag > 1 else va.fillna(0)

    c = args.cost_bps / 1e4
    turnover = (pos_es.diff().abs() + 1.5 * pos_va.diff().abs()).fillna(0)
    turnover += df["va_roll"] * 3.0 * pos_va.abs() + df["es_roll"] * 2.0 * pos_es.abs()
    cash = 0.0 if args.no_cash else df["rf"]
    ret = pos_es.shift(1) * df["r_es_ex"] + pos_va.shift(1) * df["r_va"] + cash - c * turnover
    return ret.fillna(0), pos_es, pos_va, turnover


def perf_stats(r, rf, turnover=None):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    ex = r - rf
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(252)
    return {
        "Ann. Return": eq.iloc[-1] ** (1 / yrs) - 1,
        "Ann. Vol": r.std() * np.sqrt(252),
        "Max DD": -(eq / eq.cummax() - 1).min(),
        "Sharpe": ex.mean() / r.std() * np.sqrt(252),
        "Sortino": ex.mean() * 252 / downside,
        "Avg daily turnover": np.nan if turnover is None else turnover.mean(),
    }


def bucket_table(df):
    b = pd.cut(df["D"], [-0.5, 0.5, 4.5, 7.5], labels=["Contango (0)", "Low (2-4)", "High (5-7)"])
    es_tot = df["r_es_ex"] + df["rf"]
    t = pd.DataFrame({"bucket": b, "ES_t": es_tot, "VA_t": df["r_va"],
                      "ES_t+1": es_tot.shift(-1), "VA_t+1": df["r_va"].shift(-1)})
    g = t.groupby("bucket", observed=True)
    out = g.mean() * 100
    out.insert(0, "count", g.size())
    return out.round(3)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Squid Programs backtest")
    ap.add_argument("--start", default="2006-04-05")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--mc", type=float, default=-0.10, help="critical negative slope (placeholder)")
    ap.add_argument("--vc", type=float, default=25.0, help="VIX threshold (placeholder)")
    ap.add_argument("--roll-days", type=int, default=3)
    ap.add_argument("--exec-lag", type=int, default=1, help="1 = trade at the signal close")
    ap.add_argument("--jumbo-mode", choices=["cap", "keep"], default="cap",
                    help="Jumbo: cap vol weight at 1 when ES floored at 0, or keep 2(1-w)")
    ap.add_argument("--no-cash", action="store_true", help="exclude T-bill interest on collateral")
    ap.add_argument("--es-csv", default=None, help="optional ratio-adjusted continuous ES CSV (date,close)")
    ap.add_argument("--sweep", action="store_true", help="grid over mc and Vc for refined programs")
    ap.add_argument("--refresh", action="store_true", help="re-download all data")
    ap.add_argument("--fred-api-key", default=None,
                    help="FRED API key (default: FRED_API_KEY environment variable)")
    args = ap.parse_args()
    fred_key = args.fred_api_key or os.environ.get("FRED_API_KEY")

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VIX futures...")
    prices = load_vx_futures(data_dir, start - pd.Timedelta(days=10), end, args.refresh)
    print("Loading VIX spot, T-bills, equity...")
    vix = load_vix_spot(data_dir, args.refresh)
    if args.no_cash and not fred_key and not (data_dir / "DTB3.csv").exists():
        # Interest is excluded anyway; the T-bill rate is still used for excess returns
        # and Sharpe, so fall back to zero rather than demanding a key.
        print("[info] no FRED key and --no-cash set: using a 0% risk-free rate")
        rf_ann = pd.Series(0.0, index=pd.DatetimeIndex([start - pd.Timedelta(days=30)]), name="rf_annual")
    else:
        rf_ann = load_rf(data_dir, start, end, fred_key, args.refresh)
    eq_px, eq_is_excess = load_equity(data_dir, start, args.refresh, args.es_csv)

    idx = prices.index.intersection(eq_px.index).intersection(vix.index)
    idx = idx[(idx >= start - pd.Timedelta(days=10)) & (idx <= end)]
    prices = prices.loc[idx]
    legs, vx_gaps = build_signal_and_legs(prices, args.roll_days)
    vx_gaps.to_csv(out_dir / "vx_price_gaps.csv", index=False)
    markers = sorted(p.stem for p in (data_dir / "vx").glob("*.missing"))
    main_missing = [s for s in markers if not s.endswith("_archive")]
    arch_missing = [s.replace("_archive", "") for s in markers if s.endswith("_archive")]
    if main_missing:
        print(f"[warn] {len(main_missing)} VX contracts were never downloaded "
              f"(marked *.missing in {data_dir / 'vx'}): {', '.join(main_missing)}")
    if arch_missing:
        print(f"[info] no Cboe archive copy for {len(arch_missing)} 2013-2014 contracts: {', '.join(arch_missing)}")

    df = legs.copy()
    df["vix"] = vix.reindex(idx).ffill()
    rf_d = rf_ann.reindex(idx, method="ffill").shift(1)
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(1)
    df["rf"] = (rf_d * days / 365.0).fillna(0)  # accrue over calendar gaps
    r_eq = eq_px.reindex(idx).pct_change()
    df["r_es_ex"] = r_eq if eq_is_excess else r_eq - df["rf"]
    df["es_roll"] = es_roll_days(idx)
    df = df[df.index >= start].dropna(subset=["r_es_ex"])
    df = df.loc[df["D"].first_valid_index():]  # begin at the first complete curve

    # Days without a complete UX1..UX7 curve: report them, then hold the last signal
    # through them instead of dropping the days (dropping them also discarded that
    # period's market returns and showed up as straight lines on the chart).
    df["signal_stale"] = df["D"].isna()
    stale = df["signal_stale"]
    if stale.any():
        run_id = (stale != stale.shift()).cumsum()[stale]
        runs = (df[stale].assign(run=run_id).groupby("run")
                .agg(first=("D", lambda s: s.index.min().date()), last=("D", lambda s: s.index.max().date()),
                     days=("D", "size"), missing=("curve_issue", lambda s: s.value_counts().index[0])))
        runs.to_csv(out_dir / "signal_gaps.csv", index=False)
        print(f"[warn] {int(stale.sum())} days without a complete UX1-UX7 curve; last signal held "
              f"(details: signal_gaps.csv in the results folder)")
        print(runs.sort_values("days", ascending=False).head(10).to_string(index=False))
    df[["D", "m"]] = df[["D", "m"]].ffill()
    print(f"Backtest window: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")

    print("\nReturn by dislocation bucket (mean daily %, ES total return, VA = -0.5 UX1 + UX3):")
    bt = bucket_table(df[~df["signal_stale"]])
    print(bt.to_string())
    bt.to_csv(out_dir / "bucket_stats.csv")

    results, rets, weights = {}, {}, {}
    bench = df["r_es_ex"] + df["rf"]
    results["S&P 500 (TR)"] = perf_stats(bench, df["rf"])
    rets["S&P 500 (TR)"] = bench
    for name, cfg in PROGRAMS.items():
        r, pe, pv, to = run_program(df, cfg, args)
        results[name] = perf_stats(r, df["rf"], to)
        rets[name] = r
        weights[f"{name} ES"], weights[f"{name} VA"] = pe, pv

    summary = pd.DataFrame(results).T
    fmt = summary.copy()
    for col in ["Ann. Return", "Ann. Vol", "Max DD", "Avg daily turnover"]:
        fmt[col] = (summary[col] * 100).map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    for col in ["Sharpe", "Sortino"]:
        fmt[col] = summary[col].map(lambda x: f"{x:.2f}")
    print(f"\nPerformance (mc={args.mc}, Vc={args.vc}, cost={args.cost_bps}bps, "
          f"cash={'off' if args.no_cash else 'on'}):")
    print(fmt.to_string())
    summary.to_csv(out_dir / "summary.csv")
    pd.DataFrame(rets).to_csv(out_dir / "daily_returns.csv")
    pd.DataFrame(weights).join(df[["D", "m", "vix"]]).to_csv(out_dir / "weights_and_signal.csv")

    if args.sweep:
        print("\nThreshold sweep (Sharpe):")
        for name in ["Giant Squid", "Jumbo Squid", "Colossal Squid"]:
            grid = {}
            for mc in [-0.02, -0.05, -0.10, -0.15, -0.20]:
                for vc in [20, 25, 30, 35]:
                    a = argparse.Namespace(**{**vars(args), "mc": mc, "vc": vc})
                    r, *_ = run_program(df, PROGRAMS[name], a)
                    grid[(mc, vc)] = perf_stats(r, df["rf"])["Sharpe"]
            tbl = pd.Series(grid).unstack().round(2)
            tbl.index.name, tbl.columns.name = "mc", "Vc"
            print(f"\n{name}\n{tbl.to_string()}")
            tbl.to_csv(out_dir / f"sweep_{name.replace(' ', '_')}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eq = (1 + pd.DataFrame(rets)).cumprod()
        fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2]})
        eq.plot(ax=ax[0], logy=True, lw=1)
        ax[0].set_title("Squid Programs - growth of $1 (log scale)")
        ax[0].grid(alpha=0.3)
        (eq / eq.cummax() - 1).plot(ax=ax[1], lw=0.8, legend=False)
        ax[1].set_title("Drawdowns")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / "equity_and_drawdowns.png"
        fig_path.unlink(missing_ok=True) # delete picture from prev. run if it exsists. 
        fig.savefig(fig_path, dpi=130)
        print(f"\nSaved charts and CSVs to {out_dir.resolve()}")
    except ImportError:
        print("matplotlib not installed; skipping charts")


if __name__ == "__main__":
    main()
