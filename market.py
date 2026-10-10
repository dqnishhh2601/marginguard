"""
External market data: commodity prices, energy and exchange rates.

Source: FRED (Federal Reserve Bank of St. Louis), which republishes the IMF's
Primary Commodity Prices and the Federal Reserve's exchange rates. FRED's CSV
download needs no API key.

Retrieval is live-first, with labelled fallbacks:

  1. live       download from FRED now; the response is validated (layout, dates,
                values) before it is used, and saved to market_cache/ for later.
  2. cached     if the live download fails or is invalid: the last validated live
                download in market_cache/, else the copy shipped with the app in
                market_data/. Its original retrieval time is kept, never "now",
                and the reason for the fallback is recorded.
  3. unavailable if neither exists. Nothing is ever made up.

There is no synthetic market data: only the invoices in the demo are fictional.

Two different dates are tracked and must not be confused:
  - retrieved_at: when the data was downloaded from FRED;
  - last_obs:     the latest month the market price itself refers to. Commodity
                  prices are published with a lag, so this is normally 2-3 months
                  behind today even when the download is live.

No Streamlit in this file, so it can be tested and reused from the command line.
"""
import io
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SNAPSHOT_DIR = Path(__file__).parent / "market_data"     # shipped with the app
CACHE_DIR = Path(__file__).parent / "market_cache"       # last validated live download (written at run time)
SNAPSHOT_DATE = "2026-10-10"   # when the files in market_data/ were retrieved from FRED
MIN_OBS = 24                   # a valid response has at least two years of monthly data
MAX_LAG_MONTHS = 4             # latest observation older than this (vs today) is flagged as outdated
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_PAGE = "https://fred.stlouisfed.org/series/{sid}"

IMF = "IMF Primary Commodity Prices, via FRED"

# id -> description. 'kind' says what role the series can play in an explanation.
SERIES = {
    "PPOILUSDM":    {"short": "palm oil", "title": "Palm oil, global price", "unit": "US$ per tonne", "kind": "commodity", "source": IMF},
    "PSOILUSDM":    {"short": "soybean oil", "title": "Soybean oil, global price", "unit": "US$ per tonne", "kind": "commodity", "source": IMF},
    "PWHEAMTUSDM":  {"short": "wheat", "title": "Wheat, global price", "unit": "US$ per tonne", "kind": "commodity", "source": IMF},
    "PRICENPQUSDM": {"short": "Thai rice", "title": "Rice (Thailand), global price", "unit": "US$ per tonne", "kind": "commodity", "source": IMF},
    "PSUGAISAUSDM": {"short": "raw sugar", "title": "Sugar No. 11, world price", "unit": "US cents per lb", "kind": "commodity", "source": IMF},
    "PCOFFOTMUSDM": {"short": "Arabica coffee", "title": "Coffee (other mild Arabica), global price", "unit": "US cents per lb",
                     "kind": "commodity", "source": IMF},
    "PPOULTUSDM":   {"short": "poultry", "title": "Poultry, global price", "unit": "US cents per lb", "kind": "commodity", "source": IMF},
    "POILBREUSDM":  {"short": "Brent crude", "title": "Brent crude oil (energy & freight cost signal)", "unit": "US$ per barrel",
                     "kind": "energy", "source": IMF},
    "EXSIUS":       {"short": "SGD/USD", "title": "SGD per US dollar", "unit": "S$ per US$", "kind": "fx",
                     "source": "Federal Reserve G.5 exchange rates, via FRED"},
}


def page_url(sid):
    return FRED_PAGE.format(sid=sid)


def parse_fred_csv(text, sid=None):
    """FRED CSV -> DataFrame[date, value]. Tolerates the old ('DATE') and new
    ('observation_date') header and FRED's '.' for missing values."""
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("unexpected FRED CSV layout")
    out = pd.DataFrame({"date": pd.to_datetime(df.iloc[:, 0], errors="coerce"),
                        "value": pd.to_numeric(df.iloc[:, 1], errors="coerce")})
    out = out.dropna().sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no observations for {sid}")
    return out


class FetchError(RuntimeError):
    pass


def now_utc():
    return datetime.now(timezone.utc)


def validate(df, sid, today=None):
    """Reject responses that parse but can't be right. Returns the cleaned frame."""
    today = pd.Timestamp(today or now_utc().date())
    if df is None or len(df) < MIN_OBS:
        raise FetchError(f"invalid response: only {0 if df is None else len(df)} observations")
    if df["date"].duplicated().any():
        raise FetchError("invalid response: duplicate dates")
    if df["date"].max() > today:
        raise FetchError(f"invalid response: observation dated in the future ({df['date'].max():%Y-%m})")
    if (df["value"] <= 0).any():
        raise FetchError("invalid response: non-positive prices")
    return df


def fetch_live(sid, timeout=15):
    try:
        r = requests.get(FRED_CSV.format(sid=sid), timeout=timeout,
                         headers={"User-Agent": "MarginGuard/1.0 (hackathon project)"})
    except requests.Timeout:
        raise FetchError(f"FRED did not respond within {timeout}s")
    except requests.RequestException as e:
        raise FetchError(f"could not reach FRED ({type(e).__name__})")
    if r.status_code == 429:
        raise FetchError("FRED rate limit reached (HTTP 429); try again in a few minutes")
    if r.status_code != 200:
        raise FetchError(f"FRED returned HTTP {r.status_code}")
    if "<html" in r.text[:500].lower():
        raise FetchError("invalid response: FRED returned a web page, not data")
    try:
        df = parse_fred_csv(r.text, sid)
    except Exception as e:
        raise FetchError(f"invalid response: {e}")
    return validate(df, sid)


def save_cache(sid, df, retrieved_at):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        df.rename(columns={"date": "observation_date", "value": sid}).to_csv(
            CACHE_DIR / f"{sid}.csv", index=False, date_format="%Y-%m-%d")
        (CACHE_DIR / f"{sid}.json").write_text(json.dumps({"retrieved_at": retrieved_at}))
    except OSError:
        pass   # a read-only disk only means no local cache; the live data is still used


def load_cache(sid):
    """Last validated live download: (df, retrieved_at) or (None, None)."""
    p, m = CACHE_DIR / f"{sid}.csv", CACHE_DIR / f"{sid}.json"
    if not (p.exists() and m.exists()):
        return None, None
    try:
        return validate(parse_fred_csv(p.read_text(), sid), sid), json.loads(m.read_text())["retrieved_at"]
    except Exception:
        return None, None


def months_between(a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + b.month - a.month


def freshness(meta, today=None):
    """Plain-English freshness line, plus the 'outdated' flag. Uses observation dates, not download time."""
    if meta["status"] == "unavailable":
        return "No data could be loaded."
    today = today or now_utc().date()
    lag = months_between(meta["last_obs"], today)
    meta["age_months"], meta["outdated"] = lag, lag > MAX_LAG_MONTHS
    obs = f"latest published month {pd.Timestamp(meta['last_obs']):%b %Y}"
    if meta["status"] == "live":
        text = f"Downloaded live from FRED; {obs}."
        if not meta["outdated"]:
            text += " This is the newest data FRED has; commodity prices are published with a lag."
    else:
        text = f"Cached copy ({meta['cache_source']}); {obs}. Live download failed: {meta['fallback_reason']}."
    if meta["outdated"]:
        text += f" The latest month is {lag} months old, so treat comparisons as indicative only."
    return text


def load_snapshot(sid):
    p = SNAPSHOT_DIR / f"{sid}.csv"
    if not p.exists():
        return None
    return parse_fred_csv(p.read_text(), sid)


MODE = {"live": "live", "cached": "snapshot", "unavailable": "missing"}   # older name, kept for compatibility


def get_series(sid, allow_live=True, timeout=15):
    """Returns (df or None, meta).

    meta['status'] is 'live', 'cached' or 'unavailable'. Provenance fields: source, url, api_url,
    retrieved_at (when downloaded), checked_at (when this load ran), first_obs / last_obs (the
    observation months), currency, unit, cache_source and fallback_reason when not live,
    outdated and a plain-English 'freshness' line.
    """
    info = SERIES.get(sid, {"short": sid, "title": sid, "unit": "", "kind": "other", "source": "FRED"})
    checked = now_utc().strftime("%Y-%m-%d %H:%M UTC")
    meta = {"id": sid, **info, "url": page_url(sid), "api_url": FRED_CSV.format(sid=sid),
            "currency": "S$ per US$" if info["kind"] == "fx" else "US$",
            "status": "unavailable", "retrieved_at": None, "checked_at": checked, "first_obs": None,
            "last_obs": None, "n_obs": 0, "cache_source": None, "fallback_reason": None, "outdated": False,
            "age_months": None}
    df = None
    if allow_live:
        try:
            df = fetch_live(sid, timeout=timeout)
            meta.update(status="live", retrieved_at=checked)
            save_cache(sid, df, checked)
        except FetchError as e:
            meta["fallback_reason"] = str(e)[:200]
        except Exception as e:  # anything unexpected: still fall back, and say why
            meta["fallback_reason"] = f"{type(e).__name__}: {e}"[:200]
    else:
        meta["fallback_reason"] = "live download not attempted"
    if df is None:
        cdf, when = load_cache(sid)
        if cdf is not None and when > SNAPSHOT_DATE:
            df = cdf
            meta.update(status="cached", retrieved_at=when, cache_source=f"last successful download, {when}")
        else:
            try:
                df = load_snapshot(sid)
            except Exception as e:
                meta["fallback_reason"] = (meta["fallback_reason"] or "") + f"; saved copy unreadable: {e}"
                df = None
            if df is not None:
                meta.update(status="cached", retrieved_at=SNAPSHOT_DATE,
                            cache_source=f"copy shipped with the app, downloaded from FRED on {SNAPSHOT_DATE}")
    if df is not None:
        meta.update(first_obs=df["date"].min().date().isoformat(), last_obs=df["date"].max().date().isoformat(),
                    n_obs=len(df))
    meta["mode"] = MODE[meta["status"]]
    meta["retrieved"] = meta["retrieved_at"]            # older name, kept for compatibility
    meta["error"] = meta["fallback_reason"]
    meta["freshness"] = freshness(meta)
    return df, meta


def summary(mk):
    """One line about a set of loaded series, for the refresh result."""
    metas = [m for _, m in mk.values()]
    live = sum(m["status"] == "live" for m in metas)
    latest = max((m["last_obs"] for m in metas if m["last_obs"] and m["kind"] != "fx"), default=None)
    reasons = sorted({m["fallback_reason"] for m in metas if m["status"] != "live" and m["fallback_reason"]})
    text = f"{live} of {len(metas)} series downloaded live"
    if latest:
        text += f"; latest commodity month {pd.Timestamp(latest):%b %Y}"
    if reasons:
        text += ". Fallback reason: " + "; ".join(reasons)
    return text


def fingerprint(mk, ids=None):
    """Changes whenever the market data behind a calculation changes (new month, new values, new status)."""
    out = []
    for sid in sorted(ids or mk):
        df, m = mk.get(sid, (None, None))
        if m is None:
            continue
        last_val = round(float(df["value"].iloc[-1]), 6) if df is not None else None
        out.append(f"{sid}:{m['status']}:{m['last_obs']}:{last_val}")
    return "|".join(out)


_LAST_REFRESH = [0.0]


def refresh_allowed(min_interval=60):
    """Process-wide guard so the refresh button can't hammer FRED. Returns (allowed, seconds to wait)."""
    import time
    wait = int(_LAST_REFRESH[0] + min_interval - time.time())
    if wait > 0:
        return False, wait
    _LAST_REFRESH[0] = time.time()
    return True, 0


def get_all(ids=None, allow_live=True, timeout=15):
    """Load several series in parallel so one slow request can't stall the app."""
    from concurrent.futures import ThreadPoolExecutor
    ids = list(ids or SERIES)
    with ThreadPoolExecutor(max_workers=max(len(ids), 1)) as ex:
        results = list(ex.map(lambda s: get_series(s, allow_live, timeout), ids))
    return dict(zip(ids, results))


def month_start(d):
    d = pd.Timestamp(d)
    return pd.Timestamp(year=d.year, month=d.month, day=1)


def value_at_month(df, month):
    """Value for the given month (first of month); None if not available."""
    hit = df[df["date"] == month_start(month)]
    return float(hit["value"].iloc[0]) if len(hit) else None


def change_between(df, start, end):
    """% change from month(start) to month(end), clipped to the data available.

    Returns dict(start_month, end_month, start_value, end_value, pct) or None.
    If 'end' is later than the latest observation, the latest observation is used
    and 'truncated' is True, so the caller can say so.
    """
    if df is None or df.empty:
        return None
    s, e = month_start(start), month_start(end)
    last = df["date"].max()
    truncated = e > last
    e = min(e, last)
    if e <= s:
        return None
    v0, v1 = value_at_month(df, s), value_at_month(df, e)
    if v0 is None or v1 is None or v0 == 0:
        return None
    return {"start_month": s.date().isoformat(), "end_month": e.date().isoformat(),
            "start_value": v0, "end_value": v1, "pct": v1 / v0 - 1, "truncated": bool(truncated)}


if __name__ == "__main__":
    # Refresh the offline snapshot:  python market.py
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    for sid in SERIES:
        try:
            df = fetch_live(sid)
            df.rename(columns={"date": "observation_date", "value": sid}).to_csv(
                SNAPSHOT_DIR / f"{sid}.csv", index=False, date_format="%Y-%m-%d")
            print(f"OK   {sid}: {len(df)} rows, latest {df['date'].max():%Y-%m}")
        except Exception as e:
            print(f"FAIL {sid}: {e}")
    print(f"\nNow update SNAPSHOT_DATE in market.py to {date.today().isoformat()}")
