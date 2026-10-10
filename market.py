"""
External market data: commodity prices, energy and exchange rates.

Source: FRED (Federal Reserve Bank of St. Louis), which republishes the IMF's
Primary Commodity Prices and the Federal Reserve's exchange rates. FRED's CSV
download needs no API key.

Every series is fetched live when possible. If the live fetch fails (no internet,
FRED down, rate-limited), we fall back to the snapshot in market_data/, and the
app says so. If neither exists, the series is reported as missing; nothing is
ever made up.

No Streamlit in this file, so it can be tested and reused from the command line.
"""
import io
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SNAPSHOT_DIR = Path(__file__).parent / "market_data"
SNAPSHOT_DATE = "2026-10-10"   # when the files in market_data/ were retrieved from FRED
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


def fetch_live(sid, timeout=8):
    r = requests.get(FRED_CSV.format(sid=sid), timeout=timeout,
                     headers={"User-Agent": "MarginGuard/1.0 (hackathon project)"})
    r.raise_for_status()
    return parse_fred_csv(r.text, sid)


def load_snapshot(sid):
    p = SNAPSHOT_DIR / f"{sid}.csv"
    if not p.exists():
        return None
    return parse_fred_csv(p.read_text(), sid)


def get_series(sid, allow_live=True, timeout=8):
    """Returns (df or None, meta). meta['mode'] is 'live', 'snapshot' or 'missing'."""
    info = SERIES.get(sid, {"short": sid, "title": sid, "unit": "", "kind": "other", "source": "FRED"})
    meta = {"id": sid, **info, "url": page_url(sid), "mode": "missing", "retrieved": None,
            "last_obs": None, "error": None}
    df = None
    if allow_live:
        try:
            df = fetch_live(sid, timeout=timeout)
            meta["mode"] = "live"
            meta["retrieved"] = datetime.now(timezone.utc).date().isoformat()
        except Exception as e:  # network, HTTP or parsing problem: fall back quietly but record why
            meta["error"] = f"{type(e).__name__}: {e}"[:200]
    if df is None:
        try:
            df = load_snapshot(sid)
        except Exception as e:
            meta["error"] = f"snapshot unreadable: {e}"
            df = None
        if df is not None:
            meta["mode"] = "snapshot"
            meta["retrieved"] = SNAPSHOT_DATE
    if df is not None:
        meta["last_obs"] = df["date"].max().date().isoformat()
    return df, meta


def get_all(ids=None, allow_live=True, timeout=6):
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
