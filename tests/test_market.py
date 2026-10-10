"""
Tests for live-first market data retrieval, fallbacks and provenance (market.py).
FRED is replaced by a fake HTTP layer, so no network is needed.

Run:  python -m pytest tests/   (or just: python tests/test_market.py)
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import insights as I  # noqa: E402
import market as M    # noqa: E402


def csv(sid, months=36, end="2026-08-01", start_value=100.0, step=1.0):
    dates = pd.date_range(end=end, periods=months, freq="MS")
    rows = [f"{d:%Y-%m-%d},{start_value + i * step}" for i, d in enumerate(dates)]
    return f"observation_date,{sid}\n" + "\n".join(rows) + "\n"


class Resp:
    def __init__(self, text="", status=200):
        self.text, self.status_code = text, status


class FakeHTTP:
    """Stands in for requests.get. 'plan' maps series id -> Resp or an exception to raise."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, url, timeout=None, headers=None):
        sid = url.split("id=")[1]
        self.calls.append(sid)
        r = self.plan.get(sid, requests.ConnectionError("offline"))
        if isinstance(r, Exception):
            raise r
        return r


class Sandbox:
    """Point the cache and snapshot folders at temp dirs and swap in the fake HTTP layer."""

    def __init__(self, plan, snapshot=True):
        self.plan, self.snapshot = plan, snapshot

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.old = (M.CACHE_DIR, M.SNAPSHOT_DIR, M.requests.get)
        M.CACHE_DIR = t / "cache"
        M.SNAPSHOT_DIR = t / "snap"
        M.SNAPSHOT_DIR.mkdir()
        if self.snapshot:
            (M.SNAPSHOT_DIR / "PPOILUSDM.csv").write_text(csv("PPOILUSDM", end="2026-07-01"))
        self.http = FakeHTTP(self.plan)
        M.requests.get = self.http
        return self

    def __exit__(self, *a):
        M.CACHE_DIR, M.SNAPSHOT_DIR, M.requests.get = self.old
        self.tmp.cleanup()


SID = "PPOILUSDM"


# --------------------------------------------------------------------------- live
def test_live_success_is_validated_labelled_and_cached():
    with Sandbox({SID: Resp(csv(SID, end="2026-08-01"))}):
        df, m = M.get_series(SID)
        assert m["status"] == "live" and m["mode"] == "live" and m["fallback_reason"] is None
        assert m["last_obs"] == "2026-08-01" and m["first_obs"] == "2023-09-01" and m["n_obs"] == 36
        assert m["retrieved_at"] == m["checked_at"] and m["retrieved_at"].endswith("UTC")
        assert m["source"].startswith("IMF") and m["url"].endswith(SID) and m["currency"] == "US$"
        assert m["unit"] == "US$ per tonne"
        assert "Downloaded live" in m["freshness"] and "Aug 2026" in m["freshness"]
        assert (M.CACHE_DIR / f"{SID}.csv").exists()           # kept for a later failure


# --------------------------------------------------------------------------- fallbacks
def test_failure_uses_last_good_download_with_its_original_time():
    with Sandbox({SID: Resp(csv(SID, end="2026-08-01"))}) as sb:
        _, first = M.get_series(SID)
        sb.http.plan[SID] = requests.Timeout()
        df, m = M.get_series(SID)
        assert m["status"] == "cached" and m["mode"] == "snapshot"
        assert m["retrieved_at"] == first["retrieved_at"]        # not "now": never presented as fresh
        assert "last successful download" in m["cache_source"]
        assert "did not respond" in m["fallback_reason"]
        assert m["last_obs"] == "2026-08-01"                    # newer than the shipped copy, so preferred


def test_failure_without_local_cache_uses_the_shipped_copy():
    with Sandbox({SID: Resp("", status=503)}):
        df, m = M.get_series(SID)
        assert m["status"] == "cached" and m["retrieved_at"] == M.SNAPSHOT_DATE
        assert "shipped with the app" in m["cache_source"] and "HTTP 503" in m["fallback_reason"]
        assert m["last_obs"] == "2026-07-01"
        assert "Cached copy" in m["freshness"] and "Live download failed" in m["freshness"]


def test_no_live_and_no_cache_is_unavailable():
    with Sandbox({}, snapshot=False):
        df, m = M.get_series(SID)
        assert df is None and m["status"] == "unavailable" and m["mode"] == "missing"
        assert m["last_obs"] is None and m["fallback_reason"]


def test_rate_limit_is_reported():
    with Sandbox({SID: Resp("Too many", status=429)}):
        _, m = M.get_series(SID)
        assert m["status"] == "cached" and "rate limit" in m["fallback_reason"]


def test_invalid_responses_are_rejected_and_fall_back():
    bad = {
        "html page": Resp("<!DOCTYPE html><html><body>Error</body></html>"),
        "too short": Resp(csv(SID, months=5)),
        "future date": Resp(csv(SID, end="2031-01-01")),
        "negative": Resp(csv(SID, start_value=-50)),
        "garbage": Resp("hello"),
    }
    for name, r in bad.items():
        with Sandbox({SID: r}):
            _, m = M.get_series(SID)
            assert m["status"] == "cached", name
            assert "invalid response" in m["fallback_reason"], (name, m["fallback_reason"])
            assert not (M.CACHE_DIR / f"{SID}.csv").exists(), name   # bad data is never cached


def test_bad_cache_file_is_ignored():
    with Sandbox({}) as _:
        M.CACHE_DIR.mkdir()
        (M.CACHE_DIR / f"{SID}.csv").write_text("nonsense")
        (M.CACHE_DIR / f"{SID}.json").write_text('{"retrieved_at": "2099-01-01 00:00 UTC"}')
        _, m = M.get_series(SID)
        assert m["status"] == "cached" and "shipped with the app" in m["cache_source"]


# --------------------------------------------------------------------------- freshness
def test_outdated_observations_are_flagged_on_observation_date_not_download_time():
    m = {"status": "live", "last_obs": "2026-01-01"}
    text = M.freshness(m, today=pd.Timestamp("2026-10-10").date())
    assert m["outdated"] and m["age_months"] == 9 and "indicative only" in text
    m = {"status": "live", "last_obs": "2026-07-01"}
    M.freshness(m, today=pd.Timestamp("2026-10-10").date())
    assert not m["outdated"] and m["age_months"] == 3           # normal publication lag


def test_outdated_market_data_weakens_the_benchmark():
    g = pd.DataFrame({"date": pd.to_datetime(["2026-01-10", "2026-03-10"]), "price_per_base": [1.0, 1.05],
                      "base_unit": "L"})
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=15, freq="MS"), "value": 100.0})
    df.loc[df.index[-1], "value"] = 104.0
    fresh = {"id": "PPOILUSDM", "title": "Palm", "short": "palm", "kind": "commodity", "source": "t",
             "url": "", "mode": "live", "status": "live", "outdated": False, "last_obs": "2026-03-01"}
    mk = {"PPOILUSDM": (df, fresh), "PSOILUSDM": (df, {**fresh, "id": "PSOILUSDM"})}
    b = I.benchmark(g, "Edible oils", mk, "USD")
    assert "indicative only" not in b["verdict_text"]
    stale = {k: (d, {**m, "outdated": True}) for k, (d, m) in mk.items()}
    b2 = I.benchmark(g, "Edible oils", stale, "USD")
    assert b2["verdict"] == b["verdict"] and "indicative only" in b2["verdict_text"]
    assert any("out of date" in x for x in b2["limitations"])


# --------------------------------------------------------------------------- invalidation and recalculation
def test_fingerprint_changes_when_the_data_changes():
    with Sandbox({SID: Resp(csv(SID, end="2026-07-01"))}) as sb:
        mk1 = {SID: M.get_series(SID)}
        sb.http.plan[SID] = Resp(csv(SID, end="2026-08-01"))        # a new month is published
        mk2 = {SID: M.get_series(SID)}
        sb.http.plan[SID] = requests.ConnectionError()                # same data, but now from the cache
        mk3 = {SID: M.get_series(SID)}
    fps = {M.fingerprint(mk1), M.fingerprint(mk2), M.fingerprint(mk3)}
    assert len(fps) == 3


def test_new_market_month_recalculates_the_benchmark():
    g = pd.DataFrame({"date": pd.to_datetime(["2026-04-10", "2026-06-10", "2026-08-10"]),
                      "price_per_base": [1.00, 1.03, 1.10], "base_unit": "L"})
    meta = {"title": "Palm", "short": "palm", "kind": "commodity", "source": "t", "url": "", "mode": "live",
            "status": "live"}
    to_june = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-06-01", freq="MS"), "value": 100.0})
    to_june.loc[to_june.date >= "2026-05-01", "value"] = 103.0
    to_aug = pd.concat([to_june, pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-08-01"]),
                                               "value": [106.0, 110.0]})], ignore_index=True)
    b1 = I.benchmark(g, "Edible oils", {"PPOILUSDM": (to_june, meta), "PSOILUSDM": (to_june, meta)}, "USD")
    b2 = I.benchmark(g, "Edible oils", {"PPOILUSDM": (to_aug, meta), "PSOILUSDM": (to_aug, meta)}, "USD")
    # With data to June, the August rise can't be checked; once August is published it is compared.
    assert b1["unverified"] and b1["period"] == "Apr–Jun 2026"
    assert not b2["unverified"] and b2["period"] == "Apr–Aug 2026" and b2["verdict"] == "consistent"


def test_refresh_is_rate_limited():
    M._LAST_REFRESH[0] = 0.0
    assert M.refresh_allowed(60) == (True, 0)
    ok, wait = M.refresh_allowed(60)
    assert not ok and 0 < wait <= 60
    M._LAST_REFRESH[0] = 0.0


def test_summary_reports_status_and_reason():
    with Sandbox({SID: requests.Timeout()}):
        mk = {SID: M.get_series(SID)}
    s = M.summary(mk)
    assert s.startswith("0 of 1 series downloaded live") and "Jul 2026" in s and "did not respond" in s


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok ", name)
    print(f"{n} passed")
