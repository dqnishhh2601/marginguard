"""
Tests for the decision-support maths (insights.py) and market data handling (market.py).
No network and no API key needed: market data comes from small hand-made series.

Run:  python -m pytest tests/   (or just: python tests/test_insights.py)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import insights as I  # noqa: E402
import market as M    # noqa: E402


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def monthly(values, start="2026-01-01"):
    return pd.DataFrame({"date": pd.date_range(start, periods=len(values), freq="MS"),
                         "value": [float(v) for v in values]})


def meta(sid, kind="commodity", mode="snapshot"):
    return {"id": sid, "short": sid.lower(), "title": sid, "unit": "US$", "kind": kind, "source": "test",
            "url": M.page_url(sid), "mode": mode, "retrieved": "2026-10-10", "last_obs": None, "error": None}


def supplier(prices, dates, unit="L"):
    return pd.DataFrame({"date": pd.to_datetime(dates), "price_per_base": prices, "base_unit": unit})


# --------------------------------------------------------------------------- market.py
def test_parse_fred_csv_both_headers_and_missing():
    new = "observation_date,X\n2026-01-01,10\n2026-02-01,.\n2026-03-01,12.5\n"
    old = "DATE,X\n2026-01-01,10\n2026-03-01,12.5\n"
    for text in (new, old):
        df = M.parse_fred_csv(text, "X")
        assert list(df["value"]) == [10.0, 12.5]


def test_change_between_and_truncation():
    df = monthly([100, 110, 121])                       # Jan, Feb, Mar
    ch = M.change_between(df, "2026-01-15", "2026-02-20")
    assert approx(ch["pct"], 0.10) and not ch["truncated"]
    ch = M.change_between(df, "2026-01-15", "2026-06-01")  # asks past the data: clipped to March
    assert approx(ch["pct"], 0.21) and ch["truncated"] and ch["end_month"] == "2026-03-01"
    assert M.change_between(df, "2026-03-01", "2026-03-30") is None   # same month: no change to measure


def test_get_series_falls_back_to_snapshot_and_never_invents():
    df, m = M.get_series("PWHEAMTUSDM", allow_live=False)
    assert m["mode"] == "snapshot" and m["retrieved"] == M.SNAPSHOT_DATE and len(df) > 12
    df, m = M.get_series("NOT_A_SERIES", allow_live=False)
    assert df is None and m["mode"] == "missing"


# --------------------------------------------------------------------------- classification
def test_classify():
    assert I.classify("Cooking oil") == "Edible oils"
    assert I.classify("Plain flour") == "Wheat & flour"
    assert I.classify("Jasmine rice") == "Rice"
    assert I.classify("Unsalted butter") == "Dairy"
    assert I.classify("Eggs, large") == "Eggs"
    assert I.classify("Brown onions") == "Fresh produce"
    assert I.classify("Takeaway containers 750ml") == "Packaging"
    assert I.classify("Mystery item", "Dairy & eggs") == "Dairy"
    assert I.classify("Mystery item") == "Other"


# --------------------------------------------------------------------------- benchmarking
def _market(commodity_values, fx_values=None):
    mk = {"COMM": (monthly(commodity_values), meta("COMM")),
          "POILBREUSDM": (monthly([80, 80, 80, 80]), meta("POILBREUSDM", "energy"))}
    mk["EXSIUS"] = (monthly(fx_values or [1.30, 1.30, 1.30, 1.30]), meta("EXSIUS", "fx"))
    return mk


def _bench(sup, values, fx=None, currency="SGD"):
    I.PROFILES["_test"] = {"group": I.FOOD, "primary": ["COMM"], "comparability": "Medium", "why": "test"}
    try:
        return I.benchmark(sup, "_test", _market(values, fx), currency)
    finally:
        del I.PROFILES["_test"]


def test_benchmark_exceeds_consistent_below():
    # market series runs Jan-Apr, so Jan-Mar is fully covered: matched period == the whole run
    sup = supplier([10.0, 12.0], ["2026-01-10", "2026-03-10"])          # +20%
    assert _bench(sup, [100, 100, 105, 105])["verdict"] == "exceeds"    # market +5% -> 15pt gap
    assert _bench(sup, [100, 110, 118, 118])["verdict"] == "consistent"  # market +18% -> 2pt gap
    b = _bench(sup, [100, 120, 140, 140])                               # market +40%
    assert b["verdict"] == "below" and approx(I.catch_up(b), 1.40 / 1.20 - 1)
    assert approx(b["gap_pp"], 0.20 - 0.40) and not b["unverified"]


def test_benchmark_converts_usd_to_sgd():
    sup = supplier([10.0, 11.0], ["2026-01-10", "2026-03-10"])          # +10%
    b = _bench(sup, [100, 100, 110, 110], fx=[1.30, 1.30, 1.43, 1.43])  # +10% in US$, US$ +10% vs S$
    assert approx(b["benchmark_pct"], 1.1 * 1.1 - 1)                    # +21% in S$
    b_usd = _bench(sup, [100, 100, 110, 110], fx=[1.30, 1.30, 1.43, 1.43], currency="USD")
    assert approx(b_usd["benchmark_pct"], 0.10)


def test_benchmark_judges_the_matched_period_not_the_whole_run():
    """Market data ends in February but the supplier raised again in April.

    The verdict must compare like with like (Jan–Feb), and report the April move
    separately instead of scoring it against market data that doesn't cover it.
    """
    sup = supplier([10.0, 10.5, 12.0], ["2026-01-10", "2026-02-10", "2026-04-10"])
    b = _bench(sup, [100, 105])                       # market +5% Jan–Feb
    assert approx(b["matched_pct"], 0.05) and approx(b["benchmark_pct"], 0.05)
    assert b["verdict"] == "consistent"               # NOT "exceeds": +20% vs +5% would be unfair
    assert approx(b["supplier_pct"], 0.20) and approx(b["later_pct"], 12.0 / 10.5 - 1)
    assert b["unverified"] and "cannot be compared with the market yet" in b["unverified_text"]
    assert approx(b["unverified_per_unit"], 1.5) and approx(b["above_market_per_unit"], 0.0)
    assert any("two months behind" in x for x in b["limitations"])


def test_benchmark_above_market_only_counts_the_matched_excess():
    sup = supplier([10.0, 13.0, 15.0], ["2026-01-10", "2026-02-10", "2026-04-10"])
    b = _bench(sup, [100, 105])                       # matched +30% vs market +5%
    assert b["verdict"] == "exceeds" and approx(b["gap_pp"], 0.25)
    assert approx(b["supported_price"], 10.5)         # what the market move alone would justify
    assert approx(b["above_market_per_unit"], 13.0 - 10.5)
    assert approx(b["unverified_per_unit"], 15.0 - 13.0)


def test_benchmark_is_labelled_a_market_proxy():
    sup = supplier([10.0, 12.0], ["2026-01-10", "2026-03-10"])
    b = _bench(sup, [100, 100, 105, 105])
    assert b["match_type"] == "proxy"
    assert any("not a quote for the product you buy" in x for x in b["limitations"])


def test_no_benchmark_profile_and_single_invoice():
    sup = supplier([10.0, 12.0], ["2026-01-10", "2026-03-10"])
    assert I.benchmark(sup, "Dairy", _market([100, 100, 100]))["verdict"] == "no_benchmark"
    assert I.benchmark(supplier([10.0], ["2026-01-10"]), "Dairy", {})["verdict"] == "insufficient"


def test_missing_market_data_is_reported_not_invented():
    sup = supplier([10.0, 12.0], ["2026-01-10", "2026-03-10"])
    mk = {"PWHEAMTUSDM": (None, meta("PWHEAMTUSDM", mode="missing"))}
    b = I.benchmark(sup, "Wheat & flour", mk)
    assert b["verdict"] == "no_benchmark" and b["benchmark_pct"] is None and not b["evidence"]


# --------------------------------------------------------------------------- forecasting
def test_forecast_flat_series_and_trend_series():
    flat = I.forecast_series(monthly([100] * 24))
    assert flat["method"] == "naive" and all(approx(r["pct"], 0.0) for r in flat["rows"])
    line = I.forecast_series(monthly([100 + 2 * i for i in range(24)]))
    assert line["method"] == "trend"
    r6 = next(r for r in line["rows"] if r["h"] == 6)
    assert approx(r6["point"], 146 + 12, tol=1e-6)        # last value 146, +2 per month
    assert all(r["low"] <= r["point"] <= r["high"] for r in line["rows"])


def test_forecast_short_history_is_labelled_not_forecast():
    fc = I.forecast_series(monthly([100, 101, 99]))
    assert fc["rows"][0]["low"] is None and "not a forecast" in fc["note"]
    assert I.forecast_series(None) is None


def test_scenario_horizon_changes_horizon_dependent_outputs():
    """Horizon must move the trend scenario and the cumulative money, even when a flat
    market projection leaves the projected price itself unchanged."""
    sup = supplier([10.0, 12.1], ["2026-01-01", "2026-03-02"])      # ~2 months, ~+10%/month
    sets = {h: I.scenario_set(sup, h, forecast_pct=0.0, pass_through=0.5) for h in (1, 3, 6)}
    trend = {h: next(r["pct"] for r in v["rows"] if r["scenario"] == "Recent trend continues")
             for h, v in sets.items()}
    assert trend[1] < trend[3] < trend[6]                            # compounds with the horizon
    assert all(v["horizon"] == h for h, v in sets.items())
    # a flat market projection is legitimately the same at every horizon...
    flat = {h: next(r["pct"] for r in v["rows"] if r["scenario"].startswith("Market moves"))
            for h, v in sets.items()}
    assert flat[1] == flat[3] == flat[6] == 0.0
    # ...but the money it costs is not
    cums = [I.cumulative_impact(2000, 0.06, h)["cumulative"] for h in (1, 3, 6)]
    assert cums == [120.0, 360.0, 720.0]


def test_commodity_shock_feeds_through_at_the_pass_through_rate():
    sup = supplier([10.0], ["2026-03-02"])
    def market_row(shock, pt):
        v = I.scenario_set(sup, 3, forecast_pct=0.0, commodity_shock=shock, pass_through=pt)
        return next(r["pct"] for r in v["rows"] if r["scenario"].startswith("Market moves"))
    assert approx(market_row(0.10, 0.6), 0.06)      # +10% commodity x 60% pass-through
    assert approx(market_row(0.10, 1.0), 0.10)
    assert approx(market_row(0.10, 0.0), 0.0)
    assert approx(market_row(0.0, 0.6), 0.0)        # no shock, no effect
    assert market_row(-0.10, 0.6) < 0               # a fall feeds through too


def test_shocks_compound_with_the_model_projection_not_replace_it():
    sup = supplier([10.0], ["2026-03-02"])
    v = I.scenario_set(sup, 3, forecast_pct=0.05, commodity_shock=0.10, fx_shock=0.0, pass_through=1.0)
    assert approx(v["market_pct"], 1.05 * 1.10 - 1)
    assert v["market_parts"]["forecast"] == 0.05 and v["market_parts"]["commodity"] == 0.10


def test_fx_shock_only_applies_to_items_with_usd_exposure():
    sup = supplier([10.0], ["2026-03-02"])
    exposed = I.scenario_set(sup, 3, forecast_pct=0.0, fx_shock=0.10, pass_through=1.0, fx_exposed=True)
    domestic = I.scenario_set(sup, 3, forecast_pct=0.0, fx_shock=0.10, pass_through=1.0, fx_exposed=False)
    assert approx(exposed["market_pct"], 0.10) and exposed["applicable"]["fx_shock"]
    assert approx(domestic["market_pct"], 0.0) and not domestic["applicable"]["fx_shock"]


def test_items_without_a_benchmark_mark_shocks_not_applicable():
    sup = supplier([10.0, 11.0], ["2026-01-01", "2026-03-02"])
    v = I.scenario_set(sup, 3, forecast_pct=None, commodity_shock=0.30, fx_shock=0.20,
                       pass_through=1.0, has_benchmark=False)
    assert not v["applicable"]["commodity_shock"] and not v["applicable"]["fx_shock"]
    assert v["market_pct"] is None
    assert not any(r["scenario"].startswith("Market moves") for r in v["rows"])
    # the supplier's own history still drives a scenario, so the page is not empty
    assert any(r["scenario"] == "Recent trend continues" for r in v["rows"])


def test_catch_up_scenario_responds_to_pass_through():
    sup = supplier([10.0], ["2026-03-02"])
    pcts = [next(r["pct"] for r in I.scenario_set(sup, 3, forecast_pct=0.0, pass_through=pt,
                                                  catch_up_pct=0.20)["rows"]
                 if r["scenario"].startswith("Supplier catches up")) for pt in (0.0, 0.5, 1.0)]
    assert [round(x, 4) for x in pcts] == [0.0, 0.10, 0.20]


def test_resetting_the_controls_restores_the_baseline():
    sup = supplier([10.0, 11.0], ["2026-01-01", "2026-03-02"])
    base = I.scenario_set(sup, 3, forecast_pct=0.0, pass_through=0.5)
    shocked = I.scenario_set(sup, 3, forecast_pct=0.0, commodity_shock=0.2, fx_shock=0.1, pass_through=0.5)
    back = I.scenario_set(sup, 3, forecast_pct=0.0, commodity_shock=0.0, fx_shock=0.0, pass_through=0.5)
    assert shocked["market_pct"] != base["market_pct"]
    assert back["market_pct"] == base["market_pct"]
    assert [r["pct"] for r in back["rows"]] == [r["pct"] for r in base["rows"]]


def test_scenarios_never_alter_the_underlying_history():
    sup = supplier([10.0, 11.0], ["2026-01-01", "2026-03-02"])
    before = sup.copy(deep=True)
    I.scenario_set(sup, 6, forecast_pct=0.05, commodity_shock=0.25, fx_shock=0.1, pass_through=0.8)
    assert sup.equals(before)
    df = monthly([100, 110, 120])
    snapshot = df.copy(deep=True)
    I.forecast_series(df)
    assert df.equals(snapshot)


def test_cumulative_impact_is_not_annualised():
    c = I.cumulative_impact(2000, 0.06, 1)
    assert approx(c["extra_monthly"], 120) and approx(c["cumulative"], 120)
    assert approx(c["annualised_if_sustained"], 1440)      # reported separately, never as the horizon cost
    assert approx(I.cumulative_impact(2000, 0.06, 6)["cumulative"], 720)
    assert approx(I.cumulative_impact(2000, 0.0, 6)["cumulative"], 0.0)


def test_forecast_horizons_are_honoured_so_the_chart_can_end_at_the_horizon():
    df = monthly([100 + 2 * i for i in range(24)])
    for h in (1, 3, 6):
        fc = I.forecast_series(df, horizons=tuple(range(1, h + 1)))
        assert [r["h"] for r in fc["rows"]] == list(range(1, h + 1))
        assert fc["rows"][-1]["date"] == df["date"].iloc[-1] + pd.DateOffset(months=h)


# --------------------------------------------------------------------------- money
def test_cost_impact():
    imp = I.cost_impact(monthly_qty=100, unit_price=20, pct_change=0.10)    # S$2,000/month, +10%
    assert approx(imp["extra_monthly"], 200) and approx(imp["extra_annual"], 2400)
    imp = I.cost_impact(100, 20, 0.10, monthly_revenue=10000, monthly_cogs=3000)
    assert approx(imp["gross_margin_before"], 0.70) and approx(imp["gross_margin_after"], 0.68)
    assert approx(imp["margin_change_pp"], -0.02) and approx(imp["price_rise_to_offset"], 0.02)


def test_quote_savings_like_for_like():
    # brief example: 0.30 vs 0.24 per unit, 10,000 units a month -> S$600 a month gross
    q = I.quote_savings(0.30, "item", pack_price=24.0, pack_size=100, pack_unit="each", monthly_base_qty=10000)
    assert q["ok"] and approx(q["alt_price"], 0.24) and approx(q["gross_monthly"], 600)
    q = I.quote_savings(0.30, "item", 24.0, 100, "each", 10000, extra_monthly_cost=100, one_off_cost=500)
    assert approx(q["net_monthly"], 500) and approx(q["net_annual"], 500 * 12 - 500) and approx(q["payback_months"], 1)
    q = I.quote_savings(4.0, "kg", 2.0, 500, "g", 10)                   # 500 g pack for $2 = $4/kg
    assert approx(q["alt_price"], 4.0)
    assert not I.quote_savings(4.0, "kg", 10.0, 5, "L", 10)["ok"]       # litres vs kg: refuse


# --------------------------------------------------------------------------- recommendations
class F:
    def __init__(self, id, type, item, supplier, pct, impact, detail="d", change_date="2026-06-01"):
        self.id, self.type, self.item, self.supplier = id, type, item, supplier
        self.pct_change, self.annual_impact, self.detail, self.change_date = pct, impact, detail, change_date


def test_recommendations_rules_and_no_double_counting():
    findings = [F("F1", "price_increase", "Oil", "A", 0.20, 1000), F("F2", "price_increase", "Flour", "A", 0.10, 500),
                F("F3", "supplier_gap", "Eggs", "B", 0.09, 250), F("F4", "price_increase", "Butter", "C", 0.12, 400)]
    bench = {("Oil", "A"): {"verdict": "exceeds", "supplier_pct": 0.20, "gap_pp": 0.15, "comparability": "Medium",
                            "verdict_text": "t", "drivers": [], "above_market_per_unit": 0.5, "unverified": False},
             ("Flour", "A"): {"verdict": "consistent", "verdict_text": "t", "unverified": False},
             ("Butter", "C"): {"verdict": "no_benchmark", "verdict_text": "t"}}
    stats = pd.DataFrame([{"item": "Oil", "supplier": "A", "monthly_spend": 500.0, "monthly_qty": 100.0}])
    quotes = [{"ok": True, "item": "Oil", "supplier": "Q", "net_annual": 600.0, "alt_price": 3.0,
               "landed_price": 3.1, "unit": "L", "diff_pct": -0.1, "landed_diff_pct": -0.08, "url": "https://x",
               "date": "2026-10-10", "match": "Same product", "moq": 0, "packs_per_month": 20}]
    recs = I.recommendations(findings, bench, stats, quotes=quotes)
    kinds = {(r["item"], r["kind"]) for r in recs}
    assert ("Oil", "negotiate") in kinds and ("Flour", "pass_on") in kinds and ("Butter", "quote") in kinds
    neg = next(r for r in recs if r["kind"] == "negotiate")
    assert approx(neg["impact"], 0.5 * 100 * 12)        # above-market excess x monthly quantity x 12
    total, best = I.potential_savings(recs)
    assert approx(best["Oil"], 600) and approx(best["Eggs"], 250)   # best option per item, not 600 + 600
    assert approx(total, 850)
    assert next(r for r in recs if r["kind"] == "pass_on")["saving"] == 0   # in line with market: not a saving
    assert next(r for r in recs if r["kind"] == "quote")["saving"] == 0     # no benchmark: not a saving yet
    assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))


def test_unverified_increase_is_flagged_for_review_but_not_claimed_as_saving():
    findings = [F("F1", "price_increase", "Oil", "A", 0.18, 786)]
    bench = {("Oil", "A"): {"verdict": "consistent", "verdict_text": "In line.", "unverified": True,
                            "unverified_text": "Since Jul 2026 your price moved a further +9.5%.",
                            "later_pct": 0.095, "above_market_per_unit": 0.0, "unverified_per_unit": 0.38,
                            "comparability": "Medium", "drivers": []}}
    stats = pd.DataFrame([{"item": "Oil", "supplier": "A", "monthly_spend": 400.0, "monthly_qty": 100.0}])
    r = I.recommendations(findings, bench, stats)[0]
    assert r["kind"] == "review" and approx(r["impact"], 0.38 * 100 * 12)
    assert r["saving"] == 0 and not r["counts_as_saving"]
    assert I.potential_savings([r]) == (0.0, {})
    assert approx(I.at_stake([r])[0], 456)


def test_quote_below_minimum_order_is_not_counted_as_a_saving():
    q = {"ok": True, "item": "Oil", "supplier": "Q", "net_annual": 600.0, "alt_price": 3.0, "landed_price": 3.1,
         "unit": "L", "diff_pct": -0.1, "landed_diff_pct": -0.08, "url": "", "date": "", "match": "Possible substitute",
         "moq": 50, "packs_per_month": 20}
    r = I.recommendations([], {}, pd.DataFrame(), quotes=[q])[0]
    assert r["saving"] == 0 and not r["counts_as_saving"] and r["confidence"] == "Low"
    assert "minimum order" in r["why"] and "substitute" in r["why"]


def test_on_sample_invoices_if_market_snapshot_present():
    """The demo story: oil in line to July with a later rise to question, flour in line, rice behind."""
    df_oil = supplier([3.70, 3.98, 4.36], ["2026-04-10", "2026-06-10", "2026-08-10"])
    mk = M.get_all(allow_live=False)
    oil = I.benchmark(df_oil, "Edible oils", mk)
    assert oil["verdict"] == "consistent" and oil["unverified"]
    assert approx(oil["unverified_per_unit"], 4.36 - 3.98, tol=1e-9)
    flour = supplier([1.60, 1.60, 1.7778], ["2026-04-10", "2026-06-10", "2026-07-10"], unit="kg")
    assert I.benchmark(flour, "Wheat & flour", mk)["verdict"] == "consistent"
    rice = supplier([1.68, 1.78], ["2026-04-10", "2026-07-10"], unit="kg")
    assert I.benchmark(rice, "Rice", mk)["verdict"] == "below"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
