"""
Tests for the AI Procurement Investigator (investigator.py).

The model is replaced by a scripted fake, so these run with no API key and no network,
and they check what the code guarantees whatever the model says:
tool choices are validated, the loop is bounded, fake citations are dropped, figures
that are not in the evidence are flagged, and money only ever comes from the tools.

Run:  python -m pytest tests/   (or just: python tests/test_investigator.py)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import investigator as I  # noqa: E402
import insights           # noqa: E402
import llm                # noqa: E402
import market as M        # noqa: E402

OIL, EGGS, NAPKINS = "Cooking oil", "Eggs, large", "Paper napkins"
A, B = "Alpha Foods", "Beta Supplies"


# --------------------------------------------------------------------------- fixtures
def monthly(values, start="2026-01-01"):
    return pd.DataFrame({"date": pd.date_range(start, periods=len(values), freq="MS"),
                         "value": [float(v) for v in values]})


def meta(sid, title):
    return {"id": sid, "short": sid.lower(), "title": title, "unit": "US$", "kind": "commodity",
            "source": "test", "url": M.page_url(sid), "mode": "snapshot", "retrieved": "2026-10-10",
            "last_obs": "2026-07-01", "error": None}


def market():
    # Jan..Jul 2026. Oil prices +3% from April to July; FX flat; Brent down.
    return {
        "PPOILUSDM": (monthly([100, 100, 100, 100, 101, 102, 103]), meta("PPOILUSDM", "Palm oil")),
        "PSOILUSDM": (monthly([100, 100, 100, 100, 101, 102, 103]), meta("PSOILUSDM", "Soybean oil")),
        "EXSIUS": (monthly([1.30] * 7), meta("EXSIUS", "SGD per US dollar")),
        "POILBREUSDM": (monthly([80, 80, 80, 80, 78, 76, 74]), meta("POILBREUSDM", "Brent crude")),
    }


def _line(item, sup, d, inv, price, qty, unit="L", pack=5.0, desc=None, cat="Dry goods"):
    return {"canonical_name": item, "supplier": sup, "date": pd.Timestamp(d), "invoice_number": inv,
            "description": desc or f"{item} {pack:g}{unit}", "price_per_base": price, "base_unit": unit,
            "base_qty": qty, "line_total": price * qty, "pack_size": pack, "pack_unit": unit,
            "unit_price": price * pack, "category": cat}


def dataset():
    rows = []
    # Oil: in line with the market to July, then a further jump the market data can't cover yet.
    for i, (d, p) in enumerate([("2026-04-10", 3.70), ("2026-05-10", 3.70), ("2026-06-10", 3.80),
                                 ("2026-07-10", 3.81), ("2026-08-10", 4.20), ("2026-09-10", 4.20)]):
        rows.append(_line(OIL, A, d, f"A-{100 + i}", p, 100))
    # Eggs from two suppliers: B is cheaper for the same product.
    for i, d in enumerate(["2026-04-12", "2026-06-12", "2026-09-12"]):
        rows.append(_line(EGGS, A, d, f"A-E{i}", 0.30 if i < 2 else 0.33, 800, unit="item", pack=30,
                          desc="Eggs large 30s", cat="Dairy & eggs"))
        rows.append(_line(EGGS, B, d, f"B-E{i}", 0.30, 300, unit="item", pack=30, desc="Large eggs tray 30",
                          cat="Dairy & eggs"))
    # Napkins: no market benchmark exists.
    for i, (d, p) in enumerate([("2026-04-15", 0.02), ("2026-09-15", 0.025)]):
        rows.append(_line(NAPKINS, A, d, f"A-N{i}", p, 5000, unit="item", pack=500, cat="Packaging"))
    lines = pd.DataFrame(rows)
    series = lines[["date", "canonical_name", "supplier", "price_per_base", "base_unit", "unit_price",
                    "pack_size", "pack_unit", "description"]].copy()
    return series, lines


def case(item=OIL, supplier=A, quotes=(), mk=None, profile=None):
    series, lines = dataset()
    return I.Case(item=item, supplier=supplier, series=series, lines=lines, stats=insights.item_stats(lines),
                  finding=None, market=market() if mk is None else mk,
                  profile=profile or insights.classify(item, lines[lines.canonical_name == item].category.iloc[0]),
                  currency="SGD", quotes=list(quotes))


class Script:
    """A fake model: plan replies in order, then the report. Records every prompt it was sent."""

    def __init__(self, plan, report=None, compare=None):
        self.plan, self.report, self.compare = list(plan), report or {}, compare or {}
        self.prompts = []

    def __call__(self, messages, max_tokens=0):
        self.prompts.append(messages)
        system = messages[0]["content"]
        if system == I.COMPARE_SYSTEM:
            return self.compare
        if system == I.REPORT_SYSTEM:
            return self.report
        return self.plan.pop(0) if self.plan else {"tool": "finish"}


def go(tool, **args):
    return {"tool": tool, "args": args, "reason": "test"}


def ev(inv, title):
    return next(e for e in inv.evidence if e["title"] == title)


def statuses(inv):
    return [(t["tool"], t["status"]) for t in inv.trace]


QUOTE = {"item": OIL, "vs_supplier": A, "supplier": "Gamma Oils", "spec": "Canola oil 20L, food grade",
         "match": "Possible substitute", "pack_price": 70.0, "pack_size": 20.0, "pack_unit": "L", "moq": 2,
         "delivery": 0.0, "one_off": 0.0, "costs_known": True, "url": "", "date": "2026-10-01",
         "source": "Quote entered by you"}


# --------------------------------------------------------------------------- the workflow
def test_happy_path_market_check_and_cited_report():
    fake = Script([go("market_benchmark"), go("market_context"), go("supplier_options"), go("finish")],
                  report={"status": "review_needed",
                          "assessment": "Your price rose 13.5% [E1]. Over the matched months it moved with the "
                                        "market [E4].",
                          "verified_facts": [{"text": "Price up 13.5%.", "evidence": ["E1"]}],
                          "possible_explanations": [{"text": "Fuel costs", "evidence": ["E5"]}],
                          "missing_evidence": ["The supplier's cost breakdown"],
                          "actions": [{"type": "request_breakdown", "title": "Ask why", "why": "Unexplained part",
                                       "evidence": ["E4"], "option_id": None}]})
    inv = I.investigate(case(), llm_fn=fake)
    assert [t for t, _ in statuses(inv)] == ["price_history", "market_benchmark", "market_context",
                                             "supplier_options", "finish"]
    assert inv.status == "review_needed" and inv.verified_facts[0]["evidence"] == ["E1"]
    mc = ev(inv, "Matched-period comparison")
    assert mc["content"]["verdict"] == "consistent" and mc["content"]["unverified"]
    # Money in the action comes from the matched comparison, not from the model.
    act = inv.actions[0]
    assert act["impact"] == mc["content"]["at_stake_annual"] and act["impact"] > 0
    assert not inv.validation["dropped_citations"] and not inv.validation["unsupported_figures"]
    assert inv.llm_calls == {"live": 5, "cached": 0}


def test_price_history_always_runs_first_and_is_cited():
    inv = I.investigate(case(), llm_fn=Script([go("finish")]))
    first = inv.evidence[0]
    assert first["tool"] == "price_history" and first["provenance"] == "invoice"
    assert "A-100" in first["summary"] and "A-105" in first["summary"]     # cites the actual invoices


def test_missing_invoice_data_stops_without_calling_the_model():
    fake = Script([go("market_benchmark")])
    inv = I.investigate(case(item="Truffles", profile="Other"), llm_fn=fake)
    assert inv.status == "insufficient_evidence" and fake.prompts == [] and inv.actions == []


def test_market_data_unavailable_is_reported_not_invented():
    fake = Script([go("market_benchmark"), go("finish")], report={"status": "insufficient_evidence"})
    inv = I.investigate(case(mk={}), llm_fn=fake)
    bench = [e for e in inv.evidence if e["tool"] == "market_benchmark"]
    assert bench and all(e["provenance"] in ("unavailable",) for e in bench)
    assert not any(e["title"] == "Matched-period comparison" for e in inv.evidence)


def test_no_benchmark_item_and_in_line_claim_is_a_conflict():
    fake = Script([go("market_benchmark"), go("finish")], report={"status": "in_line", "assessment": "Fine [E2]."})
    inv = I.investigate(case(item=NAPKINS), llm_fn=fake)
    assert ev(inv, "No market benchmark")["provenance"] == "unavailable"
    assert any("no market benchmark" in c for c in inv.validation["conflicts"])


def test_in_line_status_against_unverified_increase_is_a_conflict():
    fake = Script([go("market_benchmark"), go("finish")], report={"status": "in_line"})
    inv = I.investigate(case(), llm_fn=fake)
    assert any("flags part of the increase" in c for c in inv.validation["conflicts"])


def test_no_alternatives_says_so_and_invents_nothing():
    fake = Script([go("supplier_options"), go("finish")])
    inv = I.investigate(case(), llm_fn=fake)
    e = ev(inv, "No alternative offers on file")
    assert e["provenance"] == "unavailable" and "no listings were invented" in e["summary"]


def test_existing_supplier_is_comparable_by_rule_and_saving_is_deterministic():
    fake = Script([go("supplier_options"), go("assess_comparability", option_id="E2"),
                   go("calculate_savings", option_id="E2"), go("finish")],
                  report={"status": "act_now", "actions": [
                      {"type": "switch_supplier", "title": "Buy eggs from Beta", "why": "Cheaper",
                       "evidence": ["E2", "E4"], "option_id": "E2"}]})
    inv = I.investigate(case(item=EGGS), llm_fn=fake)
    comp = ev(inv, "Comparability of E2")["content"]
    assert comp["classification"] == "comparable" and comp["method"] == "deterministic"
    # The model was never asked to compare: the invoice match decides it.
    assert not any(p[0]["content"] == I.COMPARE_SYSTEM for p in fake.prompts)
    a = inv.actions[0]
    stats = insights.item_stats(dataset()[1])
    mine = stats[(stats.item == EGGS) & (stats.supplier == A)].iloc[0]
    expected = (0.33 - 0.30) * mine["monthly_qty"] * 12
    assert abs(a["impact"] - expected) < 0.01 and a["comparability"] == "comparable"


def test_unit_mismatch_is_not_comparable_whatever_the_model_says():
    q = {**QUOTE, "pack_unit": "kg", "spec": "Same oil, honest"}
    fake = Script([go("supplier_options"), go("assess_comparability", option_id="E2"),
                   go("calculate_savings", option_id="E2"), go("finish")],
                  compare={"classification": "comparable"},
                  report={"actions": [{"type": "switch_supplier", "title": "Switch", "why": "x",
                                       "evidence": ["E2"], "option_id": "E2"}]})
    inv = I.investigate(case(quotes=[q]), llm_fn=fake)
    assert ev(inv, "Comparability of E2")["content"]["classification"] == "not_comparable"
    a = inv.actions[0]
    assert a["impact"] is None and "no saving is claimed" in a["impact_label"]
    assert any("not comparable" in c for c in inv.validation["conflicts"])


def test_ai_comparability_is_validated_and_downgraded_without_a_spec():
    fake = Script([go("supplier_options"), go("assess_comparability", option_id="E2"), go("finish")],
                  compare={"classification": "definitely the same!!", "explanation": "x"})
    inv = I.investigate(case(quotes=[QUOTE]), llm_fn=fake)
    assert ev(inv, "Comparability of E2")["content"]["classification"] == "insufficient_information"

    fake = Script([go("supplier_options"), go("assess_comparability", option_id="E2"), go("finish")],
                  compare={"classification": "comparable", "explanation": "x"})
    inv = I.investigate(case(quotes=[{**QUOTE, "spec": ""}]), llm_fn=fake)
    c = ev(inv, "Comparability of E2")["content"]
    assert c["classification"] == "insufficient_information" and c["requires_manual_verification"]


def test_quote_saving_is_labelled_unconfirmed_until_the_product_matches():
    fake = Script([go("supplier_options"), go("assess_comparability", option_id="E2"),
                   go("calculate_savings", option_id="E2"), go("finish")],
                  compare={"classification": "potentially_comparable", "explanation": "Canola, not palm."},
                  report={"actions": [{"type": "get_quote", "title": "Check Gamma", "why": "x",
                                       "evidence": ["E2"], "option_id": "E2"}]})
    inv = I.investigate(case(quotes=[QUOTE]), llm_fn=fake)
    a = inv.actions[0]
    assert a["impact"] is not None and "product match not confirmed" in a["impact_label"]
    # Product text goes to the model as data inside JSON, never as instructions.
    cmp_prompt = next(p for p in fake.prompts if p[0]["content"] == I.COMPARE_SYSTEM)
    assert json.loads(cmp_prompt[1]["content"])["alternative_description"] == QUOTE["spec"]


def test_fictional_quote_is_labelled_synthetic():
    q = {**QUOTE, "source": "Fictional demo quote"}
    inv = I.investigate(case(quotes=[q]), llm_fn=Script([go("supplier_options"), go("finish")]))
    assert inv.evidence[1]["provenance"] == "synthetic"


# --------------------------------------------------------------------------- guardrails
def test_invalid_tool_missing_args_bad_option_and_repeats_are_rejected_with_feedback():
    fake = Script([go("search_the_internet"), go("calculate_savings"), go("calculate_savings", option_id="E1"),
                   go("market_context"), go("market_context"), go("price_history"), go("finish")])
    inv = I.investigate(case(), llm_fn=fake)
    got = statuses(inv)
    assert got[1] == ("search_the_internet", "rejected: unknown tool")
    assert got[2] == ("calculate_savings", "rejected: missing option_id")
    assert got[3][1].startswith("error:")                       # E1 is the price history, not an option
    assert got[4] == ("market_context", "ok") and got[5] == ("market_context", "rejected: repeat")
    assert got[6] == ("price_history", "rejected: unknown tool")
    # The model is told why, so it can correct itself.
    plan_prompts = [p[1]["content"] for p in fake.prompts if p[0]["content"] == I.PLAN_SYSTEM]
    assert "is not a tool" in plan_prompts[1] and "needs option_id" in plan_prompts[2]
    assert "already run" in plan_prompts[5]


def test_unknown_arguments_are_stripped():
    inv = I.investigate(case(), llm_fn=Script([{"tool": "market_context", "args": {"url": "http://x"}},
                                               go("finish")]))
    assert inv.trace[1]["args"] == {} and inv.trace[1]["status"] == "ok"


def test_step_cap_stops_a_model_that_never_finishes():
    fake = Script([go("search")] * 50)
    inv = I.investigate(case(), llm_fn=fake, max_steps=4)
    assert inv.trace[-1]["tool"] == "stop" and len(inv.trace) == 1 + 4 + 1
    assert len(fake.prompts) == 4 + 1                                # 4 plan calls + 1 report


def test_comparison_limit():
    quotes = [{**QUOTE, "supplier": f"S{i}"} for i in range(5)]
    plan = [go("supplier_options")] + [go("assess_comparability", option_id=f"E{i}") for i in range(2, 7)]
    fake = Script(plan, compare={"classification": "potentially_comparable"})
    inv = I.investigate(case(quotes=quotes), llm_fn=fake, max_steps=10)
    assert sum(1 for t in inv.trace if t["tool"] == "assess_comparability" and t["status"] == "ok") \
        == I.MAX_COMPARISONS
    assert any(t["status"] == "rejected: comparison limit" for t in inv.trace)


def test_garbage_model_output_does_not_crash():
    fake = Script([["not", "a", "dict"], {"tool": None, "args": "nope"}],
                  report={"status": "excellent", "verified_facts": "lots", "actions": [{"type": "launch_rockets"}]})
    inv = I.investigate(case(), llm_fn=fake, max_steps=2)
    assert inv.status == "insufficient_evidence" and inv.actions == [] and inv.verified_facts == []


# --------------------------------------------------------------------------- citation and figure checks
def test_fabricated_citations_are_dropped_and_uncited_facts_are_not_verified():
    fake = Script([go("finish")], report={
        "status": "review_needed", "assessment": "Up 13.5% [E1], says a report [E42].",
        "verified_facts": [{"text": "Supplier cited a drought.", "evidence": ["E9"]},
                           {"text": "Price up 13.5%.", "evidence": ["E1", "E77"]}]})
    inv = I.investigate(case(), llm_fn=fake)
    assert "E42" not in inv.assessment and "[E1]" in inv.assessment
    assert [f["text"] for f in inv.verified_facts] == ["Price up 13.5%."]
    assert inv.verified_facts[0]["evidence"] == ["E1"]
    assert set(inv.validation["dropped_citations"]) == {"E9", "E42", "E77"}
    assert inv.validation["downgraded_facts"] == ["Supplier cited a drought."]


def test_figures_not_in_the_evidence_are_flagged():
    fake = Script([go("market_benchmark"), go("finish")], report={
        "status": "review_needed",
        "assessment": "Your price rose 13.5% [E1] while palm oil rose 3.0% [E2] and freight jumped 40% [E2].",
        "verified_facts": [{"text": "Freight is up S$1,200 a year.", "evidence": ["E2"]}],
        "possible_explanations": [{"text": "A 25% tariff.", "evidence": []}]})
    inv = I.investigate(case(), llm_fn=fake)
    bad = inv.validation["unsupported_figures"]
    assert "40%" in bad and "S$1,200" in bad and "25%" in bad
    assert "13.5%" not in bad and "3.0%" not in bad
    assert inv.verified_facts == []                      # a "fact" with an unsupported figure is not verified
    assert inv.possible_explanations[0]["unsupported"] == ["25%"]


def test_unsupported_figures_helper():
    store = I.EvidenceStore()
    store.add(tool="t", title="x", source="s", provenance="invoice", observed="",
              summary="Went from S$3.700 to S$4.200, +13.5%. You buy 100.0 L.")
    assert I.unsupported_figures("+13.5% and S$4.20 and 100.0", store) == []
    assert I.unsupported_figures("+14% and S$5.00", store) == ["+14%", "S$5.00"]


# --------------------------------------------------------------------------- the real LLM client path
def test_live_and_replayed_calls_are_counted(tmp_path):
    """With no llm_fn, calls go through llm.chat_json; answers already on disk count as replayed."""
    old_dir, old_chat = llm.CACHE_DIR, llm.chat_json
    llm.CACHE_DIR = tmp_path
    try:
        c = case()
        plan_msgs = I._plan_messages(c, _store_after_history(c), ["price_history"], "")
        _, path = llm.cached(plan_msgs)
        path.write_text(json.dumps({"tool": "finish", "args": {}, "reason": "cached"}))

        def fake_chat(messages, max_tokens=0):
            hit, _ = llm.cached(messages)
            if hit is not None:
                return hit
            return {"status": "insufficient_evidence"}
        llm.chat_json = fake_chat
        inv = I.investigate(c)
        assert inv.llm_calls == {"live": 1, "cached": 1}
    finally:
        llm.CACHE_DIR, llm.chat_json = old_dir, old_chat


def _store_after_history(c):
    s = I.EvidenceStore()
    I.tool_price_history(c, s)
    return s


def test_model_unreachable_raises_llm_error():
    def down(messages, max_tokens=0):
        raise llm.LLMError("No LLM_API_KEY set.")
    try:
        I.investigate(case(), llm_fn=down)
    except llm.LLMError:
        return
    raise AssertionError("expected LLMError")


def test_prompts_contain_no_retrieval_timestamps():
    """Retrieval dates would change every day and break replay from the saved cache."""
    fake = Script([go("market_benchmark"), go("market_context"), go("finish")])
    I.investigate(case(), llm_fn=fake)
    assert not any("2026-10-10" in m["content"] for p in fake.prompts for m in p)


if __name__ == "__main__":
    import inspect
    import tempfile
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            n += 1
            print("ok ", name)
    print(f"{n} passed")
