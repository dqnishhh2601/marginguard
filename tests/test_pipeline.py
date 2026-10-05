"""
Offline end-to-end test with a fake LLM (no API key needed).

The fake LLM returns the known-correct answers from ground_truth.json, so this test
checks that everything AROUND the AI — validation, self-correction, matching merge,
price maths, shrinkflation and cross-supplier detection — works correctly.

Run:  python -m pytest tests/   (or just: python tests/test_pipeline.py)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import extract, normalize, recommend, pipeline  # noqa: E402

TRUTH = {t["invoice_number"]: t for t in json.loads((ROOT / "sample_invoices/ground_truth.json").read_text())}
CANON = {
    "Tomatoes, Roma": "Roma tomatoes", "Roma Tomato": "Roma tomatoes",
    "Romaine Lettuce": "Romaine lettuce", "Lettuce - Romaine (per kg)": "Romaine lettuce",
    "Onions Brown - 10kg bag": "Brown onions", "Eggs, large (tray of 30)": "Eggs, large",
    "EGGS LARGE TRAY/30": "Eggs, large", "Lemons": "Lemons", "Cooking Oil 5L tin": "Cooking oil",
    "Jasmine Rice 25kg": "Jasmine rice", "Plain Flour 10kg": "Plain flour", "Plain Flour 9kg": "Plain flour",
    "White Sugar 5kg": "White sugar", "Coffee Beans House Blend 1kg": "Coffee beans",
    "FRESH MILK 2L": "Fresh milk", "BUTTER UNSALTED 250G": "Unsalted butter",
    "CHICKEN THIGH BONELESS": "Chicken thigh", "CREAM CHEESE 1KG BLOCK": "Cream cheese",
}
calls = {"extract": 0, "retry": 0}


def fake_extract(messages, **kw):
    calls["extract"] += 1
    text = messages[1]["content"]
    num = re.search(r"(GL-\d+|HDG/2026/\d+|NDM-\d+)", text).group(1)
    t = json.loads(json.dumps(TRUTH[num]))
    if len(messages) > 2:
        calls["retry"] += 1
    elif num == "NDM-1002":  # simulate an LLM mistake on the first try
        t["line_items"][0]["line_total"] = 999.0
    return {k: t[k] for k in ("supplier_name", "invoice_number", "invoice_date", "subtotal", "line_items")}


def fake_normalize(messages, **kw):
    items = []
    for line in messages[1]["content"].splitlines()[1:]:
        m = re.match(r'(\d+)\. \[.*?\] "(.*)" \(pack', line)
        items.append({"id": int(m.group(1)), "canonical_name": CANON[m.group(2)], "category": "Other"})
    return {"items": items}


def fake_recommend(messages, **kw):
    ids = re.findall(r'"id": "(F\d+)"', messages[1]["content"])
    return {"headline": "test", "actions": [{"finding_id": i, "title": "t", "advice": "a"} for i in ids],
            "email": {"supplier": "x", "subject": "s", "body": "b"}}


def test_end_to_end():
    extract.chat_json, normalize.chat_json, recommend.chat_json = fake_extract, fake_normalize, fake_recommend
    files = sorted((ROOT / "sample_invoices").glob("*.pdf"))
    res = pipeline.run([(p.name, p) for p in files])
    s, F = res["summary"], {(f.item, f.type): f for f in res["findings"]}

    assert s["invoices"] == 24 and s["suppliers"] == 3 and s["items"] == 14, s
    assert calls["retry"] == 1, "self-correction should have fixed the planted mistake"
    assert not res["warnings"] and not res["errors"], res["warnings"] + res["errors"]

    expected = {  # item: (type, pct change)
        "Cooking oil": ("price_increase", 0.178), "Unsalted butter": ("price_increase", 0.118),
        "Plain flour": ("shrinkflation", 0.111), "Brown onions": ("price_increase", 0.10),
        "Jasmine rice": ("price_increase", 0.0595), "Fresh milk": ("price_increase", 0.0446),
        "Eggs, large": ("supplier_gap", 0.092),
    }
    for item, (typ, pct) in expected.items():
        f = F.get((item, typ))
        assert f, f"missing {typ} for {item}"
        assert abs(f.pct_change - pct) < 0.002, (item, f.pct_change)
    flagged = {f.item for f in res["findings"]}
    for stable in ["Roma tomatoes", "Romaine lettuce", "Lemons", "White sugar", "Coffee beans",
                   "Chicken thigh", "Cream cheese"]:
        assert stable not in flagged, f"{stable} should not be flagged"
    assert len(res["findings"]) == 7

    # Hand-checked: oil $3.70/L -> $4.36/L at ~100 L/month -> ~$792/yr (within 2%: months aren't all equal)
    assert abs(F[("Cooking oil", "price_increase")].annual_impact / 792 - 1) < 0.02
    assert abs(F[("Unsalted butter", "price_increase")].annual_impact / 384 - 1) < 0.02
    print("PASS", json.dumps(s, indent=1))
    for f in res["findings"]:
        print(f"  {f.id} {f.type:15} {f.item:16} {f.pct_change:+.1%} ${f.annual_impact:,.0f}/yr  since {f.change_date}")


if __name__ == "__main__":
    test_end_to_end()
