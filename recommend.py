"""
Step 4 — Recommendations: the LLM turns the computed findings into a plain-English
action plan and drafts a negotiation email to the supplier.

The LLM only explains and advises. Every dollar figure shown to the user comes from
analyze.py, not from the model, so the AI can't invent numbers.
"""
import json

from analyze import fmt_unit_price
from llm import LLMError, chat_json

PROMPT_VERSION = "recommend-v1"

SYSTEM = """You advise the owner of a small food business (café/restaurant) on supplier costs.
You get findings that were already calculated by software. Do NOT do any maths and do NOT
invent numbers; refer to findings by id. Be specific, practical and brief. Plain English,
no jargon. The owner is busy and has little negotiating power, so suggest realistic moves:
asking for the old price, requesting a quote from the other supplier, buying a different
pack size, changing order size, or adjusting a menu price.

Reply with ONE JSON object only:
{
  "headline": string,                      // one sentence, the single most important takeaway
  "actions": [                             // one per finding id you are given, most urgent first
    {"finding_id": string, "title": string (max 8 words), "advice": string (1-2 sentences)}
  ],
  "email": {                               // to the supplier with the biggest total increase
    "supplier": string,
    "subject": string,
    "body": string                         // polite, firm, under 150 words, signed "[Your name], [Your business]"
  }
}"""

TYPE_LABEL = {"price_increase": "price increase", "shrinkflation": "hidden increase (smaller pack, same price)",
              "supplier_gap": "cheaper at another supplier", "price_decrease": "price decrease"}


def _findings_for_prompt(findings):
    out = []
    for f in findings:
        out.append({
            "id": f.id, "type": TYPE_LABEL[f.type], "item": f.item, "supplier": f.supplier,
            "was" if f.type != "supplier_gap" else "other_supplier_price": fmt_unit_price(f.baseline_price, f.base_unit),
            "now" if f.type != "supplier_gap" else "this_supplier_price": fmt_unit_price(f.current_price, f.base_unit),
            "change": f"{f.pct_change:+.1%}",
            ("extra_cost_per_year" if f.type != "supplier_gap" else "possible_saving_per_year"): f"${abs(f.annual_impact):,.0f}",
            "since": f.change_date, "note": f.detail,
        })
    return out


def _fallback(findings):
    """Template advice when no LLM is available."""
    actions = []
    for f in findings:
        if f.type == "supplier_gap":
            title, advice = f"Compare {f.item.lower()} suppliers", f"{f.detail} Ask {f.supplier} to match it or move this order."
        elif f.type == "shrinkflation":
            title, advice = f"Check {f.item.lower()} pack size", f"{f.detail} Ask for the original pack or a lower price."
        elif f.type == "price_increase":
            title, advice = f"Query {f.item.lower()} increase", f"Ask {f.supplier} to explain the {f.pct_change:+.0%} rise and get a competing quote."
        else:
            title, advice = f"{f.item} got cheaper", "Consider stocking up while the price is low."
        actions.append({"finding_id": f.id, "title": title, "advice": advice})
    return {"headline": "", "actions": actions, "email": None}


def recommend(findings, summary, max_findings=8):
    top = [f for f in findings if f.type != "price_decrease"][:max_findings]
    if not top:
        return {"headline": "No meaningful price increases found. Your supplier prices look stable.",
                "actions": [], "email": None}, True
    payload = {"business_summary": {
        "monthly_supplier_spend": f"${summary['monthly_spend']:,.0f}",
        "total_extra_cost_per_year_from_increases": f"${summary['annual_extra_cost']:,.0f}",
        "period": f"{summary['date_from']} to {summary['date_to']}"},
        "findings": _findings_for_prompt(top)}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"[{PROMPT_VERSION}]\n{json.dumps(payload, indent=1)}"}]
    try:
        data = chat_json(messages, max_tokens=2000)
        valid_ids = {f.id for f in top}
        data["actions"] = [a for a in data.get("actions", []) if a.get("finding_id") in valid_ids]
        if not data["actions"]:
            raise LLMError("no usable actions")
        return data, True
    except LLMError:
        return _fallback(top), False
