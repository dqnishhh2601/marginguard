"""
Step 2 — Matching: decide which invoice lines are the same product.

Suppliers name things differently ("EGGS LARGE TRAY/30" vs "Eggs, large (tray of 30)"),
rename items when they change systems, and change pack sizes ("Plain Flour 10kg" ->
"Plain Flour 9kg"). String matching can't handle this reliably; the LLM can.
It returns one canonical name per product so prices can be tracked over time
and compared across suppliers.
"""
import re

from llm import LLMError, chat_json

PROMPT_VERSION = "normalize-v1"

SYSTEM = """You group product names from a food business's supplier invoices.
Different suppliers and systems write the same product differently. Give every input line a
short canonical product name, so that the SAME product always gets EXACTLY the same name.

Rules:
- Same product, different wording, word order, capitals, or pack size -> same canonical name.
  e.g. "Plain Flour 10kg" and "Plain Flour 9kg" -> "Plain flour".
  e.g. "EGGS LARGE TRAY/30" and "Eggs, large (tray of 30)" -> "Eggs, large".
- Genuinely different products must get different names (e.g. "Brown onions" vs "Red onions",
  "Salted butter" vs "Unsalted butter", "Chicken thigh" vs "Chicken breast").
- Canonical names: plain English, sentence case, no sizes or pack info.
- category: one of "Produce", "Dairy & eggs", "Meat & seafood", "Dry goods", "Beverages", "Other".

Reply with ONE JSON object only:
{"items": [{"id": <input id>, "canonical_name": string, "category": string}]}"""


def _fallback_name(desc):
    """Crude non-AI fallback: strip sizes/punctuation. Used only if the LLM is unavailable."""
    s = desc.lower()
    s = re.sub(r"\d+(\.\d+)?\s*(kg|g|l|ml|pcs?|x)\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    words = sorted(w for w in s.split() if w not in {"tin", "bag", "tray", "of", "per", "block", "pkt"})
    return " ".join(words).capitalize() or desc


def normalize(lines_df):
    """Adds canonical_name and category columns. Returns (df, used_ai: bool)."""
    uniq = (lines_df[["supplier", "description", "pack_size", "pack_unit"]]
            .drop_duplicates(["supplier", "description"])
            .sort_values(["supplier", "description"]).reset_index(drop=True))
    payload = "\n".join(f'{i}. [{r.supplier}] "{r.description}" (pack: {r.pack_size:g} {r.pack_unit})'
                        for i, r in uniq.iterrows())
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"[{PROMPT_VERSION}] Lines:\n{payload}"}]
    used_ai = True
    try:
        data = chat_json(messages, max_tokens=3000)
        by_id = {int(x["id"]): x for x in data.get("items", [])}
        missing = [i for i in uniq.index if i not in by_id]
        if missing:
            raise LLMError(f"matcher skipped {len(missing)} lines")
        uniq["canonical_name"] = [by_id[i]["canonical_name"].strip() for i in uniq.index]
        uniq["category"] = [by_id[i].get("category", "Other") for i in uniq.index]
    except LLMError:
        used_ai = False
        uniq["canonical_name"] = uniq["description"].map(_fallback_name)
        uniq["category"] = "Other"

    # Case-insensitive tidy-up so "Eggs, Large" and "Eggs, large" can never split
    first_spelling = {}
    uniq["canonical_name"] = [first_spelling.setdefault(n.lower(), n) for n in uniq["canonical_name"]]

    out = lines_df.merge(uniq[["supplier", "description", "canonical_name", "category"]],
                         on=["supplier", "description"], how="left")
    return out, used_ai
