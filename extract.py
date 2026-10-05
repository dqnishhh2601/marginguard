"""
Step 1 — Extraction: invoice PDF -> text -> LLM -> validated structured data.

The LLM reads messy, differently-formatted invoices and returns JSON.
Code then checks the LLM's work with arithmetic (qty x price = line total,
lines add up to the subtotal). If a check fails, the LLM is asked again with
the specific error, so mistakes are caught instead of silently trusted.
"""
import json
from datetime import date
from typing import Literal, Optional

import pdfplumber
from pydantic import BaseModel, Field, ValidationError, field_validator

from llm import chat_json

PROMPT_VERSION = "extract-v1"

SYSTEM = """You extract line items from supplier invoices for a small food business.
Reply with ONE JSON object and nothing else, in exactly this shape:
{
  "supplier_name": string,
  "invoice_number": string,
  "invoice_date": "YYYY-MM-DD",
  "subtotal": number or null,   // total BEFORE tax
  "line_items": [
    {
      "description": string,     // product name exactly as printed (drop product codes)
      "quantity": number,        // how many units were bought
      "unit_price": number,      // price for ONE unit, before tax
      "line_total": number,      // amount for this line, before tax
      "pack_size": number,       // amount of product in ONE unit bought
      "pack_unit": "kg" | "g" | "L" | "ml" | "each"
    }
  ]
}
Rules:
- Only real products. Skip subtotal, tax/GST/VAT, delivery, discount and total lines.
- pack_size/pack_unit describe ONE unit of quantity. Examples:
  "Cooking Oil 5L tin" bought by the tin -> 5, "L".
  "Tomatoes" sold per kg -> 1, "kg".   "Butter 250G" per piece -> 250, "g".
  "Eggs tray of 30" per tray -> 30, "each".  "Milk 2L" per bottle -> 2, "L".
- Dates may be written as 6 Apr 2026, 10/04/2026 (day/month/year) or 2026-04-15. Always output YYYY-MM-DD.
- Numbers must be plain numbers (no currency symbols or thousands commas)."""


class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    line_total: float = Field(ge=0)
    pack_size: float = Field(gt=0)
    pack_unit: Literal["kg", "g", "L", "ml", "each"]

    @field_validator("pack_unit", mode="before")
    @classmethod
    def norm_unit(cls, v):
        aliases = {"kg": "kg", "kgs": "kg", "g": "g", "gm": "g", "gram": "g", "grams": "g",
                   "l": "L", "ltr": "L", "litre": "L", "liter": "L", "ml": "ml",
                   "each": "each", "ea": "each", "pc": "each", "pcs": "each", "piece": "each", "unit": "each"}
        return aliases.get(str(v).strip().lower(), v)


class Invoice(BaseModel):
    supplier_name: str
    invoice_number: str
    invoice_date: date
    subtotal: Optional[float] = None
    line_items: list[LineItem]


def pdf_text(path_or_file):
    with pdfplumber.open(path_or_file) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages).strip()


def check_arithmetic(inv: Invoice):
    """Return a list of problems found by re-doing the invoice maths."""
    problems = []
    for li in inv.line_items:
        expected = li.quantity * li.unit_price
        if abs(expected - li.line_total) > max(0.05, 0.01 * li.line_total):
            problems.append(f"'{li.description}': {li.quantity} x {li.unit_price} = {expected:.2f}, "
                            f"but line_total is {li.line_total}")
    if inv.subtotal:
        s = sum(li.line_total for li in inv.line_items)
        if abs(s - inv.subtotal) > max(0.05, 0.005 * inv.subtotal):
            problems.append(f"line totals add up to {s:.2f} but subtotal is {inv.subtotal} "
                            f"(a line item may be missing or a non-product line included)")
    return problems


def extract_invoice(path_or_file, filename):
    """Returns (Invoice, warnings). Raises on unreadable files / LLM failure."""
    text = pdf_text(path_or_file)
    if len(text) < 30:
        raise ValueError(f"{filename}: no readable text (scanned image? photo/OCR support is not built yet)")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"[{PROMPT_VERSION}] Invoice text:\n\n{text}"}]
    best = None  # last schema-valid answer, used if the retry still has maths problems
    for attempt in range(2):
        data = chat_json(messages)
        try:
            inv = Invoice.model_validate(data)
        except ValidationError as e:
            problems = [f"JSON did not match the schema: {e.errors()[:3]}"]
        else:
            best = inv
            problems = check_arithmetic(inv)
            if not problems:
                return inv, []
        if attempt == 0:
            # Self-correction: tell the model exactly which check failed and ask again
            messages = messages + [
                {"role": "assistant", "content": json.dumps(data)},
                {"role": "user", "content": "Your extraction failed these checks:\n- " + "\n- ".join(problems)
                 + "\nRe-read the invoice and reply with the corrected JSON only."}]
    if best is not None:
        # Keep the data but surface the problem to the user instead of hiding it
        return best, [f"{filename}: {p}" for p in check_arithmetic(best)]
    raise ValueError(f"{filename}: could not extract valid data ({problems[0]})")
