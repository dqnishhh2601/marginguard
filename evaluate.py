"""
Measures how accurately the AI reads the sample invoices, by comparing its output
with the known-correct answers in sample_invoices/ground_truth.json.

Run (needs LLM_API_KEY in .env):   python evaluate.py

Put the resulting accuracy number in your README — judges score "correctness".
Running this also fills cache/ so the deployed demo loads the samples instantly.
"""
import json
from pathlib import Path

from extract import extract_invoice
from llm import load_env

ROOT = Path(__file__).parent
FIELDS = ["quantity", "unit_price", "line_total", "pack_size", "pack_unit"]


def close(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return abs(float(a) - float(b)) <= 0.011


def main():
    load_env()
    truth = json.loads((ROOT / "sample_invoices/ground_truth.json").read_text())
    total = correct = inv_ok = 0
    header_total = header_ok = 0
    for t in truth:
        try:
            inv, warns = extract_invoice(ROOT / "sample_invoices" / t["file"], t["file"])
        except Exception as e:
            print(f"FAIL  {t['file']}: {e}")
            total += len(t["line_items"]) * len(FIELDS)
            continue
        got = {li.description.strip().lower(): li.model_dump() for li in inv.line_items}
        file_ok = True
        for checks in [(inv.invoice_date.isoformat(), t["invoice_date"]), (inv.invoice_number, t["invoice_number"])]:
            header_total += 1
            header_ok += checks[0] == checks[1]
            file_ok &= checks[0] == checks[1]
        for exp in t["line_items"]:
            g = got.get(exp["description"].lower()) or next((v for k, v in got.items() if k.startswith(exp["description"].lower() + " ")), None)
            for f in FIELDS:
                total += 1
                if g is not None and close(g[f], exp[f]):
                    correct += 1
                else:
                    file_ok = False
                    print(f"  miss {t['file']}: '{exp['description']}' {f}: expected {exp[f]}, got {g[f] if g else 'MISSING'}")
        if len(got) != len(t["line_items"]):
            file_ok = False
            print(f"  {t['file']}: expected {len(t['line_items'])} lines, got {len(got)}")
        inv_ok += file_ok
        print(f"{'OK  ' if file_ok else 'DIFF'}  {t['file']}" + (f"  ({len(warns)} warning)" if warns else ""))

    print(f"\nLine-item fields correct: {correct}/{total} = {correct / total:.1%}")
    print(f"Invoice dates & numbers correct: {header_ok}/{header_total}")
    print(f"Invoices read perfectly: {inv_ok}/{len(truth)}")


if __name__ == "__main__":
    main()

