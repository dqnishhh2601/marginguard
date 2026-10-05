"""
Generate realistic sample supplier invoices for a fictional café, with
price increases planted on purpose so we know exactly what MarginGuard
should find.

Run:  python generate_samples.py
Writes:
  sample_invoices/*.pdf                 24 invoices from 3 suppliers (Apr–Sep 2026)
  sample_invoices/ground_truth.json     every line item on every invoice (for evaluate.py)
  sample_invoices/PLANTED_CHANGES.md    what the app is expected to detect

All businesses and people here are made up.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

OUT = Path(__file__).parent / "sample_invoices"
CUSTOMER = ("Little Lantern Café", "12 Maple Row, Unit 01-04")
GST = 0.09
random.seed(7)

# ---------------------------------------------------------------------------
# Price schedules. Each item: name as the supplier prints it, pack info,
# quantity per order, and a function month -> pack price.
# ---------------------------------------------------------------------------

def step(changes):
    """changes: list of (from_date, price) sorted by date."""
    def price(d):
        p = changes[0][1]
        for start, val in changes:
            if d >= start:
                p = val
        return p
    return price


def noisy(base, pct=0.012):
    """Fresh produce: small natural wobble, no real trend."""
    def price(d):
        return round(base * (1 + random.uniform(-pct, pct)), 2)
    return price


SUPPLIERS = {
    "greenleaf": {
        "name": "GreenLeaf Produce Co.",
        "address": "Block 8, Fresh Market Way #02-11",
        "phone": "+00 6100 2233",
        "prefix": "GL-",
        "dates": [date(2026, 4, 6) + timedelta(days=14 * i) for i in range(12)],
        "items": [
            # (description, (pack_size, pack_unit), qty, unit_label, price_fn)
            ("Tomatoes, Roma", (1, "kg"), 30, "kg", noisy(3.20)),
            ("Romaine Lettuce", (1, "kg"), 20, "kg", noisy(4.80)),
            ("Onions Brown - 10kg bag", (10, "kg"), 6, "bag",
             step([(date(2026, 1, 1), 14.00), (date(2026, 7, 13), 15.40)])),
            ("Eggs, large (tray of 30)", (30, "each"), 12, "tray", step([(date(2026, 1, 1), 9.50)])),
            ("Lemons", (1, "kg"), 10, "kg", noisy(6.00)),
        ],
        # From August GreenLeaf's new system prints some names differently
        "renames": {date(2026, 8, 1): {"Romaine Lettuce": "Lettuce - Romaine (per kg)",
                                       "Tomatoes, Roma": "Roma Tomato"}},
    },
    "harbourline": {
        "name": "Harbourline Dry Goods Pte Ltd",
        "address": "21 Wharf Industrial Park",
        "phone": "+00 6200 7788",
        "prefix": "HDG/2026/",
        "dates": [date(2026, m, 10) for m in range(4, 10)],
        "items": [
            ("Cooking Oil 5L tin", (5, "L"), 20, "tin",
             step([(date(2026, 1, 1), 18.50), (date(2026, 6, 1), 19.90), (date(2026, 8, 1), 21.80)])),
            ("Jasmine Rice 25kg", (25, "kg"), 8, "bag",
             step([(date(2026, 1, 1), 42.00), (date(2026, 7, 1), 44.50)])),
            # Shrinkflation: same price, smaller bag from July
            ("Plain Flour 10kg", (10, "kg"), 12, "bag", step([(date(2026, 1, 1), 16.00)])),
            ("White Sugar 5kg", (5, "kg"), 10, "bag", step([(date(2026, 1, 1), 7.20)])),
            ("Coffee Beans House Blend 1kg", (1, "kg"), 15, "pkt", step([(date(2026, 1, 1), 28.00)])),
        ],
        "shrink": {"Plain Flour 10kg": (date(2026, 7, 1), "Plain Flour 9kg", (9, "kg"))},
    },
    "northgate": {
        "name": "Northgate Dairy & Meats",
        "address": "5 Coldstore Road, Cold Chain Hub",
        "phone": "+00 6300 4455",
        "prefix": "NDM-",
        "dates": [date(2026, m, 15) for m in range(4, 10)],
        "items": [
            ("FRESH MILK 2L", (2, "L"), 80, "BTL",
             step([(date(2026, 1, 1), 5.60), (date(2026, 8, 1), 5.85)])),
            ("BUTTER UNSALTED 250G", (250, "g"), 80, "PC",
             step([(date(2026, 1, 1), 3.40), (date(2026, 6, 1), 3.80)])),
            ("CHICKEN THIGH BONELESS", (1, "kg"), 60, "KG", step([(date(2026, 1, 1), 9.80)])),
            ("EGGS LARGE TRAY/30", (30, "each"), 10, "TRAY", step([(date(2026, 1, 1), 8.70)])),
            ("CREAM CHEESE 1KG BLOCK", (1, "kg"), 8, "PC", step([(date(2026, 1, 1), 12.50)])),
        ],
    },
}

styles = getSampleStyleSheet()
small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)


def money(x):
    return f"{x:,.2f}"


def build_lines(key, sup, d):
    lines = []
    for desc, pack, qty, unit_label, price_fn in sup["items"]:
        price = price_fn(d)
        name = desc
        for when, mapping in sup.get("renames", {}).items():
            if d >= when and desc in mapping:
                name = mapping[desc]
        if desc in sup.get("shrink", {}):
            when, new_name, new_pack = sup["shrink"][desc]
            if d >= when:
                name, pack = new_name, new_pack
        total = round(price * qty, 2)
        lines.append({"description": name, "quantity": qty, "unit_label": unit_label,
                      "unit_price": price, "line_total": total,
                      "pack_size": pack[0], "pack_unit": pack[1]})
    return lines


def render(key, sup, d, num, lines):
    path = OUT / f"{key}_{d.isoformat()}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    subtotal = round(sum(l["line_total"] for l in lines), 2)
    tax = round(subtotal * GST, 2)
    total = round(subtotal + tax, 2)
    story = []

    if key == "greenleaf":
        # Layout A: friendly, date like "6 Apr 2026"
        story += [Paragraph(f"<b>{sup['name']}</b>", styles["Title"]),
                  Paragraph(f"{sup['address']} · Tel {sup['phone']}", small), Spacer(1, 8),
                  Paragraph(f"<b>TAX INVOICE</b> &nbsp; No. {num} &nbsp;&nbsp; Date: {d.day} {d.strftime('%b %Y')}", styles["Normal"]),
                  Paragraph(f"Bill to: {CUSTOMER[0]}, {CUSTOMER[1]}", small), Spacer(1, 10)]
        rows = [["Item", "Qty", "Unit", "Unit Price ($)", "Amount ($)"]]
        rows += [[l["description"], str(l["quantity"]), l["unit_label"], money(l["unit_price"]), money(l["line_total"])] for l in lines]
        rows += [["", "", "", "Subtotal", money(subtotal)], ["", "", "", "GST 9%", money(tax)], ["", "", "", "TOTAL", money(total)]]
        head = colors.HexColor("#2e7d32")
    elif key == "harbourline":
        # Layout B: dense wholesale style, date like 10/04/2026
        story += [Paragraph(f"<b>{sup['name'].upper()}</b>", styles["Heading2"]),
                  Paragraph(f"{sup['address']} | {sup['phone']}", small), Spacer(1, 6),
                  Table([["Invoice #", num, "Invoice Date", d.strftime("%d/%m/%Y")],
                         ["Customer", CUSTOMER[0], "Terms", "30 days"]],
                        colWidths=[25 * mm, 60 * mm, 28 * mm, 40 * mm],
                        style=[("FONTSIZE", (0, 0), (-1, -1), 8), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                               ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold")]),
                  Spacer(1, 10)]
        rows = [["Description", "Pack", "Qty", "Price/Pack", "Total"]]
        rows += [[l["description"], l["unit_label"], str(l["quantity"]), money(l["unit_price"]), money(l["line_total"])] for l in lines]
        rows += [["", "", "", "Sub-Total", money(subtotal)], ["", "", "", "Add GST", money(tax)], ["", "", "", "Amount Due", money(total)]]
        head = colors.HexColor("#1a3a5c")
    else:
        # Layout C: ERP printout with product codes, date like 2026-04-15
        story += [Paragraph(f"{sup['name']}", styles["Heading1"]),
                  Paragraph(f"{sup['address']}<br/>Phone: {sup['phone']}", small), Spacer(1, 6),
                  Paragraph(f"INVOICE {num} &nbsp; | &nbsp; DOC DATE {d.isoformat()} &nbsp; | &nbsp; ACCT: LLC-0042 ({CUSTOMER[0]})", small),
                  Spacer(1, 10)]
        rows = [["Code", "Product", "Qty", "UOM", "Rate", "Amount"]]
        for i, l in enumerate(lines):
            rows.append([f"P{1040 + i * 7}", l["description"], str(l["quantity"]), l["unit_label"], money(l["unit_price"]), money(l["line_total"])])
        rows += [["", "", "", "", "NET", money(subtotal)], ["", "", "", "", "GST @9%", money(tax)], ["", "", "", "", "GROSS", money(total)]]
        head = colors.HexColor("#6d4c41")

    ncol = len(rows[0])
    t = Table(rows, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), head), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("ALIGN", (ncol - 3, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, len(lines)), 0.4, colors.grey),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story += [t, Spacer(1, 14), Paragraph("Thank you for your business. Goods remain our property until paid in full.", small)]
    doc.build(story)
    return path.name, subtotal, total


def main():
    OUT.mkdir(exist_ok=True)
    truth = []
    for key, sup in SUPPLIERS.items():
        for i, d in enumerate(sup["dates"]):
            num = f"{sup['prefix']}{1001 + i}"
            lines = build_lines(key, sup, d)
            fname, subtotal, total = render(key, sup, d, num, lines)
            truth.append({"file": fname, "supplier_name": sup["name"], "invoice_number": num,
                          "invoice_date": d.isoformat(), "subtotal": subtotal, "total": total,
                          "line_items": [{k: l[k] for k in ("description", "quantity", "unit_price",
                                                            "line_total", "pack_size", "pack_unit")} for l in lines]})
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    (OUT / "PLANTED_CHANGES.md").write_text(PLANTED)
    print(f"Wrote {len(truth)} invoices to {OUT}")


PLANTED = """# What MarginGuard should find in the sample invoices

Fictional customer: Little Lantern Café. Three fictional suppliers, April–September 2026.

| Item | Supplier | What happened | Price per base unit |
|---|---|---|---|
| Cooking oil | Harbourline | Two quiet increases (June, August) | $3.70/L → $4.36/L (+17.8%) |
| Butter | Northgate | Increase in June | $13.60/kg → $15.20/kg (+11.8%) |
| Plain flour | Harbourline | **Shrinkflation**: bag shrank 10kg → 9kg in July, same $16.00 price | $1.60/kg → $1.78/kg (+11.1%) |
| Brown onions | GreenLeaf | Increase in mid-July | $1.40/kg → $1.54/kg (+10.0%) |
| Jasmine rice | Harbourline | Increase in July | $1.68/kg → $1.78/kg (+6.0%) |
| Fresh milk | Northgate | Increase in August | $2.80/L → $2.925/L (+4.5%) |
| Eggs (tray of 30) | GreenLeaf vs Northgate | Same product, GreenLeaf charges $9.50 vs Northgate $8.70 (+9.2%) | cross-supplier gap |

Stable items (should NOT be flagged): tomatoes, lettuce, lemons (small ±1% wobble),
sugar, coffee beans, chicken thigh, cream cheese.

Naming traps for the AI matcher:
- GreenLeaf renamed "Romaine Lettuce" → "Lettuce - Romaine (per kg)" and
  "Tomatoes, Roma" → "Roma Tomato" from August.
- "Eggs, large (tray of 30)" (GreenLeaf) and "EGGS LARGE TRAY/30" (Northgate) are the same product.
- "Plain Flour 10kg" and "Plain Flour 9kg" are the same product in different pack sizes.

Three different invoice layouts and three date formats (6 Apr 2026, 10/04/2026, 2026-04-15).
"""

if __name__ == "__main__":
    main()
