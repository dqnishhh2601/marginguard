"""
Step 3 — Analysis: pure Python/pandas, no AI.

All the money maths happens here, deterministically, so the numbers on screen are
always correct and reproducible. The LLM never does arithmetic.

Key idea: compare price PER KG / PER LITRE / PER ITEM, not price per pack.
That's how we catch "shrinkflation" — same price, smaller pack.
"""
from dataclasses import dataclass, asdict

import pandas as pd

TO_BASE = {"kg": (1.0, "kg"), "g": (0.001, "kg"), "L": (1.0, "L"), "ml": (0.001, "L"), "each": (1.0, "item")}


@dataclass
class Finding:
    id: str
    type: str            # price_increase | shrinkflation | supplier_gap | price_decrease
    item: str
    supplier: str
    base_unit: str
    baseline_price: float     # per base unit
    current_price: float      # per base unit
    pct_change: float         # e.g. 0.178 = +17.8%
    annual_impact: float      # $ per year (extra cost, or possible saving for supplier_gap)
    first_date: str
    change_date: str
    detail: str

    def to_dict(self):
        return asdict(self)


def to_lines_df(invoices):
    """invoices: list of (filename, Invoice) -> one row per line item."""
    rows = []
    for fname, inv in invoices:
        for li in inv.line_items:
            factor, base_unit = TO_BASE[li.pack_unit]
            base_per_unit = li.pack_size * factor
            rows.append({
                "file": fname, "supplier": inv.supplier_name.strip(), "invoice_number": inv.invoice_number,
                "date": pd.Timestamp(inv.invoice_date), "description": li.description.strip(),
                "quantity": li.quantity, "unit_price": li.unit_price, "line_total": li.line_total,
                "pack_size": li.pack_size, "pack_unit": li.pack_unit,
                "base_unit": base_unit, "base_per_unit": base_per_unit,
                "base_qty": li.quantity * base_per_unit,
                "price_per_base": li.unit_price / base_per_unit,
            })
    return pd.DataFrame(rows)


def _months_covered(dates):
    """Time span the purchases represent, counting the last order's own period."""
    d = sorted(dates.unique())
    if len(d) < 2:
        return 1.0
    gaps = pd.Series(d).diff().dropna().dt.days
    span_days = (d[-1] - d[0]).days + gaps.median()
    return max(span_days / 30.44, 0.5)


def fmt_unit_price(p, unit):
    return f"${p:.3f}/{unit}" if p < 1 else f"${p:.2f}/{unit}"


def analyze(df, threshold=0.03):
    """df must include canonical_name. Returns (findings, series_df, summary)."""
    findings = []
    stats = []  # per item+supplier, used for cross-supplier comparison

    for (item, supplier), g in df.sort_values("date").groupby(["canonical_name", "supplier"]):
        g = g.reset_index(drop=True)
        months = _months_covered(g["date"])
        monthly_qty = g["base_qty"].sum() / months
        base_unit = g["base_unit"].iloc[0]
        baseline, current = g["price_per_base"].iloc[0], g["price_per_base"].iloc[-1]
        pct = current / baseline - 1 if baseline else 0.0
        stats.append({"item": item, "supplier": supplier, "current": current, "monthly_qty": monthly_qty,
                      "base_unit": base_unit, "last_date": g["date"].iloc[-1]})
        if len(g) < 2 or abs(pct) < threshold:
            continue

        # When did the price first move noticeably away from the baseline?
        moved = g[(g["price_per_base"] / baseline - 1).abs() >= threshold]
        change_date = moved["date"].iloc[0] if len(moved) else g["date"].iloc[-1]

        # Shrinkflation: the pack got smaller while the pack price stayed ~the same
        first_pack, last_pack = g["base_per_unit"].iloc[0], g["base_per_unit"].iloc[-1]
        pack_price_change = g["unit_price"].iloc[-1] / g["unit_price"].iloc[0] - 1
        shrink = last_pack < first_pack * 0.99 and abs(pack_price_change) <= 0.01

        impact = (current - baseline) * monthly_qty * 12
        if shrink:
            ftype = "shrinkflation"
            detail = (f"Pack shrank from {first_pack:g}{base_unit} to {last_pack:g}{base_unit} "
                      f"but the pack price stayed at ${g['unit_price'].iloc[-1]:.2f}.")
        elif pct > 0:
            ftype = "price_increase"
            steps = (g["price_per_base"].pct_change().abs() >= 0.01).sum()
            detail = f"Price went up in {steps} step{'s' if steps != 1 else ''}."
        else:
            ftype = "price_decrease"
            detail = "Price went down."
        findings.append(Finding(
            id="", type=ftype, item=item, supplier=supplier, base_unit=base_unit,
            baseline_price=round(baseline, 4), current_price=round(current, 4), pct_change=round(pct, 4),
            annual_impact=round(impact, 2), first_date=g["date"].iloc[0].date().isoformat(),
            change_date=change_date.date().isoformat(), detail=detail))

    # Cross-supplier: same product bought from more than one supplier
    st = pd.DataFrame(stats)
    if not st.empty:
        for item, g in st.groupby("item"):
            if g["supplier"].nunique() < 2:
                continue
            cheapest = g.loc[g["current"].idxmin()]
            for _, r in g.iterrows():
                gap = r["current"] / cheapest["current"] - 1
                if r["supplier"] == cheapest["supplier"] or gap < threshold:
                    continue
                saving = (r["current"] - cheapest["current"]) * r["monthly_qty"] * 12
                findings.append(Finding(
                    id="", type="supplier_gap", item=item, supplier=r["supplier"], base_unit=r["base_unit"],
                    baseline_price=round(cheapest["current"], 4), current_price=round(r["current"], 4),
                    pct_change=round(gap, 4), annual_impact=round(saving, 2), first_date="",
                    change_date=r["last_date"].date().isoformat(),
                    detail=f"{cheapest['supplier']} sells the same item for "
                           f"{fmt_unit_price(cheapest['current'], r['base_unit'])}."))

    order = {"price_increase": 0, "shrinkflation": 0, "supplier_gap": 1, "price_decrease": 2}
    findings.sort(key=lambda f: (order[f.type], -abs(f.annual_impact)))
    for i, f in enumerate(findings, 1):
        f.id = f"F{i}"

    months_all = _months_covered(df["date"])
    increases = [f for f in findings if f.type in ("price_increase", "shrinkflation")]
    summary = {
        "invoices": int(df["file"].nunique()),
        "suppliers": int(df["supplier"].nunique()),
        "items": int(df["canonical_name"].nunique()),
        "total_spend": round(float(df["line_total"].sum()), 2),
        "monthly_spend": round(float(df["line_total"].sum() / months_all), 2),
        "date_from": df["date"].min().date().isoformat(),
        "date_to": df["date"].max().date().isoformat(),
        "n_increases": len(increases),
        "annual_extra_cost": round(float(sum(f.annual_impact for f in increases)), 2),
        "annual_switch_savings": round(float(sum(f.annual_impact for f in findings if f.type == "supplier_gap")), 2),
    }
    summary["extra_cost_pct_of_spend"] = round(
        summary["annual_extra_cost"] / (summary["monthly_spend"] * 12), 4) if summary["monthly_spend"] else 0.0

    series = df[["date", "canonical_name", "supplier", "price_per_base", "base_unit", "unit_price",
                 "pack_size", "pack_unit", "description"]].sort_values("date")
    return findings, series, summary
