"""
Step 5 — Decision support: why did a price rise, is it in line with the market,
what could happen next, what is it worth, and what should the owner do?

Everything here is deterministic Python. No AI, no network: market data is passed
in (see market.py), so every function can be unit-tested and every number traced.

Wording rule used throughout: market data can show that an increase is
*consistent with* or *larger than* market movements. It cannot prove *why* a
supplier raised a price, so we never say it does.
"""
import math
import re

import pandas as pd

import market as mkt
from analyze import TO_BASE, _months_covered

FOOD = "Food ingredients & agricultural commodities"
CONTEXT_SERIES = ["POILBREUSDM", "EXSIUS"]   # energy/freight and currency, shown for every food item
TOLERANCE_PP = 0.05                           # +/- 5 percentage points counts as "in line with the market"
SYMBOL = {"SGD": "S$", "USD": "US$"}          # what to print in front of the user's own invoice amounts

#: Every benchmark we connect is a market proxy (a raw-commodity index), never a quote for the exact
#: product bought. A direct comparable would be an actual quote for an equivalent product, which only
#: the user's own invoices and entered quotes provide.
PROXY_NOTE = ("Market proxy: a raw-commodity index, not a quote for the product you buy. "
              "Use it as context, not as the price you should be paying.")


def sym(currency="SGD"):
    return SYMBOL.get(currency, "$")

# Benchmark profiles. 'primary' series are the closest available market proxies.
PROFILES = {
    "Edible oils": {
        "group": FOOD, "primary": ["PPOILUSDM", "PSOILUSDM"], "comparability": "Medium",
        "why": "Wholesale cooking oil is mostly palm- or soybean-based, so these world prices drive its raw-material cost."},
    "Wheat & flour": {
        "group": FOOD, "primary": ["PWHEAMTUSDM"], "comparability": "Medium",
        "why": "Flour is milled wheat; wheat is its main raw-material cost."},
    "Rice": {
        "group": FOOD, "primary": ["PRICENPQUSDM"], "comparability": "Medium",
        "why": "Thai white rice export price. Jasmine rice is a premium Thai variety that tends to move with it, but not one-for-one."},
    "Sugar": {
        "group": FOOD, "primary": ["PSUGAISAUSDM"], "comparability": "Medium",
        "why": "World raw-sugar price. Refined white sugar adds refining and packing costs."},
    "Coffee": {
        "group": FOOD, "primary": ["PCOFFOTMUSDM"], "comparability": "Low",
        "why": "Green Arabica bean price. A roasted house blend also carries roasting, labour and margin, so it moves less."},
    "Poultry": {
        "group": FOOD, "primary": ["PPOULTUSDM"], "comparability": "Low",
        "why": "Global poultry export price; local chicken prices also depend on the importing country, cut and cold-chain costs."},
    "Dairy": {
        "group": FOOD, "primary": [], "comparability": "None",
        "why": "MarginGuard does not have a dairy price benchmark yet."},
    "Eggs": {
        "group": FOOD, "primary": [], "comparability": "None",
        "why": "Egg prices are set by local supply and import rules; no comparable public benchmark is connected."},
    "Fresh produce": {
        "group": FOOD, "primary": [], "comparability": "None",
        "why": "Fresh produce prices are local and seasonal; no comparable public benchmark is connected."},
    "Packaging": {
        "group": "Packaging & raw materials", "primary": [], "comparability": "None",
        "why": "Plastic packaging follows oil and petrochemical prices, but no packaging price index is connected yet."},
    "Energy & utilities": {
        "group": "Energy & utilities", "primary": [], "comparability": "None",
        "why": "Electricity and gas tariffs are set by your retailer or regulator; crude oil is not a reliable stand-in for them."},
    "Transport & logistics": {
        "group": "Transportation & logistics", "primary": [], "comparability": "None",
        "why": "Delivery charges depend on fuel, labour and route; Brent crude is shown only as a fuel-cost signal."},
    "Other": {
        "group": "Other business consumables", "primary": [], "comparability": "None",
        "why": "No market benchmark is connected for this kind of item."},
}

_KEYWORDS = [
    ("Edible oils", {"oil", "oils", "ghee", "shortening"}),
    ("Rice", {"rice"}),
    ("Wheat & flour", {"flour", "wheat", "pasta", "noodles", "noodle"}),
    ("Sugar", {"sugar"}),
    ("Coffee", {"coffee", "espresso"}),
    ("Poultry", {"chicken", "poultry", "turkey", "duck"}),
    ("Dairy", {"milk", "butter", "cheese", "cream", "yoghurt", "yogurt"}),
    ("Eggs", {"egg", "eggs"}),
    ("Fresh produce", {"onion", "onions", "tomato", "tomatoes", "lettuce", "lemon", "lemons", "lime", "limes",
                       "potato", "potatoes", "carrot", "carrots", "cabbage", "garlic", "vegetable", "vegetables",
                       "fruit", "herbs", "spinach", "cucumber", "apple", "apples", "banana", "bananas"}),
    ("Packaging", {"container", "containers", "packaging", "cup", "cups", "lid", "lids", "box", "boxes",
                   "straw", "straws", "napkin", "napkins", "wrap", "foil", "takeaway"}),
    ("Energy & utilities", {"electricity", "gas", "lpg", "water", "utilities", "power"}),
    ("Transport & logistics", {"delivery", "freight", "shipping", "courier", "transport"}),
]


def classify(item_name, ai_category=None):
    """Item name -> benchmark profile name. Keyword rules first, then the AI's category."""
    words = set(re.findall(r"[a-z]+", str(item_name).lower()))
    for profile, keys in _KEYWORDS:
        if words & keys:
            return profile
    cat = (ai_category or "").lower()
    if "dairy" in cat:
        return "Dairy"
    if "produce" in cat:
        return "Fresh produce"
    return "Other"


# ----------------------------------------------------------------------------- per-item statistics

def item_stats(lines):
    """One row per (item, supplier): monthly quantity, monthly spend, latest price."""
    rows = []
    for (item, supplier), g in lines.sort_values("date").groupby(["canonical_name", "supplier"]):
        months = _months_covered(g["date"])
        rows.append({
            "item": item, "supplier": supplier,
            "category": g["category"].iloc[0] if "category" in g else None,
            "base_unit": g["base_unit"].iloc[0],
            "monthly_qty": g["base_qty"].sum() / months,
            "monthly_spend": g["line_total"].sum() / months,
            "first_price": g["price_per_base"].iloc[0],
            "current_price": g["price_per_base"].iloc[-1],
            "first_date": g["date"].iloc[0], "last_date": g["date"].iloc[-1], "n_invoices": len(g),
        })
    return pd.DataFrame(rows)


def spend_by_group(lines, overrides=None):
    overrides = overrides or {}
    months = _months_covered(lines["date"])
    df = lines.copy()
    df["profile"] = [overrides.get(n) or classify(n, c) for n, c in
                     zip(df["canonical_name"], df.get("category", pd.Series([None] * len(df))))]
    df["group"] = df["profile"].map(lambda p: PROFILES[p]["group"])
    out = df.groupby(["group", "profile"], as_index=False)["line_total"].sum()
    out["monthly_spend"] = out["line_total"] / months
    return out.sort_values("monthly_spend", ascending=False)


# ----------------------------------------------------------------------------- benchmarking

def _price_on_or_before(series, when):
    s = series[series["date"] <= when]
    return (float(s["price_per_base"].iloc[-1]), s["date"].iloc[-1]) if len(s) else (None, None)


def benchmark(item_series, profile_name, market, currency="SGD"):
    """Compare one supplier's price change for one item with market data.

    item_series: rows of analyze's series for ONE item from ONE supplier (date, price_per_base, ...).
    market: {series_id: (df or None, meta)}.
    Returns a dict that the app renders as-is.
    """
    prof = PROFILES[profile_name]
    g = item_series.sort_values("date")
    cur = sym(currency)
    out = {"profile": profile_name, "group": prof["group"], "comparability": prof["comparability"],
           "why_relevant": prof["why"], "match_type": "proxy" if prof["primary"] else "none",
           "drivers": [], "context": [], "evidence": [], "facts": [], "associations": [], "hypotheses": [],
           "limitations": [], "verdict": "insufficient", "benchmark_pct": None, "gap_pp": None,
           "unverified": False, "unverified_text": "", "currency": currency}
    if len(g) < 2:
        out["verdict_text"] = "Only one invoice for this item, so there is no price change to explain."
        return out

    start, end = g["date"].iloc[0], g["date"].iloc[-1]
    p0, p1 = float(g["price_per_base"].iloc[0]), float(g["price_per_base"].iloc[-1])
    unit = g["base_unit"].iloc[0]
    out["supplier"] = {"start": start, "end": end, "start_price": p0, "end_price": p1, "pct": p1 / p0 - 1,
                       "unit": unit}
    out["facts"].append(f"Your price went from {cur}{p0:,.3f} to {cur}{p1:,.3f} per {unit} "
                        f"({p1 / p0 - 1:+.1%}) between {start:%d %b %Y} and {end:%d %b %Y}.")

    fx_df = market.get("EXSIUS", (None, None))[0]
    convert = currency == "SGD"

    def move(sid):
        df, meta = market.get(sid, (None, None))
        if meta is None:
            return None
        if meta["mode"] != "missing":
            out["evidence"].append(meta)
        ch = None if df is None else mkt.change_between(df, start, end)
        if ch is None:
            return {"id": sid, "title": meta["title"], "available": False, "meta": meta}
        local = ch["pct"]
        fx = None
        if convert and meta["kind"] in ("commodity", "energy") and fx_df is not None:
            fx = mkt.change_between(fx_df, ch["start_month"], ch["end_month"])
            if fx and fx["end_month"] == ch["end_month"]:
                local = (1 + ch["pct"]) * (1 + fx["pct"]) - 1
            else:
                fx = None
        return {"id": sid, "title": meta["title"], "available": True, "meta": meta, **ch,
                "local_pct": local, "fx_pct": fx["pct"] if fx else None}

    drivers = [d for d in (move(s) for s in prof["primary"]) if d]
    context = [d for d in (move(s) for s in (CONTEXT_SERIES if prof["group"] in (FOOD, "Transportation & logistics")
                                              else [])) if d]
    out["drivers"], out["context"] = drivers, context
    usable = [d for d in drivers if d["available"]]

    if not prof["primary"]:
        out["verdict"] = "no_benchmark"
        out["verdict_text"] = f"No market benchmark for this item. {prof['why']}"
    elif not usable:
        out["verdict"] = "no_benchmark"
        out["verdict_text"] = "Market data for this item couldn't be loaded, so the increase can't be compared."
    else:
        # Matched period: compare like with like. The market series stops at bench_end, so the
        # verdict uses the supplier's price on the nearest invoice on or before that date.
        # Anything the supplier did AFTER that is reported separately as not yet checkable.
        bench_end = pd.Timestamp(min(d["end_month"] for d in usable)) + pd.offsets.MonthEnd(0)
        aligned_price, aligned_date = _price_on_or_before(g, bench_end)
        matched_pct = aligned_price / p0 - 1
        later_pct = p1 / aligned_price - 1 if aligned_price else 0.0   # change since the market data ends
        sup_pct = p1 / p0 - 1
        bench_pct = sum(d["local_pct"] for d in usable) / len(usable)
        gap = matched_pct - bench_pct
        supported_price = p0 * (1 + bench_pct)       # what the market move alone would justify
        out.update(benchmark_pct=bench_pct, gap_pp=gap, supplier_pct=sup_pct, matched_pct=matched_pct,
                   supplier_aligned_pct=matched_pct, later_pct=later_pct, aligned_end=aligned_date,
                   bench_end=bench_end, supported_price=supported_price, start_price=p0, end_price=p1,
                   above_market_per_unit=max(aligned_price - supported_price, 0.0),
                   unverified_per_unit=max(p1 - aligned_price, 0.0))
        names = " and ".join(d["meta"].get("short", d["title"]) for d in usable)
        m_period = f"{pd.Timestamp(usable[0]['start_month']):%b}–{pd.Timestamp(usable[0]['end_month']):%b %Y}"
        out["period"] = m_period
        for d in usable:
            conv = f" in {cur} terms" if d["fx_pct"] is not None else ""
            out["facts"].append(f"{d['title']}: {d['local_pct']:+.1%}{conv} over {m_period} "
                                f"(source: {d['meta']['source']}).")
        out["facts"].append(f"Over the same months ({m_period}) your price changed {matched_pct:+.1%}.")

        matched = f"Over the matched period {m_period}, your price changed {matched_pct:+.1%} against "
        if abs(gap) <= TOLERANCE_PP:
            out["verdict"] = "consistent"
            out["verdict_text"] = matched + f"{names} at {bench_pct:+.1%}: broadly in line."
        elif gap > 0:
            out["verdict"] = "exceeds"
            out["verdict_text"] = (matched + f"{names} at {bench_pct:+.1%}, a gap of {gap * 100:.0f} percentage "
                                   f"points. Worth asking about; it is not by itself proof of overcharging.")
        else:
            out["verdict"] = "below"
            out["verdict_text"] = (matched + f"{names} at {bench_pct:+.1%}: the market rose more than your price, "
                                   f"so your supplier may not have passed on the full move yet.")
        out["associations"].append(out["verdict_text"])

        # The part of the supplier's move that the market data does not yet cover
        out["unverified"] = abs(later_pct) > 0.01
        if out["unverified"]:
            out["unverified_text"] = (
                f"Since {aligned_date:%b %Y} your price has moved a further {later_pct:+.1%} "
                f"(total {sup_pct:+.1%} since {start:%b %Y}). {usable[0]['meta']['title'].split(',')[0]} and other "
                f"market series are only published to {pd.Timestamp(usable[0]['end_month']):%b %Y}, so this part "
                f"cannot be compared with the market yet.")
            out["associations"].append(out["unverified_text"])
            out["limitations"].append(
                f"Commodity prices are published about two months behind, so the comparison covers {m_period} only. "
                f"Your {later_pct:+.1%} move after that is reported separately, not treated as above or below market.")
        elif any(d["truncated"] for d in usable):
            out["limitations"].append(
                f"Commodity prices are published about two months behind; the comparison covers {m_period}. "
                f"Your price has not changed materially since then.")

    for c in context:
        if not c["available"]:
            continue
        if c["id"] == "EXSIUS":
            if convert:
                direction = "stronger" if c["pct"] > 0 else "weaker"
                out["associations"].append(
                    f"The US dollar got {abs(c['pct']):.1%} {direction} against the {cur} "
                    f"({pd.Timestamp(c['start_month']):%b}–{pd.Timestamp(c['end_month']):%b %Y}), which "
                    f"{'raises' if c['pct'] > 0 else 'lowers'} the {cur} cost of imports priced in US$.")
        else:
            trend = "rose" if c["pct"] > 0.02 else ("fell" if c["pct"] < -0.02 else "was roughly flat")
            out["associations"].append(
                f"Brent crude, a signal for fuel and freight costs, {trend} ({c['pct']:+.1%}, "
                f"{pd.Timestamp(c['start_month']):%b}–{pd.Timestamp(c['end_month']):%b %Y})"
                + (" — it does not explain an increase." if c["pct"] < -0.02 else "."))

    out["hypotheses"] = [
        "Supplier-specific costs (wages, rent, delivery, wastage) may have changed. These can't be checked from public data.",
        "Contract or volume terms may differ from what the market price reflects.",
    ]
    if out["comparability"] != "None":
        out["limitations"] += [
            PROXY_NOTE + " Raw materials are only part of a wholesale price, so even a fair supplier usually "
            "passes on part of a commodity move, often 1-3 months later.",
            f"Comparability: {out['comparability']}. {prof['why']}",
        ]
        if convert:
            out["limitations"].append(f"Assumes your invoices are in {cur}; US$ prices are converted with the "
                                      f"Federal Reserve's monthly SGD/USD rate.")
    return out


# ----------------------------------------------------------------------------- forecasting

def _ses(values, alpha):
    level = values[0]
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return level


def _trend(values, window=12):
    y = list(values[-window:])
    n = len(y)
    if n < 3:
        return y[-1], 0.0
    xbar, ybar = (n - 1) / 2, sum(y) / n
    sxx = sum((i - xbar) ** 2 for i in range(n))
    slope = sum((i - xbar) * (v - ybar) for i, v in enumerate(y)) / sxx
    return ybar + slope * (n - 1 - xbar), slope   # fitted last value, slope per month


METHODS = {
    "naive": "Last value carried forward",
    "ses": "Exponential smoothing (recent months weighted more)",
    "trend": "Straight-line trend of the last 12 months",
}


def _predict(method, hist, h):
    if method == "naive":
        return hist[-1]
    if method == "ses":
        return _ses(hist, 0.5)
    last_fit, slope = _trend(hist)
    return last_fit + slope * h


def forecast_series(df, horizons=(1, 3, 6), backtest_points=6, min_history=12):
    """Simple, explainable forecast of a monthly market series.

    Picks the method with the lowest error in a rolling one-month-ahead backtest
    over the last `backtest_points` months. The range is a rough band built from
    those errors, widening with the horizon. Returns None if there's no data.
    """
    if df is None or df.empty:
        return None
    vals = [float(v) for v in df["value"].tolist()][-36:]
    last_date, last = df["date"].iloc[-1], vals[-1]
    if len(vals) < min_history:
        return {"method": "naive", "method_label": METHODS["naive"], "backtest": {}, "last_date": last_date,
                "last_value": last, "rows": [{"h": h, "date": last_date + pd.DateOffset(months=h), "point": last,
                                              "low": None, "high": None, "pct": 0.0} for h in horizons],
                "note": f"Only {len(vals)} months of data: showing a flat 'no change' line, not a forecast."}
    errors = {}
    for m in METHODS:
        errs = []
        for k in range(len(vals) - backtest_points, len(vals)):
            pred = _predict(m, vals[:k], 1)
            errs.append(abs(pred - vals[k]) / vals[k])
        errors[m] = sum(errs) / len(errs)
    # Prefer the simpler method unless a fancier one is clearly (10%+) better
    best = "naive"
    for m in ("ses", "trend"):
        if errors[m] < errors[best] * 0.9:
            best = m
    rows = []
    for h in horizons:
        point = _predict(best, vals, h)
        band = errors[best] * math.sqrt(h) * 1.5
        rows.append({"h": h, "date": last_date + pd.DateOffset(months=h), "point": point,
                     "low": point * (1 - band), "high": point * (1 + band), "pct": point / last - 1})
    return {"method": best, "method_label": METHODS[best], "backtest": errors, "last_date": last_date,
            "last_value": last, "rows": rows,
            "note": "A rough guide from past prices only. Commodity prices react to weather, policy and news "
                    "that no simple model can foresee."}


def market_outlook(forecast_pct, commodity_shock=0.0, fx_shock=0.0, fx_exposed=True):
    """Combine the model's own projection with the user's what-if shocks.

    These are three separate things and they compound rather than add:
      forecast_pct  what the historical series alone projects (often 0 under a naive model)
      commodity_shock  "what if the commodity jumps X%" - the user's assumption, not a prediction
      fx_shock  "what if the US dollar moves Y%" - only bites on items priced off a US$ commodity

    Returns (combined_pct, parts) where parts says which inputs were actually applied.
    """
    parts = {"forecast": forecast_pct or 0.0, "commodity": commodity_shock or 0.0,
             "fx": (fx_shock or 0.0) if fx_exposed else 0.0, "fx_applied": bool(fx_exposed)}
    combined = (1 + parts["forecast"]) * (1 + parts["commodity"]) * (1 + parts["fx"]) - 1
    return combined, parts


def scenario_set(item_series, horizon, forecast_pct=None, commodity_shock=0.0, fx_shock=0.0,
                 pass_through=0.5, catch_up_pct=None, fx_exposed=True, has_benchmark=True):
    """Supplier-price what-ifs at ONE horizon, with every input's applicability recorded.

    A market benchmark is what makes the commodity and FX shocks mean anything: with no
    benchmark (eggs, dairy, fresh produce) they are reported as not applicable rather than
    quietly applied to a number they have no relationship with.
    """
    g = item_series.sort_values("date")
    current = float(g["price_per_base"].iloc[-1])
    out = {"horizon": horizon, "current": current, "market_pct": None, "market_parts": None,
           "applicable": {"commodity_shock": bool(has_benchmark), "fx_shock": bool(has_benchmark and fx_exposed),
                          "pass_through": bool(has_benchmark or (catch_up_pct or 0) > 0)},
           "rows": []}

    def row(name, pct, basis):
        out["rows"].append({"scenario": name, "pct": pct, "price": current * (1 + pct), "basis": basis,
                            "h": horizon})

    row("Price holds", 0.0, "Your supplier does not move again in this period.")

    if len(g) >= 2:
        months = max((g["date"].iloc[-1] - g["date"].iloc[0]).days / 30.44, 1.0)
        r = (current / float(g["price_per_base"].iloc[0])) ** (1 / months) - 1
        pct = max(min((1 + r) ** horizon - 1, 0.5), -0.5)
        row("Recent trend continues", pct,
            f"Your own price has moved about {r:+.1%} a month; carried forward {horizon} month"
            f"{'s' if horizon > 1 else ''} and capped at +/-50%.")

    if catch_up_pct and catch_up_pct > 0:
        row(f"Supplier catches up on {pass_through:.0%} of the market rise already seen",
            catch_up_pct * pass_through,
            f"The market has already risen {catch_up_pct:+.1%} more than your price. This applies "
            f"{pass_through:.0%} of that gap.")

    if has_benchmark:
        combined, parts = market_outlook(forecast_pct, commodity_shock, fx_shock, fx_exposed)
        out["market_pct"], out["market_parts"] = combined, parts
        bits = [f"projection {parts['forecast']:+.1%}"]
        if parts["commodity"]:
            bits.append(f"your commodity shock {parts['commodity']:+.1%}")
        if parts["fx"]:
            bits.append(f"your FX shock {parts['fx']:+.1%}")
        row(f"Market moves {combined:+.1%}, supplier passes on {pass_through:.0%}",
            combined * pass_through,
            "Market move = " + " x ".join(bits) + f". Only {pass_through:.0%} reaches your invoice, because "
            f"raw materials are one part of a wholesale price.")
    return out


def catch_up(b):
    """Market rise not yet reflected in the supplier's price, from a benchmark result (or None)."""
    if b.get("verdict") != "below" or b.get("benchmark_pct") is None:
        return None
    return (1 + b["benchmark_pct"]) / (1 + b["supplier_pct"]) - 1


def cumulative_impact(monthly_spend, pct, months):
    """Money effect of a price change held for `months`.

    Assumes the change lands immediately and stays put, so the same extra cost repeats
    each month. That is an assumption, not a forecast, and the app says so on screen.
    Cumulative is deliberately NOT annualised: a 1-month horizon costs one month.
    """
    extra = monthly_spend * pct
    return {"baseline_monthly": monthly_spend, "scenario_monthly": monthly_spend + extra,
            "extra_monthly": extra, "months": months, "cumulative": extra * months,
            "annualised_if_sustained": extra * 12, "pct": pct}


# ----------------------------------------------------------------------------- money

def cost_impact(monthly_qty, unit_price, pct_change, monthly_revenue=None, monthly_cogs=None):
    """Extra procurement cost from a % price change, and its effect on gross profit.

    Gross profit = revenue − cost of goods sold. The extra cost lowers gross profit
    dollar-for-dollar unless selling prices change.
    """
    base_monthly = monthly_qty * unit_price
    extra_m = base_monthly * pct_change
    out = {"base_monthly": base_monthly, "extra_monthly": extra_m, "extra_annual": extra_m * 12}
    if monthly_revenue and monthly_revenue > 0:
        cogs = monthly_cogs if monthly_cogs is not None else 0.0
        gp0 = monthly_revenue - cogs
        gp1 = gp0 - extra_m
        out.update(gross_profit_before=gp0, gross_profit_after=gp1,
                   gross_margin_before=gp0 / monthly_revenue, gross_margin_after=gp1 / monthly_revenue,
                   margin_change_pp=(gp1 - gp0) / monthly_revenue,
                   price_rise_to_offset=extra_m / monthly_revenue)
    return out


def quote_savings(current_price_per_base, current_base_unit, pack_price, pack_size, pack_unit, monthly_base_qty,
                  extra_monthly_cost=0.0, one_off_cost=0.0):
    """Savings from an alternative quote, on a like-for-like price per kg / L / item."""
    if pack_unit not in TO_BASE:
        return {"ok": False, "reason": f"Unknown unit '{pack_unit}'."}
    factor, base_unit = TO_BASE[pack_unit]
    if base_unit != current_base_unit:
        return {"ok": False, "reason": f"Quote is per {base_unit} but you buy per {current_base_unit}; not comparable."}
    if not pack_size or pack_size <= 0 or pack_price is None or pack_price < 0:
        return {"ok": False, "reason": "Pack size and price are needed."}
    alt = pack_price / (pack_size * factor)
    gross_m = (current_price_per_base - alt) * monthly_base_qty
    net_m = gross_m - (extra_monthly_cost or 0.0)
    payback = (one_off_cost / net_m) if (one_off_cost and net_m > 0) else None
    # Landed price: the list price plus the delivery you'd actually pay, spread over what you buy.
    landed = alt + ((extra_monthly_cost or 0.0) / monthly_base_qty if monthly_base_qty else 0.0)
    packs = monthly_base_qty / (pack_size * factor) if pack_size else 0.0
    return {"ok": True, "alt_price": alt, "landed_price": landed, "unit": base_unit,
            "diff_pct": alt / current_price_per_base - 1,
            "landed_diff_pct": landed / current_price_per_base - 1,
            "gross_monthly": gross_m, "net_monthly": net_m, "net_annual": net_m * 12 - (one_off_cost or 0.0),
            "payback_months": payback, "packs_per_month": packs}


# ----------------------------------------------------------------------------- recommendations

CONF_W = {"High": 1.0, "Medium": 0.7, "Low": 0.4}
DIFF_W = {"Low": 1.0, "Medium": 0.8, "High": 0.6}
SAVING_KINDS = {"negotiate", "switch", "shrink"}   # count towards "potential savings"


def recommendations(findings, bench_by_key, stats, scenario_6m=None, quotes=None, pass_through=0.5):
    """Rules-based, ranked next actions. Every number comes from code.

    bench_by_key: {(item, supplier): benchmark dict}
    stats: item_stats DataFrame; scenario_6m: {(item, supplier): % change under the pass-through scenario at 6m}
    quotes: list of dicts from the Savings page (already priced with quote_savings).
    """
    scenario_6m = dict(scenario_6m or {})
    for key, b in bench_by_key.items():      # default 'below' scenario: catch-up on the market rise already seen
        cu = catch_up(b)
        if cu and key not in scenario_6m:
            scenario_6m[key] = cu * pass_through
    recs = []
    st_idx = {(r["item"], r["supplier"]): r for _, r in stats.iterrows()} if stats is not None else {}

    def add(**k):
        k.setdefault("saving", k["impact"] if k["kind"] in SAVING_KINDS else 0.0)
        k.setdefault("counts_as_saving", k["kind"] in SAVING_KINDS)
        k["score"] = max(k["impact"], 0) * CONF_W[k["confidence"]] * DIFF_W[k["difficulty"]]
        recs.append(k)

    for f in findings:
        key = (f.item, f.supplier)
        b = bench_by_key.get(key, {})
        v = b.get("verdict")
        if f.type == "supplier_gap":
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="switch",
                action=f"Buy {f.item.lower()} from the cheaper supplier, or ask {f.supplier} to match",
                why=f.detail, impact=f.annual_impact, confidence="High", difficulty="Low",
                evidence="Your own invoices", next_step=f"Send {f.supplier} the lower price and ask them to match it.",
                assumptions="Same product and quality; current monthly quantities continue.")
            continue
        if f.type == "price_decrease":
            continue
        if f.type == "shrinkflation":
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="shrink",
                action=f"Ask for the original {f.item.lower()} pack size or a matching price cut",
                why=f.detail + (f" {b['verdict_text']}" if b.get("verdict") in ("consistent", "exceeds", "below") else ""),
                impact=f.annual_impact, confidence="High", difficulty="Low",
                evidence="Your own invoices",
                next_step="Email the supplier quoting both invoice pack sizes.",
                assumptions="You keep buying the same weight per month.")
            continue
        # price_increase. Money is split in two: the part the market does not justify over the
        # matched period (a real negotiating target) and the part the market data cannot cover yet.
        row = st_idx.get(key)
        qty = float(row["monthly_qty"]) if row is not None else 0.0
        above = b.get("above_market_per_unit", 0.0) * qty * 12 if v == "exceeds" else 0.0
        unver = b.get("unverified_per_unit", 0.0) * qty * 12 if b.get("unverified") else 0.0
        ev = "Your invoices + " + ", ".join(d["title"] for d in b.get("drivers", []) if d.get("available"))
        if v in ("exceeds", "consistent") and unver > 0:
            lead = (b["verdict_text"] + " " + b["unverified_text"]) if v == "consistent" else \
                   (b["verdict_text"] + " On top of that, " + b["unverified_text"][0].lower() + b["unverified_text"][1:])
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="review",
                action=f"Ask {f.supplier.title()} to explain the latest {f.item.lower()} increase",
                why=lead, impact=above + unver, saving=above, counts_as_saving=above > 0,
                confidence="Medium", difficulty="Low", evidence=ev,
                next_step=f"Ask for a written breakdown of the {b['later_pct']:+.1%} move and a revised quote. "
                          f"Get one competing quote so you can compare.",
                assumptions=f"Amount at stake = the yearly cost of the part of your increase the market data "
                            f"does not explain or cannot yet cover, at current quantities. Not proof of overcharging.")
        elif v == "exceeds":
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="negotiate",
                action=f"Challenge the {f.item.lower()} increase using market data",
                why=b["verdict_text"], impact=above, saving=above, counts_as_saving=True,
                confidence="Medium" if b["comparability"] == "Medium" else "Low", difficulty="Low",
                evidence=ev,
                next_step=f"Ask {f.supplier.title()} to explain the gap and request a revised quote; get one "
                          f"competing quote.",
                assumptions="Amount = the yearly cost of the part of the increase above the market move, "
                            "if you negotiated it back.")
        elif v == "consistent":
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="pass_on",
                action=f"Plan for the {f.item.lower()} increase: it matches the market",
                why=b["verdict_text"], impact=f.annual_impact, saving=0.0, counts_as_saving=False,
                confidence="Medium", difficulty="Medium",
                evidence="Your invoices + market data",
                next_step="Check whether menu prices of dishes using this item still cover the higher cost.",
                assumptions="Arguing the price down is unlikely to work if the market moved the same way. "
                            "Amount = the full yearly cost of the increase, to absorb or price in.")
        elif v == "below":
            sc = scenario_6m.get(key)
            r0 = st_idx.get(key)
            future = float(r0["monthly_spend"]) * sc * 12 if (r0 is not None and sc) else 0.0
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="lock_in",
                action=f"Ask for a fixed {f.item.lower()} price for the next 3–6 months",
                why=b["verdict_text"], impact=max(future, 0.0), saving=0.0, counts_as_saving=False,
                confidence="Low", difficulty="Medium", evidence="Market data trend (scenario, not a prediction)",
                next_step="Request a fixed-price agreement. Only buy ahead if you have dry storage, the stock "
                          "won't spoil, and the cash isn't needed elsewhere.",
                assumptions="Impact = extra yearly cost if the supplier caught up on half of the market rise "
                            "it hasn't passed on yet. A scenario, not a prediction.")
        else:
            add(item=f.item, supplier=f.supplier, finding_id=f.id, kind="quote",
                action=f"Get a competing quote for {f.item.lower()}",
                why=f"{f.pct_change:+.1%} since {pd.Timestamp(f.change_date):%d %b}. "
                    + (b.get("verdict_text") or "No market benchmark available."),
                impact=f.annual_impact, saving=0.0, counts_as_saving=False, confidence="Low", difficulty="Low",
                evidence="Your own invoices only",
                next_step="Ask one other supplier for a quote and enter it on the Savings page.",
                assumptions="Amount = the full yearly cost of the increase. A competing quote is what turns "
                            "this into a saving you can actually bank.")

    for q in quotes or []:
        if not q.get("ok") or q["net_annual"] <= 0:
            continue
        complete = bool(q.get("url")) and bool(q.get("date"))
        same = q.get("match", "Same product") == "Same product"
        moq_ok = not q.get("moq") or q["moq"] <= q.get("packs_per_month", 0)
        conf = "Medium" if (complete and same) else "Low"
        reasons = []
        if not same:
            reasons.append("labelled a possible substitute, so check it works in your recipes")
        if not moq_ok:
            reasons.append(f"minimum order of {q['moq']:g} packs is above the "
                           f"{q.get('packs_per_month', 0):.1f} packs a month you use")
        if not complete:
            reasons.append("no link or date given, so the price isn't verifiable")
        add(item=q["item"], supplier=q["supplier"], finding_id="", kind="switch",
            action=f"Consider switching {q['item'].lower()} to {q['supplier']}",
            why=f"Landed {q['landed_price']:,.3f} per {q['unit']} against your current price "
                f"({q['landed_diff_pct']:+.1%})." + (" Caveats: " + "; ".join(reasons) + "." if reasons else ""),
            impact=q["net_annual"], saving=q["net_annual"] if moq_ok else 0.0,
            counts_as_saving=moq_ok, confidence=conf, difficulty="Medium",
            evidence=q.get("source", "Quote entered by you") + ("" if complete else " (no link/date given)"),
            next_step="Confirm quality, delivery days and minimum order before switching; order a trial batch.",
            assumptions="Net of the extra monthly and one-off costs you entered.")

    recs.sort(key=lambda r: -r["score"])
    for i, r in enumerate(recs, 1):
        r["rank"] = i
    return recs


def potential_savings(recs):
    """Best saving per item. Options for one item overlap, so they are never added together, and
    only money with evidence behind it counts (see counts_as_saving)."""
    best = {}
    for r in recs:
        if r.get("counts_as_saving") and r.get("saving", 0) > 0:
            best[r["item"]] = max(best.get(r["item"], 0.0), r["saving"])
    return float(sum(best.values())), {k: float(v) for k, v in best.items()}


def at_stake(recs):
    """Money flagged for review that is NOT counted as a saving: increases the market can't explain."""
    best = {}
    for r in recs:
        if not r.get("counts_as_saving") and r["kind"] in ("review", "quote") and r["impact"] > 0:
            best[r["item"]] = max(best.get(r["item"], 0.0), r["impact"])
    return float(sum(best.values())), {k: float(v) for k, v in best.items()}
