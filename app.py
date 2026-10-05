"""
MarginGuard — Streamlit app.

Run locally:   streamlit run app.py
"""
import os
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import llm
from analyze import fmt_unit_price
from pipeline import run

ROOT = Path(__file__).parent
SAMPLES = sorted((ROOT / "sample_invoices").glob("*.pdf"))

st.set_page_config(page_title="MarginGuard", page_icon="🛡️", layout="wide")

# --- Config: .env locally, Streamlit secrets when deployed ---------------------
llm.load_env()
try:
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass  # no secrets file locally — that's fine

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.mg-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;}
.mg-badge {display:inline-block; font-size: 12px; font-weight: 600; padding: 2px 9px; border-radius: 999px; margin-right: 6px;}
.mg-up {background: #fde2e1; color: #a12622;}
.mg-shrink {background: #fff0d6; color: #8a5300;}
.mg-gap {background: #dff1ff; color: #0b5394;}
.mg-down {background: #e0f5e4; color: #1e6b2f;}
.mg-money {font-size: 20px; font-weight: 700;}
.mg-muted {opacity: .7; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

BADGE = {"price_increase": ("Price increase", "mg-up"), "shrinkflation": ("Hidden increase: smaller pack", "mg-shrink"),
         "supplier_gap": ("Cheaper elsewhere", "mg-gap"), "price_decrease": ("Price drop", "mg-down")}

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("1 · Add invoices")
    use_samples = st.toggle(f"Use sample invoices ({len(SAMPLES)} invoices, 3 suppliers)", value=True,
                            help="Fictional café, April–September 2026. Price rises were planted on purpose.")
    uploads = st.file_uploader("…or upload your own invoice PDFs", type=["pdf"], accept_multiple_files=True,
                               disabled=use_samples)
    st.header("2 · Settings")
    threshold = st.slider("Flag price changes bigger than", 1, 10, 3, format="%d%%") / 100
    with st.expander("AI model"):
        key_in = st.text_input("Use your own API key (optional)", type="password",
                               help="Any OpenAI-compatible provider. Defaults to Featherless.")
        if key_in:
            os.environ["LLM_API_KEY"] = key_in
        cfg = llm.config()
        st.caption(f"Model: `{cfg['model']}`")
        st.caption("✅ API key set" if cfg["api_key"] else
                   "⚠️ No API key: only the cached sample invoices will work.")
    go = st.button("Analyze invoices", type="primary", use_container_width=True)

# --- Header ------------------------------------------------------------------
st.title("🛡️ MarginGuard")
st.markdown("##### Catch the supplier price rises that quietly eat your profit.")

if go:
    files = [(p.name, p) for p in SAMPLES] if use_samples else [(u.name, u) for u in (uploads or [])]
    if not files:
        st.warning("Upload at least one invoice PDF, or switch on the sample invoices.")
    else:
        bar = st.progress(0.0, "Starting…")
        try:
            st.session_state.result = run(files, threshold, progress=lambda f, m: bar.progress(min(f, 1.0), m))
            bar.empty()
        except Exception as e:
            bar.empty()
            st.error(f"Something went wrong: {e}")

res = st.session_state.get("result")

if not res:
    st.write("")
    cols = st.columns(4)
    steps = [("📄 Read", "AI reads invoices in any layout and pulls out every product, quantity and price."),
             ("🔗 Match", "AI recognises the same product across suppliers, renamed items and new pack sizes."),
             ("🧮 Measure", "Code compares price per kg / litre over time and works out the yearly cost."),
             ("✅ Act", "AI writes a plain-English action plan and a ready-to-send email to your supplier.")]
    for c, (t, d) in zip(cols, steps):
        c.markdown(f"<div class='mg-card'><b>{t}</b><br><span class='mg-muted'>{d}</span></div>", unsafe_allow_html=True)
    st.info("👈 Click **Analyze invoices** to try it with the sample invoices.")
    st.stop()

s, findings, plan = res["summary"], res["findings"], res["plan"]
advice = {a["finding_id"]: a for a in plan.get("actions", [])}

# --- Headline + KPIs ---------------------------------------------------------
if plan.get("headline"):
    st.success(f"**{plan['headline']}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Supplier spend", f"${s['monthly_spend']:,.0f}/mo",
          help=f"{s['invoices']} invoices from {s['suppliers']} suppliers, {s['date_from']} to {s['date_to']}")
k2.metric("Price rises found", s["n_increases"], help=f"Changes above {threshold:.0%}, measured per kg / litre / item")
k3.metric("Extra cost per year", f"${s['annual_extra_cost']:,.0f}",
          delta=f"{s['extra_cost_pct_of_spend']:.1%} of your spend", delta_color="inverse")
k4.metric("Could save by switching", f"${s['annual_switch_savings']:,.0f}/yr",
          help="Where another of your suppliers sells the same item for less")

st.divider()

# --- Alerts ------------------------------------------------------------------
left, right = st.columns([1.15, 1])
with left:
    st.subheader("What changed")
    if not findings:
        st.write("No price changes above your threshold. 🎉")
    for f in findings:
        label, cls = BADGE[f.type]
        if f.type == "supplier_gap":
            prices = f"You pay {fmt_unit_price(f.current_price, f.base_unit)} vs {fmt_unit_price(f.baseline_price, f.base_unit)}"
            money = f"save ${f.annual_impact:,.0f}/yr"
        else:
            prices = f"{fmt_unit_price(f.baseline_price, f.base_unit)} → {fmt_unit_price(f.current_price, f.base_unit)} since {pd.Timestamp(f.change_date):%d %b}"
            money = f"{'+' if f.annual_impact >= 0 else '−'}${abs(f.annual_impact):,.0f}/yr"
        a = advice.get(f.id)
        tip = f"<div style='margin-top:6px'>💡 <b>{escape(a['title'])}</b> — {escape(a['advice'])}</div>" if a else ""
        st.markdown(f"""<div class='mg-card'>
          <span class='mg-badge {cls}'>{label}</span><span class='mg-muted'>{escape(f.supplier)}</span>
          <div style='display:flex;justify-content:space-between;align-items:baseline;margin-top:4px'>
            <div><b style='font-size:17px'>{escape(f.item)}</b> &nbsp;<span class='mg-muted'>{f.pct_change:+.1%}</span></div>
            <div class='mg-money'>{money}</div></div>
          <div class='mg-muted'>{escape(prices)}. {escape(f.detail)}</div>{tip}</div>""", unsafe_allow_html=True)

with right:
    st.subheader("Price history")
    series = res["series"]
    items = list(dict.fromkeys([f.item for f in findings] + sorted(series["canonical_name"].unique())))
    pick = st.selectbox("Product", items, label_visibility="collapsed")
    d = series[series["canonical_name"] == pick].copy()
    unit = d["base_unit"].iloc[0]
    d["label"] = d["description"] + " · $" + d["unit_price"].map("{:.2f}".format) + " per pack"
    chart = (alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=70), interpolate="step-after")
             .encode(x=alt.X("date:T", title=None),
                     y=alt.Y("price_per_base:Q", title=f"$ per {unit}", scale=alt.Scale(zero=False)),
                     color=alt.Color("supplier:N", title=None, legend=alt.Legend(orient="bottom")),
                     tooltip=[alt.Tooltip("date:T"), "supplier", "label",
                              alt.Tooltip("price_per_base:Q", title=f"$ per {unit}", format=".3f")])
             .properties(height=320))
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Price per {unit}, so pack-size changes show up as real price changes.")

    email = plan.get("email")
    if email:
        st.subheader("Ready-to-send email")
        st.caption(f"To {email.get('supplier', 'your supplier')} · edit before sending")
        st.text_input("Subject", email.get("subject", ""))
        st.text_area("Message", email.get("body", ""), height=260)

st.divider()

# --- Transparency ------------------------------------------------------------
t1, t2, t3 = st.tabs(["📋 Extracted data", "⚠️ Checks & warnings", "🧮 How the numbers work"])
with t1:
    show = res["lines"][["date", "supplier", "invoice_number", "description", "canonical_name", "quantity",
                         "unit_price", "line_total", "pack_size", "pack_unit", "price_per_base", "base_unit"]]
    st.dataframe(show.sort_values(["date", "supplier"]), hide_index=True, use_container_width=True,
                 column_config={"date": st.column_config.DateColumn("Date"),
                                "canonical_name": "Matched product (AI)",
                                "price_per_base": st.column_config.NumberColumn("Price per base unit", format="%.3f")})
    findings_df = pd.DataFrame([f.to_dict() for f in findings])
    if not findings_df.empty:
        st.download_button("Download findings (CSV)", findings_df.to_csv(index=False), "marginguard_findings.csv")
with t2:
    st.write(f"AI product matching: {'✅ used' if res['ai_matched'] else '⚠️ unavailable — basic name matching used'}")
    st.write(f"AI action plan: {'✅ used' if res['ai_plan'] else '⚠️ unavailable — template advice used'}")
    st.write("Every invoice is re-checked with arithmetic (quantity × price = line total, lines add up to the "
             "subtotal). If the AI's reading fails a check, it is asked to correct itself once.")
    for w in res["warnings"]:
        st.warning(w)
    for e in res["errors"]:
        st.error(e)
    if not res["warnings"] and not res["errors"]:
        st.success(f"All {s['invoices']} invoices passed the arithmetic checks.")
with t3:
    st.markdown(f"""
- **Price per base unit** = pack price ÷ amount in the pack (kg, litres or items). This catches *shrinkflation*.
- **Change** = latest price vs the first price we saw for that product from that supplier. Changes under {threshold:.0%} are treated as normal noise.
- **Extra cost per year** = (current − original price per unit) × your average monthly quantity × 12.
- **Could save by switching** = where another of your suppliers charges less for the same product right now.
- The AI reads invoices, matches product names and writes advice. **All dollar figures are calculated in code**, not by the AI.
""")
