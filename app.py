"""
MarginGuard: Streamlit app.

Detect → Investigate → Forecast → Act
  Overview             what changed, what it costs, what to do first
  Cost Intelligence    the verdict and next steps first; evidence underneath
  Forecast & Scenarios what could happen next and what it does to margin
  Savings              supplier comparison workspace + action tracker
  Data & Checks        extracted invoice data, checks, sources, methods

Run locally:   streamlit run app.py
"""
import json
import os
from datetime import date
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

import altair as alt
import pandas as pd
import streamlit as st

import insights
import investigator
import llm
import market
from analyze import TO_BASE
from pipeline import run

ROOT = Path(__file__).parent
SAMPLES = sorted((ROOT / "sample_invoices").glob("*.pdf"))

# --- Design tokens: "Midnight Slate" ------------------------------------------
# The first five mirror .streamlit/config.toml, which is what styles Streamlit's own
# widgets. Everything here covers what the native theme can't reach: card surfaces,
# status pills and chart colours. Change a colour in both places or they will drift.
T = {
    "bg":       "#0B1120",   # main canvas
    "sidebar":  "#101827",   # one step lighter, so the sidebar reads as part of the app
    "surface":  "#151F30",   # cards, charts, tables
    "raised":   "#1C293B",   # hover and elevated controls
    "border":   "#263449",
    "grid":     "#1E2A3D",   # chart gridlines: quieter than borders
    "text":     "#E2E8F0",
    "text2":    "#94A3B8",
    "muted":    "#7589A1",   # lightened from #64748B so small caption text clears WCAG AA
    "accent":   "#38B6C9",
    "accent2":  "#59C8D8",
    "positive": "#5CC7AD",
    "warning":  "#F0B56B",
    "negative": "#EF8585",
    "slate":    "#8FA3BF",   # neutral series in charts
}


def tint(hex_colour, alpha):
    """Translucent version of a token, for pill backgrounds over a dark surface."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="MarginGuard", page_icon="🛡️", layout="wide")

# --- Config: .env locally, Streamlit secrets when deployed ---------------------
llm.load_env()
try:
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass  # no secrets file locally, that's fine

st.markdown(f"""
<style>
#MainMenu, footer, [data-testid="stStatusWidget"] {{visibility: hidden;}}
.block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}}
h1 {{font-size: 1.6rem !important; font-weight: 650 !important; letter-spacing: -0.01em;
     margin-bottom: .1rem; color: {T["text"]};}}
h3 {{font-size: 1.05rem !important; font-weight: 600 !important; margin-top: .4rem; color: {T["text"]};}}
hr, [data-testid="stDivider"] {{margin: 1rem 0; border-color: {T["border"]};}}

/* Financial figures: tabular numerals so columns of money line up */
[data-testid="stMetricValue"] {{font-size: 1.45rem; font-variant-numeric: tabular-nums;
                                color: {T["text"]};}}
[data-testid="stMetricLabel"] {{font-size: .82rem; color: {T["text2"]};}}
[data-testid="stMetricDelta"] {{font-size: .8rem;}}

/* Sidebar: a shade lighter than the canvas, joined by a hairline rather than a slab */
[data-testid="stSidebar"] {{background: {T["sidebar"]}; border-right: 1px solid {T["border"]};}}
[data-testid="stSidebarNav"] a {{border-radius: 6px;}}

/* Cards (st.container(border=True)) and expanders share one surface */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    border-radius: 10px;}}
[data-testid="stExpander"] details {{background: {T["surface"]}; border: 1px solid {T["border"]};
                                     border-radius: 10px;}}
[data-testid="stExpander"] summary:hover {{background: {T["raised"]};}}

.mg-kicker {{font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
             color: {T["muted"]};}}
.mg-pill {{display: inline-block; font-size: 11.5px; font-weight: 600; padding: 1px 8px;
           border-radius: 4px; border: 1px solid; margin: 0 6px 4px 0;}}
.mg-red   {{color: {T["negative"]}; background: {tint(T["negative"], .12)};
            border-color: {tint(T["negative"], .35)};}}
.mg-amber {{color: {T["warning"]};  background: {tint(T["warning"], .12)};
            border-color: {tint(T["warning"], .35)};}}
.mg-green {{color: {T["positive"]}; background: {tint(T["positive"], .12)};
            border-color: {tint(T["positive"], .35)};}}
.mg-blue  {{color: {T["accent2"]};  background: {tint(T["accent"], .12)};
            border-color: {tint(T["accent"], .35)};}}
.mg-grey  {{color: {T["text2"]};    background: {tint(T["text2"], .08)};
            border-color: {T["border"]};}}
.mg-big {{font-size: 1.3rem; font-weight: 650; color: {T["text"]}; line-height: 1.2;
          font-variant-numeric: tabular-nums;}}
.mg-muted {{color: {T["text2"]}; font-size: .86rem;}}
.mg-cite {{font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: {T["accent2"]};
           background: {tint(T["accent"], .10)}; border-radius: 3px; padding: 0 4px; margin-left: 2px;}}
.mg-verdict {{font-size: 1rem; line-height: 1.5; margin: .2rem 0 .4rem 0; color: {T["text"]};}}
.mg-step {{display: inline-block; font-size: 10.5px; font-weight: 700; letter-spacing: .06em;
           text-transform: uppercase; color: {T["accent2"]}; background: {tint(T["accent"], .12)};
           border: 1px solid {tint(T["accent"], .35)}; border-radius: 4px; padding: 1px 7px;
           margin-right: 8px; vertical-align: 2px;}}
.mg-steph {{font-size: 1.05rem; font-weight: 600; color: {T["text"]}; display: inline;}}
.mg-stepq {{color: {T["text2"]}; font-size: .88rem; margin: .25rem 0 .6rem 0;}}
</style>
""", unsafe_allow_html=True)

TYPE_LABEL = {"price_increase": ("Price increase", "red"), "shrinkflation": ("Smaller pack, same price", "amber"),
              "supplier_gap": ("Cheaper elsewhere", "blue"), "price_decrease": ("Price drop", "green")}
STATUS = {"review": ("Review needed", "red"), "exceeds": ("Above market", "red"),
          "consistent": ("In line with market", "green"), "below": ("Market rising faster", "amber"),
          "no_benchmark": ("No market benchmark", "grey"), "insufficient": ("Not enough data", "grey")}
CONF_TONE = {"High": "green", "Medium": "blue", "Low": "grey"}
TRACK = ["Not started", "Saved", "Under review", "Implemented", "Dismissed"]
TRACK_TONE = {"Not started": "grey", "Saved": "blue", "Under review": "amber", "Implemented": "green",
              "Dismissed": "grey"}

ss = st.session_state
for k, v in {"overrides": {}, "quotes": [], "pass_through": 0.5, "revenue": 0.0, "cogs": 0.0, "sel": None,
             "horizon": 3, "actions": {}, "shock": 0.0, "fx": 0.0, "investigations": {}}.items():
    ss.setdefault(k, v)


# --- Formatting --------------------------------------------------------------
def cur():
    """Currency symbol for the user's own invoice amounts."""
    return insights.sym(ss.get("currency", "SGD"))


def money(x, dp=0):
    return f"−{cur()}{abs(x):,.{dp}f}" if x < 0 else f"{cur()}{x:,.{dp}f}"


def unit_price(p, unit):
    return f"{money(p, 3)}/{unit}" if abs(p) < 1 else f"{money(p, 2)}/{unit}"


def esc(text):
    """Streamlit renders markdown, where $...$ is a maths formula: escape money amounts."""
    return str(text).replace("$", r"\$")


def pill(text, tone="grey"):
    return f"<span class='mg-pill mg-{tone}'>{escape(str(text))}</span>"


def md(html):
    st.markdown(html, unsafe_allow_html=True)


def say(text, fn=None):
    """Write prose that may contain currency amounts."""
    (fn or st.markdown)(esc(text))


# --- Cached loaders ----------------------------------------------------------
@st.cache_data(show_spinner="Reading the sample invoices…")
def load_demo(threshold):
    return run([(p.name, p) for p in SAMPLES], threshold)


@st.cache_data(ttl=6 * 3600, max_entries=2, show_spinner="Loading market data…")
def load_market():
    return market.get_all()


RETRY_FALLBACK_MIN = 10   # a failed live download is retried automatically after this many minutes


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    md("<div class='mg-kicker'>Data</div>")
    mode = st.radio("Invoices", ["Demo: sample café (synthetic)", "Upload my invoices"], key="mode",
                    label_visibility="collapsed",
                    help="The demo uses 24 fictional invoices (April–September 2026) with planted price changes.")
    demo = mode.startswith("Demo")
    uploads = None
    if not demo:
        uploads = st.file_uploader("Invoice PDFs", type=["pdf"], accept_multiple_files=True)
    run_uploads = st.button("Analyze my invoices", type="primary", use_container_width=True, disabled=demo)

    md("<div class='mg-kicker'>Settings</div>")
    threshold = st.slider("Flag price changes bigger than", 1, 10, 3, format="%d%%", key="thr") / 100
    currency = st.selectbox("Invoice currency", ["SGD", "USD", "Other"], key="currency",
                            help="Market prices are published in US$. For SGD they are converted using the "
                                 "Federal Reserve's monthly SGD/USD rate.")
    with st.expander("AI model"):
        key_in = st.text_input("Your own API key (optional)", type="password",
                               help="Any OpenAI-compatible provider. Defaults to Featherless.")
        if key_in:
            os.environ["LLM_API_KEY"] = key_in
        cfg = llm.config()
        st.caption(f"Model: `{cfg['model']}`")
        st.caption("API key set." if cfg["api_key"] else "No API key: only the sample invoices will work.")
    refresh_mk = st.button("Refresh market data", use_container_width=True,
                           help="Downloads the latest published prices from FRED now, bypassing the 6-hour cache. "
                                "Limited to once a minute.")
    mk_msg = st.empty()

# --- Load invoices + market data ---------------------------------------------
load_error = None
if demo:
    try:
        res = load_demo(threshold)
    except Exception as e:
        res, load_error = None, f"The sample analysis could not be loaded: {e}"
else:
    if run_uploads:
        files = [(u.name, u) for u in (uploads or [])]
        if not files:
            st.sidebar.warning("Upload at least one invoice PDF first.")
        else:
            bar = st.sidebar.progress(0.0, "Starting…")
            try:
                ss["upload_res"] = run(files, threshold, progress=lambda f, m: bar.progress(min(f, 1.0), m))
                ss["upload_thr"] = threshold
            except Exception as e:
                load_error = f"Something went wrong: {e}"
            bar.empty()
    res = ss.get("upload_res")
    if res and ss.get("upload_thr") != threshold:
        st.sidebar.caption("Threshold changed: click Analyze again to apply it.")

if refresh_mk:
    ok, wait = market.refresh_allowed()
    if ok:
        load_market.clear()
        ss["mk_msg"] = ("refresh", None)
    else:
        ss["mk_msg"] = ("wait", wait)
MK = load_market()
_checked = min(pd.Timestamp(m["checked_at"].replace(" UTC", "")) for _, m in MK.values())
if (any(m["status"] != "live" for _, m in MK.values())
        and pd.Timestamp.now(tz="UTC").tz_localize(None) - _checked > pd.Timedelta(minutes=RETRY_FALLBACK_MIN)):
    # Don't let one failed download pin cached data for the whole 6-hour cache window.
    load_market.clear()
    MK = load_market()
_msg = ss.pop("mk_msg", None)
if _msg and _msg[0] == "refresh":
    (mk_msg.success if all(m["status"] == "live" for _, m in MK.values()) else mk_msg.warning)(
        market.summary(MK) + ".")
elif _msg:
    mk_msg.info(f"Market data was refreshed moments ago. Try again in {_msg[1]}s.")
MK_FP = market.fingerprint(MK)

DATA_STATUS = {"live": ("Live verified", "green"), "cached": ("Cached data", "grey"),
               "demo": ("Demo data", "amber"), "unavailable": ("Unavailable", "grey")}


def data_badge(status, outdated=False):
    label, tone = DATA_STATUS.get(status, DATA_STATUS["unavailable"])
    if status == "cached" and outdated:
        label, tone = "Cached data, outdated", "amber"
    return pill(label, tone)


def status_text(m):
    """Short table cell: status plus why it isn't live."""
    label = DATA_STATUS.get(m["status"], DATA_STATUS["unavailable"])[0]
    return label + (" (outdated)" if m.get("outdated") else "")


def market_status():
    metas = [m for _, m in MK.values()]
    st_ = [m["status"] for m in metas]
    latest = max((m["last_obs"] for m in metas if m["last_obs"] and m["kind"] != "fx"), default=None)
    obs = f", latest month {pd.Timestamp(latest):%b %Y}" if latest else ""
    stale = any(m.get("outdated") for m in metas)
    if all(x == "live" for x in st_):
        return pill(f"Market data: live verified{obs}", "green")
    if any(x == "live" for x in st_):
        return pill(f"Market data: partly live, partly cached{obs}", "amber" if stale else "grey")
    if any(x == "cached" for x in st_):
        return pill(f"Market data: cached{obs} (live download failed)", "amber" if stale else "grey")
    return pill("Market data unavailable", "red")


# --- Derived analytics -------------------------------------------------------
def derive(res):
    ser, findings = res["series"], res["findings"]
    stats = insights.item_stats(res["lines"])
    cat = dict(zip(res["lines"]["canonical_name"], res["lines"].get("category", pd.Series(dtype=object))))
    bench = {}
    for f in findings:
        g = ser[(ser["canonical_name"] == f.item) & (ser["supplier"] == f.supplier)]
        prof = ss["overrides"].get(f.item) or insights.classify(f.item, cat.get(f.item))
        bench[(f.item, f.supplier)] = insights.benchmark(g, prof, MK, ss.get("currency", "SGD"))
    priced = []
    for q in ss["quotes"]:
        row = stats[(stats["item"] == q["item"]) & (stats["supplier"] == q["vs_supplier"])]
        if row.empty:
            continue
        r = row.iloc[0]
        out = insights.quote_savings(r["current_price"], r["base_unit"], q["pack_price"], q["pack_size"],
                                     q["pack_unit"], r["monthly_qty"], q["delivery"], q["one_off"])
        priced.append({**q, **out, "current_price": r["current_price"], "monthly_qty": r["monthly_qty"]})
    recs = insights.recommendations(findings, bench, stats, quotes=priced, pass_through=ss["pass_through"])
    saved, by_item = insights.potential_savings(recs)
    stake, _ = insights.at_stake(recs)
    return {"stats": stats, "bench": bench, "recs": recs, "savings_total": saved, "savings_by_item": by_item,
            "at_stake": stake, "quotes": priced, "cat": cat}


D = derive(res) if res else None


# --- Shared UI ---------------------------------------------------------------
def header(title, subtitle):
    md("<div class='mg-kicker'>MarginGuard</div>")
    st.title(title)
    st.caption(subtitle)
    md((pill("Demo · synthetic invoices", "blue") if demo else pill("Your invoices", "blue")) + market_status())
    if load_error:
        st.error(load_error)


def need_data():
    if not res:
        st.info("Choose the demo data, or upload invoices and click **Analyze my invoices** in the sidebar.")
        st.stop()


def key_of(f):
    return f"{f.item}||{f.supplier}"


def split_key(k):
    item, supplier = k.split("||", 1)
    return item, supplier


def label_key(k):
    item, sup = split_key(k)
    return f"{item}  ·  {sup.title()}"


def finding_for(k):
    item, supplier = split_key(k)
    return next((f for f in res["findings"] if f.item == item and f.supplier == supplier), None)


def item_picker(label, keys, widget_key):
    """One selected item shared across pages (held outside the widget so it survives navigation)."""
    if not keys:
        return None
    idx = keys.index(ss["sel"]) if ss["sel"] in keys else 0
    choice = st.selectbox(label, keys, index=idx, key=widget_key, format_func=label_key)
    ss["sel"] = choice
    return choice


def flagged_keys():
    return list(dict.fromkeys(key_of(f) for f in res["findings"] if f.type != "price_decrease"))


def goto(k, page):
    ss["sel"] = k
    st.switch_page(page)


def supplier_series(item, supplier):
    s = res["series"]
    return s[(s["canonical_name"] == item) & (s["supplier"] == supplier)].sort_values("date")


def status_of(b):
    """Headline status: the matched-period verdict, raised to 'review' when a later move is unexplained."""
    v = b.get("verdict", "insufficient")
    return "review" if v in ("consistent", "exceeds") and b.get("unverified") else v


def rec_for(item, supplier):
    return [r for r in D["recs"] if r["item"] == item and r["supplier"] == supplier]


def impact_label(kind):
    return {"negotiate": "could be negotiated back", "switch": "possible saving", "shrink": "possible saving",
            "review": "at stake, needs an explanation", "pass_on": "cost to absorb or price in",
            "lock_in": "cost you may avoid", "quote": "cost of the increase"}.get(kind, "")


def action_id(r):
    return f"{r['item']}||{r['supplier']}||{r['kind']}"


def tracked(r):
    return ss["actions"].get(action_id(r), {})


# --- Charts ------------------------------------------------------------------
def dark(chart):
    """One Altair theme for every chart: dark surface, quiet grid, legible labels.

    Applied at the top level so layered charts inherit it too. Kept as a function
    rather than a registered theme so it works the same across Altair versions.
    """
    return (chart
            .configure(background=T["surface"])
            .configure_view(strokeWidth=0, fill=T["surface"])
            .configure_axis(domainColor=T["border"], tickColor=T["border"], gridColor=T["grid"],
                            labelColor=T["text2"], titleColor=T["text2"], labelFontSize=11,
                            titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=T["text2"], titleColor=T["text2"], labelFontSize=11,
                              symbolStrokeWidth=2, padding=0)
            .configure_title(color=T["text"]))


def price_history_chart(item, height=260):
    d = res["series"][res["series"]["canonical_name"] == item].copy()
    unit = d["base_unit"].iloc[0]
    d["label"] = d["description"] + " · " + d["unit_price"].map(lambda v: f"{cur()}{v:,.2f}") + " per pack"
    ch = (alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=45), interpolate="step-after", strokeWidth=2)
          .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("price_per_base:Q", title=f"{cur()} per {unit}", scale=alt.Scale(zero=False)),
                    color=alt.Color("supplier:N", title=None, legend=alt.Legend(orient="bottom"),
                                    scale=alt.Scale(range=[T["accent"], T["warning"], T["slate"]])),
                    tooltip=[alt.Tooltip("date:T", title="Invoice date"), alt.Tooltip("supplier:N", title="Supplier"),
                             alt.Tooltip("label:N", title="Line"),
                             alt.Tooltip("price_per_base:Q", title=f"{cur()} per {unit}", format=".3f")])
            .properties(height=height))
    return dark(ch)


def benchmark_chart(g, b, height=300):
    """Your price vs the market proxy, both indexed to 100 at the start of the matched period."""
    start, end = g["date"].min(), g["date"].max()
    sup = g[["date", "price_per_base"]].copy()
    sup["index"] = sup["price_per_base"] / sup["price_per_base"].iloc[0] * 100
    sup["series"] = "Your price"
    frames = [sup[["date", "index", "series"]]]
    m0 = market.month_start(start)
    for d in b["drivers"] + b["context"]:
        if not d.get("available") or d["id"] == "EXSIUS":
            continue
        df = MK[d["id"]][0]
        v0 = market.value_at_month(df, m0)
        sub = df[(df["date"] >= m0) & (df["date"] <= end)].copy()
        if v0 and len(sub):
            sub["index"] = sub["value"] / v0 * 100
            sub["series"] = d["meta"]["short"].capitalize() + " (market proxy, US$)"
            frames.append(sub[["date", "index", "series"]])
    data = pd.concat(frames, ignore_index=True)
    names = list(dict.fromkeys(data["series"]))
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom"),
                      scale=alt.Scale(domain=names,
                                      range=[T["accent"], T["slate"], T["warning"], T["muted"]][: len(names)]))
    tip = [alt.Tooltip("date:T", title="Date"), alt.Tooltip("series:N", title="Series"),
           alt.Tooltip("index:Q", title="Index (start = 100)", format=".1f")]
    yaxis = alt.Y("index:Q", title="Index (start of period = 100)", scale=alt.Scale(zero=False))
    layers = [alt.Chart(pd.DataFrame({"index": [100.0]})).mark_rule(color=T["border"]).encode(y=yaxis),
              alt.Chart(data[data["series"] != "Your price"])
              .mark_line(strokeWidth=1.6, strokeDash=[5, 3], point=alt.OverlayMarkDef(size=25))
              .encode(x=alt.X("date:T", title=None), y=yaxis, color=color, tooltip=tip),
              alt.Chart(data[data["series"] == "Your price"])
              .mark_line(interpolate="step-after", strokeWidth=2.5, point=alt.OverlayMarkDef(size=40))
              .encode(x=alt.X("date:T", title=None), y=yaxis, color=color, tooltip=tip)]
    if b.get("bench_end") is not None and pd.Timestamp(b["bench_end"]) < end:
        cut = pd.DataFrame({"date": [pd.Timestamp(b["bench_end"])]})
        layers.append(alt.Chart(cut).mark_rule(color=T["negative"], strokeDash=[3, 3], opacity=0.8)
                      .encode(x=alt.X("date:T", title=None),
                              tooltip=[alt.Tooltip("date:T", title="Market data ends")]))
    return dark(alt.layer(*layers).properties(height=height))


# ============================================================================ PAGES

def page_overview():
    header("Overview", "Know why your supplier costs rise, what could come next, and what to do about it.")
    need_data()
    s = res["summary"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Supplier spend", f"{money(s['monthly_spend'])}/mo",
              help=f"{s['invoices']} invoices from {s['suppliers']} suppliers, {s['date_from']} to {s['date_to']}.")
    k2.metric("Cost exposure", f"{money(s['annual_extra_cost'])}/yr",
              delta=f"{s['extra_cost_pct_of_spend']:.1%} of spend", delta_color="inverse",
              help="Yearly cost of all price rises found, at your current quantities.")
    k3.metric("Savings identified", f"{money(D['savings_total'])}/yr",
              help="Best evidence-backed option per item, never double-counted: a cheaper supplier you already "
                   "use, a shrunk pack to recover, an increase above the market, or a quote you entered. "
                   "Before switching costs.")
    k4.metric("Flagged for review", f"{money(D['at_stake'])}/yr",
              help="Increases the market data can't explain or doesn't yet cover. Not counted as savings: "
                   "they need an explanation from your supplier first.")
    if ss["revenue"] > 0:
        say(f"At {money(ss['revenue'])}/month revenue, the rises found cost "
            f"{s['annual_extra_cost'] / (ss['revenue'] * 12):.1%} of revenue, straight off gross margin unless "
            f"selling prices change.", st.caption)

    st.subheader("Do these first")
    st.caption("Ranked by yearly value × confidence in the evidence × how easy it is.")
    for r in D["recs"][:3]:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.4])
            c1.markdown(f"**{r['rank']}. {esc(r['action'])}**")
            t = tracked(r).get("status")
            c1.markdown(pill(f"Confidence: {r['confidence']}", CONF_TONE[r["confidence"]])
                        + pill(f"Effort: {r['difficulty']}", "grey") + pill(r["evidence"], "grey")
                        + (pill(t, TRACK_TONE[t]) if t and t != "Not started" else ""), unsafe_allow_html=True)
            say(f"Next step: {r['next_step']}", c1.caption)
            c2.markdown(f"<div class='mg-big'>{escape(money(r['impact']))}/yr</div>"
                        f"<div class='mg-muted'>{escape(impact_label(r['kind']))}</div>", unsafe_allow_html=True)
            k = f"{r['item']}||{r['supplier']}"
            if finding_for(k) and c2.button("Investigate", key=f"ov_{r['rank']}", use_container_width=True):
                goto(k, PAGES["cost"])
    st.page_link(PAGES["savings"], label="All actions, supplier comparison and tracker",
                 icon=":material/arrow_forward:")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("What changed")
        rows = []
        for f in res["findings"]:
            b = D["bench"].get((f.item, f.supplier), {})
            rows.append({"Item": f.item, "Supplier": f.supplier.title(), "Signal": TYPE_LABEL[f.type][0],
                         "Change": f"{f.pct_change:+.1%}", "Per year": money(f.annual_impact),
                         "Market check": STATUS[status_of(b)][0]})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.write("No price changes above your threshold.")
    with right:
        st.subheader("Where the money goes")
        sp = insights.spend_by_group(res["lines"], ss["overrides"])
        st.altair_chart(dark(alt.Chart(sp).mark_bar(color=T["accent"], cornerRadiusEnd=2)
                        .encode(x=alt.X("monthly_spend:Q", title=f"{cur()} per month"),
                                y=alt.Y("profile:N", sort="-x", title=None),
                                tooltip=[alt.Tooltip("profile:N", title="Category"),
                                         alt.Tooltip("monthly_spend:Q", title="Per month", format=",.0f")])
                        .properties(height=230)), use_container_width=True)
        lines = res["lines"].copy()
        lines["month"] = lines["date"].dt.to_period("M").dt.to_timestamp()
        bym = lines.groupby("month", as_index=False)["line_total"].sum()
        st.altair_chart(dark(alt.Chart(bym).mark_bar(color=T["slate"], cornerRadiusEnd=2)
                        .encode(x=alt.X("month:T", title=None, timeUnit="yearmonth"),
                                y=alt.Y("line_total:Q", title=f"Invoiced {cur()} per month"),
                                tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"),
                                         alt.Tooltip("line_total:Q", title="Invoiced", format=",.0f")])
                        .properties(height=150)), use_container_width=True)
        st.caption("Monthly totals follow delivery timing, so a dip or spike isn't always a price change.")


# --- AI Procurement Investigator ----------------------------------------------
AI_STATUS = {"review_needed": ("AI: review needed", "red"), "act_now": ("AI: act now", "red"),
             "in_line": ("AI: in line with market", "green"),
             "insufficient_evidence": ("AI: not enough evidence", "grey")}
PROV = {"invoice": ("Your invoices", "blue"), "live": ("Live verified", "green"), "cached": ("Cached data", "grey"),
        "snapshot": ("Cached data", "grey"), "manual": ("Entered by you", "blue"), "synthetic": ("Demo data", "amber"),
        "derived": ("Calculated", "grey"), "unavailable": ("Unavailable", "grey")}
COMP = {"comparable": ("Comparable", "green"), "potentially_comparable": ("Possibly comparable", "amber"),
        "not_comparable": ("Not comparable", "red"), "insufficient_information": ("Not enough information", "grey"),
        "not_checked": ("Match not checked", "grey")}
TOOL_NAME = {"price_history": "Price history", "market_benchmark": "Market benchmark",
             "market_context": "Fuel and currency", "supplier_options": "Supplier options",
             "assess_comparability": "Product comparability", "calculate_savings": "Savings calculator",
             "finish": "Finish", "stop": "Stopped"}


def cites(ids):
    return "".join(f"<span class='mg-cite'>{escape(i)}</span>" for i in ids or [])


def inv_signature(b):
    """What an investigation depended on, so a stale one can be spotted after settings change."""
    return [b["profile"], ss.get("currency", "SGD"), len(ss["quotes"]), MK_FP]


def run_label(calls):
    live, cached = calls.get("live", 0), calls.get("cached", 0)
    if live and not cached:
        return f"Live AI run ({live} model calls)", "green"
    if cached and not live:
        return f"Replayed from saved AI responses ({cached} calls)", "blue"
    return f"Mixed: {live} live, {cached} replayed AI calls", "blue"


def ai_section(k, item, supplier, f, b):
    st.subheader("Investigate")
    saved = ss["investigations"].get(k)
    with st.container(border=True):
        a, bcol = st.columns([4, 1.4])
        a.markdown("An AI analyst works through this price rise step by step. It chooses which check to run next "
                   "from six tools (your invoices, market benchmarks, fuel and currency, supplier options, "
                   "product comparability, savings), then writes findings that cite the evidence behind them.")
        a.caption("Money is always calculated by MarginGuard, never by the AI. News and online catalogues are "
                  "not connected, so the AI cannot cite them.")
        go_btn = bcol.button("Run again" if saved else "Investigate", key=f"inv_{k}", type="primary",
                             use_container_width=True)
        if saved and bcol.button("Clear", key=f"invclr_{k}", use_container_width=True):
            ss["investigations"].pop(k, None)
            st.rerun()
    if go_btn:
        case = investigator.Case(item=item, supplier=supplier, series=res["series"], lines=res["lines"],
                                 stats=D["stats"], finding=f, market=MK, profile=b["profile"],
                                 currency=ss.get("currency", "SGD"), quotes=list(ss["quotes"]))
        with st.status("Investigating…", expanded=True) as box:
            try:
                inv = investigator.investigate(case, on_step=lambda msg: (box.update(label=msg), box.write(msg)))
            except llm.LLMError as e:
                box.update(label="The AI model could not be reached", state="error")
                st.error(f"The investigation needs the AI model, which could not be reached: {e} Add an API key "
                         f"under **AI model** in the sidebar. Investigations saved with the app can be replayed "
                         f"without a key; this one has not been saved yet.")
                return
            box.update(label=f"Investigation complete · {len(inv.evidence)} pieces of evidence", state="complete",
                       expanded=False)
        ss["investigations"][k] = {"inv": inv.to_dict(), "sig": inv_signature(b)}
        saved = ss["investigations"][k]
    if saved:
        render_investigation(saved["inv"], k, stale=saved["sig"] != inv_signature(b))


def render_investigation(inv, k, stale=False):
    evid = {e["id"]: e for e in inv["evidence"]}
    lab, tone = run_label(inv["llm_calls"])
    s_lab, s_tone = AI_STATUS.get(inv["status"], AI_STATUS["insufficient_evidence"])
    steps = sum(1 for t in inv["trace"] if t["status"] == "ok" and t["tool"] not in ("finish",))
    md(pill(s_lab, s_tone) + pill(lab, tone) + pill(f"{steps} checks · {len(evid)} evidence items", "grey")
       + pill(f"Run {inv['started']}", "grey"))
    if stale:
        st.warning("The benchmark, currency, quotes or market data have changed since this investigation ran, "
                   "so its findings may be out of date. Run it again to recalculate them.")
    for c in inv["validation"]["conflicts"]:
        st.warning(f"Conflicting evidence: {c} The calculated status above stands; check the evidence below.")

    # A. Executive summary
    st.markdown("**A. Summary**")
    md(f"<div class='mg-verdict'>{escape(inv['assessment']) or 'The AI gave no summary.'}</div>")

    # B. What we found
    st.markdown("**B. What we found**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("Verified by evidence")
        for x in inv["verified_facts"] or [{"text": "Nothing the AI could verify.", "evidence": []}]:
            md(f"<div class='mg-muted'>• {escape(x['text'])} {cites(x['evidence'])}</div>")
    with c2:
        st.markdown("Possible, not verified")
        for x in inv["possible_explanations"] or [{"text": "None suggested.", "evidence": []}]:
            flag = " <span class='mg-pill mg-amber'>figure not in evidence</span>" if x.get("unsupported") else ""
            md(f"<div class='mg-muted'>• {escape(x['text'])} {cites(x['evidence'])}{flag}</div>")
    with c3:
        st.markdown("Missing evidence")
        for x in inv["missing_evidence"] or ["Nothing listed."]:
            md(f"<div class='mg-muted'>• {escape(x)}</div>")

    # C. Market context
    st.markdown("**C. Market context**")
    mkt = [e for e in inv["evidence"] if e["tool"] in ("market_benchmark", "market_context")]
    if not mkt:
        st.caption("The AI did not check market data for this item.")
    details = []
    for e in mkt:
        pv = e["content"].get("provenance")
        badge = (data_badge(pv["status"], pv.get("outdated")) if pv and e["provenance"] != "unavailable"
                 else pill(*PROV[e["provenance"]]))
        fresh = (f"<div class='mg-muted' style='margin:-2px 0 6px 0;font-size:.8rem'>{escape(pv['freshness'])}</div>"
                 if pv and pv.get("freshness") and e["provenance"] != "unavailable" else "")
        md(f"<div class='mg-muted'>{cites([e['id']])} <b>{escape(e['title'])}</b> {badge} "
           f"{escape(e['summary'])}</div>{fresh}")
        if pv:
            details.append({"ID": e["id"], "Series": e["title"], "Status": DATA_STATUS.get(pv["status"], ("?",))[0]
                            + (" (outdated)" if pv.get("outdated") else ""),
                            "Latest observation": pv["last_obs"], "Period used": e["observed"] or "—",
                            "Observations held": f"{pv['first_obs']} to {pv['last_obs']}" if pv["first_obs"] else "—",
                            "Downloaded": pv["retrieved_at"] or "—", "Checked": pv["checked_at"] or "—",
                            "Currency": pv["currency"], "Unit": pv["unit"], "Source": pv["source"],
                            "Cached copy": pv["cache_source"] or "—",
                            "Why not live": pv["fallback_reason"] if pv["status"] != "live" else "—",
                            "Link": pv["source_url"]})
    if details:
        with st.expander("Source details"):
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True,
                         column_config={"Link": st.column_config.LinkColumn(display_text="Open on FRED")})
            st.caption("'Downloaded' is when MarginGuard fetched the data; 'Latest observation' is the month the "
                       "price refers to. Each comparison stops at the latest observation: supplier changes after "
                       "it are reported as not yet checkable, never explained by older market data. All are raw "
                       "commodity prices, not the wholesale product you buy.")

    # D. Alternatives
    st.markdown("**D. Alternatives**")
    opts = [e for e in inv["evidence"] if e["tool"] == "supplier_options"]
    if not opts:
        st.caption("The AI did not look for alternative suppliers.")
    rows = []
    for e in opts:
        if e["provenance"] == "unavailable":
            say(e["summary"], st.caption)
            continue
        comp = next((x["content"] for x in inv["evidence"] if x["tool"] == "assess_comparability"
                     and x["content"].get("option_id") == e["id"]), None)
        sav = next((x["content"] for x in inv["evidence"] if x["tool"] == "calculate_savings"
                    and x["content"].get("option_id") == e["id"]), None)
        cls = comp["classification"] if comp else ("comparable" if e["content"].get("kind") == "invoice"
                                                    else "not_checked")
        saving = "—"
        if sav and sav.get("ok"):
            saving = "not claimed" if cls == "not_comparable" else f"{money(sav['net_annual'])}/yr"
        rows.append({"ID": e["id"], "Option": e["title"], "Source": PROV[e["provenance"]][0],
                     "Product match": COMP[cls][0], "Saving": saving,
                     "Basis": (sav or {}).get("basis", "not calculated") if saving != "not claimed" else "—",
                     "Why": (comp or {}).get("explanation", "")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # E. Actions
    st.markdown("**E. Recommended actions**")
    if not inv["actions"]:
        st.caption("The AI recommended no actions.")
    item, supplier = split_key(k)
    for i, a in enumerate(inv["actions"]):
        aid = f"{item}||{supplier}||ai:{a['type']}:{a.get('option_id') or ''}"
        with st.container(border=True):
            left, right = st.columns([4, 1.4])
            left.markdown(f"**{esc(a['title'] or a['type'].replace('_', ' ').capitalize())}**")
            md_txt = f"<div class='mg-muted'>{escape(a['why'])} {cites(a['evidence'])}</div>"
            left.markdown(md_txt, unsafe_allow_html=True)
            if a.get("unsupported"):
                left.caption("Contains a figure that is not in the evidence: "
                             + ", ".join(esc(x) for x in a["unsupported"]))
            if a["impact"] is not None:
                right.markdown(f"<div class='mg-big'>{escape(money(a['impact']))}/yr</div>"
                               f"<div class='mg-muted'>{escape(a['impact_label'])}</div>", unsafe_allow_html=True)
            else:
                right.markdown(f"<div class='mg-muted'>{escape(a['impact_label'].capitalize())}</div>",
                               unsafe_allow_html=True)
            if aid in ss["actions"]:
                right.markdown(pill(f"In tracker: {ss['actions'][aid]['status']}", "blue"), unsafe_allow_html=True)
            elif right.button("Save to tracker", key=f"ais_{i}_{aid}", use_container_width=True):
                is_saving = a["impact"] is not None and a["impact_label"].startswith("saving")
                ss["actions"][aid] = {"item": item, "supplier": supplier, "action": a["title"] or a["type"],
                                      "status": "Saved",
                                      "expected_monthly": round(a["impact"] / 12, 2) if is_saving else 0.0,
                                      "actual_monthly": 0.0, "date": date.today().isoformat(),
                                      "note": "From AI investigation. Evidence: " + ", ".join(a["evidence"])}
                st.rerun()
    if inv["actions"]:
        st.page_link(PAGES["savings"], label="Open the action tracker on Savings", icon=":material/arrow_forward:")

    # F. Evidence and methodology
    st.markdown("**F. Evidence and method**")
    with st.expander(f"All evidence ({len(evid)} items)"):
        st.dataframe(pd.DataFrame([{"ID": e["id"], "From": TOOL_NAME.get(e["tool"], e["tool"]),
                                    "Evidence": e["title"], "Data": PROV[e["provenance"]][0],
                                    "Period": e["observed"], "Detail": e["summary"], "Source": e["source"],
                                    "Retrieved": e["retrieved"], "Link": e["url"] or None}
                                   for e in inv["evidence"]]),
                     hide_index=True, use_container_width=True,
                     column_config={"Link": st.column_config.LinkColumn(display_text="Open")})
    with st.expander("How the AI investigated (step by step)"):
        st.dataframe(pd.DataFrame([{"Step": t["step"], "Check": TOOL_NAME.get(t["tool"], t["tool"]),
                                    "Arguments": ", ".join(f"{a}={v}" for a, v in t["args"].items()) or "—",
                                    "AI's reason": t["reason"], "Result": t["status"],
                                    "Evidence": ", ".join(t["evidence"]) or "—"} for t in inv["trace"]]),
                     hide_index=True, use_container_width=True)
        st.caption(f"The AI chose each check after seeing the evidence so far, up to {investigator.MAX_STEPS} "
                   f"checks. Unknown tools, missing arguments and repeats are rejected and the AI is told why. "
                   f"Took {inv['seconds']:.0f}s.")
    with st.expander("Checks on the AI's answer"):
        v = inv["validation"]
        st.markdown("- Citations to evidence that doesn't exist: "
                    + (", ".join(v["dropped_citations"]) + " (removed)" if v["dropped_citations"] else "none"))
        say("- Figures in the AI's text that aren't in the evidence: "
            + (", ".join(v["unsupported_figures"]) + " (flagged)" if v["unsupported_figures"] else "none"))
        if v.get("downgraded_facts"):
            st.markdown("- Claims the AI called verified without valid evidence (not shown as verified): "
                        + "; ".join(esc(x) for x in v["downgraded_facts"]))
        st.markdown("- Conflicts with the calculated market check: " + ("; ".join(v["conflicts"]) or "none"))
        st.caption("Every money figure in this report comes from MarginGuard's calculations. Product "
                   "descriptions and quotes are passed to the AI as data, and it is told never to follow "
                   "instructions inside them.")


def page_cost():
    header("Cost Intelligence", "The verdict first. The evidence is underneath.")
    need_data()
    keys = flagged_keys()
    if not keys:
        st.success("No price changes above your threshold.")
        return
    k = item_picker("Item", keys, "pick_cost")
    item, supplier = split_key(k)
    f, g = finding_for(k), supplier_series(item, supplier)
    b = D["bench"][(item, supplier)]
    stat = D["stats"][(D["stats"]["item"] == item) & (D["stats"]["supplier"] == supplier)].iloc[0]
    status = status_of(b)
    slabel, stone = STATUS[status]

    # ---- Decision summary
    st.subheader(f"{item} from {supplier.title()}")
    md(pill(slabel, stone) + pill(TYPE_LABEL[f.type][0], TYPE_LABEL[f.type][1])
       + pill(f"Benchmark: {b['profile']}", "grey")
       + (pill("Market proxy, not a direct quote", "grey") if b["match_type"] == "proxy" else ""))
    c1, c2, c3, c4 = st.columns(4)
    if f.type == "supplier_gap":
        c1.metric("Your price", unit_price(f.current_price, f.base_unit))
        c2.metric("Cheaper supplier", unit_price(f.baseline_price, f.base_unit))
        c3.metric("Difference", f"{f.pct_change:+.1%}")
        c4.metric("Possible saving", f"{money(f.annual_impact)}/yr")
    else:
        c1.metric("Price now", unit_price(f.current_price, f.base_unit),
                  delta=f"{f.pct_change:+.1%} since {pd.Timestamp(f.first_date):%b %Y}", delta_color="inverse",
                  help=f"Was {unit_price(f.baseline_price, f.base_unit)}.")
        c2.metric("Extra cost", f"{money(f.annual_impact)}/yr",
                  help=f"At {stat['monthly_qty']:,.1f} {f.base_unit} a month.")
        if b["benchmark_pct"] is not None:
            c3.metric(f"Market, {b['period']}", f"{b['benchmark_pct']:+.1%}",
                      help="Change in the matching world commodity price over the same months, in your currency.")
            c4.metric(f"You, {b['period']}", f"{b['matched_pct']:+.1%}",
                      delta=f"{b['gap_pp'] * 100:+.0f} pts vs market", delta_color="inverse",
                      help="Your price change over exactly the same months, so the two are comparable.")
        else:
            c3.metric("Market comparison", "—")
            c4.metric("Benchmark", "None", help=b["why_relevant"])

    md(f"<div class='mg-verdict'>{escape(b['verdict_text'])}</div>")
    if b.get("unverified"):
        md(f"<div class='mg-verdict'>{escape(b['unverified_text'])}</div>")
    if status == "review":
        say(f"**What this means:** the part covered by market data looks reasonable, but "
            f"{money(((b['above_market_per_unit'] if b['verdict'] == 'exceeds' else 0) + b['unverified_per_unit']) * stat['monthly_qty'] * 12)}/yr of "
            f"this increase has no market explanation yet. Ask before you absorb it.")
    elif status == "exceeds":
        say(f"**What this means:** your supplier moved further than the market over the same months. "
            f"{money(b['above_market_per_unit'] * stat['monthly_qty'] * 12)}/yr is worth asking about.")
    elif status == "consistent":
        st.markdown("**What this means:** the market moved with your supplier, so a price cut is unlikely. "
                    "Plan for this cost instead.")
    elif status == "below":
        st.markdown("**What this means:** the market rose faster than your price. Your supplier may still "
                    "pass it on, so consider fixing the price now.")
    else:
        st.markdown("**What this means:** no comparable public price exists for this item, so your own invoices "
                    "are the only evidence. A competing quote is the fastest way to find out if it's fair.")

    # ---- Next steps
    st.subheader("Recommended next steps")
    mine = rec_for(item, supplier)[:3]
    if not mine:
        st.write("No action needed for this item.")
    for i, r in enumerate(mine):
        with st.container(border=True):
            a, bcol = st.columns([4, 1.4])
            a.markdown(f"**{esc(r['action'])}**")
            say(r["next_step"], a.write)
            a.markdown(pill(f"Confidence: {r['confidence']}", CONF_TONE[r["confidence"]])
                       + pill(f"Effort: {r['difficulty']}", "grey") + pill(r["evidence"], "grey"),
                       unsafe_allow_html=True)
            bcol.markdown(f"<div class='mg-big'>{escape(money(r['impact']))}/yr</div>"
                          f"<div class='mg-muted'>{escape(impact_label(r['kind']))}</div>", unsafe_allow_html=True)
            if bcol.button("Compare suppliers", key=f"cs_{i}", use_container_width=True):
                goto(k, PAGES["savings"])
            if bcol.button("See forecast", key=f"cf_{i}", use_container_width=True):
                goto(k, PAGES["forecast"])

    # ---- AI investigation
    ai_section(k, item, supplier, f, b)

    # ---- Chart
    st.subheader("Your price against the market")
    if any(d.get("available") for d in b["drivers"]):
        st.altair_chart(benchmark_chart(g, b), use_container_width=True)
        cap = ("Both lines start at 100. Market series are monthly averages in US$"
               + (f", converted to {cur()} in the figures above. " if ss.get("currency") == "SGD" else ". "))
        if b.get("unverified"):
            cap += "The dashed red line marks where market data ends."
        say(cap, st.caption)
    else:
        st.altair_chart(price_history_chart(item), use_container_width=True)
        say(f"Price per {f.base_unit}, so pack-size changes show up as real price changes. "
            f"No market benchmark: {b['why_relevant']}", st.caption)

    # ---- Analyst layer
    st.subheader("Supporting evidence")
    with st.expander("Verified facts and market comparisons"):
        st.markdown("**Verified facts**")
        for x in b["facts"]:
            say(f"- {x}")
        st.markdown("**Comparisons** — association, not proof of cause")
        for x in b["associations"] or ["No market comparison available for this item."]:
            say(f"- {x}")
    with st.expander("Possible explanations we cannot verify"):
        for x in b["hypotheses"]:
            say(f"- {x}")
    with st.expander("Method and limitations"):
        if b.get("period"):
            st.markdown(f"- Matched-period comparison: your price change and the market's over the **same "
                        f"months** ({b['period']}).")
        for x in b["limitations"] or ["No market comparison was made for this item."]:
            say(f"- {x}")
    with st.expander("Data sources"):
        if b["evidence"]:
            st.dataframe(pd.DataFrame([{"Series": m["title"], "Source": m["source"], "Units": m["unit"],
                                        "Latest month": m["last_obs"], "Data": status_text(m),
                                        "Downloaded": m["retrieved_at"], "Link": m["url"]} for m in b["evidence"]]),
                         hide_index=True, use_container_width=True,
                         column_config={"Link": st.column_config.LinkColumn(display_text="Open on FRED")})
        else:
            st.write("No external data used for this item.")
        st.caption("Invoice data comes from the uploaded PDFs"
                   + (" (fictional demo invoices)." if demo else "."))
    with st.expander("Change the benchmark"):
        opts = list(insights.PROFILES)
        auto = insights.classify(item, D["cat"].get(item))
        curp = ss["overrides"].get(item, auto)
        new = st.selectbox("Compare this item with", opts, index=opts.index(curp), key=f"prof_{item}",
                           help=f"Detected automatically as '{auto}'. Change it if that's wrong.")
        say(insights.PROFILES[new]["why"], st.caption)
        if new != curp:
            if new == auto:
                ss["overrides"].pop(item, None)
            else:
                ss["overrides"][item] = new
            st.rerun()

    advice = {a["finding_id"]: a for a in res["plan"].get("actions", [])}.get(f.id)
    if advice:
        with st.expander("AI suggestion in plain English"):
            st.markdown(f"**{esc(advice['title'])}**")
            say(advice["advice"], st.write)
            st.caption("Written by the AI from the calculated findings; it adds no numbers of its own.")


def step(n, title, question):
    md(f"<div><span class='mg-step'>Step {n}</span><span class='mg-steph'>{escape(title)}</span></div>"
       f"<div class='mg-stepq'>{escape(question)}</div>")


def page_forecast():
    header("Forecast & Scenarios", "Three questions, in order: where might the market go, what if it goes "
                                   "differently, and what would that cost you.")
    need_data()
    all_keys = list(dict.fromkeys(flagged_keys() + [f"{r['item']}||{r['supplier']}"
                                                    for _, r in D["stats"].iterrows()]))
    k = item_picker("Item", all_keys, "pick_fc")
    item, supplier = split_key(k)
    g = supplier_series(item, supplier)
    stat = D["stats"][(D["stats"]["item"] == item) & (D["stats"]["supplier"] == supplier)].iloc[0]
    b = D["bench"].get((item, supplier))
    if b is None:
        prof = ss["overrides"].get(item) or insights.classify(item, D["cat"].get(item))
        b = insights.benchmark(g, prof, MK, ss.get("currency", "SGD"))
    primaries = [d for d in b["drivers"] if d.get("available")]
    has_bench = bool(primaries)
    fx_exposed = has_bench and ss.get("currency", "SGD") == "SGD"
    unit = stat["base_unit"]

    # ============================================================= Step 1: market outlook
    with st.container(border=True):
        step(1, "Market outlook", f"Where might the world price behind {item.lower()} go?")
        fc_pct, fc, meta = None, None, None
        if has_bench:
            c1, c2 = st.columns([2, 1])
            names = {d["id"]: d["title"] for d in primaries}
            with c1:
                sid = (st.selectbox("Market series", list(names), format_func=names.get, key=f"sid_{item}")
                       if len(names) > 1 else list(names)[0])
                if len(names) == 1:
                    st.caption(f"Benchmark: {names[sid]}")
            with c2:
                horizon = st.radio("How far ahead?", [1, 3, 6], index=[1, 3, 6].index(ss["horizon"]),
                                   horizontal=True, format_func=lambda h: f"{h}m", key="hz")
                ss["horizon"] = horizon
            df, meta = MK[sid]
            # Forecast month by month, so the chart genuinely ends at the chosen horizon.
            fc = insights.forecast_series(df, horizons=tuple(range(1, horizon + 1)))
            last = fc["rows"][-1]
            fc_pct = last["pct"]
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Latest, {pd.Timestamp(fc['last_date']):%b %Y}", f"{fc['last_value']:,.0f}",
                      help=meta["unit"])
            m2.metric(f"Baseline, {pd.Timestamp(last['date']):%b %Y}", f"{last['point']:,.0f}",
                      delta=f"{last['pct']:+.1%}", delta_color="off")
            m3.metric("Uncertainty range", f"{last['low']:,.0f}–{last['high']:,.0f}"
                      if last["low"] is not None else "n/a",
                      help="Widens with the horizon: a longer projection is less certain.")
            bt = ", ".join(f"{insights.METHODS[m].split(' (')[0].lower()} {e:.1%}" for m, e in fc["backtest"].items())
            if fc["method"] == "naive" and abs(last["pct"]) < 0.005:
                say(f"**The baseline is flat by design.** Three methods were tested on the last 6 months "
                    f"({bt}); carrying the last price forward was the most accurate, so the model has no "
                    f"evidence prices will move either way. What does change with the horizon is the "
                    f"uncertainty range. To ask 'what if', use Step 2.", st.info)
            else:
                say(f"Baseline method: {fc['method_label']}, lowest error over the last 6 months ({bt}). "
                    f"{fc['note']}", st.caption)
        else:
            horizon = st.radio("How far ahead?", [1, 3, 6], index=[1, 3, 6].index(ss["horizon"]),
                               horizontal=True, format_func=lambda h: f"{h} month{'s' if h > 1 else ''}", key="hz")
            ss["horizon"] = horizon
            say(f"There is no public market benchmark for {item.lower()} ({b['profile']}). {b['why_relevant']} "
                f"Steps 2 and 3 still work from your supplier's own price history.", st.info)

    # ============================================================= Step 2: what-if
    with st.container(border=True):
        step(2, "What-if scenario", "What if the market turns out differently?")
        if has_bench:
            shock = st.slider("Assumed market price change", -30, 30, int(ss["shock"] * 100), 5, format="%d%%",
                              key="shk",
                              help="Your own assumption about the world commodity price. It does not change "
                                   "the baseline above; it sits on top of it.")
            ss["shock"] = shock / 100
            with st.expander("Advanced assumptions"):
                pt = st.slider("How much of the market price change reaches your supplier's price?",
                               0, 100, int(ss["pass_through"] * 100), 5, format="%d%%", key="pt")
                ss["pass_through"] = pt / 100
                say(f"Assumes {pt}% of the market price change shows up in what your supplier charges you. "
                    f"Your supplier also pays for processing, packing, delivery and labour, so a 20% jump in "
                    f"the raw commodity rarely means a 20% jump on your invoice. This is a sensitivity "
                    f"assumption you choose, not a relationship measured from your data.", st.caption)
                if fx_exposed:
                    fx = st.slider("US dollar shock", -20, 20, int(ss["fx"] * 100), 5, format="%d%%", key="fxs")
                    ss["fx"] = fx / 100
                    say(f"World commodity prices are quoted in US dollars. A stronger dollar raises the "
                        f"{cur()} cost of anything priced off one, even if the commodity price itself "
                        f"does not move. Applied only to items with a US dollar benchmark.", st.caption)
                else:
                    ss["fx"] = 0.0
                    say(f"US dollar shock is off: your invoice currency is {ss.get('currency', 'SGD')}, so no "
                        f"currency exposure is modelled.", st.caption)
            active = [f"{horizon}-month horizon", f"market {ss['shock']:+.0%}",
                      f"{ss['pass_through']:.0%} reaches your price"]
            if ss["fx"]:
                active.insert(2, f"US dollar {ss['fx']:+.0%}")
            say("**Your assumptions:** " + ", ".join(active) + ".")
        else:
            ss["shock"], ss["fx"] = 0.0, 0.0
            say(f"**Commodity and US dollar assumptions do not apply to {item.lower()}.** Applying them would "
                f"mean inventing a link between this item and a market we have not matched it to. The "
                f"scenarios below come from your supplier's own price history instead.")

    sset = insights.scenario_set(g, horizon, forecast_pct=fc_pct, commodity_shock=ss["shock"],
                                 fx_shock=ss["fx"], pass_through=ss["pass_through"],
                                 catch_up_pct=insights.catch_up(b), fx_exposed=fx_exposed,
                                 has_benchmark=has_bench)
    rows = sset["rows"]

    # ---- chart, now that the scenario is known, so the overlay can be drawn
    if has_bench and fc is not None:
        df, _ = MK[sid]
        hist = df.tail(24)
        fcd = pd.DataFrame([{"date": fc["last_date"], "point": fc["last_value"], "low": fc["last_value"],
                             "high": fc["last_value"]}]
                           + [{j: x[j] for j in ("date", "point", "low", "high")} for x in fc["rows"]])
        fcd["low"] = fcd["low"].fillna(fcd["point"])
        fcd["high"] = fcd["high"].fillna(fcd["point"])
        shocked = None
        if sset["market_pct"] is not None and abs(sset["market_pct"] - (fc_pct or 0)) > 1e-9:
            n = len(fcd) - 1
            shocked = fcd.copy()
            # ramp the shock in over the horizon, so the line reaches the full move at the end
            shocked["point"] = [fc["last_value"] * (1 + sset["market_pct"] * i / n) if n else fc["last_value"]
                                for i in range(len(fcd))]
        yt = meta["unit"]
        layers = [
            alt.Chart(hist).mark_line(color=T["slate"], strokeWidth=1.8)
            .encode(x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=yt, scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                             alt.Tooltip("value:Q", title="Observed", format=",.1f")]),
            alt.Chart(fcd).mark_area(opacity=0.18, color=T["accent"])
            .encode(x=alt.X("date:T", title=None), y=alt.Y("low:Q", title=yt), y2="high:Q"),
            alt.Chart(fcd).mark_line(color=T["accent2"], strokeDash=[5, 4], point=True)
            .encode(x=alt.X("date:T", title=None), y=alt.Y("point:Q", title=yt),
                    tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                             alt.Tooltip("point:Q", title="Baseline", format=",.1f"),
                             alt.Tooltip("low:Q", title="Low", format=",.1f"),
                             alt.Tooltip("high:Q", title="High", format=",.1f")]),
        ]
        if shocked is not None:
            layers.append(alt.Chart(shocked).mark_line(color=T["warning"], strokeDash=[2, 2], point=True)
                          .encode(x=alt.X("date:T", title=None), y=alt.Y("point:Q", title=yt),
                                  tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                                           alt.Tooltip("point:Q", title="Your scenario", format=",.1f")]))
        st.altair_chart(dark(alt.layer(*layers).properties(height=260)), use_container_width=True)
        legend = "Grey: observed. Blue dashed with the band: baseline projection and its uncertainty."
        if shocked is not None:
            legend += " Amber dashed: your scenario from Step 2."
        say(f"{legend} Source: {meta['source']}, "
            f"{status_text(meta).lower()}, downloaded {meta['retrieved_at']}; latest observation "
            f"{meta['last_obs']}.", st.caption)

    # ============================================================= Step 3: business impact
    with st.container(border=True):
        step(3, "Impact on your business",
             f"What would that cost you, at {money(stat['monthly_spend'])} a month on {item.lower()}?")
        names_sc = [r["scenario"] for r in rows]

        def _pref(r):
            return (abs(r["pct"]) > 1e-9, r["scenario"].startswith("Market moves"),
                    r["scenario"].startswith("Supplier catches up"))
        default = max(range(len(rows)), key=lambda i: _pref(rows[i])) if rows else 0
        f1, f2 = st.columns([2, 1])
        pick = f1.selectbox("Which scenario?", names_sc, index=default, key=f"pick_sc_{item}")
        override = f2.number_input("Or set the price change yourself (%)", value=0.0, step=1.0,
                                   help="Leave at 0 to use the scenario.")
        pct = override / 100 if override else next(r["pct"] for r in rows if r["scenario"] == pick)
        cum = insights.cumulative_impact(float(stat["monthly_spend"]), pct, horizon)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Supplier price change", f"{pct:+.1%}",
                  help=f"Now {unit_price(sset['current'], unit)}; "
                       f"scenario {unit_price(sset['current'] * (1 + pct), unit)}.")
        r2.metric("Baseline monthly spend", money(cum["baseline_monthly"]))
        r3.metric("Scenario monthly spend", money(cum["scenario_monthly"]),
                  delta=money(cum["extra_monthly"]), delta_color="inverse")
        r4.metric(f"Extra over {horizon} month{'s' if horizon > 1 else ''}", money(cum["cumulative"]),
                  help="Extra monthly cost × months in the horizon. Deliberately not annualised.")
        say(next((r["basis"] for r in rows if r["scenario"] == pick), "Your own price-change assumption."),
            st.caption)

        with st.expander("Effect on profit margin"):
            e1, e2 = st.columns(2)
            revenue = e1.number_input(f"Monthly revenue ({cur()})", min_value=0.0, value=float(ss["revenue"]),
                                      step=1000.0)
            cogs = e2.number_input(f"Monthly cost of goods sold ({cur()})", min_value=0.0,
                                   value=float(ss["cogs"]), step=1000.0,
                                   help="All food and drink purchases per month.")
            ss["revenue"], ss["cogs"] = revenue, cogs
            imp = insights.cost_impact(float(stat["monthly_qty"]), float(stat["current_price"]), pct,
                                       revenue or None, cogs if revenue else None)
            if "gross_margin_before" in imp:
                g1, g2 = st.columns(2)
                g1.metric("Gross margin now", f"{imp['gross_margin_before']:.1%}")
                g2.metric("Under this scenario", f"{imp['gross_margin_after']:.1%}",
                          delta=f"{imp['margin_change_pp'] * 100:+.2f} pts")
                say(f"Gross margin = (revenue − cost of goods sold) ÷ revenue. Not operating margin: rent, "
                    f"wages and other overheads are excluded. To offset this, average selling prices would "
                    f"need to rise {imp['price_rise_to_offset']:.2%}, assuming customers buy the same amount.",
                    st.caption)
            else:
                st.caption("Enter your monthly revenue to see the effect on gross margin.")

        say(f"Assumes the change lands now and holds for the whole {horizon} month"
            f"{'s' if horizon > 1 else ''}, on unchanged volumes. If it arrives gradually, the cumulative "
            f"figure is lower. Sustained for a full year it would be "
            f"{money(cum['annualised_if_sustained'])}, which answers a different question from the "
            f"{horizon}-month exposure above.", st.caption)

    with st.expander("All scenarios for this item"):
        st.dataframe(pd.DataFrame([{"Scenario": r["scenario"], f"Price/{unit}": unit_price(r["price"], unit),
                                    "Change": f"{r['pct']:+.1%}",
                                    f"Extra over {horizon}m": money(stat["monthly_spend"] * r["pct"] * horizon)}
                                   for r in rows]), hide_index=True, use_container_width=True)
        for r in rows:
            say(f"- **{r['scenario']}** — {r['basis']}")
    st.page_link(PAGES["savings"], label="Next: compare suppliers and act", icon=":material/arrow_forward:")


def page_savings():
    header("Savings", "Compare suppliers like for like, then track what you actually did.")
    need_data()
    exp_m = sum(a.get("expected_monthly", 0.0) for a in ss["actions"].values()
                if a.get("status") in ("Saved", "Under review", "Implemented"))
    act_m = sum(a.get("actual_monthly", 0.0) for a in ss["actions"].values() if a.get("status") == "Implemented")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Identified", f"{money(D['savings_total'])}/yr",
              help="Best evidence-backed option per item, never double-counted.")
    s2.metric("Flagged for review", f"{money(D['at_stake'])}/yr",
              help="Increases with no market explanation yet. Not savings until your supplier answers.")
    s3.metric("Expected", f"{money(exp_m * 12)}/yr", help="From actions you saved or implemented in the tracker.")
    s4.metric("Realised", f"{money(act_m * 12)}/yr",
              help="Only from actions marked Implemented with an actual monthly saving recorded.")

    tab1, tab2, tab3 = st.tabs(["Recommended actions", "Supplier comparison", "Action tracker"])

    # ---------------------------------------------------------------- actions
    with tab1:
        if not D["recs"]:
            st.write("No actions needed: no price changes above your threshold.")
        for r in D["recs"]:
            cur_status = tracked(r).get("status", "Not started")
            with st.container(border=True):
                a, bcol = st.columns([4, 1.5])
                a.markdown(f"**{r['rank']}. {esc(r['action'])}**")
                say(r["why"], a.write)
                say(f"**Next step:** {r['next_step']}", a.markdown)
                a.markdown(pill(f"Confidence: {r['confidence']}", CONF_TONE[r["confidence"]])
                           + pill(f"Effort: {r['difficulty']}", "grey") + pill(r["evidence"], "grey")
                           + pill(cur_status, TRACK_TONE[cur_status]), unsafe_allow_html=True)
                say(f"Assumptions: {r['assumptions']}", a.caption)
                bcol.markdown(f"<div class='mg-big'>{escape(money(r['impact']))}/yr</div>"
                              f"<div class='mg-muted'>{escape(impact_label(r['kind']))}</div>",
                              unsafe_allow_html=True)
                new = bcol.selectbox("Status", TRACK, index=TRACK.index(cur_status), key=f"st_{action_id(r)}",
                                     label_visibility="collapsed")
                if new != cur_status:
                    if new == "Not started":
                        ss["actions"].pop(action_id(r), None)
                    else:
                        rec = ss["actions"].setdefault(action_id(r), {})
                        rec.update(item=r["item"], supplier=r["supplier"], action=r["action"], status=new,
                                   expected_monthly=rec.get("expected_monthly", round(r["saving"] / 12, 2)),
                                   actual_monthly=rec.get("actual_monthly", 0.0), note=rec.get("note", ""),
                                   date=rec.get("date", date.today().isoformat()))
                    st.rerun()

    # ------------------------------------------------------- supplier compare
    with tab2:
        stats = D["stats"]
        opts = [f"{r['item']}||{r['supplier']}" for _, r in stats.iterrows()]
        sel = item_picker("Item to compare", opts, "pick_sav")
        item, vs_sup = split_key(sel)
        row = stats[(stats["item"] == item) & (stats["supplier"] == vs_sup)].iloc[0]
        unit = row["base_unit"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current supplier", vs_sup.title())
        c2.metric("You pay", unit_price(row["current_price"], unit))
        c3.metric("You buy", f"{row['monthly_qty']:,.1f} {unit}/mo",
                  help=f"About {money(row['monthly_spend'])} a month.")

        rows = [{"Supplier": vs_sup.title(), "Source": "Your invoices", "Match": "Current supplier",
                 "Listed price": unit_price(row["current_price"], unit),
                 f"Landed per {unit}": unit_price(row["current_price"], unit), "Min order": "—",
                 "Saving/yr": "—", "Basis": "—", "Checked": str(res["summary"]["date_to"]), "Link": None}]
        for f in res["findings"]:
            if f.type == "supplier_gap" and f.item == item and f.supplier == vs_sup:
                rows.append({"Supplier": f.detail.split(" sells")[0], "Source": "Your invoices (verified)",
                             "Match": "Same product", "Listed price": unit_price(f.baseline_price, f.base_unit),
                             f"Landed per {unit}": unit_price(f.baseline_price, f.base_unit), "Min order": "—",
                             "Saving/yr": money(f.annual_impact), "Basis": "Net (same delivery terms)",
                             "Checked": f.change_date, "Link": None})
        for q in D["quotes"]:
            if q["item"] != item or q["vs_supplier"] != vs_sup:
                continue
            if not q.get("ok"):
                rows.append({"Supplier": q["supplier"], "Source": q["source"], "Match": q.get("match", "—"),
                             "Listed price": "—", f"Landed per {unit}": "—", "Min order": "—",
                             "Saving/yr": "not comparable", "Basis": q["reason"], "Checked": q["date"],
                             "Link": q["url"] or None})
                continue
            known = q["delivery"] > 0 or q["one_off"] > 0 or q.get("costs_known")
            rows.append({"Supplier": q["supplier"], "Source": q["source"], "Match": q.get("match", "Same product"),
                         "Listed price": f"{money(q['pack_price'], 2)} / {q['pack_size']:g}{q['pack_unit']}",
                         f"Landed per {unit}": unit_price(q["landed_price"], q["unit"]),
                         "Min order": f"{q['moq']:g} packs" if q.get("moq") else "—",
                         "Saving/yr": money(q["net_annual"]),
                         "Basis": "Net of your entered costs" if known else "Gross: delivery not known",
                         "Checked": q["date"], "Link": q["url"] or None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Link": st.column_config.LinkColumn(display_text="Open")})
        say(f"Everything is compared per {unit}, so pack sizes don't distort it. The landed price adds the "
            f"delivery you entered, spread over what you buy. Where delivery is unknown the saving is "
            f"labelled gross, never assumed to be zero.", st.caption)

        ranked = [r for r in D["recs"] if r["kind"] == "switch" and r["item"] == item]
        if ranked:
            st.markdown("**Why these rank where they do**")
            for r in ranked:
                say(f"- **{r['supplier']}** — {money(r['impact'])}/yr, confidence {r['confidence']}. {r['why']}")

        st.markdown("**Add a supplier quote**")
        st.caption("MarginGuard does not search shopping sites and never invents prices. Get a quote or find a "
                   "listing, then enter it here.")
        with st.form("quote", clear_on_submit=True):
            a1, a2, a3 = st.columns(3)
            sup = a1.text_input("Supplier name")
            match = a2.selectbox("How close a match?", ["Same product", "Possible substitute"],
                                 help="Choose 'possible substitute' if the grade, brand or spec differs.")
            spec = a3.text_input("Product / spec (optional)")
            b1, b2, b3, b4 = st.columns(4)
            pack_price = b1.number_input(f"Pack price ({cur()})", min_value=0.0, step=0.5)
            pack_size = b2.number_input("Pack size", min_value=0.0, step=1.0)
            pack_unit = b3.selectbox("Unit", list(TO_BASE))
            moq = b4.number_input("Min order (packs)", min_value=0.0, step=1.0)
            d1, d2, d3 = st.columns(3)
            delivery = d1.number_input(f"Delivery or fees per month ({cur()})", min_value=0.0, step=5.0)
            one_off = d2.number_input(f"One-off switching cost ({cur()})", min_value=0.0, step=10.0)
            costs_known = d3.checkbox("Delivery is included in this price", value=False,
                                      help="Tick if the quote is delivered-in, so the saving counts as net.")
            e1, e2 = st.columns(2)
            url = e1.text_input("Link to the listing or quote (optional)")
            qdate = e2.date_input("Quote date", value=date.today())
            if st.form_submit_button("Add quote", type="primary"):
                if not sup.strip() or pack_price <= 0 or pack_size <= 0:
                    st.warning("Enter the supplier name, pack price and pack size.")
                else:
                    ss["quotes"].append({"item": item, "vs_supplier": vs_sup, "supplier": sup.strip(),
                                         "spec": spec.strip(), "match": match, "pack_price": pack_price,
                                         "pack_size": pack_size, "pack_unit": pack_unit, "moq": moq,
                                         "delivery": delivery, "one_off": one_off, "costs_known": costs_known,
                                         "url": url.strip(), "date": qdate.isoformat(),
                                         "source": "Quote entered by you"})
                    st.rerun()
        g1, g2, g3 = st.columns(3)
        g1.link_button(f"Search the web for {item.lower()} suppliers",
                       f"https://www.google.com/search?q={quote_plus(item + ' wholesale supplier Singapore')}",
                       use_container_width=True)
        if demo and g2.button("Add a fictional demo quote", use_container_width=True,
                              help="A clearly labelled made-up quote, so you can see how the comparison works."):
            ss["quotes"].append({"item": item, "vs_supplier": vs_sup,
                                 "supplier": "Example Trading (fictional demo supplier)", "spec": "demo",
                                 "match": "Same product", "pack_price": round(row["current_price"] * 0.88 * 5, 2),
                                 "pack_size": 5.0, "pack_unit": {"L": "L", "kg": "kg"}.get(unit, "each"),
                                 "moq": 4, "delivery": 15.0, "one_off": 0.0, "costs_known": True, "url": "",
                                 "date": date.today().isoformat(), "source": "Fictional demo quote"})
            st.rerun()
        if ss["quotes"] and g3.button("Clear all quotes", use_container_width=True):
            ss["quotes"] = []
            st.rerun()
        st.caption("The search button opens an ordinary web search. Its results are not checked by MarginGuard.")

    # ------------------------------------------------------------- tracker
    with tab3:
        st.caption("Set a status on any recommendation, then record what you expected and what you actually got. "
                   "Expected savings are never counted as realised.")
        if not ss["actions"]:
            st.info("Nothing tracked yet. Set a status on a recommendation in the first tab.")
        for aid, a in list(ss["actions"].items()):
            with st.container(border=True):
                head, right = st.columns([3, 1])
                head.markdown(f"**{esc(a['action'])}**")
                head.markdown(pill(a["status"], TRACK_TONE[a["status"]])
                              + pill(f"{a['item']} · {a['supplier'].title()}", "grey"), unsafe_allow_html=True)
                if right.button("Remove", key=f"rm_{aid}", use_container_width=True):
                    ss["actions"].pop(aid)
                    ss.pop(f"st_{aid}", None)   # else the status dropdown re-creates it on the next rerun
                    st.rerun()
                c1, c2, c3 = st.columns([1, 1, 2])
                a["expected_monthly"] = c1.number_input(f"Expected ({cur()}/mo)", min_value=0.0,
                                                        value=float(a.get("expected_monthly", 0.0)), step=5.0,
                                                        key=f"exp_{aid}")
                a["actual_monthly"] = c2.number_input(f"Actual ({cur()}/mo)", min_value=0.0,
                                                      value=float(a.get("actual_monthly", 0.0)), step=5.0,
                                                      key=f"act_{aid}",
                                                      help="Counts as realised only when the status is Implemented.")
                a["note"] = c3.text_input("What you did", value=a.get("note", ""), key=f"nt_{aid}")
                if a["status"] == "Implemented" and a["actual_monthly"] == 0:
                    st.caption("Marked implemented, but with no actual saving recorded it isn't counted as "
                               "realised.")
        h1, h2 = st.columns(2)
        h1.download_button("Download tracker (JSON)", json.dumps(ss["actions"], indent=1),
                           "marginguard_actions.json", use_container_width=True)
        up = h2.file_uploader("Restore a saved tracker", type=["json"])
        if up is not None:
            try:
                loaded = json.load(up)
                if isinstance(loaded, dict):
                    ss["actions"].update(loaded)
                    st.success(f"Restored {len(loaded)} tracked actions.")
                else:
                    st.error("That file isn't a MarginGuard tracker export.")
            except Exception as e:
                st.error(f"Could not read that file: {e}")
        st.caption("The tracker lives in this browser session only, so it clears when the tab closes or the app "
                   "restarts. Download it to keep a copy. A shared database would be the next step for real use.")

    email = res["plan"].get("email")
    if email:
        st.subheader("Draft email to your supplier")
        st.caption(f"To {email.get('supplier', 'your supplier')} · written by the AI from the findings · "
                   f"edit before sending")
        st.text_input("Subject", email.get("subject", ""))
        st.text_area("Message", email.get("body", ""), height=230)


def page_data():
    header("Data & Checks", "Everything MarginGuard read, checked and used, so every number can be traced.")
    need_data()
    s = res["summary"]
    t1, t2, t3, t4 = st.tabs(["Extracted invoice data", "Checks & warnings", "Market data", "How the numbers work"])
    with t1:
        show = res["lines"][["date", "supplier", "invoice_number", "description", "canonical_name", "quantity",
                             "unit_price", "line_total", "pack_size", "pack_unit", "price_per_base", "base_unit"]]
        st.dataframe(show.sort_values(["date", "supplier"]), hide_index=True, use_container_width=True,
                     column_config={"date": st.column_config.DateColumn("Date"),
                                    "canonical_name": "Matched product (AI)",
                                    "price_per_base": st.column_config.NumberColumn("Price per base unit",
                                                                                    format="%.3f")})
        fdf = pd.DataFrame([f.to_dict() for f in res["findings"]])
        if not fdf.empty:
            st.download_button("Download findings (CSV)", fdf.to_csv(index=False), "marginguard_findings.csv")
    with t2:
        st.write(f"AI product matching: {'used' if res['ai_matched'] else 'unavailable, basic name matching used'}")
        st.write(f"AI action plan: {'used' if res['ai_plan'] else 'unavailable, template advice used'}")
        st.write("Every invoice is re-checked with arithmetic (quantity × price = line total, lines add up to the "
                 "subtotal). If the AI's reading fails a check, it is asked to correct itself once.")
        for w in res["warnings"]:
            st.warning(w)
        for e in res["errors"]:
            st.error(e)
        if not res["warnings"] and not res["errors"]:
            st.success(f"All {s['invoices']} invoices passed the arithmetic checks.")
    with t3:
        md(market_status())
        st.dataframe(pd.DataFrame([{"Series": m["title"], "FRED id": sid, "Units": m["unit"], "Source": m["source"],
                                    "Data": status_text(m), "Observations": f"{m['first_obs']} to {m['last_obs']}",
                                    "Downloaded": m["retrieved_at"], "Checked": m["checked_at"],
                                    "Why not live": m["fallback_reason"] if m["status"] != "live" else "—",
                                    "Link": m["url"]}
                                   for sid, (_, m) in MK.items()]),
                     hide_index=True, use_container_width=True,
                     column_config={"Link": st.column_config.LinkColumn(display_text="Open")})
        st.caption("Live-first: every load tries FRED's free CSV download, checks the response (layout, dates, "
                   "values) and keeps it for 6 hours. If the download fails, the last good download is used, "
                   f"or the copy shipped with the app (downloaded {market.SNAPSHOT_DATE}), labelled 'Cached data' "
                   f"with the reason. A failed download is retried after {RETRY_FALLBACK_MIN} minutes, or now with "
                   "'Refresh market data' in the sidebar. 'Downloaded' is when the data was fetched; 'Observations' "
                   "is the months the prices refer to. Commodity prices are published monthly, about two to three "
                   "months behind, so even live data ends a few months before today.")
    with t4:
        say(f"""
**From your invoices (deterministic code, no AI)**
- **Price per base unit** = pack price ÷ amount in the pack (kg, litres or items). This is what catches smaller packs.
- **Change** = latest price vs the first price seen for that product from that supplier. Changes under {threshold:.0%} are treated as noise.
- **Extra cost per year** = (current − original price per unit) × average monthly quantity × 12.
- **Cheaper elsewhere** = another of your current suppliers charges less for the same product.

**Market check (matched periods)**
- Market data is published about two months late. So MarginGuard compares your price change and the market's over **exactly the same months**, then reports anything you paid after that separately, as not yet checkable.
- Within ±{insights.TOLERANCE_PP * 100:.0f} percentage points = *in line*; more = *above market*; less = *market rising faster*.
- Every benchmark here is a **market proxy** (a raw-commodity index), never a quote for the product you buy. It shows whether an increase is consistent with the market; it cannot prove why a supplier raised a price.

**Forecasts and scenarios**
- Market outlook: the simplest of three methods (last value, exponential smoothing, 12-month trend) unless another is clearly more accurate over the last 6 months. The range widens with the horizon.
- Your price: what-if scenarios (price holds, recent trend continues, supplier catches up, supplier passes on part of a projected move), never labelled forecasts.

**Money**
- *Savings identified* counts only evidence-backed money: a cheaper supplier on your own invoices, a shrunk pack, an increase above the market over the matched period, or a quote you entered. Best option per item only, so nothing is double-counted.
- *Flagged for review* is money with no market explanation yet, deliberately kept out of savings.
- *Expected* and *realised* come from the tracker, and expected is never counted as realised.

**The AI** reads invoices, matches product names and writes the plain-English advice and email. All figures are calculated in code and unit-tested.
""")


PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    "cost": st.Page(page_cost, title="Cost Intelligence", icon=":material/query_stats:", url_path="cost"),
    "forecast": st.Page(page_forecast, title="Forecast & Scenarios", icon=":material/trending_up:",
                        url_path="forecast"),
    "savings": st.Page(page_savings, title="Savings", icon=":material/savings:", url_path="savings"),
    "data": st.Page(page_data, title="Data & Checks", icon=":material/fact_check:", url_path="data"),
}
st.navigation(list(PAGES.values())).run()
