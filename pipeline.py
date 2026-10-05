"""
Runs the whole MarginGuard pipeline:  PDFs -> extract -> normalize -> analyze -> recommend.
Usable from the Streamlit app or the command line:

    python pipeline.py sample_invoices/*.pdf
"""
import sys
from pathlib import Path

from analyze import analyze, to_lines_df
from extract import extract_invoice
from normalize import normalize
from recommend import recommend


def run(files, threshold=0.03, progress=None):
    """files: list of (filename, path_or_filelike). progress: optional callback(fraction, message)."""
    invoices, warnings, errors = [], [], []
    for i, (name, f) in enumerate(files):
        if progress:
            progress(i / max(len(files), 1) * 0.8, f"Reading {name} ({i + 1}/{len(files)})")
        try:
            inv, w = extract_invoice(f, name)
            invoices.append((name, inv))
            warnings += w
        except Exception as e:  # keep going: one bad file shouldn't sink the whole run
            errors.append(f"{name}: {e}")
    if not invoices:
        raise RuntimeError("No invoices could be read.\n" + "\n".join(errors))

    if progress:
        progress(0.82, "Matching products across suppliers")
    df = to_lines_df(invoices)
    df, ai_matched = normalize(df)

    if progress:
        progress(0.9, "Calculating price changes")
    findings, series, summary = analyze(df, threshold)

    if progress:
        progress(0.95, "Writing your action plan")
    plan, ai_plan = recommend(findings, summary)
    if progress:
        progress(1.0, "Done")

    return {"lines": df, "findings": findings, "series": series, "summary": summary, "plan": plan,
            "warnings": warnings, "errors": errors, "ai_matched": ai_matched, "ai_plan": ai_plan}


if __name__ == "__main__":
    from llm import load_env
    load_env()
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("sample_invoices").glob("*.pdf"))
    res = run([(p.name, p) for p in paths], progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    s = res["summary"]
    print(f"\n{s['invoices']} invoices, {s['items']} products, ${s['monthly_spend']:,.0f}/month spend")
    print(f"Extra cost from price increases: ${s['annual_extra_cost']:,.0f}/year "
          f"({s['extra_cost_pct_of_spend']:.1%} of spend)\n")
    for f in res["findings"]:
        print(f"{f.id:4} {f.type:15} {f.item:22} {f.supplier:32} {f.pct_change:+7.1%}  ${f.annual_impact:>8,.0f}/yr")
    for w in res["warnings"] + res["errors"]:
        print("WARN", w)
