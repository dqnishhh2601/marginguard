# Devpost submission draft (fill in the _italic_ bits)

## Project title
MarginGuard

## Short description (one line)
AI that reads a small restaurant's supplier invoices, catches hidden price rises, checks them against world market prices, and tells the owner which to fight, which to plan for, and where to save.

## Track
AI + Business

## Links
- Demo video: _YouTube link (Public, 2–4 min)_
- GitHub: https://github.com/dqnishhh2601/marginguard
- Live app: https://marginguard-crossdash.streamlit.app

---

## Inspiration / Problem statement
Small independent restaurants keep only about 5 cents of every dollar of sales as pre-tax profit, and wholesale food prices have risen about 35% since 2019 (National Restaurant Association). Supplier price increases rarely arrive as one announcement: they hide in a few cents more per line, a product renamed in a new billing system, or a 10kg bag that becomes 9kg at the same price. Even when an owner spots a rise, they can't tell whether it's the market or their supplier, so they don't know whether to push back, raise menu prices or switch.

**Target users:** owners and managers of independent cafés, restaurants, bakeries and food stalls who buy from several suppliers and have no finance team.

## What it does
MarginGuard turns a pile of supplier invoices into decisions, in four steps:
1. **Detect:** reads invoice PDFs in any layout, matches the same product across suppliers and pack sizes, and flags price increases, shrinkflation and cheaper suppliers, with the yearly cost of each.
2. **Investigate:** compares each increase with real world-market prices (IMF commodity prices for palm and soybean oil, wheat, rice, sugar, coffee and poultry, plus Brent crude and SGD/USD, via FRED) over **matched periods**. Commodity data lags by about two months, so it scores only the months both sides cover and reports anything newer separately as not yet checkable. It separates verified facts from unverifiable explanations, labels every benchmark a market proxy rather than a direct quote, and shows sources and dates.
3. **Forecast:** a simple, back-tested market outlook and clearly labelled what-if scenarios for the supplier's price, plus a profit impact simulator that shows the effect on gross margin.
4. **Act:** ranked, rules-based recommendations (ask for an explanation, challenge an above-market rise, recover a shrunk pack, switch, review menu prices, ask for a fixed price, get a competing quote), a supplier comparison workspace that compares landed cost per kg or litre, and an action tracker for what you did and what it actually saved.

On our 24 sample invoices it found $2,104/year in hidden increases (4.2% of supplier spend) and, crucially, sorted them rather than lumping them together:
- **Cooking oil — review needed.** Over April–July, where market data exists, the supplier moved +7.6% against +3.1% for palm and soybean oil: broadly in line. A further +9.5% landed after the latest published market data, so about $450/year has no market explanation yet. That is flagged for review, not claimed as a saving.
- **Plain flour — in line.** The bag shrank 10kg to 9kg (+11.1% per kg) while wheat rose about 14%. Hidden, but consistent with the market, so plan for it.
- **Jasmine rice — future pressure.** Supplier +6% while Thai rice export prices rose about 22%. Ask for a fixed price.
- **Eggs — act now.** The same eggs are 9.2% cheaper at another current supplier: $250/year, verified from the user's own invoices.

That first case is the design decision we are most proud of. Comparing the supplier's April–September rise against April–July market data would have made the supplier look guilty, which is exactly the kind of confident-but-wrong analysis we wanted to avoid.

## How we built it (technical approach)
- **Extraction:** pdfplumber pulls the invoice text; an open-source LLM (Qwen2.5-72B via Featherless AI) converts it to structured JSON, validated with pydantic.
- **Verification + self-correction:** code re-does the invoice maths (quantity × price = line total; lines = subtotal). If the AI's reading fails, it's told exactly what's wrong and asked to fix it. Remaining problems are shown to the user, never hidden.
- **Product matching:** the LLM groups product names across suppliers and naming changes into canonical products.
- **Analysis (no AI):** pandas calculates price per base unit, change vs the first recorded price, yearly impact from average monthly quantities, shrinkflation and cross-supplier gaps.
- **Market benchmarks (no AI):** items are classified and mapped to FRED series; changes are aligned to the same months, converted from US$ to S$, and compared with a ±5-point tolerance. The part of a supplier's move that falls outside the published market window is split out and reported separately. Live data is cached for 6 hours, with a labelled snapshot fallback if FRED is unreachable.
- **Forecasting and scenarios (no AI):** naive, exponential smoothing and 12-month trend, chosen by a rolling 6-month backtest and biased towards the simplest method; forecast month by month to the chosen horizon, with ranges that widen. Kept strictly apart from the user's what-ifs: a commodity price shock, a US dollar shock and a cost pass-through rate compound with the projection rather than overwriting it, and are disabled with an explanation on items that have no market benchmark. Supplier paths are scenarios, not predictions, and the money panel reports the cumulative cost over the selected horizon rather than annualising a short shock.
- **Recommendations (no AI):** rules-based, scored by yearly value × evidence confidence × ease. Money is kept in four separate buckets that are never mixed: *identified savings* (evidence-backed, best option per item so nothing is double-counted), *flagged for review* (no market explanation yet, deliberately excluded from savings), *expected* and *realised* (from the tracker; expected is never counted as realised). A quote below its minimum order, or labelled a possible substitute, is downgraded rather than counted. The LLM writes the plain-English advice and email from the calculated findings, referencing them by ID, so it can't invent figures.
- **Interface:** Streamlit multi-page app (Overview, Cost Intelligence, Forecast & Scenarios, Savings, Data & Checks) with Altair charts, one shared item selection across pages, and the conclusion and next steps above the evidence. A "Midnight Slate" dark theme defined from a single set of design tokens, with WCAG AA contrast throughout and colour reserved for financial meaning. Deployed on Streamlit Community Cloud.
- **Accuracy:** 99.2% of line-item fields (595/600) extracted correctly on the 24 sample invoices, with all 48 dates and invoice numbers correct and 23 of 24 invoices read perfectly (measured with evaluate.py against known answers). The one error was a garbled product name ("J000asmine Rice") with all numbers right.
- **Tests:** an offline end-to-end pipeline test with a fake LLM, and 20 unit tests for matched-period benchmarks, currency conversion, forecasts, scenarios, profit impact, quote savings, minimum-order handling and recommendation ranking. Plus a scripted walkthrough of every page and interaction against a strict Streamlit stand-in.

**Technical components:** Python, Streamlit, pandas, Altair, pdfplumber, pydantic, requests, FRED (IMF Primary Commodity Prices, Federal Reserve exchange rates), Featherless AI (OpenAI-compatible API), Qwen2.5-72B-Instruct, reportlab (sample data).

## Real-world impact
For a café spending about $4,200 a month on supplies, MarginGuard surfaced about $2,100 a year in increases, a large share of profit when margins are about 5%. The bigger value is the decision: which increase to challenge, with market evidence and a drafted email; which to plan for in menu prices; where to ask for a fixed price before more pressure arrives; and where switching saves money now. The same approach extends to any small business buying from suppliers.

## Challenges we ran into
_In your own words. Ideas from what actually happened:_
- _Our first accuracy score was 80%, but the AI was right: our grader demanded exact product-name matches. Fixing the grader took it to 99.2%. The lesson was to check the measurement before blaming the model._
- _Our first benchmark compared a supplier's April–September rise with April–July market data and declared it "above market". It looked impressive and it was unfair. Rebuilding it around matched periods meant our headline number dropped, which was the right trade._
- _Streamlit renders text between two dollar signs as a maths formula, which quietly mangled every price on screen._
- _Reliable live supplier prices need a paid API, so we chose verified invoice data and quotes the user enters over invented listings._

## Accomplishments we're proud of
_In your own words. E.g. 99.2% measured extraction accuracy; catching shrinkflation; matched-period benchmarking that refuses to overclaim; keeping identified, flagged, expected and realised savings strictly separate; every number calculated in code and unit-tested._

## What we learned
_Your own words._

## What works and what doesn't
Works: text-based PDF invoices in varied layouts, cross-supplier matching, increase/shrinkflation/supplier-gap detection, market benchmarks for oils, wheat, rice, sugar, coffee and poultry, market outlook and supplier scenarios, profit impact simulator, ranked recommendations, quote comparison, AI action plan and email.

Not yet: photo or scanned invoices (needs OCR); live online supplier search (needs a paid shopping/search API, so the app uses your own invoices, quotes you enter and a web-search shortcut, and never invents prices); benchmarks for dairy, eggs, fresh produce, packaging and electricity; persistent storage for the action tracker (it lives in the browser session and exports to JSON, because Streamlit Community Cloud has no database); direct connection to accounting software or email; testing on large volumes of real invoices. Commodity prices lag about two months, so the newest increases can only be flagged for review, not benchmarked. Sample invoices are synthetic; market data is real. The AI occasionally garbles a product name (1 of 24 sample invoices).

## What's next
Photo invoices via a vision model, more benchmarks (dairy auctions, local wholesale market prices, electricity tariffs), a real database behind the action tracker so realised savings build up over time, live supplier catalogues where a permitted API exists, forwarding invoices by email, connecting to accounting tools, and alerts when a new invoice arrives.

## AI use disclosure
Besides the LLM inside the product (Qwen2.5-72B via Featherless AI), we used Claude (an AI assistant) to help plan the project and to write most of the code, sample data, market-data integration and documentation. We ran it, measured its accuracy, fixed the problems we found, deployed it, and reviewed and edited everything, including this write-up, which Claude helped draft. All sample invoices and businesses are fictional; market data is real and sourced from FRED.

## Built with
python, streamlit, pandas, altair, pdfplumber, pydantic, fred, featherless-ai, qwen, llm
