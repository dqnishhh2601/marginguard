# Devpost submission draft (fill in the _italic_ bits)

## Project title
MarginGuard

## Short description (one line)
AI that reads a small restaurant's supplier invoices and catches the hidden price rises eating its profit.

## Track
AI + Business

## Links
- Demo video: _YouTube link (Public, 2–4 min)_
- GitHub: _https://github.com/.../marginguard_
- Live app: _Streamlit link_

---

## Inspiration / Problem statement
Small independent restaurants keep only about 5 cents of every dollar of sales as pre-tax profit, and food costs have risen about 35% since 2019 (National Restaurant Association). Supplier price increases rarely arrive as one announcement: they hide in a few cents more per line, a product renamed in a new billing system, or a 10kg bag that becomes 9kg at the same price. Owners don't have time to compare every invoice line by line, so the increases go unnoticed and quietly eat the margin.

**Target users:** owners and managers of independent cafés, restaurants, bakeries and food stalls who buy from several suppliers and have no finance team.

## What it does
MarginGuard turns a pile of supplier invoices into clear decisions:
1. **Reads** invoice PDFs in any layout and extracts every product, quantity, pack size and price.
2. **Matches** the same product across suppliers, renamed items and changed pack sizes.
3. **Measures** the real price per kg / litre / item over time, flags increases and shrinkflation, finds where another supplier is cheaper, and calculates the cost per year.
4. **Recommends** what to do, with a prioritised action plan and a ready-to-send negotiation email.

On our 24 sample invoices it found $2,104/year in hidden increases (4.2% of supplier spend) plus $250/year in savings from switching one item to the cheaper supplier.

## How we built it (technical approach)
- **Extraction:** pdfplumber pulls the invoice text; an open-source LLM (Qwen2.5-72B via Featherless AI) converts it to structured JSON, validated with pydantic.
- **Verification + self-correction:** code re-does the invoice maths (quantity × price = line total; lines = subtotal). If the AI's reading fails, it's told exactly what's wrong and asked to fix it. Remaining problems are shown to the user, never hidden.
- **Product matching:** the LLM groups product names across suppliers and naming changes into canonical products.
- **Analysis (no AI):** pandas calculates price per base unit, change vs the first recorded price, yearly impact from average monthly quantities, shrinkflation (pack shrinks while pack price holds) and cross-supplier gaps. All figures are deterministic.
- **Recommendations:** the LLM receives the calculated findings and writes advice and an email, referencing findings by ID. The app shows our calculated numbers, so the AI can't invent figures.
- **Interface:** Streamlit + Altair, deployed on Streamlit Community Cloud. Responses are cached so the sample demo loads instantly.
- **Accuracy:** _XX%_ of line-item fields extracted correctly on the 24 sample invoices (measured with evaluate.py against known answers).

**Technical components:** Python, Streamlit, pandas, Altair, pdfplumber, pydantic, Featherless AI (OpenAI-compatible API), Qwen2.5-72B-Instruct, reportlab (sample data).

## Real-world impact
For a café spending ~$4,200/month on supplies, MarginGuard surfaced ~$2,100/year in increases, which is a large share of profit when margins are ~5%. It also gives owners evidence and a drafted email to negotiate, something they rarely have time to prepare. Independent restaurants are major local employers; protecting their margins helps them stay open. The same approach extends to any small business buying from suppliers: retail shops, salons, workshops.

## Challenges we ran into
_e.g. getting consistent JSON from the model, matching names reliably, deciding how to measure "normal" price noise._

## Accomplishments we're proud of
_e.g. the self-correcting extraction, catching shrinkflation, every number verifiable._

## What we learned
_your own words_

## What works and what doesn't
Works: text-based PDF invoices in varied layouts, cross-supplier matching, increase/shrinkflation/supplier-gap detection, AI action plan and email.
Not yet: photo/scanned invoices (needs OCR), direct connection to accounting software or email, testing on large volumes of real invoices. Sample data is synthetic.

## What's next
Photo invoices via a vision model, forwarding invoices by email (e.g. an Agentboxd inbox), connecting to accounting tools, and alerts when a new invoice arrives.

## Built with
python, streamlit, pandas, altair, pdfplumber, pydantic, featherless-ai, qwen, llm
