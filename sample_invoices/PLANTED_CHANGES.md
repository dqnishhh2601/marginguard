# What MarginGuard should find in the sample invoices

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
