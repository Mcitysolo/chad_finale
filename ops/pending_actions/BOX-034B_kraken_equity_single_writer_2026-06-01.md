---
# BOX-034B — kraken_equity Single-Writer Unification (follow-up to BOX-034A)
Date: 2026-06-01 · Status: PENDING (spec; no implementation until operator GO)
Risk class: correctness/governance (currency-truth of total_equity); value impact de-minimis (~0.06%)

## 1. Problem (discovered 2026-06-01 during BOX-034A Inc 3 Step 0b)
Two services write the same key `kraken_equity` to runtime/portfolio_snapshot.json on different timers with no coordination — the same dual-writer class as the ibkr_equity defect in BOX-034A §1:
- chad/ops/portfolio_snapshot_publisher.py — after Inc 3 Step 0b, writes kraken_equity in CAD by converting kraken_balances.json `usd_equivalent` via a live band-validated USDCAD rate (~252.85 CAD); sets kraken_equity_currency=CAD, _ok=true.
- chad/portfolio/merge_kraken_into_snapshot.py — writes kraken_equity in CAD directly from `kraken_fiat_cad` (ZCAD 1:1, ~255.59 CAD); does NOT set kraken_equity_currency_ok (null).
Last writer wins. Both are now CAD (an improvement over the pre-Step-0b USD-vs-CAD race), but they (a) disagree in value (~252.85 vs ~255.59) from different valuation methods, and (b) emit an inconsistent kraken_equity_currency_ok (true vs null).

## 2. Impact
- total_equity = ibkr + coinbase + kraken. BOX-034A Inc 3 derives total_equity_currency_ok from its legs; while kraken's _ok flickers null (merge writer), total_equity_currency_ok cannot stably be true. This GATES a clean Inc 3 total_equity tagging — until kraken has one CAD+ok=true writer, the aggregate tag fail-closes intermittently.
- Value impact de-minimis today (kraken ~0.06% of book); this is correctness/governance, not a sizing emergency.

## 3. Decision needed (before fix)
Pick the single canonical kraken_equity writer + valuation:
- Investigate whether merge's `kraken_fiat_cad` (ZCAD) includes crypto holdings or is fiat-only. The publisher's FX-converted usd_equivalent values crypto+fiat; the ~$3 gap is likely crypto inclusion and/or ZCAD-1:1 vs live-FX.
- Choose the more complete/correct valuation, make it the SOLE writer (retire the other's kraken_equity write), ensure the survivor sets kraken_equity_currency=CAD + _ok.

## 4. Acceptance (proven to bind)
- grep: exactly one writer of `kraken_equity`.
- kraken_equity carries currency=CAD + _ok=true, no flicker over a 10-min window.
- The chosen valuation documented and the ~$3 method gap reconciled.

## 5. Observability (carry-over from Inc 2 / Step 0b)
Degraded fail-closed states (ibkr_equity_currency_ok=false, kraken_equity_currency_ok=false/null, ibkr_equity_usd_display=null) are now surfaced IN-PAYLOAD, not via exit code. The health monitor must watch these flags (alert on persistence, not single blips) so a degraded-but-safe state is never silent.

## 6. Out of scope
total_equity/account_equity tagging (BOX-034A Inc 3); the allocator no-FX sum (correct once legs are CAD); coinbase (0.0, moot).

## 7. Implementation gate
No code until operator GO. Then single-writer unification + verify §4 live.
---
