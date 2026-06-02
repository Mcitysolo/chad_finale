# BOX-034B — kraken_equity Single-Writer Unification (follow-up to BOX-034A)
Date: 2026-06-01 · Status: PENDING (corrected after Step 1 investigation; no implementation until operator GO)
Risk class: correctness/governance; value impact de-minimis (~0.06% of total equity)

## 1. Problem (CORRECTED after Step 1 investigation 2026-06-01)
- LIVE writer of kraken_equity = chad/ops/portfolio_snapshot_publisher.py (5-min timer): writes ~255.55 CAD with kraken_equity_currency=CAD + _ok=true.
- chad/portfolio/ibkr_portfolio_collector_v2.py (2-min timer) PRESERVES the kraken value via read-through but RE-EMITS WITHOUT the kraken currency tags (strips kraken_equity_currency/_ok). This tag-strip — publisher(tagged) <-> collector(tag-stripped) every 2-5 min — is the real defect, NOT a publisher-vs-merge value race.
- chad/portfolio/merge_kraken_into_snapshot.py / chad-portfolio-merge.service: DISABLED, inactive, no timer (ordering-only), AND schema-broken (reads balances["ZCAD"], which no longer exists post-normalization -> would write 0.0). Effectively retired; not a live participant.

## 2. Impact
total_equity = ibkr + coinbase + kraken; BOX-034A Inc 3 derives total_equity_currency_ok from its legs. While the collector strips kraken's tags every cycle, total_equity_currency_ok cannot stably be true. This GATES a clean Inc 3 total tag. Value impact de-minimis.

## 3. Decision (after Step 1)
Canonical kraken_equity writer = portfolio_snapshot_publisher.py (only enabled, currency-tagged, fail-closed writer). Fixes:
- Collector: preserve kraken_equity_currency/_ok through its read-through (or be designated a non-kraken writer) so tags stop flickering. THIS unblocks Inc 3.
- Publisher: read NATIVE CAD fiat directly (balances["CAD"], ~252.85) instead of round-tripping usd_equivalent through FX — eliminates the stale-FX artifact (3a-a). Tag currency=CAD + _ok=true.
- merge: confirm masked/retired (already dead+broken).

## 3a. Known value-accuracy gaps (de-minimis; documented, NOT blocking Inc 3)
Currency-truth (the CAD tag) is independent of value-accuracy. Two gaps on the kraken leg (~0.06% of total), explicitly accepted and deferred:
- (a) Stale-FX round-trip: usd_equivalent = CAD fiat * 0.73 (non-reciprocal _FIAT_USD_FALLBACK) then * 1.3845 -> +1.07% (~$2.70). RESOLVED by the native-CAD read in §3.
- (b) BTC omission: kraken_balance_provider drops crypto when no live USD price is supplied; BTC 0.0012 (~$166 CAD, ~40% of true kraken ~$419) is missing from both writers. DEFERRED: value crypto in CAD when the holding grows enough to matter. Track as a follow-up.

## 4. Acceptance (proven to bind)
- grep: exactly one writer SETS kraken_equity (publisher); collector preserves both value AND tags.
- kraken_equity carries currency=CAD + _ok=true with NO tag flicker across a full collector+publisher cycle (10-min window).
- Native CAD fiat value (no round-trip); merge masked/inert.

## 5. Observability (carry-over from Inc 2 / Step 0b)
Degraded fail-closed states (ibkr/kraken _currency_ok=false, ibkr_equity_usd_display=null) are surfaced IN-PAYLOAD, not via exit code. Health monitor must watch these flags (alert on persistence, not single blips).

## 6. Out of scope
total_equity/account_equity tagging (BOX-034A Inc 3); allocator no-FX sum; coinbase (0.0); BTC/crypto CAD valuation (deferred per 3a-b).

## 7. Implementation gate
No code until operator GO. Then collector tag-preserve + publisher native-CAD read + merge mask, verify §4 live.
