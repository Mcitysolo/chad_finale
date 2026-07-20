# PA — SIM-mark fill-price freshness gate (Wave-2A item 4, report-only)

**Status:** PROPOSED (no build in Wave-2A — this PA is the deliverable for item 4).
**Filed:** 2026-07-20 (branch `goal/wave2-books-cleanup`).
**Class:** paper-fill mark fidelity. Independent of the PFF1 double-book; same defect class
PFF1-Q4 flagged.

## Observation

The exit overlay's ACTIVE close of UNH printed the SIM fill at `fill_price = 423.00` while the
broker filled the same order at **424.88** — a **$1.88/sh (~$513 on 273 sh)** divergence
between the booked paper mark and broker truth.

## Diagnosis (traced, code-verified)

The paper executor prices the SIM fill from `runtime/price_cache.json` via
`chad.execution.paper_exec_evidence_writer.build_submit_quote_stamp`
(`paper_exec_evidence_writer.py:1040`, `source="price_cache_mid_or_last"`,
`confidence="ref_only_no_nbbo"`). The stamp on this fill:

```
ref_price=423.0, quote_ts=13:50:00, quote_age_s=55.3, quote_ttl_s=300,
confidence="ref_only_no_nbbo"
```

So the SIM mark is a **price-cache mid/last that was 55 s stale but still inside the 300 s
TTL**, with no NBBO. On a name whose ATR is ~3.4 %/day, 55 s is enough to drift ~$1.9. The
broker filled at the live tape; the SIM booked the stale cache scalar. `_SUBMIT_QUOTE_TTL_S =
300` (`paper_exec_evidence_writer.py:995`) is the only staleness bound and it is far too loose
to serve as a *fill-price* gate — it was set to the price-cache `ttl_seconds` (an evidence-TTL,
not a fill-mark tolerance).

## Proposed remediation (recommend a + b)

- **(a) Tight fill-price freshness gate.** For the *fill mark* specifically (distinct from the
  300 s evidence-TTL), reject/repull when `quote_age_s` exceeds a small bound (e.g. 15–30 s).
  When unmet, book the fill `pnl_untrusted` (so the trust filter excludes it from SCR /
  Stage-2) rather than banking a stale-priced round-trip as clean edge.
- **(b) Reconcile the SIM fill price to broker truth (forward-only).** When the harvester later
  reports the real fill (it already carries `ibkr_exec_id`), treat the broker fill price as the
  authoritative mark for the evidence record. This closes the gap even when (a)'s window is met
  but the tape still moved.
- **(c) Require NBBO for a fill mark** (`confidence != ref_only_no_nbbo`). Listed for
  completeness; weaker than (a)+(b) because NBBO is not always present in paper.

Recommend **(a) + (b)**: (a) caps the per-fill error at open time; (b) makes the booked mark
converge to broker truth once the real fill lands.

## Scope / guardrails for the eventual build

- Forward-only. Does NOT rewrite historical evidence (hash-chained ledger untouched — same
  invariant as the W2A ghost-scrub).
- The freshness bound is a new constant distinct from `_SUBMIT_QUOTE_TTL_S`; do not tighten the
  evidence-TTL (other readers depend on 300 s).
- Paper-only; no order-path change. One change, tested, verified per governance.
