# CONTRACT — W5A Measurement Layer → Stage-2 Harness Handoff

**Status:** contract only. W5A (Lane A) writes the fields; **Lane B** wires the
`chad/validation/` adapter to consume them. This wave makes NO `chad/validation/`
edit — that directory is frozen here. This document names every new field, its
null semantics, the join key, and the compatibility rules Lane B must honor so
the wiring lands without rework.

Source audit: `PLAN_W5A.md §1`; blast-radius record: `audits/W5A_BASELINE.md`.

---

## 1. Where the new fields live (and why the schema was NOT bumped)

Both new blocks ride **inside a `closed_trade.v1` payload**, as additive,
**optional**, nullable sub-objects — the top-level `schema_version` stays
`closed_trade.v1`.

**Why no `.v2` bump** (operator-decided, amending rider R2): `closed_trade` is
hash-chained AND exact-matched (`== "closed_trade.v1"`) by ≥5 non-test readers,
one of them the execution-critical Bug-B-Fix-B fill-consumed dedup
(`trade_closer.py:853`, whose skip risks FIFO re-processing — the INCIDENT-0723
re-entry class), and `verify_ledger_chain` recomputes the byte-exact hash **only
for exact `closed_trade.v1`**. A `.v2` would silently disable hash verification
and ripple through execution-critical code. So the blocks are **self-versioned**
(`implementation_shortfall.v1`, `mae_mfe.v1`) and the ledger schema is unchanged.
`audits/W5A_BASELINE.md` is the blast-radius map for any future lane that DOES
need a real `closed_trade.v2`.

**Contract rule for a reader:** the ABSENCE of these blocks is never data loss.
A `closed_trade.v1` row may or may not carry them (they are flag-gated:
`CHAD_TCA_STAMP`, `CHAD_E3_EXCURSION=stamp`). Absent ⇒ "not measured for this
lap", identical to an explicit `null` leg. Treat both as unknown, never as zero.

**EXS7 enforcement:** both sub-schemas are pinned in
`config/exterminator.json → schema_contracts.embedded` and validated by the
sentinel against the newest ledger row when present (shape drift = a break),
even though the top-level schema was not bumped (condition b).

---

## 2. `payload.implementation_shortfall` (schema `implementation_shortfall.v1`)

Written by `chad/analytics/implementation_shortfall.py::compute_lap_is`, stamped
at `trade_closer.to_payload` under `CHAD_TCA_STAMP=on`. Rides only on **admitted
laps** (the adapter's `trust_exclusion` + quarantine + non-futures gate already
decide which laps count — a field on an admitted row is, by construction, on a
trusted lap).

| Field | Type | Null semantics (rider R1 — honest nulls) |
|---|---|---|
| `schema_version` | str | always `"implementation_shortfall.v1"` |
| `is_usd` | float\|null | total implementation cost $ = Σ resolved legs; **null** when NO cost leg resolved (distinct from a genuine 0) |
| `is_bps` | float\|null | `is_usd / (decision_price·qty·mult) · 1e4`; null when `is_usd` or `decision_price` null (see `is_bps_reason`) |
| `is_r` | float\|null | `is_usd / stop_width_usd`; **null + `is_r_reason="no_stop_width_usd"`** for the strategies that don't stamp `stop_width_usd` (only 4 do) |
| `slippage_usd` | float\|null | Σ per-leg realized slippage $; null-with-reason when a leg has no `slippage_bps` / non-genuine status / missing fill |
| `fees_usd` | float\|null | Σ per-leg modeled fee $ (IBKR `FEES_*` join / Kraken native); null when unjoinable |
| `funding_usd` | **null** | always null; `funding_reason="not_modeled_paper_lane"` (greenfield — no source) |
| `opportunity_cost_usd` | **null** | always null; `opportunity_cost_reason="no_unfilled_qty_evidence"` (paper fills fully; no requested-vs-filled data) |
| `decision_price` | float\|null | the entry leg's benchmark: `submit_quote.ref_price` → `expected_price` fallback; null when neither present |
| `decision_price_source` | str\|null | `submit_quote_ref_price` \| `expected_price` \| `kraken_expected_price` \| null |
| `cost_basis_status` | str | `real` (slippage+fees resolved for every leg) \| `partial` \| `unavailable` |
| `*_reason` fields | str / list | per-leg reason codes for every null (never a bare null) |
| `legs` | list | per-fill `{fill_id, role, found, slippage_usd, fee_usd, decision_price}` |

The accounting numbers (`pnl`/`gross_pnl`/`net_pnl`/`pnl_breakdown`) are
**unchanged** by this block — it is descriptive. `pnl_breakdown.cost_basis_status`
deliberately stays `"unavailable"`; the real cost decomposition is HERE.

---

## 3. `payload.mae_mfe` (schema `mae_mfe.v1`)

Written by `chad/analytics/excursion_recorder.py::read_lap_excursion`, stamped at
`trade_closer.to_payload` under `CHAD_E3_EXCURSION=stamp` (best-effort). The
**authoritative** source is the overlay close-time sidecar
`data/exit_overlay/excursion_YYYYMMDD.ndjson` (schema `mae_mfe.v1`) — the stamp
is a convenience for the common case and is **absent** when the sidecar was not
yet written at mint (the overlay-prune-vs-mint race, D8). Lane B should prefer
the sidecar for completeness and use the stamp as a fast path.

| Field | Type | Semantics |
|---|---|---|
| `schema_version` | str | `"mae_mfe.v1"` |
| `mae_pct` / `mae_usd` | float\|null | max ADVERSE excursion (≤0), entry-relative; null + reason when entry/watermark/qty unknown |
| `mfe_pct` / `mfe_usd` | float\|null | max FAVORABLE excursion (≥0) |
| `excursion_source` | str | `watermark_bar_hilo` (true bar high/low folded) \| `watermark_point_only` |
| `hwm` / `lwm` | float\|null | the high/low watermarks the metric was computed from |
| `join` (stamp only) | obj | `{on:"strategy_symbol_temporal", opened_at_utc, closed_detect_utc}` |

MAE/MFE is a true bar-resolution watermark (R4), not a point sample — but it is
bounded by the daily-bar cache + ~72s cycle cadence; `excursion_source` says so.

---

## 4. The join key

- **Lap granularity (recommended):** `record_hash` — the closed_trade row's
  hash-chain identity, already surfaced into the adapter's `provenance` and the
  same key `is_quarantined` pins on. TCA/MAE-MFE fields travel on the row, so a
  reader that has the row has the fields.
- **Sidecar → lap join:** `(strategy, symbol)` + temporal (closest
  `opened_at_utc` to entry, `closed_detect_utc` to exit). Per-`strategy|symbol`
  single-value: a FIFO partial close or fast reopen can blur which lap owns an
  excursion — the sidecar row carries `opened_at_utc`/`closed_detect_utc` so the
  join can disambiguate; flag ambiguity, don't guess.
- **Per-leg TCA (the `legs` array):** keyed on `fill_id`. **NAMED LANE-B CHANGE:**
  the adapter does not currently surface `payload.fill_ids` into `provenance`, so
  a fill-level join can't be made through the adapter today. Lane B must surface
  `fill_ids` into `provenance` to consume per-leg TCA. (This is the one adapter
  edit the audit identified; it is Lane B's, not W5A's.)

---

## 5. The double-charge resolution (rider D10 — REPLACE, binding)

The cost_model (`chad/validation/cost_model.py`) **estimates** costs today:
`participation`/`volatility` default to 0, leaving a fixed conservative haircut
(`slippage_base_bps`, half-spread by liquidity tier, per-share/contract floors).
That estimate exists ONLY because real costs were unavailable. Now they are.

**Contract clause (D10 = replace):** for a lap whose
`implementation_shortfall.cost_basis_status == "real"`, Lane B MUST feed the
realized `slippage_usd + fees_usd` as the haircut and **turn the estimated
haircut OFF** for that lap — never both (that double-charges, the hazard flagged
at `trade_log_adapter.py:557-562`). For `partial`/`unavailable`/absent laps, Lane
B keeps the estimated haircut (no realized truth to replace it). Funding and
opportunity cost are always null in W5A (paper) — Lane B adds nothing for them.

---

## 6. Stability rules Lane B must honor

- **Additive, nullable only.** No new REQUIRED top-level key on `closed_trade.v1`
  (the adapter's required-field gate must still pass old rows). The blocks are
  optional; absence == unknown.
- **Schema stays `closed_trade.v1`.** Do NOT bump to `.v2` without also updating
  `verify_ledger_chain`'s exact-match (`trade_log_adapter.py:1070`) AND the ≥5
  execution/risk readers in `audits/W5A_BASELINE.md`. The additive-sub-schema
  design exists precisely to avoid that.
- **Import isolation** (`tests/validation/test_isolation.py`): the adapter reads
  the ledger as TEXT and must not `import chad.execution.*` / runtime / broker
  modules to consume these fields.
- **Output version:** bump the adapter's `stage2_trade_log` output version when
  Lane B surfaces the new fields, so downstream consumers detect the shape change.
- **Admitted laps only:** these fields ride on rows the adapter already admits
  (trusted, non-quarantined, non-futures) — the contract makes no claim about
  excluded laps.
