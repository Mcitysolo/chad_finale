### 1. Title
CHAD Position Truth v2 Engine — Architectural Design

### 2. Status
PROPOSED — design + stub only; production wiring requires separate
operator authorization (see §12 migration plan).

### 3. Source authority
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
- `reports/forensic_audits/POSITION_AUTHORITY_INVENTORY_20260527.md`
- `ops/pending_actions/R1_canonical_position_authority_gap_2026-05-27.md`
- `CHAD_LATEST_SSOT_9.3.md` §5.1 (Reconciliation Layer) and §14.1
  (positions_snapshot.json staleness footnote)
- `CLAUDE.md` governance rules #1 (one change at a time), #3 (no
  direct config mutation — Pending Actions only), #5 (never modify
  anything inside runtime_FREEZE_* or data_FREEZE_*)

### 4. Problem statement

Two existing position-truth surfaces have no code-enforced
single-writer authority:

- `runtime/positions_snapshot.json` — periodic IBKR portfolio poll,
  ~5 min cadence (`ttl_seconds=300`), schema = `{ts_utc, source,
  positions: [<list of {symbol, position, secType, conId, ...}>],
  positions_count, ttl_seconds}`.
- `runtime/ibkr_paper_ledger_state.json` — event-driven append log,
  schema = `{<sha256_hash>: {symbol, qty, secType, conId, account_id,
  strategy, opened_at_utc, ...}}`. **No global `ts_utc`.** Per-entry
  `opened_at_utc` is the freshness signal; per-symbol totals require
  summing entries with the same `symbol`.

Today's forensic audit (`20260527T142951Z`) observed:

- `positions_truth.broker_authority_status=RED` for ~96 minutes
  during the morning soak window.
- 19 vs 18 count mismatch (ledger reported 19; snapshot reported 18
  on one cycle).
- `alpha_futures|MGC` drift classified `broker_truth_missing` —
  position present in one surface, absent in the other.
- M6E 58-vs-56 mismatch caught by the morning's
  `chad/validators/position_authority.py` — a fill landed during
  the snapshot poll window.

Picking either surface as canonical creates a known weak spot.
Snapshot-as-canonical misses just-landed fills (M6E pattern).
Ledger-as-canonical misses fills that CHAD never received an event
for (operator manual broker action; broker truth without our
acknowledgement). The institutional answer is a third surface that
merges both with explicit conflict-resolution rules and full
provenance.

### 5. Goals

- Replace the binary "which file wins" question with a per-symbol
  merged view backed by provenance.
- Fail closed on uncertainty — never guess a qty.
- Make every position state recoverable to its source events
  (which surface said what, when, with what sha256).
- Make routine race-conditions (cadence lag) **classified**, not
  surprising.
- Make real bugs (missed fill events) **explicit** and alerting.
- Preserve forward compatibility with multi-broker (Kraken, Coinbase)
  when live by parameterising the source list (this phase ships
  with IBKR-only sources but the engine is shape-agnostic per source).

### 6. Non-goals

- This design does **NOT** retire `positions_snapshot.json` or
  `ibkr_paper_ledger_state.json`. Both continue to write as today.
- This design does **NOT** automatically resolve any current
  mismatch. The engine classifies and surfaces conflicts; it does
  not patch them.
- This design does **NOT** introduce any live-trading authorization.
- This design does **NOT** mutate the existing
  `runtime/positions_truth.json` publisher.

### 7. Architecture overview

```
  positions_snapshot.json (poll, 5-min cadence) ──┐
                                                  ├──> PositionTruthEngine.run()
  ibkr_paper_ledger_state.json (events, ~realtime) ┘            │
                                                                ▼
                                                  position_truth_v2.json   (atomic write)
                                                                │
                                                                ▼
                                                  chad.validators.position_truth_v2
                                                                │
                                                                ▼
                                                  downstream consumers (READ-ONLY)
                                                  (deferred to migration §12 Step 3)
```

The engine is a **pure function** of its two input artifacts: same
inputs → same output. It writes only its own output file. It never
reads or writes any other `runtime/*.json`. It never calls the
broker. It never publishes Telegram.

### 8. Schema — `position_truth_v2.v1`

```jsonc
{
  "schema_version": "position_truth_v2.v1",
  "ts_utc": "<iso>",                          // engine run timestamp
  "engine_version": "1.0.0",
  "authority_mode": "merged_with_provenance",
  "ttl_seconds": 360,                         // consumer staleness budget
  "source_artifacts": {
    "snapshot": {
      "path": "runtime/positions_snapshot.json",
      "ts_utc": "<iso>",                      // snapshot.ts_utc (canonical)
      "sha256": "<hex>",                      // sha256 of the file at read time
      "age_seconds": <float>                  // engine ts_utc - source ts_utc
    },
    "ledger": {
      "path": "runtime/ibkr_paper_ledger_state.json",
      "ts_utc": "<iso>",                      // max(per-entry opened_at_utc)
      "sha256": "<hex>",
      "age_seconds": <float>
    }
  },
  "positions": {
    "<symbol>": {
      "qty": <int|float|null>,                // chosen by authority_decision
      "side": "LONG|SHORT|FLAT|UNKNOWN",
      "value_source": "snapshot|ledger|both|DISAGREEMENT|FAIL_CLOSED",
      "snapshot_value": <int|float|null>,
      "ledger_value": <int|float|null>,
      "agreement": <bool>,
      "delta": <int|float|null>,              // ledger_value - snapshot_value
      "delta_reason": "...",                  // see §9 (one of the M1-M5 reasons)
      "merge_rule": "M1|M2|M3|M4|M5",
      "authority_decision": "snapshot|ledger|FAIL_CLOSED",
      "fail_closed": <bool>,
      "last_reconciled_utc": "<iso>",
      "provenance_chain": [
        {"surface": "snapshot", "ref": "<conId|symbol>", "ts_utc": "<iso>"},
        {"surface": "ledger",   "ref": "<sha256_key|symbol>", "ts_utc": "<iso>"}
      ]
    }
  },
  "global_authority_health": "GREEN|YELLOW|DEGRADED|RED",
  "fail_closed_symbols": [...],
  "warnings": [...],
  "errors": [...]
}
```

Notes:
- `qty=null` ⇔ `value_source=FAIL_CLOSED` ⇔ `fail_closed=true`.
  Consumers must treat `null` as "do not trade this symbol."
- `provenance_chain` must contain **at least one** entry per symbol;
  empty chain is a validator error (§ Phase C).
- Per-symbol qty is the **collapsed sum** across all ledger entries
  sharing the same symbol. The collapse step is documented in
  `PositionTruthEngine._collapse_ledger_to_symbol`.

### 9. The 5 merge rules (verbatim — these become the engine's contract)

Notation: for each symbol present in either source, let
`s = snapshot_value` and `l = ledger_value` (either may be `None`
if the symbol is absent from that source). `delta = l - s` (using
`0` for `None` when both are not `None`; otherwise see specific
rule).

#### Rule M1 — Both surfaces agree
- **Trigger:** `s is not None AND l is not None AND s == l AND side(s) == side(l)`
- **Classification:** `value_source=both`, `agreement=true`,
  `delta=0`, `delta_reason=in_agreement`, `merge_rule=M1`.
- **Downstream effect:** trust the value; no `fail_closed`.
- **Example:** `AAPL qty=100 LONG` on both surfaces.

#### Rule M2 — Ledger newer than snapshot, integer-bounded delta
- **Trigger:** `l is not None AND s is not None AND
  ledger.ts_utc > snapshot.ts_utc AND
  abs(delta) is a whole-contract amount (i.e. delta == int(delta))
  AND (ledger.ts_utc - snapshot.ts_utc) < 2 × snapshot_cadence (600s)`.
- **Classification:** `value_source=ledger`,
  `delta_reason=ledger_lag_within_cadence`, `merge_rule=M2`,
  `authority_decision=ledger`.
- **Downstream effect:** trust ledger; no `fail_closed`;
  provenance includes both surfaces; `global_health` worsens to
  YELLOW unless something more severe fires elsewhere.
- **Example:** M6E `snapshot=58, ledger=56` (a fill landed during
  the 5-min poll window; the ledger captured it event-driven before
  the next snapshot tick).

#### Rule M3 — Ledger older than snapshot, integer-bounded delta
- **Trigger:** `l is not None AND s is not None AND
  snapshot.ts_utc > ledger.ts_utc AND
  abs(delta) is a whole-contract amount`.
- **Classification:** `value_source=snapshot`,
  `delta_reason=missed_fill_event`, `merge_rule=M3`,
  `authority_decision=snapshot`, `fail_closed=true`.
- **Downstream effect:** trust snapshot for display; **FAIL_CLOSED
  on this symbol** for new entries (CHAD missed an event we should
  have seen); emit warning; require operator review.
- **Example:** snapshot says 18 of a symbol, ledger says 17 — CHAD
  missed a fill event from the broker.

#### Rule M4 — Fractional or non-integer disagreement / side mismatch
- **Trigger:** `(l is not None AND s is not None) AND
  (abs(delta) != int(abs(delta)) OR side(s) != side(l) OR
   sign(s) != sign(l))`.
- **Classification:** `value_source=DISAGREEMENT`,
  `delta_reason=structural_mismatch`, `merge_rule=M4`,
  `authority_decision=FAIL_CLOSED`, `fail_closed=true`.
- **Downstream effect:** **FAIL_CLOSED globally on this symbol**;
  ERROR-level alert; block any new entry on the symbol; require
  operator review.
- **Example:** snapshot=1 LONG, ledger=1 SHORT (impossible without
  a bug or operator manual broker action against CHAD's intent).

#### Rule M5 — Either surface stale beyond TTL
- **Trigger (per-symbol):** `snapshot.age_seconds > snapshot_ttl OR
  ledger.age_seconds > ledger_ttl OR (s is None AND l is not None)
  OR (s is not None AND l is None)`.
- **Classification:** `value_source=FAIL_CLOSED`,
  `delta_reason=stale_source` (or `missing_in_one_source` for the
  one-sided absence case), `merge_rule=M5`,
  `authority_decision=FAIL_CLOSED`, `fail_closed=true`,
  `qty=null`.
- **Downstream effect:** **FAIL_CLOSED globally on this symbol**;
  do not guess; alert. If the staleness is at the source-file level
  (entire file aged out), every symbol gets M5 and
  `global_health=RED`.
- **Example A (file-level):** snapshot publisher hung — every symbol
  M5.
- **Example B (symbol-level):** `alpha_futures|MGC` present in
  ledger but absent in snapshot (broker_truth_missing pattern from
  this morning's audit) — symbol classified M5 with
  `delta_reason=missing_in_one_source`.

#### Rule precedence (deterministic ordering)
When more than one rule could apply, the engine evaluates in this
order and emits the first match:

1. **M5** (stale / missing in one source) — checked first because
   it's the strictest safety condition.
2. **M4** (structural mismatch).
3. **M1** (perfect agreement).
4. **M2** (ledger newer, integer delta within cadence).
5. **M3** (snapshot newer, integer delta — i.e. missed fill).

This precedence is exercised by `test_real_world_19_vs_18_classifies_correctly`.

### 10. Global authority health computation

Mapping from per-symbol classifications to `global_authority_health`,
selecting the **max severity** across all symbols:

| Per-symbol rule | Health contribution |
|---|---|
| M1 | GREEN |
| M2 | YELLOW |
| M3 | DEGRADED |
| M4 | RED |
| M5 | RED |

`global_authority_health = max severity across all per-symbol rules`,
where severity ordering is `GREEN < YELLOW < DEGRADED < RED`.

Examples:
- 19 M1 symbols, 0 others → `GREEN`.
- 18 M1 symbols, 1 M2 → `YELLOW`.
- 18 M1 symbols, 1 M3 → `DEGRADED`.
- 17 M1 symbols, 1 M3, 1 M5 → `RED`.

### 11. Failure modes (explicit table)

| Failure | Engine behavior | Recovery |
|---|---|---|
| `positions_snapshot.json` missing | every symbol M5; `global=RED`; engine still writes output | Wait for next IBKR poll; investigate `chad-positions-snapshot.timer` |
| `ibkr_paper_ledger_state.json` missing | every symbol M5; `global=RED`; engine still writes output | Restart paper executor; investigate ledger writer |
| Schema mismatch on either source | per-source `errors[]` entry; do not coerce; engine emits `global=RED` with `errors` populated | Operator review; do not auto-fix |
| Engine crash mid-write | Atomic write pattern (`tmp + os.replace`) prevents partial; consumer reads previous file | Auto-recover on next engine cycle |
| `ts_utc` parse failure on snapshot | Treat snapshot as M5-stale for all symbols; record `errors[]` | Source publisher fix; do not coerce |
| `opened_at_utc` parse failure on a ledger entry | That entry is dropped from the per-symbol collapse with a warning; **does not** crash the engine | Source publisher fix |
| Clock skew between hosts (multi-broker future) | Engine uses **server local UTC**; documented assumption. Per-source `age_seconds` is computed against `engine.ts_utc`, not against each other directly | Document only; revisit when multi-broker lands |
| Ledger entry has `qty=0` | Treated as FLAT; collapses to 0 in per-symbol sum; not an error | n/a |
| Snapshot entry has `position=0` | Treated as FLAT; not an error | n/a |
| Symbol appears in ledger N times with conflicting signs | Engine sums first; if collapsed sum is non-zero it's a valid LONG/SHORT; if 0 it's FLAT; per-entry conflicts surface in `provenance_chain` for operator review | Operator review of strategy attribution |

### 12. Migration plan (NOT executed in this phase)

The engine ships as **observation-only** in this phase. The
migration to authoritative status happens in three steps, each
separately operator-authorized per CLAUDE.md rule #3:

- **Step 1** (this phase delivers prerequisites only): engine
  writes `runtime/position_truth_v2.json` read-only. Downstream
  consumers continue reading the existing `positions_truth.json`.
  No production scheduler runs the engine yet — this phase ships
  the CLI only.
- **Step 2** (separate PA): a shadow consumer compares
  `position_truth_v2` against the existing `positions_truth` for
  **7 calendar days**; emits a daily divergence report. Both files
  coexist; the existing publisher remains authoritative.
- **Step 3** (separate PA): downstream consumers
  (`chad.core.live_loop`, `chad.risk.*`, position_guard rebuilder)
  are repointed to `position_truth_v2`. The legacy
  `positions_truth.json` publisher is retired or moved to
  diagnostic-only output.

This phase delivers Step 1's prerequisites only. Step 2 and Step 3
are separate Pending Actions. The engine's `write_truth_v2()`
method exists in code but is not wired into any systemd unit.

### 13. Test plan (delivered in Phase D below)

- One test per merge rule (M1-M5) using small synthetic fixtures.
- Real-world fixtures derived from this morning's incident:
  - `real_19_vs_18` — 19 ledger symbols, 18 snapshot symbols
    (MGC missing from snapshot), exercising rule precedence
    (M5 wins over hypothetical M2 elsewhere).
  - `mgc_drift` — minimal reproduction of the
    `alpha_futures|MGC broker_truth_missing` pattern.
  - M6E `m2_ledger_lag` — 58-vs-56 with ledger newer by 90s,
    classified M2 (not M3) because of the timestamp ordering.
- Stale TTL behavior (M5).
- Schema validation by the Phase C validator.
- Atomic-write safety (engine writes via tmp+rename).
- **Engine never mutates source files** (asserted via sha256
  before/after).
- CLI `--check` mode does not write any file.

### 14. Open questions for operator

- `snapshot_cadence`: confirmed 5 min from
  `chad-positions-snapshot.timer` (or equivalent). The engine
  defaults assume 300s.
- `snapshot_ttl`: PROPOSED **600 seconds** (2× cadence). The
  engine defaults reflect this; the live `positions_snapshot.json`
  already advertises `ttl_seconds=300` so consumers that read that
  field should not be coupled to the engine's 600s budget.
- `ledger_ttl`: PROPOSED **60 seconds** (event-driven, should be
  very fresh — older than 60s implies the writer hung).
- Both TTLs are constructor arguments to `PositionTruthEngine`;
  defaults shipped in code; runtime override is a future PA.
- Multi-broker (Kraken, Coinbase) sources are not in scope for v1
  but the per-source schema (`source_artifacts.<broker>`) is
  shaped to accept them additively.

### 15. Acceptance criteria for this design phase

- This document exists at `docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md`.
- Schema is implementable as `@dataclass` without ambiguity
  (delivered in `chad/schemas/position_truth_v2.py`).
- Each of the 5 merge rules is independently testable
  (delivered in `chad/tests/test_position_truth_engine.py`).
- No production runtime mutation occurs in this phase
  (engine's CLI is `--check` only; `write_truth_v2()` is not
  invoked against any production path).
- Tests cover M1-M5 with real-world-derived fixtures.
- Existing `runtime/positions_truth.json` publisher is **not**
  modified.
- Existing `chad/validators/position_authority.py` (this morning's
  R1 read-only validator) is **not** modified — it remains the
  current operator-facing surface until Migration Step 3.

### 16. No-live confirmation

This design document does not authorize live trading.
`ready_for_live` must remain false.
`allow_ibkr_live` must remain false.
`allow_ibkr_paper` must remain true.
No broker orders may be placed or cancelled under this design.
