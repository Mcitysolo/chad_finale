# Position-Authority Surface Inventory — 2026-05-27 (R1 / TRUTH-RECONCILE-1)

## Purpose
Read-only inventory of every code surface that **writes** or **reads** the four
paper-position truth files. Companion artefact for `chad/validators/position_authority.py`.

This document **does not** designate a canonical writer. Per CLAUDE.md
governance rule #3 ("No direct config mutation. Risk caps, live mode, and
strategy config must be prepared as Pending Actions only"), the canonical
authority designation is operator-domain and lives in
`ops/pending_actions/R1_canonical_position_authority_gap_2026-05-27.md`.

The validator is fail-closed: it reports mismatches and exits non-zero, but it
does not pick a winner.

## Files in scope

| Path | Schema shape | Writer (publisher) | Cadence |
|---|---|---|---|
| `runtime/positions_snapshot.json` | `{positions: list[{symbol, position, conId, secType, avgCost, currency}], positions_count, source, ts_utc, ttl_seconds}` | `chad/portfolio/ibkr_portfolio_collector_v2.py` (`chad-positions-snapshot.service` + `.timer`) | 5 min |
| `runtime/ibkr_paper_ledger_state.json` | `{<sha256-hash>: {symbol, qty, secType, conId, strategy, avg_cost, opened_at_utc, account_id, tags, source_strategies, ...}}` | `chad/portfolio/ibkr_paper_ledger_watcher.py` (`chad-ibkr-paper-ledger-watcher.service` + `.timer`) | 15 min |
| `runtime/position_guard.json` | `{"<strategy>|<symbol>": {open, opened_at, updated_at_utc, strategy, symbol, side, quantity, source}}` | `chad/core/live_loop.py` (`_rebuild_guard_from_paper_ledger`), `chad/core/position_guard.py`, broker-truth rebuilds | per-cycle (≈30 s) |
| `runtime/reconciliation_state.json` | `{status, broker_source, chad_state_source, counts, worst_diff, mismatches, drifts, ts_utc, ttl_seconds, ...}` | `chad/ops/reconciliation_publisher.py` (`chad-reconciliation-publisher.service` + `.timer`) | per-cycle |
| `runtime/positions_truth.json` | `{broker_authority_status, broker_authority_reason, truth_ok, truth_source, positions, cash, ts_utc, schema_version=positions_truth.v1, ...}` (classified output, downstream of the above 4) | `chad/ops/lifecycle_truth_publisher.py` | per-cycle |

## Known readers

Grep result for `positions_snapshot.json` or `ibkr_paper_ledger_state.json`
inside `chad/` (excluding `.bak` / `.deprecated`):

```
chad/risk/portfolio_var.py
chad/risk/profit_lock.py
chad/portfolio/ibkr_portfolio_collector_v2.py    [WRITER for snapshot]
chad/portfolio/portfolio_engine.py
chad/ops/lifecycle_replay_drift_audit.py
chad/portfolio/ibkr_paper_ledger_watcher.py      [WRITER for ledger]
chad/ops/execution_quality_publisher.py
chad/ops/lifecycle_replay_coverage.py
chad/ops/daily_chad_report.py
chad/ops/lifecycle_truth_publisher.py            [CLASSIFIER]
chad/execution/ibkr_client_ids.py
chad/intel/advisory_engine.py
chad/intel/synthetic_analyst.py
```

Test surfaces:
```
chad/tests/test_pr09_position_truth_contract.py
chad/tests/test_positions_truth_classifier.py
chad/tests/test_probe_bag_quotes.py
chad/tests/test_ibkr_portfolio_collector_v2_positions.py
chad/tests/test_portfolio_surface.py
```

## Canonical authority — UNDESIGNATED in code

Per the audit (§17 R1-A), there is no code-enforced single canonical writer.
Documentation hints:
- `ops/pending_actions/BOX-047_dual_ledger_authority_policy.md` — acknowledges
  the dual-authority concern; Pending Action only; no code-enforced resolution.
- `CLAUDE.md` "Position-Guard Rebuilder Policy (GAP-028 Option B — PERMISSIVE)"
  describes the rebuilder mirroring `trade_closer_state.queues` faithfully but
  does **not** pin which of snapshot/ledger is authoritative.
- SSOT v9.3 / forward errata: no explicit pin.

**Conclusion:** the validator must not pick a winner. It must surface
mismatches and exit non-zero, and the canonical-designation gap remains the
deliverable of R1.

## Validator contract

`chad/validators/position_authority.py`:
- Read-only by default; reads the 4 runtime files **only**.
- Never writes to `runtime/`.
- Optional `--output <path>` writes the structured report to an out-of-runtime path.
- Exit 0 if all surfaces agree.
- Exit 2 if any mismatch is detected.
- Prints JSON to stdout for pipe-into-health-monitor consumption.

Mismatch categories:
1. `missing_symbol` — symbol present in one surface, absent from another (per-symbol aggregation, hash-keyed ledger collapsed).
2. `extra_symbol` — inverse of (1).
3. `qty_mismatch` — same symbol, different aggregated quantity (within tolerance).
4. `side_mismatch` — same symbol, different sign of position.
5. `stale_ts` — file `ts_utc` older than configured threshold (default 1800 s).
6. `key_shape_mismatch` — one surface is symbol-keyed (snapshot positions list, guard) while another is hash-keyed (ledger); the validator collapses hashes to symbols via the `symbol` field of each ledger entry, but emits a warning if any ledger entry lacks `symbol`.

## Out of scope for this PA closeout

- Designation of the canonical writer (operator-domain, separate PA).
- Schema-version backfill (covered by SCHEMA-VERSION-1, MED).
- Cadence reconciliation (covered by the R1 follow-up PR).
- Any mutation of the runtime files themselves.
