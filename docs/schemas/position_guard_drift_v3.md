# position_guard_drift schema — v3 (P0A-A5, 2026-07-14)

Emitted by `chad.ops.reconciliation_publisher._emit_position_guard_drift` to
`runtime/position_guard_drift.json`; consumed by
`ops.live_readiness_publish._resolved_reconciliation_status`. Produced by the
pure detector `chad.core.position_guard.detect_guard_vs_broker_drift_v2`.

## What changed from v2

v3 adds the **`mixed_ownership_info`** drift kind and makes `drift_count`
count **actionable drift only**.

Symbols in the reconciliation **exclusion policy** (operator-owned pre-existing
broker positions — `config/reconciliation_exclusions.json`:
`exclusion_policy` + `broker_preexisting_symbols`, e.g. BAC/SPY/LLY/MSFT) have a
broker total that is a **mix** of the operator's shares and any CHAD lots.
Comparing CHAD-tracked lots against that combined total is not like-with-like,
so on such symbols a would-be `qty_mismatch` / `broker_untracked_position` /
`phantom_guard_entry` is reclassified to `mixed_ownership_info`. These records
are **informational** and are **excluded from `drift_count`** — they never flip
live-readiness RED. (Before v3, four operator-owned symbols inflated
`drift_count` from 5 to 9 on 2026-07-13.)

If a signed operator baseline is recorded (`exclusion_policy[SYM].operator_baseline_qty`,
optional; none set today), the residual `broker - baseline` is netted and
reported; otherwise the delta is honestly reported as unattributable (nulls).

## Top-level fields

| field | type | notes |
|---|---|---|
| `schema_version` | str | `"position_guard_drift.v3"` |
| `ts_utc` | str | emit time |
| `ttl_seconds` | int | advisory freshness TTL |
| `drift_count` | int | **ACTIONABLE** drift only = phantom + broker_untracked + qty_mismatch. The live-readiness gate: `drift_count > 0 → RED`. |
| `info_count` | int | number of `mixed_ownership_info` records |
| `excluded_symbols` | list[str] | operator-owned symbols the detector treated as mixed-ownership |
| `snapshot_generation` | int/null | `_version` of the single atomic position_guard.json read |
| `counts_by_kind` | obj | keys: `phantom_guard_entry`, `broker_untracked_position`, `qty_mismatch`, `mixed_ownership_info` |
| `drifts` | list[obj] | per-symbol records |

## Per-record fields

Common: `symbol`, `drift_kind`, `is_excluded` (bool, NEW in v3), `guard_qty`,
`broker_qty`, `qty_delta`, `guard_keys`, `guard_strategies`, `broker_side`,
`broker_present`, `snapshot_generation`.

`mixed_ownership_info` records additionally carry (NEW in v3):
`operator_baseline` (float|null), `net_broker_qty` (float|null =
`broker_qty - operator_baseline`), `chad_vs_net_broker_delta` (float|null =
`guard_qty - net_broker_qty`).

## Consumer compatibility

`ops.live_readiness_publish` accepts `position_guard_drift.v1`, `.v2`, and
`.v3`; the gate reads only `schema_version` + `drift_count`, whose semantics are
identical across versions (v3's `drift_count` already excludes the informational
mixed-ownership records).
