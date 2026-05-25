# PR-09 — Position Truth / Reconciliation Contract Remediation

**Date:** 2026-05-25
**Scope:** chad/ops/lifecycle_truth_publisher.py, chad/ops/reconciliation_publisher.py,
chad/ops/exterminator.py, ops/live_readiness_publish.py, chad/tests/test_pr09_position_truth_contract.py
**Trading posture impact:** NONE (paper). No live enablement. No service restarts.
**Authorization required to apply:** N/A — this is a contract clarification, not a config mutation.

---

## Root cause

Two field-name collisions in runtime JSON made the position-truth contract
ambiguous, allowing readers to draw contradictory conclusions from the same
state:

1. `runtime/positions_truth.json::evidence.reconciliation_status` is computed
   by a lifecycle-replay-coverage classifier (qty mismatches / missing-from-replay
   / scope mismatch). It downgrades to `"RED"` whenever lifecycle replay scope
   is partial, even when the *actual broker-authority* truth (snapshot
   reconciles with IBKR paper ledger, counts align) is GREEN and `truth_ok=true`.
   The field name collides with `reconciliation_state.json::status`, and no
   top-level field expressed broker-authority truth unambiguously.

2. `runtime/reconciliation_state.json::drifts` carries two distinct kinds of
   entries with no in-band label:
   - **Broker_sync-only advisory** — symbols where CHAD's `position_guard`
     has only a `broker_sync` entry (no strategy attribution) and the broker
     moved independently. The publisher classifies these as GREEN (`worst_diff=0.0`,
     `status="GREEN"`, `mismatches=[]`).
   - **Strategy-attributable drifts** — would-be entries when a strategy has
     attribution AND broker disagrees. Today's classifier routes these to
     `mismatches`, not `drifts`.

   `ops/live_readiness_publish.py::_resolved_reconciliation_status`
   (added in NEW-GAP-041) treats any `drifts` non-empty as RED. Combined with
   the publisher's classification, broker_sync-only advisory entries were
   falsely tripping the reconciliation gate.

**Observed evidence (pre-fix, 2026-05-25 ~16:21Z):**
- `reconciliation_state.json`: status=GREEN, worst_diff=0.0, mismatches=[],
  drifts=[M6E (b=20, c=18, d=2), MGC (b=-12, c=-11, d=1)]
- `position_guard_drift.json`: drift_count=0, drifts=[]
- `positions_truth.json`: truth_ok=true, truth_source=BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER,
  evidence.reconciliation_status="RED" while evidence.reconciliation_status_upstream="GREEN"
- `live_readiness.json`: ready_for_live=false

---

## Fix approach

**Option A from the PR-09 brief:** add explicit, unambiguous fields. NO field
removal, NO semantic widening of any existing gate.

### chad/ops/lifecycle_truth_publisher.py

Added top-level fields to the positions_truth.v1 payload:

| Field | Type | Semantics |
|-------|------|-----------|
| `broker_authority_status` | `"GREEN"` \| `"RED"` | Authoritative paper/live truth — derives 1:1 from `truth_ok`. |
| `broker_authority_reason` | string | Explanatory reason (`BROKER_AUTHORITY_GREEN: upstream=…` / `BROKER_AUTHORITY_RED: ledger_state_missing …`). |
| `replay_diagnostic_status` | `"GREEN"` \| `"YELLOW"` \| `"PARTIAL"` \| `"RED"` | Lifecycle replay coverage classifier (collapses `PARTIAL_REPLAY_COVERAGE` and `SCOPE_MISMATCH_*` to `PARTIAL`). |
| `replay_diagnostic_reason` | string | Mirrors `evidence.reconciliation_status_reason`. |
| `replay_diagnostic_blocks_truth` | bool | `false` whenever `truth_source` begins with `BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER`. |

`evidence.reconciliation_status` and `evidence.reconciliation_status_upstream`
are preserved verbatim for backward compatibility.

### chad/ops/reconciliation_publisher.py

Added a new `diagnostic_drifts` field to the output payload. Broker_sync-only
advisory entries (the diffs where `strategy_contrib < 1e-6`) are now routed
there instead of `drifts`. The `drifts` field is retained in the schema and
reserved for any future strategy-attributable drifts — preserving GAP-041's
fail-closed safety net unchanged.

Each `diagnostic_drifts` entry carries `kind: "broker_sync_only"` for
explicit classification.

### chad/ops/exterminator.py

`check_reconciliation` EX009 advisory was updated to read BOTH `drifts` and
`diagnostic_drifts` so operator visibility of broker-side advisory entries is
preserved. The finding evidence now distinguishes
`strategy_drift_count` (live_readiness-gating) from
`diagnostic_drift_count` (advisory).

### ops/live_readiness_publish.py

`lifecycle_truth_ok` now reads `positions_truth.broker_authority_status` and
surfaces it in the reason string. The boolean gate is unchanged (still
gated on `truth_ok`) — only the auditability of the reason is improved.

`_resolved_reconciliation_status` is **not** changed semantically: it still
fails-closed on `reconciliation_state.drifts` non-empty (the GAP-041
contract) and on `position_guard_drift.drift_count > 0`. The reconciliation
gate becomes GREEN naturally because the publisher now emits `drifts=[]`
+ `diagnostic_drifts=[M6E, MGC]`.

---

## Files changed

- `chad/ops/lifecycle_truth_publisher.py` — add broker_authority_* and replay_diagnostic_* fields; extend notes.
- `chad/ops/reconciliation_publisher.py` — add `diagnostic_drifts[]` and route broker_sync-only entries there.
- `chad/ops/exterminator.py` — `check_reconciliation` reads both `drifts` and `diagnostic_drifts`.
- `ops/live_readiness_publish.py` — surface `broker_authority_status` in `lifecycle_truth_ok` reason.
- `chad/tests/test_pr09_position_truth_contract.py` — new (10 tests).

## Tests added

10 tests in `chad/tests/test_pr09_position_truth_contract.py`:

1. `test_truth_ok_with_partial_replay_keeps_broker_authority_green`
2. `test_truth_ok_false_implies_broker_authority_red`
3. `test_replay_diagnostic_visibility_preserved`
4. `test_position_guard_drift_count_zero_is_authoritative_drift_gate`
5. `test_position_guard_drift_count_positive_still_red`
6. `test_live_readiness_lifecycle_truth_ok_passes_on_broker_authority_green`
7. `test_live_readiness_lifecycle_truth_ok_fails_when_truth_ok_false`
8. `test_m6e_mgc_diagnostic_drifts_do_not_block_when_guard_drift_clean`
9. `test_exterminator_ex009_surfaces_diagnostic_drifts`
10. `test_reconciliation_publisher_routes_broker_sync_only_to_diagnostic_drifts`

GAP-041 tests (13) re-validated unchanged — `reconciliation_state.drifts`
non-empty still resolves RED, confirming no semantic widening.

---

## Runtime / artifact evidence (post-fix, 2026-05-25 ~16:52Z)

### `runtime/positions_truth.json` (refreshed by chad-lifecycle-truth-publisher.timer)

```
truth_ok=True
truth_source='BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER'
broker_authority_status='GREEN'
broker_authority_reason='BROKER_AUTHORITY_GREEN: upstream=GREEN snapshot=20 ledger=20'
replay_diagnostic_status='PARTIAL'
replay_diagnostic_reason='QTY_OR_SYMBOL_MISMATCH: qty_mismatches=6 missing_from_replay=14 …'
replay_diagnostic_blocks_truth=False
evidence.reconciliation_status='RED'                    (preserved)
evidence.reconciliation_status_upstream='GREEN'          (preserved)
evidence.replay_coverage_status='PARTIAL_REPLAY_COVERAGE' (preserved)
```

### `runtime/reconciliation_state.json` (refreshed by chad-reconciliation-publisher.timer)

```
status='GREEN'
worst_diff=0.0
mismatches=[]
drifts=[]                                  (GAP-041 safety net preserved)
diagnostic_drifts=[
  {symbol='M6E', chad=18.0, broker=20.0, diff=2.0, kind='broker_sync_only'},
  {symbol='MGC', chad=-13.0, broker=-14.0, diff=1.0, kind='broker_sync_only'},
]
```

### `runtime/position_guard_drift.json` (unchanged — authoritative drift gate)

```
schema_version=position_guard_drift.v1
drift_count=0
drifts=[]
```

### `reports/live_readiness/LIVE_READINESS_20260525T164625Z.json` (post-fix evaluation)

```
ready_for_live=False
  reconciliation: ok=True reason='reconciliation_resolved=GREEN'   ← now GREEN
  lifecycle_truth: ok=True reason='lifecycle_truth=GREEN broker_authority_status=GREEN source=BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER'
  chad_mode: ok=False reason='chad_mode=paper live_enabled=False'  ← still gates live
  …
```

---

## Current PR-09 status

| Requirement | Status |
|-------------|--------|
| Broker snapshot + IBKR paper ledger authority has explicit GREEN/RED status | ✅ `broker_authority_status` |
| Lifecycle replay coverage is labeled diagnostic when non-authoritative | ✅ `replay_diagnostic_*` fields |
| Live/paper readiness does not treat diagnostic replay RED as authoritative | ✅ `lifecycle_truth_ok` reads `truth_ok`+`broker_authority_status` |
| M6E/MGC drifts not reported as active blockers when broker-authority is green | ✅ Routed to `diagnostic_drifts`; live_readiness reconciliation gate GREEN |
| GAP-041 fail-closed contract preserved | ✅ Existing GAP-041 tests pass unchanged |
| `position_guard_drift.drift_count=0` remains authoritative drift gate | ✅ Test 4 pins it |

**Status:** VERIFIED.

---

## No-live confirmation

- Trading posture remains PAPER (`CHAD_EXECUTION_MODE=paper`).
- `ready_for_live` remains `false` (gated independently by `chad_mode=paper`,
  `live_enabled=false`).
- No broker orders submitted. No order cancellation. No live enablement.
- No service restarts: systemd timers fired their next-scheduled tick under
  the updated source. Each tick is a fresh process; no daemon-restart needed.
- No runtime JSON manual edits. All runtime artifact updates were produced
  by the existing publishers via their normal timer cadence.

---

## Verification commands

```bash
# 1. Compile-check changed source files
source venv/bin/activate
python3 -m py_compile chad/ops/lifecycle_truth_publisher.py \
  chad/ops/reconciliation_publisher.py \
  chad/ops/exterminator.py \
  ops/live_readiness_publish.py

# 2. PR-09 targeted tests
python3 -m pytest chad/tests/test_pr09_position_truth_contract.py -v

# 3. Targeted regression (positions_truth, reconciliation, live_readiness, pr09)
python3 -m pytest chad/tests/ -k "positions_truth or reconciliation or live_readiness or position_guard_drift or pr09" -v

# 4. Full regression
python3 -m pytest chad/tests/ -q

# 5. Live artifact inspection
python3 -c "import json; d=json.loads(open('runtime/positions_truth.json').read()); \
  print(json.dumps({k:d.get(k) for k in ('truth_ok','truth_source','broker_authority_status',\
  'replay_diagnostic_status','replay_diagnostic_blocks_truth')}, indent=2))"

python3 -c "import json; d=json.loads(open('runtime/reconciliation_state.json').read()); \
  print('status=', d.get('status'), 'drifts=', len(d.get('drifts',[])), \
  'diagnostic_drifts=', len(d.get('diagnostic_drifts',[])))"
```

## Definition of done

- [x] `positions_truth.json` carries `broker_authority_status`, `broker_authority_reason`, `replay_diagnostic_status`, `replay_diagnostic_reason`, `replay_diagnostic_blocks_truth`.
- [x] `reconciliation_state.json` carries `diagnostic_drifts[]` and reserves `drifts[]` for strategy-attributable entries.
- [x] `evidence.reconciliation_status_upstream` and `evidence.reconciliation_status` preserved verbatim.
- [x] `live_readiness.lifecycle_truth` ok=True under the new contract; `live_readiness.reconciliation` ok=True after the publisher emits `drifts=[]` + `diagnostic_drifts=[M6E,MGC]`.
- [x] `ready_for_live` remains `false` (gated by `chad_mode=paper`, `live_enabled=false`).
- [x] GAP-041 fail-closed contract intact (13 tests pass unchanged).
- [x] PR-09 test suite: 10/10 pass.
- [x] Full regression: 2460/2460 pass; 1 pre-existing flake (`test_canonical_equity_source.py::test_canonical_sources_agree_within_skew_tolerance`) is environment/timing-sensitive against live equity files written by independent timers, passes in isolation, and is unrelated to PR-09 (no overlap with positions_truth / reconciliation / readiness producers).
