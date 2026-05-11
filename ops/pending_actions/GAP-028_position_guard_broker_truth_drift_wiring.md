# GAP-028 — Position-Guard ↔ Broker-Truth Drift Wiring

Status: PENDING
Severity: P1 (paper-soak signal yield drag; not a live-trading blocker)
Opened: 2026-05-11
Affected components: chad/core/live_loop.py, chad/core/position_guard.py,
  runtime/position_guard.json, runtime/trade_closer_state.json,
  runtime/reconciliation_state.json
Related: GAP-001 (closed NOISY_BUT_CORRECT), GAP-027 (mes_paper_ledger_stale_position),
  commit aaa438c (drift detector test landed without production wiring),
  audit_2026_05_07.md (root cause superseded by this finding)

## 1. Finding

Concrete statement: the per-strategy guard book diverges from broker truth on at least two pairs;
the existing detector and surgical close function are dead code; the paper rebuilder is unaware
of operator exclusions and never cross-checks against broker truth.

Verbatim snapshots (as of 2026-05-11T00:30:52Z):

    delta|AAPL       : qty=31  BUY  open=True   source=paper_ledger_rebuild
    omega_macro|M6E  : qty=146 BUY  open=True   source=paper_ledger_rebuild
    broker_sync|AAPL : qty=2   SELL source=broker_truth_rebuild
    broker_sync|M6E  : qty=26  BUY  source=broker_truth_rebuild
    positions_truth.AAPL : -2 SHORT  (STK)
    positions_truth.M6E  : +26 LONG  (FUT)
    reconciliation_state.exclusion_policy.AAPL : present, owner=operator, added 2026-04-01
    reconciliation_state.exclusion_policy.M6E  : absent

Observable consequence in the live-loop journal: `SKIP suppression=same_side_position_open` fires
at approximately 142 lines/hour, concentrated entirely on the two divergent pairs
(omega_macro:M6E:BUY = 78 lines/hour; delta:AAPL:BUY = 64 lines/hour; zero SELL/EXIT suppressions).

## 2. Mechanical Model (Reproduced)

- M6E: cumulative 172 BUY fills since 2026-05-09 (day breakdown: 20260509 buys=64 qty=128.0,
  20260510 buys=22 qty=44.0). Zero SELL fills. 13 closed-trade events in 30 days (total_pnl=$0.00)
  trimmed 26 contracts off the FIFO queue, leaving 146 open across 73 lots. The paper rebuilder
  at chad/core/live_loop.py:383 mirrors this state faithfully every cycle.
- AAPL: 31 contracts across 3 BUY lots, last opened 2026-05-09T23:47:29Z. Only 1 closed-trade
  record in 30 days (pnl +$1.44). AAPL is on the operator exclusion list since 2026-04-01, but
  the rebuilder does not consult exclusion_policy.

Reproduction matches observed quantities exactly. The rebuilder is correct; the divergence is
upstream (trade_closer FIFO grew monotonically because no SELL fills landed to match against
open BUY lots).

## 3. Two Structural Gaps Identified

### Gap A — REBUILDER_NO_EXCLUSION_AWARENESS

chad/core/live_loop.py:383 `_rebuild_guard_from_paper_ledger` does not consult
`reconciliation_state.exclusion_policy`. Symbols on the exclusion list can accumulate per-strategy
guard records that no other component will close. The exclusion list is currently consumed only by
chad/ops/reconciliation_publisher.py (10 grep hits) and the operator. Evidence:

    $ grep -n "exclusion" chad/core/live_loop.py
    (empty)
    $ grep -RIn "exclusion_policy" chad/risk chad/core chad/portfolio chad/execution
    (empty — zero hits outside chad/ops/reconciliation_publisher.py)

### Gap B — REBUILDER_NO_CROSS_CHECK_AGAINST_BROKER_TRUTH

The drift detector at chad/core/position_guard.py:419 `detect_guard_vs_broker_truth_drift` is
unit-tested but has zero production callers (the only references outside the function definition
are in chad/tests/test_position_guard_same_side_hardening.py:162 and
chad/tests/test_position_guard.py:285,324). The surgical close function at
chad/core/position_guard.py:491 `close_stale_position_from_broker_truth` has no CLI wrapper and
no production caller. Drift findings are not surfaced to the operator anywhere — no systemd unit,
no timer, no journal log entry in the last 24 hours.

Commit aaa438c (2026-05-09 11:31Z) added the detector function and tests but no production wiring;
its commit subject begins "Test:" rather than "Feature:" or "Fix:".

## 4. Operational Invariant (Important)

Closing a per-strategy guard entry alone is insufficient — the next live-loop cycle's
`_rebuild_guard_from_paper_ledger` will re-open it from the trade_closer FIFO queue (logic at
chad/core/live_loop.py:425-466). Any remediation tool MUST close the guard entry AND clear the
corresponding `trade_closer_state.queues[strategy|symbol]` entry in the same operation, atomically.

This invariant applies to both the proposed CLI wrapper (Subtask 5.2) and any future automated
remediation path. Violating it produces a no-op: the entry will reappear within ~60 seconds.

## 5. Proposed Wiring (Three Subtasks)

### Subtask 5.1 — Wire the drift detector into the reconciliation publisher

Add a call to `detect_guard_vs_broker_truth_drift()` inside chad-reconciliation-publisher (which
already runs on a timer and already consumes the broker_sync entries). Emit findings to a new
advisory file `runtime/position_guard_drift.json` with `schema_version=position_guard_drift.v1`
and a TTL aligned with the publisher cadence (currently active per
chad-reconciliation-publisher.timer). No automatic remediation. INFO-level entries only until a
Telegram-alert rule is separately approved.

Schema (proposed):

    {
      "schema_version": "position_guard_drift.v1",
      "ts_utc": "<iso>",
      "ttl_seconds": <int>,
      "drift_count": <int>,
      "drifts": [
        {
          "key": "<strategy>|<symbol>",
          "strategy": "<name>",
          "symbol": "<sym>",
          "guard_side": "BUY|SELL",
          "broker_side": "BUY|SELL|null",
          "broker_present": <bool>,
          "drift_kind": "broker_truth_missing|side_mismatch"
        }
      ]
    }

Read-only — no writes to position_guard.json or trade_closer_state.json from this subtask.

### Subtask 5.2 — CLI wrapper for surgical close

New file: scripts/close_guard_entry.py

Required flags: `--strategy`, `--symbol`, `--reason`, `--by`, `--confirm` (no auto-yes default).

Required behaviour:

- Calls `chad.core.position_guard.close_stale_position_from_broker_truth()` (existing function at
  chad/core/position_guard.py:491-522).
- Atomically clears `trade_closer_state.queues[strategy|symbol]` in the same op so the next
  live-loop cycle's `_rebuild_guard_from_paper_ledger` cannot re-open the entry from a stale FIFO
  (see invariant in section 4).
- Writes an audit record to data/operator_actions/ (ndjson) with the previous_entry JSON for
  rollback.
- Refuses to run if SCR is unsafe (state != CONFIDENT/CAUTIOUS), exec_mode != paper, or LiveGate
  is ALLOW_LIVE — fail-closed.
- Refuses to run on broker_sync|<sym> entries (those are owned by `_rebuild_guard_from_broker`
  at chad/core/live_loop.py:524 and reflect IBKR truth).
- Idempotent: re-running on an already-closed entry is a no-op with WARNING log, exit 0.

### Subtask 5.3 — Optional rebuilder exclusion-awareness

Conditional on the OPERATOR DECISION block (section 7). If "strict" is chosen, plumb
`reconciliation_state.exclusion_policy` into `_rebuild_guard_from_paper_ledger` (or upstream into
signal routing) so excluded symbols cannot accumulate per-strategy entries. If "permissive" is
chosen, document the current behaviour in CLAUDE.md and do not change the rebuilder.

## 6. Regression Tests Required

- test_detect_guard_vs_broker_truth_drift_fires_on_delta_AAPL_synthetic_state
- test_close_guard_entry_clears_trade_closer_fifo_in_same_op
- test_close_guard_entry_refuses_on_broker_sync_key
- test_close_guard_entry_refuses_when_exec_mode_not_paper
- test_close_guard_entry_refuses_when_livegate_allow_live
- test_close_guard_entry_idempotent_on_already_closed
- test_close_guard_entry_writes_operator_action_audit_record
- test_rebuilder_consults_exclusion_policy_when_strict_mode (conditional on section 7)
- test_reconciliation_publisher_emits_position_guard_drift_json

## 7. OPERATOR DECISION REQUIRED

Two policy options. The Pending Action does NOT pick one. Operator must mark a choice before
Subtask 5.3 is implemented.

Option A — STRICT
Strategies cannot open per-strategy guard entries for symbols on exclusion_policy. The exclusion
list is plumbed into signal routing before any rebuilder runs. Pro: structural prevention. Con:
changes the semantics of the exclusion list from "advisory to reconciler" to "enforced upstream"
and may surprise strategies that expect to attribute on excluded symbols.

Option B — PERMISSIVE (current behaviour)
Strategies continue to attribute against excluded symbols; the drift detector + CLI close
function provide the maintenance surface; CLAUDE.md is updated to document this explicitly.
Pro: minimal blast radius. Con: requires ongoing operator maintenance on the two affected pairs
and any future pairs.

Operator choice: ____  (A | B)
Operator GO date: ____

## 8. Immediate Hygiene (Operator-Only, After Subtask 5.2 Lands)

After the CLI wrapper exists, the two stale entries can be closed by the operator:

    scripts/close_guard_entry.py --strategy delta       --symbol AAPL \
      --reason "broker_truth_short_2_exclusion_AAPL"      --by operator --confirm
    scripts/close_guard_entry.py --strategy omega_macro --symbol M6E  \
      --reason "broker_truth_long_26_vs_guard_long_146"   --by operator --confirm

Expected post-close state:

- same_side_position_open journal rate drops from ~142/hour to near zero.
- omega_macro can re-evaluate M6E entries on next cycle (will likely re-block by other gates
  until trade_closer FIFO state stabilises).
- delta will not re-open AAPL if Option A is chosen; will be free to re-attribute if Option B.

Each invocation captures the previous_entry JSON for rollback into the operator audit ndjson
under data/operator_actions/.

## 9. Validation Criteria

- runtime/position_guard_drift.json emitted with TTL aligned to chad-reconciliation-publisher
  cadence; at minimum one cycle produces a non-empty `drifts` array containing the two known
  pairs (delta|AAPL, omega_macro|M6E) until they are closed.
- scripts/close_guard_entry.py exists, is executable, and refuses to run when SCR is unsafe,
  exec_mode != paper, LiveGate is ALLOW_LIVE, or target key starts with broker_sync|.
- After invoking the CLI on the two stale pairs in a controlled paper-mode window:
  - position_guard.json shows delta|AAPL.open=False and omega_macro|M6E.open=False with
    cleared_at populated and source/closed_by=stale_guard_entry reason.
  - trade_closer_state.queues no longer contains entries for delta|AAPL or omega_macro|M6E.
  - The next live-loop cycle does NOT re-open either entry (proves the atomic close + FIFO-clear
    invariant from section 4 is honoured).
- 60-minute journal sample after the close shows zero `same_side_position_open` lines for the
  closed pairs.
- All regression tests in section 6 pass under `python3 -m pytest chad/tests/ -x -q`.
- LiveGate verdict remains unchanged across the whole exercise (exec_mode=paper, live_enabled=
  false, allow_ibkr_live=false, allow_ibkr_paper=true, LIVE_READINESS_FALSE).
- No mutation of broker positions; no change to risk weights; no service restart.

## 10. Rollback

Each `close_guard_entry.py` invocation captures the previous_entry JSON into
data/operator_actions/<date>.ndjson. To revert a close, the previous_entry can be re-applied via a
follow-up tool (not in this Pending Action's scope) or by directly invoking
`chad.core.position_guard.save_state()` against a reconstructed state dict — both paths require
operator-level confirmation and a separate Pending Action.

If Subtask 5.1 produces undesired alert noise, disabling the new emit branch in
chad-reconciliation-publisher is a one-line change that does not affect the existing
reconciliation_state.json output.

## 11. Non-Goals

This Pending Action explicitly does NOT:

- Modify the rebuilder's source-of-truth (trade_closer_state.json remains canonical for
  per-strategy attribution).
- Change LiveGate behaviour or any live-mode gate.
- Alter risk weights, dynamic caps, edge-decay halts, or SCR thresholds.
- Enable or restart any service.
- Address the root cause of why omega_macro M6E BUY fills never produce matching SELL fills
  (that is a separate strategy-logic question deferred to the operator).
- Address GAP-027 (MES stale paper ledger entry) — that remains a separate Pending Action.
