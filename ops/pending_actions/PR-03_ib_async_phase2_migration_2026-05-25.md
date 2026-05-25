# PR-03 — ib_async Phase 2 Migration

**Date:** 2026-05-25
**Author:** Team CHAD (Claude Code)
**Status:** VERIFIED — last two production ib_insync importers migrated to
ib_async with import-only changes; full parity suite + new PR-03 tests +
full regression all green.
**Mode:** PAPER. ready_for_live=false. allow_ibkr_live=false. allow_ibkr_paper=true.

## Plain-English summary
GAP-A019 / GAP-007 staged the ib_async migration across two phases. Phase 1
migrated 18 read-only and execution-path files (including the hot-path
`chad/core/live_loop.py`, `chad/execution/ibkr_adapter.py`, and
`chad/execution/ibkr_trade_router.py`). Two production files remained on
`ib_insync` in the `PHASE2_DEFERRED_FILES` ledger: `paper_position_closer`
(a CLI/oneshot exit-only helper) and `paper_shadow_runner` (gated paper
order placer reached via the wrapper-v2 subprocess). PR-03 migrates both
with import-only changes, empties the deferred-files allowlist, and adds
19 new tests pinning the contract.

## Root cause
Phase 1 already proved that ib_async is a drop-in replacement: it preserves
the full ib_insync public API surface (`IB`, `Stock`, `MarketOrder`,
`LimitOrder`, `Future`, `Forex`, `Contract`, `Order`, `util.patchAsyncio`,
`ib_async.ib` submodule). The two Phase 2 deferrals were precautionary,
not technical — they sit on the order-placing execution path so they were
held back until the rest of the migration baked. With Phase 1 stable for
many weeks (and `chad-ibkr-broker-events`/`chad-ibkr-paper-fill-harvester`
journal entries continuing to show clean `ib_async.wrapper` log lines
post-Phase 1), the two remaining swaps are now safe.

## Files audited
- `chad/core/paper_position_closer.py` — module-level + lazy ib_insync
  import (lines 39/41/49 pre-PR-03)
- `chad/core/paper_shadow_runner.py` — function-local ib_insync imports
  (lines 595/774 pre-PR-03)
- All Phase-1-migrated files (re-verified clean by the existing parity
  invariants)
- `chad/tests/test_ib_async_import_parity.py` — ledger source of truth

## Files changed
- `chad/core/paper_position_closer.py` — replaced `from ib_insync import …`
  with `from ib_async import …` at module-level + lazy helper.
- `chad/core/paper_shadow_runner.py` — replaced both function-local
  `from ib_insync import …` lines (IBKRClient.connect; place_mkt_order)
  with `from ib_async import …`. Refreshed the two doc-comment
  references that said "must NOT import ib_insync" to read "must NOT
  import the broker library (ib_async)" so the import-safety contract
  stays accurate post-migration.
- `chad/tests/test_ib_async_import_parity.py` — moved both files from
  `PHASE2_DEFERRED_FILES` into `PHASE1_MIGRATED_FILES`; left
  `PHASE2_DEFERRED_FILES` as an empty tuple so the existing
  `test_remaining_ib_insync_imports_are_explicitly_allowlisted`
  invariant immediately rejects any future production importer.
- `chad/tests/test_paper_shadow_runner.py::test_module_does_not_import_ib_insync_on_safe_paths`
  — strengthened to additionally reject `ib_async` in the delta so the
  gating-layer contract reflects the post-migration broker library.
- `chad/tests/test_pr03_ib_async_phase2_migration.py` (new) — 19 tests:
  empty Phase 2 ledger; migrated targets in Phase 1; both files no
  longer import ib_insync; both files now import ib_async;
  zero-production-importer sweep; subprocess-isolated import-safety
  smoke tests for both migrated files; service entrypoint smoke imports
  for broker-events, paper-fill-harvester, paper-ledger-watcher,
  reconciliation-publisher; PR-09 contract preserved (positions_truth
  fields); PR-02 delta strategy surface present; PR-04 failure-artifact
  constants preserved; live posture sentinel.

## Imports remaining
Zero production ib_insync imports remain. `requirements.txt` continues to
pin `ib-insync` for backward compatibility (kept by the
`test_phase0_requirements_pin_ib_async` invariant) but no production
source loads it. Tests legitimately import `ib_insync` to assert
coexistence and to construct fake-broker fixtures — that is by design.

## Tests added / updated
- 19 new tests in `chad/tests/test_pr03_ib_async_phase2_migration.py`
- 1 strengthened test in `chad/tests/test_paper_shadow_runner.py`
- Existing 8 parity tests in `chad/tests/test_ib_async_import_parity.py`
  continue to enforce the ledger
- Targeted PR-03 + parity + paper_shadow_runner: 32 passed
- Protection regression (pr09/pr02/pr02b/pr04/placeholder/positions_truth/
  options_chain): 103 passed
- Full regression: 2529 passed (baseline pre-PR-03 was 2510, +19 new)

## Service restart needed
No. Both migrated files have NO active systemd unit:
- `chad-paper-position-closer.service` — does not exist at the
  systemd level.
- `chad-paper-shadow-runner.service` — MASKED
  (`/etc/systemd/system/_masked_archive_20260506/`).
- `chad-paper-shadow-exec.service` — MASKED.

The active wrapper `chad-paper-shadow-exec` would normally invoke
`python -m chad.core.paper_shadow_runner` as a subprocess, but it's
also masked. The next time the wrapper is unmasked and fired, the
fresh subprocess will pick up the ib_async migration automatically.

The neighboring already-Phase-1-migrated services were confirmed
healthy via `systemctl status`: `chad-ibkr-broker-events.service`
last exit 0 at 23:16:38 UTC, `chad-ibkr-paper-fill-harvester.service`
last exit 0 at 23:16:38 UTC — both emit `ib_async.wrapper` log lines
in the journal, proving the ib_async stack is stable end-to-end.

## Runtime proof available or deferred
- **Available now:** subprocess import smoke tests verify both migrated
  modules import cleanly without raising. The neighboring ib_async
  services emit successful `ib_async.wrapper` log lines.
- **Deferred:** an end-to-end paper-order placement through
  `paper_shadow_runner` is gated behind the operator's
  `I_UNDERSTAND_PAPER_CAN_PLACE_ORDERS` arm phrase and the masked
  `chad-paper-shadow-exec.service`. Unmasking and arming are operator
  actions outside this PR's scope. Until they happen, the migration is
  proven by code inspection + parity tests + subprocess imports — which
  is the strongest evidence the no-live posture allows.

## No-live confirmation
- `runtime/live_readiness.json::ready_for_live == false`
- `runtime/decision_trace_heartbeat.json::allow_ibkr_live == false`
- `runtime/decision_trace_heartbeat.json::allow_ibkr_paper == true`
- `runtime/decision_trace_heartbeat.json::mode == {"chad_mode": "paper",
  "live_enabled": false}`
- Stop bus inactive.
- No systemd unit modified. No config mutated. No broker orders sent.
- Prior PR contracts preserved (PR-09 broker authority truth; PR-02 delta
  abstain; PR-02b reconciler placeholder; PR-04 options chain failure
  artifact) — verified by their existing test suites continuing to pass.

## Current final status
**VERIFIED** — code, tests, and import smoke checks prove the migration
is complete and clean. The deferred ledger is empty, so any future
re-introduction of `ib_insync` to production source is caught
immediately by the parity invariants. Promotion to LOCKED requires
observing a paper-shadow-runner run through the unmasked wrapper —
that is a future operator action, not a PR-03 deliverable.
