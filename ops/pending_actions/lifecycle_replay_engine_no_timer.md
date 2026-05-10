# Lifecycle Replay Engine — No Timer (Secondary Defect)

Status: PENDING
Reviewed: 2026-05-10

## Finding
lifecycle_replay_engine.py and lifecycle_replay_coverage.py have no systemd
unit driving them. Last write: 2026-03-23 (48 days stale).

## Impact
reconciliation shows SCOPE_MISMATCH_MANUAL_VS_PAPER_EXEC with stale symbol
set. This is diagnostic-only — does NOT block truth_ok (confirmed by audit
2026-05-10). replay=3 vs snapshot=14 is obsolete data.

## Fix Path
Wire chad-lifecycle-replay-engine.service/.timer (oneshot, every 30 min)
calling: python3 -m chad.ops.lifecycle_replay_engine --once
After deploy: replay artifacts will reflect current 14-position scope.

## Blocked By
Nothing. Independent work item.
