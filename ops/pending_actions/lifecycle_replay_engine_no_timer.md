# Lifecycle Replay Engine — No Timer (Secondary Defect)

Status: CLOSED
Reviewed: 2026-05-10
Closed: 2026-05-12

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

## Resolution (2026-05-12)
Installed `/etc/systemd/system/chad-lifecycle-replay-engine.service` (oneshot)
and `chad-lifecycle-replay-engine.timer` (OnBootSec=5min, OnUnitActiveSec=30min,
Persistent=true). Service runs the engine module then coverage module via
ExecStartPost, each redirecting stdout to `runtime/lifecycle_replay_state.json`
and `runtime/lifecycle_replay_coverage.json` via atomic `.tmp` + `mv` (engine
module had no built-in disk write — it only printed JSON to stdout, so the
unit captures stdout). Both runtime artifacts refreshed from 2026-03-23 to
2026-05-12 01:46:41 UTC; service exited 0; timer scheduled (next run +30min);
journalctl clean. Note: the `--once` flag in the original fix-path note is a
no-op because the engine has no argparse — the unit invokes the module without
flags.
