# GAP-053 — Lifecycle Replay Coverage Policy

**Status:** Active policy (code-level enforcement)
**Effective:** 2026-05-19
**Owner:** CHAD operator
**Authority:** Closes Stage 2 Completion Matrix Box 19

---

## Summary

`runtime/trade_lifecycle_state.json::backlog_flag` is the canonical "lifecycle
pipeline is recording everything it should be" signal. Pre-GAP-053 it tripped
true whenever `data/fills/*.ndjson` or `data/fees/*.ndjson` had not been
modified within `CHAD_BROKER_EVENTS_MAX_AGE_SECONDS` (default 900s). That
conflates two distinct conditions:

1. **Real backlog** — fills happened at the broker but the writer fell behind.
2. **Quiet window** — no fills happened at the broker, so there is nothing to
   write.

CHAD's paper loop spends extended periods in `cooldown_active` /
`same_side_position_open` suppression where no paper executions occur. During
those windows the ledger mtimes naturally age past 15 minutes even though the
lifecycle pipeline is perfectly healthy (broker_events heartbeats are
continuing, no fills are being missed because no fills are being attempted).

## Policy

Lifecycle backlog truth is computed in
`chad/ops/lifecycle_truth_publisher.py::build_trade_lifecycle_state` and is now
gated by the following rules. Operators may verify any of them in code or via
the runtime `trade_lifecycle_state.json` audit fields.

| Condition | `backlog_flag` |
|-----------|----------------|
| Any of broker_events / fills / fees ledger directory missing or empty (`gap_flag=true`) | **true** |
| broker_events newest file mtime older than `max_age_seconds` (no heartbeats) | **true** |
| broker_events fresh AND ≥1 non-heartbeat (fill/fee/other) event observed in the trailing window AND fills or fees ledger stale | **true** |
| broker_events fresh AND zero non-heartbeat events in the trailing window AND fills or fees ledger stale | **false** (`quiet_window_accepted=true`) |
| All three fresh | **false** |

`quiet_window_accepted` is surfaced as a top-level boolean in the payload so
operators can distinguish a clean window from an accepted quiet window in a
single audit query (`jq .quiet_window_accepted`).

## What this policy does NOT do

* It does **not** authorise live trading. Live activation is gated by the
  full `LiveGate` AND `live_readiness` AND-of-many-gates. `chad_mode=paper` and
  `position_guard_drift` (and any other RED gate) still block live independently.
* It does **not** mark lifecycle replay coverage as FULL. The replay engine
  remains evidence-gated bootstrap (see `runtime/lifecycle_replay_coverage.json`
  for current `status` — typically `PARTIAL_REPLAY_COVERAGE`). The replay
  feed is diagnostic-only per the broker-authority truth path documented in
  `chad/ops/lifecycle_truth_publisher.py::build_positions_truth.notes`.
* It does **not** weaken `gap_flag`. A missing fills/fees/broker_events
  directory or empty newest file still fails closed with `gap_flag=true` →
  `backlog_flag=true`.
* It does **not** silence the writer-outage path. If real broker fill activity
  occurred within `max_age_seconds` but the writer fell behind, backlog_flag
  remains true. The mechanism for detection is parsing the trailing 400 lines
  of the newest `broker_events_*.ndjson` for `event_type != "heartbeat"`
  entries with `ts_utc` within the freshness window.

## What still blocks live trading

After GAP-053 is applied:

1. `chad_mode == "live"` AND `live_enabled == true` — currently paper-only.
2. `live_readiness.ready_for_live == true` — currently false.
3. `reconciliation_state.json::status == "GREEN"` (publisher rollup) AND
   `position_guard_drift.json::drift_count == 0` (resolved, GAP-041).
4. `lifecycle_truth_ok()` — still requires `truth_ok=true` AND `gap_flag=false`
   AND `backlog_flag=false`. The post-fix `backlog_flag` continues to fail
   closed on any of the four blocking conditions above.
5. `SCR` not paused, `paper_only=false`.
6. `operator_intent.operator_mode == "ALLOW_LIVE"`.
7. STOP bus clear, profit lock not in HARD_STOP, etc.

GAP-053 only changes the **interpretation** of fills/fees mtime aging during a
verified quiet window. Every other safety gate is untouched.

## How to audit at runtime

```bash
# Quick view
jq '{
  backlog_flag, gap_flag, quiet_window_accepted,
  events_fresh: .evidence.broker_events.events_fresh,
  recent_non_heartbeat_count: .evidence.broker_events.recent_non_heartbeat_count,
  fills_fresh: .evidence.fills.fresh,
  fees_fresh: .evidence.fees.fresh,
  had_recent_broker_fill_activity: .evidence.had_recent_broker_fill_activity
}' runtime/trade_lifecycle_state.json
```

Expected during a healthy quiet window:
```
backlog_flag = false
gap_flag = false
quiet_window_accepted = true
events_fresh = true
recent_non_heartbeat_count = 0
fills_fresh = false
fees_fresh = false
had_recent_broker_fill_activity = false
```

Expected when the writer falls behind real activity (this is the failure
mode the new policy must catch):
```
backlog_flag = true
gap_flag = false
quiet_window_accepted = false
events_fresh = true
recent_non_heartbeat_count = N>0
fills_fresh = false   # or fees_fresh = false
had_recent_broker_fill_activity = true
```

## Tests pinning the policy

See `chad/tests/test_gap053_lifecycle_backlog_quiet_window.py`. Every branch
in the truth table above is pinned, plus a regression matrix that asserts
`backlog_flag` cannot silently flip to false when any failure mode is present.

## Future work (out of GAP-053 scope)

* Implement the full deterministic replay engine
  (`chad/ops/lifecycle_replay_engine.py`) end-to-end so
  `lifecycle_replay_coverage.json::status == "REPLAY_MATCH_CONFIRMED"` and
  positions_truth can drop the broker-authority shortcut.
* Once full replay is online, retire the quiet-window relaxation: the policy
  here is a bootstrap-era tolerance, not a long-term design.

## Operator sign-off

This policy was applied as the resolution for **NEW-GAP-053** in the
Completion Matrix (Box 019). Code change is contained to a single file
(`chad/ops/lifecycle_truth_publisher.py`); behaviour change is gated by
deterministic tests; no live trading authorisation, no service restart, no
runtime JSON mutation.
