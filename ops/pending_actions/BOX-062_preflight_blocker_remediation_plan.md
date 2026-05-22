# BOX-062 — Preflight blocker remediation plan

**Timestamp (UTC):** 2026-05-21T03:00:38Z
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 062 (preflight).
**Status:** Box 062 is **BLOCKED**. This document sequences the work
that must be completed before Box 062 can be attempted. It performs
no fixes.

## 1. Scope statement

- **Box 062 is BLOCKED.** It cannot be attempted while the runtime
  blockers in §3 are active.
- **This document is NOT live trading authorization.** It does not
  set `ready_for_live=true`. It does not place orders. It does not
  modify any runtime / SQLite / ledger / position / guard / fill /
  broker-event / strategy state.
- **`ready_for_live` MUST remain `false`** until every gate in §7
  passes simultaneously and operator GO is explicitly recorded.

Box-061 (Final P0 / P1 sign-off) closed with
`PASS_FINAL_P0_P1_SIGNOFF_EVIDENCE_ONLY` — evidence integrity is
clean, no false closures remain, but 5 P1-class runtime blockers
still prevent Box 062 from proceeding. This plan sequences their
remediation.

## 2. Box 062 acceptance criteria (recap, do not satisfy here)

Per the Evidence-Locked Completion Matrix v0.1:

1. `runtime/live_readiness.json` `ready_for_live=true` from the
   canonical live-readiness publisher.
2. Operator GO is explicitly recorded under `ops/pending_actions/`.
3. First-live canary plan is approved (sizing, scope, kill-switch
   criteria, rollback path).

None of these is satisfied at the time of writing. **Do not
satisfy them via this plan.** The plan only sequences the
prerequisites.

## 3. Current runtime blocker table (as of 2026-05-21T03:00Z)

| # | Blocker | Severity | Source file / service | Maps to | Live-readiness impact | Required channel |
|---|---|---|---|---|---|---|
| 1 | `stop_bus.active=true` (`broker_latency:avg_latency_ms ≈ 8200–8270 ms > 2000 ms`) since 2026-05-20T04:20:11Z | **P1** | `runtime/stop_bus.json`; live_loop journal shows STOP_BUS_TRIGGERED every cycle | **G-26** (IB Gateway latency) | live-loop is skipping every cycle ("STOP_BUS_ACTIVE — skipping cycle") | **Manual external** (IB Gateway investigation) |
| 2 | `reconciliation_state.status=RED`, `broker_source="unavailable:"` | **P1** | `runtime/reconciliation_state.json` | **G-26** (downstream of #1) | reconciliation gate fails; readiness publisher will not flip | **Manual external** + auto-resolves when IB Gateway responsive |
| 3 | `position_guard_drift.drift_count=2` — `alpha\|MSFT` + `alpha_intraday\|MSFT` `side_mismatch` (guard SELL vs broker BUY) | **P1** | `runtime/position_guard_drift.json` | **G-01** (MSFT guard drift; GAP-028 PERMISSIVE policy) | guard-drift gate fails | **Channel 1** (operator) — `scripts/close_guard_entry.py` |
| 4 | `trade_lifecycle_state.backlog_flag=true`; `events_fresh=false`; `fills.fresh=false`; `fees.fresh=false`; last event 2026-05-20T12:45:34Z | **P1** | `runtime/trade_lifecycle_state.json` | **G-26** (downstream of failed broker-event / fill-harvester services) | lifecycle truth gate fails | **Channel 1** (operator) — recover failed services |
| 5 | `feed_state.ibkr_stocks` stale (last update 2026-05-20T12:30:02Z, ~14 h) | **P1** | `runtime/feed_state.json` | **G-26** (downstream of IB Gateway) | feed gate fails | **Manual external** + auto-resolves when IB Gateway responsive |

### 3a. Failed IBKR services (proximate causes of blockers #4, partially #2)

Read-only `systemctl show` discovered two failed units that the
Box-061 evidence did not surface explicitly:

| Unit | ActiveState | SubState | Failure surface |
|---|---|---|---|
| `chad-ibkr-broker-events.service` | **failed** | **failed** | Likely contributes to `events_fresh=false` and `BROKER_EVENTS_IBKR_20260521.ndjson` missing |
| `chad-ibkr-paper-fill-harvester.service` | **failed** | **failed** | Likely contributes to `fills.fresh=false` |

These are oneshot services; their **timers** may still be active
(per Box-060 list-timers output), but the last service-run exited
with a failure code. They must be investigated read-only before
the operator decides whether to restart them.

### 3b. Healthy supporting services (for context, no action needed)

| Unit | ActiveState |
|---|---|
| `chad-live-loop.service` | active (running) — skipping cycles due to stop_bus |
| `chad-backend.service` | active (running) |
| `chad-ibkr-bar-provider.service` | active (running) — but logs show `reqHistoricalData: Timeout` for SI / MYM contracts |
| `chad-ibkr-watchdog.service` / `chad-ibkr-health.service` / `chad-scr-sync.service` / `chad-reconciliation-publisher.service` / `chad-governor.service` / `chad-feed-watchdog.service` / `chad-health-monitor.service` | `inactive / dead` — expected for `Type=oneshot` triggered by their respective active timers (Box-060 evidence confirms timers are scheduled) |

## 4. Required action sequence (must be executed in order)

### Step 1 — Resolve **G-26** (IB Gateway latency) [Manual external]

This is the upstream root cause for blockers #1, #2, #4, #5 and the
two failed services.

1.1 Operator investigates IB Gateway health (Pre-Live Operator Task
    #3 in `CLAUDE.md`): client-id collision, network path latency,
    Gateway process responsiveness, paper account session validity.

1.2 Once IB Gateway latency returns below the 2 s threshold:
    - `stop_bus.active` will auto-clear via the durable hysteresis
      path (GAP-034 / Phase-44) after the configured clean-streak.
    - `reconciliation_state.status` will flip GREEN at the next
      `chad-reconciliation-publisher.timer` cycle (~5 min).
    - `feed_state.ibkr_stocks` will refresh at the next bar-provider
      cycle.

1.3 Operator inspects the two failed services (read-only):
    - `journalctl -u chad-ibkr-broker-events.service --since "1 day ago"`
    - `journalctl -u chad-ibkr-paper-fill-harvester.service --since "1 day ago"`
   Identify failure mode (timeout? auth? exit code?). If failure is
   transient and IB Gateway is now responsive, operator may
   `systemctl restart` each — **operator action, NOT automated**.

1.4 Expected proof at completion of Step 1:
    - `runtime/stop_bus.json` `active=false`, `cleared_by` reflects
      automatic clear.
    - `runtime/reconciliation_state.json` `status="GREEN"`,
      `broker_source` non-empty.
    - `runtime/feed_state.json` ts_utc within last 5 minutes.
    - `runtime/trade_lifecycle_state.json` `backlog_flag=false`,
      `events_fresh=true`.
    - `chad-ibkr-broker-events.service` last run `result=success`.
    - `chad-ibkr-paper-fill-harvester.service` last run `result=success`.

### Step 2 — Resolve **G-01** (MSFT position-guard drift) [Channel 1]

2.1 Operator confirms the two drift entries persist after Step 1
    completes (they likely will — drift is independent of latency).

2.2 Operator runs `scripts/close_guard_entry.py` for both entries
    per the GAP-028 PERMISSIVE policy
    (`ops/pending_actions/GAP-028_position_guard_broker_truth_drift_wiring.md`):

    ```
    python3 scripts/close_guard_entry.py --key 'alpha|MSFT'
    python3 scripts/close_guard_entry.py --key 'alpha_intraday|MSFT'
    ```

    The CLI is gated to refuse when SCR ∉ {CONFIDENT, CAUTIOUS},
    when exec_mode is not paper/dry_run, when LiveGate operator
    intent is ALLOW_LIVE [sic — guard CLI documented behavior], or
    on `broker_sync|*` keys. The operator must confirm gate state
    before invocation.

2.3 Expected proof: `runtime/position_guard_drift.json`
    `drift_count=0`, `drifts=[]` at the next reconciliation cycle.

### Step 3 — Verify the 8 Box-059 soak gates [Channel 2, audit only]

3.1 Read `runtime/live_readiness.json` — expect every check `ok=true`.

3.2 Confirm in order: `stop_bus.active=false`,
    `reconciliation_state.status=GREEN`, `position_guard_drift.drift_count=0`,
    `trade_lifecycle_state.backlog_flag=false`,
    `feed_state.ibkr_stocks` fresh, `scr_state.state=CONFIDENT &&
    paper_only=false`, `operator_intent.operator_mode=ALLOW_LIVE`,
    service uptime ≥ 1 cycle since Step 1 clearance.

3.3 Expected proof: one live-loop cycle in which all 8 gates are
    simultaneously GREEN. **Soak clock starts at this cycle.**

### Step 4 — Begin 5-trading-day clean paper soak [time only]

4.1 Per `ops/pending_actions/BOX-059_sustained_clean_paper_soak_policy.md`:
    5 consecutive trading days, all 8 gates GREEN end-to-end,
    clock resets on any gate trip.

4.2 No operator action during the soak unless a gate trips. If a
    gate trips: record root cause, resolve, re-confirm all 8 gates
    GREEN for one cycle, then restart the soak clock.

4.3 Expected proof: 5-trading-day window with zero gate trips;
    operator writes
    `runtime/completion_matrix_evidence/BOX-059_SOAK_OBSERVATION_<ts>.md`
    with the gate-by-gate report.

### Step 5 — Complete Pre-Live Operator Tasks in `CLAUDE.md`

5.1 OS reboot only if a kernel update is pending (none as of
    2026-05-19; re-check at Step-5 time).
5.2 Disk cleanup if usage > 75%.
5.3 Verify all tests pass post-reboot (~1465 tests per
    `CLAUDE.md`).
5.4 Run `chad/core/full_cycle_preview.py --dry-run` clean.
5.5 `scripts/lint_systemd_wants_symlinks.sh` exit 0.
5.6 Review any open paper positions before mode switch.

### Step 6 — First-live canary plan approval [Channel 1]

6.1 Operator authors a canary plan under `ops/pending_actions/`:
    - first-trade instrument(s), sizing factor (start small),
      maximum loss before pull-back, kill-switch criteria, rollback
      path (revert tag = `RATIFICATION_MASTER_20260402` per
      `CLAUDE.md`).
6.2 Plan is reviewed and approved in writing.

### Step 7 — Operator GO recorded

7.1 Operator files an explicit GO entry under `ops/pending_actions/`
    (e.g. `BOX-062_operator_go_<ts>.md`).
7.2 GO entry references the soak observation, the canary plan, and
    the Pre-Live Operator Tasks verification.

### Step 8 — Flip `ready_for_live=true` via the canonical publisher

8.1 The live-readiness publisher (`ops/live_readiness_publish.py`)
    flips `ready_for_live=true` automatically when all 10 of its
    gates pass simultaneously (see Box-058 §5.1 for the gate
    enumeration). The operator does NOT edit
    `runtime/live_readiness.json` by hand.
8.2 Box 062 is closed only after the publisher's flip is observed
    in `runtime/live_readiness.json`.

## 5. Channel-specific next actions

### Channel 1 — Terminal actions requiring operator GO

- Step 1.3 — restart `chad-ibkr-broker-events.service` /
  `chad-ibkr-paper-fill-harvester.service` if root cause is
  transient.
- Step 2.2 — run `scripts/close_guard_entry.py` for the two MSFT
  entries.
- Step 5.x — OS reboot, disk cleanup, full_cycle_preview, lint.
- Step 7.1 — file operator GO entry.

### Channel 2 — Audit / code / doc actions

- Step 3 — read-only verification of the 8 soak gates.
- Step 4 — passive observation of the 5-day soak window.
- Optional documentation: if any soak gate trip occurs, record root
  cause + remediation under `ops/pending_actions/`.

### Manual external

- Step 1.1 — IB Gateway investigation (network, client-id, paper
  session validity).
- Step 5.1 — OS / kernel update if applicable.

## 6. Explicit "do not" list

The following actions are **forbidden** during this Box-062
preflight phase. They would either invalidate the soak window or
constitute unauthorized live activation:

- **Do not** edit `runtime/stop_bus.json` manually to clear it.
  The stop bus must auto-clear via the durable hysteresis path —
  manual clears mask still-broken signals.
- **Do not** edit `runtime/position_guard.json`,
  `runtime/position_guard_drift.json`,
  `runtime/reconciliation_state.json`,
  `runtime/trade_lifecycle_state.json`,
  `runtime/feed_state.json`,
  `runtime/scr_state.json`, or any other runtime JSON by hand.
- **Do not** edit `data/fills/*.ndjson`,
  `data/broker_events/*.ndjson`,
  `data/fees/*.ndjson`, or any SQLite database (`exec_state.sqlite3`,
  etc.) by hand.
- **Do not** flip `runtime/live_readiness.json` `ready_for_live` to
  `true` by hand. The publisher is the only authorized writer.
- **Do not** authorize live trading. CHAD_EXECUTION_MODE must remain
  `paper` until Step 8 completes.
- **Do not** place broker orders (live or paper) outside the
  scheduled live-loop cycles.
- **Do not** mark Official Box 062 closed before all 8 gates are
  simultaneously GREEN and a publisher-driven flip is observed.
- **Do not** silence the Box-059 / Box-060 / Box-061 guard tests to
  pass a future Box-062 check; relax them only in the same commit
  that lands the corresponding evidence.

## 7. Box 062 go/no-go checklist

Box 062 may **only** be attempted when **every** row below is YES.
Today, several rows are NO.

| # | Gate | Status today | Target |
|---|---|---|---|
| 1 | `stop_bus.active=false` | **NO** (active since 2026-05-20T04:20:11) | YES |
| 2 | `reconciliation_state.status=GREEN` | **NO** (RED) | YES |
| 3 | `position_guard_drift.drift_count=0` | **NO** (=2) | YES |
| 4 | `trade_lifecycle_state.backlog_flag=false` | **NO** (=true) | YES |
| 5 | `feed_state.ibkr_stocks` fresh | **NO** (~14 h stale) | YES |
| 6 | SCR `state ∈ {CONFIDENT, CAUTIOUS}`, `paper_only=false` | YES (CONFIDENT / false) | YES |
| 7 | `chad-ibkr-broker-events.service` last run = success | **NO** (failed) | YES |
| 8 | `chad-ibkr-paper-fill-harvester.service` last run = success | **NO** (failed) | YES |
| 9 | Box-059 5-day clean soak observation recorded | **NO** (0 trading days) | YES |
| 10 | Pre-Live Operator Tasks complete | **NO** (IB Gateway latency open) | YES |
| 11 | First-live canary plan approved | **NO** (not authored yet) | YES |
| 12 | Operator GO recorded under `ops/pending_actions/` | **NO** | YES |
| 13 | Publisher-driven `ready_for_live=true` observed in `runtime/live_readiness.json` | **NO** (false) | YES |

**Today's count: 1 YES / 12 NO.** Box 062 cannot proceed.

## 8. Pending Action

This plan is **policy / sequencing only**. There is **no runtime
config change** to apply by this document. No service restart is
required by this document. The operator actions in §4 are
prerequisites the operator must execute; this document does not
execute them. CHAD remains PAPER. Live trading remains NOT
authorized. `ready_for_live=false`. Box 062 remains BLOCKED.
