> **STATUS UPDATE 2026-05-28:**
> Partial implementation landed via Fix B (commit `bbab921`).
> Channel 2 artifacts created:
> - `ops/systemd_templates/chad-ibgateway-nightly-restart.service`
> - `ops/systemd_templates/chad-ibgateway-nightly-restart.timer`
> - `scripts/post_gateway_restart_verify.py`
> - `ops/runbooks/INSTALL_chad_ibgateway_nightly_restart_2026-05-28.md`
>
> Scope of Fix B: SCHEDULED restart only (prevents the socket-leak wedge
> by restarting before it can wedge). The detect-and-auto-restart-on-wedge
> behavior (full IBKR_AUTO_RECOVERY) remains pending as a separate phase.
> Prerequisite already live: Fix A symmetric hysteresis (`f3ab3d8` + `291bb84`)
> ensures the restart window does not trip stop_bus.
>
> Awaiting Channel 1 operator-authorized install per CLAUDE.md rule #6.

# IBKR-RELIABILITY — Socket Backpressure and IB Gateway Churn (live-loop cannot self-recover from sustained latency)

# Status: PROPOSED

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §17 R7-B (root-cause grouping)
  - §19 closeout item 9 (STOP-BUS-RECOVERY-1)
  - §14 (Tier 13 journal scan)
- Parent: `ops/pending_actions/PAPER_EPOCH_3_START_2026-05-27.md` §13 (deferred hardening note)
- Day-0 commit: `02ccda7` — "Paper Epoch 3: log dirty day zero latency halt"

# Severity
**HIGH** (Effort L) per audit §18 / §19 item 9.

# Problem statement
The live-loop's stop_bus auto-recovery path (`auto_recovery:broker_latency_clean_streak=5`) is sufficient for short flutter events (≤ ~13 min observed) but **cannot self-recover from sustained IBKR latency** when the underlying IB Gateway socket has entered a backpressure / deadlocked state. On Day-0 (2026-05-27), a 7h 11m halt (04:24:23Z → 11:35:36Z) only cleared after an **operator-authorised `chad-ibgateway.service` restart**. The live-loop's internal recovery never advanced.

Behavioural pattern:
- Short flutter (rolling avg > 2000 ms for ≤ ~10 cycles): auto-recovers via `clean_streak=5` within ~5 min after latency drops below threshold.
- Sustained halt (avg > 8000 ms for an extended period): live-loop continues to publish `STOP_BUS_ACTIVE` indefinitely; socket/queue state inside the IB Gateway does not drain even after the upstream load condition passes; operator must restart `chad-ibgateway.service` to break the deadlock.

# Evidence
1. **Day-0 incident timeline (2026-05-27):**
   - 04:24:23Z: `STOP_BUS_TRIGGERED reason=broker_latency:avg_latency_ms=N>threshold=2000.0`.
   - 04:24Z → 11:35Z: continuous `STOP_BUS_ACTIVE … skipping cycle` lines, ~1 per minute, for 7h 11m.
   - 11:35:36Z: stop_bus cleared by `auto_recovery:broker_latency_clean_streak=5` — **but only after** the operator restarted `chad-ibgateway.service`.
   - Day-0 logged DIRTY / NOT COUNTABLE per Epoch 3 parent §12.

2. **Today's residual flutter (auto-recovered):**
   - 12:34:04Z → 12:47:05Z (~13 min): `avg_latency_ms=8222.2 → 8588.5`, auto-cleared via `clean_streak=5` after 240s elapsed.
   - 13:07:50Z → 13:12:51Z (~5 min): `avg_latency_ms=3217.9`, auto-cleared.

3. **IBKR connect latency last 30 min (audit Tier 1):** range 318 ms → 1589 ms, mostly < 1000 ms. Sub-threshold but visibly noisier than the 300–500 ms baseline of a clean day.

4. **Same-cause symptoms elsewhere:**
   - `chad-options-chain-refresh.service` failed at 12:33:07Z with `Error 2103/2105/2108/2157` (data farms broken).
   - `chad-live-loop` journal shows `Error 1100, reqId -1: Connectivity between IBKR and Trader Workstation has been lost`.
   - `chad-ibkr-bar-provider` emits 588 errors/24h (separate root cause — futures-roll, see FUTURES_CONTRACT_EXPIRY Pending Action — but masks IBKR-side intermittents in journal noise).

5. **No instrumentation for "consecutive cycles above stop threshold"** — `runtime/ibkr_status.json` carries `latency_ms` per tick but no running counter of how long the system has been in degraded state. Operators must infer from journal pattern.

6. **No socket-state observability** — there is no published artifact for IB Gateway Send-Q / Recv-Q depth, no count of in-flight `reqId`s, no per-client outstanding-call gauge. The deadlock state is invisible to the orchestrator.

# Affected files / services / artifacts
- `chad-live-loop.service` (auto-restart disabled per v8.9 SS01 invariant)
- `chad-ibgateway.service` (the unit that had to be restarted)
- `chad-ibkr-health.service` + `.timer` (60 s health probe)
- `chad-ibkr-watchdog.service` (current watchdog scope unclear — REQUIRES AUDIT)
- `chad/core/live_loop.py` (stop_bus evaluator)
- `chad/core/stop_bus.py` or equivalent (the bus producer)
- `chad/execution/ibkr_adapter.py` (socket/client churn)
- `chad/execution/ibkr_client_ids.py` (clientId map)
- `runtime/ibkr_status.json` (no schema_version)
- `runtime/stop_bus.json` (`stop_bus.v1`)
- `runtime/stop_bus_recovery_state.json` (`stop_bus_recovery_state.v1`)
- `runtime/ibkr_watchdog_last.json` (no schema_version)

# Root-cause hypothesis
Three candidate sub-causes (any combination):
1. **Gateway Send-Q backlog**: under sustained latency, accumulated outbound API calls (bars, market data, contract details) saturate the IB Gateway's TWS socket buffer. Even after upstream latency clears, the buffer is still flushing previously-queued requests, sustaining high observed `connect_ms`/`api_ms` until the buffer drains — which can never complete if new requests keep arriving.
2. **Client churn**: each new probe/reconnect attempt allocates a fresh clientId; if old clientIds are not properly torn down, the Gateway accumulates zombie connections that eat slots and CPU.
3. **Reader-loop backpressure**: `ib_async`'s internal reader loop may get behind on draining the inbound queue if a downstream consumer holds the GIL for too long (e.g. during a heavy bars-cache write). The Gateway sees ACKs slow down and tightens its own pacing, which the live-loop then interprets as "latency."

Any sub-cause individually could explain the failure mode; the deadlock is the combination.

# Why this matters
1. Day-0's 7h 11m halt is the **single largest operational interruption** observed in Epoch 3 prep and was the dominant reason for declaring Day-0 DIRTY.
2. Without in-process recovery, every sustained IBKR outage forces an operator-authorised `chad-ibgateway.service` restart — which itself disqualifies the day under Session 1 criterion §3.5 (auto-recovery only) and §3.17 (no operator service restarts).
3. **Live promotion is blocked** while a single farm outage requires manual intervention. The Pre-Live Operator Tasks in `CLAUDE.md` include "IB Gateway latency — investigate and resolve dangerous (>750 ms) classification" — this is the same item, escalated.
4. Repeat probability: the data-farm flutter is **daily** during this audit window (≥3 distinct events on Day-0 and the morning of 2026-05-27).
5. Observability gap: even when the system IS in deadlock, the operator has no `ibkr_status` field that says "Send-Q depth = X" or "consecutive_above_threshold_cycles = Y" — diagnosis is journal-only.

# Current safety posture
- All live gates held: `ready_for_live=false`, `allow_ibkr_live=false`, `allow_ibkr_paper=true`.
- stop_bus correctly halts the live-loop during latency events — the safety contract is honoured.
- No broker orders are placed under `STOP_BUS_ACTIVE`.
- Paper mode tolerates the halt without financial impact (PnL = $0 during halt).
- v8.9 SS01 invariant (`chad-live-loop` and `chad-orchestrator` NEVER_AUTO_RESTART) is **deliberate** and must be preserved.

# Scope for future remediation
Layered approach; each layer is independently mergeable:

1. **Observability layer** (smallest, lowest risk):
   - Add `runtime/ibkr_status.json::consecutive_cycles_above_stop_threshold` counter.
   - Add `runtime/ibkr_status.json::ib_gateway_socket_state` (best-effort, may require Gateway logfile parsing).
   - Add `runtime/ibkr_status.v1` `schema_version`.
   - Publish to Telegram on sustained breach (e.g. > 10 consecutive cycles above threshold without auto-recovery).

2. **Deadlock characterisation**:
   - Add a one-shot diagnostic Pending Action to instrument an `ibkr_adapter` probe that, during stop_bus, samples Send-Q / Recv-Q / outstanding-reqId counts.
   - Compare deadlock vs healthy state for at least one observed flutter event.

3. **In-process recovery**:
   - Define a safe in-process socket reset (`ibkr_adapter.disconnect_and_reconnect`) gated by stop_bus + `consecutive_cycles_above_threshold > N` (e.g. N=15, ~ 15 min).
   - The reset must use a fresh clientId, properly close the prior `ib_async.IB` instance, and re-register subscriptions.
   - Must NOT restart the systemd unit; must NOT cancel any pending orders; must NOT affect `chad-orchestrator`.
   - Behind a feature flag (`CHAD_IBKR_IN_PROCESS_RECOVERY=disabled` by default).

4. **Watchdog escalation**:
   - If in-process recovery itself fails after K attempts (K=3), publish a hard-fail artifact and alert the operator with a runbook link. **Do not** auto-restart any systemd unit.

5. **Test harness**:
   - Mock-IBKR fault injection that simulates a 30+ min latency excursion.
   - Assert the in-process recovery activates, completes successfully, and live-loop resumes cycling **without** any systemd state change.

# Explicitly out of scope
- Live-mode enablement.
- Restarting `chad-ibgateway.service` (operator action, separate Pending Action).
- Clearing stop_bus manually (forbidden by Session 1 tracker §3.5).
- Changing the stop_bus threshold (currently 2000 ms; this Pending Action does not propose to tune it).
- Modifying the v8.9 SS01 NEVER_AUTO_RESTART invariant for `chad-live-loop` or `chad-orchestrator`.
- Migrating off ib_async (separate Pending Action: ib_insync phase 2 closure).

# Required tests
- New: mock-IBKR latency fault-injection test (≥ 15 min sustained); assert observability counter increments and the in-process recovery activates exactly once.
- New: ensure `runtime/ibkr_status.json` carries `schema_version=ibkr_status.v1` after the observability layer lands.
- New: regression test that the v8.9 SS01 invariant is preserved (`chad-live-loop` and `chad-orchestrator` units are not restarted by any code path under load).
- Existing must continue to pass: `chad/tests/test_*ibkr*.py`, `chad/tests/test_*stop_bus*.py`, `chad/tests/test_ib_async_*.py`, plus the full baseline.

# Required runtime verification
After each layer lands:
- **Observability**: a real flutter event produces the new counter values and an artifact change visible in `runtime/ibkr_status.json`.
- **In-process recovery**: in a controlled fault-injection (paper, off-hours), a simulated 30 min latency excursion clears via the in-process path with no operator action. `chad-ibgateway.service` uptime is unchanged.
- **Watchdog escalation**: an alert lands in Telegram on K-failed recoveries; no systemd state change.
- **Day-0-style scenario** (live test): a real >30 min outage clears via the new path, and the day is no longer NOT-COUNTABLE for that reason.

# Operator approvals required
- **Approval 1 (observability)**: schema additions to `runtime/ibkr_status.json` (low risk — additive).
- **Approval 2 (in-process recovery design)**: socket reset logic, clientId churn rules, subscription re-registration. **Highest-risk approval** in this Pending Action.
- **Approval 3 (feature-flag enablement)**: which environment turns `CHAD_IBKR_IN_PROCESS_RECOVERY` on first (paper → soak → live).
- **Approval 4 (Telegram alert routing)**: alert format for sustained breach and for K-failed recoveries.
- **Approval 5 (SS01 invariant confirmation)**: explicit written confirmation that the NEVER_AUTO_RESTART invariant for `chad-live-loop` and `chad-orchestrator` is preserved by the design.

# Definition of done
1. Observability layer merged; `runtime/ibkr_status.json` carries `schema_version=ibkr_status.v1` and `consecutive_cycles_above_stop_threshold`.
2. One real flutter event (after deploy) produces the counter and an alert.
3. In-process recovery design reviewed and approved by operator.
4. In-process recovery merged behind feature flag; controlled fault-injection passes.
5. Feature flag enabled in paper; observed for at least one real sustained-latency event (or 14 days clean if no event recurs).
6. Watchdog escalation merged; K-failed alert path verified.
7. The `IBKR-RELIABILITY` deferred-hardening note in Epoch 3 §13 is closed with a link to the merged PRs.
8. Pre-Live Operator Task 3 ("IB Gateway latency") in `CLAUDE.md` is resolvable.
9. SSOT / forward-erratum doc updated.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.

# Session 1 impact
- **Does this item block Session 1 evaluation (window opens 2026-05-28T00:00:00Z)?** NO — but a recurrence DURING Session 1 would force NOT COUNTABLE.
- The remediation cannot land before Session 1 opens (~9 h from this Pending Action creation); it is an Effort-L workstream requiring design, review, fault-injection testing, and soak.
- **Mitigation for Session 1**: rely on the existing `auto_recovery:broker_latency_clean_streak=5` path. If a sustained outage occurs and an operator-authorised `chad-ibgateway.service` restart is required, Session 1 is automatically NOT COUNTABLE per §3.5 / §3.17 of the Session 1 tracker — and Session 2 candidate window starts the next clean US-equity day.
- **Latest acceptable closure date**: before the second consecutive Session that fails for this reason. If three sessions fail because of this gap, the soak window cannot complete and the live-readiness path is structurally blocked.
