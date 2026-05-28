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

# IBKR auto-recovery design (STOP-BUS-RECOVERY-1 follow-on)

# HIGH_ID: STOP-BUS-RECOVERY-1 (recovery-design portion)
# Status: PROPOSED
# Severity: HIGH (companion to IBKR_RELIABILITY_socket_backpressure_and_gateway_churn_2026-05-27.md)
# Effort: L

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §17 R7-B (Stop-bus flutter pattern + Day-0 7h 11m halt requiring operator restart)
  - §19 closeout item 9 (STOP-BUS-RECOVERY-1)
  - §13 Epoch 3 declaration "deferred hardening note IBKR-RELIABILITY"

# Problem statement
The stop_bus latches when broker avg_latency rises above the threshold for
sustained cycles, and auto-clears via `broker_latency_clean_streak=5`
(~5 min) when latency returns to normal. On Day-0 of Paper Epoch 3, a
**7h 11m halt** required an operator-authorised restart of
`chad-ibgateway.service`. The clean-streak path could not recover the
session because the gateway itself was wedged, not just the latency.

This Pending Action proposes a design — code is **not** authored here — for
a safe in-process recovery path that does not require a systemd restart.

# Observability landed in this batch (already shipped)
- New module `chad/ops/ibkr_reliability_tracker.py` augments
  `runtime/ibkr_status.json` with:
  - `consecutive_cycles_above_stop_threshold`
  - `last_above_threshold_at`
  - `max_latency_observed_in_window`
  - `current_recovery_state` (healthy / degrading / above_threshold / recovering)
  - `last_gateway_churn_at`
- New health-monitor rule `R19` raises a CRITICAL Telegram alert when the
  counter crosses a configurable threshold (default 5).
- These additions do NOT change recovery behaviour; they make the
  pattern visible to operators before the existing 5-min clean_streak
  decides what to do.

# Required design (out of scope for this batch — operator-domain)
1. **Characterise the Day-0 deadlock.** From `journalctl`:
   - What was the last successful API call before the wedge?
   - Did `reqCurrentTime` time out, hang, or return stale data?
   - Was the underlying TCP socket alive (RST/FIN/ESTAB)?
   - Did the IB Gateway Java process exhaust file descriptors / threads?
2. **Design the in-process recovery.**
   - Trigger conditions: `consecutive_cycles_above_stop_threshold >= N`
     AND `current_recovery_state == "above_threshold"` for > M minutes.
   - Action sequence (least-invasive first):
     a. `ib.disconnect()` + reconnect with the same client_id.
     b. Reconnect with a new client_id (treated as gateway churn — records
        `last_gateway_churn_at`).
     c. Escalate to operator via Telegram only — no autonomous
        `systemctl restart chad-ibgateway.service` (operator policy).
3. **Idempotency and rate-limiting.**
   - Recovery attempts must be exponential-backoff with a hard ceiling
     (e.g. max 3 attempts in 30 min before escalating to operator-only).
   - Recovery must NOT fire while broker_authority_status is RED.
4. **Telemetry.**
   - Each attempt writes a structured record to
     `runtime/ibkr_recovery_attempts.json` (new file; needs its own
     schema_version per SCHEMA-VERSION-1).
   - The health monitor reads that file and surfaces it.

# Required code/tests after the design lands (not in this PA)
- New `chad/core/ibkr_recovery.py` module (or extend `ibkr_watchdog.py`).
- Unit tests with mocked IB clients (no real broker calls).
- Integration test: simulate a deadlock in a test harness; assert
  recovery either succeeds or escalates within budget.

# Acceptance criteria (for the design closeout)
- A documented design lives in `docs/IBKR_RECOVERY_DESIGN.md` (new doc).
- The design is reviewed by the operator and converted into a tracked
  PR with explicit GO before any code lands.
- A future stop_bus halt > 30 min is recoverable via the live-loop's own
  path with no operator restart (per audit §19 item 9 acceptance).

# Required tests (placeholder for the future PR)
- `chad/tests/test_ibkr_recovery.py` (does not exist yet).
- Tests in this batch (`chad/tests/test_ibkr_reliability_tracker.py`)
  cover only the observability layer.

# Operator approvals required
- **Approval 1 (design doc):** operator approves the design before code
  is authored.
- **Approval 2 (in-process disconnect):** explicit GO for the autonomous
  `ib.disconnect()` + reconnect path.
- **Approval 3 (no systemd restart):** explicit confirmation that the
  autonomous path never invokes `systemctl restart`.

# Session 1 impact
- **Does this item block Session 1?** NO.
- Today's observability layer is already in code; the alert path goes
  through the existing health-monitor → Telegram pipeline.
- If a Session 1 halt occurs, the operator-authorised manual recovery
  path (the Day-0 procedure) remains the fallback.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.
