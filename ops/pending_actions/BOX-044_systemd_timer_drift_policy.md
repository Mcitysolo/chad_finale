# BOX-044 — Systemd timer drift policy (harmless monotonic-timer artifact)

**Status:** Closure documentation, not an open action. No Channel-1 step required.
**Authored:** 2026-05-20 UTC
**Scope:** Classifies the empty `NextElapseUSecRealtime` artifact for CHAD monotonic timers as a documented harmless systemd display behavior, and prescribes the operator verification method that supersedes the raw `systemctl show` field.

---

## 1. Background

CHAD runs 79 `chad-*` timers, of which 60 are *monotonic-only* (they use `OnBootSec=` and/or `OnUnitActiveSec=` but no `OnCalendar=`). At any given moment a small subset of these is in `SubState=running` because their backing oneshot service is mid-execution (e.g. fetching IBKR price snapshots, building a portfolio snapshot, etc.).

For each such timer, `systemctl show <timer>` returns:

```
NextElapseUSecRealtime=          (empty)
NextElapseUSecMonotonic=infinity
```

To an operator quickly glancing at unit state, this can look like a scheduling failure or like the timer has gone dormant. It is neither.

## 2. Root cause (upstream systemd semantics)

`OnUnitActiveSec=` measures the next fire **relative to the moment the unit re-enters inactive state**. While the backing oneshot service is mid-execution, that anchor does not yet exist, so systemd has no realtime clock target to compute and emits the sentinel `NextElapseUSecMonotonic=infinity` with an empty `NextElapseUSecRealtime`.

Once the oneshot returns and `OnUnitActiveSec=` starts ticking, both fields populate normally. This is upstream systemd behavior (verified against `systemd-show(1)` semantics for monotonic-only timer units) and not a CHAD-side defect, not a stale unit cache, and not a `NeedDaemonReload` condition.

## 3. Decision

The artifact is **classified as harmless**. CHAD does **not** patch unit files, drop-ins, or systemd configuration in response to this display behavior alone.

## 4. Verification method (operator-facing)

When a chad monotonic timer shows empty `NextElapseUSecRealtime`, verify health with:

1. `systemctl list-timers '<timer-name>' --all` — confirms a real recent `LAST` exists.
2. `systemctl show <timer-name> -p LastTriggerUSec,ActiveState,SubState,Persistent,NeedDaemonReload` — `ActiveState=active`, `SubState=running`, recent `LastTriggerUSec`, `Persistent=yes`, `NeedDaemonReload=no` is the expected healthy fingerprint.
3. `systemctl cat <timer-name>` — confirms the configured cadence contract (`OnUnitActiveSec=`, `OnBootSec=`).
4. `systemctl status <backing-service-name>` — verifies the oneshot is genuinely mid-fire (not stuck).

If all four are healthy, no action is required.

## 5. When the artifact becomes actionable

The empty `NextElapseUSecRealtime` graduates to actionable **only** if all of the following hold simultaneously:

- The backing oneshot service has been in `SubState=running` for substantially longer than the timer's expected wall-clock budget (e.g. > 10× cadence), AND
- `LastTriggerUSec` has gone stale (no recent fires beyond expected cadence), AND
- `systemctl list-timers` shows no recent `LAST` for the timer, AND
- The backing service does not already carry a finite `RuntimeMaxSec=` / `TimeoutStartSec=` bound that would self-terminate the stuck run.

That conjunction is the **Box 21 / NEW-GAP-043 stuck-oneshot pattern**. It is already remediated for `chad-ibkr-collector.service` by `/etc/systemd/system/chad-ibkr-collector.service.d/10-timeout-guards.conf` (`Type=oneshot` + `RuntimeMaxSec=90` + `TimeoutStartSec=60` + `TimeoutStopSec=30`) and the code-level SIGALRM guard in `chad/portfolio/ibkr_portfolio_collector_v2.py::install_wall_clock_guard`. If the same conjunction is observed for a different monotonic timer in the future, open a new pending action mirroring Box 21 — do **not** modify Box 44 closure.

## 6. Cached-vs-disk drift posture (as of 2026-05-20)

- `NeedDaemonReload=yes` count across all chad-* units: **0**.
- "unit file changed on disk" hits in 30-day journal: **0**.
- "has no effect" / "RuntimeMaxSec=" warning hits in 7-day journal: **0**.
- Repo `ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf` vs live drop-in: **byte-identical**.

No systemd reload is needed. The CHAD systemd cache and on-disk unit set are in agreement.

## 7. What this policy does NOT authorize

- It does **not** authorize any `systemctl daemon-reload`, `restart`, `start`, `stop`, or `reset-failed` invocation.
- It does **not** authorize installing or modifying any unit file or drop-in in `/etc/systemd/system/`.
- It does **not** authorize any live trading mode change, runtime JSON write, SQLite mutation, broker order action, or order cancellation.

Any future Channel-1 operator deployment touching CHAD systemd state must go through its own pending action (see precedent in `ops/pending_actions/BOX-039_chad_scr_sync_telegram_env.md` and the deploy block at the bottom of `ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf`).

## 8. References

- Evidence: `runtime/completion_matrix_evidence/BOX-044_monotonic_timer_systemd_cached_disk_drift_resolved.md`
- Box 21 closure (RuntimeMaxSec + oneshot): `runtime/completion_matrix_evidence/BOX-021_NEW-GAP-043_collector_runtime_reverify.md`, `runtime/completion_matrix_evidence/BOX-021_NEW-GAP-043_collector_systemd_deploy_verify.md`
- Drop-in source of truth: `ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf`
- Wants-symlink linter (GAP-032): `scripts/lint_systemd_wants_symlinks.sh`
