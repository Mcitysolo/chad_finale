# BOX-059 — Sustained clean paper soak policy

**Status:** Pending Action (paper / governance — defines the soak; does NOT pass it).
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 059.
**Title:** Sustained clean paper soak defined
**Acceptance criterion (verbatim):**
> "paper duration passes with no P0/P1, clean fills, clean reconciliation,
> and stable SCR."

## Decision

**SOAK POLICY DEFINED — SOAK NOT YET COMPLETE.** This document is the
official Box-059 definition of "sustained clean paper soak." It enumerates
the duration, the eight gate conditions, and the failure semantics. As of
the evidence timestamp in
`runtime/completion_matrix_evidence/BOX-059_OFFICIAL_sustained_clean_paper_soak_defined.md`,
the runtime currently fails several gates, so the soak window is **not yet
counting**. CHAD remains PAPER. `ready_for_live=false`.

## Soak window — minimum duration

**Five (5) consecutive trading days with all eight gates green, end-to-end.**

- A *trading day* is any RTH session in which `feed_state` is non-stale and
  `chad-live-loop.service` has been up the entire session. Weekends and
  holidays don't break the streak (no markets to soak through) but they
  also don't count toward the five.
- The clock resets to zero the moment **any** gate trips. Re-establishing
  cleanliness restarts the count — no carry-over.
- The clock starts at `now()` of the first cycle in which all eight gates
  are simultaneously green.

Five trading days is chosen to cover at least one full weekly options
expiry cycle, one full futures roll window for non-front-month
contracts, and four overnight broker-event reconciliations — the
smallest window that exercises every reconciliation cadence CHAD has.

## Eight gates (all must be GREEN continuously)

1. **Service uptime — no unexpected restarts.** `chad-live-loop.service`
   `ActiveState=active` / `SubState=running` for the entire window. The
   `MainPID` may change only via deliberate operator restart with an
   accompanying `ops/pending_actions/` entry.

2. **No open P0 — no P0 blocker** in `ops/pending_actions/` for any
   chad path. A P0 is anything that risks loss of capital, loss of
   data, or unsafe live execution. Existing P0 file format:
   `ops/pending_actions/P0_*.md` with `status: OPEN`.

3. **No open P1 — no P1 blocker** in `ops/pending_actions/`. A P1 is
   anything that degrades correctness or safety but does not put
   capital at immediate risk. Existing P1 file format:
   `ops/pending_actions/P1_*.md` with `status: OPEN`. Operationally
   "P0/P1" also includes any runtime-state flag that maps to those
   classes: `stop_bus.active=true`, `reconciliation_state.status=RED`,
   stale `feed_state`, or `trade_lifecycle_state.backlog_flag=true`.

4. **Fills clean.** Every paper fill in `data/fills/FILLS_<date>.ndjson`
   has a matching broker event in
   `data/broker_events/BROKER_EVENTS_IBKR_<date>.ndjson` (or a documented
   exclusion). No `pnl_untrusted=true` rows from new fills in the window.
   No fills land with `status` in `_PAPER_PENDING_STATUSES` (PendingSubmit,
   error, …) that fail normalize.

5. **Reconciliation GREEN.** `runtime/reconciliation_state.json`
   `status="GREEN"` (not RED, not YELLOW). `broker_source` is non-empty
   (`broker_source != "unavailable:"`). `mismatches=[]`. Any
   `exclusion_policy` entry must have a `reviewed_utc` within the
   trailing 30 days.

6. **position_guard_drift = 0.** `runtime/position_guard_drift.json`
   `drift_count=0` and `drifts=[]`. No `side_mismatch`, no
   `qty_mismatch`, no `chad_only_present`, no `broker_only_present`
   entries.

7. **Lifecycle backlog flag false; SCR stable.**
   `runtime/trade_lifecycle_state.json` `backlog_flag=false` AND
   `gap_flag=false`. `runtime/scr_state.json` `state ∈ {CONFIDENT,
   CAUTIOUS}` for the full window (never PAUSED), `paper_only=false`,
   `sizing_factor` is monotonic-or-rising (never dipped by SCR
   thrash). SCR transitions log a reason; "thrash" is two transitions
   in opposite directions within 24h — that breaks the streak.

8. **stop_bus auto-clear verified.** `runtime/stop_bus.json` either
   `active=false` (cleared) OR — if it tripped during the window —
   has cleared via the durable hysteresis path (GAP-034 / Phase-44)
   within the configured `clean_streak` threshold AND the `cleared_by`
   field reflects an automatic clear, not an operator override that
   masks a still-broken signal.

## Verification cadence

The soak gates are evaluated **every live-loop cycle** by the runtime
publishers that already write these state files (the live-readiness
publisher rolls eight of them into `runtime/live_readiness.json`).
Operators read the soak status from the same files — no new
publisher, no new daemon, no new instrumentation is required for
Box-059.

A soak observation may be summarized by the operator after the five
trading days complete by writing:

```
runtime/completion_matrix_evidence/BOX-059_SOAK_OBSERVATION_<ts>.md
```

with: window start/end, gate-by-gate cleanliness summary, any
sub-window perturbations, and a final pass/fail call. Only at that
point may a follow-up Box-059 closure assert
`PASS_SUSTAINED_CLEAN_PAPER_SOAK_VERIFIED`.

## Failure semantics

If any gate trips during the window, the clock resets to zero and the
operator must:

1. Record the trip and root cause in `ops/pending_actions/`.
2. Resolve the root cause (not silence it).
3. Re-confirm all eight gates are green for one full cycle before the
   new window begins.

The soak cannot be "patched through" by clearing a state file
manually; a manual clear is itself an operator override that counts
as a P1 unless paired with documented root-cause resolution.

## Live trading authorization

This policy **does not authorize live trading** and does **not** set
`ready_for_live=true`. Even after a clean five-day soak, live
activation still requires:

- The Pre-Live Operator Tasks listed in `CLAUDE.md` (OS reboot if
  kernel pending, IB Gateway latency resolution, disk cleanup, all
  1465 tests green after reboot, `full_cycle_preview.py` clean,
  `live_readiness.json` flips to `ready_for_live=true`, systemd
  `Wants` symlink lint passes, open paper positions reviewed).
- An explicit operator GO recorded in `ops/pending_actions/`.

CHAD remains PAPER until *all* of those steps are also satisfied.

## Verification command (for the policy guard tests)

```
source venv/bin/activate
python3 -m pytest chad/tests/test_box059_official_paper_soak_policy_guard.py -v
```

The guard tests verify only that the policy doc exists, enumerates
the required gates, and does not silently authorize live trading.
They do **not** assert the soak has passed.

## Pending Action

This document is a **policy** only. There is **no runtime config
change** to apply. No service restart is required. No operator
approval is needed beyond review. CHAD remains PAPER. Live trading
remains NOT authorized. `ready_for_live=false`. Box 60+ remain open.
