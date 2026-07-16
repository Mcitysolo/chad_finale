# PA — CONN-RESILIENCE: self-heal the shared truth-reading `ib` after a gateway reset

- **ID:** CONN-RESILIENCE
- **Filed:** 2026-07-16
- **Severity:** L1 (HIGH — broker truth goes dark for hours; exit checks pause
  the entire time, silent to trading correctness though loud to the operator)
- **Scope:** repo-only — code + tests + this PA + commit + push attempt.
  **NO DEPLOY. NO SERVICE RESTART.**
- **Baseline commit:** `ce0ab66` (`IR1-3: intel-repair R4`).
- **Trading posture unchanged:** paper-only; `ready_for_live=false`. Nothing here
  flips exec mode or adds/relaxes a live-order path. It only restores the ability
  to *read* broker truth after a connection drop.
- **Activation:** next `chad-live-loop` restart (operator GO). Until then the
  running process holds the pre-change code and behaviour is unchanged.
- **Kill-switch:** `CHAD_TRUTH_RECONNECT=0` (default ON) reverts to pure XOV2-1
  loud-and-dead — no code revert needed.

---

## 1. The bug (why truth stays dark)

The shared truth-reading `ib` (`chad/core/live_loop.py`, `ib.connect(...,
clientId=LIVE_LOOP=99, ...)`) is established **exactly once at module import**
and there is **no reconnect anywhere**. `position_sync.fetch_positions()` reads
`ib.positions()`, which is a pure read of ib_async's local `wrapper.positions`
cache — emptied by `wrapper.reset()` on any socket drop and repopulated **only**
by the `reqPositions` subscription issued inside `connectAsync`.

Two events drop that socket on a predictable cadence:
- **IBKR ~23:45 UTC daily session reset** (broker-side).
- **Our own 03:15 gateway nightly restart** (`chad-ibgw-restart`).

XOV-2345 already fixed the *dangerous* half of this: a dead connection used to
return `{}` (byte-identical to a flat broker) and false-flat **every** guard
entry. Post-XOV2-1, `fetch_positions()` fails **closed** — it raises
`BrokerTruthUnavailable`, the guard sweep is skipped (state preserved, no
false-flat), and the operator is alerted with `BROKER_TRUTH_UNAVAILABLE`.

But safe-and-loud is not the same as healed. Once dropped, the shared `ib`
**never came back on its own**. Observed **2026-07-16, 13:30→19:11 UTC**: 272
consecutive `BROKER_TRUTH_UNAVAILABLE` cycles with exit checks paused, ending
only when an operator manually restarted the process (which re-runs the
module-level connect). This PA makes the connection **HEAL** so a routine
gateway reset self-recovers within a cycle or two instead of waiting hours for a
human.

---

## 2. Chosen design (Option A — bounded reconnect on MainThread, in the guarded
`fetch_positions` path)

When `_rebuild_guard_from_broker` catches `BrokerTruthUnavailable`, it now calls
`_handle_broker_truth_unavailable`, which attempts a **bounded, backed-off
reconnect of the shared `ib` on MainThread** and, on success, re-reads truth and
resumes the rebuild.

New surface (all in `chad/core/live_loop.py`):
- `_attempt_shared_ib_reconnect(logger) -> bool` — up to
  `CHAD_TRUTH_RECONNECT_MAX_ATTEMPTS` (default 3) attempts, each a
  `disconnect()` then `ib.connect("127.0.0.1", 4002, clientId=LIVE_LOOP,
  timeout=CHAD_TRUTH_RECONNECT_TIMEOUT_S)` (default 30s), with exponential
  backoff between attempts (2s, 4s, capped 8s). Returns True iff
  `position_sync.api_connected()` probes healthy after a connect.
- `_emit_broker_truth_restored(logger)` — emits the **`BROKER_TRUTH_RESTORED`**
  marker + one calm, **coach-voiced** recovery note (via
  `coach_voice.format_alert`, fail-soft to a hand-written calm string).
- `_handle_broker_truth_unavailable(logger, exc)` — orchestrates: skip-mode
  → return None; heal-and-refetch → return fresh positions (+ RESTORED); else
  → stay XOV2-1 loud (`BROKER_TRUTH_UNAVAILABLE` + critical alert), return None.
- Tunables: `_truth_reconnect_enabled()` (kill-switch),
  `_truth_reconnect_max_attempts()`, `_truth_reconnect_timeout_s()` — all env
  driven with safe fallbacks.

### Why this design, argued against L1-CLD's lessons

L1-CLD (`ops/pending_actions/L1_CLD_cross_loop_deadlock_fix_2026-07-08.md`) roots
an uninterruptible cross-loop deadlock: the shared `ib`'s reader is bound to
**whatever loop ran `connectAsync`**, and a broker call dispatched onto a
*different* loop awaits a reply that only ever arrives on the connection's loop —
which, for the MainThread-homed shared `ib`, is not pumped. Its three lessons
are **never share loops, use cancellable/bounded timeouts, keep the reader
pumped**. Option A honours all three:

1. **Never share loops.** The shared `ib` was homed on MainThread, and **every**
   truth read (`fetch_positions`) runs on MainThread inside `run_once` →
   `_rebuild_guard_from_broker` (verified: the only production callers are
   `run_once`, itself only called from `run_loop` on MainThread). The reconnect
   therefore runs **on that same MainThread**, rebinding the reader to the
   **same** loop `run_once` already drives. No cross-loop hand-off is ever
   introduced. If the function is somehow entered off MainThread it **refuses**
   (`BROKER_TRUTH_RECONNECT_REFUSED`) and stays loud — it never rebinds the
   shared reader to a foreign loop.
2. **Cancellable / bounded timeouts.** Every `ib.connect` carries an explicit
   `timeout` (default 30s, deliberately well under the 120s boot connect so a
   mid-cycle heal cannot stall the loop for minutes). Attempts are bounded (no
   reconnect storm); on exhaustion it stays loud.
3. **Reader pumped.** `ib.connect` on MainThread runs `connectAsync` to
   completion (pumping the MainThread loop) and re-issues `reqPositions`; the
   very next `fetch_positions` reads that freshly-populated cache **within the
   same cycle**. This is exactly the mechanism XOV-2345 documents.

Blast-radius note: the reconnect happens at the **top of the cycle** (guard
rebuild is `run_once` step 1), before any downstream shared-`ib` read, on a
single-threaded MainThread cycle — no concurrent reader can race the
`disconnect`.

### Rejected alternative — Option B (dedicated truth-connection keeper thread)

Mirror the owner-loop pattern: a daemon thread with its **own** clientId running
`run_forever()`, reconnecting a second truth connection.

Rejected because it **reintroduces the exact hazard L1-CLD warns against**, for
no correctness gain:
- A keeper thread cannot reconnect the **shared** `ib` (that would rebind its
  reader onto the keeper's loop while `run_once` still reads it on MainThread —
  the cross-loop deadlock verbatim). So it would need a **separate** IB object on
  a **new** clientId — and then `fetch_positions` (MainThread) reading that
  keeper-loop-homed object is itself a cross-loop read unless **all** truth reads
  are marshalled onto the keeper loop via `run_coroutine_threadsafe`. That is a
  wholesale re-plumbing of the truth path (the entire owner-loop machinery), not
  a surgical heal.
- It **fragments broker truth** across two connections (99 + new id) — the same
  reason L1-CLD itself rejected per-worker connections (§3 of that PA):
  position/open-order/idempotency state assumes a single authoritative view.
- It adds a thread + socket + clientId + lifecycle for a failure mode
  (gateway reset) that a MainThread reconnect already covers.

The single-connection, MainThread, in-path reconnect keeps one authoritative
truth connection, one loop, and the smallest possible diff.

---

## 3. Scope boundary — what is explicitly NOT touched

- The **dedicated execution connection** (clientId `EXECUTION=9007`,
  `chad/execution/broker_loop.py`, the L1-CLD U7 owner loop) is **separate and
  healthy** and is **never referenced** by this change. Tests pin that the heal
  only ever reconnects `LIVE_LOOP=99`, never `9007`, and never calls
  `_home_execution_connection` or the execution adapter's `ensure_connected`.
- `chad/core/broker_position_sync.py` is unchanged (it already exposes
  `api_connected()` and `.ib`).
- **Reader-stall coverage is out of scope.** This heals a *dropped socket*
  (`isConnected()==False`, which is what raises `BrokerTruthUnavailable`). A
  reader-stall where `isConnected()` stays True but the cache goes stale is the
  owner-loop watchdog's job for the execution path (`BROKER_READER_STALLED`);
  for the truth path it is a smaller, separate risk and would need its own
  staleness probe. Not addressed here — stated so the boundary is honest.

---

## 4. Markers

- **New:** `BROKER_TRUTH_RESTORED` (heal + re-read succeeded → truth resumes;
  paired with a coach-voiced recovery note), `BROKER_TRUTH_RECONNECT_ATTEMPT`,
  `BROKER_TRUTH_RECONNECT_OK`, `BROKER_TRUTH_RECONNECT_EXHAUSTED`,
  `BROKER_TRUTH_RECONNECT_REFUSED` (off-MainThread), `BROKER_TRUTH_RECONNECT_DISABLED`
  (kill-switch off).
- **Preserved:** `BROKER_TRUTH_UNAVAILABLE` (XOV2-1 loud path, now fired only
  after a heal fails/exhausts).

---

## 5. Tests

`chad/tests/test_conn_resilience_truth_reconnect.py` (12 tests, no real IB I/O —
an in-memory healing fake models the ib_async cache-reset-on-drop contract):
- disconnect → reconnect → probe recovers; handler returns fresh truth +
  `BROKER_TRUTH_RESTORED` + coach note; full `_rebuild_guard_from_broker`
  resumes reconciliation from healed truth (open entry stays open).
- reconnect storm is **bounded** (exactly `max_attempts` connects) and
  **backed-off** (`sleep` 2s, 4s, none after the last); exhaustion stays XOV2-1
  loud.
- **owner-loop untouched:** refuses off-MainThread (no I/O); only ever uses
  `LIVE_LOOP` id, never `EXECUTION`; never calls the execution homing/connect.
- kill-switch off ⇒ no heal; default ON when unset; **hard env gate** — with
  `CHAD_SKIP_IB_CONNECT=1` set, no real I/O even if `_skip_ib_connect()` is
  forced False (a suite run can never touch the live gateway / clientId 99).
- invalid tunables fall back to defaults; `max_attempts` clamps to ≥1.

Regression: `test_xov2345_exit_overlay_silent_death.py` (18) and the
`_rebuild_guard_from_broker`/`run_once` callers
(`test_p0a_broker_sync_mirror.py`, `test_gap039_...`) stay green — the hard env
gate keeps the loud path firing under pytest exactly as before.

---

## 6. Verification

- `python3 -m py_compile chad/core/live_loop.py` — clean.
- `CHAD_SKIP_IB_CONNECT=1 pytest chad/tests/test_conn_resilience_truth_reconnect.py` — 12 pass.
- `CHAD_SKIP_IB_CONNECT=1 pytest chad/tests/` — green except pre-existing/
  date-driven reds (recorded in the commit body).
- `CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview` — clean.
- No systemctl / runtime/ / config / systemd changes. **NOT_DEPLOYED — awaiting
  operator GO (next chad-live-loop restart).**

---

## 7. Rollback

Code-only. Either set `CHAD_TRUTH_RECONNECT=0` (keeps the code, restores pure
XOV2-1 loud-and-dead — no restart needed once the switch is read next cycle), or
revert the CONN-RESILIENCE commit. No runtime state, config, or systemd unit is
touched.
