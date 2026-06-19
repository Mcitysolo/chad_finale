# Pending Action — Bug A (L1): root-fix the per-call event-loop/fd leak in ibkr_adapter
Date: 2026-06-04  •  Author: TEAM CHAD (issued) / SOLO (executes)  •  Status: IMPLEMENTED / LANDED (2026-06-19)  •  Priority: HIGH (the leak's only current mask is the Bug B env gate — see §5)

## 0. Implementation status (2026-06-19) — LANDED
The original scope of this PA was L-01/L-02 only (the executor / event-loop leak in ibkr_adapter._call_with_timeout). On GO the scope was extended to the **full leak class** surfaced by the 2026-06-09 fd-leak audit (L-03/L-04/L-05), and all four items have landed; two are also live (services restarted):

| Item | Leak site | Commit | State |
|------|-----------|--------|-------|
| L-01/L-02 | executor / event-loop (ibkr_adapter._call_with_timeout, shared bounded broker executor) | 275acff | **LANDED + LIVE** — live-loop + orchestrator restarted; epoll fds 19 → 1 |
| L-03 | `_SQLiteIdempotencyStore` per-call SQLite connection (fd leak on order-submit hot path) | ce9fad7 | **LANDED** |
| L-04 | kraken_executor `ExecStateStore` per-call SQLite connection (twin of L-03) | 710e262 | **LANDED** |
| L-05 | backend ai_surface `GPTClient` per-request Session/socket leak (now a process singleton) | 87c002f | **LANDED + LIVE** — backend restarted |

The `CHAD_DISABLE_FUTURES_EXECUTION` env-gate (the Bug B mask, §5) **REMAINS ON** as defense-in-depth. This fix removes the leak's load-bearing role but does NOT unlock futures re-enable: removing the env-gate is a separate future decision (Bug B disposition), explicitly NOT granted by this PA. This PA does not mark `ready_for_live` and removes no gate.

## 1. Objective
Root-fix the event-loop / file-descriptor leak behind the 2026-05-30 fd-exhaustion (Errno 24) silent freeze. Replace per-broker-call thread+loop minting in ibkr_adapter._call_with_timeout with one bounded module-level executor, so loop/fd count stays constant regardless of broker-call volume. Retires the LimitNOFILE band-aid's load-bearing role.

## 2. Current state — DORMANT, not absent, not fixed
Running live_loop (PID 3633421, 23h): fd=37 flat, 10 epoll fds = 10 live (leaked) asyncio loops, FDSize=512 high-water (bursts >256 fds occurred this run). The profile IS the leak signature at low volume. Dormant ONLY because CHAD_DISABLE_FUTURES_EXECUTION=1 (the Bug B env gate, installed 2026-05-30 as a same-day sibling of the LimitNOFILE drop-in) removed the high-frequency caller. The 05-30 incident: a repeated futures-submission loop outran GC -> 1024 fds -> process-wide freeze. Mask and trigger were installed together; the leak was never root-fixed.

## 3. Root cause (smoking gun)
ibkr_adapter.py:93-123 _call_with_timeout spawns a NEW daemon thread per broker call; inside each, ib_async sync API -> util.run() -> util.getLoop() -> asyncio.new_event_loop()+set_event_loop(), never closed. Every broker call = 1 thread + 1 loop (3 fds), reclaimed only by nondeterministic GC after the thread dies; a TIMED-OUT call leaks a live hung thread+loop permanently. Touchpoints: qualifyContracts (1554,2954), whatIfOrder (2401), placeOrder (2480), openTrades (2796). Normal cadence: GC keeps pace (plateau). Submission/timeout storm: outruns GC -> exhaustion.
Retry amplification: call-site retry-on-timeout makes the storm mode superlinear — each retry mints a fresh thread+loop while the timed-out predecessor still holds its leaked thread+loop, so one slow-gateway window multiplies fd burn; post-fix, retries queue against the bounded pool instead (observable via the §4 saturation marker).
Duplicate pattern (2026-06-07 audit): chad/execution/ibkr_trade_router.py:17 _call_with_timeout repeats the per-call daemon-thread mint (same leak-on-timeout mode; any ib_async sync call routed through it mints a loop via util.run() in the throwaway thread). Fixing ibkr_adapter alone leaves this twin armed — extract the bounded executor into a shared module consumed by both (§4).
NOT the leak (confirmed): live_loop _ensure_thread_event_loop (bounded long-lived listeners); context_builder asyncio.run fallbacks (churn, no accumulation, tool paths only); CLI-entrypoint asyncio.run (one-shot).

## 4. The change (recommended: bounded executor)
One bounded executor in a shared module (consumed by ibkr_adapter.py AND the router twin flagged in §3): a module-level ThreadPoolExecutor, `max_workers=4` (pinned), worker-initializer installs one event loop per worker thread once. Lifecycle: created at module import, daemon workers, per-worker loops live for the process lifetime; no shutdown/teardown path and never recreated (process exit reaps daemon workers). _call_with_timeout becomes: future = _BROKER_EXECUTOR.submit(fn, *args); future.result(timeout=timeout_s) — SAME signature, SAME BrokerTimeoutError, SAME fail-soft classification -> ZERO call-site changes. Loops become constant ~10-12 fds forever; the 05-30 incident class becomes structurally impossible.
Deliberate semantic change to accept: if all 4 workers hang, subsequent calls fail fast — correct fail-closed against a dead gateway, but a change from "always mint a fresh thread." Document in the PA. Instrument it: when a submit finds the pool saturated, emit log marker `BUG_A_POOL_SATURATED` so this fail-fast mode is journald-observable rather than silent.
Rejected alternative: per-call thread with loop.close() in finally — kills GC-dependent accumulation but still leaks on every timeout (the danger mode). Executor is the real fix.

## 5. Sequencing — COUPLED to the Bug B disposition (critical)
The Bug B env gate is the leak's current mask. Therefore the Bug A fix MUST land before (or with) any futures-execution re-enable / gate removal (Bug B disposition options b, c). Re-enabling futures while the leak is unfixed re-arms the high-frequency caller that caused the 05-30 freeze; the Fix A cap blocks over-cap submits but does not address the leak. Disposition options that KEEP the gate armed (a, d) are unaffected.

## 6. Risk / deploy
Risk-tightening stability fix; touches the live submit hot path (ibkr_adapter.py, one function + one executor); no config mutation. -> gated PA + gated chad-live-loop restart (rules #6/#7). Its activation-restart can BUNDLE with the L3 restart (and, only after Bug A is verified, any gate-removal restart).

## 7. Procedure (on GO)
1. (Ch2) Add module-level executor + initializer; rewrite _call_with_timeout to submit/result; preserve the BrokerTimeoutError + fail-soft contract.
2. py_compile; new tests (N sequential _call_with_timeout calls do NOT grow fd/loop count; timeout-path raises BrokerTimeoutError without leaving a live leaked thread); full suite (expect zero regression); full_cycle_preview clean.
3. Commit (rule #8).
4. (Ch1, gated) chad-live-loop restart; post-restart soak: epoll-fd/loop count flat across cycles including with broker activity.

## 8. Verification
Tests green. Post-restart: loop/fd count flat over time; an observed broker timeout leaves no live leaked thread; ideally a higher-volume window (or post futures-re-enable) shows no fd growth.
Mechanical loop-count assertion: count `anon_inode:[eventpoll]` entries under /proc/<live_loop_pid>/fd. Today's adapter shows ~10 (one per leaked loop); after the fix this number must be ≤ max_workers + 1 (one per persistent worker loop + the main-thread loop) and stay flat across cycles, including ones with broker activity. For unit-test fd assertions use `len(os.listdir('/proc/self/fd'))` with a ±2 tolerance band to avoid flaking on pytest plugin fds.

## 9. Out of scope (separate)
- Publisher silent-hang (lifecycle/snapshot/reconciliation oneshots lack RuntimeMaxSec): related family, DIFFERENT mechanism (unbounded blocking call -> unit stuck "activating" -> timer won't re-fire). Separate tiny systemd PA (RuntimeMaxSec=), rule #6, explicit instruction required.
- 90-fd-limit.conf: keep as defense-in-depth; comment-only amendment after the fix soaks (do not remove the safety margin).

## 10. Status log
- 2026-06-04: authored from L1/Bug A read-first (leak DORMANT not fixed; root cause = per-call thread+loop mint in ibkr_adapter._call_with_timeout; mask = Bug B env gate). PENDING operator GO.
- 2026-06-07: refined with audit deltas — router-twin shared-module extraction (§3/§4), retry-amplification note (§3), executor size pinned + lifecycle (§4), pool-saturation log marker (§4), mechanical epoll-fd assertion (§8). Status unchanged: PENDING operator GO.
- 2026-06-19: **IMPLEMENTED / LANDED.** Scope extended from L-01/L-02 to the full 2026-06-09 leak class (L-01/L-02 executor → 275acff, LIVE [live-loop + orchestrator restarted, epoll 19→1]; L-03 _SQLiteIdempotencyStore → ce9fad7; L-04 kraken ExecStateStore → 710e262; L-05 backend GPTClient session → 87c002f, LIVE [backend restarted]). CHAD_DISABLE_FUTURES_EXECUTION env-gate REMAINS ON as defense-in-depth (removal is a separate future decision, not unlocked here). No `ready_for_live` flip; no gate removed. See §0.
