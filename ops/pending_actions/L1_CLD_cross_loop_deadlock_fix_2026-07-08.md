# PA — L1-CLD: Cross-Loop Deadlock Fix (Connection-Owner Loop Architecture)

- **ID:** L1-CLD
- **Filed:** 2026-07-08 (autopsy) / implemented 2026-07-09
- **Severity:** L1 (HIGH — broker I/O can hard-deadlock, uninterruptible)
- **Scope:** repo-only — code + tests + this PA. **NO DEPLOY. NO SERVICE RESTART.**
- **Baseline commit:** `3c80d40` (`feat(margin-block): Phase B2 ALLOW/BLOCK gate`)
- **Trading posture unchanged:** paper-only; `ready_for_live=false`; `allow_ibkr_live=false`.
  Nothing here flips exec mode or adds a live-order path.

---

## 1. Autopsy summary (the bug)

An operator py-spy autopsy dated **2026-07-08 (HIGH confidence)** identified a
**cross-loop deadlock** in the IBKR execution path: broker calls that require a
request/response round-trip (contract-details / `qualifyContracts`) can hang
forever, uninterruptibly, pinning the broker worker pool until the process is
killed.

> Grounding note: the py-spy stack dump is the operator-provided autopsy input.
> This PA does not reproduce that dump verbatim; instead it corroborates the
> autopsy's conclusion against the **current source** (citations below), all of
> which are consistent with the described cross-loop mechanism. ib_async
> Context7 docs were **not available** in this session (Context7 MCP absent), so
> the library-side claims are grounded against the installed **ib_async 2.1.0**
> source in the venv (`venv/lib/python3.12/site-packages/ib_async/`).

### Mechanism (why it deadlocks)

1. **The connection is established on MainThread.**
   `chad/core/live_loop.py:106` calls `util.patchAsyncio()` (nest_asyncio),
   `:108` creates the shared `ib = IB()`, and `:121` calls the **sync**
   `ib.connect("127.0.0.1", 4002, clientId=…, timeout=120)` on MainThread.
   ib_async binds the socket transport + the incoming-data reader to **whatever
   loop is running when `connectAsync` executes** — here, MainThread's loop
   (`ib_async/connection.py:67 data_received → hasData.emit`; processed at
   `ib_async/client.py:357 _onSocketHasData`). After the sync `connect()`
   returns, that loop is **not run again** — nobody pumps the reader.

2. **The pre-connected `ib` is injected into the execution adapter.**
   `chad/core/live_loop.py:141 ib_factory=lambda: ib`. The live execution path
   is `live_loop.py:2355 _paper_adapter.submit_strategy_trade_intents(...)` →
   `IbkrAdapter` (NOT `ibkr_trade_router`).

3. **Broker calls are dispatched onto *different* loops.**
   `chad/execution/ibkr_adapter.py:96 _call_with_timeout` routes every broker
   call through `chad/execution/broker_executor.py`, a
   `ThreadPoolExecutor(max_workers=4)` whose worker-initializer installs **one
   persistent event loop per worker thread** (`broker_executor.py:88-89`). So a
   call like `ib.qualifyContracts(...)` (adapter `_qualify_if_possible`,
   `ibkr_adapter.py:3006` / `2453 whatIfOrder` / `2519 _place_and_wait` /
   `2848 openTrades`) runs on a **worker's** loop, not the connection's loop.

4. **The response never arrives on the caller's loop.**
   ib_async's sync surface resolves the loop via `util.getLoop()`
   (`ib_async/util.py:484-523`), which returns the **calling thread's** loop and
   is explicitly **not cached**. `qualifyContracts` →
   `qualifyContractsAsync` (`ib.py:2110`) → `reqContractDetailsAsync`
   (`ib.py:2124-2126`, **the killer**) awaits a contract-details response. That
   response is delivered by the socket reader — which lives on MainThread's
   **un-pumped** loop. The worker's loop drives the request but never sees the
   reply → the await never completes.

5. **Uninterruptible + pool saturation.**
   The worker is stuck inside `util.run(...).run_until_complete(...)` on a future
   that can never resolve. `broker_executor`'s `future.result(timeout)` returns
   control to the caller (fail-fast), but the **worker thread stays pinned**.
   Four such hangs → all four pool workers dead → every subsequent broker call
   is starved (`BUG_A_POOL_SATURATED`) → total broker I/O deadlock.

`isConnected()` stays `True` throughout (the socket is open; only the reader is
starved), so liveness checks do not detect it — a silent hang.

---

## 2. Chosen design (Option A — connection-owner loop)

A single **dedicated connection-owner event-loop thread** owns the IB object end
to end. The reader runs continuously (`loop.run_forever()`), so responses are
always processed. Everything else marshals onto that loop.

1. **`chad/execution/broker_loop.py`** — one daemon thread created at adapter
   init, running its own asyncio loop via `run_forever()`. The IB instance is
   created and `connectAsync`'d **on that loop**. Nothing else touches the IB
   object directly.
2. **All broker calls become coroutines** submitted via
   `asyncio.run_coroutine_threadsafe(coro, owner_loop)`. Callers block on the
   returned `concurrent.futures.Future` with a **bounded timeout**; on timeout
   the coroutine is **cancelled** (`fut.cancel()` propagates to the owner loop).
   No call can ever be uninterruptible again.
3. **`broker_executor.py`** retires the per-worker persistent loops. It becomes
   a bounded-concurrency admission gate (Semaphore, max 4 in-flight) in front of
   the owner loop. `BUG_A_POOL_SATURATED` now means "4 in-flight calls" (not "4
   dead workers"); `started=` telemetry still distinguishes admission-wait
   (saturation) from a slow admitted call. Thread reuse preserves the Bug-A
   fix (loops are lazily created once per reused thread by `util.getLoop`, never
   per call).
4. **Bounded timeouts on every broker coroutine** — `qualifyContracts`
   (the killer), the open-trades snapshot, `whatIfOrder`, and `placeOrder`
   (existing `_place_and_wait` semantics preserved).
5. **Reader-progress watchdog on the owner loop** — tracks `last_read_ts` from
   the reader path (`ib.client.conn.hasData` / `ib.updateEvent`). If calls are
   pending and the reader makes **no progress** for N seconds, the connection is
   marked **DEAD**, force disconnect+reconnect happens on the owner loop, and
   marker `BROKER_READER_STALLED` is emitted. `isConnected()` alone is never
   trusted.
6. **`ensure_connected()` reroutes through the owner loop** — the sync connect
   on MainThread is removed for the owner-loop-homed path.
7. **Fail-closed** — if the owner thread/loop dies, every subsequent call fails
   fast with marker `BROKER_LOOP_DOWN`. No silent hang, no auto trade-through.

### Backward-compatibility bridge (why the suite stays green)

The existing test suite injects **sync-only fake IBs** (`IBLike` protocol —
`isConnected/connect/qualifyContracts/whatIfOrder/placeOrder/openTrades/sleep`,
no `*Async` twins). The adapter therefore routes to the owner loop **only** when
the IB exposes async twins *and* the adapter itself homed the connection on the
owner loop (`_owner_loop_homed`). Sync fakes and any already-connected injected
IB fall back to the pre-existing bounded-executor path unchanged. This makes the
migration inert for today's live injection (see §5) and fully exercised by new
async-capable fake-IB tests.

---

## 3. Rejected alternative — per-worker connections

Give each broker worker its **own** IB connection (its own clientId), so the
call and its connection share a loop.

Rejected because:
- **N× clientIds / N× sockets** against IB Gateway — Gateway churn and pacing
  violations are already an open reliability item
  (`ops/pending_actions/IBKR_RELIABILITY_socket_backpressure_and_gateway_churn_2026-05-27.md`).
- **Broker-truth fragmentation** — open-order / position / fill state would be
  split across connections; the open-order guard (`_enforce_open_order_guard`)
  and idempotency reconciliation assume a single authoritative view. Per the
  saved methodology, `reqAllOpenOrders` is clientId-scoped nuance-laden already.
- **No shared order stream** — `placeOrder` returns a `Trade` whose status
  events arrive on its connection's loop; scattering connections scatters the
  event bus the idempotency/status handlers subscribe to.
- **Cost without benefit** — a single owner loop with a continuously-pumped
  reader already removes the deadlock; multiplying connections adds failure
  surface for no correctness gain.

The single connection-owner loop keeps one authoritative connection and one
event stream, which every downstream guard already assumes.

---

## 4. Rollout plan

Repo-only now (this PA). Activation is a **separate, gated** operator step.

1. **U0** — this PA. `git add` explicit path; commit `L1-CLD U0`.
2. **U1** — `broker_loop.py`: owner thread + `submit_coro` + lifecycle +
   `BROKER_LOOP_DOWN`; unit tests (cancel-on-timeout). Commit.
3. **U2** — reader-progress watchdog + forced reconnect + `BROKER_READER_STALLED`;
   tests (stall triggers reconnect; healthy reader does not). Commit.
4. **U3** — `ibkr_adapter.py` migration: connection creation + `ensure_connected`
   + `qualifyContracts` + snapshots + `placeOrder` through `submit_coro` with
   bounded timeouts; async-capable fake-IB tests. Commit.
5. **U4** — `broker_executor.py`: remove per-worker loops; semaphore admission
   model; preserve markers + telemetry; regression tests (5th concurrent call
   saturates; a timed-out call frees its slot). Commit.
6. **U5** — integration test with a scripted fake IB (no network): hang-inject
   contract-details → timeout + slot recovery + watchdog reconnect → subsequent
   call succeeds. Commit.
7. **U6** — full suite; fix fallout; final commit + push `origin/main`.

### Activation — WIRED AT U7 (2026-07-09)

**Status flip:** the migration is **no longer dormant** — U7 wired the
activation as the **production default** (repo-only; see the caveat below). U0–U6
built the owner-loop machinery; U7 lets `live_loop.py` hand the execution
adapter its OWN connection so the machinery actually engages.

What U7 changed (all repo-only — **no deploy, no restart**):

1. **Dedicated execution clientId — from config.** New registry constant
   `chad/execution/ibkr_client_ids.py::EXECUTION = 9007` (the **config key**),
   overridable at runtime via env **`CHAD_EXECUTION_CLIENT_ID`** (safe default =
   the registry constant). `9007` is collision-free by construction:
   `assert_no_collisions()` covers it, it differs from `LIVE_LOOP` (99, the
   shared MainThread `ib`), and it was absent from the 2026-07-08 IB Gateway
   client census (observed 80, 99, 9001, 9003, 9021, 9035). It is a NEW slot, so
   the shared `ib` (position_sync + market-data) is untouched.

2. **params-mode wiring in `live_loop.py`.** `_paper_adapter` is now built with
   `ib_factory=lambda: IB()` (an UNCONNECTED IB) and `IbkrConfig(client_id=
   _execution_client_id())`. The adapter's `ensure_connected()` then homes it on
   the broker owner loop via `connectAsync` (the U3 machinery) — the MainThread
   sync `connect()` is never used for the execution path. Legacy adoption remains
   available for instant rollback via **`CHAD_EXECUTION_OWN_CONNECTION=0`** (no
   code revert needed); sync-fake tests keep the adoption path unchanged.

3. **Boot fail-closed (`_home_execution_connection`).** `run_loop()` homes the
   execution connection before the first cycle. On ANY failure it sets
   `_EXECUTION_DISABLED=True`, emits the **`BROKER_LOOP_DOWN`** marker, and the
   `run_once` submit path skips execution (`SUBMIT_SKIPPED_NO_EXECUTION`) — it
   never falls back to a MainThread connect.

4. **Reconnect parity.** The reader-progress watchdog's forced reconnect
   (`ibkr_adapter.py::_wire_reader_watchdog._reconnect`) reconnects with
   `self._config.client_id` = the SAME dedicated `9007`, via `connectAsync` on
   the owner loop. No path re-introduces `util.getLoop()` / a sync connect on a
   foreign thread.

**CAVEAT — activation trigger:** the wiring is committed but **NOT deployed**.
It takes effect only at the **next `chad-live-loop` restart** (operator GO).
Treat that restart as the gated activation. Until then production runs the
already-loaded (pre-U7) code and behavior is unchanged. Post-restart proof
markers: `EXECUTION_OWNER_LOOP_ENGAGED clientId=9007`, a `chad-broker-loop`
thread, and IB Gateway showing a client `9007` connection.

---

## 5. Rollback plan

All units are additive or self-contained. The landed L1-CLD commit range
(oldest→newest) is:

```
8472862  U0  PA doc
725ed71  U1  broker_loop.py (owner loop + submit_coro)
06b30f7  U2  reader-progress watchdog
f7b1a3f  U3  ibkr_adapter owner-loop migration
2f431a3  U4  broker_executor semaphore admission gate
a28e237  U5  cross-loop deadlock recovery integration test
36ed7bc  U6  full-suite verification record + rollback range
<U7>     U7  activation — live_loop params-mode + EXECUTION clientId (this commit)
```

**Instant rollback without a code revert (preferred, since U7 is repo-only):**
because U7's activation is gated behind an env switch, an operator who has not
yet restarted `chad-live-loop` needs to do nothing; one who wants to keep the
code but run legacy behavior sets `CHAD_EXECUTION_OWN_CONNECTION=0` (the adapter
reverts to adopting the shared MainThread `ib`, owner-loop routing OFF).

To revert the entire change set in git:

```
git revert --no-edit 8472862^..HEAD           # revert the whole L1-CLD range incl. U7
# or hard-reset to the pre-L1-CLD baseline:
git reset --hard 3c80d40
```

`broker_loop.py` is a new module; reverting the adapter/executor/live_loop edits
plus deleting it restores the `3c80d40` baseline. No runtime state, no
`config/` file, no systemd unit is touched, so rollback is code-only and needs
no restart. **Until the next `chad-live-loop` restart the U7 wiring is inert in
production** (the running process holds the pre-U7 code), so this remains a
NOT_DEPLOYED change awaiting operator GO.

---

## 6. Verification criteria

- `python3 -m py_compile` clean on every changed file.
- New unit tests per unit pass (cancel-on-timeout; watchdog reconnect vs healthy;
  saturation marker; **timed-out call frees its slot** — the regression test).
- U5 integration test: injected contract-details hang → bounded timeout + slot
  recovery + watchdog reconnect → a subsequent call **succeeds** (this test
  failing == the old bug).
- Full suite (`pytest chad/tests/ -q`) green except pre-existing/date-driven reds,
  each explained.
- `CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview` runs clean.
- No systemctl / runtime/ / config / systemd changes. Deploy status:
  **NOT_DEPLOYED — awaiting operator GO.**

---

## 7. Markers

- **Preserved:** `BUG_A_POOL_SATURATED` (re-scoped to "4 in-flight", not "4 dead
  workers"; `started=` telemetry retained).
- **New:** `BROKER_LOOP_DOWN` (owner loop/thread dead → fail fast),
  `BROKER_READER_STALLED` (reader made no progress while calls pending → forced
  reconnect).
