# PA — Soak Status-History Writer + companion audit logs (RED-window mechanical verifiability)

**Status:** PROPOSED — spec/doc only. NOT built, NOT committed (operator commits),
NOT deployed, NO runtime write. CHAD remains PAPER; `ready_for_live=false`,
`allow_ibkr_live=false`.
**Date:** 2026-06-20
**Type:** Design specification for a prerequisite observability writer. No production
code in this document.
**Author authority:** read-before-write audit recorded in memory
`soak_red_classifier_8_blocked`; unblocks the locked RED-window rule.

---

## 1. Purpose

The locked rule
`ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md`
(sha256 `800480a570e8b2e50d115b9a62d52d6b7022cd5dc0a4421a740528061ef60723`,
**unchanged** by this PA) lets a transient
`positions_truth.json::broker_authority_status=RED` window be classified
**EXPLAINED** only when **all eight** criteria are *mechanically verifiable from
artifacts written during the window*. A read-before-write audit found that six of
the eight criteria cannot be graded post-hoc today, because the state they depend on
is written **point-in-time and overwritten every cycle** — there is no durable,
timestamped history. The RED-window classifier was therefore stopped at the
build-gate (memory `soak_red_classifier_8_blocked`).

This PA specifies the prerequisite that unblocks it:

- **PRIMARY — a status-history writer:** an append-only, timestamped ndjson of all
  soak-relevant state, one row per cadence tick, so the classifier can read the
  *across-window* and *at-cycle* values after the fact.
- **COMPANION #2 — a signal-router emission log** (criterion 2).
- **COMPANION #3 — an ENTRY-intent audit log** (criterion 3).

It is a **specification only**. No writer is built, wired, or deployed here. The
evaluator/classifier itself remains a separate, later PA that must reference the
locked rule by sha256 (rule §8).

---

## 2. Criterion coverage — what each writer unblocks, and how it is graded

Window bounds `[window_open_utc, window_close_utc]` are taken from the
`broker_authority_status` transitions recorded by the **primary** writer (the first
row that reads `RED` opens the window; the first subsequent row that reads
non-`RED` closes it).

| # | Rule criterion (verbatim source) | Verifiable today? | Writer that unblocks it | Post-hoc grading procedure |
|---|---|---|---|---|
| 1 | Duration ≤ 15 min wall-clock | **Yes** | none needed | `window_close_utc − window_open_utc ≤ 900s`, computed from primary-writer rows. |
| 2 | Zero `RoutedSignal` emissions, from the signal_router audit log | No (no log exists) | **Companion #2** | Count emission rows with `ts_utc ∈ window`; criterion passes iff count == 0. |
| 3 | Zero `intent_type=ENTRY` items in the execution pipeline | No (no log exists) | **Companion #3** | Count audit rows with `intent_type=="ENTRY"` and `ts_utc ∈ window`; passes iff 0. |
| 4 | `dynamic_caps.json` `ts_utc` outside window **OR** per-strategy `sizing_factor` unchanged | No (overwritten) | **Primary** | All in-window rows share one `sizing_digest`, **and** every `dynamic_caps_ts_utc ∉ window` OR the digest is constant across the window. |
| 5 | `scr_state.json` state band unchanged across window | No (overwritten) | **Primary** | `scr_band` identical across all in-window rows. |
| 6 | `ts_utc` on `profit_lock_state.json`, `stop_bus.json`, `tier_state.json` all unchanged | No (overwritten) | **Primary** | `profit_lock_ts_utc`, `stop_bus_triggered_at`/`stop_bus_cleared_at`, `tier_ts_utc` constant across in-window rows; `stop_bus_active` stays `false`. (See §5 note on `stop_bus.json`.) |
| 7 | Zero new closed-trade rows in `data/trades/trade_history_<date>.ndjson` | **Yes** | none needed | Count append rows in the dated file with `ts_utc ∈ window`; passes iff 0. |
| 8 | Next reconciliation-publisher cycle returned `broker_authority_status=GREEN` | No (overwritten) | **Primary** | **De-dup by `reconciliation_ts_utc`** (the primary samples at 60s, so ~5 consecutive rows carry one 300s reconciliation snapshot): collapse all rows that share a `reconciliation_ts_utc` into one logical reconciliation cycle, order the *distinct* values ascending; the **next reconciliation cycle after the window** is the first distinct `reconciliation_ts_utc > window_close_utc`. Criterion passes iff that cycle's `reconciliation_status=="GREEN"` **and** the contemporaneous `broker_authority_status=="GREEN"`. See §4.1. |

Criteria **1 and 7 are already gradable** from durable artifacts and need no new
writer. The primary writer covers **4, 5, 6, 8**; the two companions cover **2 and 3**.

---

## 3. Output schemas

### 3.1 PRIMARY — status-history row (`soak_status_history.v1`)

One JSON object per line, append-only. Proposed path (dated rotation, matching the
`trade_history_<YYYYMMDD>.ndjson` convention):

```
data/soak/status_history_<YYYYMMDD>.ndjson      # e.g. status_history_20260620.ndjson
```

Row shape (`>=` these fields; field names are the spec contract):

```json
{
  "schema_version": "soak_status_history.v1",
  "ts_utc": "2026-06-20T13:30:36Z",            // writer sample time (row clock)
  "writer_identity": "chad.ops.soak.status_history_writer",
  "cadence_source": "lifecycle_truth_publisher",  // provenance of this tick — 60s native authority cadence (§4)

  "broker_authority_status": "GREEN",            // positions_truth.broker_authority_status
  "broker_authority_as_of_utc": "2026-06-20T13:30:00.333713Z", // positions_truth.as_of_event_ts_utc
  "broker_authority_publish_ts_utc": "2026-06-20T13:30:36.807927Z", // positions_truth.ts_utc
  "broker_authority_age_seconds": 0,             // row ts − positions_truth.ts_utc — ~0 by construction: row is emitted in the same 60s cycle that just authored positions_truth.json (§4.1)

  "reconciliation_status": "GREEN",              // reconciliation_state.status
  "reconciliation_ts_utc": "2026-06-20T13:30:00Z", // reconciliation_state.ts_utc

  "scr_band": "CAUTIOUS",                        // scr_state.state
  "scr_ts_utc": "2026-06-20T13:29:30Z",          // scr_state.ts_utc

  "sizing_digest": "sha256:<hex>",               // sha256 of canonical-JSON(dynamic_caps.strategy_caps)
  "dynamic_caps_ts_utc": "2026-06-20T13:29:58.740740Z", // dynamic_caps.ts_utc

  "profit_lock_mode": "NORMAL",                  // profit_lock_state.mode
  "profit_lock_sizing_factor": 1.0,              // profit_lock_state.sizing_factor (context)
  "profit_lock_ts_utc": "2026-06-20T13:34:46Z",  // profit_lock_state.ts_utc

  "stop_bus_active": false,                       // stop_bus.active
  "stop_bus_triggered_at": "",                    // stop_bus.triggered_at (file has NO ts_utc — §5)
  "stop_bus_cleared_at": "2026-05-28T18:38:41.794703+00:00", // stop_bus.cleared_at

  "tier_name": "PRO_GROWTH",                      // tier_state.tier_name (context)
  "tier_ts_utc": "2026-06-20T04:06:38.290073Z",   // tier_state.ts_utc

  "source_sha256": {                              // rule §5 evidence-citation requirement
    "positions_truth.json":      "sha256:<hex>",
    "reconciliation_state.json": "sha256:<hex>",
    "scr_state.json":            "sha256:<hex>",
    "dynamic_caps.json":         "sha256:<hex>",
    "profit_lock_state.json":    "sha256:<hex>",
    "stop_bus.json":             "sha256:<hex>",
    "tier_state.json":           "sha256:<hex>"
  },
  "read_errors": []                               // names of any source file that failed to read this tick
}
```

**`sizing_digest` (criterion 4):** sha256 of the canonical (sorted-key, compact)
JSON serialization of `dynamic_caps.json::strategy_caps` — the per-strategy dollar
caps (a 16-key dict, verified 2026-06-20). A stable digest across in-window rows
proves "per-strategy sizing_factor values are unchanged." `dynamic_caps_ts_utc`
independently satisfies the rule's OR-branch ("ts_utc lies outside the RED window").
Capturing both makes either branch gradable.

**Field-mapping note — source of truth per field (verified 2026-06-20):**

| Row field | Source file | Source key | Writer/cadence of source |
|---|---|---|---|
| `broker_authority_status`, `*_as_of_utc`, `*_publish_ts_utc` | `runtime/positions_truth.json` | `broker_authority_status`, `as_of_event_ts_utc`, `ts_utc` | `chad-lifecycle-truth-publisher` — **60s** |
| `reconciliation_status`, `reconciliation_ts_utc` | `runtime/reconciliation_state.json` | `status`, `ts_utc` | `chad-reconciliation-publisher` — **300s** |
| `scr_band`, `scr_ts_utc` | `runtime/scr_state.json` | `state`, `ts_utc` | SCR shadow (ttl 180s) |
| `sizing_digest`, `dynamic_caps_ts_utc` | `runtime/dynamic_caps.json` | `strategy_caps`, `ts_utc` | allocator publish |
| `profit_lock_mode/_sizing_factor/_ts_utc` | `runtime/profit_lock_state.json` | `mode`, `sizing_factor`, `ts_utc` | profit-lock publish |
| `stop_bus_active/_triggered_at/_cleared_at` | `runtime/stop_bus.json` | `active`, `triggered_at`, `cleared_at` | stop-bus (event-driven) |
| `tier_name`, `tier_ts_utc` | `runtime/tier_state.json` | `tier_name`, `ts_utc` | tier publish |

### 3.2 COMPANION #2 — signal-router emission log (`soak_signal_router.v1`)

Proposed path: `data/soak/signal_router_emissions_<YYYYMMDD>.ndjson`. One row per
emitted `RoutedSignal`:

```json
{"schema_version":"soak_signal_router.v1","ts_utc":"2026-06-20T13:30:36Z","symbol":"AAPL","side":"BUY","primary_strategy":"alpha","source_strategies":["alpha","beta"],"net_size":100.0}
```

Minimal contract for criterion 2 is `{ts_utc, symbol}`; `side`,
`primary_strategy`, `source_strategies`, `net_size` are included for forensic value
and cost nothing.

### 3.3 COMPANION #3 — ENTRY-intent audit log (`soak_entry_intent.v1`)

Proposed path: `data/soak/entry_intent_audit_<YYYYMMDD>.ndjson`. This is the rule's
named `execution_plan_audit.ndjson` "(or equivalent audit artifact)". One row per
intent that reaches the execution decision point:

```json
{"schema_version":"soak_entry_intent.v1","ts_utc":"2026-06-20T13:30:36Z","intent_type":"ENTRY","symbol":"AAPL","side":"BUY","strategy":"alpha","admitted":true}
```

`intent_type ∈ {ENTRY, EXIT, FLIP}`. Criterion 3 counts only
`intent_type=="ENTRY"`. `admitted` records whether the intent passed all guards and
reached broker submission (see §4.3).

---

## 4. Cadence + attach points (file:line)

### 4.1 PRIMARY writer — cadence decision

**Sample the window-defining field at its native rate.** The field that *defines*
the RED window — `positions_truth.json::broker_authority_status` (rule §4 scope
line) — is authored every **60s** by `chad-lifecycle-truth-publisher.timer`
(`OnUnitActiveSec=60`). It is computed at
`chad/ops/lifecycle_truth_publisher.py:592`
(`broker_authority_status = "GREEN" if truth_ok else "RED"`) and emitted into the
`positions_truth.json` payload at `:630`. `reconciliation_state.json::status` is a
separate, slower **300s** signal (`chad-reconciliation-publisher`). The two are
independent publishers on different clocks. To sample the window-defining field at
its native resolution, the primary history writer is attached to the **60s
lifecycle-truth publisher** (this revises the earlier 300s recommendation).

**Options considered:**

- **(C) Hook `lifecycle_truth_publisher.publish_once()` — 60s, no new timer.
  RECOMMENDED.** Samples `broker_authority_status` at the exact cadence it is
  authored, so every authority transition lasting ≥1 tick is observed and
  `broker_authority_age_seconds ≈ 0` by construction (the row reads the file the
  same cycle authored it). No new systemd timer (honors the GAP-032 lineage of not
  adding itself-corruptible timers). The earlier objection — "edits the
  safety-critical authority publisher" — is fully mitigated by **ordering +
  isolation** (see attach point below): the history write runs only *after* both
  authority artifacts are durably written, and is wrapped in a `try/except` that
  never raises, so it cannot corrupt, delay, or alter the publisher's primary job.
- **(A) Hook `reconciliation_publisher.main()` — 300s, no new timer.** Idiomatic
  (this file already hosts `_emit_position_guard_drift`,
  `chad/ops/reconciliation_publisher.py:244`/called `:293`, and the
  `systemd_wants_lint` refresh `:302`, as best-effort side-emitters for the in-code
  reason of *"detect … without adding a new (itself-corruptible) systemd timer"*,
  `:297-300`). But 300s **undersamples** the 60s authority field: a RED window
  shorter than one ~5-min reconciliation cycle is invisible, and even a captured
  window carries a `broker_authority` sample up to ~60s stale (read off a different
  publisher's clock). Retained as the documented **fallback** if the 60s hook
  proves unsafe.
- **(B) Dedicated writer + dedicated 60s timer** (`chad-soak-status-history.timer`).
  Pro: 60s resolution with a fully isolated failure domain. Con: adds a systemd
  timer the codebase deliberately avoids (GAP-032 lineage); more surface to
  maintain. Documented as the **Phase-2 upgrade** if the in-process 60s hook must
  later be decoupled from the publisher.

**Recommendation: Option C.** Native authority cadence, no new timer, and the
window-defining field is sampled at the rate it actually changes.

**Feasibility — all state needed for criteria 4/5/6/8 is readable at this attach
point (verified 2026-06-20):** `publish_once(*, repo_root, runtime_dir, data_dir)`
(`chad/ops/lifecycle_truth_publisher.py:689`) already holds `runtime_dir` and
`data_dir` in scope. The helper reads, fresh from `runtime_dir`, exactly as the
publisher already reads `reconciliation_state.json` at `:477`:

  - `positions_truth.json` — the file **just written at `:720`** (its
    `broker_authority_status`, `as_of_event_ts_utc`, and the injected `ts_utc` are
    all on disk; confirmed present 2026-06-20);
  - `reconciliation_state.json` — the latest 300s snapshot (`status` + `ts_utc`;
    confirmed present). **Read directly** — *not* via the embedded
    `evidence.reconciliation_status`, which is the replay-coverage classifier, not
    the raw upstream `reconciliation_state.json::status` that criterion 8 names;
  - `scr_state.json` (`state`, `ts_utc`), `dynamic_caps.json`
    (`strategy_caps`, `ts_utc`), `profit_lock_state.json`, `stop_bus.json`,
    `tier_state.json` — all confirmed present.

Of these seven, the lifecycle publisher itself writes only `positions_truth.json`;
the other six are independent latest snapshots at the 60s tick — exactly the
semantics Option A would have had, minus the staleness penalty. **No needed state is
unreachable at this hook.**

**Criterion 8 from 60s samples (verified).** `reconciliation_state.json` is
rewritten every cycle by `reconciliation_publisher._write()`
(`chad/ops/reconciliation_publisher.py:235-241`), which sets `payload["ts_utc"] =
_utc_now_iso()` at `:237` — so its `ts_utc` is a fresh wall-clock value that
**changes per real 300s cycle** (current on-disk value
`2026-06-20T21:44:12.233224Z`). At 60s sampling ~5 consecutive rows carry the same
`reconciliation_ts_utc`; the evaluator **de-duplicates by that value** to recover the
true 300s cycle boundaries (grading procedure in §2, criterion 8). The locked rule's
criterion 8 — "the next *reconciliation-publisher* cycle" — is therefore fully
satisfiable from 60s samples without changing the rule.

**Residual limitation (must be carried into the classifier PA).** 60s sampling
*removes* the Option-A "shorter than one ~5-min reconciliation cycle is NOT
COUNTABLE" gap for every window that spans at least one 60s tick (i.e. all windows
≥60s are now graded). The only residual is a RED window **shorter than the 60s tick
interval** that opens and closes entirely between two samples — it is still **not
captured** and therefore **NOT COUNTABLE** (the rule's safe default; it does not
become EXPLAINED). This residual is acceptable: it is far below the 15-min ceiling
(criterion 1 caps a countable window at 900s), and a sub-60s authority RED flap is
within the broker-latency flutter envelope (memory `broker_latency_flutter_pattern`),
not a durable RED state the rule is designed to grade. Every row still records the
authority sample's native `broker_authority_publish_ts_utc` and
`broker_authority_age_seconds` (≈0 here) so the evaluator can confirm it graded on a
fresh read.

**Attach point (Option C):** add a new best-effort helper `_emit_status_history(*,
repo_root, runtime_dir, data_dir, truth_path)`, structurally mirroring the
established side-emitter idiom
(`reconciliation_publisher._emit_position_guard_drift`,
`chad/ops/reconciliation_publisher.py:244`), and invoke it inside `publish_once()`
**immediately before the function returns — after the `LOG.info(...)` block at
`chad/ops/lifecycle_truth_publisher.py:722-723`, i.e. after both
`write_runtime_state_json(...)` calls complete at `:719-720`** — so both authority
artifacts are durably on disk before the row is built and the row reads the
just-written `positions_truth.json`. Wrap the entire call in a `try/except` that logs
and swallows (best-effort), exactly as `reconciliation_publisher.py:292-295`; a
history-write failure must never propagate into `publish_once()` or perturb the 60s
publisher. The helper also `mkdir(parents=True, exist_ok=True)`s `data_dir/"soak"`
(§5.5).

### 4.2 COMPANION #2 — signal-router emission log

**Attach point: inside `SignalRouter.route()` at the return,
`chad/utils/signal_router.py:210-211`** (`routed.sort(...)` / `return routed`).

**Correction to the brief.** The brief proposed `route_signals():215`. Verification
shows the live pipeline calls the **method** `self.router.route()`
(`chad/utils/pipeline.py:335`), **not** the `route_signals()` functional wrapper
(`signal_router.py:215`). The `route_signals` import at `pipeline.py:37` is unused on
the hot path. Logging in `route_signals()` would be **blind in production**. Logging
at the `route()` return (`:210-211`) captures **both** the live pipeline path and the
compat wrapper (which delegates to `route()`), so it is strictly more correct.
Iterate `routed` and append one emission row per `RoutedSignal` before returning.

### 4.3 COMPANION #3 — ENTRY-intent audit log

**Attach point: the per-intent execution decision tree in
`chad/core/live_loop.py:2306-2328`**, with the recommended write **just before the
broker submit at `:2328`** (`submitted = _paper_adapter.submit_strategy_trade_intents([intent])`).

`intent_type` is fully determinable at this point (verified 2026-06-20):

- **FLIP** — `is_flip_signal(intent)` true → `replace_position(intent)`
  (`live_loop.py:2306`, `:2314`).
- **EXIT** — `_is_exit_intent` true via explicit `side ∈ {EXIT, CLOSE}`
  (`live_loop.py:2108`).
- **ENTRY** — else branch → `mark_position_open(intent)` (`live_loop.py:2323`).

Same-side-duplicate intents are suppressed earlier (`is_same_side_open(intent)` →
`continue`, `live_loop.py:2088`) and are **correctly excluded** from ENTRY counts —
they were blocked as duplicates, not admitted as fresh entries. To also surface
ENTRY intents that were *present but suppressed downstream*, set `admitted=false` on
rows for intents that do not reach `:2328`; criterion 3 ("zero ENTRY *in the
pipeline*") is satisfied by `count(intent_type=="ENTRY") == 0` regardless of
`admitted`, so the field is informational. Minimal contract is
`{ts_utc, intent_type, symbol}`.

---

## 5. Hard constraints + fail-safety

1. **Append-only.** All three writers open in append mode and only ever append. They
   **never** open a source/state file for write, never truncate, never overwrite
   history. Dated rotation by UTC date in the filename; no in-place mutation.
2. **Secondary to live state — must never break the source.** Each writer is invoked
   inside a `try/except` that swallows and logs (best-effort), mirroring the existing
   `reconciliation_publisher.py:292-295` pattern. For the **primary** writer
   specifically, the call is placed *after* both `write_runtime_state_json(...)` calls
   (`chad/ops/lifecycle_truth_publisher.py:719-720`), so the lifecycle publisher's
   primary job — authoring `positions_truth.json`/`broker_authority_status` and
   `trade_lifecycle_state.json` — is already complete and durable before any history
   work begins; **ordering + the `try/except` together** guarantee a history-write
   failure (disk full, permission, serialization error) cannot corrupt, delay, or
   alter that authoring. The same isolation MUST hold for the signal router (§4.2) and
   the live loop (§4.3). On any per-source read failure the writer records the file
   name in `read_errors[]` and emits the row anyway (a partial row is better than no
   row).
3. **Read-only on all live state.** The writers READ
   `positions_truth.json`, `reconciliation_state.json`, `scr_state.json`,
   `dynamic_caps.json`, `profit_lock_state.json`, `stop_bus.json`, `tier_state.json`
   and the in-memory `RoutedSignal`/intent objects. They WRITE **only** under
   `data/soak/`. They MUST NOT mutate any live-gating state, `ready_for_live`,
   `allow_ibkr_live`, `epoch_state.json`, the position guard, trade-closer queues, or
   any source file.
4. **`stop_bus.json` has no `ts_utc` (verified).** Its keys are
   `active, triggered_at, cleared_at, triggered_by, cleared_by, reason,
   schema_version`. Criterion 6 says "ts_utc on … stop_bus.json"; the faithful proxy
   is the pair `triggered_at`/`cleared_at` plus `active`. The writer records all
   three; the classifier PA must read criterion 6 against these fields for stop_bus,
   not a non-existent `ts_utc`. (This nuance is flagged here so the classifier does
   not silently treat a missing key as "unchanged.")
5. **`data/soak/` does not yet exist** (verified) — the writer must
   `mkdir(parents=True, exist_ok=True)` on first run; that is the only directory it
   creates, and it is outside `runtime/`.
6. **Atomic-append discipline.** One `json.dumps(...) + "\n"` per row, single
   `write()` per row; no temp-file replace needed for append. No concurrent writer to
   the same file (the primary is single-cadence; companions write their own files).

---

## 6. Deployment / backfill consequence (read this before scheduling any soak)

**History begins at deployment. There is no retroactive history.** The classifier can
only grade a RED window for which the writer was **already live and appending** across
the window and the following reconciliation cycle. Therefore:

- **The soak clock cannot start until this writer is deployed and confirmed
  appending.** Any RED window before first append is **NOT COUNTABLE** (the rule's
  safe default) because criteria 4/5/6/8 are unprovable for it.
- A clean BOX-059 soak session that contains an ungraded (pre-writer) RED window is
  not a clean session; it must restart after the writer is live.
- First-row sanity must be confirmed (a row lands in
  `data/soak/status_history_<date>.ndjson` on the next 60s lifecycle-truth tick)
  **before** the operator declares Session 1 open.

This is a one-way prerequisite ordering: **writer live → soak clock starts →
classifier (separate PA) can grade.**

---

## 7. Proposed unit + verification plan

This PA writes **no code**. When the writer is later built under a separate
implementation PA, that PA should include:

**Unit tests (proposed):**
- `status_history_writer`: given seven fixture JSON files, emits one
  `soak_status_history.v1` row with every mapped field populated and a
  deterministic `sizing_digest` (same `strategy_caps` → same digest; one cap changed
  → different digest).
- Partial-read resilience: one unreadable source file → row still emitted, file named
  in `read_errors[]`, no exception escapes.
- Append-only: two invocations append two lines; line 1 is byte-identical afterward.
- Fail-safety: writer raising internally does NOT propagate out of
  `lifecycle_truth_publisher.publish_once()` (assert `publish_once()` returns normally
  and that `positions_truth.json` + `trade_lifecycle_state.json` are still written
  when the history write is forced to throw — the authority artifacts are produced at
  `:719-720`, before the history call).
- Companion #2: a 3-signal `route()` call appends exactly 3 emission rows; an empty
  route appends 0.
- Companion #3: ENTRY/EXIT/FLIP fixtures each produce the correct `intent_type`;
  same-side-suppressed intent produces no ENTRY row.

**Cross-condition replay test (proposed):** synthesize a 60s-cadence history with a
known 12-minute RED window (broker_authority RED→GREEN), stable sizing_digest/
scr_band/profit_lock/stop_bus/tier across it, zero companion rows in-window, and — to
exercise the §2 criterion-8 de-dup — several consecutive rows sharing one pre-window
`reconciliation_ts_utc` followed by a **new, distinct** `reconciliation_ts_utc` after
`window_close_utc` carrying `reconciliation_status=="GREEN"`; assert a stub classifier
grades all of 1,4,5,6,8 as PASS, that it collapses same-`reconciliation_ts_utc` rows
into one logical cycle, and that flipping any one field to "changed" flips that
criterion to FAIL.

**Live verification after deployment (operator):**
- `python3 -m py_compile` the changed files; full `pytest chad/tests/` green;
  `CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview` clean
  (CLAUDE.md verification sequence).
- Confirm a fresh row in `data/soak/status_history_<date>.ndjson` on the next 60s
  lifecycle-truth tick, with non-null `broker_authority_status`,
  `reconciliation_status`, `scr_band`, `sizing_digest`, and a populated
  `source_sha256` map.
- Confirm `positions_truth.json` (and `trade_lifecycle_state.json`) continue to update
  normally on the 60s cadence (writer did not perturb the lifecycle publisher's
  primary job).
- Confirm `data/soak/` is the only new write target; no `runtime/` file changed
  mtime as a result of the writer.

---

## 8. Explicit non-effects

This PA, and the writer it specifies:

- does **NOT** change, weaken, add, or remove any of the eight RED-window criteria,
  and does **NOT** alter the locked rule (sha256 `800480a5…` unchanged);
- does **NOT** flip `ready_for_live`, `allow_ibkr_live`, or `allow_ibkr_paper`;
- does **NOT** mutate `epoch_state.json`, the position guard, trade-closer queues,
  profit-lock, stop-bus, tier, SCR, or any live-gating state — it **reads** them;
- does **NOT** pass, fail, or count any soak session, and is **not** the classifier
  (that is a separate, later PA that must reference the locked rule by sha256);
- does **NOT** place, cancel, or modify any broker order;
- writes **only** under `data/soak/`; touches no `runtime/`, config, or systemd file.

CHAD remains PAPER. Live trading remains NOT authorized.

---

## 9. Constraints honored by this document

Doc/spec only. No production code, no git commit (operator commits), no service
restart, no runtime write. All attach points cited by `file:line` and verified
against the working tree on 2026-06-20. Two findings shape the design: (a) the
window-defining field `broker_authority_status` is authored at 60s
(`lifecycle_truth_publisher.py:592`/`:630`), so the primary writer is hooked into that
60s publisher's `publish_once()` (attach after the authority writes at `:719-720`,
Option C) rather than the 300s reconciliation publisher — sampling the field at its
native rate; recommendation, feasibility, criterion-8 de-dup, and the residual
sub-60s limitation are documented in §4.1; (b) the signal-router attach point
(`SignalRouter.route():210-211`, not `route_signals():215` — §4.2).
