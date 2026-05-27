# PAPER EPOCH 3 — 5-SESSION SOAK TRACKER

**Status:** Pending Action (paper / governance — declarative).
**Created:** 2026-05-27T13:33Z
**Author:** Team CHAD (operator: indiemusicteam7@gmail.com)
**Tracker type:** Per-session evidence ledger for the Paper Epoch 3 5-session clean-soak window.

---

## 1. Linked Epoch declaration

- Parent declaration: `ops/pending_actions/PAPER_EPOCH_3_START_2026-05-27.md`
- Day-0 dirty record: §12 of the parent declaration (2026-05-27, 7h 11m stop_bus halt)
- Day-0 commit: `02ccda7` — "Paper Epoch 3: log dirty day zero latency halt"
- Clean-soak count **before this tracker**: **0 / 5**
- Clean-soak count **at tracker creation**: **0 / 5**

This tracker does **not** retroactively count Day-0. Day-0 (2026-05-27) is permanently classified DIRTY / NOT COUNTABLE.

## 2. Session 1 candidate

| Field | Value |
|---|---|
| Candidate date | **2026-05-28** |
| Candidate start (UTC) | **2026-05-28T00:00:00Z** |
| Candidate end (UTC) | 2026-05-28T23:59:59Z (US-equity calendar session, evaluated end-to-end) |
| Candidate type | **First post-Day-0 clean-soak candidate** |
| Status at tracker creation | **PENDING** |
| Verdict locked | **No** — session evidence not yet collected |

Session 1 will not be marked PASS until §3 criteria are met across the full candidate window and the §5 evidence fields are populated.

## 3. Per-session pass criteria

A session is marked **PASS** only if **every** criterion below holds (or is explicitly justified with linked evidence):

1. `ready_for_live = false` throughout the entire session.
2. `allow_ibkr_live = false` throughout the entire session.
3. `allow_ibkr_paper = true` throughout the entire session.
4. **stop_bus halt time** total active duration **< 1h / day** (soft budget) unless explicitly justified.
5. **No sustained stop_bus active window** — any trigger must auto-recover via the live-loop's own `auto_recovery:broker_latency_clean_streak=5` path; operator-authorized service restarts during the session disqualify it.
6. **SCR remains CONFIDENT** (or any transition is explained with linked publisher / runtime evidence — e.g. transient `scr_sync_error:TimeoutError` self-resolve is acceptable; a real performance-driven downgrade is not).
7. **Reconciliation remains GREEN** (`reconciliation_state.status=GREEN`, `drifts=[]`), or any drift is resolved and explained.
8. **position_guard_drift.drift_count = 0**, or each drift is resolved (via `scripts/close_guard_entry.py` under the §4 GAP-028 permissive policy) and explained.
9. **trade_lifecycle_state.backlog_flag = false**, or backlog self-clears within one cycle with explanation.
10. **No public `fill_price = 100.0` fingerprint** in any `data/fills/FILLS_*.ndjson` row written during the session (PO-03 success criterion).
11. **No trusted fake placeholder evidence** — any placeholder row must continue to carry `status=rejected` + `pnl_untrusted=true` + tags `placeholder, broker_rejected` (P0-1 defense-in-depth must remain intact at trade_closer + paper_exec_evidence_writer + paper_trade_executor).
12. **Options chains / Greeks healthy** (`options_greeks.json::status=ok` within TTL; `options_chains_cache.json` refreshed or honestly classified as degraded with no impact on routed strategies).
13. **regime_state and strategy_health refresh within TTL** except during justified stop_bus windows (no silent staleness >2× TTL outside halt cover).
14. **Live-loop cycles resume after any transient halt** (no stuck/skipping cycles after stop_bus clears).
15. **All 17/18 strategy classifications remain known** (no new `UNKNOWN — REQUIRES AUDIT` entries; reclassifications during the session must carry explicit evidence).
16. **No live enablement** — no flip of `ready_for_live` or `allow_ibkr_live`; no `chad_mode` change away from `paper`.
17. **No operator service restarts** during the session window (a `chad-ibgateway.service`, `chad-live-loop`, or any `chad-*` restart disqualifies the day as NOT COUNTABLE).
18. **No idempotency clearing** — no `ibkr_adapter_state.sqlite3` operator mutation.
19. **No signal_guard clearing** — no `runtime/signal_guard.json` operator edit.
20. **No position_guard manual closes** except via `scripts/close_guard_entry.py` with fail-closed gates honoured.

## 4. Tracker baseline (at creation, 2026-05-27T13:33Z)

| Gate | Observed | Source |
|---|---|---|
| HEAD | `02ccda7` (Paper Epoch 3: log dirty day zero latency halt) | `git rev-parse --short HEAD` |
| ready_for_live | **false** | `runtime/live_readiness.json` (mtime ≈83s) |
| allow_ibkr_live | **false** | `runtime/decision_trace_heartbeat.json` (mtime ≈24s) |
| allow_ibkr_paper | **true** | `runtime/decision_trace_heartbeat.json` |
| chad_mode | paper, live_enabled=false | `runtime/decision_trace_heartbeat.json` |
| stop_bus.active | **false** (last cleared by `auto_recovery:broker_latency_clean_streak=5` at 2026-05-27T13:12:51Z) | `runtime/stop_bus.json` (mtime ≈22.6 min) |
| reconciliation_state.status | **GREEN**, drifts=[] | `runtime/reconciliation_state.json` |
| position_guard_drift.drift_count | **0** | `runtime/position_guard_drift.json` |
| positions_truth.broker_authority_status | **GREEN**, truth_ok=true | `runtime/positions_truth.json` |
| positions_truth.replay_diagnostic_status | PARTIAL (visibility-only per PR-09; replay_diagnostic_blocks_truth=false) | `runtime/positions_truth.json` |
| trade_lifecycle_state.backlog_flag | **false** | `runtime/trade_lifecycle_state.json` |
| scr_state | **CONFIDENT**, sizing_factor=1.0, paper_only=false | `runtime/scr_state.json` |
| regime_state | trending_bull (fresh, ts 13:33:50Z) | `runtime/regime_state.json` |
| strategy_health | fresh (ts 13:34:35Z) | `runtime/strategy_health.json` |
| options_chains_cache | stale (mtime ≈62 min; chains_count=0) — observation-only at tracker creation | `runtime/options_chains_cache.json` |
| options_greeks | status=ok (ts 2026-05-26T23:49Z; TTL=90000s = 25h, still within TTL) | `runtime/options_greeks.json` |
| ibkr_bars_cache | fresh (ts 13:35:08Z, 34 symbols) | `runtime/ibkr_bars_cache.json` |
| Public `fill_price=100.0` since Epoch 3 declaration | **0** rows | `data/fills/FILLS_*.ndjson` scan |
| Trusted fake placeholder rows since Epoch 3 declaration | **0** rows | `data/fills/FILLS_*.ndjson` scan |
| Placeholder-tagged rows since Epoch 3 declaration | 3 rows (all `status=rejected` + `pnl_untrusted=true`, P0-1 defense-in-depth catches) | `data/fills/FILLS_*.ndjson` scan |
| Pytest | 2538 passed, **1 failed** (`test_phase_a_item4_setup_tagging.py::test_02_alpha_intraday_momentum_surge_setup_family`), 8 warnings, 99.78s | `python3 -m pytest chad/tests/ -q` |

### Tracker-creation flutter observations (precondition baseline, NOT Session 1 evidence)

Two stop_bus flutter events in the 60 min before tracker creation:
- **12:34:04Z → 12:47:05Z** (~13m 1s active), broker_latency avg ~8220ms, auto-cleared via `clean_streak=5`.
- **13:07:50Z → 13:12:51Z** (~5m 1s active), broker_latency avg ~3217ms, auto-cleared via `clean_streak=5`.

These occurred **before** Session 1's candidate window opens (2026-05-28T00:00:00Z) and are recorded here as precondition context only. They do **not** disqualify Session 1.

### Baseline observation items to monitor into Session 1

- **Pytest regression** (1 failure: alpha_intraday momentum-surge setup family) — record as PARTIAL baseline observation; if the failure persists through Session 1 close, it is a separate Pending Action, not a session disqualifier.
- **options_chains_cache staleness** (~62 min, 0 chains) — monitor for self-refresh; degrade verdict to "options chain degraded" if not refreshed within Session 1 window AND options strategies are routed.
- **IBKR latency residual flutter** — connect_ms recently observed up to 1588ms / 3217ms / 8220ms; continue to monitor §3.4 (halt budget) and §3.5 (auto-recovery only).

## 5. Session 1 evidence fields (to be filled at session close)

The following fields will be populated at 2026-05-28T23:59:59Z (or earlier if a disqualifying event closes the session). Until then, each field is `[PENDING]`.

| Field | Value |
|---|---|
| Session window (UTC start) | `[PENDING]` |
| Session window (UTC end) | `[PENDING]` |
| Cumulative stop_bus halt time | `[PENDING]` |
| Number of stop_bus trigger events | `[PENDING]` |
| Largest single halt window | `[PENDING]` |
| Auto-recovery only? (no operator restart) | `[PENDING]` |
| IBKR latency summary (min / median / p95 / max connect_ms) | `[PENDING]` |
| Dangerous-classification cycle fraction (target <10%) | `[PENDING]` |
| SCR transitions (CONFIDENT / CAUTIOUS / UNKNOWN counts) | `[PENDING]` |
| SCR publisher timeout events | `[PENDING]` |
| regime_state freshness violations | `[PENDING]` |
| strategy_health freshness violations | `[PENDING]` |
| reconciliation_state status transitions | `[PENDING]` |
| position_guard_drift events (count, resolved/unresolved) | `[PENDING]` |
| trade_lifecycle backlog events (count, self-clear/escalated) | `[PENDING]` |
| Real broker-confirmed fills (count, per strategy) | `[PENDING]` |
| Placeholder fingerprint counts (public 100.0 / trusted fake / placeholder-tagged) | `[PENDING]` |
| Options chain / Greeks status (windowed) | `[PENDING]` |
| Strategy classification changes (with evidence link) | `[PENDING]` |
| Carryover open-order activity (M6E 5137, AAPL 5142, MSFT 5144, SPY 5146) | `[PENDING]` |
| Live enablement attempted / executed? | `[PENDING]` |
| Idempotency / signal_guard / position_guard manual mutations? | `[PENDING]` |
| Service restarts during session? | `[PENDING]` |
| **Final verdict** | `[PENDING]` — **PASS / FAIL / NOT COUNTABLE** |

## 6. Day-0 carryover note

Day-0 (2026-05-27) is **DIRTY / NOT COUNTABLE** per §12 of the parent Epoch 3 declaration. Reasons:

- stop_bus active 2026-05-27T04:24:23Z → 2026-05-27T11:35:36Z (7h 11m)
- Root cause: IB Gateway connect latency / socket backpressure deadlock
- Recovery: operator-authorized `chad-ibgateway.service` restart (disqualifying under §3.5 and §3.17)
- stop_bus cleared naturally via `auto_recovery:broker_latency_clean_streak=5` after Gateway restart
- No live enablement, no broker orders, no cancellations during the halt

Clean-soak count remains **0 / 5** at tracker creation. Day-0 carries the deferred hardening note `IBKR-RELIABILITY: investigate socket draining / Gateway backpressure under stop_bus` (parent §13).

## 7. Tracker scope and non-live statement

This tracker:

- **Does not** authorize live trading.
- **Does not** flip `ready_for_live` to true.
- **Does not** flip `allow_ibkr_live` to true.
- **Does not** mutate `runtime/epoch_state.json`.
- **Does not** modify any runtime JSON, configuration, service file, or broker state.
- **Does not** stage or commit any code or runtime change.

It exists solely as the per-session evidence ledger for the Paper Epoch 3 5-session clean-soak proof window. Operator GO remains separately required for any future live transition.

## 8. Future session rows

Sessions 2–5 will be appended to this tracker as additional `## 5b. Session N evidence` blocks following the same template, each preceded by its own per-session pass-criteria checklist (same as §3, frozen at session start). The tracker is not closed until either:

- 5 sessions are recorded as PASS (advances proof toward 95%+), or
- A FAIL / NOT COUNTABLE verdict is recorded, in which case the clean-soak counter is reset to 0 and a fresh tracker is started (or this tracker continues with a re-baselined Session 1 candidate at the next clean window).

## 9. Next action

Open Session 1 at **2026-05-28T00:00:00Z** if the runtime gates remain clean at that moment. The first session-evaluation pass will run after the session closes on 2026-05-28T23:59:59Z; until then, this tracker remains PENDING.

---

**Declared by:** Team CHAD
**Declared at:** 2026-05-27T13:33Z
**Effective:** at 2026-05-28T00:00:00Z (Session 1 candidate window opens)
**Status label:** VERIFIED (tracker creation); Session 1 status PENDING until evidence is collected.
