# PAPER EPOCH 3 — FULL-PRODUCTION PAPER MODE SOAK START

**Status:** Pending Action (paper / governance — declarative).
**Date:** 2026-05-27
**Author:** Team CHAD (operator: indiemusicteam7@gmail.com)
**Companion artifacts:**

- `reports/parity_audit/CHAD_FULL_PRODUCTION_PAPER_COMPLETION_REGISTER_20260527T001309Z.md` — last committed register (build ≈97%, proof ≈70%, 17/17 strategies classified).
- `reports/parity_audit/PAPER_FILL_PRODUCTION_BLOCKER_AUDIT_20260527T020000Z.md` — blocker classification (M6E `REAL_BROKER_ORDER_ACTIVE`, M2K `EXPECTED_SAFETY`, MYM `INVALID_QUANTITY_EXPECTED`, regime-gated 12/cycle, 4 strategies halted).
- `reports/parity_audit/M6E_ORDER_5137_BROKER_TRUTH_PROBE_20260527T024500Z.md` — broker truth probe verifying order `5137` is genuinely open at IBKR Paper.

This Pending Action declares Paper Epoch 3 as the canonical post-repair Full-Production Paper Mode validation window. It does not authorize live trading and does not mutate runtime config.

---

## 1. Definition

**Paper Epoch 3** is the post-repair CHAD Full-Production Paper Mode validation window. It begins a clean forward-looking proof period while preserving the historical Epoch 2 record in place. Epoch 3 is the scoreboard from which paper-complete proof advances toward 95%+.

This document is the **source-of-truth declaration** for Paper Epoch 3's start. The runtime/epoch_state.json flip from `CHAD_v8.9_Paper_Epoch_2 → CHAD_v8.9_Paper_Epoch_3` (and any associated archive of `runtime/` state into `runtime/archive/epoch_2_pre_20260527*/`) is a **separate operator-controlled action** to be raised as its own Pending Action when desired — production code reads `runtime/epoch_state.json` and changing it materially alters per-strategy-loss-guard, trade-stats-engine, dashboard, and exterminator labelling.

## 2. Start reason

Epoch 2 contains dirty pre-repair history that must not contaminate forward proof:

- **Placeholder-evidence era** — fill_price=100.0 placeholders previously written to FILLS_*.ndjson before P0-1 / PO-03 hardening.
- **Stale M2K/MYM bar era** — before PR-M2K-MYM mapped micro-futures in the IBKR bar provider.
- **IBKR latency halt era** — extended stop_bus windows before PR-04 / PR-09 and watchdog tuning.
- **Options-chain degraded period** — before PR-04 recovered the chains cache and Greeks publisher to `status=ok`.
- **Strategy unknown period** — before the 17/17 classification reached `UNKNOWN=0`.
- **Reconciliation / truth contract ambiguity** — before PR-09 separated `broker_authority_status` from `replay_diagnostic_status`.

Epoch 3 does **not** erase any of the above. It creates a new measurement boundary for forward evaluation; Epoch 2 records remain on disk and continue to be readable.

## 3. Baseline evidence (as of declaration)

| Field | Value | Source |
|---|---|---|
| Declaration UTC | 2026-05-27T03:27Z | this document |
| Baseline HEAD | `617fcdb` (Pending Action: PR-12 paper mode closeout archive) | `git rev-parse --short HEAD` |
| Pytest | **2539 passed**, 8 warnings, 120.25s | `python3 -m pytest chad/tests/ -q` |
| ready_for_live | **False** | runtime/live_readiness.json |
| allow_ibkr_live | **False** | runtime/decision_trace_heartbeat.json |
| allow_ibkr_paper | **True** | runtime/decision_trace_heartbeat.json |
| chad_mode | paper, live_enabled=false | runtime/decision_trace_heartbeat.json |
| stop_bus.active | **False** (last cleared 2026-05-27T03:20:36Z via `auto_recovery:broker_latency_clean_streak=5`) | runtime/stop_bus.json |
| reconciliation_state.status | **GREEN** (drifts=[]) | runtime/reconciliation_state.json |
| position_guard_drift.drift_count | **0** | runtime/position_guard_drift.json |
| positions_truth.broker_authority_status | **GREEN**, truth_ok=true, truth_source=BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER | runtime/positions_truth.json |
| positions_truth.replay_diagnostic_status | PARTIAL (visibility-only per PR-09; replay_diagnostic_blocks_truth=false) | runtime/positions_truth.json |
| trade_lifecycle_state.backlog_flag | **False** | runtime/trade_lifecycle_state.json |
| scr_state | **CONFIDENT**, sizing_factor=1.0, paper_only=false (effective_trades=196, win_rate=0.755, sharpe_like=5.73, total_pnl=$9,905.13, max_drawdown=−$290) | runtime/scr_state.json |
| tier_state | SCALE | runtime/tier_state.json |
| IB connect latency | ~329ms (well under 750ms dangerous threshold; rolling avg recovered from earlier flutter) | runtime/ibkr_status.json |

Pre-declaration latency flutter: stop_bus triggered at 03:14:35Z (`avg_latency_ms=2514.8`) and auto-recovered at 03:20:36Z after a 5-cycle clean streak. SCR shadow timed out briefly at 03:13/03:18 and re-synced at 03:14/03:26. Both behaviours are tracked as Epoch 3 observation items below.

## 4. Strategy classification baseline (17/17)

Source: `reports/parity_audit/CHAD_FULL_PRODUCTION_PAPER_COMPLETION_REGISTER_20260527T001309Z.md`.

| Class | Count | Strategies |
|---|---|---|
| ACTIVE_PRODUCTIVE | 5 | alpha, alpha_futures, alpha_intraday, gamma_futures, omega_macro |
| ACTIVE_BLOCKED_BY_DATA | 1 | delta |
| REGIME_SILENT | 10 | alpha_intraday_micro, alpha_options, beta, beta_trend, delta_pairs, gamma, gamma_reversion, omega, omega_momentum_options, omega_vol |
| HALTED_EDGE_DECAY | 1 | alpha_crypto |
| **UNKNOWN — REQUIRES AUDIT** | **0** | — |
| **Total** | **17** | — |

All 17 strategies are classified with explicit evidence-backed reasoning. Zero UNKNOWN at Epoch 3 start.

## 5. Carryover open-order note (M6E order 5137)

Verified live at IBKR Paper at 2026-05-27T02:43Z (read-only probe, clientId=9208, `reqAllOpenOrdersAsync`, no orders placed or cancelled):

| Field | Value |
|---|---|
| account | DUK902770 |
| orderId | **5137** |
| permId | 1525012073 |
| clientId (placer) | 99 (chad-live-loop) |
| contract | M6E FUT (`localSymbol=M6EM6`, `tradingClass=M6E`, `conId=838628915`, `lastTradeDateOrContractMonth=20260615`, multiplier 12500) |
| exchange | CME |
| action | BUY |
| totalQuantity | 2.0 |
| orderType | LMT |
| lmtPrice | 1.1624 |
| tif | DAY |
| status | **Submitted**, filled=0, remaining=2.0, whyHeld=`` |
| Local adapter row | `ibkr_exec_state` row `c193d882…` matches 1-to-1 (status=Submitted, sym=M6E, side=BUY, qty=2.0) |

Classification: **REAL_BROKER_ORDER_ACTIVE.**

Implications for Epoch 3:

- `duplicate_blocked` on M6E per cycle is correct safety behaviour.
- **No idempotency cleanup required.**
- **No signal_guard cleanup required.**
- **Do not cancel** unless a separate operator-approved cancellation Pending Action is created. The `DAY` TIF will expire naturally at CME session close if not filled; current spot EUR/USD (~1.08–1.10) is far below the 1.1624 limit.
- While 5137 stays live, the omega_macro M6E BUY 2.0 intent will continue to be the only intent reaching the executor each cycle and will continue to record `duplicate_blocked` + `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS` + `COOLDOWN_NOT_REARMED_UNCONFIRMED_STATUS`. Three other open orders co-exist (`5142 AAPL BUY 21 @ 308.82 LMT DAY PreSubmitted`, `5144 MSFT BUY 3 @ 418.57 LMT DAY PreSubmitted`, `5146 SPY BUY 15 @ 745.64 LMT DAY PreSubmitted`); all match local adapter state and are tracked but not acted on by this declaration.

## 6. Broker-order audit methodology note

For any future broker-side open-order audit:

- **Use `reqAllOpenOrdersAsync()`** (cross-client). This returns open orders placed by **every** clientId on the account, including the live-loop's clientId=99.
- **Do not rely on `reqOpenOrdersAsync()`** for cross-client audits. That call only exposes orders bound to the current calling clientId; using it from a fresh probe clientId can produce a false-negative 0/0 result while real orders remain open. This was observed during the 2026-05-27 02:43Z probe (clientId=9207 → 0/0; clientId=9208 with `reqAllOpenOrdersAsync` → 4 open orders including the target M6E 5137).
- Choose a fresh non-live clientId not present in `chad/execution/ibkr_client_ids.client_id_map()`.
- Connect via ib_async, dump only, disconnect cleanly. Do not place, modify, or cancel orders.

This note must be preserved as a permanent methodology reference; an existing audit that returns 0/0 without cross-client coverage is not authoritative.

## 7. Epoch 3 observation requirements

Across **5 clean US-equity sessions**, the following must hold (or be honestly classified):

1. **No live enablement.** ready_for_live remains false; allow_ibkr_live remains false; allow_ibkr_paper remains true.
2. **stop_bus halt time.** Cumulative active time tracked per session; soft budget **< 1h/day** unless justified. Today's two intra-day flutter triggers (01:48–01:53Z and 03:14–03:20Z, both `broker_latency` rolling-avg >2000ms, both auto-recovered via `clean_streak=5`) count as the precondition baseline and will be the first datapoints of the flutter ledger.
3. **SCR stability.** SCR remains CONFIDENT/sizing_factor=1.0, or any downgrade is explained (e.g. publisher timeout vs. real performance change). Today's transient `scr_sync_error:TimeoutError` → UNKNOWN at 03:13/03:18 with rapid re-sync at 03:14/03:26 is the precondition baseline for the SCR-publisher flutter ledger.
4. **No public `fill_price=100.0` fingerprint** in any FILLS_*.ndjson (PO-03 success criterion).
5. **No trusted fake placeholder evidence.** Placeholder rows must continue to be flagged `status=rejected` + `pnl_untrusted=true` + tagged `placeholder, broker_rejected` (P0-1 defense-in-depth must remain intact).
6. **position_guard_drift.drift_count = 0**, or every drift is explained against the GAP-028 permissive rebuilder policy and resolved via `scripts/close_guard_entry.py` if appropriate.
7. **trade_lifecycle_state.backlog_flag = false**, or self-clears within one cycle.
8. **IBKR latency stays mostly below the 2000ms stop threshold**; flutter tracked and not sustained (target dangerous-classification fraction <10% of cycle ticks).
9. **Strategy classifications remain valid.** Any strategy that produces unexpected fills, halts, or silence is re-classified within the epoch with explicit evidence.
10. **Options chain / Greeks remain healthy** (status=ok) or are honestly classified as degraded.
11. **No idempotency clearing.** No `ibkr_adapter_state.sqlite3` mutation outside the live-loop's own writes.
12. **No signal_guard clearing.** No `runtime/signal_guard.json` operator edits.
13. **No position_guard closes** except via `scripts/close_guard_entry.py` with the fail-closed gates honoured.
14. **No live restarts.** No `systemctl restart` on live services without explicit GO.

## 8. Epoch 3 success definition

Paper-complete **proof** advances toward 95%+ only **after 5 clean sessions pass** under §7. A clean session is one where:

- All §7 requirements hold for that calendar US-equity session.
- Cumulative stop_bus halt time < 1h.
- ≥1 real broker-confirmed fill produced (per strategy that is classified ACTIVE_PRODUCTIVE and not currently encumbered by an upstream open-order or same-side condition), OR an explanation for zero fills that is consistent with the strategy's classification.
- No new unexplained drift, no new halted-strategy entries, no new untruth (broker_authority_status remains GREEN).

## 9. Explicit non-live statement

This document **does not** authorize live trading. It **does not** flip `ready_for_live` to true. It **does not** flip `allow_ibkr_live` to true. It **does not** mutate `runtime/epoch_state.json`. It **does not** alter risk caps, sizing factors, or strategy gating. It **does not** restart any service.

Operator GO is still required, separately and explicitly, for any future live transition. CHAD remains paper-only.

## 10. Carryover open-order watchpoints (informational only)

These rows are observed by this declaration and will be referenced when assessing the §7 / §8 conditions, but no action is authorized here:

| orderId | symbol | secType | side | qty | LMT | TIF | status |
|---|---|---|---|---|---|---|---|
| 5137 | M6E | FUT (20260615) | BUY | 2 | 1.1624 | DAY | Submitted |
| 5142 | AAPL | STK | BUY | 21 | 308.82 | DAY | PreSubmitted |
| 5144 | MSFT | STK | BUY | 3 | 418.57 | DAY | PreSubmitted |
| 5146 | SPY | STK | BUY | 15 | 745.64 | DAY | PreSubmitted |

All four match local `ibkr_adapter_state.sqlite3 / ibkr_exec_state` rows 1-to-1.

## 11. Next action

Start (re-base) the 5-session paper soak at the next clean US-equity open. No service restart required. No code or runtime mutation required. Each session's evidence will be amended to this Pending Action (or to a per-session sub-record) until 5 clean sessions are recorded.

If at any time during Epoch 3 the operator wishes to flip `runtime/epoch_state.json::active_epoch` from `CHAD_v8.9_Paper_Epoch_2` to `CHAD_v8.9_Paper_Epoch_3` (and to archive Epoch 2 state into `runtime/archive/epoch_2_pre_20260527*/`), that is a separate Pending Action requiring explicit operator approval — it is not authorized by this declaration because it would immediately re-label per_strategy_loss_guard, trade_stats_engine, dashboard, and exterminator records.

---

**Declared by:** Team CHAD
**Declared at:** 2026-05-27T03:27Z
**Effective:** at next clean US-equity open
**Status label:** VERIFIED (declaration); paper-soak status will be PARTIAL until 5 clean sessions are recorded.
