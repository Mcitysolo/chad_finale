# PENDING ACTION — Equity round-trips do not close: forensic finding + minimal-risk fix

- **Filed:** 2026-07-13
- **Type:** Forensic finding + Pending Action. SHADOW-first fix BUILT + wired 2026-07-13 (see §6);
  **DEPLOYED IN SHADOW** (corrected 2026-07-18); ACTIVE flip remains PROPOSED.
- **Author:** EXIT-AUDIT (read-only forensic) + EXIT-OVERLAY (implementation)
- **Status:** SHADOW implementation landed repo-side and is **deployed in shadow** (default-safe;
  submits nothing — corrected 2026-07-18, was "not deployed"). The ACTIVE flip requires typed
  operator GO against the §4 pre-registered criteria.
- **Governance:** One change at a time · no direct config mutation · default-OFF, shadow-first · no
  flip without pre-registered written activation criteria (per dormant-census governance).

---

## FU1-B6 DOC CORRECTIONS (2026-07-18) — 5 stale claims fixed

`docs/ULTRA_CLOSE_AUDIT_2026-07-17.md` §F found five drifted claims in this PA. All five are
corrected inline below; recorded here for the audit trail (each re-verified against the live tree,
not merely copied from the audit):

1. **§6 "NOT DEPLOYED" was STALE.** The overlay **is deployed and running in SHADOW** — heartbeat
   fresh `runtime/exit_overlay_heartbeat.json` `mode=shadow ts=2026-07-18T01:55:20Z evaluated=7`.
   §6 header and deploy-status line corrected.
2. **§3 "honors `_EFFECTIVE_NON_CHAD_SYMBOLS`" was FALSE as written.** No exclusion term existed in
   the module; exclusions were honoured only accidentally at the `apply_close_intents` chokepoint.
   **FU1-B6 makes it TRUE** — `evaluate_positions` now skips excluded symbols in-module
   (`SKIP_EXCLUDED`), before the phantom guard. The doc claim now matches the code.
3. **§3 "default OFF" contradicted the shipped `config/position_exit_overlay.json` `mode: "shadow"`**
   (env unset → `resolve_mode` returns the config's `shadow`, not OFF). Corrected to "shipped
   default = shadow".
4. **§3 evidence path `data/exit_overlay_shadow/` was wrong** — the code writes `data/exit_overlay/`
   (`_DEFAULT_EVIDENCE_SUBDIR = ("data", "exit_overlay")`; §6 and the markers block already said so).
   Corrected.
5. **§6 hot-path wiring cited `live_loop.py:1933-1953`** — the actual block is
   `live_loop.py:2270-2278` (verified). Corrected.

The criteria-amendment items the audit also raised (C5 proxy ruling, C7 rollback procedure, the
WARMUP re-grade paradox) are handled separately in the FU1-B7 amendment PA, not here.

---

## 0. STOP-clause disposition — the hypothesis is REFUTED at the premise, CONFIRMED at the outcome

The finding-under-test hypothesised that "some or all equity strategies are entry-only" and that
"gamma.py greps show no exit/stop/take-profit defs." **Both premises are false.** Exits exist at two
layers and *did* fire historically:

- `gamma.py` contains **five** `side=SignalSide.SELL` branches (`gamma.py:257,322,338,354,372`):
  `no_history_exit_only`, `vol_spike_exit`, `trend_break_exit`, `time_stop_exit`, `atr_trail_exit`.
- **6 of 16** active heads carry a first-class EXPLICIT_EXIT path; **5 more** can emit the opposite
  side as a directional entry (see §Q2).
- In **Epoch 2** (Mar–May 2026) CHAD booked **~1,428 confirmed equity SELLs** — round-trips *did* close
  (`data/fills/FILLS_*.ndjson`: Mar 928 / Apr 275 / May 225 confirmed EQ SELLs).

Per the STOP instruction, this contradiction **is** the finding. The outcome (zero *confirmed* closes
this epoch) is real, but the cause is **not** "entry-only strategies." It is three compounding
mechanical faults documented below. **The fix narrative is not forced onto a false premise.**

---

## 1. Root cause (three compounding faults)

### Fault A — Strategy-native exits are runtime-blind (they never fire)
Every position-aware strategy exit (gamma/alpha/delta/omega/alpha_options) gates its SELL on
`qty > 0`, where `qty` derives from `portfolio.positions.get(symbol)`:

```
gamma.py:302   pos = portfolio.positions.get(symbol)
gamma.py:303   qty = float(pos.quantity) if pos is not None else 0.0
gamma.py:314   if qty > 0:          # ← entire exit block lives under this guard
gamma.py:386   if qty <= 0:         # ← entry block; always true when positions is empty
```

But the strategy-facing `portfolio.positions` is **hardwired empty**:

- `chad/core/live_execution_router.py:50` — `ctx = ContextBuilder().build().context` (no positions arg)
- Every other caller is identical: `routed_execution_runner.py:76`, `ibkr_execution_runner.py:402`,
  `full_execution_cycle.py:201` — **none passes `current_positions`.**
- `context_builder.py:547` `build()` defaults `current_positions=None` → `build_async` `:388`
  `current_positions or {}` → `{}` → `_build_portfolio_for_context(positions={})` `:482-485` →
  `SimpleNamespace(positions={})` `:859`.
- `git log -S "current_positions=load_open_positions"` returns **nothing** — this feed has **never**
  been wired, in any epoch.

Consequence: `portfolio.positions == {}` on every cycle → `qty == 0` → the exit block is never
entered → gamma/alpha/delta/omega/alpha_options are **functionally entry-only at runtime despite
having full exit code.** This is the true meaning behind "the accumulator that never sells."

### Fault B — The only working close path is *incidental*, not an exit
The close path that actually reads real positions is the **reconciler**
(`chad/core/position_reconciler.py:93 reconcile_positions_with_signals`), wired in the hot path at
`chad/core/live_loop.py:1916-1918`. It reads authoritative open positions from `load_open_positions()`
(guard/ledger — NOT the empty `portfolio.positions`) and emits a close **only when a net *opposing*
signal exists for that exact symbol** (`position_reconciler.py:178-185`), attributing the close to the
position's owning strategy via `position_key.split("|")[0]` (`:187-188`).

- This is why the **72 "gamma" MA SELLs (2026-06-20/21)** exist: they are **reconciler closes**
  attributed to gamma — triggered by *another* strategy's directional MA SELL against gamma's open MA
  long — **not** gamma-handler exits. (The Q3 sub-agent's "stuck gamma exit loop" reading is corrected
  here.) Record: `data/fills/FILLS_20260620.ndjson` fill_id `aa640191…`, `source=paper_trade_executor`,
  `source_strategies=[gamma]`, 2 sh @ $490.92, `status=rejected`, `broker_rejected_status:error`.
- Companion close paths — regime-reduction (`live_loop.py:1955-1984`) and flip-executor
  (`live_loop.py:1830-1836`) — also read real positions but require rarer triggers (regime change /
  `FLIP_ALLOWED`). None fired to a confirmed close this window.

The reconciler is not a risk-based exit: it closes **only** when a *different* strategy happens to emit
an opposing directional signal on the same symbol. In Epoch 2, against a real broker book, that was
frequent enough to produce ~1,428 closes. In the current routing regime (gamma+alpha monoculture,
RTH/min-size gating) opposing signals are scarce, so even this incidental path rarely triggers.

### Fault C — Post-reset broker/guard split-brain rejects the closes that *do* fire
At the Epoch-3 reset (2026-05-27, commit `4c5cd3b`) the **broker book was flattened** but the CHAD-side
paper ledger/guard retained the open longs (phantoms). Close attempts therefore target positions the
broker does not hold → broker returns `error` / no confirmed fill price → the placeholder/price-truth
defense demotes them to `status=rejected, pnl_untrusted`. Commit `29e7197` ("strict trusted-fake
placeholder rejection"), landed at the reset, tightened this rejection.

Confirmed vs rejected **equity SELLs** by month (`data/fills/FILLS_*.ndjson`):

| Month | Confirmed EQ SELL | Rejected/untrusted EQ SELL |
|------|------:|------:|
| 2026-03 | 928 | — |
| 2026-04 | 275 | 297 |
| 2026-05 | 225 | 875 |
| 2026-06 | **8** | **1,115** |
| 2026-07 | **0** | 30 |

Round-trips did not "stop being emitted" — they **stopped confirming**. The evidence pipeline
(FIFO round-trips → effective_trades → SCR → edge harness) is starved because closes are rejected,
not because SELLs are absent.

---

## 2. Why the pipeline can never be fed (summary chain)

1. Strategy-native exits blind (Fault A) → gamma/alpha/delta/etc. emit **only BUY** → inventory
   accumulates.
2. The one authoritative close path (reconciler, Fault B) fires only on an **incidental opposing
   signal** — no deterministic risk exit exists.
3. When any close *does* fire, the **broker/guard split-brain** (Fault C) rejects it → no confirmed
   round-trip → no realized PnL → SCR/edge-harness unfed.
4. **No unconditional safety net:** EOD flatten is futures-only + SKIPPED at SCALE tier + has no
   ledger drainer; `ibkr_adapter` places no stop/TP/bracket; `exit_only_executor` places no broker
   orders; the options monitor is options-only paper-bookkeeping.

---

## 3. Recommended fix — Option (b): position-aware exit overlay (default OFF, shadow-first)

**Chosen: (b)** — a deterministic, position-aware exit overlay at the portfolio layer.

- **Where:** new `chad/risk/position_exit_overlay.py`, invoked from the hot path immediately after the
  reconciler block (`live_loop.py:~1918`), on the SAME `load_open_positions()` authoritative source and
  emitting through the SAME `apply_close_intents(..., paper_adapter)` chokepoint.
- **Rule:** for each real open position, emit a close intent on either (i) **ATR stop** — adverse
  excursion > `k · ATR` from a persisted entry anchor, or (ii) **time-based max-hold** — position age
  > `max_hold_bars`. Entry anchor/age persisted to `runtime/position_exit_overlay_state.json` (survives
  restart — unlike gamma's in-process `_STATE`).
- **Safety:** config-gated `CHAD_POSITION_EXIT_OVERLAY`; the **shipped default is `mode: "shadow"`**
  (env unset → `resolve_mode` returns the config's `shadow`, not OFF — corrected 2026-07-18, was
  "default OFF"); shadow writes marker evidence to `data/exit_overlay/` (corrected, was
  `data/exit_overlay_shadow/`) and emits **no** intents; kill-switch; fail-closed on missing
  ATR/price; honors `_EFFECTIVE_NON_CHAD_SYMBOLS` operator exclusions — enforced **in the overlay
  itself** (`evaluate_positions` → `SKIP_EXCLUDED`, before the phantom guard) since FU1-B6, not only
  at the downstream `apply_close_intents` chokepoint — and paper/dry-run gating.

**Why (b) over the alternatives**

- **Rejected (a) — wire `current_positions` into ContextBuilder.** This is the deeper root-cause fix,
  but as a *first* step it has an unacceptable blast radius: it simultaneously arms the native exits of
  **6** strategies; gamma's in-memory `_STATE` (`gamma.py:215`) will not match guard-sourced positions,
  so `time_stop`/`atr_trail` misbehave; and it does nothing about the confirmation-rejection (Fault C).
  **Sequence it AFTER (b) proves the close path confirms** and the guard/broker split-brain is
  reconciled. It also violates one-change-at-a-time.
- **Rejected (c) — extend EOD flatten to equities.** `ops/micro_eod_flatten.py` has **no runtime
  drainer** for its intents ledger (writer + one test only), is futures-scoped, and blunt-flattens
  overnight-legitimate positions. Blocked and semantically wrong.

Option (b) touches no existing strategy, uses the authoritative position source and the proven close
chokepoint, and can validate in shadow before it ever emits an order — the minimal-risk path.

---

## 4. Pre-registered activation criteria (no flip without these — ALL required)

The overlay may be authored default-OFF at any time, but **arming** (`shadow → active`) requires,
in writing, ALL of:

1. `CHAD_EXECUTION_MODE=paper` and `ibkr_dry_run=true` at flip time.
2. SCR state ∈ {CONFIDENT, CAUTIOUS} (re-read `runtime/scr_state.json`, honor TTL).
3. **Guard/broker reconciliation GREEN:** `runtime/position_guard_drift.json` shows **0** qty_mismatch
   and reconciliation status not RED — i.e. Fault C's split-brain is resolved (no phantom longs).
4. **Shadow proof:** ≥ 5 sessions of shadow evidence in which every overlay-proposed close maps to a
   position that broker-truth confirms is actually held (no phantom targets).
5. **Confirmation proof:** ≥ 1 shadow-window equity close that, if it had been live, would have
   confirmed at the broker (a non-placeholder, non-`error` fill price is obtainable) — demonstrating
   Fault C no longer universally rejects equity closes.
6. Verification sequence green: `py_compile` + full `pytest` + `full_cycle_preview` clean.
7. Kill-switch + rollback documented; first 3 live cycles under manual oversight.

If any criterion is unmet, the overlay remains in shadow (evidence-only). Silent truncation/caps are
logged, never inferred as coverage.

---

## 5. Files cited (evidence)
- `chad/strategies/gamma.py:215,257,302,303,314,322,338,354,372,386`
- `chad/utils/context_builder.py:364,388,482-485,547,859`
- `chad/core/live_execution_router.py:43-60`
- `chad/core/live_loop.py:1492,1548,1830-1836,1916-1918,1955-1984`
- `chad/core/position_reconciler.py:93,167-199`
- `ops/micro_eod_flatten.py:49-51,199-222`; `chad/execution/exit_only_executor.py:160,221`
- `data/fills/FILLS_*.ndjson` (Epoch-2 vs Epoch-3 confirmed/rejected SELL counts; gamma MA record)
- reset commit `4c5cd3b`; strict placeholder rejection `29e7197`; futures-disable gate `82136d7`

---

## 6. Implementation status — Q5(b) SHADOW-first landed 2026-07-13; DEPLOYED IN SHADOW (corrected 2026-07-18)

The recommended overlay is **built and wired in SHADOW mode** (U0–U3). It activates in shadow
only at the next live-loop restart; the **ACTIVE flip remains a separate future PA** gated on the
§4 pre-registered criteria. Commits (main): U0 `268d5da` (this doc), U1 `1628211`, U2 `80ed444`.

Implementation citations:
- **Core (pure):** `chad/risk/position_exit_overlay.py` — `evaluate_positions` (:401) applies the
  three OR'd conditions (hard-stop → ATR-trailing → max-hold; first hit) and emits reduce-only
  close-intent dicts. Phantom-awareness via `_broker_signed_by_symbol` (:353), which aggregates
  `broker_sync|<symbol>` truth signed by side — mirroring `chad/core/position_guard._signed_qty`
  / `detect_guard_vs_broker_drift_v2`; a position with no same-side broker truth yields
  `EXIT_OVERLAY_SKIP_UNCONFIRMED`. Reduce-only invariant enforced at eval (`min(guard_qty,
  broker_held)`) and re-checked at submit by `_reduce_only_reclamp` (:728).
- **Config (pre-registered, default shadow):** `config/position_exit_overlay.json` (frozen
  thresholds; `mode` is the sole post-freeze flip). Kill-switch env `CHAD_POSITION_EXIT_OVERLAY`
  (OFF|SHADOW|ACTIVE) via `resolve_mode`.
- **Runner + factory:** `PositionExitOverlay` (:567) does markers + `data/exit_overlay/*.ndjson`
  evidence + anchor persistence (`runtime/position_exit_overlay_state.json`) and, only in ACTIVE,
  routes through `chad.core.position_reconciler.apply_close_intents` (reused, not forked → inherits
  the adapter RTH gate `ibkr_adapter._evaluate_rth_gate`, idempotency, margin-shadow).
  `build_default_overlay` (:846) fail-opens to `None` on bad config and carries the margin-gate
  pytest evidence/state leak guard. A cycle error logs `EXIT_OVERLAY_ERROR` and submits nothing.
- **Hot-path wiring:** `chad/core/live_loop.py:2270-2278` (corrected 2026-07-18, was `1933-1953`) —
  inside `run_once`, immediately after the reconciler block and before the market-metrics publisher
  / intent planning.
- **Tests:** `chad/tests/test_position_exit_overlay.py` (23) + `..._wiring.py` (4) — every
  condition, all skip classes, reduce-only clamp + reclamp, short mirror, anchor seed/update/prune,
  config validation, kill-switch, SHADOW-writes-evidence-submits-nothing, OFF inert, error
  fail-open, ACTIVE reaches the real submit path + is RTH-gate-subject, phantom dropped at submit,
  structural wiring guard.

New markers: `EXIT_OVERLAY_SHADOW`, `EXIT_OVERLAY_SKIP_UNCONFIRMED`, `EXIT_OVERLAY_ERROR`,
`EXIT_OVERLAY_ACTIVE_CLOSE`. Evidence: `data/exit_overlay/exit_overlay_YYYYMMDD.ndjson`
(schema `exit_overlay.v1`).

**Deploy status: DEPLOYED IN SHADOW** (corrected 2026-07-18; was "NOT DEPLOYED"). The overlay is
live in shadow — `runtime/exit_overlay_heartbeat.json` fresh `mode=shadow ts=2026-07-18T01:55:20Z
evaluated=7`; it emits marker/heartbeat evidence and submits **nothing** (SHADOW). The ACTIVE flip
is still its own authorization PA per §4. No service files touched by this correction; no restart
performed. (The FU1 B-2..B-6 fixes below land at the next gated restart; SHADOW keeps submitting
nothing until the ACTIVE flip.)
