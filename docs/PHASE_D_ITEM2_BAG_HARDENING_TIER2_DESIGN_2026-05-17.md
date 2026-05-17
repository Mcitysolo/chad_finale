# Phase D Item 2 — BAG Hardening Tier 2 Design

## Status

**DESIGN ONLY — NO LIVE BAG EXECUTION.**

This document is the output of an audit-only task on 2026-05-17. It contains
no code, no runtime mutation, no deploy changes, and no commit. The Tier 2
implementation it scopes is itself a paper-only / dry-run hardening; it does
not authorize live BAG order placement.

---

## 1. Tier 1 baseline

Phase D Item 2 Tier 1 (commit 51166d4, 2026-05-17) introduced:

- `chad/options/spread_spec.py` — the frozen `OptionsSpreadSpec` dataclass
  carrying `symbol`, `expiry`, `long_strike`, `short_strike`, `long_right`,
  `short_right`, `ratio_long`, `ratio_short`, `exchange`, `currency`,
  `spread_type`, `max_loss_per_contract`, `net_debit_estimate`, `spread_id`,
  `dte`, plus `to_legacy_meta()` / `from_legacy_meta()` adapters.
- `chad/options/__init__.py` — public re-export.
- `chad/strategies/alpha_options.py` — emits the typed spec under
  `meta["spread_spec"]` alongside the legacy flat keys (lines 473–530).
- `chad/execution/ibkr_adapter.py` — `_resolve_combo` consumes the typed
  spec first and falls back to legacy meta only when absent (lines
  1050–1107).
- `chad/execution/paper_exec_evidence_writer.py` — hydrates legacy keys
  from a typed spec under `extra["spread_spec"]` when callers omit the
  flat fields (lines 977–1027).
- `scripts/preview_bag_intent.py` — dry-run CLI; mode-gated; contains no
  IBKR imports and never calls `placeOrder` (asserted by
  `chad/tests/test_bag_preview.py:165–239`).
- Expanded test coverage: `test_options_spread_spec.py`, the typed-spec
  cases in `test_alpha_options*.py`, `test_alpha_options_bag_paper_fill.py`,
  `test_bag_preview.py`.

Full pytest baseline after Tier 1: **2054 passed**.

Tier 1 created a *typed contract*. It did **not** modify any
live-execution path beyond letting the adapter prefer the typed spec when
present.

---

## 2. Live BAG blockers (recap)

Five blockers prevent live BAG promotion. None are addressed by Tier 1:

1. **Force LMT for BAGs.** Default `order_type` in some intent intake
   paths is `"MKT"`. MKT BAG legs cross independently against the touch
   and slip violently.
2. **Broker-mid quote check.** The strategy's `net_debit_estimate` is a
   heuristic (`0.50 × width × 100`); a stale broker mid can submit far
   off the live combo mid.
3. **Bracket / OCA or fail-safe exit.** Only exit today is
   `alpha_options.max_hold_seconds=3600` SELL emitted by `live_loop`. If
   `live_loop` is down, a live BAG sits unmanaged.
4. **`spread_id`-aware reconciliation.** `position_guard` keys on
   `(strategy, symbol)`. Concurrent BAGs on the same symbol collide.
5. **Live BAG fill harness.** Zero tests today exercise a
   broker-confirmed live BAG round-trip through `trade_closer` /
   `profit_lock`.

---

## 3. Audit findings (Steps 2–5)

### 3.1 Order type / limit-price path (Step 2)

- `chad/execution/order_type_selector.py` is hard-coded to **never emit
  MKT** (`ORDER_TYPE_LIMIT="LMT"`, `order_type_selector.py:72`). Anything
  routed through `execution_pipeline._build_intent` → `select_order_type`
  receives LMT.
- `chad/execution/execution_pipeline.py:1192–1317` invokes
  `select_order_type` and stamps `e4_order_type` / `e4_limit_price` onto
  the `IBKRStrategyTradeIntent`. Strategy-emitted `net_debit_estimate`
  is **not** the source of `e4_limit_price` for BAGs in this path;
  instead `compute_aggressive_limit_price(side, reference_price=…)` or
  `float(order.price)` is used. There is no BAG-specific guard tying the
  limit price back to `net_debit_estimate`.
- `chad/execution/ibkr_adapter.py:1670–1693` (`_intent_from_routed_signal`)
  copies `meta["net_debit_estimate"]` into `meta["limit_price"]` for
  OPTIONS/BAG (line 1673–1674) — but defaults `order_type` to `"MKT"`
  when meta omits it (line 1685).
- `chad/execution/ibkr_adapter.py:1695–1744` (`_intent_from_trade_intent`)
  also defaults `order_type` to `"MKT"` when the `StrategyTradeIntent`
  omits it (line 1699). `limit_price` is read from the raw intent and
  may be `None`.
- `_OrderFactory.build` (`ibkr_adapter.py:1266–1296`) already validates:
  rejects MKT when `IbkrConfig.allow_market_orders=False` (default
  `True` — `ibkr_adapter.py:303`), and requires a positive `limit_price`
  for LMT. There is **no BAG-specific gate**.
- Net: the *pipeline* path is LMT-by-construction for all asset classes
  including BAG, but the *legacy / direct-intent* path is MKT-by-default
  for BAG when the producer omits `order_type`. This is the live
  footgun.

### 3.2 Broker-mid / quote check (Step 3)

- `chad/market_data/ibkr_price_provider.py` exposes
  `IBKRPriceProvider.get_snapshot(symbol, sec_type)` with
  `_make_contract` handling `"STK" / "FUT" / "FX"` only
  (`ibkr_price_provider.py:165–172`). **No OPT or BAG path.**
- No module under `chad/options/` or `chad/execution/` queries combo
  mids or per-leg option bids/asks today. Search for
  `BAG.*quote|combo.*quote|option.*bid|option.*ask` returns no hits in
  execution/options/strategies.
- The strategy's `net_debit_estimate` is the only price that ever
  reaches the adapter for BAG (`strike_selector.py:218`,
  `0.50 × actual_width × 100`).
- A broker-mid check would require new code: either
  (a) `reqMktData` on the BAG `Contract` (IBKR supports combo
  market data with `genericTickList=""`), or (b) two per-leg snapshot
  calls (long bid/ask, short bid/ask) and synthesise
  `debit_mid = long_mid − short_mid`. Both can be unit-tested with a
  fake `IB` shim and never need a live connection in CI.
- Bag mid is not currently used anywhere; introducing it is purely
  additive.

### 3.3 Exit / bracket / failsafe (Step 4)

- The sole exit mechanism for `alpha_options` today is the loop in
  `chad/strategies/alpha_options.py:583–685`: when bar age exceeds
  `tuning.max_hold_seconds` (default 3600 s) the strategy emits a SELL
  intent with `meta.reason="max_hold_exit"` and reconstructs `bag_legs`
  from the last opening fill.
- Grep for `bracket|OCA|parentId|transmit` across
  `chad/execution`, `chad/core`, `chad/risk` returns **no matches**.
  The adapter has no bracket/OCA wiring. Every order is a single
  parentless submission.
- The drift detector (`chad/core/position_guard.py
  detect_guard_vs_broker_truth_drift`, wired into
  `chad-reconciliation-publisher` per CLAUDE.md) does *detect* stale
  guard entries but does not close BAG positions. Closure relies
  entirely on a healthy `live_loop`.
- If `live_loop` is down (gateway disconnect, process restart, kernel
  reboot, IB Gateway latency crash) a live BAG cannot be exited.

### 3.4 spread_id-aware reconciliation (Step 5 partial)

- `spread_id = str(uuid.uuid4())` is minted in
  `alpha_options.py:439` and propagated through:
    - `OptionsSpreadSpec.spread_id`
    - `meta["spread_id"]` on the BAG intent
    - `extra["spread_id"]` on the paper fill record
      (`paper_exec_evidence_writer.py:1028`)
    - `_OPT_META_KEYS` flattening in `execution_pipeline.py:1240`
    - the BAG meta keys hoisted to `live_loop.py:2048`.
- `position_guard.STATE_PATH` (`runtime/position_guard.json`) keys
  entries by `f"{strategy}|{symbol}"` only. `spread_id` is not part of
  the key. Two simultaneous SPY verticals from `alpha_options` would
  share a single guard slot and clobber one another's state.
- A migration to `(strategy, symbol, spread_id)` keying is a
  cross-cutting reconciler change touching `position_guard.py`,
  `trade_closer.py`, `live_loop.py` reconcile paths,
  `scripts/close_guard_entry.py`, and every test that pokes the
  guard. Higher blast radius than Option A.

### 3.5 Live fill / paper evidence (Step 5)

- `paper_exec_evidence_writer.py:1340–1467` (`simulate_bag_paper_fill`)
  stamps `fill_price = net_debit_estimate`, `bag_legs`, and
  `extra["pnl_untrusted"] = False` for the opener. The closer
  (`simulate_bag_close_paper_fill`, lines 1103–1339) uses
  `close_credit = original_debit × _BAG_CLOSE_CREDIT_RATIO` (synthetic
  30% credit, never a market quote) and **always** marks
  `pnl_untrusted=True` with reason `bag_close_synthetic_credit_ratio_30pct`.
  Quarantine sweep `test_quarantine_20260511_alpha_options.py` keys off
  that reason string.
- Live fill path: `chad/execution/ibkr_adapter.py:1396` reads
  `trade.orderStatus`, line 1951 inspects `order_status`. There is **no
  test** in `chad/tests/` that exercises a `placeOrder → orderStatus →
  fill` cycle for a BAG `Contract`. All BAG tests are paper-fill or
  resolution tests.
- `spread_id` is the natural correlation handle to reconcile a live BAG
  round-trip, but the live evidence writer does not yet stamp it on the
  open-side fill payload in the same shape the paper simulator does.
- A live BAG fill harness can be built around a fake `IB` shim that
  returns a synthetic `Trade` object — no live connection required —
  and exercise adapter → evidence writer → trade_closer →
  position_guard in one integration test.

---

## 4. Blocker recommendation table

| # | Blocker | Current evidence | Tier 2 recommendation | Testability | Risk |
|---|---|---|---|---|---|
| 1 | Force LMT for BAGs | Pipeline path is LMT-by-construction via `select_order_type`. Legacy intake paths in `_intent_from_routed_signal` / `_intent_from_trade_intent` default `order_type` to `"MKT"` (`ibkr_adapter.py:1685`, `1699`). `_OrderFactory` already has an `allow_market_orders=False` switch but it is global, not BAG-scoped. | Add a **BAG-scoped** validation: reject any BAG intent whose `order_type != "LMT"` and any BAG LMT whose `limit_price` is missing / non-positive. Default `limit_price` from `meta["net_debit_estimate"]` (or `spread_spec.net_debit_estimate`) when not explicitly supplied. Non-BAG order types untouched. | Pure unit tests; no IBKR needed; new tests in `test_options_spread_spec.py` + `test_ibkr_adapter_*.py`. | **LOW.** Validation-only; surgical. |
| 2 | Broker-mid quote check | No combo / OPT quote helper exists. `IBKRPriceProvider` covers STK/FUT/FX only. `net_debit_estimate` is a 0.50×width heuristic with no broker corroboration. | Defer to Tier 3. Define the interface (`get_bag_mid(spec) -> Optional[float]`) and the divergence policy (`max_deviation_pct`) in a follow-on design; no code in Tier 2. | Requires a fake `IB` shim once implemented. | **MED.** New IBKR call, new failure modes (stale ticks, halted contracts). |
| 3 | Bracket / OCA or failsafe | Sole exit is `alpha_options.max_hold_seconds=3600` driven by `live_loop`. No `parentId`/`OCA` references anywhere in execution. | Defer to Tier 4. Choose between native IBKR bracket (parentId/transmit chain) and an out-of-band failsafe closer (cron-driven, independent of `live_loop`). | Bracket: needs broker mock that respects parent/child semantics. Failsafe: pure unit-testable. | **MED-HIGH.** Bracket changes adapter primitives; failsafe adds a new daemon. |
| 4 | spread_id-aware reconciliation | `spread_id` flows through meta/fills but `position_guard` keys are `(strategy, symbol)`. Concurrent SPY spreads would collide. | Defer to Tier 5. Migration affects `position_guard.py`, `trade_closer.py`, `live_loop.py`, `scripts/close_guard_entry.py`, and ~all guard tests. Schema bump required. | Wide test surface but offline-testable. | **HIGH.** Cross-cutting reconciler change with persisted-state migration. |
| 5 | Live BAG fill harness | Zero tests exercise `placeOrder → Trade → orderStatus` for a BAG contract. All BAG tests are paper-fill / resolution. | Defer to Tier 6. Builds on Tier 2's LMT guarantee, Tier 3's quote check, and Tier 4's exit semantics — meaningful only once they exist. | Pure offline test with a fake `IB`. | **LOW once dependencies land.** Premature today: would lock in a contract before LMT/quote/exit shape is decided. |

---

## 5. Recommended Tier 2 target

**Option A — Force BAG LMT discipline (tests only, paper-only).**

### Justification

- **Smallest blast radius.** Pure pre-submit validation, no new
  modules, no IBKR calls, no schema changes, no runtime mutation.
- **Highest single-step leverage.** Eliminates the most dangerous
  live BAG footgun (MKT slippage on independently-crossed legs)
  before any live combo order can be sent.
- **Cleanly composable with later tiers.** A broker-mid check
  (Option B) becomes "compare `limit_price` to `bag_mid`", which is a
  natural extension of Option A's "BAG must have a positive
  `limit_price`". Without Option A first, B has no anchor.
- **Builds on Tier 1's typed contract.** `OptionsSpreadSpec.net_debit_estimate`
  is already wired through the adapter (`ibkr_adapter.py:1057–1067`),
  giving Tier 2 a typed source for the BAG LMT price.
- **No live broker dependency.** Every check is offline-testable.
- **Reversibility.** A purely additive validation can be reverted
  with a single revert; nothing persisted, nothing live.
- Options B / C / D each carry materially higher coupling
  (new market-data path, persisted-state migration, or new daemon).
  Sequencing the easy, contained win first reduces stack risk on the
  later tiers.

---

## 6. Proposed implementation scope (Option A — Tier 2)

### In scope

1. **BAG LMT enforcement gate.** In `chad/execution/ibkr_adapter.py`,
   add a BAG-scoped pre-build check (most likely in `_OrderFactory.build`
   or in the two intent-intake paths
   `_intent_from_routed_signal` / `_intent_from_trade_intent`) that:
   - rejects any intent with `sec_type == "BAG"` and
     `order_type != "LMT"` (via `ValidationError`), regardless of the
     global `allow_market_orders` flag, **and**
   - rejects any BAG LMT whose `limit_price` is `None`, `NaN`,
     non-finite, or `<= 0`.
2. **Default BAG `limit_price` from `net_debit_estimate`.** When the
   intent has `sec_type == "BAG"` and `limit_price is None`, hydrate
   it from (in order):
   - `intent.meta["spread_spec"].net_debit_estimate` if a typed
     `OptionsSpreadSpec` is present and the field is non-null,
   - else `intent.meta["net_debit_estimate"]` if non-null,
   - else **reject** the intent. No silent fallback to `0.0`.
3. **Non-BAG paths untouched.** STK, FUT, FX, single-leg OPT order
   types and limit prices flow exactly as today.
4. **Tests.** Add to `chad/tests/`:
   - BAG MKT intent is rejected with a clear `ValidationError`.
   - BAG LMT with no `limit_price` and no `net_debit_estimate` is
     rejected.
   - BAG LMT with no `limit_price` but a typed
     `OptionsSpreadSpec.net_debit_estimate` is hydrated and accepted.
   - BAG LMT with no `limit_price` but a legacy
     `meta["net_debit_estimate"]` is hydrated and accepted.
   - BAG LMT with an explicit `limit_price > 0` passes through
     unchanged (no override).
   - STK MKT and STK LMT behaviour is identical to pre-change baseline
     (no regression in existing tests).
5. **Preview-only verification of the gate.** Extend
   `scripts/preview_bag_intent.py` with a `--simulate-validation` flag
   (broker-disconnected) that runs the new validation against a
   constructed intent and prints the result. No `placeOrder`, no
   `connectAsync`.
6. **Docs.** A short Tier 2 close-out doc summarising the change,
   tests, and full-suite pytest count.

### Out of scope (explicitly)

- No `reqMktData` / combo-mid pull.
- No bracket / OCA / parentId wiring.
- No `position_guard` key schema change.
- No live-fill harness.
- No strategy logic change (alpha_options behaviour unchanged).
- No runtime mutation (no edits under `runtime/`).
- No deploy / systemd / service-file change.
- No live posture flip.

---

## 7. What remains after Tier 2

After Tier 2 lands, the open live-BAG blockers are:

- **Tier 3:** Broker-mid quote pull and divergence gate.
- **Tier 4:** Bracket / OCA or out-of-band fail-safe closer.
- **Tier 5:** `spread_id`-aware `position_guard` reconciliation.
- **Tier 6:** Live BAG fill harness (offline `IB` shim, full
  adapter → evidence → trade_closer → profit_lock round-trip).
- **Tier 7 (post-implementation):** Operator-initiated live BAG
  promotion in a *separate* governance change — not part of Tier 2.

---

## 8. Forbidden actions

- **No MKT BAG live orders** — Tier 2 makes this a hard adapter-level
  rejection.
- **No live BAG execution** authorized by this design.
- **No strategy behaviour change** beyond pre-submit validation.
- **No runtime mutation** (no edits under `runtime/`).
- **No systemd / deploy changes.**
- **No config-file edits.** Risk caps, posture, strategy
  configuration remain governed by the Pending Action mechanism per
  CLAUDE.md.

---

## 9. Next implementation prompt (Tier 2 — outline)

When the operator is ready to execute Tier 2:

> **Phase D Item 2 Tier 2 — Force LMT discipline for BAG intents
> (paper-only, validation-only).**
>
> Scope:
> 1. In `chad/execution/ibkr_adapter.py`, add a BAG-scoped pre-submit
>    validation. For `sec_type == "BAG"`:
>    - require `order_type == "LMT"`;
>    - require finite, positive `limit_price`;
>    - hydrate `limit_price` from
>      `meta["spread_spec"].net_debit_estimate` (typed) or
>      `meta["net_debit_estimate"]` (legacy) when not supplied;
>    - otherwise raise `ValidationError`.
> 2. Leave non-BAG paths untouched.
> 3. Add unit tests covering accept / reject / hydrate cases.
> 4. Extend `scripts/preview_bag_intent.py` with a
>    `--simulate-validation` flag (broker-disconnected).
> 5. Run the verification sequence from CLAUDE.md.
> 6. Document the outcome in
>    `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER2.md`.
>
> Forbidden:
> - no `placeOrder` from preview or tests;
> - no `ib_async` connection;
> - no live posture flip;
> - no runtime / deploy / systemd changes;
> - no commit until verification sequence passes.

---

*End of design.*
