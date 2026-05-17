# Phase D Item 2 — BAG Execution Hardening, Tier 1

**Status:** PAPER-ONLY HARDENING — live BAG execution remains **NOT AUTHORIZED**.
**Date:** 2026-05-17
**Scope:** additive only; no strategy behavior change beyond typed metadata.

---

## Context

The Phase D Item 2 audit classified CHAD's BAG (combo) options execution path
as `READY FOR PAPER BAG HARDENING ONLY`. The audit found:

- `alpha_options` already emits one BAG `TradeSignal` per spread with the
  required leg meta (`expiry`, `long_strike`, `short_strike`, `long_right`,
  `short_right`, `net_debit_estimate`, `spread_id`).
- The IBKR adapter already builds `Contract(secType="BAG")` with two
  `ComboLeg`s and guards against `Error 321` (conId=0).
- The paper-fill simulator already produces a trusted open and an explicitly
  `pnl_untrusted` synthetic close.
- BAG meta is carried across the strategy → pipeline → adapter → writer path
  as a **string-keyed dict**. A single typo silently kills a BAG. No typed
  contract enforces the leg shape at the boundary.

Tier 1 adds that typed contract — `chad.options.spread_spec.OptionsSpreadSpec` —
and a paper-only dry-run preview CLI. It does not touch any of the live-blocker
gaps below.

---

## What changed

### New files
- `chad/options/spread_spec.py` — frozen `OptionsSpreadSpec` dataclass with
  validators (YYYYMMDD expiry, strikes > 0, rights ∈ {C, P}, ratios ≥ 1,
  long ≠ short, non-empty exchange/currency), plus `to_legacy_meta()`,
  `from_legacy_meta()`, `bag_leg_dicts()`, and `as_dict()` helpers.
- `chad/tests/test_options_spread_spec.py` — validator + projection tests.
- `chad/tests/test_bag_preview.py` — preview CLI behavior + safety tests.
- `scripts/preview_bag_intent.py` — dry-run CLI; mode-gated; no IBKR imports.
- `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER1.md` — this file.

### Modified files (additive)
- `chad/options/__init__.py` — exports `OptionsSpreadSpec`.
- `chad/strategies/alpha_options.py` — stamps `meta["spread_spec"] =
  OptionsSpreadSpec(...)` alongside existing legacy keys. Sizing,
  confidence, side, exits, and spread selection are unchanged.
- `chad/execution/ibkr_adapter.py` — `_resolve_combo` prefers
  `meta["spread_spec"]` when present; otherwise rebuilds it via
  `OptionsSpreadSpec.from_legacy_meta(...)`. All pre-existing
  `ContractResolutionError` messages and `BAG_INTENT_SKIPPED_UNQUALIFIED_LEG`
  log markers are preserved.
- `chad/execution/paper_exec_evidence_writer.py` — `_hydrate_legacy_bag_meta_from_spec`
  helper backfills missing legacy keys from `extra["spread_spec"]` for both
  the OPEN-side simulator (`simulate_bag_paper_fill`) and the SELL-close
  helper (`_simulate_bag_sell_close`). PnL math, close-credit ratio
  (0.30), and `pnl_untrusted` semantics are unchanged.
- `chad/tests/test_alpha_options*.py` — extensions only.

---

## What did NOT change

- No live broker code path was modified. `placeOrder`, `qualifyContracts`,
  bracket logic — all untouched.
- No `runtime/`, `deploy/`, `ops/`, `config/`, `shared/`, or `chad/core/`
  files were modified.
- No systemd unit files were touched. No services were restarted.
- No `git commit` was executed by this task.
- `alpha_options` continues to emit the exact same legacy dict meta
  (`sec_type=BAG`, `spread_id`, strikes, rights, `net_debit_estimate`,
  Greeks). The typed `spread_spec` is an additional channel.
- The paper-fill simulator's `0.50 * width * 100` open price heuristic
  and `0.30 * original_debit` close-credit ratio are unchanged.
- The strategy's `max_hold_seconds=3600` exit timer is unchanged.

---

## Why live BAG is still NOT authorized

The audit identified five gaps that must close before any live BAG can be
considered. Tier 1 addresses **none of them directly**; it only creates the
typed contract on which Tier 2 will build.

### Remaining live blockers

1. **Force LMT for BAGs, never MKT.** Today the default `order_type` flows
   through as `MKT`. Combo MKT orders slip badly because IBKR fills each
   leg independently against the touch. Tier 2 must:
   - Require `order_type="LMT"` for any `sec_type=="BAG"` intent.
   - Refuse to submit when `limit_price` is missing or stale.
2. **Broker-mid quote check before submit.** The strategy stamps a
   `net_debit_estimate = 0.50 * width * 100` heuristic. The live BAG path
   must:
   - Pull a live combo mid via `reqMktData` on the BAG contract.
   - Refuse to submit when the strategy estimate deviates from the broker
     mid by more than a configured tolerance.
3. **Bracket / OCA or fail-safe exit.** Today the only close mechanism is
   `alpha_options`'s 1-hour `max_hold_seconds` SELL emitted by
   `live_loop.run_once`. If `live_loop` is down (gateway disconnect,
   process restart), live BAGs sit unmanaged. Tier 2 must add either:
   - A native IBKR bracket (parent BAG + child stop / target), or
   - A cron-triggered fail-safe close path independent of `live_loop`.
4. **`spread_id`-aware position tracking.** `position_guard` keys on
   `(strategy, symbol)`. Multiple concurrent SPY spreads from
   `alpha_options` would collide. The reconciler must move to
   `(strategy, symbol, spread_id)` before more than one open BAG per
   symbol can be permitted.
5. **Live BAG fill test harness.** Zero tests today exercise a
   broker-confirmed live BAG round-trip producing trusted PnL.
   Tier 2 must add a mock `ib.placeOrder` returning a `Trade` with
   fills, end-to-end through `trade_closer` and `profit_lock`.

---

## Verification evidence

Tier 1 verification (see paste-back):

1. `python3 -m py_compile` on all new and modified files → success.
2. Targeted pytest on `test_options_spread_spec.py`,
   `test_alpha_options.py`, `test_alpha_options_bag_paper_fill.py`,
   `test_alpha_options_meta_preservation.py`,
   `test_alpha_options_bag_no_etf_downgrade.py`, `test_bag_preview.py`
   → all pass.
3. Full suite `python3 -m pytest chad/tests/ -q` → no regressions
   relative to the 2006-passing baseline.
4. `python3 -m chad.core.full_cycle_preview` → clean dry-run.
5. `CHAD_EXECUTION_MODE=paper python3 scripts/preview_bag_intent.py ...`
   → emits the expected JSON with `spread_spec` + `legacy_meta` +
   `bag_legs` and exits 0.
6. `grep` over `scripts/preview_bag_intent.py` and
   `chad/tests/test_bag_preview.py` for `placeOrder`, `connectAsync`,
   `ib_async`, `IB(` → no matches.
7. `git status --short` after task → only Tier 1 files modified /
   created. No commit.

---

## Next steps (Tier 2 — NOT in this task)

When the operator is ready to advance toward live BAG execution:

1. LMT enforcement + limit-price discipline in
   `chad/execution/execution_pipeline.py` for `sec_type == "BAG"`.
2. Pre-submit broker-mid quote pull (new module
   `chad/options/bag_mid_quote.py`).
3. Bracket / OCA wiring in `chad/execution/ibkr_adapter.py` OR a
   cron-driven `chad/risk/bag_failsafe_closer.py`.
4. Migrate `position_guard` keying to `(strategy, symbol, spread_id)`.
5. Live BAG fill mock harness in
   `chad/tests/test_ibkr_live_bag_roundtrip.py`.
6. After all five land, **then** request live posture promotion in a
   separate change.
