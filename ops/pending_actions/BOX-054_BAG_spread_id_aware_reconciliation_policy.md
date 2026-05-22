# Box 054 — BAG spread_id-aware reconciliation policy

**Status:** Pending Action — DOCUMENTATION ONLY. No production code change.
No runtime mutation. No order placement. No live authorization.

**Scope:** Pins the contract that **concurrent BAG spreads carry an
immutable spread_id end-to-end**, AND documents the
`(strategy, symbol)`-only keying gap on
`position_guard._position_key` and on
`paper_exec_evidence_writer._find_opening_bag_fill` as the unblock
condition for true concurrent-BAG support. Closure path =
`PASS_LIVE_BAG_BLOCKED_BUT_POLICY_READY`.

---

## 1. Identity-keying inventory by subsystem

| Subsystem                                                       | Key shape today                                                                                                  | spread_id used?                | Collision risk under concurrent BAGs                       |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------- |
| `chad/strategies/alpha_options.py:440-534`                        | Stamps `spread_id = str(uuid.uuid4())` onto every emitted BAG `TradeSignal`; one signal per call (`return [signal]`) | **YES (produced)**             | None — producer emits one BAG per call.                     |
| `chad/options/spread_spec.py::OptionsSpreadSpec`                  | Typed `spread_id: Optional[str]` field; `to_legacy_meta()` projects it (line 172-173); `from_legacy_meta()` round-trips it (line 212-216) | **YES (carried)**              | None — preservation channel.                                 |
| `chad/execution/execution_pipeline.py::BAG_META_KEYS` (line 948)   | Strict allow-list includes `"spread_id"` (line 949) and `"spread_type"`                                            | **YES (preserved end-to-end)** | None — preservation channel.                                 |
| `chad/execution/execution_pipeline.py::_OPT_META_KEYS` (line 1236) | Single-leg-and-BAG allow-list includes `"spread_id"` (line 1240)                                                   | **YES (preserved end-to-end)** | None — preservation channel.                                 |
| `chad/execution/paper_exec_evidence_writer.py::_hydrate_legacy_bag_meta_from_spec` (line 985-1033) | Hydrates `spread_id` (line 1028) from typed `OptionsSpreadSpec` under `extra["spread_spec"]`                       | **YES (hydrated)**             | None — preservation channel.                                 |
| `chad/execution/ibkr_adapter.py::_stable_idempotency_payload` (line 2940-3002) | For `sec_type == "BAG"` includes `bag.expiry`, `bag.long_strike`, `bag.short_strike`, `bag.long_right`, `bag.short_right` | Indirect (leg-tuple-aware)     | None — different leg tuples → different idempotency keys.   |
| `chad/core/position_guard.py::_position_key` (line 184)            | Returns `f"{strategy}|{symbol}"` only                                                                              | **NO**                          | **YES** — two BAGs with same `(strategy, symbol)` but distinct spread_id collide. |
| `chad/execution/paper_exec_evidence_writer.py::_find_opening_bag_fill` (line 1036-1100) | Signature `(strategy, symbol)`; matches by strategy + symbol + side="BUY" + status="paper_fill"                    | **NO**                          | **YES** — returns most-recent matching fill regardless of `spread_id`. |
| `chad/strategies/alpha_options.py` max-hold-exit loop (line 582-693) | Iterates paper-ledger entries keyed by `f"{strategy}|{symbol}"`                                                    | **NO**                          | Implicit one-BAG-per-(strategy,symbol).                       |
| `chad/core/flip_executor.py:179`                                   | `position_key = f"{existing_strategy}|{symbol}"`                                                                  | **NO**                          | Not BAG-specific; non-BAG keying.                            |
| `chad/core/position_reconciler.py`                                | Iterates `open_positions` keyed by the `position_guard` key                                                       | **NO**                          | Inherits position_guard's `(strategy, symbol)` gap.            |

**Summary:**
- **Preservation channel is complete.** `spread_id` flows from
  `alpha_options.py` → `OptionsSpreadSpec` → planner artifact
  (`BAG_META_KEYS` + `_OPT_META_KEYS`) → paper-fill writer
  (`_hydrate_legacy_bag_meta_from_spec`) → fill payload (`extra.spread_id`).
- **Identity-keying surface has a gap.** `position_guard._position_key`,
  `_find_opening_bag_fill`, and the max-hold-exit ledger loop key by
  `(strategy, symbol)` only — they do NOT consult `spread_id`.

---

## 2. Why this is structurally safe TODAY

Three structural blocks make the keying gap **prospective only**:

1. **Box 053 live BAG block.** `IbkrConfig.dry_run` defaults to True;
   `runtime/live_readiness.json::ready_for_live=false`. No live BAG
   can open.
2. **`alpha_options.py:534`: `return [signal]`.** The producer emits
   AT MOST ONE BAG signal per call. There is no code path that emits
   two concurrent BAGs for the same `(strategy, symbol)` in a single
   cycle.
3. **same_side_position_open guard.** The
   `position_guard.is_same_side_open(intent)` check blocks a second
   BUY for an already-open `(strategy, symbol)` position before the
   intent reaches the IBKR adapter. While one BAG is open, a second
   alpha_options BAG BUY for the same symbol is refused.

The combination forces an implicit "one alpha_options BAG per symbol"
contract that holds without `spread_id`-aware keying.

---

## 3. Hard rules

1. **`spread_id` MUST be stamped at emission.** `alpha_options.py`
   stamps a fresh uuid4; no BAG signal may reach the planner with a
   missing or blank `spread_id`. Pinned by
   `test_alpha_options_signal_meta_has_uuid_spread_id`.
2. **`spread_id` MUST survive the planner allow-lists.** Both
   `BAG_META_KEYS` and `_OPT_META_KEYS` MUST contain `"spread_id"`.
   Pinned by `test_execution_pipeline_bag_meta_keys_include_spread_id`.
3. **`spread_id` MUST hydrate from the typed spec.** The paper-fill
   writer's `_hydrate_legacy_bag_meta_from_spec` MUST list
   `"spread_id"` in its backfill key list. Pinned by
   `test_paper_evidence_writer_hydrates_spread_id_from_spec`.
4. **BAG idempotency MUST be leg-tuple-aware.** The IBKR adapter
   idempotency key MUST differ when expiry / strikes / rights differ.
   Pinned by three IBKR adapter tests.
5. **The `(strategy, symbol)`-only keys MUST be widened before
   concurrent live BAGs are allowed.** Specifically:
   - `position_guard._position_key` MUST gain a BAG branch that
     includes `spread_id` (or an equivalent immutable leg-hash),
     while preserving the non-BAG `f"{strategy}|{symbol}"` shape.
   - `_find_opening_bag_fill` MUST accept an optional `spread_id`
     argument and filter on it when provided.
   - `alpha_options.max_hold_exit` MUST iterate by
     `(strategy, symbol, spread_id)` when multiple BAGs are open.
6. **Until those widenings land, live BAG entry remains blocked** by
   Box 053's structural mechanisms (dry_run default + LiveGate
   fail-closed).

---

## 4. When concurrent-BAG support is enabled (future)

The Box 054 anchor tests fail by design when the
`(strategy, symbol)`-only keys widen, forcing a deliberate refresh:

| Anchor test                                                                  | Asserts                                                                            | Fails when (refresh signal)                                              |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `test_position_guard_key_is_strategy_symbol_only`                              | `_position_key` returns `f"{strategy}|{symbol}"` exactly                            | When `_position_key` is widened to consume `spread_id` (or anything else)  |
| `test_position_guard_key_collides_on_same_strategy_and_symbol`                  | Two identical `(strategy, symbol)` queries yield identical keys                     | When the key is sec_type-branched                                         |
| `test_position_guard_key_for_non_bag_is_unchanged`                              | Non-BAG callers still get `f"{strategy}|{symbol}"`                                  | When non-BAG keying changes inadvertently                                  |
| `test_find_opening_bag_fill_signature_is_strategy_symbol_only`                   | `_find_opening_bag_fill(strategy, symbol)` (2 params)                                | When a third `spread_id` parameter is added                                |
| `test_alpha_options_signal_emit_block_returns_single_signal_list`                | Source contains `return [signal]`                                                  | When the strategy is widened to emit multiple BAGs per call                |

---

## 5. Acceptance criteria — Box 054

| Criterion                                                                                       | Status |
| ----------------------------------------------------------------------------------------------- | ------ |
| `spread_id` is stamped at producer (alpha_options uuid4)                                          | PASS   |
| `spread_id` is round-tripped via `OptionsSpreadSpec.to_legacy_meta`                                | PASS   |
| `spread_id` is in both `BAG_META_KEYS` and `_OPT_META_KEYS` preservation allow-lists                | PASS   |
| `spread_id` is hydrated by `_hydrate_legacy_bag_meta_from_spec` into the fill payload              | PASS   |
| IBKR adapter idempotency key is leg-tuple-aware (per-spread identity at broker submission boundary) | PASS   |
| `position_guard._position_key` keying gap is pinned and documented                                  | PASS (gap recorded; future-wiring refresh signals in place) |
| `_find_opening_bag_fill` signature gap is pinned and documented                                    | PASS (gap recorded; future-wiring refresh signals in place) |
| Non-BAG keying behavior is unchanged                                                              | PASS   |
| Live BAG is structurally blocked (cross-ref Box 053)                                              | PASS   |
| Tests cover BAG spread identity behavior                                                          | PASS (15 new + IBKR idempotency family) |

---

## 6. Runtime / live invariants

- No `runtime/*.json` mutation.
- No SQLite mutation.
- No order placement.
- No BAG order built / previewed / submitted.
- No live authorization.
- No `systemctl daemon-reload` / restart / start / stop.
- `chad-live-loop.service` remains `active (running)`.
- `runtime/live_readiness.json` `ready_for_live` remains `false`.
- HEAD invariant `bbe7525` before/after.

---

## 7. Cross-references

- Box 051 (Official) — BAG per-share / contract-dollar unit normalization:
  `runtime/completion_matrix_evidence/BOX-051_OFFICIAL_BAG_Tier_3C_limit_price_unit_normalization.md`
- Box 052 (Official) — BAG adapter quote enforcement:
  `runtime/completion_matrix_evidence/BOX-052_OFFICIAL_BAG_adapter_quote_enforcement.md`
- Box 053 (Official) — BAG bracket/OCA or fail-safe exit:
  `runtime/completion_matrix_evidence/BOX-053_OFFICIAL_BAG_bracket_OCA_or_fail_safe_exit.md`
- IBKR adapter idempotency contract: `chad/tests/test_ibkr_idempotency_key_stability.py` (BAG section).
- BAG meta preservation contract: `chad/tests/test_alpha_options_meta_preservation.py`.
- BAG paper-fill simulator contract: `chad/tests/test_alpha_options_bag_paper_fill.py`.
