# W2A Item-2 build-blocker (D7) — PROVEN: UNH re-attribution creates no over-count

**Decision D7:** *build-step 1 proves (test or sandbox replay) that re-attribution will NOT
leave a `broker_sync|UNH` duplicate summing to 451. No script until proven.*

This document is that proof. The executable half is
`chad/tests/test_w2a_reattribution_guard_dedup.py` (all assertions below are pinned there,
using the real reader functions — not a re-implementation).

---

## The concern

Item-2 re-attributes the real +228 UNH long to `gamma` by writing an OPEN `gamma|UNH` lot
(qty 228, trusted basis) into `runtime/trade_closer_state.json`, from which the position guard
is rebuilt. The guard **already** carries a `broker_sync|UNH` mirror (broker truth, 228). The
fear: a reader that SUMS the strategy entry and its broker_sync mirror would see
`228 + 228 = 456` (the plan wrote "451" against the older broker qty of 223) — a brand-new
2.0× phantom.

## Live starting state (verified 2026-07-20, runtime/position_guard.json)

| key | open | side | qty | source |
|-----|------|------|-----|--------|
| `gamma|UNH` | **false** (CLOSED) | SELL | 223 | paper_ledger_rebuild / position_reconciler |
| `broker_sync|UNH` | true | BUY | 228 | broker_truth_rebuild |

`runtime/trade_closer_state.json`: **no** `gamma UNH` queue (empty) → item-2's idempotency
precondition ("gamma UNH FIFO empty, no `gamma|UNH` lot") holds. `broker_sync UNH` FIFO = 143.

## Cycle mechanics (chad/core/live_loop.py, paper mode)

Per cycle (lines 1636-1647): `_rebuild_guard_from_paper_ledger` **then**
`_rebuild_guard_from_broker`.

1. `_rebuild_guard_from_paper_ledger` (L797-838): for the new `gamma|UNH` queue it nets the
   matching `broker_sync|UNH` down by the strategy qty (ISSUE-56 v2, L802-822: same side,
   residual `228 − 228 = 0 ≤ 0` → broker_sync soft-closed `strategy_ownership_assumed`), then
   writes `gamma|UNH` open BUY 228.
2. `_rebuild_guard_from_broker` (L1352-1377): **unconditionally** re-opens `broker_sync|UNH` =
   228 from live IBKR truth (second loop runs regardless of the soft-close above).

**End state after any full cycle:** `gamma|UNH` open 228 **and** `broker_sync|UNH` open 228.
So the two keys DO coexist. The question is only whether any reader sums them.

## The codebase-wide invariant: the guard DUAL-BOOKS; readers compare, never sum

A `<strategy>|<sym>` entry and its `broker_sync|<sym>` mirror represent the **same** shares.
This is stated in the source and enforced by every reader:

- `chad/core/position_guard.py:817` — `_agg_guard_strategy` docstring: *"NEVER summed with the
  broker_sync mirror (the guard dual-books the same shares — summing invents a 2.0x phantom)."*
- `position_guard.py:856-858` — v4 detector: *"the legs are NEVER summed (the guard dual-books
  `gamma|UNH 273` AND `broker_sync|UNH 273` for one 273-share position; summing invents a 2.0×
  phantom on every mixed symbol)."*

### Reader 1 — `detect_guard_vs_broker_drift_v2` (the position_guard_drift emitter)

Aggregates `guard_qty[sym]` = signed sum of OPEN non-broker_sync entries, and
`broker_qty[sym]` = signed sum of `broker_sync|` entries, then compares **symbol-total vs
symbol-total** (L677-726). It never adds the two.

| state | guard_qty[UNH] | broker_qty[UNH] | classification |
|-------|----------------|-----------------|----------------|
| **today** (gamma closed) | 0 | +228 | `broker_untracked_position` (**actionable drift**) |
| **after re-attribution** | +228 | +228 | equal → **no drift** |

Re-attribution therefore **removes** an actionable drift; it never produces a 456 mismatch.

### Reader 2 — `detect_guard_vs_independent_snapshot_drift` (v4, three-leg)

Compares three separate aggregates — independent broker (positions_snapshot), guard broker
mirror, guard strategy — each **against the independent broker leg**, never each other, never
summed (L913-960). After re-attribution with independent UNH=228: broker=228, mirror=228,
strategy=228 → `mirror_delta=0`, `strat_delta=0` → `continue` (both agree) → **no drift**.
Neither leg is ever 456.

### Reader 3 — the exit overlay (item-2's consumer)

`chad/risk/position_exit_overlay.py:633` (and `:866` in `run_cycle`): the enumeration
**skips** any `broker_sync|` key. So the overlay manages only `gamma|UNH`; it never
double-manages the mirror. Its reduce-only cross-check reads broker truth via
`_broker_signed_by_symbol` (L403/627) = `{UNH: 228}`, and a close is capped at
`min(open_qty, broker_held) = min(228, 228) = 228` (L758) — **never 456**, no oversell. This
is exactly the outcome item-2 wants: gamma + the overlay finally SEE and MANAGE the 228.

## Conclusion

No reader sums the strategy entry and its broker_sync mirror; all three compare like-with-like.
Re-attributing `gamma|UNH = 228 (open, BUY, trusted)`:

- **removes** the current `broker_untracked_position` drift (drift_v2), and
- leaves all three v4 legs in agreement at 228, and
- lets the exit overlay manage `gamma|UNH` reduce-only, capped at broker truth 228.

**Corollary — item-2 must NOT clear `broker_sync|UNH`.** It is the broker-truth anchor every
reader uses, and `_rebuild_guard_from_broker` re-creates it every cycle regardless; clearing it
would be both futile and wrong. The plan's fallback ("script must additionally clear any stale
`broker_sync|UNH`") is therefore **not required** — proven here. Item-2 writes exactly one
`gamma|UNH` lot and the matching guard entry; it touches no broker_sync key.

**D7 satisfied → the re-attribution script (W2A-4) is unblocked.**
