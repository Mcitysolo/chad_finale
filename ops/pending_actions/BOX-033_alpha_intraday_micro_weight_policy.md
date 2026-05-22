# BOX-033 — alpha_intraday_micro weight policy (Pending Action)

Generated: 2026-05-20T00:30:00Z
Origin: Box 033 strategy registry reconciliation

## Background

The strategy registry reconciliation in Box 033 confirms one explicit
mismatch between the active strategy registry and the canonical allocator
weights:

- `chad/strategies/__init__.py` registers `ALPHA_INTRADAY_MICRO` as
  active (`_REGISTRY` includes the handler at line 240).
- `chad/risk/tier_manager.py::_CANONICAL_STRATEGY_NAMES` (line 95)
  includes `alpha_intraday_micro`.
- `chad/types.py::StrategyName.ALPHA_INTRADAY_MICRO` (line 67) is in the
  enum.
- `chad/strategies/DEFERRED_STRATEGIES` does NOT include
  `alpha_intraday_micro` (only `alpha_forex`).
- BUT `config/strategy_weights.json` has only 16 weights and does NOT
  contain a key for `alpha_intraday_micro`.

Effect: `chad.risk.dynamic_risk_allocator` and `chad.risk.allocator_v3`
load weights from `config/strategy_weights.json` and filter to "eligible
strategies … present in base weights, base weight > 0". A registered
strategy with no weight entry is silently dropped at the allocator gate
even if it emits signals — i.e., the strategy is **soft-deferred**
without that state being captured anywhere authoritative.

## SSOT context

`docs/CHAD_UNIFIED_SSOT_v9.1_2026-05-13.md` line 1285 documents this
state as an open residual:

> "alpha_intraday_micro registered but conditionally inactive at runtime.
>  The handler is in the canonical registry (chad/strategies/__init__.py:240)
>  and is invoked each cycle by live_execution_router._build_available_signals
>  → iter_strategy_registrations. However, the handler short-circuits to []
>  when ctx.tier_profile is not a TierRiskProfile instance. ContextBuilder
>  must populate tier_profile and tier_name on MarketContext for this
>  strategy to emit signals. **Status: REQUIRES CONTEXT-BUILDER WIRING
>  VERIFICATION** before MICRO/STARTER/PRO_GROWTH testing."

The Box-033 parity-test allowlist (`chad/tests/test_strategy_registry_parity.py::WEIGHT_DEFERRED_ALLOWLIST`)
records `alpha_intraday_micro` as an audited soft-deferred entry pending
operator decision below.

## Operator decision required

Choose ONE:

### Option A — Formally activate with an allocator weight

1. Edit `config/strategy_weights.json`:
   - Decide an allocation percentage (e.g. carve a small slice from
     `alpha_intraday` since they share intent class).
   - Add `"alpha_intraday_micro": <value>` to the `weights` dict.
   - Renormalize remaining weights so the dict sums to 1.0.
   - Update `applied_action_id`, `previous_weights`, `reason`, and
     `updated_ts_utc`.
2. Confirm `ContextBuilder` populates `tier_profile` and `tier_name` on
   `MarketContext` (SSOT v9.1 residual line 1285 prerequisite).
3. Remove `"alpha_intraday_micro"` from
   `WEIGHT_DEFERRED_ALLOWLIST` in `chad/tests/test_strategy_registry_parity.py`.
4. Re-run the Box-033 parity tests.
5. Channel-1: restart `chad-live-loop.service` so the allocator picks
   up the new weights.

### Option B — Formally defer (no allocator weight; no handler)

1. Edit `chad/strategies/__init__.py`:
   - Add `StrategyName.ALPHA_INTRADAY_MICRO` to `DEFERRED_STRATEGIES`.
   - Comment out the `_build_registry` entry for ALPHA_INTRADAY_MICRO
     (mirror the existing alpha_forex pattern at lines 195–198).
   - Update the module docstring listing.
2. Remove `"alpha_intraday_micro"` from `WEIGHT_DEFERRED_ALLOWLIST`.
3. Re-run the Box-033 parity tests.
4. Channel-1: restart `chad-live-loop.service` so the registry change
   takes effect.

### Option C — Hold the current soft-deferred state

If neither A nor B is operator-approved yet, the current state is
acceptable as the explicit "audited soft-deferred" state recorded by the
Box-033 parity test allowlist. Box 033 closes on this status. No code
or config change required until A or B is selected.

## Risk assessment

- **Option A risk**: weight reallocation changes sizing across all
  strategies (because allocator normalizes). Requires careful operator
  review of allocation impact. Live trading remains gated by
  `live_readiness.json`.
- **Option B risk**: minimal — alpha_intraday_micro is already
  effectively zero-weighted; formal defer just makes it deterministic.
  No allocator sizing change.
- **Option C risk**: none for closure; preserves status quo with
  explicit audit trail.

## Recommendation

Default to **Option C (hold)** until operator confirms whether
ContextBuilder tier_profile wiring is in place. If tier wiring is not
verified, prefer **Option B (formal defer)** to remove the registered-
but-can't-emit confusion. **Option A (activate)** only when both
wiring and a deliberate allocation decision are in hand.

## Status

- Recorded: 2026-05-20T00:30:00Z (Box 033)
- Decision pending operator review
- Box 033 closure relies on Option C (current state explicitly captured
  by WEIGHT_DEFERRED_ALLOWLIST + this pending-action artifact)
- This artifact will be deleted (and the allowlist entry removed) when
  Option A or B is applied.
