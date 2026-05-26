# PR-02b — Reconciler Upstream $100 Placeholder Fix

**Date:** 2026-05-25
**Branch:** `main` (commit pending; see Tier 6)
**Author:** Team CHAD
**Governance rule references:** §1 (one-change-at-a-time), §3 (no direct
config mutation), §4 (verification sequence after every change).

---

## 1. Scope

Silence the second of the two upstream `fill_price=100.0` producers
identified in the 2026-05-22 placeholder audit. PR-02 (commit `139d275`)
closed the *delta* strategy producer; PR-02b closes the *reconciler*
producer in `chad/core/position_reconciler.apply_close_intents`.

No other behaviour is intentionally changed. PR-09 (`2d454ed`) and the
GAP-058 / P0-1 writer-level defense gates remain unchanged and are
regression-locked by the new tests.

---

## 2. Root Cause

`chad/core/position_reconciler.py:343` (`apply_close_intents`) called
the legacy `_load_price(order.symbol)` helper to seed the synthesized
close fill's `fill_price`. `_load_price` reads
`runtime/price_cache.json` **without a TTL freshness check**. When the
price cache was stale, missing the symbol, or briefly unavailable, the
synthesized `PaperExecEvidence` carried `fill_price=0.0` /
`expected_price=0.0`. Downstream, the writer's
`normalize_paper_fill_evidence` re-resolved the price via
`_lookup_paper_fill_price` / `expected_price`, and in the incident
window of 2026-05-16..05-23 produced a `strategy="reconciler"` row with
the canonical `fill_price=100.0` placeholder on IWM and SPY.

The 10-day pre-fix audit measured:

```
PRE_FIX_RECONCILER_BASELINE (last 10 days):
  total reconciler fills: 82
  reconciler $100 fills by symbol: {'SPY': 30, 'IWM': 42}
```

72 of 82 reconciler-tagged fills were $100 placeholders. The
GAP-058 / P0-1 defense gates correctly demoted them to
`status=rejected, pnl_untrusted=true` — but the upstream emission
itself pollutes `data/fills/FILLS_*.ndjson` and prevents the PO-03
zero-placeholder-day clock from starting.

---

## 3. Files Changed

Exhaustive list (no other paths touched):

| File | Nature of change |
|------|-------------------|
| `chad/core/position_reconciler.py` | Added `_load_fresh_cache_price`, `_load_recent_broker_fill_price`, `_resolve_close_fill_price`; replaced `_load_price(order.symbol)` callsite in `apply_close_intents` with the new resolver + abstain branch. |
| `chad/tests/test_position_guard_close_confirmation.py` | One-line update: monkey-patch target moved from `_load_price` to `_resolve_close_fill_price` to keep the ISSUE-29 reject-skip regression coverage intact. |
| `chad/tests/test_pr02b_reconciler_upstream_placeholder.py` | **New** — T1..T6 regression-lock suite. |
| `ops/pending_actions/PR-02b_reconciler_upstream_placeholder_fix_2026-05-25.md` | **New** — this file. |

The legacy `_load_price` helper is **intentionally left callable** for
back-compat; it is no longer on the production code path.

---

## 4. Behaviour Change

`apply_close_intents` now resolves a close-fill price via a strict
cascade with explicit abstain:

1. **Tier 1 — broker-confirmed fills.** Walk the most-recent two daily
   `data/fills/FILLS_*.ndjson` files for the latest non-rejected,
   not-`pnl_untrusted`, strictly-positive fill matching the symbol.
2. **Tier 2 — fresh `runtime/price_cache.json`.** Read the cache,
   parse `ts_utc`, compute age, and use the symbol's price only when
   `age <= ttl_seconds` (default 300s).
3. **Tier 3 — abstain.** Log
   `RECONCILER_CLOSE_ABSTAIN_NO_PRICE symbol=… side=… qty=… reason=no_broker_fill_and_no_fresh_price_cache — evidence write skipped; guard remains open`
   and `continue` the outer loop without writing any
   `PaperExecEvidence`. The existing ISSUE-29 positive-confirmation
   gate keeps the position guard open, so the next reconciler cycle
   retries naturally.

The `100.0` magic constant is never substituted at any tier.

---

## 5. Tests Added (regression-lock)

`chad/tests/test_pr02b_reconciler_upstream_placeholder.py`:

| ID | Test | Asserts |
|----|------|---------|
| T1 | `test_t1_reconciler_abstains_when_no_real_price_available` | No broker fills + no cache → resolver returns 0.0 AND `apply_close_intents` writes no evidence AND logs `RECONCILER_CLOSE_ABSTAIN_NO_PRICE`. |
| T2 | `test_t2_reconciler_uses_fresh_price_cache_not_100` | Fresh cache with `IWM=283.8` → resolver returns 283.8, not 100.0. |
| T2b | `test_t2b_reconciler_rejects_stale_cache` | Cache age > ttl → `_load_fresh_cache_price` returns 0.0 AND cascade abstains. |
| T3 | `test_t3_reconciler_uses_last_broker_fill_when_cache_stale` | Stale cache + real broker fill at 285.20 → resolver returns 285.20 (tier-1 wins). |
| T3b | `test_t3b_broker_fill_resolver_rejects_untrusted_and_rejected` | Tier-1 ignores `reject=True`, `extra.pnl_untrusted`, and `fill_price <= 0` rows. |
| T4 | `test_t4_writer_defense_gate_still_demotes_100_dollar_placeholder` | GAP-058 / P0-1 writer-level zeroing for `fill_price=100.0` is intact (regression lock). |
| T5 | `test_t5_pr02_delta_resolve_positive_price_still_rejects_invalid` | PR-02 delta `_resolve_positive_price` still rejects None/0/negative/NaN/Inf (regression lock for commit 139d275). |
| T6 | `test_t6_pr09_positions_truth_exposes_broker_authority_fields` | PR-09 `runtime/positions_truth.json` schema still exposes `broker_authority_status` and `replay_diagnostic_status` (regression lock for commit 2d454ed). |

Targeted run (PR-02b / reconciler / placeholder / PR-02 / PR-09):
**73 passed** in 3.40s.
Full regression: **2492 passed**, 8 warnings, in 109.74s.

---

## 6. Runtime Evidence

Pre-fix audit of last 10 daily `FILLS_*.ndjson` files:

```
total reconciler fills:           82
reconciler $100 fills by symbol:  {'SPY': 30, 'IWM': 42}   # 72 of 82
```

Post-fix expectation (forward-looking once the live-loop process
reloads the new code — see §10 Deferred): zero new reconciler-tagged
rows with `fill_price=100.0`. When `_resolve_close_fill_price` returns
0.0, the abstain branch fires and the evidence write is skipped
entirely, so neither the `100.0` nor `0.0` fingerprint enters
`FILLS_*.ndjson` under `strategy="reconciler"`.

The legacy demoted rows already in the ledger remain (audit chain
preserved); PR-02b is a forward fix, not a back-fill.

---

## 7. Safety Envelope (Tier 4)

```
ready_for_live          = False
truth_ok                = True
broker_authority_status = GREEN
replay_diagnostic_status = PARTIAL
replay_diagnostic_blocks_truth = False
stop_bus_active         = False
```

No runtime JSON was edited. No broker orders placed or cancelled. No
service restarts. No live enablement. `ready_for_live` remains
`False`.

---

## 8. Contract Preservation Checklist

- [x] **PR-02 delta abstain** (commit `139d275`) — file untouched;
  T5 smoke import + invalid-price reject passes.
- [x] **PR-09 broker-authority / replay-diagnostic separation**
  (commit `2d454ed`) — file untouched; T6 smoke read passes; live
  `runtime/positions_truth.json` still exposes both fields.
- [x] **GAP-058 / P0-1 writer defense gate** (commits `d924eea`,
  `80f44da`, `ddbc79f`) — `chad/execution/paper_exec_evidence_writer.py`
  untouched; T4 regression lock passes.
- [x] **ISSUE-29 positive-confirmation guard gate** — the new abstain
  branch happens *before* the post-confirmation guard mutation, so
  the existing "no confirmed fills → guard stays open" semantics are
  unchanged.
- [x] **GAP-001 corrected-scope exclusion chokepoint** — the
  excluded-symbol skip at `apply_close_intents` line 307 fires
  *before* the resolver, so excluded symbols never reach the new
  cascade.

---

## 9. Operator Notes

- **No-live confirmation:** `CHAD_EXECUTION_MODE` remains `paper`;
  `ready_for_live=False`; no operator action required.
- **Branch invariant:** the change is surgical — one file in
  production, one new test file, one one-line test update, this PA.
  No refactors, no renames, no API surface changes.

---

## 10. Deferred

- **Production observation lock requires a live-loop restart.**
  The fix is on disk; the running `chad-live-loop` service still has
  the pre-fix `apply_close_intents` cached in memory. Confirming
  zero new `fill_price=100.0` reconciler rows in
  `data/fills/FILLS_$(date +%Y%m%d).ndjson` after the live-loop
  reloads the new code is out of scope for PR-02b (same pattern as
  PR-02). Operator authorisation required before
  `systemctl restart chad-live-loop` — see governance §1, §7.

---

## 11. Verification Commands (idempotent, read-only)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
export PYTHONPATH=/home/ubuntu/chad_finale CHAD_SKIP_IB_CONNECT=1

# Compile the touched module.
python3 -m py_compile chad/core/position_reconciler.py

# Targeted scope (8 PR-02b + PR-02/PR-09/GAP-058 regression locks):
python3 -m pytest chad/tests/ -k "pr02b or reconciler or placeholder or pr02 or pr09" -v | tail -10

# Full regression (expect 2492 passed):
python3 -m pytest chad/tests/ -q | tail -5

# Safety envelope (must remain ready_for_live=False / GREEN / inactive):
python3 -c "import json; print('ready_for_live=', json.load(open('runtime/live_readiness.json')).get('ready_for_live'))"
python3 -c "import json; d=json.load(open('runtime/positions_truth.json')); print({k:d.get(k) for k in ('truth_ok','broker_authority_status','replay_diagnostic_status','replay_diagnostic_blocks_truth')})"
python3 -c "import json; print('stop_bus_active=', json.load(open('runtime/stop_bus.json')).get('active'))"
```

---

## PO-03 success-criterion closure (appended 2026-05-26)

Per `PO-03_zero_public_placeholder_fingerprint_success_2026-05-26.md`, the
operator's binding success criterion for PO-03 is:

> "Zero public `fill_price=100.0` fingerprint AND zero trusted fake placeholder
> evidence" — placeholder-tagged rows that are rejected, untrusted, public-price-scrubbed,
> and writer-quarantined are explicitly NOT a paper-complete blocker.

Under that criterion, observed evidence since 2026-05-26T14:45:17Z:

- `public_fill_price_100_since_start = 0`
- `trusted_fake_placeholder_since_start = 0`
- `placeholder_tagged_rows_since_start = 21` (delta=14, reconciler=7; all `status=rejected`,
  `reject=true`, `pnl_untrusted=true`, public `fill_price` = real cache price, never `100.0`)

**PR-02b status for paper-complete: VERIFIED.** The reconciler-level synthesized
close-fill placeholder emission was silenced at the reconciler layer (commit
`5c5507e`); the public `$100` fingerprint is zero; downstream PnL / SCR / trade
evidence consumers are protected by `status=rejected` + `pnl_untrusted=true` + tags.

The "zero placeholder-tagged rows" stricter goal is recorded as DEFERRED
hardening (executor-layer placeholder emitter trace) — non-blocking for
paper-complete. See `PO-03_zero_public_placeholder_fingerprint_success_2026-05-26.md` §5.

No live posture change. `ready_for_live=false`, `allow_ibkr_live=false`,
`allow_ibkr_paper=true` preserved at declaration time.
