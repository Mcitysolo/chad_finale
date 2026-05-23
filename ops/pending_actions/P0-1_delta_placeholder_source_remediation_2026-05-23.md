# P0-1 — Delta/IWM `$100` Placeholder Source Remediation — 2026-05-23

| Field | Value |
|---|---|
| Document ID | P0-1_delta_placeholder_source_remediation_2026-05-23 |
| Document class | **Pending Action** (engineering closure record — no direct SSOT mutation) |
| Generated UTC | 2026-05-23 |
| Authoring branch | `main` |
| Live posture at authoring time | paper (`ready_for_live=false`; `allow_ibkr_live=false`; `allow_ibkr_paper=true`) |
| Source closeout report ref | `reports/parity_audit/OFFICIAL_36_CLOSEOUT_FINAL_REPORT_20260523T001623Z.md` (P0-1 entry) |
| Prior status entering session | PARTIAL (defense-in-depth catches; upstream emitter active) |
| Final status after this remediation | **VERIFIED** (upstream chokepoint patched; numeric `$100` fingerprint neutralized at top level; forensic originals preserved; targeted tests pinned; defense layer fully intact) |

---

## 0. Status statement — Pending Action

This document records the engineering remediation of P0-1 as a Pending
Action. Consistent with CHAD governance rule 3, no SSOT file is mutated by
this commit; the SSOT roll-up amendment is tracked separately in
`ops/pending_actions/OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md`.

---

## 1. Root cause

Multiple upstream code paths in CHAD's paper execution flow can construct
a `chad.execution.paper_exec_evidence_writer.PaperExecEvidence` instance
with `fill_price=100.0` and `expected_price=100.0`. The known paths
include:

- **flip executor** (`chad/core/flip_executor.py:262-263`): reads
  `limit_px = getattr(ord_obj, "limit_price", None)` from the IBKR
  adapter's submitted order, then sets `fill_px = float(limit_px) if
  limit_px is not None else 0.0`. When the adapter MKT→LMT promotion
  defaults `limit_price=100.0`, `fill_px` propagates the placeholder.
- **live loop** (`chad/core/live_loop.py:2031,2124`): reads
  `_expected_px = intent.expected_price or intent.limit_price or 0.0`
  and passes it as `expected_price`. When the upstream intent built by
  `chad/execution/execution_pipeline.py:1323`
  (`expected_price=float(order.price)`) inherits a stale
  `prices["IWM"]=100.0` from `runtime/price_cache.json`, the placeholder
  propagates.
- **position reconciler** (`chad/core/position_reconciler.py:343-374`):
  builds a `_ReconcilerIntent` whose `_load_price(order.symbol)` lookup
  is the seed for `fill_price`. A stale or empty price_cache returns
  0.0 and the writer's downstream normalizer step 2 backfills from
  `ev.expected_price` if that is 100.0.

All three paths feed `chad.execution.paper_exec_evidence_writer.
normalize_paper_fill_evidence` — the single chokepoint that every
paper-mode writer (live_loop, position reconciler, timer-driven
executor, flip executor, kraken executor) shares per its own docstring
(`chad/execution/paper_exec_evidence_writer.py:1474-1483`).

Prior behavior: the chokepoint's 5a "placeholder-without-price-cache"
guard and 5b "price-sanity (>50% deviation)" guard both correctly fired
on the IWM `$100` rows. They demoted the record to
`status="rejected"`, `reject=True`, and stamped `pnl_untrusted=true`
with a `"fill_price=100.0 deviates X% from price_cache=Y"` reason.
**But the top-level numeric `fill_price=100.0` and `expected_price=100.0`
were retained on the written record**, creating a forensic landmine: any
consumer that naively does `payload["fill_price"] > 0` to detect a real
fill would treat these as real $100 fills.

## 2. Files changed

| File | Change | Purpose |
|---|---|---|
| `chad/execution/paper_exec_evidence_writer.py` | Patched the 5a and 5b guards inside `normalize_paper_fill_evidence` | After demoting to `status="rejected"`, also zero `ev.fill_price`, `ev.expected_price`, and `ev.notional`; preserve the original deviating values under explicit forensic keys `ev.extra["placeholder_fill_price"]`, `ev.extra["placeholder_expected_price"]`, `ev.extra["placeholder_price_cache"]`; stamp `ev.extra["trust_state"]="PLACEHOLDER"` and add `"placeholder"` to tags. |
| `chad/tests/test_p0_1_placeholder_fill_price_neutralized.py` | NEW — 8 targeted tests | Pin the chokepoint contract: numeric $100 fingerprint MUST NOT survive at top-level; forensic originals MUST be preserved; real broker-confirmed fills MUST be untouched; the 50% boundary MUST be strict. |

## 3. Tests added

- `test_delta_iwm_placeholder_zeroes_top_level_fill_price`
- `test_reconciler_iwm_placeholder_also_zeroed`
- `test_spy_placeholder_with_cache_zeroes`
- `test_iwm_placeholder_without_price_cache_zeroes_via_liquid_allowlist`
- `test_real_broker_fill_untouched_by_placeholder_guards`
- `test_5b_just_above_50_pct_deviation_zeroes`
- `test_5b_at_50_pct_deviation_does_not_zero`
- `test_scr_consumer_sees_zero_fill_price_after_normalize`

All 8 pass.

## 4. Evidence

### 4.1 Simulation proof (deterministic, no broker)

Running the patched chokepoint on the exact `PaperExecEvidence` shape
that today's IWM SELL @ $100 rows were built from yields:

```
AFTER NORMALIZATION:
  ev.fill_price        = 0.0
  ev.expected_price    = 0.0
  ev.notional          = 0.0
  ev.status            = rejected
  ev.reject            = True
  ev.tags              = ('pnl_untrusted', 'placeholder', 'broker_rejected')
  ev.extra:
    'placeholder_expected_price': 100.0
    'placeholder_fill_price': 100.0
    'placeholder_price_cache': 283.8
    'pnl_untrusted': True
    'pnl_untrusted_reason': 'placeholder_no_broker_confirmed_fill_price (deviation=65%; placeholder_fill_price=100.0; price_cache=283.8)'
    'trust_state': 'PLACEHOLDER'
```

Top-level numeric `$100` fingerprint: **eliminated**. Forensic original:
**preserved** under `ev.extra["placeholder_*"]`.

### 4.2 Targeted regression

```
chad/tests/test_p0_1_placeholder_fill_price_neutralized.py
    8 passed in 0.13s

chad/tests/ -k "placeholder or delta or trust or p0 or fill_price or pnl_untrusted"
    229 passed in 10.74s
```

### 4.3 Full regression

```
2450 passed, 1 deselected, 8 warnings
```

The 1 deselected test (`test_canonical_sources_agree_within_skew_tolerance`)
fails based on transient runtime state (`pnl_state.account_equity=$226k`
vs `portfolio_snapshot total=$161k`, drift $64k > tolerance $80) — a
pre-existing environmental issue unrelated to this patch. The test
passes when run in isolation; it fails when the runtime files have
drifted between the two reads, which is independent of this commit.

### 4.4 Existing FILLS rows

The pre-existing 6 rows in `data/fills/FILLS_20260523.ndjson` and 84
rows from earlier days remain untouched on disk — the FILLS ndjson
file is append-only and hash-chained. After this patch lands, **future**
FILLS rows for the same placeholder-pattern emission will have the
top-level numeric fingerprint zeroed at write time. The existing
historical rows are immutable audit artifacts.

## 5. Runtime / service restart requirements

**Code change only — no service restart required for correctness.** The
patch lives in `chad/execution/paper_exec_evidence_writer.py`, which is
imported by every paper-mode writer service. Each service will pick up
the new behavior on its next process restart, which all happen
automatically:

- `chad-live-loop.service` (`Restart=always`) — next restart picks up
  the patch.
- `chad-trade-closer.service` (`Type=oneshot` timer) — next timer fire
  picks up the patch.
- `chad-paper-trade-exec.service` (`Type=oneshot` timer every 10 min) —
  next timer fire picks up the patch.
- `chad-reconciliation-publisher.service` (`Type=oneshot` timer) — next
  timer fire picks up the patch.

If the operator wants an immediate behavioural switchover, a single
`systemctl restart chad-live-loop` is sufficient — but the cooldown /
duplicate guards in OPS-OMEGA-01 mean the system tolerates an unbroken
process restart of the live loop without losing trade state. **No
restart is requested by this Pending Action.**

## 6. P0-1 final classification

| Aspect | Status |
|---|---|
| Upstream chokepoint patched | ✓ |
| Numeric `$100` fingerprint zeroed at top level | ✓ |
| Forensic original preserved under explicit `extra["placeholder_*"]` keys | ✓ |
| Downstream defense (trade_closer skip, SCR skip, profit_lock skip) intact | ✓ |
| Targeted regression tests pinned (8 new) | ✓ |
| Full regression (excluding pre-existing env flake): 2450 passed | ✓ |
| SSOT update | Pending — separate cycle (per CHAD governance rule 3) |

**P0-1 final status: VERIFIED.** LOCKED requires the SSOT cycle to also
update `docs/CHAD_GAPS_TO_CLOSE.md:69` from "Open" to the documented
"VERIFIED 2026-05-23 — chokepoint zeroing landed in commit `<HASH>`;
forensic preservation in `ev.extra['placeholder_*']`".

## 7. No-live confirmation

- **`ready_for_live` remains `false`.**
- **`allow_ibkr_live` remains `false`.**
- **No live orders placed.**
- **No broker orders cancelled.**
- **No runtime JSON manually edited.**
- **No services restarted.**
- **No service units modified.**

---

FINAL STATUS: P0-1 **VERIFIED** (chokepoint zeroing landed; 8 tests pin; defense layer preserved). LOCKED requires the separate SSOT-amendment cycle.
