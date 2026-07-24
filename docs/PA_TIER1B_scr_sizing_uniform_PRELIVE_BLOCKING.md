# PA TIER1-B — SCR sizing factor applied uniformly across ALL states

## ⛔ PRE-LIVE BLOCKING — MUST be closed before `ready_for_live` is ever discussed

**Class:** decorative control (published-but-unapplied risk parameter) — the
exact defect class this month's campaign exists to eliminate.
**Posture at authoring:** PAPER / SCR=WARMUP / `ready_for_live=false`.
**Runtime mutation:** none. Repo-side only; activates on the next gated
live-loop restart (operator-controlled).

---

## The defect

The SCR governor (`chad/analytics/shadow_confidence_router.py`) publishes a
`sizing_factor` to `runtime/scr_state.json` for **every** state:

| SCR state | published `sizing_factor` |
|-----------|---------------------------|
| WARMUP    | **0.10** |
| CAUTIOUS  | 0.25 |
| CONFIDENT | 1.00 |
| PAUSED    | 0.00 (hard-block) |

The execution path (`chad/core/live_loop.py`) applied that factor to intent
quantity **only when the state was `CAUTIOUS`**:

```python
if _scr_state_val == "CAUTIOUS" and _scr_sizing > 0.0:   # ← the bug
```

In WARMUP — the current live state — `0.10` was published, displayed on the
dashboard as "Trading at 10% size", and applied to **no order quantity**. Paper
intents rode at 100% size while every surface claimed 10%. A published risk
throttle that no code path enforces is a decorative control.

Not live-risk *today* (WARMUP ⇒ `paper_only`), but it is a false risk-control
that would silently carry into any state transition, and it corrupts the paper
sizing that feeds SCR/Stage-2 evidence. Hence **PRE-LIVE BLOCKING**.

## The fix (this PA)

1. **`chad/risk/scr_sizing.py`** (new) — single source of truth:
   - `scr_sizing_should_apply(state, factor)` — policy predicate: apply iff
     enabled, non-PAUSED, and `0 < factor < 1.0` (1.0 is a no-op).
   - `apply_scr_sizing(raw, factor, sec_type)` — scaling math, extracted
     **verbatim** from the former CAUTIOUS block (FUT round·min1, else
     floor·min1). CAUTIOUS behaviour is byte-identical.
   - `scr_sizing_apply_enabled()` — kill-switch `CHAD_SCR_SIZING_APPLY`,
     **default ON**. `0/false/no/off` disables (and the sentinel then reports
     the factor as decorative).

2. **`chad/core/live_loop.py`** — the in-loop scaler now gates on
   `scr_sizing_should_apply(...)` instead of the `== "CAUTIOUS"` literal, so the
   published factor is applied in **every** non-PAUSED throttled state (WARMUP
   included). PAUSED is still hard-blocked above; CONFIDENT (1.0) is a no-op.

3. **Sentinel-visible assertion (EX021)** — the live loop writes
   `runtime/scr_sizing_application.v1` once per cycle, recording the state, the
   published factor, `application_enabled`, and `applied` — the latter computed
   from the **same** `scr_sizing_should_apply` predicate the scaler uses, so the
   applied path and the evidence marker cannot drift. New Exterminator check
   `check_scr_sizing_applied` (`chad/ops/exterminator.py`) reads the marker and:
   - `application_enabled == false` while a sub-unity factor is published →
     **CRITICAL** "SCR sizing factor is DECORATIVE".
   - `applied == false` (policy did not apply a published throttle) →
     **CRITICAL** "published but NOT applied".
   - marker absent → WARNING (unverifiable); marker factor ≠ canonical → WARNING
     (stale).

   A future re-narrowing of the gate flips the marker's `applied` to false and
   trips CRITICAL — a published-but-unapplied factor **can never recur
   silently**.

## Behavioural change on activation

- **WARMUP** (current state): paper intent quantities scale to 10% (they rode at
  100% before). This changes paper fills / paper P&L and the SCR evidence built
  from them — intended: the published throttle becomes real.
- **CAUTIOUS:** byte-identical (same predicate result, same math).
- **CONFIDENT/PAUSED:** unchanged (no-op / hard-block).

## Tests

`chad/tests/test_tier1_scr_sizing_uniform.py` (15 cases): per-state predicate,
kill-switch, equity/futures math, CAUTIOUS byte-identical regression,
intent-quantity-per-state (incl. WARMUP-no-longer-full-size), a **source pin**
that fails if the CAUTIOUS-only literal returns, and the six EX021 sentinel
branches.

## Operator decision required at review — the score-ladder amounts are fiction

Separate from this fix (flagged, not touched here): the dashboard's cosmetic
score ladder (`_next_level`, `chad/dashboard/api.py`) advertises sizing amounts
that **do not match the real governor**:

- Card "Confident" = **50%**; real `CONFIDENT` `sizing_factor` = **100%**.
- Card "Full Send" (100%) and "Max" rungs are **phantom** — no such SCR states
  exist (real ladder is WARMUP/CAUTIOUS/CONFIDENT/PAUSED).

TIER1-A anchors the "Next:" *label* to the state, but the *amounts* remain
fictional. **Align to the governor (10/25/100, drop Full Send/Max) or delete the
amount claims — operator's call at review time.**

## Activation / rollback
- Activation: next gated live-loop restart (operator). No config change needed.
- Kill-switch: `CHAD_SCR_SIZING_APPLY=0` disables application (sentinel will then
  flag it CRITICAL by design).
- Rollback: `git revert` the TIER1-B commit(s); no runtime state to unwind.
