# PA TIER1-A — dashboard "Next:" tracks SCR state, not the runaway score

**Class:** cosmetic display bug (dashboard only; no execution/risk path).
**Posture:** PAPER. **Runtime mutation:** none. Repo-side only.

## The defect

The Training card's "Next:" line is computed by `_next_level(score)` where
`score = _sharpe_to_score(sharpe_like)` (`chad/dashboard/api.py`). The score is
a 0–100 mapping of `sharpe_like`, **decoupled** from the SCR state machine
(whose real WARMUP→CAUTIOUS gate is `effective_trades ≥ 100`, not a sharpe
score).

When `sharpe_like` rose to 0.444 (**score 53**), `_next_level` returned the next
rung above the *score* — "Confident — Unlocks 50% size at score 60" — while the
governor is still **WARMUP** (10% size). The score had outrun the state, so the
card **skipped the real next rung** (Cautious/25%) and showed the rung after it.
Two days earlier, `sharpe_like < 0.15` (score < 30) so it correctly read
"Cautious"; the change the operator noticed was this score crossing 30, **not a
promotion** (no tier/state change occurred).

## The fix (this PA)

Clamp the score that feeds `_next_level`/progress to just below the current
state's own rung, so the card always shows the **immediate successor** of the
current state:

```python
_state_now = (scr.get("state") or "").upper()
_state_ceiling = {"WARMUP": 29, "PAUSED": 29, "CAUTIOUS": 59}.get(_state_now)
_next_score = min(score, _state_ceiling) if _state_ceiling is not None else score
next_name, next_detail, next_threshold = _next_level(_next_score)
progress = int(min(100, round(_next_score * 100 / max(1, next_threshold))))
```

- WARMUP/PAUSED → "Cautious (25% at score 30)".
- CAUTIOUS → "Confident (50% at score 60)".
- CONFIDENT/other → raw score passes through unchanged (no matching real rung).
- `performance_score` still reports the **real** score (53) — only the "Next:"
  label and its progress bar are steered.

## Scope note

This fixes the state-skip only. The ladder's advertised *amounts* remain fiction
(Confident=50% vs governor's 100%; phantom Full Send/Max) — see the operator
decision in **PA TIER1-B**. Align or delete is the operator's call.

## Tests

`chad/tests/test_tier1_next_level_clamp.py` (6 cases), including the exact
reported-bug regression: WARMUP @ sharpe 0.444 must render "Cautious", not
"Confident".

## Rollback
`git revert` the TIER1-A commit; dashboard-only, no runtime state.
