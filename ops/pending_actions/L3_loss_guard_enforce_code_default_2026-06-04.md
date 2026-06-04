# Pending Action — L3 loss-guard: make ENFORCE the code default (not env-var only)
Date: 2026-06-04  •  Author: TEAM CHAD (issued) / SOLO (executes)  •  Status: PENDING operator GO  •  Priority: MEDIUM (safety-brake durability; risk-tightening only; brake binds today)

## 1. Objective
Make the per-strategy loss-guard ENFORCE the code default so the brake cannot silently revert to report-only if CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE is dropped (drop-in edit, service change, or a unit rewrite that forgets it). Same "code enforcement over runtime override" hardening Fix A applied to the futures cap. Risk-tightening only; zero behavior change in current operation.

## 2. Current state / root cause
- The decision bit is the env var ALONE. is_enforce_enabled() (per_strategy_loss_guard.py:122-126) reads CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE; unset -> "" -> False -> report-only (log-only; brake does not bind). Bound at :342 (self.enforce); consumed once at live_loop.py:1461-1474 (enforce -> drop fresh entry signals from breached strategies; report-only -> WARNING only; exits/protectives always pass; failure-soft).
- config default_mode: "report_only" (config/per_strategy_loss_limits.json, loaded :105-107) is DEAD WEIGHT — never consulted by any decision. Editing it changes nothing (spec != reality).
- Running live_loop (PID 3633421) env has the var =1 -> self.enforce=True every cycle. The brake binds TODAY; the risk is silent future loosening with no alarm.

## 3. The change (shape B — enforce-default + loud explicit-disable)
File: chad/risk/per_strategy_loss_guard.py, is_enforce_enabled() (~line 122). Return True (enforce) when the var is absent OR truthy; return False ONLY for an explicit falsy value, logging loudly each call:
    raw = str(src.get(ENFORCEMENT_ENV, "")).strip().lower()
    if raw in {"0","false","no","off"}:
        LOG.warning("PER_STRATEGY_LOSS_GUARD enforcement EXPLICITLY DISABLED via %s=%s", ENFORCEMENT_ENV, raw)
        return False
    return True
Net: code is the primary guard; the env var demotes to redundant backup / explicit, announced escape hatch. Loosening can no longer be silent.
Companion truth-ups (same PA, cosmetic): module docstring (:8-13), class docstring (:326-329), the dead config default_mode (drop it or set "enforce" for honesty), live_loop comment (:1447-1454).
SHARP EDGE: do NOT "fix" this by editing config default_mode — that field is decorative; the change MUST land in is_enforce_enabled().

## 4. No-behavior-change confirmation
Running env =1 (truthy) -> is_enforce_enabled returns True under both old and new logic. Only the counterfactual changes: var deleted -> enforce STAYS ON (new) vs silently OFF (old). The service env var becomes harmless belt-and-suspenders.

## 5. Consumers + tests
- One runtime consumer: live_loop.py:1456-1495. No SCR/orchestrator/strategy logic reads the mode.
- Tests to update IN THIS CHANGE: test_per_strategy_loss_guard_warns_report_only_by_default (:152) -> invert to enforce-by-default; test_per_strategy_loss_guard_does_not_suppress_when_enforce_off (:186) -> set explicit-disable value; NEW test asserting the loud WARNING on explicit disable. Unaffected: :214/:245/:284; fixture delenvs the var (:67), isolation clean.

## 6. Risk classification + deploy
Risk-MUTATING (edits a safety brake's decision logic; new code activates only on a gated chad-live-loop restart, rules #6/#7) -> Channel 1 PA, explicit GO, never auto-applied (rule #3). Risk-TIGHTENING only (removes a silent-loosening failure mode) -> low-controversy.

## 7. Procedure (on GO)
1. (Ch2) Edit is_enforce_enabled() to shape B + companion truth-ups; update the 2 pinned tests + add the explicit-disable test.
2. py_compile; CHAD_SKIP_IB_CONNECT=1 pytest chad/tests/test_per_strategy_loss_guard.py -q, then full suite (expect zero regression); full_cycle_preview clean.
3. Commit (rule #8).
4. (Ch1, gated) Activate via chad-live-loop restart — STANDALONE, or bundled with the next gated live_loop restart (e.g. gate removal) to avoid dedicated restart churn. Post-restart: guard reports ENFORCING, no enforcement-disabled WARNING (env still =1), no behavior regression.

## 8. Verification (acceptance)
Targeted + full pytest green (3 updated/new tests). Post-restart: running guard ENFORCING (not REPORT_ONLY); no enforcement-disabled WARNING; no regression.

## 9. Rollback
Revert the one-file change + restart; the env var (=1) restores enforce either way.

## 10. Out of scope (separate PA)
Buying-power / margin pre-trade check: CONFIRMED absent (no buying_power/AvailableFunds/ExcessLiquidity gate in chad/; the 17.9x book formed with all existing layers in place). A NEW control needing account-snapshot plumbing + margin estimation — file as its own read-first/PA, not on this change.

## 11. Status log
- 2026-06-04: authored from L3 read-first (PROCEED-TO-PA, no blockers). PENDING operator GO.
