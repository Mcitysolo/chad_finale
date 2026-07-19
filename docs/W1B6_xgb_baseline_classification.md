# W1B-6 — xgb "8th failure" classification (baseline entry)

**Date:** 2026-07-19
**Owner:** W1B Wave-1 forensics (CHAD ops / mcitysolo)
**Node:** `chad/tests/test_xgb_promotion_workflow.py::test_trainer_candidate_dir_named_with_utc_timestamp`
**Disposition:** NOT an independent failure — SIGALRM collateral, **resolved by W1B-0**. No code fix required.

## Context

The pre-W1B full-suite baseline (2026-07-19) was `8 failed, 3770 passed`:

- **5 deterministic known-5** — `test_tier_manager.py` ×3 (uncommitted working-tree
  currency-audit config `config/tiers.json` + `config/withdrawal_policy.json`),
  `test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop`,
  `test_quarantine_sidecar.py::test_per_strategy_loss_guard_excludes_quarantined_delta_trades`.
  These are unchanged by W1B (currency config is an explicit non-goal).
- **2 Item-1 flakes** — the two `live_posture` tests (pr03/pr04), fixed by W1B-1.
- **1 xgb row (this entry)** — the 8th "failure".

## Classification

The xgb node is **order-gated collateral of the leaked collector SIGALRM**, not an
independent deterministic failure:

1. **Passes in isolation** — `pytest <node>` → `1 passed in 0.54s`.
2. **The test body uses no alarm** — it is a pure victim: it calls `train_xgb_model`
   (`xgb.train`), and the 60s `SIGALRM` leaked by
   `chad.portfolio.ibkr_portfolio_collector_v2.main()` (see W1B-0) fired *inside*
   `xgb.train` during this test, surfacing as a spurious failure.
3. **Absent from the failing set post-W1B-0** — across the four post-W1B-0 full-suite
   runs (after commits W1B-0, W1B-1, W1B-2, W1B-3, W1B-4) the failing set was exactly the
   deterministic known-5 every time; the xgb node never appeared.

Root cause and mechanism are the same leaked SIGALRM documented in `PLAN_W1B.md` §1a and
fixed by W1B-0 (`fcc107e`, autouse `signal.alarm(0)` teardown). Since W1B-0 removed the
leak, the collateral is gone.

## Baseline decision

The **deterministic known-5** is the standing accepted-failure baseline for W1B "green vs
known-5". The xgb node is **removed from the failure baseline** — it is expected to PASS in
every full-suite run now that W1B-0 has landed. If it ever reappears in a full-suite run
while passing in isolation, re-open W1B-0 (suspect a *new* alarm/timeout leak from another
module calling a wall-clock guard without teardown) before treating it as an xgb regression.
