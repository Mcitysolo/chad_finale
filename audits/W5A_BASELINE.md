# W5A-0 — Baseline Capture (set-diff reference)

Captured 2026-07-23 in worktree `/home/ubuntu/chad_w5a`, branch
`goal/wave5-measurement`, base `main@e86eaaf` (the merged W4A fuse box).

Command: `python3 -m pytest chad/tests/ -q`.
Result: **16 failed, 4323 passed, 5 skipped, 22 warnings**.

Identical to the W4A post-merge failing set — every W5A commit must leave the
failing-test set ⊆ this named set (set-diff, not count-diff).

## Baseline failing set (16)

| # | Test |
|---|---|
| 1 | `test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes` |
| 2 | `test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage` |
| 3 | `test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop` |
| 4 | `test_kraken_execution.py::test_intent_builder_btc_basic` |
| 5 | `test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified` |
| 6 | `test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only` |
| 7 | `test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only` |
| 8–13 | `test_repo_write_guard.py` (6 tests) |
| 14 | `test_routing_gates.py::test_e4_kraken_passive_order_params` |
| 15 | `test_routing_gates.py::test_e4_kraken_aggressive_order_params` |
| 16 | `test_w4b8_flatten_bare_terminal.py::test_bare_terminal_drill_reaches_drill_complete` (worktree has no `venv/`; the test subprocess-spawns `<repo>/venv/bin/python3`) |

## Schema-bump blast-radius finding (R2 vs the freeze — needs an operator call)

Rider R2 directs a `closed_trade.v1 → .v2` version bump "following the W4A/D6
precedent." That precedent (`drawdown_state.v2`) touched ONE reader (the EXS7
pin). `closed_trade` is **not** analogous: it is hash-chained AND read by
**exact `== "closed_trade.v1"` match** in ≥5 non-test sites, one of them
execution-critical and one of them in frozen territory:

- `chad/execution/trade_closer.py:853` — reads back closed rows to mark their
  `fill_ids` **consumed** (Bug-B-Fix-B). A `.v2` row skipped here is NOT marked
  consumed ⇒ the FIFO can re-process the close ⇒ **the futures-runaway
  re-entry gap the comment explicitly warns about** (the INCIDENT-0723 class).
  This makes a naive bump NOT observer-class.
- `chad/execution/trade_closer.py:1191` — pnl aggregation (under-counts v2).
- `chad/risk/tier_risk_enforcer.py:328`, `chad/risk/symbol_performance_blocker.py:101`
  (dormant), `chad/portfolio/ibkr_paper_ledger_watcher.py:240`,
  `chad/portfolio/ibkr_paper_fill_harvester.py:53` (writer const),
  `chad/analytics/train_xgb_model.py:79`.
- **FROZEN:** `chad/validation/trade_log_adapter.py:1070` `verify_ledger_chain`
  recomputes the byte-exact `record_hash` **only for exact `closed_trade.v1`**.
  A `.v2` row keeps linkage+sequence checks but SKIPS the tamper-recompute —
  restoring it requires editing `chad/validation/`, which the wave freezes.
- ~2–18 test files pin `closed_trade.v1`.

The decision (additive-no-bump vs full bump incl. the frozen edit vs bump with
the verifier deferred to Lane B) is recorded in the W5A-0 commit and escalated
to the operator before any code that depends on it (W5A-2 onward). W5A-1
(the IS joiner core), W5A-3/-4 (E3), and W5A-5 (DQ2) do not depend on it.
