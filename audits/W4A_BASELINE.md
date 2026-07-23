# W4A-0 — Baseline Capture (set-diff reference)

Captured 2026-07-23 in worktree `/home/ubuntu/chad_w4a`, branch `goal/wave4-fusebox`,
immediately after rebasing the Phase-1 plan commit onto `main@5013893` (W4B-8f tip;
rebase was clean — plan commit `173df9b` → `975d5a9`, 17 main commits replayed under it).

Command: `python3 -m pytest chad/tests/ -q` (canonical venv).
Result: **16 failed, 4153 passed, 5 skipped, 22 warnings in 164.53s**.

Every W4A commit must leave the failing-test set ⊆ this named set (set-diff, not
count-diff — W2A/W3B methodology).

## Baseline failing set (16)

| # | Test | Note |
|---|---|---|
| 1 | `test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes` | pre-existing (W3B baseline-15) |
| 2 | `test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage` | pre-existing |
| 3 | `test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop` | pre-existing |
| 4 | `test_kraken_execution.py::test_intent_builder_btc_basic` | pre-existing |
| 5 | `test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified` | pre-existing |
| 6 | `test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only` | pre-existing |
| 7 | `test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only` | pre-existing |
| 8 | `test_repo_write_guard.py::test_guard_blocks_direct_write_under_data` | pre-existing (6-test module) |
| 9 | `test_repo_write_guard.py::test_guard_blocks_write_under_runtime` | pre-existing |
| 10 | `test_repo_write_guard.py::test_guard_blocks_mkdir_under_data` | pre-existing |
| 11 | `test_repo_write_guard.py::test_guard_reraises_even_when_caller_swallows` | pre-existing |
| 12 | `test_repo_write_guard.py::test_baseline_blocks_all_but_records_only_new_sinks` | pre-existing |
| 13 | `test_repo_write_guard.py::test_grandfathered_write_is_blocked_but_not_recorded` | pre-existing |
| 14 | `test_routing_gates.py::test_e4_kraken_passive_order_params` | pre-existing |
| 15 | `test_routing_gates.py::test_e4_kraken_aggressive_order_params` | pre-existing |
| 16 | `test_w4b8_flatten_bare_terminal.py::test_bare_terminal_drill_reaches_drill_complete` | NEW vs W3B baseline-15 — arrived with the W4B-8e rebase. Verified cause: the test subprocess-spawns `<repo>/venv/bin/python3`, which exists in the canonical checkout but not in this worktree (`FileNotFoundError: /home/ubuntu/chad_w4a/venv/bin/python3`). Worktree-environmental, not a code defect; W4B-8e landed set-diff-green on main's own checkout. Pinned here so any W4A regression against it is still caught by name. |

## Provenance

- W3B baseline-15 reference: `PLAN_W3B` methodology (4008 passed at 6979b99);
  suite has since grown to 4153 passing under the W4B merge.
- Raw tail of the run: see git history of this file's commit message; run log
  was captured off-tree (scratchpad) per no-runtime-writes rule.
