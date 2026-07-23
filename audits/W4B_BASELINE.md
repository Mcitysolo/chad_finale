# W4B-0 — Baseline test capture (set-diff reference)

Worktree `/home/ubuntu/chad_w4b` @ `6979b99` (pristine, before any W4B change).
Command: `python3 -m pytest chad/tests/ -q` (venv: chad_finale/venv).
Result: **15 failed, 4008 passed, 5 skipped** in 167.00s — identical to the
W3B baseline-15 set. Every W4B commit must keep its failure set a subset of
the list below (named-test diff, not count diff).

## Baseline failing set (15)

```
chad/tests/test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage
chad/tests/test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes
chad/tests/test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop
chad/tests/test_kraken_execution.py::test_intent_builder_btc_basic
chad/tests/test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified
chad/tests/test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only
chad/tests/test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only
chad/tests/test_repo_write_guard.py::test_baseline_blocks_all_but_records_only_new_sinks
chad/tests/test_repo_write_guard.py::test_grandfathered_write_is_blocked_but_not_recorded
chad/tests/test_repo_write_guard.py::test_guard_blocks_direct_write_under_data
chad/tests/test_repo_write_guard.py::test_guard_blocks_mkdir_under_data
chad/tests/test_repo_write_guard.py::test_guard_blocks_write_under_runtime
chad/tests/test_repo_write_guard.py::test_guard_reraises_even_when_caller_swallows
chad/tests/test_routing_gates.py::test_e4_kraken_aggressive_order_params
chad/tests/test_routing_gates.py::test_e4_kraken_passive_order_params
```

(These 15 are pre-existing on main — carried across W2A/W3B waves; the
repo_write_guard 6 are self-tests of the guard sensitive to runner context,
the rest are known environment/config-coupled reds. Not W4B's to fix.)
