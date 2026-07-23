# W5B-0 — Baseline Capture (set-diff reference)

Captured 2026-07-23 in worktree `/home/ubuntu/chad_w5b`, branch
`goal/wave5-allocator`, base `main@b5d0ee2` (the merged W5A measurement layer)
via merge commit `7cfca0c`.

Command: `python3 -m pytest chad/tests/ -q -p no:randomly`.
Result: **16 failed, 4381 passed, 5 skipped, 22 warnings** in 168.89s.

The failing set is **name-for-name identical** to `audits/W5A_BASELINE.md`
(itself identical to the W4A post-merge set). The passed count rose 4323 → 4381
because the merge brought in W5A's six new test files. Every W5B commit must
leave the failing-test set **⊆ this named set** — set-diff, never count-diff.

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
| 8 | `test_repo_write_guard.py::test_guard_blocks_direct_write_under_data` |
| 9 | `test_repo_write_guard.py::test_guard_blocks_write_under_runtime` |
| 10 | `test_repo_write_guard.py::test_guard_blocks_mkdir_under_data` |
| 11 | `test_repo_write_guard.py::test_guard_reraises_even_when_caller_swallows` |
| 12 | `test_repo_write_guard.py::test_baseline_blocks_all_but_records_only_new_sinks` |
| 13 | `test_repo_write_guard.py::test_grandfathered_write_is_blocked_but_not_recorded` |
| 14 | `test_routing_gates.py::test_e4_kraken_passive_order_params` |
| 15 | `test_routing_gates.py::test_e4_kraken_aggressive_order_params` |
| 16 | `test_w4b8_flatten_bare_terminal.py::test_bare_terminal_drill_reaches_drill_complete` |

All 16 are worktree-environment artifacts, not regressions: the worktree has no
`venv/`, no `runtime/`, and no `data/` (they are gitignored and live only in
`/home/ubuntu/chad_finale`), so tests that subprocess-spawn `<repo>/venv/bin/python3`,
assert on live runtime artifacts, or exercise the repo write-guard against real
`runtime/`/`data/` paths cannot pass here.

**Note for W5B specifically:** the six `test_repo_write_guard.py` failures are in
the baseline, which means the write-guard suite cannot be relied on to catch a
W5B evidence-path leak. W5B therefore carries its **own** leak guard — the W4A
`_guard_test_write` / `_under_pytest` pattern (`fuse_box.py:1059-1073`), which
raises `RuntimeError` when a default `runtime/` or `data/` path is reached under
pytest without an explicit override. That guard is exercised by W5B's own tests.

## Premise re-verification

The Phase-1 plan was written against `main@e86eaaf`, before W5A merged. Five of
its premises did not survive re-audit (the shadow-gate template file does not
exist; the stage-3 line anchors moved; the book gained a symbol; the equity
basis was attributed to the wrong artifact; and a correlation threshold *is*
sourced). Each correction, with evidence, is recorded in `PLAN_W5B.md §12`.
The operator GO and its four amendments are in `PLAN_W5B.md §13`; the revised
commit sequence is `§14`.
