# W2A standing finding (D5/D8) — the test suite is not hermetic

**Status:** FINDING (documented for a future test-hygiene lane). W2A fixed exactly ONE instance
(the tier fixture, W2A-6); the rest are recorded here, not touched, per D5/D8.

## The finding

Running the full suite **in a worktree** does not reproduce the live tree's failing set. In
`/home/ubuntu/chad_w2a` the baseline is **18 failed, 3838 passed, 5 skipped**, whereas the live
tree (`/home/ubuntu/chad_finale`, main) fails a different 5. The failing SET is
environment-dependent because many tests are coupled to the live tree via **absolute
`/home/ubuntu/chad_finale` paths** and reads of live `runtime/` / `config/` / `data/bars`.

Scale (measured 2026-07-20): **46 occurrences across 23 test files** hardcode
`/home/ubuntu/chad_finale`. Categories:

- **Absolute repo-root constants** — e.g. `REPO_ROOT = Path("/home/ubuntu/chad_finale")`
  (`test_tier_manager.py` — FIXED in W2A-6; `test_repo_write_guard.py`,
  `test_phase_b_item4_crypto_derivatives.py`, `chad_behavioral_audit_harness.py`,
  `test_margin_shadow_gate.py`, `test_margin_gate_integration.py`, …).
- **Subprocess env / interpreter paths** — `PYTHONPATH`, `CHAD_RUNTIME_DIR`,
  `CHAD_CONFIG_DIR`, `ExecStart=/home/ubuntu/chad_finale/venv/bin/python3`, absolute `SCRIPT`
  (`test_rotation_publish.py`, `test_rebalance_autonomy_bounds.py`,
  `test_rebalance_auto_executor_paper.py`, `test_portfolio_var_drawdown_report_only.py`, …).
- **Direct live-runtime/config reads** — `runtime/positions_truth.json`,
  `runtime/live_readiness.json`, `config/crypto_exit_overlay.json`,
  `runtime/setup_family_expectancy.json` (`test_pr02b_…`, `test_box054_…`,
  `test_crypto_exit_overlay.py`, `test_alpha_intraday_micro.py`, …).
- **Systemd-unit literal assertions** — `WorkingDirectory=/home/ubuntu/chad_finale` etc. (these
  are arguably CORRECT to pin to the deployed path; call out but likely leave).

Full inventory: `scripts`-free grep — `grep -rn "/home/ubuntu/chad_finale" chad/tests/ tests/`.

## Consequence

- Worktree-based CI for ANY lane sees spurious failures. This is why W2A adopted a
  **failing-test-ID SET diff** against the recorded worktree-18 baseline (require ⊆; each commit
  must add no new failing ID), NOT a green-by-count gate (D8). The 14 non-tier artifacts in the
  worktree-18 baseline are live-tree coupling, verified NOT to overlap any W2A-touched test, so
  they cannot mask a W2A regression.
- A repo-root collection error also exists OUTSIDE `chad/tests/`:
  `tests/validation/test_backtest_engine.py` fails at collection, which interrupts a
  `pytest chad/tests/ tests/` run. W2A scopes its gate to `chad/tests/` (matching CLAUDE.md's
  verification sequence) + `tests/validation/test_isolation.py` run separately.

## Recommendation (future hygiene lane — NOT W2A)

1. Replace every `Path("/home/ubuntu/chad_finale")` with `Path(__file__).resolve().parents[N]`
   (the pattern W2A-6 applied to `test_tier_manager.py`).
2. Replace subprocess `PYTHONPATH`/`SCRIPT`/interpreter literals with values derived from the
   test file's repo root and `sys.executable`.
3. Route live-runtime/config reads through `tmp_path` fixtures or `CHAD_RUNTIME_DIR` /
   `CHAD_CONFIG_DIR` env overrides so a test never reads the deployed tree.
4. Fix or quarantine the `tests/validation/test_backtest_engine.py` collection error.
5. Decide policy on systemd-unit literal path assertions (pin-to-deploy vs parametrize).

Until then, worktree CI for any lane must use the SET-diff methodology, not green-by-count.
