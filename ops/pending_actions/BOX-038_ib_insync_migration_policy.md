# BOX-038 — ib_insync Migration Policy (deferred-importer ledger)

- **Box:** 038 — GAP-004 / GAP-021 ib_insync migration finished
- **Stage:** 3 — Engineering, tests, SSOT, hidden-gap closure
- **Document timestamp (UTC):** 2026-05-20T02:03:14Z
- **Document type:** migration ledger + policy (no code mutation by this audit; no service restart; no live trading authorization)
- **Authoritative ledger:** `chad/tests/test_ib_async_import_parity.py` (`PHASE1_MIGRATED_FILES`, `PHASE2_DEFERRED_FILES`). The static-check tests in that file are the gate — this document is the human-readable mirror.

---

## 1. Migration status (summary)

| Bucket | Count | Source-of-truth |
| --- | --- | --- |
| Production files migrated to `ib_async` (Phase 1, complete) | **20** | `PHASE1_MIGRATED_FILES` |
| Production files still importing `ib_insync` (Phase 2, deferred) | **2** | `PHASE2_DEFERRED_FILES` |
| Unclassified production importers | **0** | enforced by `test_remaining_ib_insync_imports_are_explicitly_allowlisted` |
| Test files referencing either library | not enforced (allowed) | `_iter_production_python_files` excludes `tests/` |
| Pinned versions | `ib_async==2.1.0`, `ib-insync==0.9.86` (coexistence) | `requirements.txt`; enforced by `test_phase0_requirements_pin_ib_async` |

## 2. Phase 1 — migrated to `ib_async` (20 files)

| Batch | File | Profile |
| --- | --- | --- |
| 1B.1 | `backend/ibkr.py` | READ_ONLY_BROKER health endpoint (managedAccounts, reqCurrentTime) |
| 1B.2 | `chad/core/broker_position_sync.py` | READ_ONLY_BROKER positions read |
| 1B.2 | `chad/core/ibkr_healthcheck.py` | CLI healthcheck |
| 1B.2 | `chad/dashboard/api.py` | Dashboard snapshot |
| 1B.4 | `chad/market_data/ibkr_bar_provider.py` | reqHistoricalData / reqMktData |
| 1B.4 | `chad/market_data/ibkr_historical_provider.py` | reqHistoricalData |
| 1B.4 | `chad/market_data/ibkr_price_provider.py` | reqMktData / FX (Forex) |
| 1B.4 | `chad/market_data/nightly_bars_refresh.py` | bulk historical bar refresh |
| 1B.4 | `chad/market_data/options_chain_refresh.py` | reqSecDefOptParams / contract metadata |
| 1B.4 | `chad/market_data/price_cache_refresh.py` | price cache fetch |
| 1B.4 | `chad/options/chain_provider.py` | options chain metadata |
| 1B.5 | `chad/ops/ibkr_broker_events_collector.py` | fills + commission report read |
| 1B.5 | `chad/ops/portfolio_snapshot_publisher.py` | portfolio snapshot read |
| 1B.5 | `chad/ops/reconciliation_publisher.py` | reconciliation snapshot read |
| 1B.5 | `chad/portfolio/ibkr_paper_fill_harvester.py` | paper fill harvester read |
| 1B.5 | `chad/portfolio/ibkr_paper_ledger_watcher.py` | ledger watcher read |
| 1B.5 | `chad/portfolio/ibkr_portfolio_collector_v2.py` | portfolio collector read |
| 1B.6 | `chad/intel/advisory_engine.py` | USDCAD FX read for advisory context |
| 2    | `chad/core/live_loop.py` | execution hot-path (migrated; import-only delta — `ib_async` preserves the `ib_insync` public API surface) |
| 2    | `chad/execution/ibkr_adapter.py` | execution-path adapter |
| 2    | `chad/execution/ibkr_trade_router.py` | execution-path router |

Static check pinning this list: `test_phase1_migrated_files_do_not_import_ib_insync` — any file in this list that re-introduces an `ib_insync` import fails the test.

## 3. Phase 2 — explicitly deferred (2 files)

The remaining production `ib_insync` importers are the two execution-path runners that place orders directly. They are tracked in `PHASE2_DEFERRED_FILES` and are governed by `test_phase2_high_risk_files_remain_allowlisted_on_ib_insync` (must continue to import) and `test_remaining_ib_insync_imports_are_explicitly_allowlisted` (no silent additions).

### 3.1 `chad/core/paper_position_closer.py`

| Field | Value |
| --- | --- |
| Import lines | `:39` (TYPE_CHECKING), `:41` (runtime module-top), `:49` (lazy helper) |
| Imports used | `IB`, `MarketOrder`, `LimitOrder`, `Stock` |
| Profile | EXECUTION_PATH — `ib.placeOrder(...)` for paper-mode position close |
| Why deferred | Hot-path order placement on the paper-position closer. Migrating requires a dedicated execution-path safety harness so a regression cannot silently flip order side / type / quantity. Per CHAD governance §3/§7 this needs an operator-approved Pending Action with a dedicated test pass on placeOrder semantics, not a routine import swap. |
| Risk level | HIGH (places paper orders; broker call surface) |
| Runtime services that execute this code | Invoked by `paper_shadow_runner` indirectly (calendar helpers) and by manual operator runs; no dedicated systemd timer/service runs `paper_position_closer.main` today. |
| Owner / module | `chad/core` (execution / paper closer) |
| Future migration action | Swap each `from ib_insync import IB, MarketOrder, LimitOrder, Stock` to `from ib_async import IB, MarketOrder, LimitOrder, Stock`. `ib_async` preserves the public API surface (per Phase 2 `ibkr_adapter` / `ibkr_trade_router` precedent) so the call sites do not change. Verification: full repo test suite + targeted paper-mode integration smoke (no live broker). Add a fixture-based placeOrder regression test before the swap. |

### 3.2 `chad/core/paper_shadow_runner.py`

| Field | Value |
| --- | --- |
| Import lines | `:595` (lazy in `IBKRClient.connect`), `:774` (lazy in `place_mkt_order`) |
| Imports used | `IB`, `Stock`, `MarketOrder` |
| Profile | EXECUTION_PATH — `ib.placeOrder(...)` for paper-mode shadow execution |
| Why deferred | Same as 3.1 — places paper orders via `ib.placeOrder`. This module is the entry point that systemd actually runs (`chad-paper-shadow-exec.service` / `chad-paper-shadow-runner.service` drop-ins both invoke `python -m chad.ops.paper_shadow_runner_wrapper_v2 --execute`, which shells out to `chad.core.paper_shadow_runner`). Highest blast radius of the remaining importers. |
| Risk level | HIGH (places paper orders; systemd-invoked; primary paper-execution path) |
| Runtime services that execute this code | `deploy/drop-ins/chad-paper-shadow-exec.service.d_60-wrapper-v2.conf` and `chad-paper-shadow-runner.service.d_90-wrapper-v2.conf` both invoke `chad.ops.paper_shadow_runner_wrapper_v2 --execute`, which in turn spawns `python -m chad.core.paper_shadow_runner` (see `paper_shadow_runner_wrapper_v2.py:271`). |
| Owner / module | `chad/core` (paper shadow execution) |
| Future migration action | Swap both lazy imports to `from ib_async import ...`. Behavior identical (preserved API). Verification: pre-flight `full_cycle_preview.py --dry-run`, then run the parity test, then run `test_paper_shadow_runner.py` + `test_paper_shadow_execute_gating.py` + `tests/core/test_paper_shadow_runner_selection.py`. Must NOT be combined with the closer migration — single-file commits only, each with operator approval (governance §1). |

## 4. Test-only references (allowed)

Test files are excluded from the `_iter_production_python_files` walker by `skip_parts = {"__pycache__", "tests"}` and may freely import either library. Notable references:

- `chad/tests/test_ib_async_import_parity.py` — the parity test itself; imports both libraries to verify coexistence.
- `chad/tests/test_paper_shadow_runner.py` — fixtures, plus an assertion that the safe path does NOT trigger an `ib_insync` import.
- `chad/tests/test_options_chain_refresh.py`, `chad/tests/test_options_quote_check.py`, `chad/tests/test_bag_preview.py`, `chad/tests/test_probe_bag_quotes.py`, `chad/tests/test_exterminator.py`, `chad/tests/test_fmp_earnings_intel_publisher.py`, `chad/tests/test_salary_withdrawal_safety.py` — install fake-IB harnesses that patch both `ib_async` and `ib_insync` namespaces.
- `tests/core/test_paper_shadow_runner_selection.py` — documents that `paper_shadow_runner` imports `ib_insync` and may need IBKR runtime deps under tests.

These are intentionally allowed; the parity test enforces production scope only.

## 5. Doc / comment references (not imports)

The following files mention `ib_insync` only in comments / docstrings, never in an import line. They are correctly excluded from the deferred ledger.

- `chad/options/quote_check.py:14` (docstring: explicitly states "No IBKR / ib_async / ib_insync imports")
- `chad/ops/ibkr_broker_events_collector.py:11-12` (docstring describing migrated source semantics)
- `chad/core/paper_shadow_runner.py:74,104` (comments documenting safe paths that do NOT import ib_insync)
- `chad/execution/ibkr_adapter.py:1122` (comment "Lazy ib_insync imports" — predates the Phase 2 migration; the actual lazy import on `:1140` uses `ib_async`, so this comment is stale-but-harmless)

A future Phase-2 cleanup commit may update the stale comment in `ibkr_adapter.py:1122` to read `ib_async`; this audit deliberately makes no source edit.

## 6. Static-check contract (locked in code)

| Test | Invariant |
| --- | --- |
| `test_ib_async_imports_core_api` | `ib_async` installed, core symbols (`IB`, `Stock`, `Contract`, `Order`) importable |
| `test_ib_insync_still_imports_during_migration` | `ib_insync` still installed during coexistence |
| `test_ib_async_and_ib_insync_can_coexist` | both libraries co-exist; distinct classes |
| `test_phase1_migrated_files_do_not_import_ib_insync` | no Phase 1 file may re-introduce `ib_insync` |
| `test_phase2_high_risk_files_remain_allowlisted_on_ib_insync` | Phase 2 files must still import `ib_insync` (no silent drift) |
| `test_remaining_ib_insync_imports_are_explicitly_allowlisted` | **every** production importer must be in `PHASE2_DEFERRED_FILES` — new undeclared importers fail the test |
| `test_proposed_phase1_candidates_are_subset_of_phase2_deferred` | the proposed-candidates list cannot diverge from the deferred ledger |
| `test_phase0_requirements_pin_ib_async` | `requirements.txt` pins both `ib_async==<exact>` and `ib-insync==<exact>` |

8/8 tests pass at audit time.

## 7. Forbidden by this document

- Do not migrate the two deferred files in a batched commit — each is a single-file operator-approved migration with its own placeOrder regression test (governance §1, §3).
- Do not silently add a new production `ib_insync` importer; the parity test will fail and Box-038 closure is invalidated.
- Do not remove `ib-insync` from `requirements.txt` while either Phase 2 file still imports it — `test_phase0_requirements_pin_ib_async` enforces both pins.
- Do not edit the parity test allowlists outside the explicit migration commit that promotes a file from Phase 2 → Phase 1.
- Do not modify the systemd `chad-paper-shadow-*` units without an operator-approved Pending Action (CHAD governance §6/§7).

## 8. Cross-references

- Parity test: `chad/tests/test_ib_async_import_parity.py`
- Phase-2 deferred files: `chad/core/paper_position_closer.py`, `chad/core/paper_shadow_runner.py`
- Wrapper that invokes the deferred shadow runner: `chad/ops/paper_shadow_runner_wrapper_v2.py`
- systemd drop-ins: `deploy/drop-ins/chad-paper-shadow-exec.service.d_60-wrapper-v2.conf`, `deploy/drop-ins/chad-paper-shadow-runner.service.d_90-wrapper-v2.conf`
- Box-038 evidence: `runtime/completion_matrix_evidence/BOX-038_GAP-004_021_ib_insync_migration_finished.md`
- CLAUDE.md migration note: "Phase 1 complete (18 files migrated to ib_async), Phase 2 pending (5 files remaining)" — this is **stale**. Audit re-counted today: Phase 1 holds **20** files; Phase 2 holds **2** files. Recommend a future CLAUDE.md edit (out of Box-038 scope) to refresh the counts.
