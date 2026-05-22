# BOX-028 deterministic failure cluster — classification artifact

Generated: 2026-05-20T00:02:00Z (CHAD Box 028 audit)
Baseline command: `python3 -m pytest -q chad/tests tests --tb=long --disable-warnings -p no:cacheprovider`
Aggregate result: `18 failed, 2285 passed, 14 warnings in 85.80s (0:01:25)`

Live `runtime/stop_bus.json` at audit instant:
```json
{
  "active": true,
  "reason": "broker_latency:avg_latency_ms=2760.4>threshold=2000.0",
  "triggered_at": "2026-05-19T23:57:48.028581+00:00",
  "triggered_by": "live_loop.run_once"
}
```

Shared root cause across all 18 failures: the failing tests invoke
production intent-builders (`build_ibkr_intents_from_plan`,
`build_kraken_intents`, and equivalents) without monkeypatching
`chad.risk.stop_bus_state.is_stop_bus_active`. Production correctly
short-circuits when the LIVE stop bus is active
(`chad/execution/execution_pipeline.py:1060-1063` for IBKR;
`chad/execution/execution_pipeline.py:1629-1631` for Kraken) and logs
`STOP_BUS_ACTIVE — *** intent building skipped`. The pytest log
contains 19 `STOP_BUS_ACTIVE` occurrences (one per affected intent build
plus one in a test that captures the warning text).

Shared mapped gap: **NEW-GAP-033b** — extends GAP-033 (Box 027). The
Box-027 isolation proof verified the dedicated stop-bus suites and
hash-invariance (tests don't WRITE live runtime), but the broader
deterministic baseline includes 4 test files that READ live
`runtime/stop_bus.json` indirectly via production intent-builders.
Those tests need stop-bus isolation (monkeypatch or tmp_path
override), or a session-level autouse fixture that pins
`is_stop_bus_active → False` for non-stop-bus tests.

Severity: HIGH (blocks deterministic baseline) but NOT a production
bug — production behavior is correct; only test isolation is missing.

Deterministic? **YES** when live stop_bus.json is active=true; **NO**
otherwise (this is exactly why the cluster is intermittent across
sessions — when broker latency is normal and stop_bus.json is
active=false, all 18 tests pass).

Environment-dependent? **YES** — depends on live runtime state.

---

| Failure ID | Test nodeid | Traceback location | Immediate error/assertion | Root cause | Mapped gap / matrix box | Owner / module | Severity | Deterministic? | Env-dependent? | Recommended next action |
|------|------|------|------|------|------|------|------|------|------|------|
| F-01 | chad/tests/test_alpha_options_meta_preservation.py::test_alpha_options_bag_intent_preserves_contract_meta_and_sec_type | test_alpha_options_meta_preservation.py:116 | `assert len(intents) == 1` (got 0) | Calls `build_ibkr_intents_from_plan` w/o monkeypatching stop_bus; live bus active → `[]` | NEW-GAP-033b → Box 029 (`alpha_options_BAG_meta_tests_fixed`) | chad/strategies/alpha_options + chad/execution/execution_pipeline (test file) | HIGH | YES (when live bus active) | YES | Box 029: add `monkeypatch.setattr("chad.risk.stop_bus_state.is_stop_bus_active", lambda *a, **kw: False)` or an autouse fixture in the test file |
| F-02 | chad/tests/test_alpha_options_meta_preservation.py::test_alpha_options_bag_intent_resolves_through_ibkr_adapter | test_alpha_options_meta_preservation.py:145 | `IndexError: list index out of range` (`_build_intents(routed)[0]` on empty list) | Same — `_build_intents` returns `[]` | NEW-GAP-033b → Box 029 | alpha_options test file | HIGH | YES | YES | Box 029 |
| F-03 | chad/tests/test_alpha_options_meta_preservation.py::test_single_leg_opt_intent_preserves_expiry_strike_right | test_alpha_options_meta_preservation.py:173 | `assert len(intents) == 1` (got 0) | Same | NEW-GAP-033b → Box 029 | alpha_options test file | HIGH | YES | YES | Box 029 |
| F-04 | chad/tests/test_alpha_options_meta_preservation.py::test_options_intent_missing_expiry_is_skipped_pre_submit | test_alpha_options_meta_preservation.py:241 | `assert any(...)` — expected OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META audit log | Same — production short-circuited before the OPTIONS_INTENT_SKIPPED log site | NEW-GAP-033b → Box 029 | alpha_options test file | HIGH | YES | YES | Box 029 |
| F-05 | chad/tests/test_alpha_options_meta_preservation.py::test_options_intent_missing_strikes_is_skipped_pre_submit | test_alpha_options_meta_preservation.py:255 | `assert any(...)` (no matching log) | Same | NEW-GAP-033b → Box 029 | alpha_options test file | HIGH | YES | YES | Box 029 |
| F-06 | chad/tests/test_alpha_options_meta_preservation.py::test_futures_contract_month_safety_unaffected | test_alpha_options_meta_preservation.py:280 | `assert len(intents) == 1` (got 0) | Same | NEW-GAP-033b → Box 029 | alpha_options test file | HIGH | YES | YES | Box 029 |
| F-07 | chad/tests/test_alpha_options_meta_preservation.py::test_alpha_options_bag_no_signal_meta_falls_back_safely | test_alpha_options_meta_preservation.py:305 | `assert any(...)` (no fallback log) | Same — production short-circuited | NEW-GAP-033b → Box 029 | alpha_options test file | HIGH | YES | YES | Box 029 |
| F-08 | chad/tests/test_futures_contract_resolver.py::test_build_intents_attaches_contract_month_for_futures[MES] | test_futures_contract_resolver.py:136 | `assert len(intents) == 1` (got 0) | Same | NEW-GAP-033b → Box 030+ (futures_contract_resolver tests stop_bus isolation) | chad/market_data/futures_contract_resolver (test file) | HIGH | YES | YES | Apply same monkeypatch pattern as Box 029 |
| F-09 | chad/tests/test_futures_contract_resolver.py::test_build_intents_attaches_contract_month_for_futures[MGC] | test_futures_contract_resolver.py:136 | `assert len(intents) == 1` (got 0) | Same | NEW-GAP-033b → Box 030+ | futures_contract_resolver test file | HIGH | YES | YES | Same fix |
| F-10 | chad/tests/test_futures_contract_resolver.py::test_build_intents_does_not_attach_contract_month_for_equities | test_futures_contract_resolver.py:164 | `assert len(intents) == 1` (got 0) | Same | NEW-GAP-033b → Box 030+ | futures_contract_resolver test file | HIGH | YES | YES | Same fix |
| F-11 | chad/tests/test_kraken_execution.py::test_build_kraken_intents_skips_non_crypto_and_missing_prices | test_kraken_execution.py:164 | `assert len(intents) == 1` (got 0) | `build_kraken_intents` short-circuits on live stop_bus active | NEW-GAP-033b → Box 030+ (kraken intent test isolation) | chad/execution/execution_pipeline Kraken branch (test file) | HIGH | YES | YES | Same monkeypatch (Kraken branch also calls `_stop_bus_active`) |
| F-12 | chad/tests/test_omega_macro_execution_lane.py::test_execution_pipeline_omega_macro_intent[M6E-CME] | test_omega_macro_execution_lane.py:93 | `assert len(intents) == 1` (got 0); "registry must include M6E as a FUT spec" | Same — message misattributes the cause (registry IS fine; intent build short-circuited by stop_bus) | NEW-GAP-033b → Box 030+ (omega_macro test isolation) | chad/strategies/omega_macro (test file) | HIGH | YES | YES | Same fix |
| F-13 | chad/tests/test_omega_macro_execution_lane.py::test_execution_pipeline_omega_macro_intent[ZB-CBOT] | test_omega_macro_execution_lane.py:93 | `assert len(intents) == 1` (got 0); ZB | Same | NEW-GAP-033b → Box 030+ | omega_macro test file | HIGH | YES | YES | Same fix |
| F-14 | chad/tests/test_omega_macro_execution_lane.py::test_execution_pipeline_omega_macro_intent[ZN-CBOT] | test_omega_macro_execution_lane.py:93 | `assert len(intents) == 1` (got 0); ZN | Same | NEW-GAP-033b → Box 030+ | omega_macro test file | HIGH | YES | YES | Same fix |
| F-15 | chad/tests/test_omega_macro_execution_lane.py::test_intent_builder_logs_unknown_symbol | test_omega_macro_execution_lane.py:162 | "expected a WARNING containing INTENT_DROPPED_NO_SPEC ... got: ['STOP_BUS_ACTIVE — IBKR intent building skipped']" | **SMOKING GUN** — pytest captured the live stop_bus warning instead of the expected unknown-symbol warning | NEW-GAP-033b → Box 030+ | omega_macro test file | HIGH | YES | YES | Same fix |
| F-16 | chad/tests/test_omega_macro_execution_lane.py::test_execution_pipeline_omega_macro_end_to_end | test_omega_macro_execution_lane.py:206 | `assert len(intents) == 1, "omega_macro M6E: expected 1 intent, got 0"` | Same | NEW-GAP-033b → Box 030+ | omega_macro test file | HIGH | YES | YES | Same fix |
| F-17 | chad/tests/test_omega_macro_execution_lane.py::test_execution_pipeline_m2k_intent | test_omega_macro_execution_lane.py:232 | `assert len(intents) == 1` (M2K) | Same | NEW-GAP-033b → Box 030+ | omega_macro test file | HIGH | YES | YES | Same fix |
| F-18 | chad/tests/test_omega_macro_execution_lane.py::test_execution_pipeline_mym_intent | test_omega_macro_execution_lane.py:273 | `assert len(intents) == 1` (MYM) | Same | NEW-GAP-033b → Box 030+ | omega_macro test file | HIGH | YES | YES | Same fix |

---

## Cluster summary

- **Total failures**: 18 (matches the historical "19/18 deterministic failure cluster" label —
  19 STOP_BUS_ACTIVE warnings captured, 18 distinct test failures; the 19th warning is the
  one inspected by `test_intent_builder_logs_unknown_symbol` itself).
- **Distinct files**: 4
  - chad/tests/test_alpha_options_meta_preservation.py (7 failures → Box 029)
  - chad/tests/test_futures_contract_resolver.py (3 failures → future box)
  - chad/tests/test_kraken_execution.py (1 failure → future box)
  - chad/tests/test_omega_macro_execution_lane.py (7 failures → future box)
- **Distinct root causes**: 1 (missing stop_bus monkeypatch isolation in intent-builder tests)
- **Production-code defects**: 0 — production behavior is correct
- **Suggested generic fix** (per test file):
  ```python
  @pytest.fixture(autouse=True)
  def _disable_stop_bus_for_intent_tests(monkeypatch):
      import chad.risk.stop_bus_state as sbs
      monkeypatch.setattr(sbs, "is_stop_bus_active", lambda *args, **kwargs: False)
      # Defensive: also patch the cached references in execution_pipeline if any
      import chad.execution.execution_pipeline as ep
      if hasattr(ep, "_stop_bus_active"):
          monkeypatch.setattr(ep, "_stop_bus_active", lambda: False)
  ```
- **Alternative**: session-level autouse fixture in `chad/tests/conftest.py` that disables
  stop-bus globally for any test NOT in the dedicated stop-bus suites — but that risks
  over-disabling, so per-file fixtures are preferred.
