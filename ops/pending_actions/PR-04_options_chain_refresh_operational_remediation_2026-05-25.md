# PR-04 — Options Chain Refresh Operational Remediation

**Date:** 2026-05-25
**Author:** Team CHAD (Claude Code)
**Status:** BLOCKED — code/tests/artifacts harden the failure surface; full service
success requires IBKR contract-details endpoint to come back from its 2026-05-25
(US Memorial Day) degradation.
**Mode:** PAPER. ready_for_live=false. allow_ibkr_live=false. allow_ibkr_paper=true.

## Plain-English summary
`chad-options-chain-refresh.service` (oneshot, Mon–Fri 12:30 UTC) was logging a
single bare `SPY=timeout_after_30.0s` on a transient IBKR farm hiccup and giving
up. Today's 12:30 UTC run hit that single-attempt path on Memorial Day and the
operator surface degraded to "Options chain refresh failed" plus
"options_greeks.json::symbols is {}". The cache contract was already honest
(empty `chains` + `error` field, monitored by R17/R18) but there was no retry,
no per-attempt log, and no structured failure artifact carrying
`blocked_reason` / `last_successful_ts`. PR-04 adds those, leaves the existing
contracts intact, and proves the upgraded path via tests AND a live operational
run.

## Root cause
- `chad/market_data/options_chain_refresh.py::run` made exactly one
  `reqContractDetailsAsync` per symbol per timer firing. When IBKR's contract
  details endpoint stalled past `CHAD_OPTIONS_CHAIN_TIMEOUT_SECONDS` (default
  30s) the run exited with no retry budget and no structured downstream
  artifact beyond the legacy cache `error` string.
- Today (2026-05-25, US Memorial Day) IBKR's contract-details endpoint for
  SPY options was unresponsive: a fresh post-fix operational run with 2
  attempts at 20s each both timed out, confirming the external blocker.

## Files changed
- `chad/market_data/options_chain_refresh.py`
  - Add env-driven retry/backoff (`CHAD_OPTIONS_CHAIN_REFRESH_ATTEMPTS`
    default 3, capped at 10; `CHAD_OPTIONS_CHAIN_REFRESH_BACKOFF_SECONDS`
    default 5s) with per-attempt structured logging.
  - Add structured `runtime/options_chain_refresh_failure.json`
    (schema_version `options_chain_refresh_failure.v1`) carrying:
    `ts_utc`, `status`, `provider`, `service_entrypoint`, `max_attempts`,
    `per_call_timeout_seconds`, `backoff_seconds`, `symbol_errors`,
    `attempts` (per-symbol history), `error_type`, `error_message`,
    `last_successful_ts` (best-effort from prior healthy cache),
    `blocked_reason` (classified from per-symbol error shape).
  - Preserve the existing empty-cache-with-`error`-field semantics so
    `chad/ops/health_monitor_rules.py::rule_options_chain_refresh_health`
    (R17) still fires CRITICAL and `rule_options_greeks_freshness` (R18)
    still fires its WARNING/CRITICAL ladder.
  - On a fully healthy run, scrub a stale failure artifact so the operator
    surface reflects current truth.
- `chad/tests/test_pr04_options_chain_refresh_remediation.py` (new)
  - 18 tests covering: structured failure artifact on full timeout;
    per-attempt log records; flaky-symbol recovery via retry; healthy run
    clears stale artifact; full failure never fakes a healthy cache; env
    knob plumbing (attempts/backoff) rejects bad values; failure artifact
    records `last_successful_ts` from prior cache; alpha_options fails
    closed on missing/empty cache; omega_momentum_options marks fallback
    signals `synthetic_pricing=True`; live posture artifacts remain paper.

## Tests added / updated
- Targeted suite (`-k "options_chain or options_chains or greeks or
  alpha_options or omega_momentum_options or health_monitor or pr04"`):
  167 passed.
- Full regression: 2510 passed (up from 2492 baseline, +18 new tests).
  One flake in `test_canonical_equity_source.py` (concurrent positions_truth
  publisher between samples) cleared on rerun — unrelated to PR-04.

## Operational run result
- Command: `python3 -m chad.market_data.options_chain_refresh` against the
  running paper `chad-ibgateway.service` (port 4002) with
  `CHAD_OPTIONS_CHAIN_TIMEOUT_SECONDS=20`,
  `CHAD_OPTIONS_CHAIN_REFRESH_ATTEMPTS=2`,
  `CHAD_OPTIONS_CHAIN_REFRESH_BACKOFF_SECONDS=3`.
- Wall-clock: ~47s (2 × 20s + 3s backoff + a few seconds of spot fallback).
- Spot price for SPY successfully fell back to bar cache (745.64,
  2026-05-22) when live MktData returned IBKR Error 10089 / 10168
  (no realtime market-data subscription).
- Both contract-details attempts timed out.
- Run exited 1; wrote both artifacts.

## Artifact paths (post-run)
- `runtime/options_chains_cache.json`
  - `schema_version=options_chain_cache.v2`
  - `chains={}` (empty)
  - `error="all_symbols_failed: SPY=timeout_after_20.0s"`
  - `ts_utc=2026-05-25T22:11:03Z`
- `runtime/options_chain_refresh_failure.json` (NEW)
  - `schema_version=options_chain_refresh_failure.v1`
  - `status=failed`
  - `provider=ibkr_contract_details`
  - `blocked_reason=ibkr_contract_details_unresponsive`
  - `service_entrypoint=python -m chad.market_data.options_chain_refresh`
  - `max_attempts=2`, `per_call_timeout_seconds=20.0`, `backoff_seconds=3.0`
  - `attempts.SPY=[2 entries with ts_utc + result=timeout]`
  - `symbol_errors={"SPY":"timeout_after_20.0s"}`
  - `last_successful_ts=null` (prior cache was also in failed state)
- `runtime/options_greeks.json` — unchanged from prior daily publisher run
  (status=partial, symbols={} — synthetic publisher hasn't re-run since
  yesterday's 23:49 UTC firing). R18 WARNING continues to fire.

## Service / strategy state
- `chad-options-chain-refresh.service`: BLOCKED (external) — fails honestly
  on every attempt because IBKR contract-details is unresponsive today.
  Health monitor R17 will continue to fire CRITICAL until the cache has
  non-empty chains. R18 will continue to fire WARNING until the next greeks
  publisher run reads a non-empty chain cache.
- `alpha_options`: fails closed — returns zero TradeSignals while
  `options_chains_cache.json::chains == {}`. Verified by tests
  `test_alpha_options_returns_no_signals_when_chain_cache_missing` and
  `test_alpha_options_returns_no_signals_when_chain_cache_empty`.
- `omega_momentum_options`: continues to operate using
  `estimate_contract_price` synthetic pricing and labels emitted signals
  `meta.synthetic_pricing=True` so downstream consumers never mistake the
  price for real market data. Verified by
  `test_omega_momentum_options_marks_synthetic_pricing_true`.

## No-live confirmation
- `runtime/live_readiness.json::ready_for_live == false`
- `runtime/decision_trace_heartbeat.json::allow_ibkr_live == false`
- `runtime/decision_trace_heartbeat.json::allow_ibkr_paper == true`
- `runtime/decision_trace_heartbeat.json::mode == {"chad_mode": "paper",
  "live_enabled": false}`
- No service restart performed. No systemd unit file modified. No
  configuration mutated. No broker orders sent. No runtime JSON edited
  by hand. Stop bus inactive.

## Next action if external proof is needed
1. Wait for the next non-holiday timer firing (Tue 2026-05-26 12:30 UTC).
   The retry/backoff should self-recover any transient IBKR farm
   degradation.
2. If 2026-05-26's run still fails, the structured failure artifact will
   show whether the `blocked_reason` flips (e.g.
   `ibkr_qualify_contracts_failed` or `mixed_failures`) and operators can
   investigate the IBKR Gateway side directly. The exact failed-attempt
   timestamps in `attempts.SPY` give cross-correlation against IBKR
   farm-status logs.
3. Once a healthy run produces non-empty chains, the next greeks publisher
   run (daily 23:49 UTC timer, or run on demand) will populate
   `options_greeks.json::symbols` and clear R18.

## Closure status
**BLOCKED** — local code/tests/artifacts harden the failure surface and
the system now reports the IBKR contract-details unresponsiveness
honestly via `runtime/options_chain_refresh_failure.json`. The only
remaining blocker is external IBKR contract-details endpoint
responsiveness, which is expected to clear on the next non-holiday
trading day.
