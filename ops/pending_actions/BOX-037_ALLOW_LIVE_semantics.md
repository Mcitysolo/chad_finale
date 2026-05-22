# BOX-037 — ALLOW_LIVE Semantics (locked)

- **Box:** 037 — GAP-016 ALLOW_LIVE semantics locked
- **Stage:** 3 — Engineering, tests, SSOT, hidden-gap closure
- **Document timestamp (UTC):** 2026-05-20T01:40:43Z
- **Document type:** policy/contract attestation (no live runtime mutation, no service restart, no live trading authorization)

This document defines the authoritative semantics of `operator_mode=ALLOW_LIVE`, the paper-mode auto-refresh behaviour, the deliberate-human-intent requirement for live trading, and the fail-closed behaviour for missing/expired/unknown intent. It is sourced from in-tree code, in-tree tests, the live systemd units, and read-only runtime evidence.

---

## 1. What `ALLOW_LIVE` means

`ALLOW_LIVE` is **one of three** canonical operator-intent modes recorded in `runtime/operator_intent.json` (`OperatorMode`):

| Mode | Meaning at the LiveGate boundary |
| --- | --- |
| `ALLOW_LIVE` | Operator-intent gate passes. Live trading **may** be authorized **only if** every other gate also passes (execution mode, live readiness, execution quality, SCR, mutation state, change canary, reconciliation, lifecycle truth). |
| `EXIT_ONLY` | Operator-intent gate denies new entries; exits may flow if downstream config explicitly allows. |
| `DENY_ALL` | Operator-intent gate denies everything. |

`ALLOW_LIVE` is **not** authorization to trade live — it is a *one-of-many* preconditions. The CHAD live gate is a deterministic AND of all gates (`chad/core/live_gate.py:776-1121`).

## 2. Who/what can write `operator_intent.json`

| Writer | Path | Mode it may write | Live-mode guard |
| --- | --- | --- | --- |
| Authority service (timer) | `chad/ops/operator_intent_refresher.py refresh` | `ALLOW_LIVE` (paper/dry_run only) | `refresh_mode_for_execution(live)` raises `refresh_refused_live_mode` — the timer cannot widen permissions in live |
| Authority service (set) | `chad/ops/operator_intent_refresher.py set --mode <…>` | any of `ALLOW_LIVE` / `EXIT_ONLY` / `DENY_ALL` | `write_intent` raises `write_refused_live_mode_without_allow_live_write` unless the operator passes `--allow-live-write` |
| Legacy CLI | `chad/core/operator_intent.py set --mode ALLOW` | maps to `ALLOW_LIVE` (back-compat) | n/a — pre-dates the authority service; deprecated; canonical writer is the authority service |
| Store API | `backend.operator_intent_store.OperatorIntentStore.set_intent` | any canonical mode | callers are themselves gated (the only callers are the two writers above) |

The systemd unit driving the timer-driven refresher:

```
/etc/systemd/system/chad-operator-intent-refresh.timer  → OnUnitActiveSec=10min
/etc/systemd/system/chad-operator-intent-refresh.service:
  Environment=CHAD_OPERATOR_INTENT_TTL_SECONDS=900     # 15-minute TTL
  Environment=CHAD_OPERATOR_INTENT_MODE=ALLOW_LIVE
  Environment=CHAD_OPERATOR_INTENT_REASON=auto_refresh_allow_entries_non_live
  ExecStart=… python3 -m chad.ops.operator_intent_refresher refresh
```

The combination produces a fresh `ALLOW_LIVE` entry every 10 minutes with a 15-minute TTL — **only** while execution mode is `paper`/`dry_run`. The moment execution mode is flipped to `live`, `refresh_mode_for_execution` raises and the timer can no longer write — the operator must use the explicit `set --mode ALLOW_LIVE --allow-live-write` path.

## 3. TTL / expiry rules

| Layer | Rule | Source |
| --- | --- | --- |
| Refresher default TTL | 900 s (15 min) | systemd env `CHAD_OPERATOR_INTENT_TTL_SECONDS=900`; refresher default `DEFAULT_TTL_SECONDS=900` |
| Refresher cadence | every 10 min | `chad-operator-intent-refresh.timer` `OnUnitActiveSec=10min` |
| Store hard ceiling | 7 days | `backend.operator_intent_store.MAX_TTL_SECONDS = 7 * 86400` (rejects larger TTLs at write time) |
| LiveGate fallback TTL when JSON omits `ttl_seconds` | 2 h | `chad/core/live_gate.py:422` — `default_ttl_s=2 * 60 * 60` |
| Freshness check | `age <= ttl` against UTC `ts_utc`; otherwise treated as stale | `backend.operator_intent_store.compute_freshness` + `chad/core/live_gate.py:_freshness_from_obj` |
| Stale handling | collapses to `DENY_ALL` (store) or `OperatorIntent(operator_mode="DENY_ALL", operator_reason="operator_intent_stale_or_invalid:…")` (LiveGate) | proven by `chad/tests/test_gap016_allow_live_semantics.py` |

## 4. Paper-mode auto-refresh behaviour

| Item | Value |
| --- | --- |
| When it runs | Every 10 minutes via `chad-operator-intent-refresh.timer` |
| What it writes (paper/dry_run) | `operator_mode=ALLOW_LIVE`, `operator_reason=auto_refresh_allow_entries_non_live`, `ttl_seconds=900`, fresh `ts_utc` |
| What it writes (live mode) | nothing — `refresh_mode_for_execution("live")` raises and exit code 2 is returned |
| Why this is safe | `ALLOW_LIVE` in `operator_intent.json` is one of many AND-gates; the LiveGate also requires `exec_mode=="live"` AND `live_readiness.ready_for_live=true` AND every other green gate. With `CHAD_EXECUTION_MODE=paper` (live state today), the `EXEC_MODE` gate at `live_gate.py:1090-1094` denies regardless of the operator-intent value. |
| Operator-visible reason string | `auto_refresh_allow_entries_non_live` — searchable in `runtime/operator_intent.json` and journals |

## 5. Live-mode deliberate human intent

To authorize live trading, an operator must:

1. **Promote execution mode** — change `CHAD_EXECUTION_MODE` in the systemd unit/drop-in from `paper` to `live` (governance Pending Action — never direct).
2. **Explicitly write operator intent in live mode** with the live-mode guard satisfied:
   ```
   python3 -m chad.ops.operator_intent_refresher set \
       --mode ALLOW_LIVE \
       --reason "<audit-grade reason text>" \
       --ttl-seconds 900 \
       --allow-live-write
   ```
   Without `--allow-live-write`, the call raises `write_refused_live_mode_without_allow_live_write` and writes nothing.
3. **Pass every other LiveGate gate**: stop, profit lock, live_readiness=true, execution quality (not "dangerous" or "unknown"), SCR (not paper_only/PAUSED), mutation state (no pending/rejected/lockout), change canary (factor=1.0 and no future expiry), reconciliation=GREEN, lifecycle truth (truth_ok=true, gap_flag=false, backlog_flag=false), `exec_mode=="live"`.
4. **Pass the final-allow flags**: `exec_cfg.ibkr_enabled AND not exec_cfg.ibkr_dry_run AND operator.allow_ibkr_live` (`live_gate.py:1099`).

Auto-refresh and stale state cannot reach this path. The timer-driven refresher cannot widen in live mode; the store-level loader collapses stale ALLOW_LIVE to `DENY_ALL`; the LiveGate loader independently collapses missing/unknown intent to `DENY_ALL`.

## 6. Fail-closed contract (locked by tests in `chad/tests/test_gap016_allow_live_semantics.py`)

| Path | Result | Test |
| --- | --- | --- |
| Refresher in live mode | `refresh_mode_for_execution("live")` raises | `test_refresh_mode_refuses_to_widen_in_live_mode` |
| Refresher in paper/dry_run | returns `ALLOW_LIVE` | `test_refresh_mode_returns_allow_live_in_non_live[paper]`, `[dry_run]` |
| Write in live mode without flag | `write_intent(..., allow_live_write=False)` raises, no file written | `test_write_intent_blocks_live_mode_without_allow_live_write` |
| Write in live mode with flag | writes ALLOW_LIVE with audit reason | `test_write_intent_allows_live_mode_with_explicit_allow_live_write` |
| Write in paper mode | writes without `--allow-live-write` | `test_write_intent_paper_mode_does_not_require_allow_live_write` |
| Missing `operator_intent.json` | store → `DENY_ALL`; LiveGate → `DENY_ALL` | `test_store_load_missing_file_returns_deny_all`, `test_live_gate_operator_intent_missing_file_yields_deny_all` |
| Malformed JSON | store → `DENY_ALL` | `test_store_load_malformed_json_returns_deny_all` |
| Unknown mode | store → `DENY_ALL`; LiveGate → `DENY_ALL` | `test_store_load_unknown_mode_returns_deny_all`, `test_live_gate_operator_intent_unknown_mode_yields_deny_all` |
| Stale TTL | store → `DENY_ALL` (reason `stale_or_missing`); LiveGate → `DENY_ALL` (reason `stale_or_invalid`) | `test_store_load_stale_ttl_returns_deny_all`, `test_live_gate_operator_intent_stale_yields_deny_all` |
| Legacy `operator_mode=ALLOW` | normalized to `ALLOW_LIVE` (back-compat) | `test_live_gate_normalizes_legacy_allow_to_allow_live` |
| Stale ALLOW_LIVE end-to-end | `evaluate_live_gate()` returns `DENY_ALL`, operator-intent gate trips first | `test_evaluate_live_gate_stale_allow_live_returns_deny_all` |
| Fresh paper-mode ALLOW_LIVE | `evaluate_live_gate()` returns `DENY_ALL` (paper exec_mode + ready_for_live=false fail upstream) | `test_evaluate_live_gate_paper_mode_blocks_live_even_with_fresh_allow_live` |

## 7. Current runtime state (read-only, 2026-05-20)

```json
// runtime/operator_intent.json
{
  "operator_mode": "ALLOW_LIVE",
  "operator_reason": "auto_refresh_allow_entries_non_live",
  "ts_utc": "2026-05-20T01:31:38Z",
  "ttl_seconds": 900
}

// runtime/live_readiness.json
{
  "latest_report_path": "/home/ubuntu/chad_finale/reports/live_readiness/LIVE_READINESS_20260520T012759Z.json",
  "latest_report_sha256": "sha256:b46eadaaa7ca36d27d5c237c9cf76670ecef15ed5929a84919fba5725967c8b8",
  "ready_for_live": false,
  "schema_version": "live_readiness_state.v1",
  "ts_utc": "2026-05-20T01:27:59Z",
  "ttl_seconds": 604800
}
```

| Signal | Value | Interpretation |
| --- | --- | --- |
| `operator_intent.operator_mode` | `ALLOW_LIVE` | passes the operator-intent gate |
| `operator_intent.operator_reason` | `auto_refresh_allow_entries_non_live` | written by the paper-mode timer-driven refresher |
| `live_readiness.ready_for_live` | `false` | **denies** at the live-readiness gate |
| `CHAD_EXECUTION_MODE` (systemd) | `paper` | **denies** at the exec-mode gate |
| Net live authorization | **NO** | gates fail-closed; no live execution authorized |

## 8. How to audit live state (commands)

```bash
# Current operator intent body (read-only):
cat runtime/operator_intent.json

# Authority-service view (freshness + canonical-mode validation):
python3 -m chad.ops.operator_intent_refresher show

# Strict verification (exit code 3 if anything fails contract):
python3 -m chad.ops.operator_intent_refresher verify --require-canonical-mode

# Live-readiness snapshot:
cat runtime/live_readiness.json

# Compute and print the LiveGate decision (does not mutate state):
python3 -m chad.core.live_gate

# Pretty live-gate summary with reasons:
python3 -m chad.core.show_live_gate

# Confirm live env does NOT carry live-mode promotion:
systemctl show chad-live-loop.service --property=Environment

# Test suite that pins these contracts:
python3 -m pytest chad/tests/test_gap016_allow_live_semantics.py -v
```

## 9. Forbidden by this document

- Do not edit `runtime/operator_intent.json` by hand. Use the authority service (`chad/ops/operator_intent_refresher.py set`) so the canonical schema, atomic write, audit reason, and TTL are enforced.
- Do not run `set --mode ALLOW_LIVE --allow-live-write` outside a governance-approved Pending Action.
- Do not flip `CHAD_EXECUTION_MODE` from `paper`/`dry_run` to `live` without a governance-approved Pending Action.
- Do not extend `CHAD_OPERATOR_INTENT_TTL_SECONDS` beyond the 7-day store ceiling. Re-apply intent periodically (auditable) instead.
- Do not weaken the live-mode write guard (`allow_live_write`).
- Do not weaken the refresher live-mode refusal (`refresh_mode_for_execution`).
- Do not introduce a writer that bypasses `OperatorIntentStore` — atomic writes + freshness validation are the single source of truth.

## 10. Cross-references

- LiveGate evaluator: `chad/core/live_gate.py` (`evaluate_live_gate`, `_load_operator_intent`)
- Operator-intent authority service: `chad/ops/operator_intent_refresher.py`
- Operator-intent store: `backend/operator_intent_store.py`
- Legacy operator-intent writer (deprecated): `chad/core/operator_intent.py`
- Live-mode toggle state: `chad/core/live_mode.py`, `chad/core/set_live_mode.py`
- Execution mode resolver: `chad/execution/execution_config.py`
- LiveGate readiness TTL contract: `chad/tests/test_live_readiness_ttl.py`
- ALLOW_LIVE semantics tests (new): `chad/tests/test_gap016_allow_live_semantics.py`
- systemd units: `/etc/systemd/system/chad-operator-intent-refresh.{timer,service}`
- Evidence: `runtime/completion_matrix_evidence/BOX-037_GAP-016_ALLOW_LIVE_semantics_locked.md`
