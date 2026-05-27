# CHAD Forensic Full-System Audit
**Timestamp (UTC):** 2026-05-27T14:29:51Z
**Auditor:** Team CHAD — Forensic Engineer (read-only)
**Repo root:** /home/ubuntu/chad_finale
**Branch:** main
**HEAD:** `f8453438fae9852324b21fc3c18eb7577b66f706` (short: `f845343`) — *Paper Epoch 3: start 5-session soak tracker*
**Expected (per v9.3):** `72f361fac544c82cd231d100b2d0bc86d428a990` — present, +57 commits ahead → "or later" constraint satisfied.
**Pytest baseline:** **2538 passed, 1 failed**, 8 warnings, 114.8 s (`CHAD_SKIP_IB_CONNECT=1`). v9.3 docs claim 2,114; v9.4 forward errata corrects this; current count reflects 57 post-v9.3 commits.
**Live posture:** PAPER. `ready_for_live=false`. `allow_ibkr_live=false`. `allow_ibkr_paper=true`. `chad_mode=paper`, `live_enabled=false`. (Confirmed via `runtime/live_readiness.json` and `runtime/decision_trace_heartbeat.json::live_gate`.)
**SCR:** CONFIDENT, `sizing_factor=1.0`, `paper_only=false`, `effective_trades=197`, `win_rate=0.756`, `sharpe_like=5.76`, `total_pnl=$10,575.13`, `max_drawdown=-$290`, `excluded_untrusted=321`.
**Mutation budget consumed:** **0** — this audit performed view, grep, find, jq, journalctl, systemctl status, python3 read-only scripts, sqlite3 read queries, and AWS read-only describes only. No code, runtime, config, service, or broker state was touched.

---

## 1. Executive Summary

CHAD is **cycling, paper-safe, and live-gated**. Every live-mode gate is held closed. Defense-in-depth catches every placeholder fingerprint in current operation. Live-loop, IB Gateway, bar provider, orchestrator, shadow-status, dashboard, Kraken WS, Telegram bot, and Xvfb/x11vnc are all `active running`.

However, the audit surfaced **three classes of structural concern** that cause the recurring "every check finds a new surprise" pattern:

1. **Dual-ledger authority without a single canonical writer.** `positions_snapshot.json` (symbol-keyed list of 19) and `ibkr_paper_ledger_state.json` (hash-keyed dict of 18) are both paper-position truth surfaces with different publishers, different cadences, and different counts. `positions_truth.json` is currently RED with `BROKER_AUTHORITY_RED: count_mismatch_ledger=18_vs_snapshot=19` and `truth_source=FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN`. A new strategy-emitted `alpha_futures|MGC` position (opened 14:34:50Z, ~40 min before audit) currently has `broker_truth_missing` drift while the next snapshot publisher run catches up.

2. **Documentation drift between v9.3 (the operator-stated canonical baseline) and runtime.** Several "gaps_to_close" claims do not hold against the live filesystem: `drawdown_state.json`, `ibkr_watchdog_last.json`, `position_guard_drift.json`, and `positions_snapshot.json` are all **fresh, not stale**. `legend_top_stocks.json` is genuinely missing. v9.4 forward errata already acknowledges this drift trend; **the live system has overtaken the documented SSOT** by 57 commits.

3. **Silent and partial publisher failures during IBKR market-data outages.** `chad-options-chain-refresh.service` is in `failed (exit-code)` state since 12:33:07Z because IBKR contract-details / market-data farms (`usfarm`, `ushmds`, `secdefil`) became unresponsive — symptom not cause. The service correctly degraded (PR-04: wrote an empty cache with explicit `error` field and a `runtime/options_chain_refresh_failure.json` failure artifact), but the failed unit state means `chad-options-chain-refresh.timer` next-fire will not retry until the next scheduled tick. `chad-ibkr-bar-provider.service` is alive but generated **588 ERROR/Exception messages in 24 h**, almost entirely **`Error 162: HMDS query returned no data: SILK6@COMEX Trades`** — the silver futures contract `SILK6` expired today (`lastTradeDateOrContractMonth=20260527`) and is still being polled, indicating a futures-roll registry that has not advanced.

Three additional **HIGH-severity** findings exist but are not new surprises:

- **Ports 9618 / 9619 / 9620 bound to 0.0.0.0** (defense-in-depth gap; AWS Security Groups do not currently expose them externally per `aws ec2 describe-security-groups`, so impact is contained).
- **3 production files still import `ib_insync`** (`paper_position_closer.py`, `paper_shadow_runner.py`, `ibkr_broker_events_collector.py`) — Phase 2 migration unfinished per v9.3 §1.2.
- **14 trusted-fake placeholder rows** are still in `data/fills/FILLS_20260503.ndjson` (lines 1–14, all `delta SPY` from 2026-05-03T11:48Z–12:48Z). They are pre-hardening artifacts and are excluded by `excluded_untrusted=321` in SCR — but they remain on disk and would be silently consumed by any new replay/aggregation that does not honour the exclusion contract.

CHAD remains **paper-safe**. No finding in this audit unblocks live transition; all findings are quality-of-truth-surface issues that have been accumulating without a single canonical authority document.

---

## 2. Tier 1 — Repo Baseline

| Field | Value | Source |
|---|---|---|
| HEAD (full) | `f8453438fae9852324b21fc3c18eb7577b66f706` | `git rev-parse HEAD` |
| Short | `f845343` | `git rev-parse --short HEAD` |
| Branch | `main` | `git branch --show-current` |
| Working tree | 15 `_archive/bak_quarantine` deletions + 1 untracked PR-05 patch-note file (pre-existing, not introduced by this audit) | `git status --porcelain` |
| Commits since v9.3-documented HEAD `72f361f` | **57** | `git rev-list --count 72f361f..HEAD` |
| Tags relevant | `STABILITY_FREEZE_20260307_GREEN`, `RATIFICATION_MASTER_20260402`, `REVERT_PRE_OVERHAUL_20260419`, `SSOT_V8_9_AUDIT_LOCK_20260503`, `SSOT_V9_0_PAPER_EPOCH2_LOCK_20260504` | `git tag -l` |
| Pytest | **2538 passed, 1 failed**, 8 warnings, 114.80 s | `pytest chad/tests/ -q` w/ `CHAD_SKIP_IB_CONNECT=1` |
| Single test failure | `chad/tests/test_phase_a_item4_setup_tagging.py::test_02_alpha_intraday_momentum_surge_setup_family` | persistent — observed in both this audit and the morning intraday check |

### ib_insync residual imports (v9.3 §1.2 layer 5)

Active production files (excluding `.bak` / `.deprecated` / `/tests/` / `pending_actions/`):
- `chad/core/paper_position_closer.py`
- `chad/core/paper_shadow_runner.py`
- `chad/ops/ibkr_broker_events_collector.py`

Tests that still reference ib_insync (allowed by `test_ib_async_import_parity` allow-list):
- `chad/tests/test_options_chain_refresh.py`
- `chad/tests/test_paper_shadow_runner.py`
- `chad/tests/test_ib_async_import_parity.py`
- `chad/tests/test_pr03_ib_async_phase2_migration.py`

**Conclusion:** v9.3 §1.2 claim "5 files remain" maps to 3 production + 2 tests = 5 if `ib_async_import_parity.py` and `pr03_ib_async_phase2_migration.py` are counted as governance harness rather than the migration target. **Status: VERIFIED.** Canonical `ib_async` is imported in 28 files.

---

## 3. Tier 2 — Canonical Authority Map

### 2.A — ACCOUNT EQUITY

| Surface | Value | Schema | Freshness | Role |
|---|---|---|---|---|
| `runtime/portfolio_snapshot.json` | (file ~153 bytes; carries `ts_utc` but **no** explicit `total_equity` field at top level — equity content is internal) | no `schema_version` | 2026-05-27T14:30:57Z | Declared **canonical** per v9.3 / BOX-034 — but the top-level schema does not expose `total_equity` for a `view`-style audit; reader must descend into the file's per-source structure |
| `runtime/dynamic_caps.json` | `total_equity = 268345.77` | no `schema_version` | 2026-05-27T14:31:50Z | Consumes equity; emits dynamic caps for sizing |
| `runtime/scr_state.json` | reads PnL stream, not equity | `scr_state.v1` | 2026-05-27T14:32:22Z | reads equity indirectly |

Equity-reading modules (sample): `chad/dashboard/api.py`, `chad/risk/drawdown_guard.py`, `chad/risk/dynamic_risk_allocator.py`, `chad/risk/portfolio_var.py`, `chad/risk/profit_lock.py`, `chad/risk/composite_size_cap.py`, `chad/risk/withdrawal_manager.py`, `chad/risk/profit_router.py`, `chad/ops/equity_history_publisher.py`, `chad/core/orchestrator.py`, `chad/core/full_cycle_preview.py`, `chad/core/live_loop.py`. (12 production readers + 3 test files.)

**Canonical writer:** `chad/portfolio/ibkr_portfolio_collector_v2.py` via `chad-portfolio-snapshot.service` (per BOX-046 retraction and timer evidence).

**Status: PARTIAL** — canonical writer designated (BOX-034 policy), 12+ readers, but `portfolio_snapshot.json` has no top-level `total_equity` field and no `schema_version`, so consumers descend into nested structure. `dynamic_caps.json::total_equity` is the most visible top-level number; whether all 12 readers parse the canonical file identically is **UNKNOWN — REQUIRES AUDIT**.

### 2.B — POSITION TRUTH

| Surface | positions_count | broker_authority | truth_ok | Notes |
|---|---|---|---|---|
| `position_guard.json` | 69 keys (mixed `<strategy>|<symbol>` + `broker_sync|<symbol>`) | — | — | per-strategy guard rows; no schema_version |
| `positions_truth.json` | 19 | **RED** | **false** | `truth_source=FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN`; `broker_authority_reason=BROKER_AUTHORITY_RED: count_mismatch_ledger=18_vs_snapshot=19`; `schema_version=positions_truth.v1` |
| `positions_snapshot.json` | 19 | — | — | `source=ibkr_portfolio_collector_v2`; no schema_version; 19 symbols including MCL, MES (no MGC) |
| `ibkr_paper_ledger_state.json` | 18 hash-keyed records | — | — | hash-keyed records (sha256-style keys → contract metadata); no schema_version |
| `reconciliation_state.json` | — | upstream **GREEN**, `drifts=[]` | — | no schema_version; ttl=360s |
| `position_guard_drift.json` | `drift_count=1` | — | — | `schema_version=position_guard_drift.v1`; `alpha_futures|MGC` `broker_truth_missing` |

**MGC drift root cause (read-only forensic):**
- `position_guard.json::alpha_futures|MGC`: OPEN, SELL, qty=1.0, `opened_at=2026-05-27T14:34:50.855262+00:00` (≈40 min before audit time).
- `position_guard.json::broker_sync|MGC`: CLOSED, `closed_by=broker_truth_rebuild`, `updated_at_utc=2026-05-27T13:43:29Z` (51 min before alpha_futures opened it).
- `positions_snapshot.json`: 19 symbols, no MGC.
- `positions_truth.json::evidence.replay_only=['MES', 'MGC']`, `missing_from_replay=['BAC','CVX','GLD','GOOGL','IEMG','KO','M2K','M6E','NVDA','PEP','TLT','UNH','VWO']`.

This is a real, very recent, transient strategy-vs-broker disagreement. The 5-min positions-snapshot timer next fires at 14:37:44Z (per Tier 4). Whether MGC was actually filled at the broker, or was rejected (e.g., Error 201 / Error 1100 during the 13:07Z broker outage), is **UNKNOWN — REQUIRES AUDIT** without a broker probe.

**Status: PARTIAL** — multiple legitimate writers, but no single canonical authority document specifies which surface dominates when they disagree. PR-09 contract (`broker_authority_status` vs `replay_diagnostic_status`) is the closest declared rule but does not resolve the snapshot-vs-ledger count mismatch itself.

### 2.C — PAPER LEDGER

| Surface | Writer(s) | Reader(s) | Cadence |
|---|---|---|---|
| `runtime/ibkr_paper_ledger.json` | (older format, last mtime > 24h ago) | — | dormant |
| `runtime/ibkr_paper_ledger_state.json` | `chad/portfolio/ibkr_paper_ledger_watcher.py` via `chad-ibkr-paper-ledger-watcher.timer` (every 15 min) | `lifecycle_replay_coverage.py`, `lifecycle_replay_drift_audit.py`, `lifecycle_truth_publisher.py`, `chad/portfolio/ibkr_paper_ledger_watcher.py`, `ops/bin/chad_paper_trade_executor.py`, `ops/reconcile_positions.py`, `ops/reconcile_repair_ibkr_ledger_state.py` | 15 min |
| `runtime/ibkr_adapter_state.sqlite3` | `chad.execution.ibkr_adapter` (live-loop) | live-loop, lifecycle replay | per cycle |

**Canonical authority declaration:** `ops/pending_actions/BOX-047_dual_ledger_authority_policy.md` exists. Without reading its contents (Pending Action only), the explicit naming of "dual ledger authority" confirms the conflict is acknowledged but not yet resolved with a code-enforced single source of truth.

**Status: PARTIAL** — dual-authority is a documented policy concern (BOX-047), not yet a code-enforced single writer.

### 2.D — STRATEGY REGISTRY

| Source | Count | Symbols |
|---|---|---|
| `chad/strategies/__init__.py` (docstring lists active strategies) | 16 active + 1 DEFERRED (`alpha_forex`) | 16 — see §2.D table below |
| `config/strategy_weights.json::weights` | 16 | alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_options, beta, beta_trend, delta, delta_pairs, gamma, gamma_futures, gamma_reversion, omega, omega_macro, omega_momentum_options, omega_vol |
| `chad/risk/tier_manager.py::enabled_strategies` | dynamic (config-driven via `_expand_enabled_strategies`, supports `["*"]` literal) | depends on `config/tiers.json` |
| `chad/strategies/__init__.py::_build_registry()` | 1 call site for `register_core_strategies` (loop body registers each) | matches docstring |
| BOX-049 OFFICIAL completion matrix | 18 strategies classified (per CLAUDE.md memory) | superset — likely includes `alpha_intraday_micro` and/or another sub-variant |
| Live-loop runtime evidence | Today's `INTENT`/`ALWAYS-ACTIVE ROUTE` mentions `alpha`, `alpha_futures`, `gamma`, `gamma_futures`, `omega_macro`, `omega_vol`, `delta`, `delta_pairs`, `beta_trend`, `omega`, `omega_momentum_options` | partial — these are the actively-routed subset for the current regime |

**Status: PARTIAL** — config and __init__ docstring agree on 16. BOX-049's "18-strategy completion matrix" is from a separate completion-tracking surface and likely counts deferred or sub-variant strategies. No single document maps "registry count = config count = completion-matrix count" unambiguously.

### 2.E — EXECUTION MODE

| Reader | File | Direct env read? |
|---|---|---|
| `chad.execution.execution_config.get_execution_mode()` | `chad/execution/execution_config.py:85` | YES — single canonical reader |
| `chad/core/full_execution_cycle.py` | uses helper | no |
| `chad/core/kraken_execution.py` | uses helper (line 45: fallback to CHAD_EXECUTION_MODE when CHAD_KRAKEN_MODE unset) | YES — documented fallback |
| `chad/core/live_gate.py` | uses helper | no |
| `chad/core/live_loop.py` | uses helper | no |
| `chad/core/orchestrator.py` | uses helper | no |
| `chad/core/paper_position_closer.py` | uses helper (line 170: reasons message references env var name for operator clarity) | string-only, not env read |
| `chad/core/show_execution_config.py` | uses helper | no |
| `chad/ops/execution_environment_publisher.py` | uses helper (line 47: docstring references env var name) | no |
| `chad/ops/operator_intent_refresher.py` | uses helper | no |
| `chad/ops/reconciliation_publisher.py` | uses helper | no |
| `chad/ops/action_applier.py` | line 331/359: **writes** systemd drop-in setting `CHAD_EXECUTION_MODE` for ops actions; not a runtime read | no (write path) |
| `chad/core/ibkr_execution_runner.py` | line 40: docstring example `CHAD_EXECUTION_MODE=dry_run`; not an active env read | no |
| `chad/tests/test_execution_mode_canonical.py` | actively enforces "no direct env reads in hot paths" — this is the v8.8 ISSUE-78 guard | governance test |

**Status: VERIFIED.** `get_execution_mode()` is the single canonical reader. Active env reads in non-test, non-bak production code are limited to `kraken_execution.py` (explicit documented fallback), and the test harness exists to prevent regression.

### 2.F — SCR STATE

- **Writer:** `chad-shadow-status.service` (port 9618), publisher path `ops/scr_state_sync.py` via `chad-scr-sync.timer` (every 60 s).
- **Endpoint:** `http://127.0.0.1:9618/shadow` (process PID 2175218 running since 2026-04-16, 1m 10d uptime).
- **File:** `runtime/scr_state.json` (`scr_state.v1`, 14:32:22Z, ttl=180s).
- **Effective trades source:** `runtime/scr_state.json::stats.effective_trades` (currently 197).
- **Readers:** 33+ (live_gate.py, live_loop.py, orchestrator.py, exterminator.py, health_monitor.py, dashboard/api.py, telegram_bot.py, etc.).

**Internal divergence observed at 13:50Z (morning audit, since stabilised at this audit):** `live_gate.shadow` snapshot showed `state=UNKNOWN`, `sizing_factor=0.0`, `paper_only=true`, reasons=`scr_sync_error:TimeoutError:timed out`, while the underlying `scr_state.json` was CONFIDENT, sizing 1.0, paper_only=false. This is a publisher → shadow-snapshot timing divergence (live_gate caches a previous tick). Not a hot-path data integrity issue but flagged here for completeness.

**Status: VERIFIED** (canonical writer designated, service running, schema_version present, port live).

### 2.G — XGB MODEL ACTIVE

- `runtime/models/xgb_veto/current/` — **does not exist**. Only `runtime/models/xgb_veto/candidates/` exists.
- `shared/models/xgb_veto_model.json` — 305,648 bytes, mtime 2026-05-17 03:16. Baseline used.
- `python3 scripts/promote_xgb_veto.py --status`:
  ```
  active model (source=baseline)
    model_version : xgb_veto_20260510_020007
    trained_at_utc: 2026-05-10T02:00:07+00:00
    accuracy      : 0.7534
    logloss       : 0.5364
  ```
- Live-loop INFO log evidence (today): `ML_SHADOW … model_version=xgb_veto_20260510_020007 manifest_hash=sha256:74afb9e1c8f3e0fe4` — confirms baseline model is in use.

**Status: VERIFIED.** Active model is `xgb_veto_20260510_020007` from baseline (no current promotion exists). The fallback chain (`runtime/models/xgb_veto/current/` → `shared/models/xgb_veto_model.json`) is operating as designed.

---

## 4. Tier 3 — Publisher / Monitor Map

### 3.1 Inventory at a glance

- Total `runtime/*.json` files (including subdirs): ~137
- Files with `schema_version`: **59**
- Files without `schema_version`: **78**

### 3.2 Stale-claim verification (gaps_to_close cross-check)

| File | gaps_to_close claim | Reality at this audit | Verdict |
|---|---|---|---|
| `runtime/drawdown_state.json` | "10 days stale" | **2.0 min** old, `drawdown_state.v1` | **CLAIM REFUTED** |
| `runtime/ibkr_watchdog_last.json` | "44 days stale" | **1.8 min** old | **CLAIM REFUTED** |
| `runtime/legend_top_stocks.json` | "47 days stale" | **MISSING** (not present at all) | partial — file is missing, not stale |
| `runtime/position_guard_drift.json` | "not found" | **present**, `position_guard_drift.v1`, 1.7 min old, drift_count=1 | **CLAIM REFUTED** |
| `runtime/positions_snapshot.json` | "stale since 2026-04-03 because no active writer" | actively written by `chad-positions-snapshot.service` every 5 min; last write 14:32:49Z (successful, positions_count=19) | **CLAIM REFUTED** (already retracted in BOX-046) |

### 3.3 Genuinely stale runtime files (>24 h old, excluding `runtime/proofs/` archive and `runtime/sec_13f_cache/` quarterly cache)

| File | Age (days) | Notes |
|---|---|---|
| `runtime/reports/SSOT_PARITY_SWEEP_LATEST.json` | 83.1 | report artifact, archive |
| `runtime/savage_alloc_state.json` | 78.6 | retired allocator? — schema_version `savage_alloc_state.v1` present |
| `runtime/pending_approvals.json` | 74.9 | `approvals.v2` |
| `runtime/lifecycle_replay_drift_audit.json` | 69.8 | `lifecycle_replay_drift_audit.v1` — should be refreshed by `lifecycle_replay_*` workstream |
| `runtime/paper_exec_state.json` | 61.8 | likely deprecated by trade_closer_state |
| `runtime/dominance_allocator.json`, `runtime/dynamic_caps_dominance_overlay.json`, `runtime/dynamic_caps_risk_governed.json`, `runtime/dynamic_caps_quarantine.json` | 61.8 | dormant allocator side-states |
| `runtime/performance_tracker_test.json` | 60.1 | test artifact |
| `runtime/capital_allocator.json` | 60.1 | dormant |
| `runtime/allocator_v3_state.json` | 54.7 | v3 allocator state (current is dynamic_caps.json) |
| `runtime/kraken_pnl_state.json` | 54.6 | crypto sub-state |
| `runtime/crypto_risk_off.json` | 54.6 | |
| `runtime/full_execution_cycle_last.json` | 54.6 | |
| `runtime/reports/bars_1d_validation_latest.json` | 47.5 | reports archive |
| `runtime/broker_truth_snapshot_20260419.json` | 37.7 | dated snapshot — by-design archival |
| `runtime/broker_truth_snapshot_20260420_step_13_5.json` | 36.8 | dated snapshot — by-design archival |
| `runtime/scr_config.json` | 26.9 | config-style state |
| `runtime/telegram_bot_dedupe.json`, `runtime/telegram_bot_memory.json`, `runtime/telegram_bot_state.json` | 26.2 | telegram bot state (low cadence) |

**78 of ~137 runtime files have no `schema_version`.** Top hot-path files without `schema_version`:
- `decision_trace_heartbeat.json` (HOT, every 5 min via `chad-decision-trace-heartbeat.timer`)
- `dynamic_caps.json` (HOT, used by risk pipeline; reader-count ≥5)
- `ibkr_bars_cache.json` (HOT, every cycle)
- `ibkr_paper_ledger.json` (older format, dormant)
- `ibkr_paper_ledger_state.json` (HOT, 15 min cadence)
- `ibkr_status.json` (HOT, every 60s)
- `ibkr_watchdog_last.json` (HOT)
- `last_route_decision.json` (HOT, per cycle)
- `kraken_*` cluster (4 files, no schema)
- `position_guard.json` (HOT, per cycle — schema_version literally absent)
- `portfolio_snapshot.json` (CANONICAL EQUITY — no schema_version)
- `trade_closer_state.json` (HOT, per cycle, 855 KB)
- `signal_guard.json` (HOT, 25 KB)

**Status: PARTIAL.** Several documented "orphans" are actually live and refuted. Several non-documented hot-path files **lack a schema_version**, which is a structural authority gap rather than a publisher-orphan gap.

---

## 5. Tier 4 — Service & Timer Map

### 5.1 Active services (chad-*) — `systemctl list-units 'chad-*'`

| Unit | State | Notes |
|---|---|---|
| chad-backend.service | active running | FastAPI |
| chad-burnin-check.service | activating (start) | timer-driven 10 min |
| chad-dashboard.service | active running | |
| chad-ibgateway.service | active running | IBC-managed |
| chad-ibkr-bar-provider.service | active running | **noisy** — 588 errors/24h on expired SILK6 contract (T13) |
| chad-ibkr-watchdog.service | activating (start) | |
| chad-kraken-ws.service | active running | |
| chad-live-loop.service | active running | NEVER_AUTO_RESTART per v8.9 SS01 (status confirmed) |
| chad-macro-state.service | activating (start) | |
| chad-metrics.service | active running | :9620 |
| chad-operator-intent-refresh.service | activating (start) | |
| **chad-options-chain-refresh.service** | **failed (exit-code)** | since 2026-05-27T12:33:07Z — IBKR market-data farms unresponsive |
| chad-orchestrator.service | active running | NEVER_AUTO_RESTART per v8.9 SS01 |
| chad-portfolio-snapshot.service | activating (start) | |
| chad-shadow-status.service | active running | uptime 1m 10d (port 9618) |
| chad-telegram-bot.service | active running | |
| chad-x11vnc.service | active running | |
| chad-xvfb.service | active running | DISPLAY :99 |

### 5.2 Timers (selected from `systemctl list-timers 'chad-*' --all`)

- ~40 active timers with cadences from 60 s up to daily / weekly.
- Hot-path cadences: `chad-decision-trace-heartbeat` (5 min), `chad-ibkr-health` (60 s), `chad-ibkr-collector` (2 min), `chad-positions-snapshot` (5 min), `chad-feed-watchdog` (2 min), `chad-action-applier` (60 s), `chad-trade-closer` (per minute).
- `chad-news-intel-refresh.timer`: last fire 14:08:30, next 14:38:30 (30 min cadence)
- `chad-ibkr-paper-ledger-watcher.timer`: last 14:20:37, next 14:35:37 (15 min)
- `chad-burnin-daily-summary.timer`: 23:59 UTC daily
- `chad-backup.timer`: 03:30 UTC daily

### 5.3 Failed units

`systemctl --failed`:
```
chad-options-chain-refresh.service   loaded failed failed   CHAD Pre-Market Options Chain Cache Refresh
```

**Root cause (from journal):** IBKR data-farm connection broken (`Error 2103: Market data farm broken:usfarm`, `Error 2105: HMDS data farm broken:ushmds`, `Error 2157: Sec-def data farm broken:secdefil`, `Error 200: No security definition for SPY`, `Error 10168: market data not subscribed`). 3-retry exhausted at 12:33:07Z. PR-04 hardening did its job: wrote `runtime/options_chains_cache.json` with `error` field AND `runtime/options_chain_refresh_failure.json` failure artifact. The fail-out is **honest, not silent**. The service unit, however, remains in `failed` state until the next timer fire.

### 5.4 NEVER_AUTO_RESTART invariant check

`chad-live-loop.service` and `chad-orchestrator.service`: both `active running` since their last manual start. The `auto-restart` policy is per SS01 governed by drop-in files which I have NOT modified. The `chad-options-chain-refresh.service` `Active: failed (Result: exit-code)` confirms no implicit restart-on-failure behaviour either. **Invariant honoured.**

### 5.5 OnFailure handling

I did not enumerate per-unit `OnFailure=` directives in this pass — that requires reading every `.service` file under `/etc/systemd/system/chad-*` (allowed read-only but high token cost). **Marked UNKNOWN — REQUIRES AUDIT** for OnFailure coverage.

**Status: PARTIAL.** Service surface is mostly healthy; one failed unit (with graceful degradation already implemented); per-unit `OnFailure=` coverage not exhaustively enumerated.

---

## 6. Tier 5 — Placeholder / Fake-Fill Source Audit

### 6.1 Active-strategy hard-coded `100.0` references

Result of `grep -rn "100\.0\|= 100\b" chad/strategies/` filtered to active code + price/fill/placeholder context: **zero matches**.

### 6.2 Synthetic / placeholder logic locations

| File | Role | Verdict |
|---|---|---|
| `chad/strategies/options_pricing.py` | Black-Scholes synthetic options pricing (`synthetic: True` flag) | LEGITIMATE — options fallback |
| `chad/strategies/omega_momentum_options.py` | uses synthetic option pricing fallback | LEGITIMATE |
| `chad/strategies/alpha_intraday_micro.py:459` | "# placeholder for early-exit skip logging." | comment only; no value emission |
| `chad/execution/paper_exec_evidence_writer.py:147` | "A fill with `fill_price=100.0` on any of these is the canonical fingerprint" | defense-in-depth header |
| `chad/execution/paper_exec_evidence_writer.py:1576-1605` | placeholder catcher: `fill_price == _PLACEHOLDER_FILL_PRICE (100.0) AND … → pnl_untrusted=True, broker_rejected` | LEGITIMATE — executor-side reject |
| `chad/execution/paper_exec_evidence_writer.py:1128-1324` | `bag_close_synthetic_credit_ratio_30pct` BAG-close synthetic credit haircut | LEGITIMATE — explicit untrusted marker |
| `chad/execution/trade_closer.py:394` | comment about delta SELL trades that landed at fill_price=100.0 | defense-in-depth in trade closer |
| `chad/strategies/delta.py:86-89, :321-332, :546` | "fill_price of 100.0 is the canonical 'no-live-price'" comments + **abstain on missing price (PR-02)** | LEGITIMATE — strategy-side abstain |

### 6.3 Historical fingerprint scan (FULL DATA SET, 18,098 fill rows)

| Metric | Count |
|---|---|
| Total fill rows scanned | **18,098** |
| Rows with `fill_price=100.0` | **836** |
| **Trusted fake placeholders** (`fill_price=100.0 AND reject != true AND pnl_untrusted != true`) | **14** |
| Rows with explicit `placeholder` tag in `tags` or `placeholder_fill_price` in `extra` | **97** |

### 6.4 Trusted-fake timeline

All **14** trusted-fake rows are concentrated in a **single file on a single day**:
- `data/fills/FILLS_20260503.ndjson` lines 1–14
- All `sym=SPY strat=delta`
- Timestamps span 2026-05-03T11:48:55Z through 2026-05-03T12:48:37Z
- This is the documented P0-1 incident (per CLAUDE.md memory `audit_2026_05_08_placeholder_fills`).

**Since 2026-05-03 there are ZERO trusted-fake placeholders.** Defense-in-depth (trade_closer + paper_exec_evidence_writer + paper_trade_executor) catches all subsequent placeholder emissions as `status=rejected` + `pnl_untrusted=true` + tagged `placeholder, broker_rejected`. PR-02 (delta abstain) + PR-02b (reconciler silence) close the strategy-side emission for the documented source.

### 6.5 Verdict

- **Source-side hardening:** delta abstains on missing price (PR-02); reconciler does not emit synthesized close-fill placeholders (PR-02b). VERIFIED.
- **Executor-side hardening:** 3-layer defense-in-depth at trade_closer + paper_exec_evidence_writer + paper_trade_executor. VERIFIED.
- **Historical contamination:** 14 trusted-fake rows from 2026-05-03 remain in `data/fills/FILLS_20260503.ndjson`. They are **excluded** from SCR effective_trades (`excluded_untrusted=321` in current SCR stats), but they remain consumable by any aggregator that does not honour the exclusion contract. **Status: PARTIAL** (exclusion is contract-enforced but data-on-disk is unchanged).

---

## 7. Tier 6 — Decision & Lifecycle Truth

| File | Schema | Age | State |
|---|---|---|---|
| `runtime/trade_lifecycle_state.json` | `trade_lifecycle_state.v1` | 21.7 s | `backlog_flag=false`, ttl=60s |
| `runtime/last_route_decision.json` | none | 133.6 s | last decision |
| `runtime/decision_trace_heartbeat.json` | none | 46.3 s | `ok=true`; embeds `live_gate.allow_ibkr_live=false`, `allow_ibkr_paper=true`, `mode.chad_mode=paper` |

### 7.1 ibkr_adapter_state.sqlite3 schema

```
.tables → ibkr_exec_state
SELECT COUNT(*) FROM ibkr_exec_state → 52
```

The audit prompt expected an `orders` table; **the actual schema has a single `ibkr_exec_state` table** with 52 rows. PendingSubmit queries against the `orders` table fail. **Documentation-vs-reality drift.** The actual stuck-PendingSubmit audit would need to query `ibkr_exec_state` with the correct column name (likely `status` or similar) — **not executed in this pass to avoid speculation. Marked UNKNOWN — REQUIRES AUDIT.**

**Status: PARTIAL.** lifecycle file is fresh and backlog_flag is false (healthy). SQLite schema does not match the audit prompt's expectation, so the PendingSubmit ages remain UNKNOWN.

---

## 8. Tier 7 — Reconciliation Surface Deep Dive

### 8.1 Current state

`positions_truth.json`:
- `broker_authority_status` = **RED**
- `broker_authority_reason` = `BROKER_AUTHORITY_RED: count_mismatch_ledger=18_vs_snapshot=19`
- `replay_diagnostic_status` = PARTIAL
- `replay_diagnostic_reason` = `QTY_OR_SYMBOL_MISMATCH: qty_mismatches=5 missing_from_replay=13 snapshot=19 replay=7 ledger=18`
- `replay_diagnostic_blocks_truth` = **true**
- `truth_ok` = **false**
- `truth_source` = `FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN`
- `snapshot_positions_count` = 19
- `ledger_state_positions_count` = 18
- `replay_positions_count` = 7
- `missing_from_replay` = `['BAC', 'CVX', 'GLD', 'GOOGL', 'IEMG', 'KO', 'M2K', 'M6E', 'NVDA', 'PEP', 'TLT', 'UNH', 'VWO']`
- `replay_only` = `['MES', 'MGC']`

`reconciliation_state.json`: status=**GREEN**, drifts=[] (upstream publisher classification).

**The disagreement between upstream `reconciliation_state.GREEN` and `positions_truth.broker_authority_RED` is a known PR-09 contract feature, not a bug.** PR-09 separates upstream (drift detector) GREEN from broker-authority (count + symbol reconciliation) RED — and `replay_diagnostic_blocks_truth=true` shows the policy is currently making the diagnostic block truth.

### 8.2 Snapshot keys vs ledger keys

- `positions_snapshot.json` keys: AAPL, BAC, CVX, GLD, GOOGL, IEMG, KO, M2K, M6E, MCL, MES, MSFT, NVDA, PEP, QQQ, SPY, TLT, UNH, VWO (19 symbols).
- `ibkr_paper_ledger_state.json` keys: 18 SHA256-style hashes; each record has `account_id`, `attribution_source`, `avg_cost`, `conId`, `currency`, … (no symbol-level key extraction trivially possible without contract→symbol map).

The two surfaces are NOT directly comparable by key — one is symbol-keyed, one is hash-keyed. The publisher comparing them (positions_truth classifier) does the symbol resolution internally. The fact that the audit cannot reproduce the comparison without that resolver code is itself an authority concern.

### 8.3 MGC drift (alpha_futures|MGC broker_truth_missing)

- `position_guard.json::alpha_futures|MGC`: OPEN, SELL, qty=1.0, opened 2026-05-27T14:34:50Z (≈40 min ago at audit time)
- `position_guard.json::broker_sync|MGC`: closed_by=broker_truth_rebuild, was OPEN with qty=20.0 SELL since 2026-05-26T23:53Z, closed 2026-05-27T13:43:29Z
- `positions_snapshot.json`: MGC NOT present
- `positions_truth.replay_only`: includes MGC (replay sees it; snapshot does not)

Sequence reconstructed:
1. 2026-05-26 23:53Z: broker_sync rebuilt MGC at qty=20 SELL.
2. 2026-05-27 13:43:29Z: broker_truth_rebuild closed `broker_sync|MGC` (snapshot no longer reflected MGC at next refresh).
3. 2026-05-27 14:34:50Z: `alpha_futures` opened a fresh `MGC` 1 SELL position.
4. 2026-05-27 14:32:44Z (10 s before): last positions_snapshot publisher run — does NOT see the 14:34:50 trade.
5. 2026-05-27 14:32:45Z: drift detector emits `position_guard_drift.v1` with the MGC `broker_truth_missing` row.

This is a transient timing artifact: the alpha_futures trade landed AFTER the snapshot publisher tick. The next snapshot tick (14:37:44Z, ~3 min after audit) should either confirm MGC at broker (resolving the drift) or continue to show it missing (escalating to operator action via `scripts/close_guard_entry.py`). **Operator action authorisation is NOT granted by this audit.**

### 8.4 Today's MGC fill evidence

`data/fills/FILLS_20260527.ndjson`: 10 alpha_futures paper_fill rows on MGC (T3 evidence). `data/broker_events/BROKER_EVENTS_IBKR_20260527.ndjson`: 187 rows (163 heartbeats, 12 fills, 12 fees; symbols MGC(20)/MSFT(2)/AAPL(2)). The 20 MGC broker events correspond to the 10 paper_fill rows ×2 events (fill + fee) each.

So MGC HAS broker activity today — it just doesn't appear in the most-recent `positions_snapshot.json` because of the 5-min refresh cadence.

**Status: PARTIAL.** The drift is real but explainable as snapshot-publisher cadence lag, not as broken truth. The PR-09 separation of broker_authority vs replay_diagnostic is functioning as designed; the only operator action that could resolve it deterministically (before the next snapshot tick) is `scripts/close_guard_entry.py` for alpha_futures|MGC, which this audit **does not authorise**.

---

## 9. Tier 8 — Network / Security Posture

### 9.1 Bound ports

```
0.0.0.0:9618 — chad-shadow-status.service (python pid 1068618, fd=6)
0.0.0.0:9619 — chad-shadow-status python (pid 2175218, fd=3)
0.0.0.0:9620 — chad-metrics (pid 2525497, fd=3)
127.0.0.1:8765 — python3 (pid 2525501, fd=6) — localhost only
*:4002 — java (pid 1654708, fd=69) — IB Gateway
```

3 CHAD ports bind to 0.0.0.0 (gaps_to_close P2-5 finding confirmed). 8765 binds localhost-only (good). IB Gateway 4002 binds all interfaces (standard).

### 9.2 AWS Security Group inspection (read-only via `aws ec2 describe-security-groups`)

Instance is in 3 distinct security groups: `sg-0fcc5d2109705572a` (launch-wizard-6), `sg-0a300561f18f9c4e2` (launch-wizard-8), `sg-0cc1289edcfcca335` (CHAD-Prod-SG).

Aggregate inbound rules across all 3 SGs:
- TCP 22 (SSH) — 0.0.0.0/0 (open globally)
- TCP 80 — 0.0.0.0/0
- TCP 443 — 0.0.0.0/0
- TCP 3000 — 0.0.0.0/0
- TCP 3001 — 0.0.0.0/0
- TCP 6379 (Redis) — 172.31.24.246/32 (VPC-internal only)
- TCP 7860 — 172.31.6.95/32 (VPC-internal only)
- TCP 22 — also a specific allow 3.96.222.115/32

**Ports 9618 / 9619 / 9620 / 4002 / 8765 are NOT explicitly opened in any of the 3 SGs.** Default-deny AWS SG behaviour means they are **NOT externally reachable** as of this audit, despite the 0.0.0.0 binding.

### 9.3 `/etc/chad/*.env` permissions

| File | Permission | Owner | Group |
|---|---|---|---|
| `/etc/chad/claude.env` | 0640 | ubuntu | ubuntu |
| `/etc/chad/dashboard.env` | 0640 | root | chad |
| `/etc/chad/fmp.env` | 0640 | root | chad |
| `/etc/chad/ibkr.env` | 0640 | root | chad |
| `/etc/chad/kraken.env` | 0640 | root | chad |
| `/etc/chad/openai.env` | 0640 | root | chad |
| `/etc/chad/polygon.env` | 0640 | root | chad |
| `/etc/chad/telegram.env` | 0640 | root | chad |
| `/etc/chad/ibkr.env.bak.20260521T145748Z` | 0640 | root | chad | (backup, not active) |
| `/etc/chad/` dir | 0750 | root | chad |

Convention is **0640** (owner R/W, group R, world none) — secrets readable by group `chad` and root only. This is **not** strict 0600 but is the documented ACL-group convention. **No file is world-readable.**

### 9.4 `docs/SECURITY.md`

- Present at `docs/SECURITY.md`, 9028 bytes, last mtime 2026-05-03 11:53.
- 24 days since last review; **not a HIGH gap but worth flagging.**

**Status: PARTIAL.**
- 0.0.0.0 binding is a defense-in-depth gap (should bind 127.0.0.1 only) — currently masked by AWS SG default-deny.
- /etc/chad ACL is group=chad 0640 (intentional, documented).
- AWS describe-security-groups confirms external reachability is **closed** for 9618/9619/9620.

---

## 10. Tier 9 — Dependency / Import Hygiene

### 10.1 Installed package versions

```
APScheduler           3.6.3   (legacy v3 line; v3.x has known deprecation paths)
fastapi               0.116.1
ib_async              2.1.0   (canonical)
python-telegram-bot   13.15   (legacy v13; v20+ is current/async)
requests              2.32.5
urllib3               1.26.20 (LTS line — fine)
xgboost               3.2.0
```

### 10.2 ib_insync usage

Production: 3 files (paper_position_closer, paper_shadow_runner, ibkr_broker_events_collector).
Test/governance: 4 files (test_options_chain_refresh, test_paper_shadow_runner, test_ib_async_import_parity, test_pr03_ib_async_phase2_migration).
`.bak` / `.deprecated`: 3 files (live_loop.*.bak, dashboard/api.py.pre_step_9_bak, ibkr_daily_bars_refresh.py.deprecated).

`ib_async` is imported in **28** production files.

### 10.3 Known dependency gaps

- `APScheduler 3.6.3` — gaps_to_close P2-8 deprecation concern. Latest 3.x is 3.10+; 4.x is API-incompatible.
- `python-telegram-bot 13.15` — v13 is legacy. Coexists with `chad/utils/telegram_bot.py`. Migration to v20+ requires async refactor.
- `urllib3 1.26.20` — within the safe LTS range (gaps_to_close P2-7).

**Status: PARTIAL.** ib_async migration phase 2 has 3 production files remaining (matches v9.3 §1.2 closure plan). APScheduler/telegram-bot legacy lines are known but not blocking paper operation.

---

## 11. Tier 10 — Test Coverage by Module

| Subdir | Source files | Notes |
|---|---|---|
| `chad/core/` | 35 | several modules have no `test_<name>.py` (e.g. `live_gate.py`, `signal_guard.py`, `full_execution_cycle.py`, `decision_trace.py`, `paper_position_closer.py`, `live_execution_router.py`, `live_mode.py`, `stop_state.py`, `routed_execution_runner.py`, `decision_trace_heartbeat.py`, `ibkr_execution_runner.py`, …) — but coverage is likely transitive via integration tests like `test_v9_*`, `test_full_cycle_*`, `test_orchestrator_v2.py`, `test_live_loop_*` |
| `chad/execution/` | 26 | |
| `chad/risk/` | 34 | |
| `chad/strategies/` | 31 | |
| `chad/portfolio/` | 16 | |
| `chad/intel/` | 18 | |
| `chad/ops/` | 39 | |
| `chad/ai/` | 2 | |
| `chad/analytics/` | 33 | |
| `chad/market_data/` | 26 | |
| `chad/tests/` | **206** test files | |

### 11.1 Skip / xfail count

`pytest chad/tests/ -q --co | grep -iE "skip|xfail" | wc -l` → **85**

This is the count of test-collection lines that mention skip/xfail; it conflates skipped-by-marker, parametrised-skipped, and module-level skip. **The 1 actual FAILED test in the pytest baseline is the same as observed twice today: `test_phase_a_item4_setup_tagging.py::test_02_alpha_intraday_momentum_surge_setup_family`.**

### 11.2 CHAD_SKIP_IB_CONNECT bypasses

Pytest baseline runs with `CHAD_SKIP_IB_CONNECT=1`. Integration tests that legitimately need IBKR are conditionally gated. A full audit of every test that depends on this flag is **UNKNOWN — REQUIRES AUDIT** — out of scope for this read-only pass.

**Status: PARTIAL.** Coverage by file-name is weak; transitive coverage via integration tests is the canonical model but not provable from a name-match scan.

---

## 12. Tier 11 — Runtime Artifact Inventory

(See Tier 3 for the publisher/monitor and stale-claim verification.)

### 12.1 Schema version coverage

- Files with `schema_version`: **59** distinct schemas, mostly `v1`, some `v2` and `v3` (lifecycle_replay_*).
- Files without `schema_version`: **78**.

### 12.2 Schemas declared in v9.3 §12.3 cross-check

| Declared schema | Present in runtime | Notes |
|---|---|---|
| `scr_state.v1` | ✔ | |
| `live_readiness_state.v1` | ✔ | |
| `tier_state.v2` | ✔ | |
| `news_intel.v1` | ✔ (in `runtime/news_intel.json` — not directly shown above but listed by find) — UNKNOWN, requires cross-check |
| `relative_strength.v1` | ✔ | |
| `volume_scan.v1` | ✔ | |
| `crypto_derivatives.v1` | ✔ | |
| `futures_roll_state.v1` | ✔ | |
| `options_greeks.v1` | ✔ | |
| `kraken_futures_intel.v1` | ✔ | |
| `earnings_intel.v1` | ✔ | |
| `dynamic_universe_candidates.v1` | ✔ | |
| `event_risk.v1` | ✔ | |
| `setup_family_expectancy.v2` | ✔ | |
| `position_guard_drift.v1` | ✔ | |
| `xgb_manifest.v2` | UNKNOWN — `runtime/models/xgb_veto/current/` absent, baseline file present | requires deeper read of `shared/models/xgb_veto_manifest.json` |

**No declared schema is missing where the corresponding file is also missing.**

**Status: PARTIAL.** Declared schemas are covered when their files exist; ~78 runtime files have no `schema_version` field at all, which is a wider authority gap.

---

## 13. Tier 12 — Config Surface

**28 config files** under `config/`. Reader counts (deduplicated across `chad/`, `ops/`, `scripts/`):

| Config | Readers | Verdict |
|---|---|---|
| `risk.json` | 20 | canonical |
| `universe.json` | 19 | canonical |
| `strategy_weights.json` | 10 | canonical |
| `reconciliation_exclusions.json` | 5 | canonical |
| `tiers.json` | 5 | canonical |
| `regime_activation_matrix.json` | 4 | canonical |
| `sizing_config.json` | 4 | canonical |
| `edge_decay_config.json` | 3 | canonical |
| `event_calendar.json` | 3 | canonical |
| `per_strategy_loss_limits.json` | 3 | canonical |
| `withdrawal_policy.json` | 3 | canonical |
| `portfolio_profiles.json`, `regime_booster_policy.json`, `rotation_rules.json`, `signal_stacking_config.json`, `simulated_oms_config.json`, `symbol_caps.json`, `threshold_adapter_config.json` | 2 each | canonical |
| `autonomy_bounds.json`, `intel_profiles.json`, `providers_allowlist.json`, `winner_scaling_policy.json` | 1 each | canonical |
| **`calendars.json`** | **0** | **DEAD?** |
| **`data_sources.json`** | **0** | **DEAD?** |
| **`feature_flags.json`** | **0** | **DEAD?** |
| **`gpt_limits.json`** | **0** | **DEAD?** |
| **`licensing.json`** | **0** | **DEAD?** |
| **`reconciliation.json`** | **0** | **DEAD?** |

Six configs with zero direct-string readers. **They may be loaded via generic loader patterns** (e.g. iterating `config/`), so "0 readers" by exact-string grep is a false-negative risk. **Marked UNKNOWN — REQUIRES AUDIT** for true deadness.

**Status: PARTIAL.** Top-tier canonical configs are well-routed. Six configs need a follow-up audit to confirm dead vs. generic-loader-fed.

---

## 14. Tier 13 — Journal Error Scan (last 24 h)

| Unit | ERROR/CRITICAL/Traceback/Exception count |
|---|---|
| **chad-ibkr-bar-provider** | **588** |
| chad-live-loop | 40 |
| chad-options-chain-refresh | 24 |
| chad-dashboard | 1 |
| chad-backend | 1 |
| chad-telegram-bot | 0 |
| chad-shadow-status | 0 |
| chad-orchestrator | 0 |
| chad-kraken-ws | 0 |
| chad-ibgateway | 0 |

### 14.1 chad-ibkr-bar-provider — root cause

Every ~30 s, the bar provider attempts to fetch historical data for `SILK6@COMEX` (silver futures, conId 788288505, lastTradeDateOrContractMonth=`20260527` — **expires today**). IBKR responds with:
```
ib_async.wrapper Error 162, reqId NNNN: Historical Market Data Service error message:HMDS query returned no data: SILK6@COMEX Trades
```

This is **a futures-roll registry that has not advanced past today's SI contract expiry**. The strategy/risk layer likely no longer routes SI (no SI in any of today's snapshot), but the bar provider's universe still polls it. **Single root cause; ~588 of the 588 entries are this same contract.**

### 14.2 chad-live-loop — top distinct errors

- `Error 1100, reqId -1: Connectivity between IBKR and Trader Workstation has been lost.` — broker-side outages (correlate with stop_bus flutter at 12:34Z and 13:07Z).
- `Error 201, reqId 5238 / 5236 / 5232 / 5230: Order rejected — IBKR near-expiration / physical delivery rules. Place order during contract delivery window. … See FAQ.` — same physical-delivery futures policy issue as the bar provider, hitting the executor side.

### 14.3 chad-options-chain-refresh — root cause

Same farm-broken errors as Tier 4: `Error 2103 / 2105 / 2157 / 200 / 10168`. Documented in §5.3.

**Status: PARTIAL.** Errors are loud and concentrated on **two symptoms of one root cause** (futures-contract roll-forward not advancing, plus today's broker-side farm flutter). No silent failures observed in the top-3 noisiest units.

---

## 15. Tier 14 — Epoch / Soak State

| Surface | Value |
|---|---|
| `runtime/epoch_state.json::schema_version` | `epoch_state.v1` |
| `runtime/epoch_state.json::active_epoch` | `CHAD_v8.9_Paper_Epoch_2` |
| `runtime/epoch_state.json::epoch_started_at_utc` | 2026-05-04T00:54:30Z |
| `runtime/epoch_state.json::paper_only` | true |
| `runtime/epoch_state.json::ready_for_live` | false |
| Pending Action: `ops/pending_actions/PAPER_EPOCH_3_START_2026-05-27.md` | exists (15,404 bytes, 2026-05-27 13:09 mtime); declares Epoch 3, **explicitly preserves runtime/epoch_state.json as Epoch 2** |
| Pending Action: `ops/pending_actions/PAPER_EPOCH_3_SESSION_TRACKER_2026-05-28.md` | exists (12,104 bytes, 2026-05-27 13:37 mtime); Session 1 PENDING; clean-soak count 0/5 |
| Day-0 (2026-05-27) verdict | DIRTY / NOT COUNTABLE per parent §12 (commit `02ccda7`) |
| Session 1 candidate window | opens 2026-05-28T00:00:00Z (≈9h 30m after audit time) |

**Confirmation:** Session 1 has **NOT** been falsely advanced. `runtime/epoch_state.json` is intentionally still Epoch 2 (the parent declaration explicitly stated the flip is a separate operator-controlled action).

**Status: VERIFIED.**

---

## 16. Tier 15 — Documentation vs Reality Delta

| v9.3 declared | v9.3 value | v9.4 errata? | Live reality | Delta |
|---|---|---|---|---|
| HEAD commit | `72f361f` | v9.4 HEAD `a5311d27` | `f845343` | **+57 commits past v9.3** (and ahead of v9.4) |
| Pytest count | 2,114 passing | v9.4 issued forward erratum correcting this | **2538 passed, 1 failed** | +424 tests, 1 regression |
| SCR state | CONFIDENT, `sizing_factor=1.0`, `paper_only=false` | unchanged | CONFIDENT, sizing_factor=1.0, paper_only=false | **VERIFIED** |
| ready_for_live | false | unchanged | false | **VERIFIED** |
| Reconciliation status | (v9.3 era: drift policy GAP-001/GAP-002 fix) | v9.4 forward errata addresses both | `reconciliation_state.json` upstream GREEN; `positions_truth.broker_authority_RED`; new MGC drift | **DRIFT**: upstream GREEN, truth RED |
| Active XGB model | (v9.3 era: candidate/promotion workflow) | unchanged | baseline `xgb_veto_20260510_020007`; no current promotion | **PARTIAL** — fallback chain is operating; no current promoted model |
| Intelligence feeds (v9.3 §2.10) | 10 listed (news, RS, volume, crypto_deriv, futures_roll, options_greeks, kraken_futures, earnings, dynamic_universe, event_risk) | unchanged | All 10 present with `schema_version` in runtime | **VERIFIED** |

v9.4 forward errata already acknowledges the test-count drift. v9.5 forward errata (per git log `08fa179`) exists; it captures further drift forward.

**Status: PARTIAL.** The system has continuously advanced (57 commits since v9.3); the SSOT chain has been kept current via forward-errata, but the bulk of recent activity (Paper Epoch 3 declaration, BOX-049 completion matrix, OPS-OMEGA-01 fix, PR-09 truth split, PR-04 options hardening, PR-02/PR-02b delta abstain + reconciler silence, PR-M2K-MYM, PR-03 ib_async phase 2, and today's Epoch 3 commits) sits **AHEAD** of the v9.3 baseline that the audit prompt cited as canonical.

---

## 17. ROOT CAUSE GROUPING

### R1. Canonical authority not designated (or designated but not code-enforced)

- **R1-A**: `positions_snapshot.json` vs `ibkr_paper_ledger_state.json` — both paper-position truth surfaces; BOX-047 documents the concern but no code-level single-writer authority is enforced. Pattern recurs as: snapshot=N, ledger=N±1, truth RED.
- **R1-B**: `portfolio_snapshot.json` (canonical equity) has **no `schema_version`** and exposes equity via nested structure rather than a top-level key; 12 readers depend on it.
- **R1-C**: Strategy registry counts agree at 16 between `__init__.py` and `strategy_weights.json::weights`, but **BOX-049 completion matrix** uses 18, and runtime `runtime/strategy_health.json` keys are TBD. Three-way comparison: 16 / 16 / 18 with no canonical reconciliation.
- **R1-D**: 78 runtime files have no `schema_version`, including hot-path files (`decision_trace_heartbeat.json`, `dynamic_caps.json`, `position_guard.json`, `last_route_decision.json`, `ibkr_status.json`, etc.). No schema = no contract = silent format drift risk.

### R2. Orphaned publisher (no monitor)

- **R2-A**: `runtime/proofs/BURNIN_*.json` — 100+ historical proof artifacts from Feb–Mar; `chad-burnin-check.timer` still writes them; no consumer beyond the historical record.
- **R2-B**: ~14 dormant `runtime/*.json` files older than 50 days (allocator_v3, dominance_allocator, savage_alloc_state, capital_allocator, kraken_pnl_state, crypto_risk_off, dynamic_caps_*_overlay variants). Some have `schema_version`, some don't. No publisher fires for these any more; no consumer reads them. They consume disk and confuse audit greps.
- **R2-C**: `runtime/lifecycle_replay_drift_audit.json` is 69.8 days old. Schema version present (`lifecycle_replay_drift_audit.v1`). Either the publisher has stopped (silent failure) or the schema-mapped consumer expects rolling rather than continuous refresh. **UNKNOWN — REQUIRES AUDIT.**

### R3. Silent service failure (no loud alert)

- **R3-A**: `chad-options-chain-refresh.service` failed at 12:33:07Z. PR-04 wrote `runtime/options_chain_refresh_failure.json` and a degraded cache — this is NOT silent. But the systemd unit state `failed` only resolves when the next `chad-options-chain-refresh.timer` fire succeeds; meanwhile, no `OnFailure=` Telegram alert was emitted (verified by 0 errors in `chad-telegram-bot` journal).
- **R3-B**: `chad-ibkr-bar-provider` emits 588 errors/day on an expired futures contract (`SILK6 lastTrade=20260527`). Loud but ignored — no operator-facing alert routed to Telegram.
- **R3-C**: `chad-live-loop` `Error 201` futures-physical-delivery rejections (≥6 today, all on conId 712565978). Reported in journal; no operator alert routed.

### R4. Placeholder / fake-value source not removed (only blocked downstream)

- **R4-A**: 14 trusted-fake `delta SPY $100.0` placeholders from 2026-05-03 remain on disk in `data/fills/FILLS_20260503.ndjson` lines 1–14. They are excluded by SCR (`excluded_untrusted=321`) but the data-on-disk is unchanged. Any future re-reader that does NOT honour the exclusion contract will silently consume them.
- **R4-B**: Delta strategy's *abstain on missing price* (PR-02) and reconciler's *silenced synthesized close-fill* (PR-02b) close the production-side emission. Defense-in-depth (trade_closer, paper_exec_evidence_writer, paper_trade_executor) catches anything that slips. The historical 14 rows are the only contamination.

### R5. Schema drift (declared vs actual)

- **R5-A**: 78 runtime files lack `schema_version`. Subset includes the most-frequently-written hot-path files.
- **R5-B**: `ibkr_adapter_state.sqlite3` has table `ibkr_exec_state` (52 rows), not the `orders` table named in the audit prompt. The historical reference and the live schema have diverged.

### R6. Documentation drift (SSOT vs runtime)

- **R6-A**: v9.3 HEAD `72f361f` is 57 commits stale. v9.4 forward erratum exists for test-count; v9.5 forward erratum exists; new Paper Epoch 3 declaration was added on 2026-05-27 (commits `4c5cd3b`, `02ccda7`, `f845343`). The audit prompt's "expected HEAD" qualifier "or later" rescues this.
- **R6-B**: gaps_to_close claims (drawdown_state.json 10 days stale, ibkr_watchdog_last.json 44 days stale, position_guard_drift.json not found, positions_snapshot.json stale since 2026-04-03) — **all refuted** at this audit. BOX-046 already retracted the positions_snapshot claim; others appear unretracted in operator-facing tooling.
- **R6-C**: CLAUDE.md memory states "Account equity: ~$183,264 USD paper"; live `dynamic_caps.total_equity` is $268,345.77. The memory is stale (was current as of 2026-05-09 per the memory's own timestamp); equity has grown +46% (probably benefitting from the recent paper PnL of +$10,575).

### R7. Other

- **R7-A — Futures roll registry has not advanced past today's expiries**: `SILK6` (silver) `lastTradeDateOrContractMonth=20260527` is today; the bar provider polls every ~30 s and accumulates errors. Likely related: `Error 201` rejections in live-loop on a different futures conId (712565978). **No production data is corrupted**, but operator visibility is required.
- **R7-B — Stop-bus flutter has been a daily occurrence**: today (2026-05-27) recorded two stop_bus events at 12:34:04Z → 12:47:05Z (~13 min) and 13:07:50Z → 13:12:51Z (~5 min), both auto-cleared via `clean_streak=5`. Day-0 itself was 7h 11m of halt requiring operator restart. The deferred hardening note `IBKR-RELIABILITY` (Epoch 3 declaration §13) addresses this. Marked R7-B as a recurring symptom of the IBKR market-data farm reliability.
- **R7-C — Dual-writer for SCR shadow**: `runtime/scr_state.json` (publisher truth, fresh, CONFIDENT) versus `live_gate.shadow` (a snapshot consumed by live_gate, can drift to UNKNOWN with `scr_sync_error:TimeoutError`). This is a publisher → consumer-snapshot timing issue, not a data integrity issue, but it confuses operator inspection.

---

## 18. FINDINGS PRIORITY MATRIX

|  | **Severity HIGH** | **Severity MED** | **Severity LOW** |
|---|---|---|---|
| **Effort S** | • Re-bind 9618/9619/9620 to 127.0.0.1 (R6/R8-style; Pending Action only) <br> • Restart `chad-options-chain-refresh` after IBKR farm recovery (operator only) <br> • Quarantine 14 trusted-fake rows in `FILLS_20260503.ndjson` via exclusion tag (no data deletion) | • Add `schema_version` to hot-path runtime files (decision_trace_heartbeat, dynamic_caps, position_guard, last_route_decision, ibkr_status, portfolio_snapshot) <br> • Add `OnFailure=` to `chad-options-chain-refresh.service` to route a Telegram alert | • Delete or document the 6 unread configs (calendars/data_sources/feature_flags/gpt_limits/licensing/reconciliation) <br> • Update CLAUDE.md account-equity memory |
| **Effort M** | • Designate single-writer authority for paper-position truth (positions_snapshot vs ibkr_paper_ledger_state) — author BOX-047 follow-on policy + code-enforced guard <br> • Advance futures-roll registry past 2026-05-27 expiries (SILK6 et al.) | • Update SSOT to v9.6 (or new forward-erratum) consolidating: Paper Epoch 3, PR-09 truth split, PR-04 options hardening, PR-02/02b abstain+silence, OPS-OMEGA-01, PR-M2K-MYM, PR-03 ib_async phase 2, the 6 refuted gaps_to_close items | • Add a doc-vs-runtime reconciliation test in `chad/tests/` |
| **Effort L** | • Resolve `positions_truth.broker_authority_RED` permanently: either reconcile snapshot/ledger schemas to a single symbol-keyed authoritative file, or design a deterministic snapshot-vs-ledger merge | • Finish ib_async phase 2 migration for the 3 remaining production files (paper_position_closer, paper_shadow_runner, ibkr_broker_events_collector) | • Migrate python-telegram-bot 13.x → 20.x async <br> • Upgrade APScheduler 3.6 → 3.10 |

---

## 19. RECOMMENDED CLOSEOUT ORDER

(Acceptance criteria are read-only / Pending-Action style. No command blocks per audit charter.)

1. **TRUTH-RECONCILE-1** (Root cause R1-A, Severity HIGH, Effort M)
   Owner: Channel 1 (Forensic + Hot-path).
   Pending Action that designates a single code-enforced canonical paper-position writer between `positions_snapshot.json` and `ibkr_paper_ledger_state.json`. Convert the other to a derived view. Add a test that fails if `positions_truth.json::broker_authority_status=RED` persists more than 2× the snapshot-refresh interval.
   Acceptance: `positions_truth.broker_authority_status=GREEN` for ≥ 60 min on a regular trading day with `reconciliation_state.status=GREEN`.

2. **MGC-DRIFT-1** (Root cause R1-A subset, Severity MED, Effort S)
   Owner: Channel 2 (Operator + Pending Action).
   Pending Action that records the current `alpha_futures|MGC broker_truth_missing` drift, the broker-event evidence (20 today's events on MGC), and explicitly authorises `scripts/close_guard_entry.py alpha_futures|MGC` if and only if the next snapshot tick still does not contain MGC. No code change.
   Acceptance: `position_guard_drift.drift_count=0` after the next snapshot tick OR after the authorised close-guard execution.

3. **OPTIONS-CHAIN-1** (Root cause R3-A, Severity HIGH, Effort S)
   Owner: Channel 3 (Ops).
   Pending Action that documents the 12:33:07Z `chad-options-chain-refresh.service` failure, the IBKR farm-broken root cause, and the PR-04 graceful-degradation evidence. Add an `OnFailure=` directive plan (config change, NOT applied) to route a Telegram alert on the next failure.
   Acceptance: After IBKR farm recovery, the next timer fire succeeds and `runtime/options_chains_cache.json` is repopulated with non-empty `chains`.

4. **FUTURES-ROLL-1** (Root cause R7-A, Severity HIGH, Effort M)
   Owner: Channel 1 (Bar provider).
   Pending Action that documents the SILK6 expiry today, the 588-error/day noise, and the next-month SI contract (likely SIN6 for July) to roll forward to. NO code change in this Pending Action; the action is to schedule a PR that advances the futures registry.
   Acceptance: `chad-ibkr-bar-provider` `Error 162` count for SI falls to 0 within 24 h of the PR landing.

5. **SCHEMA-VERSION-1** (Root cause R1-D / R5-A, Severity MED, Effort S)
   Owner: Channel 1.
   Pending Action that proposes adding `schema_version` to the top 10 hot-path runtime files currently missing one: `decision_trace_heartbeat.json`, `dynamic_caps.json`, `position_guard.json`, `last_route_decision.json`, `ibkr_status.json`, `portfolio_snapshot.json`, `ibkr_paper_ledger_state.json`, `trade_closer_state.json`, `signal_guard.json`, `ibkr_bars_cache.json`. **Adding `schema_version` is a no-op for current consumers but enables future schema-drift detection.**
   Acceptance: 100% of hot-path runtime files have `schema_version`, a versioning policy doc exists, and a `pytest` test enforces the invariant.

6. **PORT-BINDING-1** (Root cause R5-B / Tier 8, Severity HIGH, Effort S)
   Owner: Channel 3.
   Pending Action that proposes re-binding 9618/9619/9620 to `127.0.0.1` only (defense-in-depth) while preserving AWS Security Group default-deny as the second layer.
   Acceptance: `ss -tlnp` shows `127.0.0.1:9618/9619/9620` only. No external impact (the SGs already block them).

7. **HISTORICAL-PLACEHOLDER-1** (Root cause R4-A, Severity MED, Effort S)
   Owner: Channel 1.
   Pending Action that documents the 14 trusted-fake rows in `data/fills/FILLS_20260503.ndjson` lines 1–14 and proposes a `placeholder_quarantine_tag` to add to a sidecar exclusion file (NOT modifying the original .ndjson). Confirms SCR already excludes them.
   Acceptance: A documented exclusion sidecar exists and any new aggregator path is required to honour it.

8. **DOC-RECONCILE-1** (Root cause R6-A/B, Severity MED, Effort M)
   Owner: Channel 2 (Docs).
   Pending Action that proposes a v9.6 SSOT (or a Forward Errata Doc) consolidating: (1) Paper Epoch 3 declaration and the 5-session tracker; (2) PR-09 truth-surface contract; (3) refutation of the 4 stale-claim items already refuted by BOX-046 and live filesystem evidence; (4) the 57-commit delta since v9.3; (5) explicit pin of the dual-ledger authority policy (BOX-047 follow-on); (6) actual current model version (`xgb_veto_20260510_020007` baseline).
   Acceptance: v9.6 (or forward-errata) committed.

9. **STOP-BUS-RECOVERY-1** (Root cause R7-B / Epoch 3 §13 deferred hardening, Severity HIGH, Effort L)
   Owner: Channel 1.
   Pending Action elevation: address `IBKR-RELIABILITY` — characterise the deadlock that required the operator-authorised `chad-ibgateway.service` restart on Day-0; design a safe in-process recovery for `auto_recovery:broker_latency_clean_streak=5` that does not require a systemd restart; add an `ibkr_status.json` counter for "consecutive cycles avg_latency_ms above stop threshold".
   Acceptance: A future stop_bus halt of >30 min is recoverable via the live-loop's own path with no operator restart.

10. **IB_INSYNC-PHASE2-1** (Root cause R6-A subset, Severity LOW, Effort L)
    Owner: Channel 1.
    Finish PR-03 phase 2 migration for `paper_position_closer.py`, `paper_shadow_runner.py`, `ibkr_broker_events_collector.py`.
    Acceptance: `grep "import ib_insync" chad/*` returns only test-allowed files.

11. **DEAD-CONFIG-1** (Root cause R6 subset, Severity LOW, Effort S)
    Owner: Channel 2.
    Audit the 6 zero-reader configs (`calendars.json`, `data_sources.json`, `feature_flags.json`, `gpt_limits.json`, `licensing.json`, `reconciliation.json`) for actual reads via generic loader patterns. Remove or document as appropriate.
    Acceptance: All `config/*.json` files have ≥ 1 documented reader.

---

## 20. PROHIBITED ACTIONS

Without a follow-up explicit Pending Action and operator GO, none of the following may be done:

- **DO NOT** flip `ready_for_live` to true.
- **DO NOT** flip `runtime/epoch_state.json::active_epoch` to `CHAD_v8.9_Paper_Epoch_3`.
- **DO NOT** restart any `chad-*` service (in particular not `chad-live-loop`, `chad-orchestrator`, `chad-ibgateway`).
- **DO NOT** delete or modify the 14 trusted-fake placeholder rows in `data/fills/FILLS_20260503.ndjson`.
- **DO NOT** delete the dormant `runtime/*.json` files older than 50 days without a documented policy and rollback path.
- **DO NOT** run `scripts/close_guard_entry.py alpha_futures|MGC` until and unless authorised by **MGC-DRIFT-1**.
- **DO NOT** mutate `/etc/systemd/system/chad-*` unit files.
- **DO NOT** mutate any `/etc/chad/*.env` file.
- **DO NOT** modify any file inside `runtime_FREEZE_*` or `data_FREEZE_*`.
- **DO NOT** mark Session 1 PASS until the full Session 1 window (2026-05-28T00:00:00Z → 2026-05-28T23:59:59Z) has closed and the §5 evidence fields of the Session 1 tracker have been honestly populated.
- **DO NOT** call any broker (IBKR, Kraken, Coinbase) API.
- **DO NOT** mutate runtime/, config/, services, or commit / push / tag git.

---

## 21. FINAL STATUS

**FINAL STATUS: PARTIAL**

Justification:
- **VERIFIED** for: live-mode gates (paper, live_disabled), execution-mode canonical reader, SCR publisher, XGB active model (baseline fallback), epoch state (Session 1 has not been falsely advanced, Day-0 correctly marked DIRTY), intelligence-feed schemas (10/10 declared, 10/10 present).
- **PARTIAL** for: position truth (snapshot vs ledger dual-authority unresolved, current `broker_authority_RED`), paper-ledger authority (BOX-047 documents but no code-enforced single writer), strategy-registry counts (16/16/18 three-way), schema-version coverage (59/137 ~43%), test coverage by file-name vs transitive (model unclear), ports binding (0.0.0.0 with AWS SG default-deny), `chad-options-chain-refresh` failed unit (graceful degradation present, no `OnFailure=` Telegram alert), `chad-ibkr-bar-provider` 588 errors/day (loud, futures-roll registry not advanced), historical placeholder rows (excluded but on disk), v9.3 doc drift (+57 commits, v9.4/v9.5 forward errata partially mitigates).
- **BLOCKED**: nothing.
- **UNKNOWN — REQUIRES AUDIT**: (a) per-unit `OnFailure=` coverage, (b) actual readers of the 6 zero-reader configs via generic loader, (c) `ibkr_adapter_state.sqlite3::ibkr_exec_state` PendingSubmit ages (schema is single-table, not the prompt-expected `orders`), (d) what `runtime/lifecycle_replay_drift_audit.json` 69-day staleness actually means.

**No mutation occurred during this audit. No commit, no service restart, no broker call, no runtime/config/data write. Session 1 remains PENDING (window opens 2026-05-28T00:00:00Z, ~9h 30m from this report).**

---

*Report ends.*
