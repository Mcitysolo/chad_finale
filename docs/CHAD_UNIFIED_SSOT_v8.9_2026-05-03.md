# CHAD Unified SSOT v8.9

**Version:** 8.9
**Date:** 2026-05-03
**Status:** Active — Paper Trading / Audit Remediation Locked / SCR PAUSED
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v8.8_2026-05-02.md` (commit `fd96133`, 2026-05-02)
**Repository HEAD:** `02280094d85ebac3643b41967ec0542c87f4b339` (`0228009`)
**Branch:** `main`
**Test status:** 1064 passed / 0 failed (verified `CHAD_SKIP_IB_CONNECT=1 pytest`, 2026-05-03)
**Full-cycle preview:** PASS — *logical preview only, no broker calls*
**Live readiness:** `runtime/live_readiness.json:ready_for_live=false`
**SCR posture:** `PAUSED`, `sizing_factor=0.0`

---

## v8.9 signature: SSOT External Audit Remediation Lock

An external forensic SSOT audit identified critical production-readiness
gaps in v8.8. v8.9 documents the **remediation lock**: critical audit
blockers are either fixed in code/tests or explicitly documented as
accepted residual risks with mitigation/deferred owner. **This is an
audit-remediation lock, not a live-trading approval.**

The system passed code/test validation (1064 passed / 0 failed) and the
full-cycle preview produced a clean logical pass with no broker calls.
Reconciliation is GREEN. The stop bus is clear. The flip executor's
broker-confirmed close-first/open-second invariant is enforced and
covered by 6 dedicated tests. The signal throttle now persists across
restarts. The choppy regime detector is asset-class scoped. The
per-symbol daily loss limit accounts for unrealized P&L. The two
trading engine units are NEVER_AUTO_RESTART. Model versioning is
proven end-to-end.

**Operational trading remains disabled by SCR risk controls.** SCR is
PAUSED with `sizing_factor=0.0` because measured paper performance has
breached the cautious thresholds (`sharpe_like=-1.287 < 0.500`,
`max_drawdown=-10314.12 < -10000.00`). This is the risk control
working correctly — it is *not* a code failure, and v8.9 does **not**
override it.

> If this document and the code disagree, either the code drifted or
> this revision needs another pass — either way, revise the document
> before relying on the disagreement.

---

## Current Runtime Truth — 2026-05-03

```
Repo state                     : clean at pre-SSOT lock (git status empty)
Repository HEAD                : 02280094d85ebac3643b41967ec0542c87f4b339
Branch                         : main
Tests                          : 1064 passed / 0 failed (CHAD_SKIP_IB_CONNECT=1)
Compile                        : OK for flip_executor.py and live_loop.py
Full-cycle preview             : PASS, logical preview only, no broker calls
Reconciliation                 : GREEN (broker=ibkr:clientId=83, chad_open=14, broker=14, worst_diff=0.0)
SCR state                      : PAUSED
SCR sizing_factor              : 0.0
SCR effective_trades           : 321
SCR win_rate                   : 0.6978
SCR sharpe_like                : -1.287   (< 0.500 cautious gate, < 0.100 cautious-allow gate)
SCR max_drawdown               : -10314.12 (< -10000.00 cautious gate)
SCR total_pnl                  : -20855.54 over 5000 paper_trades, 0 live_trades
Stop bus                       : false (cleared 2026-04-22 by smoke_test)
Profit lock                    : NORMAL  (sizing_factor=1.0; daily_loss_limit_hit=true; stop_new_entries=true)
Regime                         : trending_bull (confidence 0.795)  with choppy_overlay ACTIVE (score 0.55, sizing_multiplier 0.25)
Choppy regime                  : ACTIVE since 2026-05-02T13:06:20Z, proxy_symbol=SPY, consecutive_choppy_reads=285
Live readiness                 : false (requirements_remaining: operator_GO, 14_day_equity_history, 60_day_paper_performance)
Failed chad services           : none
Account equity (paper)         : ibkr=$177,761.48 + kraken=$184.58 ≈ $177,946.06 USD
Operational conclusion         : audit remediation is locked, but trading remains disabled by SCR risk controls.
```

---

## Table of Contents

0. [Preamble](#0-preamble)
1. [Mission & Architecture](#1-mission--architecture)
2. [Runtime State](#2-runtime-state)
3. [Strategies](#3-strategies)
4. [Execution Pipeline](#4-execution-pipeline)
5. [Risk & Governance](#5-risk--governance)
6. [Business Framework](#6-business-framework)
7. [Reconciliation](#7-reconciliation)
8. [Intelligence Layer](#8-intelligence-layer)
9. [Telegram Operator Interface](#9-telegram-operator-interface)
10. [Dashboard](#10-dashboard)
11. [Services & Timers](#11-services--timers)
12. [Data & Storage](#12-data--storage)
13. [Change Log](#13-change-log)
14. [Known Issues / Accepted Residual Risks](#14-known-issues--accepted-residual-risks)
15. [Phase Roadmap](#15-phase-roadmap)
16. [Appendices](#16-appendices)

---

## 0. PREAMBLE

### Document metadata

| Field | Value |
|---|---|
| Document version | 8.9 |
| Date written | 2026-05-03 (UTC) |
| Predecessor | v8.8 — `docs/CHAD_UNIFIED_SSOT_v8.8_2026-05-02.md` (`fd96133`) |
| Repository HEAD at write time | `0228009` — *"SS08: model versioning artifacts — retrained xgb_veto_model + manifest (7672 samples, accuracy=54.98%)"* |
| Branch | `main` |
| Audit cycle | 2026-05-03 SSOT external forensic audit |
| Lock type | Audit remediation lock (NOT live promotion) |

### Why a v8.9

v8.8 was the post-issue-closure SSOT — every numbered ISSUE was closed
and the test suite hit zero failures. While v8.8 was being lived in,
an external forensic audit graded the SSOT against production-readiness
criteria and produced a finding tree:

- **CB-class** (Critical Blockers): code or schema invariants that
  must be fixed before any live exposure. CB01–CB08.
- **CF-class** (Critical Findings): correctness/observability gaps
  with material safety implications. CF01, CF03.
- **DS-class** (Data Schema): canonical-form / migration concerns.
  DS03, DS05, DS06, DS07, DS08.
- **BG-class** (Bugs): functional defects with non-trivial blast
  radius. BG04, BG06, BG10, BG11, BG13.
- **SS-class** (Security/Safety): operational safety controls.
  SS01, SS02, SS03, SS05, SS08.
- **OP-class** (Ops): startup/dependency wiring. OP01.

This document — v8.9 — is the **lock state** showing what was fixed
and what is documented as an accepted residual risk with an owner and
a mitigation plan. It is *not* a live-readiness statement.

### Server / repo / mode (verified)

- **Server:** AWS EC2, Ubuntu 24.04, kernel `6.17.0-1009-aws`.
- **Repo root:** `/home/ubuntu/chad_finale`.
- **Python:** `python3` via `/home/ubuntu/chad_finale/venv` (governance
  rule in `CLAUDE.md` — never invoke `python`).
- **Execution posture:** PAPER — `CHAD_EXECUTION_MODE=paper`, single
  canonical reader at
  `chad/execution/execution_config.get_execution_mode()`.
- **IBKR paper account:** Canadian-domiciled, `$177,761.48 USD` net
  liquidation (source: `runtime/portfolio_snapshot.json:ibkr_equity`
  at `2026-05-03T12:45:07Z` after IBKR FX-quoted CAD→USD conversion).
- **Kraken paper:** connected — 0.0012 BTC + 252.85 CAD ≈ `$184.58 USD`
  (source: `runtime/portfolio_snapshot.json:kraken_equity`).
- **Total equity (USD):** `$177,946.06`.
- **Live readiness:** `runtime/live_readiness.json:ready_for_live=false`,
  evaluated `2026-05-03T01:58:28Z`, requirements remaining:
  `operator_GO`, `14_day_equity_history`, `60_day_paper_performance`.
- **SCR posture:** `PAUSED`, `sizing_factor=0.0`. Reasons:
  `sharpe_like=-1.287 < 0.500`, `max_drawdown=-10314.12 < -10000.00`.
  Stat snapshot: `total_trades=5000, effective_trades=321,
  win_rate=0.698, total_pnl=-20855.54`.
- **Profit lock:** `mode=NORMAL`, `sizing_factor=1.0`, but
  `daily_loss_limit_hit=true` (`-$14,880.00` realized today vs
  `-$5,338.38` 3% limit) → `stop_new_entries=true`.
- **Reconciliation:** `GREEN`, `chad_open=14`, `broker_positions=14`,
  `worst_diff=0.0`, excluded symbols `AAPL`, `NVDA` under bounded
  exclusion policy (DS08).

### Commit chain since v8.8 (`fd96133..HEAD`)

In topological order (oldest → newest):

```
36b94f6  Fix: CB02 exit-exempt throttle, CB03 persist throttle state, CF03 TTL 360s, CB06 recon blocks all fresh, BG10 halt writes, BG04 choppy+macro watched
71c0d11  Fix: CB07 unrealized PnL loss limit, CB08 version field position guard, BG11 FLIP safety, BG13 asset-class choppy scoping, BG06 live_readiness, SS01 NEVER_AUTO_RESTART, SS05 throttle audit, SS08 model versioning
0060ce4  Fix: CF01 contributors field, DS05 next_event schema, DS07 canonical net_pnl, DS08 exclusion policy, CB05 fail-conservative stale, OP01 startup deps, SS02/SS03 security doc
b424748  Audit remediation Batch 4: BG11 flip executor close-first, SS02 security doc, Kraken test isolation — 1064 passed 0 failed
df3b519  Housekeeping: untrack runtime/pnl_state.json, gitignore *.bak and *.pre_* backup files
0228009  SS08: model versioning artifacts — retrained xgb_veto_model + manifest (7672 samples, accuracy=54.98%)
```

Six functional commits across four remediation batches plus one
housekeeping and one model-artifact commit. All on 2026-05-03 and
preceded by `fd96133` (v8.8 docs landing on 2026-05-02).

### Governance rules (unchanged from CLAUDE.md)

1. One change at a time; baseline → change → verify → proceed.
2. No full rewrites; surgical changes only.
3. No direct config mutation. Risk caps, live mode, strategy config
   prepared as Pending Actions only.
4. Verification sequence after every code change.
5. Never modify `runtime_FREEZE_*` or `data_FREEZE_*`.
6. Never modify systemd service files without explicit instruction.
7. Never restart live services without explicit instruction.
8. Commit and tag git after each completed P0/P1/P2 item.

v8.9 was produced under these rules; nothing in this document
authorizes a live mode flip, a config mutation, or a service restart.

---

## 1. MISSION & ARCHITECTURE

### Mission

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a multi-strategy
algorithmic trading engine that runs in paper today and is being
hardened toward live capital deployment. The objective is consistent
risk-adjusted compounding — not headline returns — under explicit
governance with operator-in-the-loop authorization and a defensive
Stack of Risk Controls (SCR) that pauses trading when measured
performance breaches thresholds.

### Architecture (no structural change in v8.9)

The architecture inherited from v8.8 stands:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CHAD Live Loop                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. Strategy signal generation (16 strategies)                  │  │
│  │ 2. Signal router → contributors metadata (CF01)                │  │
│  │ 3. Net Exposure Conflict Gate (per-symbol loss limit CB07)     │  │
│  │ 4. Smart Strategy Throttle Gate (persistent state CB03)        │  │
│  │ 5. Signal throttle (R10 churn cap; persistent)                 │  │
│  │ 6. ML Veto (shadow only; CHAD_ML_VETO_ENABLED off)             │  │
│  │ 7. Choppy regime overlay (asset-class scoped BG13)             │  │
│  │ 8. Risk allocation (Profit Lock, SCR, Choppy multiplier)       │  │
│  │ 9. Execution plan build                                        │  │
│  │10. FLIP Executor close-first → broker confirm → open (BG11)    │  │
│  │11. IBKR / Kraken adapter                                       │  │
│  │12. Paper exec evidence writer (gross/commission/slippage/net)  │  │
│  │13. Position guard mutation on broker confirm (CB08 versioned)  │  │
│  │14. Reconciliation (GREEN gate for fresh entries CB06)          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

The new v8.9 modules and surgical edits are layered into the existing
pipeline; no module was introduced that displaces an existing
component. The flip executor (`chad/core/flip_executor.py`) is the
only NEW module in v8.9.

### Component map (additions / changes from v8.8)

| Component | Path | v8.9 change |
|---|---|---|
| Flip executor | `chad/core/flip_executor.py` | NEW (BG11) |
| Signal router | `chad/utils/signal_router.py` | `meta["contributors"]` field (CF01) |
| Net exposure gate | `chad/execution/net_exposure_gate.py` | unrealized PnL in per-symbol loss (CB07); recon blocks ALL fresh (CB06) |
| Strategy throttle gate | `chad/execution/strategy_throttle_gate.py` | persistent state file (CB03) |
| Live loop | `chad/core/live_loop.py` | exit-exempt throttle (CB02); confidence DESC trim (CB02) |
| Position guard | `chad/core/position_guard.py` | top-level `_version`, per-entry `_entry_version` (CB08/DS03) |
| Regime classifier / detector | `chad/analytics/regime_classifier.py`, `chad/analytics/choppy_regime_detector.py` | TTL 360s aligned (CF03/CB04); choppy proxy by asset class (BG13) |
| Macro state publisher | `ops/macro_state_publish.py` | normal + fail-closed schema alignment (DS06) |
| Event risk publisher | `ops/event_risk_publish.py` | `next_event` structured object (DS05) |
| Reconciliation publisher | `chad/ops/reconciliation_publisher.py` | bounded exclusion policy (DS08) |
| Health monitor | `chad/ops/health_monitor.py`, `health_monitor_rules.py` | NEVER_AUTO_RESTART for live engines (SS01); `choppy_regime_state.json` and `macro_state.json` watched (BG04); `regime_state.json` NOTIFY_ONLY (SS01) |
| Live-loop systemd | `/etc/systemd/system/chad-live-loop.service` | `PartOf=`/`Requires=` removed; `ExecStartPre` 60s probe (OP01) |
| Live readiness publisher | `chad/ops/live_readiness_publisher.py` | cadence/requirements fields (BG06) |
| Halt writer | `chad/risk/edge_decay_monitor.py` | writes `strategy_allocations.json` via `set_strategy_halted()` (BG10) |
| Trade record schema | `chad/execution/paper_exec_evidence_writer.py` | `net_pnl` canonical; `pnl` deprecated alias retained (DS07) |
| ML veto manifest | `shared/models/xgb_veto_manifest.json` | NEW model metadata + timestamped backups (SS08) |
| Signal throttle audit | `runtime/signal_throttle_audit.json` | NEW state-change audit trail (SS05) |
| Security doc | `docs/SECURITY.md` | NEW (SS02/SS03) — 246 lines |

---

## 2. RUNTIME STATE

### Live snapshot (verified 2026-05-03)

#### `reconciliation_state.json`

```
status                : GREEN
broker_source         : ibkr:clientId=83
chad_state_source     : position_guard.json
counts                : chad_open=14, chad_strategy_open=15, broker_positions=14
worst_diff            : 0.0
mismatches            : []
drifts                : []
excluded_symbols      : ["AAPL", "NVDA"]
exclusion_policy      : bounded {reason, owner, added_utc, expires_utc, reviewed_utc}
ts_utc                : 2026-05-03T12:45:04Z
ttl_seconds           : 360
```

Exclusion policy entries: AAPL, MSFT, NVDA — all `reason="pre-existing
broker position"`, `owner="operator"`, `added_utc=2026-04-01`,
`reviewed_utc=2026-05-03`, `expires_utc=null`. The bounded policy
(DS08) means each exclusion now carries a review timestamp; the
operator must keep `reviewed_utc` current.

#### `scr_state.json`

```
state                 : PAUSED
sizing_factor         : 0.0
paper_only            : true
schema_version        : scr_state.v1
source.url            : http://127.0.0.1:9618/shadow
ts_utc                : 2026-05-03T12:48:35Z
ttl_seconds           : 180

stats:
  total_trades        : 5000
  effective_trades    : 321
  excluded_untrusted  : 4241
  excluded_manual     : 0
  excluded_nonfinite  : 0
  paper_trades        : 5000
  live_trades         : 0
  win_rate            : 0.6978
  sharpe_like         : -1.2872
  max_drawdown        : -10314.12
  total_pnl           : -20855.54

reasons (verbatim):
  CONFIDENT blocked: sharpe_like=-1.287 < 0.500
  CONFIDENT blocked: max_drawdown=-10314.12 < -10000.00
  CAUTIOUS  blocked by sharpe_like=-1.287 < 0.100
  PAUSED:   performance below cautious thresholds — live trading should be disabled.
  Summary:  total_trades=5000, effective_trades=321, win_rate=0.698, sharpe_like=-1.287, max_drawdown=-10314.12, total_pnl=-20855.54
```

This is the operational blocker: SCR has measured paper drawdown past
the cautious gate and is correctly refusing live promotion. v8.9 does
not override SCR.

#### `live_readiness.json`

```
ready_for_live              : false
schema_version              : live_readiness_state.v1
ts_utc                      : 2026-05-03T01:58:28Z
last_evaluated_utc          : 2026-05-03T01:58:28Z
next_evaluation_cadence     : weekly
evaluated_by                : operator_session_2026-05-03
requirements_remaining      : ["operator_GO", "14_day_equity_history", "60_day_paper_performance"]
latest_report_path          : reports/live_readiness/LIVE_READINESS_20260409T134029Z.json
latest_report_sha256        : sha256:715fd4d2…
```

The cadence/requirements fields (BG06) are present. `ready_for_live`
remains `false`; three independent gates must clear before that flips.

#### `stop_bus.json`

```
active           : false
cleared_at       : 2026-04-22T01:56:50Z
cleared_by       : smoke_test
schema_version   : stop_bus.v1
```

#### `profit_lock_state.json`

```
mode                       : NORMAL
sizing_factor              : 1.0
profit_lock_active         : false
daily_loss_limit_hit       : true
daily_loss_today           : -14880.00
daily_loss_limit_dollars   : -5338.38   (3% of equity 177,946.06)
daily_loss_limit_pct       : 3.0
stop_new_entries           : true
explain                    : "profit lock inactive | daily_loss_limit_hit"
inputs.account_equity      : 177946.06
inputs.equity_source       : runtime/dynamic_caps.json
inputs.realized_pnl        : -14880.00
inputs.trade_count         : 16
ts_utc                     : 2026-05-03T12:48:35Z
```

The profit lock mode is NORMAL but the daily loss limit has been
breached for today (paper). `stop_new_entries=true` is the
authoritative trade-side block while SCR holds the system at PAUSED.

#### `regime_state.json`

```
regime               : trending_bull
previous_regime      : trending_bull
confidence           : 0.795
inputs_used          : [realized_vol_percentile, adx, trend_slope, market_breadth]
choppy_overlay       : active=true, score=0.55, sizing_multiplier=0.25, confidence_floor_add=0.15, block_trend_following=true
ttl_seconds          : 360   (CF03/CB04)
source               : live_loop.run_once
ts_utc               : 2026-05-03T12:48:51Z
```

#### `choppy_regime_state.json`

```
choppy_active                  : true
choppy_score                   : 0.55
proxy_symbol                   : SPY
entered_choppy_at_utc          : 2026-05-02T13:06:20Z
consecutive_choppy_reads       : 285
consecutive_clean_reads        : 0
ttl_seconds                    : 300
indicators:
  adx                          : 17.56     (adx_weak=true)
  direction_flips_5d           : 3         (high=true)
  failed_breakouts_10d         : 0         (high=false)
  small_loss_churn_ratio       : 0.0       (churn_high=false)
  trend_followthrough_rate     : 0.636     (weak=false)
thresholds:
  enter_threshold              : 0.55
  exit_threshold               : 0.35
  consecutive_to_enter         : 3
  consecutive_to_exit          : 4
  min_hold_minutes             : 60
```

Choppy has been active for ~24 hours; the BG13 fix means the proxy
asset is asset-class scoped (equity/options→SPY, futures→MES,
crypto/forex exempt) so the overlay does not accidentally suppress
24/7 markets.

#### `portfolio_snapshot.json`

```
ibkr_equity      : 177761.48
coinbase_equity  : 0.0
kraken_equity    : 184.58
ts_utc           : 2026-05-03T12:45:07Z
ttl_seconds      : 300
```

Total equity ≈ `$177,946.06 USD`.

### Verification commands run for this SSOT

```bash
git status --short                 # → empty (clean)
git rev-parse HEAD                 # → 02280094…
git log -8 --oneline               # → confirms 6 functional commits since fd96133
CHAD_SKIP_IB_CONNECT=1 \
  python3 -m pytest chad/tests/ -q # → 1064 passed, 29 warnings in 49.58s
PYTHONPATH=/home/ubuntu/chad_finale \
  CHAD_SKIP_IB_CONNECT=1 \
  python3 chad/core/full_cycle_preview.py
                                   # → orders_count=0, ibkr_intents=0, alpha_futures executed=False
                                   #   "[full_cycle_preview] NOTE: No broker calls were made. This is a logical preview only."
systemctl --failed --no-pager | grep chad
                                   # → no failed chad services
python3 -m py_compile chad/core/flip_executor.py
python3 -m py_compile chad/core/live_loop.py
                                   # → compile OK
```

---

## 3. STRATEGIES

The 16 wired strategies inherited from v8.8 are unchanged in v8.9. The
inventory and behavioural notes carry forward from v8.8 §3 verbatim
except where the audit-remediation work touched them; the touch-list
is short:

- **alpha_options** — `max_hold_seconds=3600` from v8.8 still in effect.
  No v8.9 change.
- **gamma_futures** — universe (`MCL`, `MYM`, `M2K`, `ZN`, `ZB`)
  unchanged. BG13's choppy proxy now resolves to `MES` for this
  asset class so a futures session is not falsely choppy-suppressed
  when SPY is choppy.
- **alpha / alpha_intraday / alpha_futures / alpha_options /
  alpha_crypto** — alpha cluster correlated-degradation page (R12)
  remains gated by `sample_count >= 10`.
- **alpha_crypto / kraken** — BG11's flip executor is the safety
  layer for crypto reversals (any FLIP intent must close-first
  with broker confirmation before opening the new side).
- All 16 strategies remain enabled in `config/strategy_weights.json`.

The full per-strategy detail (target universe, sizing rules, signal
shape, telemetry) lives unchanged in v8.8 §3; nothing in v8.9 has
modified per-strategy semantics.

---

## 4. EXECUTION PIPELINE

The post-fix execution path is now:

1. **Signal generation** — strategies emit `RawSignal` with side,
   confidence, intent_type, target notional, and `meta`.
2. **Signal router** (`chad/utils/signal_router.py`) — per `(symbol,
   side, asset_class)` bucket. Routes signals into `RoutedSignal`,
   preserving `meta["contributors"]` (CF01) — every contributing
   strategy's `(strategy, side, size, confidence)` tuple is retained
   for downstream attribution and audit.
3. **Net Exposure Conflict Gate**
   (`chad/execution/net_exposure_gate.py`):
   - Six-action enum (ALLOW / MERGE / REDUCE / CLOSE_ONLY /
     FLIP_ALLOWED / BLOCK).
   - Strategy priority table; reversal requires confidence ≥ 0.70
     AND delta ≥ 0.15 over incumbent.
   - **CB06**: when reconciliation is non-GREEN, ALL fresh entries
     are blocked (not just opposite-direction). Exits, reductions,
     hedges still allowed.
   - **CB07**: per-symbol daily loss limit now includes open-position
     **unrealized** drawdown (price from `runtime/price_cache.json`).
     Same-side fresh exposure is blocked once the realized + unrealized
     loss on `(strategy, symbol, side)` breaches the limit. Closes,
     reductions and exits remain permitted.
4. **Smart Strategy Throttle Gate**
   (`chad/execution/strategy_throttle_gate.py`):
   - Performance-aware: progressively throttles losing strategies.
   - **CB03**: state persisted atomically to
     `runtime/strategy_throttle_state.json`; reloaded on module
     import. Restart no longer clears pause/entry state.
5. **Signal throttle (R10 churn cap)** — `runtime/signal_throttle.json`
   read every cycle in `chad/core/live_loop.py:_apply_signal_throttle`.
   - **CB02**: protected signals (`intent_type` in
     `{exit, reduction, hedge, liquidation}`) bypass throttle trim
     entirely. Fresh entries are sorted by `confidence DESC` *before*
     trim so the highest-confidence signals survive the cap.
   - **SS05**: state changes write `runtime/signal_throttle_audit.json`
     so a programmatic activation/expiry leaves an audit trail.
     (Manual `rm signal_throttle.json` remains unobservable —
     documented residual risk §14.2.)
6. **ML Veto (shadow only)** — `chad/analytics/ml_veto_predictor.py`
   scores every intent. `ML_SHADOW` log line emitted regardless.
   `CHAD_ML_VETO_ENABLED` defaults OFF; veto execution requires
   shadow soak first (residual §14.3).
7. **Choppy regime overlay** — `regime_state.json:choppy_overlay`
   composed in `chad/risk/dynamic_risk_allocator.py` as a sizing
   multiplier (currently 0.25) and a confidence floor add (0.15).
   - **BG13**: proxy asset is asset-class scoped:
     - `equity`/`options` → SPY
     - `futures` → MES
     - `crypto`/`forex` → exempt (24/7 markets where the choppy
       concept is inapplicable)
8. **Risk allocation** — Profit Lock, SCR sizing factor, Choppy
   multiplier, dynamic caps. SCR PAUSED forces sizing 0.0 here.
9. **Execution plan build** — produces `ExecutionPlan` with per-symbol
   orders.
10. **FLIP Executor (NEW v8.9, `chad/core/flip_executor.py`)**:
    - `enforce_flip_close_first(intents, broker_state)` walks intents
      and, for any FLIP (opposite-direction across an existing
      position):
      - emits the close intent first;
      - **blocks the new flipped open intent until the broker
        confirms the close**;
      - **does not** mutate position guard / strategy attribution
        to "flat" pre-confirmation;
      - re-enqueues the open intent for the next cycle once the
        broker confirm arrives.
    - 6 dedicated tests in `chad/tests/test_flip_executor.py`.
11. **IBKR / Kraken adapter** — submits the surviving intents.
12. **Paper exec evidence writer** —
    `chad/execution/paper_exec_evidence_writer.py` writes per-trade
    `gross_pnl`, `commission`, `slippage`, `net_pnl` (DS07 — `net_pnl`
    is canonical; `pnl` deprecated alias retained for backwards
    compatibility with downstream readers).
13. **Position guard mutation** — only on broker fill confirmation
    (ISSUE-29 fix carried forward). Entries stamped with `_version`
    (top-level), `_written_by`, and per-entry `_entry_version`
    (CB08/DS03; migration stamped 44 existing entries).
14. **Reconciliation** — `chad/ops/reconciliation_publisher.py`
    compares broker state to `position_guard.json`. Status published
    to `runtime/reconciliation_state.json`. Status feeds back into
    step 3 (Net Exposure Gate).

The full-cycle preview (`chad/core/full_cycle_preview.py`) walks
steps 1–9 and reports `orders_count`, `ibkr_intents_count`,
`futures_orders`, dynamic caps, and a smoke probe of alpha_futures.
**Verified 2026-05-03**: `orders_count=0`, `ibkr_intents_count=0`,
`alpha_futures.executed=False`, ending banner `"[full_cycle_preview]
NOTE: No broker calls were made. This is a logical preview only."`

---

## 5. RISK & GOVERNANCE

### The full chain

The risk chain (top to bottom) is:

```
LiveGate          → CHAD_EXECUTION_MODE flip authorization
                    (single canonical reader; v8.9 unchanged)
SCR               → measured paper performance gate
                    (currently PAUSED, sizing_factor=0.0)
Profit Lock       → per-day equity drawdown gate
                    (NORMAL today; daily_loss_limit_hit=true)
Reconciliation    → broker-vs-guard equality gate (CB06: blocks
                    ALL fresh entries when non-GREEN)
Net Exposure Gate → per-symbol conflict & loss limits (CB07
                    realized + unrealized)
Strategy Throttle → per-strategy performance-aware throttle (CB03
                    persistent across restart)
Signal Throttle   → R10 churn cap (CB02 exits/reductions exempt;
                    SS05 audit trail)
Choppy Overlay    → regime-conditioned sizing/conf floor (BG13
                    asset-class scoped)
Flip Executor     → BG11 close-first/open-second invariant for
                    direction reversals
Edge Decay        → per-strategy halt enforcement
                    (BG10 writes strategy_allocations.json)
```

Each is independent and can independently block / down-size a trade.
`stop_new_entries=true` (currently set by Profit Lock daily loss
breach) is a hard veto.

### SCR as today's authoritative blocker

SCR is in **PAUSED**. The verbatim reason strings (preserved):

```
CONFIDENT blocked: sharpe_like=-1.287 < 0.500
CONFIDENT blocked: max_drawdown=-10314.12 < -10000.00
CAUTIOUS  blocked by sharpe_like=-1.287 < 0.100
PAUSED:   performance below cautious thresholds – live trading should be disabled.
Summary:  total_trades=5000, effective_trades=321, win_rate=0.698, sharpe_like=-1.287, max_drawdown=-10314.12, total_pnl=-20855.54
```

The path out is *natural recovery*: as paper performance improves,
sharpe_like rises and max_drawdown decays, the cautious gate (and
later the confident gate) reopens. Under governance rule #3 the
operator may not directly mutate SCR thresholds; the unlock is the
metrics moving.

### CB07 — Realized + unrealized loss limit

Pre-v8.9: per-symbol daily loss limit looked only at realized PnL.
Open-position drawdown could push the total loss far past the limit
without blocking.

v8.9: `chad/execution/net_exposure_gate.py` reads
`runtime/price_cache.json` for the current mark, computes unrealized
PnL for each open `(strategy, symbol, side)` triple, sums it with the
realized day's PnL, and applies the same `BLOCK` if the *combined*
loss exceeds `−CHAD_MAX_SYMBOL_DAILY_LOSS` (default `$300`). Same-side
fresh exposure is blocked; closes and reductions remain permitted.
Existing tests in `chad/tests/test_net_exposure_gate.py` were updated
to reflect the new reason string.

### CB06 — Reconciliation blocks all fresh entries

Pre-v8.9: when reconciliation was not GREEN, only opposite-direction
fresh exposure was blocked.

v8.9: any non-GREEN reconciliation status blocks ALL fresh entries
(same-side or opposite). Exits, reductions, hedges, and liquidations
remain allowed because they reduce risk.

### BG11 — Flip executor close-first invariant

Pre-v8.9: a FLIP could in principle race — the close intent and the
new open intent could be submitted in the same cycle, and intermediate
attribution / guard state could be mutated to "flat" before the broker
confirmed the close.

v8.9: `chad/core/flip_executor.py:enforce_flip_close_first()` is the
chokepoint. For any FLIP:

1. The close intent is emitted *alone*.
2. The new flipped open intent is held back.
3. Position guard / strategy attribution is **not** mutated to flat.
4. On broker fill confirmation of the close, the open intent is
   re-enqueued for the next cycle.
5. If the close fails or times out, the open intent is dropped — no
   false flat, no orphan position direction.

Six tests in `chad/tests/test_flip_executor.py` cover the
close-first/open-second sequence, the broker-confirmation gate, the
no-false-flat invariant, the close-failure path, and the
re-enqueue-after-confirm path.

### Profit Lock and daily-loss-limit interaction

Profit Lock currently reports:

- `mode = NORMAL`
- `sizing_factor = 1.0` (NORMAL mode)
- `daily_loss_limit_hit = true` (`-$14,880` today vs `-$5,338` 3% of
  equity)
- `stop_new_entries = true`

Even though mode is NORMAL (no profit lock thresholds tripped on the
upside), the daily-loss-limit branch independently sets
`stop_new_entries=true`. This is the intended behaviour: profit lock
participates in the chain whether it's compressing on profit or
braking on drawdown.

### LiveGate

Single canonical CHAD_EXECUTION_MODE reader at
`chad/execution/execution_config.get_execution_mode()`. v8.9 unchanged
from v8.8. Mode flip from `paper` → `live` requires:

1. Operator-initiated config edit (governance rule #3 — Pending
   Action).
2. `live_readiness.json:ready_for_live=true`.
3. Reconciliation GREEN.
4. SCR not PAUSED.
5. Paper soak per phase roadmap.
6. Operator GO via governance rule #3.

None of these are met today.

---

## 6. BUSINESS FRAMEWORK

The business framework (capital tiering, regime booster, winner
scaling, withdrawal manager, business phase tracker, profit router
drain) is unchanged from v8.8. v8.9 added no new tier rule, no new
booster path, no new withdrawal trigger.

| Component | Status (v8.9) |
|---|---|
| `TierManager` | BUILT |
| `RegimeBooster` | BUILT (active 1.35×) |
| `WinnerScaler` | BUILT (mixed multipliers; alpha=0.5, alpha_options=1.133, others 1.0) |
| `WithdrawalManager` | BUILT (deferred until live; residual §14.6) |
| `BusinessPhaseTracker` | BUILT (timestamp-diff `days_in_phase` from v8.8) |
| Profit router drain | BUILT v8.6 |
| Beta injection | flag OFF (residual §14.4) |
| Amplifier bucket | pending non-trivial multipliers (residual §14.5) |

**Salary withdrawal automation** remains deferred until live/PAY
phase (§14.6).

---

## 7. RECONCILIATION

Reconciliation publishes `runtime/reconciliation_state.json`. The
publisher is `chad/ops/reconciliation_publisher.py`.

### v8.9 changes

- **DS08 — bounded exclusion policy**: every entry in
  `exclusion_policy` carries `{reason, owner, added_utc, expires_utc,
  reviewed_utc}`. The publisher refuses to honor an exclusion whose
  `reviewed_utc` is older than the policy's review cadence (operator
  responsibility to keep `reviewed_utc` current).

### Current state (verified 2026-05-03)

- Status: GREEN.
- Counts: `chad_open=14`, `chad_strategy_open=15`,
  `broker_positions=14`. The one-position delta between
  `chad_open` and `chad_strategy_open` is the standard
  multi-strategy-on-one-symbol case (one strategy is hedging
  inside the same physical broker line).
- Worst diff: 0.0.
- Mismatches: none.
- Drifts: none.
- Excluded symbols: AAPL, NVDA. (MSFT also has policy entry but is
  no longer held — entry retained for audit.)

CB06 ensures that if this status flips to non-GREEN, all fresh entries
are blocked downstream. Exits, reductions, hedges remain permitted.

---

## 8. INTELLIGENCE LAYER

### Macro state (DS06)

`ops/macro_state_publish.py` publishes `runtime/macro_state.json` from
6 FRED series (DGS2, DGS10, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y)
with proper User-Agent header. Schema alignment (DS06) ensures the
*normal* publish path and the *fail-closed* path emit the same shape,
so downstream readers don't have to special-case fail-closed records.

### Event risk (DS05)

`ops/event_risk_publish.py` publishes `runtime/event_risk.json`. The
`next_event` field is now a structured object:

```
next_event = {
  name      : str,
  ts_utc    : str (ISO 8601),
  hours_until : float,
  severity  : str ("low"|"medium"|"high"|"critical"),
  source    : str ("fed_calendar"|"operator_curated"|...)
} | null
```

The `null` value is explicit when no upcoming event is in the window.

### Choppy regime (BG13)

`chad/analytics/choppy_regime_detector.py` writes
`runtime/choppy_regime_state.json`. v8.9 introduces **asset-class
scoping** of the proxy symbol:

- `equity` / `options` → `SPY`
- `futures` → `MES`
- `crypto` / `forex` → exempt (the choppy concept maps to a daily
  trading session; 24/7 markets do not have one)

The current state is `choppy_active=true` (since 2026-05-02T13:06Z),
proxy `SPY`, score `0.55`, `consecutive_choppy_reads=285`. Hysteresis
is `enter_threshold=0.55, exit_threshold=0.35,
consecutive_to_enter=3, consecutive_to_exit=4, min_hold_minutes=60`.

### CB05 — Fail-conservative on stale inputs

When upstream intelligence inputs go stale, the live loop now scales
*conservative* (not aggressive):

- Stale **winner scaler** → effective multiplier scaled to 0.5×.
- Stale **choppy state** → assume `active=True`, `sizing=0.50`,
  `confidence_add=0.10` (sizes down, raises confidence floor).
- Stale **booster** → no boost applied; warning logged.

The principle: when we don't know, we trade smaller, not bigger.

### ML veto (shadow)

`chad/analytics/ml_veto_predictor.py` is unchanged in API. The model
itself was retrained in v8.9 (commit `0228009`):

- 7,672 samples.
- Accuracy: 54.98%.
- Manifest: `shared/models/xgb_veto_manifest.json` records training
  metadata, sample count, accuracy, training timestamp, source dataset
  hash.
- **SS08 — model versioning**: the trainer creates a timestamped
  backup of the prior model file *before* overwriting; rollback is a
  single `cp` of the backup over the current model.
- `CHAD_ML_VETO_ENABLED` remains OFF. Shadow soak required before
  enabling (residual §14.3).

### Reconciliation feedback into intelligence

The `RECONCILED_PHASE2` exclusion remains removed from the
`strategy_health` computation (v8.8 carryover). Tier 2 Claude snapshot
still filtered by `sample_count >= 10`.

---

## 9. TELEGRAM OPERATOR INTERFACE

The Telegram operator interface from v8.8 is unchanged in v8.9.
Commands, allowed-chat-id gating, and the bot token sourced from
`/etc/chad/telegram.env` are all carried forward.

The audit-remediation work in v8.9 did not add a Telegram surface;
operator-facing changes (`live_readiness` cadence/requirements, SCR
state visibility) are all observable via the existing dashboard and
log channels.

---

## 10. DASHBOARD

Dashboard at `chadtrades.com` carries forward unchanged from v8.8.

The relevant v8.9 hardening is documented in `docs/SECURITY.md`
(SS02/SS03):

- Session TTL on operator login.
- Cookies marked `HttpOnly` and `SameSite=Strict`.
- Brute-force lockout on repeated authentication failures.
- Audit log for login attempts and privileged actions.

These were already wired pre-v8.9; the security doc formalizes the
contract so the audit can review it as a single artifact.

---

## 11. SERVICES & TIMERS

The systemd inventory from v8.8 carries forward. v8.9 changes are
limited and surgical:

### `chad-live-loop.service` (OP01)

```
[Unit]
After=...       (ordering retained)
# PartOf= REMOVED — live loop no longer cascades stop with IB Gateway
# Requires= REMOVED

[Service]
ExecStartPre=... 60s bounded price-cache readiness probe
ExecStart=...
```

Rationale: `PartOf=` and `Requires=` previously meant a transient IB
Gateway restart could cascade and stop the live loop. The audit
flagged that as a SS01-class concern (the live-trading engine is the
last unit that should be brought down by an upstream blip). After=
ordering is retained so the live loop still *prefers* IB Gateway to
be up first.

The `ExecStartPre` probe blocks startup for up to 60 seconds waiting
for `runtime/price_cache.json` to be present and recent. If the probe
times out, startup proceeds anyway with a warning.

### Health monitor (SS01 NEVER_AUTO_RESTART)

```python
NEVER_AUTO_RESTART = {
    "chad-live-loop.service",
    "chad-orchestrator.service",
}
```

The health monitor's auto-remediation refuses to `systemctl restart`
either of these. `regime_state.json` staleness is downgraded to
`NOTIFY_ONLY` because its publisher is `chad-live-loop`, and we do
not want the health monitor to bounce the trading engine.

### New units carried from v8.8

Unchanged — `chad-choppy-regime.timer`, `chad-xgb-train.timer`,
`chad-trade-closer.timer`. v8.9 added no new units.

### Failed services

Verified via `systemctl --failed --no-pager | grep chad`: **none**.

---

## 12. DATA & STORAGE

### Runtime artifacts (verified)

```
runtime/reconciliation_state.json       GREEN  (DS08 bounded exclusion policy)
runtime/scr_state.json                  PAUSED, sizing 0.0
runtime/live_readiness.json             ready_for_live=false (BG06 cadence/reqs)
runtime/stop_bus.json                   active=false
runtime/profit_lock_state.json          NORMAL, daily_loss_limit_hit=true
runtime/regime_state.json               trending_bull, choppy overlay active, TTL 360s (CF03)
runtime/choppy_regime_state.json        active=true, proxy=SPY (BG13 asset-class scoping)
runtime/portfolio_snapshot.json         ibkr=$177,761 + kraken=$184 ≈ $177,946
runtime/strategy_throttle_state.json    persisted across restart (CB03)
runtime/signal_throttle.json            present-or-absent; auto-expiry honored
runtime/signal_throttle_audit.json      NEW — state-change audit trail (SS05)
runtime/price_cache.json                read by Net Exposure Gate (CB07 unrealized)
runtime/macro_state.json                6-series FRED, schema-aligned (DS06)
runtime/event_risk.json                 next_event structured object (DS05)
```

### Position guard schema (CB08/DS03)

```json
{
  "_version": "position_guard.v2",
  "_written_by": "chad/core/position_guard.py:save_state",
  "entries": {
    "<symbol>": {
      "_entry_version": "position_guard_entry.v2",
      "strategy": "...",
      "side": "...",
      ...
    }
  }
}
```

Migration stamped 44 existing entries in this session. Future writes
include the version fields automatically. Downstream readers can
trust `_version` for compatibility branching.

### Trade record schema (DS07)

Every closed-trade record now carries:

```json
{
  "gross_pnl":   <float>,
  "commission":  <float>,
  "slippage":    <float>,
  "net_pnl":     <float>,    // canonical
  "pnl":         <float>     // deprecated alias retained = net_pnl
}
```

`net_pnl` is the canonical field. `pnl` is preserved as a deprecated
alias so v8.7-era readers don't crash; callers should be migrated to
`net_pnl`.

### Git tracking hygiene

- `runtime/pnl_state.json` removed from index (`git rm --cached`).
  File continues to exist on disk; history retained.
- `.gitignore` augmented with:
  - `runtime/pnl_state.json`
  - `*.bak`
  - `*.pre_*`
- `scripts/retry3_checkpoint.sh` deleted (old debug tool, unrelated to
  any current path).

### Storage paths (unchanged)

| Item | Path |
|---|---|
| Repo root | `/home/ubuntu/chad_finale` |
| Runtime state | `runtime/` |
| Daily bars | `data/bars/1d` |
| Minute bars | `data/bars/1m` |
| Fills | `data/fills` |
| Trade history | `data/trades` |
| Slippage ledgers | `data/slippage` |
| Reports | `reports/` |
| Revert tarballs | `/home/ubuntu/chad_revert_points/` |
| Models | `shared/models/` |
| Systemd units | `/etc/systemd/system/chad-*` |
| Env files | `/etc/chad/*.env` |

---

## 13. CHANGE LOG

### Commits since v8.8 (`fd96133..HEAD`)

In topological order (oldest → newest):

```
36b94f6  Fix: CB02 exit-exempt throttle, CB03 persist throttle state, CF03 TTL 360s, CB06 recon blocks all fresh, BG10 halt writes, BG04 choppy+macro watched
71c0d11  Fix: CB07 unrealized PnL loss limit, CB08 version field position guard, BG11 FLIP safety, BG13 asset-class choppy scoping, BG06 live_readiness, SS01 NEVER_AUTO_RESTART, SS05 throttle audit, SS08 model versioning
0060ce4  Fix: CF01 contributors field, DS05 next_event schema, DS07 canonical net_pnl, DS08 exclusion policy, CB05 fail-conservative stale, OP01 startup deps, SS02/SS03 security doc
b424748  Audit remediation Batch 4: BG11 flip executor close-first, SS02 security doc, Kraken test isolation — 1064 passed 0 failed
df3b519  Housekeeping: untrack runtime/pnl_state.json, gitignore *.bak and *.pre_* backup files
0228009  SS08: model versioning artifacts — retrained xgb_veto_model + manifest (7672 samples, accuracy=54.98%)
```

Six functional commits, plus housekeeping. Below — the audit-finding
view, grouped by class.

### 13.1 Critical Blockers (CB)

| ID | Description | Files | Resolution |
|---|---|---|---|
| **CB01 / CF01** | Signal provenance / contributors field absent | `chad/utils/signal_router.py` | `RoutedSignal.meta["contributors"]` populated with every contributing strategy's `(strategy, side, size, confidence)` tuple. Bucket key clarified to `(symbol, side, asset_class)`. |
| **CB02** | Exit signals subject to throttle trim | `chad/core/live_loop.py` | Protected signals (`intent_type` in `{exit, reduction, hedge, liquidation}`) bypass throttle entirely. Fresh entries sorted by `confidence DESC` *before* trim so highest-confidence entries survive the cap. |
| **CB03** | Strategy throttle state lost on restart | `chad/execution/strategy_throttle_gate.py`, `runtime/strategy_throttle_state.json` | State persisted with atomic write; reload on module import. |
| **CB04 / CF03** | TTL drift across regime publishers/consumers | `chad/analytics/regime_classifier.py`, `chad/ops/health_monitor_rules.py`, `chad/ops/feed_watchdog.py` | All callers aligned at `regime_state.json` TTL 360s. |
| **CB05** | Stale inputs caused aggressive (not conservative) sizing | live loop overlay composition | Stale winner → 0.5× scaling; stale choppy → assume `active=True, sizing=0.50, confidence_add=0.10`; stale booster → no boost; warning logged in all three. |
| **CB06** | Reconciliation only blocked opposite-direction fresh entries | `chad/execution/net_exposure_gate.py` | Non-GREEN reconciliation now blocks ALL fresh entries (same- or opposite-side). Exits / reductions / hedges still allowed. |
| **CB07** | Per-symbol daily loss limit ignored unrealized | `chad/execution/net_exposure_gate.py`, `runtime/price_cache.json` | Realized + unrealized aggregated per `(strategy, symbol, side)`; same-side fresh exposure blocked once combined loss exceeds limit. |
| **CB08 / DS03** | Position guard not versioned | `chad/core/position_guard.py` | Top-level `_version` and `_written_by`; per-entry `_entry_version`. Migration stamped 44 existing entries in this session. |

### 13.2 Bugs (BG)

| ID | Description | Files | Resolution |
|---|---|---|---|
| **BG04** | `choppy_regime_state.json` and `macro_state.json` not in feed watchdog | `chad/ops/feed_watchdog.py`, `chad/ops/health_monitor_rules.py` | Added with TTL 900s (choppy) and 7200s (macro). Remediation map updated. |
| **BG06** | `live_readiness.json` lacking cadence/requirements fields | `chad/ops/live_readiness_publisher.py` | Refreshed with `next_evaluation_cadence`, `requirements_remaining`, `last_evaluated_utc`, `evaluated_by`. |
| **BG10** | `HALT_DEFER_TO_EDGE_DECAY` did not write to allocations | `chad/risk/edge_decay_monitor.py` | `set_strategy_halted()` now writes to `runtime/strategy_allocations.json` so downstream allocator sees the halt. |
| **BG11** | FLIP could mutate state before broker confirmation | `chad/core/flip_executor.py` (NEW), 6 tests in `chad/tests/test_flip_executor.py` | Close-first/open-second invariant enforced; new flipped open intent blocked until broker confirms close; no false-flat mutation. |
| **BG13** | Choppy proxy was always SPY regardless of asset class | `chad/analytics/choppy_regime_detector.py` | Proxy resolves by asset class: equity/options→SPY, futures→MES, crypto/forex→exempt. |

### 13.3 Schema / Data (DS)

| ID | Description | Resolution |
|---|---|---|
| **DS05** | `event_risk.next_event` not structured | Now `{name, ts_utc, hours_until, severity, source}` object or `null`. |
| **DS06** | `macro_state` normal vs fail-closed schema diverged | Aligned — both paths emit the same field shape. |
| **DS07** | `pnl` field ambiguous (gross? net?) | `net_pnl` canonical; `pnl` preserved as deprecated alias = `net_pnl`. `gross_pnl`, `commission`, `slippage` explicit on every record. |
| **DS08** | Reconciliation exclusions free-form | Bounded policy: each entry carries `{reason, owner, added_utc, expires_utc, reviewed_utc}`. |

### 13.4 Security / Safety (SS)

| ID | Description | Resolution |
|---|---|---|
| **SS01** | Health monitor could auto-restart trading engines | `NEVER_AUTO_RESTART = {chad-live-loop.service, chad-orchestrator.service}`. `regime_state.json` staleness downgraded to NOTIFY_ONLY. |
| **SS02 / SS03** | No formal security policy doc | `docs/SECURITY.md` (246 lines) — file ownership shapes (`root:chad 640` for service-shared, `ubuntu:ubuntu 600/640` for operator-only); forbidden modes (world-readable, group-writable, unexpected owners); dashboard auth (session TTL, HttpOnly+SameSite=Strict cookies, brute-force lockout, audit log); live/paper promotion workflow; no-import-time-broker-connection policy. |
| **SS05** | Manual `signal_throttle.json` clear unobservable | `runtime/signal_throttle_audit.json` written on every state change. Manual `rm` of the throttle file remains unauditable — documented as accepted residual risk §14.2. |
| **SS08** | Retraining could overwrite the live model irreversibly | Trainer takes a timestamped backup before overwriting; `shared/models/xgb_veto_manifest.json` records metadata; rollback is `cp <backup> <model>`. |

### 13.5 Operations (OP)

| ID | Description | Resolution |
|---|---|---|
| **OP01** | `chad-live-loop.service` cascaded with IB Gateway via `PartOf=`/`Requires=` | Removed both. `After=` ordering retained. Added `ExecStartPre` 60s bounded price-cache readiness probe. |

### 13.6 Housekeeping

- Removed `runtime/pnl_state.json` from git tracking
  (`git rm --cached`); file remains on disk.
- `.gitignore` extended: `runtime/pnl_state.json`, `*.bak`, `*.pre_*`.
- Deleted `scripts/retry3_checkpoint.sh` (old debug tool).

### 13.7 Tests

- **1064 passed / 0 failed** (verified `CHAD_SKIP_IB_CONNECT=1
  pytest`, 49.58s).
- New: `chad/tests/test_flip_executor.py` — 6 BG11 tests
  (close-first/open-second, broker-confirmation gate, no-false-flat
  invariant, close-failure handling, re-enqueue-after-confirm).
- Fixed: `chad/tests/test_kraken_execution.py` — runtime isolation
  fix (was using a shared runtime path that other tests mutated).
- Updated: `chad/tests/test_net_exposure_gate.py` — updated reason
  string for CB07 realized + unrealized aggregation.
- Compile: OK for `chad/core/flip_executor.py` and
  `chad/core/live_loop.py` (verified via `python3 -m py_compile`).
- Full-cycle preview: PASS, no broker calls made.

### 13.8 Aggregate

Six functional commits, no module deletions of consequence (one
unrelated debug script removed). Net code change: heavily additive —
one new module (flip executor), one new audit artifact path (signal
throttle audit), one new model manifest, one new security policy
document. All other changes are surgical edits to existing modules to
satisfy audit-finding semantics.

---

## 14. KNOWN ISSUES / ACCEPTED RESIDUAL RISKS

v8.9 explicitly does **not** hide these. The audit lock requires that
each unresolved item be named, classified, and given a mitigation /
deferred owner.

### 14.1 SCR PAUSED (current operational blocker)

| Field | Value |
|---|---|
| Severity | OPERATIONAL — current trading block |
| Status | Risk control working as designed |
| State | `PAUSED`, `sizing_factor=0.0` |
| Reason | `sharpe_like=-1.287 < 0.500` (CONFIDENT gate); `max_drawdown=-10314.12 < -10000.00` (CONFIDENT gate); `sharpe_like=-1.287 < 0.100` (CAUTIOUS gate) |
| Required unlock | SCR natural recovery — measured paper performance must improve until cautious thresholds are no longer breached |
| Owner | SCR governance |
| Note | This is **not** a code failure. Per CLAUDE.md governance rule #3, the operator may not directly mutate SCR thresholds. v8.9 does not authorize a workaround. |

### 14.2 SS05 — Manual signal-throttle clear is unauditable

| Field | Value |
|---|---|
| Severity | OPERATIONAL |
| Status | Accepted residual risk |
| Description | `rm runtime/signal_throttle.json` from the shell bypasses `signal_throttle_audit.json` because no module is invoked. |
| Mitigation (deferred) | Operator CLI wrapper that wraps the rm + writes to the audit file, OR an `inotify` watcher daemon that observes file deletion and writes a `manual_clear` event. |
| Owner | Ops (deferred) |

### 14.3 ML veto in shadow only

| Field | Value |
|---|---|
| Severity | OPERATIONAL |
| Status | Behind flag, soak required |
| Flag | `CHAD_ML_VETO_ENABLED=off` (default) |
| Model accuracy | 54.98% (commit `0228009`, 7,672 samples) |
| Threshold | `CHAD_ML_VETO_THRESHOLD=0.65` |
| Required unlock | Shadow soak (1–2 weeks of `ML_SHADOW` log accumulation), threshold tuning, canary on a single strategy first |
| Owner | Intelligence |

### 14.4 Beta injection

| Field | Value |
|---|---|
| Severity | OPERATIONAL |
| Status | Behind flag, paper soak required |
| Flag | `CHAD_PROFIT_ROUTER_BETA_INJECTION=off` (default) |
| Required unlock | 1–2 weeks of clean drain telemetry in test mode first |
| Owner | Risk |

### 14.5 Amplifier bucket

| Field | Value |
|---|---|
| Severity | DESIGN |
| Status | Pending meaningful winner-scaler multipliers |
| Description | `winner_scaler` currently produces `alpha=0.5`, `alpha_options=1.133`, others 1.0. Amplifier bucket should inject into the top-multiplier strategy similarly to the beta path once more non-trivial multipliers re-emerge. |
| Owner | Risk |

### 14.6 Salary withdrawal automation

| Field | Value |
|---|---|
| Severity | NOT YET BUILT |
| Status | Deferred until live / PAY phase |
| Description | `WithdrawalManager` authorizes; operator still moves money manually. Automation is out of scope until live capital is engaged and PAY phase is sustained. |
| Owner | Business |

### 14.7 Live promotion not authorized

| Field | Value |
|---|---|
| Severity | INFORMATIONAL |
| Status | v8.9 is NOT live approval |
| Description | Live promotion requires (in any order): operator GO via governance rule #3; `live_readiness.json:ready_for_live=true`; reconciliation GREEN (currently true); SCR not PAUSED (currently PAUSED — blocker); paper soak per phase roadmap; LiveGate authorization. |
| Owner | Operator |

### 14.8 Verification gaps

None encountered during the v8.9 verification sequence. Every command
executed cleanly; every runtime JSON parsed.

---

## 15. PHASE ROADMAP

### IMMEDIATE (next session)

- **Watch SCR recover.** SCR PAUSED is the current operational
  blocker. Track `effective_trades`, `sharpe_like`, `max_drawdown`
  per cycle. SCR will lift when measured performance crosses back
  above the cautious thresholds.
- **Profit lock daily-loss-limit reset.** `daily_loss_today=-14,880`
  is today's paper realized; `stop_new_entries=true` clears at UTC
  midnight (then resets per-day).
- **Choppy regime recovery.** Currently active for ~24 hours,
  consecutive_choppy_reads=285. Hysteresis requires 4 consecutive
  clean reads + 60-min minimum hold to exit.
- **Equity history.** `live_readiness.requirements_remaining`
  includes `14_day_equity_history` and `60_day_paper_performance` —
  these are wall-clock requirements regardless of SCR.
- **ML veto shadow soak.** Continue accumulating `ML_SHADOW` log
  entries.
- **Backup cleanup.** `*.bak` and `*.pre_*` are now gitignored;
  operator may clean up the working tree at leisure.

### Phase 5B — ML veto loop (BUILT v8.8 — soak phase)

Status: BUILT, SHADOW MODE, model retrained v8.9 to 54.98% accuracy.

1. Shadow soak (1–2 weeks).
2. Threshold tuning.
3. Canary promotion (single strategy, e.g. `alpha`, for 1 week).
4. Full promotion.

### Phase 9 — pre-live calibration

- ~~Choppy detector~~ — built v8.8, asset-class scoped v8.9.
- Kelly fraction tuning (`CHAD_ALLOC_V3_KELLY_MAX`).
- Live feature distribution drift monitoring.
- Net-EV gate opt-in (`expected_pnl` at strategy level).

### Phase 10 — live capital flip (CARRIED — NOT AUTHORIZED IN v8.9)

Entry criteria (none of these are met today):

- All Phase-9 items complete.
- **SCR ≠ PAUSED** (currently PAUSED).
- 60-90 days consistent paper performance.
- Explicit operator GO (governance rule #3).
- WithdrawalManager phase = PAY for ≥ 14 days.
- ML veto shadow soak complete; threshold tuned.
- Beta injection flag enabled and validated.
- Live readiness `ready_for_live=true` (currently false).

When live (for reference only — not active):

- `CHAD_EXECUTION_MODE` flips paper → live (single canonical reader).
- `LiveGate` accepts the posture change.
- First 3 cycles run with manual oversight.
- Profit routing flips from advisory to actual capital movement.

---

## 16. APPENDICES

### Appendix A — File inventory of changed paths since v8.8

| File | Touched in commits | Change |
|---|---|---|
| `chad/core/live_loop.py` | `36b94f6`, `0060ce4` | M (CB02 exit-exempt, conf-DESC trim; CB05 stale handling) |
| `chad/core/flip_executor.py` | `71c0d11`, `b424748` | A (NEW — BG11 close-first/open-second) |
| `chad/core/position_guard.py` | `71c0d11` | M (CB08/DS03 versioning, 44 entries migrated) |
| `chad/utils/signal_router.py` | `0060ce4` | M (CF01 contributors field) |
| `chad/execution/net_exposure_gate.py` | `36b94f6`, `71c0d11` | M (CB06 recon blocks all fresh; CB07 unrealized PnL) |
| `chad/execution/strategy_throttle_gate.py` | `36b94f6` | M (CB03 persistent state) |
| `chad/execution/paper_exec_evidence_writer.py` | `0060ce4` | M (DS07 canonical net_pnl) |
| `chad/analytics/regime_classifier.py` | `36b94f6` | M (CF03/CB04 TTL 360s) |
| `chad/analytics/choppy_regime_detector.py` | `71c0d11` | M (BG13 asset-class proxy) |
| `chad/risk/edge_decay_monitor.py` | `36b94f6` | M (BG10 set_strategy_halted writes allocations) |
| `chad/risk/dynamic_risk_allocator.py` | `0060ce4` | M (CB05 stale-input handling in overlay) |
| `chad/ops/feed_watchdog.py` | `36b94f6` | M (BG04 choppy + macro watched) |
| `chad/ops/health_monitor_rules.py` | `71c0d11`, `36b94f6` | M (SS01 NEVER_AUTO_RESTART; BG04 remediation map; CF03 360s) |
| `chad/ops/health_monitor.py` | `71c0d11` | M (SS01 enforcement) |
| `chad/ops/live_readiness_publisher.py` | `71c0d11` | M (BG06 cadence/requirements) |
| `chad/ops/reconciliation_publisher.py` | `0060ce4` | M (DS08 bounded exclusion policy) |
| `ops/event_risk_publish.py` | `0060ce4` | M (DS05 next_event structured) |
| `ops/macro_state_publish.py` | `0060ce4` | M (DS06 schema alignment) |
| `chad/analytics/train_xgb_model.py` | `71c0d11`, `0228009` | M (SS08 timestamped backups, manifest write) |
| `shared/models/xgb_veto_model.json` | `0228009` | M (retrained — 7,672 samples, 54.98%) |
| `shared/models/xgb_veto_manifest.json` | `0228009` | A (NEW — model metadata) |
| `chad/tests/test_flip_executor.py` | `b424748` | A (NEW — 6 BG11 tests) |
| `chad/tests/test_net_exposure_gate.py` | `36b94f6`, `71c0d11` | M (reason string update for CB06/CB07) |
| `chad/tests/test_kraken_execution.py` | `b424748` | M (runtime isolation fix) |
| `docs/SECURITY.md` | `0060ce4`, `b424748` | A (NEW — 246 lines, SS02/SS03) |
| `/etc/systemd/system/chad-live-loop.service` | `0060ce4` | M (OP01 — PartOf/Requires removed; ExecStartPre probe) |
| `runtime/strategy_throttle_state.json` | `36b94f6` | A (created on first persistent write) |
| `runtime/signal_throttle_audit.json` | `71c0d11` | A (created on first state-change write) |
| `runtime/pnl_state.json` | `df3b519` | D (removed from git index; file remains on disk) |
| `.gitignore` | `df3b519` | M (`runtime/pnl_state.json`, `*.bak`, `*.pre_*`) |
| `scripts/retry3_checkpoint.sh` | `df3b519` | D (deleted — old debug tool) |

### Appendix B — Environment files (names only)

Unchanged from v8.8.

```
/etc/chad/dashboard.env
/etc/chad/claude.env
/etc/chad/ibkr.env
/etc/chad/kraken.env
/etc/chad/openai.env
/etc/chad/polygon.env
/etc/chad/telegram.env
/etc/chad/chad.env
```

### Appendix C — Critical paths cheat sheet (v8.9 deltas highlighted)

| Item | Path |
|---|---|
| Repo root | `/home/ubuntu/chad_finale` |
| Hot-path entry | `chad/core/orchestrator.py`, `chad/core/live_loop.py` |
| **Flip executor (NEW v8.9)** | **`chad/core/flip_executor.py`** |
| Execution adapter | `chad/execution/ibkr_adapter.py` |
| Net Exposure Gate (CB06/CB07) | `chad/execution/net_exposure_gate.py` |
| Strategy Throttle Gate (CB03) | `chad/execution/strategy_throttle_gate.py` |
| Position guard (CB08/DS03) | `chad/core/position_guard.py` |
| Signal router (CF01) | `chad/utils/signal_router.py` |
| Choppy detector (BG13) | `chad/analytics/choppy_regime_detector.py` |
| ML veto predictor | `chad/analytics/ml_veto_predictor.py` |
| **Model manifest (NEW v8.9)** | **`shared/models/xgb_veto_manifest.json`** |
| **Security policy (NEW v8.9)** | **`docs/SECURITY.md`** |
| Health monitor (SS01) | `chad/ops/health_monitor.py`, `health_monitor_rules.py` |
| Feed watchdog (BG04) | `chad/ops/feed_watchdog.py` |
| Live readiness publisher (BG06) | `chad/ops/live_readiness_publisher.py` |
| Reconciliation publisher (DS08) | `chad/ops/reconciliation_publisher.py` |
| Event risk publisher (DS05) | `ops/event_risk_publish.py` |
| Macro state publisher (DS06) | `ops/macro_state_publish.py` |
| Edge decay halt (BG10) | `chad/risk/edge_decay_monitor.py` |
| Allocator (CB05 stale handling) | `chad/risk/dynamic_risk_allocator.py` |
| Evidence writer (DS07) | `chad/execution/paper_exec_evidence_writer.py` |
| Live-loop systemd (OP01) | `/etc/systemd/system/chad-live-loop.service` |

### Appendix D — Operator one-liners (v8.9 additions)

```bash
# Are flip executor invariants holding? Look for FLIP-related events.
journalctl -u chad-live-loop --since=today | grep -E "flip_executor|FLIP_"

# What does the signal throttle audit say happened last?
cat /home/ubuntu/chad_finale/runtime/signal_throttle_audit.json 2>/dev/null \
  | python3 -m json.tool | tail -40

# Current ML veto model metadata
cat /home/ubuntu/chad_finale/shared/models/xgb_veto_manifest.json \
  | python3 -m json.tool

# Verify NEVER_AUTO_RESTART is honored (should not see restart actions on these)
journalctl -u chad-health-monitor --since=today \
  | grep -E "chad-live-loop|chad-orchestrator"

# Check reconciliation exclusion policy review timestamps
python3 -c "
import json; from pathlib import Path
d = json.loads(Path('runtime/reconciliation_state.json').read_text())
for sym, pol in d.get('exclusion_policy', {}).items():
    print(f'{sym}: reviewed_utc={pol[\"reviewed_utc\"]}, owner={pol[\"owner\"]}')
"

# SCR snapshot — current pause reasons
python3 -c "
import json; from pathlib import Path
d = json.loads(Path('runtime/scr_state.json').read_text())
print(f'state={d[\"state\"]} sizing={d[\"sizing_factor\"]}')
for r in d['reasons']: print(' -', r)
"
```

### Appendix E — Verification log (executed for v8.9)

```
$ git status --short
                                          # (empty — clean)
$ git rev-parse HEAD
02280094d85ebac3643b41967ec0542c87f4b339
$ git log -8 --oneline
0228009 SS08: model versioning artifacts — retrained xgb_veto_model + manifest (7672 samples, accuracy=54.98%)
df3b519 Housekeeping: untrack runtime/pnl_state.json, gitignore *.bak and *.pre_* backup files
b424748 Audit remediation Batch 4: BG11 flip executor close-first, SS02 security doc, Kraken test isolation — 1064 passed 0 failed
0060ce4 Fix: CF01 contributors field, DS05 next_event schema, DS07 canonical net_pnl, DS08 exclusion policy, CB05 fail-conservative stale, OP01 startup deps, SS02/SS03 security doc
71c0d11 Fix: CB07 unrealized PnL loss limit, CB08 version field position guard, BG11 FLIP safety, BG13 asset-class choppy scoping, BG06 live_readiness, SS01 NEVER_AUTO_RESTART, SS05 throttle audit, SS08 model versioning
36b94f6 Fix: CB02 exit-exempt throttle, CB03 persist throttle state, CF03 TTL 360s, CB06 recon blocks all fresh, BG10 halt writes, BG04 choppy+macro watched
fd96133 Docs: CHAD Unified SSOT v8.8 — issue closure session, all gates built, 0 failing tests, choppy regime active
a8556d8 Fix: per-symbol daily loss limit in Net Exposure Gate — blocks repeated entries after $300 loss (LLY scenario)

$ CHAD_SKIP_IB_CONNECT=1 python3 -m pytest chad/tests/ -q | tail -5
1064 passed, 29 warnings in 49.58s

$ python3 -m py_compile chad/core/flip_executor.py chad/core/live_loop.py
                                          # (no output — compile OK)

$ PYTHONPATH=/home/ubuntu/chad_finale CHAD_SKIP_IB_CONNECT=1 python3 chad/core/full_cycle_preview.py | tail -3
intents_count: 0
artifact_exists: True
[full_cycle_preview] NOTE: No broker calls were made. This is a logical preview only.

$ systemctl --failed --no-pager | grep chad
                                          # (no output — no failed chad services)
```

### Appendix F — Audit-finding cross-reference

| Audit ID | Class | Status v8.9 | Where it lives in this doc |
|---|---|---|---|
| CB01 | Critical Blocker | FIXED | §13.1, §4 step 2 |
| CB02 | Critical Blocker | FIXED | §13.1, §4 step 5 |
| CB03 | Critical Blocker | FIXED | §13.1, §4 step 4 |
| CB04 | Critical Blocker | FIXED | §13.1, §2 (regime_state) |
| CB05 | Critical Blocker | FIXED | §13.1, §8 |
| CB06 | Critical Blocker | FIXED | §13.1, §4 step 3, §5 |
| CB07 | Critical Blocker | FIXED | §13.1, §4 step 3, §5 |
| CB08 | Critical Blocker | FIXED | §13.1, §12 |
| CF01 | Critical Finding | FIXED | §13.1, §4 step 2 |
| CF03 | Critical Finding | FIXED | §13.1, §2 (regime_state) |
| DS03 | Data Schema | FIXED (paired with CB08) | §12 |
| DS05 | Data Schema | FIXED | §13.3, §8 |
| DS06 | Data Schema | FIXED | §13.3, §8 |
| DS07 | Data Schema | FIXED | §13.3, §12 |
| DS08 | Data Schema | FIXED | §13.3, §7 |
| BG04 | Bug | FIXED | §13.2, §11 |
| BG06 | Bug | FIXED | §13.2, §2 (live_readiness) |
| BG10 | Bug | FIXED | §13.2 |
| BG11 | Bug | FIXED + 6 tests | §13.2, §4 step 10, §5 |
| BG13 | Bug | FIXED | §13.2, §4 step 7, §8 |
| SS01 | Security/Safety | FIXED | §13.4, §11 |
| SS02 | Security/Safety | FIXED — `docs/SECURITY.md` | §13.4, §10 |
| SS03 | Security/Safety | FIXED — `docs/SECURITY.md` | §13.4, §10 |
| SS05 | Security/Safety | PARTIAL — programmatic audit + documented residual | §13.4, §14.2 |
| SS08 | Security/Safety | FIXED | §13.4, §8 |
| OP01 | Operations | FIXED | §13.5, §11 |

### Appendix G — Active git tags

- `STABILITY_FREEZE_20260307_GREEN` — original stable baseline.
- `PRE_HARDENING_20260402` — snapshot before P0 hardening began.
- `RATIFICATION_MASTER_20260402` — all hardening and GAP items
  complete.
- `REVERT_PRE_OVERHAUL_20260419` — snapshot before 2026-04-19/21
  overhaul.

v8.9 has not yet been tagged; the lock document lands first, then the
operator may apply an `AUDIT_REMEDIATION_LOCK_20260503` tag at their
discretion.

### Appendix H — Rollback

Per CLAUDE.md: `git checkout RATIFICATION_MASTER_20260402` returns
the codebase to the pre-overhaul ratified baseline. The
`REVERT_PRE_OVERHAUL_20260419` tag (commit `45f3728`; tarball
`/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`)
is the more recent escape hatch.

### Appendix I — Things v8.9 deliberately does NOT claim

To make the lock state unambiguous:

- v8.9 does NOT claim "live trading ready."
- v8.9 does NOT claim "SCR CONFIDENT."
- v8.9 does NOT authorize a `CHAD_EXECUTION_MODE` flip.
- v8.9 does NOT authorize a config mutation.
- v8.9 does NOT authorize a service restart.
- v8.9 does NOT override SCR's PAUSED state.
- v8.9 does NOT lift the `daily_loss_limit_hit` profit-lock branch.
- v8.9 does NOT enable `CHAD_ML_VETO_ENABLED`.
- v8.9 does NOT enable `CHAD_PROFIT_ROUTER_BETA_INJECTION`.
- v8.9 does NOT mark any of the residual risks (§14) as resolved.

### Appendix J — At-a-glance status

```
tests                              : 1064 / 0
audit remediation                  : LOCKED
live readiness                     : false
SCR                                : PAUSED, sizing 0.0
critical audit blockers            : fixed or residual documented
flip executor (BG11)               : ACTIVE — 6 tests passing
signal throttle persistence (CB03) : ACTIVE
choppy asset scoping (BG13)        : ACTIVE
per-symbol loss limit (CB07)       : realized + unrealized
NEVER_AUTO_RESTART (SS01)          : live-loop, orchestrator
model versioning (SS08)            : code path proven
reconciliation                     : GREEN
stop bus                           : false
profit lock                        : NORMAL but daily_loss_limit_hit=true → stop_new_entries
operational trading                : disabled by SCR (correctly)
v8.9 lock type                     : audit remediation, NOT live promotion
```

### Appendix K — v8.8 → v8.9 delta table

| Area | v8.8 state | v8.9 state |
|---|---|---|
| Signal contributors | absent | `meta["contributors"]` populated (CF01) |
| Throttle exit-exempt | exits could be trimmed | exits/reductions/hedges/liquidations exempt; fresh sorted by confidence DESC (CB02) |
| Strategy throttle persistence | lost on restart | persisted to `strategy_throttle_state.json` (CB03) |
| Regime TTL alignment | drift across publishers/consumers | all aligned at 360s (CB04/CF03) |
| Stale input handling | could scale aggressively | conservative defaults across winner/choppy/booster (CB05) |
| Recon non-GREEN gating | only opposite-direction blocked | ALL fresh entries blocked (CB06) |
| Per-symbol loss limit | realized only | realized + unrealized via `price_cache.json` (CB07) |
| Position guard versioning | unversioned | `_version`/`_entry_version`/`_written_by`; 44 entries migrated (CB08/DS03) |
| FLIP safety | could mutate state pre-confirm | dedicated executor; close-first/open-second; broker-confirmed; 6 tests (BG11) |
| Choppy proxy | always SPY | asset-class scoped: equity→SPY, futures→MES, crypto/forex exempt (BG13) |
| `event_risk.next_event` | unstructured | `{name, ts_utc, hours_until, severity, source}` or null (DS05) |
| Macro state schema | normal vs fail-closed diverged | aligned (DS06) |
| Trade record canonical PnL | `pnl` ambiguous | `net_pnl` canonical; `pnl` deprecated alias; gross/commission/slippage explicit (DS07) |
| Reconciliation exclusions | free-form | bounded `{reason, owner, added_utc, expires_utc, reviewed_utc}` (DS08) |
| Watchdog coverage | choppy/macro absent | added with TTLs (BG04) |
| `live_readiness` schema | bare ready_for_live | + cadence + requirements_remaining + last_evaluated_utc + evaluated_by (BG06) |
| Halt write path | did not write allocations | `set_strategy_halted()` writes `strategy_allocations.json` (BG10) |
| Health monitor restart policy | could restart trading engines | NEVER_AUTO_RESTART for live-loop and orchestrator (SS01) |
| Security policy doc | absent | `docs/SECURITY.md` 246 lines (SS02/SS03) |
| Signal throttle audit | none | `signal_throttle_audit.json` on state changes (SS05 partial) |
| Model versioning | overwrite-in-place | timestamped backup + `xgb_veto_manifest.json` (SS08) |
| Live-loop systemd | `PartOf`/`Requires` cascade | removed; After= retained; ExecStartPre 60s probe (OP01) |
| Git tracking hygiene | `pnl_state.json` tracked; `*.bak`/`*.pre_*` not ignored | `pnl_state.json` untracked; `*.bak`/`*.pre_*` ignored |
| Tests | 1053 passed (v8.8 close) | 1064 passed (v8.9; +11 net incl. flip executor) |
| Live readiness | false | false (unchanged — three independent gates remain) |
| SCR | varied across v8.8 lifetime | PAUSED at v8.9 lock — operational blocker |

---

End of CHAD Unified SSOT v8.9.
