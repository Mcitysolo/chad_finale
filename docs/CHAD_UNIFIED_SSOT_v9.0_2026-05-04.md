# CHAD Unified SSOT v9.0 — Paper Epoch 2 Clean Runtime / Missing-List Closure Lock

**Version:** 9.0
**Date:** 2026-05-04
**Status:** Active — Paper Epoch 2 / Missing-List Closure Lock / Clean Evidence Collection
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v8.9_2026-05-03.md` (commit `0228009`, 2026-05-03)
**Repository HEAD:** `46245a5ec4d2a3ec8d0856719037181108a4cb30` (`46245a5`)
**Branch:** `main`
**Test status:** 1219 passed / 0 failed, 34 warnings (verified `CHAD_SKIP_IB_CONNECT=1 pytest`, 2026-05-04)
**Live readiness:** `runtime/live_readiness.json:ready_for_live=false`
**SCR posture:** `WARMUP` (Paper Epoch 2), `sizing_factor=0.10`, `effective_trades=5`
**Lock type:** Post-v8.9 remediation closure lock — **NOT a live-trading approval**

> **Live promotion not authorized.** This document captures the post-v8.9
> missing-list closure and the start of clean Paper Epoch 2 evidence
> collection. CHAD is **not** live-approved. SCR is not CONFIDENT. Strategy
> performance is not yet proven on clean Epoch 2 data. Live promotion still
> requires clean paper performance, `live_readiness.ready_for_live=true`,
> SCR promotion out of WARMUP, operator GO, and any remaining governance
> gates.

---

## 1. Executive Summary

### What v8.9 was

v8.9 was the **external audit remediation lock**. An external forensic SSOT
audit identified critical production-readiness gaps in v8.8; v8.9 closed
the audit blockers in code/tests or accepted them with documented mitigation.
At v8.9 lock, SCR was `PAUSED` with `sizing_factor=0.0` because Epoch 1
paper performance had breached cautious thresholds.

### What v9.0 captures

v9.0 captures the **post-v8.9 cleanup and missing-list closure**. After v8.9
locked, a sweep of seventeen residual missing-list items was completed:
six numbered ISSUE tickets, plus eleven thematic items spanning ML veto
loop wiring, per-trade P&L breakdown, salary/beta/amplifier safety gates,
strategy intelligence (real VIX and event risk), Alpha Options BAG
downgrade prevention, and strategy-maturity gates for low-sample evidence.

Paper Epoch 1 was archived after corrupted SPY paper-fill quarantine.
**Paper Epoch 2** is now active (`epoch_2_reset_attempt_2_2026-05-04`)
on the quarantine-aware HEAD, and runtime publishers are quarantine-aware.

### What v9.0 is **not**

v9.0 is **not** a live-trading approval. The runtime confirms:

- `live_readiness.ready_for_live=false`
- SCR `state=WARMUP`, `sizing_factor=0.10`, only `effective_trades=5`
- `requirements_remaining` includes `epoch_2_warmup_in_progress`,
  `operator_GO`, `14_day_clean_equity_history`,
  `60_day_clean_paper_performance`
- Withdrawal authorizer is in `GROW` phase but blocked from withdrawals
  because SCR is not yet `CONFIDENT`

CHAD is now ready for **clean Paper Epoch 2 evidence collection**.
Live promotion still requires:

1. Clean paper performance on Epoch 2 (no quarantined fills, no untrusted placeholders)
2. `live_readiness.ready_for_live` flips to `true`
3. SCR promotes out of WARMUP toward CONFIDENT on clean evidence
4. Operator `GO`
5. Any remaining governance gates from prior SSOT versions

---

## 2. Current Runtime Truth — 2026-05-04

Source: live runtime files at the time of v9.0 lock. Trust the runtime
files; if a future reader observes drift, the runtime is authoritative.

```
Repo state                     : clean at pre-SSOT lock (git status empty)
Repository HEAD                : 46245a5ec4d2a3ec8d0856719037181108a4cb30
Branch                         : main
Tests                          : 1219 passed / 0 failed, 34 warnings (CHAD_SKIP_IB_CONNECT=1, 56.59s)
Failed chad services           : NO_FAILED_CHAD_SERVICES
Active epoch                   : CHAD_v8.9_Paper_Epoch_2 (started 2026-05-04T00:54:30Z)
Epoch attempt                  : 2 (attempt 2 on quarantine-aware HEAD a9f9732)
Previous epoch archive         : runtime/archive/epoch_1_pre_20260503_attempt2
Quarantine manifest            : runtime/quarantine_manifest_20260503.json
SCR state                      : WARMUP
SCR sizing_factor              : 0.10
SCR effective_trades           : 5 (< 100 required)
SCR win_rate                   : 0.800
SCR sharpe_like                : -0.0999
SCR max_drawdown               : -6.20
SCR total_pnl                  : -0.77 (16 paper_trades, 0 live_trades)
SCR excluded_untrusted         : 6 (placeholder/untrusted fills correctly excluded)
Live readiness                 : false (epoch_2_warmup_in_progress, operator_GO, 14d_equity, 60d_perf)
Stop bus                       : false (cleared 2026-04-22 by smoke_test)
Profit lock                    : NORMAL (sizing_factor=1.0, daily_loss_today=-0.77, stop_new_entries=false)
Reconciliation                 : GREEN, worst_diff=0.0, mismatches=[], 6 documented drifts
Reconciliation drifts          : BAC, CVX, LLY, PEP, QQQ, SPY (broker side has positions CHAD does not — drifts only, not mismatches)
Reconciliation exclusions      : AAPL, NVDA, MSFT (operator-owned pre-existing broker positions)
Regime                         : trending_bull (confidence 0.793) with choppy_overlay ACTIVE (score 0.55, sizing_multiplier 0.25)
Choppy regime                  : ACTIVE since 2026-05-02T13:06:20Z, proxy=SPY, consecutive_choppy_reads=704
Macro state                    : risk_on / composite_risk_label=low_risk (provider=fed_fred_csv, real)
Event risk                     : low (real provider; next event FOMC 2026-05-07T18:00:00Z, hours_until=66.48)
Withdrawal phase               : GROW; authorized_withdrawal_usd=0.0 (blocked: SCR is WARMUP, need CONFIDENT)
Account equity (paper)         : ibkr=$182,763.28 + kraken=$184.58 ≈ $182,947.86 USD
Strategy health (samples)      : alpha=88, alpha_futures=376, alpha_intraday=19, alpha_options=88, beta=4, delta=82, gamma_futures=2, omega_momentum_options=3, omega_vol=3
Operational conclusion         : missing-list closed; clean Paper Epoch 2 evidence collection in progress; trading remains paper-only.
```

### Per-file runtime truth

- **`runtime/scr_state.json`** — `state=WARMUP`, `sizing_factor=0.10`,
  `effective_trades=5`, `excluded_untrusted=6`, ts=`2026-05-04T23:46:11Z`,
  ttl=180s. Reasons: `Warmup: only 5 effective trades (< 100 required)`,
  `win_rate=0.800, sharpe_like=-0.100, max_drawdown=-6.20, total_pnl=-0.77`.
- **`runtime/profit_lock_state.json`** — mode `NORMAL`, `sizing_factor=1.0`,
  `profit_lock_active=false`, `daily_loss_today=-0.77`,
  `daily_loss_limit_hit=false`, `stop_new_entries=false`, account
  equity `$182,947.86`, ts=`2026-05-04T23:46:12Z`.
- **`runtime/reconciliation_state.json`** — `status=GREEN`,
  `broker_source=ibkr:clientId=83`, `chad_open=9`, `chad_strategy_open=17`,
  `broker_positions=15`, `worst_diff=0.0`, `mismatches=[]`. Six **drift**
  entries (BAC, CVX, LLY, PEP, QQQ, SPY) — broker has positions CHAD does
  not; these are tracked drifts, not mismatches. Excluded by policy:
  AAPL, MSFT, NVDA (pre-existing operator broker positions).
  ts=`2026-05-04T23:42:56Z`, ttl=360s.
- **`runtime/stop_bus.json`** — `active=false`, cleared
  `2026-04-22T01:56:50Z` by `smoke_test`.
- **`runtime/regime_state.json`** — `regime=trending_bull`,
  `confidence=0.793`, `choppy_overlay.active=true` (score 0.55,
  sizing_multiplier 0.25, block_trend_following=true), source
  `live_loop.run_once`, ts=`2026-05-04T23:45:33Z`.
- **`runtime/choppy_regime_state.json`** — `choppy_active=true`,
  score 0.55, `consecutive_choppy_reads=704`, entered choppy
  `2026-05-02T13:06:20Z`, proxy `SPY`. ADX=17.56 (weak),
  direction_flips_5d=3 (high), failed_breakouts_10d=0.
- **`runtime/live_readiness.json`** — `ready_for_live=false`,
  `epoch=CHAD_v8.9_Paper_Epoch_2`, `epoch_attempt=2`, started
  `2026-05-04T00:54:30Z`. `requirements_remaining`: `epoch_2_warmup_in_progress`,
  `operator_GO`, `14_day_clean_equity_history`,
  `60_day_clean_paper_performance`.
- **`runtime/portfolio_snapshot.json`** — `ibkr=$182,763.28`,
  `coinbase=$0.00`, `kraken=$184.58`, ts=`2026-05-04T23:42:59Z`.
- **`runtime/epoch_state.json`** — `active_epoch=CHAD_v8.9_Paper_Epoch_2`,
  `paper_only=true`, `ready_for_live=false`, started
  `2026-05-04T00:54:30Z`, previous archive
  `runtime/archive/epoch_1_pre_20260503_attempt2`, quarantine manifest
  `runtime/quarantine_manifest_20260503.json`.
- **`runtime/strategy_health.json`** — present. Sample maturity wide
  range: `alpha_futures=376`, `alpha=88`, `alpha_options=88`, `delta=82`,
  `alpha_intraday=19`, `beta=4`, `omega_momentum_options=3`,
  `omega_vol=3`, `gamma_futures=2`. Low-sample strategies are subject
  to maturity gates (item 17 below).
- **`runtime/winner_scaling.json`** — `min_trades_for_scaling=5`,
  `n_strategies_scaled=5`, `n_strategies_neutral=5`, `median_expectancy=13.13`,
  multipliers floor 0.5, cap 1.5. Strategies below 5 trades are not scaled
  yet (low-sample maturity gate).
- **`runtime/withdrawal_authorization.json`** — `phase=GROW`,
  `current_equity_usd=$183,054.29`, `seed_capital_usd=$50,000`,
  `high_water_mark_usd=$183,874.27`, `drawdown_from_hwm_pct=0.45%`,
  `authorized_withdrawal_usd=0.0`, reason: *GROW phase: equity above
  build threshold but SCR is WARMUP (need CONFIDENT). Reinvesting
  profits.* ts=`2026-05-04T19:00:39Z`.
- **`runtime/macro_state.json`** — provider `fed_fred_csv` (real),
  `risk_label=risk_on`, `composite_risk_label=low_risk`, no risk flags
  raised, US10Y=4.39, US2Y=3.88, curve_slope_10y_2y=0.51, CPI YoY=3.32%,
  unemployment=4.3%, HY spread=2.77%. ts=`2026-05-04T23:41:12Z`, ttl=1800s.
- **`runtime/event_risk.json`** — provider `EconomicCalendarRiskProvider`
  (real), `severity=low`, `elevated_risk=false`. Next event: **FOMC Rate
  Decision 2026-05-07T18:00:00Z** (66.48 hours out). 15 merged windows
  (10 rule-derived + 5 operator calendar). ts=`2026-05-04T23:31:11Z`,
  ttl=1800s.

### Reconciliation nuance (do not overstate)

Reconciliation is `GREEN` because there are zero counted mismatches and
`worst_diff=0.0` against the policy gate, but the runtime explicitly
records six **drift** entries (BAC, CVX, LLY, PEP, QQQ, SPY) where the
broker side holds positions CHAD's `position_guard` does not. These are
documented drifts — they do not count as mismatches under the current
policy and do not flip the status — but they are not "no drift" either.
This nuance is preserved here intentionally and should not be summarized
to "fully clean."

---

## 3. Commit Chain Since v8.9

Range: `0228009..HEAD` (HEAD = `46245a5`). Listed chronologically and
grouped by remediation theme.

### 3.1 — SSOT v8.9 lock (anchor commit, included for reference)
- `c780b01` — Docs: CHAD Unified SSOT v8.9 — audit remediation lock, SCR PAUSED, 1064/0 tests

### 3.2 — Pre-Epoch blockers / paper executor trust
- `0b78345` — Fix: pre-epoch blockers — paper fill price trust guard, asset-scoped choppy gate, winner scaler artifact filter

### 3.3 — Paper Epoch 2 reset / quarantine awareness
- `a9f9732` — Fix: make runtime publishers quarantine-aware for Paper Epoch 2
- `a7977c0` — Fix: add SCR epoch boundary filtering for Paper Epoch 2

### 3.4 — Edge decay / health monitor schema
- `2ae2a9e` — Fix: edge decay alert UnboundLocalError with non-rebinding set cleanup
- `202a428` — Fix: health monitor R09 reads strategy_allocations allocations schema

### 3.5 — Paper executor placeholder price hardening + repo-managed executor
- `d924eea` — Fix: reject placeholder paper fill prices and harden trade closer
- `6523224` — Ops: source-control fixed paper trade executor

### 3.6 — Closed-trade contamination fix
- `80f44da` — Fix: exclude closed trades derived from untrusted fills

### 3.7 — 1m bar freshness fix
- `437e5c4` — Fix: prefer fresh 1m bars for execution intent freshness

### 3.8 — Position guard / atomic writer fixes
- `fdfcef8` — Fix: position guard closes only after confirmed fills and uses atomic writer

### 3.9 — Options chain timeout (ISSUE-50)
- `2be2ac6` — Fix: ISSUE-50 — options-chain refresh env-tunable timeout, typed error, failure artifact

### 3.10 — Execution mode canonical reader / trade closer timer audit
- `4b69834` — Fix: pin canonical execution mode reader and close trade-closer timer drift

### 3.11 — ISSUE-22 formal close
- `7c1b578` — Test: close ISSUE-22 reconciliation exclusion unification

### 3.12 — Alpha Options BAG downgrade prevention
- `f0b8220` — Fix: prevent alpha options BAG fills from downgrading to ETF paper fills

### 3.13 — Strategy intelligence real VIX / event risk
- `cd98150` — Fix: use real VIX and event risk context in strategy intelligence

### 3.14 — Per-trade P&L breakdown
- `c461c5d` — Feature: add canonical per-trade PnL breakdown

### 3.15 — ML veto loop (shadow-first)
- `71ccdb2` — Feature: complete safe shadow-first ML veto loop

### 3.16 — Beta / amplifier gated close
- `91894c8` — Test: lock beta and amplifier injection safety gates

### 3.17 — Salary propose-only gated close
- `955a54a` — Test: lock salary withdrawal propose-only safety gates

### 3.18 — Strategy maturity gates
- `46245a5` — Test: lock strategy maturity gates for low-sample evidence (HEAD)

---

## 4. Missing-List Closure Table

All seventeen missing-list items are closed. Closure type indicates
whether closure is enforced in code, locked by tests, gated by safety
controls, or accepted as a documented maturity gate.

| Item | v9.0 status | Closure type | Commit | Notes |
|------|-------------|--------------|--------|-------|
| ISSUE-29 | Closed | Code fix + atomic writer | `fdfcef8` | Position guard closes only after confirmed fills; writes go through atomic writer; eliminates partial-write windows. |
| ISSUE-50 | Closed | Code fix + typed error | `2be2ac6` | Options-chain refresh has env-tunable timeout, typed error path, failure artifact written for ops visibility. |
| ISSUE-75 | Closed | Code fix | `437e5c4` | Execution intent freshness now prefers fresh 1m bars over stale 1d bars; resolves stale-bar regression. |
| ISSUE-78 | Closed | Code fix | `4b69834` | Canonical execution mode reader pinned; trade-closer timer drift closed; wall-clock status surfaced. |
| ISSUE-22 | Closed | Test lock | `7c1b578` | Reconciliation exclusion unification — operator-owned pre-existing positions (AAPL, MSFT, NVDA) handled by single exclusion path; behavior locked by tests. |
| ISSUE-58 | Closed | Code fix + lock | `d924eea`, `80f44da` | $100 placeholder paper fill prices rejected by paper executor; closed trades derived from untrusted fills excluded from SCR; `excluded_untrusted=6` in current SCR state confirms enforcement. |
| ML veto loop | Closed (shadow-first) | Feature, enforcement disabled by default | `71ccdb2` | Shadow-first ML veto loop wired end-to-end. Vetoes are computed and logged; enforcement is **disabled by default**. Promotion to canary requires shadow soak + retraining. |
| Per-trade P&L breakdown | Closed | Feature + schema | `c461c5d` | Canonical `pnl_breakdown.v1` per-trade structure added; profit lock continues to read net P&L after the breakdown lands. |
| Salary automation | Closed (propose-only) | Safety gate locked by tests | `955a54a` | Salary withdrawal automation is **propose-only**; tests lock the safety gates. Requires SCR CONFIDENT and operator authorization to act. Currently `authorized_withdrawal_usd=0.0`. |
| Amplifier injection | Closed (gated) | Safety gate locked by tests | `91894c8` | Amplifier injection paths are gated; tests lock that the gates cannot be bypassed without explicit safety conditions. |
| Beta injection activation | Closed (gated) | Safety gate locked by tests | `91894c8` | Beta injection activation is gated; tests lock the safety gates. Beta strategy currently has only 4 samples — under maturity gate. |
| Alpha Options stuck review | Closed (downgrade prevented) | Code fix | `f0b8220` | Alpha Options BAG fills can no longer silently downgrade to ETF paper fills. Real BAG/COMBO execution architecture remains future work (see §7). |
| Omega Vol low sample | Closed (maturity gate) | Documented gate | `46245a5` | Omega Vol has only 3 samples; subject to formal low-sample strategy maturity gate. Health score 0.10 reflects regime non-alignment + low evidence. |
| Macro intelligence placeholder | Closed | Real provider wired | `cd98150` | Strategy intelligence now consumes real VIX from `price_cache` and real macro context (`macro_state.json` provider=`fed_fred_csv`). Placeholder removed. |
| Event calendar placeholder | Closed | Real provider wired | `cd98150` | Strategy intelligence now consumes real structured `event_risk.json` from `EconomicCalendarRiskProvider` (operator calendar + rule-derived events). Placeholder removed. |
| Winner scaling/sample maturity | Closed (maturity gate) | Documented gate + tests | `46245a5` | Winner scaling enforces `min_trades_for_scaling=5`; strategies below threshold remain at neutral 1.0×. Maturity gates locked by tests. |
| Strategy health sample maturity | Closed (maturity gate) | Documented gate + tests | `46245a5` | Low-sample strategies (gamma_futures=2, omega_vol=3, omega_momentum_options=3, beta=4) are not treated as evidence-mature. Tests lock that low-sample evidence cannot promote allocations. |

**Total closed: 17 of 17.**

---

## 5. Safety Contracts Now Locked

The following invariants are enforced in code and locked by tests in
the post-v8.9 cleanup. None of these may be loosened without an
explicit governance change.

1. **Position guard closes only after confirmed fills.** The position
   guard does not record a close on the basis of an order being
   submitted or staged — it requires a confirmed broker fill. (`fdfcef8`)
2. **Position guard writes go through the atomic writer.** No partial-
   write windows on `position_guard.json`; readers cannot observe a
   half-updated state. (`fdfcef8`)
3. **Options-chain refresh has a bounded timeout.** Env-tunable timeout,
   typed error, failure artifact written for ops visibility — refresh
   cannot hang indefinitely. (`2be2ac6`)
4. **Canonical execution mode reader is pinned.** All execution paths
   read execution mode through one canonical reader; ad-hoc reads are
   removed. (`4b69834`)
5. **Trade closer timer wall-clock status is surfaced.** Trade-closer
   timer drift is closed and the wall-clock status is observable. (`4b69834`)
6. **No trusted $100 placeholder fills.** Paper executor rejects $100
   placeholder paper fill prices — they are no longer treated as
   trustworthy fill evidence. (`d924eea`)
7. **Closed trades from untrusted fills are excluded.** Trades closed
   on the basis of untrusted fills are excluded from SCR statistics.
   Current SCR shows `excluded_untrusted=6` confirming enforcement is
   active. (`80f44da`)
8. **1m bar freshness preferred for execution intent.** Execution
   freshness checks prefer fresh 1m bars over stale daily bars; daily
   data is not used as a proxy for current intent. (`437e5c4`)
9. **Alpha Options BAG/COMBO cannot silently downgrade to ETF paper
   fills.** A BAG that cannot be filled as a true combo does not get
   silently rerouted to an ETF paper fill — failures are visible. (`f0b8220`)
10. **Strategy intelligence uses real VIX and structured event risk.**
    Strategy intelligence reads real VIX from `price_cache` and real
    structured `event_risk` (operator calendar + rule-derived events).
    No placeholders. (`cd98150`)
11. **Per-trade `pnl_breakdown.v1` exists.** Canonical per-trade P&L
    breakdown is recorded with each closed trade; profit lock continues
    to read net P&L correctly after the breakdown is added. (`c461c5d`)
12. **ML veto is shadow-first; enforcement disabled by default.**
    Vetoes are computed, logged, and observable, but they do **not**
    block trades by default. Promotion to enforcement requires shadow
    soak and retraining. (`71ccdb2`)
13. **Beta / amplifier / salary are gated or propose-only.** Beta
    injection and amplifier injection are gated; salary withdrawal is
    propose-only. None of these can act without explicit safety
    conditions met. (`91894c8`, `955a54a`)
14. **Low-sample strategy maturity gates prevent overreaction.** Winner
    scaling enforces a minimum trade count; strategy health treats
    low-sample strategies as evidence-immature. Allocations cannot be
    promoted on low-sample evidence. (`46245a5`)
15. **SCR epoch boundary filtering.** SCR statistics filter on the
    active epoch boundary — Epoch 1 contaminated history does not
    leak into Epoch 2 evaluation. (`a7977c0`)
16. **Runtime publishers are quarantine-aware.** Publishers do not
    re-emit quarantined fills as fresh runtime state. (`a9f9732`)
17. **Edge decay alert correctness.** Non-rebinding set cleanup
    eliminated the `UnboundLocalError` in the edge decay alert path. (`2ae2a9e`)
18. **Health monitor R09 schema-correct.** Reads `strategy_allocations`
    via the `allocations` schema; no silent schema mismatches. (`202a428`)

---

## 6. Remaining Non-Live Blockers / Paper-Soak Requirements

These are **not bugs**. They are governance gates and evidence-collection
requirements that must be satisfied on clean Epoch 2 data before live
promotion can be considered.

1. **`live_readiness.ready_for_live=false`.** Authoritative gate;
   currently `false`.
2. **SCR is WARMUP, not CONFIDENT.** Current state `WARMUP` with
   `sizing_factor=0.10`. Promotion path requires sufficient effective
   trades and clean performance metrics.
3. **`epoch_2_warmup_in_progress`.** Listed in
   `requirements_remaining`. Epoch 2 must mature.
4. **14-day clean equity history.** Listed in `requirements_remaining`.
5. **60-day clean paper performance.** Listed in `requirements_remaining`.
6. **`operator_GO` requirement.** Listed in `requirements_remaining`.
7. **Sufficient clean samples for strategy health / winner scaling.**
   Current low-sample strategies (`gamma_futures=2`, `omega_vol=3`,
   `omega_momentum_options=3`, `beta=4`) are below scaling threshold;
   `alpha_intraday=19` is also light. Allocations from these
   strategies cannot be promoted on this evidence.
8. **ML veto shadow soak and retraining before any canary promotion.**
   The ML veto loop runs in shadow only. Promoting to enforcement
   requires a shadow soak and a retraining cycle on clean data.
9. **Alpha Options real BAG/COMBO execution architecture.** Downgrade
   to ETF paper fills is now blocked, but a real BAG/COMBO execution
   path remains future work.
10. **Reconciliation drift nuance.** Six drift entries (BAC, CVX, LLY,
    PEP, QQQ, SPY) are tracked drifts where the broker holds positions
    CHAD does not. Status is GREEN under current policy, but these
    drifts must be reviewed and reconciled (or formally excluded) before
    live promotion.
11. **Withdrawal authorization is `GROW` but blocked.** Authorizer
    explicitly requires SCR `CONFIDENT`; `authorized_withdrawal_usd=0.0`
    until that promotion happens.
12. **Delta edge-decay halt / boost contradiction requires paper-soak audit before live.**
    The health monitor has flagged an active edge-decay halt on the `delta`
    strategy at the same time that winner scaling reports a boosted
    multiplier / boost context for `delta`. This contradiction must be
    audited and resolved on paper before live promotion. See §6.1 below
    for full detail.

### 6.1 — Delta Edge-Decay Halt vs Winner-Scaling Boost — Active Paper-Soak Warning

**Observed contradiction.** The health monitor has warned that the `delta`
strategy currently has an **active edge-decay halt** while winner scaling
simultaneously reports a **boosted multiplier / boost context** for the
same `delta` strategy. Two runtime signals that should agree on whether
`delta` is allowed to scale up are pointing in opposite directions.

**Why this is not an immediate live-risk.** CHAD is in **Paper Epoch 2**
with **SCR `WARMUP`**, **`sizing_factor=0.10`**, and
**`live_readiness.ready_for_live=false`**. No live capital is exposed to
the contradiction; any boosted allocation that might flow from this
contradiction is bounded by the WARMUP sizing floor and the paper-only
posture. The contradiction therefore lands as a **paper-soak warning**,
not a live blocker — but it **must be resolved before live promotion**.

**Do not manually clear the halt.** The halt is a safety signal. Manually
clearing it without an audit would mask whatever caused the halt to be
emitted in the first place and would defeat the rule-engine / edge-decay
recovery path. The halt must only ever be cleared through a verified
recovery path, never by hand.

**Required future resolution path** (must be completed before live):

1. **Audit whether the delta halt is stale or valid.** Inspect the
   edge-decay rule path and the evidence that produced the halt. Decide
   whether the halt reflects a real, current edge-decay condition on
   `delta` or whether it is a stale signal that has not been cleared by
   the recovery path.
2. **If the halt is valid**, the winner-scaling / allocator path must be
   updated so it **suppresses boost for halted strategies**. A halted
   strategy must not receive an aggressive boost multiplier — the boost
   context should be neutralized (or floored at 1.0×) for any strategy
   under an active halt.
3. **If the halt is stale**, it must be cleared **only through a verified
   rule-engine / edge-decay recovery path** — never by manual edit of
   runtime state. The recovery path itself is the authoritative clearer.
4. **Add a regression test** locking the invariant that **a halted
   strategy cannot receive an aggressive winner-scaling boost**. The
   test must fail if a strategy with an active edge-decay halt is
   simultaneously assigned a boosted multiplier above the neutral floor.

**Status.** This contradiction is recorded as an **active paper-soak
warning**, not as a missing-list item and not as a live blocker. It is
acceptable to continue Paper Epoch 2 evidence collection while the
contradiction is being audited, because the WARMUP / `sizing_factor=0.10`
posture and `ready_for_live=false` together cap the practical exposure.
Live promotion **must not** proceed while this contradiction is unresolved.

---

## 7. Exterminator Read-Only Sentinel — Next Build

The next build directly after v9.0 is the **Exterminator Read-Only
Sentinel (Stage 1)**. This is intentionally narrow.

### Stage 1 scope (read-only, this build)

- **Read-only scanner.** No mutation, no patching, no restarts.
- **Detects**:
  - Stale feeds (TTL exceeded on runtime files)
  - Placeholder fill prices (e.g., $100 trust violations)
  - Untrusted fills (closed trades on untrusted source)
  - Reconciliation drift (drifts/mismatches against broker)
  - Failed services (`systemctl --failed | grep chad`)
  - Dirty git (uncommitted state at sentinel run time)
  - Runtime schema breaks (schema_version mismatches, missing required keys)
  - Shadow ML anomalies (veto-rate anomalies, model-version drift)
- **Writes report only.** A single report artifact — no runtime
  mutation, no service restart, no config change.

### Stage 2 (future, NOT this build)

Stage 2 may propose patches — but only after Stage 1 has accumulated
enough trust to justify a propose-only patch path. Stage 2 is **not**
part of v9.0 and is not authorized to run.

### Hard constraints

- The sentinel **must not** modify any file under `runtime/`,
  `data/`, or any `runtime_FREEZE_*` / `data_FREEZE_*` archive.
- The sentinel **must not** restart any service.
- The sentinel **must not** stage or commit any change.
- The sentinel **must not** call any broker.

---

## 8. Verification Appendix

### 8.1 — Git state at v9.0 lock

```
$ git status --short
(empty — clean)

$ git rev-parse HEAD
46245a5ec4d2a3ec8d0856719037181108a4cb30

$ git rev-parse --abbrev-ref HEAD
main

$ git log 0228009..HEAD --oneline --reverse
c780b01 Docs: CHAD Unified SSOT v8.9 — audit remediation lock, SCR PAUSED, 1064/0 tests
0b78345 Fix: pre-epoch blockers — paper fill price trust guard, asset-scoped choppy gate, winner scaler artifact filter
a9f9732 Fix: make runtime publishers quarantine-aware for Paper Epoch 2
a7977c0 Fix: add SCR epoch boundary filtering for Paper Epoch 2
2ae2a9e Fix: edge decay alert UnboundLocalError with non-rebinding set cleanup
202a428 Fix: health monitor R09 reads strategy_allocations allocations schema
d924eea Fix: reject placeholder paper fill prices and harden trade closer
6523224 Ops: source-control fixed paper trade executor
80f44da Fix: exclude closed trades derived from untrusted fills
437e5c4 Fix: prefer fresh 1m bars for execution intent freshness
fdfcef8 Fix: position guard closes only after confirmed fills and uses atomic writer
2be2ac6 Fix: ISSUE-50 — options-chain refresh env-tunable timeout, typed error, failure artifact
4b69834 Fix: pin canonical execution mode reader and close trade-closer timer drift
7c1b578 Test: close ISSUE-22 reconciliation exclusion unification
f0b8220 Fix: prevent alpha options BAG fills from downgrading to ETF paper fills
cd98150 Fix: use real VIX and event risk context in strategy intelligence
c461c5d Feature: add canonical per-trade PnL breakdown
71ccdb2 Feature: complete safe shadow-first ML veto loop
91894c8 Test: lock beta and amplifier injection safety gates
955a54a Test: lock salary withdrawal propose-only safety gates
46245a5 Test: lock strategy maturity gates for low-sample evidence
```

### 8.2 — Test result

```
$ CHAD_SKIP_IB_CONNECT=1 PYTHONPATH=/home/ubuntu/chad_finale \
    /home/ubuntu/chad_finale/venv/bin/python3 -m pytest chad/tests/ -q
...
1219 passed, 34 warnings in 56.59s
```

The 34 warnings are `DeprecationWarning` for `datetime.datetime.utcnow()`
in legacy paths; behavior is correct, no test fails.

### 8.3 — Service health

```
$ systemctl --failed --no-pager | grep chad || echo "NO_FAILED_CHAD_SERVICES"
NO_FAILED_CHAD_SERVICES
```

### 8.4 — `runtime/scr_state.json` (head)

```json
{
  "paper_only": true,
  "reasons": [
    "Warmup: only 5 effective trades (< 100 required).",
    "Current win_rate=0.800, sharpe_like=-0.100, max_drawdown=-6.20, total_pnl=-0.77 (total_trades=16, effective_trades=5)"
  ],
  "schema_version": "scr_state.v1",
  "sizing_factor": 0.1,
  "source": { "url": "http://127.0.0.1:9618/shadow" },
  "state": "WARMUP",
  "stats": {
    "effective_trades": 5,
    "excluded_manual": 0,
    "excluded_nonfinite": 0,
    "excluded_untrusted": 6,
    "live_trades": 0,
    "max_drawdown": -6.2,
    "paper_trades": 16,
    "paused_recovery_ticks": 0,
    "sharpe_like": -0.09990450532179314,
    "total_pnl": -0.77,
    "total_trades": 16,
    "win_rate": 0.8
  },
  "ts_utc": "2026-05-04T23:46:11Z",
  "ttl_seconds": 180
}
```

### 8.5 — `runtime/live_readiness.json`

```json
{
  "epoch_metadata": {
    "epoch": "CHAD_v8.9_Paper_Epoch_2",
    "epoch_started_at_utc": "2026-05-04T00:54:30Z",
    "epoch_attempt": 2,
    "previous_epoch_archive": "/home/ubuntu/chad_finale/runtime/archive/epoch_1_pre_20260503_attempt2",
    "prior_attempt_archive": "/home/ubuntu/chad_finale/runtime/archive/epoch_1_pre_20260503",
    "reset_reason": "fresh paper evaluation after v8.9 audit remediation lock and corrupted SPY paper-fill quarantine; attempt 2 on quarantine-aware HEAD a9f9732 (publishers now quarantine-aware)"
  },
  "evaluated_by": "epoch_2_reset_attempt_2_2026-05-04",
  "last_evaluated_utc": "2026-05-04T00:54:30Z",
  "next_evaluation_cadence": "weekly",
  "ready_for_live": false,
  "requirements_remaining": [
    "epoch_2_warmup_in_progress",
    "operator_GO",
    "14_day_clean_equity_history",
    "60_day_clean_paper_performance"
  ],
  "schema_version": "live_readiness_state.v1",
  "ts_utc": "2026-05-04T00:54:30Z"
}
```

### 8.6 — `runtime/epoch_state.json`

```json
{
  "schema_version": "epoch_state.v1",
  "active_epoch": "CHAD_v8.9_Paper_Epoch_2",
  "epoch_started_at_utc": "2026-05-04T00:54:30Z",
  "paper_only": true,
  "ready_for_live": false,
  "previous_epoch_archive": "/home/ubuntu/chad_finale/runtime/archive/epoch_1_pre_20260503_attempt2",
  "quarantine_manifest": "/home/ubuntu/chad_finale/runtime/quarantine_manifest_20260503.json"
}
```

---

## 9. Appendix — v8.9 to v9.0 Delta

| Area | v8.9 (2026-05-03) | v9.0 (2026-05-04) |
|------|-------------------|-------------------|
| Repo HEAD | `0228009` | `46245a5` |
| Branch | `main` | `main` |
| Tests | 1064 passed / 0 failed | **1219 passed / 0 failed**, 34 warnings |
| SCR state | `PAUSED` | `WARMUP` (Paper Epoch 2) |
| SCR sizing_factor | 0.0 | 0.10 |
| SCR effective_trades | 321 (Epoch 1, contaminated) | 5 (Epoch 2, clean) |
| SCR sharpe_like | -1.287 | -0.100 |
| SCR max_drawdown | -10314.12 | -6.20 |
| SCR total_pnl | -20855.54 | -0.77 |
| SCR untrusted-fill exclusion | not enforced / unknown | **enforced** (`excluded_untrusted=6`) |
| Active epoch | (not yet declared in v8.9 SSOT — Epoch 1 implicit) | `CHAD_v8.9_Paper_Epoch_2` (attempt 2) |
| Placeholder $100 fills | unresolved / not blocked | **blocked / rejected / excluded** |
| Bar freshness | 1d stale issue (ISSUE-75) | **1m freshness** preferred |
| Position guard close semantics | (pre-fix) | **confirmed fills only**, atomic writer |
| Options-chain refresh | unbounded | **env-tunable timeout, typed error** |
| Execution mode reader | ad-hoc | **canonical reader pinned** |
| Trade closer timer | drift open | **wall-clock status surfaced** |
| Reconciliation exclusion (ISSUE-22) | open | **closed (test-locked)** |
| Alpha Options BAG | could silently downgrade | **downgrade blocked** |
| Strategy intelligence — VIX | placeholder | **real VIX from price_cache** |
| Strategy intelligence — event risk | placeholder | **real `event_risk.json`, EconomicCalendarRiskProvider** |
| Per-trade P&L breakdown | absent | **`pnl_breakdown.v1`** |
| ML veto loop | not wired | **wired (shadow-first, enforcement off)** |
| Salary / Beta / Amplifier | deferred | **formally gated / propose-only**, locked by tests |
| Low-sample strategy treatment | implicit | **formal maturity gates**, locked by tests |
| Live readiness | `false` | `false` (unchanged) |
| Live readiness requirements_remaining | operator_GO, 14d_equity, 60d_perf | **+ `epoch_2_warmup_in_progress`** |
| Withdrawal authorization | (not surfaced in v8.9 lock) | `GROW` phase, blocked (need SCR CONFIDENT) |
| Failed CHAD services | none | none |
| Lock type | external audit remediation lock | **post-v8.9 missing-list closure lock** |
| Live promotion authorized | **NO** | **NO** |

---

**Document end. v9.0 is a paper-soak / clean-evidence-collection / post-remediation lock. Live promotion is not authorized by this document.**
