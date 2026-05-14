# CHAD Unified SSOT v9.1 — Tier-aware micro-futures, setup-family expectancy, stop-distance risk enforcement (architecture baseline)

**Version:** 9.1
**Date:** 2026-05-13
**Status:** Active — Paper Epoch 2 / Elite Signal Layer Phase A Baseline
**Supersedes:** docs/CHAD_UNIFIED_SSOT_v9.0_2026-05-04.md (commit 938ec2c)
**Repository HEAD:** 67ba391be234987c9fd1b41f3f4dbd6d8c5344e8
**Branch:** main
**Test status:** 1652 passed / 0 failed (reported at HEAD by commit 67ba391)
**Live readiness:** ready_for_live=false (runtime/live_readiness.json, ts_utc=2026-05-13T22:37:04Z)
**SCR posture:** CONFIDENT (sizing_factor=1.0, paper_only=false; ts_utc=2026-05-13T22:39:59Z)
**Lock type:** Architecture baseline lock — **NOT** a live-trading approval

---

## 0. Preamble

### 0.1 Document Metadata

| Field | Value |
|---|---|
| Version | 9.1 |
| Date | 2026-05-13 |
| Predecessor | docs/CHAD_UNIFIED_SSOT_v9.0_2026-05-04.md (commit 938ec2c) |
| Repository HEAD | 67ba391be234987c9fd1b41f3f4dbd6d8c5344e8 |
| Branch | main |
| Tests | 1652 passed / 0 failed |
| Live readiness | ready_for_live=false |
| SCR posture | CONFIDENT, sizing_factor=1.0 |
| Lock type | Architecture baseline lock (NOT live-trading approval) |

### 0.2 What v9.0 was

v9.0 captured CHAD's Paper Epoch 2 clean runtime lock following the v8 hardening cycle (P0-1..P0-4, P1-1..P1-3, P2-1..P2-3) and the GAP-1..GAP-25 governance audit. It documented 16 production strategies, the dynamic risk allocator overlay chain (tier × winner × regime × choppy), the SCR shadow ledger, profit lock, reconciliation, and the 50/30/20 ALPHA/BETA/ADAPTIVE chassis. v9.0 confirmed paper-only operation with live activation gated by `live_readiness.json`.

### 0.3 What v9.1 captures

v9.1 captures the **91 commits** merged since v9.0 (commit 938ec2c → 67ba391). The headline additions are:

1. **Tier architecture v2** — schema_version `tier_state.v2` and `tiers.v9_1`. Four tiers (MICRO / STARTER / PRO_GROWTH / SCALE) replace the prior name/range structure, each with an explicit `TierRiskProfile` (max_contracts_per_trade, max_risk_per_trade_usd, max_daily_loss_usd, max_weekly_loss_usd, max_trades_per_day, primary_session_only, flatten_before_eod, flatten_eod_minutes_before_close, stop_width_gate_enabled). Legacy "PRO" tier is migrated to "SCALE" transparently on read. Demotion has a mid-session deferral rule (apply at next market open).
2. **alpha_intraday_micro** — 18th strategy (`StrategyName.ALPHA_INTRADAY_MICRO = "alpha_intraday_micro"`). Tier-aware MES/MNQ intraday brain with **5 setup families**: ORB, VWAP_RECLAIM, VWAP_REJECTION, PULLBACK_CONTINUATION, SWEEP_REVERSAL. All session-window arithmetic computed in America/New_York. Registered in the canonical registry (`chad/strategies/__init__.py`).
3. **TierRiskEnforcer** — runtime stop-distance risk enforcement infrastructure (`chad/risk/tier_risk_enforcer.py`). Reads `data/trades/trade_history_YYYYMMDD.ndjson` and writes `runtime/tier_enforcement_state.json`. Provides `check()` (daily/weekly loss + trade-count + max-contracts) and `validate_stop_width()` (proposed stop fits per-trade risk budget). Five deterministic skip reasons: SKIP_INVALID_CONTRACT_COUNT / SKIP_MAX_CONTRACTS_EXCEEDED / SKIP_MAX_TRADES_REACHED / SKIP_DAILY_LOSS_LIMIT / SKIP_WEEKLY_LOSS_LIMIT.
4. **TierInstrumentGate** — pipeline-level allowlist gate (`chad/execution/tier_instrument_gate.py`). Fail-open block reason `INSTRUMENT_NOT_IN_TIER_ALLOWLIST`; writes `runtime/tier_instrument_gate_state.json` (schema_version `tier_instrument_gate.v1`).
5. **setup_family_expectancy_updater** — analytics writer (`chad/analytics/setup_family_expectancy_updater.py`). Computes per-setup-family expectancy (trades, wins, win_rate, avg_r, expectancy_r, skip_count_stop_too_wide) for alpha_intraday_micro from the trade ledger. Writes `runtime/setup_family_expectancy.json` (schema_version `setup_family_expectancy.v2`).
6. **EOD flatten service** — `ops/micro_eod_flatten.py`, wired through `deploy/chad-micro-eod-flatten.{service,timer}` (fires Mon–Fri 15:30 America/New_York).
7. **Setup expectancy timer** — `deploy/chad-setup-expectancy.{service,timer}` (fires Mon–Fri 21:00 America/New_York).
8. **Health rules R15 (tier daily loss budget) and R16 (setup family skip rate)** added to `chad/ops/health_monitor_rules.py`.
9. **Gap-4 meta forwarding** — alpha_options option meta preserved through the execution pipeline (commit a949756).
10. **ib_async migration complete on primary execution path** — `chad/core/live_loop.py`, `chad/execution/ibkr_adapter.py`, and `chad/execution/ibkr_trade_router.py` migrated from ib_insync to ib_async 2.1.0 (commit 132f4e9 and Phase 1B sub-commits 8294369 / 33a9628 / 7e5dec2 / fbf627e / b531270). Two paper-runner files remain on ib_insync — see §14.

### 0.4 What v9.1 is NOT

v9.1 is **NOT** a live-trading approval. `live_readiness.json` shows `ready_for_live: false` (ts_utc 2026-05-13T22:37:04Z). The pre-live operator tasks listed in CLAUDE.md (OS reboot if kernel update lands, disk cleanup below 75 %, IB Gateway latency resolution, full_cycle_preview clean, MES short review) still gate any mode flip.

### 0.5 Server / Repo / Mode

| Field | Value |
|---|---|
| Server | EC2 Linux 6.17.0-1009-aws (paper) |
| Repo root | /home/ubuntu/chad_finale |
| Python | venv at /home/ubuntu/chad_finale/venv (python3) |
| Execution mode | CHAD_EXECUTION_MODE=paper |
| Broker | Interactive Brokers Paper (ib_async 2.1.0); Kraken (crypto) |
| Tier | SCALE (current_equity_usd $162,445.36, min $160,000) |
| Business phase | GROW (current $162,224.07 vs HWM $183,874.27) |

---

## 1. Mission & Architecture

### 1.1 Mission

> CHAD (Compounding Hedge-Fund Algorithmic Desk) is a single-operator automated trading system that compounds a six-figure equity book across 17 production strategy heads. The strategies emit deterministic TradeSignals; a multi-layer gate/overlay stack converts them into broker-ready orders under hard risk and governance constraints. CHAD trades paper today and is gated for live promotion only when SCR confidence, equity history, and operator GO converge.

### 1.2 Architecture (ASCII diagram, v9.1)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  MARKET DATA / IBKR                     │
                    │  (ib_async 2.1.0 primary path · Kraken for crypto)      │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────┐
                          │            ContextBuilder                │
                          │  bars/bars_1m, prices, portfolio, regime │
                          │  tier_profile, tier_name (NEW v9.1)      │
                          └──────────────────────┬───────────────────┘
                                                 │
                                                 ▼
                           ┌─────────────────────────────────────────┐
                           │     STRATEGY REGISTRY (17 active)       │
                           │  alpha / alpha_intraday / alpha_micro * │
                           │  alpha_futures / alpha_options / crypto │
                           │  beta / beta_trend                      │
                           │  gamma / gamma_futures / gamma_reversion│
                           │  delta / delta_pairs                    │
                           │  omega / omega_vol / omega_macro        │
                           │  omega_momentum_options                 │
                           │  (alpha_forex DEFERRED — see §3)        │
                           └──────────────────────┬──────────────────┘
                                                  │ TradeSignal[]
                                                  ▼
                ┌──────────────────────────────────────────────────────────┐
                │  StrategyRouter — single-winner OR always-active mode    │
                │                                                          │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │     Edge-decay halt filter (halted → dropped)      │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      TierInstrumentGate (NEW v9.1)                 │  │
                │  │      INSTRUMENT_NOT_IN_TIER_ALLOWLIST              │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      NetExposureGate                               │  │
                │  │      ALLOW/MERGE/REDUCE/CLOSE_ONLY/FLIP/BLOCK      │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      StrategyThrottleGate                          │  │
                │  │      ALLOW/THROTTLE/UPSHIFT/PAUSE/HALT_DEFER       │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      Routing gates: data freshness / stale_intent  │  │
                │  │      / too_late_to_chase / net_ev                  │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      ML veto (shadow ON; enforce gated)            │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      TierRiskEnforcer (NEW v9.1)                   │  │
                │  │      validate_stop_width() · check(daily/weekly/   │  │
                │  │      max_contracts/max_trades)                     │  │
                │  └────────────────────────────────────────────────────┘  │
                │  ┌────────────────────────────────────────────────────┐  │
                │  │      Position guard / flip executor (BG11)         │  │
                │  └────────────────────────────────────────────────────┘  │
                └──────────────────────────┬───────────────────────────────┘
                                           │ PlannedOrder[] / IBKR intents
                                           ▼
                          ┌─────────────────────────────────────┐
                          │   IBKR adapter (ib_async 2.1.0)     │
                          │   Paper exec evidence writer        │
                          │   Trade closer · reconciliation     │
                          │   setup_family_expectancy_updater * │
                          │   (* NEW v9.1, EOD flatten + 21:00) │
                          └─────────────────────────────────────┘
```

### 1.3 The 17 Strategy Heads

`StrategyName` enum in `chad/types.py` exposes 18 values but **ALPHA_FOREX is deferred** (commented out in `chad/strategies/__init__.py:_build_registry`). Active heads:

1. ALPHA (`alpha`) — phase-4 regime/momentum equity & ETF
2. BETA (`beta`) — institutional-consensus 13F compounder
3. BETA_TREND (`beta_trend`) — legend-weighted multi-week trend
4. GAMMA (`gamma`) — range/trend dual-mode ETF/equity
5. OMEGA (`omega`) — SH/PSQ hedging engine
6. DELTA (`delta`) — high-conviction convexity hunter
7. ALPHA_CRYPTO (`alpha_crypto`) — BTC/ETH/SOL momentum
8. ALPHA_INTRADAY (`alpha_intraday`) — Delta high-convexity day trading
9. ALPHA_FUTURES (`alpha_futures`) — MES/MNQ/MGC momentum
10. GAMMA_FUTURES (`gamma_futures`) — MCL/MYM/M2K mean reversion
11. OMEGA_MACRO (`omega_macro`) — ZN/ZB/M6E macro regime
12. GAMMA_REVERSION (`gamma_reversion`) — SPY/QQQ/GLD/TLT confluence reversion
13. ALPHA_OPTIONS (`alpha_options`) — SPY vertical-spread combos (BAG)
14. OMEGA_MOMENTUM_OPTIONS (`omega_momentum_options`) — single-leg intraday options
15. OMEGA_VOL (`omega_vol`) — SVXY/UVXY regime-driven vol
16. DELTA_PAIRS (`delta_pairs`) — SPY/QQQ/IWM market-neutral pairs
17. **ALPHA_INTRADAY_MICRO** (`alpha_intraday_micro`) — **NEW v9.1**, tier-aware MES/MNQ intraday with 5 setup families

`StrategyName.ALPHA_FOREX` is enumerated but its registry entry is commented out — universe (EUR-USD, GBP-USD, USD-CAD, USD-JPY) is not mapped to active bar/price context (see `chad/strategies/__init__.py` comment lines 192-201).

---

## 2. Current Runtime Truth — 2026-05-13

All values read from `/home/ubuntu/chad_finale/runtime/` files at the time of writing.

### 2.1 Repo & Tests

| Field | Value |
|---|---|
| HEAD | 67ba391 (Build: v9.1 workstream …, 2026-05-13T20:16:04Z) |
| Branch | main |
| Working tree | clean (git status --short empty) |
| Tests | 1652 passed / 0 failed (as reported at HEAD) |

### 2.2 SCR (`runtime/scr_state.json`)

| Field | Value |
|---|---|
| state | CONFIDENT |
| sizing_factor | 1.0 |
| paper_only | false |
| effective_trades | 194 |
| paper_trades | 3873 |
| live_trades | 0 |
| total_trades | 3873 |
| excluded_manual | 5 |
| excluded_nonfinite | 0 |
| excluded_untrusted | 321 |
| win_rate | 0.7629 |
| sharpe_like | +5.7519 |
| max_drawdown | -290.0 |
| total_pnl | +$9,934.165 |
| ts_utc | 2026-05-13T22:39:59Z |
| ttl_seconds | 180 |
| reasons | "CONFIDENT: win_rate, sharpe, and drawdown all within confident band." |

### 2.3 Live Readiness (`runtime/live_readiness.json`)

| Field | Value |
|---|---|
| ready_for_live | **false** |
| latest_report_path | /home/ubuntu/chad_finale/reports/live_readiness/LIVE_READINESS_20260513T223704Z.json |
| latest_report_sha256 | sha256:220d7200e055bb53d2d23d38209b7ec164b68725ebdbd3c8f69906c721fd0b4b |
| ts_utc | 2026-05-13T22:37:04Z |
| ttl_seconds | 604800 (1 week) |

### 2.4 Regime (`runtime/regime_state.json`)

| Field | Value |
|---|---|
| regime | trending_bull |
| previous_regime | trending_bull |
| confidence | 0.7366 |
| ok | true |
| inputs_used | realized_vol_percentile, adx, trend_slope, market_breadth |
| source | live_loop.run_once |
| ts_utc | 2026-05-13T22:39:14Z |

### 2.5 Choppy Regime (`runtime/choppy_regime_state.json`)

| Field | Value |
|---|---|
| choppy_active | false |
| choppy_score | 0.45 |
| consecutive_choppy_reads | 0 |
| consecutive_clean_reads | 958 |
| entered_choppy_at_utc | null |
| proxy_symbol | SPY |
| adx | 29.27 |
| direction_flips_5d | 3 |
| failed_breakouts_10d | 2 |
| trend_followthrough_rate | 0.625 |
| thresholds | enter=0.55, exit=0.35, enter_reads=3, exit_reads=4, min_hold=60min |

### 2.6 Profit Lock (`runtime/profit_lock_state.json`)

| Field | Value |
|---|---|
| mode | NORMAL |
| profit_lock_active | false |
| stop_new_entries | false |
| sizing_factor | 1.0 |
| account_equity | $162,358.55 |
| daily_loss_today | $0.00 |
| daily_loss_limit_dollars | $4,870.76 (3.0%) |
| daily_loss_limit_hit | false |
| trade_count | 996 |
| thresholds_pct | warn=1.5, lock1=3.0, lock2=5.0, lock3=8.0, hard_stop=10.0 |
| factors | lock1=0.50, lock2=0.25, lock3=0.10, hard_stop=0.00 |
| pnl_sources | trade_history_20260513.ndjson, FILLS_20260513.ndjson |
| ts_utc | 2026-05-13T22:40:01Z |

### 2.7 Stop Bus (`runtime/stop_bus.json`)

| Field | Value |
|---|---|
| active | false |
| cleared_at | 2026-04-22T01:56:50Z |
| cleared_by | smoke_test |
| triggered_at | "" |
| triggered_by | "" |

### 2.8 Reconciliation (`runtime/reconciliation_state.json`)

| Field | Value |
|---|---|
| status | GREEN |
| broker_source | ibkr:clientId=83 |
| chad_state_source | position_guard.json |
| chad_open positions | 15 |
| chad_strategy_open | 14 |
| broker_positions | 15 |
| worst_diff | 0.0 |
| mismatches | [] |
| drifts | [] |
| excluded_symbols | AAPL, BAC, CVX, NVDA, PEP, QQQ |
| futures_excluded_symbols | M2K, MES |
| exclusion_policy (full set) | AAPL, BAC, CVX, LLY, MSFT, NVDA, PEP, QQQ, SPY (all pre-existing broker positions, owner=operator) |
| ts_utc | 2026-05-13T22:38:59Z |

### 2.9 Portfolio Snapshot (`runtime/portfolio_snapshot.json`)

| Field | Value |
|---|---|
| ibkr_equity | $162,173.97 |
| coinbase_equity | $0.00 |
| kraken_equity | $184.58 |
| Total | ~$162,358.55 |
| ts_utc | 2026-05-13T22:39:01Z |
| ttl_seconds | 300 |

### 2.10 Tier State (`runtime/tier_state.json`)

| Field | Value |
|---|---|
| schema_version | tier_state.v2 |
| **tier_name** | **SCALE** |
| description | Full stack — all 16 strategies, existing sizing pipeline unchanged |
| current_equity_usd | $162,445.36 |
| tier_min_equity | $160,000 |
| tier_max_equity | $10,000,000 |
| enabled_strategies | alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_options, beta, beta_trend, delta, delta_pairs, gamma, gamma_futures, gamma_reversion, omega, omega_macro, omega_momentum_options, omega_vol (16; **note: alpha_intraday_micro NOT in expanded list** — wildcard expansion uses canonical 16-name list, see §5.2) |
| allowed_instruments | ["*"] |
| risk_profile | all caps NULL (SCALE leaves caps to base sizing pipeline) |
| previous_tier | SCALE |
| promoted_at_utc | 2026-04-27T19:00:35Z |
| demotion_pending | false |
| ts_utc | 2026-05-13T22:38:55Z |

### 2.11 Business Phase (`runtime/business_phase.json`)

| Field | Value |
|---|---|
| phase | GROW |
| description | Engine is built. Now growing the account before salary starts. SCR must reach CONFIDENT first. |
| current_equity_usd | $162,224.07 |
| seed_capital_usd | $50,000 |
| growth_pct_from_seed | 224.45% |
| days_in_phase | 16 |
| high_water_mark_usd | $183,874.27 |
| next_phase_requirement | "To enter PAY phase: Recover above high water mark $183,874." |
| ts_utc | 2026-05-13T22:14:41Z |

### 2.12 Withdrawal Authorization (`runtime/withdrawal_authorization.json`)

| Field | Value |
|---|---|
| phase | GROW |
| current_equity_usd | $162,898.25 |
| high_water_mark_usd | $183,874.27 |
| drawdown_from_hwm_pct | 11.41% |
| spendable_surplus_usd | $0.00 |
| authorized_withdrawal_usd | $0.00 |
| scr_state | CONFIDENT |
| history_days | 16 |
| reason | "GROW phase override: 30d drawdown 11.4% exceeds veto threshold 5.0%. No salary during drawdown." |
| ts_utc | 2026-05-13T19:02:35Z |

### 2.13 Macro State (`runtime/macro_state.json`)

| Field | Value |
|---|---|
| risk_label | risk_on |
| composite_risk_label | moderate |
| cpi_yoy_pct | 3.947 |
| unemployment_rate_pct | 4.3 |
| us_10y / us_2y / 10-2 spread | 4.46% / 4.00% / 0.48% |
| high_yield_spread_pct | 2.82 |
| risk_flags | credit_stress=false, inflation_elevated=true, recession_risk=false, yield_curve_inverted=false |
| provider | FredYieldProvider (DGS2/DGS10/UNRATE/CPIAUCSL/BAMLH0A0HYM2/T10Y2Y) |
| ts_utc | 2026-05-13T22:29:15Z |
| ttl_seconds | 1800 |

### 2.14 Event Risk (`runtime/event_risk.json`)

| Field | Value |
|---|---|
| elevated_risk | true |
| severity | high |
| risk_score | 0.8 |
| risk_description | CPI Release |
| next_event | Retail Sales (2026-05-15T12:30Z, 38h away, medium) |
| operator_calendar_present | true |
| operator_events | 5 |
| rule_events | 11 |
| merged | 16 events |
| windows | 16 forward (FOMC, CPI, NFP, GDP through 2026-09-04) |

### 2.15 Winner Scaling (`runtime/winner_scaling.json`)

| Field | Value |
|---|---|
| schema_version | winner_scaling.v1 |
| max_multiplier | 1.5 |
| min_multiplier | 0.5 |
| median_expectancy | 0.28 |
| n_strategies_scaled | 7 |
| n_strategies_neutral | 5 |
| min_trades_for_scaling | 5 |
| excluded_strategies | broker_sync, manual, paper_exec, unknown |
| multipliers (selected) | alpha 0.5, alpha_intraday 0.5, alpha_options 1.5, beta 1.0, beta_trend 1.0, delta 1.5, gamma_futures 1.5, omega_macro 0.5; broker_sync/manual=1.0 |
| ts_utc | 2026-05-13T22:40:05Z |

### 2.16 Strategy Health (`runtime/strategy_health.json`, sampled)

| Strategy | health_score | sharpe_norm | win_rate | sample_count |
|---|---|---|---|---|
| alpha | 0.4293 | 0.00 | 0.431 | 111 |
| alpha_futures | 0.8827 | 1.00 | 0.609 | 585 |
| alpha_intraday | 0.3642 | 0.00 | 0.214 | 19 |
| alpha_options | 1.0000 | 1.00 | 1.000 | 143 |
| beta | 0.9250 | 1.00 | 0.750 | 4 |
| delta | 0.9025 | 1.00 | 0.675 | 87 |
| gamma_futures | 1.0000 | 1.00 | 1.000 | 12 |
| manual | 0.3500 | 0.50 | 0.000 | 5 |
| omega_macro | 0.4000 | 0.50 | 0.000 | 14 |
| omega_momentum_options | 0.4000 | 0.50 | 0.000 | 3 |
| omega_vol | 0.1000 | 0.00 | 0.000 | 3 |

Weights: sharpe 0.4 / win_rate 0.3 / slippage 0.2 / regime 0.1. Regime state at health computation: trending_bull. ts_utc 2026-05-13T22:39:33Z.

### 2.17 Active Epoch

Paper Epoch 2 (CLAUDE.md). SCR sizing_factor 1.0 sustained from 2026-05-09 (CONFIDENT promotion); paper_only flipped to false at that promotion. Effective_trades 194 (>180 cutoff for CONFIDENT band).

### 2.18 Operational Conclusion

System is in GREEN reconciliation, CONFIDENT SCR, NORMAL profit lock, trending_bull regime, choppy_active=false, stop_bus inactive, paper mode, withdrawal phase=GROW with 11.4 % drawdown from HWM. Live activation continues to be blocked by `live_readiness.json` (ready_for_live=false) and operator approval. Account is **above** SCALE tier floor by ~$2,400.

---

## 3. Strategies — All 17

For each: source path, sleeve, weight, multiplier breakdown from `runtime/dynamic_caps.json` where available, regime activation matrix membership, condition guards, status.

### 3.1 ALPHA

| Field | Value |
|---|---|
| Source | chad/strategies/alpha.py |
| Sleeve | ALPHA (50/30/20 chassis) |
| Weight (config) | 0.16 (config/strategy_weights.json) |
| normalized_weight | 0.1316 (dynamic_caps.json) |
| dollar_cap | $534.26 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / **0.5** / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | Always-on within active regimes; max 3 signals/symbol/day |
| Universe | Legend-filtered (legend.weights keys) or raw ctx.prices keys |
| Conditions | Regime detect on EMA(12)/EMA(48); uptrend → BUY trend_momentum; recovery → BUY recovery_long; downtrend → SELL trend_short; chop → fade midpoint. Gates: 0.002 ≤ atr_pct ≤ 0.050; range/ATR ≤ 3.0; momentum/ATR ≥ 0.35. Position-aware exits (vol spike >6%, ATR trail 2.0×, trend break, time stop 30 bars). |
| Status | ACTIVE (regime trending_bull, sample_count 111, health 0.4293 — sub-healthy but not halted) |

### 3.2 ALPHA_INTRADAY

| Field | Value |
|---|---|
| Source | chad/strategies/alpha_intraday.py |
| Sleeve | ALPHA |
| Weight (config) | 0.03 |
| normalized_weight | 0.0248 |
| dollar_cap | $100.82 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / **0.5** / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | 10-min cooldown per symbol; 1-min bars with daily fallback |
| Universe | SPY, QQQ, AAPL, NVDA, MSFT, GOOGL, BAC, MES, MNQ, BTC-USD |
| Conditions | Three OR-triggers: (1) vol explosion (current_ATR/baseline_ATR ≥ 2.0); (2) momentum surge (5-bar move ≥ 0.5 % AND aligned 20-bar AND \|m_short\| ≥ 1.5 ×\|m_long\|); (3) mean-reversion snap on equities (BB 2σ penetration with bb_pen=0.003 + RSI <25 or >75). All signals tagged high_convexity=true, stop_loss_pct=1.5, take_profit_pct=4.5, max_hold_bars=30. |
| Status | ACTIVE (sample_count 19, health 0.3642) |

### 3.3 ALPHA_INTRADAY_MICRO (NEW v9.1)

| Field | Value |
|---|---|
| Source | chad/strategies/alpha_intraday_micro.py + alpha_intraday_micro_config.py |
| Sleeve | ALPHA |
| Weight (config) | Not present in config/strategy_weights.json (additive at registration time) |
| dynamic_caps overlay | Not present in current dynamic_caps.json (strategy not in `tier_enabled_strategies` set when SCALE wildcard expands to canonical 16-name list — see §5.2) |
| Active in | All regimes (regime activation matrix unaltered; the strategy is tier- and session-gated, not regime-gated). |
| Silent in | Outside America/New_York primary window (09:35–11:00) and secondary (13:30–15:00); inside EOD-flatten window (configurable by tier); ranging/adverse implicit via downstream filters. |
| Window | PRIMARY 09:35–11:00 ET; SECONDARY 13:30–15:00 ET; HARD_EOD_EXIT 15:30 ET; ORB valid 45 min after open; SWEEP valid 60 min after open |
| Universe | MES, MNQ (FUTURES) — DOLLARS_PER_POINT MES=5.0, MNQ=2.0 |
| Conditions (entry guard, per setup family) | All five evaluators run per cycle; signals emitted in setup-priority order (ORB=1 < VWAP=2 < PULLBACK=3 < SWEEP=4). Gate chain: session window → flatten window → setup-specific guards → tier `validate_stop_width` → tier `check()` (max_contracts/trades/daily/weekly loss) → priority suppression → duplicate suppression. Confidence fixed at 0.65. Sizing always 1 contract clamped against tier `max_contracts_per_trade`. |
| **Setup families (all 5)** | **ORB** — opening-range breakout (volume ≥ 1.3× OR avg; extension ≤ 1.0%; prior-day clearance ≥ 1.5 ATR). **VWAP_RECLAIM** — prev_close < VWAP, current ≥ VWAP, initial drive ≥ 0.3%, ≤ 3 crosses in last 10 bars. **VWAP_REJECTION** — prev_close > VWAP, current ≤ VWAP, same gates. **PULLBACK_CONTINUATION** — 3-bar monotonic trend then retrace ≤ 50% of impulse range. **SWEEP_REVERSAL** — sweep of overnight high/low + snap back within 2 bars (breach ≤ 0.5%, not a trend day i.e. extension < 1.5%). |
| Status | ARMED (registered in canonical strategy registry at `chad/strategies/__init__.py:240`; handler is invoked through `live_execution_router._build_available_signals → iter_strategy_registrations`. Handler short-circuits to `[]` when `ctx.tier_profile` is not a `TierRiskProfile` — see §14 residual). |

### 3.4 ALPHA_FUTURES

| Field | Value |
|---|---|
| Source | chad/strategies/alpha_futures.py |
| Sleeve | ALPHA |
| Weight (config) | 0.09 |
| normalized_weight | 0.0745 |
| dollar_cap | $604.94 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | Overnight gate on MCL/MGC (block entries outside 13:30–20:00 UTC); MES/MNQ exempt |
| Universe | MES, MNQ, MGC (disjoint from gamma_futures) |
| Conditions | Position-aware. Exit checks first: ATR stop (stop_loss_atr_multiple=2.0), trend exit (price vs EMA_slow), time stop (≥ 20 bars). Entry: trend regime via EMA(12)/EMA(26)+price; breakout overrides via 20-bar HH/LL. Confidence ≥ 0.65 floor. Sizing via ATR ($150 min risk; 1.5% pct; $50k max notional). |
| Status | ACTIVE (sample 585, health 0.8827) |

### 3.5 ALPHA_OPTIONS

| Field | Value |
|---|---|
| Source | chad/strategies/alpha_options.py |
| Sleeve | ALPHA |
| Weight (config) | 0.04 |
| normalized_weight | 0.0331 |
| dollar_cap | $403.29 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / **1.5** / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | 21–45 DTE; max_hold_seconds=3600 (forced SELL exit); max 4 open spreads |
| Universe | SPY only |
| Conditions | Read upstream alpha/gamma/gamma_reversion directional signals (confidence ≥ 0.70) → bull call (bullish) or bear put (bearish) vertical spread. If no upstream signal, derive from EMA(12)>EMA(26) on bars. Build SpreadSpec via chad.options.strike_selector (otm_offset 2%, width 1%). Emit ONE BAG combo signal sec_type="BAG" sized by 0.5% equity risk per spread. |
| Status | ACTIVE (sample 143, health 1.0) |

### 3.6 ALPHA_CRYPTO

| Field | Value |
|---|---|
| Source | chad/strategies/alpha_crypto.py |
| Sleeve | ADAPTIVE |
| Weight (config) | 0.04 |
| normalized_weight | 0.0139 |
| dollar_cap | $113.14 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | Max 2 signals/cycle; regime read from runtime/regime_state.json fallback |
| Universe | BTC-USD, ETH-USD, SOL-USD (+ BTC-CAD/ETH-CAD when Kraken CAD balance > 1.0) |
| Conditions | Long-only momentum: (1) price > SMA20 (20-day simple), (2) price > prior_close (up-confirm), (3) 3d move ≥ 1.5 %, (4) vol5/vol20 ratio ≥ 0.7 (skip compression), boost 1.2× when > 1.5. Regime multiplier 0.5× in trending_bear, else 1.0; ranging/adverse → []. Strength clamp [0.3, 1.0]; size = target_notional / price (notional $1500–$5000 by strength). |
| Status | ACTIVE |

### 3.7 BETA

| Field | Value |
|---|---|
| Source | chad/strategies/beta.py |
| Sleeve | BETA |
| Weight (config) | 0.05 (carve-out from beta_trend 2026-04-23) |
| normalized_weight | 0.1995 (in dynamic_caps — note: dynamic_caps.json mixes beta+beta_trend allocation differently; see strategy_weights.json for source-of-truth weights) |
| dollar_cap | $1,619.49 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, ranging, volatile, unknown |
| Silent in | adverse |
| Window | Once per UTC day per symbol; ≤ 2 signals/cycle; ≤ 3 signals/week; 7-day cooldown between rebalances per symbol |
| Universe | runtime/institutional_consensus.json (13F-driven, refreshed weekly) |
| Conditions | gap = target_capped − current_weight where target_capped = min(consensus_weight, 0.02). If gap ≥ 0.005 (underweight_gap) → BUY sized to half-gap (int(equity × 0.5 × gap / price)). Fail-closed if consensus stale > 45 days. ProfitRouter beta-injection optional (CHAD_PROFIT_ROUTER_BETA_INJECTION env). |
| Status | ACTIVE (sample 4, health 0.925) |

### 3.8 BETA_TREND

| Field | Value |
|---|---|
| Source | chad/strategies/beta_trend.py |
| Sleeve | BETA |
| Weight (config) | 0.20 |
| normalized_weight | 0.1005 |
| dollar_cap | $815.89 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | Once per UTC day per symbol; max 20 signals/day hard cap; 14-day cooldown after exit; 21-day min hold before add-on |
| Universe | Legend consensus top-10 (SPY, QQQ, IWM, DIA, TLT, IEF, GLD, LQD, VWO, IEMG by weight) |
| Conditions | Legend weight ≥ 0.05 threshold. EWMA smoothing (α=0.35) of legend weights. If flat → BUY; if holding ≥ 21 days → BUY add-on (size × 0.5). Confidence = 0.50 + (w × 2.0) clamped [0.50, 0.95]. |
| Status | ACTIVE |

### 3.9 GAMMA

| Field | Value |
|---|---|
| Source | chad/strategies/gamma.py |
| Sleeve | ALPHA |
| Weight (config) | 0.07 |
| normalized_weight | 0.1365 |
| dollar_cap | $1,108.40 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | ranging, volatile, unknown (regime matrix) |
| Silent in | trending_bull, trending_bear (NOT in matrix), adverse |
| Window | Time-stop 60 bars on long positions |
| Universe | get_trade_universe() (default ETF/equity liquid set) |
| Conditions | Regime split on EMA slope (in ATR units): if |slope| ≤ 0.06 → RANGE regime, BUY when price deviates > 0.75 ATR below EMA_slow (mean reversion). If > 0.06 → TREND regime, BUY when ema_fast > ema_slow AND price > ema_fast AND momentum/ATR ≥ 0.35. Exits: vol spike (atr_pct > 0.055), trend break (ema_fast ≤ ema_slow), time stop, ATR trail 2.4×. |
| Status | REGIME-SILENT in trending_bull (current regime) — strategy is not in the trending_bull/bear allowed list in config/regime_activation_matrix.json |

### 3.10 GAMMA_FUTURES

| Field | Value |
|---|---|
| Source | chad/strategies/gamma_futures.py |
| Sleeve | ALPHA |
| Weight (config) | 0.05 |
| normalized_weight | 0.0414 |
| dollar_cap | $336.08 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / **1.5** / 1.0 / 1.0; halt_clamp_applied=true |
| Active in | trending_bull, trending_bear, volatile |
| Silent in | ranging, adverse, unknown |
| Window | None explicit; mean-reversion driven |
| Universe | MCL (primary), MYM, M2K (with ZN/ZB fallback) — disjoint from alpha_futures |
| Conditions | Four mean-reversion triggers (any): (1) RSI > 75 AND price ≥ BB_upper → SELL; (2) RSI < 25 AND price ≤ BB_lower → BUY; (3) (price − EMA_slow)/EMA_slow > 0.02 → SELL; (4) (EMA_slow − price)/EMA_slow > 0.02 → BUY. Confidence = 0.65 + 0.15×RSI_extremity + 0.10×BB_penetration. Risk budget 1.2 %, $40k max notional. Liquidity gate ≥ $10M. |
| Status | ACTIVE but halt_clamp_applied (winner_factor 1.5 clamped to 1.0 if halt active). Sample 12, health 1.0. |

### 3.11 GAMMA_REVERSION

| Field | Value |
|---|---|
| Source | chad/strategies/gamma_reversion.py |
| Sleeve | ALPHA |
| Weight (config) | 0.04 |
| normalized_weight | 0.0331 |
| dollar_cap | $268.86 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | ranging |
| Silent in | trending_bull, trending_bear (NOT in matrix), adverse |
| Window | 15-bar max hold; SMA-20 target; 2.5× ATR stop |
| Universe | SPY, QQQ, GLD, TLT |
| Conditions | 3-of-3 confluence. SHORT: RSI > 72 AND (px > BB_upper OR zscore > 1.8) AND ROC > 0. LONG: RSI < 28 AND (px < BB_lower OR zscore < −1.8) AND ROC < 0. GLD strict: requires BOTH BB AND zscore (not OR). Confidence 0.55 + 0.10 if extreme RSI + 0.10 if \|z\| > 2.2 + 0.05 if \|z\| > 2.5. |
| Status | REGIME-SILENT in trending_bull |

### 3.12 DELTA

| Field | Value |
|---|---|
| Source | chad/strategies/delta.py |
| Sleeve | ADAPTIVE |
| Weight (config) | 0.02 |
| normalized_weight | 0.0668 |
| dollar_cap | $542.05 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / **1.5** / 1.0 / 1.0; halt_clamp_applied=true |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | 40-bar time stop; ATR trail 2.8× |
| Universe | Legend top weights or ctx.delta_universe (max 4 symbols/cycle) |
| Conditions | Three-of-three convexity gate: (1) EMA_fast > EMA_slow AND price > EMA_fast; (2) price > rolling_high[-2] + 0.10 ATR (breakout); (3) (price − EMA_fast)/ATR ≥ 0.45. Plus optional vol expansion (ATR_now ≥ 1.35× baseline_ATR). Conviction score 0–1; size = min(max_size, base_size + score × 18). Long-only. |
| Status | ACTIVE, halt_clamp_applied. Sample 87, health 0.9025. |

### 3.13 DELTA_PAIRS

| Field | Value |
|---|---|
| Source | chad/strategies/delta_pairs.py |
| Sleeve | ADAPTIVE |
| Weight (config) | 0.05 |
| normalized_weight | 0.0281 |
| dollar_cap | $227.95 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | ranging |
| Silent in | trending_bull, trending_bear (NOT in matrix), adverse |
| Window | Exit when \|Z\| ≤ 0.5; hard stop when \|Z\| ≥ 3.5 |
| Universe | SPY/QQQ, SPY/IWM, QQQ/IWM (3 pairs, 6 legs) |
| Conditions | Ratio z-score crossing: \|Z\| ≥ 2.0 → fade divergence. Z > 0 → long_A/short_B; Z < 0 → short_A/long_B. Dollar-neutral, equal units. Confidence = min(1.0, \|z\|/2.0) × correlation (e.g. SPY/QQQ corr 0.993). Defined risk 0.8 % equity per pair. |
| Status | REGIME-SILENT in trending_bull |

### 3.14 OMEGA

| Field | Value |
|---|---|
| Source | chad/strategies/omega.py |
| Sleeve | ADAPTIVE |
| Weight (config) | 0.05 |
| normalized_weight | 0.0463 |
| dollar_cap | $375.73 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | volatile, unknown |
| Silent in | trending_bull, trending_bear, ranging (NOT in matrix), adverse |
| Window | 60-minute hedge cooldown |
| Universe | SH (inverse SPY), PSQ (inverse QQQ) |
| Conditions | Activate hedge on ≥2/3 danger sensors: drawdown ≤ −6 %, ATR% spike ≥ 3 %, VIX ≥ 25. Deactivate on ≥2/3 calm: dd > −3 %, ATR% ≤ 2 %, VIX ≤ 20. Fixed confidence 0.85 (BUY hedge) / 0.75 (SELL hedge). Sizing 5–25 units. |
| Status | REGIME-SILENT in trending_bull |

### 3.15 OMEGA_VOL

| Field | Value |
|---|---|
| Source | chad/strategies/omega_vol.py |
| Sleeve | ADAPTIVE |
| Weight (config) | 0.05 |
| normalized_weight | 0.0281 |
| dollar_cap | $227.95 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | volatile |
| Silent in | trending_bull, trending_bear, ranging (NOT in matrix), adverse |
| Window | UVXY time stop 10 bars; SVXY cap 3 % equity |
| Universe | SVXY (short vol), UVXY (long vol) |
| Conditions | 5-state classifier: LOW_VOL (VIX<15) → SHORT vol; NORMAL (15–22) → mild SHORT if momentum<0; ELEVATED (22–30) → no signal; CRISIS (≥30) → LONG vol; VOL_CRUSH (VIX −20% from peak) → SHORT vol (highest conviction). UVXY size halved (structural decay). |
| Status | REGIME-SILENT in trending_bull. Sample 3, health 0.10. |

### 3.16 OMEGA_MACRO

| Field | Value |
|---|---|
| Source | chad/strategies/omega_macro.py |
| Sleeve | ADAPTIVE |
| Weight (config) | 0.03 |
| normalized_weight | 0.0168 |
| dollar_cap | $68.39 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / **0.5** / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, ranging, volatile, unknown |
| Silent in | adverse |
| Window | None; regime-direction only |
| Universe | ZN (10Y), ZB (30Y), M6E (Micro Euro FX) |
| Conditions | Macro regime: RISK_OFF (VIX>25 or dd≤−5%) → ZN BUY, ZB BUY, M6E SELL; RISK_ON (VIX<18 or dd>−2%) → ZN SELL, ZB SELL, M6E BUY; STAGFLATION → ZN BUY, ZB BUY, M6E SELL; NEUTRAL → no signals. Confidence base 0.55 + alignments. Risk budget 1.0 %, $35k max notional. |
| Status | ACTIVE. Sample 14, health 0.4 (degraded). |

### 3.17 OMEGA_MOMENTUM_OPTIONS

| Field | Value |
|---|---|
| Source | chad/strategies/omega_momentum_options.py |
| Sleeve | ALPHA |
| Weight (config) | 0.03 |
| normalized_weight | 0.0248 |
| dollar_cap | $201.65 |
| tier_factor / winner_factor / regime_factor / choppy_factor | 1.0 / 1.0 / 1.0 / 1.0 |
| Active in | trending_bull, trending_bear, volatile, unknown |
| Silent in | ranging, adverse |
| Window | Hard exit 15:45 ET; 15-min cooldown per symbol; max 3 concurrent positions; market hours only |
| Universe | SPY, QQQ, AAPL, NVDA, MSFT |
| Conditions | 5-bar move ≥ 0.3 % + EMA_10 slope aligned + volume ≥ 1.5× average; VIX ≤ 40 gate. BUY_CALL on bullish + EMA slope up; BUY_PUT on bearish + slope down. 50 % profit target / 25 % stop loss. Regime-dependent contracts (NORMAL=2, LOW/ELEVATED=1). |
| Status | ACTIVE. Sample 3, health 0.4. |

---

## 4. Execution Pipeline

### 4.1 Pipeline Diagram (v9.1)

```
TradeSignal[] from 17 active strategies (via iter_strategy_registrations)
         │
         ▼
[1] Edge-decay halt filter
    drops signals from strategies marked halted=true in
    runtime/strategy_allocations.json (edge_decay_monitor)
         │
         ▼
[2] StrategyRouter — single-winner OR always-active (CHAD_ALWAYS_ACTIVE_ROUTING)
         │
         ▼
[3] TierInstrumentGate  (NEW v9.1, Gap-1)
    reads runtime/tier_state.json → allowed_instruments
    blocks INSTRUMENT_NOT_IN_TIER_ALLOWLIST, fail-open
    writes runtime/tier_instrument_gate_state.json
         │
         ▼
[4] NetExposureGate
    decisions: ALLOW / MERGE / REDUCE / CLOSE_ONLY / FLIP_ALLOWED / BLOCK
    reads position_guard, reconciliation_state, portfolio_snapshot, price_cache
         │
         ▼
[5] StrategyThrottleGate
    levels: ALLOW / THROTTLE / CONFIDENCE_UPSHIFT / PAUSE_TEMPORARILY /
            HALT_DEFER_TO_EDGE_DECAY
    reads today's trade_history; writes runtime/strategy_throttle_state.json
         │
         ▼
[6] Routing gates (execution_pipeline.build_ibkr_intents_from_plan)
    data_freshness (A4) / stale_intent (E2) / too_late_to_chase (E5) /
    net_ev (R7)
         │
         ▼
[7] ML veto (shadow-on by default, enforcement gated by canary)
    Booster scores 10 features → ShadowDecision
    Protective intents (exit/reduce/hedge/liquidate) always PASS
         │
         ▼
[8] TierRiskEnforcer  (NEW v9.1)
    validate_stop_width() before sizing
    check(strategy, instrument, contracts):
      SKIP_INVALID_CONTRACT_COUNT, SKIP_MAX_CONTRACTS_EXCEEDED,
      SKIP_MAX_TRADES_REACHED, SKIP_DAILY_LOSS_LIMIT,
      SKIP_WEEKLY_LOSS_LIMIT
    writes runtime/tier_enforcement_state.json (TTL 300s)
         │
         ▼
[9] Position guard + flip executor (BG11 close-first/open-second)
         │
         ▼
[10] build_execution_plan → PlannedOrder[]
     net by symbol; validate price/size; whole-unit enforcement on STK/FUT/OPT
         │
         ▼
[11] build_ibkr_intents_from_plan → IBKRStrategyTradeIntent[]
         │
         ▼
[12] IBKRAdapter (ib_async 2.1.0) → broker
         │
         ▼
[13] Paper exec evidence writer → fills ledger + SCR shadow update
```

### 4.2 Stages newly added since v9.0

- **TierInstrumentGate** (stage 3) — Gap-1 closure. Prevents alpha_crypto BTC/ETH/SOL signals from reaching a STARTER account whose allowed_instruments list does not include ETH-USD/SOL-USD.
- **TierRiskEnforcer.validate_stop_width** (stage 8) — gates each candidate trade on `stop_width_usd ≤ tier.max_risk_per_trade_usd` (or pass-through when cap is None, which is SCALE today).
- **TierRiskEnforcer.check** (stage 8) — gates each candidate trade on `daily_loss / weekly_loss / trades_today / max_contracts_per_trade`.

---

## 5. Risk & Governance — The Full Chain

### 5.1 Cap calculation chain (`chad/risk/dynamic_risk_allocator.py`)

For each strategy, the per-cycle dollar cap is computed as:

```
base_cap         = total_equity × daily_risk_fraction × normalized_weight
profit_lock_mult = profit_lock_state.json.sizing_factor          (0.0–1.0, default 1.0)
tier_factor      = 1.0 if strategy ∈ tier_state.enabled_strategies else 0.0
winner_factor    = winner_scaling.json.multipliers[strategy]     (0.5–1.5, default 1.0)
                 + halt-clamp: if strategy halted AND winner_factor > 1.0 → clamp to 1.0
regime_mult      = regime_booster.json (clamped to [1.0, 1.5])
choppy_mult      = choppy_regime_state.json overlay.sizing_multiplier (1.0 inactive / 0.25 active)
final_cap        = base_cap × profit_lock_mult × tier_factor × winner_factor × regime_mult × choppy_mult
```

`portfolio_risk_cap = total_equity × daily_risk_fraction` (today: `162,358.55 × 0.05 = 8,117.93`, matches `dynamic_caps.json`). Sum of `strategy_caps` over 16 strategies = $7,449.99 (some dollar_caps in dynamic_caps are post-winner — e.g. alpha base $1068.52 × 0.5 = $534.26 matches the file).

**NEW v9.1 — TierRiskEnforcer overlay (stage 8)** is enforced per-trade *after* the dollar-cap chain produces a size: the stop-width gate clamps trades where `entry−stop × $/point × contracts > max_risk_per_trade_usd`. Strategy must shrink size or skip. The check() gate hard-stops new trades on tier-level loss/trade-count limits.

### 5.2 Tier System v2 — full table from `config/tiers.json`

Schema: `tiers.v9_1`. Hysteresis: `5.0 %`.

#### MICRO

| Field | Value |
|---|---|
| description | Proving account — single instrument, minimum frequency |
| min_equity_usd | $0 |
| max_equity_usd | $2,500 |
| demotion_equity_usd | null (no tier below) |
| enabled_strategies | ["alpha_intraday_micro"] |
| allowed_instruments | ["MES"] |
| max_contracts_per_trade | 1 |
| max_risk_per_trade_usd | $10 |
| max_daily_loss_usd | $20 |
| max_weekly_loss_usd | $30 |
| max_trades_per_day | 2 |
| primary_session_only | true |
| flatten_before_eod | true (30 min before EOD) |
| stop_width_gate_enabled | true |

#### STARTER

| Field | Value |
|---|---|
| description | Controlled scaling — two instruments, primary session only |
| min_equity_usd | $2,500 |
| max_equity_usd | $25,000 |
| demotion_equity_usd | $2,250 |
| enabled_strategies | alpha_intraday_micro, alpha_intraday, beta_trend, alpha_crypto |
| allowed_instruments | MES, MNQ, SPY, QQQ, IWM, BTC-USD |
| max_contracts_per_trade | 2 |
| max_risk_per_trade_usd | $40 |
| max_daily_loss_usd | $150 |
| max_weekly_loss_usd | $300 |
| max_trades_per_day | 4 |
| primary_session_only | true |
| flatten_before_eod | true (30 min before EOD) |
| stop_width_gate_enabled | true |

#### PRO_GROWTH

| Field | Value |
|---|---|
| description | Full futures + equities — primary and secondary session |
| min_equity_usd | $25,000 |
| max_equity_usd | $160,000 |
| demotion_equity_usd | $22,500 |
| enabled_strategies | alpha, alpha_intraday, alpha_intraday_micro, alpha_futures, alpha_crypto, beta, beta_trend, gamma_futures, delta, omega_macro |
| allowed_instruments | ["*"] |
| max_contracts_per_trade | 5 |
| max_risk_per_trade_usd | $200 |
| max_daily_loss_usd | $500 |
| max_weekly_loss_usd | $1,500 |
| max_trades_per_day | 10 |
| primary_session_only | false |
| flatten_before_eod | false |
| stop_width_gate_enabled | true |

#### SCALE (current)

| Field | Value |
|---|---|
| description | Full stack — all 16 strategies, existing sizing pipeline unchanged |
| min_equity_usd | $160,000 |
| max_equity_usd | $10,000,000 |
| demotion_equity_usd | $144,000 |
| enabled_strategies | ["*"] (wildcard) |
| allowed_instruments | ["*"] |
| max_contracts_per_trade | null |
| max_risk_per_trade_usd | null |
| max_daily_loss_usd | null |
| max_weekly_loss_usd | null |
| max_trades_per_day | null |
| primary_session_only | false |
| flatten_before_eod | false |
| stop_width_gate_enabled | false |

> **Wildcard semantics**: in `chad/risk/tier_manager.py`, `_expand_enabled_strategies` replaces `["*"]` with the canonical 16-name list `_CANONICAL_STRATEGY_NAMES`. That list **does NOT include `alpha_intraday_micro`** (it lists alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_options, beta, beta_trend, delta, delta_pairs, gamma, gamma_futures, gamma_reversion, omega, omega_macro, omega_momentum_options, omega_vol). Therefore the published `tier_state.json` enabled_strategies omits alpha_intraday_micro under SCALE — see §14 residual.

### 5.3 Stop-width validation gate (NEW v9.1)

`chad/risk/tier_risk_enforcer.py:validate_stop_width`:

```
stop_width_points = |entry_price − stop_price|
stop_width_usd    = stop_width_points × dollars_per_point × contracts
budget_usd        = tier.max_risk_per_trade_usd          (or None at SCALE)
fits_budget       = True  if budget_usd is None
                  = (stop_width_usd ≤ budget_usd) otherwise
```

Returns `StopWidthValidation(stop_width_points, stop_width_usd, budget_usd, fits_budget)`. Currently a no-op at SCALE; will gate trades for MICRO/STARTER/PRO_GROWTH.

### 5.4 LiveGate (unchanged from v9.0)

`chad/core/live_gate.py` requires explicit operator intent (`ALLOW_LIVE`) and matching execution mode (`paper`/`live`). Paper soak runs unaffected. Live activation is procedurally gated:

1. Set CHAD_EXECUTION_MODE=live (Pending Action — must be operator-approved)
2. LiveGate accepts only when (operator_intent=ALLOW_LIVE) AND (live_readiness.ready_for_live=true)
3. Monitor first 3 execution cycles manually
4. Confirm broker truth reconciliation on first fill

### 5.5 Profit Lock (`runtime/profit_lock_state.json`)

| Tier | Threshold (pct of equity) | sizing_factor | stop_new_entries |
|---|---|---|---|
| WARN | 1.5 % | 1.00 | false |
| LOCK1 | 3.0 % | 0.50 | false |
| LOCK2 | 5.0 % | 0.25 | false |
| LOCK3 | 8.0 % | 0.10 | false |
| HARD_STOP | 10.0 % | 0.00 | true |
| daily_loss_limit | −3.0 % | n/a | true |

Current: mode=NORMAL, sizing_factor=1.0, daily_loss_today=$0.00, daily_loss_limit=$4,870.76. TTL 60s. PnL sources today: `trade_history_20260513.ndjson`, `FILLS_20260513.ndjson`.

### 5.6 SCR State (`runtime/scr_state.json`)

State machine: NEW_BUILD → SHADOW → CAUTIOUS → CONFIDENT → PAUSED. Bands keyed by sharpe_like, win_rate, max_drawdown, effective_trades, paused_recovery_ticks.

Current: **CONFIDENT** (sizing_factor 1.0, paper_only=false). Reasons: "win_rate, sharpe, and drawdown all within confident band." Effective trades 194 (paper 3873, live 0; excluded_untrusted 321, excluded_manual 5). win_rate 0.7629, sharpe_like +5.7519, max_drawdown −$290.00, total_pnl +$9,934.165. TTL 180s.

---

## 6. Business Framework

### 6.1 Components (all 7, unchanged from v9.0 structure)

1. **Tier state (`runtime/tier_state.json`)** — equity-band → enabled strategies/instruments/risk profile (now v2 schema).
2. **Business phase (`runtime/business_phase.json`)** — GROW / PAY phase machine with HWM tracking.
3. **Withdrawal authorization (`runtime/withdrawal_authorization.json`)** — salary-eligible surplus computation.
4. **Capital allocator (`runtime/capital_allocator.json`)** — strategy weight distribution.
5. **Winner scaling (`runtime/winner_scaling.json`)** — per-strategy multipliers (0.5–1.5).
6. **Profit lock (`runtime/profit_lock_state.json`)** — daily-PnL sizing factors.
7. **SCR shadow (`runtime/scr_state.json`)** — paper performance gate for live promotion.

### 6.2 Current state values (snapshot 2026-05-13)

| Component | Value |
|---|---|
| Tier | SCALE @ $162,445.36 (above floor $160k by ~$2.4k) |
| Business phase | GROW, day 16, growth_pct_from_seed 224.45 % |
| Withdrawal | $0 spendable (GROW phase override; dd 11.4 % > veto 5 %) |
| Capital allocator | beta_carveout_from_beta_trend_20260423 last applied |
| Winner scaling | 7 scaled, 5 neutral; max 1.5, min 0.5; median expectancy 0.28 |
| Profit lock | NORMAL, sizing 1.0 |
| SCR | CONFIDENT, sizing 1.0 |

---

## 7. Reconciliation

`runtime/reconciliation_state.json` (status GREEN; ts_utc 2026-05-13T22:38:59Z).

### 7.1 Tracked drifts (6 by name)

The reconciliation publisher tracks divergence across these six surfaces. Drift formalization documented in commit `d0f75dc` (Fix: formalize reconciliation drift exclusions for paper soak):

1. **strategy_open vs broker_positions** — CHAD-internal position guard vs IBKR account positions (excluding owner=operator exclusion_policy entries).
2. **paper_ledger vs broker_positions** — paper fill ledger reconstructed positions vs broker truth.
3. **position_guard vs trade_closer_state** — guard open flag vs trade_closer FIFO queue head.
4. **broker_sync entries vs broker_positions** — manual/external (broker_sync|*) entries reconciled against truth (excluded from rebuilder per GAP-028 Option B PERMISSIVE).
5. **positions_truth.json staleness** — drift detector emits YELLOW when positions_truth scope differs from current ledger (GAP-A009).
6. **position_guard_drift.json** — emitted by drift detector wired into chad-reconciliation-publisher (schema_version `position_guard_drift.v1`).

Excluded symbols (operator pre-existing): AAPL, BAC, CVX, LLY, MSFT, NVDA, PEP, QQQ, SPY (futures: M2K, MES). All `owner=operator`.

### 7.2 Operator close CLI

`scripts/close_guard_entry.py` — atomic guard-close + trade_closer FIFO clear. Refuses when SCR ∉ {CONFIDENT, CAUTIOUS}, when exec_mode is not paper/dry_run, when LiveGate operator intent is `ALLOW_LIVE`, or on `broker_sync|*` keys.

---

## 8. Intelligence Layer

### 8.1 Runtime intel feeds

| Feed file | Producer | TTL | Schema version |
|---|---|---|---|
| runtime/price_cache.json | price_cache_refresh | 180s | (legacy) |
| runtime/regime_state.json | live_loop.run_once | 360s | regime_state.v1 |
| runtime/choppy_regime_state.json | choppy_regime_detector | 300s | (none, dict shape) |
| runtime/dynamic_caps.json | dynamic_risk_allocator | 300s | (none, dict shape) |
| runtime/regime_booster.json | regime_booster_publisher | 240s | regime_booster.v1 |
| runtime/kraken_prices.json | kraken_ws | 120s | (legacy) |
| runtime/reconciliation_state.json | chad-reconciliation-publisher | 480s | (none, dict shape) |
| runtime/macro_state.json | macro_state_publish | 1800s (7200 watchdog) | macro_state.v1 |
| runtime/event_risk.json | event_risk_publish | 1800s | event_risk.v1 |
| runtime/scr_state.json | scr publisher | 180s | scr_state.v1 |
| runtime/profit_lock_state.json | profit_lock engine | 60s | (none, dict shape) |
| runtime/stop_bus.json | (manual / smoke_test) | n/a | stop_bus.v1 |
| runtime/strategy_health.json | strategy_health publisher | n/a | strategy_health.v1 |
| runtime/winner_scaling.json | winner_scaling publisher | n/a | winner_scaling.v1 |
| runtime/tier_state.json | tier_manager | n/a | tier_state.v2 |
| runtime/tier_enforcement_state.json | tier_risk_enforcer | 300s | (none) |
| runtime/tier_instrument_gate_state.json | tier_instrument_gate | 300s | tier_instrument_gate.v1 |
| runtime/setup_family_expectancy.json | setup_family_expectancy_updater (timer) | 86400s | setup_family_expectancy.v2 — **[NOT PRESENT — publisher first run pending; timer not yet installed]** |
| runtime/position_guard_drift.json | reconciliation publisher | n/a | position_guard_drift.v1 |
| runtime/business_phase.json | business_phase publisher | n/a | business_phase.v1 |
| runtime/withdrawal_authorization.json | withdrawal publisher | n/a | (none) |
| runtime/live_readiness.json | live_readiness publisher | 604800s (1 wk) | live_readiness_state.v1 |

### 8.2 ib_async migration status

**Primary execution path: COMPLETE** (commit 132f4e9).
- `chad/core/live_loop.py` — migrated to ib_async (Phase 2)
- `chad/execution/ibkr_adapter.py` — migrated to ib_async
- `chad/execution/ibkr_trade_router.py` — migrated to ib_async

Phase 1B sub-commits (8294369, 33a9628, 7e5dec2, 19b2fd5, fbf627e, b531270) handled advisory, reconciliation, market-data, and read-only utility imports.

**Two paper-runner files remain on ib_insync (deferred)**:
1. `chad/core/paper_position_closer.py`
2. `chad/core/paper_shadow_runner.py`

Justification: paper-only path with no live capital risk; coexistence validated by GAP-007 Phase 0 (commit 7f05965).

---

## 9. Telegram Operator Interface

Slash commands (unchanged from v9.0):

| Command | Purpose |
|---|---|
| `/status` | Composite health: SCR, profit_lock, regime, reconciliation, stop_bus, tier |
| `/scr` | Current SCR state + stats (sharpe_like, win_rate, drawdown, effective_trades) |
| `/regime` | Current regime + confidence + inputs |
| `/positions` | Open positions per strategy from position_guard |
| `/pnl` | Today's realized PnL, daily limit headroom |
| `/halt` | Operator-initiated stop_bus activation |
| `/resume` | Clear stop_bus |
| `/preview` | Run full_cycle_preview --dry-run |
| `/livegate` | Inspect LiveGate state (operator_intent, exec_mode, ready_for_live) |
| `/health` | Run health_monitor_rules (R01–R16) |

---

## 10. Dashboard

Backend service `chad-dashboard.service` (FastAPI). No structural changes since v9.0 except the dashboard PnL fix (commit eb54773 — show today realized and total paper PnL).

---

## 11. Services & Timers

### 11.1 Active services (`deploy/*.service`)

| Service | Purpose |
|---|---|
| chad-advisory-pre-market.service | Pre-market intel refresh |
| chad-dashboard.service | FastAPI operator dashboard |
| chad-expectancy-tracker.service | Per-strategy expectancy tracker (legacy, distinct from setup_family) |
| chad-ibkr-bar-provider.service | IBKR bar collector daemon |
| chad-kraken-ws.service | Kraken websocket price feed |
| **chad-micro-eod-flatten.service** | **NEW v9.1 — alpha_intraday_micro EOD flatten** (ops/micro_eod_flatten.py) |
| chad-options-monitor.service | Options chain monitor |
| chad-reconciliation-publisher.service | Position-guard drift detector + reconciliation state |
| chad-reddit-sentiment-refresh.service | Reddit sentiment refresh |
| **chad-setup-expectancy.service** | **NEW v9.1 — setup_family_expectancy_updater** (ops/update_setup_family_expectancy.py) |
| chad-short-interest-refresh.service | Short interest refresh |
| chad-strategy-intelligence-refresh.service | Strategy intelligence refresh |
| chad-symbol-blocker.service | Symbol blocklist refresh |
| chad-trade-closer.service | Position close / fill confirmation |
| chad-trends-refresh.service | Trends refresh |
| ibkr_paper_fill_harvester.service | IBKR paper fill harvester |
| options_chain_refresh.service | Options chain refresh |

### 11.2 Active timers (`deploy/*.timer`)

| Timer | Cadence |
|---|---|
| chad-advisory-pre-market.timer | (daily, pre-market window) |
| chad-expectancy-tracker.timer | (per-strategy expectancy) |
| **chad-micro-eod-flatten.timer** | **Mon–Fri 15:30 America/New_York** (AccuracySec 30s, Persistent=false) — **NEW v9.1** |
| chad-options-monitor.timer | (intraday) |
| chad-reconciliation-publisher.timer | (frequent) |
| chad-reddit-sentiment-refresh.timer | (periodic) |
| **chad-setup-expectancy.timer** | **Mon–Fri 21:00 America/New_York** (AccuracySec 60s, Persistent=true) — **NEW v9.1** |
| chad-short-interest-refresh.timer | (periodic) |
| chad-strategy-intelligence-refresh.timer | (periodic) |
| chad-symbol-blocker.timer | (periodic) |
| chad-trade-closer.timer | (frequent) |
| chad-trends-refresh.timer | (periodic) |
| ibkr_paper_fill_harvester.timer | (frequent) |
| options_chain_refresh.timer | (intraday) |

### 11.3 Lifecycle replay engine

Wired by commit d7672f7 (`Wire: chad-lifecycle-replay-engine systemd unit + timer`). Status: timer still pending install per f7ebae2 docs note.

### 11.4 Retired / intentionally disabled

- `chad-feed-watchdog` (script exists at `chad/ops/feed_watchdog.py`, intended to run via systemd timer; not currently installed — runs ad-hoc).
- ALPHA_FOREX strategy (commented out in `chad/strategies/__init__.py` registry, cf01ba8 formalize).

---

## 12. Data & Storage

### 12.1 Universe (`config/universe.json`)

**Equities/ETFs (38 symbols):** AAPL, SPY, MSFT, GOOGL, BAC, IEMG, QQQ, VWO, NVDA, GLD, SH, PSQ, SVXY, UVXY, VIXY, IWM, TLT, AMZN, META, V, MA, UNH, KO, AVGO, LLY, JNJ, WMT, CVX, NFLX, AMD, ORCL, CRM, QCOM, PEP, ADBE, MCD, INTC, VXX.

**Futures (10 symbols):** MES (CME), MNQ (CME), MCL (NYMEX), MGC (COMEX), ZN (CBOT), ZB (CBOT), M6E (CME), SIL (COMEX), MYM (CBOT), M2K (CME).

Crypto universe is strategy-local: alpha_crypto uses BTC-USD/ETH-USD/SOL-USD (plus BTC-CAD/ETH-CAD when Kraken CAD balance is present).

### 12.2 Bar provider

IBKR is sole market data source for equities/ETFs/futures (Polygon provider dead code removed commit dbf1842). `chad-ibkr-bar-provider.service` writes to `runtime/ibkr_bars_cache.json` and `data/bars/{1d,1m}/`. Crypto bars from Kraken WS.

### 12.3 setup_family_expectancy state (NEW v9.1)

- Path: `runtime/setup_family_expectancy.json`
- Schema: `setup_family_expectancy.v2`
- TTL: 86400s
- Writer: `chad/analytics/setup_family_expectancy_updater.py` invoked by `chad-setup-expectancy.service`
- Lookback: 90 days
- Schema fields per family: `trades, wins, win_rate, avg_r, expectancy_r, skip_count_stop_too_wide, status (ACTIVE / LOW_SAMPLE / NO_DATA)`
- Families: ORB, VWAP_RECLAIM, VWAP_REJECTION, PULLBACK_CONTINUATION, SWEEP_REVERSAL, UNKNOWN
- Current state: **[NOT PRESENT — publisher not yet run; timer pending install]**

### 12.4 Fills ledger

Unchanged: `data/trades/trade_history_YYYYMMDD.ndjson` (closed trades, schema `closed_trade.v1`), `data/fills/FILLS_YYYYMMDD.ndjson` (raw fill stream).

---

## 13. Change Log — Delta from v9.0

**91 commits** from 938ec2c (exclusive) to 67ba391 (inclusive). Grouped by theme.

### 13.1 v9.1 workstream (HEAD commit)

`67ba391` — Build: v9.1 workstream — tier arch v2 (MICRO/STARTER/PRO_GROWTH/SCALE), alpha_intraday_micro (5 setup families), tier_risk_enforcer, tier_instrument_gate, setup_family_expectancy, EOD flatten, health rules R15/R16, Gap-4 meta forwarding — 1652 tests pass.

Expanded components:
- Tier architecture v2 (tier_manager.py: TierRiskProfile, _LEGACY_TIER_ALIASES, _CANONICAL_STRATEGY_NAMES wildcard expansion, schema bump tier_state.v1 → v2)
- alpha_intraday_micro strategy + alpha_intraday_micro_config (5 setup families, 19 deterministic skip reason codes)
- chad/risk/tier_risk_enforcer.py (check + validate_stop_width, 5 skip reasons)
- chad/execution/tier_instrument_gate.py (Gap-1 INSTRUMENT_NOT_IN_TIER_ALLOWLIST)
- chad/analytics/setup_family_expectancy_updater.py (v2 schema)
- ops/micro_eod_flatten.py + deploy/chad-micro-eod-flatten.{service,timer}
- ops/update_setup_family_expectancy.py + deploy/chad-setup-expectancy.{service,timer}
- chad/ops/health_monitor_rules.py: added R15 (tier daily loss budget approaching), R16 (setup family skip rate)
- Gap-4: alpha_options option meta forwarding through execution pipeline (commit a949756)

### 13.2 ib_async migration (Phase 1B / Phase 2)

- `132f4e9` Fix: migrate ib_insync → ib_async on execution path (Phase 2)
- `8294369` Fix: GAP-A019 Phase 1B.6 migrate advisory IBKR import to ib_async
- `33a9628` Fix: GAP-A019 Phase 1B.5 migrate reconciliation imports to ib_async
- `7e5dec2` Fix: GAP-A019 Phase 1B.4 migrate market-data imports to ib_async
- `19b2fd5` Test: support ib_async fakes for GAP-A019 migration
- `fbf627e` Fix: GAP-A019 Phase 1B.2 Batch 1 migrate read-only utilities to ib_async
- `b531270` Fix: GAP-A019 Phase 1B.1 migrate backend IBKR health import to ib_async
- `ce51338` Audit: GAP-A019 reclassify ib_insync migration risk
- `fb59b8c` Docs: document qualify timeout as ib_async Phase 2 dependency
- `e876afb` Test: GAP-007 Phase 1 classify ib_insync migration risk
- `7f05965` Fix: GAP-007 Phase 0 validate ib_async coexistence

### 13.3 GAP closures (audit items)

- `ad26b53` Fix: GAP-028 wire drift detector and add close_guard_entry CLI (Option B)
- `0a0145c` Fix: normalize flat IBKR ledger schema in positions truth publisher (GAP-A001-adjacent)
- `68da794` Fix: GAP-A001 — BAG paper SELL close path for alpha_options max_hold exit
- `86fb679` Fix: reconcile stale alpha_options SPY guard after GAP-A001
- `723dad9` Fix: GAP-A003 — pnl_untrusted fill-matcher for snapshot-diff closes
- `68ceacb` Fix: GAP-A009 — positions_truth YELLOW when replay scope mismatch
- `651b32c` Fix: GAP-010 centralize IBKR client ID registry
- `39c01c4` Fix: GAP-015/016 add report-only portfolio VaR and drawdown state
- `2d171bf` Fix: GAP-025/026 add routing diagnostics and report-only strategy loss guard
- `3661aef` Fix: GAP-017A add backend healthz liveness endpoint
- `a1ef61f` Ops: document GAP-027 MES stale paper ledger position
- `7040b3b` Ops: document GAP-020 IB Gateway bindaddress maintenance

### 13.4 Bug fixes (execution / reconciliation / data integrity)

- `dbf1842` Remove: Polygon provider dead code — IBKR is sole market data source
- `5f10a0f` Fix: raise local rate limit default 10→30 for all-Haiku fallback mode
- `13ba99c` Fix: make OLLAMA_TIMEOUT_SEC env-configurable; set 2s fast-fail via drop-in
- `d7672f7` Wire: chad-lifecycle-replay-engine systemd unit + timer
- `6b7bc01` Fix: edge_decay_monitor respects quarantine manifests
- `bca8d33` Fix: close_guard_entry livegate check uses allow_ibkr_live not operator_mode
- `ac88ca1` Fix: respect options_multiplier in trade closer PnL
- `cc71755` Fix: mark synthetic alpha_options BAG closes as pnl_untrusted
- `c884a90` Fix: quarantine alpha_options synthetic BAG close SCR contamination
- `f7ebae2` Docs: document lifecycle replay engine missing timer as pending action
- `83543ef` Model: update XGB veto model after quarantine-aware retraining
- `f25dbca` Fix: resolve Pydantic V2 and FastAPI lifespan deprecation warnings
- `56c173d` Fix: block broker-side duplicate working orders before IBKR submit
- `31577cc` Fix: stabilize IBKR idempotency key to block duplicate submissions
- `e9c9954` Fix: skip paper evidence for unconfirmed IBKR order statuses
- `c94673e` Fix: prevent unqualified BAG combo legs from reaching IBKR
- `d99931d` Fix: normalize futures contract months in broker_sync guard keys
- `1ccde6a` Fix: add WEEKEND_GATE_A_FILTERED log for pipeline A/B diagnostic clarity
- `d65a9da` Fix: replace deprecated datetime.utcnow() with timezone-aware equivalent
- `2f4cb8c` Fix: gamma uses valid AssetClass enum
- `98c5bc1` Fix: add MYM to futures spec registry and IBKR adapter
- `a2a5841` Fix: add M2K to futures spec registry and IBKR adapter
- `9e784ee` Fix: add omega_macro futures support to IBKR adapter
- `53fdb98` Fix: omega_macro execution lane — add ZN/ZB/M6E to futures spec registry
- `087a7dc` Fix: separate excluded_validate_only from excluded_untrusted in SCR stats
- `71aa5de` Fix: P1 universe provider prefers live-screened runtime/universe.json
- `eb54773` Fix: show today realized and total paper PnL on dashboard
- `4c34e10` Fix: harden same-side suppression logging and side normalization
- `05c2447` Fix: wire IBKR positions snapshot collector timer
- `a949756` Fix: preserve alpha_options option metadata through execution pipeline (Gap-4)
- `b204337` Fix: quarantine placeholder SPY fills and phantom delta PnL
- `cbded80` Fix: prevent placeholder fills from corrupting trade closer PnL
- `d9936fe` Fix: write paper evidence for alpha_crypto Kraken validate-only fills
- `7ce9fa6` Fix: register ledger watcher IBKR client ID
- `9ccbebb` Fix: cache IBKR qualified contracts to reduce broker timeouts
- `7b58921` Fix: restore correlation overlay producer-consumer path
- `17a6bcc` Fix: publish execution environment from active runtime mode
- `aeb93d2` Fix: resolve futures contract_month before IBKR submission
- `201155b` Fix: guard rebuild skips non-dict entries in position_guard
- `cc0d802` Fix: align live readiness TTL with weekly cadence
- `4129601` Fix: align health monitor with effective halted-strategy caps
- `63cc5b9` Fix: suppress winner boost for halted strategies

### 13.5 Tests

- `aaa438c` Test: add position guard broker-sync drift detector
- `3726212` Test: isolate rebalance tests from live event risk
- `95efbf3` Test: fix event calendar stale datetime fixture

### 13.6 Ops / hygiene

- `72be893` Ops: add SQLite retention pruner and weekly timer (UNK-06)
- `5ff4bba` Ops: P3 hygiene sweep — bak files, stale logs, proof files, shadow copies
- `bfdc348` Fix: P3 batch 3 — remove tracked backup staging clutter
- `0dad936` Fix: normalize IBKR stock contracts with SMART exchange
- `51e7c14` Fix: P2 batch 1 — model versions, CLAUDE.md test count, full-cycle-refresh mode
- `286b786` Ops: remove deprecated requirements.backend.txt
- `7cc00e4` Ops: archive .bak snapshot files from active source tree
- `39ca232` Ops: add authoritative requirements freeze
- `d0f75dc` Fix: formalize reconciliation drift exclusions for paper soak

### 13.7 Features

- `08c14b4` Feature: simulate alpha options BAG paper fills with net debit
- `8d73985` Feature: add Exterminator read-only sentinel Stage 1
- `cf5f0e7` Fix: formalize ALPHA_FOREX as deferred strategy
- `6eac8eb` Fix: guard legacy Polygon paths and update IBKR bar docs

### 13.8 Documentation

- `2cc4a65` Docs: update CLAUDE.md to reflect current system state
- `f31ca86` Docs: update CLAUDE runtime baseline for Paper Epoch 2

---

## 14. Known Issues / Accepted Residual Risks

### 14.1 Carried forward from v9.0

- **Excluded operator positions** (AAPL, BAC, CVX, LLY, MSFT, NVDA, PEP, QQQ, SPY) remain owner=operator and outside CHAD reconciliation. Manual flat-only.
- **Open MES short** (pre-live operator task in CLAUDE.md) — operator review required before any live-mode flip.
- **IB Gateway latency** classified dangerous (>750ms) in some windows — pre-live task pending.
- **Disk usage** below 75 % required before live flip (pre-live task).
- **alpha_intraday daily-bar fallback path** — `alpha_intraday.py` falls back to daily bars when 1-min bars are absent; daily fallback is documented but produces materially different trigger sensitivities. Acceptable for paper.
- **GAP-027 stale paper MES position** — documented in commit a1ef61f as a known stale ledger entry (operator-tracked).

### 14.2 New residual risks (v9.1)

1. **alpha_intraday_micro registered but conditionally inactive at runtime.** The handler is in the canonical registry (`chad/strategies/__init__.py:240`) and is invoked each cycle by `live_execution_router._build_available_signals → iter_strategy_registrations`. However, the handler short-circuits to `[]` when `ctx.tier_profile` is not a `TierRiskProfile` instance. ContextBuilder must populate `tier_profile` and `tier_name` on `MarketContext` for this strategy to emit signals. **Status: REQUIRES CONTEXT-BUILDER WIRING VERIFICATION** before MICRO/STARTER/PRO_GROWTH testing.
2. **SCALE wildcard expansion omits alpha_intraday_micro.** `_CANONICAL_STRATEGY_NAMES` in `chad/risk/tier_manager.py:90` lists 16 names and does not include `alpha_intraday_micro`. Therefore `tier_state.json` `enabled_strategies` under SCALE shows 16 names (no micro). Downstream consumers that gate on `tier_state.enabled_strategies` membership will exclude alpha_intraday_micro at SCALE. Acceptable as a deliberate scoping choice (micro is intended for lower tiers), but should be documented in the next workstream.
3. **EOD flatten service not yet installed to systemd.** Unit files exist (`deploy/chad-micro-eod-flatten.{service,timer}`) but installation status to active systemd is not confirmed by `systemctl list-timers` output; flatten window arithmetic is fully implemented inside alpha_intraday_micro (`_in_flatten_window`), so the service is for cross-cycle EOD position closure not handled by strategy emit logic alone.
4. **Setup expectancy timer not yet installed to systemd.** `deploy/chad-setup-expectancy.{service,timer}` exist but not yet active; `runtime/setup_family_expectancy.json` is therefore [NOT PRESENT — publisher not yet run].
5. **Stop-distance sizing (Phase A Item 1) NOT yet implemented.** Position size is still computed from notional weight chain (dynamic_caps overlays). The standard "size = dollar_risk / stop_distance" risk-per-trade-driven formula is not in place anywhere except inside alpha_intraday_micro where contracts always = 1. All other strategies still use weight-based sizing.
6. **Setup tagging across all strategies (Phase A Item 4) — PARTIALLY DONE.** Only alpha_intraday_micro tags meta.setup_family. Other strategies do not yet emit setup_family in TradeSignal.meta, so the setup_family_expectancy updater observes only one strategy.
7. **Pre-entry R:R gate (Phase A Item 3) NOT yet implemented.** No standalone gate enforces reward-to-risk before entry; strategies emit signals on individual trigger conditions and R:R is implicit in their take_profit/stop_loss meta fields.
8. **Float-aware routing (Phase A Item 5) NOT yet implemented.**
9. **ib_async paper-runner files still on ib_insync (deferred).** `chad/core/paper_position_closer.py` and `chad/core/paper_shadow_runner.py` remain on the legacy ib_insync library. Paper-only path; no live capital risk.
10. **Lifecycle replay engine missing timer install** — per docs commit f7ebae2.

### 14.3 Acceptance criteria

All items above are accepted as residuals under Paper Epoch 2 / Elite Signal Layer Phase A baseline. None block paper operation. Items (5), (6), (7), (8) are tracked under Phase A roadmap (§15).

---

## 15. Phase Roadmap

### 15.1 Track 1 — Paper Epoch 2 → Live promotion (carried from v9.0)

Required for live activation:
- [x] SCR reaches CONFIDENT band — **DONE** (CONFIDENT, sizing 1.0 sustained since 2026-05-09)
- [ ] 14-day clean equity history — in progress (currently 16 days in GROW phase but 11.4 % dd from HWM)
- [ ] 60-day clean paper performance — paper_trades 3873 over Epoch 2; tracking
- [ ] Operator GO — pending operator signoff
- [ ] live_readiness.json flips to ready_for_live=true — currently false
- [ ] All pre-live operator tasks (OS reboot if kernel update; disk < 75 %; IB Gateway latency; MES short review)

### 15.2 Track 2 — Elite Signal Layer (NEW in v9.1)

#### Phase A — Bucket 1 (pure code, no new data)

| Item | Status | Notes |
|---|---|---|
| Item 1 — Stop-distance-driven sizing | **NOT STARTED** | Strategies still size by weight chain; alpha_intraday_micro is the only place TierRiskEnforcer.validate_stop_width is consulted, and it always sizes to 1 contract. |
| Item 2 — Time-of-day session zones | **PARTIALLY DONE — alpha_intraday_micro only** | Primary 09:35–11:00 / Secondary 13:30–15:00 / EOD-flatten 15:30 ET implemented in alpha_intraday_micro. Other intraday strategies (alpha_intraday, omega_momentum_options) have hard cutoffs but no formal session-zone abstraction. |
| Item 3 — Pre-entry R:R gate | **NOT STARTED** | No standalone gate; R:R lives implicitly inside individual strategies' meta fields. |
| Item 4 — Setup tagging across all strategies | **PARTIALLY DONE — micro only** | Only alpha_intraday_micro emits meta.setup_family. Other strategies tag reason / engine / regime but not setup_family. |
| Item 5 — Float-aware routing | **NOT STARTED** | Strategy router doesn't consult float; future bucket-1 item. |

#### Phase B — Bucket 2 (new data feeds)

All items NOT STARTED. Anticipated feeds: short interest deltas, options flow, dark-pool prints, real-time sentiment, futures order-book imbalance.

#### Phase C — Bucket 3 (new exchange connections)

All items NOT STARTED. Anticipated: micro-options venue, alternate crypto exchanges, futures clearing routing.

#### Phase D — Bucket 4 (architecture decisions)

All items NOT STARTED. Anticipated: sub-second execution lane, separate risk allocator per asset class, multi-account routing.

---

## 16. Appendices

### Appendix A — Tier Risk Profiles (full)

(From `config/tiers.json` — schema `tiers.v9_1`.)

| Tier | min_equity | max_equity | demotion | max_contracts | max_risk_trade | max_daily_loss | max_weekly_loss | max_trades_day | primary_session_only | flatten_before_eod | flatten_eod_min | stop_width_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MICRO | $0 | $2,500 | null | 1 | $10 | $20 | $30 | 2 | true | true | 30 | true |
| STARTER | $2,500 | $25,000 | $2,250 | 2 | $40 | $150 | $300 | 4 | true | true | 30 | true |
| PRO_GROWTH | $25,000 | $160,000 | $22,500 | 5 | $200 | $500 | $1,500 | 10 | false | false | null | true |
| SCALE | $160,000 | $10,000,000 | $144,000 | null | null | null | null | null | false | false | null | false |

Enabled strategies and allowed instruments per tier — see §5.2.

### Appendix B — Setup Family Definitions

(From `chad/strategies/alpha_intraday_micro_config.py`.)

```
SESSION_TIMEZONE          = "America/New_York"
PRIMARY_SESSION_START     = "09:35"
PRIMARY_SESSION_END       = "11:00"
SECONDARY_SESSION_START   = "13:30"
SECONDARY_SESSION_END     = "15:00"
HARD_EOD_EXIT_TIME        = "15:30"

OPENING_RANGE_BARS        = 5       (5-min OR computed from first 5 × 1m bars)
OPENING_RANGE_BAR_SIZE    = "1m"

ORB_VOLUME_MULTIPLIER     = 1.3     (volume must be ≥ 1.3× OR average)
ORB_PRIOR_DAY_CLEARANCE_ATR = 1.5
ORB_MAX_EXTENSION_PCT     = 0.01    (max 1% extension past breakout level)
ORB_VALID_UNTIL_MINUTES_AFTER_OPEN = 45

VWAP_MESSY_TEST_COUNT     = 3       (≥3 crosses in last 10 bars → skip)
VWAP_INITIAL_DRIVE_MIN_PCT = 0.003  (require ≥ 0.3% initial drive)

PULLBACK_MAX_RETRACE_RATIO = 0.50   (retrace ≤ 50% of impulse range)
PULLBACK_MIN_TREND_BARS    = 3      (3-bar monotonic trend required)

SWEEP_MAX_BREACH_PCT       = 0.005  (overnight breach ≤ 0.5%)
SWEEP_SNAP_BARS            = 2      (snap-back must occur within 2 bars)
TREND_DAY_EXTENSION_PCT    = 0.015  (skip when extension >1.5%)
SWEEP_VALID_UNTIL_MINUTES_AFTER_OPEN = 60

R_TARGET_PARTIAL           = 1.5
R_TARGET_MAIN              = 2.0

MES_DOLLARS_PER_POINT      = 5.0
MNQ_DOLLARS_PER_POINT      = 2.0

SETUP_PRIORITY = {
    "ORB": 1,
    "VWAP_RECLAIM": 2,
    "VWAP_REJECTION": 2,
    "PULLBACK_CONTINUATION": 3,
    "SWEEP_REVERSAL": 4,
}
```

Deterministic skip reason codes (19 total):
`SKIP_OUTSIDE_PRIMARY_WINDOW`, `SKIP_OUTSIDE_SECONDARY_WINDOW`, `SKIP_EOD_FLATTEN_WINDOW`, `SKIP_INSUFFICIENT_BARS`, `SKIP_ORB_VOLUME_TOO_LOW`, `SKIP_ORB_TOO_EXTENDED`, `SKIP_ORB_NO_CLEARANCE`, `SKIP_VWAP_MESSY`, `SKIP_VWAP_NO_INITIAL_DRIVE`, `SKIP_PULLBACK_TOO_DEEP`, `SKIP_PULLBACK_TREND_NOT_CONFIRMED`, `SKIP_SWEEP_TOO_DEEP`, `SKIP_SWEEP_NO_SNAPBACK`, `SKIP_SWEEP_TREND_DAY`, `SKIP_STOP_TOO_WIDE`, `SKIP_TIER_ENFORCEMENT`, `SKIP_PRIORITY_SUPPRESSED`, `SKIP_DUPLICATE_SIGNAL`, `SKIP_INVALID_CONTRACT_COUNT` (from tier_risk_enforcer).

### Appendix C — Health Monitor Rules R01–R16

From `chad/ops/health_monitor_rules.py`:

| Rule | Title | Severity | Remedy type |
|---|---|---|---|
| R01 | Critical services must be active | CRITICAL | SERVICE_RESTART |
| R02 | Critical feeds within TTL | CRITICAL/WARNING | SERVICE_RESTART (regime_state→NOTIFY_ONLY per SS01) |
| R03 | SCR not PAUSED | CRITICAL | NOTIFY_ONLY |
| R04 | Stop bus not active | CRITICAL | NOTIFY_ONLY |
| R05 | Reconciliation GREEN | CRITICAL | NOTIFY_ONLY |
| R06 | Profit lock escalation (LOCK2/LOCK3/HARD_STOP) | CRITICAL | NOTIFY_ONLY |
| R07 | Disk usage < 75 % (warn) / < 85 % (critical) | WARNING/CRITICAL | SAFE_AUTO (archive_old_fills) |
| R08 | Runtime JSON not zero-byte or corrupt | CRITICAL | SAFE_AUTO (restore_from_backup) |
| R09 | No active edge decay halts | WARNING | NOTIFY_ONLY |
| R10 | High trade churn (>300 trades, PnL < −$500, weekday-only) | CRITICAL | SAFE_AUTO (write_signal_throttle) |
| R11 | Stale reconciliation artifact penalties (>7 days, weekday-only) | WARNING | SAFE_AUTO (clear_reconciliation_artifact) |
| R12 | Alpha cluster degrading (≥3 alphas < 0.5 health, weekday-only) | WARNING | NOTIFY_ONLY |
| R13 | SCR effective-trades gap (raw > 2× effective) | INFO | NOTIFY_ONLY |
| R14 | Halted strategy boost clamped (effective winner_factor ≤ 1.0) | WARNING | NOTIFY_ONLY |
| **R15** | **Tier daily loss budget approaching (budget_remaining ≥ 30 % of max)** | **WARNING** | **NOTIFY_ONLY** |
| **R16** | **Setup family skip rate (skip_count ≤ 2× trades, min 5 trades)** | **WARNING** | **NOTIFY_ONLY** |

Feeds watched by R02: `price_cache.json` (180s TTL), `regime_state.json` (360s), `dynamic_caps.json` (180s), `regime_booster.json` (240s), `kraken_prices.json` (120s), `reconciliation_state.json` (480s), `choppy_regime_state.json` (900s), `macro_state.json` (7200s).

### Appendix D — Verification Sequence

```
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

For CI-friendly invocation:

```
CHAD_SKIP_IB_CONNECT=1 python3 -m pytest chad/tests/ -q --tb=no 2>&1 | tail -3
```

### Appendix E — Rollback commands

Active tags:

- `STABILITY_FREEZE_20260307_GREEN` — original stable baseline
- `PRE_HARDENING_20260402` — snapshot before P0 hardening began
- `RATIFICATION_MASTER_20260402` — all hardening and GAP items complete
- `REVERT_PRE_OVERHAUL_20260419` — snapshot before 2026-04-19/21 overhaul (commit 45f3728)

Rollback to RATIFICATION_MASTER:

```
git checkout RATIFICATION_MASTER_20260402
```

Tarball at `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`.

The v9.0 SSOT can be diffed against this v9.1 SSOT at `docs/CHAD_UNIFIED_SSOT_v9.0_2026-05-04.md` (commit 938ec2c).

---

**End of CHAD Unified SSOT v9.1 — 2026-05-13.**
