# CHAD Unified SSOT v9.2 — Elite Signal Layer Phase A + Phase B Complete

**Version:** 9.2
**Date:** 2026-05-15
**Status:** Active — Paper Epoch 2 / Elite Signal Layer Phases A+B Complete
**Supersedes:** docs/CHAD_UNIFIED_SSOT_v9.1_2026-05-13.md (1465 lines)
**Repository HEAD:** `48d90ec90f559b0dc23d7647ffb4ea29e6b7198c` (short: `48d90ec`)
**Branch:** `main`
**Total commits:** 445
**Test status:** 1,874 passed in 56.99s (`CHAD_SKIP_IB_CONNECT=1`)
**Live readiness:** `ready_for_live: false` (last report 2026-05-16T00:14:08Z)
**SCR posture:** `CONFIDENT` — `sizing_factor=1.0`, `paper_only=false`, sharpe_like=+5.727, effective_trades=196, win_rate=0.7551, total_pnl=+$9,905.13 (snapshot 2026-05-16T00:23:49Z)

---

## 0. Preamble

| Field                       | Value                                                            |
|-----------------------------|------------------------------------------------------------------|
| Document version            | 9.2                                                              |
| Author                      | CHAD engineering (operator + Claude Code)                        |
| Date                        | 2026-05-15                                                       |
| Supersedes                  | CHAD_UNIFIED_SSOT_v9.1_2026-05-13.md                             |
| Output path                 | `docs/CHAD_UNIFIED_SSOT_v9.2_2026-05-15.md`                      |
| Repo HEAD                   | `48d90ec`                                                        |
| Branch                      | `main`                                                           |
| Commits since v9.1 (`ac3f58e`) | 17                                                            |
| Test count                  | 1,874 passing                                                    |
| Server                      | EC2 (Ubuntu 24.x, kernel 6.17.0-1009-aws)                        |
| Canonical root              | `/home/ubuntu/chad_finale`                                       |
| Python                      | `python3` (venv at `/home/ubuntu/chad_finale/venv`)               |
| Execution mode              | `CHAD_EXECUTION_MODE=paper`                                      |
| SCR state                   | `CONFIDENT`, sizing_factor=1.0, paper_only=false                 |
| Account equity              | ~$175,740 USD paper (ibkr) + $185 (kraken) → ≈$175,925           |

### What v9.1 was

v9.1 (2026-05-13) ratified the Tier Architecture v2 rollout (MICRO / STARTER /
PRO_GROWTH / SCALE), the addition of `alpha_intraday_micro` as the 17th
strategy head, the `tier_risk_enforcer` / `tier_instrument_gate` /
`setup_family_expectancy` plumbing, EOD flatten support, health rules R15/R16,
and the GAP-4 meta-forwarding fix. It established the "Elite Signal Layer
Phase A baseline" at 1,652 tests passing.

### What v9.2 captures

Two coherent groups of work landed on top of v9.1, plus a forward-looking
provider scaffolding:

**Phase A — Signal Quality (5 items, all complete)**
1. Stop-distance sizing — `alpha_futures`, `alpha`, `alpha_intraday` now size
   contracts/shares against the tier's `max_risk_per_trade_usd` budget using
   `contracts = tier_max_risk_usd / (ATR × stop_mult × point_value)`, with the
   legacy sizing preserved as a fallback when no tier profile is attached.
2. Session-zone gating — `chad/utils/session.py` is the canonical
   primary/secondary-window classifier. Strategies consult it for **entries
   only**; exits, stops, and position reductions never block on session zones.
3. R:R pre-entry gate — `chad/utils/risk_reward.py` rejects new entries whose
   reward-to-risk ratio falls below 1.5; degenerate or missing target estimates
   fail open (never block a trade).
4. Setup-family tagging — every emitted `TradeSignal.meta` now carries a
   coarse `setup_family` label so the expectancy tracker (v9.1) can group fills
   by setup taxonomy.
5. Float-aware liquidity gate — `chad/utils/liquidity.py` reads daily bars at
   `data/bars/1d/{SYM}.json`, classifies as LARGE / STANDARD / THIN / UNKNOWN
   by 20-day ADDV, and blocks THIN equities below an absolute 0.80 confidence
   floor. UNKNOWN is the fail-open path.

**Phase B — Intelligence Feeds (6 items + FMP scaffolding, all complete)**
1. Catalyst news intelligence — `news_intel.json` produced by
   `news_intel_publisher`, consumed by `catalyst_gate.py` to block trades
   contrary to a HIGH or MEDIUM catalyst when headline relevance is confirmed.
2. Relative strength — `relative_strength.json` produced from existing daily
   bars; `rs_gate.py` provides a ±0.10 confidence modifier (never a hard block).
3. Intraday RVOL — `volume_scan.json` produced by `volume_scan_publisher`
   from a Polygon snapshot or a rolling 1-minute fallback; `rvol_gate.py`
   gives a ±0.05 confidence modifier for `alpha_intraday` only.
4. Crypto derivatives — `crypto_derivatives.json` produced from Kraken Futures
   public tickers; `crypto_signal_filter.py` shaves up to 0.20 confidence when
   the proposed crypto entry is aligned with crowded funding.
5. Futures roll calendar — `futures_roll_state.json` static third-Friday
   quarterly calendar; `roll_gate.py` can block new entries on supported
   quarterly equity-index micros (MES, MNQ, MYM, M2K) during the warning
   window. Unsupported symbols (MCL, MGC, ZN, ZB, M6E) emit informational
   `unsupported_v1` rows and never block.
6. Options Greeks — `options_greeks.json` produced by
   `options_greeks_publisher` from the options chain cache + VIX, exposing
   synthetic Black–Scholes Greeks read by `options_greeks_gate.py` as
   metadata-only annotations on `alpha_options` signals.

**Phase B bonus** — `chad/market_data/fmp_client.py` and
`chad/market_data/market_intel_provider.py` add a Financial Modeling Prep
stable-API scaffold (quote / profile / earnings-calendar / price-target /
analyst-estimates / SEC filings). No publisher migration has happened yet —
the abstraction layer is in place to allow one-at-a-time migrations later.

### What v9.2 is NOT

This document does **not**:
- Authorize live trading. `live_readiness.json` still reads
  `ready_for_live: false` and the LIVE_GATE remains paper-only by
  configuration. The four pre-live operator tasks (reboot when next kernel
  update lands, disk cleanup, IBKR latency investigation, full 1874-test
  re-verification after reboot) are unchanged from v9.1.
- Modify the canonical risk caps, the strategy config, or any systemd unit.
- Touch any `runtime_FREEZE_*` or `data_FREEZE_*` artifact.

### Server / repo / mode

- Host: EC2 (Ubuntu 24.x); kernel 6.17.0-1009-aws current — no pending
  kernel update as of 2026-04-21 (reboot deferred to the next real update).
- Repo: `/home/ubuntu/chad_finale` (git, branch `main`).
- Python: `python3` via `/home/ubuntu/chad_finale/venv` (never `python`).
- Execution mode: `CHAD_EXECUTION_MODE=paper`.

---

## 1. Mission & Architecture

### 1.1 Mission

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a fully autonomous
multi-strategy paper-trading system whose mandate is to compound a small seed
balance into operator salary. v9.2 advances the system from "validated paper
engine" toward "elite-grade signal generation" by layering Phase A signal-
quality gates and Phase B intelligence feeds on top of the existing
broker-truth-anchored execution and reconciliation core.

### 1.2 Architecture (v9.2)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INTELLIGENCE LAYER                              │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐    │
│  │ Regime      │  │ Event Risk  │  │ Macro       │  │ Choppy       │    │
│  │ regime_     │  │ event_      │  │ macro_      │  │ choppy_      │    │
│  │ state.json  │  │ risk.json   │  │ state.json  │  │ regime.json  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘    │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐    │
│  │ News Intel  │  │ Rel.        │  │ Volume Scan │  │ Crypto       │    │
│  │ (Phase B1)  │  │ Strength    │  │ (Phase B3)  │  │ Derivatives  │    │
│  │ news_intel  │  │ (Phase B2)  │  │ volume_     │  │ (Phase B4)   │    │
│  │ .json       │  │ relative_   │  │ scan.json   │  │ crypto_      │    │
│  │             │  │ strength.j  │  │             │  │ derivatives.j│    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘    │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ Futures     │  │ Options     │  │ Short Interest / Reddit /       │  │
│  │ Roll (B5)   │  │ Greeks (B6) │  │ Trends / Setup Expectancy /     │  │
│  │ futures_    │  │ options_    │  │ Strategy Intelligence (legacy)  │  │
│  │ roll_state  │  │ greeks.json │  │                                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL GENERATION                                │
│                                                                          │
│   17 strategy heads (alpha, alpha_intraday, alpha_intraday_micro,        │
│   alpha_futures, alpha_crypto, alpha_options, beta, beta_trend,          │
│   gamma, gamma_futures, gamma_reversion, delta, delta_pairs,             │
│   omega, omega_macro, omega_momentum_options, omega_vol)                 │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐  │
│   │ ENTRY GATE CHAIN (Phase A + Phase B — entry-only, fail-open)      │  │
│   │ ─────────────────────────────────────────────────────────────────  │  │
│   │  1. Session zone        chad/utils/session.py                     │  │
│   │  2. Stop-distance size  tier_risk_enforcer + strategy hook        │  │
│   │  3. R:R                 chad/utils/risk_reward.py    (≥1.5)        │  │
│   │  4. Liquidity (eq)      chad/utils/liquidity.py      (THIN <0.80)  │  │
│   │  5. Catalyst (eq)       chad/utils/catalyst_gate.py                │  │
│   │  6. RS modifier (eq)    chad/utils/rs_gate.py        (±0.10)       │  │
│   │  7. RVOL modifier       chad/utils/rvol_gate.py      (±0.05) [INTR]│  │
│   │  8. Crypto crowding     chad/utils/crypto_signal_filter.py [CRYP] │  │
│   │  9. Roll gate           chad/utils/roll_gate.py            [FUT]  │  │
│   │ 10. Greeks metadata     chad/utils/options_greeks_gate.py  [OPT]  │  │
│   └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO / EXECUTION HOT PATH                        │
│                                                                          │
│  Orchestrator → Net Exposure Gate → Strategy Throttle Gate →             │
│  Per-symbol daily loss limit → Risk allocator → IBKR adapter →           │
│  paper_exec_evidence_writer → trade_closer → reconciliation publisher    │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  GOVERNANCE / OBSERVABILITY                              │
│                                                                          │
│  LiveGate · SCR · Profit Lock · Stop Bus · Tier Manager ·                │
│  Position Guard + Drift Detector · Withdrawal Manager · Business Phase · │
│  Telegram operator interface · Dashboard (nginx + Flask)                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Strategy heads (17)

| #  | Strategy                       | Asset class    | Primary gate set                             |
|----|--------------------------------|----------------|----------------------------------------------|
| 1  | `alpha`                        | Equities/ETFs  | Session, sizing, R:R, liq, catalyst, RS      |
| 2  | `alpha_intraday`               | Equities/ETFs  | Session, sizing, R:R, liq, catalyst, RS, RVOL |
| 3  | `alpha_intraday_micro`         | MES (micro)    | Session(primary-only), tier risk, stop width  |
| 4  | `alpha_futures`                | Index micros   | Session, sizing, R:R, roll gate              |
| 5  | `alpha_crypto`                 | BTC/ETH/SOL    | Crypto crowding filter                       |
| 6  | `alpha_options`                | Options (vert) | Greeks metadata                              |
| 7  | `beta`                         | Equities       | Existing v9.1 gates                          |
| 8  | `beta_trend`                   | Trend-follow   | Existing v9.1 gates                          |
| 9  | `gamma`                        | Mean-rev       | Existing v9.1 gates                          |
| 10 | `gamma_futures`                | Futures MR     | Existing v9.1 gates                          |
| 11 | `gamma_reversion`              | Reversion      | Existing v9.1 gates                          |
| 12 | `delta`                        | Cross-asset    | Existing v9.1 gates                          |
| 13 | `delta_pairs`                  | Stat-arb       | Existing v9.1 gates                          |
| 14 | `omega`                        | Macro overlay  | Existing v9.1 gates                          |
| 15 | `omega_macro`                  | Macro          | Existing v9.1 gates                          |
| 16 | `omega_momentum_options`       | Options-mom    | Existing v9.1 gates                          |
| 17 | `omega_vol`                    | Vol overlay    | Existing v9.1 gates                          |

Note: the v8.x→v9.1 transition moved `alpha_intraday_micro` from a config-driven
override into a canonical strategy head; v9.2 makes no further roster changes.

---

## 2. Current Runtime Truth — 2026-05-15

All values below are read directly from `runtime/*.json` at v9.2 cut time.

### 2.1 Repository / tests

| Field                         | Value                                       |
|-------------------------------|---------------------------------------------|
| HEAD (full)                   | `48d90ec90f559b0dc23d7647ffb4ea29e6b7198c`  |
| HEAD (short)                  | `48d90ec`                                   |
| Branch                        | `main`                                      |
| `git status --short`          | clean                                       |
| Total commits                 | 445                                         |
| Commits since v9.1 baseline   | 17 (from `ac3f58e`)                         |
| Test count                    | 1,874 passing                               |
| Test runtime                  | 56.99 s (`CHAD_SKIP_IB_CONNECT=1`)          |

### 2.2 SCR — `runtime/scr_state.json`

| Field            | Value                          |
|------------------|--------------------------------|
| state            | `CONFIDENT`                    |
| sizing_factor    | 1.0                            |
| paper_only       | false                          |
| effective_trades | 196                            |
| live_trades      | 0                              |
| paper_trades     | 3,884                          |
| sharpe_like      | +5.727005                      |
| win_rate         | 0.7551 (75.51%)                |
| total_pnl        | +$9,905.125                    |
| max_drawdown     | -$290.0                        |
| excluded_manual  | 6                              |
| excluded_nonfinite | 0                            |
| excluded_untrusted | 321                          |
| paused_recovery_ticks | 0                         |
| reason           | "CONFIDENT: win_rate, sharpe, and drawdown all within confident band." |
| ts_utc           | 2026-05-16T00:23:49Z           |
| ttl_seconds      | 180                            |

### 2.3 Live readiness — `runtime/live_readiness.json`

| Field            | Value                                                   |
|------------------|---------------------------------------------------------|
| ready_for_live   | **false**                                               |
| latest_report    | `LIVE_READINESS_20260516T001408Z.json`                  |
| sha256           | `3d584784da70abc076f34a3215c7d158f5baa561922ce21fd5144032564c3f29` |
| ts_utc           | 2026-05-16T00:14:08Z                                    |
| ttl_seconds      | 604800                                                  |

### 2.4 Regime — `runtime/regime_state.json`

| Field            | Value                                                              |
|------------------|--------------------------------------------------------------------|
| regime           | `trending_bull`                                                    |
| previous_regime  | `trending_bull`                                                    |
| confidence       | 0.74502                                                            |
| ok               | true                                                               |
| inputs_used      | realized_vol_percentile, adx, trend_slope, market_breadth          |
| source           | `live_loop.run_once`                                               |
| choppy_overlay.active | false                                                         |
| choppy_overlay.score  | 0.2                                                           |
| choppy_overlay.sizing_multiplier | 1.0                                                |
| choppy_overlay.confidence_floor_add | 0.0                                             |
| ts_utc           | 2026-05-16T00:22:40Z                                               |
| ttl_seconds      | 360                                                                |

### 2.5 Choppy overlay — `runtime/choppy_regime_state.json`

| Field            | Value                                                              |
|------------------|--------------------------------------------------------------------|
| choppy_active    | false                                                              |
| choppy_score     | 0.2                                                                |
| consecutive_clean_reads | 1,501                                                       |
| consecutive_choppy_reads | 0                                                          |
| proxy_symbol     | SPY                                                                |
| indicators.adx   | 36.65 (not weak)                                                   |
| indicators.direction_flips_5d | 1 (low)                                               |
| indicators.failed_breakouts_10d | 2 (above the high threshold)                        |
| indicators.trend_followthrough_rate | 0.571                                           |
| thresholds.enter_threshold | 0.55                                                     |
| thresholds.exit_threshold  | 0.35                                                     |
| ts_utc           | 2026-05-16T00:22:34Z                                               |
| ttl_seconds      | 300                                                                |

### 2.6 Profit Lock — `runtime/profit_lock_state.json`

| Field                     | Value                                          |
|---------------------------|------------------------------------------------|
| mode                      | NORMAL                                         |
| profit_lock_active        | false                                          |
| sizing_factor             | 1.0                                            |
| stop_new_entries          | false                                          |
| daily_loss_today          | $0.00                                          |
| daily_loss_limit_pct      | 3.0%                                           |
| daily_loss_limit_dollars  | $5,272.19                                      |
| daily_loss_limit_hit      | false                                          |
| account_equity (input)    | $175,739.51                                    |
| equity_source             | `runtime/dynamic_caps.json`                    |
| thresholds (warn/lock1/lock2/lock3/hard) | 1.5 / 3.0 / 5.0 / 8.0 / 10.0 %  |
| factors (lock1/lock2/lock3/hard) | 0.5 / 0.25 / 0.1 / 0.0                  |
| ts_utc                    | 2026-05-16T00:23:51Z                           |
| ttl_seconds               | 60                                             |

### 2.7 Stop Bus — `runtime/stop_bus.json`

| Field        | Value                                       |
|--------------|---------------------------------------------|
| active       | false                                       |
| cleared_at   | 2026-04-22T01:56:50Z                        |
| cleared_by   | `smoke_test`                                |
| triggered_at | (empty)                                     |

### 2.8 Reconciliation — `runtime/reconciliation_state.json`

| Field                        | Value                                                 |
|------------------------------|-------------------------------------------------------|
| status                       | **GREEN**                                             |
| broker_source                | `ibkr:clientId=83`                                    |
| chad_state_source            | `position_guard.json`                                 |
| counts.chad_open             | 15                                                    |
| counts.chad_strategy_open    | 14                                                    |
| counts.broker_positions      | 15                                                    |
| worst_diff                   | 0.0                                                   |
| mismatches                   | [] (empty)                                            |
| drifts                       | [] (empty)                                            |
| excluded_symbols             | AAPL, CVX, NVDA, PEP, QQQ                             |
| futures_excluded_symbols     | M2K, MCL, MES                                         |
| exclusion_policy keys        | AAPL, MSFT, NVDA, BAC, CVX, LLY, PEP, QQQ, SPY        |
| ts_utc                       | 2026-05-16T00:22:38Z                                  |
| ttl_seconds                  | 360                                                   |

### 2.9 Portfolio snapshot — `runtime/portfolio_snapshot.json`

| Field            | Value                |
|------------------|----------------------|
| ibkr_equity      | $175,554.92          |
| coinbase_equity  | $0.00                |
| kraken_equity    | $184.58              |
| total (derived)  | $175,739.51          |
| ts_utc           | 2026-05-16T00:22:40Z |
| ttl_seconds      | 300                  |

### 2.10 Tier state — `runtime/tier_state.json`

| Field                    | Value                                                  |
|--------------------------|--------------------------------------------------------|
| schema_version           | `tier_state.v2`                                        |
| tier_name                | **SCALE**                                              |
| current_equity_usd       | $175,739.51                                            |
| tier_min_equity          | $160,000                                               |
| tier_max_equity          | $10,000,000                                            |
| previous_tier            | SCALE (no recent demotion)                             |
| demotion_pending         | false                                                  |
| risk_profile fields      | all `null` / false (SCALE = unrestricted)              |
| primary_session_only     | false                                                  |
| flatten_before_eod       | false                                                  |
| stop_width_gate_enabled  | false                                                  |
| promoted_at_utc          | 2026-04-27T19:00:35Z                                   |
| ts_utc                   | 2026-05-16T00:22:34Z                                   |
| enabled_strategies (17)  | alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_intraday_micro, alpha_options, beta, beta_trend, delta, delta_pairs, gamma, gamma_futures, gamma_reversion, omega, omega_macro, omega_momentum_options, omega_vol |

### 2.11 Business phase — `runtime/business_phase.json`

| Field                    | Value                                  |
|--------------------------|----------------------------------------|
| phase                    | **GROW**                               |
| current_equity_usd       | $175,739.51                            |
| seed_capital_usd         | $50,000.00                             |
| growth_pct_from_seed     | 251.48%                                |
| days_in_phase            | 19                                     |
| high_water_mark_usd      | $183,874.27                            |
| total_return_pct         | 251.48%                                |
| annualized_return_pct    | 1,000.0% (annualization placeholder)   |
| days_active              | 18                                     |
| next_phase_requirement   | Recover above HWM ($183,874.27) to enter PAY |
| ts_utc                   | 2026-05-16T00:20:41Z                   |

### 2.12 Withdrawal authorization — `runtime/withdrawal_authorization.json`

| Field                       | Value                                                  |
|-----------------------------|--------------------------------------------------------|
| phase                       | GROW                                                   |
| current_equity_usd          | $166,213.84                                            |
| high_water_mark_usd         | $183,874.27                                            |
| drawdown_from_hwm_pct       | 9.60%                                                  |
| spendable_surplus_usd       | $0.00                                                  |
| authorized_withdrawal_usd   | $0.00                                                  |
| scr_state                   | CONFIDENT                                              |
| history_days                | 18                                                     |
| reason                      | "GROW phase override: 30d drawdown 9.6% exceeds veto threshold 5.0%. No salary during drawdown." |
| ts_utc                      | 2026-05-15T19:03:06Z                                   |

Note: `withdrawal_authorization.current_equity_usd` ($166.2k) is the 30-day
trough used for the salary calculation; `business_phase.current_equity_usd`
($175.7k) is the live mark used for tier/sizing decisions. The two values are
intentionally different and update on different cadences.

### 2.13 Macro state — `runtime/macro_state.json`

| Field                          | Value                                       |
|--------------------------------|---------------------------------------------|
| risk_label                     | `risk_on`                                   |
| composite_risk_label           | `moderate`                                  |
| curve_slope_10y_2y             | +0.470                                      |
| us_10y / us_2y                 | 4.47% / 4.00%                               |
| unemployment_rate_pct          | 4.3%                                        |
| cpi_yoy_pct                    | 3.947% (elevated)                           |
| high_yield_spread_pct          | 2.76%                                       |
| credit_stress                  | false                                       |
| inflation_elevated             | **true**                                    |
| recession_risk                 | false                                       |
| yield_curve_inverted           | false                                       |
| provider                       | `FredYieldProvider`                         |
| series fetched                 | DGS2, DGS10, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y |
| ts_utc                         | 2026-05-16T00:16:45Z                        |
| ttl_seconds                    | 1800                                        |

### 2.14 Event risk — `runtime/event_risk.json`

| Field                  | Value                                                |
|------------------------|------------------------------------------------------|
| elevated_risk          | false                                                |
| severity               | low                                                  |
| risk_score             | 0.0                                                  |
| next_event             | Non-Farm Payrolls (high) — 2026-05-30T12:30Z (≈348.6h) |
| operator_calendar_present | true                                              |
| provider               | `EconomicCalendarRiskProvider`                       |
| rule_events / operator_events / merged | 10 / 5 / 15                          |
| ts_utc                 | 2026-05-15T23:56:41Z                                 |
| ttl_seconds            | 1800                                                 |
| windows ahead (sample) | FOMC 2026-06-17/18, CPI 2026-06-15, NFP 2026-06-05   |

### 2.15 News intel — `runtime/news_intel.json` (Phase B Item 1)

| Field                          | Value                                |
|--------------------------------|--------------------------------------|
| schema_version                 | `news_intel.v1`                      |
| status                         | ok                                   |
| symbols_processed              | 25                                   |
| symbols_with_catalyst          | 3 (BAC, LLY, TLT)                    |
| high_strength_count            | 3                                    |
| provider primary / fallback    | polygon / yahoo                      |
| provider_breakdown             | polygon=5, yahoo=20, none=0          |
| ts_utc                         | 2026-05-16T00:14:11Z                 |
| ttl_seconds                    | 1800                                 |

Catalysts (active): BAC bearish/high, LLY bullish/high (earnings, ratings),
TLT bearish/high. All carry `confirmed_gate_relevant=true`; symbols whose
relevance is `unknown` or `broad_market` keep the gate fail-open even when a
headline is present.

### 2.16 Relative strength — `runtime/relative_strength.json` (Phase B Item 2)

| Field                          | Value                                       |
|--------------------------------|---------------------------------------------|
| schema_version                 | `relative_strength.v1`                      |
| market_direction               | `up`                                        |
| benchmark_qqq_return_5d        | +3.5758%                                    |
| benchmark_spy_return_5d        | +2.2677%                                    |
| symbols_computed               | 23                                          |
| strong_count / neutral_count / weak_count / unknown_count | 3 / 5 / 17 / 0     |
| strong symbols                 | AVGO, NVDA, UNH                             |
| provider                       | `daily_bars` (`data/bars/1d`)               |
| lookback_days                  | 5                                           |
| ts_utc                         | 2026-05-15T12:52:35Z                        |
| ttl_seconds                    | 90,000                                      |

### 2.17 Futures roll — `runtime/futures_roll_state.json` (Phase B Item 5)

| Field                  | Value                                              |
|------------------------|----------------------------------------------------|
| schema_version         | `futures_roll_state.v1`                            |
| status                 | ok                                                 |
| symbols_tracked        | 9                                                  |
| supported_count        | 4 (MES, MNQ, MYM, M2K)                             |
| unsupported_count      | 5 (MCL, MGC, ZN, ZB, M6E)                          |
| roll_warning_count     | 0                                                  |
| roll_critical_count    | 0                                                  |
| blocked_symbols        | [] (none)                                          |
| MES / MNQ / MYM / M2K  | current_expiry=2026-06-19, days_to_expiry=35       |
| provider               | `static_cme_calendar`                              |
| ts_utc                 | 2026-05-15T21:50:33Z                               |
| ttl_seconds            | 86,400                                             |

### 2.18 Crypto derivatives — `runtime/crypto_derivatives.json` (Phase B Item 4)

| Field                   | Value                                              |
|-------------------------|----------------------------------------------------|
| schema_version          | `crypto_derivatives.v1`                            |
| provider                | `kraken_futures_public` (tickers endpoint)         |
| symbols_fetched         | 3 (BTC-USD, ETH-USD, SOL-USD)                      |
| long_crowded_count      | 3                                                  |
| short_crowded_count     | 0                                                  |
| BTC-USD market_bias     | `long_crowded` (funding_8h=27.04%, last=$79,116)   |
| ETH-USD market_bias     | `long_crowded` (funding_8h=1.80%, last=$2,226)     |
| SOL-USD market_bias     | `long_crowded` (funding_8h=0.12%, last=$89.26)     |
| ts_utc                  | 2026-05-16T00:22:35Z                               |
| ttl_seconds             | 600                                                |

Phase B4 has BTC-USD funding flagged `funding_extreme_long=true` (the
threshold is 0.0003 / 8h while the reading is 0.27044) — `alpha_crypto`
sees `-0.20` applied to BUY confidence on all three names this cycle.

### 2.19 Strategy health samples

`SCALE` tier is unrestricted, so the active universe = `config/universe.json`.
A representative recent fill picture (paper):

- Sharpe-like (effective): +5.727
- Win rate: 75.51% on 196 effective trades
- Open positions (chad_open): 15
- Broker positions (excluded + active): 15
- Worst broker-vs-state diff: 0.0

---

## 3. Strategies — All 17

For every strategy head, the table below documents the Phase A/B integration
points added in v9.2. "Existing v9.1 gates" means the strategy still benefits
from the v9.1 stack (tier_risk_enforcer, tier_instrument_gate, setup-family
expectancy, SCR throttle, profit lock, etc.) and has no new v9.2 surface.

### 3.1 `alpha`  (`chad/strategies/alpha.py`)

Asset: equities / ETFs.
v9.2 deltas:
- Phase A1: stop-distance sizing — risk budget `tier_max_risk_usd` /
  (stop_per_share) produces the share count when a tier profile is attached.
- Phase A2: session-zone gate — `session_decision(...)` consulted at line 255;
  `tier_profile.primary_session_only` decides whether secondary-window
  entries are allowed.
- Phase A3: R:R gate — `passes_rr_gate(target_pts, stop_pts)` at line 368.
  Fail-open when target/stop are degenerate.
- Phase A4: `setup_family` emitted in `meta` (line 449) via
  `_setup_family_for_alpha(reason, side, regime)`.
- Phase A5: liquidity gate — `blocks_thin_entry(sym, confidence, ...)` at
  line 377; THIN symbols are blocked below the 0.80 confidence floor.
- Phase B1: catalyst gate — `check_catalyst_gate(sym, side)` at line 396;
  blocks BUY-vs-bearish-HIGH and SELL-vs-bullish-HIGH (and MEDIUM symmetric).
- Phase B2: RS adjustment — `get_rs_adjustment(sym, side)` produces a
  ±0.10 confidence modifier (never a hard block).
- Meta fields added: `setup_family`, `stop_distance_pts`,
  `stop_distance_usd`, `rr_gate`, `liquidity_class`,
  `liquidity_required_confidence`, `catalyst_strength`, `catalyst_direction`,
  `catalyst_gate`.

### 3.2 `alpha_intraday`  (`chad/strategies/alpha_intraday.py`)

Asset: equities / ETFs (intraday).
v9.2 deltas:
- Phase A1: stop-distance sizing via `_size_for(sym, confidence, atr, tier_max_risk_usd)` at line 189; `tier_max_risk_usd` activates the stop-aligned path.
- Phase A2: session-zone gate (line 468). Fail-open if `session_decision` raises.
- Phase A3: R:R gate (line 246).
- Phase A4: `setup_family` = `trigger` (the bar trigger reason).
- Phase A5: liquidity gate (line 253).
- Phase B1: catalyst gate (line 266).
- Phase B2: RS adjustment.
- Phase B3: RVOL adjustment via `get_rvol_adjustment(sym)` (only consumer).
- Meta fields added (parallel to alpha): `setup_family`, `stop_distance_pts`,
  `stop_distance_usd`, `catalyst_strength`, `catalyst_direction`,
  `catalyst_gate`, plus the RS/RVOL adjusted confidence outputs.

### 3.3 `alpha_intraday_micro`  (`chad/strategies/alpha_intraday_micro.py`)

Asset: MES micro futures (single-instrument MICRO tier strategy).
v9.2 deltas: none beyond the v9.1 baseline. The strategy already uses its
own primary-session-only window logic that
`chad/utils/session.py` was modeled on, so the shared utility is now the
canonical source of those constants.

### 3.4 `alpha_futures`  (`chad/strategies/alpha_futures.py`)

Asset: index micro futures (MES, MNQ, MYM, M2K plus MCL/MGC observability).
v9.2 deltas:
- Phase A1: `_compute_contract_size(...)` at line 262 — implements
  `contracts = tier_max_risk_usd / (stop_pts × point_value)` with the legacy
  contract sizer preserved as fallback. Outputs `(contracts, alloc_wt,
  risk_budget_usd, risk_per_contract_usd)`.
- Phase A2: session-zone gate at line 716 (with try/except fail-open).
- Phase A3: R:R gate (line 560).
- Phase A4: `_setup_family_for_alpha_futures(side, breakout=...)` at line 243.
- Phase B5: roll gate — `check_roll_gate(symbol)` at line 567. Supported
  quarterly symbols can be hard-blocked inside the 5-day warning window;
  unsupported symbols fail open.
- Meta fields added: `setup_family`, `stop_distance_pts`,
  `stop_distance_usd`, plus the legacy futures fields.

### 3.5 `alpha_crypto`  (`chad/strategies/alpha_crypto.py`)

Asset: BTC-USD, ETH-USD, SOL-USD (perp-style direction via spot/Kraken).
v9.2 deltas:
- Phase B4: `get_crypto_filter(symbol, side)` at line 734. Confidence is
  shaved by 0.20 when proposed BUY is into `long_crowded` (or SELL into
  `short_crowded`); 0.05 penalty for `long_leaning`. Never a hard block.
- Meta fields added on emit: `crypto_market_bias`, `crypto_funding_extreme`.

### 3.6 `alpha_options`  (`chad/strategies/alpha_options.py`)

Asset: vertical-spread option structures on equity/ETF underlyings.
v9.2 deltas:
- Phase B6: `get_option_greeks(symbol, expiry, strike, option_type)` at
  line 399 (with a stand-in dataclass shim if the gate is unavailable).
- Meta fields added: `long_delta`, `short_delta`, `net_delta_estimate`,
  `long_theo_price`, `short_theo_price`, `long_delta_source`,
  `short_delta_source`. The metadata is informational only; it does not
  participate in sizing or in the entry decision.

### 3.7 `beta`, `beta_trend`, `gamma`, `gamma_futures`, `gamma_reversion`, `delta`, `delta_pairs`, `omega`, `omega_macro`, `omega_momentum_options`, `omega_vol`

Asset classes vary. No new Phase A/B gates wire into these strategies in
v9.2. They continue to use the v9.1 tier_risk_enforcer + tier_instrument_gate
+ setup_family_expectancy stack, the SCR throttle, the Profit Lock, the Stop
Bus, and the Net Exposure / Strategy Throttle gates on the orchestrator's
side. The shared `chad/utils/session.py` will pick them up automatically as
they migrate, but no per-strategy wiring is in place yet.

---

## 4. Execution Pipeline

The pipeline is identical to v9.1 at the orchestrator layer (the broker-truth
anchor, paper exec writer, trade closer, and reconciliation publisher are
unchanged). The signal-generation layer in front of the orchestrator has
become substantially richer.

### 4.1 Entry gate chain (entry-only, in canonical order)

Each gate listed below applies to **new** entries only. Exits, position
reductions, and stop emissions never pass through this chain.

1. **Session zone gate** — `chad/utils/session.py::session_decision(...)`
   * Canonical America/New_York source of `PRIMARY` (09:35–11:00) and
     `SECONDARY` (13:30–15:00) windows + 15:30 hard EOD flatten boundary.
   * `tier_profile.primary_session_only=true` (MICRO + STARTER tiers) blocks
     SECONDARY-window entries.
   * Pure utility, stdlib only, test-injectable `now` parameter.
2. **Stop-distance sizing** — `tier_risk_enforcer` + strategy hook.
   * When a tier profile attaches `max_risk_per_trade_usd`, contract /
     share count solves `risk_per_unit × units = max_risk_per_trade_usd`.
   * Fallback path preserves legacy proportional sizing when no tier budget
     is present (e.g., SCALE tier).
3. **R:R pre-entry gate** — `chad/utils/risk_reward.py::passes_rr_gate`
   * Minimum ratio: `MIN_RR_RATIO = 1.5`.
   * Fail-open: degenerate target/stop (≤0) returns ratio=None → True.
4. **Liquidity gate** — `chad/utils/liquidity.py::blocks_thin_entry`
   * LARGE (ADDV ≥ $500M) / STANDARD (≥ $50M) / THIN / UNKNOWN.
   * THIN symbols block when `confidence < THIN_MIN_CONFIDENCE (0.80)`.
   * Per-(symbol, repo_root) cache; UNKNOWN classifications always pass.
   * Equities/ETFs only (futures/crypto pre-empt the call).
5. **Catalyst gate** — `chad/utils/catalyst_gate.py::check_catalyst_gate`
   * Reads `runtime/news_intel.json`. Fail-open on missing/stale/unparseable.
   * Blocks BUY-into-bearish-HIGH and SELL-into-bullish-HIGH (also MEDIUM
     symmetric). Requires `confirmed_gate_relevant=true` — without an
     explicit headline-relevance confirmation, the gate fails open even when
     a record exists.
   * Equities/ETFs only.
6. **RS confidence modifier** — `chad/utils/rs_gate.py::get_rs_adjustment`
   * Reads `runtime/relative_strength.json`. Fail-open.
   * BUY + rs_class=`weak` + market=`up` → -0.10 confidence.
   * SELL + rs_class=`strong` + market=`down` → -0.10 confidence.
   * Never a hard block.
7. **RVOL confidence modifier** — `chad/utils/rvol_gate.py::get_rvol_adjustment`
   * Reads `runtime/volume_scan.json`. Fail-open. **alpha_intraday only.**
   * `rvol_class=high` → +0.05; `low` → -0.05; everything else 0.0.
8. **Crypto crowding filter** —
   `chad/utils/crypto_signal_filter.py::get_crypto_filter`
   * Reads `runtime/crypto_derivatives.json`. Fail-open. **alpha_crypto only.**
   * BUY + `long_crowded` → -0.20; SELL + `short_crowded` → -0.20;
     BUY + `long_leaning` → -0.05.
   * Never a hard block; metadata-only `funding_extreme` flag for telemetry.
9. **Roll gate** — `chad/utils/roll_gate.py::check_roll_gate`
   * Reads `runtime/futures_roll_state.json`. Fail-open. **alpha_futures only.**
   * Blocks new entries when `block_new_entries=true` and
     `roll_supported=true`. Unsupported symbols and missing/stale files
     never block.
10. **Options Greeks metadata** —
    `chad/utils/options_greeks_gate.py::get_option_greeks`
    * Reads `runtime/options_greeks.json`. Fail-open with shaped defaults
      (call=+0.5, put=-0.5). **alpha_options only.**
    * No sizing or block effect — annotates `TradeSignal.meta` only.

### 4.2 Downstream orchestrator gates (unchanged from v9.1)

* Net Exposure Gate
* Strategy Throttle Gate
* Per-symbol daily loss limit
* SCR throttle (`sizing_factor` ∈ {1.0, 0.5, 0.25, 0.0})
* ML veto (mature strategies)
* Profit Lock (lock1/lock2/lock3/hard) — currently NORMAL
* Stop Bus — currently inactive
* Tier Risk Enforcer / Tier Instrument Gate
* LiveGate operator intent (paper-only by configuration)

### 4.3 Reconciliation cycle (unchanged from v9.1)

The Position Guard rebuilder (`chad/core/live_loop.py:383`) intentionally
does NOT consult `reconciliation_state.exclusion_policy` (GAP-028 Option B —
PERMISSIVE). The drift detector
(`chad.core.position_guard.detect_guard_vs_broker_truth_drift`) is wired into
`chad-reconciliation-publisher` and emits `runtime/position_guard_drift.json`
every cycle. Operators close stale entries with `scripts/close_guard_entry.py`.

---

## 5. Risk & Governance

### 5.1 Tier system v2 — `config/tiers.json` (schema `tiers.v9_1`)

Hysteresis: 5.0% (a tier change cannot toggle back inside this band).

| Tier        | Equity band       | Demote at | Strategies                                       | Max risk/trade | Daily loss | Trades/day | Primary-only | EOD flatten | Stop-width gate |
|-------------|-------------------|-----------|--------------------------------------------------|----------------|------------|------------|--------------|-------------|-----------------|
| MICRO       | $0–$2,500         | n/a       | alpha_intraday_micro                             | $10            | $20        | 2          | true         | true (30m)  | true            |
| STARTER     | $2,500–$25,000    | $2,250    | +alpha_intraday, beta_trend, alpha_crypto         | $40            | $150       | 4          | true         | true (30m)  | true            |
| PRO_GROWTH  | $25,000–$160,000  | $22,500   | +alpha, alpha_futures, beta, gamma_futures, delta, omega_macro | $200 | $500 | 10 | false | false | true |
| **SCALE**   | $160,000–$10,000,000 | $144,000 | all 17 (`*`)                                   | unrestricted   | unrestricted | unrestricted | false      | false       | false           |

Current tier: **SCALE** (since 2026-04-27T19:00:35Z, equity $175,739).

### 5.2 Stop-distance sizing formula (Phase A Item 1)

For futures:

```
contracts = tier_max_risk_usd / (stop_pts × point_value)
```

For equities/ETFs:

```
shares = tier_max_risk_usd / stop_per_share
```

Where:
* `tier_max_risk_usd` comes from `tier_state.risk_profile.max_risk_per_trade_usd`.
* `stop_pts` is derived from ATR × `stop_mult` (the strategy's chosen multiple).
* `point_value` is the contract specification (MES=5, MNQ=2, MYM=0.5, M2K=5,
  MCL=1000, MGC=10).

When `tier_max_risk_usd` is `None` (SCALE tier), the strategy falls back to the
legacy sizing path, preserving v9.1 behavior exactly. This is verified by the
Phase A regression tests in `chad/tests/`.

### 5.3 Profit Lock — `chad/risk/profit_lock.py`

(Unchanged from v9.1.)

| Mode      | Trigger (PnL % of equity) | sizing_factor | Allows entries |
|-----------|---------------------------|---------------|----------------|
| NORMAL    | < +1.5%                   | 1.0           | yes            |
| WARN      | +1.5% to +3.0%            | 1.0           | yes            |
| LOCK1     | +3.0%                     | 0.5           | yes            |
| LOCK2     | +5.0%                     | 0.25          | yes            |
| LOCK3     | +8.0%                     | 0.1           | yes            |
| HARD_STOP | +10.0% or -3.0% daily     | 0.0           | no             |

### 5.4 SCR — `runtime/scr_state.json`

(Unchanged from v9.1.)

| State        | sizing_factor | paper_only           | Conditions                                      |
|--------------|---------------|----------------------|-------------------------------------------------|
| CAUTIOUS     | 0.5           | true                 | early epoch, insufficient evidence              |
| CONFIDENT    | 1.0           | false                | win_rate / sharpe / drawdown inside band        |
| RECOVER      | 0.25          | true                 | drawdown band breach                            |
| HALT         | 0.0           | true                 | severe condition; new entries disabled          |

### 5.5 Other gates

* **Stop Bus** — operator emergency stop, currently inactive.
* **Net Exposure Gate** — caps gross exposure by strategy and globally.
* **Strategy Throttle Gate** — caps simultaneous fills per strategy.
* **Per-symbol daily loss limit** — closes trade attribution after a
  per-symbol daily loss threshold is hit.
* **ML veto** — strategy-specific Bayesian model approvals.

---

## 6. Business Framework

Phase ladder (v9.1, unchanged):
1. **BUILD** — engine construction (completed pre-v8.0).
2. **GROW** — compounding seed → high water mark. **Current.**
3. **PAY** — equity above HWM allows partial salary withdrawals.
4. **SCALE_OUT** — engine paid; profits compounded into capital.

Current snapshot (`runtime/business_phase.json`):
* Phase: **GROW**
* days_in_phase: 19
* growth_pct_from_seed: 251.48% (from $50,000 to $175,739.51)
* high_water_mark: $183,874.27
* next phase requirement: re-cross HWM (currently 4.4% below).

Withdrawal authorization (`runtime/withdrawal_authorization.json`) shows
`spendable_surplus_usd=0` and `authorized_withdrawal_usd=0` because the 30-day
trailing drawdown (9.6%) exceeds the 5.0% GROW-phase veto threshold. No
salary withdrawal can be authorized in this state.

---

## 7. Reconciliation

`runtime/reconciliation_state.json` is the broker-truth anchor (the
chad-reconciliation-publisher writes it on every cycle).

| Field                       | Value                                                  |
|-----------------------------|--------------------------------------------------------|
| status                      | **GREEN**                                              |
| broker_source               | `ibkr:clientId=83`                                     |
| chad_state_source           | `position_guard.json`                                  |
| chad_open                   | 15                                                     |
| chad_strategy_open          | 14                                                     |
| broker_positions            | 15                                                     |
| worst_diff                  | 0.0                                                    |
| mismatches                  | empty                                                  |
| drifts                      | empty                                                  |
| excluded equity symbols     | AAPL, CVX, NVDA, PEP, QQQ                              |
| excluded futures symbols    | M2K, MCL, MES                                          |
| exclusion_policy entries    | AAPL, MSFT, NVDA, BAC, CVX, LLY, PEP, QQQ, SPY         |
| ts_utc                      | 2026-05-16T00:22:38Z                                   |

The Position-Guard rebuilder remains GAP-028 Option B (PERMISSIVE): the
rebuilder mirrors `trade_closer_state.queues` faithfully without consulting
the exclusion policy. The drift detector
(`chad.core.position_guard.detect_guard_vs_broker_truth_drift`) publishes
`runtime/position_guard_drift.json` (schema `position_guard_drift.v1`) for
operator review. The CLI close-tool (`scripts/close_guard_entry.py`) is gated
fail-closed on `SCR ∉ {CONFIDENT,CAUTIOUS}`, exec mode not in `{paper,dry_run}`,
LiveGate operator intent `ALLOW_LIVE`, or `broker_sync|*` keys.

The 2026-05-09 audit memory note (`positions_snapshot.json` stale since
2026-04-03) is preserved: that issue is observability-side. `broker_sync|*`
entries are still correct and the reconciliation publisher remains the
authoritative truth source.

---

## 8. Intelligence Layer — MAJOR UPDATE

### 8.1 Feed catalog

| Feed                  | File                                          | Publisher                                      | Cadence  | TTL (s)  | Phase     |
|-----------------------|-----------------------------------------------|------------------------------------------------|----------|----------|-----------|
| Regime                | `regime_state.json`                           | `chad-live-loop`                               | 60 s     | 360      | legacy    |
| Choppy overlay        | `choppy_regime_state.json`                    | `chad-choppy-regime`                           | 5 m      | 300      | legacy    |
| Event risk            | `event_risk.json`                             | `chad-event-risk` (EconomicCalendar)           | 10 m     | 1,800    | v8.8      |
| Macro state           | `macro_state.json`                            | `chad-macro-state` (FRED)                      | 10 m     | 1,800    | v8.8      |
| Short interest        | `short_interest.json`                         | `chad-short-interest-refresh`                  | 6 h      | -        | v8.x      |
| Reddit sentiment      | `reddit_sentiment.json`                       | `chad-reddit-sentiment-refresh`                | 2 h      | -        | v8.x      |
| Trends                | `trends.json`                                 | `chad-trends-refresh`                          | 4 h      | -        | v8.x      |
| **News intel**        | `news_intel.json`                             | `chad-news-intel-refresh`                      | 30 m     | 1,800    | **B1**    |
| **Relative strength** | `relative_strength.json`                      | `chad-rs-refresh`                              | daily    | 90,000   | **B2**    |
| **Volume scan**       | `volume_scan.json`                            | `chad-volume-scan`                             | 5 m      | 600      | **B3**    |
| **Crypto derivatives**| `crypto_derivatives.json`                     | `chad-crypto-derivatives-refresh`              | 5 m      | 600      | **B4**    |
| **Futures roll**      | `futures_roll_state.json`                     | `chad-futures-roll-refresh`                    | daily    | 86,400   | **B5**    |
| **Options Greeks**    | `options_greeks.json`                         | `chad-options-greeks-refresh`                  | hourly   | 3,600    | **B6**    |
| Setup expectancy      | `setup_family_expectancy.json`                | `chad-setup-expectancy`                        | daily    | 86,400   | v9.1      |
| Liquidity (ADDV)      | (computed in-process via `liquidity.py`)      | (process-local cache; no publisher)            | per-sig  | -        | A5        |
| Strategy intelligence | `strategy_intelligence.json`                  | `chad-strategy-intelligence-refresh`           | 15 m     | -        | v8.x      |

### 8.2 Phase B feed deep-dives

#### 8.2.1 News intel (Phase B Item 1)

* Schema: `news_intel.v1`.
* File: `runtime/news_intel.json`.
* Source: Polygon news primary, Yahoo fallback.
* Module: `chad/market_data/news_intel_publisher.py` orchestrates per-symbol
  fetches via `catalyst_news_provider.get_catalyst_intel`.
* Per-symbol fields:
  - `has_catalyst` (bool)
  - `catalyst_strength` ∈ {none, low, medium, high, unknown}
  - `catalyst_direction` ∈ {none, bullish, bearish, neutral, unknown}
  - `catalyst_categories` (e.g., earnings, ratings, fda, merger)
  - `catalyst_count`, `news_count`, `relevant_news_count`
  - `confirmed_gate_relevant` (bool — must be true for the gate to consider blocking)
  - `symbol_relevance` ∈ {direct, broad_market, unknown}
  - `latest_headline`, `latest_ts_utc`, `source_provider`
* Consumer: `chad/utils/catalyst_gate.py` → `alpha`, `alpha_intraday`.
* Effect: HARD BLOCK on opposing high-strength catalyst (BUY-vs-bearish-HIGH,
  SELL-vs-bullish-HIGH; symmetric for MEDIUM); other paths fail open.
* Symbol scoping: futures (MES, MNQ, …) and crypto (`*-USD`) are excluded
  from the publisher entirely.

#### 8.2.2 Relative strength (Phase B Item 2)

* Schema: `relative_strength.v1`.
* File: `runtime/relative_strength.json`.
* Source: daily bars at `data/bars/1d/{SYM}.json` (no external API).
* Module: `chad/market_data/relative_strength_publisher.py`.
* Top-level fields: `market_direction`, `lookback_days`,
  `benchmark_spy_return_5d`, `benchmark_qqq_return_5d`, `summary`.
* Per-symbol fields: `return_5d`, `spy_return_5d`, `qqq_return_5d`,
  `excess_vs_spy_5d`, `excess_vs_qqq_5d`, `rs_vs_spy`, `rs_vs_qqq`,
  `rs_class` ∈ {strong, neutral, weak, unknown}, `bars_used`, `data_available`.
* Consumer: `chad/utils/rs_gate.py` → `alpha`, `alpha_intraday`.
* Effect: ±0.10 confidence (never a hard block).

#### 8.2.3 Volume scan / RVOL (Phase B Item 3)

* Schema: `volume_scan.v1`.
* File: `runtime/volume_scan.json`.
* Source: Polygon snapshot primary (`/v2/snapshot/locale/us/markets/stocks/tickers`)
  with a `rolling_1m` fallback that sums recent 1-minute bars from
  `data/bars/1m/{SYM}.json` and divides by an estimated expected volume.
* Top-level: `market_open` (bool), `fraction_elapsed`, `minutes_into_session`,
  `source.volume_provider` ∈ {polygon_snapshot, rolling_1m},
  `source.provider_status`.
* Per-symbol fields: `avg_daily_volume`, `current_volume`, `expected_volume`,
  `metric_type` ∈ {session_rvol, rolling_rvol}, `rvol`, `rvol_class` ∈ {high,
  above, normal, low, unavailable}, `window_minutes`, `data_available`.
* Consumer: `chad/utils/rvol_gate.py` → `alpha_intraday` only.
* Effect: +0.05 boost (`high`), -0.05 penalty (`low`); 0.0 otherwise. Never a
  hard block.
* Off-hours behavior: outside RTH with no rolling 1m data, all symbols are
  `unavailable`; the gate then returns a zero adjustment.

#### 8.2.4 Crypto derivatives (Phase B Item 4)

* Schema: `crypto_derivatives.v1`.
* File: `runtime/crypto_derivatives.json`.
* Source: Kraken Futures public tickers endpoint
  (`https://futures.kraken.com/derivatives/api/v3/tickers`) — no auth needed.
* Module: `chad/market_data/crypto_derivatives_publisher.py`. Uses stdlib
  HTTP only; writes atomically; preserves an existing good payload on fetch
  failure unless `--dry-run` is set.
* Symbol map: `PF_XBTUSD` → BTC-USD, `PF_ETHUSD` → ETH-USD,
  `PF_SOLUSD` → SOL-USD.
* Per-symbol fields: `last`, `bid`, `ask`, `index_price`,
  `funding_rate_8h`, `funding_rate_annualized`, `funding_elevated_long`,
  `funding_extreme_long`, `funding_extreme_short`, `oi_change_pct`,
  `open_interest_contracts`, `open_interest_usd`, `vol_24h`, `market_bias`
  ∈ {long_crowded, long_leaning, balanced, short_leaning, short_crowded,
  unknown}.
* Thresholds: `FUNDING_HIGH_THRESHOLD=0.0001`, `FUNDING_EXTREME_THRESHOLD=0.0003` (per-8h).
* Consumer: `chad/utils/crypto_signal_filter.py` → `alpha_crypto`.
* Effect: -0.20 confidence on aligned crowded entries, -0.05 on leaning.

#### 8.2.5 Futures roll calendar (Phase B Item 5)

* Schema: `futures_roll_state.v1`.
* File: `runtime/futures_roll_state.json`.
* Source: static third-Friday quarterly calendar (no external API,
  `provider="static_cme_calendar"`).
* Module: `chad/market_data/futures_roll_publisher.py`.
* Supported (gating) symbols: MES, MNQ, MYM, M2K (quarterly equity-index).
* Unsupported (informational only): MCL, MGC, ZN, ZB, M6E. These emit
  `roll_pattern="unsupported_v1"` and `block_new_entries=False`.
* Thresholds: `ROLL_WARNING_DAYS=5`, `ROLL_CRITICAL_DAYS=2`.
* Per-symbol fields: `current_expiry`, `next_expiry`, `days_to_expiry`,
  `roll_pattern`, `roll_supported`, `roll_warning`, `roll_critical`,
  `block_new_entries`.
* Consumer: `chad/utils/roll_gate.py` → `alpha_futures`.
* Effect: hard block on new entries when `roll_supported=True` and
  `block_new_entries=True`.

#### 8.2.6 Options Greeks (Phase B Item 6)

* Schema: `options_greeks.v1`.
* File: `runtime/options_greeks.json`.
* Source: synthetic Black–Scholes derived from
  `runtime/options_chains_cache.json` (chain cache v2), VIX from
  `data/bars/1d/VIX.json`, and underlying spot from
  `runtime/price_cache.json`.
* Module: `chad/market_data/options_greeks_publisher.py`.
* Provider: `synthetic_black_scholes_vix`, `provider_status="approximated"`.
* Per-symbol nesting: `symbols[SYM] → expirations[YYYYMMDD] → strikes[K]` with
  per-strike `call_delta`, `put_delta`, optional `call_gamma` / `put_gamma` /
  `call_theta` / `put_theta` / `call_theo_price` / `put_theo_price`, `near_atm`,
  `source`.
* Bounds: call delta clamped to [0.01, 0.99]; put delta clamped to [-0.99, -0.01].
* Consumer: `chad/utils/options_greeks_gate.py` → `alpha_options`.
* Effect: metadata only — annotates `TradeSignal.meta` with `long_delta`,
  `short_delta`, `net_delta_estimate`, `long_theo_price`, `short_theo_price`,
  `long_delta_source`, `short_delta_source`. The gate never blocks.

### 8.3 FMP scaffolding (Phase B bonus)

`chad/market_data/fmp_client.py` exposes the stable-API endpoints confirmed
working on the current plan:
- `/stable/quote` → `FMPQuote`
- `/stable/profile` → `FMPProfile`
- `/stable/earnings-calendar` → `FMPEarningsEvent`
- `/stable/price-target-consensus` → `FMPPriceTargetConsensus`
- `/stable/analyst-estimates` (`period=annual`) → `FMPAnalystEstimate`
- `/stable/sec-filings-search/symbol` (with `from`/`to`) → `FMPSecFiling`

Legacy v3 endpoints are blocked for this account and are intentionally never
referenced. `/stable/news/stock` is restricted on the current plan and is
exposed via the higher-level provider as a no-op stub. The key file lives at
`/etc/chad/fmp.env`; the placeholder value `YOUR_REAL_FMP_KEY_HERE` is
treated as "no key" and short-circuits all calls to an empty list.

`chad/market_data/market_intel_provider.py` is the abstraction layer:
- `CHAD_INTEL_PROVIDER` (default `fmp`) — routes equity/earnings fact lookups.
- `CHAD_NEWS_PROVIDER` (default `existing`) — keeps Polygon/Yahoo flow.
- `CHAD_VOLUME_PROVIDER` (default `existing`) — keeps `rolling_1m` fallback.

No publisher has migrated to FMP yet; the abstraction is the seam for later
one-at-a-time migrations.

---

## 9. Telegram Operator Interface

Unchanged from v9.1 — the bot reads runtime artifacts and can:
- `/status` — SCR + business phase + tier + reconciliation summary.
- `/positions` — current open positions (broker truth + chad guard).
- `/risk` — Profit Lock state, daily loss, Stop Bus.
- `/recon` — reconciliation status (`runtime/reconciliation_state.json`).
- `/halt` — operator-emergency stop (sets `stop_bus.active=true`).
- `/clear` — clears the stop bus after manual review.
- Notifications fire on tier promotions / demotions, SCR transitions,
  stop bus triggers, and reconciliation drift > threshold.

No new commands were added in v9.2. The operator UI does not yet surface
catalyst, RS, RVOL, crypto-funding, or roll-window state; that is a follow-up
candidate.

---

## 10. Dashboard

Unchanged from v9.1 — nginx fronts the Flask dashboard
(`deploy/chad-dashboard-nginx.conf` + `deploy/chad-dashboard.service`). The
dashboard surfaces runtime artifacts (SCR, tier, profit lock, regime, choppy,
event risk, macro, business phase, reconciliation, recent fills). Phase B
intelligence feeds are written to `runtime/` and are available to any future
panel without further plumbing.

---

## 11. Services & Timers

The deploy directory (`deploy/`) holds all systemd units shipped to
`/etc/systemd/system`. Active timers as of v9.2 (selected from
`systemctl list-timers --all | grep chad`):

| Timer                                       | Service unit                          | Cadence (approx.) | Phase  |
|---------------------------------------------|---------------------------------------|-------------------|--------|
| `chad-live-readiness.timer`                 | chad-live-readiness                   | 10 m              | v8.x   |
| `chad-reconciliation-publisher.timer`       | chad-reconciliation-publisher         | 5 m               | v8.x   |
| `chad-position-snapshot.timer`              | (chad-positions-snapshot)             | 5 m               | v8.x   |
| `chad-portfolio-snapshot.timer`             | chad-portfolio-snapshot               | 5 m               | v8.x   |
| `chad-trade-closer.timer`                   | chad-trade-closer                     | 1 m               | v8.x   |
| `chad-paper-trade-exec.timer`               | chad-paper-trade-exec                 | 10 m              | v8.x   |
| `chad-event-risk.timer`                     | chad-event-risk                       | 10 m              | v8.8   |
| `chad-macro-state.timer`                    | chad-macro-state                      | 10 m              | v8.8   |
| `chad-choppy-regime.timer`                  | chad-choppy-regime                    | 5 m               | v8.x   |
| `chad-business-phase.timer`                 | chad-business-phase                   | 30 m              | v8.x   |
| `chad-withdrawal-manager.timer`             | chad-withdrawal-manager               | 6 h               | v8.x   |
| `chad-burnin-check.timer`                   | chad-burnin-check                     | 10 m              | v8.x   |
| `chad-operator-intent-refresh.timer`        | chad-operator-intent-refresh          | 10 m              | v8.x   |
| `chad-rebalance-auto-executor-paper.timer`  | chad-rebalance-auto-executor-paper    | 10 m              | v8.x   |
| `chad-expectancy-tracker.timer`             | chad-expectancy-tracker               | 5 m               | v9.x   |
| `chad-setup-expectancy.timer`               | chad-setup-expectancy                 | daily             | v9.1   |
| `chad-tier-manager.timer`                   | chad-tier-manager                     | 5 m               | v9.x   |
| `chad-symbol-blocker.timer`                 | chad-symbol-blocker                   | 5 m               | v8.x   |
| `chad-strategy-intelligence-refresh.timer`  | chad-strategy-intelligence-refresh    | 15 m              | v8.x   |
| `chad-winner-scaler.timer`                  | chad-winner-scaler                    | 15 m              | v8.x   |
| `chad-feed-watchdog.timer`                  | chad-feed-watchdog                    | 2 m               | v8.x   |
| `chad-portfolio-artifacts.timer`            | chad-portfolio-artifacts              | hourly            | v8.x   |
| `chad-disk-guard.timer`                     | chad-disk-guard                       | 30 m              | v8.x   |
| `chad-health-monitor.timer`                 | chad-health-monitor                   | 5 m               | v8.x   |
| `chad-lifecycle-replay-engine.timer`        | chad-lifecycle-replay-engine          | 30 m              | v8.x   |
| `chad-calendar-state-publisher.timer`       | chad-calendar-state-publisher         | 5 m               | v8.x   |
| `chad-ibkr-broker-events.timer`             | chad-ibkr-broker-events               | 5 m               | v8.x   |
| `chad-ibkr-paper-fill-harvester.timer`      | chad-ibkr-paper-fill-harvester        | 5 m               | v8.x   |
| `chad-ibkr-paper-ledger-watcher.timer`      | chad-ibkr-paper-ledger-watcher        | 15 m              | v8.x   |
| `chad-ibkr-daily-bars-refresh.timer`        | chad-ibkr-daily-bars-refresh          | daily             | v8.x   |
| `chad-universe-refresh.timer`               | chad-universe-refresh                 | hourly            | v8.x   |
| `chad-trends-refresh.timer`                 | chad-trends-refresh                   | 4 h               | v8.x   |
| `chad-short-interest-refresh.timer`         | chad-short-interest-refresh           | 6 h               | v8.x   |
| `chad-reddit-sentiment-refresh.timer`       | chad-reddit-sentiment-refresh         | 2 h               | v8.x   |
| `chad-proofs-cleanup.timer`                 | chad-proofs-cleanup                   | daily             | v8.x   |
| `chad-advisory-pre-market.timer`            | chad-advisory-pre-market              | daily             | v8.x   |
| `chad-micro-eod-flatten.timer`              | chad-micro-eod-flatten                | daily             | v9.1   |
| `chad-options-monitor.timer`                | chad-options-monitor                  | 5 m               | v8.x   |
| `options_chain_refresh.timer`               | options_chain_refresh                 | hourly            | v8.x   |
| **`chad-news-intel-refresh.timer`**         | chad-news-intel-refresh               | 30 m              | **B1** |
| **`chad-rs-refresh.timer`**                 | chad-rs-refresh                       | daily             | **B2** |
| **`chad-volume-scan.timer`**                | chad-volume-scan                      | 5 m               | **B3** |
| **`chad-crypto-derivatives-refresh.timer`** | chad-crypto-derivatives-refresh       | 5 m               | **B4** |
| **`chad-futures-roll-refresh.timer`**       | chad-futures-roll-refresh             | daily             | **B5** |
| **`chad-options-greeks-refresh.timer`**     | chad-options-greeks-refresh           | hourly            | **B6** |

All six Phase B timers + their service units are present in `deploy/` and
have fired at least once on the host (verified via `systemctl list-timers`).
The next fire windows visible at v9.2 cut time include:
- `chad-volume-scan.timer` — next 2026-05-16T00:29:34Z (5 min cadence).
- `chad-crypto-derivatives-refresh.timer` — next 2026-05-16T00:27:34Z.
- `chad-news-intel-refresh.timer` — next 2026-05-16T00:44:06Z (30 min).
- `chad-rs-refresh.timer` — next 2026-05-16T12:52:35Z (daily).
- `chad-futures-roll-refresh.timer` — next 2026-05-16T21:50:33Z (daily).
- `chad-options-greeks-refresh.timer` — daily.

---

## 12. Data & Storage

### 12.1 Universe — `config/universe.json`

Equities/ETFs (38): AAPL, SPY, MSFT, GOOGL, BAC, IEMG, QQQ, VWO, NVDA, GLD,
SH, PSQ, SVXY, UVXY, VIXY, IWM, TLT, AMZN, META, V, MA, UNH, KO, AVGO, LLY,
JNJ, WMT, CVX, NFLX, AMD, ORCL, CRM, QCOM, PEP, ADBE, MCD, INTC, VXX.

Futures (10): MES (CME), MNQ (CME), MCL (NYMEX), MGC (COMEX), ZN (CBOT),
ZB (CBOT), M6E (CME), SIL (COMEX), MYM (CBOT), M2K (CME).

Crypto (publisher-tracked, 3): BTC-USD, ETH-USD, SOL-USD (Kraken Futures
mirror).

### 12.2 Runtime files produced by Phase B

| File                                  | Phase | Schema                          | TTL    |
|---------------------------------------|-------|---------------------------------|--------|
| `runtime/news_intel.json`             | B1    | `news_intel.v1`                 | 1,800  |
| `runtime/relative_strength.json`      | B2    | `relative_strength.v1`          | 90,000 |
| `runtime/volume_scan.json`            | B3    | `volume_scan.v1`                | 600    |
| `runtime/crypto_derivatives.json`     | B4    | `crypto_derivatives.v1`         | 600    |
| `runtime/futures_roll_state.json`     | B5    | `futures_roll_state.v1`         | 86,400 |
| `runtime/options_greeks.json`         | B6    | `options_greeks.v1`             | 3,600  |

### 12.3 FMP client (Phase B bonus)

- Key file: `/etc/chad/fmp.env` (env-style `FMP_API_KEY=...`).
- Base URL: `https://financialmodelingprep.com/stable`.
- Timeout: 8 s default.
- User-Agent: `CHAD/1.0`.
- Placeholder sentinel: `YOUR_REAL_FMP_KEY_HERE` causes every public method
  to return an empty list (no HTTP call).

### 12.4 Historical bars

* `data/bars/1d/{SYM}.json` — daily bars (used by RS publisher, liquidity
  classifier, and options Greeks publisher for VIX).
* `data/bars/1m/{SYM}.json` — 1-minute bars (used by volume_scan_publisher
  rolling_1m fallback).
* `runtime/options_chains_cache.json` — chain cache (v2 schema), input to
  the synthetic Greeks publisher.

---

## 12.5 Phase A and Phase B utility surface — code-level reference

This section documents the exact public API of each Phase A and Phase B
utility module. It exists so that future strategy authors can wire into the
gates without reading the implementation files.

### 12.5.1 `chad/utils/session.py` (Phase A Item 2)

Canonical America/New_York session model. Constants:

```
SESSION_TIMEZONE            = "America/New_York"
PRIMARY_SESSION_START       = "09:35"
PRIMARY_SESSION_END         = "11:00"
SECONDARY_SESSION_START     = "13:30"
SECONDARY_SESSION_END       = "15:00"
HARD_EOD_EXIT_TIME          = "15:30"

SKIP_OUTSIDE_PRIMARY_WINDOW = "SKIP_OUTSIDE_PRIMARY_WINDOW"
SKIP_OUTSIDE_SESSION_WINDOW = "SKIP_OUTSIDE_SESSION_WINDOW"
SKIP_EOD_FLATTEN_WINDOW     = "SKIP_EOD_FLATTEN_WINDOW"

SESSION_PRIMARY             = "PRIMARY"
SESSION_SECONDARY           = "SECONDARY"
```

`SessionDecision` dataclass:

```
session_window: Optional[str]    # "PRIMARY" | "SECONDARY" | None
is_primary: bool
is_secondary: bool
is_eod_flatten_window: bool
entry_allowed: bool
skip_reason: Optional[str]
```

Entry point:

```
session_decision(
  now=None,                              # default = datetime.now(UTC)
  timezone_name="America/New_York",
  primary_start="09:35", primary_end="11:00",
  secondary_start="13:30", secondary_end="15:00",
  hard_eod_exit_time="15:30",
  primary_session_only=False,
) -> SessionDecision
```

Behavioral matrix (entry-only):

| Local time              | `primary_session_only=true` | `primary_session_only=false` |
|-------------------------|------------------------------|------------------------------|
| 09:35 ≤ t < 11:00       | entry_allowed=True (primary) | entry_allowed=True (primary) |
| 11:00 ≤ t < 13:30       | False, `SKIP_OUTSIDE_PRIMARY_WINDOW` | False, `SKIP_OUTSIDE_SESSION_WINDOW` |
| 13:30 ≤ t < 15:00       | False, `SKIP_OUTSIDE_PRIMARY_WINDOW` | True (secondary)             |
| 15:00 ≤ t < 15:30       | False, `SKIP_OUTSIDE_PRIMARY_WINDOW` | False, `SKIP_OUTSIDE_SESSION_WINDOW` |
| 15:30 ≤ t (EOD flatten) | False, `SKIP_EOD_FLATTEN_WINDOW` | False, `SKIP_EOD_FLATTEN_WINDOW` |
| Otherwise               | False, `SKIP_OUTSIDE_SESSION_WINDOW` | False, `SKIP_OUTSIDE_SESSION_WINDOW` |

### 12.5.2 `chad/utils/risk_reward.py` (Phase A Item 3)

```
MIN_RR_RATIO = 1.5

compute_rr_ratio(target_pts, stop_pts) -> Optional[float]
  Returns target_pts/stop_pts, or None if either is <= 0.

passes_rr_gate(target_pts, stop_pts, *, min_ratio=MIN_RR_RATIO) -> bool
  Returns True if ratio is None (fail-open) OR ratio >= min_ratio.
```

The fail-open contract is critical: degenerate or absent target estimates
must never block a valid signal.

### 12.5.3 `chad/utils/liquidity.py` (Phase A Item 5)

```
class LiquidityClass(str, Enum):
    LARGE    = "large"      # ADDV >= $500M
    STANDARD = "standard"   # ADDV >= $50M
    THIN     = "thin"
    UNKNOWN  = "unknown"

THIN_MIN_CONFIDENCE = 0.80

classify(symbol, *, repo_root=None, refresh=False) -> LiquidityClass
blocks_thin_entry(symbol, confidence, *, repo_root=None,
                  thin_min_confidence=0.80) -> (bool, LiquidityClass, float)
clear_cache()
```

ADDV computation: average of `close * volume` over the last 20 valid bars,
requires ≥5 valid bars to commit a classification. Bar lookup tries:

1. `data/bars/1d/<SYM>.json` ← canonical
2. `data/daily_bars/<SYM>.ndjson` ← dead fallback
3. `data/daily_bars/<SYM>_daily.ndjson` ← dead fallback
4. `data/<SYM>_daily_bars.ndjson` ← dead fallback

The classification cache is keyed by `(normalized_symbol, repo_root)` so
tests using temp paths cannot contaminate production lookups.

### 12.5.4 `chad/utils/catalyst_gate.py` (Phase B Item 1)

```
DEFAULT_RUNTIME_DIR = "/home/ubuntu/chad_finale/runtime"
NEWS_INTEL_FILENAME = "news_intel.json"
DEFAULT_TTL_SECONDS = 3600

class CatalystGateResult:
    allowed: bool
    catalyst_strength: str           # none|low|medium|high|unknown
    catalyst_direction: str          # none|bullish|bearish|neutral|unknown
    block_reason: Optional[str]

check_catalyst_gate(symbol, signal_side, *, runtime_dir=None) -> CatalystGateResult
```

Block-reason codes:
- `high_catalyst_bullish_opposes_bearish`
- `high_catalyst_bearish_opposes_bullish`
- `medium_catalyst_bullish_opposes_bearish`
- `medium_catalyst_bearish_opposes_bullish`

Pre-conditions for any block:
1. File exists, parses, and is not stale (`ttl_seconds`).
2. `payload["symbols"][SYM]` exists.
3. `confirmed_gate_relevant == True` on that record.
4. `has_catalyst == True`.
5. `strength ∈ {"high", "medium"}`.
6. `direction ∈ {"bullish", "bearish"}`.

If any pre-condition fails, the gate fails open with `allowed=True`.

### 12.5.5 `chad/utils/rs_gate.py` (Phase B Item 2)

```
RS_FILENAME            = "relative_strength.json"
RS_FILE_TTL            = 90000
RS_CONFIDENCE_PENALTY  = 0.10

class RSGateResult:
    confidence_adjustment: float
    rs_class: str            # strong|neutral|weak|unknown
    rs_vs_spy: Optional[float]
    excess_vs_spy_5d: Optional[float]
    market_direction: str    # up|down|flat|unknown

get_rs_adjustment(symbol, signal_side, *, runtime_dir=None) -> RSGateResult
```

Penalty rule (all else returns 0.0):
- `BUY  + rs_class="weak"  + market_direction="up"`   → -0.10
- `SELL + rs_class="strong"+ market_direction="down"` → -0.10

### 12.5.6 `chad/utils/rvol_gate.py` (Phase B Item 3)

```
RVOL_FILENAME    = "volume_scan.json"
RVOL_FILE_TTL    = 600
RVOL_HIGH_BOOST  = 0.05
RVOL_LOW_PENALTY = 0.05

class RvolGateResult:
    confidence_adjustment: float
    rvol: Optional[float]
    rvol_class: str          # high|above|normal|low|unavailable

get_rvol_adjustment(symbol, *, runtime_dir=None) -> RvolGateResult
```

Mapping: `high` → +0.05; `low` → -0.05; everything else → 0.0.

### 12.5.7 `chad/utils/crypto_signal_filter.py` (Phase B Item 4)

```
CROWDED_PENALTY    = 0.20
LEANING_PENALTY    = 0.05
CRYPTO_DERIV_TTL   = 1200

class CryptoFilterResult:
    confidence_adjustment: float
    market_bias: str             # long_crowded|short_crowded|long_leaning|...
    funding_rate_8h: Optional[float]
    funding_extreme: bool

get_crypto_filter(symbol, signal_side, *, runtime_dir=None) -> CryptoFilterResult
```

Penalty rule:
- `BUY  + long_crowded`  → -0.20
- `SELL + short_crowded` → -0.20
- `BUY  + long_leaning`  → -0.05
- All other combinations → 0.0

`funding_extreme` is True when bias is `long_crowded` or `short_crowded`,
purely for telemetry.

### 12.5.8 `chad/utils/roll_gate.py` (Phase B Item 5)

```
ROLL_FILENAME = "futures_roll_state.json"
ROLL_FILE_TTL = 172800

class RollGateResult:
    blocked: bool
    days_to_expiry: Optional[int]
    roll_pattern: Optional[str]    # quarterly_3rd_friday|unsupported_v1|None
    roll_supported: bool
    block_reason: Optional[str]    # "ROLL_WARNING_WINDOW" or None

check_roll_gate(symbol, *, runtime_dir=None) -> RollGateResult
```

Block requires `roll_supported=True` AND `block_new_entries=True`. The
publisher's static calendar currently never sets `block_new_entries=True`
ahead of more than 5 days to expiry; the supported quarterly micros (MES,
MNQ, MYM, M2K) all read `block_new_entries=False` at v9.2 cut time
(days_to_expiry=35).

### 12.5.9 `chad/utils/options_greeks_gate.py` (Phase B Item 6)

```
GREEKS_FILE_NAME    = "options_greeks.json"
GREEKS_FILE_TTL     = 7200
DEFAULT_CALL_DELTA  = +0.5
DEFAULT_PUT_DELTA   = -0.5

class GreeksResult:
    delta: float
    gamma: Optional[float]
    theta: Optional[float]
    theo_price: Optional[float]
    source: str              # "default" | "synthetic" | publisher source string
    near_atm: bool

get_option_greeks(symbol, expiry, strike, option_type, *,
                  runtime_dir=None) -> GreeksResult
```

Lookup order:
1. File missing / stale / unreadable → default result (`source="default"`).
2. Symbol missing in payload → default.
3. Expiry missing under that symbol → default.
4. Exact strike match → use that strike block.
5. Otherwise → nearest strike in the same expiry.

The gate never raises and is metadata-only.

---

## 13. Change Log — Delta from v9.1

v9.1 baseline tag: commit `ac3f58e` (2026-05-13).
v9.2 HEAD: commit `48d90ec`. Total of **17 commits**.

### 13.1 Phase A — Signal Quality (5 items, 6 commits)

| # | Commit    | Item                          | Summary                                                                 |
|---|-----------|-------------------------------|-------------------------------------------------------------------------|
| 1 | `f819cd3` | (baseline polish)             | Wire alpha_intraday_micro into canonical strategy names; tier_profile/tier_name on MarketContext — 1,652 tests. |
| 2 | `57c791c` | Phase A Item 1                | Stop-distance sizing (alpha_futures, alpha, alpha_intraday) + 4 regression tests — 1,656 tests. Tier budget activates stop-aligned sizing; fallback preserves legacy. |
| 3 | `dd6d7fa` | Phase A Item 2                | Session-zone gating (alpha, alpha_intraday, alpha_futures) + shared `chad/utils/session.py` + 15 regression tests — 1,671 tests. Tier `primary_session_only` controls entry windows; exits never blocked. |
| 4 | `4eadf6e` | Phase A Item 3                | Pre-entry R:R gate `chad/utils/risk_reward.py` (`MIN_RR_RATIO=1.5`); fail-open on degenerate inputs. |
| 5 | `759ec5d` | Phase A Item 4                | Setup-family tagging across alpha / alpha_intraday / alpha_futures meta. |
| 6 | `6c73191` | Phase A Item 5                | Float-aware liquidity gate `chad/utils/liquidity.py`; LARGE/STANDARD/THIN/UNKNOWN; THIN blocks below 0.80 confidence. |

### 13.2 Phase B — Intelligence Feeds (6 items, 10 commits)

| #  | Commit    | Item                          | Summary                                                                 |
|----|-----------|-------------------------------|-------------------------------------------------------------------------|
| 7  | `452b604` | Phase B Item 1                | Catalyst news intelligence gate: `catalyst_news_provider.py`, `news_intel_publisher.py`, `chad/utils/catalyst_gate.py`. Polygon primary, Yahoo fallback. Wires into alpha + alpha_intraday. |
| 8  | `c61bcd5` | (Item 1 follow-up)            | Tightened catalyst news symbol relevance: only `symbol_relevance="direct"` and `confirmed_gate_relevant=true` participate in gating. |
| 9  | `4b1da8e` | (Item 1 follow-up)            | Require confirmed catalyst relevance before blocking trades — eliminates false-positive blocks on broad-market headlines. |
| 10 | `2e9957c` | Phase B Item 2                | Relative-strength intelligence: `relative_strength_publisher.py`, `chad/utils/rs_gate.py`. ±0.10 confidence modifier, never hard-blocks. |
| 11 | `cdcaa14` | Phase B Item 3                | Intraday RVOL scanner: `volume_scan_publisher.py`, `chad/utils/rvol_gate.py`. ±0.05 modifier; alpha_intraday only. |
| 12 | `bfcb9fc` | (Item 3 follow-up)            | Rolling 1m fallback for RVOL scanner when Polygon snapshot is unavailable / off-hours. |
| 13 | `29e2699` | Phase B bonus                 | FMP stable-endpoint client scaffold (`chad/market_data/fmp_client.py`, `chad/market_data/market_intel_provider.py`); no publisher migrated. |
| 14 | `156bfe7` | Phase B Item 4                | Crypto derivatives intelligence: `crypto_derivatives_publisher.py` (Kraken Futures public), `chad/utils/crypto_signal_filter.py`. -0.20 on aligned crowding for alpha_crypto. |
| 15 | `5ccc06c` | Phase B Item 5                | Futures roll calendar gate: `futures_roll_publisher.py` (static CME calendar), `chad/utils/roll_gate.py`. Hard-blocks supported quarterly micros in warning window. |
| 16 | `48d90ec` | Phase B Item 6                | Options Greeks metadata publisher: `options_greeks_publisher.py` (synthetic Black–Scholes), `chad/utils/options_greeks_gate.py`. alpha_options TradeSignal meta annotations only. |

### 13.3 Test growth

| Milestone                | Tests passing |
|--------------------------|---------------|
| v9.1 baseline (`ac3f58e`)| 1,652         |
| After Phase A Item 1     | 1,656         |
| After Phase A Item 2     | 1,671         |
| v9.2 HEAD (`48d90ec`)    | **1,874**     |

Net new tests during the v9.1 → v9.2 window: **+222**, of which the bulk come
from Phase B regression suites (catalyst gate fail-open paths, RS gate
adjustment matrix, RVOL session windows, crypto derivatives schema, futures
roll boundary conditions, and synthetic Greeks bounds).

---

### 13.4 What did NOT change between v9.1 and v9.2

For completeness, the following surfaces are untouched between v9.1 and v9.2:

* `chad/core/orchestrator.py` — entry/exit pipeline unchanged.
* `chad/core/live_loop.py` — Position Guard rebuilder unchanged (GAP-028
  Option B PERMISSIVE policy preserved).
* `chad/execution/ibkr_adapter.py` — broker adapter unchanged.
* `chad/execution/paper_exec_evidence_writer.py` — paper evidence writer
  hardened in commit `cbded80` (pre-v9.2) and unchanged since.
* `chad/risk/dynamic_risk_allocator.py` — allocator unchanged.
* `chad/risk/profit_lock.py` — profit-lock thresholds unchanged.
* `chad/core/live_gate.py` — LiveGate unchanged (paper-only by config).
* `chad/core/position_guard.py` — drift detector unchanged.
* `scripts/close_guard_entry.py` — guard-close CLI unchanged.
* `runtime_FREEZE_*` and `data_FREEZE_*` — untouched (per governance).
* All systemd `*.service` units shipped to `/etc/systemd/system` (per
  governance, never modified without explicit instruction).
* Risk caps, strategy config, and the live-mode toggle — untouched (per
  governance, only Pending Actions are permitted).

### 13.5 Migration / dependency notes for v9.3

When v9.3 begins (Phase C, or live promotion):
1. The Phase B publishers are all stdlib-only on the hot path, but several
   include optional HTTPS clients. Any future container or VM image change
   should preserve outbound HTTPS to `futures.kraken.com`,
   `api.polygon.io`, and `financialmodelingprep.com`.
2. The FMP scaffold is wired but not consumed; activating it requires
   creating a real key in `/etc/chad/fmp.env` (the placeholder sentinel
   blocks all calls today) and setting `CHAD_INTEL_PROVIDER=fmp` on the
   relevant publishers.
3. The Options Greeks pipeline depends on three input files
   (`runtime/options_chains_cache.json`, `data/bars/1d/VIX.json`,
   `runtime/price_cache.json`). All three are produced by existing v8.x
   publishers; any deprecation of those sources must be sequenced after
   migrating the Greeks consumer.
4. The crypto derivatives publisher uses a public endpoint and requires no
   credentials; it remains the lowest-risk Phase B feed to keep enabled
   across infra changes.

---

## 14. Known Issues / Accepted Residual Risks

### 14.1 Carried from v9.1 (still open)

* **IBKR Gateway latency** — periodic classification spikes above 750 ms
  ("dangerous") still observed. Live-promotion pre-task; not yet diagnosed.
* **Disk usage** — backup archives have not been pruned below 75%; pre-task
  for live promotion.
* **Kernel reboot** — deferred to the next actual kernel update
  (`6.17.0-1009-aws` is current).
* **`positions_snapshot.json` stale since 2026-04-03** — observability-side
  bug. `broker_sync|*` entries remain CORRECT in `reconciliation_state.json`;
  the publisher of `positions_snapshot.json` has no active writer.
  `positions_truth.json` reads from the stale snapshot.
* **`same_side_position_open` over-blocking** (2026-05-07 audit) — central
  regression analysis still pending; affects entry frequency for some
  strategies.

### 14.2 New residuals introduced in v9.2

* **Liquidity gate bar path** — the live path is `data/bars/1d/{SYM}.json`
  (which works). The NDJSON fallback paths
  (`data/daily_bars/{SYM}.ndjson`, `data/daily_bars/{SYM}_daily.ndjson`,
  `data/{SYM}_daily_bars.ndjson`) are dead code in the current deployment
  (no live data exists at those paths). Acceptable — UNKNOWN fail-open
  handles any miss.
* **RVOL scanner off-hours** — outside RTH and with no rolling 1m data, all
  symbols are marked `rvol_class="unavailable"`; the gate yields zero
  adjustment. Expected behavior. The most recent `volume_scan.json` we
  inspected shows `market_open=false` and `provider_status=fallback_rolling_1m`.
* **Crypto derivatives OI delta on first run** — `oi_change_pct` is `null`
  until a second sample is available (the publisher needs the prior cycle to
  compute the delta). Subsequent runs populate it; the current snapshot shows
  delta values of order 1e-5 to 1e-3.
* **Options Greeks: synthetic only** — the `provider_status` reads
  `approximated`. Actual broker Greeks (IBKR `reqMktData` Greeks ticks) are
  a Phase C consideration; the synthetic numbers are usable as ranking
  metadata but should not be treated as broker-accurate.
* **Phase B publisher installation status** — all six Phase B service +
  timer pairs are present in `deploy/` and active in
  `systemctl list-timers`. No outstanding install gap as of v9.2 HEAD.
* **ib_async Phase 2 migration** — 5 files remain on mixed
  `ib_insync`/`ib_async` imports per the CLAUDE.md note. No new files in this
  state introduced by v9.2.

---

## 15. Phase Roadmap

### Track 1 — Live promotion (carried from v9.1)

Status: still gated.

Required to flip `ready_for_live=true`:
1. SCR posture remains `CONFIDENT` (✅ currently CONFIDENT).
2. ≥14-day equity history captured (in progress — `business_phase.days_in_phase=19`,
   `withdrawal_authorization.history_days=18`).
3. ≥60-day paper performance with stable Sharpe band (in progress — Paper
   Epoch 2).
4. Operator GO (verbal/explicit instruction from operator; never auto).

Pre-live tasks (carried):
* OS reboot when next kernel update lands.
* Disk cleanup (backup pruning).
* IB Gateway latency investigation.
* Full re-verification of 1,874 tests post-reboot.
* Live readiness clean run.
* Confirm `live_readiness.json.ready_for_live=true`.
* Review open paper positions (MES short was the v9.1 callout).

### Track 2 — Elite Signal Layer

**Phase A — Signal Quality**: ✅ COMPLETE (5 / 5 items).
- Item A1 — Stop-distance sizing ✅
- Item A2 — Session zone gating ✅
- Item A3 — R:R pre-entry gate ✅
- Item A4 — Setup-family tagging ✅
- Item A5 — Float-aware liquidity gate ✅

**Phase B — Intelligence Feeds**: ✅ COMPLETE (6 / 6 items + FMP scaffold).
- Item B1 — Catalyst news intelligence ✅
- Item B2 — Relative strength ✅
- Item B3 — Intraday RVOL ✅
- Item B4 — Crypto derivatives ✅
- Item B5 — Futures roll calendar ✅
- Item B6 — Options Greeks ✅
- FMP stable-API client scaffold ✅ (no publisher migration yet)

**Phase C — Bucket 3 (new exchange connections)**: NOT STARTED.
- Item C1 — Kraken Futures adapter → `alpha_crypto_perps` (a perp-native
  twin of `alpha_crypto` that can both fetch funding and place reduced-size
  perp orders against Kraken's spot/perp surface).
- Item C2 — Liquidation heatmap (Coinglass public API) → crypto strategies'
  exit/stop heuristics.
- Item C3 — DOM / order-flow consumer (IBKR `reqMktDepth`) → microstructure
  context for `alpha_intraday` and `alpha_intraday_micro`.

**Phase D — Bucket 4 (architecture decisions)**: NOT STARTED.
- Item D1 — Dynamic universe scanner / premarket scan (replaces the static
  `config/universe.json` symbol list with a daily-scoped scan that filters by
  liquidity, gap, premarket volume, and event-risk score).
- Item D2 — Real BAG / COMBO options execution (today `alpha_options` emits
  vertical-spread legs; D2 would move to native combo orders against IBKR).

---

## 16. Appendices

### Appendix A — Tier Risk Profiles (`config/tiers.json`)

```
schema_version: tiers.v9_1
hysteresis_pct: 5.0

MICRO       0..2,500
  enabled:  alpha_intraday_micro
  insts:    MES only
  risk:     1 contract / $10 / $20-day / $30-week / 2 trades-day
  primary_session_only=true, flatten_before_eod=true (30m before close),
  stop_width_gate_enabled=true

STARTER     2,500..25,000  (demote at 2,250)
  enabled:  alpha_intraday_micro, alpha_intraday, beta_trend, alpha_crypto
  insts:    MES, MNQ, SPY, QQQ, IWM, BTC-USD
  risk:     2 / $40 / $150-day / $300-week / 4 trades-day
  primary_session_only=true, flatten_before_eod=true (30m),
  stop_width_gate_enabled=true

PRO_GROWTH  25,000..160,000  (demote at 22,500)
  enabled:  alpha, alpha_intraday, alpha_intraday_micro, alpha_futures,
            alpha_crypto, beta, beta_trend, gamma_futures, delta, omega_macro
  insts:    *
  risk:     5 / $200 / $500-day / $1,500-week / 10 trades-day
  primary_session_only=false, flatten_before_eod=false,
  stop_width_gate_enabled=true

SCALE       160,000..10,000,000  (demote at 144,000)
  enabled:  * (all 17)
  insts:    *
  risk:     unrestricted (null/null/null/null/null)
  primary_session_only=false, flatten_before_eod=false,
  stop_width_gate_enabled=false
```

### Appendix B — Phase A gate chain summary

| Gate            | Module                                | Active on                                    | Effect          |
|-----------------|---------------------------------------|----------------------------------------------|------------------|
| Session zone    | `chad/utils/session.py`               | alpha, alpha_intraday, alpha_futures, alpha_intraday_micro | hard block on entries outside window |
| Stop-distance   | tier_risk_enforcer + strategy hook    | alpha, alpha_intraday, alpha_futures         | size cap         |
| R:R             | `chad/utils/risk_reward.py`           | alpha, alpha_intraday, alpha_futures         | hard block if RR<1.5 |
| Liquidity       | `chad/utils/liquidity.py`             | alpha, alpha_intraday                        | hard block if THIN and conf<0.80 |
| Setup family    | (in-strategy helpers)                 | alpha, alpha_intraday, alpha_futures         | meta tag only    |

### Appendix C — Phase B intelligence-feed consumer map

| Feed                  | Module gate                                  | Strategy consumer           | Effect                                |
|-----------------------|----------------------------------------------|-----------------------------|----------------------------------------|
| News intel            | `chad/utils/catalyst_gate.py`                | alpha, alpha_intraday       | hard block on opposing HIGH/MEDIUM    |
| Relative strength     | `chad/utils/rs_gate.py`                      | alpha, alpha_intraday       | ±0.10 confidence (never blocks)       |
| Volume scan / RVOL    | `chad/utils/rvol_gate.py`                    | alpha_intraday only         | ±0.05 confidence (never blocks)       |
| Crypto derivatives    | `chad/utils/crypto_signal_filter.py`         | alpha_crypto                | -0.20 crowded / -0.05 leaning         |
| Futures roll          | `chad/utils/roll_gate.py`                    | alpha_futures               | hard block in warning window (supported syms) |
| Options Greeks        | `chad/utils/options_greeks_gate.py`          | alpha_options               | meta annotation only                  |

### Appendix D — FMP stable endpoints (Phase B bonus)

| Endpoint                                     | Dataclass                  | Notes                            |
|----------------------------------------------|----------------------------|----------------------------------|
| `GET /stable/quote?symbol=AAPL`              | `FMPQuote`                 | quote-style snapshot             |
| `GET /stable/profile?symbol=AAPL`            | `FMPProfile`               | issuer / sector / market cap     |
| `GET /stable/earnings-calendar?...`          | `FMPEarningsEvent`         | window-based                     |
| `GET /stable/price-target-consensus?symbol=` | `FMPPriceTargetConsensus`  | rating bucket counts             |
| `GET /stable/analyst-estimates?period=annual`| `FMPAnalystEstimate`       | per-year EPS / rev consensus     |
| `GET /stable/sec-filings-search/symbol`      | `FMPSecFiling`             | requires `from` and `to`         |
| `GET /stable/news/stock`                     | (stubbed)                  | restricted on current plan       |

Key handling:
* File: `/etc/chad/fmp.env`.
* Placeholder sentinel: `YOUR_REAL_FMP_KEY_HERE` → empty-result short-circuit.
* No method raises; all failure paths return an empty list.

### Appendix E — Verification Sequence (unchanged)

```
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
CHAD_SKIP_IB_CONNECT=1 python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

Governance gates (non-negotiable):
1. One change at a time.
2. No full rewrites.
3. No direct config mutation (risk caps, live mode, strategy config are
   Pending Actions only).
4. After every code change, run the verification sequence.
5. Never modify `runtime_FREEZE_*` or `data_FREEZE_*`.
6. Never modify systemd service files without explicit instruction.
7. Never restart live services without explicit instruction.
8. Commit and tag after each completed P0/P1/P2 item.

### Appendix F — Rollback commands

| Tag                                      | Purpose                                                       |
|------------------------------------------|---------------------------------------------------------------|
| `STABILITY_FREEZE_20260307_GREEN`        | Original stable baseline                                      |
| `PRE_HARDENING_20260402`                 | Snapshot before P0 hardening                                  |
| `RATIFICATION_MASTER_20260402`           | All P0–P2 + GAP items complete                                |
| `REVERT_PRE_OVERHAUL_20260419`           | Pre 2026-04-19/21 overhaul snapshot (commit 45f3728)          |
| `SSOT_V8_9_AUDIT_LOCK_20260503`          | v8.9 audit lock                                               |
| `SSOT_V9_0_PAPER_EPOCH2_LOCK_20260504`   | v9.0 paper Epoch 2 lock                                       |
| (v9.1 baseline)                          | commit `ac3f58e` (Docs: v9.1 …)                               |
| (v9.2 baseline)                          | commit `48d90ec` (Phase B Item 6 / current HEAD)              |

Rollback to most recent stable lock:

```
git checkout RATIFICATION_MASTER_20260402
```

Roll back v9.2 to v9.1 baseline (if needed):

```
git checkout ac3f58e
```

---

## End of v9.2 SSOT

This document captures CHAD state at HEAD `48d90ec` on 2026-05-15, with
Paper Epoch 2 in CONFIDENT posture, Phase A (5 items) and Phase B (6 items +
FMP scaffold) complete, 1,874 tests passing, and `ready_for_live=false`
gated on the operator pre-live checklist.

v9.3 should land when either Phase C begins (new exchange connections) or
when live promotion is granted — whichever comes first.
