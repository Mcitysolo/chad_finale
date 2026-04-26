# CHAD Unified SSOT v8.2

**Version:** 8.2
**Date:** 2026-04-26
**Status:** Active — Paper Trading
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v8.1_2026-04-23.md` (commit `93bbd70`, 2026-04-23)

This document is the master reference for the CHAD trading system as it
exists at HEAD (`6af971d`, 2026-04-26). It captures every post-v8.1 fix,
every wired strategy, every runtime invariant, and the live state of the
machine at the moment of writing. If this document and the code disagree,
either the code drifted or this revision needs another pass — either way,
revise the document before relying on the disagreement.

---

## Table of Contents

0. [Preamble](#0-preamble)
1. [Mission & Architecture](#1-mission--architecture)
2. [Runtime State (live snapshot)](#2-runtime-state-live-snapshot)
3. [The 16 Strategies](#3-the-16-strategies)
4. [Execution Pipeline (post-fix path)](#4-execution-pipeline-post-fix-path)
5. [Risk & Governance](#5-risk--governance)
6. [Reconciliation](#6-reconciliation)
7. [Intelligence Layer](#7-intelligence-layer)
8. [Telegram Operator Interface](#8-telegram-operator-interface)
9. [Dashboard (chadtrades.com)](#9-dashboard-chadtradescom)
10. [Services & Timers](#10-services--timers)
11. [Data & Storage](#11-data--storage)
12. [Change Log (delta from v8.1)](#12-change-log-delta-from-v81)
13. [Known Issues](#13-known-issues)
14. [Phase Roadmap](#14-phase-roadmap)
15. [Appendices](#15-appendices)

---

## 0. PREAMBLE

### Document metadata

| Field | Value |
|---|---|
| Document version | 8.2 |
| Date written | 2026-04-26 (UTC) |
| Predecessor | v8.1 — `docs/CHAD_UNIFIED_SSOT_v8.1_2026-04-23.md` (`93bbd70`) |
| Repository HEAD at write time | `6af971d` — *"Fix: pre-Monday wiring — broker truth + universe expansion"* |
| Branch | `main` |

### Server / repo / mode

- **Server:** AWS EC2 `ip-172-31-8-43`, Ubuntu 24.04, kernel 6.17.0-1009-aws.
- **Repo root:** `/home/ubuntu/chad_finale`.
- **Python:** `python3` via `/home/ubuntu/chad_finale/venv` (governance rule
  in `CLAUDE.md` — never invoke `python`).
- **IBKR paper account:** `DUK902770`, $998,109.89 equity (source:
  `runtime/dynamic_caps.json:total_equity` at `2026-04-26T12:45:38Z`).
- **Kraken paper:** connected — 0.0012 BTC + 252.8538 CAD ≈ $184.58 USD
  (source: `runtime/kraken_balances.json` at `2026-04-26T12:42:48Z`).
- **Execution posture:** PAPER — `CHAD_EXECUTION_MODE=paper`. The systemd
  unit ships `dry_run` (`/etc/systemd/system/chad-live-loop.service:13`)
  but `chad/core/live_loop.py:312:_is_paper_mode` treats both `paper` and
  `dry_run` as paper for guard-rebuild purposes.

### Today's session summary (15 commits between v8.1 and v8.2)

In topological order (oldest → newest). The 10 commits *after* v8.1 was
cut form the canonical v8.2 delta; commits 1–5 below predate `93bbd70`
and are listed for completeness because they were already incorporated
into v8.1's commit table.

1. **`20d6f59`** — *Rename: chad/strategies/beta.py → beta_trend.py
   (pure rename)* — clears the `beta.py` filename for the new
   institutional compounder.
2. **`2c6a16b`** — *Build: Beta — institutional long-term compounder
   (original concept)* — adds `chad/strategies/beta.py` and the SEC 13F
   data pipeline.
3. **`3e9db6d`** — *Fix: update Citadel and Appaloosa CIKs in SEC 13F
   fetcher* — corrects two stale CIK numbers that broke 13F downloads.
4. **`c021ccd`** — *Fix: dedup institutional_consensus entries by
   resolved symbol* — collapses duplicate CUSIP rows that resolved to
   the same ticker.
5. **`51c190e`** — *Fix: switch bar provider from reqRealTimeBars to
   polling* — drops the IBKR real-time-data subscription requirement.
6. **`8cd644a`** — *Fix: fill recording — asset_class resolver +
   PendingSubmit timeout* — wires `resolve_asset_class` into evidence
   writes and times out PendingSubmit so it cannot bleed into FILLS.
7. **`e150c0b`** — *Upgrade: complete Telegram intelligence layer* —
   adds the elite-prodigy take, free-text NL routing, alerts.
8. **`81dafce`** — *Build: alpha_crypto real momentum signals* —
   replaces the placeholder `block-all` skeleton with a 3-filter
   momentum signal pipeline (commit `81dafce`).
9. **`2935eea`** — *Feature: Claude chat panel on chadtrades.com
   dashboard* — adds `/api/chat` endpoint backed by Sonnet 4.6.
10. **`064efc7`** — *Tighten chat voice: plain English only, no trading
    jargon* — locks the chat persona.
11. **`d666b5d`** — *Fix: execution pipeline — futures cancellation +
    options routing* — splits buckets by `(symbol, asset_class)` and
    fixes the OPTIONS branch.
12. **`56eaeea`** — *Fix: allocator sleeves, dashboard regime, service
    health, position_guard opened_at, MCL contract* — five-headed fix
    pass.
13. **`f6caf3d`** — *Fix: meta propagation + reconciliation GREEN in
    paper mode* — primary strategy meta now flows from `TradeSignal`
    through `RoutedSignal`.
14. **`21166c9`** — *Fix: 100% PaperExecEvidence parity — futures fills*
    — single chokepoint via `normalize_paper_fill_evidence`.
15. **`6af971d`** — *Fix: pre-Monday wiring — broker truth + universe
    expansion* — paper-mode broker rebuild + universe IWM/TLT.

### What's different from v8.1 (executive delta)

- **Reconciliation:** GREEN in paper mode again. Strategy ledger and
  broker truth tracked separately.
  (`chad/ops/reconciliation_publisher.py:91-92`).
- **Single normalizer chokepoint** for paper fills — three callsites all
  funnel through `normalize_paper_fill_evidence`. PendingSubmit / error
  / unknown can no longer leak into FILLS_*.ndjson.
- **Bucketing fix** — `(symbol, asset_class)` keys. Opposing signals on
  the same futures symbol no longer net to zero
  (`chad/utils/signal_router.py:102` and
  `chad/execution/execution_pipeline.py:680-685`).
- **Meta propagation** — primary strategy's metadata now reaches the
  routed bucket (`chad/utils/signal_router.py:179`).
- **`alpha_crypto` is real** — three-filter momentum logic replaces
  the prior `block-all` placeholder (`chad/strategies/alpha_crypto.py`,
  +453 LoC).
- **Dashboard chat** — Claude-powered advisory chat panel with strict
  plain-English voice (`chad/dashboard/api.py:910-948`,
  `:1121` `/api/chat`).
- **MorningBrief** — elite-prodigy voice (Simons / Dalio / Druckenmiller
  flavour) (`chad/ops/daily_chad_report.py:1196` ff.).
- **`opened_at` invariant** — every `mark_position_open` write carries a
  monotonic `opened_at` timestamp
  (`chad/core/position_guard.py:141-165`).
- **Universe expansion** — `IWM` and `TLT` now in
  `config/universe.json:18-19`, unlocking `delta_pairs` (SPY/IWM,
  QQQ/IWM) and `gamma_reversion` (TLT).
- **Futures separation** — `alpha_futures` (`MES, MNQ, MGC`) and
  `gamma_futures` (`MCL, MYM, M2K, ZN, ZB`) no longer share symbols.
  `zero_net_size` cancellations from opposing signals on the same
  contract are gone.
- **Bar provider** — pure polling via `reqHistoricalData`, no
  `reqRealTimeBars` subscription required.

---

## 1. MISSION & ARCHITECTURE

### Mission (preserved from v8.1)

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a **business**, not a
bot. The business earns money by running a diversified portfolio of
systematic trading strategies over multiple asset classes, compounding
the profits through a structured allocation rule, and eventually paying
a salary to the operator without touching the capital base.

The product goal is plain: **the operator checks their phone at the end
of the day, sees what CHAD made, and does nothing**. Everything
downstream — risk management, execution, reconciliation, health
monitoring — is automation designed to make "do nothing" the safe
default.

### Architecture (ASCII)

```
┌────────────────────┐
│  16 strategy heads │  raw TradeSignals (one bucket per strategy)
└─────────┬──────────┘
          ▼
┌────────────────────────────────────────────────────────────┐
│  signal_router.route                                       │
│   - bucket on (symbol, side, asset_class)                  │
│   - additive netting within bucket                         │
│   - primary_strategy = largest size contributor            │
│   - meta from primary strategy carried into RoutedSignal   │
└─────────┬──────────────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────┐
│  execution_pipeline.split_signals_by_asset_class           │
│       CRYPTO          everything else                      │
│         │                  │                               │
│         ▼                  ▼                               │
│   Kraken lane        IBKR lane                             │
│   build_kraken_     build_execution_plan                   │
│     intents_from_   ↓                                      │
│     routed_signals  build_intents_from_plan                │
│         │           ↓                                      │
│         ▼           routing_gates.run_all_gates            │
│   KrakenExecutor    (A4 / E2 / E5 / R7 / S5)               │
│         │           ↓                                      │
│         │           vote_collector  (S1, min_votes=1)      │
│         │           ↓                                      │
│         │           sizing pipeline  (R3 → R5 → R6 → S5)   │
│         │           ↓                                      │
│         │           IbkrAdapter.submit_strategy_intents    │
│         ▼           │                                      │
│   Kraken fills      IBKR fills                             │
│         │           │                                      │
│         └───────────┴─→ normalize_paper_fill_evidence ─────┤
│                                                            │
│                       ↓                                    │
│                 PaperExecutionEvidenceWriter               │
│                       ↓                                    │
│            data/fills/FILLS_YYYYMMDD.ndjson                │
│                       ↓                                    │
│       position_reconciler / reconciliation_publisher       │
└────────────────────────────────────────────────────────────┘
```

### Built / Degraded / Not yet

| Component | Status | Note |
|---|---|---|
| 16 strategy registry | **BUILT** | Forex commented out (`chad/strategies/__init__.py` registry block) — 16 of 17 active. |
| Signal router (bucket meta carry) | **BUILT** | `chad/utils/signal_router.py:179` (commit `f6caf3d`). |
| Asset-class splitter | **BUILT** | `chad/execution/execution_pipeline.py:1164`. |
| IBKR EMS / OMS | **BUILT** | A1 separation `chad/execution/ems.py` + `oms.py` (Phase-8 Session 9). |
| Kraken EMS / OMS | **BUILT** | REST altname maps `chad/execution/execution_pipeline.py:1228-1264`. |
| Routing gates 5-stack | **BUILT** | `chad/execution/routing_gates.py:run_all_gates`. |
| Sizing R3/R5/R6 | **BUILT** | `target_daily_vol=0.015`, `correlation_threshold=0.65`. |
| Profit lock | **BUILT** | `runtime/profit_lock_state.json` mode=NORMAL, sizing 1.0. |
| SCR | **BUILT** | WARMUP, sizing_factor 0.10, 79 effective trades. |
| Stop bus | **BUILT** | `runtime/stop_bus.json` active=false (cleared 2026-04-22). |
| Edge decay (F4) | **BUILT** | `consecutive_threshold=5`, `min_trades=20`. |
| Reconciliation publisher | **BUILT** | GREEN; paper-mode skip of strategy entries `chad/ops/reconciliation_publisher.py:91-92`. |
| Paper evidence normalizer | **BUILT** | Single chokepoint `chad/execution/paper_exec_evidence_writer.py:925`. |
| Position guard `opened_at` | **BUILT** | `chad/core/position_guard.py:155-157` (commit `56eaeea`). |
| Broker-truth rebuild in paper | **BUILT** | `chad/core/live_loop.py:641-651` (commit `6af971d`). |
| Telegram intelligence layer | **BUILT** | Free-text router + elite voice + alerts (commit `e150c0b`). |
| Dashboard chat | **BUILT** | `/api/chat`, model `claude-sonnet-4-6`. |
| Bar provider polling | **BUILT** | `chad-ibkr-bar-provider.service`. |
| Strategy intelligence cache | **BUILT** | `runtime/strategy_intelligence.json` (48h TTL). |
| Profit-routing 50/30/20 | **DEGRADED** (advisory) | Ledger-only until live (`runtime/profit_routing.json`). |
| `alpha_options` new entries | **DEGRADED** | Existing SPY long blocks new spreads (MAINTAINED state). |
| `omega_vol` health | **DEGRADED** | health_score=0.10 (3 samples). |
| Live trading | **NOT YET** | `runtime/live_readiness.json:ready_for_live=false`. |
| Salary withdrawal automation | **NOT YET** | Manual-only until SCR=CONFIDENT. |

---

## 2. RUNTIME STATE (live snapshot)

All values pulled at write time from the canonical state files. Each row
cites its source.

### SCR — Self-Calibrating Risk

Source: `runtime/scr_state.json` (ts `2026-04-26T12:45:10Z`).

| Field | Value |
|---|---|
| state | `WARMUP` |
| sizing_factor | `0.10` |
| effective_trades | `79` |
| paper_trades | `5000` |
| excluded_untrusted | `4559` |
| win_rate | `0.5443` (≈ 54.4%) |
| sharpe_like | `+0.572` |
| max_drawdown | `-1416.34` |
| total_pnl (effective) | `-6444.64` |
| paper_only | `true` |
| reasons[0] | "Warmup: only 79 effective trades (< 100 required)." |

`warmup_min_trades = 100` per `runtime/scr_config.json`. CAUTIOUS gate
also requires `win_rate ≥ 0.35` AND `sharpe_like ≥ 0.10` AND
`max_drawdown ≥ −$15,000` — all three currently satisfied; only the
trade-count gate keeps us in WARMUP.

### Regime classifier

Source: `runtime/regime_state.json` (ts `2026-04-26T12:45:53Z`).

| Field | Value |
|---|---|
| regime | `trending_bull` |
| previous_regime | `trending_bull` (stable transition) |
| confidence | `0.6864` |
| inputs_used | `realized_vol_percentile, adx, trend_slope, market_breadth` |
| source | `live_loop.run_once` |
| ttl_seconds | 60 |

Inputs from `runtime/market_metrics.json`:
- `adx` (proxy) `33.32`
- `market_breadth` `0.40`
- per-symbol values for 30 symbols (AAPL ADX 36.6, SPY ADX 18.0, MCL ADX
  90.7, etc.)

Active strategies in `trending_bull` (see
`config/regime_activation_matrix.json:9-13`):
`alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_forex,
alpha_options, beta, beta_trend, delta, gamma_futures, omega_macro,
omega_momentum_options`.
*(`alpha_forex` is registry-disabled — see §3.)*

### Equity / PnL

| Field | Value | Source |
|---|---|---|
| account_equity | `$998,109.89` | `runtime/dynamic_caps.json:total_equity` |
| portfolio_risk_cap | `$49,905.49` (5% daily-risk fraction) | `runtime/dynamic_caps.json:portfolio_risk_cap` |
| daily realized PnL today | `$0.00` | `runtime/profit_lock_state.json:inputs.realized_pnl` |
| trade_count today | `2` | `runtime/profit_lock_state.json:inputs.trade_count` |

### Profit lock

Source: `runtime/profit_lock_state.json` (ts `2026-04-26T12:46:15Z`).

- mode = `NORMAL`
- sizing_factor = `1.00`
- stop_new_entries = `false`
- daily_loss_limit_dollars = `$29,943.30` (3% of equity)
- daily_loss_today = `$0.00`
- profit_lock_active = `false`

### Reconciliation

Source: `runtime/reconciliation_state.json` (ts
`2026-04-26T12:42:02Z`).

- status = **`GREEN`**
- broker_source = `ibkr:clientId=83`
- chad_open = `1`, broker_positions = `1`
- worst_diff = `0.0`
- mismatches = `[]`, drifts = `[]`
- excluded_symbols = `[]`, futures_excluded_symbols = `[]`
- ttl_seconds = 300

GREEN/YELLOW/RED criteria
(`chad/ops/reconciliation_publisher.py:10-13`):
- GREEN — every open CHAD position matches broker within 1 unit.
- YELLOW — minor discrepancy ≤ 2 units OR broker has no-guard symbols.
- RED — major discrepancy OR IBKR unavailable.

### Stop bus

Source: `runtime/stop_bus.json`.

- active = `false`
- cleared_at = `2026-04-22T01:56:50Z`
- cleared_by = `smoke_test`

### Open positions

From `runtime/position_guard.json` — entries with `open: true`:

| Key | Strategy | Symbol | Side | Qty | opened_at |
|---|---|---|---|---|---|
| `broker_sync\|SPY` | broker_sync | SPY | BUY | 13 | 2026-04-26T12:45:51Z |
| `alpha\|SPY` | alpha | SPY | BUY | 26 | 2026-04-24T20:18:23Z |
| `alpha_options\|SPY` | alpha_options | SPY | BUY | 1 | 2026-04-24T20:18:23Z |
| `delta\|SPY` | delta | SPY | BUY | 49 | 2026-04-24T21:49:23Z |

Total open in guard: 4 entries (1 broker_sync echo + 3 strategy
positions, all on SPY). Notional ≈ 13 × 714.01 = $9,282 (broker view);
strategies hold an additional 76 SPY paper shares ≈ $54,265.

### Market snapshot (price_cache.json @ `2026-04-26T12:45:36Z`)

23 symbols cached, 60 s TTL. Selected:

| Symbol | Last | Symbol | Last |
|---|---|---|---|
| SPY | 714.01 | QQQ | 663.46 |
| AAPL | 270.90 | MSFT | 423.88 |
| GOOGL | 343.59 | NVDA | 208.10 |
| GLD | 432.60 | TLT | 86.72 |
| IWM | 276.65 | VWO | 59.07 |
| MES | 7143.50 | MNQ | 26934.00 |
| MCL | 95.85 | MGC | 4705.10 |
| ZN | 111.046875 | ZB | 113.84375 |
| SVXY | 49.71 | UVXY | 39.77 |
| VIXY | 28.66 | M6E | 1.1717 |

VIX is not currently in `price_cache.json`. Daily close is at
`data/bars/1d/VIX.json` (consumed by `omega`, `omega_vol`,
`omega_macro`, `omega_momentum_options`, `alpha_options` via the
context wiring shipped in commit `6ea83c6`).

Kraken (`runtime/kraken_balances.json`): BTC `0.0012`, CAD `252.85`,
USD-equivalent `$184.58`.

### Services

Total `chad-*` units active: **47** (services + timers, output of
`systemctl list-units 'chad-*' --no-legend`).
Failed units: **0**.

### Bar freshness

- 1d bars: `data/bars/1d/` — 30 symbols (incl. SOL-USD, VXX, SIL).
- 1m bars: `data/bars/1m/` — 10 symbols polled via
  `chad-ibkr-bar-provider.service` every 30 s.
- VIX: `data/bars/1d/VIX.json` (CBOE nightly).

### Strategy intelligence

Source: `runtime/strategy_intelligence.json` — most recent
`regime_profile` entries `2026-04-26T03:27Z`. AI cache TTL 48h
(`chad/ops/daily_chad_report.py:1353`). All current entries fall back
to `profile=normal` because the macro context provider returned
`unknown` for VIX and macro-risk inputs at the snapshot time — known
classifier-input gap, not a fault.

### Institutional consensus

Source: `runtime/institutional_consensus.json` (updated
`2026-04-26T00:00:05Z`). 7 funds, 25 holdings ranked by conviction.

Top 10:

| Rank | Symbol | Funds | Conviction |
|---|---|---|---|
| 1 | AAPL | 6 | 0.4030 |
| 2 | AMZN | 6 | 0.3887 |
| 3 | GOOGL | 7 | 0.3856 |
| 4 | META | 5 | 0.3219 |
| 5 | V | 5 | 0.3164 |
| 6 | MA | 5 | 0.3157 |
| 7 | UNH | 5 | 0.3156 |
| 8 | KO | 5 | 0.3155 |
| 9 | SPY | 5 | 0.3143 |
| 10 | NVDA | 4 | 0.2776 |

### Profit routing ledger

Source: `runtime/profit_routing.json` (updated `2026-04-26T11:14:13Z`).

- decisions logged: 19
- totals (advisory):
  - trading_capital = `$938.88`
  - beta_allocation = `$563.33`
  - amplifier_allocation = `$375.55`

Largest single decision: `alpha_futures` `$1,058.00` realized →
`(529.00, 317.40, 211.60)` at `2026-04-24T21:51:22Z`.

---

## 3. THE 16 STRATEGIES

For each strategy: source path, sleeve, weight, dollar cap, regime
gating, fire window, universe, signal conditions, current health,
status, and any caveats. Weights from `config/strategy_weights.json`,
caps from `runtime/dynamic_caps.json`, regime gating from
`config/regime_activation_matrix.json`, sleeve membership from
`chad/risk/dynamic_risk_allocator.py:392-399`.

> **Sleeve recap** — the 50/30/20 chassis:
> - **ALPHA sleeve** (50%): `alpha, alpha_futures, alpha_intraday,
>   alpha_options, omega_momentum_options, gamma, gamma_futures,
>   gamma_reversion`.
> - **BETA sleeve** (30%): `beta, beta_trend`.
> - **ADAPTIVE sleeve** (20%): `omega, omega_macro, omega_vol, delta,
>   delta_pairs, alpha_crypto`.
>
> `omega_momentum_options` lives in ALPHA per
> `chad/risk/dynamic_risk_allocator.py:393-395`. The docstring on
> `enforce_chassis` (`:415`) is a stale summary — code is authoritative.

### alpha — Intraday tactical momentum brain

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha.py` |
| Sleeve | ALPHA |
| Weight | 0.16 (largest) |
| Dollar cap | `$7,984.88` (`runtime/dynamic_caps.json:strategies.alpha.dollar_cap`) |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Per-symbol 3 signals/day cap; max 8 signals per cycle (`alpha.py:196,319`). |
| Universe | Legend-driven via `ctx.legend.weights`, fallback to `ctx.prices` keys (`alpha.py:216`). |
| Conditions | Four entry regimes: uptrend (EMA fast > slow, momentum ≥ 0.35 ATR), recovery (price > EMA slow, momentum ≥ 0.175 ATR), downtrend (mirrored shorts), chop (fade mid-band) (`alpha.py:254-333`). |
| Health | 0.8167 on 32 samples; win_rate 0.389; normalized_sharpe 1.0 (`runtime/strategy_health.json`). |
| Open positions | 1 — `alpha\|SPY` 26 BUY (opened 2026-04-24T20:18Z). |
| Status | **ACTIVE**. |
| Notes | Position MAINTAINED on SPY blocks repeat opens until current closes — by design (no flip). |

### alpha_intraday — Delta high-convexity day-trading brain

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_intraday.py` |
| Sleeve | ALPHA |
| Weight | 0.03 |
| Dollar cap | `$1,497.16` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | 1m bars with daily fallback; 10-min per-symbol cooldown (`alpha_intraday.py:40,324-328`). |
| Universe | SPY, QQQ, AAPL, NVDA, MSFT, GOOGL, BAC, MES, MNQ, BTC-USD (`alpha_intraday.py:31-34`). |
| Conditions | Any one of: vol explosion (current ATR ≥ 2× baseline), momentum surge (5-bar accel aligned with 20-bar trend), mean-reversion snap (Bollinger + RSI extreme, equities only) (`alpha_intraday.py:6-12,244-296`). |
| Health | 0.40 on 4 samples (low confidence). |
| Open positions | None (3 closed: `NVDA, QQQ, SPY` — last closed 2026-04-25T23:02Z). |
| Status | **ACTIVE**. |
| Notes | All signals tagged `high_convexity=True`; 1.5% stop, 4.5% target, 30-bar max hold. |

### alpha_options — Defined-risk vertical spreads

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_options.py` (+ `alpha_options_config.py`) |
| Sleeve | ALPHA |
| Weight | 0.04 |
| Dollar cap | `$1,996.22` |
| Active in | `trending_bull, trending_bear, volatile` |
| Silent in | `ranging, unknown, adverse` |
| Window | Triggered by alpha/gamma/gamma_reversion signals at confidence ≥ 0.70 (`alpha_options.py:96`). Max 4 open spreads. |
| Universe | SPY only — most liquid options book (`alpha_options_config.py:25-27`). |
| Conditions | Bullish source signal → bull call spread (21–45 DTE, 2% OTM, 5% width). Bearish → bear put spread (mirror) (`alpha_options.py:206-262,460-468`). |
| Health | n/a (not computed — sample threshold). |
| Open positions | 1 — `alpha_options\|SPY` 1 BUY (opened 2026-04-24T20:18Z). |
| Status | **CONDITION-GATED** (existing position MAINTAINED until close → no new entries). |
| Notes | Pricing from `runtime/options_chains_cache.json` with Black-Scholes synthetic fallback. |

### alpha_futures — Futures momentum engine

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_futures.py` (+ `alpha_futures_config.py`) |
| Sleeve | ALPHA |
| Weight | 0.09 (second-largest) |
| Dollar cap | `$4,491.49` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | MES, MNQ continuous; **MCL/MGC restricted 13:30–20:00 UTC for new entries** (overnight liquidity gate `alpha_futures.py:488-497`). |
| Universe | **MES, MNQ, MGC** (`alpha_futures.py:98` — `ALPHA_FUTURES_UNIVERSE = (MES, MNQ, MGC)`). MCL was removed and assigned to `gamma_futures` to eliminate the zero-net-size cancellation observed pre-`6af971d`. The legacy 4-symbol default still appears in `alpha_futures_config.py:55-60` but the strategy module's hardcoded tuple wins. |
| Conditions | Momentum: price > EMA_fast > EMA_slow (BUY) or mirror (SELL). Breakout override: 20-bar high/low. Exits on 2× ATR stop, EMA_slow cross, 20-bar time stop (`alpha_futures.py:376-412,507-523`). |
| Health | 0.4713 on **329 samples** — largest sample base in the system; win_rate 0.571. |
| Open positions | None (`MGC` and `MES` both closed 2026-04-25T23-26 UTC). |
| Status | **ACTIVE**. |
| Notes | 1.25× risk multiple, max 5 contracts per instrument, $1M ADV gate. |

### alpha_crypto (weight key: `crypto`) — Crypto momentum signals

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_crypto.py` (commit `81dafce`) |
| Sleeve | ADAPTIVE |
| Weight | 0.04 (key `crypto`) |
| Dollar cap | `$1,996.22` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Always armed when regime allows; signal pipeline emits long-only (`alpha_crypto.py:641-645,627-758`). |
| Universe | Default `BTC-USD, ETH-USD, SOL-USD`; CAD pairs (`BTC-CAD, ETH-CAD`) added when Kraken paper account holds ZCAD (`alpha_crypto.py:75-86`). Kraken REST altname routing via `normalize_kraken_pair` (`execution_pipeline.py:1228-1264`). |
| Conditions | Three stacked filters: (1) SMA20 momentum breakout — price > SMA20, 3-day return ≥ 1.5%; (2) 5d/20d vol-ratio expansion ≥ 0.7 (boost ≥ 1.5); (3) regime multiplier — 0.5 in trending_bear, 1.0 elsewhere. Min strength 0.3, all long-only. |
| Health | not computed (no samples yet under the new logic). |
| Open positions | None. |
| Status | **ACTIVE**, awaiting first vol-compression-cleared signal under the new logic. |
| Notes | Weight key in config is `crypto`; the dynamic-caps shim normalises this to `alpha_crypto` (`chad/risk/dynamic_caps.py:67`). |

### beta — Institutional long-term compounder

| Field | Value |
|---|---|
| Source | `chad/strategies/beta.py` (commit `2c6a16b`) |
| Sleeve | BETA |
| Weight | 0.05 (carved from beta_trend in action `beta_carveout_from_beta_trend_20260423`) |
| Dollar cap | `$2,495.27` |
| Active in | `trending_bull, trending_bear, ranging, volatile, unknown` (every non-adverse regime) |
| Silent in | `adverse` |
| Window | Max 2 signals per cycle; max 3/week rolling; per-symbol 7-day rebalance gate (`beta.py:73-95`). |
| Universe | Whatever ranks in `runtime/institutional_consensus.json` (currently 25 names, top 10 listed in §2). Consensus must be ≤ 45 days old (13F quarterly cycle). |
| Conditions | If `target_weight − current_weight ≥ 2%`, emit a BUY sized to fill ~50% of the gap (conservative). Long-only. |
| Health | n/a. |
| Open positions | None. |
| Status | **ACTIVE** (advisory funding via 30%-of-realized-profit slice). |
| Notes | Funded by `profit_router` 30% slice; ledger-only until live. |

### beta_trend — Legend-driven long-term ETF / equity allocator

| Field | Value |
|---|---|
| Source | `chad/strategies/beta_trend.py` (renamed from old `beta.py`, commit `20d6f59`) |
| Sleeve | BETA |
| Weight | 0.20 (reduced from 0.25 after Beta carve-out) |
| Dollar cap | `$9,981.10` (largest single dollar cap) |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Once-per-UTC-day per symbol (hard gate); max 20 signals/day; 21-day hold before add-ons; 14-day cooldown after exit (`beta_trend.py:57-62`). |
| Universe | Top legend-weighted names (min 5%, max 10 symbols), EWMA-smoothed (`beta_trend.py:172-186`). |
| Conditions | Entry if flat: BUY size = clamp(3 + legend_weight × 10, 3, 8). Add-on after 21d: 50% of entry size, 95% confidence (`beta_trend.py:240-306`). |
| Health | n/a. |
| Open positions | None. |
| Status | **ACTIVE**. |
| Notes | The biggest single dollar bucket in the book. |

### gamma — Activated swing engine

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma.py` |
| Sleeve | ALPHA |
| Weight | 0.07 |
| Dollar cap | `$3,493.38` |
| Active in | `ranging, volatile, unknown` |
| Silent in | `trending_bull, trending_bear, adverse` |
| Window | No fixed schedule; bar-driven. Min 60 bars (`gamma.py:77-89`). |
| Universe | Pulled dynamically from `ctx.bars` keys. |
| Conditions | Trend regime: EMA_fast > slow, price > fast, momentum ≥ 0.35 ATR. Range regime: deviates from EMA_slow by ≥ 0.75 ATR (mean reversion). Vol gate: ATR% ∈ [0.15%, 4.0%]. Anti-chase: range/ATR ≤ 3.2 (`gamma.py:70-106,369-428`). |
| Health | n/a. |
| Open positions | None. |
| Status | **REGIME-SILENT** (currently `trending_bull`); awaiting a `ranging` or `volatile` regime. |

### gamma_futures — Futures mean-reversion counterpart

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma_futures.py` (+ config) |
| Sleeve | ALPHA |
| Weight | 0.05 |
| Dollar cap | `$2,495.27` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Continuous (energy 24/5; bonds liquid hours). |
| Universe | **MCL** primary; extends with `MYM, M2K` if bars exist; otherwise `ZN, ZB` (`gamma_futures.py:86-122`). **Disjoint from alpha_futures** post-`6af971d`. |
| Conditions | Short when `RSI > overbought (default 70)` AND `price > BB_upper`; Long when `RSI < oversold (default 30)` AND `price < BB_lower`. ATR-based sizing: 1.2% risk, max $40k notional (`gamma_futures.py:14-50`). |
| Health | 1.0 on **2 samples** — too small to be meaningful. |
| Open positions | None (`MCL` closed `2026-04-26T12:45Z` via broker_truth_rebuild). |
| Status | **ACTIVE**. |
| Notes | The 1058 USD profit-routing event on 2026-04-24 came from this strategy. |

### gamma_reversion — ETF statistical mean-reversion

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma_reversion.py` (+ config) |
| Sleeve | ALPHA |
| Weight | 0.04 |
| Dollar cap | `$1,996.22` |
| Active in | `ranging` |
| Silent in | every other regime |
| Window | Continuous within active regime. Min 40 bars; RSI(14), Boll(20), Z(20). |
| Universe | **SPY, QQQ, GLD, TLT** (`gamma_reversion_config.py:28-34`). IWM removed after `-1.06` Sharpe in backtest. |
| Conditions | 3/3 confluence — SHORT: RSI > 72 AND (price > BB_upper OR z > 1.8) AND ROC > 0; LONG: mirror with 28/-1.8/<0. GLD requires strict 3/3 (backtest showed 2/3 ⇒ 38% win). 2.5× ATR stop, 15-bar time stop, SMA20 cross = target (`gamma_reversion.py:19-27,103`). |
| Health | n/a. |
| Open positions | None. |
| Status | **REGIME-SILENT** (currently `trending_bull`); waits for `ranging`. |

### delta — Cross-asset convexity hunter

| Field | Value |
|---|---|
| Source | `chad/strategies/delta.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.02 (smallest) |
| Dollar cap | `$998.11` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | No per-symbol cooldown (conviction-driven, wealth-mode). Cash floor $2.5k (`delta.py:108,115`). |
| Universe | `ctx.delta_universe` override → legend top weights → price keys; max 4 symbols (`delta.py:243-266`). |
| Conditions | Conviction ≥ 0.65 required. Trend: EMA_fast > slow, price > fast. Breakout: price > 20-bar high + 0.10 × ATR. Momentum: (price − EMA_fast)/ATR ≥ 0.45. Conviction = vol_expanding 0.35 + breakout 0.25 + momentum 0.25 + event_ok 0.10 + liq_ok 0.05 (`delta.py:312-328,508-569`). |
| Health | 0.925 on 7 samples; win_rate 0.75. |
| Open positions | 1 — `delta\|SPY` 49 BUY (opened 2026-04-24T21:49Z). |
| Status | **ACTIVE**, strongest health score in the book. |

### delta_pairs — Market-neutral ETF pairs trader

| Field | Value |
|---|---|
| Source | `chad/strategies/delta_pairs.py` (+ config) |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| Dollar cap | `$2,495.27` |
| Active in | `ranging` |
| Silent in | every other regime |
| Window | Continuous within active regime. 60-day lookback, min 40 bars. |
| Universe | **SPY/QQQ** (r 0.993), **SPY/IWM** (0.967), **QQQ/IWM** (0.941) — pair list `delta_pairs.py:71-75`. IWM newly available after universe expansion in commit `6af971d`. |
| Conditions | z = (ratio_now − mean_60) / std_60. Entry at \|z\| ≥ 2.0; exit ≤ 0.5; stop ≥ 3.5. Both legs same unit count, max 50 units/leg (`delta_pairs.py:86-92,212,251-284`). |
| Health | n/a. |
| Open positions | None. |
| Status | **REGIME-SILENT**. |
| Notes | Linked by `pair_id`; SHORT high ratio + LONG low ratio. |

### omega — Wealth-safe hedge sleeve

| Field | Value |
|---|---|
| Source | `chad/strategies/omega.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| Dollar cap | `$2,495.27` |
| Active in | `volatile, unknown` |
| Silent in | trending regimes (by design — hedge is dormant in calm seas) |
| Window | Cooldown 60 minutes between activation/deactivation cycles. |
| Universe | **SH** (inverse SPY) and **PSQ** (inverse QQQ) (`omega.py:74-75`). |
| Conditions | Activation requires ≥ 2 sensor agreement: drawdown ≤ -6% (deactivate -3%), ATR% ≥ 3% (deactivate 2%), VIX ≥ 25 (deactivate 20). 5 base hedge units, max 25 per symbol (`omega.py:82-130`). |
| Health | n/a. |
| Open positions | None. |
| Status | **ARMED**, currently dormant — `trending_bull` regime + VIX below 20 (last close ~18.92). |

### omega_vol — VIX-linked volatility alpha

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_vol.py` (+ config) |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| Dollar cap | `$2,495.27` |
| Active in | `volatile` |
| Silent in | every other regime |
| Window | 5-state VIX regime: `LOW_VOL` (< 15), `NORMAL_VOL` (15-22), `ELEVATED` (22-30), `CRISIS` (≥ 30), `VOL_CRUSH` (≥ 20% drop from peak) (`omega_vol.py:67-101`). UVXY hard 10-bar time stop (structural decay). |
| Universe | **SVXY** (short vol, contango) and **UVXY** (long vol, spike) — both long & short legal (`omega_vol_config.py:20-23`). |
| Conditions | LOW_VOL → SVXY BUY; NORMAL_VOL → mild SVXY; ELEVATED → skip; CRISIS → UVXY BUY; VOL_CRUSH → SVXY mean-revert. 3-8 units, max 3% equity per position. Confidence ≥ 0.65. |
| Health | **0.10 on 3 samples** — flagged. Win rate 0.0; slippage_ratio 0.5; regime_alignment 0.0. |
| Open positions | None (last `omega_vol\|SVXY` 5 BUY closed via broker_truth_rebuild this cycle). |
| Status | **DEGRADED** (severity = sample-too-small but health score is the lowest in the book). Watch list. |

### omega_macro — Macro regime futures

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_macro.py` (+ config) |
| Sleeve | ADAPTIVE |
| Weight | 0.03 |
| Dollar cap | `$1,497.16` |
| Active in | every regime except `adverse` (universally eligible). |
| Silent in | `adverse` |
| Window | Continuous; min 40 bars. Risk budget 1.0% (tighter than alpha's 1.5%). |
| Universe | **ZN, ZB, M6E** (`omega_macro_config.py:29-33`). |
| Conditions | 4-state macro regime: `RISK_OFF` (VIX>25, DD<-5% → ZN/ZB BUY, M6E SELL), `RISK_ON` (VIX<18, DD>-2% → mirror), `STAGFLATION` (ZN/ZB BUY, M6E SELL), `NEUTRAL` (no signals). ATR sizing × point_value; max 3 contracts each (`omega_macro.py:148-160,200-250`). |
| Health | n/a. |
| Open positions | None. |
| Status | **ACTIVE**, currently `RISK_ON`-leaning per low VIX. |

### omega_momentum_options — Intraday single-leg options momentum

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_momentum_options.py` |
| Sleeve | ALPHA (per `dynamic_risk_allocator.py:393-395`) |
| Weight | 0.03 |
| Dollar cap | `$1,497.16` |
| Active in | `trending_bull, trending_bear, volatile` |
| Silent in | `ranging, unknown, adverse` |
| Window | Market hours 9:45 AM ET → 3:30 PM ET; hard exit by 3:45 PM ET; 15-min per-symbol cooldown; max 3 concurrent (`omega_momentum_options.py:59-70,102-112`). |
| Universe | **SPY, QQQ, AAPL, NVDA, MSFT** (`omega_momentum_options.py:55`). |
| Conditions | Both required: (1) momentum (0.3% in 5 bars) + EMA slope + 1.5× volume; (2) VIX regime filter (skip if VIX > 40). 50% profit target, 25% stop loss. ATM/nearest, 1-3 DTE. |
| Health | n/a. |
| Open positions | None. |
| Status | **ARMED** within session window. |

### Strategy summary table

| # | Strategy | Sleeve | Weight | Cap (USD) | Health | Open | Status |
|---|---|---|---|---|---|---|---|
| 1 | alpha | ALPHA | 0.16 | 7,985 | 0.82 (32) | 1 | ACTIVE |
| 2 | alpha_intraday | ALPHA | 0.03 | 1,497 | 0.40 (4) | 0 | ACTIVE |
| 3 | alpha_options | ALPHA | 0.04 | 1,996 | n/a | 1 | CONDITION-GATED |
| 4 | alpha_futures | ALPHA | 0.09 | 4,491 | 0.47 (329) | 0 | ACTIVE |
| 5 | alpha_crypto | ADAPTIVE | 0.04 | 1,996 | n/a | 0 | ACTIVE |
| 6 | beta | BETA | 0.05 | 2,495 | n/a | 0 | ACTIVE |
| 7 | beta_trend | BETA | 0.20 | 9,981 | n/a | 0 | ACTIVE |
| 8 | gamma | ALPHA | 0.07 | 3,493 | n/a | 0 | REGIME-SILENT |
| 9 | gamma_futures | ALPHA | 0.05 | 2,495 | 1.00 (2) | 0 | ACTIVE |
| 10 | gamma_reversion | ALPHA | 0.04 | 1,996 | n/a | 0 | REGIME-SILENT |
| 11 | delta | ADAPTIVE | 0.02 | 998 | 0.93 (7) | 1 | ACTIVE |
| 12 | delta_pairs | ADAPTIVE | 0.05 | 2,495 | n/a | 0 | REGIME-SILENT |
| 13 | omega | ADAPTIVE | 0.05 | 2,495 | n/a | 0 | ARMED |
| 14 | omega_vol | ADAPTIVE | 0.05 | 2,495 | **0.10 (3)** | 0 | DEGRADED |
| 15 | omega_macro | ADAPTIVE | 0.03 | 1,497 | n/a | 0 | ACTIVE |
| 16 | omega_momentum_options | ALPHA | 0.03 | 1,497 | n/a | 0 | ARMED |
| — | **Σ** | — | **1.00** | **49,907** | — | **3** | — |

(Caps total ≈ portfolio_risk_cap of $49,905.49.)

---

## 4. EXECUTION PIPELINE (post-fix path)

### Stages

```
raw signals (TradeSignal[])
  → signal_router.route                       # bucket on (symbol,side,asset_class)
  → split_signals_by_asset_class              # CRYPTO vs IBKR
  → IBKR lane: build_execution_plan           # net opposing legs, drop missing prices
              → build_intents_from_plan       # StrategyTradeIntent[]
              → routing_gates.run_all_gates   # A4 / E2 / E5 / R7 / S5
              → vote_collector.submit_intent  # S1 (min_votes=1)
              → sizing pipeline               # R3 → R5 → R6 → S5
              → IbkrAdapter.submit_strategy_trade_intents
  → Kraken lane: build_kraken_intents_from_routed_signals
                → KrakenExecutor.submit
  → fills → normalize_paper_fill_evidence → PaperExecutionEvidenceWriter
  → position_guard, slippage_tracker, edge_decay_monitor, expectancy_tracker
```

### Bucket key change (commit `d666b5d`)

Before: bucket key was `(symbol, side)`. SPY/ETF and SPY/OPTIONS merged
into a single bucket; the ETF won and the option leg was lost. Opposing
side signals on the same futures symbol from different strategies
netted to zero and dropped silently.

After: bucket key is `(symbol, asset_class)` —
`chad/execution/execution_pipeline.py:680-685`:

```python
# Bucket key: (symbol, asset_class). Bucketing by symbol alone caused
# SPY/ETF and SPY/OPTIONS to merge (ETF wins, options identity lost) and
# caused opposing-side signals from different strategies on the same
# futures symbol to net to zero and silently drop.
```

The same bucketing also applies in the upstream router
(`chad/utils/signal_router.py:102`):

```python
key = (sig.symbol, sig.side, sig.asset_class)
```

### Meta propagation (commit `f6caf3d`)

Before: each `TradeSignal.meta` was discarded when buckets merged into
`RoutedSignal`. Reconciliation lost provenance — the publisher couldn't
attribute a fill to the originating strategy if multiple signals
collided in a bucket.

After: the router tracks meta per-strategy in the bucket
(`signal_router.py:109` `meta_by_strategy`), then on emission selects
the meta from the bucket's `primary_strategy` (largest size
contributor) (`signal_router.py:179`):

```python
meta=data.get("meta_by_strategy", {}).get(primary) or {},
```

This is what allows `position_reconciler` to assert paper-mode
strategy ownership and skip drift checks against IBKR for
non-`broker_sync` entries.

### Asset class router

`chad/execution/execution_pipeline.py:1164-1187` —
`split_signals_by_asset_class` returns `(ibkr_signals, kraken_signals)`:

- `AssetClass.CRYPTO` → Kraken lane.
- Everything else (or missing/unknown) → IBKR lane.

The Kraken lane uses `normalize_kraken_pair`
(`execution_pipeline.py:1228-1264`) which maps CHAD canonical symbols
(`BTC-USD, ETH-USD, SOL-USD`) to Kraken REST altnames (`XBTUSD, ETHUSD,
SOLUSD`) — wsname (`XBT/USD`) is rejected by `/0/private/AddOrder`.

CAD-quoted alternates (`XBTCAD, ETHCAD`) are wired so that when USD
buying power is empty and Kraken holds ZCAD, `alpha_crypto` can still
emit fillable intents.

### OPTIONS branch

`chad/execution/execution_pipeline.py:616-665` —
`resolve_ibkr_instrument_spec` dispatches on `AssetClass`:

```python
if asset_class == AssetClass.OPTIONS:
    return _resolve_options_spec(symbol)
```

Pre-`d666b5d`, `OPTIONS` signals were proxied through the underlying
ETF — losing the options identity. The OPT spec resolver now produces
an IBKR `Contract` with `sec_type="OPT"` and a real expiry/strike
(when the strategy supplies it). `alpha_options` populates expiry from
the cached chain (`runtime/options_chains_cache.json`); when the chain
is unavailable the synthetic Black-Scholes fallback sizes correctly
but the spec carries placeholder strike/expiry — see Known Issues
§13 for the open metadata gap.

### Paper fill normalization (commit `21166c9`)

Single chokepoint `normalize_paper_fill_evidence` at
`chad/execution/paper_exec_evidence_writer.py:925-983` enforces four
invariants in paper mode:

1. `asset_class` is never blank/unknown when the symbol is recognisable
   (lazy import of `resolve_asset_class` with inline futures-root
   fallback).
2. `fill_price > 0` when `runtime/price_cache.json` has a price (with
   futures contract-month normalisation, e.g. `MGCK6 → MGC`); falls
   back to `expected_price`.
3. `status` is rewritten to `paper_fill` when raw status is
   `PendingSubmit`, `error`, etc. and a positive price is available.
4. **Hard invariant** — raises `ValueError` if status would still be
   pending after normalisation. Untrusted records cannot persist.

`is_live=True` records pass through unchanged (broker is the source of
truth — never overwrite).

### Three writers, one chokepoint

| Writer | Path |
|---|---|
| Live loop | `chad/core/live_loop.py:1129-1151` |
| Position reconciler | `chad/core/position_reconciler.py:217,282` |
| Timer-driven paper executor | `/usr/local/bin/chad_paper_trade_executor.py:11,222` |

All three import `normalize_paper_fill_evidence` directly. Plus the
public `write_paper_exec_evidence` (`paper_exec_evidence_writer.py:998`)
calls the normalizer one more time as a safety net for any caller that
bypasses the explicit helper.

---

## 5. RISK & GOVERNANCE

### 50/30/20 chassis

Module: `chad/risk/dynamic_risk_allocator.py:392-485`. Frozen sets
define sleeve membership; `enforce_chassis(weights)` only fires when
sleeve drift exceeds `CHASSIS_TOLERANCE = 0.05`.

```python
ALPHA_STRATEGIES = frozenset({
    "alpha", "alpha_futures", "alpha_intraday", "alpha_options",
    "omega_momentum_options", "gamma", "gamma_futures", "gamma_reversion",
})
BETA_STRATEGIES = frozenset({"beta", "beta_trend"})
ADAPTIVE_STRATEGIES = frozenset({
    "omega", "omega_macro", "omega_vol", "delta", "delta_pairs",
    "alpha_crypto",
})
ALPHA_TARGET = 0.50
BETA_TARGET = 0.30
ADAPTIVE_TARGET = 0.20
```

Targets are 50% / 30% / 20% with a 5% tolerance band. Strategies not
in any sleeve get zeroed (`:472-475`). Disable via env
`CHAD_CHASSIS_ENFORCEMENT=0`.

### Dynamic risk allocator pipeline

```
config/strategy_weights.json
   ↓  base weights (16 strategies, sum=1)
correlation overlay  (gracefully fails-soft to base after stale file
                       was archived 2026-03-26)
   ↓
chassis enforcement (50/30/20)
   ↓
dollar caps:
   dollar_cap = total_equity × daily_risk_fraction × normalized_weight
   total_equity = $998,109.89
   daily_risk_fraction = 0.05
   ⇒ portfolio_risk_cap = $49,905.49 (5% of equity)
   ↓
runtime/dynamic_caps.json
```

### Sizing — equity / ETF intents

`chad/execution/execution_pipeline.py:_apply_sizing_layer`:

```
base → R3 vol_adjusted → R5 composite_cap → R6 correlation_monitor → S5 event_gate → OMS
```

| Layer | Module | Behavior |
|---|---|---|
| **R3** | `chad/risk/vol_adjusted_sizer.py` | `mult = clamp(target_daily_vol/realized_vol, 0.1, 2.0)`, `target=0.015`, 20-day lookback (`config/sizing_config.json`). |
| **R5** | `chad/risk/composite_size_cap.py` | `min(vol_adj, max_per_symbol=300, sector_remaining=$5k, 0.5%×ADV, 5%×equity/ref_px)`. |
| **R6** | `chad/risk/correlation_monitor.py` | Multiplies by `threshold/avg_corr` (floor 0.1) when book \|r\| > 0.65. |
| **S5** | `chad/analytics/event_calendar.py` | Inside event window: `urgency=high` reject; `normal` reduce 50%. |

Futures and forex bypass — they have specialised upstream sizers.
Crypto rides a Kraken builder path with S3 confidence attenuation +
volume gate.

### SCR — Self-Calibrating Risk

State machine in `chad/risk/scr_state.py`. Runtime at
`runtime/scr_state.json`. Config at `runtime/scr_config.json`.

| State | sizing_factor | What changes |
|---|---|---|
| WARMUP (current) | 0.10 | All trades 10% of planned size; live blocked. |
| CAUTIOUS | 0.25 | 2.5× upsize; still paper-only. |
| CONFIDENT | 1.00 | Full size; live-eligible (still gated by operator GO). |
| PAUSED | 0.00 | Hard stop. |

Currently we are 21 effective trades short of the 100 needed for
CAUTIOUS at WARMUP fill cadence; both other CAUTIOUS gates (win_rate
≥ 0.35, sharpe_like ≥ 0.10) are already satisfied (54.4% / +0.572).

### Profit lock — circuit breaker

Module: `chad/risk/profit_lock.py`. State:
`runtime/profit_lock_state.json`. Six modes, equity-driven.

| Mode | Trigger (% of equity) | Sizing factor |
|---|---|---|
| NORMAL | default | 1.00 |
| WARN | profit ≥ 1.5% | 1.00 (flagged) |
| LOCK1 | profit ≥ 3.0% | 0.50 |
| LOCK2 | profit ≥ 5.0% | 0.25 |
| LOCK3 | profit ≥ 8.0% | 0.10 |
| HARD_STOP | profit ≥ 10.0% | 0.00 |

Currently NORMAL: realized_pnl_today=$0, daily_loss_limit=$29,943.30.

### Position guard `opened_at` invariant (commit `56eaeea`)

Every `mark_position_open` write now stamps a UTC `opened_at`
(`chad/core/position_guard.py:155-157`):

```python
state[key] = {
    "open": True,
    "opened_at": prior_opened or now_iso,
    "updated_at_utc": now_iso,
    ...
}
```

When the prior entry was already open and same-side, `prior_opened` is
preserved so the open-time clock doesn't reset on size adjustments.

### Broker truth rebuild in paper mode (commit `6af971d`)

`chad/core/live_loop.py:641-651`:

```python
if _is_paper_mode():
    _rebuild_guard_from_paper_ledger(logger)
    try:
        _rebuild_guard_from_broker(logger)
    except Exception as bexc:
        logger.warning("Broker guard rebuild failed in paper mode (non-fatal): %s", bexc)
else:
    _rebuild_guard_from_broker(logger)
```

In paper mode the guard is rebuilt twice each cycle: first from the
local trade-closer ledger so simulated positions survive across
cycles, then from IBKR so `broker_sync` entries reflect the real
paper-account positions. Strategy entries are tracked separately, so
reconciliation can split "drift" (IBKR-initiated) from "mismatch"
(strategy-vs-broker).

### Stop bus

Triggers (`chad/risk/stop_bus_triggers.py`):
1. `daily_loss_breach`
2. `reject_rate_spike`
3. `data_staleness`
4. `broker_latency_spike`

State at `runtime/stop_bus.json`. Currently `active=false`.

### Edge decay (F4)

Module: `chad/risk/edge_decay_monitor.py`. Halts a strategy when ≥ 5
consecutive losses on ≥ 20-trade base. Recovery via
`scripts/clear_edge_decay.py --strategy <name>`.

### Routing gates (5 gates)

| # | Gate | What it checks | Block reason |
|---|---|---|---|
| 1 | A4 `data_freshness` | Bar age ≤ 300s intraday / 172800s daily | `bar_stale` |
| 2 | E2 `stale_intent` | `utcnow − created_at ≤ ttl_seconds` (300s) | `intent_expired` |
| 3 | E5 `too_late_to_chase` | \|current − intent_price\|/intent_price ≤ 0.5% | `price_moved` |
| 4 | R7 `net_ev` | `expected_pnl − commission − spread ≥ min_edge` | `net_ev_below_min_edge` |
| 5 | S5 `event_risk` | Inside calendar window | reject (high urgency) / reduce 50% (normal) |

---

## 6. RECONCILIATION

### Publisher path

`chad/ops/reconciliation_publisher.py:run_publish`. Runs every cycle
of `chad-reconciliation-publisher.timer`. Reads
`runtime/position_guard.json` + IBKR positions and produces
`runtime/reconciliation_state.json`.

### Paper-mode behavior (commit `f6caf3d`)

`chad/ops/reconciliation_publisher.py:91-92` and `:125-127`:

```python
# In paper mode: only reconcile broker_sync (real IBKR positions)
if is_paper and strategy != "broker_sync":
    continue
```

In paper mode, only `broker_sync|*` entries are compared to IBKR
truth. Strategy entries (`alpha|SPY`, `delta|SPY`, etc.) are tracked
in the paper ledger and excluded from broker reconciliation. This
prevents the prior failure mode where every strategy-tracked paper
position appeared as a "missing" broker leg and tipped the reconciler
into RED.

### Status thresholds

`chad/ops/reconciliation_publisher.py:225-233`:

- `worst_diff ≤ 1.0` → **GREEN**
- `worst_diff ≤ 2.0` → **YELLOW**
- otherwise → **RED**

### Excluded symbols

| Set | Symbols | Why |
|---|---|---|
| `excluded_symbols` | `AAPL, MSFT` | Pre-existing paper-account positions before CHAD's tracking horizon. |
| `futures_excluded_symbols` | from `KNOWN_FUTURES_SYMBOLS` (`reconciliation_publisher.py:58`) | Futures contract-resolution path is incomplete; gaps are reality, not bugs. |

`KNOWN_FUTURES_SYMBOLS = frozenset({"MCL", "ES", "NQ", "CL", "GC",
"RTY", "MES", "MNQ"})`.

### Current state

`runtime/reconciliation_state.json` (ts `2026-04-26T12:42:02Z`):

```
status               GREEN
broker_source        ibkr:clientId=83
chad_open            1
broker_positions     1
worst_diff           0.0
mismatches           []
drifts               []
excluded_symbols     []
futures_excluded     []
```

---

## 7. INTELLIGENCE LAYER

Each runtime intel feed is independent — failures degrade gracefully
(stale data is preferred over no data; consumers test
`max_age_hours`).

| File | Schema | Purpose | Fresh? |
|---|---|---|---|
| `runtime/regime_state.json` | `regime_state.v1` | Live classifier inputs + regime label. | 60s TTL. |
| `runtime/strategy_intelligence.json` | n/a | AI-generated regime profile per strategy. | 48h cache (`daily_chad_report.py:1353`). |
| `runtime/expectancy_state.json` | n/a | Per-strategy rolling expectancy. | Refreshed by `chad-expectancy-tracker.timer` every 5 min. |
| `runtime/trends_state.json` | n/a | Google Trends ratios per symbol. | `chad-trends-refresh.timer`. |
| `runtime/reddit_sentiment.json` | n/a | Reddit mention/sentiment per symbol. | `chad-reddit-sentiment-refresh.timer`. |
| `runtime/short_interest.json` | n/a | Short float % per symbol. | `chad-short-interest-refresh.timer`. |
| `runtime/event_risk.json` | `event_risk.v1` | US session-edge windows. | 1800s TTL. |
| `runtime/institutional_consensus.json` | `institutional_consensus.v1` | Top 25 holdings across 7 funds. | Sunday 00:00 UTC weekly. |
| `runtime/profit_routing.json` | `profit_routing.v1` | 50/30/20 splits ledger. | Per-realised-close. |
| `runtime/intel_cache/*` | various | Brain-returns + macro_state + sector_rotation. | Various. |

### Strategy intelligence current state

`runtime/strategy_intelligence.json` — every entry is `profile=normal`
because the macro context provider returned `unknown` for VIX and
macro-risk inputs at snapshot time. The reasoning string (e.g.
`"Insufficient data due to macro context provider error; VIX and
macro risk label unknown; defaulting to normal regime per rules."`)
makes the gap explicit.

This is a known classifier-input wiring gap, not a fault — see Phase
roadmap §14.

---

## 8. TELEGRAM OPERATOR INTERFACE

### Free-text routing

`chad/utils/telegram_bot.py:1365` — `handle_free_text`. Wired in
`:1471`:

```python
dp.add_handler(MessageHandler(Filters.text & ~Filters.command,
                              handle_free_text))
```

A natural-language router classifies the message intent and dispatches
to the appropriate slash-command handler (`why_blocked`, `readiness`,
`risk`, etc.). Failures are logged and a friendly fallback is sent.

### Slash commands (registered handlers)

From `chad/utils/telegram_bot.py:1455-1468`:

- `/ping`
- `/help`
- `/coach_mode`
- `/status` (also default fallback)
- `/readiness`
- `/why_blocked`
- `/risk`
- `/perf`
- `/live_gate`
- `/shadow`
- `/portfolio_active`
- `/portfolio_targets <profile>`
- `/portfolio_rebalance <profile>`
- `/price <symbol>`
- (plus advisory + AI research commands `:1248`, `:1263`)

### Morning brief — elite prodigy voice

`chad/ops/daily_chad_report.py:MorningBrief` (`:1196`).

System prompt (`:1247-1257`):

> "You are CHAD — an elite autonomous trading system. You think with
> the quantitative precision of Jim Simons, the macro vision of Ray
> Dalio, and the opportunistic instincts of Stanley Druckenmiller.
> You are speaking to a non-technical operator who does not need
> jargon — use plain English only. Analyze the pre-market conditions
> provided and give a sharp, specific 2-3 sentence assessment.
> Reference actual data. Be confident and direct. No generic
> statements like 'stay disciplined' or 'markets can be volatile'.
> Wrap your response in a JSON object with a single key 'text'."

Schedule: `chad-morning-brief.timer` — Mon-Fri 9:00 AM ET (13:00 UTC).

Inputs to the take include the live regime, intelligence feeds (max
age 6h via `_load_fresh_feed`), price snapshots, Trends + Reddit
signals, and recent SCR stats.

### End-of-day brief

`chad-daily-report.timer` — Mon-Fri 4:35 PM ET (21:35 UTC).
`chad/ops/daily_chad_report.py:run_morning_brief` and the daily report
generator (`run_daily_report`) share the same elite-prodigy voice.

### Real-time alerts

`chad/utils/telegram_notify.py` — read-only side effects, deterministic
retries, dedupe via runtime files. Triggered for:

- Trade fills (live_loop best-effort send right after evidence write).
- SCR milestone transitions (WARMUP → CAUTIOUS → CONFIDENT).
- Stop bus activation.
- Edge-decay halts.
- IBKR / data-feed staleness watchdog.

### Weekly summary

`chad/ops/daily_chad_report.py:WeeklySummary` (`:1570`),
`run_weekly_summary` (`:1763`). Schedule:
`chad-weekly-report.timer` — Sundays 20:00 UTC. Includes the live
beta_allocation total from `runtime/profit_routing.json`.

### Voice lock

The CHAD voice is enforced by the system prompts in
`MorningBrief._chads_take` (elite prodigy) and the dashboard chat
(strict no-jargon plain English, see §9). If the LLM call fails the
brief degrades silently to data-only — no fallback paragraph that
could break voice.

---

## 9. DASHBOARD (chadtrades.com)

### Auth

- Password-protected via `/etc/chad/dashboard.env` (basic auth +
  session token in cookie).
- Public health endpoint `/health` (no auth) for systemd / monitoring.

### Routing

- TLS via Certbot (cert valid through 2026-07-19).
- nginx → `127.0.0.1:8765` (FastAPI/Uvicorn).

### Panels (current)

From `chad/dashboard/api.py:DashboardSnapshot.build`:

- **Training Mode card** — driven by `_chad_status` (`:327`).
- **Account Value** — `_portfolio` (`:275`).
- **Realized PnL** — `runtime/pnl_state.json`.
- **Market** — `_intelligence` (`:455`): VIX, BTC, SPY trend, top
  Reddit mention, market regime.
- **What CHAD Is Watching** — strategy_intelligence + intel feeds.
- **Open Positions** — `_open_positions` (`:301`).
- **Recent Trades** — `_iter_recent_closed_trades` (`:625`).
- **Strategy Performance** — `_strategies` (`:356`).
- **Ask CHAD chat** — `/api/chat` (commit `2935eea`).

### Chat (`/api/chat`, commit `2935eea` + `064efc7`)

- Endpoint `chad/dashboard/api.py:1121`.
- Model: `claude-sonnet-4-6` (`:948`).
- Context snapshot: `_chat_context_snapshot` (`:977`) — ~400 tokens of
  SCR, regime, portfolio, top 5 active strategies, stop bus, VIX.
- History cap: `CHAT_HISTORY_MAX` (sanitised in `:1056`).
- Plain-English voice (system prompt `:910-948`): no percentages, no
  trading terms (Sharpe, regime, drawdown, etc.), no jargon, max 3
  sentences unless asked, "warm, direct, confident — like a smart
  friend, not a banker". Translations baked into the prompt
  (`sizing_factor` → "how aggressively we trade",
  `effective_trades` → "practice runs").

### Dashboard regime card (commit `56eaeea`)

The regime panel reads `runtime/regime_state.json` directly
(`api.py:798-836`) — was previously inferring from `event_risk.json`
which lagged the classifier.

### Service health (commit `56eaeea`)

`api.py:_system_health` correctly counts oneshot service success.
Pre-fix, oneshot units that exited 0 between dashboard polls were
classified as failed. The new logic distinguishes oneshot completion
from running-service failure.

---

## 10. SERVICES & TIMERS

47 `chad-*` units (services + timers) loaded and active. 0 failed.

### Hot-path services (always running)

| Unit | Purpose |
|---|---|
| `chad-live-loop.service` | Hot-path: rebuild guard → signals → gates → execution. |
| `chad-orchestrator.service` | Risk-budget publisher (writes `runtime/dynamic_caps.json`). |
| `chad-ibgateway.service` | IBC-managed IB Gateway paper port 4002. |
| `chad-ibkr-bar-provider.service` | Polls IBKR `reqHistoricalData` every 30s for 1m + 1d bars. |
| `chad-kraken-ws.service` | Kraken WebSocket crypto feed. |
| `chad-shadow-status.service` | HTTP endpoint `:9618/shadow` (SCR sizing source). |
| `chad-metrics.service` | Prometheus-style metrics on `:9620/metrics`. |
| `chad-backend.service` | FastAPI backend for dashboard/reports. |
| `chad-dashboard.service` | Public dashboard on `127.0.0.1:8765`. |
| `chad-telegram-bot.service` | Operator alerts + briefs + free-text routing. |
| `chad-x11vnc.service`, `chad-xvfb.service` | Virtual display for IB Gateway. |
| `chad-ibkr-price-refresh.service` | Snapshots → `runtime/price_cache.json`. |

### Timers — hot-path

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-trade-closer.timer` | OnBootSec=45, OnUnitActiveSec=60 | Scheduled exits (stops/targets/time stops). |
| `chad-scr-sync.timer` | every 60s | Refresh `runtime/scr_state.json` from shadow. |
| `chad-reconciliation-publisher.timer` | recurring | Publishes reconciliation snapshots. |
| `chad-orchestrator` (Redis pub/sub) | continuous | Distributes dynamic caps. |
| `chad-paper-trade-exec.timer` | every 10m | Backstop paper executor. |
| `chad-paper-trade-executor.timer` | recurring | Alternate paper-trade pipeline. |
| `chad-ibkr-paper-fill-harvester.timer` | every 5m | Harvests broker fills into evidence ledger. |
| `chad-ibkr-broker-events.timer` | every 5m | Collects broker events. |
| `chad-ibkr-price-refresh.timer` | 60s market hours / 300s off-hours | Price cache refresh cadence. |
| `chad-options-monitor.timer` | every 60s during market hours | Monitors options positions. |
| `chad-options-chain-refresh.timer` | Mon-Fri 12:30 UTC | Refresh options chain cache. |

### Timers — analytics & feeds

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-expectancy-tracker.timer` | every 5m | Per-strategy expectancy. |
| `chad-symbol-blocker.timer` | every 5m | Symbol perf-based blocking. |
| `chad-strategy-intelligence-refresh.timer` | recurring | AI per-strategy regime profile. |
| `chad-trends-refresh.timer` | recurring | Google Trends signal refresh. |
| `chad-reddit-sentiment-refresh.timer` | recurring | Reddit sentiment scrape. |
| `chad-short-interest-refresh.timer` | recurring | Short interest data. |
| `chad-burnin-check.timer` | every 10m | Burn-in health check. |
| `chad-burnin-daily-summary.timer` | 23:59 UTC daily | Burn-in summary. |
| `chad-disk-guard.timer` | every 30m | Disk usage guard. |
| `chad-portfolio-artifacts.timer` | every 60m | Portfolio artifact publisher. |
| `chad-rebalance-auto-executor-paper.timer` | every 10m | Phase 12 rebalance preview (paper). |
| `chad-operator-intent-refresh.timer` | every 10m | Operator intent file refresh. |
| `chad-action-applier.timer` | every 60s | SSOT v5 ActionApplier. |
| `chad-execq-publisher.timer` | every 60s | execution_quality + latency_state publisher. |
| `chad-mutation-state-publisher.timer` | every 60s | action_state + change_canary_state. |
| `chad-lifecycle-truth-publisher.timer` | every 60s | trade_lifecycle_state + positions_truth. |
| `chad-profit-lock-publisher.timer` | every 60s | profit_lock + pnl_state publisher. |
| `chad-calendar-state-publisher.timer` | every 5m | calendar_state.json publisher. |

### Timers — reporting

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-morning-brief.timer` | Mon-Fri 13:00 UTC (9:00 ET) | Elite-prodigy pre-market brief. |
| `chad-daily-report.timer` | Mon-Fri 21:35 UTC (4:35 ET) | End-of-day report. |
| `chad-weekly-report.timer` | Sunday 20:00 UTC | Weekly summary including beta_allocation. |
| `chad-advisory-pre-market.timer` | recurring | Pre-market advisory. |
| `chad-ibkr-daily-bars-refresh.timer` | nightly | Daily bars refresh (futures). |
| `chad-proofs-cleanup.timer` | daily | Proof artifacts cleanup. |

### Intentionally disabled / retired

- `chad-reconciliation.timer` — masked (dual-writer risk).
- `chad-options-chain-refresh.service` (the standalone service form) —
  intentionally disabled; the timer `chad-options-chain-refresh.timer`
  is the canonical entry.
- `chad-polygon-stocks`, `chad-bars-validate`, `chad-daily-bars-backfill`
  — masked.

---

## 11. DATA & STORAGE

### Universe

`config/universe.json`:
- **Equities/ETFs (17):** AAPL, SPY, MSFT, GOOGL, BAC, IEMG, QQQ, VWO,
  NVDA, GLD, SH, PSQ, SVXY, UVXY, VIXY, **IWM**, **TLT**.
  IWM and TLT added in commit `6af971d` (unlocking `delta_pairs`'s
  IWM-leg pairs and `gamma_reversion`'s TLT slot).
- **Futures (8):** MES (CME), MNQ (CME), MCL (NYMEX), MGC (COMEX),
  ZN (CBOT), ZB (CBOT), M6E (CME), SIL (COMEX).

### Bar provider

- Service: `chad-ibkr-bar-provider.service`.
- Mechanism: `reqHistoricalData` polling every 30s — no real-time
  subscription required (commit `51c190e`).
- 1m bars: 10 symbols stored in `data/bars/1m/`.
- 1d bars: 30 symbols stored in `data/bars/1d/`.
- Daily bar refresh: `chad-ibkr-daily-bars-refresh.timer` runs
  nightly. The expired MCLK6 contract was removed from the daily
  refresher in commit `56eaeea`.

### Crypto data

- Kraken 1d bars via REST: `BTC-USD, ETH-USD, SOL-USD`.
- Kraken WS feed: real-time prices + balances
  (`runtime/kraken_prices.json`, `runtime/kraken_balances.json`).
- Altname routing in `chad/execution/execution_pipeline.py:1228-1264`.

### Price cache

`runtime/price_cache.json` — 23 symbols, 60s TTL, refreshed by
`chad-ibkr-price-refresh.timer` (60s in market hours, 300s off-hours).
The normalizer (`paper_exec_evidence_writer.py:925`) reads from this
cache to backfill paper fill prices.

### Fills ledger

- Equity / futures / options: `data/fills/FILLS_YYYYMMDD.ndjson` (one
  line per fill, hash-chained by writer).
- Crypto: `data/fills/kraken_fills_YYYYMMDD.ndjson`.

Recent days present: `FILLS_20260423.ndjson` … `FILLS_20260426.ndjson`,
plus `kraken_fills_20260422.ndjson`.

### SEC 13F refresh

Weekly cron Sunday 00:00 UTC via `scripts/update_institutional_consensus.py`.
Output: `runtime/institutional_consensus.json`. Citadel and Appaloosa
CIKs were corrected in commit `3e9db6d`; CUSIP-based dedup added in
commit `c021ccd`.

### Other key runtime files

- `runtime/dynamic_caps.json` — orchestrator-published per-strategy
  dollar caps.
- `runtime/profit_lock_state.json` — circuit breaker state.
- `runtime/stop_bus.json` — halt flag.
- `runtime/strategy_health.json` — F3 composite per strategy.
- `runtime/expectancy_state.json` — F1 rolling expectancy.
- `runtime/options_chains_cache.json` — cached options chains.
- `runtime/last_route_decision.json` — DecisionTrace bridge.
- `runtime/live_readiness.json` — `ready_for_live=false`.

---

## 12. CHANGE LOG (delta from v8.1)

In commit order (oldest first since v8.1 was cut). Items 1-5 predate
v8.1's commit table and are listed for completeness; items 6-15 are
the canonical post-v8.1 delta.

### Pre-v8.1 (already in v8.1 §13)

1. **`20d6f59`** — *Rename: chad/strategies/beta.py → beta_trend.py*
   Pure rename to free the `beta.py` filename for the new
   institutional compounder.
2. **`2c6a16b`** — *Build: Beta — institutional long-term compounder*
   New `chad/strategies/beta.py` reading `runtime/institutional_consensus.json`,
   pulled by `scripts/update_institutional_consensus.py`. Weight
   carved from `beta_trend` (action `beta_carveout_from_beta_trend_20260423`).
3. **`3e9db6d`** — *Fix: update Citadel and Appaloosa CIKs in SEC 13F
   fetcher.* Two stale CIK numbers were causing `chad/analytics/institutional_consensus.py`
   to silently drop those funds.
4. **`c021ccd`** — *Fix: dedup institutional_consensus entries by
   resolved symbol.* Multiple CUSIP rows resolving to the same ticker
   collapsed into one weighted entry.
5. **`51c190e`** — *Fix: switch bar provider from reqRealTimeBars to
   polling.* Drops the IBKR real-time-data subscription requirement.

### Post-v8.1 (new in v8.2)

6. **`8cd644a`** — *Fix: fill recording — asset_class resolver +
   PendingSubmit timeout.*
   - Was broken: paper FILLS sometimes carried `asset_class=""` or
     `unknown`; `PendingSubmit` status could persist for a full cycle
     and then be hash-chained as a "fill".
   - Was fixed: wired `chad/execution/ibkr_adapter.resolve_asset_class`
     into the writer and added a PendingSubmit timeout.
   - Citation: `chad/execution/ibkr_adapter.py` (resolver) +
     `chad/execution/paper_exec_evidence_writer.py:104` (used in
     `chad/core/live_loop.py`).

7. **`e150c0b`** — *Upgrade: complete Telegram intelligence layer.*
   - Was broken: the Telegram bot was alert-only; the morning brief
     was templated text without an LLM voice; free-text messages from
     the operator returned `unknown command`.
   - Was fixed: NL routing in `handle_free_text`
     (`chad/utils/telegram_bot.py:1365`); elite-prodigy voice in
     `MorningBrief._chads_take`
     (`chad/ops/daily_chad_report.py:1247-1257`); rich notifier
     module (`chad/utils/telegram_notify.py`, +173 LoC).

8. **`81dafce`** — *Build: alpha_crypto real momentum signals.*
   - Was broken: `alpha_crypto` was a placeholder that returned `[]`
     in every regime — the strategy was wired but emitted no signals.
   - Was fixed: 3-filter momentum pipeline (SMA20 breakout + 3-day
     return ≥ 1.5%; vol expansion 5d/20d ≥ 0.7; regime multiplier).
   - Citation: `chad/strategies/alpha_crypto.py:627-758` (+453 / -363
     LoC).

9. **`2935eea`** — *Feature: Claude chat panel on chadtrades.com.*
   - Was missing: dashboard had no operator-facing chat.
   - Was added: `/api/chat` endpoint backed by `claude-sonnet-4-6`
     with a 400-token state snapshot and a sanitised history
     (`chad/dashboard/api.py:901-1146`, `chad/dashboard/static/index.html`).

10. **`064efc7`** — *Tighten chat voice: plain English only.*
    - Was wrong: chat replies still leaked terms like "Sharpe",
      "drawdown", "PnL".
    - Was fixed: hardened `CHAT_SYSTEM_PROMPT`
      (`chad/dashboard/api.py:910-948`) — strict no-jargon, no-percentages
      rules, plain-English translation map, max 3 sentences.

11. **`d666b5d`** — *Fix: execution pipeline — futures cancellation +
    options routing.*
    - Was broken: bucket key `(symbol, side)` caused opposing-side
      futures signals from `alpha_futures` and `gamma_futures` on the
      same symbol to net to zero and silently drop. SPY/ETF and
      SPY/OPTIONS merged into a single bucket; the ETF won and the
      option leg was lost.
    - Was fixed: bucket key changed to `(symbol, asset_class)` in
      both `chad/utils/signal_router.py:102` and
      `chad/execution/execution_pipeline.py:680-685`. OPTIONS branch
      properly resolves via `_resolve_options_spec` at
      `:664-665`.

12. **`56eaeea`** — *Fix: allocator sleeves, dashboard regime, service
    health, position_guard opened_at, MCL contract.*
    - **Allocator sleeves:** `chad/risk/dynamic_risk_allocator.py:392-485`
      now correctly assigns `omega_momentum_options` to ALPHA and
      `delta_pairs` + `alpha_crypto` to ADAPTIVE.
    - **Dashboard regime:** `chad/dashboard/api.py:798-836` reads
      `runtime/regime_state.json` directly (was inferring from
      `event_risk.json`).
    - **Service health:** `chad/dashboard/api.py:_system_health` now
      counts oneshot success.
    - **`position_guard.opened_at`:** every `mark_position_open` write
      stamps a UTC `opened_at`
      (`chad/core/position_guard.py:155-157`).
    - **MCL contract:** removed expired `MCLK6` from the daily-bars
      refresher.

13. **`f6caf3d`** — *Fix: meta propagation + reconciliation GREEN in
    paper mode.*
    - Was broken: each `TradeSignal.meta` was lost when the router
      merged buckets. Reconciliation in paper mode flipped to RED
      because every strategy paper position appeared as a missing
      broker leg.
    - Was fixed: meta carried per-strategy in the bucket
      (`chad/utils/signal_router.py:109`), primary strategy's meta
      emitted into the `RoutedSignal` (`:179`).
      `chad/ops/reconciliation_publisher.py:91-92` now skips
      strategy entries in paper mode.

14. **`21166c9`** — *Fix: 100% PaperExecEvidence parity — futures
    fills.*
    - Was broken: futures fills sometimes wrote
      `status="PendingSubmit"` and `fill_price=0` to FILLS, forcing
      SCR to mark them `excluded_untrusted`. The four-callsite
      writer pattern guaranteed at least one path bypassed any
      single fix.
    - Was fixed: single chokepoint `normalize_paper_fill_evidence` at
      `chad/execution/paper_exec_evidence_writer.py:925-983`. All
      three production writers (live_loop, position_reconciler, the
      paper_trade_executor binary) call it directly; the public
      `write_paper_exec_evidence` calls it again as a safety net
      (`:998`).

15. **`6af971d`** — *Fix: pre-Monday wiring — broker truth + universe
    expansion.*
    - **Paper-mode broker rebuild:** `chad/core/live_loop.py:641-651`
      now rebuilds the guard from BOTH the local paper ledger AND
      IBKR positions in paper mode (was: local-only).
    - **Universe expansion:** `config/universe.json:18-19` adds IWM
      and TLT, unlocking `delta_pairs`'s IWM-leg pairs and
      `gamma_reversion`'s TLT slot.
    - **Futures separation:** `alpha_futures` and `gamma_futures` no
      longer share symbols (alpha = MES/MNQ/MGC; gamma = MCL +
      MYM/M2K/ZN/ZB), eliminating the zero-net-size cancellations.

---

## 13. KNOWN ISSUES

### DEGRADED

| ID | Severity | Summary |
|---|---|---|
| **`alpha_options` stuck on SPY** | DEGRADED | `alpha_options\|SPY` is `open=true` from 2026-04-24T20:18Z. Position MAINTAINED → no new spreads emit until the existing one closes. By design (no flip on options spreads) but blocks signal generation — investigate whether a time-based exit should be added. |
| **`omega_vol` health = 0.10** | DEGRADED (low confidence) | 3 samples, 0 wins, slippage_ratio 0.5, regime_alignment 0.0. Sample size is too small to act on, but watch list for the next 10 trades — if no recovery, F4 edge-decay should halt at the 5-loss threshold. |
| **strategy_intelligence shows all profile=normal** | DEGRADED (input-feed gap) | `runtime/strategy_intelligence.json` reasoning string says "Insufficient data due to macro context provider error; VIX and macro risk label unknown". Macro context wiring gap, not classifier fault. |

### TEST FAILURES (pre-existing)

| Test | Cause | Severity |
|---|---|---|
| 3 × `position_guard rebuild` | `clientId=99` collision with another test client. | Cosmetic — production uses `clientId=83`. |
| 1 × `regime_classifier` | Config drift (calibration constants changed). | Cosmetic — classifier behavior is verified via integration. |

### COSMETIC

- **Weight key mismatch.** `config/strategy_weights.json` uses key
  `"crypto"` while every other surface (`runtime/dynamic_caps.json`,
  the registry) uses `alpha_crypto`. Normalised via shim
  `chad/risk/dynamic_caps.py:67`. Could be renamed for clarity but
  the shim is harmless.
- **`alpha_futures` docstring lists `MCL`.** The strategy file's
  module-level docstring still names the legacy 4-symbol universe;
  the live tuple is `(MES, MNQ, MGC)` per `:98`. Documentation
  drift — code is authoritative.
- **`alpha_futures_config.py:55-60` defaults to 4 symbols.** Override
  is the hardcoded `ALPHA_FUTURES_UNIVERSE` tuple; legacy default is
  effectively dead code.
- **`strategy_health.json` covers 6 of 16 strategies.** F3 only scores
  strategies that have crossed its sample threshold (`sample_count`).
  Quiet strategies are simply absent — not a fault.
- **`_chassis_enabled()` docstring** mentions `crypto` in the ADAPTIVE
  list (`dynamic_risk_allocator.py:417`); the frozenset uses
  `alpha_crypto`. Both refer to the same strategy after the shim.

### MONDAY VERIFICATION REQUIRED (next session)

- Per-strategy fill counts during a full active US session.
- `alpha_crypto`'s first Kraken intent under the new momentum logic
  (will only fire when 5d/20d vol-ratio ≥ 0.7 — vol compression
  may suppress for several days).
- `effective_trades` climb rate toward the CAUTIOUS threshold (need
  21 more, current pace ≈ 1–3/day).
- Whether `alpha_options\|SPY` releases on the next options-cycle
  exit, freeing the strategy to emit new spreads.
- That `omega_vol` doesn't accumulate a 4th and 5th consecutive loss
  this week (would trigger F4 edge-decay halt).

### TRACKED ISSUES (from v8.1 §14, still open)

| ID | Sev | Status | Summary |
|---|---|---|---|
| ISSUE-22 | P2 | OPEN | Legacy placeholder audit item. |
| ISSUE-29 | P1 | PARTIAL | `apply_close_intents` mutates guard before broker confirms — emits spurious close intents each cycle. Reconciler-side fix landed (`b672042`); root cause untreated. |
| ISSUE-50 | P1 | OPEN | `chad-options-chain-refresh` hangs when IBKR ushmds farm is down. Timeout wrapper pending. |
| ISSUE-54 | P2 | OPEN | `runtime/pnl_state.json` still tracked in git (currently dirty in working tree). |
| ISSUE-58 | P2 | OPEN | `chad-trade-closer.timer` uses OnBootSec/OnUnitActiveSec; needs OnCalendar or documented seed step. |
| ISSUE-75 | P1 | OPEN | Multiple call sites write `position_guard.json` directly; unify via a single setter. |
| ISSUE-78 | P2 | OPEN | Two code paths read `CHAD_EXECUTION_MODE`. |

---

## 14. PHASE ROADMAP

### IMMEDIATE (next session)

- Monday session audit (see §13 verification list).
- Fix the 4 pre-existing test failures.
- `alpha_options` expiry/strike metadata for actual fills (currently
  uses chain cache; metadata gap when cache is stale or missing).
- Decide on optional time-based exit for `alpha_options` to release
  the SPY MAINTAINED block.

### PHASE 11 — multi-market expansion

- Full options chain integration (per-symbol chains, not just SPY/QQQ
  cache).
- `alpha_statarb` — stat-arb basket engine.
- `alpha_crypto_alt` — altcoin momentum (DOT, LINK, etc. when Kraken
  pairs are available and minimum-volume gates pass).

### PHASE 12 — policy automation

- Capital `TierManager` (`config/tiers.json` is already present and
  unused — wire to risk allocator).
- `RegimeBooster` — bounded multiplier on per-strategy size when
  regime alignment is confirmed.
- `WinnerScaling` — top-quartile-strategy size boost from amplifier
  bucket.
- `WithdrawalManager` — high-water mark logic for the salary account.

### PHASE 9 — pre-live calibration (carried over from v8.1)

- Regime classifier tuning (ADX proxy → Wilder ADX or threshold
  re-calibration).
- Kelly fraction tuning (`CHAD_ALLOC_V3_KELLY_MAX`).
- Slippage model fit per asset class.
- Live feature distribution drift monitoring.
- Net-EV gate opt-in (populate `expected_pnl` at strategy level).
- Halt-on-reconciliation-mismatch.
- ISSUE-29, 50, 75, 78, 54, 58.

### PHASE 10 — live capital flip (carried over)

Entry criteria:
- All Phase-9 items complete.
- SCR = CONFIDENT.
- 60-90 days consistent paper performance.
- Explicit operator GO via governance rule #3 (`CLAUDE.md`).

When live:
- `CHAD_EXECUTION_MODE` flips from `paper` to `live`.
- `LiveGate` accepts the posture change.
- First 3 cycles run with manual oversight.
- Profit routing flips from advisory to actual capital movement.

---

## 15. APPENDICES

### Appendix A — File inventory of changed paths since v8.1

From `git diff 93bbd70..HEAD --stat`:

| File | Δ |
|---|---|
| `chad/core/live_loop.py` | +147 |
| `chad/core/position_guard.py` | +12 |
| `chad/core/position_reconciler.py` | +13 |
| `chad/dashboard/api.py` | +288 |
| `chad/dashboard/static/index.html` | +246 |
| `chad/execution/execution_pipeline.py` | +88 |
| `chad/execution/ibkr_adapter.py` | +122 |
| `chad/execution/paper_exec_evidence_writer.py` | +176 |
| `chad/market_data/ibkr_historical_provider.py` | +4 |
| `chad/ops/daily_chad_report.py` | +527 |
| `chad/ops/reconciliation_publisher.py` | +21 |
| `chad/risk/dynamic_risk_allocator.py` | +6 |
| `chad/strategies/alpha_crypto.py` | +453 / -363 |
| `chad/strategies/alpha_futures.py` | +43 |
| `chad/strategies/gamma_futures.py` | +74 |
| `chad/tests/test_normalize_paper_fill_evidence.py` | +177 (new) |
| `chad/utils/signal_router.py` | +8 |
| `chad/utils/telegram_bot.py` | +40 |
| `chad/utils/telegram_notify.py` | +173 (new module) |
| `config/universe.json` | +6 |
| `ops/scr_state_sync.py` | +16 |
| **Total** | **+2188 / -452 across 21 files** |

### Appendix B — Environment files (names only — never values)

These files live outside the repo, are gitignored, and contain
secrets. Listed by name only.

- `/etc/chad/dashboard.env` — dashboard basic-auth credentials.
- `/etc/chad/claude.env` — Anthropic API key for chat + morning brief.
- `/etc/chad/ibkr.env` — IBKR Gateway credentials.
- `/etc/chad/kraken.env` — Kraken REST + WS keys.
- `/etc/chad/openai.env` — fallback / non-Claude model creds (legacy).
- `/etc/chad/polygon.env` — Polygon API key (currently unused; bar
  provider has switched to IBKR polling).
- `/etc/chad/telegram.env` — bot token + allowed chat id.
- `/etc/chad/chad.env` — additional envs loaded by `chad-live-loop`.

### Appendix C — Critical paths cheat sheet

| Item | Path |
|---|---|
| Repo root | `/home/ubuntu/chad_finale` |
| Virtualenv | `/home/ubuntu/chad_finale/venv` (always `python3`) |
| Runtime state | `/home/ubuntu/chad_finale/runtime` |
| Daily bars | `/home/ubuntu/chad_finale/data/bars/1d` |
| Minute bars | `/home/ubuntu/chad_finale/data/bars/1m` |
| Fills | `/home/ubuntu/chad_finale/data/fills` |
| Trade history | `/home/ubuntu/chad_finale/data/trades` |
| Slippage ledgers | `/home/ubuntu/chad_finale/data/slippage` |
| Reports | `/home/ubuntu/chad_finale/reports` |
| Revert tarballs | `/home/ubuntu/chad_revert_points/` |
| Systemd units | `/etc/systemd/system/chad-*` |
| Env files | `/etc/chad/*.env` |
| Hot-path entry | `chad/core/orchestrator.py`, `chad/core/live_loop.py` |
| Execution adapter | `chad/execution/ibkr_adapter.py` |
| Paper executor (timer) | `/usr/local/bin/chad_paper_trade_executor.py` |
| LiveGate | `chad/core/live_gate.py` |
| Risk allocator | `chad/risk/dynamic_risk_allocator.py` |
| Profit Lock | `chad/risk/profit_lock.py` |
| Evidence writer | `chad/execution/paper_exec_evidence_writer.py` |
| Full preview | `chad/core/full_cycle_preview.py` |

### Appendix D — Git log of commits since v8.1 (full SHAs)

From `git log 93bbd70..HEAD --pretty=format:"%H %ad %s" --date=short`:

```
6af971d2... 2026-04-26 Fix: pre-Monday wiring — broker truth + universe expansion
21166c97... 2026-04-25 Fix: 100% PaperExecEvidence parity — futures fills
f6caf3df... 2026-04-25 Fix: meta propagation + reconciliation GREEN in paper mode
56eaeea7... 2026-04-24 Fix: allocator sleeves, dashboard regime, service health, position_guard opened_at, MCL contract
d666b5d3... 2026-04-24 Fix: execution pipeline — futures cancellation + options routing
064efc70... 2026-04-24 Tighten chat voice: plain English only, no trading jargon
2935eea8... 2026-04-24 Feature: Claude chat panel on chadtrades.com dashboard
81dafce4... 2026-04-24 Build: alpha_crypto real momentum signals
e150c0b9... 2026-04-23 Upgrade: complete Telegram intelligence layer
8cd644a1... 2026-04-23 Fix: fill recording — asset_class resolver + PendingSubmit timeout
```

(Truncated to 7-character SHAs in the body for readability;
`git rev-parse <short-sha>` resolves the full hash.)

### Appendix E — Verification sequence (CLAUDE.md governance rule)

After every change:

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

### Appendix F — Active git tags

- `STABILITY_FREEZE_20260307_GREEN` — original stable baseline.
- `PRE_HARDENING_20260402` — snapshot before P0 hardening.
- `RATIFICATION_MASTER_20260402` — all hardening + GAP items complete.
- `REVERT_PRE_OVERHAUL_20260419` — snapshot before 2026-04-19/21
  overhaul (commit `45f3728`; runtime tarball at
  `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`).

### Appendix G — Rollback commands (governance-approved)

```bash
# Roll back to post-hardening stable
git checkout RATIFICATION_MASTER_20260402

# Roll back to pre-overhaul stable (restore runtime from tarball too)
git checkout REVERT_PRE_OVERHAUL_20260419
tar -xzf /home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz \
        -C /home/ubuntu/chad_finale
# Steps: /home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt
```

### Appendix H — Operator daily checks (≈ 2 minutes)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Core health
cat runtime/scr_state.json | python3 -m json.tool | head -15
cat runtime/profit_lock_state.json | python3 -m json.tool | head -15
cat runtime/stop_bus.json | python3 -m json.tool

# 2. Regime + activation
cat runtime/regime_state.json
cat runtime/market_metrics.json | python3 -m json.tool | head -15

# 3. Reconciliation
cat runtime/reconciliation_state.json | python3 -m json.tool

# 4. Strategy health (only those with samples)
python3 -c "import json; \
  [print(f'{k}: {v[\"health_score\"]:.3f} ({v[\"sample_count\"]})') \
   for k,v in json.load(open('runtime/strategy_health.json'))['strategies'].items()]"

# 5. Dashboard
curl -s https://chadtrades.com/health
```

What to look for (today's expected values):

- SCR `state` = `WARMUP` (expected until ~100 effective trades; 79
  now).
- Profit lock `mode` = `NORMAL`, `stop_new_entries` = `false`.
- Stop bus `active` = `false`.
- Reconciliation `status` = `GREEN`.
- No strategy with `health_score < 0.3` AND `sample_count ≥ 20`
  (`omega_vol` is below 0.3 but only 3 samples; not actionable yet).
- Dashboard returns HTTP 200.

---

**End of CHAD Unified SSOT v8.2.**

This document is the truth as of 2026-04-26. If the code disagrees,
either the code drifted or this revision needs another pass. Cut
v8.3 before relying on the disagreement.
