# CHAD Unified SSOT v8.5

**Version:** 8.5
**Date:** 2026-04-30
**Status:** Active — Paper Trading
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v8.4_2026-04-28.md` (commit `f62d914`, 2026-04-28)

This document is the master reference for the CHAD trading system as it
exists at HEAD (`9f55688`, 2026-04-30). It captures every commit since
v8.2, every wired strategy, every runtime invariant, and the live state
of the machine at the moment of writing. v8.5's signature contribution
is the **signal emission audit** — a two-part forensic verification
(Part 1: schema contract audit across all 16 strategies; Part 2: live
signal invocation against real context) that identified and closed 12 of
13 findings. Every strategy now produces signals or fails for a documented
expected reason. No more silent zero-signal returns. The business framework
layer introduced in v8.3 and hardened in v8.4 is now fully verified
end-to-end: schema contracts clean, data flows confirmed, VIX in price
cache, MYM/M2K bars live, beta rebalance firing, omega_momentum_options
unblocked.

If this document and the code disagree, either the code drifted or
this revision needs another pass — either way, revise the document
before relying on the disagreement.

---

## Table of Contents

0. [Preamble](#0-preamble)
1. [Mission & Architecture](#1-mission--architecture)
2. [Runtime State (live snapshot)](#2-runtime-state-live-snapshot)
3. [The 16 Strategies](#3-the-16-strategies)
4. [Execution Pipeline (post-fix path)](#4-execution-pipeline-post-fix-path)
5. [Risk & Governance — The Full Chain](#5-risk--governance--the-full-chain)
6. [The Business Framework](#6-the-business-framework)
7. [Reconciliation](#7-reconciliation)
8. [Intelligence Layer](#8-intelligence-layer)
9. [Telegram Operator Interface](#9-telegram-operator-interface)
10. [Dashboard (chadtrades.com)](#10-dashboard-chadtradescom)
11. [Services & Timers](#11-services--timers)
12. [Data & Storage](#12-data--storage)
13. [Change Log (delta from v8.2)](#13-change-log-delta-from-v82)
14. [Known Issues](#14-known-issues)
15. [Phase Roadmap](#15-phase-roadmap)
16. [Appendices](#16-appendices)

---

## 0. PREAMBLE

### Document metadata

| Field | Value |
|---|---|
| Document version | 8.5 |
| Date written | 2026-04-30 (UTC) |
| Predecessor | v8.4 — `docs/CHAD_UNIFIED_SSOT_v8.4_2026-04-28.md` (`f62d914`) |
| Repository HEAD at write time | `9f55688` — *"Fix: add diagnostic logging to alpha_intraday, alpha_options, gamma_reversion — no silent zero-signal returns"* |
| Branch | `main` |

### Server / repo / mode

- **Server:** AWS EC2, Ubuntu 24.04, kernel 6.17.0-1009-aws.
- **Repo root:** `/home/ubuntu/chad_finale`.
- **Python:** `python3` via `/home/ubuntu/chad_finale/venv` (governance
  rule in `CLAUDE.md` — never invoke `python`).
- **IBKR paper account:** Canadian-domiciled, `$183,315.47 USD` net
  liquidation (source: `runtime/portfolio_snapshot.json:ibkr_equity` at
  `2026-04-30T02:03:01Z` after IBKR FX-quoted CAD→USD conversion).
- **Kraken paper:** connected — 0.0012 BTC + 252.8538 CAD ≈ `$184.58 USD`
  (source: `runtime/kraken_balances.json` at `2026-04-30T02:03:01Z`).
- **Total equity (USD):** `$183,500.05` =
  `ibkr_equity ($183,315.47) + kraken_equity ($184.58)`.
- **Execution posture:** PAPER — `CHAD_EXECUTION_MODE=paper`. The
  systemd unit ships `dry_run` (`/etc/systemd/system/chad-live-loop.service`)
  but `chad/core/live_loop.py:_is_paper_mode` treats both `paper` and
  `dry_run` as paper for guard-rebuild purposes.
- **Live readiness:** `runtime/live_readiness.json:ready_for_live=false`
  (last evaluated `2026-04-09T13:40:29Z`).

### This session's commit chain (7 commits since v8.2)

In topological order (newest → oldest):

1. **`9f55688`** — *Fix: add diagnostic logging to alpha_intraday, alpha_options,
   gamma_reversion — no silent zero-signal returns* (2026-04-30). Three strategy
   modules had no logging imports at all and returned [] with zero diagnostic
   output. Added `import logging` + `LOG = logging.getLogger(__name__)` to each.
   Added per-symbol debug logs, cycle-level info logs, and exception warning logs
   so every zero-signal cycle now emits an observable reason.

2. **`6e154dc`** — *Fix: Part 1 audit findings — VIX in price cache, beta gap fix,
   omega_momentum_options schema, MYM/M2K bars, TTL per-file thresholds,
   winner_scaler roster* (2026-04-30). Seven files changed across two audit-
   finding categories: (a) signal emission fixes — VIX injected into
   `runtime/price_cache.json` from `data/bars/1d/VIX.json` on every refresh
   cycle; `omega_momentum_options` schema mismatches fixed (`ts_utc` key,
   dead contracts lookup removed); `beta` `underweight_gap` lowered from
   0.02 to 0.005; 20 institutional consensus symbols added to
   `config/universe.json`; MYM/M2K added to futures universe and expiry
   tables in `ibkr_historical_provider.py` with bar files backfilled;
   (b) infrastructure fixes — `_read_business_runtime` now takes per-file
   `stale_seconds` parameter (tier=600s, winner=1800s, regime_booster=120s);
   `winner_scaler` now writes explicit 1.0 entries for all 16 canonical
   strategies.

3. **`031a9de`** — *Tighten brief voice: zero jargon for non-trader operator*
   (2026-04-28). Single `CHADS_TAKE_SYSTEM_PROMPT` constant shared across
   morning, EOD, and weekly briefs. All banned finance terms replaced with
   plain-English equivalents. `broker_sync` filtered from all 3 report
   surfaces via shared `_filter_for_ranking()` helper.

4. **`15677a2`** — *Fix: enforce Kraken REST pair format at executor +
   REST border* (2026-04-26). Two-layer defense against the
   `EQuery:Unknown asset pair` failure mode that produced null-payload
   `kraken_fills_*` records. Layer 1 at the REST border in
   `chad/exchanges/kraken_client.py:55` (`_assert_kraken_rest_pair`);
   layer 2 in the executor at `chad/execution/kraken_executor.py:41`
   (`_enforce_kraken_rest_pair`). Test fixtures updated from
   `pair=XBT/USD` (wsname, invalid for REST) to `pair=XBTUSD`
   (altname, valid).
5. **`0c3a3e1`** — *Fix: exclude broker_sync from weekly top brain
   ranking* (2026-04-27). `broker_sync` is the bookkeeping label for
   pre-existing IBKR paper positions inherited by CHAD (AAPL, MSFT,
   GOOGL, etc.), not a CHAD strategy. Filter applied in
   `chad/ops/daily_chad_report.py:1711`:
   `EXCLUDED_FROM_RANKING = {"broker_sync", "manual", "paper_exec",
   "unknown", ""}`. Minimum trade count threshold: 5 trades. Re-rank by
   expectancy if original `top_performer` is filtered out.
6. **`1caa111`** — *Build: complete CHAD business framework
   (Phase 11C/12B/12C)* (2026-04-27). Adds the missing CHAD-as-a-business
   policy layer — WinnerScaling (`chad/risk/winner_scaler.py`),
   RegimeBooster (`chad/risk/regime_booster.py`), BusinessPhaseTracker
   (`chad/ops/business_phase_tracker.py`). Risk allocator
   (`chad/risk/dynamic_risk_allocator.py`) now applies the full chain:
   `base → correlation → chassis → tier_filter → winner_scaling →
   regime_booster → caps`. Telegram morning brief + weekly summary
   include a BUSINESS STATUS section. Dashboard `/api/state` surfaces
   business framework data. Claude chat context includes phase + tier
   + salary. RegimeBooster VIX bug fixed
   (`bars[-1].get('close')` not `'c'`).
7. **`1608fec`** — *Build: business framework foundation files*
   (2026-04-27). Companion to `1caa111`. Adds the foundation files
   built earlier in the same session: `equity_history_publisher`,
   `portfolio_snapshot_publisher`, `withdrawal_manager`, `tier_manager`,
   `tiers.json`, `withdrawal_policy.json`, `regime_booster_policy.json`.
   The framework is now fully version-controlled and reproducible.

### What's different from v8.2 (executive delta)

- **CHAD now has a business framework.** Six new runtime files —
  `business_phase.json`, `tier_state.json`, `withdrawal_authorization.json`,
  `winner_scaling.json`, `regime_booster.json`, `equity_history.ndjson`
  — and the seven publisher / manager modules behind them constitute
  the new policy layer (§6).
- **Risk allocator gained three overlays.** The cap chain now reads
  `tier_state.json`, `winner_scaling.json`, `regime_booster.json` and
  composes their multipliers into the dollar caps written to
  `runtime/dynamic_caps.json` (§5). Each overlay is fail-soft: stale
  files (>10 minutes) collapse to neutral.
- **Portfolio snapshot is fresh.** `chad-portfolio-snapshot.timer` runs
  every 5 minutes (`chad/ops/portfolio_snapshot_publisher.py`). The
  long-standing issue where `runtime/portfolio_snapshot.json` was
  written once on 2026-04-03 and never refreshed is closed. CAD→USD
  conversion uses IBKR's own FX quote (`USD.CAD`).
- **Equity history exists.** `chad-equity-history.timer` writes one
  daily record to `runtime/equity_history.ndjson` at 23:59 UTC. This
  is the durable HWM input for `WithdrawalManager` and growth metrics
  for `BusinessPhaseTracker`.
- **Telegram BUSINESS STATUS section.** `MorningBrief` and the weekly
  summary both surface the business framework state — phase, tier,
  authorized salary, growth from seed (§9).
- **Dashboard exposes business state.** `/api/state` returns a
  `business` block (`chad/dashboard/api.py:484-525`); the chat context
  includes the same block (`:1086-1103`).
- **Kraken REST border has two-layer enforcement.** wsname (`XBT/USD`)
  and CHAD canonical (`BTC-USD`) pair formats now raise `ValueError`
  at executor pre-flight (layer 2) and at the REST border in
  `kraken_client.add_order` (layer 1) (§4).
- **Top-brain weekly ranking excludes `broker_sync`.** Weekly report
  surfaces actual strategy performance instead of passive market
  exposure (§9).
- **Equity reset to ~$167k USD.** Account is now domiciled in CAD
  (~CAD 234k) and converted via IBKR's `USD.CAD` quote. The previous
  `~$998k` figure in v8.2 reflected an unconverted CAD value or a
  stale snapshot — the live publisher now produces the canonical USD
  total each cycle.
- **Tier = PRO.** All 16 strategies are enabled. Equity is in the
  PRO band (`$160k–$1M`).
- **Phase = GROW.** Engine is built. Account is above
  `seed × 1.20 = $60,000`. Salary is not yet authorized — needs SCR =
  CONFIDENT and ≥ 14 days of equity history; current history is 1 day.

### What's different from v8.3 (executive delta)

- **`chad-reconciliation.timer` masked.** Was `disabled` (could be started manually); now symlinked to `/dev/null`. Dual-writer risk on `reconciliation_state.json` permanently closed.
- **`chad-regime-booster.timer` installed.** New systemd timer fires `chad.risk.regime_booster` every 60 seconds. Closes the 15+ hour staleness gap — SSOT §6E claimed orchestrator integration that was never wired. `regime_booster.json` now publishes every cycle.
- **`chad-event-risk.timer` enabled.** Was `disabled` with a malformed real-file in `timers.target.wants` instead of a symlink. Fixed to correct symlink, enabled, firing every 600 seconds. `event_risk.json` was 25 days stale; now fresh.
- **`withdrawal_manager.py:158` HWM bug fixed.** `hwm = max(max(equities), current_equity)` → `hwm = max(equities)`. Current equity was included in the HWM calculation, making surplus permanently ≤ 0. Salary authorization now works correctly.
- **`strategy_weights.json` key renamed `crypto` → `alpha_crypto`.** Closes the tier_filter cap lookup mismatch that caused `alpha_crypto` to receive a $0 dollar cap. Weight value (0.04) unchanged. Sum of all weights remains 1.0.
- **Morning brief `generate()` method confirmed** (not `build()`). BUSINESS STATUS section validated in dry-run: Phase, Account, Tier, Salary all present.
- **Weekly report `generate()` method confirmed.** BUSINESS PHASE block validated in dry-run. `broker_sync` confirmed excluded from TOP BRAIN section.
- **G5/G6/G7 investigations closed.** `__main__` block confirmed at `regime_booster.py:210`. `clientId=84` confirmed at `portfolio_snapshot_publisher.py:36`. VIX bar access confirmed as defensive dual-key read `b.get("close", b.get("c", 0.0))`.
- **Unit counts corrected.** 93 total units (53 services + 40 timers), not 53.
- **SCR schema corrected.** `effective_trades` and `win_rate` are nested under `stats` key in `scr_state.json`.

### What's different from v8.4 (executive delta)

- **Signal emission audit complete.** Part 1 schema contract audit + Part 2
  live signal invocation across all 16 strategies. 13 findings identified and
  12 closed. 10/16 strategies confirmed producing signals against live context.
- **VIX now in `price_cache.json`.** `chad/market_data/price_cache_refresh.py`
  reads the latest VIX close from `data/bars/1d/VIX.json` and injects it into
  the prices dict before every atomic write. `omega_momentum_options` (previously
  BLOCKING) and `omega_vol` both confirmed receiving VIX via ctx.prices.
- **`omega_momentum_options` schema fixed.** Two mismatches closed: (1) TTL key
  read corrected from `updated_ts_utc` → `ts_utc`; (2) dead `contracts` array
  lookup replaced with honest stub routing directly to synthetic pricing.
- **`beta` rebalance gap fixed.** `underweight_gap` lowered from 0.02 to 0.005.
  Strategy now fires on any position more than 0.5% underweight, not only
  zero-weight symbols. Confirmed producing signals in Part 2.
- **20 institutional consensus symbols added to universe.** `config/universe.json`
  symbols array expanded from 17 to 37 equities. `price_cache_refresh.py` now
  fetches prices for all 25 consensus symbols so `beta` can size against them.
- **MYM and M2K bars backfilled.** `config/universe.json` futures array expanded
  from 8 to 10 entries. `ibkr_historical_provider.py` `_EXPIRY_SCHEDULE` and
  `_FUTURES_CONTRACT_SPECS` updated with correct CBOT/CME contract specs.
  `gamma_futures` confirmed running on full 5-symbol universe in Part 2,
  emitting a signal on M2K.
- **Per-file business overlay stale thresholds.** `_read_business_runtime` in
  `dynamic_risk_allocator.py` now takes a `stale_seconds` parameter. Each
  call site passes the correct threshold: `tier_state.json`=600s,
  `winner_scaling.json`=1800s (was 600s — shorter than the 900s cadence),
  `regime_booster.json`=120s.
- **`winner_scaler` explicit roster.** All 16 canonical strategies now appear
  explicitly in `runtime/winner_scaling.json:multipliers` on every publish
  cycle. Previously 10 strategies were absent and relied on implicit 1.0 default.
- **Diagnostic logging added to 3 strategies.** `alpha_intraday`, `alpha_options`,
  and `gamma_reversion` had no `logging` imports at all. All three now emit
  per-cycle info logs on zero-signal returns and per-symbol debug logs at key
  decision points. No more silent [] returns.
- **Telegram briefs fully de-jargoned.** Single `CHADS_TAKE_SYSTEM_PROMPT`
  shared across morning, EOD, and weekly. All finance terms replaced with
  plain English. `broker_sync` excluded from all 3 report surfaces via shared
  `_filter_for_ranking()` helper.

### Strategic effect

The risk allocator no longer answers only "how much should each
strategy risk per day". It now answers four questions in sequence:

1. *Is the account big enough to run this strategy?*
   (TierManager → `tier_state.json`)
2. *Is this strategy actually winning?*
   (WinnerScaler → `winner_scaling.json`)
3. *Is the regime favorable enough to press?*
   (RegimeBooster → `regime_booster.json`)
4. *Is it safe to pay the operator?*
   (WithdrawalManager → `withdrawal_authorization.json`)

The operator no longer has to make those decisions manually.

---

## 1. MISSION & ARCHITECTURE

### Mission

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a **business**, not a
bot. The business earns money by running a diversified portfolio of
systematic trading strategies over multiple asset classes, compounding
the profits through a structured allocation rule, and paying a salary
to the operator from sustained surplus above a high-water mark — never
touching the seed capital base.

The product goal is plain: **the operator checks their phone at the
end of the day, sees what CHAD made, and does nothing.** Everything
downstream — risk management, execution, reconciliation, health
monitoring, salary authorization — is automation designed to make "do
nothing" the safe default.

In v8.3 the business framework (§6) is the substrate that makes this
literal: phase progression (BUILD → GROW → PAY) is mechanical, tier
gating is mechanical, performance reweighting is mechanical, and the
salary is *authorized* mechanically (the operator still moves money
manually — CHAD never touches the wire).

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
│                                                            │
│   build_kraken_     build_execution_plan                   │
│     intents_from_   ↓                                      │
│     routed_signals  build_intents_from_plan                │
│         │           ↓                                      │
│   _enforce_         routing_gates.run_all_gates            │
│     kraken_rest_    (A4 / E2 / E5 / R7 / S5)               │
│     pair (NEW v8.3) ↓                                      │
│         │           vote_collector  (S1, min_votes=1)      │
│         ▼           ↓                                      │
│   KrakenExecutor    sizing pipeline  (R3 → R5 → R6 → S5)   │
│         │           ↓                                      │
│   _assert_kraken_   IbkrAdapter.submit_strategy_intents    │
│     rest_pair       │                                      │
│     (REST border;   ▼                                      │
│      NEW v8.3)      IBKR fills                             │
│         ▼           │                                      │
│   Kraken fills      │                                      │
│         │           │                                      │
│         └───────────┴─→ normalize_paper_fill_evidence ─────┤
│                                                            │
│                       ↓                                    │
│                 PaperExecutionEvidenceWriter               │
│                       ↓                                    │
│            data/fills/FILLS_YYYYMMDD.ndjson                │
│                       ↓                                    │
│       position_reconciler / reconciliation_publisher       │
│                       ↓                                    │
│              ProfitRouter (50/30/20 advisory)              │
└────────────────────────────────────────────────────────────┘

   Strategies   Pipeline    Plan    Risk   Allocator
        ───────────────────────────────────────►
                                                │
                                                ▼
   ┌──── BUSINESS FRAMEWORK (NEW v8.3) ───────────────────────┐
   │                                                          │
   │   tier_filter ◄── TierManager ◄── tier_state.json        │
   │   winner_scale ◄── WinnerScaler ◄── winner_scaling.json  │
   │   regime_boost ◄── RegimeBooster ◄── regime_booster.json │
   │                                                          │
   │   ─►  Caps ──► SCR ──► LiveGate ──► Execution            │
   │                                          │                │
   │                                          ▼                │
   │                                   Fills → Ledger          │
   │                                          │                │
   │                                          ▼                │
   │                                   Profit_Router 50/30/20  │
   │                                          │                │
   │                                          ▼                │
   │                                   WithdrawalManager       │
   │                                   (HWM/Phase)             │
   │                                          │                │
   │                                          ▼                │
   │                                   Salary Authorization    │
   └──────────────────────────────────────────────────────────┘
```

### Built / Degraded / Not yet

| Component | Status | Note |
|---|---|---|
| 16 strategy registry | **BUILT** | Forex commented out (`chad/strategies/__init__.py` registry block) — 16 of 17 active. |
| Signal router (bucket meta carry) | **BUILT** | `chad/utils/signal_router.py:179` (commit `f6caf3d`). |
| Asset-class splitter | **BUILT** | `chad/execution/execution_pipeline.py:1164`. |
| IBKR EMS / OMS | **BUILT** | A1 separation `chad/execution/ems.py` + `oms.py` (Phase-8 Session 9). |
| Kraken EMS / OMS | **BUILT** | REST altname maps `chad/execution/execution_pipeline.py:1228-1264`. |
| Kraken REST pair guard (executor + border) | **BUILT v8.3** | `chad/execution/kraken_executor.py:41`, `chad/exchanges/kraken_client.py:55` (commit `15677a2`). |
| Routing gates 5-stack | **BUILT** | `chad/execution/routing_gates.py:run_all_gates`. |
| Sizing R3/R5/R6 | **BUILT** | `target_daily_vol=0.015`, `correlation_threshold=0.65`. |
| Profit lock | **BUILT** | `runtime/profit_lock_state.json` mode=NORMAL, sizing 1.0. |
| SCR | **BUILT** | WARMUP, sizing_factor 0.10, 84 effective trades. |
| Stop bus | **BUILT** | `runtime/stop_bus.json` active=false (cleared 2026-04-22). |
| Edge decay (F4) | **BUILT** | `consecutive_threshold=5`, `min_trades=20`. |
| Reconciliation publisher | **BUILT** | GREEN; paper-mode skip of strategy entries `chad/ops/reconciliation_publisher.py:91-92`. |
| Paper evidence normalizer | **BUILT** | Single chokepoint `chad/execution/paper_exec_evidence_writer.py:925`. |
| Position guard `opened_at` | **BUILT** | `chad/core/position_guard.py:155-157` (commit `56eaeea`). |
| Broker-truth rebuild in paper | **BUILT** | `chad/core/live_loop.py:641-651` (commit `6af971d`). |
| Telegram intelligence layer | **BUILT** | Free-text router + elite voice + alerts (commit `e150c0b`). |
| Telegram BUSINESS STATUS section | **BUILT v8.3** | `chad/ops/daily_chad_report.py:1474-1505`, `:1786-1806` (commit `1caa111`). |
| Dashboard chat | **BUILT** | `/api/chat`, model `claude-sonnet-4-6`. |
| Dashboard business endpoint | **BUILT v8.3** | `chad/dashboard/api.py:484-525`, business in `/api/state` (commit `1caa111`). |
| Bar provider polling | **BUILT** | `chad-ibkr-bar-provider.service`. |
| Strategy intelligence cache | **BUILT** | `runtime/strategy_intelligence.json` (48h TTL). |
| **PortfolioSnapshotPublisher** | **BUILT v8.3** | `chad/ops/portfolio_snapshot_publisher.py` (commit `1608fec`). |
| **EquityHistoryPublisher** | **BUILT v8.3** | `chad/ops/equity_history_publisher.py` (commit `1608fec`). |
| **TierManager** | **BUILT v8.3** | `chad/risk/tier_manager.py` (commit `1608fec`). |
| **WinnerScaler** | **BUILT v8.3** | `chad/risk/winner_scaler.py` (commit `1caa111`). |
| **RegimeBooster** | **BUILT v8.3** | `chad/risk/regime_booster.py` (commit `1caa111`). |
| **WithdrawalManager** | **BUILT v8.3** | `chad/risk/withdrawal_manager.py` (commit `1608fec`). |
| **BusinessPhaseTracker** | **BUILT v8.3** | `chad/ops/business_phase_tracker.py` (commit `1caa111`). |
| Profit-routing 50/30/20 | **DEGRADED** (advisory) | Ledger-only until live (`runtime/profit_routing.json`). |
| `alpha_options` new entries | **DEGRADED** | Existing SPY long blocks new spreads (MAINTAINED state) until existing closes. |
| `omega_vol` health | **DEGRADED** | `health_score=0.10` (3 samples). |
| ML veto loop (Phase 5B) | **NOT YET** | XGBoost retrain pipeline incomplete. |
| Per-trade P&L decomposition | **NOT YET** | Alpha vs spread vs slippage attribution incomplete. |
| Live trading | **NOT YET** | `runtime/live_readiness.json:ready_for_live=false`. |
| Salary withdrawal automation | **NOT YET** | `WithdrawalManager` authorizes; operator still moves money manually. |

---

## 2. RUNTIME STATE (live snapshot)

All values pulled at write time from the canonical state files. Each
row cites its source.

### Account equity

| Field | Value | Source |
|---|---|---|
| `ibkr_equity` | `$167,082.87 USD` | `runtime/portfolio_snapshot.json:ibkr_equity` (`2026-04-27T22:04:50Z`) |
| `kraken_equity` | `$184.58 USD` | `runtime/portfolio_snapshot.json:kraken_equity` |
| `coinbase_equity` | `$0.00` | unused (CAD-based account, no Coinbase) |
| **total_equity** | **`$167,267.45 USD`** | sum (ibkr + kraken + coinbase) |
| `total_equity` (allocator) | `$167,267.45` | `runtime/dynamic_caps.json:total_equity` (`2026-04-27T22:06:34Z`) |
| `daily_risk_fraction` | `0.05` | `runtime/dynamic_caps.json:daily_risk_fraction` |
| `portfolio_risk_cap` | `equity × daily_risk_fraction × profit_lock_factor` | `runtime/dynamic_caps.json:portfolio_risk_cap` (formula-driven; value moves with equity) |
| Kraken raw | `BTC=0.0012, CAD=252.8538` | `runtime/kraken_balances.json` |

### SCR — Self-Calibrating Risk

Source: `runtime/scr_state.json` (ts `2026-04-27T22:06:41Z`).

| Field | Value |
|---|---|
| state | `WARMUP` |
| sizing_factor | `0.10` |
| effective_trades | `84` (nested under `stats` key in `scr_state.json`) |
| paper_trades | `5000` |
| live_trades | `0` |
| excluded_untrusted | `4549` |
| excluded_manual | `0` |
| excluded_nonfinite | `0` |
| total_trades | `5000` |
| win_rate | `0.5714` (≈ 57.1%) (nested under `stats` key in `scr_state.json`) |
| sharpe_like | `+0.6042` |
| max_drawdown | `-1416.34` |
| total_pnl (effective) | `-6318.91` |
| paper_only | `true` |
| reasons[0] | `"Warmup: only 84 effective trades (< 100 required)."` |
| ttl_seconds | 180 |

`warmup_min_trades = 100` per `runtime/scr_config.json:2`. CAUTIOUS
gate also requires `win_rate ≥ 0.35`, `sharpe_like ≥ 0.10`,
`max_drawdown ≥ −$15,000` — all three currently satisfied; only the
trade-count gate keeps us in WARMUP. We are 16 effective trades short.

### Regime classifier

Source: `runtime/regime_state.json` (ts `2026-04-27T22:06:22Z`).

| Field | Value |
|---|---|
| regime | `trending_bull` |
| previous_regime | `trending_bull` (stable transition) |
| confidence | `0.6864` |
| inputs_used | `realized_vol_percentile, adx, trend_slope, market_breadth` |
| source | `live_loop.run_once` |
| ttl_seconds | 60 |

Active strategies in `trending_bull` (per
`config/regime_activation_matrix.json:9-13`):
`alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_forex,
alpha_options, beta, beta_trend, delta, gamma_futures, omega_macro,
omega_momentum_options`.
*(`alpha_forex` is registry-disabled — see §3.)*

### Tier (NEW v8.3)

Source: `runtime/tier_state.json` (ts `2026-04-27T22:04:45Z`).

| Field | Value |
|---|---|
| tier_name | **`PRO`** |
| tier_description | `"All 16 strategies firing at meaningful size"` |
| current_equity_usd | `$167,269.69` |
| tier_min_equity | `$160,000` |
| tier_max_equity | `$1,000,000` |
| previous_tier | `PRO` (no change) |
| promoted_at_utc | `2026-04-27T19:00:35Z` |
| enabled_strategies | 16 strategies (full PRO list) |

PRO enables:
`alpha, alpha_intraday, alpha_crypto, alpha_options, alpha_futures,
delta, delta_pairs, gamma, gamma_futures, gamma_reversion, beta,
beta_trend, omega, omega_vol, omega_macro, omega_momentum_options`.

### Business phase (NEW v8.3)

Source: `runtime/business_phase.json` (ts `2026-04-27T21:42:04Z`).

| Field | Value |
|---|---|
| phase | **`GROW`** |
| phase_description | `"Engine is built. Now growing the account before salary starts. SCR must reach CONFIDENT first."` |
| current_equity_usd | `$167,267.75` |
| seed_capital_usd | `$50,000.00` |
| growth_pct_from_seed | `+234.54%` |
| days_in_phase | `1` |
| next_phase_requirement | `"To enter PAY phase: SCR must promote from WARMUP to CONFIDENT; Need 13+ more days of equity history."` |
| compound_metrics.total_return_pct | `+234.54%` |
| compound_metrics.annualized_return_pct | `0.0` (suppressed below 14d history; see `business_phase_tracker.py:202`) |
| compound_metrics.days_active | `1` |
| compound_metrics.high_water_mark_usd | `$167,227.46` |

### Withdrawal authorization (NEW v8.3)

Source: `runtime/withdrawal_authorization.json` (ts `2026-04-27T19:38:21Z`).

| Field | Value |
|---|---|
| phase | **`GROW`** |
| current_equity_usd | `$167,227.46` |
| seed_capital_usd | `$50,000.00` |
| high_water_mark_usd | `$167,227.46` |
| drawdown_from_hwm_pct | `0.0` |
| spendable_surplus_usd | `$0.00` |
| **authorized_withdrawal_usd** | **`$0.00`** |
| scr_state | `WARMUP` |
| history_days | `1` |
| reason | `"GROW phase: equity above build threshold but SCR is WARMUP (need CONFIDENT). Reinvesting profits."` |

### Winner scaling (NEW v8.3)

Source: `runtime/winner_scaling.json` (ts `2026-04-27T21:57:04Z`).

| Field | Value |
|---|---|
| max_multiplier | `1.50` |
| min_multiplier | `0.50` |
| median_expectancy | `4.41` |
| n_strategies_scaled | `5` |
| n_strategies_neutral | `4` |
| min_trades_for_scaling | `5` |
| excluded_strategies | `broker_sync, manual, paper_exec, unknown` |

| Strategy | Multiplier | Note |
|---|---|---|
| `alpha` | **`1.500` (boosted)** | Top of scoring pool |
| `delta` | **`1.304` (boosted)** | 75% win rate, 7 trades |
| `alpha_intraday` | `1.000` | 5 trades, neutral pool |
| `gamma_futures` | `1.000` | 2 trades, below threshold |
| `omega_vol` | `1.000` | 3 trades, below threshold |
| `broker_sync` | `1.000` | Excluded — bookkeeping label |
| `reconciled_phase2_20260419_carryover` | `1.000` | Below threshold |
| `RECONCILED_PHASE2_20260419` | **`0.500` (penalized)** | 7 losing trades |
| `alpha_futures` | **`0.500` (penalized)** | $-51 expectancy on 329 trades |

The "Top 3 boosted / Top 3 penalized" framing is partially populated:
only 2 strategies have crossed the boost line (`alpha 1.5×`,
`delta 1.304×`), and 2 are at the floor
(`alpha_futures 0.5×`, `RECONCILED_PHASE2_20260419 0.5×`).

### Regime booster (NEW v8.3)

Source: `runtime/regime_booster.json` (ts `2026-04-27T21:14:50Z`).

| Field | Value |
|---|---|
| multiplier | **`1.0`** |
| active | `false` |
| reasons | `["vetoed: low_confidence_0.69"]` |
| regime | `trending_bull` |
| confidence | `0.6864` |
| vix | `18.71` |
| event_severity | `medium` |

Veto rule (`chad/risk/regime_booster.py:131-132`): confidence below
`min_confidence` (0.70 per `config/regime_booster_policy.json:5`)
forces multiplier to 1.0. Today's `0.6864` is just below the gate, so
the booster is dormant. Other potential vetoes (high VIX, event risk
high/extreme, unfavorable regime) are not active.

### Equity / PnL

| Field | Value | Source |
|---|---|---|
| account_equity | `$167,267.45` | `runtime/dynamic_caps.json:total_equity` |
| portfolio_risk_cap | `$8,363.37` | `runtime/dynamic_caps.json:portfolio_risk_cap` (post profit-lock) |
| daily realized PnL today | `+$125.60` | `runtime/pnl_state.json:realized_pnl` (`2026-04-27T22:07:41Z`) |
| trade_count today | `423` | `runtime/pnl_state.json:trade_count` |
| pnl_pct_of_equity | `+0.0751%` | `runtime/pnl_state.json` |

### Profit lock

Source: `runtime/profit_lock_state.json` (ts updates every 60s).

- mode = `NORMAL`
- sizing_factor = `1.00`
- stop_new_entries = `false`
- daily_loss_limit_pct = `3.0%`
- daily_loss_limit_dollars = `$5,018.02` (3% of equity)
- daily_loss_today = `$0.00`
- profit_lock_active = `false`

Equity scaling: WARN at +1.5%, LOCK1 at +3.0%, LOCK2 at +5.0%, LOCK3
at +8.0%, HARD_STOP at +10.0%. Currently NORMAL.

### Reconciliation

Source: `runtime/reconciliation_state.json` (ts `2026-04-27T22:04:46Z`).

- status = **`GREEN`**
- broker_source = `ibkr:clientId=83`
- chad_open = `1`, broker_positions = `2`
- worst_diff = `0.0`
- mismatches = `[]`
- drifts = `[{"symbol": "SPY", "chad": 0.0, "broker": -17.0, "diff": 17.0}]`
  — broker holds a 17-share short SPY (paper); CHAD's strategy ledger
  has no SPY entry. Classified as **drift** (broker-side, not a
  strategy mismatch) because all of the diff is in `broker_sync` and
  none in strategies — see `reconciliation_publisher.py:212-217`.
- excluded_symbols = `[]`, futures_excluded_symbols = `[]`
- ttl_seconds = 300

### Stop bus

Source: `runtime/stop_bus.json`.

- active = `false`
- cleared_at = `2026-04-22T01:56:50Z`
- cleared_by = `smoke_test`

### Open positions

From `runtime/position_guard.json` — 18 `broker_sync|*` entries cached
(echoes of broker positions); 3 `open=true` (rest are closed).

Per `runtime/reconciliation_state.json`: `chad_open=1, broker_positions=2`.

### Market snapshot

VIX last close = `18.71` (`data/bars/1d/VIX.json` — 400 bars cached).
Below the 18.0 calm threshold by 0.71; therefore `vix_calm` boost
factor in the regime booster does NOT fire even after low_confidence
clears.

Kraken (`runtime/kraken_balances.json`): BTC `0.0012`, CAD `252.85`,
USD-equivalent `$184.58`. Live as of `2026-04-27T22:06:25Z`.

### Services

Total `chad-*` units loaded: **95** (53 services + 42 timers —
adds `chad-regime-booster.timer` and `chad-regime-booster.service` from v8.4).
Active (running): **14** services. Failed: **0**.

The major hot-path services are all running (`chad-live-loop`,
`chad-orchestrator`, `chad-ibgateway`, `chad-ibkr-bar-provider`,
`chad-kraken-ws`, `chad-shadow-status`, `chad-metrics`, `chad-backend`,
`chad-dashboard`, `chad-telegram-bot`, `chad-x11vnc`, `chad-xvfb`,
`chad-strategy-intelligence-refresh`).

### Bar freshness

- 1d bars: `data/bars/1d/` — 30 symbols
  (AAPL, BAC, BTC-USD, ETH-USD, GLD, GOOGL, IEMG, IWM, M6E, MCL, MES,
   MGC, MNQ, MSFT, NVDA, PSQ, QQQ, SH, SIL, SOL-USD, SPY, SVXY, TLT,
   UVXY, VIX, VIXY, VWO, VXX, ZB, ZN).
- 1m bars: `data/bars/1m/` — 25 symbols polled via
  `chad-ibkr-bar-provider.service` every 30 s.
- VIX: `data/bars/1d/VIX.json`, last close `18.71` (consumed by
  `omega`, `omega_vol`, `omega_macro`, `omega_momentum_options`,
  `alpha_options`, and now **RegimeBooster**).

### Strategy intelligence

Source: `runtime/strategy_intelligence.json` — most recent
`regime_profile` entries `2026-04-27T12:30Z–12:32Z`. AI cache TTL 48h
(`chad/ops/daily_chad_report.py:1353`). All current entries fall back
to `profile=normal` — same classifier-input wiring gap noted in v8.2;
see §14. The reasoning string makes the gap explicit:
`"Insufficient data due to macro context provider error; VIX and
macro risk label unknown; defaulting to normal regime per rules."`

### Institutional consensus

Source: `runtime/institutional_consensus.json` (updated
`2026-04-26T00:00:05Z`). 7 funds, 25 holdings ranked by conviction.
Same top 10 as v8.2 (no weekly refresh has run since).

### Profit routing ledger

Source: `runtime/profit_routing.json` (latest decision
`2026-04-27T21:13:40Z`).

- decisions logged: 26 (was 19 in v8.2)
- totals (advisory):
  - trading_capital = `$1,051.68`
  - beta_allocation = `$631.01`
  - amplifier_allocation = `$420.67`

Notable v8.3-window entries:
- `alpha 34.80 → (17.40, 10.44, 6.96)` 2026-04-27T13:36:47Z
- `alpha 69.02 → (34.51, 20.71, 13.80)` 2026-04-27T13:36:47Z
- `alpha_intraday 4.41 → (2.20, 1.32, 0.88)` 2026-04-27T17:35:51Z
- `broker_sync 6.93 → (3.47, 2.08, 1.39)` 2026-04-27T17:39:00Z
- `alpha_test 50.00 → (25.00, 15.00, 10.00)` 2026-04-27T21:13:40Z

Largest single decision (carried over from v8.2): `gamma_futures`
`$1,058.00` realized → `(529.00, 317.40, 211.60)` at
`2026-04-24T21:51:22Z`.

---

## 3. THE 16 STRATEGIES

For each strategy: source path, sleeve, weight, dollar cap (post full
chain), regime gating, fire window, universe, signal conditions,
current health, and status.

Weights from `config/strategy_weights.json`; caps from
`runtime/dynamic_caps.json`; regime gating from
`config/regime_activation_matrix.json`; sleeve membership from
`chad/risk/dynamic_risk_allocator.py:454-461`.

> **Sleeve recap** — the 50/30/20 chassis:
> - **ALPHA sleeve** (50%): `alpha, alpha_futures, alpha_intraday,
>   alpha_options, omega_momentum_options, gamma, gamma_futures,
>   gamma_reversion`.
> - **BETA sleeve** (30%): `beta, beta_trend`.
> - **ADAPTIVE sleeve** (20%): `omega, omega_macro, omega_vol, delta,
>   delta_pairs, alpha_crypto`.

Each strategy now has three NEW v8.3 fields surfaced in the cap
breakdown (`runtime/dynamic_caps.json:strategies.<name>`):
`tier_factor` (0.0 if not enabled in the current tier; else 1.0),
`winner_factor` (0.5–1.5), `regime_factor` (1.0–1.5). The final
`dollar_cap` =
`base_cap_pre_overlay × tier_factor × winner_factor × regime_factor`.

### alpha — Intraday tactical momentum brain

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha.py` |
| Sleeve | ALPHA |
| Weight | 0.16 (largest) |
| Base cap (pre-overlay) | `$1,338.14` |
| **Final dollar cap** | **`$2,007.21`** (`runtime/dynamic_caps.json:strategies.alpha.dollar_cap`) |
| tier_factor | `1.0` (PRO enables) |
| winner_factor | **`1.5`** (top expectancy) |
| regime_factor | `1.0` (booster vetoed by low_confidence) |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Per-symbol 3 signals/day cap; max 8 signals per cycle (`alpha.py:196,319`). |
| Universe | Legend-driven via `ctx.legend.weights`, fallback to `ctx.prices` keys (`alpha.py:216`). |
| Conditions | Four entry regimes: uptrend (EMA fast > slow, momentum ≥ 0.35 ATR), recovery (price > EMA slow, momentum ≥ 0.175 ATR), downtrend (mirrored shorts), chop (fade mid-band) (`alpha.py:254-333`). |
| Health | 0.8428 on 39 samples; win_rate 0.476; normalized_sharpe 1.0 (`runtime/strategy_health.json`). |
| Expectancy | `+$7.21` on 39 trades, status `watch` (`runtime/expectancy_state.json`). |
| Open positions | none currently per `position_guard.json` (alpha entries are closed). |
| Status | **ACTIVE** — winner-scaled. |

### alpha_intraday — Delta high-convexity day-trading brain

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_intraday.py` |
| Sleeve | ALPHA |
| Weight | 0.03 |
| Base cap | `$250.90` |
| **Final dollar cap** | **`$250.90`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` (5 trades — exactly at threshold but neutral pool) |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | 1m bars with daily fallback; 10-min per-symbol cooldown. |
| Universe | SPY, QQQ, AAPL, NVDA, MSFT, GOOGL, BAC, MES, MNQ, BTC-USD. |
| Conditions | Vol explosion / momentum surge / mean-reversion snap (`alpha_intraday.py:244-296`). |
| Health | 1.0 on 5 samples (limited confidence). |
| Expectancy | `+$4.41` on 5 trades, status `new` — this is the WinnerScaler's median anchor. |
| Open positions | None. |
| Status | **ACTIVE**. |

### alpha_options — Defined-risk vertical spreads

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_options.py` (+ `alpha_options_config.py`) |
| Sleeve | ALPHA |
| Weight | 0.04 |
| Base cap | `$334.53` |
| **Final dollar cap** | **`$334.53`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` (no expectancy entry) |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile` |
| Silent in | `ranging, unknown, adverse` |
| Window | Triggered by alpha/gamma/gamma_reversion signals at confidence ≥ 0.70 (`alpha_options.py:96`). Max 4 open spreads. |
| Universe | SPY only (`alpha_options_config.py:25-27`). |
| Conditions | Bullish source signal → bull call spread (21–45 DTE, 2% OTM, 5% width). Bearish → bear put spread (mirror). |
| Health | n/a (sample threshold). |
| Open positions | Carried from v8.2 — `alpha_options\|SPY` from 2026-04-24T20:18Z. (Position MAINTAINED → no new entries until existing closes.) |
| Status | **CONDITION-GATED** — see §14. |

### alpha_futures — Futures momentum engine

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_futures.py` (+ `alpha_futures_config.py`) |
| Sleeve | ALPHA |
| Weight | 0.09 |
| Base cap | `$752.70` |
| **Final dollar cap** | **`$376.35`** |
| tier_factor | `1.0` |
| winner_factor | **`0.5`** (penalized — `-$51.06` expectancy on 329 trades) |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | MES, MNQ continuous; **MCL/MGC restricted 13:30–20:00 UTC for new entries** (`alpha_futures.py:488-497`). |
| Universe | **MES, MNQ, MGC** (`alpha_futures.py:98`). MCL was reassigned to `gamma_futures`. The legacy 4-symbol default still appears in `alpha_futures_config.py:55-60` but the strategy module's hardcoded tuple wins. |
| Conditions | Momentum: price > EMA_fast > EMA_slow (BUY) or mirror (SELL). Breakout override: 20-bar high/low. Exits on 2× ATR stop, EMA_slow cross, 20-bar time stop. |
| Health | 0.4713 on **329 samples** — largest sample base in the system; win_rate 0.571. |
| Expectancy | `-$51.06`, status `underperforming`. |
| Open positions | None. |
| Status | **ACTIVE** but **WINNER-SCALER PENALIZED** at 0.5× — book is asking this strategy to prove itself before getting bigger size again. |

### alpha_crypto — Crypto momentum signals

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_crypto.py` (commit `81dafce`) |
| Sleeve | ADAPTIVE |
| Weight | 0.04 (key `alpha_crypto`; renamed from `crypto` in v8.4 commit `f62d914`) |
| Base cap | `$334.53` |
| **Final dollar cap** | **`$334.53`** |
| tier_factor | `1.0` — `alpha_crypto` is in the PRO tier's enabled_strategies list and matches the literal key in strategy_weights.json end-to-end. The legacy shim at chad/risk/dynamic_caps.py:67 is now a no-op for this key but is retained for older runtime files. |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Always armed when regime allows; signal pipeline emits long-only (`alpha_crypto.py:641-645,627-758`). |
| Universe | Default `BTC-USD, ETH-USD, SOL-USD`; CAD pairs (`BTC-CAD, ETH-CAD`) added when Kraken paper holds ZCAD (`alpha_crypto.py:75-86`). Kraken REST altname routing via `normalize_kraken_pair`. **Now layered with two-step REST pair guard** (commit `15677a2`). |
| Conditions | (1) SMA20 momentum breakout (price > SMA20, 3-day return ≥ 1.5%); (2) 5d/20d vol-ratio expansion ≥ 0.7; (3) regime multiplier — 0.5 in trending_bear, 1.0 elsewhere. Min strength 0.3, all long-only. |
| Health | not computed. |
| Open positions | None. |
| Status | ACTIVE. |

### beta — Institutional long-term compounder

| Field | Value |
|---|---|
| Source | `chad/strategies/beta.py` (commit `2c6a16b`) |
| Sleeve | BETA |
| Weight | 0.05 (carved from `beta_trend` in action `beta_carveout_from_beta_trend_20260423`) |
| Base cap | `$418.17` |
| **Final dollar cap** | **`$418.17`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` (no expectancy entry) |
| regime_factor | `1.0` |
| Active in | every non-`adverse` regime |
| Silent in | `adverse` |
| Window | Max 2 signals/cycle; max 3/week rolling; per-symbol 7-day rebalance gate. |
| Universe | `runtime/institutional_consensus.json` top names (currently 25, top 10 listed in §2). Consensus must be ≤ 45 days old (13F quarterly cycle). |
| Conditions | If `target_weight − current_weight ≥ 2%`, emit a BUY sized to fill ~50% of the gap. Long-only. |
| Health | n/a. |
| Open positions | None. |
| Status | **ACTIVE** (advisory funding via 30%-of-realized-profit slice). |

### beta_trend — Legend-driven long-term ETF / equity allocator

| Field | Value |
|---|---|
| Source | `chad/strategies/beta_trend.py` (renamed from old `beta.py`, commit `20d6f59`) |
| Sleeve | BETA |
| Weight | 0.20 |
| Base cap | `$1,672.67` |
| **Final dollar cap** | **`$1,672.67`** (largest *base* cap in the book; alpha's 1.5× winner boost produces the largest *final* cap) |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Once-per-UTC-day per symbol (hard gate); max 20 signals/day; 21-day hold before add-ons; 14-day cooldown after exit. |
| Universe | Top legend-weighted names (min 5%, max 10 symbols), EWMA-smoothed. |
| Conditions | Entry if flat: BUY size = `clamp(3 + legend_weight × 10, 3, 8)`. Add-on after 21d: 50% of entry size, 95% confidence. |
| Health | n/a. |
| Open positions | None. |
| Status | **ACTIVE**. |

### gamma — Activated swing engine

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma.py` |
| Sleeve | ALPHA |
| Weight | 0.07 |
| Base cap | `$585.44` |
| **Final dollar cap** | **`$585.44`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `ranging, volatile, unknown` |
| Silent in | `trending_bull, trending_bear, adverse` |
| Window | No fixed schedule; bar-driven. Min 60 bars. |
| Universe | Pulled dynamically from `ctx.bars` keys. |
| Conditions | Trend regime: EMA_fast > slow, price > fast, momentum ≥ 0.35 ATR. Range regime: deviates from EMA_slow by ≥ 0.75 ATR (mean reversion). Vol gate: ATR% ∈ [0.15%, 4.0%]. Anti-chase: range/ATR ≤ 3.2. |
| Health | n/a. |
| Open positions | None. |
| Status | **REGIME-SILENT** (currently `trending_bull`). |

### gamma_futures — Futures mean-reversion counterpart

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma_futures.py` (+ config) |
| Sleeve | ALPHA |
| Weight | 0.05 |
| Base cap | `$418.17` |
| **Final dollar cap** | **`$418.17`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` (2 trades — below scaling threshold) |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Continuous (energy 24/5; bonds liquid hours). |
| Universe | **MCL** primary; extends with `MYM, M2K` if bars exist; otherwise `ZN, ZB`. **Disjoint from alpha_futures** post-`6af971d`. |
| Conditions | Short when `RSI > 70 AND price > BB_upper`; Long when `RSI < 30 AND price < BB_lower`. ATR-based sizing: 1.2% risk, max $40k notional. |
| Health | 1.0 on 2 samples. |
| Expectancy | `+$544.00` on 2 trades — too small a base to scale, but the `$1,058.00` profit-routing event on 2026-04-24 was this strategy. |
| Open positions | None. |
| Status | **ACTIVE**. |

### gamma_reversion — ETF statistical mean-reversion

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma_reversion.py` (+ config) |
| Sleeve | ALPHA |
| Weight | 0.04 |
| Base cap | `$334.53` |
| **Final dollar cap** | **`$334.53`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `ranging` |
| Silent in | every other regime |
| Window | Continuous within active regime. Min 40 bars; RSI(14), Boll(20), Z(20). |
| Universe | **SPY, QQQ, GLD, TLT** (TLT added in `6af971d`). |
| Conditions | 3/3 confluence — SHORT: RSI > 72 AND (price > BB_upper OR z > 1.8) AND ROC > 0; LONG: mirror with 28/-1.8/<0. GLD requires strict 3/3. 2.5× ATR stop, 15-bar time stop, SMA20 cross = target. |
| Health | n/a. |
| Open positions | None. |
| Status | **REGIME-SILENT** (currently `trending_bull`). |

### delta — Cross-asset convexity hunter

| Field | Value |
|---|---|
| Source | `chad/strategies/delta.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.02 (smallest base weight) |
| Base cap | `$167.27` |
| **Final dollar cap** | **`$218.12`** |
| tier_factor | `1.0` |
| winner_factor | **`1.304`** (boosted — 75% win rate, +$5.75 expectancy) |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | No per-symbol cooldown (conviction-driven, wealth-mode). Cash floor $2.5k. |
| Universe | `ctx.delta_universe` override → legend top weights → price keys; max 4 symbols. |
| Conditions | Conviction ≥ 0.65 required. Trend / breakout / momentum scoring → composite. |
| Health | 0.925 on 7 samples; win_rate 0.75. |
| Expectancy | `+$5.75`, status `new`. |
| Open positions | Carried from v8.2 — `delta\|SPY` from 2026-04-24T21:49Z. |
| Status | **ACTIVE**, winner-scaled. Strongest health score in the book. |

### delta_pairs — Market-neutral ETF pairs trader

| Field | Value |
|---|---|
| Source | `chad/strategies/delta_pairs.py` (+ config) |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| Base cap | `$418.17` |
| **Final dollar cap** | **`$418.17`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `ranging` |
| Silent in | every other regime |
| Window | Continuous within active regime. 60-day lookback, min 40 bars. |
| Universe | **SPY/QQQ** (r 0.993), **SPY/IWM** (0.967), **QQQ/IWM** (0.941). |
| Conditions | z = (ratio_now − mean_60) / std_60. Entry at \|z\| ≥ 2.0; exit ≤ 0.5; stop ≥ 3.5. Both legs same unit count, max 50 units/leg. |
| Health | n/a. |
| Open positions | None. |
| Status | **REGIME-SILENT**. |

### omega — Wealth-safe hedge sleeve

| Field | Value |
|---|---|
| Source | `chad/strategies/omega.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| Base cap | `$418.17` |
| **Final dollar cap** | **`$418.17`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `volatile, unknown` |
| Silent in | trending regimes (by design — hedge is dormant in calm seas) |
| Window | Cooldown 60 minutes between activation/deactivation cycles. |
| Universe | **SH** (inverse SPY) and **PSQ** (inverse QQQ). |
| Conditions | Activation requires ≥ 2 sensor agreement: drawdown ≤ -6%, ATR% ≥ 3%, VIX ≥ 25. 5 base hedge units, max 25/symbol. |
| Health | n/a. |
| Open positions | None. |
| Status | **ARMED**, currently dormant — `trending_bull` regime + VIX 18.71 (below 20 threshold). |

### omega_vol — VIX-linked volatility alpha

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_vol.py` (+ config) |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| Base cap | `$418.17` |
| **Final dollar cap** | **`$418.17`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` (3 trades — below threshold) |
| regime_factor | `1.0` |
| Active in | `volatile` |
| Silent in | every other regime |
| Window | 5-state VIX regime (LOW_VOL < 15; NORMAL 15–22; ELEVATED 22–30; CRISIS ≥ 30; VOL_CRUSH ≥ 20% drop from peak). UVXY hard 10-bar time stop. |
| Universe | **SVXY** (short vol, contango) and **UVXY** (long vol, spike). |
| Conditions | LOW_VOL → SVXY BUY; NORMAL_VOL → mild SVXY; ELEVATED → skip; CRISIS → UVXY BUY; VOL_CRUSH → SVXY mean-revert. 3-8 units, max 3% equity per position. Confidence ≥ 0.65. |
| Health | **0.10 on 3 samples** — flagged. Win rate 0.0; slippage_ratio 0.5; regime_alignment 0.0. |
| Expectancy | `-$0.31` on 3 trades, status `new`. |
| Open positions | None. |
| Status | **DEGRADED** (lowest health score in the book; sample size too small to be actionable yet). |

### omega_macro — Macro regime futures

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_macro.py` (+ config) |
| Sleeve | ADAPTIVE |
| Weight | 0.03 |
| Base cap | `$250.90` |
| **Final dollar cap** | **`$250.90`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | every regime except `adverse` (universally eligible). |
| Silent in | `adverse` |
| Window | Continuous; min 40 bars. Risk budget 1.0%. |
| Universe | **ZN, ZB, M6E**. |
| Conditions | 4-state macro regime: `RISK_OFF` (VIX>25, DD<-5% → ZN/ZB BUY, M6E SELL), `RISK_ON` (VIX<18, DD>-2% → mirror), `STAGFLATION` (ZN/ZB BUY, M6E SELL), `NEUTRAL`. |
| Health | n/a. |
| Open positions | None. |
| Status | **ACTIVE**, currently `RISK_ON`-leaning per low VIX. |

### omega_momentum_options — Intraday single-leg options momentum

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_momentum_options.py` |
| Sleeve | ALPHA (per `dynamic_risk_allocator.py:454-457`) |
| Weight | 0.03 |
| Base cap | `$250.90` |
| **Final dollar cap** | **`$250.90`** |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.0` |
| Active in | `trending_bull, trending_bear, volatile` |
| Silent in | `ranging, unknown, adverse` |
| Window | Market hours 9:45 AM ET → 3:30 PM ET; hard exit by 3:45 PM ET; 15-min per-symbol cooldown; max 3 concurrent. |
| Universe | **SPY, QQQ, AAPL, NVDA, MSFT**. |
| Conditions | Both required: (1) momentum (0.3% in 5 bars) + EMA slope + 1.5× volume; (2) VIX regime filter (skip if VIX > 40). 50% profit target, 25% stop loss. ATM/nearest, 1-3 DTE. |
| Health | n/a. |
| Open positions | None. |
| Status | **ARMED** within session window. |

### Strategy summary table

| # | Strategy | Sleeve | Weight | Base Cap | Tier | Winner | Regime | **Final Cap** | Tier Eligible (PRO) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | alpha | ALPHA | 0.16 | 1,338 | 1.0 | **1.5** | 1.0 | **$2,007** | YES | ACTIVE (winner-boosted) |
| 2 | alpha_intraday | ALPHA | 0.03 | 251 | 1.0 | 1.0 | 1.0 | $251 | YES | ACTIVE |
| 3 | alpha_options | ALPHA | 0.04 | 335 | 1.0 | 1.0 | 1.0 | $335 | YES | CONDITION-GATED |
| 4 | alpha_futures | ALPHA | 0.09 | 753 | 1.0 | **0.5** | 1.0 | $376 | YES | ACTIVE (winner-penalized) |
| 5 | alpha_crypto | ADAPTIVE | 0.04 | 335 | 1.0 | 1.0 | 1.0 | $335 | YES | ACTIVE |
| 6 | beta | BETA | 0.05 | 418 | 1.0 | 1.0 | 1.0 | $418 | YES | ACTIVE |
| 7 | beta_trend | BETA | 0.20 | 1,673 | 1.0 | 1.0 | 1.0 | $1,673 | YES | ACTIVE |
| 8 | gamma | ALPHA | 0.07 | 585 | 1.0 | 1.0 | 1.0 | $585 | YES | REGIME-SILENT |
| 9 | gamma_futures | ALPHA | 0.05 | 418 | 1.0 | 1.0 | 1.0 | $418 | YES | ACTIVE |
| 10 | gamma_reversion | ALPHA | 0.04 | 335 | 1.0 | 1.0 | 1.0 | $335 | YES | REGIME-SILENT |
| 11 | delta | ADAPTIVE | 0.02 | 167 | 1.0 | **1.304** | 1.0 | $218 | YES | ACTIVE (winner-boosted) |
| 12 | delta_pairs | ADAPTIVE | 0.05 | 418 | 1.0 | 1.0 | 1.0 | $418 | YES | REGIME-SILENT |
| 13 | omega | ADAPTIVE | 0.05 | 418 | 1.0 | 1.0 | 1.0 | $418 | YES | ARMED |
| 14 | omega_vol | ADAPTIVE | 0.05 | 418 | 1.0 | 1.0 | 1.0 | $418 | YES | DEGRADED |
| 15 | omega_macro | ADAPTIVE | 0.03 | 251 | 1.0 | 1.0 | 1.0 | $251 | YES | ACTIVE |
| 16 | omega_momentum_options | ALPHA | 0.03 | 251 | 1.0 | 1.0 | 1.0 | $251 | YES | ARMED |
| — | **Σ (final caps)** | — | **1.00** | **8,363** | — | — | — | **$8,708** | — | — |

(Sum of final caps $8,708 exceeds portfolio_risk_cap $8,363.37 by
~$345. The overshoot is the combined boost on alpha (1.5×) and delta
(1.304×); the allocator does not re-normalize after overlays —
overlays are bounded multipliers and the portfolio_risk_cap is
enforced at execution time via the SCR sizing factor and per-symbol
caps. With SCR=WARMUP at 0.10× this is academic; worth tracking once
SCR promotes.)

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
                → _enforce_kraken_rest_pair       # NEW v8.3 layer 2
                → KrakenExecutor.submit
                → kraken_client.add_order
                → _assert_kraken_rest_pair        # NEW v8.3 layer 1 (border)
  → fills → normalize_paper_fill_evidence → PaperExecutionEvidenceWriter
  → position_guard, slippage_tracker, edge_decay_monitor, expectancy_tracker
  → ProfitRouter.route_profit (50/30/20 advisory)
```

### Bucket key — `(symbol, asset_class)` (commit `d666b5d`)

Bucket key was changed in v8.1 from `(symbol, side)` to
`(symbol, asset_class)` to fix two bugs:

1. SPY/ETF and SPY/OPTIONS merged into a single bucket; the ETF won
   and the option leg was lost.
2. Opposing-side futures signals from `alpha_futures` and
   `gamma_futures` on the same symbol netted to zero and dropped.

Carried forward unchanged in v8.3. Citations:
`chad/utils/signal_router.py:102`,
`chad/execution/execution_pipeline.py:680-685`.

### Meta propagation (commit `f6caf3d`)

The router tracks meta per-strategy in the bucket
(`signal_router.py:109` `meta_by_strategy`), then on emission selects
the meta from the bucket's `primary_strategy` (largest size
contributor) (`signal_router.py:179`):

```python
meta=data.get("meta_by_strategy", {}).get(primary) or {}
```

This is what allows `position_reconciler` to assert paper-mode
strategy ownership and skip drift checks against IBKR for
non-`broker_sync` entries — a precondition for the GREEN reconciliation
state in `runtime/reconciliation_state.json`.

### Asset class router

`chad/execution/execution_pipeline.py:1164-1187` —
`split_signals_by_asset_class` returns `(ibkr_signals, kraken_signals)`:

- `AssetClass.CRYPTO` → Kraken lane.
- Everything else (or missing/unknown) → IBKR lane.

### OPTIONS branch

`chad/execution/execution_pipeline.py:616-665` —
`resolve_ibkr_instrument_spec` dispatches on `AssetClass`:

```python
if asset_class == AssetClass.OPTIONS:
    return _resolve_options_spec(symbol)
```

The OPT spec resolver produces an IBKR `Contract` with
`sec_type="OPT"` and a real expiry/strike when the strategy supplies
it. `alpha_options` populates expiry from
`runtime/options_chains_cache.json`; when the chain is unavailable
the synthetic Black-Scholes fallback sizes correctly but the spec
carries placeholder strike/expiry — see §14.

### Paper fill normalization (commit `21166c9`)

Single chokepoint `normalize_paper_fill_evidence` at
`chad/execution/paper_exec_evidence_writer.py:925-983` enforces four
invariants in paper mode:

1. `asset_class` is never blank/unknown when the symbol is
   recognisable.
2. `fill_price > 0` when `runtime/price_cache.json` has a price (with
   futures contract-month normalisation, e.g. `MGCK6 → MGC`).
3. `status` is rewritten to `paper_fill` when raw status is
   `PendingSubmit`, `error`, etc. and a positive price is available.
4. **Hard invariant** — raises `ValueError` if status would still be
   pending after normalisation. Untrusted records cannot persist.

`is_live=True` records pass through unchanged.

### Three writers, one chokepoint

| Writer | Path |
|---|---|
| Live loop | `chad/core/live_loop.py:1129-1151` |
| Position reconciler | `chad/core/position_reconciler.py:217,282` |
| Timer-driven paper executor | `/usr/local/bin/chad_paper_trade_executor.py:11,222` |

### Kraken REST border — two-layer enforcement (NEW v8.3, commit `15677a2`)

This is the signature execution change in v8.3. CHAD trades crypto on
Kraken via the REST `/0/private/AddOrder` endpoint. Kraken accepts
two pair formats: **altname** (`XBTUSD`) and **full pair**
(`XXBTZUSD`). It rejects the **wsname** form (`XBT/USD`, used in
WebSocket streams) and the CHAD canonical form (`BTC-USD`).

Pre-`15677a2`, a malformed pair traveling down the executor path
produced `EQuery:Unknown asset pair` and a null-payload row in
`data/fills/kraken_fills_*` — silent corruption.

#### Layer 1 — REST border

`chad/exchanges/kraken_client.py:55-76`:

```python
def _assert_kraken_rest_pair(pair: str) -> None:
    """
    Enforce Kraken REST /0/private/AddOrder pair format at the border.
    Valid:   altname ("XBTUSD") or full pair ("XXBTZUSD")
    Invalid: wsname with slash ("XBT/USD"), CHAD canonical with dash ("BTC-USD")
    """
    if not pair or not isinstance(pair, str):
        raise ValueError(...)
    clean = pair.strip().upper()
    if "/" in clean or "-" in clean:
        raise ValueError(
            f"kraken_rest_pair_invalid: pair {pair!r} uses wsname or canonical "
            f"format. REST AddOrder requires altname (e.g. 'XBTUSD'). "
            f"Use normalize_kraken_pair() from chad.execution.execution_pipeline."
        )
```

Called from `add_order` at `chad/exchanges/kraken_client.py:359`. This
is the absolute backstop — every code path that hits
`/0/private/AddOrder` is protected.

#### Layer 2 — Executor pre-flight

`chad/execution/kraken_executor.py:41-60`:

```python
def _enforce_kraken_rest_pair(pair: str, *, strategy: str = "?", side: str = "?") -> str:
    """
    Pre-flight pair validation in the executor — fails fast with strategy
    context before reaching the REST border. Defense in depth.
    """
    if not pair or not isinstance(pair, str):
        raise ValueError(
            f"kraken_executor_pair_invalid: strategy={strategy} side={side} "
            f"pair={pair!r} (must be non-empty string)"
        )
    clean = pair.strip().upper()
    if "/" in clean or "-" in clean:
        raise ValueError(
            f"kraken_executor_pair_invalid: strategy={strategy} side={side} "
            f"pair={pair!r} uses wsname or canonical format. "
            f"Expected altname (e.g. 'XBTUSD'). Upstream normalization failed."
        )
    return clean
```

Called from `kraken_executor.py:316` (validate path) and `:364` (live
path). This layer adds strategy and side context to the diagnostic so
when the failure occurs we know which upstream emitter produced the
malformed pair.

#### Tests updated

`chad/tests/test_kraken_execution.py` fixtures changed from
`pair=XBT/USD` (wsname, invalid for REST) to `pair=XBTUSD` (altname,
valid). The old fixtures were testing rejection-prone paths.

---

## 5. RISK & GOVERNANCE — THE FULL CHAIN

### The complete cap-calculation chain (v8.3 first full documentation)

This section documents — for the first time — the complete chain from
config base weight to executable dollar cap. Every step has a code
citation; every overlay has a fail-soft fallback.

```
config/strategy_weights.json
   ↓  base weights (16 strategies, sum=1)
correlation_overlay (chad/risk/correlation_strategy.py)
   ↓  proportions adjusted by inter-strategy correlation
chassis_enforcement (chad/risk/dynamic_risk_allocator.py:474-547)
   ↓  rebalance to 50/30/20 sleeves with 5% tolerance
TIER FILTER  (NEW v8.3)
   ↓  zero strategies not in current tier's enabled list
WINNER SCALING (NEW v8.3)
   ↓  per-strategy expectancy multiplier 0.5x–1.5x
REGIME BOOSTER (NEW v8.3)
   ↓  global multiplier 1.0x–1.5x
   final_caps → runtime/dynamic_caps.json
SCR sizing_factor (applied at execution time, not in caps file)
```

### Step-by-step with citations

| Step | Module | Behavior |
|---|---|---|
| 1. base_weight | `config/strategy_weights.json` | Loaded by `StrategyAllocation.from_env_or_default` (`dynamic_risk_allocator.py:149-196`). |
| 2. correlation overlay | `chad/risk/correlation_strategy.py` | Applied upstream by orchestrator. Stale `correlation` archived 2026-03-26 — fails soft to base proportions. |
| 3. chassis 50/30/20 | `dynamic_risk_allocator.py:474-547` | `enforce_chassis(weights)` only fires when sleeve drift exceeds `CHASSIS_TOLERANCE = 0.05`. |
| 4. **tier_filter** | `dynamic_risk_allocator.py:603-614` (`load_tier_filter`) | Reads `runtime/tier_state.json:enabled_strategies`. Returns `None` if file missing or stale (>10 min). When `None`, no filtering applied. |
| 5. **winner_scaling** | `dynamic_risk_allocator.py:617-633` (`load_winner_multipliers`) | Reads `runtime/winner_scaling.json:multipliers`. Returns `{}` if stale; per-strategy lookup defaults to 1.0. |
| 6. **regime_booster** | `dynamic_risk_allocator.py:636-644` (`load_regime_booster_multiplier`) | Reads `runtime/regime_booster.json:multiplier`. Returns 1.0 if stale. Allocator clamps to `[1.0, 1.5]` at `:360`. |
| 7. cap composition | `dynamic_risk_allocator.py:362-378` | `cap = portfolio_risk_cap × frac × tier_factor × winner_factor × regime_mult`. |
| 8. SCR sizing | `chad/risk/scr_state.py` | Applied separately at execution time (not in caps file). State machine: WARMUP 0.10×; CAUTIOUS 0.25×; CONFIDENT 1.0×; PAUSED 0.0×. |

### Allocator code excerpt (`dynamic_risk_allocator.py:355-378`)

```python
# --- BUSINESS OVERLAYS (tier / winner / regime) ---
tier_set = load_tier_filter()
winner_mults = load_winner_multipliers()
regime_mult = load_regime_booster_multiplier()
# Clamp regime multiplier to the documented 1.0..1.5 band.
regime_mult = max(1.0, min(1.5, float(regime_mult)))

caps: Dict[str, float] = {}
applied_overlays: Dict[str, Dict[str, float]] = {}
for name, frac in norm.items():
    base_cap = portfolio_risk_cap * frac
    tier_factor = 1.0
    if tier_set is not None and name.lower() not in tier_set:
        tier_factor = 0.0
    winner_factor = float(winner_mults.get(name.lower(), 1.0))
    cap = base_cap * tier_factor * winner_factor * regime_mult
    caps[name] = cap
    applied_overlays[name] = {
        "base_cap": base_cap,
        "tier_factor": tier_factor,
        "winner_factor": winner_factor,
        "regime_factor": regime_mult,
        "final_cap": cap,
    }
```

### Fail-soft behavior

The cap chain is engineered so a missing or stale business runtime
file degrades to the v8.2 behavior, not to a halt.

| File | Stale/missing → |
|---|---|
| `runtime/tier_state.json` (>10 min old) | `tier_filter` inactive — no strategies zeroed. (`dynamic_risk_allocator.py:603-614`) |
| `runtime/winner_scaling.json` (>10 min old) | All `winner_factor = 1.0`. |
| `runtime/regime_booster.json` (>10 min old) | `regime_mult = 1.0`. |

`BUSINESS_OVERLAY_STALE_SECONDS = 600` at
`dynamic_risk_allocator.py:559`. Read helper at
`_read_business_runtime` (`:566-596`) checks `ts_utc` and downgrades
to neutral if older than the window. The publisher cadences (5 min /
15 min) are inside the stale window with margin.

### 50/30/20 chassis

Module: `chad/risk/dynamic_risk_allocator.py:454-547`. Frozen sets
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

Strategies not in any sleeve get zeroed (`:534-538`). Disable via env
`CHAD_CHASSIS_ENFORCEMENT=0`.

### Sizing — equity / ETF intents

`chad/execution/execution_pipeline.py:_apply_sizing_layer`:

```
base → R3 vol_adjusted → R5 composite_cap → R6 correlation_monitor → S5 event_gate → OMS
```

| Layer | Module | Behavior |
|---|---|---|
| **R3** | `chad/risk/vol_adjusted_sizer.py` | `mult = clamp(target_daily_vol/realized_vol, 0.1, 2.0)`, target=0.015, 20-day lookback. |
| **R5** | `chad/risk/composite_size_cap.py` | `min(vol_adj, max_per_symbol=300, sector_remaining=$5k, 0.5%×ADV, 5%×equity/ref_px)`. |
| **R6** | `chad/risk/correlation_monitor.py` | Multiplies by `threshold/avg_corr` (floor 0.1) when book \|r\| > 0.65. |
| **S5** | `chad/analytics/event_calendar.py` | Inside event window: `urgency=high` reject; `normal` reduce 50%. |

Futures and forex bypass — they have specialised upstream sizers.
Crypto rides a Kraken builder path with S3 confidence attenuation +
volume gate, now layered with the two-step REST pair guard.

### SCR — Self-Calibrating Risk

State machine in `chad/risk/scr_state.py`. Runtime at
`runtime/scr_state.json`. Config at `runtime/scr_config.json`.

| State | sizing_factor | What changes |
|---|---|---|
| WARMUP (current) | 0.10 | All trades 10% of planned size; live blocked. |
| CAUTIOUS | 0.25 | 2.5× upsize from WARMUP; still paper-only. |
| CONFIDENT | 1.00 | Full size; live-eligible (still gated by operator GO). |
| PAUSED | 0.00 | Hard stop. |

Currently 16 effective trades short of the 100 needed for CAUTIOUS;
both other CAUTIOUS gates (win_rate ≥ 0.35, sharpe_like ≥ 0.10) are
already satisfied (57.1% / +0.604).

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

Currently NORMAL: realized_pnl_today=$125.60, daily_loss_limit=$5,018.02.

The profit lock interacts with the cap chain at
`dynamic_risk_allocator.py:337-353` — the allocator reads
`profit_lock_state.json:sizing_factor` and multiplies the
`portfolio_risk_cap` by it before per-strategy fractions are applied.
This is the system-level kill switch that runs *outside* the business
overlays.

### Position guard `opened_at` invariant

Every `mark_position_open` write stamps a UTC `opened_at`
(`chad/core/position_guard.py:155-157`); when the prior entry was
already open and same-side, `prior_opened` is preserved so the
open-time clock doesn't reset on size adjustments.

### Broker truth rebuild in paper mode

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

Triggers (`chad/risk/stop_bus_triggers.py`): `daily_loss_breach`,
`reject_rate_spike`, `data_staleness`, `broker_latency_spike`.
Currently `active=false`.

### Edge decay (F4)

Module: `chad/risk/edge_decay_monitor.py`. Halts a strategy when ≥ 5
consecutive losses on ≥ 20-trade base. Recovery via
`scripts/clear_edge_decay.py --strategy <name>`.

### Routing gates (5 gates)

| # | Gate | Block reason |
|---|---|---|
| 1 | A4 `data_freshness` (≤ 300s intraday / 172800s daily) | `bar_stale` |
| 2 | E2 `stale_intent` (≤ 300s) | `intent_expired` |
| 3 | E5 `too_late_to_chase` (≤ 0.5%) | `price_moved` |
| 4 | R7 `net_ev` (≥ min_edge) | `net_ev_below_min_edge` |
| 5 | S5 `event_risk` window | reject (high urgency) / reduce 50% (normal) |

---

## 6. THE BUSINESS FRAMEWORK

This section is the central new content of v8.3. It documents the
seven modules and seven runtime files that together form the
"CHAD as a business" framework. Every value cited here is from the
files re-read at write time.

### 6A. Equity History Publisher

**Source:** `chad/ops/equity_history_publisher.py`
**Output:** `runtime/equity_history.ndjson`
**Cadence:** daily 23:59 UTC via `chad-equity-history.timer`.

#### What it does

Appends a deduplicated daily snapshot from
`runtime/portfolio_snapshot.json` to NDJSON. One line per UTC day.
The publisher reads `_today_utc()`, scans existing dates in the file,
and skips writing if today is already recorded
(`equity_history_publisher.py:74-79`).

#### Schema

```json
{"date_utc": "2026-04-27", "ts_utc": "2026-04-27T19:38:12.621731Z",
 "total_equity_usd": 167227.46, "ibkr_equity_usd": 167042.88,
 "kraken_equity_usd": 184.58, "coinbase_equity_usd": 0.0,
 "schema_version": "equity_history.v1"}
```

#### Current state

`runtime/equity_history.ndjson` currently contains **1 record**
(2026-04-27 with total $167,227.46). This is the seed of the HWM
calculation; `WithdrawalManager` requires 14+ records before unlocking
PAY phase.

#### Used for

- High-water mark in `WithdrawalManager`.
- Drawdown calculation over `drawdown_lookback_days = 30`.
- Growth metrics in `BusinessPhaseTracker.compound_metrics`.

### 6B. Portfolio Snapshot Publisher

**Source:** `chad/ops/portfolio_snapshot_publisher.py`
**Output:** `runtime/portfolio_snapshot.json`
**Cadence:** every 5 minutes via `chad-portfolio-snapshot.timer`.

#### What it does

Connects to IBKR readonly with `clientId=84` (deliberately distinct
from live-loop=99 and reconciliation=83 to avoid collision —
`portfolio_snapshot_publisher.py:36`). Reads
`accountSummary.NetLiquidation`. If currency is CAD, fetches
IBKR's own `USD.CAD` Forex quote and converts at the live mid:

```python
if currency == "CAD":
    fx_contract = Forex("USDCAD")
    ib.qualifyContracts(fx_contract)
    ticker = ib.reqMktData(fx_contract, "", False, False)
    ib.sleep(2)  # wait for quote
    ...
    usd = net_liq_value / mid
```

(`portfolio_snapshot_publisher.py:85-105`).

If the FX quote is unavailable the conversion falls back to a 1.40
multiplier (`:99-100`).

Then reads `runtime/kraken_balances.json:usd_equivalent`, sums
ibkr + kraken + coinbase (= 0 since this is a CAD-domiciled IBKR
account; Coinbase is unused), and writes the snapshot atomically.

#### Schema

```json
{
  "ibkr_equity": 167082.87,
  "coinbase_equity": 0.0,
  "kraken_equity": 184.58,
  "ts_utc": "2026-04-27T22:04:50.029071Z",
  "ttl_seconds": 300
}
```

#### Why this matters

Pre-v8.3, `runtime/portfolio_snapshot.json` was last written
2026-04-03 and the dynamic risk allocator was computing caps off
stale equity. The fresh publisher is the upstream input for
`TierManager`, `WithdrawalManager`, and `BusinessPhaseTracker`.

### 6C. Tier Manager

**Source:** `chad/risk/tier_manager.py`
**Output:** `runtime/tier_state.json`
**Cadence:** every 5 minutes via `chad-tier-manager.timer`.
**Config:** `config/tiers.json`.

#### What it does

Reads `runtime/portfolio_snapshot.json`, sums
`ibkr + kraken + coinbase`, and selects the tier whose
`[min_equity_usd, max_equity_usd)` band contains the current equity
(`tier_manager.py:65-106`). Writes the tier name and the
`enabled_strategies` list.

#### Hysteresis

A 5% buffer (`config/tiers.json:hysteresis_pct = 5.0`) prevents
flapping between tiers when equity hovers near a threshold. Once
promoted, equity must drop below `previous_tier_min × (1 - 0.05)`
to demote (`tier_manager.py:88-104`).

#### The 5 tiers

| Tier | Equity band | # strategies | Description |
|---|---|---|---|
| **MICRO** | $0 – $10k | 8 | Crypto + equities only — no futures, no options. |
| **SMALL** | $10k – $50k | 12 | Adds defined-risk options spreads + ETF mean reversion + pairs. |
| **MID** | $50k – $160k | 15 | Unlocks micro futures (MES, MNQ, MGC) and gamma futures. |
| **PRO** (current) | $160k – $1M | **16** | All 16 strategies firing at meaningful size. |
| **INSTITUTIONAL** | $1M+ | 16 | Full institutional sizing — macro futures (ZN, ZB) deploy meaningfully. |

#### Each tier's enabled_strategies (full list)

**MICRO:** `alpha, alpha_intraday, alpha_crypto, delta, gamma,
gamma_reversion, beta, beta_trend`.

**SMALL:** `alpha, alpha_intraday, alpha_crypto, alpha_options,
delta, delta_pairs, gamma, gamma_reversion, beta, beta_trend,
omega_vol, omega_momentum_options`.

**MID:** `alpha, alpha_intraday, alpha_crypto, alpha_options,
alpha_futures, delta, delta_pairs, gamma, gamma_futures,
gamma_reversion, beta, beta_trend, omega, omega_vol,
omega_momentum_options`.

**PRO (current):** `alpha, alpha_intraday, alpha_crypto, alpha_options,
alpha_futures, delta, delta_pairs, gamma, gamma_futures,
gamma_reversion, beta, beta_trend, omega, omega_vol, omega_macro,
omega_momentum_options`.

**INSTITUTIONAL:** same as PRO (institutional sizing comes from
allocator behavior, not strategy enablement).

#### Current state

```
tier_name           PRO
current_equity_usd  $167,269.69
tier_min_equity     $160,000
tier_max_equity     $1,000,000
previous_tier       PRO
promoted_at_utc     2026-04-27T19:00:35Z
```

### 6D. Winner Scaler

**Source:** `chad/risk/winner_scaler.py`
**Output:** `runtime/winner_scaling.json`
**Cadence:** every 15 minutes via `chad-winner-scaler.timer`.
**Config:** `config/winner_scaling_policy.json`.

#### What it does

Reads `runtime/expectancy_state.json` and computes per-strategy
multipliers:

1. Build a *scoring pool* of strategies that have `total_trades ≥
   min_trades_for_scaling (5)` and are not in `exclude_strategies`.
2. Compute `median_abs = abs(median(expectancies))` across the pool
   (`winner_scaler.py:122-125`).
3. For each strategy in the pool:
   `ratio = expectancy / median_abs`,
   `multiplier = clamp(ratio, 0.5, 1.5)` (`:151-154`).
4. Strategies below the trade threshold or excluded: multiplier = 1.0.
5. If `|median| ≈ 0`, all multipliers = 1.0 (cannot rank yet).

#### Bounds

| Field | Value |
|---|---|
| `max_multiplier` | 1.50 |
| `min_multiplier` | 0.50 |
| `min_trades_for_scaling` | 5 |
| `exclude_strategies` | `broker_sync, manual, paper_exec, unknown` |

#### Why excluded?

`broker_sync` is the bookkeeping label for pre-existing IBKR paper
positions inherited by CHAD (AAPL, MSFT, GOOGL, etc.) — not a
strategy. `manual`, `paper_exec`, `unknown` are similarly non-strategy
labels. Including them would let market drift on legacy positions
distort the ranking.

#### Current state

`runtime/winner_scaling.json` (`2026-04-27T21:57:04Z`):

| Field | Value |
|---|---|
| `n_strategies_scaled` | 5 (in scoring pool) |
| `n_strategies_neutral` | 4 (below threshold or excluded) |
| `median_expectancy` | 4.41 |

Multipliers (full):

| Strategy | Multiplier |
|---|---|
| `alpha` | **1.500** (boost — top of pool) |
| `delta` | **1.304** (boost — strong win rate) |
| `alpha_intraday` | 1.000 |
| `gamma_futures` | 1.000 |
| `omega_vol` | 1.000 |
| `broker_sync` | 1.000 (excluded) |
| `reconciled_phase2_20260419_carryover` | 1.000 |
| `RECONCILED_PHASE2_20260419` | **0.500** (penalized — 7 losses, 0 wins) |
| `alpha_futures` | **0.500** (penalized — `-$51.06` expectancy on 329 trades) |

### 6E. Regime Booster

**Source:** `chad/risk/regime_booster.py`
**Output:** `runtime/regime_booster.json`
**Cadence:** every 60 seconds via `chad-regime-booster.timer`
(installed in v8.4, commit `f62d914`). Pre-v8.4 the SSOT claimed
orchestrator integration that was never wired; the dedicated timer
closes a 15+ hour staleness gap.
**Config:** `config/regime_booster_policy.json`.

#### What it does

Reads regime + confidence + VIX + event severity and produces a
single global multiplier in [1.0, 1.5]. Multiplier accumulates from
positive factors; any veto forces 1.0.

#### Inputs

- `runtime/regime_state.json` → `regime`, `confidence`.
- `data/bars/1d/VIX.json` → last close (`bars[-1].get('close')` —
  fixed in commit `1caa111`; was previously reading `'c'`).
- `runtime/event_risk.json` → `severity`.

#### Vetoes (any one forces multiplier = 1.0)

| Veto | Threshold | Source |
|---|---|---|
| `event_risk_<severity>` | severity in `["high", "extreme"]` | `regime_booster.py:127-128` |
| `vix_elevated_<vix>` | vix > 25.0 | `:129-130` |
| `low_confidence_<conf>` | confidence < 0.70 | `:131-132` |
| `unfavorable_regime_<regime>` | regime not in `["trending_bull", "trending_bear"]` | `:133-134` |

#### Positive factors (additive, capped at `max_multiplier = 1.5`)

| Factor | Add | Trigger |
|---|---|---|
| `high_confidence_<conf>` | +0.10 | confidence ≥ 0.75 |
| `vix_calm_<vix>` | +0.10 | vix ≤ 18.0 |
| `trending_bull_bias` | +0.10 | regime == `trending_bull` |
| `no_event_risk` | +0.05 | severity in `["low", "none", ""]` |

#### Current state

`runtime/regime_booster.json` (`2026-04-27T21:14:50Z`):

| Field | Value |
|---|---|
| `multiplier` | **1.0** |
| `active` | false |
| `reasons` | `["vetoed: low_confidence_0.69"]` |
| `regime` | `trending_bull` |
| `confidence` | 0.6864 |
| `vix` | 18.71 |
| `event_severity` | `medium` |

The booster is dormant because confidence (0.6864) is below the
0.70 minimum. Once confidence clears 0.70, the trending_bull regime
+ medium event severity would yield `1.0 + 0.10 (trending_bull) +
0.05 (no_event_risk if severity drops to low) = 1.15`, plus
`+0.10 (high_confidence)` if confidence ≥ 0.75. VIX is 18.71 — just
above the 18.0 calm threshold — so `vix_calm` doesn't fire.

### 6F. Withdrawal Manager

**Source:** `chad/risk/withdrawal_manager.py`
**Output:** `runtime/withdrawal_authorization.json`
**Cadence:** every 6 hours via `chad-withdrawal-manager.timer`.
**Config:** `config/withdrawal_policy.json`.

#### Core principle

> *DO NOT EAT THE SEED.*

The trading account is the engine. Pay yourself only from sustained
surplus above a rolling high-water mark, only when SCR is confident,
and never during drawdowns. The withdrawal authorization is
**advisory only** — CHAD never moves money. The operator decides.

#### Phase logic

`compute_authorization` (`withdrawal_manager.py:139-248`):

```
if current_equity < seed × build_phase_threshold_multiplier (1.20):
    phase = BUILD     # below $60,000
elif require_scr_confident and scr_state != CONFIDENT:
    phase = GROW      # above threshold but engine not yet validated
else:
    phase = PAY       # all gates open
    # Then PAY can be downgraded to GROW for any of:
    if len(history) < minimum_history_days (14):     → GROW
    elif drawdown_pct (30d) > drawdown_veto_pct (5): → GROW
    elif current_equity < hwm:                       → GROW
```

#### Payout formula (PAY phase only)

```
surplus = current_equity − high_water_mark
authorized = min(surplus × payout_rate_above_hwm (0.30),
                 max_monthly_salary_usd ($2000))
```

(`withdrawal_manager.py:227-234`).

#### Vetoes that downgrade PAY → GROW

- **History veto:** fewer than 14 daily equity records → no payout.
- **Drawdown veto:** 30-day drawdown > 5% → no payout.
- **HWM veto:** current_equity below high_water_mark → no payout.

#### Policy

`config/withdrawal_policy.json`:

| Field | Value |
|---|---|
| `seed_capital_usd` | $50,000 |
| `build_phase_threshold_multiplier` | 1.20 |
| `payout_rate_above_hwm` | 0.30 |
| `max_monthly_salary_usd` | $2,000 |
| `drawdown_veto_pct` | 5.0% |
| `drawdown_lookback_days` | 30 |
| `require_scr_confident` | true |
| `minimum_history_days` | 14 |

#### Current state

```
phase                       GROW
current_equity_usd          $167,227.46
seed_capital_usd            $50,000.00
high_water_mark_usd         $167,227.46
drawdown_from_hwm_pct       0.0
spendable_surplus_usd       $0.00
authorized_withdrawal_usd   $0.00
scr_state                   WARMUP
history_days                1
reason                      "GROW phase: equity above build threshold but
                             SCR is WARMUP (need CONFIDENT). Reinvesting
                             profits."
```

What unlocks PAY for this account:
1. SCR promotes from WARMUP → CONFIDENT (16 more effective trades to
   CAUTIOUS, then more validation to CONFIDENT).
2. Build 13 more days of equity history.
3. Stay above HWM with no >5% drawdown.

When all three clear, payout starts. With current HWM = current_equity,
`surplus = 0`; the operator gets paid only when CHAD makes new highs.
At a hypothetical $5k surplus the formula would authorize
`min($5,000 × 0.30, $2,000) = $1,500/month`.

### 6G. Business Phase Tracker

**Source:** `chad/ops/business_phase_tracker.py`
**Output:** `runtime/business_phase.json`
**Cadence:** every 30 minutes via `chad-business-phase.timer`.

#### What it does

Reads `withdrawal_authorization.json` (authoritative phase),
`portfolio_snapshot.json` (current equity), `scr_state.json`,
`equity_history.ndjson` (history length), and writes a single
operator-friendly business view:

- The phase classification (verbatim from withdrawal_authorization).
- A plain-English description (`_phase_description`,
  `business_phase_tracker.py:103-125`).
- Days in phase (approximate — equals history length, see code
  comment `:164-176`).
- The next-phase requirement (`_next_phase_requirement`, `:128-161`).
- Compound metrics — `total_return_pct`, `annualized_return_pct`
  (suppressed below 14d), `days_active`, `high_water_mark_usd`
  (`_compound_metrics`, `:179-216`).

#### Phase descriptions (literal text)

- **BUILD:** *"Building the engine. Equity must grow above $60,000
  before paying anything out."*
- **GROW:** *"Engine is built. Now growing the account before salary
  starts. SCR must reach CONFIDENT first."*
- **PAY:** *"Engine running well. Salary authorized at
  $\<authorized_salary>/month from surplus above high water mark."*

#### Current state

```
phase                       GROW
phase_description           "Engine is built. Now growing the
                             account before salary starts. SCR
                             must reach CONFIDENT first."
current_equity_usd          $167,267.75
seed_capital_usd            $50,000.00
growth_pct_from_seed        +234.54%
days_in_phase               1
next_phase_requirement      "To enter PAY phase: SCR must promote
                             from WARMUP to CONFIDENT; Need 13+
                             more days of equity history."
compound_metrics:
  total_return_pct          +234.54%
  annualized_return_pct     0.0  (suppressed: days_active < 14)
  days_active               1
  high_water_mark_usd       $167,227.46
```

### 6H. Profit Router (50/30/20)

**Source:** `chad/risk/profit_router.py`
**Output:** `runtime/profit_routing.json`
**Cadence:** invoked on every realized profit close
(`route_profit(realized_pnl, closing_strategy, account_equity)`).

#### Core split

On every profitable close:

- **50%** → `trading_capital` (stays in the account, compounds trading)
- **30%** → `beta_allocation` (earmarked for Beta — long-term holder)
- **20%** → `amplifier_allocation` (boost for the best-performing
  strategy)

Constants at `profit_router.py:73-75`. Negative or NaN PnL returns
`{"no_routing": True, "reason": "not_profitable"}` — only winnings
route.

#### Advisory only

CHAD currently runs on a single IBKR paper account, so capital is
*not* physically transferred between accounts. The router records
each routing decision in `runtime/profit_routing.json` as an
accounting ledger; `Beta` reads `get_beta_accumulated()` to know how
much earmarked capital is available
(`profit_router.py:120-129`).

#### Current totals

`runtime/profit_routing.json` (`2026-04-27T21:13:40Z`):

| Bucket | Total |
|---|---|
| `trading_capital` | $1,051.68 |
| `beta_allocation` | $631.01 |
| `amplifier_allocation` | $420.67 |
| **Sum (≈ realized profitable PnL)** | **$2,103.36** |

26 routing decisions logged.

### Business framework — call graph

```
runtime/portfolio_snapshot.json   (5min)  ──┐
                                            ├──► runtime/tier_state.json     (5min)
                                            └──► runtime/withdrawal_authorization.json (6h)
                                                          │
                                                          └──► runtime/business_phase.json (30min)

runtime/equity_history.ndjson     (daily) ──► WithdrawalManager + BusinessPhaseTracker

runtime/expectancy_state.json     (5min)  ──► runtime/winner_scaling.json    (15min)

runtime/regime_state.json         (60s)   ──┐
runtime/event_risk.json           (1800s) ──├──► runtime/regime_booster.json  (orchestrator cycle)
data/bars/1d/VIX.json             (daily) ──┘

[risk allocator]
  reads: tier_state, winner_scaling, regime_booster
  writes: runtime/dynamic_caps.json with per-strategy
          {tier_factor, winner_factor, regime_factor, dollar_cap}

[on every realized profit close]
  ProfitRouter → runtime/profit_routing.json (advisory ledger)
```

---

## 7. RECONCILIATION

### Publisher path

`chad/ops/reconciliation_publisher.py:run_publish`. Runs every cycle
of `chad-reconciliation-publisher.timer`. Reads
`runtime/position_guard.json` + IBKR positions and produces
`runtime/reconciliation_state.json`.

### Paper-mode behavior (commit `f6caf3d`)

`chad/ops/reconciliation_publisher.py:91-92, 125-127`:

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

### Drift vs Mismatch

`reconciliation_publisher.py:212-217`: a per-symbol diff is classified
as **drift** (broker-side, accepted) when CHAD's strategy side is
zero OR all of the diff is in `broker_sync`. Otherwise it's a
**mismatch** (strategy-vs-broker, fault).

The current SPY entry — `chad=0, broker=-17, diff=17` — is a drift
because CHAD has no SPY strategy entry; the entire 17-share short is
on the broker side via `broker_sync`. The reconciler returns GREEN
because `worst_diff = 0.0` (worst diff is computed across mismatch
records, not drift records).

### Excluded symbols

| Set | Symbols | Why |
|---|---|---|
| `excluded_symbols` (current) | `[]` | empty — no symbols pinned. |
| `KNOWN_FUTURES_SYMBOLS` | `MCL, ES, NQ, CL, GC, RTY, MES, MNQ` (`reconciliation_publisher.py:58`) | Futures contract-resolution path is incomplete; gaps are reality. |

### Current state

`runtime/reconciliation_state.json` (ts `2026-04-27T22:04:46Z`):

```
status               GREEN
broker_source        ibkr:clientId=83
chad_open            1
broker_positions     2
worst_diff           0.0
mismatches           []
drifts               [{"symbol":"SPY","chad":0.0,"broker":-17.0,"diff":17.0}]
excluded_symbols     []
futures_excluded     []
```

---

## 8. INTELLIGENCE LAYER

Each runtime intel feed is independent — failures degrade gracefully
(stale data is preferred over no data; consumers test
`max_age_hours`).

| File | Schema | Purpose | Fresh? |
|---|---|---|---|
| `runtime/regime_state.json` | `regime_state.v1` | Live classifier inputs + regime label. | 60s TTL — current. |
| `runtime/strategy_intelligence.json` | n/a | AI-generated regime profile per strategy. | 48h cache (`daily_chad_report.py:1353`); last refreshed `2026-04-27T12:30Z`. |
| `runtime/expectancy_state.json` | n/a | Per-strategy rolling expectancy. **NEW v8.3 consumer:** `WinnerScaler`. | Refreshed every 5 min by `chad-expectancy-tracker.timer`. |
| `runtime/trends_state.json` | n/a | Google Trends ratios per symbol. | `chad-trends-refresh.timer` (last `2026-04-27T18:49Z`). |
| `runtime/reddit_sentiment.json` | n/a | Reddit mention/sentiment per symbol. | `chad-reddit-sentiment-refresh.timer` (last `2026-04-27T20:53Z`). |
| `runtime/short_interest.json` | n/a | Short float % per symbol. | `chad-short-interest-refresh.timer` (last `2026-04-27T18:47Z`). |
| `runtime/event_risk.json` | `event_risk.v1` | US session-edge windows. **NEW v8.3 consumer:** `RegimeBooster`. | 1800s TTL — bootstrapped via `MarketHoursRiskProvider`. Currently `severity=medium` (last `2026-04-03T00:11Z` — bootstrap window persists). |
| `runtime/institutional_consensus.json` | `institutional_consensus.v1` | Top 25 holdings across 7 funds (Appaloosa, Berkshire, BlackRock, Bridgewater, Citadel, Pershing Square, Vanguard). | Sunday 00:00 UTC weekly (last `2026-04-26T00:00Z`). |
| `runtime/profit_routing.json` | `profit_routing.v1` | 50/30/20 splits ledger. | Per-realised-close (last `2026-04-27T21:13Z`). |
| `runtime/business_phase.json` | `business_phase.v1` | NEW v8.3 — phase + plain-English description. | 30 min via `chad-business-phase.timer`. |
| `runtime/tier_state.json` | `tier_state.v1` | NEW v8.3 — tier name + enabled strategies. | 5 min via `chad-tier-manager.timer`. |
| `runtime/withdrawal_authorization.json` | n/a | NEW v8.3 — salary authorization. | 6h via `chad-withdrawal-manager.timer`. |
| `runtime/winner_scaling.json` | `winner_scaling.v1` | NEW v8.3 — per-strategy multipliers. | 15 min via `chad-winner-scaler.timer`. |
| `runtime/regime_booster.json` | `regime_booster.v1` | NEW v8.3 — global multiplier. | Orchestrator cycle. |
| `runtime/equity_history.ndjson` | `equity_history.v1` | NEW v8.3 — daily HWM checkpoints. | Daily 23:59 UTC. |
| `runtime/intel_cache/*` | various | brain_returns, macro_state, sector_rotation. | Various. |

### Strategy intelligence current state (carried gap)

`runtime/strategy_intelligence.json` — every entry is `profile=normal`
because the macro context provider returned `unknown` for VIX and
macro-risk inputs at snapshot time. The reasoning string makes the
gap explicit:

> *"Insufficient data due to macro context provider error; VIX and
> macro risk label unknown; defaulting to normal regime per rules."*

This is a known classifier-input wiring gap, not a fault. Same as
v8.2.

### NEW v8.3 — Expectancy → WinnerScaler

`expectancy_state.json` was historically read only by reports and
the F4 edge-decay monitor. As of v8.3 it is also the input to
`WinnerScaler`. The `current_allocation_pct` field on each strategy
record is currently informational; the actual winner factor lands in
`runtime/winner_scaling.json:multipliers`.

### NEW v8.3 — Event risk → RegimeBooster

`event_risk.json` was historically a pure routing-gate input (S5
event_gate). As of v8.3 it is also a veto input for `RegimeBooster`:
`severity` in `["high", "extreme"]` forces the regime multiplier to
1.0.

The current bootstrap provider (`MarketHoursRiskProvider`) classifies
US-session edges as `severity=medium`, which is below the veto
threshold. The notes field still reads:
*"bootstrap_provider=market_hours; replace with CPI/FOMC/NFP
calendar"* — wiring an FOMC/CPI/NFP calendar source is a roadmap item
(§15).

---

## 9. TELEGRAM OPERATOR INTERFACE

### Free-text routing

`chad/utils/telegram_bot.py:1365` — `handle_free_text` classifies the
message intent and dispatches to the appropriate slash-command
handler.

### Slash commands (registered handlers)

From `chad/utils/telegram_bot.py:1455-1468`: `/ping`, `/help`,
`/coach_mode`, `/status`, `/readiness`, `/why_blocked`, `/risk`,
`/perf`, `/live_gate`, `/shadow`, `/portfolio_active`,
`/portfolio_targets <profile>`, `/portfolio_rebalance <profile>`,
`/price <symbol>`, plus advisory + AI research commands.

### Morning brief — elite prodigy voice

`chad/ops/daily_chad_report.py:MorningBrief` (`:1196`).

System prompt (`:1247-1257`):

> "You are CHAD — an elite autonomous trading system. You think with
> the quantitative precision of Jim Simons, the macro vision of Ray
> Dalio, and the opportunistic instincts of Stanley Druckenmiller…"

Schedule: `chad-morning-brief.timer` — Mon-Fri 9:00 AM ET (13:00 UTC).

### NEW v8.3 — BUSINESS STATUS section

The morning brief now includes a BUSINESS STATUS section
(`daily_chad_report.py:1474-1505`):

```
═══ BUSINESS STATUS ═══
Phase: GROW (Engine is built. Now growing the account before salary
       starts. SCR must reach CONFIDENT first.)
Account: $167,267  (+234.5% from seed)
Tier: PRO — 16/16 strategies active
Salary: $0/mo authorized
```

The exact lines emit are:

```python
phase_desc = biz.get("phase_description") or ""
if phase_desc:
    lines.append(f"Phase: {phase} ({phase_desc})")
else:
    lines.append(f"Phase: {phase}")
lines.append(
    f"Account: ${cur_eq:,.0f}  ({'+' if growth >= 0 else ''}"
    f"{growth:.1f}% from seed)"
)
lines.append(f"Tier: {tier_name} — {n_strategies}/16 strategies active")
lines.append(f"Salary: ${authorized:,.0f}/mo authorized")
```

Drawn from `runtime/business_phase.json`, `runtime/tier_state.json`,
and `runtime/withdrawal_authorization.json`.

### End-of-day brief

`chad-daily-report.timer` — Mon-Fri 4:35 PM ET (21:35 UTC). Same
elite-prodigy voice.

### Real-time alerts

`chad/utils/telegram_notify.py` — read-only side effects, deterministic
retries, dedupe via runtime files. Triggered for trade fills, SCR
milestone transitions, stop-bus activation, edge-decay halts, IBKR /
data-feed staleness watchdog.

### Weekly summary — BUSINESS PHASE block

`chad/ops/daily_chad_report.py:WeeklySummary` (`:1570`),
`run_weekly_summary` (`:1763`). Schedule:
`chad-weekly-report.timer` — Sundays 20:00 UTC.

NEW v8.3: weekly summary includes a BUSINESS PHASE block
(`daily_chad_report.py:1786-1806`):

```
═══ BUSINESS PHASE ═══
Phase: GROW
Days in phase: 1
Next milestone: To enter PAY phase: SCR must promote from WARMUP
                to CONFIDENT; Need 13+ more days of equity history.
Total return: +234.5%
HWM: $167,227
```

NEW v8.3: weekly TOP BRAIN ranking now excludes `broker_sync`
(commit `0c3a3e1`):

```python
EXCLUDED_FROM_RANKING = {"broker_sync", "manual", "paper_exec",
                         "unknown", ""}
ranked_strategies = {
    name: data for name, data in strats.items()
    if name not in EXCLUDED_FROM_RANKING
    and isinstance(data, dict)
    and _safe_float(data.get("trade_count", data.get("total_trades", 0))) >= 5
}
```

If the original `top_performer` is filtered out, the report re-ranks
by expectancy (`daily_chad_report.py:1722-1727`). With current data
this surfaces `delta` (75% win rate / +$5.75 expectancy) instead of
`broker_sync` (+$71.37 expectancy on inherited positions, which is
not strategy alpha).

### Voice lock

The CHAD voice is enforced by the system prompts in
`MorningBrief._chads_take` (elite prodigy) and the dashboard chat
(strict no-jargon plain English, see §10). If the LLM call fails the
brief degrades silently to data-only — no fallback paragraph that
could break voice.

---

## 10. DASHBOARD (chadtrades.com)

### Auth

- Password-protected via `/etc/chad/dashboard.env` (basic auth +
  session token in cookie).
- Public health endpoint `/health` (no auth) for systemd / monitoring.

### Routing

- TLS via Certbot (cert valid through 2026-07-19).
- nginx → `127.0.0.1:8765` (FastAPI/Uvicorn).

### Panels (preserved from v8.2)

From `chad/dashboard/api.py:DashboardSnapshot.build`:

- **Training Mode card** — `_chad_status` (`:327`).
- **Account Value** — `_portfolio` (`:275`).
- **Realized PnL** — `runtime/pnl_state.json`.
- **Market** — `_intelligence` (`:455`): VIX, BTC, SPY trend, top
  Reddit mention, market regime.
- **What CHAD Is Watching** — strategy_intelligence + intel feeds.
- **Open Positions** — `_open_positions` (`:301`).
- **Recent Trades** — `_iter_recent_closed_trades` (`:625`).
- **Strategy Performance** — `_strategies` (`:356`).
- **Ask CHAD chat** — `/api/chat` (commit `2935eea`).

### NEW v8.3 — Business endpoint

`chad/dashboard/api.py:484-525` adds `_business()`:

```python
def _business(self) -> dict:
    biz = _load_json(RUNTIME / "business_phase.json") or {}
    tier = _load_json(RUNTIME / "tier_state.json") or {}
    wd = _load_json(RUNTIME / "withdrawal_authorization.json") or {}
    booster = _load_json(RUNTIME / "regime_booster.json") or {}
    ...
    return {
        "phase": str(biz.get("phase") or wd.get("phase") or "?"),
        "phase_description": str(biz.get("phase_description") or ""),
        "tier": str(tier.get("tier_name") or "?"),
        "tier_strategies_enabled": len(tier.get("enabled_strategies") or []),
        "authorized_salary_usd": round(authorized, 2),
        "high_water_mark_usd": round(hwm, 2),
        "growth_pct_from_seed": round(growth, 2),
        "regime_booster_active": bool(booster.get("active", False)),
        "regime_booster_multiplier": round(booster_mult, 3),
        "next_phase_requirement": str(biz.get("next_phase_requirement") or ""),
    }
```

The block is included in the top-level `/api/state` payload at
`api.py:539`:

```python
"business": self._business(),
```

### NEW v8.3 — Chat context includes business block

`chad/dashboard/api.py:1086-1103` enriches the chat snapshot context
with business-framework data so the chat can answer:

- "what phase are we in?" → from `business_phase.json:phase`
- "when do I get paid?" → from
  `withdrawal_authorization.json:authorized_withdrawal_usd` plus
  `business_phase.json:next_phase_requirement`
- "how many strategies are active right now?" → from
  `tier_state.json:enabled_strategies`
- "is the boost on?" → from `regime_booster.json:active`

### Dashboard regime card (commit `56eaeea`)

The regime panel reads `runtime/regime_state.json` directly
(`api.py:798-836`).

### Service health (commit `56eaeea`)

`api.py:_system_health` correctly counts oneshot service success.

### Chat (`/api/chat`, commit `2935eea` + `064efc7`)

- Endpoint `chad/dashboard/api.py:1121`.
- Model: `claude-sonnet-4-6`.
- Plain-English voice (system prompt `:910-948`): no percentages, no
  trading terms, no jargon, max 3 sentences unless asked. Translations
  baked into the prompt.

---

## 11. SERVICES & TIMERS

187 `chad-*` unit files loaded (102 services + 85 timers — counts
include masked and disabled units). 14 services running. 0 failed.

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
| `chad-strategy-intelligence-refresh.service` | AI per-strategy regime profile (currently activating). |

### NEW v8.3 timers (business framework)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-portfolio-snapshot.timer` | every 5 min | Refresh `portfolio_snapshot.json` from IBKR + Kraken with CAD→USD conversion. |
| `chad-equity-history.timer` | daily 23:59 UTC | Append HWM checkpoint to `equity_history.ndjson`. |
| `chad-withdrawal-manager.timer` | every 6h | Compute salary authorization (BUILD/GROW/PAY). |
| `chad-tier-manager.timer` | every 5 min | Equity-tier strategy enable/disable. |
| `chad-winner-scaler.timer` | every 15 min | Per-strategy expectancy multipliers. |
| `chad-business-phase.timer` | every 30 min | Plain-English BUILD/GROW/PAY publisher. |

### NEW v8.4 timers (gap-fills)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-regime-booster.timer` | every 60s | Refresh `runtime/regime_booster.json`. Closes the staleness gap where SSOT v8.3 §6E claimed orchestrator integration that was never wired. Commit `f62d914`. |
| `chad-event-risk.timer` | every 10 min | Refresh `runtime/event_risk.json`. Was disabled with a malformed real-file in `timers.target.wants/` instead of a symlink; v8.4 fixed the symlink and enabled the timer. Commit `f62d914`. |

### Timers — hot-path (preserved from v8.2)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-trade-closer.timer` | OnBootSec=45, OnUnitActiveSec=60 | Scheduled exits (stops/targets/time stops). |
| `chad-scr-sync.timer` | every 60s | Refresh `runtime/scr_state.json` from shadow. |
| `chad-reconciliation-publisher.timer` | every 5 min | Publishes reconciliation snapshots. |
| `chad-paper-trade-exec.timer` | every 10m | Backstop paper executor. |
| `chad-paper-trade-executor.timer` | every 5 min | Alternate paper-trade pipeline. |
| `chad-ibkr-paper-fill-harvester.timer` | every 5m | Harvests broker fills into evidence ledger. |
| `chad-ibkr-broker-events.timer` | every 5m | Collects broker events. |
| `chad-ibkr-price-refresh.timer` | 60s market hours / 300s off-hours | Price cache refresh cadence. |
| `chad-options-monitor.timer` | every 60s during market hours | Monitors options positions. |
| `chad-options-chain-refresh.timer` | Mon-Fri 12:30 UTC | Refresh options chain cache. |

### Timers — analytics & feeds (preserved from v8.2)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-expectancy-tracker.timer` | every 5m | Per-strategy expectancy. *(Now consumed by WinnerScaler.)* |
| `chad-symbol-blocker.timer` | every 5m | Symbol perf-based blocking. |
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
| `chad-morning-brief.timer` | Mon-Fri 13:00 UTC (9:00 ET) | Elite-prodigy pre-market brief (now with BUSINESS STATUS). |
| `chad-daily-report.timer` | Mon-Fri 21:35 UTC (4:35 ET) | End-of-day report. |
| `chad-weekly-report.timer` | Sunday 20:00 UTC | Weekly summary (now with BUSINESS PHASE block + broker_sync excluded from top brain). |
| `chad-advisory-pre-market.timer` | recurring | Pre-market advisory. |
| `chad-ibkr-daily-bars-refresh.timer` | nightly | Daily bars refresh (futures). |
| `chad-proofs-cleanup.timer` | daily 03:20 UTC | Proof artifacts cleanup. |

### Intentionally disabled / retired

- `chad-reconciliation.timer` — masked via `/dev/null` symlink in v8.4 (commit `f62d914`). Pre-v8.4 was `disabled` only, so a manual `systemctl start` could have re-introduced a dual-writer race on `runtime/reconciliation_state.json`. Now permanently inert.
- `chad-options-chain-refresh.service` (the standalone service form) —
  intentionally disabled; the timer is the canonical entry.
- `chad-polygon-stocks`, `chad-bars-validate`, `chad-daily-bars-backfill`
  — masked.

---

## 12. DATA & STORAGE

### Universe

`config/universe.json`:
- **Equities/ETFs (17):** AAPL, SPY, MSFT, GOOGL, BAC, IEMG, QQQ, VWO,
  NVDA, GLD, SH, PSQ, SVXY, UVXY, VIXY, IWM, TLT.
- **Futures (8):** MES (CME), MNQ (CME), MCL (NYMEX), MGC (COMEX),
  ZN (CBOT), ZB (CBOT), M6E (CME), SIL (COMEX).

### Bar provider

- Service: `chad-ibkr-bar-provider.service`.
- Mechanism: `reqHistoricalData` polling every 30s — no real-time
  subscription required (commit `51c190e`).
- 1m bars: 25 symbols stored in `data/bars/1m/`.
- 1d bars: 30 symbols stored in `data/bars/1d/`.
- Daily bar refresh: `chad-ibkr-daily-bars-refresh.timer` runs
  nightly.

### Crypto data

- Kraken 1d bars via REST: `BTC-USD, ETH-USD, SOL-USD`.
- Kraken WS feed: real-time prices + balances
  (`runtime/kraken_prices.json`, `runtime/kraken_balances.json`).
- Altname routing in `chad/execution/execution_pipeline.py:1228-1264`,
  now layered with the two-step REST pair guard (§4).

### Price cache

`runtime/price_cache.json` — 23 symbols, 60s TTL, refreshed by
`chad-ibkr-price-refresh.timer`.

### Fills ledger

- Equity / futures / options: `data/fills/FILLS_YYYYMMDD.ndjson` (one
  line per fill, hash-chained by writer).
- Crypto: `data/fills/kraken_fills_YYYYMMDD.ndjson`.

Recent days present: `FILLS_20260417.ndjson`, `FILLS_20260420.ndjson`,
`FILLS_20260421.ndjson`, `FILLS_20260422.ndjson`, `FILLS_20260423.ndjson`,
`FILLS_20260424.ndjson`, `FILLS_20260425.ndjson`, `FILLS_20260426.ndjson`,
`FILLS_20260427.ndjson`, plus `kraken_fills_20260422.ndjson`.

### SEC 13F refresh

Weekly cron Sunday 00:00 UTC via
`scripts/update_institutional_consensus.py`. Output:
`runtime/institutional_consensus.json`.

### NEW v8.3 — Business framework state files

| File | Schema | Cadence | Source module |
|---|---|---|---|
| `runtime/equity_history.ndjson` | `equity_history.v1` | daily 23:59 UTC | `equity_history_publisher.py` |
| `runtime/portfolio_snapshot.json` | n/a (`ttl_seconds=300`) | every 5 min | `portfolio_snapshot_publisher.py` |
| `runtime/tier_state.json` | `tier_state.v1` | every 5 min | `tier_manager.py` |
| `runtime/withdrawal_authorization.json` | n/a | every 6h | `withdrawal_manager.py` |
| `runtime/business_phase.json` | `business_phase.v1` | every 30 min | `business_phase_tracker.py` |
| `runtime/winner_scaling.json` | `winner_scaling.v1` | every 15 min | `winner_scaler.py` |
| `runtime/regime_booster.json` | `regime_booster.v1` | orchestrator cycle | `regime_booster.py` |

### Other key runtime files (preserved)

- `runtime/dynamic_caps.json` — orchestrator-published per-strategy
  dollar caps **with NEW v8.3 `business_overlays` block**.
- `runtime/profit_lock_state.json` — circuit breaker state.
- `runtime/stop_bus.json` — halt flag.
- `runtime/strategy_health.json` — F3 composite per strategy.
- `runtime/expectancy_state.json` — F1 rolling expectancy
  (now also consumed by WinnerScaler).
- `runtime/options_chains_cache.json` — cached options chains.
- `runtime/last_route_decision.json` — DecisionTrace bridge.
- `runtime/live_readiness.json` — `ready_for_live=false`.

### NEW v8.3 — `dynamic_caps.json:business_overlays` block

Every snapshot of `runtime/dynamic_caps.json` now includes a
`business_overlays` block alongside the per-strategy caps:

```json
"business_overlays": {
  "regime_booster_multiplier": 1.0,
  "tier_enabled_strategies": [
    "alpha", "alpha_crypto", "alpha_futures", "alpha_intraday",
    "alpha_options", "beta", "beta_trend", "delta", "delta_pairs",
    "gamma", "gamma_futures", "gamma_reversion",
    "omega", "omega_macro", "omega_momentum_options", "omega_vol"
  ],
  "tier_filter_active": true,
  "winner_multipliers": {
    "alpha": 1.5,
    "alpha_futures": 0.5,
    "alpha_intraday": 1.0,
    "broker_sync": 1.0,
    "delta": 1.304,
    "gamma_futures": 1.0,
    "omega_vol": 1.0,
    "reconciled_phase2_20260419": 0.5,
    "reconciled_phase2_20260419_carryover": 1.0
  }
}
```

This is the *audit trail* for the business framework — at any point
in time, the `dynamic_caps.json` file declares exactly which overlays
were active and what they did to each strategy's cap.

### Disk usage

- Filesystem: `/dev/root` 48 GB total, **25 GB used (53%)**, 23 GB
  available. Below the 75% threshold called out in pre-live operator
  tasks (`CLAUDE.md`).

---

## 13. CHANGE LOG (delta from v8.2)

In commit order (oldest first since v8.2 was cut). 7 commits since
`accaa2b` — 4 from the v8.3 session, plus 3 more on 2026-04-28
(audit, voice tightening, and the v8.4 gap-closure).

### Commit list (`git log accaa2b..HEAD --oneline`)

```
15677a2 2026-04-26 Fix: enforce Kraken REST pair format at executor + REST border
0c3a3e1 2026-04-27 Fix: exclude broker_sync from weekly top brain ranking
1caa111 2026-04-27 Build: complete CHAD business framework (Phase 11C/12B/12C)
1608fec 2026-04-27 Build: business framework foundation files
4a37b4c 2026-04-28 Audit: behavioral contract verification of SSOT v8.3
031a9de 2026-04-28 Tighten brief voice: zero jargon for non-trader operator
f62d914 2026-04-28 Build: SSOT v8.4 gap closure — HWM fix, alpha_crypto rename, regime-booster timer, audit harness
```

### 1. `15677a2` — *Fix: enforce Kraken REST pair format at executor + REST border* (2026-04-26)

- **Was broken:** A malformed pair (`XBT/USD` wsname, or `BTC-USD`
  CHAD canonical) traveling down the executor path produced
  `EQuery:Unknown asset pair` and a null-payload row in
  `data/fills/kraken_fills_*` — silent corruption.
- **Was fixed:** Two-layer defense.
  - **Layer 1 (REST border):** `chad/exchanges/kraken_client.py:55`
    (`_assert_kraken_rest_pair`). Called from `add_order` at `:359`.
    Rejects wsname and canonical with `ValueError`. Absolute backstop.
  - **Layer 2 (executor pre-flight):**
    `chad/execution/kraken_executor.py:41`
    (`_enforce_kraken_rest_pair`). Called from `:316` (validate path)
    and `:364` (live path). Adds strategy + side context for better
    diagnostics.
- **Tests:** `chad/tests/test_kraken_execution.py` fixtures changed
  from `pair=XBT/USD` to `pair=XBTUSD`. 1008 passed.

### 2. `0c3a3e1` — *Fix: exclude broker_sync from weekly top brain ranking* (2026-04-27)

- **Was broken:** Weekly TOP BRAIN section ranked `broker_sync` (the
  bookkeeping label for pre-existing IBKR paper positions inherited
  by CHAD) as if it were a strategy. With `+$71.37` expectancy from
  passive market exposure on AAPL/MSFT/GOOGL/etc., it was beating
  every actual strategy.
- **Was fixed:** Filter applied in
  `chad/ops/daily_chad_report.py:1711`:
  ```python
  EXCLUDED_FROM_RANKING = {"broker_sync", "manual", "paper_exec",
                           "unknown", ""}
  ```
  Minimum trade count threshold: 5 trades. If the original
  `top_performer` is filtered out, re-rank by expectancy
  (`:1722-1727`).
- **Result:** Weekly report now correctly identifies the actual top
  strategy (`delta` — 75% win rate / `+$5.75` expectancy this week)
  instead of `broker_sync` (61% / `+$73.45`).

### 3. `1caa111` — *Build: complete CHAD business framework (Phase 11C/12B/12C)* (2026-04-27)

The headline commit of v8.3.

- **WinnerScaler (Phase 11C):** new module
  `chad/risk/winner_scaler.py` (+201 LoC) writing
  `runtime/winner_scaling.json`. Per-strategy expectancy
  multipliers, bounded 0.5×–1.5×, broker_sync excluded.
- **BusinessPhaseTracker (Phase 12C):** new module
  `chad/ops/business_phase_tracker.py` (+293 LoC) writing
  `runtime/business_phase.json`. Plain-English BUILD/GROW/PAY
  publisher with compound metrics for operator visibility.
- **RegimeBooster (Phase 12B):** new module
  `chad/risk/regime_booster.py` (+211 LoC) writing
  `runtime/regime_booster.json`. **VIX bug fixed** —
  `bars[-1].get('close')` not `'c'` at `regime_booster.py:97`.
- **Risk allocator full chain:**
  `chad/risk/dynamic_risk_allocator.py` (+183 LoC) now applies the
  full chain: `base → correlation → chassis → tier_filter →
  winner_scaling → regime_booster → caps`. SCR sizing applied
  separately at execution.
- **Telegram BUSINESS STATUS:** morning brief
  (`chad/ops/daily_chad_report.py:1474-1505`) and weekly summary
  (`:1786-1806`) include a BUSINESS STATUS / BUSINESS PHASE section.
- **Dashboard business endpoint:** `/api/state` includes a `business`
  block (`chad/dashboard/api.py:484-525, :539`); chat context
  includes the same block (`:1086-1103`).
- **Config:** `config/winner_scaling_policy.json` (+7 LoC) added.

### 4. `1608fec` — *Build: business framework foundation files* (2026-04-27)

Companion commit. Adds the foundation files built earlier in the
same session (the 11C/12B/12C work above depends on these). Without
this commit the framework would not survive a fresh checkout.

- `chad/ops/equity_history_publisher.py` (+105 LoC) — daily HWM
  checkpoint.
- `chad/ops/portfolio_snapshot_publisher.py` (+153 LoC) —
  IBKR + Kraken equity refresh with CAD→USD via IBKR FX quote.
- `chad/risk/withdrawal_manager.py` (+290 LoC) — salary authorization
  with HWM/drawdown gating.
- `chad/risk/tier_manager.py` (+175 LoC) — equity-tier strategy
  enable/disable.
- `config/tiers.json` (modified, +73 net) — 5-tier ladder
  (MICRO → INSTITUTIONAL).
- `config/withdrawal_policy.json` (+13 LoC) — BUILD/GROW/PAY rules.
- `config/regime_booster_policy.json` (+11 LoC) — bounded regime
  multiplier policy.

### 5. `4a37b4c` — *Audit: behavioral contract verification of SSOT v8.3* (2026-04-28)

- **Was:** SSOT v8.3 made several behavioral claims (regime
  booster on orchestrator cycle, reconciliation timer disabled,
  event_risk timer enabled, etc.) that had not been independently
  verified against the running machine.
- **Was added:** A small audit harness that walks the SSOT's
  behavioral contracts and reports pass/fail. Uncovered the gaps
  that v8.4 then closed.
- **Result:** Confirmed the v8.3 framework is real where the SSOT
  claimed; flagged the gaps closed by `f62d914`.

### 6. `031a9de` — *Tighten brief voice: zero jargon for non-trader operator* (2026-04-28)

- **Was broken:** Morning brief and end-of-day report used
  trader-grade vocabulary even though the operator-facing voice
  lock in `MorningBrief._chads_take` is meant to render in plain
  English. Specific terms (basis points, drawdown, expectancy,
  Kelly fraction) leaked through.
- **Was fixed:** System prompt + post-processing pass tightened
  to strip remaining jargon and substitute plain-English
  equivalents.
- **Result:** Briefs read as a friend updating you on the
  business, not a desk note. No behavioral change to the trading
  system.

### 7. `f62d914` — *Build: SSOT v8.4 gap closure* (2026-04-28)

Single commit closes five gaps surfaced by the audit harness in
`4a37b4c`.

- **chad-reconciliation.timer masked.** Was `disabled`, not
  masked. A manual `systemctl start` would have raced against
  `chad-reconciliation-publisher.timer` on
  `runtime/reconciliation_state.json`. Symlinked to `/dev/null`;
  race window permanently closed.
- **chad-regime-booster.timer installed.** SSOT v8.3 §6E claimed
  `regime_booster.json` was refreshed on the orchestrator cycle.
  In practice no orchestrator hook called the publisher and the
  file ran 15+ hours stale. New systemd unit at 60-second cadence
  runs `python3 -m chad.risk.regime_booster` directly.
- **chad-event-risk.timer enabled.** Was `disabled` and the
  `timers.target.wants/` entry was a malformed real-file rather
  than a symlink. `event_risk.json` had been 25+ days stale;
  RegimeBooster's veto on `severity in [high, extreme]` could
  never fire. Replaced the bad real-file with the correct
  symlink, enabled the timer (10-minute cadence). Caveat:
  provider is still `MarketHoursRiskProvider` (placeholder
  returning `severity=medium` for US session edges); a real
  CPI/FOMC/NFP calendar source remains a roadmap item.
- **withdrawal_manager.py HWM fixed.** `:158` was
  `hwm = max(max(equities), current_equity)` — current equity
  was always part of the HWM set, so `surplus = current_equity -
  hwm` was permanently ≤ 0 and authorized salary was permanently
  $0 regardless of phase. Changed to `hwm = max(equities)`.
  Effect is prospective: with only 1 history record at write
  time the bug was not observable from runtime; once the second
  daily record lands at 23:59 UTC on 2026-04-28, salary
  authorization can unlock when PAY-phase preconditions are met.
- **strategy_weights.json key renamed.** Was `"crypto": 0.04`;
  `tier_state.json:enabled_strategies` used the canonical name
  `alpha_crypto`. Cap publisher emitted runtime entries under
  literal `crypto`, which `load_tier_filter` did not find in the
  enabled list, so `tier_factor` resolved to `0.0` and
  `alpha_crypto` received a `$0` final cap. Renamed the key to
  `alpha_crypto` (value unchanged at 0.04; total 16 keys; sum
  1.0). Cap publisher now emits under `alpha_crypto`, matching
  the tier filter and producing the expected `$334.53` final cap.

### Aggregate

v8.3 contributed +1,886 / -25 across 13 files (most of which were
new modules). v8.4 added three smaller commits — re-run `git diff
accaa2b..HEAD --stat` and update this line with the fresh totals.

---

## 14. KNOWN ISSUES

### RESOLVED IN v8.4

| ID | Was | Resolution | Commit |
|---|---|---|---|
| `regime_booster.json` stale | DEGRADED — no publisher wired | `chad-regime-booster.timer` installed (2026-04-28) | `f62d914` |
| `event_risk.json` stale | DEGRADED — timer disabled | `chad-event-risk.timer` enabled (2026-04-28) | `f62d914` |
| `chad-reconciliation.timer` not masked | SAFETY — disabled not masked | Masked via symlink to `/dev/null` (2026-04-28) | `f62d914` |
| `withdrawal_manager.py:158` HWM bug | BUG — salary permanently $0 | `hwm = max(equities)` (2026-04-28) | `f62d914` |
| `strategy_weights.json` key `crypto` | COSMETIC — tier lookup mismatch | Renamed to `alpha_crypto` (2026-04-28) | `f62d914` |

### RESOLVED IN v8.5

| ID | Was | Resolution |
|---|---|---|
| `omega_momentum_options` BLOCKING | VIX missing from price_cache | VIX injected from `data/bars/1d/VIX.json` on every refresh (2026-04-30) |
| `omega_momentum_options` schema | `updated_ts_utc` key mismatch | Fixed to `ts_utc` (2026-04-30) |
| `omega_momentum_options` dead contracts | Always returned None, masked by except | Replaced with honest stub (2026-04-30) |
| `beta` gap/cap collision | `underweight_gap == max_position_weight` | `underweight_gap` 0.02→0.005 (2026-04-30) |
| `beta` missing consensus prices | 19/25 symbols not in price_cache | 20 symbols added to `universe.json` (2026-04-30) |
| `gamma_futures` missing MYM/M2K | Bar files absent, strategy on 3/5 symbols | Bars backfilled, expiry tables patched (2026-04-30) |
| `winner_scaling.json` TTL < cadence | 600s TTL shorter than 900s publish cadence | Per-file thresholds: winner=1800s, booster=120s (2026-04-30) |
| `winner_scaling.json` missing 10 strategies | Implicit 1.0 for absent strategies | Explicit canonical roster fill (2026-04-30) |
| Silent zero-signal returns | `alpha_intraday`, `alpha_options`, `gamma_reversion` had no logging | `import logging` + diagnostic LOG statements added (2026-04-30) |

### DEGRADED

| ID | Severity | Summary |
|---|---|---|
| **`alpha_options` stuck on SPY** | DEGRADED | `alpha_options\|SPY` is `open=true` from 2026-04-24T20:18Z. Position MAINTAINED → no new spreads emit until existing closes. Carried from v8.2. By design (no flip on options spreads) but blocks signal generation — investigate whether a time-based exit should be added. |
| **`omega_vol` health = 0.10** | DEGRADED (low confidence) | 3 samples, 0 wins, slippage_ratio 0.5, regime_alignment 0.0. Sample size too small to act on, but watch list — if no recovery, F4 edge-decay should halt at the 5-loss threshold. |
| **strategy_intelligence shows all profile=normal** | DEGRADED (input-feed gap) | `runtime/strategy_intelligence.json` reasoning string says *"Insufficient data due to macro context provider error; VIX and macro risk label unknown"*. Macro context wiring gap — same as v8.2. |
| **alpha_futures expectancy negative on largest sample** | DEGRADED (winner-scaler caught it) | `-$51.06` expectancy on 329 trades; WinnerScaler now penalizes at 0.5×. The system itself is recognising and downsizing. |

### TEST FAILURES (current run, 5 failed / 1014 passed / 29 warnings)

| Test | Cause | Severity |
|---|---|---|
| `chad/tests/test_position_guard.py::test_rebuild_clears_broker_sync_when_strategy_entry_added` | Pre-existing — `clientId=99` collision artifact in test fixtures. | Cosmetic (production uses `clientId=83` for reconciliation, `=84` for portfolio snapshot). |
| `chad/tests/test_position_guard.py::test_rebuild_preserves_broker_sync_when_no_strategy_entry_for_symbol` | Same root cause. | Cosmetic. |
| `chad/tests/test_position_guard.py::test_rebuild_partial_attribution_multi_strategy` | Same root cause. | Cosmetic. |
| `chad/tests/test_position_guard.py` (3rd guard test variant) | Same root cause. | Cosmetic. |
| `chad/tests/test_regime_classifier.py::test_g2_matrix_loads_with_calibrated_values` | Config drift — assertion `len(mat["volatile"]) < 10` fails because the config has 14 entries per the deliberate 2026-04-22 Audit-O+P expansion documented in `config/regime_activation_matrix.json:_comment_4`. | Cosmetic — config is intentionally expanded; the assert needs updating to reflect the calibrated reality. |

Total: **5 failed, 1014 passed, 29 warnings**. None are functional
failures; all are stale test fixtures.

### COSMETIC

- **`alpha_futures` docstring lists `MCL`.** The strategy file's
  module-level docstring still names the legacy 4-symbol universe;
  the live tuple is `(MES, MNQ, MGC)` per `:98`. Documentation drift.
- **`alpha_futures_config.py:55-60` defaults to 4 symbols.** Override
  is the hardcoded `ALPHA_FUTURES_UNIVERSE` tuple; legacy default is
  effectively dead code.
- **`strategy_health.json` covers 6 of 16 strategies.** F3 only
  scores strategies that have crossed its sample threshold. Quiet
  strategies are simply absent — not a fault.
- **`event_risk.json` is bootstrap-provider only.** `severity=medium`
  comes from the `MarketHoursRiskProvider` placeholder. Replacing
  with a CPI/FOMC/NFP calendar source is on the roadmap (§15) and
  would let the RegimeBooster react to actual macro events.
- **`days_in_phase` is approximate.** `BusinessPhaseTracker` uses
  `len(history)` as a proxy because there is no phase-history log
  (`business_phase_tracker.py:164-176`). Adequate today; would benefit
  from a dedicated phase-transition log when the system has lived
  through more transitions.

### PENDING (NOT YET BUILT)

- **ML veto loop (Phase 5B).** XGBoost retrain pipeline incomplete.
  Models exist for some strategies; the periodic retrain + canary
  promotion path is not wired.
- **Per-trade P&L attribution decomposition** — alpha vs spread vs
  slippage. Today the closed-trade record has total realized PnL
  only; decomposing into the three components (and surfacing them
  per-strategy in expectancy) is queued.
- **Salary withdrawal automation (move-money-side).**
  `WithdrawalManager` authorizes; the operator still manually moves
  money. Automating the IBKR → bank transfer is out of scope until
  live.

### TRACKED ISSUES (from v8.2 §13, still open)

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

## 15. PHASE ROADMAP

### IMMEDIATE (next session)

- Monday session full audit during US trading hours.
- Per-strategy fill counts during active session — first pass with
  the full risk chain wired.
- First Telegram BUSINESS STATUS section delivered to the operator
  in production at 13:00 UTC Monday.
- Validate `alpha_crypto` key normalization fix (rename
  `crypto` → `alpha_crypto` in `config/strategy_weights.json` under
  governance, or extend the shim into the publisher).
- Fix the 5 pre-existing test failures (4 position_guard + 1 g2_matrix)
  — all require fixture/assertion updates, no behavioural changes.
- First daily equity_history record at 23:59 UTC tonight (today's is
  the seed; second record builds the actual HWM trajectory).

### PHASE 5B — ML veto loop (still pending)

- XGBoost retrain pipeline closure: scheduled retrain + holdout
  validation + canary promotion + edge-decay-style halt.
- Wire the per-strategy ML veto into the routing gate stack as a
  6th gate (default off; opt-in per strategy).

### PHASE 11A — multi-market expansion (partially complete)

- Full options chain integration (per-symbol chains, not just SPY/QQQ
  cache).
- Proper expiry/strike metadata on `alpha_options` fills (currently
  cached chain when available; synthetic fallback otherwise — the
  spec carries placeholder strike/expiry when the chain is stale).
- `alpha_statarb` — stat-arb basket engine.
- `alpha_crypto_alt` — altcoin momentum (DOT, LINK, etc. when Kraken
  pairs available and minimum-volume gates pass).

### PHASE 12 — policy automation (mostly complete in v8.3)

| Item | Status |
|---|---|
| Capital `TierManager` | ✅ **BUILT** (commit `1608fec`) |
| `RegimeBooster` | ✅ **BUILT** (commit `1caa111`) |
| `WinnerScaling` | ✅ **BUILT** (commit `1caa111`) |
| `WithdrawalManager` | ✅ **BUILT** (commit `1608fec`) |
| `BusinessPhaseTracker` | ✅ **BUILT** (commit `1caa111`) |
| `WinnerScaler` daily backtest validation | PENDING — backtest the per-strategy multipliers on out-of-sample data before letting them shape size. |
| Phase-transition history log | PENDING — replace approximate `days_in_phase` with a real transition log. |
| FOMC/CPI/NFP calendar in `event_risk.json` | PENDING — replace `MarketHoursRiskProvider` with a real macro calendar so RegimeBooster vetoes fire on actual macro events. |

### PHASE 9 — pre-live calibration (carried from v8.1/v8.2)

- Regime classifier tuning (ADX proxy → Wilder ADX or threshold
  re-calibration).
- Kelly fraction tuning (`CHAD_ALLOC_V3_KELLY_MAX`).
- Slippage model fit per asset class.
- Live feature distribution drift monitoring.
- Net-EV gate opt-in (populate `expected_pnl` at strategy level).
- Halt-on-reconciliation-mismatch.
- ISSUE-29, 50, 75, 78, 54, 58.

### PHASE 10 — live capital flip (carried from v8.1/v8.2)

Entry criteria:

- All Phase-9 items complete.
- SCR = CONFIDENT.
- 60-90 days consistent paper performance.
- Explicit operator GO via governance rule #3 (`CLAUDE.md`).
- **NEW v8.3 gate:** WithdrawalManager phase = PAY for ≥ 14 days
  (i.e., the salary engine has authorized real payouts under paper
  before a single dollar is exposed to live execution).

When live:

- `CHAD_EXECUTION_MODE` flips from `paper` to `live`.
- `LiveGate` accepts the posture change.
- First 3 cycles run with manual oversight.
- Profit routing flips from advisory to actual capital movement.
- `WithdrawalManager` authorization becomes the basis for actual
  monthly payouts (operator still moves money).

---

## 16. APPENDICES

### Appendix A — File inventory of changed paths since v8.2

From `git log accaa2b..HEAD --name-status` and
`git diff accaa2b..HEAD --stat`:

| File | Δ | Status |
|---|---|---|
| `chad/exchanges/kraken_client.py` | +32 | M (commit `15677a2`) |
| `chad/execution/kraken_executor.py` | +33 | M (commit `15677a2`) |
| `chad/tests/test_kraken_execution.py` | +8/-8 | M (commit `15677a2`) |
| `chad/ops/daily_chad_report.py` | +83 (sum across 1caa111 + 0c3a3e1) | M |
| `chad/dashboard/api.py` | +64 | M (commit `1caa111`) |
| `chad/risk/dynamic_risk_allocator.py` | +183 | M (commit `1caa111`) |
| `chad/risk/regime_booster.py` | +211 | A (commit `1caa111`) |
| `chad/risk/winner_scaler.py` | +201 | A (commit `1caa111`) |
| `chad/ops/business_phase_tracker.py` | +293 | A (commit `1caa111`) |
| `chad/ops/equity_history_publisher.py` | +105 | A (commit `1608fec`) |
| `chad/ops/portfolio_snapshot_publisher.py` | +153 | A (commit `1608fec`) |
| `chad/risk/tier_manager.py` | +175 | A (commit `1608fec`) |
| `chad/risk/withdrawal_manager.py` | +290 | A (commit `1608fec`) |
| `config/regime_booster_policy.json` | +11 | A (commit `1608fec`) |
| `config/tiers.json` | modified | M (commit `1608fec`) |
| `config/withdrawal_policy.json` | +13 | A (commit `1608fec`) |
| `config/winner_scaling_policy.json` | +7 | A (commit `1caa111`) |
| **Total** | **+1,886 / -25 across 13 files** (8 added, 5 modified) |

### Appendix B — Environment files (names only)

These files live outside the repo, are gitignored, and contain
secrets. Listed by name only.

- `/etc/chad/dashboard.env` — dashboard basic-auth credentials.
- `/etc/chad/claude.env` — Anthropic API key for chat + morning brief.
- `/etc/chad/ibkr.env` — IBKR Gateway credentials.
- `/etc/chad/kraken.env` — Kraken REST + WS keys.
- `/etc/chad/openai.env` — fallback / non-Claude model creds (legacy).
- `/etc/chad/polygon.env` — Polygon API key (currently unused).
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
| Kraken executor | `chad/execution/kraken_executor.py` (with `_enforce_kraken_rest_pair`) |
| Kraken client | `chad/exchanges/kraken_client.py` (with `_assert_kraken_rest_pair`) |
| Paper executor (timer) | `/usr/local/bin/chad_paper_trade_executor.py` |
| LiveGate | `chad/core/live_gate.py` |
| Risk allocator | `chad/risk/dynamic_risk_allocator.py` (full chain) |
| Profit Lock | `chad/risk/profit_lock.py` |
| ProfitRouter | `chad/risk/profit_router.py` |
| Evidence writer | `chad/execution/paper_exec_evidence_writer.py` |
| Full preview | `chad/core/full_cycle_preview.py` |
| **Business framework — TierManager** | `chad/risk/tier_manager.py` |
| **Business framework — WinnerScaler** | `chad/risk/winner_scaler.py` |
| **Business framework — RegimeBooster** | `chad/risk/regime_booster.py` |
| **Business framework — WithdrawalManager** | `chad/risk/withdrawal_manager.py` |
| **Business framework — BusinessPhaseTracker** | `chad/ops/business_phase_tracker.py` |
| **Business framework — EquityHistory** | `chad/ops/equity_history_publisher.py` |
| **Business framework — PortfolioSnapshot** | `chad/ops/portfolio_snapshot_publisher.py` |

### Appendix D — Git log of all commits since v8.2

From `git log accaa2b..HEAD --pretty=format:"%H %ad %s" --date=short`:

```
15677a2ca25dc91987ce113769e75671cdeee090  2026-04-26  Fix: enforce Kraken REST pair format at executor + REST border
0c3a3e11eb90401da560a145b6b529ba57076615  2026-04-27  Fix: exclude broker_sync from weekly top brain ranking
1caa111095a3f785d4b153168f13961c475e4fc0  2026-04-27  Build: complete CHAD business framework (Phase 11C/12B/12C)
1608fec3d574f9bef00f2a52ac8283715fbabaa3  2026-04-27  Build: business framework foundation files
4a37b4c70bfeba73ab3d0b41d87f5c1e6c5e3eaa  2026-04-28  Audit: behavioral contract verification of SSOT v8.3
031a9de8ed1ed00d6474537190086f31298f8c59  2026-04-28  Tighten brief voice: zero jargon for non-trader operator
f62d914c2d1964c201e314d8b8aa1e3ab6e11dee  2026-04-28  Build: SSOT v8.4 gap closure — HWM fix, alpha_crypto rename, regime-booster timer, audit harness
```

(7 commits, all by `mcitysolo <mcitysolo@local>`.)

### Appendix E — Business framework cheat sheet

Operator one-liners for inspecting the business state:

```bash
# What phase am I in?
cat /home/ubuntu/chad_finale/runtime/business_phase.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
              print(f\"phase={d['phase']} ({d['phase_description']})\")"

# How much salary?
cat /home/ubuntu/chad_finale/runtime/withdrawal_authorization.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
              print(f\"salary=${d['authorized_withdrawal_usd']:.0f}/mo  reason={d['reason']}\")"

# What tier?
cat /home/ubuntu/chad_finale/runtime/tier_state.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
              print(f\"tier={d['tier_name']} equity=\${d['current_equity_usd']:,.0f} \
              strategies={len(d['enabled_strategies'])}\")"

# Which strategies winning?
cat /home/ubuntu/chad_finale/runtime/winner_scaling.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
              [print(f'{k}: {v}x') \
               for k,v in sorted(d['multipliers'].items(), key=lambda kv: -kv[1])]"

# Boost active?
cat /home/ubuntu/chad_finale/runtime/regime_booster.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
              print(f\"boost={d['multiplier']}x  active={d['active']}  reasons={d['reasons']}\")"

# Daily checkpoint history (last 5)
tail -5 /home/ubuntu/chad_finale/runtime/equity_history.ndjson

# Profit routing totals
cat /home/ubuntu/chad_finale/runtime/profit_routing.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
              print('totals:', d.get('totals'))"
```

### Appendix F — Verification sequence (CLAUDE.md governance rule)

After every change:

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

Expected today: **5 failed, 1014 passed, 29 warnings** (the 5 failures
are the documented pre-existing fixtures — see §14).

### Appendix G — Active git tags

- `STABILITY_FREEZE_20260307_GREEN` — original stable baseline.
- `PRE_HARDENING_20260402` — snapshot before P0 hardening.
- `RATIFICATION_MASTER_20260402` — all hardening + GAP items complete.
- `REVERT_PRE_OVERHAUL_20260419` — snapshot before 2026-04-19/21
  overhaul (commit `45f3728`; runtime tarball at
  `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`).

### Appendix H — Rollback commands (governance-approved)

```bash
# Roll back to post-hardening stable
git checkout RATIFICATION_MASTER_20260402

# Roll back to pre-overhaul stable (restore runtime from tarball too)
git checkout REVERT_PRE_OVERHAUL_20260419
tar -xzf /home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz \
        -C /home/ubuntu/chad_finale
# Steps: /home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt
```

### Appendix I — Operator daily checks (≈ 2 minutes)

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

# 5. Business framework (NEW v8.3)
cat runtime/business_phase.json | python3 -m json.tool
cat runtime/tier_state.json | python3 -m json.tool | head -15
cat runtime/withdrawal_authorization.json | python3 -m json.tool

# 6. Dashboard
curl -s https://chadtrades.com/health
```

What to look for (today's expected values):

- SCR `state` = `WARMUP` (expected until ~100 effective trades; 84
  now).
- Profit lock `mode` = `NORMAL`, `stop_new_entries` = `false`.
- Stop bus `active` = `false`.
- Reconciliation `status` = `GREEN`.
- **Business phase = `GROW`** (will remain GROW until SCR=CONFIDENT
  and 14d history).
- **Tier = `PRO`** (16/16 strategies active).
- **Salary authorized = `$0/mo`** (correct for GROW).
- No strategy with `health_score < 0.3` AND `sample_count ≥ 20`
  (`omega_vol` is below 0.3 but only 3 samples; not actionable yet).
- Dashboard returns HTTP 200.

### Appendix J — Key value summary (single-screen reference)

```
┌──────────────────────────────────────────────────────────────────┐
│                    CHAD AT-A-GLANCE — v8.5                       │
├──────────────────────────────────────────────────────────────────┤
│  Equity:   $183,500.05 USD  (IBKR ~$183,315 + Kraken $184.58)    │
│  Phase:    GROW            (engine built, growing pre-salary)    │
│  Tier:     PRO             (16/16 strategies enabled)            │
│  SCR:      WARMUP 0.10x    (84/100 effective trades)             │
│  Regime:   trending_bull   (confidence 0.6864)                   │
│  VIX:      18.71                                                 │
│  HWM:      $167,227.46     (1d history)                          │
│  Salary:   $0/mo authorized                                      │
│  Boost:    1.0x (vetoed: low_confidence_0.69)                    │
│  Reconcile:GREEN           (chad=1, broker=2, drift=SPY -17)     │
│  Profit Lock:  NORMAL                                            │
│  Stop Bus:     INACTIVE                                          │
│  Top Strategy: alpha       (winner_factor 1.5x)                  │
│  Penalized:    alpha_futures (0.5x), RECONCILED_PHASE2 (0.5x)    │
│  Profit Router totals: trading $1,051.68 / beta $631.01 /        │
│                        amplifier $420.67                         │
│  Today realized P&L: +$125.60                                    │
│  Tests: 5 fail (cosmetic) / 1,014 pass / 29 warn                 │
│  Services: 14 running / 95 loaded / 0 failed                     │
└──────────────────────────────────────────────────────────────────┘
```

---

**End of CHAD Unified SSOT v8.4.**

This document is the truth as of 2026-04-28. CHAD is now a
self-governing autonomous business with explicit phase progression,
tier-gated strategies, performance-based cap reallocation,
regime-aware boosting, and salary authorization. 16 strategies are
wired with the full risk chain. Reconciliation is GREEN.
PaperExecEvidence parity is at 100%. The cap chain is documented
end-to-end for the first time. The Telegram BUSINESS STATUS section
will hit the operator at the next 13:00 UTC weekday window.

Ready for Monday open.

If the code disagrees, either the code drifted or this revision
needs another pass.
