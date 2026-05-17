# CHAD Unified SSOT v9.3
# Elite Upgrade Closeout — Phase A/B Complete, Phase C Max-Buildable, Phase D Paper-Scope Hardened

**Version:** 9.3
**Date:** 2026-05-17
**Status:** Active — Paper Epoch 2 / Elite Upgrade Closeout
**Supersedes:** docs/CHAD_UNIFIED_SSOT_v9.2_2026-05-15.md
**HEAD commit:** `72f361fac544c82cd231d100b2d0bc86d428a990` (short: `72f361f`)
**Branch:** `main`
**Test count:** 2,114 passing (`CHAD_SKIP_IB_CONNECT=1`, 60.10 s, 4 warnings)
**Lock type:** Post-elite-upgrade closeout lock
**Live order posture:** No new live execution authorization granted by this SSOT

---

## 0. Preamble and Version Delta

### 0.1 What v9.2 was

v9.2 (2026-05-15, `5392fc7`, 1,752 lines) closed out the **Elite Signal
Layer Phase A + Phase B** workstreams that began with the v9.1 tier
architecture v2 baseline. v9.2 ratified:

- **Phase A (signal quality, 5/5 items complete):** stop-distance sizing
  (`alpha`, `alpha_intraday`, `alpha_futures`); shared
  `chad/utils/session.py` entry-only session-zone gating; pre-entry R/R
  floor at 1.5 with fail-open semantics; setup-family tagging into the
  expectancy tracker; float-aware liquidity gate
  (`chad/utils/liquidity.py`) with THIN/UNKNOWN handling.
- **Phase B (intelligence feeds, 6/6 publishers + FMP scaffold):** catalyst
  `news_intel`, daily-bar `relative_strength`, intraday `volume_scan`
  with rolling-1m fallback, Kraken-Futures-derived
  `crypto_derivatives`, static `futures_roll_state`, synthetic
  Black–Scholes `options_greeks`, plus the `chad/market_data/fmp_client.py`
  stable-API scaffold.

At v9.2 cutover, the test baseline was **1,874 passing**, the runtime
was `CONFIDENT` with `sizing_factor=1.0`, and `live_readiness` was
`ready_for_live: false`.

### 0.2 What v9.3 captures

v9.3 is the **Elite Upgrade Closeout** SSOT. It captures **27 commits
since v9.2** (`5392fc7..HEAD`) grouped into six coherent workstreams that
together close out:

- Phase C (cross-venue / depth intelligence) at its **max-buildable**
  perimeter, with three external blockers explicitly pinned.
- An FMP-stable analyst/earnings intelligence publisher and its
  read-only dashboard surface.
- Phase D Item 1 (dynamic universe scanner v1) in **observation-only**
  mode, with the runtime artifact and dashboard panel live but **no**
  strategy wiring.
- XGB veto model governance — a candidate/promotion workflow that
  prevents silent overwrite of the active model file.
- Phase D Item 2 (BAG / COMBO execution hardening) ladder through
  **Tier 1**, **Tier 2**, **Tier 3A**, and **Tier 3B**, with the
  Tier 3B live-readonly probe validated 2026-05-17. Tier 3C
  (unit normalisation) is the next authorised workstream.
- Pre-SSOT gap-closure audit and external-blocker pinning that
  underwrite every "complete" claim in this document.

### 0.3 What v9.3 is NOT

v9.3 is **not** a "live everything" authorization. Specifically v9.3
does **not** grant or imply any of the following:

- It does **not** flip the live-order posture. `CHAD_EXECUTION_MODE`
  remains `paper`. `live_readiness.json` continues to read
  `ready_for_live: false`.
- It does **not** authorize live BAG / COMBO execution. The BAG
  hardening ladder is paper-scope only through Tier 3B; Tiers
  3C → 6 remain.
- It does **not** unblock Phase C external blockers. IBKR DOM is
  pending market-hours proof. Kraken Futures live is blocked for the
  current Canadian deployment. Coinglass is blocked until a paid API
  plan/key is procured.
- It does **not** wire the dynamic universe scanner into any
  strategy. The scanner is **observation-only** and does not write
  `runtime/universe.json`.
- It does **not** change any risk cap, posture, strategy config,
  systemd unit, or `runtime_FREEZE_*` / `data_FREEZE_*` artifact.
  All such changes remain Pending Actions per CLAUDE.md §3.

### 0.4 Commit delta from v9.2 (5392fc7..HEAD)

All 27 commits since the v9.2 cut, in reverse chronological order
(newest first), grouped by workstream. The change-log section
(Section 13) repeats this list with one line per commit; this
preamble is the workstream-grouped view.

#### Group A — Phase C and Exchange Extensions

- `42baf16` *(per Section 13)* / first surfaced as commit `42daa1f`
  "Document Phase C IBKR DOM entitlement blocker" — formally pins C3 as
  PENDING / BLOCKED until live MES/MNQ DOM rows return after the CME
  open. No code change.
- `0204cba` "Add Phase C Kraken Futures public intel publisher" — new
  `chad/market_data/kraken_futures_intel_publisher.py` and
  `chad/exchanges/kraken_futures_client.py`; publishes
  `runtime/kraken_futures_intel.json` (schema
  `kraken_futures_intel.v1`) every 5 minutes on the
  `chad-kraken-futures-intel-refresh.timer`. 306 perps covered at
  audit time.
- `759fd32` "Align options Greeks TTL with daily refresh cadence" —
  adjusts the `options_greeks` TTL to match its daily refresh cadence
  so dashboard staleness classifications align.
- `69d3a95` "Add Phase C Kraken Futures adapter scaffold" — adds
  `chad/execution/kraken_futures_adapter.py`. Scaffold only; **never
  wired** into any execution path.
- `c9da3b9` "Add Kraken Futures authenticated smoke scaffold" —
  `chad/tools/kraken_futures_auth_smoke.py`; **fails closed** in the
  absence of credentials.
- `2dd0f24` "Docs: lock Phase C status after Kraken Futures scaffold"
  — `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md`.
- `9eaad5f` "Docs: mark Kraken Futures blocked for Canadian
  deployment" — `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`.
- `db077c8` "Docs: close Phase C and lock Phase D readiness" —
  `docs/CHAD_PHASE_C_CLOSEOUT_AND_PHASE_D_READINESS_2026-05-16.md`.

#### Group B — FMP Earnings / Analyst Intelligence

- `a98a165` "Add FMP earnings and analyst intelligence publisher" —
  new `chad/market_data/fmp_earnings_intel_publisher.py`; publishes
  `runtime/earnings_intel.json` (schema `earnings_intel.v1`) every
  6 hours on `chad-fmp-earnings-intel-refresh.timer`. Uses FMP stable
  endpoints (`earnings-calendar`, `price-target-consensus`,
  `analyst-estimates annual`, `sec-filings-search`). `status=partial`
  is expected for ETFs / unsupported tickers under the current FMP
  plan and is fail-open by design.
- `aca5a1f` "Docs: record FMP earnings intelligence status" —
  `docs/CHAD_FMP_EARNINGS_INTEL_STATUS_2026-05-16.md`.
- `fcb8656` "Surface earnings intelligence in dashboard context" —
  read-only `earnings_intel` block on
  `chad.dashboard.api.StateBuilder._intelligence()`. No strategy
  consumer.

#### Group C — Phase D Item 1 Dynamic Universe

- `99a5084` "Docs: design Phase D dynamic universe scanner" —
  `docs/PHASE_D_DYNAMIC_UNIVERSE_SCANNER_DESIGN_2026-05-16.md`. Design
  pinned the scanner as observation-only and forbade writing
  `runtime/universe.json`.
- `ce0c4d1` "Add Phase D dynamic universe candidate scanner" —
  new `chad/market_data/dynamic_universe_scanner.py`; publishes
  `runtime/dynamic_universe_candidates.json` (schema
  `dynamic_universe_candidates.v1`) every 5 minutes on
  `chad-dynamic-universe-scanner-refresh.timer`.
- `c4144f5` "Surface dynamic universe candidates in dashboard
  context" — read-only `dynamic_universe_candidates` block on
  `chad.dashboard.api.StateBuilder._intelligence()` and module-level
  `_load_dynamic_universe_candidates_context` in
  `chad.intel.strategy_intelligence`. **No strategy consumer**
  (enforced by `chad/tests/test_dynamic_universe_candidates_context.py`).

#### Group D — XGB Model Governance

- `be0155d` "Docs: record XGB veto retrain dirty-tree decision" —
  `docs/XGB_VETO_WEEKLY_RETRAIN_DIRTY_TREE_DECISION_2026-05-17.md`.
  Documents that the 2026-05-17 weekly retrain regressed
  (accuracy 0.7534 → 0.7112, logloss 0.5364 → 0.5953) and that the
  dirty tracked artifacts were intentionally **not** committed.
- `e7e083e` "Docs: plan XGB veto model artifact hygiene" —
  `docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md`. Defines
  the target architecture: candidates under
  `runtime/models/xgb_veto/candidates/<UTC ts>/`, active under
  `runtime/models/xgb_veto/current/`, baseline in `shared/models/`,
  promotion via `scripts/promote_xgb_veto.py`.
- `5463c63` "Add XGB veto model promotion workflow" — implements the
  promotion script and the predictor's runtime-first / baseline-fallback
  loader. The active model surface today is the baseline
  (`xgb_veto_20260510_020007`, accuracy 0.7534, logloss 0.5364),
  because the regressed 2026-05-17 retrain was not promoted.

#### Group E — Phase D Item 2 BAG / COMBO Hardening

- `51166d4` "Add Phase D BAG spread spec hardening" — Tier 1; adds
  frozen `chad/options/spread_spec.py::OptionsSpreadSpec` dataclass,
  `to_legacy_meta()` / `from_legacy_meta()` adapters, and stamps
  `meta["spread_spec"]` on `alpha_options` BAG signals. Adapter and
  paper-fill writer prefer the typed spec when present.
- `c7c7406` "Docs: design BAG hardening Tier 2 LMT discipline" —
  `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER2_DESIGN_2026-05-17.md`.
- `2ab03e7` "Enforce BAG limit-order discipline" — Tier 2; adapter
  coerces MKT BAG intents to LMT (marker
  `BAG_MKT_COERCED_TO_LMT`) and skips BAG intents missing a positive
  limit price (marker `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`).
  Hydrates `limit_price` from `meta["spread_spec"].net_debit_estimate`
  or `meta["net_debit_estimate"]` when not explicitly supplied.
- `c620119` "Docs: design BAG quote check Tier 3" —
  `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER3_QUOTE_CHECK_DESIGN_2026-05-17.md`.
- `a5c5b35` "Add offline BAG quote check module" — Tier 3A;
  `chad/options/quote_check.py` (the offline `bag_quote_check`
  validator). Module-only; **not wired into the adapter**.
- `4523032` "Docs: design BAG live quote probe Tier 3B" —
  `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER3B_LIVE_QUOTE_PROBE_DESIGN_2026-05-17.md`.
- `9fb163b` "Add BAG quote probe script" — Tier 3B;
  `scripts/probe_bag_quotes.py`. Supports `--dry-run-fake` and
  `--live-readonly`; never places orders, never imports any
  execution adapter.
- `a9fb34a` "Fix BAG quote probe async contract qualification" —
  qualifies contracts via `ib_async.qualifyContractsAsync`. Required
  to make `--live-readonly` succeed under the current `ib_async`
  surface.
- *(no code commit yet)* Tier 3B results doc
  `docs/PHASE_D_ITEM2_BAG_QUOTE_PROBE_RESULTS_2026-05-17.md`
  captures the **BAG limit-price unit-mismatch finding**
  (`--limit-price 350` → rejected; `--limit-price 3.97` → accepted).

#### Group F — Pre-SSOT and External Blocker Closure

- `42daa1f` "Document Phase C IBKR DOM entitlement blocker" — pinned
  C3 as PENDING / BLOCKED.
- *(doc only)* `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md` — Phase C
  status lock after Kraken Futures scaffold landed.
- *(doc only)* `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`
  — pinned C1 live execution as permanently blocked for the current
  Canadian deployment.
- *(doc only)* `docs/CHAD_PHASE_C_CLOSEOUT_AND_PHASE_D_READINESS_2026-05-16.md`
  — closed Phase C and locked Phase D as the next workstream.
- `72f361f` "Docs: close pre-SSOT elite upgrade gaps" — current HEAD;
  `docs/CHAD_PRE_SSOT_GAP_CLOSURE_AUDIT_2026-05-17.md` and
  `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`. These two
  documents underwrite every "complete" claim in this SSOT.

---

## 1. Mission and Core Architecture

### 1.1 Mission

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a **self-compounding
quantitative trading system** currently in **Paper Epoch 2**. Per
CLAUDE.md, paper-equity SCR has reached the `CONFIDENT` band,
`sizing_factor=1.0`, `paper_only=false`, and full sizing is applied —
**but live activation remains gated by `live_readiness.json`**, which
this SSOT does not flip.

CHAD uses **real IBKR market data and real broker connectivity** today
(via `ib_async` for execution and `IBKRPriceProvider` for prices). It
also uses real Kraken-Futures public intelligence (read-only) for
crowded-funding crypto signals and real Financial-Modeling-Prep stable
endpoints for earnings/analyst metadata. **This SSOT does not newly
authorize any live execution path** beyond what was already paper-mode
operational at the v9.2 cut.

### 1.2 Architecture Layers

The v9.3 architecture is composed of eight layers, each governed by the
existing CLAUDE.md change-control discipline:

1. **Intelligence Layer.** Out-of-band publishers writing JSON artifacts
   under `runtime/`. Examples: `regime_state`, `event_risk`,
   `macro_state`, `choppy_regime`, `news_intel`, `relative_strength`,
   `volume_scan`, `crypto_derivatives`, `futures_roll_state`,
   `options_greeks`, `kraken_futures_intel`, `earnings_intel`,
   `dynamic_universe_candidates`. Every consumer reads through
   fail-open helpers; missing / stale / partial payloads never raise.

2. **Signal Layer.** Sixteen named strategy heads (see Section 3) plus
   the experimental `alpha_intraday_micro` head added in v9.1. Each
   strategy emits zero or more `TradeSignal` objects with typed
   `meta` payloads. Phase A's entry-only gate chain (session, R/R,
   liquidity, catalyst, RS, RVOL, crypto-crowding, roll, Greeks-metadata)
   is enforced per-strategy per-entry; exits never block on Phase A
   gates.

3. **Risk Layer.** Per-strategy risk budgets are computed by
   `dynamic_risk_allocator` and published into
   `runtime/dynamic_caps.json` (normalized weights, portfolio risk
   cap). Tier-aware risk-per-trade is enforced by
   `tier_risk_enforcer`. The Profit Lock guards realized PnL and the
   Stop Bus serialises stop adjustments. The XGB veto model (a
   secondary, shadow-able predictor) shaves confidence when a probable
   loser is detected, but **never** generates trades.

4. **Execution Layer.** `execution_pipeline.py` builds intents,
   `order_type_selector.py` picks `LMT` for all asset classes
   (including BAG — Tier 2 enforcement),
   `chad/execution/ibkr_adapter.py` qualifies contracts and submits
   orders. The `kraken_futures_adapter.py` scaffold exists but is
   **dormant** (never wired). `paper_exec_evidence_writer.py` stamps
   attribution records for every paper fill; PnL math is unchanged
   from v9.1 for non-BAG, and BAG closes remain `pnl_untrusted=True`
   (synthetic 30 % credit ratio).

5. **Broker Layer.** IBKR Gateway via `ib_async` (Phase 1 of the
   ib_insync→ib_async migration is complete; Phase 2 has five files
   remaining per CLAUDE.md). `IBKRPriceProvider` handles `STK / FUT /
   FX` snapshots. There is **no OPT or BAG path** in the price
   provider today; the Tier 3B probe established the
   `qualifyContractsAsync` path on `ib_async` outside the production
   adapter. Kraken spot remains accessible; **Kraken Futures live
   trading is blocked** for the current Canadian deployment.

6. **Reconciliation Layer.** `trade_closer`, `position_guard`,
   `reconciliation_publisher`, and `lifecycle_truth_publisher` together
   build the live ledger. The guard drift detector
   (`chad/core/position_guard.py::detect_guard_vs_broker_truth_drift`)
   is wired into `chad-reconciliation-publisher` and emits
   `runtime/position_guard_drift.json` every cycle. Operators close
   stale entries with `scripts/close_guard_entry.py`. BAG
   reconciliation is still per-leg — **spread_id-aware reconciliation
   is pending Tier 5**.

7. **Dashboard / Operator Layer.** Flask dashboard exposing the
   `/intelligence` block (now including read-only `earnings_intel` and
   `dynamic_universe_candidates`). Telegram operator interface unchanged
   from v9.2. No strategy is routed from a dashboard panel — every
   panel is read-only.

8. **Model Governance Layer.** The XGB veto promotion workflow
   (`scripts/promote_xgb_veto.py`) is the canonical surface for
   accepting or rejecting weekly retrains. Active model is **always**
   the runtime-current entry under `runtime/models/xgb_veto/current/`
   when present, otherwise the tracked baseline under `shared/models/`.
   Today (2026-05-17), the active model is the baseline:
   `xgb_veto_20260510_020007`, accuracy 0.7534, logloss 0.5364.

### 1.3 Architecture Stack (v9.3)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INTELLIGENCE LAYER                              │
│                                                                          │
│  Regime · EventRisk · Macro · Choppy                                      │
│  News · RelStrength · VolumeScan · CryptoDerivs · FuturesRoll · Greeks   │
│  KrakenFuturesIntel (read-only) · EarningsIntel (FMP) ·                  │
│  DynamicUniverseCandidates (observation-only)                            │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          SIGNAL LAYER (entries)                          │
│                                                                          │
│  Phase A gate chain (entry-only, fail-open):                             │
│   session · stop-distance sizing · R/R ≥ 1.5 · liquidity (eq) ·          │
│   catalyst (eq) · RS modifier (eq) · RVOL modifier (intraday eq) ·       │
│   crypto-crowding (crypto) · roll gate (futures) ·                       │
│   Greeks metadata (options)                                              │
│                                                                          │
│  16 strategy heads + alpha_intraday_micro                                │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           RISK LAYER                                     │
│                                                                          │
│  DynamicRiskAllocator (per-strategy weights) · TierManager ·             │
│  ProfitLock · StopBus · XGB veto (shadow) ·                              │
│  Net exposure gate · per-symbol daily-loss limit · winner scaler         │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION LAYER                                  │
│                                                                          │
│  ExecutionPipeline → IBKRStrategyTradeIntent → IBKRAdapter               │
│   · BAG LMT discipline (Tier 2 — coerces MKT BAG → LMT, skips           │
│     BAG with no positive limit price)                                    │
│   · OptionsSpreadSpec (Tier 1 — typed BAG contract)                     │
│   · offline bag_quote_check (Tier 3A — module exists; not wired)        │
│   · live-readonly quote probe (Tier 3B — script only; validated)        │
│  KrakenFuturesAdapter — scaffold only; DORMANT                          │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          BROKER LAYER                                    │
│                                                                          │
│  IBKR Gateway (ib_async) · IBKRPriceProvider (STK/FUT/FX only) ·        │
│  Kraken spot (active) · Kraken Futures live (BLOCKED — Canada) ·        │
│  IBKR DOM (PENDING — Error 354 / market-hours probe required)           │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       RECONCILIATION LAYER                               │
│                                                                          │
│  PaperExecEvidenceWriter · TradeCloser · PositionGuard ·                 │
│  ReconciliationPublisher · LifecycleTruthPublisher ·                     │
│  PositionGuardDriftDetector → runtime/position_guard_drift.json          │
│  (BAG reconciliation is per-leg today; spread_id-aware pending Tier 5)   │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  DASHBOARD / OPERATOR LAYER                              │
│                                                                          │
│  Flask dashboard (/intelligence — earnings_intel and                     │
│  dynamic_universe_candidates read-only) · Telegram operator interface    │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       MODEL GOVERNANCE LAYER                             │
│                                                                          │
│  XGB veto: runtime/models/xgb_veto/current/  (active when present)       │
│            runtime/models/xgb_veto/candidates/<UTC ts>/  (weekly)        │
│            shared/models/  (baseline fallback)                           │
│  Promotion: scripts/promote_xgb_veto.py                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Cross-cutting safety promises

- **No live BAG execution authorization** is granted by this SSOT.
- **No IBKR DOM authorization or DOM-consuming code** is in scope until
  DOM rows are proven for MES + MNQ at the next CME open without an
  `Error 354`.
- **No Kraken Futures live authorization** for the current Canadian
  deployment. The Kraken Futures public intel feed remains read-only and
  is the only Kraken Futures surface that runs in production.
- **No Coinglass dependency.** No keyless scaffold is permitted; until
  a paid API plan/key is procured, the publisher does not exist.
- **No dynamic-scanner writeback** to `runtime/universe.json` or any
  strategy `target_universe`. The scanner is observation-only.

---

## 2. Current Runtime Truth — 2026-05-17

Values in this section come from direct `runtime/` reads at audit time
(2026-05-17 ~17:19 UTC). They are not derived from previous SSOTs.

### 2.1 Equity, risk cap, and weights

`runtime/dynamic_caps.json` (`ts_utc=2026-05-17T17:19:17.581375Z`,
`ttl_seconds=300`):

- `total_equity`: **$172,578.62 USD** (paper).
- `portfolio_risk_cap`: **$8,628.93 USD** (~5.0 % of equity).
- 16 strategies present in `normalized_weights`. Top five by weight:
  - `beta`: 0.19949501928755517
  - `gamma`: 0.13653728507167728
  - `alpha`: 0.13162482338010820
  - `beta_trend`: 0.10050498071244483
  - `alpha_futures`: 0.07451932228335467
- Sum of normalized weights = 1.0 by construction.

### 2.2 SCR posture

`runtime/scr_state.json` (`ts_utc=2026-05-17T17:19:35Z`,
`ttl_seconds=180`, source `http://127.0.0.1:9618/shadow`):

- `state`: **CONFIDENT**
- `schema_version`: `scr_state.v1`
- Per CLAUDE.md, `sizing_factor=1.0`, `paper_only=false`,
  `sharpe_like=+5.825`, `effective_trades=190`, `win_rate=0.774`,
  `total_pnl=+$9,880` (snapshot 2026-05-09). The current runtime
  payload confirms `state=CONFIDENT`; per-cycle stats vary as new
  paper fills land.

### 2.3 Live readiness

`runtime/live_readiness.json` (`ts_utc=2026-05-17T17:21:14Z`,
`ttl_seconds=604800`):

- `schema_version`: `live_readiness_state.v1`
- `ready_for_live`: **false**
- `latest_report_path`: `/home/ubuntu/chad_finale/reports/live_readiness/LIVE_READINESS_20260517T172114Z.json`
- `latest_report_sha256`: `sha256:4edc5814c58514d0886100f32e8f4d34774cb36d688f8ad864b2e3ae40b2317d`

This is **the canonical gate** for live-order authorization. v9.3 does
not flip it.

### 2.4 Tier state

`runtime/tier_state.json` (`schema_version=tier_state.v2`,
`ts_utc=2026-05-17T17:15:04.951230Z`). Present and fresh. Schema fields
include `tier_name`, `tier_description`, `current_equity_usd`,
`tier_min_equity`, `tier_max_equity`, `enabled_strategies`,
`allowed_instruments`, `risk_profile`, `previous_tier`,
`promoted_at_utc`, `demotion_pending`, `demotion_pending_to`,
`demotion_pending_reason`, `demotion_pending_since_utc`,
`demotion_applies_at`. Detailed per-tier risk profiles are unchanged
from v9.1 Appendix A and reproduced in Appendix A of this SSOT.

### 2.5 Portfolio snapshot

`runtime/portfolio_snapshot.json` (`ts_utc=2026-05-17T17:15:09.884199Z`,
`ttl_seconds=300`). Top keys: `ibkr_equity`, `coinbase_equity`,
`kraken_equity`. Schema is the per-broker-equity snapshot consumed by
`portfolio_engine.py`.

### 2.6 Dynamic universe candidates

`runtime/dynamic_universe_candidates.json`
(`schema_version=dynamic_universe_candidates.v1`,
`ts_utc=2026-05-17T17:16:11Z`, `ttl_seconds=300`):

- `status`: ok
- `source.provider`: `local_runtime_intelligence`
- `source.provider_status`: real
- Inputs freshness at this read: `active_universe` runtime-fresh;
  `news_intel`, `event_risk`, `earnings_intel`, `volume_scan` fresh;
  `relative_strength` available but classified `fresh=false` by the
  scanner's TTL check (still consumed fail-open).
- `summary`: candidates_published=25, confirmed_catalyst_count=5,
  earnings_warning_count=1, high_rvol_count=0, strong_rs_count=3,
  symbols_considered=25.

Top 10 candidates (rank, symbol, score, top reason):

| Rank | Symbol | Score | Reasons | Warnings |
|---:|:---|---:|:---|:---|
| 1 | LLY    | 0.63 | rs_strong, confirmed_high_catalyst | rvol_low |
| 2 | MA     | 0.43 | rs_neutral, confirmed_high_catalyst | rvol_low |
| 3 | NVDA   | 0.38 | rs_strong | rvol_low, earnings_within_7d |
| 4 | UNH    | 0.38 | rs_strong | rvol_low |
| 5 | AVGO   | 0.36 | rs_neutral, confirmed_medium_catalyst | rvol_low |
| 6 | BAC    | 0.28 | confirmed_high_catalyst | rs_weak, rvol_low |
| 7 | TLT    | 0.28 | confirmed_high_catalyst | rs_weak, rvol_low |
| 8 | AAPL   | 0.18 | rs_neutral | rvol_low |
| 9 | GOOGL  | 0.18 | rs_neutral | rvol_low |
| 10 | IWM    | 0.18 | rs_neutral | rvol_low |

These candidates are **observation-only**. No strategy consumes this
artifact; the dashboard surfaces the top-N for operator review during
the §9 observation period defined in the design doc.

### 2.7 XGB active model status

`scripts/promote_xgb_veto.py --status`:

```
[promote_xgb_veto] active model (source=baseline)
  model_version : xgb_veto_20260510_020007
  trained_at_utc: 2026-05-10T02:00:07.712729+00:00
  accuracy      : 0.7534
  logloss       : 0.5364
```

`source=baseline` means the predictor is loading
`shared/models/xgb_veto_model.json` because no candidate has been
promoted into `runtime/models/xgb_veto/current/` yet. The 2026-05-17
retrain regressed and was **not** promoted; the regressed weights live
in the working tree only as untracked timestamped backups and the
baseline remains active.

### 2.8 Full test suite baseline

```
2114 passed, 4 warnings in 60.10s (0:01:00)
```

Command:

```
source venv/bin/activate
export CHAD_SKIP_IB_CONNECT=1
export PYTHONPATH=/home/ubuntu/chad_finale
python3 -m pytest chad/tests/ -q --tb=no
```

This matches the green baseline reported in
`docs/CHAD_PRE_SSOT_GAP_CLOSURE_AUDIT_2026-05-17.md` (2,114 passed,
4 warnings, 69.39 s under `--tb=short`).

### 2.9 full_cycle_preview baseline

```
python3 -m chad.core.full_cycle_preview
```

End-of-run summary:

- `artifact_exists: True`
- `counts.evaluated_signals: 0`, `counts.ibkr_intents_count: 0`,
  `counts.orders_count: 0`, `counts.raw_signals: 0`,
  `counts.routed_signals: 0`, `counts.total_notional: 0`
- ExecutionPlan: `orders_count: 0`, `total_notional: 0.00`,
  `futures_orders: 0`
- IBKR StrategyTradeIntents (preview-only): `intents_count: 0`
- Dynamic caps preview snapshot: `total_equity=172578.62`,
  `portfolio_risk_cap=8628.93`, normalized_weights as above.
- Alpha Futures Smoke Test: `executed=False`, `mode=disabled`,
  `ok=True`, `signal_count=0`.
- Footer: `[full_cycle_preview] NOTE: No broker calls were made. This
  is a logical preview only.`

### 2.10 Intelligence feed freshness snapshot

| Feed | `ts_utc` | `ttl_seconds` | Status |
|:---|:---|---:|:---|
| `news_intel.json` | 2026-05-17T17:19:40Z | 1800 | ok (25 symbols; 5 with catalyst; 4 high-strength) |
| `relative_strength.json` | 2026-05-17T12:52:45Z | 90000 | ok (23 computed; 3 strong, 6 weak, 16 neutral) |
| `volume_scan.json` | 2026-05-17T17:16:54Z | 300 | ok (25 scanned; 0 high RVOL; rolling_1m fallback active) |
| `crypto_derivatives.json` | 2026-05-17T17:15:05Z | 600 | ok (3 mapped symbols; 1 long-crowded, 1 short-crowded) |
| `futures_roll_state.json` | 2026-05-16T21:50:34Z | 86400 | ok (9 tracked; 4 supported, 5 unsupported; 0 in warning window) |
| `options_greeks.json` | 2026-05-16T23:48:44Z | 90000 | ok (SPY only; 3 expiries; 21 strikes; approximated) |
| `kraken_futures_intel.json` | 2026-05-17T17:16:54Z | 600 | ok (306 perps; 3 mapped; 14 long-crowded, 19 short-crowded) |
| `earnings_intel.json` | 2026-05-17T17:02:01Z | 21600 | partial (25 processed; 1 next-earnings, 4 price-targets, 4 analyst-estimates, 5 sec-filings) |
| `event_risk.json` | 2026-05-17T16:54:08Z | 1800 | present (operator calendar loaded) |
| `setup_family_expectancy.json` | 2026-05-16T01:00:05.689369Z | 86400 | within 24 h cadence; 1,574 trades processed across 6 families |

All feeds within their declared TTL (`relative_strength` is within its
25 h TTL but `dynamic_universe_scanner` classified it as `fresh=false`
on its own internal 1 h check — this is the scanner's own conservative
freshness gate, and it consumes it fail-open).

### 2.11 Git baseline

- HEAD: `72f361fac544c82cd231d100b2d0bc86d428a990` (short: `72f361f`).
- Branch: `main`.
- Working tree at SSOT-creation time: clean.
- Total commits behind in the audit window since v9.2 (`5392fc7`): 27.

---

## 3. Strategy Inventory

The 16 named strategies plus `alpha_intraday_micro` (the 17th head added
in v9.1) are the canonical CHAD strategy surface. Modes reflect Paper
Epoch 2 status as of 2026-05-17.

| Strategy | Asset Class | Mode | Current Status | Notes |
|:---|:---|:---|:---|:---|
| `alpha` | Equities / ETFs | Paper | Active | Phase A entries (session, R/R, stop-distance sizing, liquidity, catalyst, RS). |
| `alpha_intraday` | Equities / ETFs | Paper | Active | Phase A entries + RVOL modifier (±0.05). |
| `alpha_intraday_micro` | MES (micro futures) | Paper | Active | 5 setup families; primary-session-only via `primary_session_only`; EOD flatten via `chad-micro-eod-flatten.timer`. |
| `alpha_futures` | Index micros (MES, MNQ, MGC) | Paper | Active | Session, sizing, R/R, roll gate (quarterly micros only). |
| `alpha_crypto` | BTC-USD, ETH-USD, SOL-USD | Paper | Active | Crypto crowding filter against `crypto_derivatives.json`. |
| `alpha_options` | Options (SPY vertical spreads) | Paper | Active | Emits `OptionsSpreadSpec` typed BAG meta. Tier 2 LMT discipline enforced. Offline `bag_quote_check` module exists; **adapter does not yet consult it**. Quote probe script exists for live-readonly validation. **Live BAG execution remains unauthorized** pending Tier 3C / 3D / 4 / 5 / 6. |
| `beta` | Equities | Paper | Active | Consensus-driven; existing v9.1 gates. |
| `beta_trend` | Trend-following equities/ETFs | Paper | Active | Legend-driven universe; existing v9.1 gates. |
| `gamma` | Mean-reversion equities | Paper | Active | Uses `get_trade_universe()` canonical loader. |
| `gamma_futures` | Futures MR (MCL, MYM, M2K, ZN, ZB) | Paper | Active | Existing v9.1 gates; unsupported quarterly micros log `unsupported_v1` rows only. |
| `gamma_reversion` | ETF reversion (SPY, QQQ, GLD, TLT) | Paper | Active | Existing v9.1 gates. |
| `delta` | Cross-asset (SPY, QQQ) | Paper | Active | Existing v9.1 gates. |
| `delta_pairs` | Stat-arb pairs (SPY, QQQ, IWM) | Paper | Active | Existing v9.1 gates. |
| `omega` | Macro overlay (SPY, QQQ, SH, PSQ) | Paper | Active | Existing v9.1 gates. |
| `omega_macro` | Macro futures (ZN, ZB, M6E) | Paper | Active | Existing v9.1 gates. |
| `omega_momentum_options` | Options momentum (SPY, QQQ, AAPL, NVDA, MSFT) | Paper | Active | Existing v9.1 gates. |
| `omega_vol` | Vol ETPs (SVXY, UVXY) | Paper | Active | Strict allow-list `ALLOWED_VOL_SYMBOLS`. |
| `alpha_crypto_perps` | Crypto perpetuals | **Planned / Deferred** | Not implemented | **Kraken Futures live trading is BLOCKED** for the current Canadian deployment. The Kraken Futures public intel feed (`kraken_futures_intel.v1`) is **read-only**. Re-evaluation only after a separate operator-led jurisdiction unlock. |

### 3.1 alpha_options — current paper-scope hardening

`alpha_options` is the highest-touch strategy in this SSOT because the
Phase D Item 2 BAG hardening ladder lives inside it. The current state:

- **OptionsSpreadSpec emitted.** Tier 1 (commit `51166d4`) stamps a
  frozen `OptionsSpreadSpec` under `meta["spread_spec"]` on every BAG
  signal alongside the legacy flat keys.
- **Legacy BAG meta preserved.** Strategy continues to emit the
  legacy string-keyed dict (`sec_type=BAG`, `spread_id`, strikes,
  rights, `net_debit_estimate`, Greeks). Tier 1 was strictly additive.
- **LMT discipline enforced.** Tier 2 (commit `2ab03e7`) coerces MKT
  BAG intents to LMT (marker `BAG_MKT_COERCED_TO_LMT`) and skips BAG
  intents with no positive limit price (marker
  `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`). Limit price is hydrated from
  `meta["spread_spec"].net_debit_estimate` (typed) or
  `meta["net_debit_estimate"]` (legacy) when not explicitly supplied.
- **Offline `bag_quote_check` module exists.** Tier 3A (commit
  `a5c5b35`) added `chad/options/quote_check.py`. The module is
  importable and unit-tested but **not wired into the adapter** — it
  exists today only as a precondition for Tier 3D adapter enforcement.
- **Quote probe script exists.** Tier 3B (commit `9fb163b`, async fix
  `a9fb34a`) added `scripts/probe_bag_quotes.py` with `--dry-run-fake`
  and `--live-readonly` modes. The probe **never** places orders and
  **never** imports any execution adapter.
- **Live BAG execution remains unauthorized.** The 2026-05-17 probe
  surfaced a contract-dollar vs per-share unit mismatch (see §5.2
  "BAG Execution Governance" and §14 "RISK-BAG-01"). Tier 3C unit
  normalisation must land before adapter quote enforcement (Tier 3D)
  is wired; Tier 4 (bracket/failsafe), Tier 5 (spread_id
  reconciliation), and Tier 6 (live BAG fill harness) remain pending
  before any live BAG fill is attempted.

### 3.2 alpha_crypto_perps — planned / deferred

- **Status:** planned / deferred.
- **Reason for deferral:** the Kraken Futures venue is **blocked for
  live trading** under the current Canadian deployment
  (`docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`). The
  blocker is jurisdictional, not engineering; no Channel 2 work can
  unblock it.
- **What exists today (read-only only):**
  - `chad/market_data/kraken_futures_intel_publisher.py` — 5 min
    public-ticker publisher writing `runtime/kraken_futures_intel.json`.
  - `chad/exchanges/kraken_futures_client.py` — public client.
- **What exists today (dormant scaffold, not wired):**
  - `chad/execution/kraken_futures_adapter.py` — adapter scaffold.
    Never imported by any execution path.
  - `chad/tools/kraken_futures_auth_smoke.py` — authenticated smoke
    that **fails closed** without credentials. Not invoked in
    production.
- **What is forbidden:** configuring the current Canadian Kraken
  account for futures/perps/derivatives/margin; enabling the
  adapter; or running the auth smoke with futures-trading intent.

---

## 4. Execution Pipeline and Gate Chain

The CHAD entry-pipeline is a chain of fail-open gates. v9.3 preserves
the v9.2 gate-chain order and adds BAG LMT discipline at the adapter
boundary. Every gate is documented either in source or in a published
design doc; nothing in the chain is implicit.

### 4.1 Canonical entry gate chain

1. **Quarantine / trust gate.** Symbols listed in
   `runtime/dynamic_caps_quarantine.json` or otherwise quarantined are
   rejected before any strategy logic runs. Quarantine is the highest-
   priority filter and is never overridden.
2. **Strategy enabled gate.** Strategy must be `enabled=True` in the
   active config and present in `dynamic_caps.normalized_weights`.
3. **Signal confidence gate.** The raw `TradeSignal.confidence` must
   meet the strategy-level floor (`min_confidence`) before downstream
   modifiers fire.
4. **Stop-distance sizing.** For `alpha_futures` / `alpha` /
   `alpha_intraday`, when a `tier_profile` is attached, position size
   = `tier_max_risk_per_trade_usd / (ATR × stop_mult × point_value)`.
   No tier profile → legacy sizing fallback.
5. **Session zone gate.** `chad/utils/session.py` classifies the
   current wall-clock instant as primary or secondary. Strategies that
   opt into `primary_session_only` (e.g. `alpha_intraday_micro`) reject
   new entries outside the primary window. Exits are **never** blocked
   on session zones.
6. **R:R gate.** `chad/utils/risk_reward.py` rejects entries with
   `target_estimate / stop_distance < 1.5`. Missing or degenerate
   target estimates fail open.
7. **Liquidity / float-aware gate.** `chad/utils/liquidity.py` reads
   `data/bars/1d/{SYM}.json`, classifies LARGE / STANDARD / THIN /
   UNKNOWN by 20-day ADDV, and blocks THIN equities whose confidence
   < 0.80. UNKNOWN fails open.
8. **Net exposure gate.** The orchestrator-level cap on aggregate
   gross/net exposure across strategies. Symbols already at the cap
   are skipped.
9. **Dynamic risk allocator.** `dynamic_risk_allocator` (per-strategy
   normalized weights) and the per-symbol daily-loss limit gate.
10. **Tier manager.** `chad/risk/tier_risk_enforcer.py` enforces the
    active tier's caps (`max_risk_per_trade_usd`,
    `max_concurrent_positions`, `max_strategy_positions`).
11. **XGB veto gate.** `chad/analytics/ml_veto_predictor.py` runs the
    active XGBoost booster on the candidate signal's features. The
    veto shaves confidence; it never blocks an entry outright (today
    it is structured as a shadow-able modifier).
12. **Winner scaler / sizing logic.** `chad/risk/winner_scaler.py`
    upscales position size on recent winners within the tier budget;
    downscales after stop-outs.
13. **Execution router.** `execution_pipeline._route` chooses the
    asset-class-appropriate intent builder
    (`_intent_from_routed_signal` for the standard path,
    `_intent_from_trade_intent` for the legacy / direct path).
14. **BAG LMT discipline for `sec_type=BAG`.** Adapter-level gate
    introduced in Tier 2 (`2ab03e7`):
    - MKT BAG is coerced to LMT (marker `BAG_MKT_COERCED_TO_LMT`).
    - Missing or non-positive BAG limit price → intent is skipped
      with marker `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`.
    - `limit_price` is hydrated from `meta["spread_spec"].net_debit_estimate`
      (typed `OptionsSpreadSpec`), else from
      `meta["net_debit_estimate"]` (legacy), else the intent is
      skipped.
    - **CAUTION:** contract-dollar debit must **not** be passed
      directly as IBKR BAG `lmtPrice`. See §5.2 and Appendix H.
15. **IBKR adapter contract qualification / submit path.** Qualifies
    the contract via `ib_async.qualifyContractsAsync` (BAG: per-leg
    qualification before `ComboLeg` assembly), guards against `Error
    321` (conId=0), and submits via `ib.placeOrder`. Paper fills are
    simulated by `paper_exec_evidence_writer` rather than waiting for
    a broker fill.

### 4.2 What v9.3 does NOT claim about the gate chain

- The offline `bag_quote_check` (Tier 3A) **is not** enforced at the
  adapter today. The module exists, is unit-tested, and is callable —
  but no execution path consults it. Adapter enforcement is the Tier
  3D work item, which depends on Tier 3C.
- Live BAG execution **is not** authorized. The LMT-discipline gate
  prevents the worst MKT-BAG footgun but does not by itself make BAG
  live-safe.
- The quote probe script `scripts/probe_bag_quotes.py` is **not** part
  of the production execution path. It is an out-of-band Channel 1
  diagnostic that operates on its own `ib_async` connection.

### 4.3 BAG markers reference

The following logging markers are emitted by the adapter and are the
canonical signals operators look for in logs / dashboards when
debugging BAG behaviour:

- `BAG_MKT_COERCED_TO_LMT` — adapter coerced an MKT BAG intent to LMT.
- `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE` — adapter skipped a BAG intent
  because no positive limit price could be hydrated.
- `BAG_INTENT_SKIPPED_UNQUALIFIED_LEG` — pre-existing marker for a leg
  that failed contract qualification (typically `Error 321`).
- `BAG_INTENT_HYDRATED_FROM_SPEC` — adapter hydrated `limit_price`
  from the typed `OptionsSpreadSpec.net_debit_estimate`.
- `BAG_INTENT_HYDRATED_FROM_LEGACY_META` — adapter hydrated
  `limit_price` from the legacy `meta["net_debit_estimate"]` key.

---

## 5. Risk and Governance

v9.3 preserves the v9.2 risk posture in summary form (see Appendix A
for full tier risk profiles) and adds two new governance surfaces:
the XGB Veto Model Promotion Workflow and the BAG Execution Governance
unit rules.

### 5.1 Risk posture summary (carried from v9.2)

- **Paper Epoch 2.** `CHAD_EXECUTION_MODE=paper`. SCR `CONFIDENT`,
  `sizing_factor=1.0`, `paper_only=false`. Account equity
  ≈ $172,578 USD paper (live runtime value as of 2026-05-17
  17:19 UTC).
- **Portfolio risk cap.** `portfolio_risk_cap=$8,628.93` (~5 % of
  paper equity at this snapshot). Per-strategy caps are normalised
  fractions of this cap, surfaced in `runtime/dynamic_caps.json` ->
  `normalized_weights`.
- **Tier architecture v2.** MICRO / STARTER / PRO_GROWTH / SCALE
  tiers each carry their own risk profile (max risk per trade, max
  concurrent positions, primary-session-only flag, etc.). Active
  tier is published into `runtime/tier_state.json`.
- **Profit Lock.** `chad/risk/profit_lock.py` continues to lock
  realized PnL and is published on `chad-profit-lock-publisher.timer`.
- **Stop Bus.** Serialises stop adjustments across the live loop.
- **Position guard drift detector.** Live in
  `chad-reconciliation-publisher`; emits
  `runtime/position_guard_drift.json` (`schema_version=
  position_guard_drift.v1`) every cycle. Operators close stale
  entries via `scripts/close_guard_entry.py`.
- **Quarantine surface.** `runtime/dynamic_caps_quarantine.json` is
  the canonical exclusion list. The position-guard rebuilder
  (`_rebuild_guard_from_paper_ledger`) intentionally does **not**
  consult `reconciliation_state.exclusion_policy` (PERMISSIVE option
  per GAP-028).

### 5.2 BAG Execution Governance — unit rules

The Tier 3B live-readonly probe (2026-05-17, results in
`docs/PHASE_D_ITEM2_BAG_QUOTE_PROBE_RESULTS_2026-05-17.md`) surfaced
the **most important BAG-execution finding to date**: a unit mismatch
between strategy-level `net_debit_estimate` and IBKR BAG `lmtPrice`.

The rules below are **non-negotiable**. They must be incorporated into
the Tier 3C unit-normalisation design and into the Tier 3D adapter
quote enforcement when both land.

- `net_debit_estimate` currently represents **contract-dollar debit
  per spread**.
- IBKR BAG `lmtPrice` expects **per-share option-price debit**.
- Contract-dollar debit must be divided by the 100 options multiplier
  before use as IBKR BAG `lmtPrice`.
- Example: `$350 contract debit → 3.50 IBKR BAG lmtPrice`.

These rules imply a field-level split on `OptionsSpreadSpec` (and any
downstream order intent), recommended in the Tier 3B results doc:

- `net_debit_contract_dollars` — strategy-facing field, in $/contract.
- `net_debit_per_share` — broker-facing field, in $/option-price unit.

**Live BAG remains unauthorized until all of the following are
complete:**

- **Tier 3C** — BAG limit-price unit normalisation design and
  field-level split landed.
- **Tier 3D** — adapter quote enforcement reads only
  `net_debit_per_share` and rejects contract-dollar values.
- **Tier 4** — bracket / OCA wiring or out-of-band failsafe closer.
- **Tier 5** — `spread_id`-aware position guard / reconciliation.
- **Tier 6** — live BAG fill harness (offline `ib_async` shim
  exercising `placeOrder → Trade → orderStatus → fill` round-trip).
- **Tier 7** — operator-initiated live BAG promotion in a separate
  governance change, with `live_readiness.json` re-issued green.

Additional paper-scope BAG governance:

- Paper BAG closes remain `pnl_untrusted=True`. The synthetic 30 %
  credit ratio (`_BAG_CLOSE_CREDIT_RATIO=0.30`) is the canonical
  close-side simulator; quarantine sweeps key off the
  `bag_close_synthetic_credit_ratio_30pct` reason string.
- Synthetic close-credit paper PnL must **not** pollute SCR,
  profit-lock, or live-readiness evidence. The `pnl_untrusted=True`
  flag is the load-bearing filter; downstream consumers must respect
  it.

### 5.3 XGB Veto Model Promotion Workflow

The model-governance layer prevents the weekly retrain from silently
overwriting the live predictor. The workflow as committed in
`5463c63` ("Add XGB veto model promotion workflow"):

**Weekly retrain schedule (per `systemctl list-timers --all`):**

- Service: `chad-xgb-train.service`
- Timer: `chad-xgb-train.timer`
- Next firing: `Sun 2026-05-24 02:00:00 UTC`
- Last firing: `Sun 2026-05-17 02:00:00 UTC`

**Filesystem layout (post-promotion-workflow):**

- Candidate output path:
  `runtime/models/xgb_veto/candidates/<YYYYMMDD_HHMMSS>/`
  containing the candidate `model.json` and `manifest.json`.
- Active runtime path:
  `runtime/models/xgb_veto/current/`
  containing the **promoted** `model.json` and `manifest.json`. The
  manifest carries the new audit triple `promoted_from_candidate`,
  `promoted_at_utc`, `promoted_by`.
- Baseline fallback:
  `shared/models/xgb_veto_model.json` and
  `shared/models/xgb_veto_manifest.json` — tracked-in-git baseline,
  rotated rarely by deliberate operator decision.

**Promotion script:**

- `scripts/promote_xgb_veto.py`
- `--status` prints the active model (source = `current` or
  `baseline`, model_version, trained_at_utc, accuracy, logloss).
- `--promote <candidate-ts>` evaluates the candidate against the
  currently active manifest. Promotion gate (default):
  `new_accuracy >= current_accuracy AND new_logloss <= current_logloss`.
- Operator override:
  `--operator-approve "reason"` — records the reason in the new
  manifest's `promoted_by` / audit fields and bypasses the metric gate.
- Failure mode: refuses to promote if the candidate manifest is
  missing, malformed, or the gate fails without an override.

**Current active model status (live `--status` output):**

```
[promote_xgb_veto] active model (source=baseline)
  model_version : xgb_veto_20260510_020007
  trained_at_utc: 2026-05-10T02:00:07.712729+00:00
  accuracy      : 0.7534
  logloss       : 0.5364
```

**Why the 2026-05-17 regressed retrain was not committed:**

The 2026-05-17 weekly retrain produced metrics that regressed on both
canonical axes:

- `validation_accuracy`: 0.7534 → 0.7112 (−4.22 pp)
- `validation_logloss`: 0.5364 → 0.5953 (+0.0589)

Under the old in-place-overwrite workflow, these regressed weights
would have been the live predictor's weights until the next retrain.
The new workflow's metric gate would have refused promotion of the
regressed candidate; the operator decision was to **not commit** the
dirty tracked artifacts (per
`docs/XGB_VETO_WEEKLY_RETRAIN_DIRTY_TREE_DECISION_2026-05-17.md`) and
to wait for either (a) a future retrain that beats the baseline, or
(b) an explicit operator review and override. The baseline weights
remain the active predictor today.

**Forbidden actions (carried from CLAUDE.md §3 and the XGB plan):**

- Do not commit the regressed model blindly.
- Do not `git checkout` model files without controlled rollback.
- Do not disable `chad-xgb-train.timer` without operator approval.
- Do not modify `chad-xgb-train.service` or `chad-xgb-train.timer`
  (CLAUDE.md §6).

---

## 6. Business Framework / Capital Chassis

v9.3 preserves the business-framework posture from prior SSOTs. No new
funding events, grants, or business decisions are recorded in this
SSOT; if any prior SSOT documented such items, they remain authoritative
in their own doc.

### 6.1 50/30/20 chassis (preserved from prior SSOT scope)

The CHAD capital chassis allocates withdrawals from realized paper PnL
across three buckets — **operator salary**, **business reserve**, and
**system reinvestment** — at the 50/30/20 ratios documented in prior
SSOTs. The `chad/portfolio/withdrawal_manager.py` daemon governs
withdrawal eligibility; `chad-withdrawal-manager.timer` runs at the
documented cadence.

This SSOT does not modify the chassis ratios, the withdrawal eligibility
rules, the operator salary cap, or any business reserve constraint.

### 6.2 Portfolio risk cap

Current runtime value (2026-05-17 17:19 UTC):

- `total_equity`: $172,578.62 USD paper.
- `portfolio_risk_cap`: $8,628.93 USD (~5 % of equity).
- Sum of `normalized_weights` = 1.0 across 16 strategies.

The cap is **dynamic** (the `dynamic_risk_allocator` recomputes it
every cycle from the live `portfolio_snapshot.json` and any active
business overlays) and **not authoritative for live trading** until
`live_readiness.json` reads `ready_for_live: true`.

### 6.3 Dry-run / paper-epoch posture

- `CHAD_EXECUTION_MODE=paper` — unchanged.
- Paper Epoch 2 — SCR `CONFIDENT`, full sizing applied.
- `live_readiness.json` — `ready_for_live: false`. Operator GO has not
  been requested. The Pre-Live Operator Tasks per CLAUDE.md remain
  unchanged from v9.2 (reboot deferred to next kernel update; disk
  cleanup; IBKR latency investigation; full 2,114-test re-verification
  after reboot).

### 6.4 No new live authorization

This SSOT does **not** grant or imply any live execution authorization.
The Live Activation Sequence per CLAUDE.md (set posture from DRY_RUN to
LIVE in a Pending Action, validate LiveGate, monitor first 3 cycles,
confirm broker truth on first fill) still requires an explicit operator
GO that v9.3 does not provide.

### 6.5 Funding / grant / business framework

No funding events or grant decisions are documented in this SSOT. Any
prior documentation in earlier SSOTs (e.g. v8.x business framework
notes) remains authoritative in its own location; v9.3 does not invent
new business-framework values.

---

## 7. Reconciliation and PnL

v9.3 preserves the v9.2 reconciliation summary and adds BAG-specific
clarifications surfaced by Phase D Item 2.

### 7.1 v9.2 reconciliation summary (carried)

- `paper_exec_evidence_writer.py` stamps an attribution record for
  every paper fill, keyed by `(strategy, symbol, intent_id)`.
- `trade_closer` matches paper closes to opens via the `spread_id` /
  `intent_id` chain and emits `trade_closer_state.queues` updates.
- `position_guard` keys on `(strategy, symbol)` for non-BAG fills and
  is the canonical "what does CHAD think it holds" surface.
- `reconciliation_publisher` reconciles the guard against broker truth
  every cycle. The drift detector emits
  `runtime/position_guard_drift.json` on any divergence.
- `lifecycle_truth_publisher` writes the canonical lifecycle ledger.
- `broker_sync|*` keys remain CORRECT (per the 2026-05-09 audit memory:
  `audit_2026_05_09_positions_snapshot_stale.md`); the
  `positions_snapshot.json` staleness issue is independent of broker
  truth and does not affect the reconciliation path.

### 7.2 BAG-specific reconciliation status

- **BAG paper opens** are **trusted** when the simulator context is
  complete (i.e. the strategy emitted a populated `bag_legs` array, a
  valid `spread_id`, and a positive `net_debit_estimate`). The open
  records use `extra["pnl_untrusted"]=False`.
- **BAG paper closes** use the synthetic close-credit ratio
  (`_BAG_CLOSE_CREDIT_RATIO=0.30`) and are **always** marked
  `extra["pnl_untrusted"]=True` with reason
  `bag_close_synthetic_credit_ratio_30pct`. Quarantine sweeps
  (e.g. `test_quarantine_20260511_alpha_options.py`) key off that
  reason string.
- **`spread_id`-aware position tracking is still pending.** Today the
  position guard keys on `(strategy, symbol)`. Two concurrent SPY
  BAGs from `alpha_options` would collide under a single guard slot
  and clobber each other's state. Tier 5 of the BAG hardening ladder
  is the cross-cutting reconciler change required to fix this.
- Until Tier 5 lands, **multiple concurrent SPY BAGs can collide**
  under the strategy/symbol guard. Operators should not assume the
  guard's BAG state is per-spread.
- **XGB model version and promotion metadata** are tracked through
  the active manifest (`runtime/models/xgb_veto/current/manifest.json`
  when populated, else `shared/models/xgb_veto_manifest.json`). Any
  attribution / replay that crosses a model rotation can look up the
  exact version via `manifest.model_version`.
- **Weekly retrains no longer auto-overwrite tracked active model
  files.** Per the promotion workflow, the trainer's output target is
  `runtime/models/xgb_veto/candidates/<UTC ts>/`; the tracked baseline
  in `shared/models/` is rotated only by deliberate operator action.

---

## 8. Intelligence Layer Feed Catalog

| Feed | Runtime file | Schema | Cadence | Timer/Service | Status | Notes |
|:---|:---|:---|:---|:---|:---|:---|
| News intel (catalyst news) | `runtime/news_intel.json` | `news_intel.v1` | ~30 min | `chad-news-intel-refresh.timer` / `chad-news-intel-refresh.service` | ok | 25 symbols processed; 5 with catalyst; 4 high-strength; provider primary=polygon, fallback=yahoo. Consumed by `catalyst_gate.py`. |
| Relative strength | `runtime/relative_strength.json` | `relative_strength.v1` | daily (computed from `data/bars/1d`) | `chad-rs-refresh.timer` / `chad-rs-refresh.service` | ok | 23 computed; 3 strong, 6 weak, 16 neutral. ±0.10 confidence modifier via `rs_gate.py`. |
| Intraday RVOL scanner | `runtime/volume_scan.json` | `volume_scan.v1` | 5 min | `chad-volume-scan.timer` / `chad-volume-scan.service` | ok | 25 scanned; 0 high RVOL at audit time; `provider_status=fallback_rolling_1m` (rolling-1m fallback active when Polygon snapshot unavailable). ±0.05 confidence modifier for `alpha_intraday` only. |
| Crypto derivatives | `runtime/crypto_derivatives.json` | `crypto_derivatives.v1` | 5 min | `chad-crypto-derivatives-refresh.timer` / `chad-crypto-derivatives-refresh.service` | ok | 3 mapped symbols (BTC-USD, ETH-USD, SOL-USD); 1 long-crowded, 1 short-crowded. Consumed by `crypto_signal_filter.py`. Source: Kraken Futures public tickers. |
| Futures roll calendar | `runtime/futures_roll_state.json` | `futures_roll_state.v1` | 24 h | `chad-futures-roll-refresh.timer` / `chad-futures-roll-refresh.service` | ok | 9 tracked (MES, MNQ, MCL, MGC, ZN, ZB, M6E, MYM, M2K); 4 supported (quarterly equity micros), 5 unsupported (informational only). 0 in warning window at audit. Consumed by `roll_gate.py`. |
| Options Greeks metadata | `runtime/options_greeks.json` | `options_greeks.v1` | daily | `chad-options-greeks-refresh.timer` / `chad-options-greeks-refresh.service` | ok | TTL aligned with daily cadence (commit `759fd32`). SPY only at audit time; 3 expiries; 21 strikes; synthetic Black–Scholes from VIX + chain cache. Metadata-only annotation on `alpha_options` signals via `options_greeks_gate.py`. |
| Kraken Futures public intel | `runtime/kraken_futures_intel.json` | `kraken_futures_intel.v1` | 5 min | `chad-kraken-futures-intel-refresh.timer` / `chad-kraken-futures-intel-refresh.service` | ok | 306 perps published; 14 long-crowded, 19 short-crowded; 3 mapped to CHAD crypto universe. **Public, read-only intel.** **Not live trading.** |
| FMP earnings intel | `runtime/earnings_intel.json` | `earnings_intel.v1` | 6 h | `chad-fmp-earnings-intel-refresh.timer` / `chad-fmp-earnings-intel-refresh.service` | partial | 25 symbols processed; 1 next-earnings, 4 price-targets, 4 analyst-estimates, 5 sec-filings. `partial` is the **expected** status under the current FMP plan; ETFs and unsupported tickers fail open. Dashboard-visible. **No strategy gate.** |
| Dynamic universe candidates | `runtime/dynamic_universe_candidates.json` | `dynamic_universe_candidates.v1` | 5 min | `chad-dynamic-universe-scanner-refresh.timer` / `chad-dynamic-universe-scanner-refresh.service` | ok | 25 candidates; top 10 surfaced in §2.6. **Observation mode only.** **Does not replace `runtime/universe.json`.** **No strategy wiring.** Promotion logic intentionally deferred to v2. |
| Event risk | `runtime/event_risk.json` | `event_risk.v1` | 10 min | `chad-event-risk.timer` / `chad-event-risk.service` | present | Top-level market-wide risk score / severity / `next_event` / `windows`. Operator calendar at `config/event_calendar.json` is loaded. Consumed at the orchestrator level, not per-symbol. |
| Setup-family expectancy | `runtime/setup_family_expectancy.json` | `setup_family_expectancy.v2` | 24 h | `chad-setup-expectancy.timer` / `chad-setup-expectancy.service` | within cadence | 6 families processed across `alpha`, `alpha_futures`, `alpha_intraday`. 1,574 trades processed; 0 skipped corrupt. Last trade at audit: 2026-05-15T02:47:58Z. |

### 8.1 Dynamic universe candidates — clarifications

- **Observation mode only.** The scanner publishes a ranked list of
  candidates. No strategy reads this list. No code under
  `chad/strategies/`, `chad/execution/`, `chad/risk/`, or
  `chad/core/` imports the helper or the runtime file (enforced by
  `chad/tests/test_dynamic_universe_candidates_context.py`).
- **Does not replace `runtime/universe.json`.** The canonical writer of
  `runtime/universe.json` remains `chad.analytics.universe_builder`.
  The scanner only **reads** the active universe (via
  `chad.utils.universe_provider.load_active_universe`); it does not
  write it.
- **No strategy wiring in v1.** Phase D Item 1 explicitly defers any
  scanner→strategy wiring to a v2 promotion-logic design that has not
  yet been authorised. The §9 observation period in the scanner design
  doc (≥ 5 trading days of side-by-side comparison) must complete
  before v2 is even scheduled.

### 8.2 Earnings intel — clarifications

- **FMP stable endpoints.** Provider `fmp_stable`. Endpoints in use:
  `earnings-calendar`, `price-target-consensus`,
  `analyst-estimates annual`, `sec-filings-search`. FMP news is
  intentionally **not** consumed under the current plan.
- **Partial status expected.** `status=partial` is the **expected**
  state when ETFs (SPY, QQQ, IWM, etc.) or other unsupported tickers
  appear in the universe; per-symbol `provider_errors` arrays are
  populated but the publisher succeeds.
- **Dashboard-visible.** Surfaced via
  `chad.intel.strategy_intelligence._load_earnings_intel_context` and
  the `earnings_intel` block on
  `chad.dashboard.api.StateBuilder._intelligence()`.
- **No strategy gate.** No file under `chad/strategies/` reads
  `earnings_intel.json` (enforced by
  `chad/tests/test_earnings_intel_context.py`). The dynamic scanner
  reads it for the `earnings_warning_count` warning band only.

### 8.3 Kraken Futures intel — clarifications

- **Public read-only intel.** The publisher uses Kraken's public
  `tickers` endpoint; no authenticated calls and no order submission.
- **306 perps** at audit time (`summary.perps_total=306`). Top open
  interest: PF_XBTUSD, PF_ETHUSD, PF_SOLUSD, PF_XRPUSD, PF_XAUTUSD.
- **Not live trading.** The corresponding adapter scaffold
  (`chad/execution/kraken_futures_adapter.py`) is dormant; the
  authenticated smoke (`chad/tools/kraken_futures_auth_smoke.py`) fails
  closed without credentials and is not invoked in production.
- **Mapped symbols.** Three (BTC-USD, ETH-USD, SOL-USD) project onto
  CHAD's existing `alpha_crypto` universe; the rest are surfaced in
  summary metadata only.

---

## 9. Telegram and Bot

The Telegram operator interface and the bot surface are **unchanged
from v9.2** and were **not re-audited** in this v9.3 cut. The v9.2
description remains authoritative for:

- Telegram bot command surface.
- Telegram alert routing for SCR transitions, profit-lock events,
  drift detector alerts, and live readiness transitions.
- Operator command authorisation (Pending Action submission via
  Telegram + dashboard confirmation).
- Rate-limiting and dedup logic for repeated alerts.

If the operator wants the Telegram surface re-audited, that should be
scheduled as a separate Channel 2 audit pass and recorded in its own
doc; v9.3 deliberately does not invent or restate Telegram-surface
facts that were not directly read in this audit.

---

## 10. Dashboard and Operator Visibility

### 10.1 `/intelligence` endpoint

The Flask dashboard's `/intelligence` endpoint (served via
`chad.dashboard.api.StateBuilder._intelligence()`) is the canonical
operator surface for runtime intelligence. v9.3 adds two new
read-only blocks to this endpoint:

- **`earnings_intel` block** (new in commit `fcb8656`). Surfaces the
  FMP earnings publisher's payload: per-symbol
  `days_to_next_earnings`, `price_target_consensus`,
  `annual_eps_avg_estimate`, `latest_filing_type`, plus the
  publisher's `summary` and `provider_status`.
- **`dynamic_universe_candidates` block** (new in commit `c4144f5`).
  Surfaces `freshness`, `status`, `summary`, the top-10
  `top_candidates`, `ts_utc`, `ttl_seconds`, and `source_provider`.

Both blocks are read-only. Both helpers are fail-open: missing /
malformed / stale / partial / empty payloads never raise; they
degrade gracefully to an `unavailable` indicator on the dashboard.

### 10.2 Existing intelligence blocks (preserved from v9.2)

- `regime_state`, `event_risk`, `macro_state`, `choppy_regime` —
  market-wide posture.
- `news_intel`, `relative_strength`, `volume_scan`,
  `crypto_derivatives`, `futures_roll_state`, `options_greeks` —
  per-symbol or per-instrument intelligence.
- `kraken_futures_intel` — public read-only crypto perps intel.
- `setup_family_expectancy` — per-strategy / per-setup expectancy
  metrics.

### 10.3 Current limitations

- **No strategy wiring from dashboard views.** The dashboard is a
  read-only operator surface. No panel mutates strategy config,
  posture, or risk caps. All such mutations remain Pending Actions per
  CLAUDE.md §3.
- **Dynamic universe candidates is observation-only.** The scanner
  panel is for operator side-by-side comparison with the current
  static-then-runtime universe during the design doc's §9 observation
  period; it does not feed routing or sizing.
- **Earnings intel is metadata-only.** No earnings-proximity guard is
  wired. The dynamic scanner emits a `warnings=[earnings_within_7d]`
  badge on candidates with `days_to_next_earnings ∈ [0, 7]`, but no
  strategy gates on it.
- **Telegram surface is not modified.** Dashboard alerting paths into
  Telegram are unchanged from v9.2.

---

## 11. Services and Timers

### 11.1 CHAD timer / service inventory (live `systemctl list-timers --all`)

The following timers were observed active on the host at audit time
(2026-05-17 ~17:22 UTC). Times are UTC. Cadence is implicit in the
next/last triggers.

| Timer | Service | Last trigger | Next trigger | Cadence |
|:---|:---|:---|:---|:---|
| `chad-trade-closer.timer` | `chad-trade-closer.service` | 2026-05-17 17:21:34 | 2026-05-17 17:22:30 | ~1 min |
| `chad-ibkr-price-refresh.timer` | `chad-ibkr-price-refresh.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-options-monitor.timer` | `chad-options-monitor.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-profit-lock-publisher.timer` | `chad-profit-lock-publisher.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-regime-booster.timer` | `chad-regime-booster.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-scr-sync.timer` | `chad-scr-sync.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-action-applier.timer` | `chad-action-applier.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-execq-publisher.timer` | `chad-execq-publisher.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-lifecycle-truth-publisher.timer` | `chad-lifecycle-truth-publisher.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-mutation-state-publisher.timer` | `chad-mutation-state-publisher.service` | 2026-05-17 17:21:36 | 2026-05-17 17:22:36 | 1 min |
| `chad-feed-watchdog.timer` | `chad-feed-watchdog.service` | 2026-05-17 17:21:36 | 2026-05-17 17:23:36 | 2 min |
| `chad-event-risk.timer` | `chad-event-risk.service` | 2026-05-17 17:14:14 | 2026-05-17 17:24:14 | 10 min |
| `chad-macro-state.timer` | `chad-macro-state.service` | 2026-05-17 17:14:14 | 2026-05-17 17:24:14 | 10 min |
| `chad-paper-trade-exec.timer` | `chad-paper-trade-exec.service` | 2026-05-17 17:14:14 | 2026-05-17 17:24:14 | 10 min |
| `chad-rebalance-auto-executor-paper.timer` | `chad-rebalance-auto-executor-paper.service` | 2026-05-17 17:14:14 | 2026-05-17 17:24:14 | 10 min |
| `chad-operator-intent-refresh.timer` | `chad-operator-intent-refresh.service` | 2026-05-17 17:14:14 | 2026-05-17 17:24:14 | 10 min |
| `chad-burnin-check.timer` | `chad-burnin-check.service` | 2026-05-17 17:14:14 | 2026-05-17 17:24:14 | 10 min |
| `chad-business-phase.timer` | `chad-business-phase.service` | 2026-05-17 16:54:54 | 2026-05-17 17:24:54 | 30 min |
| `chad-choppy-regime.timer` | `chad-choppy-regime.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-crypto-derivatives-refresh.timer` | `chad-crypto-derivatives-refresh.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-expectancy-tracker.timer` | `chad-expectancy-tracker.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-health-monitor.timer` | `chad-health-monitor.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-ibkr-broker-events.timer` | `chad-ibkr-broker-events.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-ibkr-paper-fill-harvester.timer` | `chad-ibkr-paper-fill-harvester.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-portfolio-snapshot.timer` | `chad-portfolio-snapshot.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-positions-snapshot.timer` | `chad-positions-snapshot.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-reconciliation-publisher.timer` | `chad-reconciliation-publisher.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-symbol-blocker.timer` | `chad-symbol-blocker.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-tier-manager.timer` | `chad-tier-manager.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-calendar-state-publisher.timer` | `chad-calendar-state-publisher.service` | 2026-05-17 17:20:14 | 2026-05-17 17:25:14 | 5 min |
| `chad-dynamic-universe-scanner-refresh.timer` | `chad-dynamic-universe-scanner-refresh.service` | 2026-05-17 17:21:14 | 2026-05-17 17:26:14 | 5 min |
| `chad-strategy-intelligence-refresh.timer` | `chad-strategy-intelligence-refresh.service` | 2026-05-17 17:11:41 | 2026-05-17 17:26:41 | 15 min |
| `chad-winner-scaler.timer` | `chad-winner-scaler.service` | 2026-05-17 17:11:41 | 2026-05-17 17:26:41 | 15 min |
| `chad-kraken-futures-intel-refresh.timer` | `chad-kraken-futures-intel-refresh.service` | 2026-05-17 17:21:54 | 2026-05-17 17:26:54 | 5 min |
| `chad-volume-scan.timer` | `chad-volume-scan.service` | 2026-05-17 17:21:54 | 2026-05-17 17:26:54 | 5 min |
| `chad-disk-guard.timer` | `chad-disk-guard.service` | 2026-05-17 16:59:14 | 2026-05-17 17:29:14 | 30 min |
| `chad-live-readiness.timer` | `chad-live-readiness.service` | 2026-05-17 17:21:14 | 2026-05-17 17:31:14 | 10 min |
| `chad-ibkr-paper-ledger-watcher.timer` | `chad-ibkr-paper-ledger-watcher.service` | 2026-05-17 17:16:54 | 2026-05-17 17:31:54 | 15 min |
| `chad-lifecycle-replay-engine.timer` | `chad-lifecycle-replay-engine.service` | 2026-05-17 17:07:11 | 2026-05-17 17:37:11 | 30 min |
| `chad-portfolio-artifacts.timer` | `chad-portfolio-artifacts.service` | 2026-05-17 16:42:14 | 2026-05-17 17:42:14 | 1 h |
| `chad-news-intel-refresh.timer` | `chad-news-intel-refresh.service` | 2026-05-17 17:19:34 | 2026-05-17 17:49:34 | 30 min |
| `chad-universe-refresh.timer` | `chad-universe-refresh.service` | 2026-05-17 17:01:14 | 2026-05-17 18:04:24 | ~1 h |
| `chad-short-interest-refresh.timer` | `chad-short-interest-refresh.service` | 2026-05-17 12:50:24 | 2026-05-17 18:50:24 | 6 h |
| `chad-trends-refresh.timer` | `chad-trends-refresh.service` | 2026-05-17 14:55:09 | 2026-05-17 18:55:09 | 4 h |
| `chad-reddit-sentiment-refresh.timer` | `chad-reddit-sentiment-refresh.service` | 2026-05-17 17:02:34 | 2026-05-17 19:02:34 | 2 h |
| `chad-withdrawal-manager.timer` | `chad-withdrawal-manager.service` | 2026-05-17 13:03:34 | 2026-05-17 19:03:34 | 6 h |
| `chad-weekly-report.timer` | `chad-weekly-report.service` | 2026-05-10 20:00:00 | 2026-05-17 20:00:00 | 7 d |
| `chad-futures-roll-refresh.timer` | `chad-futures-roll-refresh.service` | 2026-05-16 21:50:34 | 2026-05-17 21:50:34 | 24 h |
| `chad-fmp-earnings-intel-refresh.timer` | `chad-fmp-earnings-intel-refresh.service` | 2026-05-17 17:01:54 | 2026-05-17 23:01:54 | 6 h |
| `chad-options-greeks-refresh.timer` | `chad-options-greeks-refresh.service` | 2026-05-16 23:48:44 | 2026-05-17 23:48:44 | 24 h |
| `chad-burnin-daily-summary.timer` | `chad-burnin-daily-summary.service` | 2026-05-16 23:59:05 | 2026-05-17 23:59:00 | 24 h |
| `chad-equity-history.timer` | `chad-equity-history.service` | 2026-05-16 23:59:05 | 2026-05-17 23:59:00 | 24 h |
| `chad-sqlite-retention.timer` | `chad-sqlite-retention.service` | 2026-05-11 00:02:17 | 2026-05-18 00:01:56 | 7 d |
| `chad-ibkr-daily-bars-refresh.timer` | `chad-ibkr-daily-bars-refresh.service` | 2026-05-17 02:45:05 | 2026-05-18 02:45:00 | 24 h |
| `chad-proofs-cleanup.timer` | `chad-proofs-cleanup.service` | 2026-05-17 03:20:05 | 2026-05-18 03:20:00 | 24 h |
| `chad-options-chain-refresh.timer` | `chad-options-chain-refresh.service` | 2026-05-15 12:30:00 | 2026-05-18 12:30:00 | weekdays |
| `chad-advisory-pre-market.timer` | `chad-advisory-pre-market.service` | 2026-05-15 12:45:00 | 2026-05-18 12:45:00 | weekdays |
| `chad-rs-refresh.timer` | `chad-rs-refresh.service` | 2026-05-17 12:52:44 | 2026-05-18 12:52:44 | 24 h |
| `chad-morning-brief.timer` | `chad-morning-brief.service` | 2026-05-15 13:00:04 | 2026-05-18 13:00:00 | weekdays |
| `chad-micro-eod-flatten.timer` | `chad-micro-eod-flatten.service` | 2026-05-15 19:30:00 | 2026-05-18 19:30:00 | weekdays |
| `chad-daily-report.timer` | `chad-daily-report.service` | 2026-05-15 21:35:04 | 2026-05-18 21:35:00 | weekdays |
| `chad-setup-expectancy.timer` | `chad-setup-expectancy.service` | 2026-05-16 01:00:05 | 2026-05-19 01:00:00 | 24 h+ |
| `chad-xgb-train.timer` | `chad-xgb-train.service` | 2026-05-17 02:00:00 | 2026-05-24 02:00:00 | 7 d (Sun 02:00 UTC) |

### 11.2 Phase B publishers (live)

All six Phase B publishers from v9.2 remain live and on their published
cadences:

- `chad-news-intel-refresh.timer` — 30 min.
- `chad-rs-refresh.timer` — daily.
- `chad-volume-scan.timer` — 5 min.
- `chad-crypto-derivatives-refresh.timer` — 5 min.
- `chad-futures-roll-refresh.timer` — 24 h.
- `chad-options-greeks-refresh.timer` — 24 h (TTL aligned per `759fd32`).

### 11.3 v9.3 new timers / services

- `chad-kraken-futures-intel-refresh.timer` — 5 min (Group A).
- `chad-fmp-earnings-intel-refresh.timer` — 6 h (Group B).
- `chad-dynamic-universe-scanner-refresh.timer` — 5 min (Group C).
- `chad-xgb-train.timer` — already existed; behaviour change in
  Group D: weekly retrain now writes candidates under
  `runtime/models/xgb_veto/candidates/<UTC ts>/` instead of
  overwriting the tracked active files. **Note:** this behaviour
  change will be observable on Sun 2026-05-24 02:00 UTC firing; at v9.3
  cut time (2026-05-17), the new behaviour is expected based on the
  Phase 2 implementation in `5463c63`. Verification will be re-asserted
  after the 2026-05-24 firing.

### 11.4 Timer-name correctness note

Names in this section are taken verbatim from
`systemctl list-timers --all`. Any prior SSOT that uses a different
short-name for a timer should be treated as superseded by these names.

---

## 12. Data and Storage

### 12.1 New major files since v9.2

The following files are new since `5392fc7` (v9.2 cut). All paths are
relative to `/home/ubuntu/chad_finale`.

**Exchanges and market data:**

- `chad/exchanges/kraken_futures_client.py` — Kraken Futures public
  HTTP client (read-only). Tracked.
- `chad/market_data/kraken_futures_intel_publisher.py` — 5 min public
  intel publisher. Tracked.
- `chad/market_data/fmp_earnings_intel_publisher.py` — 6 h FMP
  earnings/analyst publisher. Tracked.
- `chad/market_data/dynamic_universe_scanner.py` — 5 min dynamic
  universe candidate scorer. Tracked.

**Execution and options:**

- `chad/execution/kraken_futures_adapter.py` — adapter scaffold
  (DORMANT, never wired). Tracked.
- `chad/options/spread_spec.py` — frozen `OptionsSpreadSpec` dataclass.
  Tracked.
- `chad/options/quote_check.py` — offline `bag_quote_check` validator
  (Tier 3A). Tracked.

**Analytics and tools:**

- `chad/analytics/ml_veto_predictor.py` — updated to load
  runtime-current first with baseline fallback (Group D).
- `chad/analytics/train_xgb_model.py` — updated to write candidates
  to `runtime/models/xgb_veto/candidates/<UTC ts>/` (Group D).
- `chad/tools/kraken_futures_auth_smoke.py` — fail-closed auth smoke.
  Tracked.

**Scripts:**

- `scripts/promote_xgb_veto.py` — XGB veto promotion CLI. Tracked.
- `scripts/preview_bag_intent.py` — BAG dry-run preview CLI
  (Tier 1). Tracked.
- `scripts/probe_bag_quotes.py` — BAG live-readonly quote probe
  (Tier 3B). Tracked.

**Runtime paths (untracked / runtime-only):**

- `runtime/dynamic_universe_candidates.json` — generated; runtime-only.
  Gitignored via `runtime/` rule in `.gitignore`.
- `runtime/earnings_intel.json` — generated; runtime-only.
- `runtime/kraken_futures_intel.json` — generated; runtime-only.
- `runtime/models/xgb_veto/candidates/` — new directory tree;
  candidates written by the weekly retrain; gitignored via `runtime/`.
- `runtime/models/xgb_veto/current/` — new directory tree; populated
  only by `scripts/promote_xgb_veto.py`; gitignored via `runtime/`.

### 12.2 Tracked vs untracked classification

| Path | Tracked? | Notes |
|:---|:---|:---|
| `chad/exchanges/kraken_futures_client.py` | yes | Source. |
| `chad/execution/kraken_futures_adapter.py` | yes | Source (DORMANT). |
| `chad/market_data/kraken_futures_intel_publisher.py` | yes | Source. |
| `chad/tools/kraken_futures_auth_smoke.py` | yes | Source. |
| `chad/market_data/fmp_earnings_intel_publisher.py` | yes | Source. |
| `chad/market_data/dynamic_universe_scanner.py` | yes | Source. |
| `chad/options/spread_spec.py` | yes | Source. |
| `chad/options/quote_check.py` | yes | Source. |
| `chad/analytics/ml_veto_predictor.py` | yes | Source (modified in `5463c63`). |
| `chad/analytics/train_xgb_model.py` | yes | Source (modified in `5463c63`). |
| `scripts/promote_xgb_veto.py` | yes | Script. |
| `scripts/preview_bag_intent.py` | yes | Script. |
| `scripts/probe_bag_quotes.py` | yes | Script. |
| `runtime/dynamic_universe_candidates.json` | no | Runtime-only; gitignored. |
| `runtime/earnings_intel.json` | no | Runtime-only; gitignored. |
| `runtime/kraken_futures_intel.json` | no | Runtime-only; gitignored. |
| `runtime/models/xgb_veto/candidates/` | no | Runtime-only directory tree; gitignored. |
| `runtime/models/xgb_veto/current/` | no | Runtime-only directory tree; gitignored. |
| `shared/models/xgb_veto_model.json` | yes | Baseline (rotated rarely by deliberate operator action). |
| `shared/models/xgb_veto_manifest.json` | yes | Baseline manifest. |

### 12.3 Schema versions in use

- `scr_state.v1`
- `live_readiness_state.v1`
- `tier_state.v2`
- `news_intel.v1`
- `relative_strength.v1`
- `volume_scan.v1`
- `crypto_derivatives.v1`
- `futures_roll_state.v1`
- `options_greeks.v1`
- `kraken_futures_intel.v1`
- `earnings_intel.v1`
- `dynamic_universe_candidates.v1`
- `event_risk.v1`
- `setup_family_expectancy.v2`
- `position_guard_drift.v1`
- `xgb_manifest.v2`

---

## 13. Change Log from v9.2

Full list of commits since `5392fc7` (v9.2 cut), in reverse
chronological order, with workstream grouping from Section 0.4.

| # | Commit | Group | Title |
|---:|:---|:---|:---|
| 1 | `72f361f` | F | Docs: close pre-SSOT elite upgrade gaps |
| 2 | `a9fb34a` | E | Fix BAG quote probe async contract qualification |
| 3 | `9fb163b` | E | Add BAG quote probe script |
| 4 | `4523032` | E | Docs: design BAG live quote probe Tier 3B |
| 5 | `a5c5b35` | E | Add offline BAG quote check module |
| 6 | `c620119` | E | Docs: design BAG quote check Tier 3 |
| 7 | `2ab03e7` | E | Enforce BAG limit-order discipline |
| 8 | `c7c7406` | E | Docs: design BAG hardening Tier 2 LMT discipline |
| 9 | `51166d4` | E | Add Phase D BAG spread spec hardening |
| 10 | `5463c63` | D | Add XGB veto model promotion workflow |
| 11 | `e7e083e` | D | Docs: plan XGB veto model artifact hygiene |
| 12 | `be0155d` | D | Docs: record XGB veto retrain dirty-tree decision |
| 13 | `c4144f5` | C | Surface dynamic universe candidates in dashboard context |
| 14 | `ce0c4d1` | C | Add Phase D dynamic universe candidate scanner |
| 15 | `99a5084` | C | Docs: design Phase D dynamic universe scanner |
| 16 | `db077c8` | F | Docs: close Phase C and lock Phase D readiness |
| 17 | `fcb8656` | B | Surface earnings intelligence in dashboard context |
| 18 | `aca5a1f` | B | Docs: record FMP earnings intelligence status |
| 19 | `a98a165` | B | Add FMP earnings and analyst intelligence publisher |
| 20 | `9eaad5f` | A/F | Docs: mark Kraken Futures blocked for Canadian deployment |
| 21 | `2dd0f24` | A/F | Docs: lock Phase C status after Kraken Futures scaffold |
| 22 | `c9da3b9` | A | Add Kraken Futures authenticated smoke scaffold |
| 23 | `69d3a95` | A | Add Phase C Kraken Futures adapter scaffold |
| 24 | `759fd32` | A | Align options Greeks TTL with daily refresh cadence |
| 25 | `0204cba` | A | Add Phase C Kraken Futures public intel publisher |
| 26 | `42daa1f` | A/F | Document Phase C IBKR DOM entitlement blocker |

Total: 26 commits since `5392fc7`. (The change-log table reads back the
exact `git log 5392fc7..HEAD --oneline` content captured in this audit.)

---

## 14. Known Issues and Residual Risks

### 14.1 v9.2 risks (summarised, carried forward)

- Live execution remains paper-only. SCR `CONFIDENT` band reached but
  not propagated to live posture. Pre-Live Operator Tasks (reboot,
  disk cleanup, IBKR latency investigation, full 2,114-test
  re-verification after reboot) remain open.
- Disk pressure: prune backup archives below 75 % when convenient.
- IBKR Gateway latency: dangerous (>750 ms) classification still
  flagged; under investigation per CLAUDE.md Pre-Live Operator Tasks.
- positions_snapshot.json staleness (`audit_2026_05_09`): the
  snapshot has been stale since 2026-04-03 because no active writer
  exists; `broker_sync|*` reconciliation entries are CORRECT, but
  `positions_truth.json` reads the stale snapshot and mis-reports
  broker holdings. This is observability-only and does not affect
  paper execution.

### 14.2 New v9.3 risks

#### RISK-BAG-01 — BAG limit-price unit mismatch

- **Symptom:** `--limit-price 350` failed; `--limit-price 3.97`
  passed (Tier 3B live-readonly probe, 2026-05-17).
- **Root cause:** `net_debit_estimate` represents **contract-dollar
  debit per spread**, while IBKR BAG `lmtPrice` expects **per-share
  option-price debit**. The 100× options multiplier is the missing
  conversion.
- **Severity:** HIGH for any live BAG path. Currently mitigated by
  the fact that **no live BAG path is authorised**.
- **Action:** Tier 3C unit-normalisation design — introduce a
  `net_debit_contract_dollars` / `net_debit_per_share` field-level
  split on `OptionsSpreadSpec`; the adapter (when Tier 3D wires it)
  must consume `net_debit_per_share` only.

#### RISK-BAG-02 — no real bid/ask broker-mid yet

- **Symptom:** Tier 3B live-readonly probe observed delayed last
  prices on both SPY legs; bid/ask unavailable; IBKR `Error 10091`
  ("additional market data subscription required for some option
  quote fields").
- **Severity:** MEDIUM. Adapter-level quote enforcement cannot be
  trusted with last-price fallback alone.
- **Action:** Future market-data / quote path — either an additional
  IBKR market-data subscription (Channel 3 decision) or a
  best-effort `reqMktData` path that gates on bid/ask presence
  before enforcing.

#### RISK-BAG-03 — no bracket/OCA or fail-safe close

- **Symptom:** The only exit mechanism for `alpha_options` BAG today
  is the strategy's `max_hold_seconds=3600` SELL emitted by
  `live_loop.run_once`. If `live_loop` is down (gateway disconnect,
  process restart), a live BAG would sit unmanaged.
- **Severity:** HIGH for any live BAG path. Currently mitigated by
  the fact that **no live BAG path is authorised**.
- **Action:** Tier 4 — choose between native IBKR bracket
  (parent BAG + child stop/target with `parentId`/`transmit`) and a
  cron-driven failsafe closer (independent of `live_loop`).

#### RISK-BAG-04 — no spread_id-aware position tracking

- **Symptom:** `position_guard` keys on `(strategy, symbol)`.
  Multiple concurrent SPY BAGs from `alpha_options` would share a
  single guard slot and clobber each other's state.
- **Severity:** HIGH once concurrent BAGs are permitted.
  Currently mitigated by `alpha_options` emitting at most one open
  BAG per symbol at a time in paper mode.
- **Action:** Tier 5 — migrate guard / reconciler keying to
  `(strategy, symbol, spread_id)`. Cross-cutting change touching
  `position_guard.py`, `trade_closer.py`, `live_loop.py`,
  `scripts/close_guard_entry.py`, and every guard test. Persisted
  state migration required.

#### RISK-XGB-01 — regressed retrain blocked

- **Symptom:** 2026-05-17 weekly retrain regressed (accuracy
  0.7534 → 0.7112, logloss 0.5364 → 0.5953). Under the pre-Group-D
  workflow, this would have silently become the live predictor.
- **Severity:** Resolved by the candidate/promotion workflow in
  `5463c63`. The regressed weights live only as untracked
  timestamped backups; the baseline remains active.
- **Action:** None required. The workflow now prevents silent
  promotion; future regressions will fail the metric gate without
  `--operator-approve`.

### 14.3 External blockers (pinned)

Per `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`:

- **Kraken Futures live trading is BLOCKED** for the current Canadian
  deployment. Channel 3 only. Permanent until a separate operator-led
  jurisdiction unlock.
- **Coinglass is BLOCKED** until a paid API plan/key is procured.
  Channel 3 only. **No keyless scaffold is permitted.**
- **IBKR DOM is BLOCKED / PENDING** until live MES + MNQ DOM rows
  return from the Gateway during regular CME hours without an
  `Error 354`. Channel 1 (+3) — retry at approximately Sunday
  6 PM ET / 22:00 UTC.

### 14.4 Dynamic scanner risks

- **Observation-only.** The scanner publishes a ranked candidate list
  that is **not** consumed by any strategy. Risk of accidental wiring
  is mitigated by `chad/tests/test_dynamic_universe_candidates_context.py`.
- **Does not replace `runtime/universe.json`.** Risk of accidental
  promotion is mitigated by §8 design-doc safety rules and the
  absence of any writer in the scanner module.
- **Promotion logic pending.** Any v2 promotion path must respect
  the static `config/universe.json` fallback, the quarantine surface,
  and the strict-allow-list strategies (`omega_vol`,
  `alpha_options`).

---

## 15. Roadmap and Live Promotion Track

### 15.1 Phase status

| Phase | Status | Notes |
|:---|:---|:---|
| **Phase A — Signal quality** | ✅ Complete | 5/5 items (stop-distance sizing; session-zone gating; R/R floor; setup-family tagging; float-aware liquidity gate). |
| **Phase B — Intelligence feeds** | ✅ Complete / live | 6/6 publishers + FMP stable scaffold + FMP earnings publisher (Group B). All publishers fresh by their declared TTL. |
| **Phase C — Cross-venue / depth intelligence** | ✅ Max-buildable / closed with external blockers pinned | See §15.2. |
| **Phase D — Universe + complex execution hardening** | Active, paper-scope hardened | See §15.3. |

### 15.2 Phase C item status

- **C1A — Kraken Futures public intel** — ✅ Complete / live (Group A
  `0204cba`).
- **C1B — Kraken Futures adapter scaffold** — ✅ Complete / DORMANT
  (Group A `69d3a95`). Never wired.
- **C1C — Kraken Futures authenticated smoke** — ✅ Complete /
  fail-closed (Group A `c9da3b9`). Not invoked in production.
- **C1 live Kraken Futures** — ❌ Blocked for Canada (Group A/F
  `9eaad5f`; pinned in
  `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`).
- **C2 Coinglass** — ❌ Blocked until paid API plan/key (pinned).
- **C3 IBKR DOM** — ⏸ Pending actual API DOM rows after CME open /
  entitlement proof (pinned; `42daa1f`).

### 15.3 Phase D item status

- **Dynamic scanner v1** — ✅ Live / read-only observation
  (Group C `99a5084`, `ce0c4d1`, `c4144f5`). Observation period per
  §9 of the design doc is in progress.
- **D2 Tier 1 — typed OptionsSpreadSpec** — ✅ Complete (Group E
  `51166d4`).
- **D2 Tier 2 — LMT discipline** — ✅ Complete (Group E `2ab03e7`).
- **D2 Tier 3A — offline `bag_quote_check`** — ✅ Complete (Group E
  `a5c5b35`).
- **D2 Tier 3B — live-readonly quote probe** — ✅ Complete /
  validated 2026-05-17 (Group E `9fb163b`, async fix `a9fb34a`).
- **D2 Tier 3C — BAG unit normalisation design** — ⬜ Next authorised
  workstream (design + docs only).
- **D2 Tier 3D — adapter quote enforcement** — pending Tier 3C.
- **D2 Tier 4 — bracket / OCA or failsafe closer** — pending.
- **D2 Tier 5 — `spread_id`-aware reconciliation** — pending
  (cross-cutting; persisted-state migration required).
- **D2 Tier 6 — live BAG fill harness** — pending; depends on
  Tier 3C/3D/4/5.

### 15.4 Live Promotion Checklist

- ✅ Test baseline green (2,114 passed, 60.10 s).
- ✅ `full_cycle_preview` clean (0 intents, 0 orders, no broker calls).
- ❌ **No new live execution authorized** by this SSOT.
- ❌ Operator GO still required.
- ❌ **Live BAG not authorized.** Pending Tier 3C / 3D / 4 / 5 / 6.
- ❌ **DOM not authorized.** Pending Channel 1 probe at next CME open.
- ❌ **Kraken Futures live not authorized for Canada.** Permanent
  for current deployment.

### 15.5 Pre-Live Operator Tasks (unchanged from v9.2 / CLAUDE.md)

1. OS reboot — no pending kernel update as of 2026-04-21 (kernel
   6.17.0-1009-aws current; reboot deferred to next actual kernel
   update).
2. Disk cleanup — prune backup archives to below 75 % usage.
3. IB Gateway latency — investigate and resolve dangerous (>750 ms)
   classification.
4. Verify all 2,114 tests pass after reboot (test count grew from
   1,465 / 1,874 in earlier checkpoints to 2,114 today).
5. Run `full_cycle_preview.py --dry-run` clean.
6. Confirm `live_readiness.json` flips to `ready_for_live: true`.
7. Review open paper positions (MES short, per CLAUDE.md) before mode
   switch.

### 15.6 Next authorised workstream (recap)

Per `docs/CHAD_PRE_SSOT_GAP_CLOSURE_AUDIT_2026-05-17.md` §9, the next
authorised workstream is **Phase D Item 2 Tier 3C — BAG limit-price
unit normalisation design**. This is **design + docs only** — it must
not modify the adapter, must not place orders, and must not enable
adapter quote enforcement. Tier 3C is the only Phase D BAG step that
can land without depending on an external blocker (it does not need
broker-mid bid/ask data; it only needs the per-share / contract-dollar
field split).

---

## 16. Appendices

### Appendix A — Tier risk profiles

CHAD's Tier Architecture v2 (introduced in v9.1 commit `67ba391`)
defines four tiers, each with its own risk profile. Field names below
are taken verbatim from `runtime/tier_state.json`
(`schema_version=tier_state.v2`):

- `tier_name` ∈ {MICRO, STARTER, PRO_GROWTH, SCALE}.
- `tier_description` — operator-facing description.
- `current_equity_usd` — live equity from `portfolio_snapshot.json`.
- `tier_min_equity`, `tier_max_equity` — promotion/demotion thresholds.
- `enabled_strategies` — strategies that may emit signals in this tier.
- `allowed_instruments` — instruments that may be traded in this tier.
- `risk_profile`:
  - `max_risk_per_trade_usd`
  - `max_concurrent_positions`
  - `max_strategy_positions`
  - `primary_session_only` (entry-only)
  - `stop_distance_sizing_enabled`
  - `rr_floor` (default 1.5)
- `previous_tier`, `promoted_at_utc` — history.
- `demotion_pending`, `demotion_pending_to`, `demotion_pending_reason`,
  `demotion_pending_since_utc`, `demotion_applies_at` — drift
  bookkeeping.

For the canonical numeric values per tier, see the v9.1 appendix
which remains authoritative. The active tier is published into
`runtime/tier_state.json` every 5 minutes by
`chad-tier-manager.timer`.

### Appendix B — Execution gate chain

Reproduced from §4.1 for offline reference:

1. Quarantine / trust gate.
2. Strategy enabled gate.
3. Signal confidence gate.
4. Stop-distance sizing.
5. Session zone gate (entry-only).
6. R:R gate (≥ 1.5).
7. Liquidity / float-aware gate (THIN < 0.80 confidence blocks).
8. Net exposure gate.
9. Dynamic risk allocator.
10. Tier manager.
11. XGB veto gate (shadow-able modifier).
12. Winner scaler.
13. Execution router.
14. BAG LMT discipline (sec_type=BAG).
15. IBKR adapter contract qualification / submit path.

### Appendix C — Intelligence feed catalog

Reproduced from §8 with the same per-feed schema, cadence, timer,
status, and notes columns.

### Appendix D — FMP endpoints used

Per `docs/CHAD_FMP_EARNINGS_INTEL_STATUS_2026-05-16.md` and
`chad/market_data/fmp_earnings_intel_publisher.py`:

- `earnings-calendar` — window: `date_from=2026-05-02`,
  `date_to=2026-06-30` at audit time; `forward_days=45`,
  `lookback_days=14`.
- `price-target-consensus` — per-symbol analyst price target.
- `analyst-estimates` (annual) — per-symbol EPS estimates.
- `sec-filings-search` — recent SEC filings by symbol.

Forbidden (current plan): **FMP news endpoint** is not enabled. Do not
add it without an explicit plan-level decision.

`status=partial` is the expected state when ETFs (SPY, QQQ, IWM, etc.)
or unsupported tickers are in the universe; per-symbol
`provider_errors` arrays are populated but the publisher succeeds.

### Appendix E — Phase C unlock conditions

Per `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`:

- **C3 — IBKR DOM:** all four conditions must hold —
  - `MES domBids > 0`
  - `MES domAsks > 0`
  - `MNQ domBids > 0`
  - `MNQ domAsks > 0`
  - No `Error 354` observed.
  - Retry at approximately Sunday 6 PM ET / 22:00 UTC.
- **C1 — Kraken Futures live:** legally eligible entity + account +
  jurisdiction + separate operator-led approval. Permanent for the
  current Canadian deployment.
- **C2 — Coinglass:** operator decision on a paid plan; if yes,
  Channel 3 delivers the key, then Channel 2 designs the publisher.

### Appendix F — BAG live authorization checklist

All items below must be complete and independently verified before any
live BAG order may be authorised. v9.3 grants no authorisation.

- **Tier 1 — typed `OptionsSpreadSpec`** — ✅ Complete (`51166d4`).
- **Tier 2 — LMT discipline (no MKT BAG submits)** — ✅ Complete
  (`2ab03e7`). Markers `BAG_MKT_COERCED_TO_LMT`,
  `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE` observable in adapter logs.
- **Tier 3A — offline `bag_quote_check` validator** — ✅ Complete
  (`a5c5b35`). Module exists; adapter does not yet consult it.
- **Tier 3B — live-readonly quote probe** — ✅ Complete and validated
  2026-05-17 (`9fb163b`, fix `a9fb34a`). Results in
  `docs/PHASE_D_ITEM2_BAG_QUOTE_PROBE_RESULTS_2026-05-17.md`.
- **Tier 3C — BAG limit-price unit normalisation design** — ⬜ Next
  authorised workstream. Must introduce field-level split:
  `net_debit_contract_dollars` (strategy-facing, $/contract) and
  `net_debit_per_share` (broker-facing, $/option-price unit).
- **Tier 3D — adapter quote enforcement** — ⬜ Pending Tier 3C.
  Adapter must consume `net_debit_per_share` only and reject
  contract-dollar values.
- **Tier 4 — bracket / OCA or out-of-band failsafe closer** — ⬜
  Pending.
- **Tier 5 — `spread_id`-aware reconciliation** — ⬜ Pending. Migrates
  `position_guard` keying to `(strategy, symbol, spread_id)`.
- **Tier 6 — live BAG fill harness** — ⬜ Pending. Offline `ib_async`
  shim exercising `placeOrder → Trade → orderStatus → fill`.
- **Tier 7 — operator-initiated live BAG promotion** — ⬜ Pending; a
  separate governance change with `live_readiness.json` re-issued
  green.

### Appendix G — XGB promotion procedure

To promote a candidate XGB veto model:

1. Run `python3 scripts/promote_xgb_veto.py --status` to confirm the
   current active model.
2. Locate the candidate directory:
   `runtime/models/xgb_veto/candidates/<UTC ts>/`.
3. Run the promotion gate:
   `python3 scripts/promote_xgb_veto.py --promote <UTC ts>`.
   The gate requires `new_accuracy >= current_accuracy` AND
   `new_logloss <= current_logloss`. If the gate fails, the script
   refuses to promote.
4. Operator override (only when justified):
   `python3 scripts/promote_xgb_veto.py --promote <UTC ts> --operator-approve "reason"`.
   The reason is recorded in the new manifest's audit fields
   (`promoted_by`, `promoted_at_utc`).
5. After promotion, the predictor's in-process cache is invalidated /
   reload-signalled per the script's implementation. On the next
   predictor call, the new active model is loaded.
6. To roll back, demote the current promoted model back to the
   previous candidate via the same script's documented demotion path.
   The demotion is recorded in the same manifest stream.
7. To roll all the way back to the tracked baseline, remove the
   `runtime/models/xgb_veto/current/` directory. The predictor falls
   back to `shared/models/`.

### Appendix H — BAG unit rules and forbidden actions

**Unit rules:**

- `net_debit_estimate` currently represents **contract-dollar debit
  per spread**.
- IBKR BAG `lmtPrice` expects **per-share option-price debit**.
- Contract-dollar debit must be divided by the 100 options multiplier
  before use as IBKR BAG `lmtPrice`.
- Example: `$350 contract debit → 3.50 IBKR BAG lmtPrice`.

**Forbidden actions until Tier 3C/3D land:**

- Do not assume contract-dollar debit equals IBKR BAG lmtPrice.
- Do not pass `net_debit_estimate` contract-dollar values directly to
  the IBKR adapter's `lmtPrice` field.
- Do not wire adapter-level quote enforcement before Tier 3C unit
  normalisation lands.
- Do not authorise any live BAG order placement. The Tier 3B probe
  remains `--live-readonly` only.

**Forbidden actions for the strategy-facing surface (always):**

- Do not assume that paper BAG close PnL is trusted — the synthetic
  30 % credit ratio marks every BAG close as
  `pnl_untrusted=True` with reason
  `bag_close_synthetic_credit_ratio_30pct`.
- Do not assume the position guard tracks per-spread state — until
  Tier 5 lands, two concurrent SPY BAGs share a single guard slot.

### Appendix I — External blocker channel map

From `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md` §2:

- **Channel 1 — Runtime probes.** `systemctl`, IBKR Gateway probes,
  `scripts/probe_*` read-only scripts, runtime artifact inspection.
  Cannot perform paid-subscription unlocks or grant jurisdictional
  approval.
- **Channel 2 — Code, docs, tests.** All engineering work in the
  repository. The only channel Claude operates in.
- **Channel 3 — External / subscriptions / accounts.** API keys,
  market-data subscriptions, account permissions, jurisdictional
  onboarding, broker entitlements. Requires the operator (and often a
  counterparty).

Blocker → channel mapping:

| Blocker | Channel | Resolvable by Claude alone? | Trigger to retry |
|:---|:---|:---|:---|
| IBKR DOM (C3) | 1 (+ 3) | No — requires market-hours probe | CME open ~Sun 6 PM ET / 22:00 UTC |
| Kraken Futures live (C1) | 3 only | No — jurisdiction | New eligible entity / account + operator approval |
| Coinglass (C2) | 3 only | No — paid key | Operator decision on paid plan |
| Live BAG execution (D2) | 2 (+ 3) | Partially — Channel 2 tiers continue | Full Tier 3C→6 ladder + operator GO |

---

## 17. Closing Notes

v9.3 is an **Elite Upgrade Closeout** SSOT. It does not flip live
posture, it does not authorise any new live execution path, and it
does not wire the dynamic universe scanner into any strategy. It is
the canonical lock for the state of the system after Phase C reached
its max-buildable perimeter (with three external blockers explicitly
pinned), after Phase D Item 1 landed in observation-only mode, and
after Phase D Item 2 reached Tier 3B with the live-readonly probe
validated.

The next authorised workstream is **Phase D Item 2 Tier 3C — BAG
limit-price unit normalisation design** (design + docs only).

All other open items are either bounded Channel 2 work that depends
on a prior tier, or true external blockers (Channel 3) that this
document explicitly pins and does not pretend to resolve.

— *End of CHAD Unified SSOT v9.3.*
