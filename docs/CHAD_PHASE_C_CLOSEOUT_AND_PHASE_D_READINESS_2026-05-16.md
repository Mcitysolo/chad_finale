# CHAD Phase C Closeout and Phase D Readiness — 2026-05-16

Audit + documentation only. No source, runtime, deploy, ops, or config files
were modified. No services were touched. No commit was created.

---

## 1. Baseline

- HEAD commit: `fcb8656` — "Surface earnings intelligence in dashboard context"
- Working tree at start: clean (`git status --short` empty)
- Test baseline (per operator brief): **1959 passed** after the earnings_intel
  read-only dashboard context landed.
- Branch: `main`

---

## 2. Phase C closeout

### 2.1 Status table

| Phase C item | Status | Evidence | Remaining blocker | Closeout action |
|---|---|---|---|---|
| **C1A** — Kraken Futures public intel publisher | **COMPLETE / LIVE** | `chad/market_data/kraken_futures_intel_publisher.py`, runtime `kraken_futures_intel.json` (306 perps, schema `kraken_futures_intel.v1`, age 156 s, ttl 600 s, status `ok`), timer `chad-kraken-futures-intel-refresh.timer` firing every 5 min (last run 2 m 8 s ago, next in 2 m 51 s), tests `test_phase_c_item1_kraken_futures_intel.py`. Documented in `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md`. | None. Public-data only — no auth, no order path. | Closed. Monitor for `provider_status` regressions; no further code work. |
| **C1B** — Kraken Futures adapter scaffold | **COMPLETE / DORMANT** | `chad/execution/kraken_futures_adapter.py` (`KrakenFuturesAdapter`, `submit_intent`), `chad/exchanges/kraken_futures_client.py`, tests `test_kraken_futures_adapter.py` + `test_kraken_futures_client.py`. Adapter is **not** referenced by `ibkr_adapter.py`, the orchestrator, or any systemd unit (grep across `chad/` and `deploy/` returns only the file itself + tests). Documented in `docs/PHASE_C_C1B_KRAKEN_FUTURES_ADAPTER_SCAFFOLD.md`. | C1 live trading blocked (see C1 row). | Closed as dormant scaffold. Do not import from live path. |
| **C1C** — Kraken Futures authenticated smoke scaffold | **COMPLETE / FAIL-CLOSED** | `chad/tools/kraken_futures_auth_smoke.py`, tests `test_kraken_futures_auth_smoke.py`. Returns `None` / skips when `KRAKEN_FUTURES_API_KEY` / `KRAKEN_FUTURES_API_SECRET` env vars are absent (which they are in the current Canadian deployment). Documented in `docs/PHASE_C_C1C_KRAKEN_FUTURES_AUTH_SMOKE.md`. | Same as C1B — keys absent because of jurisdiction. | Closed. Tool is operator-invocable only. |
| **C1** — Kraken Futures live trading | **BLOCKED** | `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md` lines 12 + 47: "**BLOCKED FOR CURRENT CANADIAN DEPLOYMENT**" / "Kraken Futures live trading is blocked for the current Canadian deployment". Reinforced in `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md` lines 103, 202, 228-229. | Kraken Futures is not available to retail users in the operator's jurisdiction; no auth credentials issued; no LiveGate authorization. | Stay blocked. C1A intel continues to provide read-only context for non-Kraken crypto strategies. |
| **C2** — Coinglass liquidation heatmap | **BLOCKED** | `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md` §4 lines 138-144: public Coinglass endpoints return HTTP 500 / deprecation; no API key provisioned; **no Coinglass code exists** (grep across `chad/`, `deploy/`, `config/`, `ops/` finds zero `.py`/`.service`/`.timer`/`.json` references — only doc mentions). | Requires paid Coinglass API plan + key. | Stay blocked. Do not begin scaffolding. |
| **C3** — IBKR DOM consumer | **BLOCKED / PENDING** | `docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md` lines 31-33: `reqMktDepth` returned **IBKR Error 354** ("Requested market data is not subscribed") and `domBids`/`domAsks` both length 0 for MES and MNQ. No `orderflow_state`, `orderflow_gate`, `ibkr_dom_provider`, or `reqMktDepth` references in `chad/` or `deploy/`. | CME Real-Time L2 entitlement not provisioned; until a live probe shows `MES.domBids > 0 ∧ MES.domAsks > 0 ∧ MNQ.domBids > 0 ∧ MNQ.domAsks > 0 ∧ no Error 354`, no consumer is implementation-ready. | Stay blocked. Re-probe only after operator confirms entitlement. |

### 2.2 Orphan check

No orphaned Phase C artifacts. The Coinglass surface is entirely absent (correct
— it was never authorized). The Kraken Futures scaffold (C1B/C1C) is properly
isolated: no live caller, no systemd unit, no orchestrator import.

---

## 3. Runtime intelligence layer

Snapshot taken 2026-05-16 ~21:05 UTC.

| File | schema_version | status | age (s) | TTL (s) | fresh | symbols | Note |
|---|---|---|---:|---:|---|---:|---|
| `runtime/news_intel.json` | `news_intel.v1` | `ok` | 1 113 | 1 800 | ✅ | 25 | polygon primary, yahoo fallback |
| `runtime/relative_strength.json` | `relative_strength.v1` | `ok` | 29 565 | 90 000 | ✅ | 25 | daily-bar provider |
| `runtime/volume_scan.json` | `volume_scan.v1` | `ok` | 156 | 300 | ✅ | 25 | rolling-1m fallback active |
| `runtime/crypto_derivatives.json` | `crypto_derivatives.v1` | `ok` | 32 | 600 | ✅ | 3 | Kraken Futures public tickers |
| `runtime/futures_roll_state.json` | `futures_roll_state.v1` | `ok` | 83 688 | 86 400 | ✅ | 9 | static CME calendar |
| `runtime/options_greeks.json` | `options_greeks.v1` | `ok` | 76 605 | 3 600 | ⚠️ stale by self-declared TTL | 1 (SPY) | Refresh timer is **daily** at 23:48 UTC (next in ~2 h 43 m); the published `ttl_seconds=3600` understates the actual cadence. Not a Phase C blocker but worth a Phase D follow-up to align the published TTL with the timer cadence (Phase B item carried over). |
| `runtime/kraken_futures_intel.json` | `kraken_futures_intel.v1` | `ok` | 156 | 600 | ✅ | 306 | C1A publisher live |
| `runtime/earnings_intel.json` | `earnings_intel.v1` | **`partial`** | 14 609 | 21 600 | ✅ | 25 | FMP stable endpoints; partial = expected (some endpoints rate-limited / not subscribed). Read-only dashboard surface only. |
| `runtime/setup_family_expectancy.json` | `setup_family_expectancy.v2` | n/a (no `status` field) | 72 315 | 86 400 | ✅ | — | daily learner |
| `runtime/event_risk.json` | `event_risk.v1` | n/a (no `status` field) | 1 486 | 1 800 | ✅ | — | operator calendar present |

**Health verdict**: the intelligence layer is healthy enough to begin Phase D.
The single anomaly is the `options_greeks.json` self-declared TTL vs. the
daily timer cadence; the underlying data is acceptable because the publisher
runs daily and the consumer (alpha_options metadata annotation) does not gate
on freshness. Earnings intel `status=partial` is the expected state during the
first collection week and is consumed read-only.

---

## 4. Phase D readiness audit

### 4.1 Existing surfaces (do not duplicate)

- **Universe source of truth**: `runtime/universe.json` (schema `universe.v1`,
  25 active symbols, intersection of `config/universe.json` static list and
  `runtime/full_execution_cycle_last.json` plan symbols, capped at 25).
- **Active-universe loader**: `chad.utils.universe_provider.load_active_universe()`
  / `get_trade_universe()` (freshness window 5 400 s, falls back to static
  config with a `universe_provider.using_static_fallback` warning). Already
  consumed by `news_intel_publisher`, `volume_scan_publisher`,
  `relative_strength_publisher`, `fmp_earnings_intel_publisher`,
  `ibkr_bar_provider`, `portfolio_engine`, `price_cache_refresh`, and
  `strategies.gamma`. **Any Phase D scanner MUST plug into this provider, not
  bypass it.**
- **Existing scanners**: `chad/market_data/volume_scan_publisher.py` (RVOL),
  `chad/ops/exterminator.py` (read-only anomaly detector — 18 categories).
  No premarket / top-mover scanner exists.
- **Options BAG/COMBO execution**: already functional end-to-end —
  `chad/strategies/alpha_options.py` emits one BAG TradeSignal per spread
  (`sec_type=BAG`, asset_class `options`); `chad/options/strike_selector.py`
  provides `select_vertical_spread()` / `SpreadSpec`;
  `chad/execution/ibkr_adapter.py` resolves BAG contracts with two
  `ComboLeg`s, qualifies legs individually, refuses to submit BAGs with
  `conId=0` legs, and skips `qualifyContracts` on BAGs (IBKR Error 321).
  Regression test `test_alpha_options_bag_no_etf_downgrade.py` pins the
  SPY/BAG → options asset-class mapping. Hardening would be incremental, not
  greenfield.
- **Dashboard read-only intel surface**: `chad/dashboard/api.py` exposes
  `earnings_intel` via `chad.intel.strategy_intelligence._load_earnings_intel_context`
  (fail-open, freshness/status classified). New Phase D feeds should reuse
  the same `chad.intel.strategy_intelligence` pattern.

### 4.2 Phase D readiness table

| Phase D candidate | Current support | Missing pieces | Risk | Recommended order |
|---|---|---|---|---|
| **Dynamic universe / premarket scanner** | `universe_provider`, `runtime/universe.json`, `volume_scan`, `relative_strength`, `news_intel`, `earnings_intel`, `exterminator` already exist. | No premarket gap-up/down provider; no top-mover ranking; no scanner→universe writeback contract; needs an explicit promotion/demotion ledger so static `config/universe.json` stays authoritative for fallback. | Medium. Pure-read audit first; writeback path is the riskiest leg and must not collide with the static fallback. | **1st — recommended.** Begin with audit-first design doc, no code. Reuses every Phase B/C feed; introduces no new external dependencies; no blocked-item assumptions. |
| **Options BAG/COMBO execution hardening** | Full BAG path already works; regression test pins SPY/BAG. Live evidence in paper. | Greeks TTL/cadence alignment, expanded BAG symbol coverage (currently SPY-only chain cached), per-leg slippage telemetry, possibly four-leg condor/butterfly support. | Low-Medium. Incremental; touches the live execution path so each change needs a regression test. | **2nd.** Schedule after the universe scanner has stabilized — they share no surface area, but ordering keeps execution-path churn separated from intel-layer churn. |
| **Earnings-intel strategy metadata** | Read-only dashboard context live since `fcb8656`. Publisher live since `a98a165`. | Need one full collection week of `runtime/earnings_intel.json` history before wiring to strategy gates (per operator instruction); no consumer in strategies/* yet. | High if rushed (event-risk false negatives could let strategies trade through earnings). | **3rd — observation only**. Do not wire to any gate until ≥ 7 calendar days of clean `status=ok|partial` history. |
| **Kraken Futures live trading** | C1A intel live; C1B/C1C scaffolds dormant. | Jurisdictional access; LiveGate authorization; risk caps; broker truth reconciliation for non-IBKR venue. | **External blocker.** | **Do not schedule.** Stays in BLOCKED until jurisdiction changes. |
| **IBKR DOM orderflow gate** | None. | CME Real-Time L2 entitlement; live `reqMktDepth` probe must pass; orderflow gate + state file design. | **External blocker.** | **Do not schedule.** Re-probe only after entitlement is confirmed. |
| **Coinglass heatmap** | None. | Paid Coinglass API key; stable endpoint contract. | **External blocker + cost.** | **Do not schedule.** |

### 4.3 Chosen next authorized Phase D action

**A — Dynamic universe / scanner audit-first implementation.**

Rationale:
- It reuses the existing `universe_provider`, `runtime/universe.json`,
  `volume_scan`, `relative_strength`, `news_intel`, and `earnings_intel`
  surfaces — no duplicate data paths.
- It does not depend on any blocked Phase C item (C1 live, C2, C3).
- It is audit-first (design doc, no code), matching the governance rule of
  one change at a time with a verification sequence between changes.
- It unblocks the second-tier Phase D item (BAG/COMBO hardening) by
  separating intel-layer churn from execution-path churn.

---

## 5. Forbidden assumptions

The following are explicitly **not** permitted as Phase D premises:

1. Kraken Futures live trading is **not** available. Do not assume `submit_intent` on `KrakenFuturesAdapter` can be wired to any live caller.
2. DOM is **not** available. Do not assume `domBids`/`domAsks` populate; do not write an `orderflow_state` consumer until MES/MNQ books populate with no Error 354.
3. Coinglass cannot be built keyless. Do not assume a public endpoint can replace the paid API.
4. Do not wire `earnings_intel` into strategy gates before one full collection week of clean `runtime/earnings_intel.json` history.
5. Do not duplicate the universe scanner. `chad.utils.universe_provider.load_active_universe()` is the source of truth for the active universe and must be reused.

---

## 6. Next authorized action

Next authorized Phase D action is: **Dynamic universe / scanner audit-first implementation** (design-doc only; no code, no runtime, no deploy changes).

---

## 7. Verification evidence

- `git status --short`: empty at start, empty at end.
- `git log --oneline -12`: HEAD `fcb8656` "Surface earnings intelligence in dashboard context".
- Runtime health: all consumed feeds fresh by their own TTL except `options_greeks.json` (self-declared `ttl=3600 s`, actual cadence daily; non-blocking, noted as a Phase B carry-over).
- `full_cycle_preview.py --dry-run`: not executed — the operator-defined verification sequence applies "after every code change", and this audit changed no code. Re-run by the operator before any Phase D code lands.
- Files modified outside `docs/`: **none**. Only new file is this doc.
- No commit was created.
