# BOX-049 (Official Matrix) — Zero-fill strategies classified

- **Box number (Official Matrix):** 049
- **Box title (Official Matrix):** Zero-fill strategies classified — Each zero-fill strategy is marked intentional dormant, regime-silent, halted, or broken
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T19:37:45Z
- **HEAD at cut:** `bbe7525`
- **Branch:** `main`

> **Numbering disambiguation:** Official Matrix Box 049 here is
> distinct from the Supplemental Annex Box 049
> ("COMMIT_PLAN_OR_PATCHSET_READY"). Filename uses `_OFFICIAL_`
> prefix in the evidence file.

---

## 0. Scope and safety statement

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This policy does not flip
  `ready_for_live`, does not enable any strategy, does not change
  weights, and does not clear any halt.
- **No runtime mutation.** No fills/trades/ledgers modified.

---

## 1. Canonical strategy universe (18 enum entries; 17 active + 1 DEFERRED)

Source: `chad/types.py` enum `StrategyName` + `chad/strategies/__init__.py`
`_REGISTRY` + `DEFERRED_STRATEGIES` + `chad/risk/tier_manager.py`
`_CANONICAL_STRATEGY_NAMES`.

- **Enum:** 18 entries (alpha, beta, beta_trend, gamma, omega, delta, alpha_crypto, alpha_intraday, alpha_forex, alpha_futures, gamma_futures, omega_macro, gamma_reversion, alpha_options, omega_vol, delta_pairs, omega_momentum_options, alpha_intraday_micro).
- **Active `_REGISTRY`:** 17 entries (all minus `alpha_forex`, which is in `DEFERRED_STRATEGIES` and has its `_REGISTRY` entry commented out at line 195).
- **Tier canonical:** 17 entries (matches active registry; `alpha_forex` omitted).
- **`config/strategy_weights.json` weights dict:** 16 entries (missing `alpha_forex` and `alpha_intraday_micro`).

---

## 2. Evidence sources (read-only)

### 2.1 Fill / trade counts (per-strategy, sourced from `data/fills/`, `data/trades/`)

- **Fills:** 53 daily files (2026-03-17 → 2026-05-20).
- **Trades:** 64 daily files (2026-03-02 → 2026-05-20).
- **Recent window:** last 7 days (file dates ≥ `20260513`).

### 2.2 Runtime evidence

- `runtime/regime_state.json` — current regime = **`trending_bull`**, confidence 0.73, source `live_loop.run_once` at 2026-05-20T04:18:47Z.
- `runtime/last_route_decision.json` — most recent route decision (2026-05-20T04:19); `available_strategies` count per strategy + `rejected_strategies` reason map.
- `runtime/strategy_allocations.json` — edge_decay state: **3 halted** strategies via `consecutive_negative_*`.
- `runtime/stop_bus.json` — `active: true` since 2026-05-20T04:20:11Z (broker_latency 8210ms > 2000ms); explains journal silence across **all** strategies (not strategy-specific).

### 2.3 Config evidence

- `config/regime_activation_matrix.json` — per-regime allowlists (consumed by `chad.portfolio.regime_activation.filter_intents_by_regime`).
- `config/strategy_weights.json` — per-strategy allocator weights.
- `config/per_strategy_loss_limits.json` — per-strategy loss limits (delta_pairs has -150.0 explicit cap).

### 2.4 Current regime → allowed strategy set

```
trending_bull: alpha, alpha_crypto, alpha_forex, alpha_futures, alpha_intraday,
               alpha_options, beta, beta_trend, delta, gamma_futures, omega_macro,
               omega_momentum_options
```
NOT in trending_bull (regime-silenced this regime): gamma, omega, delta_pairs, gamma_reversion, omega_vol, alpha_intraday_micro.

---

## 3. Per-strategy classification (all 18)

| # | Strategy | All-time fills | Recent fills (7d) | Halted? | In trending_bull? | Weight | **Classification** | Owner / next action |
| - | -------- | -------------- | ----------------- | ------- | ----------------- | ------ | ------------------ | ------------------- |
| 1 | alpha | 780 | 3 | NO | YES | 0.16 | **ACTIVE_WITH_FILLS** | none |
| 2 | beta | 319 | 0 | NO | YES | 0.05 | **ACTIVE_NO_RECENT_FILLS_REGIME_SILENT** (multi-week 13F-driven holder; regime-allowed; signal cadence is multi-week by design per `regime_activation_matrix.json` _comment_5) | observe; no action |
| 3 | beta_trend | **0** | 0 | NO | YES | **0.20** | **ACTIVE_NO_RECENT_FILLS_REGIME_SILENT** (regime-allowed; `last_route_decision` shows 0 available / "no_signal" rejection — signal generator has not produced a TradeSignal in the 64-day audit window despite the 0.20 weight; not regime-blocked, not halted, not broken — just quiet) | **operator review: confirm beta_trend signal cadence vs 0.20 weight is intended; or revisit weight allocation** (followup G-NN candidate) |
| 4 | gamma | 8 | 0 | NO | NO | 0.07 | **ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES** (regime-blocked under trending_bull; allowed in ranging/volatile/unknown) | observe; activates on regime change |
| 5 | omega | **0** | 0 | NO | NO | 0.05 | **ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES** (regime-blocked under trending_bull; allowed in volatile/unknown) | observe; activates on regime change |
| 6 | delta | 772 | 156 | **YES** (`consecutive_negative_5`) | YES | 0.02 | **ACTIVE_WITH_FILLS** (recent fills include placeholder / rejected SELLs from Box-012 GAP-046 era; halt just triggered) | **operator decision: clear halt via `scripts/clear_edge_decay.py --strategy delta --reason …` (per Box 035) or accept halt** |
| 7 | alpha_crypto | 3476 | 0 | **YES** (`consecutive_negative_10`) | YES | 0.04 | **ACTIVE_NO_RECENT_FILLS_HALTED** | **operator decision: clear halt or accept halt** |
| 8 | alpha_intraday | 38 | 1 | NO | YES | 0.03 | **ACTIVE_WITH_FILLS** | none |
| 9 | alpha_forex | **0** | 0 | n/a | YES (in matrix) but `_REGISTRY` entry commented out | None | **INTENTIONAL_DORMANT** (`DEFERRED_STRATEGIES` member; comment: "symbol-translation layer is not implemented; re-enable when the FX universe is formally defined") | none — formally deferred |
| 10 | alpha_futures | 2428 | 3 | NO | YES | 0.09 | **ACTIVE_WITH_FILLS** | none |
| 11 | gamma_futures | 476 | 7 | **YES** (`consecutive_negative_10`) | YES | 0.05 | **ACTIVE_WITH_FILLS** (recent fills before halt; halt just triggered) | **operator decision: clear halt or accept halt** |
| 12 | omega_macro | 87 | 1 | NO | YES | 0.03 | **ACTIVE_WITH_FILLS** | none |
| 13 | gamma_reversion | **0** | 0 | NO | NO | 0.04 | **ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES** (regime-blocked under trending_bull; allowed in `ranging` only; `last_route_decision` shows 1 available — emitted in last cycle but filtered by regime gate) | observe; activates in `ranging` regime |
| 14 | alpha_options | 453 | 0 | NO | YES | 0.04 | **ACTIVE_NO_RECENT_FILLS_REGIME_SILENT** | observe |
| 15 | omega_vol | 12 | 0 | NO | NO | 0.05 | **ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES** (regime-blocked under trending_bull; allowed in `volatile` only) | observe; activates in `volatile` regime |
| 16 | delta_pairs | **0** | 0 | NO | NO | 0.05 | **ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES** (regime-blocked under trending_bull; allowed in `ranging` only) | observe; activates in `ranging` regime |
| 17 | omega_momentum_options | 6 | 0 | NO | YES | 0.03 | **ACTIVE_NO_RECENT_FILLS_REGIME_SILENT** | observe |
| 18 | alpha_intraday_micro | **0** | 0 | NO | NO | None | **CONDITIONALLY_SILENT** (per Official Box 048 — registered + tier-wired but missing `strategy_weights.json` entry; also regime-blocked under trending_bull) | per Box 048 + Box 033 — operator Option A/B/C decision pending |

`strategies_total: 18`; `strategies_with_fills_total: 12` (all-time fills > 0).

---

## 4. Zero-fill strategies summary (all-time 0)

| Strategy | Classification | Why zero-fill |
| -------- | -------------- | ------------- |
| **beta_trend** | ACTIVE_NO_RECENT_FILLS_REGIME_SILENT | regime-allowed but signal generator has not fired in 64-day audit window (last_route_decision: 0 available / "no_signal"). Weight 0.20. **Operator follow-up recommended.** |
| **omega** | ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES | regime-blocked under trending_bull (only in volatile/unknown). |
| **gamma_reversion** | ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES | regime-blocked under trending_bull (only in `ranging`). |
| **delta_pairs** | ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES | regime-blocked under trending_bull (only in `ranging`). |
| **alpha_forex** | INTENTIONAL_DORMANT | `DEFERRED_STRATEGIES` member; `_REGISTRY` entry commented out. |
| **alpha_intraday_micro** | CONDITIONALLY_SILENT | per Official Box 048; missing weight + regime-blocked. |

`zero_fill_strategies_total: 6`; `zero_fill_strategies_classified_total: 6`; `unknown_zero_fill_strategies_total: 0`; `broken_zero_fill_strategies_total: 0`.

**No silent orphan remains.** Every zero-fill strategy has an explicit classification + owner/next-action note.

---

## 5. Recent-zero-fill strategies (all-time > 0 but last-7d = 0)

Not strictly "zero-fill" but worth tracking:

| Strategy | All-time | Last 7d | Classification |
| -------- | -------- | ------- | -------------- |
| beta | 319 | 0 | ACTIVE_NO_RECENT_FILLS_REGIME_SILENT (multi-week holder) |
| gamma | 8 | 0 | ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES |
| alpha_crypto | 3476 | 0 | **ACTIVE_NO_RECENT_FILLS_HALTED** |
| alpha_options | 453 | 0 | ACTIVE_NO_RECENT_FILLS_REGIME_SILENT |
| omega_vol | 12 | 0 | ACTIVE_NO_RECENT_FILLS_BLOCKED_BY_GATES |
| omega_momentum_options | 6 | 0 | ACTIVE_NO_RECENT_FILLS_REGIME_SILENT |

---

## 6. Halted strategies (operator follow-up)

3 strategies are currently HALTED via `runtime/strategy_allocations.json`:

| Strategy | Halt reason | Recent fills before halt | Operator path to clear |
| -------- | ----------- | ------------------------ | ---------------------- |
| **alpha_crypto** | `consecutive_negative_10` | 0 in last 7d, 3476 all-time | `scripts/clear_edge_decay.py --strategy alpha_crypto --reason …` (per Box 035 GAP-018 semantics: clear resets `consecutive_negative` to 0, preserves prior streak as `previous_consecutive_negative`) |
| **delta** | `consecutive_negative_5` | 156 in last 7d (incl. Box-012-era placeholder/rejected $100 SELLs) | same path |
| **gamma_futures** | `consecutive_negative_10` | 7 in last 7d | same path |

**These halts are NOT being cleared by Box 049.** They remain under
operator decision. Box 049's role is to **classify** the state, not
to clear it.

---

## 7. Patches summary

| Patch class            | Action                                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Production code        | **None** — all classifications are observational.                                                                        |
| Live config            | **None** — no `config/strategy_weights.json`, no `regime_activation_matrix.json`, no `per_strategy_loss_limits.json`, no `runtime/*.json` mutated. |
| Documentation          | **Added** `ops/pending_actions/BOX-049_zero_fill_strategies_classified.md` (this file).                                  |
| Evidence               | **Added** `runtime/completion_matrix_evidence/BOX-049_OFFICIAL_zero_fill_strategies_classified.md`.                       |
| Tests                  | **None added** — existing strategy_registry parity test (Box 033) + alpha_intraday_micro classification (Box 048) cover the structural invariants. No static test added since classifications are dependent on regime state which changes regime-to-regime (a static test would be over-fitted to today's `trending_bull`). |
| Halts                  | **None cleared** — operator decision (Box 035 + this Box 049 §6).                                                         |
| Frozen historical SSOT | **Unchanged** — forward-only.                                                                                            |
| Staged / committed     | **None.**                                                                                                                |

---

## 8. False-closure guardrails

- Does NOT enable any strategy.
- Does NOT clear any halt.
- Does NOT add `alpha_intraday_micro` or `alpha_forex` weight to
  `config/strategy_weights.json`.
- Does NOT modify `DEFERRED_STRATEGIES`.
- Does NOT change `config/regime_activation_matrix.json`.
- Does NOT make a current-positions claim.
- Does NOT authorize live trading.

**live trading not authorized. CHAD remains PAPER. `ready_for_live=false`.**
