# CHAD Phase D — Dynamic Universe Scanner Design

**Date:** 2026-05-16
**Author:** Phase D audit-first design pass
**Phase:** D, item D1 (per `docs/CHAD_UNIFIED_SSOT_v9.2_2026-05-15.md` line 1613)
**Predecessor doc:** `docs/CHAD_PHASE_C_CLOSEOUT_AND_PHASE_D_READINESS_2026-05-16.md`

---

## 1. Status

**DESIGN ONLY — NOT IMPLEMENTED.**

This document defines the contract, scope, scoring inputs, safety rules,
and implementation order for the Phase D `dynamic_universe_scanner`. It
intentionally contains no implementation. No code, runtime, deploy, or
config files were modified to produce it.

Acceptance for this design pass:

- A single new file under `docs/` (this file).
- No edits under `chad/`, `deploy/`, `ops/`, `config/`, or `runtime/`.
- No commits.
- No services created or restarted.
- No dependence on blocked items C1 (live Kraken Futures), C2 (Coinglass),
  or C3 (IBKR DOM).

---

## 2. Existing universe system

CHAD already has a canonical, two-tier universe system. **The Phase D
scanner MUST reuse this system rather than introducing parallel universe
logic.**

### 2.1 On-disk artifacts

- `config/universe.json` — operator-controlled static list. Shape:
  ```json
  { "symbols": [ "AAPL", "SPY", ... ], "futures": [ {...} ] }
  ```
  38 equities/ETFs + 10 futures specs (MES, MNQ, MCL, MGC, ZN, ZB, M6E,
  SIL, MYM, M2K) on disk at audit time.
- `runtime/universe.json` — live-screened active list, written every
  ~30 min by `chad.analytics.universe_builder` (schema
  `universe.v1`). At audit time: 25 symbols, ok=true, sources include
  `config_universe` (38) and `plan_tick_symbols` (15) from
  `runtime/full_execution_cycle_last.json`.

### 2.2 Canonical loader

`chad.utils.universe_provider.load_active_universe()` is the single
source of truth for "what symbols a strategy should consider." Its
documented preference order is:

1. `runtime/universe.json` when fresh (≤ 5400 s old).
2. `cfg.static_symbols` (STATIC mode) or legend top-N (LEGEND_TOP_N).
3. `config/universe.json` read as a bare symbol-list document.
4. `cfg.fallback` / `DEFAULT_FALLBACK` as last resort.

Stale or malformed runtime files log a single structured warning
(`universe_provider.using_static_fallback`) and degrade to config. The
return type `UniverseLoad` carries full source provenance (`source_path`,
`source_type`, `symbol_count`, `stale`, `reason`).

`get_trade_universe()` is a thin compatibility wrapper preserving the
`List[str]` shape for legacy strategy callers (currently only
`chad/strategies/gamma.py`).

### 2.3 Current `load_active_universe` consumers

Producers (publishers that read the active universe to score symbols):

- `chad/market_data/volume_scan_publisher.py`
- `chad/market_data/relative_strength_publisher.py`
- `chad/market_data/news_intel_publisher.py`
- `chad/market_data/fmp_earnings_intel_publisher.py`
- `chad/market_data/price_cache_refresh.py`
- `chad/market_data/ibkr_bar_provider.py`
- `chad/portfolio/portfolio_engine.py`

Strategy direct consumer:

- `chad/strategies/gamma.py` (via `get_trade_universe`).

All other strategies pin their universe via their own
`*_config.py` builder (`DEFAULT_*_UNIVERSE`) and `target_universe`
field on `StrategyConfig`. **This is the existing contract the scanner
must respect.**

### 2.4 Duplicate-universe risk

Static lists exist in:

- `chad/ops/short_interest_refresh.py` — `UNIVERSE = ["SPY","QQQ",...]`
- `chad/ops/reddit_sentiment_refresh.py` — `UNIVERSE = [...]`
- `chad/ops/trends_refresh.py` — `UNIVERSE = [...]`
- `chad/ops/advisory_pre_market.py` — `UNIVERSE = [...]`
- `chad/ops/strategy_intelligence_refresh.py` — `UNIVERSE = [...]`
- `chad/market_data/ibkr_bar_provider.py` — `DEFAULT_UNIVERSE` fallback
  (only used when `load_active_universe` itself fails).

These are out of scope for Phase D v1 — the scanner does not need to
unify them, but it must not add a fourth list.

---

## 3. Existing scanner state

**No premarket / top-mover dynamic scanner exists in the repo.**

What *does* exist (per audit and per the Phase C closeout doc):

- **`chad/market_data/volume_scan_publisher.py`** — intraday RVOL scan
  over the active universe. Output `runtime/volume_scan.json`
  (`volume_scan.v1`). Per-symbol `rvol`, `rvol_class`, `metric_type`.
- **`chad/market_data/relative_strength_publisher.py`** — 5-day RS vs
  SPY/QQQ. Output `runtime/relative_strength.json`
  (`relative_strength.v1`). Per-symbol `rs_class`, `excess_vs_spy_5d`,
  `excess_vs_qqq_5d`.
- **`chad/market_data/news_intel_publisher.py`** — equity catalyst
  detection. Output `runtime/news_intel.json` (`news_intel.v1`). Per-
  symbol `catalyst_strength`, `catalyst_direction`, `has_catalyst`,
  `confirmed_gate_relevant`.
- **`chad/market_data/fmp_earnings_intel_publisher.py`** — earnings /
  analyst intel. Output `runtime/earnings_intel.json`
  (`earnings_intel.v1`). Per-symbol `days_to_next_earnings`,
  `price_target_consensus`, `annual_eps_avg_estimate`.
- **`chad/ops/exterminator.py`** — read-only anomaly scanner. Not a
  universe ranker.

The current universe is **hybrid**: operator-controlled static list
(`config/universe.json`) is the primary source; `runtime/universe.json`
narrows it via the universe_builder (which appends symbols from the
orchestrator's `summary.tick_symbols`, capped at `max_symbols=25`).

`runtime/full_execution_cycle_last.json` feeds the builder's
`plan_tick_symbols` source — i.e. orchestrator tick-symbol selection
already feeds back into the active universe one tier above the scanner.

---

## 4. Intelligence inputs available

Audit of each candidate intel feed at `2026-05-16` (live values):

| Feed | Schema | Status | Scope | Per-symbol keys |
|------|--------|--------|-------|-----------------|
| `news_intel.json` | `news_intel.v1` | ok | equities/ETFs (25 syms) | `catalyst_strength`, `catalyst_direction`, `catalyst_categories`, `has_catalyst`, `confirmed_gate_relevant`, `news_count`, `relevant_news_count`, `latest_ts_utc`, `source_provider` |
| `relative_strength.json` | `relative_strength.v1` | ok | equities/ETFs (25 syms) | `rs_class`, `return_5d`, `excess_vs_spy_5d`, `excess_vs_qqq_5d`, `rs_vs_spy`, `rs_vs_qqq`, `bars_used`, `data_available` |
| `volume_scan.json` | `volume_scan.v1` | ok | equities/ETFs (25 syms) | `rvol`, `rvol_class`, `avg_daily_volume`, `current_volume`, `expected_volume`, `metric_type` (`rolling_rvol` or snapshot), `window_minutes`, `data_available` |
| `earnings_intel.json` | `earnings_intel.v1` | partial | equities (25 syms) | `days_to_next_earnings`, `next_earnings_date`, `last_earnings_date`, `eps_actual/estimated`, `annual_eps_avg_estimate`, `price_target_consensus/high/low/median`, `sec_filings_count`, `latest_filing_type` |
| `event_risk.json` | `event_risk.v1` | (top-level) | market-wide | `risk_score`, `severity`, `elevated_risk`, `next_event`, `windows[]` (macro events, not per-symbol) |
| `kraken_futures_intel.json` | `kraken_futures_intel.v1` | ok | crypto perps (306 syms) | OI / volume / funding / crowding flags |

### 4.1 Classification

- **Equity-only / safe for v1:** `news_intel`, `relative_strength`,
  `volume_scan`, `earnings_intel`.
- **Crypto-only / out of v1:** `kraken_futures_intel`. (Also gated by
  blocked C1 live deployment.)
- **Market-wide, not symbol-rankable:** `event_risk` — useful as a
  market posture warning / metadata field but NOT a per-symbol score
  input in v1.
- **Equity per-symbol with mixed coverage:** `earnings_intel` — at
  audit time only 1/25 symbols had a populated `next_earnings_date`
  (FMP `stable` endpoint coverage is partial). Earnings is therefore
  metadata + a warning flag for v1, not a hard score component.

### 4.2 Safe to consume in v1

`news_intel.json`, `relative_strength.json`, `volume_scan.json`, plus
`earnings_intel.json` strictly as metadata + a "do not score" warning
band near earnings.

Liquidity is partially derivable today from
`volume_scan.avg_daily_volume` × last price. A dedicated ADDV/liquidity
gate (`chad/utils/liquidity_gate.py`) exists in the codebase
(referenced from `alpha.py` and `alpha_intraday.py` for the EQUITY/ETF
pre-entry gate). The scanner v1 should reuse the same liquidity helper
rather than reinvent it; **no new liquidity provider is in scope.**

---

## 5. Strategy scope

### 5.1 Strategy → asset class map (audited)

Pure equity / ETF strategies (in-scope candidates for v1 observation):

- `chad/strategies/alpha.py` — `target_universe=None` (uses
  legend / raw_prices keys). EQUITY + ETF (`_asset_class_for_symbol`).
- `chad/strategies/alpha_intraday.py` — fixed `UNIVERSE` list, but
  asset class branches EQUITY/ETF vs FUTURES vs CRYPTO; the EQUITY/ETF
  branch already consumes RS/RVOL/news gates.
- `chad/strategies/beta_trend.py` — `target_universe=None`,
  legend-driven, EQUITY/ETF.
- `chad/strategies/gamma.py` — uses `get_trade_universe()`, EQUITY/ETF.
- `chad/strategies/delta_pairs.py` — `target_universe=["SPY","QQQ","IWM"]`,
  ETF pairs.
- `chad/strategies/gamma_reversion.py` —
  `target_universe=["SPY","QQQ","GLD","TLT"]`, ETF.
- `chad/strategies/omega.py` — `target_universe=["SPY","QQQ","SH","PSQ"]`,
  inverse-ETF hedge.
- `chad/strategies/omega_vol.py` — `target_universe=["SVXY","UVXY"]`,
  vol ETPs (strict allow-list `ALLOWED_VOL_SYMBOLS`).
- `chad/strategies/omega_momentum_options.py` — `UNIVERSE=["SPY","QQQ",
  "AAPL","NVDA","MSFT"]`, options on equity / ETF.
- `chad/strategies/alpha_options.py` — `target_universe=["SPY"]`, strict
  `ALLOWED_OPTIONS_SYMBOLS={"SPY","QQQ","IWM","GLD","TLT","AAPL","MSFT",
  "NVDA","GOOGL"}`.
- `chad/strategies/execution_intelligence.py` —
  `target_universe=["SPY","QQQ"]`.
- `chad/strategies/beta.py` — `target_universe=None`, consensus-driven.
- `chad/strategies/delta.py` — `target_universe=["SPY","QQQ"]`,
  cross-asset.

Futures strategies (out-of-scope for v1):

- `alpha_futures.py` — `ALPHA_FUTURES_UNIVERSE=("MES","MNQ","MGC")`.
- `gamma_futures.py` — `GAMMA_FUTURES_UNIVERSE=("MCL","MYM","M2K","ZN","ZB")`.
- `omega_macro.py` — `target_universe=["ZN","ZB","M6E"]`.
- `alpha_intraday_micro.py` — `UNIVERSE=("MES","MNQ")`.

Crypto / FX strategies (out-of-scope for v1):

- `alpha_crypto.py` — `CRYPTO_UNIVERSE_DEFAULT=["BTC-USD","ETH-USD","SOL-USD"]`.
- `alpha_forex.py` — `FOREX_UNIVERSE_DEFAULT=["EUR-USD","GBP-USD",
  "USD-CAD","USD-JPY"]` (currently deferred per `strategies/__init__.py`).

### 5.2 Safest Phase D v1 target strategies

The v1 scanner emits **candidates only** — no strategy is rewired in
v1. For the future v2 promotion path, the natural first consumers are
strategies whose `target_universe` is already `None` AND whose asset
class is EQUITY/ETF:

- `alpha` (legend-driven; would observe candidates as auxiliary feed).
- `alpha_intraday` (already gates EQUITY/ETF on RS+RVOL+news).
- `beta_trend` (legend-driven).

Strategies with **strict allow-lists** (`omega_vol`, `alpha_options`,
`alpha_options_config`) must NEVER receive scanner-driven symbol
expansion in v1 or v2 without an explicit allow-list update —
their `_validate_universe` will raise.

Strategies with **explicit pair / hedge** universes (`delta_pairs`,
`gamma_reversion`, `omega`) require a pair-aware promotion path and
are not v1 candidates.

Futures and crypto strategies are out of v1 scope by construction.

---

## 6. Proposed scanner schema

### 6.1 Identity

- **Module name (future):** `chad.market_data.dynamic_universe_scanner`
- **Output artifact (future):** `runtime/dynamic_universe_candidates.json`
- **Schema version:** `dynamic_universe_candidates.v1`

### 6.2 Document shape

```json
{
  "schema_version": "dynamic_universe_candidates.v1",
  "ts_utc": "2026-05-16T22:08:40Z",
  "ttl_seconds": 300,
  "status": "ok",
  "source": {
    "provider": "dynamic_universe_scanner",
    "provider_status": "real",
    "active_universe_source": "runtime|config|legend|default_fallback",
    "inputs": {
      "relative_strength": "fresh|stale|missing",
      "volume_scan":      "fresh|stale|missing",
      "news_intel":       "fresh|stale|missing",
      "earnings_intel":   "fresh|stale|missing|partial"
    }
  },
  "candidates": [
    {
      "symbol":            "AAPL",
      "score":             0.0,
      "rank":              1,
      "reasons":           ["rs_class=strong", "rvol_class=high"],
      "warnings":          ["earnings_in_2d"],
      "rs_class":          "strong|neutral|weak|unknown",
      "rvol_class":        "high|elevated|normal|low|unavailable",
      "catalyst_strength": "high|medium|low|none",
      "earnings_days":     null,
      "data_available":    true
    }
  ],
  "summary": {
    "candidates_total":     25,
    "candidates_with_data": 25,
    "candidates_top_n":     10,
    "strong_count":         3,
    "high_rvol_count":      0,
    "catalyst_count":       7,
    "event_risk_active":    false,
    "event_risk_severity":  "low",
    "active_universe_size": 25
  }
}
```

Conventions:

- `ts_utc` is ISO-8601 Zulu, atomic-write timestamp.
- `ttl_seconds` matches RVOL cadence (~5 min) so dashboard staleness
  classifications align with sister feeds.
- `candidates` is the **entire** active equity/ETF universe, ranked, so
  consumers can apply their own top-N cuts. The scanner does NOT
  pre-truncate.
- `reasons` and `warnings` are stable string codes (e.g.
  `rs_class=strong`, `earnings_in_2d`, `rvol_unavailable`).

### 6.3 Atomic write contract

Same pattern as the existing publishers (`tmp + os.replace`). On any
hard failure the scanner publishes `status=error` with an empty
`candidates` list rather than crashing — fail-open consumers treat
missing/error/stale identically.

---

## 7. Scoring design (conceptual)

v1 scoring is **a transparent, linear, bounded combination of
already-published, already-validated signals**. The exact weights
will be tuned during the observation period (Phase D, item C below);
this section pins the conceptual shape, not the numbers.

### 7.1 Component scores (each in `[0, 1]` unless noted)

1. **Relative strength** — from `relative_strength.json`.
   `rs_class` mapped to band score: `strong → 1.0`, `neutral → 0.5`,
   `weak → 0.0`, `unknown → 0.5` (neutral default to avoid penalising
   missing data). Continuous refinement uses `excess_vs_spy_5d`
   clamped to a sensible band.
2. **RVOL** — from `volume_scan.json`. `rvol_class` mapped to band
   score with `unavailable → 0.5` (neutral fail-open). Continuous
   refinement uses `rvol` clamped, but only when `metric_type` is the
   real intraday snapshot, not the rolling-1m fallback.
3. **Catalyst** — from `news_intel.json`. `catalyst_strength` mapped
   to band score (`high → 1.0`, `medium → 0.6`, `low → 0.3`,
   `none → 0.0`). Negative-direction catalysts contribute the same
   magnitude but flip `reasons` to bearish — the scanner is
   side-agnostic; downstream strategies decide directionality.
4. **Liquidity** — from `volume_scan.avg_daily_volume * last_price`
   (reusing the existing `chad/utils/liquidity_gate.py` helper).
   Below floor → score 0; above floor → 1. Acts as a hard gate (any
   symbol failing liquidity is still listed but `score=0` with a
   `warning=below_liquidity_floor`).

### 7.2 Metadata-only / warnings (not scored in v1)

5. **Earnings proximity** — from `earnings_intel.json`. When
   `days_to_next_earnings ∈ [0, 2]`, emit `warning=earnings_in_2d`
   and `score *= 0.5` (penalty multiplier, not additive). Coverage at
   audit time is 1/25; until FMP coverage improves, this is metadata
   on the candidates that *do* have it.
6. **Event risk** — from `event_risk.json` top-level. Surfaced in
   `summary.event_risk_*` only. Does not enter per-symbol score in v1
   (no per-symbol mapping).

### 7.3 Composition

```
score = w_rs * rs_band
      + w_rv * rvol_band
      + w_cat * catalyst_band
      + w_liq * liquidity_band
```

with weights summing to 1.0. Initial defaults (to be tuned, not
shipped as gospel): `w_rs=0.35, w_rv=0.30, w_cat=0.25, w_liq=0.10`.
Earnings penalty applies after the linear combination.

Determinism: stable sort by `(-score, symbol)` for tie-breaking.

---

## 8. Safety rules

The following are **non-negotiable** for v1:

1. **No trading.** The scanner publishes a JSON artifact. It does not
   build signals, does not call IBKR, and is not in any execution
   path.
2. **No direct universe replacement.** The scanner MUST NOT write to
   `runtime/universe.json` or `config/universe.json`. The existing
   `chad.analytics.universe_builder` remains the sole writer of
   `runtime/universe.json`.
3. **No strategy config mutation.** No strategy `target_universe` is
   modified by the scanner. No env-var injection.
4. **Reuse the canonical loader.** The scanner reads its input
   universe via `chad.utils.universe_provider.load_active_universe()`
   and respects the returned `UniverseLoad.symbols` exactly.
5. **Fail-open.** Missing or stale inputs degrade gracefully: a
   symbol with no intel still appears in `candidates` with
   `data_available=false`, `score=0`, and `warnings` populated. The
   scanner does not raise on missing intel files.
6. **Dashboard-only first.** The first consumer is the dashboard,
   read-only. No strategy code reads
   `dynamic_universe_candidates.json` in v1.
7. **Quarantine-respectful.** The scanner never overrides
   `runtime/dynamic_caps_quarantine.json` exclusions. If a future v2
   promoter is added, it must AND against the quarantine set.
8. **Atomic JSON writes only.** Same tmp+replace pattern as sister
   publishers.

---

## 9. Phase D implementation order

After this design is accepted, the recommended implementation
sequence is:

- **A — Scanner publisher.** New module
  `chad/market_data/dynamic_universe_scanner.py`. Single function:
  read active universe + four intel files, emit
  `runtime/dynamic_universe_candidates.json`. New unit tests under
  `chad/tests/test_dynamic_universe_scanner.py`. **No deploy / systemd
  changes in this step** — runnable as `python3 -m
  chad.market_data.dynamic_universe_scanner` for the first soak.
- **B — Dashboard surface.** Add a read-only panel in
  `chad/dashboard/` displaying the latest candidates and the
  `summary` block. Show input freshness (RS / RVOL / news /
  earnings) so an operator can see which scores are stale.
- **C — Observation period.** Run for at least 5 trading days
  side-by-side with the current static-then-runtime universe.
  Compare: top-N candidate stability, agreement with manual
  watchlist picks, edge cases (earnings days, RS-without-RVOL
  symbols, weekend / off-hours). No code changes during this
  window unless a hard bug surfaces.
- **D — Optional promotion logic (v2).** Only after observation,
  introduce an explicit operator-approved promotion path. Two
  candidate designs to evaluate at that point:
  1. Promote into a new `runtime/universe_candidates.json` that
     `universe_builder` *optionally* unions into
     `runtime/universe.json` behind a flag.
  2. Promote per-strategy via a new opt-in
     `target_universe_override` field for the strategies listed in
     §5.2, gated by a feature flag.
  Neither design is locked in by this doc.

---

## 10. Blocked dependencies

The following Phase C blockers are **explicitly not** dependencies of
this v1 scanner:

- **C1 — Live Kraken Futures trading.** Blocked for Canadian
  deployment (per `docs/CHAD_PHASE_C_CLOSEOUT_AND_PHASE_D_READINESS_
  2026-05-16.md`). The scanner v1 ignores crypto perps entirely.
  Public intel feed (`kraken_futures_intel.json`) is also unused in
  v1.
- **C2 — Coinglass funding intel.** Blocked. Not consumed by v1.
- **C3 — IBKR DOM bids/asks.** Blocked / pending live proof. Not
  consumed by v1. Liquidity in v1 uses `volume_scan.avg_daily_volume`
  (a daily-bar derivative), not live DOM depth.

If any of these unblock later, they slot in as v2 enhancements
(crypto candidates, funding-aware scoring, true bid/ask depth as a
liquidity component) — but they are not required for v1.

---

## 11. Recommended next implementation task

**Build the `dynamic_universe_scanner` publisher only — no strategy
wiring, no dashboard wiring, no deploy changes.**

Concretely, the next Phase D code task (under standard governance:
one change at a time, no full rewrites, no direct config mutation,
verification sequence per CLAUDE.md) is:

1. Add `chad/market_data/dynamic_universe_scanner.py` implementing
   §6 schema and §7 scoring with the v1 defaults.
2. Add `chad/tests/test_dynamic_universe_scanner.py` covering: full
   coverage, missing RS file, missing RVOL file, missing news file,
   partial earnings file, empty active universe, atomic-write
   semantics.
3. Document the new artifact in the SSOT under Phase D item D1
   (separate doc edit; do not bundle).
4. Do not register a systemd timer for the scanner yet — the
   observation period in §9.C is done by manual / cron-only
   invocation first.

This task is fully self-contained: it adds one publisher and one
test file, reuses every existing intel feed without changing them,
and writes a new artifact with a new schema name that no existing
consumer reads. **No existing test should change behaviour; the
1959-passed full-suite baseline should still hold.**
