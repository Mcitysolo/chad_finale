# CHAD WIRING CENSUS — Intelligence/Feedback Feeds: Publisher → Consumer

**Date:** 2026-07-15 (compiled 2026-07-16) · **HEAD:** `a11eeaa` · **Mode:** READ-ONLY forensic — no code changed, no restarts.
**Method:** 7 parallel trace agents. For each feed: publisher (module + timer + cadence + artifact mtime + external API cost), then every production read site (grep imports + path literals across `chad/` + `ops/`, excluding tests and the publisher itself), each read classified **TRADING-PATH** (reaches signal / sizing / gating / execution) · **DASHBOARD-ONLY** · **SELF/INTEL-CACHE-ONLY** · **NONE**. A read counts as TRADING-PATH only when the value provably influences a signal/size/gate/order — not when it is dead-assigned, log-only, or behind a permanently-off flag.
**Verdicts:** WIRED · DECORATIVE (only dashboard/self/none consumers) · STARVED (has a trading consumer but effect is null because an upstream input is dead) · BROKEN_SILENT (consumed but stale/wrong/wrong-epoch, unnoticed) · RETIRED-CANDIDATE (publisher runs, artifact read nowhere in production).
**Cross-cutting fact:** every roster timer is `enabled + active` (verified `systemctl is-active`), so every DECORATIVE/STARVED verdict below is a *live* timer burning cycles/API for zero trading effect.

---

## Executive Summary

**Totals (35 feeds classified):** WIRED **13** · DECORATIVE **14** · STARVED **5** · BROKEN_SILENT **1** · RETIRED-CANDIDATE **2**.

**The suspicion in the motivation is confirmed:** roughly **two-thirds of CHAD's offense-enhancing intelligence is publisher-only** — timers feeding dashboards and a quarantined advisory cache, read by nothing that trades. Only 13 feeds reach a live signal/size/gate/order, and several of those are minor fail-open confidence nudges.

### Two systemic root causes explain almost every dead feed

**1. The advisory bias cache is dead — and it is the sole channel for 4 intel feeds.**
`strategy_intelligence_cache.json` → `chad/core/live_loop.py:2356 _apply_intelligence_bias()` is a *real, env-enabled* (`CHAD_STRATEGY_INTELLIGENCE_ENABLED=true`) trading-path consumer that suppresses intents whose confidence+adjustment < 0.20. But it is **inert**: the cache's `confidence[*].ts_utc` entries are dated **2026-04-04/07 (~100 days stale)**, and the per-entry 300 s freshness gate (`live_loop.py:1198/1246`) rejects 100 % of them. The 15-min refresh (`chad-strategy-intelligence-refresh`) that should re-stamp them is failing its LLM call (`POST api.anthropic.com/v1/messages → HTTP 400`, Ollama tier-0 timeout) and carrying forward the April seed. Everything routed through this one channel is therefore starved: **strategy_intelligence (16), trends (8), reddit (9), short_interest (10)**. Fix the refresh and all four re-arm at once.

**2. The exit defect starves the expectancy → winner-scaling amplifier.**
Per the EXIT-AUDIT, strategy exits are runtime-blind (`portfolio.positions` hardwired empty) so round-trips rarely close. The expectancy→winner_scaling→`dynamic_caps`→executor chain is **wired end-to-end and applies a live multiplier**, but its discriminative power is starved: only `gamma` clears `min_trades_for_scaling=5`, and only because of the 2026-07-13 TLT oversell-incident closes — not real strategy exits. With a one-strategy scoring pool the winner-**boost** path (>1.0) is structurally impossible; 16/17 strategies are pinned to a neutral 1.0. **setup_family_expectancy (18)** is empty for the same reason (0 closed trades for `alpha_intraday_micro`). Unblock: exit overlay ACTIVE → real closed round-trips across ≥2 strategies → expectancy repopulates.

### Money notes
- **Paid APIs feeding zero trading effect:** `earnings_intel` (**FMP**, 6 h refresh, key at `/etc/chad/fmp.env`) — DECORATIVE, consumed only by dashboard + a scanner-candidates file that is itself dashboard-only. And `strategy_intelligence` (**Anthropic Claude + OpenAI gpt-5 fallback**, 15-min) — STARVED, refresh failing, consumer inert.
- **Paid APIs that ARE wired:** `news_intel` and `volume_scan` both hit **Polygon.io** (key `/etc/chad/polygon.env`) and both feed live entry gates (catalyst / RVOL) — money well spent.
- **No paid API sits behind any straggler** (all local-file or localhost live-gate). FRED (macro), SEC EDGAR (13F), Kraken-public (crypto/futures intel), Finviz/Reddit/pytrends scrapes are all keyless/free.

---

## Master Table (one row per feed)

| # | Feed | Verdict | TP cons. | API | Publisher (module · timer · cadence) | Key evidence |
|---|------|---------|:------:|-----|--------------------------------------|--------------|
| 1 | regime_state (+activation_matrix) | **WIRED** | 3 | none | `analytics/regime_classifier.py:241`, in-loop (no timer) | live_loop.py:2289-2310 activation gate DROPS non-allowed strategies; alpha_crypto.py:642; regime_booster |
| 2 | choppy_regime_state | **WIRED** | 1 | none | `analytics/choppy_regime_detector.py:367` · choppy · 300s | live_loop.py:2129-2202 CHOPPY_GATE drops sub-0.5 non-exempt intents |
| 3 | event_risk | **WIRED** | 1 | none | `ops/event_risk_publish.py:540` · event-risk · 600s | regime_booster.py:190 hard veto → dynamic_caps |
| 3b | calendar_state | **RETIRED-CANDIDATE** | 0 | none | `ops/calendar_state_publisher.py` · 300s | only readers are masked/inactive paper_shadow_runner; payload empty (items_total:0) |
| 4 | macro_state (FRED) | **DECORATIVE** | 0 | free | `ops/macro_state_publish.py` · macro-state · 600s | 15 reader files, all dead: allocator_v3/savage never instantiated; only live gate = inert advisory cache |
| 5 | news_intel / catalyst gate | **WIRED** | 2 | **paid** | `market_data/news_intel_publisher.py` · 1800s | catalyst_gate.py:83 → alpha.py:396 & alpha_intraday.py:266 hard-block; Polygon.io + Yahoo fallback |
| 6 | relative_strength | **WIRED** | 2 | none | `market_data/relative_strength_publisher.py` · rs-refresh · 86400s | rs_gate.py:85 → alpha.py:437, alpha_intraday.py:276 ±0.10 conf nudge; local IBKR bars |
| 7 | volume_scan / RVOL | **WIRED** | 1 | **paid** | `market_data/volume_scan_publisher.py` · 300s | rvol_gate.py:82 → alpha_intraday.py:287 ±0.05 conf; Polygon.io snapshot |
| 8 | trends_state (Google Trends) | **STARVED** | 0 | free | `ops/trends_refresh.py`→`intel/trends_provider.py` · 14400s | signals:{} empty (pytrends blocked); feeds dead advisory cache; suppression-only |
| 9 | reddit_sentiment | **STARVED** | 0 | free | `ops/reddit_sentiment_refresh.py` · 7200s | 9 signals all NEUTRAL→0.0; feeds dead advisory cache |
| 10 | short_interest | **STARVED** | 0 | free | `ops/short_interest_refresh.py`→`intel/short_interest_provider.py` · 21600s | Finviz scrape degraded (uniform 0.08→0.0 adj); feeds dead advisory cache |
| 11 | institutional_consensus (13F) | **WIRED** | 1 | free | `scripts/update_institutional_consensus.py` (**cron** Sun 00:00, not systemd) | beta.py:61/298 → live always-active routing; SEC EDGAR keyless |
| 12a | earnings_intel (FMP) | **DECORATIVE** | 0 | **paid** | `market_data/fmp_earnings_intel_publisher.py` · 21600s | dashboard + scanner-candidates only; no earnings blackout gate reads it (gate uses config/event_calendar.json) |
| 12b | earnings_state | **DECORATIVE** | 0 | none | `ops/earnings_state_publish.py` · 1800s | empty stub `bootstrap_no_provider`; zero readers — orphaned |
| 13 | kraken_futures_intel | **DECORATIVE** | 0 | free | `market_data/kraken_futures_intel_publisher.py` · 300s | 260 KB republished every 5 min; **zero code readers anywhere** (trading uses crypto_derivatives instead) |
| 14a | options_greeks | **DECORATIVE** | 0 | none | `market_data/options_greeks_publisher.py` · 86400s | alpha_options.py:471 stamps `net_delta_estimate` into meta; code says "NEVER touches size/execution" |
| 14b | options_chains_cache | **WIRED** | 2 | free | `market_data/options_chain_refresh.py` · Mon-Fri 12:30Z | alpha_options.py:237 & omega_momentum_options.py:53 build tradable spreads; IBKR structural (no per-contract fee) |
| 15 | dynamic_universe_candidates | **DECORATIVE** | 0 | none | `market_data/dynamic_universe_scanner.py:548` · 300s | dashboard only; tradeable universe is STATIC (strategies hardwire target_universe); candidates never ingested |
| 16 | strategy_intelligence (Claude advisory) | **STARVED** | 1 | **paid** | `ops/strategy_intelligence_refresh.py` · 900s | consumer live_loop.py:2356 enabled but cache 100d stale; refresh Claude HTTP-400; June "SHADOW" verdict corrected |
| 17 | expectancy → winner_scaling → amplifier | **STARVED** | 1 | none | `analytics/expectancy_tracker.py` · 300s → `risk/winner_scaler.py` · winner-scaler | chain wired to executor (dynamic_risk_allocator.py:392) but only gamma scoreable (TLT incident); boost path structurally dead |
| 18 | setup_family_expectancy | **DECORATIVE** | 0 | none | `ops/update_setup_family_expectancy.py` · Mon-Fri 21:00 ET | 0 trading consumers (health-rule NOTIFY-only) AND empty (0 closed trades) |
| 19 | symbol_blocker | **WIRED** | 1 | none | `risk/symbol_performance_blocker.py` · 300s | live_loop.py:2481 `is_symbol_blocked()` → `continue` suppresses new entries (TLT blocked now) |
| 20 | regime_booster | **WIRED** | 1 | none | `risk/regime_booster.py` · 60s | dynamic_risk_allocator.py:366/392 cap = base·tier·winner·**regime_mult**; enforced on Kraken lane (see caveat) |
| 21a | execution_quality | **WIRED** | 2 | none | `ops/execution_quality_publisher.py` · 60s | live_gate.py:547/894 EXECUTION_QUALITY LIVE-gate → IBKR/Kraken execution runners |
| 21b | latency_state | **DECORATIVE** | 0 | none | (same publisher) · 60s | **zero readers**; stop_bus >2000ms trip reads `ibkr_status.json` directly (live_loop.py:644), not this file |
| 22 | crypto_derivatives / crowding filter | **WIRED** | 1 | free | `market_data/crypto_derivatives_publisher.py` · 300s | crypto_signal_filter.py:100 → alpha_crypto.py:734 confidence −0.20 (long_crowded) → gates/throttle/sizing; Kraken public |
| 23 | futures_roll_state / roll gate | **WIRED** | 2 | none | `market_data/futures_roll_publisher.py` · 86400s | roll_gate.py:90 → alpha_futures.py:567 hard entry skip; futures_expiry_gate → ibkr_bar_provider data-gate; static CME calendar |

### Row 24 — Straggler sweep (dedicated-timer feeds with 0 trading-path readers)

| # | Feed | Verdict | API | Publisher · cadence | Evidence |
|---|------|---------|-----|---------------------|----------|
| S1 | sector_rotation | DECORATIVE | none | `ops/sector_rotation_publish.py` · 3600s | `bootstrap_no_provider` stub → rotation_publish → nobody |
| S2 | rotation_state | DECORATIVE | none | `ops/rotation_publish.py` · 3600s | dead-end pointer; zero readers |
| S3 | business_phase | DECORATIVE | none | `ops/business_phase_tracker` · 1800s | dashboard/report/health only |
| S4 | governor_state | DECORATIVE | none | `ops/governor_publish.py` · 600s | feeds only advisory-LLM narrative (macro_interpreter→advisory_engine) |
| S5 | brain_returns | DECORATIVE | none | `ops/brain_returns_publish.py` · 300s | sole consumer allocator_v3.py:246 — **never instantiated** (orchestrator.py:592) |
| S6 | lifecycle_replay (state+coverage) | **BROKEN_SILENT** | none | `ops/lifecycle_replay_engine` · 1800s | observability-only drift artifact; ignores Epoch-3 (RED); embedded in lifecycle_truth as diagnostic, drives nothing |
| S7 | clean_soak_state | DECORATIVE | none | `ops/clean_soak_evaluator` · 300s | pure dead-end; zero readers (live_readiness does not read it) |
| S8 | decision_trace_heartbeat | **RETIRED-CANDIDATE** | none | `core/decision_trace_heartbeat` · 300s | no code reads the JSON (orchestrator.py:966 dual-writes it); timer survives only as sole source of `decision_trace_livegate_down` alert |

**Write-only self-logs (no timer — in-process diagnostics, noted not counted):** `market_metrics.json` (written by vol_adjusted_sizer + live_loop, zero readers), `correlation_overlay_health.json` (written by the live correlation overlay, zero readers), `strategy_routing_diagnostics.json` (written by live_loop, comment-only reference), `profit_routing.json` (trade_closer on close, report-only, stale 07-13), `feed_state.json` (price pipeline, only reader = advisory dashboard).

**WIRED stragglers (dedicated timer + real trading consumer — verified, not dead):** `change_canary_state.json` + `action_state.json` (mutation-state-publisher 60s → LiveGate gates live_gate.py:698/729), `live_readiness.json` (600s → live_gate.py:522/885 LIVE-readiness gate), `tier_state.json` (tier-manager 300s → dynamic_risk_allocator.py:637 + tier_instrument_gate.py:106), `strategy_throttle_state.json` (self-cache read by strategy_throttle_gate.py:122).

---

## Detailed findings by cluster

### Cluster 1 — Regime & risk-context gates (rows 1-4, 20) — mostly WIRED, macro dead
- **regime_state (1)** is the most load-bearing feed in the census. Written inside every live-loop cycle (`regime_classifier.py:241` ← `live_loop.py:2108`), it drives **two independent hard gates**: the activation matrix (`live_loop.py:2289-2310` → `portfolio/regime_activation.py:104` reads `config/regime_activation_matrix.json` and drops intents from non-allowed strategies) and alpha_crypto's own check (`alpha_crypto.py:642`, returns `[]` under `ranging`/`adverse`). Under the current `ranging` regime the matrix allows only `[beta, delta_pairs, gamma_reversion, gamma, omega_macro]` and alpha_crypto emits nothing — both actively suppressing flow right now.
- **choppy_regime_state (2)** — `live_loop.py:2129-2202` drops sub-0.5-confidence equity/options/futures intents when choppy or stale; crypto/forex exempt. Live gate, currently a no-op (`choppy_active=false`, 750 clean reads) but armed.
- **event_risk (3)** — reaches trading only via `regime_booster.py:190` (severity = hard veto forcing multiplier→1.0). The `rebalance_auto_executor_paper` reader is NOT trading-path (produces paper receipts only, no orders).
- **macro_state / FRED (4) — DECORATIVE despite 15 reader files.** Every concrete trading path is dead: `allocator_v3` and `savage_allocator` are never instantiated (`_allocator_factory` hard-returns `CorrelationOverlayStrategy`, orchestrator.py:617-620); `regime_tag` is attribution-only (labels closed-trade rows); the `live_loop.py:574` read is a fallback for keys that don't exist in the file; its one nominal live gate is the inert advisory cache. FRED is free (keyless public CSV).
- **regime_booster (20) — WIRED with an enforcement caveat.** `dynamic_risk_allocator.py:392` applies `regime_mult` to every per-strategy cap → `dynamic_caps.json`. This is enforced as a hard order chokepoint on the **Kraken lane** (`kraken_executor.py:371 check_risk`), but the **IBKR/equities-futures submit path** (`ibkr_adapter.submit_strategy_trade_intents`) has zero `check_risk`/`dynamic_caps` references — so on the IBKR lane the booster-scaled cap is not enforced at submission. Currently vetoed to 1.0 (ranging + conf 0.63), which is legitimate live logic.

### Cluster 2 — Equity entry gates (rows 5-7) — WIRED, the clean path
`news_intel`, `relative_strength`, `volume_scan` run entry-only through three fail-open gates (`catalyst_gate`, `rs_gate`, `rvol_gate`) into `alpha`/`alpha_intraday` (both `Status.ACTIVE`). Catalyst is a **hard block**; RS/RVOL are ±0.10/±0.05 confidence nudges. All three artifacts currently carry gate-relevant content, so they can fire. **Catalyst-gate question resolved:** the runtime wiring (`alpha.py:396`, `alpha_intraday.py:266`) is intact — only the 4 *tests* were stubbed, not production. Caveat: real firing frequency depends on alpha/alpha_intraday emitting EQUITY entries, and the 07-12 audit flagged a routing monoculture / thin equity flow — wiring is live, firing rate is likely low.

### Cluster 3 — The dead advisory channel (rows 8, 9, 10, 16) — 4 STARVED, 1 shared cause
All four converge on `strategy_intelligence_cache.json` → `live_loop._apply_intelligence_bias`. The mechanism is armed (env-enabled, real consumer) but produces zero effect because (a) the source data is dead/degraded (pytrends empty, reddit all-NEUTRAL, Finviz uniform-0.08), and (b) the cache is 100-day-stale and rejected wholesale by the 300 s freshness gate because the 15-min refresh's Claude call is failing (HTTP 400) + Ollama tier-0 times out. **This is the single highest-leverage fix in the census:** repair `chad-strategy-intelligence-refresh` so the cache re-stamps fresh `ts_utc`, and the one enabled consumer immediately begins suppressing sub-0.20 intents driven by all four feeds. Even healthy, the effect is suppression-only (never sizing).

**Row 16 corrects the June "SHADOW — no consumer in strategies" verdict:** there IS a trading-path consumer and it IS enabled; it is simply inert on stale data. Net effect today = shadow-in-practice, but the mechanism is wired and armed, not absent.

### Cluster 4 — The exit-starved amplifier (rows 17, 18) — STARVED by the exit defect
See Executive Summary §2. Chain trace (row 17), all TRADING-PATH, fully wired to the executor:
```
expectancy_tracker.compute() (reads data/trades/trade_history_*.ndjson)
  → runtime/expectancy_state.json
  → winner_scaler.py:216 reads → :228 runtime/winner_scaling.json
  → dynamic_risk_allocator.py:357 load_winner_multipliers → :392 cap = base·tier·winner_factor·regime_mult
  → orchestrator.py:931 build_payload → runtime/dynamic_caps.json
  → ibkr_executor.py:288 / kraken_executor.py:371 check_risk clamps order qty
BREAK: not in the wiring — at the SOURCE. Only gamma (67 TLT-incident closes) has ≥5 clean trades;
16/17 strategies get a static 1.0 and the winner-boost (>1.0) never fires with a 1-strategy pool.
```
`alpha_crypto`'s 2148 ledger rows are excluded from expectancy (validate_only / pnl_untrusted filter, `expectancy_tracker.py:81-90`). Row 18 `setup_family_expectancy` is doubly dead: 0 closed trades for `alpha_intraday_micro` (empty) AND no trading-path consumer even if populated (only a NOTIFY-only health rule). Unblock: exit overlay ACTIVE → real round-trips → both repopulate; row 18 additionally needs wiring into a sizing/gate to matter.

### Cluster 5 — Instrument/microstructure gates (rows 14b, 19, 21a, 22, 23) — WIRED
- **options_chains_cache (14b)** — supplies the tradable strike/expiry universe to `alpha_options` and `omega_momentum_options` (both SCALE-tier active); fail-closed via a 26 h usability window (the in-record `ttl_seconds=3600` is a red herring). **options_greeks (14a)** is decorative metadata (`net_delta_estimate` into signal meta; explicitly never sizing/execution).
- **symbol_blocker (19)** — real suppression at `live_loop.py:2481` (`is_symbol_blocked → continue`); TLT actively blocked (3 consecutive losses). Exit/flip/CLOSE intents correctly bypass.
- **execution_quality (21a)** — LIVE-authorization gate (`live_gate.py:894`) consumed by IBKR + Kraken execution runners. **latency_state (21b)** is decorative — zero readers; the >2000 ms stop_bus halt reads `ibkr_status.json` directly.
- **crypto_derivatives (22)** — the crowding filter (`crypto_signal_filter.py:100`) shaves −0.20 off `alpha_crypto` BUY confidence when `long_crowded` (BTC/ETH currently are); that confidence reaches net-exposure gate, throttle, and sizing. WIRED despite a comment claiming otherwise.
- **futures_roll_state (23)** — two trading-path consumers: `alpha_futures.py:567` roll entry-gate (hard skip) and `ibkr_bar_provider` expiry data-gate (skips dead contracts). Static CME calendar, no API.

### Cluster 6 — Fundamental feeds (rows 11, 12) — 1 WIRED, rest DECORATIVE
- **institutional_consensus / 13F (11)** — the only fundamental feed that trades, via `beta.py` under always-active routing (SEC EDGAR, free, weekly cron). Caveat: `beta` reads `portfolio.positions` for current-weight; the hardwired-empty-positions issue makes every consensus name read as underweight → persistent BUY bias (a latent BROKEN_SILENT flavor worth a follow-up, though the feed itself is live-wired).
- **earnings_intel / FMP (12a)** — the marquee waste: a **paid** feed refreshed every 6 h that gates/sizes nothing. **earnings_state (12b)** — an orphaned empty `bootstrap_no_provider` stub published every 30 min. The real event gate uses the operator file `config/event_calendar.json` + `event_risk.json`, not either earnings artifact.

---

## U-12 / INTEL-AUDIT (row 16 addendum)

**U-12 VERDICT — CONFIRMED.** `chad/intel/*` is an architecturally quarantined leaf. Proven both directions:
- **Forward** (who imports `chad.intel`): only `dashboard/api.py`, `ops/*` publishers & reports, and human-coach surfaces (`utils/telegram_bot.py:74`, `utils/coach_intents.py:814`). Zero trading-path importers. The lone `chad/execution/ibkr_client_ids.py` hit is a **comment** (clientId doc), not an import.
- **Reverse** (does `chad.intel` import execution/risk/live_loop/orchestrator/broker): **NO.** The only edge into `chad.core` is `intel/risk_explainer.py:29 → chad.core.mode` (read-only mode status) plus read-only analytics stats. No import of `chad.execution`, `chad.risk` (sizing/allocator), `chad.core.live_loop`, `chad.core.orchestrator`, or any broker/`ibkr_adapter`. Files self-declare the boundary (`intel/strategy_intelligence.py:45`, `intel/gpt_client.py:43`). Independently re-verified: `grep import chad/intel/*.py | grep -E "execution|chad.risk|live_loop|orchestrator|broker"` → empty.

**The only cross-boundary channel** by which advisory output can reach trading is the file-decoupled `strategy_intelligence_cache.json` → `live_loop._apply_intelligence_bias` edge — currently STARVED. The `regime_booster → strategy_intelligence.json` size-multiplier edge is a **phantom** (`STRAT_INTEL_PATH` defined once at `regime_booster.py:48`, never read; the booster actually reads regime_state/VIX/event_risk).

**INTEL-AUDIT Q1-Q4:**
- **Q1 (v8.5 wiring gap):** June "SHADOW — no consumer in strategies" is **superseded**. The narrative `strategy_intelligence.json` is dashboard/report only, but the sibling `strategy_intelligence_cache.json` has a real, enabled trading consumer (`live_loop.py:2356`). It is inert on stale data → shadow-in-practice, mechanism armed.
- **Q2 (U-12 import-graph proof):** above — no import edge either direction; one starved file channel.
- **Q3 (model-ID inventory, GAP-037/038):** In `chad/intel/`: `claude-haiku-4-5-20251001` (`claude_client.py:51/52/69` — **current/valid**); `claude-sonnet-4-6` (`:53/70` — active but **previous-gen**, current is `claude-sonnet-5`); `claude-opus-4-7` (`:71` — **vestigial**, present only in the cost table, absent from `TIER_MODELS`, never routed; previous-gen vs `claude-opus-4-8`); `gpt-5` (`advisory_engine.py:498`, OpenAI fallback, env-overridable); `phi3:mini` (`claude_client.py:729`, local tier-0). **No retired/deprecated (404-ing) IDs** — the concern is one-generation lag on the Claude tiers plus one vestigial cost-table entry, not broken IDs.
- **Q4 (machine-actionable output?):** Exactly one — the per-symbol confidence `adjustment` in `strategy_intelligence_cache.json`, consumed by `_apply_intelligence_bias` to suppress intents. No pending action, config mutation, or size multiplier is emitted for any trading component (the size-mult path is the phantom `STRAT_INTEL_PATH`). Everything else is narrative prose for humans.

---

## STOP-conditions / UNKNOWNs (not guessed)
- **Equity entry firing frequency (rows 5-7):** the catalyst/RS/RVOL gates are provably wired and can fire on current artifact content, but *how often* `alpha`/`alpha_intraday` actually emit EQUITY entries in the live loop was not measured (would require live-loop signal-log inspection). Wiring = live; real-world firing rate = unverified, likely thin per the 07-12 routing-monoculture finding.
- **beta ↔ empty-positions (row 11):** `institutional_consensus` is live-wired to `beta`, but whether the hardwired-empty `portfolio.positions` turns every consensus name into a false "underweight → BUY" is a latent BROKEN_SILENT that this census flags but did not fully trace to an executed order (resolve by inspecting a live beta signal batch against real broker positions).

---

## Summary JSON

```json
{
  "generated": "2026-07-16",
  "head": "a11eeaa",
  "rows": [
    {"feed": "regime_state", "verdict": "WIRED", "trading_path_consumers": 3, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "choppy_regime_state", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "event_risk", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "calendar_state", "verdict": "RETIRED-CANDIDATE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "macro_state", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "free", "unblocked_by": ""},
    {"feed": "news_intel", "verdict": "WIRED", "trading_path_consumers": 2, "monthly_api_cost_class": "paid", "unblocked_by": ""},
    {"feed": "relative_strength", "verdict": "WIRED", "trading_path_consumers": 2, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "volume_scan", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "paid", "unblocked_by": ""},
    {"feed": "trends_state", "verdict": "STARVED", "trading_path_consumers": 0, "monthly_api_cost_class": "free", "unblocked_by": "fix strategy_intelligence refresh (Claude HTTP-400) + pytrends emits signals + tighten 300s-consume vs 900s-refresh TTL"},
    {"feed": "reddit_sentiment", "verdict": "STARVED", "trading_path_consumers": 0, "monthly_api_cost_class": "free", "unblocked_by": "fix strategy_intelligence refresh + non-NEUTRAL reddit signals"},
    {"feed": "short_interest", "verdict": "STARVED", "trading_path_consumers": 0, "monthly_api_cost_class": "free", "unblocked_by": "fix strategy_intelligence refresh + repair degraded Finviz scrape (uniform 0.08)"},
    {"feed": "institutional_consensus", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "free", "unblocked_by": ""},
    {"feed": "earnings_intel", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "paid", "unblocked_by": ""},
    {"feed": "earnings_state", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "kraken_futures_intel", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "free", "unblocked_by": ""},
    {"feed": "options_greeks", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "options_chains_cache", "verdict": "WIRED", "trading_path_consumers": 2, "monthly_api_cost_class": "free", "unblocked_by": ""},
    {"feed": "dynamic_universe_candidates", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": "wire candidates into universe_builder or a per-strategy selector"},
    {"feed": "strategy_intelligence", "verdict": "STARVED", "trading_path_consumers": 1, "monthly_api_cost_class": "paid", "unblocked_by": "repair chad-strategy-intelligence-refresh so cache confidence ts_utc re-stamps current (300s gate rejects the April seed; consumer is enabled)"},
    {"feed": "expectancy_state_winner_chain", "verdict": "STARVED", "trading_path_consumers": 1, "monthly_api_cost_class": "none", "unblocked_by": "exit overlay ACTIVE -> closed round-trips across >=2 strategies -> expectancy repopulates"},
    {"feed": "setup_family_expectancy", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": "exit overlay populates ledger AND wire into a sizing/gate"},
    {"feed": "symbol_blocker", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "regime_booster", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "execution_quality", "verdict": "WIRED", "trading_path_consumers": 2, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "latency_state", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": "retire — stop_bus latency reads ibkr_status.json directly"},
    {"feed": "crypto_derivatives", "verdict": "WIRED", "trading_path_consumers": 1, "monthly_api_cost_class": "free", "unblocked_by": ""},
    {"feed": "futures_roll_state", "verdict": "WIRED", "trading_path_consumers": 2, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "sector_rotation", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "rotation_state", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "business_phase", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "governor_state", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "brain_returns", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": "allocator_v3 never instantiated (orchestrator.py:592)"},
    {"feed": "lifecycle_replay", "verdict": "BROKEN_SILENT", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": "lifecycle-replay Epoch-3 rebuild (PA-EP8v2 netting)"},
    {"feed": "clean_soak_state", "verdict": "DECORATIVE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": ""},
    {"feed": "decision_trace_heartbeat", "verdict": "RETIRED-CANDIDATE", "trading_path_consumers": 0, "monthly_api_cost_class": "none", "unblocked_by": "migrate decision_trace_livegate_down alert off the timer"}
  ],
  "totals": {"WIRED": 13, "DECORATIVE": 14, "STARVED": 5, "BROKEN_SILENT": 1, "RETIRED_CANDIDATE": 2},
  "paid_apis_feeding_nothing": [
    "earnings_intel (FMP earnings-calendar/price-target, 6h refresh) — DECORATIVE, dashboard + scanner-candidates only",
    "strategy_intelligence (Anthropic Claude + OpenAI gpt-5 fallback, 15min) — STARVED, refresh failing HTTP-400, consumer inert on 100-day-stale cache"
  ],
  "paid_apis_wired": [
    "news_intel (Polygon.io news + Yahoo fallback) — WIRED, catalyst hard-block",
    "volume_scan (Polygon.io snapshot) — WIRED, RVOL confidence gate"
  ],
  "starved_by_exit_defect": ["expectancy_state_winner_chain", "setup_family_expectancy"],
  "starved_by_dead_advisory_channel": ["strategy_intelligence", "trends_state", "reddit_sentiment", "short_interest"],
  "single_highest_leverage_fix": "repair chad-strategy-intelligence-refresh (Claude HTTP-400) — re-arms 4 feeds through strategy_intelligence_cache.json -> live_loop._apply_intelligence_bias",
  "u12_verdict": "CONFIRMED — chad/intel/* is a quarantined leaf: zero import edges into chad.execution/chad.risk/chad.core.live_loop/orchestrator/broker in either direction (only read-only chad.core.mode + analytics stats). The sole advisory->trading channel is the file-decoupled strategy_intelligence_cache.json -> live_loop._apply_intelligence_bias edge (currently STARVED). The regime_booster->strategy_intelligence.json size-multiplier edge is a phantom (dead STRAT_INTEL_PATH).",
  "commit": "committed to main (local); see git log — census commit is the WIRING-CENSUS 2026-07-15 entry",
  "push": "attempted, DENIED by permission policy (local-only, consistent with prior CHAD census docs)"
}
```
