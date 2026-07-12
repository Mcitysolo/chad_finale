# CHAD DORMANT-FEATURE CENSUS — 2026-07-12

**Authoritative inventory of everything BUILT/WIRED but intentionally (or defectively) NOT ACTIVE.**

- **HEAD:** `b49e82b` (SSOT v9.8, post-Epoch-3-reset)
- **Runtime anchors (re-read at sweep time 2026-07-12 ~20:45 UTC):**
  - **Regime = `ranging`** (`runtime/regime_state.json:19`, conf 0.623, `choppy_overlay.active=true`) — the regime gate is LIVE (`chad/core/live_loop.py:2145-2150`). Only **5 of 16** active heads are routed this cycle.
  - **Tier = `SCALE`** (`runtime/tier_state.json`; equity 1,000,521.76 CAD) — all 16 active strategies tier-enabled; no active head is tier-excluded right now.
  - **Registry = 18 declared / 16 active / 2 dormant** (`chad/strategy_registry.py`), confirmed.
  - **Execution posture = paper**, `CHAD_EXECUTION_MODE=paper`; not live-capable.
- **Method:** 5 parallel READ-ONLY sweeps — S1 code flags (`chad/`), S2 config (`config/*.json` + regime matrix + weights), S3 systemd (230 `chad-*` units + `.d/` drop-ins), S4a known defense/risk rows, S4b known offense/data rows. Every row cites `file:line` or unit name freshly verified. **Nothing was modified.**

---

## 0. EXECUTIVE SUMMARY

**46 dormant rows.** By state: 12 REGIME_GATED, 9 DISABLED_UNIT, 6 INERT_CONFIG, 5 BROKEN_SILENT, 4 UNWIRED, 4 ENV_GATED, 3 SHADOW, 2 REPORT_ONLY, 1 WEIGHTLESS.

**The five BROKEN_SILENT rows are the operator's priority** — bugs wearing a dormancy costume, where a dead/failed thing is presented as fresh/healthy:

1. **VaR publisher** — output 66 days stale, no scheduler exists, yet exported as live Prometheus gauges with `status_ok=1.0`.
2. **`chad-ibkr-daily-bars-refresh`** — service failed (exit-1) for 17h, no `OnFailure`; downstream reads `data/bars/1d/` as fresh.
3. **`chad-service-alert@chad-ibkr-bar-provider`** — the alert handler itself is latched-failed (exit-4) for ~2 weeks → bar-provider failures never alert.
4. **`chad-lifecycle-replay-engine`** — runs on schedule but emits `matched_symbols=0` RED (ignores Epoch-3); a fresh-but-empty artifact read as truth.
5. **`CHAD_ALLOCATOR_MODE=V3` drop-in** — set on the orchestrator unit but read by ZERO code; advertises a "V3 allocator" that never runs (the live overlay is `correlation_overlay`).

**Two prior-audit premises were CORRECTED by fresh evidence (see §7):**
- **EP6 bar-ts fix is NOT dormant — it is ACTIVE.** The bar-provider process (PID 1148488) started **2026-06-27**, 13 days after the fix landed (`a0f8c06`, 06-14); the running process has the fix, and the sibling `ibkr_historical_provider` was also fixed (`ee3e5f1`, PA-EP6b). The "committed-not-activated / sibling still buggy" claim is stale.
- **Drawdown publisher is NOT broken.** Its timer fires every 120s (last run 20:44:01, file seconds-fresh, CAD-denominated). The report-only guard is dormant *by design* (§4 R25), but the publisher is healthy — the "disabled unit" is only the timer-triggered oneshot `.service`.

**Systemd caveat (S3 headline correction):** ~30 `chad-*` services show `disabled` but are **timer-driven oneshots whose timers are enabled and firing every 1-9 min** — they are NOT dormant. Only masked units, disabled-with-timer-disabled units, and units whose *timer* is masked are genuinely dark. See §8 for the false-positive guard.

**Governance gap:** ~34 of 46 rows have **no pre-registered activation criteria** (`NONE_DEFINED`) — activation is left to undocumented operator judgement. See summary JSON `no_activation_criteria_defined`.

---

## 1. OFFENSE — STRATEGY HEADS (regime-gated, TRANSIENT dormancy)

In `ranging`, the allowed head list (`config/regime_activation_matrix.json:19-21`) is ONLY `beta, delta_pairs, gamma_reversion, gamma, omega_macro`. The 11 heads below are dropped from routed intents **this cycle** and wake when the regime classifier flips. `activation_mechanism` for all = "regime transitions to a listed regime"; `activation_criteria` = classifier flip (defined mechanism, no readiness gate). `risk_class` = OFFENSE.

| # | Head | State | Evidence | Wakes in regime |
|---|---|---|---|---|
| R1 | alpha | REGIME_GATED | `regime_activation_matrix.json:19-21` (ranging omits) | trending_bull/bear, volatile, unknown |
| R2 | alpha_crypto | REGIME_GATED | `regime_activation_matrix.json:19-21` | trending_bull/bear, volatile, unknown |
| R3 | alpha_futures | REGIME_GATED **+ futures-exec-disabled** (R17) | `regime_activation_matrix.json:19-21` | trending_bull/bear, volatile, unknown **AND** futures re-enable |
| R4 | alpha_intraday | REGIME_GATED | `regime_activation_matrix.json:19-21` | trending_bull/bear, volatile, unknown |
| R5 | alpha_options | REGIME_GATED | `regime_activation_matrix.json:22-26` (volatile only) | trending_bull/bear, volatile |
| R6 | beta_trend | REGIME_GATED | `regime_activation_matrix.json:19-21` | trending_bull/bear, volatile, unknown |
| R7 | delta | REGIME_GATED | `regime_activation_matrix.json:19-21` (ranging uses delta_pairs) | trending_bull/bear, volatile, unknown |
| R8 | gamma_futures | REGIME_GATED **+ futures-exec-disabled** (R17) | `regime_activation_matrix.json:19-21` | trending_bull/bear, volatile, unknown **AND** futures re-enable |
| R9 | omega | REGIME_GATED | `regime_activation_matrix.json` (volatile/unknown only) | volatile, unknown |
| R10 | omega_momentum_options | REGIME_GATED | `regime_activation_matrix.json:19-21` | trending_bull/bear, volatile |
| R11 | omega_vol | REGIME_GATED | `regime_activation_matrix.json` (volatile/unknown only) | volatile, unknown |

**Notes:** `adverse` regime = `[]` → silences ALL 16 (design kill-switch, matrix comment `_2`). `beta` is the only head enabled in every non-adverse regime. Secondary suppressor `choppy_overlay` (active now) raises the per-signal confidence floor by +0.15 → 0.65 (`live_loop.py:2040-2058`) but is not a hard family veto (crypto/forex exempt), so it does not by itself make a head dormant. **R3 and R8 are DOUBLE-dormant** (regime-gated AND futures-execution-killed at the executor).

---

## 2. OFFENSE — REGISTRY WEIGHTLESS / UNWIRED HEADS & ALLOCATION OVERLAYS

| # | Name | Layer | State | Evidence | Why dormant | Activation mechanism | Criteria |
|---|---|---|---|---|---|---|---|
| R12 | alpha_intraday_micro | strategy | **WEIGHTLESS** | `strategy_registry.py:120-130` (Status.DORMANT, "handler registered, runs every cycle, no allocator weight → 0 fills"); absent from `config/strategy_weights.json` weights; excluded from `tier_manager.py:96-100` SCALE-wildcard set | Handler registered but no weight → allocator cap=0; weight policy deferred (`ops/pending_actions/BOX-033_...`) | Add weight key to `strategy_weights.json` (as Pending Action) + flip registry → ACTIVE | NONE_DEFINED (BOX-033 unresolved) |
| R13 | alpha_forex | strategy | **UNWIRED** | Handler exists (`chad/strategies/alpha_forex.py:264/331/383`) but registration commented out (`chad/strategies/__init__.py:195-199`, import `:75-76`); in `DEFERRED_STRATEGIES` (`:63-65`); policy `enabled=False` (`chad/policy.py:602-604`); "emits no signals" (`alpha_forex.py:16`) | FX universe not mapped to bar/price context; would emit 0 signals & pollute audits | Uncomment registration block + import; remove from `DEFERRED_STRATEGIES`; assign weight; map FX universe | NONE_DEFINED |
| R14 | allocator_v3 / Kelly ceiling | risk/allocation | **INERT_CONFIG** | `AllocV3Strategy` def `chad/core/orchestrator.py:591-614` — **zero construction sites**; `_allocator_factory()` (`:617-619`) hard-returns `CorrelationOverlayStrategy()`; sole call `:853`. `CHAD_ALLOCATOR_MODE` read-count in code = **0** (env set only in drop-in `40-allocator-v3.conf`, see R46). `CHAD_ALLOC_V3_ENABLED` default True (`allocator_v3.py:136`) gates never-executed code | Kelly/V3 built but factory never constructs it; the switching env was never wired to a reader | Code edit to `_allocator_factory()` to construct `AllocV3Strategy()` | NONE_DEFINED |
| R15 | savage_allocator | risk/allocation | **UNWIRED** | `SavageAllocatorStrategy` def `chad/core/orchestrator.py:572-588` — zero construction sites; `apply_savage_overlay` reachable only from that class (`:583`); `CHAD_SAVAGE_ALLOCATOR_ENABLED` default True (`savage_allocator.py:133`) gates never-run code | Only `correlation_overlay` is instantiated by the factory | Code edit to `_allocator_factory()` | NONE_DEFINED |
| R16 | ProfitRouter beta-budget injection | strategy | **ENV_GATED** (default OFF) | `chad/strategies/beta.py:281-287` (`CHAD_PROFIT_ROUTER_BETA_INJECTION` default OFF; "When OFF, no new code paths execute"), consumer `:359`; not set in any drop-in | Experimental beta-budget cap tied to realized-PnL router | `Environment=CHAD_PROFIT_ROUTER_BETA_INJECTION=1` | NONE_DEFINED |

---

## 3. EXECUTION LANES (dormant / kill-switched / unwired)

| # | Name | State | Evidence | Why dormant | Activation mechanism | Criteria | Risk |
|---|---|---|---|---|---|---|---|
| R17 | Futures execution lane (FUT/FOP submit) | **ENV_GATED (forced OFF)** | Gate `chad/execution/futures_gate.py:31-51`; chokepoints `ibkr_adapter.py:2858`, `live_loop.py:2513`; router `ibkr_trade_router.py:136`. Forced off by `/etc/systemd/system/chad-live-loop.service.d/91-disable-futures-exec.conf` (`CHAD_DISABLE_FUTURES_EXECUTION=1` + `CHAD_DISABLE_FUTURES=1` + `CHAD_FUTURES_EXECUTION_ENABLED=0`). **Code default (no env) = futures ENABLED** | Emergency safety guard (drop-in header, 2026-05-30) — stops repeated MCL/M6E FUT submission loop pending code-level cooldown/rejection fix; also load-bearing for Bug-A leak dormancy | Remove/flip the 3 env vars in the drop-in + `daemon-reload` + restart `chad-live-loop` | NONE_DEFINED (blocker = cooldown patch + Bug-B book disposition) | OFFENSE→DEFENSE (guard) |
| R18 | Exit-only execution lane | **UNWIRED** (fail-closed) | `chad/execution/exit_only_consumer_paper.py:183-191` (requires LiveGate `operator_mode==EXIT_ONLY` AND `allow_exits_only` AND entry lanes disabled); `exit_only_router.py:14`. No `*exit*` systemd unit exists | Capability reserved for operator EXIT_ONLY posture; current posture is paper with entry lanes enabled | LiveGate `operator_mode`→EXIT_ONLY + run consumer/router | Three-part LiveGate gate in-code (`:185-190`) — HAS criteria | DEFENSE |
| R19 | Legacy shared-`ib` execution path (own-connection kill-switch OFF-side) | **INERT_CONFIG** (rollback lever) | `chad/core/live_loop.py:167-174` (`_execution_owns_connection()` default ON) + legacy else-branch `:221-229`. `CHAD_EXECUTION_OWN_CONNECTION` absent from all drop-ins | Own-connection (L1-CLD U7) is the production default; legacy shared-`ib` is the dormant rollback path | Set `CHAD_EXECUTION_OWN_CONNECTION=0/false/off` to activate legacy path | NONE_DEFINED (rollback lever only) | DEFENSE |
| R20 | Paper-shadow canary runner (micro-order arm) | **ENV_GATED** (not armed) + masked units | `chad/core/paper_shadow_runner.py:78` (`ARM_ENV_NAME="CHAD_PAPER_SHADOW_ARM"`), `:93-99` (`is_armed()` false unless exact `ARM_PHRASE`); units `chad-paper-shadow-{exec,runner,tick}` are masked/disabled (see §5) | Fail-safe — places real micro paper orders only when explicitly armed with exact phrase | Set `CHAD_PAPER_SHADOW_ARM=<exact phrase>` + run `--execute` (units also need unmask) | Exact-phrase match (in-code) | OFFENSE/DATA |
| R21 | Kraken realized-fill lane | **SHADOW** (validate_only) | Kraken exec is ON in paper mode (drop-ins `30-kraken-mode.conf`/`30-kraken.conf`, `CHAD_KRAKEN_MODE=paper_kraken`), but the roundtrip runner is `validate_only` only (`chad/portfolio/kraken_roundtrip_runner.py:277/435/554-557`; `kraken_trade_result_logger.py:142-143`) → no realized fills | Crypto lane wired for validation/evidence, not realized paper fills | Flip the roundtrip runner off validate_only (code/config) | NONE_DEFINED | OFFENSE/DATA |

---

## 4. DEFENSE — RISK GATES (shadow / report-only / disabled-default / unwired)

| # | Name | State | Evidence | Why dormant | Activation mechanism | Criteria |
|---|---|---|---|---|---|---|
| R22 | Margin / buying-power BLOCK gate (Phase C) | **SHADOW** | `config/margin_block.json` `mode="shadow"` (frozen 2026-07-06, `no_env_overrides=true`). Wired `ibkr_adapter.py:2544-2620` (called at `:2692` before idempotency claim); shadow never-block branches `:2555-2561`, fail-OPEN on error `:2600-2601`; built at boot `live_loop.py:203-207`. `should_block` False when `not is_enforce` (`margin_shadow_gate.py:392-400`). **IBKR-lane only** (imports only IBKR `buying_power_provider`, `margin_shadow_gate.py:56`); Kraken not wired | Part-7 shadow→enforce ladder step 2 — evaluate + log + record evidence to `data/margin_shadow/`, block nothing | Edit `config/margin_block.json` `mode`→`enforce_paper`→`enforce_live` (separate authorization commit each; no env override) | Thresholds frozen/pre-registered; **go-decision NONE_DEFINED**. Prereqs dormant: account-summary publisher + persistent ledger + burst protection |
| R23 | Kraken margin/BP lane | **UNWIRED** | `chad/risk/kraken_bp_provider.py` built (parse/cache/report, fail-closed) but **zero functional importers**; only a docstring ref in `margin_shadow_gate.py:3`; `chad/core/kraken_execution.py` has no margin gate | Provider built ahead of a Kraken-lane gate that was never wired; crypto lane has no margin gate at all | Build/wire a Kraken-lane margin gate consuming `kraken_bp_provider` (code) | NONE_DEFINED |
| R24 | ML / XGB loss-probability veto | **SHADOW** (double env-gated) | Shadow scoring always runs (`live_loop.py:2359-2410`, logs `ML_SHADOW`); only skips ENTRY when both gates pass. `enforcement_enabled()` reads `CHAD_ML_VETO_ENABLED` (default OFF, `ml_veto_predictor.py:405-411`); `_canary_strategies()` empty→enforcement denied for every strategy (`:392-402`). Neither env set. Model `shared/models/xgb_veto_model.json`. Prior ~99% MES loss-prob unchanged | Fail-open shadow soak to tune threshold before enforcing; ENTRY-only, canary allowlist | Set `CHAD_ML_VETO_ENABLED=1` AND `CHAD_ML_VETO_CANARY_STRATEGIES=<csv>` | In-code gates: threshold 0.65, ≥100 manifest samples, ≤30d stale — HAS criteria; **numeric go-bar NONE_DEFINED** |
| R25 | Global drawdown guard enforcement | **REPORT_ONLY** | `chad/risk/drawdown_guard.py:11-19`; `enforcement_active=False` hardcoded (`:233/258/277`); no `stop_bus`/halt reader in live_loop/orchestrator. Output `runtime/drawdown_state.json` seconds-fresh, CAD-denominated (currency bug FIXED), no phantom drawdown today. Publisher timer is LIVE (fires 120s) — the guard, not the publisher, is dormant | GAP-016A report-only batch; enforcement deferred | Code change: set `enforcement_active` + wire halt into stop_bus (no env exists) | NONE_DEFINED (`CHAD_DRAWDOWN_HALT_PCT` default -15.0 feeds only the report) |
| R26 | Global kill-switch (default limits) | **INERT_CONFIG** (default off) | `chad/policy.py:667` `kill_switch_enabled=False` in `build_default_global_limits` | Conservative default; enabled only on operator action | Set `kill_switch_enabled=True` in global limits path | NONE_DEFINED |
| R27 | Signal-stacking vote quorum | **INERT_CONFIG** (neutralized) | `config/signal_stacking_config.json:6` `min_votes=1`; `vote_collector.py:90` "With min_votes=1, submit() is effectively pass-through" | Quorum lowered 2→1 (Audit-O) because the mix fires only 1-2 families/cycle → quorum is a no-op | Set `min_votes:2` in config | "raise back to 2 when intraday+crypto+reversion consistently firing" (soft) |
| R28 | Auto-rebalancer (autonomy_bounds) | **INERT_CONFIG** (default-off) | `config/autonomy_bounds.json:5` `enabled_default=false`; read by `ops/rebalance_auto_executor_paper.py`; "No live trading. Paper receipts only." | Phase-12 bounded automation intentionally off | Set `enabled_default:true` | Drift gate defined: `max_position_weight_drift ≥0.08` OR `turnover ≥0.20` (`:9-11`) — HAS criteria |

---

## 5. DATA & OBSERVABILITY — masked / disabled units, report-only publishers, inert config

**Genuinely dark systemd units** (masked → `/dev/null`, or disabled with timer also disabled/masked). `activation_mechanism` = `systemctl unmask/enable --now <unit>` unless noted; `activation_criteria` = NONE_DEFINED for all.

| # | Name | State | Evidence (unit → ExecStart) | Why dormant | Risk |
|---|---|---|---|---|---|
| R29 | Kraken polling collectors | DISABLED_UNIT (masked) | `chad-kraken-collector` (`chad.portfolio.kraken_portfolio_collector`), `chad-kraken-market-collector`, `chad-kraken-ohlc-collector` — all masked | Superseded by `chad-kraken-ws` (enabled+active WS feed) | DATA |
| R30 | Kraken paper lane | DISABLED_UNIT (masked) | `chad-kraken-paper-trader` (`chad.core.kraken_paper_trader`), `-paper-notify`, `-pnl-watcher`, `-validate` — all masked | Crypto paper-trading/reporting lane dark (paper realized-fills happen in the live-loop lane, R21) | OFFENSE/OBS |
| R31 | Polygon lane | DISABLED_UNIT (masked/decommissioned) | `chad-polygon-stocks` masked (→/dev/null since Apr 20); `chad-bars-builder` (`ndjson_bars_builder`), `chad-bars-validate`, `chad-daily-bars-backfill` (`polygon_daily_bars_backfill`) masked | Migrated to IBKR-native market data (`ibkr_price_provider.py:7` "Replaces Polygon…"). Embedded polygon helpers (`volume_scan_publisher.py:348-393`, `catalyst_news_provider.py:253-321`) remain key-gated LIVE (`/etc/chad/polygon.env` present) — those are NOT dormant | DATA |
| R32 | Portfolio-merge unit (Kraken→snapshot) | DISABLED_UNIT | `chad-portfolio-merge` disabled, no timer, `[Install]` never enabled; `After=chad-kraken-collector` which is MASKED → doubly dark. `chad.portfolio.merge_kraken_into_snapshot` | **Kraken legs are NOT merged into the portfolio snapshot by this unit** — currency/portfolio audits must not assume it runs | DATA |
| R33 | IBKR cash collector | DISABLED_UNIT (timer disabled) | `chad-ibkr-cash-collector` (`ibkr_cash_collector collect`) | Superseded by active `chad-ibkr-collector` | DATA |
| R34 | Price-cache refresh | DISABLED_UNIT (timer disabled) | `chad-price-cache-refresh` (`price_cache_refresh --feed-dir …`) | Superseded by active `chad-ibkr-price-refresh --provider ibkr` | DATA |
| R38 | Crypto risk-off / risk-notify publishers | DISABLED_UNIT (svc+timer disabled) | `chad-crypto-risk-off` (`crypto_risk_off_publisher`), `chad-crypto-risk-notify` (`crypto_risk_notifier`) | Crypto risk-off publishers dark — **DEFENSE gap if crypto risk-off is assumed live** | DEFENSE |
| R39 | Stop-refresh | DISABLED_UNIT (svc+timer disabled) | `chad-stop-refresh` (`/usr/local/bin/chad-stop-refresh.sh`) | Stop-loss refresh dark; **coverage elsewhere NOT confirmed — flag for follow-up** | DEFENSE |
| R40 | Superseded/completed dark units (consolidated) | DISABLED_UNIT (masked/timer-off) | `chad-paper-trade-executor` (masked; superseded by active `chad-paper-trade-exec`), `chad-tier-sync` (masked; superseded by `chad-tier-manager`), `chad-reconciliation.timer` (masked; superseded by `chad-reconciliation-publisher`), `chad-weekly-investor-report` (masked; → `chad-weekly-report`), `chad-symbol-bench` (timer-off; → `chad-symbol-blocker`), `chad-warmup` (timer-off; completed), `chad-full-cycle-refresh` (timer-off; → live-loop), `chad-ibgateway-ibc` (disabled; → `chad-ibgateway.service`), `chad-paper-shadow-{exec,runner,tick}` (masked) | Each superseded by a live equivalent or a completed one-shot | OBS/mixed |

**Report-only / inert config publishers** (config IS read; path off):

| # | Name | State | Evidence | Why dormant | Activation mechanism | Criteria | Risk |
|---|---|---|---|---|---|---|---|
| R35 | Soak evidence writers | ENV_GATED (default OFF) | `chad/ops/soak/evidence_writers.py:30-41,81-88` (`writers_enabled()` false unless `CHAD_SOAK_EVIDENCE_WRITERS` truthy; "ACTIVATION GATE (default OFF)") | Default-off activation gate, prepared as Pending Action | `Environment=CHAD_SOAK_EVIDENCE_WRITERS=1` drop-in | NONE_DEFINED | OBS |
| R36 | Sector-rotation tilt (rotation_rules) | REPORT_ONLY | `config/rotation_rules.json:5-6` `advisory_only=true, write_pointer=true`; read by `ops/rotation_publish.py` (not hot path); `max_abs_tilt 0.05` never applied to orders | Designed advisory-only; writes pointer/report, no broker calls | Set `advisory_only:false` (still needs a tilt consumer) | NONE_DEFINED | OBS |
| R37 | Regime size booster | REGIME_GATED (neutral now) | `config/regime_booster_policy.json:6` `favorable_regimes=[trending_bull,trending_bear]`; `runtime/regime_booster.json` multiplier=1.0, active=false, "vetoed: unfavorable_regime_ranging" | Booster lifts size only in trending regimes; ranging → forced 1.0 | Regime → trending_bull/bear | conf ≥0.70 + favorable regime + VIX/event vetoes clear — HAS criteria | OFFENSE |
| R41 | `config/feature_flags.json` | INERT_CONFIG | File is a stub (`schema_version` + "Bootstrap config… Replace with production values"); grep of `chad/`+`ops/`+`scripts/` for readers = **ZERO** | Placeholder from an SSOT-parity audit; gates nothing | Requires code to load+consume it | NONE_DEFINED | OBS |

---

## 6. BROKEN_SILENT — DEFECTS WEARING A DORMANCY COSTUME (priority)

These look dormant/healthy but are actually failures where a dead/failed thing is presented as fresh/healthy. **Not intentional dormancy — bugs.**

| # | Name | Evidence | The defect | Fix mechanism | Risk |
|---|---|---|---|---|---|
| R42 | **VaR publisher** | `runtime/var_state.json` mtime **2026-05-07 13:39:30 UTC → 66 days stale**; file's own `status="ok"`, `ttl_seconds=3600`. Compute script `ops/var_publisher.py` exists but **NO systemd unit/timer for it exists at all** (zero `var` publisher files in `/etc/systemd/system/`). Consumer `chad/ops/metrics_server.py:491-505` reads the 66-day file and emits `chad_var_95_1day_usd=2325.94`, `chad_var_99_1day_usd=3289.63`, `chad_var_status_ok=1.0` with **no mtime/ttl/staleness guard**, via the active `chad-metrics.service` | A dead-66-days VaR compute is scraped as a fresh, healthy risk gauge (`status_ok=1`). TTL is ignored | Add a `chad-var-publisher.timer/.service` (mirror drawdown); add a staleness guard in `metrics_server._var_drawdown_lines()` so stale→`status_ok=0` | OBSERVABILITY (DEFENSE-intent metric silently dead) |
| R43 | **`chad-ibkr-daily-bars-refresh`** | Timer ENABLED+active (next 02:45), but **service Active=failed (exit-1) since 2026-07-12 02:46 UTC, ~17h**; no `OnFailure=`. ExecStart `chad.market_data.nightly_bars_refresh` | Nightly EOD-bars writer to `data/bars/1d/` fails at runtime (matches "5/10 futures contract-resolution fail"); downstream reads `data/bars/1d/` as fresh → stale-as-fresh | Fix the exit-1 root cause (contract resolution); add `OnFailure=` | DATA |
| R44 | **`chad-service-alert@chad-ibkr-bar-provider`** | Template `chad-service-alert@.service`; **instance Active=failed (exit-4), latched since 2026-06-27 (~2 wk)** | The `OnFailure` alert HANDLER itself is dead → bar-provider failures do NOT alert (meta-broken alerting) | `systemctl reset-failed` + fix the exit-4 in the alert script | OBSERVABILITY |
| R45 | **`chad-lifecycle-replay-engine`** | Timer active (ran 20:18, every 30 min); service RUNS but emits `matched_symbols=0` RED (ignores Epoch-3) → writes a fresh `runtime/lifecycle_replay_state.json` that is functionally empty | Consumers may read a fresh-looking but empty/RED artifact as truth | PA-EP8v2 netting/Epoch-3 rebuild (code) | OBSERVABILITY |
| R46 | **`CHAD_ALLOCATOR_MODE=V3` drop-in** | `/etc/systemd/system/chad-orchestrator.service.d/40-allocator-v3.conf:2` sets `CHAD_ALLOCATOR_MODE=V3`; grep of `chad/`+`ops/` for the var = **ZERO readers**; V3 path never constructed (R14) | An operator drop-in signals "V3 allocator active"; the code never reads the var and never runs V3 — a silent, operator-facing false capability signal (subtype: config-misrepresentation) | Remove/correct the drop-in; or wire `_allocator_factory()` to actually honor it | OFFENSE (config integrity) |

**Related non-broken clarifications:** `CHAD_ALLOC_V3_ENABLED` / `CHAD_SAVAGE_ALLOCATOR_ENABLED` both default True but gate only never-executed code (dead-code flags, folded into R14/R15) — misleading but not a runtime hazard. `config/feature_flags.json` (R41) is an inert stub, not a masked-failure, so it is classed INERT_CONFIG, not BROKEN_SILENT.

---

## 7. CORRECTIONS — prior "dormant" claims REFUTED by fresh evidence (NOT dormant)

| Claim (prior audit/brief) | Fresh evidence | Verdict |
|---|---|---|
| **EP6 bar-ts fix committed but NOT activated (bar-provider unrestarted); sibling still buggy** | Fix present `chad/market_data/ibkr_bar_provider.py:164-178`; commit `a0f8c06` 2026-06-14. Bar-provider **PID 1148488 started 2026-06-27 11:36 UTC** (`ActiveEnterTimestamp`/`ps lstart`) — 13 days AFTER the fix → running process has it. Sibling `ibkr_historical_provider.py:38,311` now imports the shared helper (fixed by `ee3e5f1` PA-EP6b, 06-14) | **ACTIVE / DEPLOYED — not dormant.** Both writers fixed |
| **drawdown-publisher / feed-watchdog / ibkr-watchdog / health-monitor / decision-trace-heartbeat are BROKEN_SILENT (disabled units)** | All five have ENABLED+active timers that fired <2 min before sweep (e.g. drawdown 20:44:01, feed-watchdog 20:44:01, ibkr-watchdog 20:44:13, health-monitor 20:44:36, heartbeat 20:45:02). The `disabled` state is only the timer-triggered oneshot `.service` | **REFUTED — running on schedule, output fresh.** (The drawdown *guard* is still report-only by design, R25) |
| **dynamic universe screener is a dead/decorative unit (BROKEN_SILENT?)** | `chad/market_data/dynamic_universe_scanner.py:546` "intentionally never writes universe.json"; only observability consumers (`dashboard/api.py:616`, `strategy_intelligence.py:450`); timer fired today 20:44:36, `Result=success` | **DORMANT-BY-DESIGN (REPORT_ONLY), not broken.** Decorative candidates freshly published |

---

## 8. FALSE-POSITIVE GUARD — looks dormant, is OPERATIONALLY ENABLED (do NOT count)

- **~30 "disabled" services are timer-driven oneshots with enabled+firing timers** (S3): brain-returns, choppy-regime, clean-soak-evaluator, crypto-derivatives-refresh, dynamic-universe-scanner-refresh, intel-cache, news-intel-refresh, futures-roll-refresh, options-chain/greeks-refresh, fmp-earnings, kraken-futures-intel, micro-eod-flatten, ibkr-paper-fill-harvester, ibkr-paper-ledger-watcher, reddit-sentiment, rs-refresh, setup-expectancy, shadow-snapshot, short-interest, strategy-intelligence-refresh, trends-refresh, universe-refresh, volume-scan, advisory-pre-market, ibgateway-nightly-restart, reconciliation-publisher, drawdown-publisher, feed-watchdog, ibkr-watchdog, health-monitor, decision-trace-heartbeat.
- **Code-default-OFF flags that are ON in prod via live drop-ins:** `CHAD_ALWAYS_ACTIVE_ROUTING=1` (`10-always-active-routing.conf`), `CHAD_STRATEGY_INTELLIGENCE_ENABLED=true` (live-loop + orchestrator `override.conf`), `CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1` (`80-loss-guard-enforce.conf` — loss guard is **ENFORCING**, despite config text saying `report_only`), `CHAD_SPAM_GOVERNOR_ENABLED` (orchestrator `50-spam-governor.conf`; *note:* orchestrator-scoped — live_loop's pipeline may still see OFF, worth a cross-check), `CHAD_KRAKEN_MODE=paper_kraken`/`CHAD_KRAKEN_ENABLED=1` (Kraken lane ON in paper — but realized fills still validate_only, R21).
- **Code-default-ON gates (active, not dormant):** `CHAD_RTH_GATE` (equity/ETF market-hours gate, `rth_gate.py:46-57`; only pytest sets it 0), `CHAD_CHASSIS_ENFORCEMENT` (default "1", `dynamic_risk_allocator.py:498-500`), `allow_market_orders` (IbkrConfig default True).

---

## 9. SWEEP SOURCES

- **S1 (code):** grep `chad/` for shadow/report_only/validate_only/enforce/dry_run/`enabled=False`/`CHAD_*` env reads with inactive defaults + kill-switches.
- **S2 (config):** `config/*.json` mode fields, `regime_activation_matrix.json` per-regime exclusions (mapped against current `ranging`), `strategy_weights.json`/`tiers.json` weightless heads, reader-grep for inert configs.
- **S3 (systemd):** 230 `chad-*` unit files, masked vs disabled vs timer-live triage via `list-timers`/`is-active`, `.d/` env drop-ins.
- **S4a/S4b (known rows):** fresh re-verification of ML veto, drawdown guard, VaR, margin gate, Kraken margin lane, alpha_forex, alpha_intraday_micro, dynamic screener, Kelly/V3, EP6, Polygon — with the DORMANT-BY-DESIGN vs BROKEN_SILENT determination for each.

*Read-only census. No code, config, unit, mask, enable, or service state was changed. Runtime figures (SCR, equity, regime) are moving targets — re-read before relying on them.*
