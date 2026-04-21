# CHAD Architecture Reference

Generated: 2026-04-21. Phase-8 sprint reference.

Companion document to `docs/post_overhaul_state_20260421.md` and to the
Phase-8 build survey at `reports/audit_m_build_survey_20260421.json`.

This document exists to close five Phase-8 items that the survey
confirmed are already fully implemented. Naming them explicitly, with
file paths and public entry points, converts them from "somewhere in
the codebase" to "documented, supported, next-contributor ready."

---

## Completed Capabilities (Phase-8 audit confirmed)

### E6 — Intent Audit Log

**Modules**
- `chad/core/decision_trace.py` — `DecisionTrace` recorder, append-only NDJSON, day-partitioned, schema-versioned.
- `chad/core/decision_trace_heartbeat.py` — periodic liveness marker so "no-decision" cycles are still accounted for.
- `chad/execution/paper_exec_evidence_writer.py` — `PaperExecEvidence` captured per fill (symbol, side, quantity, fill_price, strategy, source_strategies, broker, status, asset_class).
- `chad/portfolio/ibkr_paper_trade_result_logger.py` — final trade result after close.
- `chad/analytics/edge_quality_report.py` — rolls the per-trade evidence into edge-quality statistics.
- `chad/ops/daily_chad_report.py` — `IntentRow` dataclass used by the daily report renderer.

**What is captured.** Decision Trace is written as the "black box recorder" even when no trade occurs: LiveGate inputs, gate results, final decision, cycle id, timestamps. Paper exec evidence records each submitted/filled order with attribution to source strategies. Together they cover signal → intent → submission → fill → close.

**How to use.**
```python
from chad.core.decision_trace import DecisionTrace
trace = DecisionTrace(...)  # see module constants for required fields
trace.append()              # atomic NDJSON append, day-partitioned
```
NDJSON files live under `runtime/decision_trace/YYYYMMDD.ndjson` and `runtime/exec_evidence/`. They are rotated by UTC day.

**Phase-8 extension.** After E1 is fully populated (see below), audit rows automatically carry `confidence`, `entry_reason`, `regime_state`, `expected_pnl`, `created_at`, `ttl_seconds`. No writer changes are required for those six fields — `asdict()` on the intent picks them up.

---

### R1 — Graduated Circuit Breaker

**Module.** `chad/risk/profit_lock.py`

**Six-mode enum** (`ProfitLockMode`):

| Mode | Trigger | Effect |
|---|---|---|
| `NORMAL` | Default | `sizing_factor = 1.0` |
| `WARN` | Realised profit crosses warn threshold | Sizing factor reduced mildly |
| `LOCK1` / `LOCK2` / `LOCK3` | Progressive profit thresholds breached | Progressively tighter sizing factors |
| `HARD_STOP` | Drawdown threshold breached | Sizing factor → near-zero; new entries blocked |
| `INACTIVE_EQUITY_UNKNOWN` | Equity lookup failed and `CHAD_PROFIT_LOCK_ENABLE_ON_UNKNOWN_EQUITY` unset | Fail-closed: sizing factor → 0 |

**Trigger inputs.** Realised profit from trade ledgers (NDJSON) + account equity from a composite provider chain (first successful source wins).

**Env knobs.**
- `CHAD_PROFIT_LOCK_WARN_PCT`, `CHAD_PROFIT_LOCK_1_PCT`, `CHAD_PROFIT_LOCK_2_PCT`, `CHAD_PROFIT_LOCK_3_PCT`, `CHAD_PROFIT_LOCK_HARD_STOP_PCT` — mode thresholds (percent of equity).
- `CHAD_PROFIT_LOCK_1_FACTOR`, `CHAD_PROFIT_LOCK_2_FACTOR`, `CHAD_PROFIT_LOCK_3_FACTOR`, `CHAD_PROFIT_LOCK_HARD_STOP_FACTOR` — sizing factors per mode.
- `CHAD_PROFIT_LOCK_TTL_SECONDS` — state file freshness TTL.
- `CHAD_PROFIT_LOCK_ENABLE_ON_UNKNOWN_EQUITY` — override fail-closed default.

**State file.** `runtime/profit_lock_state.json` (atomic temp-and-rename writer). Consumers read `mode`, `sizing_factor`, `ts_utc`, `ttl_seconds`.

**How strategies interact.** The allocator (`allocator_v3.py`) reads the sizing factor and uses it as the top of the cap stack. Strategies do not call profit_lock directly — sizing is applied through the allocator pipeline.

**Phase-8 extension (R2).** The current six modes are driven only by equity drawdown. Phase-8 item R2 expands the trigger set with reject-rate, broker-latency, and data-staleness inputs while leaving the mode enum intact.

---

### R4 — Fractional Kelly Sizing

**Module.** `chad/risk/allocator_v3.py`

**Concept.** Kelly acts as a **ceiling** on per-strategy weight, never as a booster beyond the other caps.

**Formula** (`allocator_v3.py:423–434`):

```
kelly_raw[s] = mean[s] / variance[s]        # per-strategy, on active-minute returns
kelly_cap[s] = 0                           if kelly_raw[s] <= 0
             = clamp(kelly_raw[s], 0, kelly_max)  otherwise

target_weight[s] ∝ base_weight[s]
                × regime_mult[s]
                × vol_w[s]
                × corr_factor[s]
                × (1 + kelly_cap[s])
```

**Default.** `CHAD_ALLOC_V3_KELLY_MAX = 0.25` (25% fractional Kelly).

**To tune.** Set `CHAD_ALLOC_V3_KELLY_MAX` in the allocator's environment. Lower values are more conservative; setting to `0` turns Kelly off entirely (strategies still receive `base × regime × vol × corr`).

**Output.** The per-strategy kelly cap is written to `runtime/allocator_v3_state.json` under `metrics.kelly_cap` for audit.

**Phase-8 note.** Kelly is production-ready. Extending the composite cap pipeline into per-symbol and per-cluster tiers is Phase-8 item R5, which leaves the Kelly formula unchanged but places R5's limits lower in the stack.

---

### A2 — Strategy Router

**Modules.**
- `chad/portfolio/strategy_router.py` — pure decision function.
- `chad/core/live_execution_router.py` — live-loop adapter.
- `chad/core/routed_execution_runner.py` — runner for paper/test harnesses.
- `chad/core/suppression.py` — `SuppressionReason` enum consumed by the router for diagnostic codes (`NO_SIGNAL`, `LOWER_PRIORITY_THAN_SELECTED`, etc.).

**Public entry point.**

```python
from chad.portfolio.strategy_router import choose_strategy_route, RouteDecision

decision: RouteDecision = choose_strategy_route(
    available_signals={"alpha": [...], "beta": [...]},
    weights={"alpha": 0.6, "beta": 0.4},
)
# decision.selected_strategy : str | None
# decision.selected_symbols  : list[str]
# decision.reason            : str (SuppressionReason when no selection)
# decision.available_counts  : dict[str, int]
# decision.weights           : dict[str, float]
# decision.rejected_strategies : dict[str, str]
```

**Rules** (`strategy_router.py:84`):
1. Strategy must have ≥ 1 available signal.
2. Higher allocator weight wins.
3. Tie-break by `PREFERRED_ORDER` tuple.
4. **Fail-closed**: any exception or invariant violation returns `selected_strategy=None` with a reason. Never raises past the API boundary.

**Preferred order** (`strategy_router.py:45`):
```
alpha_futures, alpha, alpha_crypto, alpha_forex, alpha_options,
gamma, gamma_futures, gamma_reversion, beta,
omega, omega_macro, omega_vol, delta, delta_pairs
```

**Adding a new strategy.** Append the strategy name to `PREFERRED_ORDER` (decides tie-break priority) and ensure the allocator emits a weight for it. No other code change is required.

**Phase-8 extension (S1 signal stacking).** The router currently selects a single primary strategy per cycle. Stacking (N-of-M voting across families) is Phase-8 item S1 and extends `choose_strategy_route` with a vote threshold parameter.

---

### F1 — Expectancy Tracker

**Module.** `chad/analytics/expectancy_tracker.py`

**Inputs.** `data/trades/trade_history_*.ndjson` — skips trades tagged `pnl_untrusted` or `historical_pre_rebuild`.

**Output.** `runtime/expectancy_state.json` — per-strategy block with:

```
total_trades, wins, losses,
win_pnl_sum, loss_pnl_sum, total_pnl,
win_rate, avg_win, avg_loss, expectancy,
best_trade, worst_trade,
status   # "new" | "performing" | "watch" | "underperforming"
```

**Status gate** (`expectancy_tracker.py:23`):
- `new` — < 10 trades
- `performing` — win_rate ≥ 0.55 and expectancy > 0
- `watch` — win_rate ≥ 0.45 and expectancy > 0
- `underperforming` — otherwise

**Update cadence.** Runs on a systemd timer (`chad-expectancy-tracker.timer`, every 5 minutes per `reports/audit_m` inventory). The hot path does not block on it.

**How to query.**
```python
import json
state = json.load(open("runtime/expectancy_state.json"))
alpha_status = state["alpha"]["status"]
```

**Phase-8 extension (F1 → per-setup).** The current tracker is keyed by `strategy`. Per-setup expectancy requires a `setup_id` on each trade record, which comes from the E1 intent schema's `entry_reason` field once it is uniformly populated across strategies.

---

## Intent Object Schema (E1) — post-extension

Six canonical fields are now present on both `StrategyTradeIntent` dataclasses (IBKR + Kraken). Defaults keep every existing construction site backward-compatible.

| Field | Type | Default | Purpose (blocks which Phase-8 items) |
|---|---|---|---|
| `confidence` | `float` | `0.5` | S3 uniform confidence; S1 voting; E4 passive/aggressive |
| `entry_reason` | `str` | `""` | F1 per-setup scorecard; audit readability |
| `regime_state` | `str` | `"unknown"` | S4 adaptive thresholds; G3 regime-reduction |
| `expected_pnl` | `float` | `0.0` | R7 net-EV gate; E3 slippage (vs realized) |
| `created_at` | `str` (ISO8601 UTC) | auto (`utc_now_iso()`) | E2 stale expiry; E5 too-late-to-chase |
| `ttl_seconds` | `int` | `300` | E2 stale expiry |

**Helpers** (`chad/execution/intent_schema.py`):
- `utc_now_iso()` — timezone-aware ISO8601 UTC timestamp with microsecond precision.
- `validate_intent(obj) -> (ok, errors)` — duck-typed validator; works on dataclass, slotted class, or dict.
- `intent_is_fresh(intent, now_iso=...) -> bool` — reads `created_at + ttl_seconds`. Returns True when `created_at` is empty so partially-populated intents still flow.
- `INTENT_SCHEMA` — dict describing each field's type, default, range, and purpose.

**Populated at creation.** `chad/execution/execution_pipeline.py:566` (IBKR) and `execution_pipeline.py:748` (Kraken) now populate the six fields from upstream signal/order metadata where available (`getattr(..., default)` fall-through) and always stamp `created_at = utc_now_iso()`.

---

## Phase-8 Sprint Status (25 items)

Status codes: `DONE` = already implemented and now documented; `PARTIAL` = skeleton exists; `OPEN` = unbuilt. Session numbers reference `reports/audit_m_build_survey_20260421.json` → `build_sessions`.

| ID | Item | Status | Session |
|---|---|---|---|
| E1 | Intent object canonical schema | PARTIAL → **EXTENDED (this commit)** | 1 |
| E2 | Stale intent expiry | OPEN | 2 |
| E3 | Per-trade slippage tracking | PARTIAL | 6 |
| E4 | Passive/aggressive order logic by market state | OPEN (adjacent) | 8 |
| E5 | Too-late-to-chase drop | OPEN (adjacent at signal layer) | 2 |
| E6 | Intent audit log | **DONE** | — |
| R1 | Graduated circuit breaker | **DONE** | — |
| R2 | STOP bus trigger expansion | PARTIAL | 3 |
| R3 | Vol-adjusted per-trade sizing | PARTIAL | 3 |
| R4 | Fractional Kelly | **DONE** | — |
| R5 | Composite size cap | PARTIAL | 5 |
| R6 | Correlation cluster exposure cap | PARTIAL | 5 |
| R7 | Net-EV gate on signals | OPEN | 2 |
| S1 | Signal stacking with vote threshold | OPEN | 8 |
| S2 | Multi-timeframe confirmation | OPEN | 9 |
| S3 | Confidence score on every intent | PARTIAL → supported by this commit | 1 |
| S4 | Adaptive thresholds by regime | OPEN (adjacent) | 4 |
| S5 | Event-risk suppression calendar | PARTIAL | 4 |
| G1 | Regime classifier composite | PARTIAL | — |
| G2 | Strategy activation matrix (declarative) | PARTIAL | 4 |
| G3 | Regime mismatch = position reduction | OPEN | 3 |
| F1 | Per-setup expectancy scorecard | **DONE (strategy-level); per-setup OPEN** | 6 |
| F2 | Signal decay measurement | OPEN | 7 |
| F3 | Strategy health score | OPEN | 6 |
| F4 | Edge decay auto-reduce | OPEN (adjacent) | 7 |
| A1 | OMS / EMS separation | OPEN | 10 |
| A2 | Strategy router module | **DONE** | — |
| A3 | Backtest/paper/live unified interface | OPEN | 10 |
| A4 | Data freshness gate at routing | OPEN (adjacent) | 2 |

---

## Build Order

From `reports/audit_m_build_survey_20260421.json`:

1. **Session 1 (this commit)** — E1 intent schema extension + this doc. Unblocks 11 downstream items.
2. **Session 2** — E2, E5, R7, A4. Four gate items slot into a single routing interceptor.
3. **Session 3** — R2, R3, G3. Risk hardening.
4. **Session 4** — S4, S5, G2. Signal + regime tightening.
5. **Session 5** — R5, R6. Unified per-trade sizing pipeline.
6. **Session 6** — E3, F1 (per-setup extension), F3. Measurement primitives.
7. **Session 7** — F2, F4. Decay measurement + auto-reduce.
8. **Session 8** — E4, S1. Microstructure-aware execution + signal stacking.
9. **Session 9** — S2. Higher-timeframe confirmation.
10. **Session 10** — A1, A3. OMS/EMS separation + unified backtest/paper/live interface. MULTI_SESSION.

---

## References

- Phase-8 survey: `reports/audit_m_build_survey_20260421.json`
- Post-overhaul state: `docs/post_overhaul_state_20260421.md`
- Revert point: git tag `REVERT_PRE_OVERHAUL_20260419`, tarball `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`
- Rollback instructions: `/home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt`
