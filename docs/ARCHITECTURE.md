# CHAD Architecture Reference

Last updated: 2026-04-22 (after Phase-8 Session 10 — Phase-8 complete).

Companion document to `docs/post_overhaul_state_20260421.md` and to the
Phase-8 build survey at `reports/audit_m_build_survey_20260421.json`.

This document maps the 29 Phase-8 survey items to concrete modules,
public entry points, and the sessions that delivered them. **Phase-8
is complete**: 29 of 29 items implemented across 10 sessions. The
remaining forward work is Phase-9 pre-live-capital calibration (see
"Phase-8 Complete — What This Means" at the bottom).

---

## Phase-8 Sprint — Session Summary

| Session | Commit    | Items                           | Description |
|---------|-----------|---------------------------------|-------------|
| 1       | `18d5b4e` | E1 + docs                       | Canonical intent schema extension (six fields + helpers) and this ARCHITECTURE.md. Unblocks 11 downstream items. |
| 2       | `575e11d` | E2, E5, R7, A4                  | Four pre-OMS routing gates (stale intent, too-late-to-chase, net-EV, data freshness). |
| 3       | `b02d06f` | E3, S3, R2                      | Per-trade slippage tracker + per-intent confidence + STOP-bus trigger module. |
| 4       | `d428ee3` | R2 wiring, G1, G2, G3           | STOP-bus wired into live_loop; composite regime classifier + declarative activation matrix + regime-transition reduction. |
| 5       | `1cf9f44` | S1, S2, S4, F4                  | Signal stacking (vote collector), multi-timeframe confirmation, adaptive regime thresholds, edge-decay auto-halt. |
| 6       | `d281848` | E4 (IBKR), F2, F3, G1-feed      | Passive/aggressive order selector (IBKR), signal-decay measurement, composite strategy-health score, market-metrics publisher. |
| 7       | `864b128` | R3, R5, R6, S5                  | Vol-adjusted sizer, composite size cap, correlation monitor, event-risk calendar (Gate 5). |
| 8       | `0b9930f` | E5 threading, A4 threading, E4 (Kraken), docs | Real `current_price` + `bar_timestamp` threaded into E5 and A4; E4 order-type selector wired into Kraken builder. |
| 9       | `81b8ddf` | A1 (protocols)                   | OMSInterface / EMSInterface Protocols + IbkrOMS / KrakenOMS / IbkrEMS / KrakenEMS wrappers introduced without moving implementation. PRESERVED_STATUS_STRINGS pinned. |
| 10      | (this)    | A1 (orchestrator), A3 (unified)  | execute_ibkr_cycle / execute_kraken_cycle thin orchestrators added; SimulatedOMS + SimulatedFillLedger + compare_backtest_to_paper utility; backtest_engine routed through EMS → gates → SimulatedOMS. |

---

## Phase-8 Sprint Status — all 29 items

Status codes:
- ✅ **COMPLETE** — implemented, wired, and tested. Includes items that were already DONE pre-sprint plus those delivered across Sessions 1-8.
- 🔄 **MULTI_SESSION** — scoped as a multi-session refactor; deferred.

| ID | Item | Status | Session(s) | Primary module |
|----|------|--------|------------|----------------|
| E1 | Intent canonical schema | ✅ COMPLETE | 1 | `chad/execution/intent_schema.py` |
| E2 | Stale intent expiry | ✅ COMPLETE | 2 | `chad/execution/routing_gates.py:stale_intent_gate` |
| E3 | Per-trade slippage tracking | ✅ COMPLETE | 3 | `chad/analytics/slippage_tracker.py` |
| E4 | Passive/aggressive order logic | ✅ COMPLETE | 6, 8 | `chad/execution/order_type_selector.py` |
| E5 | Too-late-to-chase drop | ✅ COMPLETE | 2, 8 | `chad/execution/routing_gates.py:too_late_to_chase_gate` |
| E6 | Intent audit log | ✅ COMPLETE | — | `chad/core/decision_trace.py` (+ paper evidence writer) |
| R1 | Graduated circuit breaker | ✅ COMPLETE | — | `chad/risk/profit_lock.py` |
| R2 | STOP bus trigger expansion | ✅ COMPLETE | 3, 4 | `chad/risk/stop_bus_triggers.py`, `chad/risk/stop_bus_state.py` |
| R3 | Vol-adjusted per-trade sizing | ✅ COMPLETE | 7 | `chad/risk/vol_adjusted_sizer.py` |
| R4 | Fractional Kelly | ✅ COMPLETE | — | `chad/risk/allocator_v3.py` |
| R5 | Composite size cap | ✅ COMPLETE | 7 | `chad/risk/composite_size_cap.py` |
| R6 | Correlation-cluster exposure cap | ✅ COMPLETE | 7 | `chad/risk/correlation_monitor.py` |
| R7 | Net-EV gate on signals | ✅ COMPLETE | 2 | `chad/execution/routing_gates.py:net_ev_gate` |
| S1 | Signal stacking with vote threshold | ✅ COMPLETE | 5 | `chad/analytics/vote_collector.py` |
| S2 | Multi-timeframe confirmation | ✅ COMPLETE | 5 | `chad/analytics/timeframe_confirmation.py` |
| S3 | Confidence score on every intent | ✅ COMPLETE | 3 | `chad/analytics/signal_confidence.py` |
| S4 | Adaptive thresholds by regime | ✅ COMPLETE | 5 | `chad/analytics/signal_confidence.py:regime_quality_from_state` + `chad/analytics/timeframe_confirmation.py` |
| S5 | Event-risk suppression calendar | ✅ COMPLETE | 7 | `chad/analytics/event_calendar.py` + `config/event_calendar.json` |
| G1 | Regime classifier composite | ✅ COMPLETE | 4, 6 | `chad/analytics/regime_classifier.py` + `chad/analytics/market_metrics_publisher.py` |
| G2 | Strategy activation matrix | ✅ COMPLETE | 4 | `chad/portfolio/regime_activation.py` + `config/regime_activation_matrix.json` |
| G3 | Regime mismatch = position reduction | ✅ COMPLETE | 4 | `chad/risk/regime_reduction.py` |
| F1 | Per-strategy expectancy scorecard | ✅ COMPLETE | — | `chad/analytics/expectancy_tracker.py` |
| F2 | Signal decay measurement | ✅ COMPLETE | 6 | `chad/analytics/signal_decay.py` |
| F3 | Strategy health score | ✅ COMPLETE | 6 | `chad/analytics/strategy_health.py` |
| F4 | Edge-decay auto-reduce | ✅ COMPLETE | 5 | `chad/risk/edge_decay_monitor.py` |
| A1 | OMS/EMS separation | ✅ COMPLETE | 9, 10 | `chad/execution/oms.py`, `chad/execution/ems.py`, `execute_ibkr_cycle` / `execute_kraken_cycle` orchestrators |
| A2 | Strategy router module | ✅ COMPLETE | — | `chad/portfolio/strategy_router.py` |
| A3 | Unified backtest/paper/live interface | ✅ COMPLETE | 10 | `chad/execution/oms.py:SimulatedOMS` + `chad/analytics/backtest_engine.py` routed through EMS → gates → SimulatedOMS |
| A4 | Data freshness gate at routing | ✅ COMPLETE | 2, 8 | `chad/execution/routing_gates.py:data_freshness_gate` |

---

## Intent Object Schema — post-Session-8

`StrategyTradeIntent` carries the canonical fields on both the IBKR
(`chad/execution/ibkr_executor.py`) and Kraken
(`chad/execution/kraken_executor.py`) dataclasses. Every field has a
default so pre-Session-1 construction sites continue to work; new
sessions added the fields additively.

| Field | Type | Default | Added in | Purpose |
|-------|------|---------|----------|---------|
| `confidence` | `float` | `0.5` | Session 1 | S3 uniform confidence; S1 voting; E4 passive/aggressive |
| `entry_reason` | `str` | `""` | Session 1 | F1 per-setup scorecard; audit readability |
| `regime_state` | `str` | `"unknown"` | Session 1 | S4 adaptive thresholds; G3 regime-reduction |
| `expected_pnl` | `float` | `0.0` | Session 1 | R7 net-EV gate |
| `created_at` | `str` (ISO8601 UTC) | `utc_now_iso()` | Session 1 | E2 stale expiry; E5 degraded mode |
| `ttl_seconds` | `int` | `300` | Session 1 | E2 stale expiry |
| `expected_price` | `float` | `0.0` | Session 3 | E3 slippage (vs realized fill) |
| `signal_strength` | `float` | `0.0` | Session 3 | S3 confidence input, sizing attenuation |
| `signal_family` | `str` | `"unknown"` | Session 5 | S1 vote aggregation across families |
| `order_urgency` | `str` | `"normal"` | Session 6 | E4 passive/aggressive order-type selection |
| `bar_timestamp` | `str` | `""` | Session 8 | A4 data-freshness validation |

**Helpers** in `chad/execution/intent_schema.py`:
- `utc_now_iso()` — timezone-aware ISO8601 UTC timestamp.
- `validate_intent(obj) -> (ok, errors)` — duck-typed validator.
- `intent_is_fresh(intent, now_iso=...) -> bool` — reads `created_at + ttl_seconds`.
- `INTENT_SCHEMA` — dict describing each field's type, default, range.

---

## Routing Gates — five-stage interceptor

`chad/execution/routing_gates.py:run_all_gates` applies five gates in
order before an intent reaches the OMS:

1. **A4 data_freshness** — rejects when the source bar is older than
   `max_bar_age_seconds` (default 172800s / 48h for the current daily-
   bar data source; tighten to 300s for intraday bars).
2. **E2 stale_intent** — rejects when `created_at + ttl_seconds` has
   elapsed.
3. **E5 too_late_to_chase** — rejects when the live price has drifted
   beyond `price_tolerance_pct` (default 0.5%) from the intent's
   reference price. Session 8 threads real `current_price` from the
   latest 1d bar; missing bars still degrade to the Session-2
   time-based path.
4. **R7 net_ev** — rejects when `expected_pnl − commission − spread <
   min_edge`. Inactive when `expected_pnl == 0.0` (not opted in).
5. **S5 event_risk** — suppresses entries near macro catalysts read
   from `config/event_calendar.json`. `urgency="high"` → reject;
   `urgency="normal"` → reduce quantity/volume/notional by 50% in
   place and pass.

Wire sites: `chad/execution/execution_pipeline.py:build_ibkr_intents_from_plan`
and `build_kraken_intents_from_routed_signals` both call `run_all_gates`
with the same config and event_calendar singleton.

---

## Per-Trade Sizing Pipeline — Session 7 composite

For EQUITY / ETF intents, `chad/execution/execution_pipeline.py:_apply_sizing_layer`
applies in order:

1. **R3 vol_adjusted_sizer** — `size × clamp(target_vol / realized_vol, 0.1, 2.0)`.
2. **R5 composite_size_cap** — `min(vol_adjusted, max_per_symbol, sector_remaining, adv_pct, equity_pct/ref_price)`. Each cap optional.
3. **R6 correlation_monitor** — if mean `|pairwise r|` of combined open+new book > threshold (0.7), multiply by `threshold / avg_corr` (floor 0.1).

Futures and forex bypass this layer (they have specialised upstream
sizers). Crypto rides the Kraken builder path which applies only the
S3 confidence attenuation plus minimum-volume gates.

Config: `config/sizing_config.json`. Monitors read daily bars from
`data/bars/1d/` — no live-price dependency.

---

## Stop Bus + Profit Lock — halt architecture

Two orthogonal halt mechanisms cooperate:

- **R1 profit_lock** (`chad/risk/profit_lock.py`) — equity-driven six-mode
  circuit breaker (NORMAL → WARN → LOCK1/2/3 → HARD_STOP). Writes
  `runtime/profit_lock_state.json`; the allocator reads `sizing_factor`
  from the file.
- **R2 stop_bus** (`chad/risk/stop_bus_state.py`, `chad/risk/stop_bus_triggers.py`) —
  operator- or trigger-set halt flag. Triggers: daily_loss, reject_rate,
  data_staleness, broker_latency. `chad/core/live_loop.py:run_once`
  evaluates triggers each cycle; any active trigger halts the cycle
  before any order is submitted. `runtime/stop_bus.json` (gitignored);
  `scripts/clear_stop_bus.py` clears the flag.

---

## Regime Layer — Session 4 / 6

- **G1 market_metrics_publisher** (`chad/analytics/market_metrics_publisher.py`)
  — each cycle, reads `data/bars/1d/*.json` and writes
  `runtime/market_metrics.json` with `realized_vol_percentile`, `adx`
  (proxy), `trend_slope`, `market_breadth`.
- **G1 regime_classifier** (`chad/analytics/regime_classifier.py`) —
  reads the metrics file and emits one of `trending_bull |
  trending_bear | ranging | volatile | unknown` to
  `runtime/regime_state.json`.
- **G2 activation_matrix** (`chad/portfolio/regime_activation.py` +
  `config/regime_activation_matrix.json`) — declarative
  `{regime → {strategy: allowed?}}` table. Fail-open on missing config.
- **G3 regime_reduction** (`chad/risk/regime_reduction.py`) —
  on any → adverse transition close 50% of open quantity; any →
  volatile close 30%; any → unknown warn only.

---

## Measurement Surfaces — Session 5/6

- **F1 expectancy_tracker** (`chad/analytics/expectancy_tracker.py`)
  → `runtime/expectancy_state.json` (per-strategy win rate, expectancy,
  status).
- **F2 signal_decay** (`chad/analytics/signal_decay.py`) — at fill
  time, records the entry price; later cycles compute alpha at T+1 /
  T+5 / T+15 / T+30 trading days from daily bars. Ledger at
  `data/signal_decay/<strategy>_decay.ndjson`.
- **F3 strategy_health** (`chad/analytics/strategy_health.py`) —
  composite `0.4·sharpe + 0.3·win_rate + 0.2·exec_quality +
  0.1·regime_alignment`, written to `runtime/strategy_health.json`.
- **F4 edge_decay_monitor** (`chad/risk/edge_decay_monitor.py`) —
  halts a strategy after N consecutive losses. Writes
  `runtime/strategy_allocations.json`; cleared via
  `scripts/clear_edge_decay.py` or file edit.

---

## OMS / EMS Separation — Session 9 / 10

After Session 10 the execution layer has an explicit seam::

    signals                                  OMSInterface
       │                                          │
       ▼                                          ▼
    EMS.build_intents_from_plan     ─►     OMS.submit(OrderRequest)
       │                                          │
       ▼                                          ▼
    StrategyTradeIntent             ─►          OrderResult
                         │
                         ▼
              routing_gates + vote_collector
                         │
                         ▼
            execute_ibkr_cycle / execute_kraken_cycle

**Public entry points** (`chad/execution/execution_pipeline.py`):

```python
from chad.execution.execution_pipeline import (
    build_execution_plan,      # routed signals → ExecutionPlan
    execute_ibkr_cycle,        # ExecutionPlan → list[OrderResult]
    execute_kraken_cycle,      # routed signals → list[OrderResult]
)
from chad.execution.ems import IbkrEMS, KrakenEMS
from chad.execution.oms import (
    IbkrOMS, KrakenOMS, SimulatedOMS, NullOMS,
    OMSInterface, OrderRequest, OrderResult,
    PRESERVED_STATUS_STRINGS,
    SimulatedFillLedger, compare_backtest_to_paper,
)
```

**Protocols**
- `OMSInterface` — `submit(OrderRequest) → OrderResult`, `cancel`, `get_status`.
- `EMSInterface` — `build_order_request(intent)`, `select_venue(intent)`.

**Status vocabulary pinned** (Session 9): `submitted`, `dry_run`,
`what-if`, `duplicate_blocked`, `error`, `unknown`. Tests assert set
equality on `PRESERVED_STATUS_STRINGS` so no future session can silently
rename a status string without failing the regression guard.

---

## Backtest — unified with paper/live (Session 10)

Pre-Session-10 the backtest engine (`chad/analytics/backtest_engine.py`)
bypassed the entire execution layer. As of Session 10 it routes signals
through the same Session-1-8 wiring that paper/live uses:

```python
engine = BacktestEngine(unified_execution=True, slippage_bps=5.0)
results = engine.run(strategy_name="alpha", universe=["SPY", "QQQ"])
fills = engine.fill_ledger.get_fills()       # SimulatedFillLedger
rejections = engine.fill_ledger.get_rejections()
```

Every signal now passes through:
1. `IbkrEMS.build_order_request(intent)` — same E4 / R3 / R5 / R6 logic
   as paper (via the Session-9 Protocol seam).
2. `routing_gates.run_all_gates(intent, ...)` — same five gates as paper
   (A4 data freshness, E2 stale intent, E5 price drift, R7 net EV, S5
   event risk).
3. `SimulatedOMS.submit(order_request)` — immediate fill at
   `limit_price ± slippage_bps`; records to `SimulatedFillLedger`.

**Validation utility:** `chad.execution.oms.compare_backtest_to_paper(
bt_fills, paper_fills)` returns a summary dict (`n_backtest`,
`n_paper`, `signal_overlap_pct`, `mean_fill_price_diff`,
`mean_slippage_diff_bps`) so operators can verify numerical agreement
between a backtest run and a paper-trading session over the same window.

**Legacy fallback:** `BacktestEngine(unified_execution=False)` preserves
the pre-Session-10 zero-slippage / no-gates fill path for numerical
baselines. Default is `unified_execution=True` — operators opt into the
legacy path.

---

## Phase-8 Complete — What This Means

All 29 items from the Phase-8 build survey
(`reports/audit_m_build_survey_20260421.json`) are implemented, wired,
and tested. CHAD now has:

- **Canonical intent schema** (E1) with 11 fields across IBKR and Kraken
  dataclasses, every field backed by a specific downstream use case.
- **Five-stage pre-OMS interceptor** (A4 / E2 / E5 / R7 / S5) validating
  every intent before submission, with real `bar_timestamp` and
  `current_price` values threaded through since Session 8.
- **Graduated halt system** (R1 profit_lock + R2 stop_bus) with four
  triggers (daily_loss, reject_rate, data_staleness, broker_latency).
- **Composite per-trade sizing** (R3 vol-adjusted + R5 composite_cap +
  R6 correlation_monitor) applied to EQUITY/ETF intents.
- **Signal-quality layer** (S1 voting + S2 MTF confirmation + S3
  confidence + S4 regime-adjusted thresholds + S5 event calendar).
- **Regime layer** (G1 classifier + market_metrics_publisher + G2
  declarative activation matrix + G3 transition reduction).
- **Measurement surfaces** (F1 expectancy + F2 signal decay + F3 health
  score + F4 edge-decay auto-halt).
- **Order-type selector** (E4) wired into both IBKR and Kraken paths —
  passive LMT at mid / aggressive LMT through market — never emits MKT.
- **OMS/EMS separation** (A1) with explicit Protocols and venue
  wrappers (IbkrOMS, KrakenOMS, SimulatedOMS).
- **Unified backtest interface** (A3) — backtests now exercise the
  same execution-layer logic as paper/live.

### What comes next (Phase-9 pre-live-capital)

Phase-8 finished the plumbing. Phase-9 is calibration:

1. **Regime classifier tuning** — the Session-6 ADX proxy is a simple
   ATR-scaled stand-in. Compare its classifications against real
   market days and either calibrate the 25 threshold or swap in a
   Wilder ADX implementation.
2. **Kelly fraction tuning** — `CHAD_ALLOC_V3_KELLY_MAX` defaults to
   0.25. With one month of paper expectancy data, re-evaluate.
3. **Slippage model calibration** — SimulatedOMS uses a flat bps
   slippage. Use `compare_backtest_to_paper` to fit the bps value to
   observed paper fill distributions.
4. **Live performance data** — accumulate at least 30 paper trading
   days before considering live capital.
5. **See** `docs/post_overhaul_state_20260421.md` for the go-live
   checklist and `CLAUDE.md` for the governance rules that still apply
   after Phase-8.

---

## References

- Phase-8 survey: `reports/audit_m_build_survey_20260421.json`
- Phase-8 session reports: `reports/phase8_session{1..8}_20260421.json`
- Post-overhaul state: `docs/post_overhaul_state_20260421.md`
- Revert point: git tag `REVERT_PRE_OVERHAUL_20260419`, tarball
  `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`
- Rollback instructions: `/home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt`
