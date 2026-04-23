# CHAD Unified SSOT v8.1
**Version:** 8.1
**Date:** 2026-04-23
**Status:** Active — Paper Trading
**Supersedes:** All previous SSOT versions

This is the master reference for the CHAD trading system. If it is in this
document, it is the authoritative answer. If another document disagrees,
this one wins until the next SSOT revision is cut.

---

## 1. VISION & BUSINESS MODEL

### What CHAD Is

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a **business**, not a bot.
The business earns money by running a diversified portfolio of systematic
trading strategies over multiple asset classes, compounding the profits
through a structured allocation rule, and eventually paying a salary to the
operator without touching the capital base.

The product goal is plain: **the operator checks their phone at the end
of the day, sees what CHAD made, and does nothing**. Everything downstream
of that — risk management, execution, reconciliation, health monitoring —
is automation designed to make "do nothing" the safe default.

### The Three-Phase Model

- **Phase 1 — Build**: assemble the engines, prove they work on paper, pass
  every self-test. (Current phase.)
- **Phase 2 — Stabilize**: 60–90 days of consistent paper/live performance
  with real measurement data flowing through every feedback loop.
- **Phase 3 — Salary**: automatic distributions from realized profit only,
  governed by the salary rules below. Core capital is never touched.

### The Business Structure

CHAD runs three simultaneous money-making surfaces that feed one another:

1. **Trading Engine (14 active strategies)** — generates daily cash flow
   from short- and medium-horizon edge: intraday momentum, mean reversion,
   options, futures, crypto, and macro hedging.
2. **Wealth Builder — Beta (institutional-consensus compounder)** — a
   long-term holder that rebalances slowly into the most-held large-cap
   names across the world's top institutions (Berkshire, Bridgewater,
   Citadel, BlackRock, Vanguard, Appaloosa, Pershing Square). Funded by
   30% of realized profits via the profit router.
3. **Profit Routing (50/30/20 rule)** — on every profitable close, CHAD
   splits the realized PnL three ways:
   - **50% → trading_capital**: stays in the trading account and
     compounds immediately.
   - **30% → beta_allocation**: earmarked for Beta to deploy against the
     institutional-consensus basket.
   - **20% → amplifier_allocation**: boost for the best-performing
     strategy.

   Routing is advisory-only today (single IBKR paper account); decisions
   are written to `runtime/profit_routing.json` as an accounting ledger.

### Capital Buckets

- **Core Capital** — the principal. Compounds, never withdrawn.
- **Growth Capital** — reinvested profit used to size trades up over time
  via the SCR and allocator.
- **Salary Account** — holds withdrawn profit destined for the operator.
  Funded only by the salary rule.

### Salary Rules (when CHAD can pay you)

All four conditions must hold:

1. **60–90 days consistent profit** rolling window.
2. **Drawdown under control** — no profit-lock LOCK2/3 events in the window.
3. **Multiple strategies profitable** — not a one-strategy outlier.
4. **SCR is CONFIDENT**, not WARMUP or CAUTIOUS.

**Withdrawal formula:** withdraw 20–40% of *net monthly profit only* from
the amplifier bucket. Never withdraw from the core balance. Never
withdraw during a locked profit state.

---

## 2. SYSTEM OVERVIEW

### Architecture Summary

- **Multi-engine profit system.** 16 strategies run in parallel, each
  producing signals on its own schedule and universe.
- **Dynamic capital allocation.** The dynamic risk allocator + allocator_v3
  + savage overlay set per-strategy caps each cycle based on weights,
  performance, Kelly fraction, and regime.
- **Regime-aware activation.** A composite regime classifier and a
  declarative matrix gate which strategies fire in which regime.
- **Full risk stack.** Five pre-OMS routing gates, composite sizing,
  profit-lock circuit breaker, STOP bus, edge-decay auto-halt, correlation
  cap, and event-risk suppression all sit between a signal and a fill.

### Infrastructure

- **Server:** AWS EC2 `ip-172-31-8-43` (Ubuntu 24.04, kernel 6.17.0-1009-aws).
- **Repo root:** `/home/ubuntu/chad_finale` (git `main` branch).
- **Python:** `python3` via `/home/ubuntu/chad_finale/venv`.
- **IBKR paper account:** `DUK902770`, ~$998,109.89 equity (as of
  2026-04-23T19:55Z).
- **Kraken paper:** connected, 0.0012 BTC + 252.85 CAD ≈ $184.58 USD.
- **Dashboard:** https://chadtrades.com (TLS valid through 2026-07-19,
  Let's Encrypt auto-renewal via `certbot.timer`).
- **Execution mode:** PAPER — `CHAD_EXECUTION_MODE=paper`.

### Active Services (13)

| Service | Role |
|---|---|
| `chad-live-loop.service` | Hot-path: broker truth → signals → gates → execution each cycle. |
| `chad-trade-closer.timer` | Scheduled exits (stops, targets, time stops). |
| `chad-scr-sync.timer` | Refreshes `runtime/scr_state.json` from shadow endpoint. |
| `chad-reconciliation-publisher.timer` | Publishes broker-vs-guard reconciliation snapshots. |
| `chad-ibgateway.service` | IB Gateway paper port 4002 process. |
| `chad-ibkr-bar-provider.service` | Polls IBKR `reqHistoricalData` every 30s to build 1m + 1d bars. |
| `chad-dashboard.service` | Fintech UI behind nginx/TLS. |
| `chad-backend.service` | Dashboard + reports JSON backend. |
| `chad-orchestrator.service` | Risk-budget publisher (dynamic caps). |
| `chad-kraken-ws.service` | Kraken websocket for crypto prices/balances. |
| `chad-metrics.service` | Prometheus-style metrics scraper. |
| `chad-shadow-status.service` | Shadow (sizing_factor source) HTTP endpoint on :9618. |
| `chad-telegram-bot.service` | Operator alerts + morning brief. |

Plus infra: `chad-xvfb / chad-x11vnc`, `nginx.service`, `certbot.timer`.

Intentionally disabled: `chad-reconciliation.timer` (dual-writer risk),
`chad-options-chain-refresh.service`. Retired/masked: `chad-polygon-stocks`,
`chad-bars-validate`, `chad-daily-bars-backfill`.

---

## 3. THE 16 STRATEGIES

Each strategy is a pure signal generator: it reads the live context, emits
`TradeSignal` objects, and never touches the broker or disk. Weights below
come from `config/strategy_weights.json` (applied action
`beta_carveout_from_beta_trend_20260423`, 2026-04-23).

### alpha — Intraday tactical momentum brain
- **Weight:** 16% of capital (largest single bucket).
- **What it does:** Buys trending large-caps when momentum and regime
  agree, exits on ATR trail / trend break / vol spike / time stop.
- **Signal family:** momentum.
- **Venue:** IBKR (equity / ETF).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** firing; 32 samples in health tracker.
- **Health score:** 0.717 (normalized Sharpe 1.0, win rate 0.389).
- **Technical:** trend + vol regime gate, ATR-normalized momentum filter,
  anti-chase on range/ATR, liquidity proxy via dollar volume, per-day
  per-symbol signal cap = 3.

### beta — Institutional long-term compounder (the wealth builder)
- **Weight:** 5% (seeded 2026-04-23 from a 5-pt carve-out of beta_trend).
- **What it does:** Slowly accumulates the most-held large-caps across
  7 tracked institutional funds. Holds for weeks. Emits add-signals only
  when materially underweight the consensus target. See §4 for the full
  concept.
- **Signal family:** trend / allocation.
- **Venue:** IBKR (equity).
- **Active regimes:** every regime except `adverse`.
- **Current status:** seeded; consensus snapshot last refreshed
  2026-04-23T13:24Z.
- **Health score:** n/a (no trades yet).
- **Technical:** signals capped per cycle; once-per-day per-symbol gate
  inherited from beta_trend pattern; funded further by 30% of realized
  profits via `profit_router`.

### beta_trend — Legend-driven long-term ETF / equity allocator
- **Weight:** 20% (reduced from 25% on 2026-04-23 when Beta was carved out).
- **What it does:** Builds a core ETF + large-cap holder based on a
  "legend" consensus weight. Once-per-day per-symbol hard gate.
- **Signal family:** trend.
- **Venue:** IBKR (equity / ETF).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** active; inherits the WEALTH-MODE throttle.
- **Technical:** `max_signals_per_day=20`, `min_weight=0.05`,
  `max_symbols=10`, `min_cash=$10k` guard.

### gamma — Activated swing engine
- **Weight:** 7%.
- **What it does:** Regime-switching swing trader. In trend regimes it
  continues the move; in range regimes it mean-reverts to EMA. Works
  with or without an existing position (no anchor dependency).
- **Signal family:** trend / mean_reversion (regime-switched).
- **Venue:** IBKR (equity).
- **Active regimes:** ranging, volatile, unknown.
- **Current status:** armed; waits for a clean trend or range signature.
- **Technical:** EMA + ATR entries; deterministic exits (time stop, ATR
  trail, trend break, vol spike).

### gamma_futures — Futures mean-reversion counterpart
- **Weight:** 5%.
- **What it does:** Trades the same micro-futures universe as alpha_futures
  (MES, MNQ, MCL, MGC) but fades overextension instead of chasing trend.
- **Signal family:** mean_reversion.
- **Venue:** IBKR (futures).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** active; 1 sample in health tracker.
- **Health score:** 0.90 (win rate 1.0 on 1 trade — tiny sample).
- **Technical:** RSI > OB band + price > upper Bollinger for shorts; RSI <
  OS + price < lower Bollinger for longs; 1.2% risk budget, $40k max
  notional per instrument.

### gamma_reversion — ETF statistical mean-reversion
- **Weight:** 4%.
- **What it does:** Fades overextended ETFs (SPY, QQQ, GLD, TLT) using
  RSI + Bollinger + Z-score + ROC confluence. Long and short.
- **Signal family:** mean_reversion.
- **Venue:** IBKR (ETF).
- **Active regimes:** ranging.
- **Current status:** waiting for ranging regime (currently volatile).
- **Technical:** 3/3 confluence required (RSI > 72 AND (price > BBupper OR
  z > 1.8) AND ROC > 0 for shorts, mirror for longs). 2.5×ATR stop, 15-bar
  time stop, SMA20 cross = target.

### delta — Cross-asset convexity hunter
- **Weight:** 2% (lowest — lowest-frequency by design).
- **What it does:** Low-frequency, high-convexity opportunistic hitter
  across the legend universe. Regime- and event-aware.
- **Signal family:** momentum / trend / breakout blend.
- **Venue:** IBKR (cross-asset).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** firing selectively; 7 samples in health tracker.
- **Health score:** 0.825 (win rate 0.75 on 7 trades — strong early read).
- **Technical:** trend via EMA crossover, breakout via rolling-high + ATR
  buffer, momentum via (price − EMAfast)/ATR; dynamic conviction sizing
  with floors/caps; first to shut down when risk rises.

### delta_pairs — Market-neutral ETF pairs trader
- **Weight:** 5%.
- **What it does:** Dollar-neutral pairs trades on correlated ETF pairs
  (SPY/QQQ, SPY/IWM, QQQ/IWM) using z-score of the price ratio.
- **Signal family:** mean_reversion.
- **Venue:** IBKR (equity, two-leg).
- **Active regimes:** ranging.
- **Current status:** waiting for ranging regime.
- **Technical:** entry at |Z| ≥ zscore_entry, exit at |Z| ≤ zscore_exit,
  stop at |Z| ≥ zscore_stop; equal-notional legs; linked by `pair_id`.

### alpha_futures — Futures momentum engine
- **Weight:** 9% (second-largest allocation).
- **What it does:** ATR-based momentum follower on MES, MNQ, MCL, MGC.
  Buy when price > EMAfast > EMAslow; sell on break-down; breakout entries
  at highest-high.
- **Signal family:** momentum.
- **Venue:** IBKR (futures).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** firing; 44 samples in health tracker.
- **Health score:** 0.699 (normalized Sharpe 1.0, win rate 0.333).
- **Technical:** 1.25× risk multiple, max 5 contracts per instrument,
  $1M min dollar-volume gate, EMA(12/26) + ATR(14).

### alpha_options — Defined-risk vertical spreads
- **Weight:** 4%.
- **What it does:** Reads strong directional signals from alpha/gamma and
  emits vertical spreads on SPY (bull call / bear put).
- **Signal family:** options / directional.
- **Venue:** IBKR (options BAG combos).
- **Active regimes:** trending_bull, trending_bear, volatile.
- **Current status:** armed; dependent on alpha/gamma directional signals.
- **Technical:** defined risk = width × 100 × contracts; position size =
  floor(equity × max_risk_pct / max_loss); max 4 open spreads; uses cached
  options chain from `runtime/options_chains_cache.json`.

### alpha_intraday — Delta high-convexity day-trading brain
- **Weight:** 3%.
- **What it does:** Event-driven asymmetric payoff engine on 1-minute
  bars with daily fallback. Three entry triggers (any one sufficient):
  volatility explosion (ATR ≥ 2× baseline), momentum surge, mean-reversion
  snap (equities only).
- **Signal family:** momentum / volatility.
- **Venue:** IBKR (equity + futures + crypto).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** armed; 10-minute per-symbol cooldown.
- **Technical:** 1.5% stop, 4.5% target (3:1 asymmetric), max 30 bars
  hold, `high_convexity=True` tag on every signal.

### alpha_crypto (weight key: "crypto") — Crypto signal engine
- **Weight:** 4%.
- **What it does:** Regime-gated crypto signals on BTC/ETH/SOL with
  liquidity gating, anti-chase, deterministic exits. No network calls
  in signal path — consumes bar data only.
- **Signal family:** trend / momentum.
- **Venue:** Kraken (spot).
- **Active regimes:** trending_bull, trending_bear, volatile, unknown.
- **Current status:** firing on the live Kraken price (fixed 2026-04-15).
- **Technical:** trend + vol regime gate, dollar-volume liquidity gate,
  time stop + ATR trail + trend break + vol spike exits. Audit metadata
  on every signal (`blocked_by`, `reasons`, `exit_reason`).

### omega — Wealth-safe hedge sleeve
- **Weight:** 5%.
- **What it does:** Insurance, not profit. Buys inverse equity ETFs
  (SH, PSQ) when multiple danger sensors agree — portfolio drawdown,
  SPY/QQQ ATR spike, VIX level/spike, macro regime flag.
- **Signal family:** macro / hedge.
- **Venue:** IBKR (equity inverse ETF).
- **Active regimes:** volatile, unknown (stays dormant in trending regimes).
- **Current status:** armed; VIX 18.92 — below activation.
- **Technical:** 2+ sensors must agree to hedge; unwinds automatically when
  sensors clear; strict hedge caps and cooldowns prevent over-hedging.

### omega_vol — VIX-linked volatility alpha
- **Weight:** 5%.
- **What it does:** Directional vol bets via SVXY (short vol, contango
  harvest) and UVXY (long vol, spike capture). Uses a 5-state VIX regime
  classifier.
- **Signal family:** volatility.
- **Venue:** IBKR (equity volatility ETPs).
- **Active regimes:** volatile.
- **Current status:** active in current volatile regime; 3 samples in
  health tracker.
- **Health score:** 0.20 (struggling — 0 wins on 3 trades, 50% slippage
  ratio). Flagged for investigation.
- **Technical:** VIX level + VIX 5-day ROC + VIX z-score(20) = 3-factor
  signal; UVXY capped short-duration (10-bar time stop) due to structural
  decay; SVXY capped at 3% of equity.

### omega_macro — Macro regime futures
- **Weight:** 3%.
- **What it does:** Trades ZN, ZB, M6E based on a 4-state macro regime
  classifier (RISK_OFF / RISK_ON / STAGFLATION / NEUTRAL).
- **Signal family:** macro.
- **Venue:** IBKR (rates + FX futures).
- **Active regimes:** trending_bull, trending_bear, ranging, volatile,
  unknown (universally eligible — macro runs in every non-adverse regime).
- **Current status:** armed; waits for regime agreement with VIX signal.
- **Technical:** ATR-based sizing via `alpha_futures._compute_contract_size`,
  1.0% risk budget; NEUTRAL regime = flat (no signals).

### omega_momentum_options — Intraday single-leg options momentum
- **Weight:** 3%.
- **What it does:** Buys calls on bullish momentum, puts on bearish
  momentum, on SPY/QQQ/AAPL/NVDA/MSFT. Hard time exit by 3:45 PM ET, 50%
  profit target, 25% stop loss.
- **Signal family:** options / momentum.
- **Venue:** IBKR (single-leg options).
- **Active regimes:** trending_bull, trending_bear, volatile.
- **Current status:** armed within market hours (9:45 AM – 3:30 PM ET);
  15-min per-symbol cooldown; 3 max concurrent.
- **Technical:** two required triggers — (1) price momentum: 0.3% 5-bar
  move + EMA slope + volume confirmation; (2) IV timing: VIX regime
  filter (skip if VIX > 40); uses `options_chains_cache.json` with
  Black-Scholes synthetic fallback.

---

## 4. BETA — INSTITUTIONAL WEALTH BUILDER

### Concept

Beta is the **anchor of the business model**. While the trading engines
chase daily edge, Beta piggybacks on the world's smartest money — the
long-only large-cap positions that multiple top institutions have
independently chosen to hold.

Why this works:

- 13F filings are quarterly, so the data is slow and stable — no intraday
  noise, no event risk.
- "Most held" reveals genuine cross-institutional conviction, not the
  idiosyncratic bets of any single fund.
- The holding horizon is weeks-to-quarters, which means churn is
  structurally low and transaction costs are negligible versus returns.
- The basket rebalances with the institutions themselves, so Beta drifts
  gently with the smart-money consensus.

Beta is therefore **the anchor**: consistent, low-touch, reliably
compounding — the return stream that lets the trading engines take on
higher-variance strategies without destabilizing the business.

### Data Pipeline

- **Source:** SEC EDGAR 13F filings via `chad/analytics/institutional_consensus.py`.
- **Updater:** `scripts/update_institutional_consensus.py`, scheduled weekly.
- **Output:** `runtime/institutional_consensus.json` (schema
  `institutional_consensus.v1`), last updated 2026-04-23T13:24Z.
- **Aggregation:** by CUSIP (normalizes issuer-name variance across
  filings), with symbol resolution via a name-to-ticker map.
- **Conviction score:** `0.5 × (fund_count / N) + 0.5 × (share_of_total_value)`.

### Tracked Funds (7)

- `appaloosa` — Appaloosa LP
- `berkshire_hathaway` — Berkshire Hathaway
- `blackrock` — BlackRock
- `bridgewater` — Bridgewater Associates
- `citadel` — Citadel Advisors
- `pershing_square` — Pershing Square Capital
- `vanguard` — Vanguard Group

### Current Consensus — Top 10 by Conviction (2026-04-23)

| Rank | Symbol | Issuer | Fund count | Conviction |
|---|---|---|---|---|
| 1 | AAPL | Apple Inc | 6 | 0.4030 |
| 2 | AMZN | Amazon Com Inc | 6 | 0.3887 |
| 3 | GOOGL | Alphabet Inc | 7 | 0.3856 |
| 4 | META | Meta Platforms Inc | 5 | 0.3219 |
| 5 | V | Visa Inc | 5 | 0.3164 |
| 6 | MA | Mastercard Inc | 5 | 0.3157 |
| 7 | UNH | UnitedHealth Group Inc | 5 | 0.3156 |
| 8 | KO | Coca Cola Co | 5 | 0.3155 |
| 9 | SPY | SPDR S&P 500 ETF | 5 | 0.3143 |
| 10 | NVDA | Nvidia Corp | 4 | 0.2776 |

### Profit Routing → Beta Funding

The **30%** slice of every profitable close is earmarked for Beta under
the 50/30/20 rule. Until live capital flips on, the routing is advisory:
decisions are recorded in `runtime/profit_routing.json` so that when Beta
starts deploying it knows exactly how much earmarked capital it has.

**Current accumulated (as of 2026-04-23T11:13Z, paper):**
- `trading_capital`: $25.00
- `beta_allocation`: $15.00
- `amplifier_allocation`: $10.00
- Source decisions: 1 (realized_pnl=$50 from `alpha_test`).

The ledger is the accounting truth; actual cash movement only begins when
CHAD is promoted to live.

---

## 5. RISK ARCHITECTURE

Every signal must survive five pre-OMS gates, three sizing layers, a
regime filter, a STOP bus, a six-mode circuit breaker, and the SCR
sizing factor before it becomes an order. Each layer exists because a
past failure taught us it had to.

### Routing Gates (5 gates, applied in order)

Module: `chad/execution/routing_gates.py:run_all_gates`.

| # | Gate | What it checks | Block reason | Config |
|---|---|---|---|---|
| 1 | A4 `data_freshness` | Source bar age ≤ `max_bar_age_seconds` | `bar_stale` | 300s intraday / 172800s daily |
| 2 | E2 `stale_intent` | `utcnow − created_at ≤ ttl_seconds` | `intent_expired` | default TTL 300s |
| 3 | E5 `too_late_to_chase` | `|current − intent_price| / intent_price ≤ tolerance` | `price_moved` | 0.5% default; 60s degraded-mode TTL |
| 4 | R7 `net_ev` | `expected_pnl − commission − spread ≥ min_edge` | `net_ev_below_min_edge` | `min_edge=0` (inactive when `expected_pnl=0`) |
| 5 | S5 `event_risk` | Inside an event-calendar window | `event_risk_reject` (high urgency) / `reduce_50pct` (normal) | `config/event_calendar.json` |

Each gate is backward compatible: missing fields degrade to pass with a
flagged reason rather than crash the pipeline.

### Sizing Pipeline

EQUITY / ETF intents pass through three composite sizing layers in order
(`chad/execution/execution_pipeline.py:_apply_sizing_layer`):

```
base_size → R3 vol_adjusted → R5 composite_cap → R6 correlation_monitor → S5 event_gate → OMS
```

**R3 vol_adjusted_sizer** (`chad/risk/vol_adjusted_sizer.py`):
- `multiplier = clamp(target_daily_vol / realized_vol, 0.1, 2.0)`.
- `target_daily_vol = 0.015` (calibrated 2026-04-22 from 1.86% median
  realized vol in the universe).
- 20-day lookback, close-to-close log-returns.

**R5 composite_cap** (`chad/risk/composite_size_cap.py`):
```
final = min(
    vol_adjusted,
    max_per_symbol,                       # 300 shares
    remaining_sector_exposure,            # $5,000
    adv × max_adv_pct,                    # 0.5% of ADV
    equity × max_position_pct / ref_px,   # 5% of equity
)
```

**R6 correlation_monitor** (`chad/risk/correlation_monitor.py`):
- If `mean(|pairwise r|)` of the combined open+new book > 0.65,
  multiply by `threshold / avg_corr`, floored at 0.1.
- 20-day lookback, bar data only.

Futures and forex bypass this layer (specialised upstream sizers).
Crypto rides the Kraken builder path with only S3 confidence attenuation
plus a minimum-volume gate.

### Regime System

**Classifier inputs** (from `runtime/market_metrics.json`):

| Input | Current value (2026-04-23T19:55Z) |
|---|---|
| `realized_vol_percentile` | 0.7754 |
| `adx` (proxy) | 35.88 |
| `trend_slope` | 0.00481 |
| `market_breadth` | 0.4667 |
| `symbols_covered` | 30 |

**Regime labels** (`chad/analytics/regime_classifier.py`):

- `trending_bull` — ADX > 25 AND positive trend_slope AND breadth > 0.
- `trending_bear` — ADX > 25 AND negative trend_slope AND breadth < 0.
- `ranging` — ADX ≤ 25 AND vol_pct ≤ 0.75.
- `volatile` — vol_pct > 0.75 (dominates other signals).
- `unknown` — missing inputs or no match.

**Current regime:** `volatile`, confidence 0.5508, previous `volatile`
(stable). Active strategies under volatile:

```
alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_options,
beta, beta_trend, delta, gamma, gamma_futures,
omega, omega_vol, omega_momentum_options, omega_macro
```

**Activation matrix** (`config/regime_activation_matrix.json`) — declarative
`{regime → [allowed strategies]}` table; unknown regimes fall through to
the `unknown` bucket; `adverse` silences everything.

**Regime-transition reduction** (`chad/risk/regime_reduction.py`):
- any → `adverse`: close 50% of open quantity per symbol.
- any → `volatile`: close 30% of open quantity per symbol.
- any → `unknown`: log warning only; no auto reduction.

### STOP Bus

**Module:** `chad/risk/stop_bus_state.py` + `chad/risk/stop_bus_triggers.py`.
**State:** `runtime/stop_bus.json` (gitignored; currently `active=false`).

Four triggers (any one active halts the cycle before order submission):

1. `daily_loss_breach` — realized PnL below configured daily floor.
2. `reject_rate_spike` — order rejects dominate recent attempts.
3. `data_staleness` — Session-2 A4 gate is rejecting most intents
   (data feed almost certainly down).
4. `broker_latency_spike` — broker submission latency above threshold.

**Clear command:** `python3 scripts/clear_stop_bus.py`.

### Circuit Breaker — Profit Lock

**Module:** `chad/risk/profit_lock.py`. Six modes, equity-driven, written
to `runtime/profit_lock_state.json`. The allocator reads `sizing_factor`
from this file.

| Mode | Trigger (of equity) | Sizing factor |
|---|---|---|
| NORMAL | default | 1.00 |
| WARN | profit ≥ 1.5% | 1.00 (flagged) |
| LOCK1 | profit ≥ 3.0% | 0.50 |
| LOCK2 | profit ≥ 5.0% | 0.25 |
| LOCK3 | profit ≥ 8.0% | 0.10 |
| HARD_STOP | profit ≥ 10.0% | 0.00 |

**Current state (2026-04-23T19:55Z):**
- Mode: `NORMAL`; sizing_factor = 1.0; stop_new_entries = false.
- Account equity: $998,109.89.
- Daily realized PnL: −$303.52 (−0.03% of equity).
- Daily loss limit: 3.0% / $29,943.30 — not hit.

### SCR — Self-Calibrating Risk

**State:** `runtime/scr_state.json`. **Config:** `runtime/scr_config.json`.
The SCR is CHAD's autonomous confidence meter: until it has enough trade
evidence, it sizes conservatively; once metrics confirm it is working,
it sizes up.

**Current (2026-04-23T19:53Z):**
- State: `WARMUP`.
- Sizing factor: 0.10.
- Effective trades: 59 (excluded_untrusted = 4,850 of 5,000 total).
- Win rate: 52.54%.
- Sharpe-like: +0.727.
- Max drawdown: −$930.21.
- Total (effective) PnL: −$6,022.39.
- `paper_only: true` — not eligible for live capital yet.

**Thresholds (from `scr_config.json`):**
- `warmup_min_trades`: 100 effective trades to exit WARMUP.
- CAUTIOUS entry: `win_rate ≥ 0.35` AND `sharpe_like ≥ 0.10` AND
  `max_drawdown ≥ −$15,000`.
- CONFIDENT entry: `win_rate ≥ 0.50` AND `sharpe_like ≥ 0.50` AND
  `max_drawdown ≥ −$10,000`.

**Progression:**

| State | Sizing factor | What changes |
|---|---|---|
| WARMUP (current) | 0.10 | Trades fire at 10% of planned size. No live allowed. |
| CAUTIOUS | 0.25 | Sizes up 2.5×. Still paper-only. Live promotion requires explicit operator GO. |
| CONFIDENT | 1.00 | Full size. Eligible for live consideration if other checks pass. |
| PAUSED | 0.00 | Hard stop (manual or rule-driven). |

**ETA at current paper cadence:** CAUTIOUS in 2–4 weeks.

---

## 6. EXECUTION ARCHITECTURE

Post-Phase-8, CHAD has an explicit seam between **what to trade** (EMS)
and **how orders behave** (OMS).

### OMS / EMS Separation (A1)

**EMS — Execution Management System.** Owns signal-to-order transformation:
instrument resolution, venue selection, sizing (R3/R5/R6), order-type
selection (E4), intent construction. Entry points live in
`chad/execution/ems.py`.

- `EMSInterface` — Protocol with `build_order_request(intent)` and
  `select_venue(intent)`.
- `IbkrEMS`, `KrakenEMS` — venue-specific implementations.

**OMS — Order Management System.** Owns order lifecycle: idempotent
submission, state tracking, fill recording, cancellation. Entry points
live in `chad/execution/oms.py`.

- `OMSInterface` — Protocol with `submit(OrderRequest) → OrderResult`,
  `cancel`, `get_status`.
- `IbkrOMS` — wraps `IbkrAdapter.submit_strategy_trade_intents`.
- `KrakenOMS` — wraps `KrakenExecutor`.
- `SimulatedOMS` — fills at `limit ± slippage_bps` into a
  `SimulatedFillLedger`. Used by backtest + A3 unified interface.
- `NullOMS` — no-op, used for dry-runs.

**Preserved status vocabulary** (pinned by regression guard):
`submitted`, `dry_run`, `what-if`, `duplicate_blocked`, `error`, `unknown`.

### Intent Schema (11 fields, E1)

Module: `chad/execution/intent_schema.py`. Every field is optional on
both IBKR and Kraken dataclasses so legacy callers keep working.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `confidence` | float [0,1] | 0.5 | S3 confidence; S1 voting; E4 passive/aggressive. |
| `entry_reason` | str | `""` | F1 scorecard; audit readability. |
| `regime_state` | str | `"unknown"` | S4 adaptive thresholds; G3 regime reduction. |
| `expected_pnl` | float | 0.0 | R7 net-EV gate. |
| `created_at` | str ISO8601 UTC | `utc_now_iso()` | E2 stale expiry; E5 degraded mode. |
| `ttl_seconds` | int | 300 | E2 stale expiry. |
| `expected_price` | float | 0.0 | E3 slippage (fill vs expected). |
| `signal_strength` | float | 0.0 | S3 input, sizing attenuation. |
| `signal_family` | str | `"unknown"` | S1 vote aggregation. |
| `order_urgency` | str | `"normal"` | E4 order-type; S5 reduction vs reject. |
| `bar_timestamp` | str | `""` | A4 data-freshness validation. |

Helpers: `utc_now_iso()`, `validate_intent(intent)`,
`intent_is_fresh(intent)`, `INTENT_SCHEMA`.

### Execution Pipeline

```
raw signals
   ↓
EMS.build_execution_plan
   ↓
EMS.build_intents_from_plan  (routed_signals → StrategyTradeIntent[])
   ↓
routing_gates.run_all_gates  (A4 / E2 / E5 / R7 / S5)
   ↓
vote_collector.submit_intent  (S1 stacking, min_votes=1 currently)
   ↓
sizing pipeline  (R3 → R5 → R6 for equity/ETF)
   ↓
OMS.submit(OrderRequest)
   ↓
fill → slippage_tracker → position_guard → paper_exec_evidence_writer
```

Thin orchestrators:
- `execute_ibkr_cycle(ExecutionPlan)` → `list[OrderResult]`.
- `execute_kraken_cycle(routed_signals)` → `list[OrderResult]`.

### Venues

- **IBKR paper (account DUK902770)**: equities, ETFs, futures, options,
  forex. Gateway on port 4002; adapter at `chad/execution/ibkr_adapter.py`.
  Bars polled every 30s via `reqHistoricalData` (no real-time subscription
  needed since commit `51c190e`, 2026-04-23).
- **Kraken (paper connected)**: crypto spot (BTC, ETH, SOL — pairs via
  Kraken REST altnames). 24/7 market; websocket for prices/balances.

---

## 7. SIGNAL QUALITY LAYER

Five modules sit between raw strategy output and the routing gates and
make every intent carry quality metadata.

### S1 — Vote Collector

Module: `chad/analytics/vote_collector.py`. Accumulates intents keyed by
`(symbol, side)`. An intent is released when ≥ `min_votes` DISTINCT
`signal_family` values have submitted that pair within `window_seconds`.

**Config** (`config/signal_stacking_config.json`):
- `min_votes`: **1** (current — lowered from 2 on 2026-04-22 because the
  active strategy mix only produces 1–2 distinct families per cycle).
- `window_seconds`: 300.

Raise `min_votes` to 2 once intraday + crypto + reversion are consistently
firing. Infrastructure is ready; the tighter threshold just currently
starves the router.

### S2 — Timeframe Confirmation

Module: `chad/analytics/timeframe_confirmation.py`. Compares the intent
side against the higher-timeframe bias (20-bar SMA on 1d bars) and
attenuates confidence:

| Intent side vs higher-TF bias | Multiplier |
|---|---|
| Agrees | 1.0 |
| Disagrees | 0.6 |
| Neutral / missing data | 0.85 |

Minimum 0.1 — S2 never zeroes confidence.

### S3 — Confidence Scoring

Module: `chad/analytics/signal_confidence.py`. Canonical formula:

```
confidence = signal_strength × regime_quality × liquidity_quality × tf_multiplier
final_size = base_size × max(SIZING_FLOOR, confidence)    # SIZING_FLOOR = 0.1
```

Low confidence attenuates size but never eliminates the trade; hard
rejects belong upstream at signal generation.

### S4 — Threshold Adapter

Module: `chad/analytics/threshold_adapter.py`.
Config: `config/threshold_adapter_config.json`. Regime multipliers
applied to entry thresholds (z-scores, RSI bands, etc.):

| Regime | Multiplier |
|---|---|
| trending_bull / trending_bear / trending / risk_on / risk_off | 1.0 |
| ranging | 0.9 |
| volatile / choppy | 1.3 |
| unknown | 1.1 |
| adverse | 2.0 |
| (fallback) default | 1.1 |

Strategies opt in by calling `adjust(base, regime)` or
`adjust_rsi(oversold, overbought, regime)` — no hardcoded cutoffs.

### S5 — Event Calendar

Module: `chad/analytics/event_calendar.py`.
Config: `config/event_calendar.json` (operator-maintained).

**Upcoming events (2026):**

| Date (UTC) | Time | Event | Suppress (hrs) |
|---|---|---|---|
| 2026-05-07 | 18:00 | FOMC Rate Decision | 24 |
| 2026-05-13 | 12:30 | CPI Release | 12 |
| 2026-05-15 | 12:30 | Retail Sales | 6 |
| 2026-05-30 | 12:30 | Non-Farm Payrolls | 12 |
| 2026-06-18 | 18:00 | FOMC Rate Decision | 24 |

**Behavior inside a window:** `urgency="high"` → reject; `urgency="normal"`
→ reduce quantity/volume/notional 50% in place and pass.

---

## 8. FEEDBACK & LEARNING LAYER

Four measurement modules close the loop between fills and allocation.

### F3 — Strategy Health Score

Module: `chad/analytics/strategy_health.py`.
Output: `runtime/strategy_health.json`.

Composite in [0, 1]:

```
health = 0.40 × normalized_sharpe
       + 0.30 × win_rate
       + 0.20 × (1 − slippage_ratio)
       + 0.10 × regime_alignment
```

**Current scores (2026-04-23T19:52Z):**

| Strategy | Score | Samples | Notes |
|---|---|---|---|
| gamma_futures | 0.900 | 1 | Too-small sample; ignore signal. |
| delta | 0.825 | 7 | Strong early read (win rate 0.75). |
| alpha | 0.717 | 32 | Meaningful sample. |
| alpha_futures | 0.699 | 44 | Largest sample; win rate 0.333. |
| omega_vol | 0.200 | 3 | **Flagged** — 0 wins, 50% slippage ratio. |

### F4 — Edge Decay Monitor

Module: `chad/risk/edge_decay_monitor.py`.
Config: `config/edge_decay_config.json`.

Halts a strategy (allocation → 0 in `runtime/strategy_allocations.json`)
when:

- `consecutive_threshold` = **5** losses in a row (calibrated 2026-04-22;
  current max observed streak is 3 on alpha), AND
- `min_trades` = **20** — strategies below this still in warm-up.

**Recovery:** `python3 scripts/clear_edge_decay.py --strategy <name>` or
direct edit of `runtime/strategy_allocations.json`.

### F2 — Signal Decay Measurement

Module: `chad/analytics/signal_decay.py`.
Ledger: `data/signal_decay/<strategy>_decay.ndjson`.

At fill time records a pending entry (strategy, symbol, side, entry price,
entry ts). Later cycles compute side-adjusted unrealized alpha at T+1,
T+5, T+15, T+30 **trading days** from 1d bars. Retrospective by design —
no live polling.

### F1 — Expectancy Tracker

Module: `chad/analytics/expectancy_tracker.py`.
Output: `runtime/expectancy_state.json`.

Reads `data/trades/trade_history_*.ndjson`, skips untrusted/pre-rebuild
records, writes per-strategy rolling win rate, avg win, avg loss,
expectancy, and a plain-English status label (`new`, `warming_up`,
`healthy`, `concerning`, etc.).

### E3 — Slippage Tracker

Module: `chad/analytics/slippage_tracker.py`.
Ledger: `data/slippage/SLIPPAGE_YYYYMMDD.ndjson`.

Records per-fill, side-adjusted slippage so positive always means
adverse (`BUY: fill − expected; SELL: expected − fill`). Skips records
where `expected_price=0` (intent-schema default) so the default never
skews rolling stats.

---

## 9. MARKET DATA

### Bar Provider

- **Service:** `chad-ibkr-bar-provider.service`.
- **1m bars:** polling IBKR `reqHistoricalData` every 30 seconds — no
  real-time subscription required (switched from `reqRealTimeBars` in
  commit `51c190e`).
- **1d bars:** nightly refresh.

### Equity / ETF / Futures Universe (from `data/bars/1d/`)

30 symbols tracked:

```
AAPL, BAC, GLD, GOOGL, IEMG, IWM, MSFT, NVDA, PSQ, QQQ,
SH, SPY, SVXY, TLT, UVXY, VIXY, VWO, VXX,
M6E, MCL, MES, MGC, MNQ, SIL, ZB, ZN,
BTC-USD, ETH-USD, SOL-USD, VIX
```

- **Configured tradeable equity universe** (`config/universe.json`):
  AAPL, SPY, MSFT, GOOGL, BAC, IEMG, QQQ, VWO, NVDA, GLD, SH, PSQ,
  SVXY, UVXY, VIXY.
- **Configured futures:** MES, MNQ, MCL, MGC, ZB, ZN, M6E, SIL (via
  CME / CBOT / ECBOT contracts).

### Crypto (Kraken)

- 1d bars only: **BTC-USD, ETH-USD, SOL-USD**. No 1m bar feed from Kraken
  today.
- Altname-resolved via Kraken REST (fix shipped in commit `e8f3c06`).

### VIX

- **Source:** CBOE nightly, stored at `data/bars/1d/VIX.json`.
- **Current close:** **18.92** (2026-04-22).
- **Consumers:** `omega`, `omega_vol`, `omega_macro`,
  `omega_momentum_options`, `alpha_options`.
- **Wiring:** live ctx injection fixed in commit `6ea83c6` — strategies
  that read `ctx.vix` now see the latest close on every cycle.

---

## 10. CONFIGURATION REFERENCE

### Key Config Files

| File | Purpose | Key values (current) |
|---|---|---|
| `config/strategy_weights.json` | Per-strategy capital weights | 16 keys summing to 1.00 (see §3). |
| `config/regime_activation_matrix.json` | `{regime → [allowed strategies]}` gating | 6 regimes, `adverse=[]` silences all. |
| `config/sizing_config.json` | R3/R5/R6 parameters | `target_daily_vol=0.015`, `correlation_threshold=0.65`. |
| `config/signal_stacking_config.json` | S1 vote collector | `min_votes=1`, `window_seconds=300`. |
| `config/event_calendar.json` | S5 suppression windows | 5 macro events May–June 2026. |
| `config/edge_decay_config.json` | F4 halt thresholds | `consecutive_threshold=5`, `min_trades=20`. |
| `config/threshold_adapter_config.json` | S4 regime multipliers | see §7. |
| `config/simulated_oms_config.json` | Backtest slippage bps by asset class | equity 3.0, futures 1.5, crypto 8.0. |
| `config/universe.json` | IBKR-tradeable symbol list | 15 equities + 8 futures. |
| `config/risk.json` | Risk policy overrides | — |
| `config/portfolio_profiles.json` | Portfolio snapshot shapes | — |

### Key Runtime Files

| File | Purpose |
|---|---|
| `runtime/scr_state.json` | SCR state (WARMUP/CAUTIOUS/CONFIDENT/PAUSED) + sizing_factor. |
| `runtime/scr_config.json` | SCR thresholds by state. |
| `runtime/regime_state.json` | Current regime label + confidence. |
| `runtime/strategy_health.json` | F3 composite health per strategy. |
| `runtime/institutional_consensus.json` | Beta's 13F basket (weekly). |
| `runtime/profit_routing.json` | 50/30/20 accounting ledger. |
| `runtime/profit_lock_state.json` | Circuit breaker mode + sizing factor. |
| `runtime/stop_bus.json` | Halt flag + trigger reasons. |
| `runtime/position_guard.json` | Per-symbol open-position state. |
| `runtime/market_metrics.json` | G1 inputs (vol_pct, ADX, slope, breadth). |
| `runtime/dynamic_caps.json` | Orchestrator-published per-strategy dollar caps. |
| `runtime/live_readiness.json` | Live-promotion checklist state (`ready_for_live=false`). |
| `runtime/expectancy_state.json` | F1 per-strategy rolling expectancy. |
| `runtime/options_chains_cache.json` | Cached options chains for alpha_options + omega_momentum_options. |
| `runtime/kraken_balances.json` | Kraken paper balances (BTC/CAD). |
| `runtime/last_route_decision.json` | Bridge file for DecisionTrace pickup. |

### Key Environment Variables

| Variable | Description | Current value |
|---|---|---|
| `CHAD_EXECUTION_MODE` | Execution posture | `paper` |
| `CHAD_ALLOC_V3_KELLY_MAX` | Max Kelly fraction cap | 0.25 (default) |
| `CHAD_ALWAYS_ACTIVE_ROUTING` | Enable all-strategy routing (vs single-winner) | OFF by default |
| `LOOP_INTERVAL_SECONDS` | `chad-live-loop` cycle cadence | 60 |
| `CHAD_PROFIT_LOCK_*` | Profit-lock thresholds / factors | defaults (see §5) |
| `CHAD_ORCH_RUN_FOREVER` | Run orchestrator as a service | set in systemd unit |

---

## 11. OPERATOR GUIDE

### Daily Checks (≈ 2 minutes)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Core health
cat runtime/scr_state.json | python3 -m json.tool | head -15
cat runtime/profit_lock_state.json | python3 -m json.tool | head -15
cat runtime/stop_bus.json | python3 -m json.tool

# 2. Regime + activation
cat runtime/regime_state.json
cat runtime/market_metrics.json

# 3. Strategy health
python3 -c "import json; [print(f'{k}: {v[\"health_score\"]:.3f} ({v[\"sample_count\"]})') for k,v in json.load(open('runtime/strategy_health.json'))['strategies'].items()]"

# 4. Dashboard
curl -s https://chadtrades.com/health
```

**What to look for:**
- SCR `state` still `WARMUP` (expected until ~100 effective trades).
- Profit lock `mode=NORMAL`, `stop_new_entries=false`.
- Stop bus `active=false`.
- No strategy with `health_score < 0.3` AND `sample_count ≥ 20` (that's
  a genuine edge-decay candidate).
- Dashboard returns HTTP 200.

### Weekly Tasks

- **Sunday — institutional consensus:** `scripts/update_institutional_consensus.py`
  runs automatically (via cron). Verify
  `runtime/institutional_consensus.json:updated_ts_utc` < 7 days old.
- **Strategy health:** review `runtime/strategy_health.json`; any
  strategy below 0.3 on a ≥ 20-sample base is an action item.
- **Profit routing:** check `runtime/profit_routing.json:totals` —
  beta_allocation should be growing at 30% of realized net profit.

### Monthly Tasks

- **Salary eligibility:** verify all four salary conditions (§1); if all
  hold and SCR = CONFIDENT, operator may withdraw 20–40% of net monthly
  profit from the amplifier bucket.
- **SCR progression review:** compare SCR stats to the CAUTIOUS /
  CONFIDENT thresholds in `runtime/scr_config.json`.
- **Position sizing:** if capital has grown materially, re-evaluate the
  composite caps in `config/sizing_config.json`.

### When SCR Advances to CAUTIOUS

- `sizing_factor` rises from 0.10 → 0.25 automatically — trades get 2.5×
  larger immediately.
- Still `paper_only=true`. No live capital.
- Re-verify health and profit-lock for the first 3 cycles at the new size.

### When SCR Advances to CONFIDENT

- `sizing_factor` rises to 1.00 — full planned size.
- The system becomes eligible for live consideration, but **live is
  still gated by explicit operator GO** (see §12).
- Before flipping live:
  1. All 82 paper tests pass.
  2. `full_cycle_preview.py --dry-run` clean.
  3. `live_readiness.json:ready_for_live=true`.
  4. Manual review of open paper positions.
  5. Operator issues the one-change-at-a-time config mutation (governance
     rule #3 in CLAUDE.md).

### Emergency Procedures

```bash
# Stop all trading (hot-path halt)
sudo systemctl stop chad-live-loop.service

# Soft halt (STOP bus — leaves services up but blocks new orders)
# (set via ops tooling — flag lives at runtime/stop_bus.json)

# Clear the STOP bus once you've addressed the cause
python3 scripts/clear_stop_bus.py

# Clear edge-decay halt for a specific strategy
python3 scripts/clear_edge_decay.py --strategy alpha

# Full revert to pre-overhaul state (nuclear option)
git checkout REVERT_PRE_OVERHAUL_20260419
# Then restore runtime from:
#   /home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz
# Full steps in /home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt
```

### Verification Sequence (after any code change — from CLAUDE.md)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

---

## 12. PHASE ROADMAP

### Phase 1 — Build & Validate (current)

**Status:** COMPLETE on build; ongoing on validate.

- P0/P1/P2 hardening complete.
- GAP-1 through GAP-25 complete.
- 2026-04-19/21 Overhaul complete (see commit `37989b7`).
- Phase-8 sprint (29/29 items across 10 sessions) complete — see
  `docs/ARCHITECTURE.md`.
- Beta (institutional compounder) shipped 2026-04-23.

**Watch:**
- SCR progression to CAUTIOUS.
- omega_vol health score (0.20 — flagged).
- Daily reconciliation stays GREEN.

### Phase 2 — Stabilize (target: 60–90 days)

**Entry criteria:**
- SCR ≥ CAUTIOUS.
- ≥ 30 paper trading days with zero LOCK2/3 breaches.
- At least 4 strategies have health_score ≥ 0.5 on ≥ 20 samples.

**What changes:**
- Raise `min_votes` from 1 to 2 once enough families are active (S1).
- Calibrate SimulatedOMS slippage bps from real slippage tracker data
  (see Phase-9 item 3).
- Re-evaluate SCR thresholds against actual win-rate distribution.

### Phase 3 — Salary

**Entry criteria:**
- All four salary conditions (§1) hold for 2+ consecutive months.
- SCR = CONFIDENT.
- Live capital switch has flipped (see Phase 10).

**Withdrawal formula:** 20–40% of net monthly profit from the amplifier
bucket only. Never touch core. Automate as a monthly cron once battle-
tested.

### Phase 9 — Pre-Live-Capital Checklist

Phase-8 finished the plumbing. Phase-9 is calibration:

1. **Regime classifier tuning** — the Session-6 ADX proxy is an
   ATR-scaled stand-in. Compare classifications against real market days
   and either tune the 25 threshold or swap in a Wilder ADX
   implementation.
2. **Kelly fraction tuning** — `CHAD_ALLOC_V3_KELLY_MAX=0.25` default;
   re-evaluate after one month of paper expectancy data.
3. **Slippage model calibration** — SimulatedOMS uses flat bps slippage;
   fit bps per asset class to the Session-3 slippage tracker ledger once
   ≥ 100 records per class exist. Use
   `compare_backtest_to_paper(bt_fills, paper_fills)` to validate.
4. **Live performance data** — accumulate ≥ 30 paper trading days before
   considering live.
5. **Net-EV gate opt-in** — strategies start populating `expected_pnl` so
   R7 moves from "inactive when 0.0" to actively filtering on EV.
6. **Live feature distribution monitoring** — drift alerts comparing
   paper-fit distributions to live context.
7. **Implementation shortfall decomposition** — modeled vs realized.
8. **Halt-on-reconciliation-mismatch** (instead of log-only RED).

### Phase 10 — Live Capital

**Entry criteria:**
- All Phase-9 items complete.
- Explicit operator GO via governance rule #3 (pending-action → apply).

**What changes:**
- `CHAD_EXECUTION_MODE` flips from `paper` to `live`.
- `LiveGate` accepts the posture change.
- First 3 execution cycles run with manual oversight.
- Broker truth reconciliation on first fill.
- Profit routing flips from advisory to actual capital movement between
  trading / beta / amplifier accounts.

---

## 13. COMMIT HISTORY (Overhaul + Phase-8 + Session 11)

Most recent first. All SHAs from `main`.

| SHA | Description |
|---|---|
| `51c190e` | Fix: switch bar provider from `reqRealTimeBars` to polling (no subscription required) |
| `c021ccd` | Fix: dedup `institutional_consensus` entries by resolved symbol |
| `3e9db6d` | Fix: update Citadel and Appaloosa CIKs in SEC 13F fetcher |
| `2c6a16b` | Build: Beta — institutional long-term compounder (original concept) |
| `20d6f59` | Rename: `chad/strategies/beta.py` → `beta_trend.py` (pure rename) |
| `6ea83c6` | Fix: wire VIX into live ctx + expand options chain refresh window |
| `e8f3c06` | Fix: unblock delta_pairs / alpha_intraday / omega_momentum_options + Kraken REST altnames |
| `93bcb5b` | Fix: regime matrix `volatile` list + futures bar ambiguity |
| `33d82eb` | Fix: wire regime matrix, unblock Kraken, fix vote starvation, signal_decay timezone, strategy_health stray keys |
| `982b779` | Calibration: apply Audit-O data-grounded config values |
| `2ce1e7b` | Phase-8 Session 10 (final): A1 thin orchestrator + A3 unified interface |
| `81b8ddf` | Phase-8 Session 9: A1 OMS/EMS separation |
| `0b9930f` | Phase-8 Session 8: E5/A4 full threading + E4 Kraken + ARCHITECTURE |
| `864b128` | Phase-8 Session 7: risk sizing layer (R3/R5/R6/S5) |
| `d281848` | Phase-8 Session 6: order logic (E4) + feedback loop (F2/F3) + market metrics publisher (G1 feed) |
| `1cf9f44` | Phase-8 Session 5: signal quality (S1/S2/S4/F4) |
| `d428ee3` | Phase-8 Session 4: R2 wiring + regime layer (G1/G2/G3) |
| `b02d06f` | Phase-8 Session 3: slippage tracking (E3) + confidence (S3) + STOP bus triggers (R2) |
| `575e11d` | Phase-8 Session 2: routing gates (E2/E5/R7/A4) |
| `18d5b4e` | Phase-8 Session 1: ARCHITECTURE.md + E1 intent schema extension |
| `37989b7` | Overhaul complete: post-overhaul state doc + CLAUDE.md update |
| `b672042` | FIX: ISSUE-29 — reconciler respects partial_attribution_residual |
| `0e3e007` | ISSUE-56 fix v2: reduce-not-close for partial broker_sync attribution |
| `e5db43a` | ISSUE-74 fix: synthesize fill_id for broker_sync lots |
| `709be69` | ISSUE-56 fix extension: cover rebuild path at `live_loop.py:284` |
| `1b850c6` | FIX: ISSUE-56 — broker_sync anchor yields to strategy ownership |
| `92b48df` | HARDEN: Dashboard TLS + fail-closed auth + retire 3 obsolete services |
| `8c82580` | FIX: Crypto signals — use live Kraken price for entry gates |
| `2dadcd7` | BUILD: Daily loss limit + per-symbol performance blocker |
| `0238292` | FIX: Alpha daily signal limit 3 per symbol per day |

---

## 14. KNOWN ISSUES & BACKLOG

### Active Issues (from post-overhaul)

| ID | Sev | Status | Summary |
|---|---|---|---|
| ISSUE-22 | P2 | OPEN | Legacy placeholder audit item; revisit during next Phase-8 follow-up. |
| ISSUE-29 | P1 | PARTIAL | `apply_close_intents` mutates guard before broker confirms. Reconciler-side fix in `b672042` prevents state corruption; root-cause mutate-after-confirm still untreated. Emits spurious close intents each cycle. |
| ISSUE-50 | P1 | OPEN | `chad-options-chain-refresh` hangs when IBKR ushmds farm is down. Timeout wrapper pending. |
| ISSUE-54 | P2 | OPEN | `runtime/pnl_state.json` still tracked in git; needs `git rm --cached` + gitignore. |
| ISSUE-56 | P0 | CLOSED | `broker_sync` anchor stayed open while strategy closed; fixed by v2 + reconciler skip (`1b850c6` → `709be69` → `0e3e007` + `b672042`). |
| ISSUE-58 | P2 | OPEN | `chad-trade-closer.timer` uses `OnBootSec=45 + OnUnitActiveSec=60`; needs `OnCalendar` or documented seed step. |
| ISSUE-74 | P0 | CLOSED | `broker_sync` lots lacked `fill_id` → KeyError in trade-closer; synthesized at lot creation (`e5db43a`). |
| ISSUE-75 | P1 | OPEN | Multiple call sites write `position_guard.json` directly; unify via a single setter. |
| ISSUE-78 | P2 | OPEN | Two code paths read `CHAD_EXECUTION_MODE`; can diverge under rapid transitions; unify into one accessor. |

### Phase-9 Items (pre-live-capital calibration)

- Regime classifier tuning (ADX proxy → Wilder ADX or calibrate threshold).
- Kelly fraction tuning (`CHAD_ALLOC_V3_KELLY_MAX`).
- Slippage model fit per asset class.
- Live feature distribution drift monitoring.
- Net-EV gate opt-in (populate `expected_pnl` at strategy level).
- Halt-on-reconciliation-mismatch.
- ISSUE-29 proper fix (mutate-after-confirm).
- ISSUE-50 options-chain timeout wrapper.
- ISSUE-75 unified guard writer.
- ISSUE-78 unified `CHAD_EXECUTION_MODE` accessor.
- ISSUE-54 `pnl_state.json` removal from git.
- ISSUE-58 trade-closer timer pattern.

### Phase-10 Items (live-capital feedback loop)

- Per-setup rolling expectancy scorecard.
- Signal decay measurement at T+1 / T+5 / T+15 / T+30 days.
- Strategy health composite (already exists — needs live-data recalibration).
- Edge-decay auto-allocation reduction (exists — needs live-data tuning).
- Implementation shortfall decomposition (modeled vs realized).
- Profit routing flips from advisory to actual inter-account capital movement.

---

## 15. REVERT & RECOVERY

**Primary revert point (pre-overhaul):**
- **Git tag:** `REVERT_PRE_OVERHAUL_20260419`
- **Commit:** `45f3728fa512c27d77453489c11d95ca0e075cb9`
- **Runtime tarball:** `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`
- **Tarball SHA-256:** `2be81cbae94fb86c266f1c519f208f058ac3a99b99032f2fec7a2eefbcc10a53`
- **Instructions:** `/home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt`

**Other active tags:**
- `STABILITY_FREEZE_20260307_GREEN` — original stable baseline.
- `PRE_HARDENING_20260402` — snapshot before P0 hardening began.
- `RATIFICATION_MASTER_20260402` — all hardening + GAP items complete.

**Rollback command (governance-approved):**

```bash
git checkout RATIFICATION_MASTER_20260402     # post-hardening stable
# or
git checkout REVERT_PRE_OVERHAUL_20260419     # pre-overhaul stable
```

Always restore `runtime/` from the matching tarball before restarting
services — code and runtime state must be consistent.

---

**End of CHAD Unified SSOT v8.1.**

This document is the truth. If the code disagrees, either the code has
drifted or this document needs a new revision — revise it before
relying on the disagreement.
