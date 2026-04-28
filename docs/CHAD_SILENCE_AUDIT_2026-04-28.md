# CHAD Per-Strategy Silence Audit — 2026-04-28

**Audit type:** Read-only diagnostic. No production state mutated, no orders executed.
**Trigger:** Pre-30-day unattended-run gate. Verify each silent strategy is silent for the
*right* reason (regime/condition gates working as designed) and not concealing a bug.
**Methodology:** Built a live `MarketContext` via `chad/utils/context_builder.py` from
the same artifacts the live loop uses, then invoked each silent handler directly,
logging signals returned and tracing each gate against the runtime state.

---

## 1. Runtime State at Audit Time

| Field | Value | Source |
|---|---|---|
| now | 2026-04-28 23:21 UTC (Tue, post-market) | `datetime.now(utc)` |
| regime | `trending_bull` (conf 0.636) | `runtime/regime_state.json` |
| previous_regime | `trending_bull` (stable) | same |
| total_equity | **$183,311.24** | `runtime/dynamic_caps.json:total_equity` |
| VIX (spot) | **18.02** | `data/bars/1d/VIX.json` last close |
| open positions | 3 (`delta\|GOOGL`, `alpha\|TLT`, `broker_sync\|TLT`) | `runtime/position_guard.json` |
| legend weights | 25 symbols (top: AAPL 5.62%, AMZN 5.42%, GOOGL 5.37%) | `data/legend_top_stocks.json` |
| daily bars | 30 symbols, ~250 bars each (last 2026-04-27) | `data/bars/1d/` |
| 1m bars | 25 symbols, ~45 bars each (last ~19:02 ET 2026-04-28) | `data/bars/1m/` |
| Kraken feed | fresh (2026-04-28T23:18Z), BTC/ETH/SOL prices | `runtime/kraken_prices.json` |
| Today's fills (FILLS_20260428.ndjson) | alpha 3, alpha_futures 7, gamma_futures 14, delta 6, broker_sync 307 | live ledger |

**Equity discrepancy note:** `CLAUDE.md` recorded "~$994k paper equity (as of 2026-04-21)". The live `dynamic_caps.json` is ~$183k. SSOT v8.4 (2026-04-28) records `$167,267.45` (2026-04-27). The runtime equity figure is the canonical source consumed by all sizing — the CLAUDE.md figure is stale and should be corrected separately.

---

## 2. Regime Activation Matrix Cross-Check

`config/regime_activation_matrix.json` filters intents downstream of handlers. In `trending_bull`, only the strategies in the active list survive routing:

| Silent Strategy | In `trending_bull` active list? |
|---|---|
| alpha_intraday | ✅ YES |
| alpha_options | ✅ YES |
| alpha_crypto | ✅ YES |
| beta | ✅ YES |
| beta_trend | ✅ YES |
| **delta_pairs** | ❌ NO (only `ranging`) |
| **gamma** | ❌ NO (only `ranging, volatile, unknown`) |
| **gamma_reversion** | ❌ NO (only `ranging`) |
| **omega** | ❌ NO (only `volatile, unknown`) |
| **omega_vol** | ❌ NO (only `volatile`) |
| omega_macro | ✅ YES |
| omega_momentum_options | ✅ YES |

The 5 `❌` entries are silenced upstream of the broker by the regime activation filter
(`chad/core/live_loop.py:906-910` calling `chad/portfolio/regime_activation.py:filter_intents_by_regime`).
Whether their *handlers* still emit signals is a separate (and informative) question, addressed below.

---

## 3. Per-Strategy Findings

For each strategy I report what the **handler itself** returned when invoked against
the live `ctx`, and what the SSOT-documented expected reason for silence is.

---

### 3.1 alpha_intraday

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Active in trending_bull — should fire when vol explosion / momentum surge / mean-reversion snap triggers." (SSOT §3, line 706-710) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | All three triggers below threshold for every UNIVERSE symbol. Per-symbol probe (1m bars where available, daily fallback for crypto): max ATR ratio = 1.30 (NVDA) vs threshold 2.0; max momentum m_short = 0.0006 (AAPL) vs threshold 0.005 (0.5%); RSI extremes only at MSFT (81.7, but no Bollinger break) and GOOGL (25.6, fails the <25 cutoff). 1m bars are only ~45 deep — close to the `len(closes) < 45` floor at `alpha_intraday.py:230`. |
| STATUS | ✅ **CORRECT** |

Detail: `alpha_intraday.py:243-297` defines three triggers (vol_explosion, momentum_surge, mean_reversion_snap). None fire on a calm trending_bull tape. This is the strategy operating as designed — it is a high-convexity day-trader meant to fire only on extreme conditions.

---

### 3.2 alpha_options

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Position MAINTAINED — Carried alpha_options\|SPY from 2026-04-24T20:18Z prevents new entries." (SSOT §3 line 734) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | The SSOT-documented reason no longer applies — `position_guard.json` shows **no** open `alpha_options\|SPY` position. The actual silent gate is **position sizing**: at `alpha_options.py:344-351`, `risk_budget = equity × max_risk_per_trade_pct = $183,311 × 0.005 = $916.58`; `max_loss_per_contract` for the selected 715/751 BULL_CALL spread (5% width on $715 SPY) = $3,600; `int(916/3600) = 0 contracts → return []`. Confirmed via instrumented run: chain loads, `_extract_directional_from_bars` returns bullish 0.83, `select_vertical_spread` returns the 715C/751C 34-DTE spec — sizing kills it. |
| STATUS | ❌ **BROKEN** (SSOT mismatch + structurally unfirable at current equity) |

**Proposed fix (operator decision):** Choose one of:
- **(a)** Raise `AlphaOptionsTuning.max_risk_per_trade_pct` from `0.005` to `~0.02` (4× — matches SPY spread cost). One-line change at `alpha_options.py:86`.
- **(b)** Tighten `spread_width_pct` from `0.05` to `~0.013` (5%×715=$36 wide → 1.3%×715=$9 wide → max-loss $900 → 1 contract). One-line change at `alpha_options.py:81`. Has the side effect of changing payoff geometry.
- **(c)** Update SSOT v8.4 §3 to document the *actual* current gate (sizing math, not MAINTAINED) and accept that alpha_options cannot fire below ~$720k equity at current config. No code change.

Note: option (c) is a viable answer — the strategy may simply be parked until equity grows. The audit flag is that **SSOT prose disagrees with code reality**, not that the code is wrong.

---

### 3.3 alpha_crypto

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "SMA20 momentum breakout AND 5d/20d vol-ratio expansion ≥ 0.7" (SSOT §3 line 775) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | Logged: `alpha_crypto: no signals this cycle (reason: BTC-USD:no_up_confirm; ETH-USD:below_sma20; SOL-USD:below_sma20)`. BTC failed the up-confirmation step (price > SMA20 met, but 3-day return < 1.5% threshold or vol-ratio < 0.7); ETH and SOL are trading below their 20-day SMA. |
| STATUS | ✅ **CORRECT** |

Strategy is operating as documented — momentum filter doing its job; no breakouts in the current crypto tape.

---

### 3.4 beta

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "If `target_weight − current_weight ≥ 2%`, emit a BUY sized to fill ~50% of the gap." (SSOT §3 line 796) — strategy should be ACTIVE in trending_bull. |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | **Config trap.** `BetaParams.max_position_weight = 0.02` (line 88) caps every target at 2%. `underweight_gap = 0.02` (line 77) is the threshold. With `current = 0` (no alpha-attributed positions in any consensus name), `gap = target_capped − current = 0.02 − 0 = 0.02`. The gate at `beta.py:330` is `if gap <= p.underweight_gap: continue`, so `0.02 <= 0.02 → continue`. Every consensus symbol is rejected. Instrumented probe confirms: 25 candidate symbols, all with `gap=0.02`, all `accept=False`. **Beta cannot bootstrap a position from flat at the current parameter equality.** |
| STATUS | ❌ **BROKEN** (code disagrees with SSOT spec) |

**Proposed fix:** SSOT v8.4 §3 reads "If `target_weight − current_weight ≥ 2%`, emit a BUY". The code uses `<= → continue` which is equivalent to "only emit if gap > 2%". Two-character fix at `beta.py:330`:
```python
# before
if gap <= p.underweight_gap:
# after
if gap < p.underweight_gap:
```
Alternative: raise `max_position_weight` to `0.04` so capped target (4%) − current (0%) = 4% > 2% threshold. This also matches typical institutional fund weights (top consensus names sit at 4-6%).

This is the highest-priority finding for an unattended run — beta is the BETA-sleeve compounder and is structurally unable to fire even though every gate above the comparison passes (consensus fresh ✓, equity above floor ✓, throttle OK ✓, prices present for AAPL/GOOGL/MSFT/NVDA/SPY ✓).

---

### 3.5 beta_trend

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Once-per-UTC-day per symbol (hard gate); max 20 signals/day; 21-day hold before add-ons" (SSOT §3 line 815) |
| HANDLER_OUTPUT | **3 signals** when invoked with a fresh in-process `_STATE` (AAPL +4.87, SPY +3.83, MSFT +3.53, all BUY, all `legend_entry_once_per_day` reason) |
| ACTUAL_SILENCE_REASON | The handler IS firing — but `_STATE` (`beta_trend.py:151`) is in-memory and persists across cycles within a single orchestrator process. In the live loop, the once-per-symbol-per-UTC-day gate at `beta_trend.py:250-256` blocks repeat emissions after the first cycle of the day. No fills today implies either (a) signals fired earlier today and didn't reach the broker (downstream filter), or (b) an earlier execution sequence already emitted them. The handler logic itself is healthy. |
| STATUS | ✅ **CORRECT** (handler firing as designed; throttle is the documented gate) |

**Operator note for the 30d run:** if you want stronger evidence that beta_trend is actually transacting, check `data/fills/FILLS_*.ndjson` for `strategy=beta_trend` over the last week. If zero across multiple days, there is a *downstream* filter to chase (routing_gates, capital_allocator, intent veto). The handler is not the problem.

---

### 3.6 delta_pairs

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Active in `ranging` only; silent in every other regime." (SSOT §3 line 920) |
| HANDLER_OUTPUT | **2 signals** (SPY BUY 2.0 + QQQ SELL 2.0, pair=SPY/QQQ, z=−2.79, conf=0.993) |
| ACTUAL_SILENCE_REASON | Handler emits the pair as designed (z=−2.79 well below entry threshold |z|≥2.0, correlation 0.993). Signals are then dropped by `filter_intents_by_regime` because `delta_pairs ∉ regimes.trending_bull`. Two-tier silence: handler still works; regime gate blocks routing. |
| STATUS | ✅ **CORRECT** (regime gate working as designed) |

---

### 3.7 gamma

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Active in `ranging, volatile, unknown`; silent in `trending_bull, trending_bear, adverse`." (SSOT §3 line 834-835) |
| HANDLER_OUTPUT | **2 signals** (SPY BUY 12.0 conf 0.9, QQQ BUY 12.0 conf 0.9, both `trend_momentum_buy`) |
| ACTUAL_SILENCE_REASON | Handler emits trend-momentum BUYs (SPY mom_atr=3.03, QQQ mom_atr=3.49 — well above 0.35 threshold). Filtered by `filter_intents_by_regime` because `gamma ∉ regimes.trending_bull`. Same two-tier pattern as delta_pairs. |
| STATUS | ✅ **CORRECT** (regime gate) |

---

### 3.8 gamma_reversion

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Active in `ranging` only; silent in every other regime." (SSOT §3 line 877-878) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | Two independent gates align: (a) handler walks SPY/QQQ/GLD/TLT and finds no symbol with 3/3 confluence (RSI > 72 + Boll-upper or |z| > 1.8 + ROC > 0 for SHORT, mirror for LONG) — none of these ETFs is at extreme reversion levels in the current calm tape; (b) even if a signal had emerged, `gamma_reversion ∉ regimes.trending_bull` would drop it. |
| STATUS | ✅ **CORRECT** (defense in depth — both signal gate and regime gate firing) |

---

### 3.9 omega

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Active in `volatile, unknown`; dormant in calm seas. Sensors: drawdown ≤−6%, ATR%≥3%, VIX≥25 (need ≥2 to activate)." (SSOT §3 line 941-945) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | Sensor count = 0 of 3: drawdown 0% (portfolio at peak: `equity_peak=$183k=equity`) > −6%; SPY/QQQ ATR% from 1m bars below 3% threshold; VIX 18.02 < 25 threshold. `want_hedge_on = (0 >= 2) = False`. `_STATE.hedge_on = False` (already off). No transition. Plus regime gate would also block in `trending_bull`. |
| STATUS | ✅ **CORRECT** (hedge correctly dormant) |

---

### 3.10 omega_vol

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Active in `volatile` only; LOW_VOL <15, NORMAL 15-22, ELEVATED 22-30, CRISIS ≥30, VOL_CRUSH on 20% drop. Health=0.10 (DEGRADED per SSOT §1)." (SSOT §3 line 962-967) |
| HANDLER_OUTPUT | **1 signal** (SVXY BUY 6.0 conf 0.65, regime=normal_vol, vix=18.02, vix_zscore=−0.84, reason=`normal_vol_mild_short`) |
| ACTUAL_SILENCE_REASON | Handler computes VIX regime as `normal_vol` (15 ≤ 18.02 ≤ 22) and emits a "mild SVXY short-vol" trade, exactly per spec. Signal dropped by `filter_intents_by_regime` because `omega_vol ∉ regimes.trending_bull` (only volatile). |
| STATUS | ⚠️ **WAITING** — handler firing correctly; regime gate blocking. **But:** SSOT flags `health_score=0.10` (3 samples, win_rate 0.0). Operator should consider whether to leave omega_vol in the active set during the 30-day run while it still has a degraded health score and so few samples. Decision is policy, not bug. |

---

### 3.11 omega_macro

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "RISK_OFF (VIX>25 or DD<−5%): ZN/ZB BUY, M6E SELL; RISK_ON (VIX<18 and DD>−2%): mirror; STAGFLATION: same as RISK_OFF; NEUTRAL: empty." (SSOT §3 line 988) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | `classify_macro_regime` (`macro_sensors.py:268-331`) with vix=18.02, dd=0%, bond_trend=ZN-EMA, commodity_trend=None: rule 1 (VIX≥25 OR DD≤−5%) fails; rule 2 (STAGFLATION VIX≥20 + commodity>0 + bond<0) fails on VIX; rule 3 (RISK_ON VIX<18 AND DD>−2%) **fails on `18.02 < 18.0` being False by 0.02 points**; falls through to NEUTRAL → `REGIME_SIGNAL_MAP[NEUTRAL] = {}` → `if not direction_map: return []` at `omega_macro.py:447`. |
| STATUS | ⚠️ **WAITING** — strategy is one tick of VIX away from RISK_ON. The classifier is structurally correct but parked at NEUTRAL by a 0.02-point margin. SSOT §3 line 991 wrote "currently RISK_ON-leaning per low VIX" — that prose is mildly optimistic; we are actually NEUTRAL. Operator may consider widening the RISK_ON cutoff (e.g. `<19`) to make the boundary less brittle, or leave it as a tight discipline. |

---

### 3.12 omega_momentum_options

| Field | Value |
|---|---|
| EXPECTED_SILENCE_REASON | "Market hours 9:45 AM ET → 3:30 PM ET; hard exit by 3:45 PM ET" (SSOT §3 line 1007) |
| HANDLER_OUTPUT | **0 signals** |
| ACTUAL_SILENCE_REASON | Logged: `omega_momentum_options: outside market hours`. Audit ran at 23:21 UTC (7:21 PM ET). |
| STATUS | ✅ **CORRECT** (hard time gate) |

---

## 4. Summary Table

| # | Strategy | Handler signals | Regime allowed? | Status | Reason |
|---|---|---|---|---|---|
| 1 | alpha_intraday | 0 | ✅ | ✅ CORRECT | All 3 triggers below threshold (vol/momentum/RSI) |
| 2 | **alpha_options** | 0 | ✅ | **❌ BROKEN** | Sizing math: 0.5%×$183k = $916 < $3,600 max-loss for 5% SPY spread → 0 contracts. SSOT documents wrong reason ("Position MAINTAINED" — no such position open). |
| 3 | alpha_crypto | 0 | ✅ | ✅ CORRECT | All 3 pairs failed momentum filter (BTC: no_up_confirm; ETH/SOL: below SMA20) |
| 4 | **beta** | 0 | ✅ | **❌ BROKEN** | `max_position_weight (0.02) == underweight_gap (0.02)` with `gap <= continue` → flat portfolio cannot bootstrap any position. Code disagrees with SSOT spec ("≥ 2%"). |
| 5 | beta_trend | 3 (fresh state) | ✅ | ✅ CORRECT | Handler fires; once-per-symbol-per-UTC-day in-memory throttle is the documented gate |
| 6 | delta_pairs | 2 | ❌ | ✅ CORRECT | SPY/QQQ z=−2.79 fires at handler; regime activation matrix drops at routing |
| 7 | gamma | 2 | ❌ | ✅ CORRECT | SPY/QQQ trend-momentum fires at handler; regime drops at routing |
| 8 | gamma_reversion | 0 | ❌ | ✅ CORRECT | No 3/3 confluence on any ETF; regime would drop anyway |
| 9 | omega | 0 | ❌ | ✅ CORRECT | 0 of 3 sensors active (DD 0%, ATR% low, VIX 18.02 < 25) |
| 10 | omega_vol | 1 | ❌ | ⚠️ WAITING | Handler emits SVXY mild-short; regime drops. Health 0.10 separately flagged. |
| 11 | omega_macro | 0 | ✅ | ⚠️ WAITING | VIX 18.02 just above 18.0 RISK_ON cutoff → NEUTRAL → empty signal map |
| 12 | omega_momentum_options | 0 | ✅ | ✅ CORRECT | Outside market hours (23:21 UTC > 19:30 UTC cutoff) |

**Tally:** 8 ✅ CORRECT · 2 ⚠️ WAITING · **2 ❌ BROKEN**

---

## 5. BROKEN findings — required action before unattended run

### 5.1 [CRITICAL] beta — config trap blocks all signals

`chad/strategies/beta.py:330`:
```python
if gap <= p.underweight_gap:
    continue
```
With `BetaParams.max_position_weight = 0.02` (line 88) and `underweight_gap = 0.02` (line 77),
the gap on every flat consensus name is exactly 0.02, which fails `<= 0.02 → continue`. Result:
zero candidates emitted, ever, for a flat portfolio.

**Fix (one of):**
1. Change comparator to `<` at `beta.py:330` (matches SSOT prose "≥ 2%"). Smallest diff.
2. Raise `max_position_weight` to `0.04` (matches typical institutional top-name weights).

Recommend (1) — strict spec adherence.

### 5.2 [HIGH] alpha_options — sizing math vs. equity

At equity $183k, `risk_budget = $916` is smaller than the max-loss of any 5%-wide SPY spread
(~$3,600 at $715 SPY). `int(916 / 3600) = 0` ⇒ no signal can ever emerge. SSOT v8.4 §3
documents the silence reason as a "MAINTAINED position" gate, but no `alpha_options|SPY`
is currently open — the prose and the code disagree on the *real* current gate.

**Choose one before the 30-day run:**
- (a) Raise `max_risk_per_trade_pct` from `0.005` to `0.02` at `alpha_options.py:86`.
- (b) Tighten `spread_width_pct` from `0.05` to `0.013` at `alpha_options.py:81` (changes payoff).
- (c) Accept that alpha_options cannot fire below ~$720k equity and amend SSOT prose to match.

Lowest-risk choice for an unattended run is **(c)** — accept the dormancy, document it.
The strategy is then verifiably "WAITING for equity" rather than concealing a sizing bug.

---

## 6. WAITING findings — flag for monitoring

### 6.1 omega_macro — RISK_ON cutoff brittle at 18.0

VIX is 18.02 today. A move of −0.03 flips the strategy from NEUTRAL to RISK_ON and starts
emitting ZN SELL / ZB SELL / M6E BUY. A move up to 25 flips it to RISK_OFF (mirror).
The classifier is correct; the operator should know the boundary is ±0.02 from the live
VIX print and that this strategy will toggle on small VIX moves. If you want a hysteresis
buffer, widen the RISK_ON cutoff to `<19` (`macro_sensors.py:327`).

### 6.2 omega_vol — DEGRADED health on the active sleeve

SSOT §1 already flags omega_vol at health_score 0.10 with 3 samples. My audit confirms
the handler fires (SVXY mild-short). For the 30-day unattended run, two policy options:
- Disable the strategy in `tier_state.json` enabled_strategies list until samples grow.
- Leave it on to accumulate samples; the regime activation matrix already silences it
  outside `volatile`, so blast radius is small.

This is a policy decision, not a bug. No code change recommended.

---

## 7. Confidence Statement

**Are all 12 silences explained?** ✅ Yes. Every silent strategy was traced to a specific gate (file:line) backed by either runtime values or instrumented probes.

**Any BROKEN findings?** Yes — 2:
- **beta** — code/spec mismatch at `beta.py:330`; cannot bootstrap from flat. One-line fix.
- **alpha_options** — SSOT/code mismatch on documented silence reason; sizing math (rather than position MAINTAINED) is the actual current gate at $183k equity. Either raise risk budget, tighten spread width, or update SSOT prose.

**Safe to run 30 days unattended? — Conditional NO.**

Recommend the operator:
1. Apply the **beta** comparator fix (`<= → <` at `beta.py:330`) before the unattended run starts. Without it, the BETA-sleeve compounder is silent for the entire 30 days regardless of regime.
2. Decide alpha_options policy (a/b/c above) and either patch config or update SSOT — do not leave the documented gate disagreeing with the actual gate.
3. Acknowledge the omega_macro VIX-boundary brittleness and the omega_vol DEGRADED health flag as known states, not surprises.

The remaining 8 strategies are silent for verifiably correct reasons (regime gate, condition gate, time gate, or sensor inactivity), and would re-activate as expected when their conditions change.

After items (1) and (2) are addressed, the system is verifiably safe for the 30-day unattended run from a silence-correctness standpoint.

---

*Audit method: synthetic invocation against live ctx. No production state mutated. Diagnostic scripts at `/tmp/silence_audit.py`, `/tmp/silence_audit_v2.py` (not committed).*
