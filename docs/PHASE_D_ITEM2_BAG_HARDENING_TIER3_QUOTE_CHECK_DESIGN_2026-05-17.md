# Phase D Item 2 — BAG Hardening Tier 3 Quote Check Design

**Date:** 2026-05-17
**Author:** CHAD engineering (audit + design pass)
**Scope:** chad/options/, chad/execution/ibkr_adapter.py, chad/market_data/,
chad/strategies/alpha_options.py
**Companion docs:**
- docs/PHASE_D_ITEM2_BAG_HARDENING_TIER1.md
- docs/PHASE_D_ITEM2_BAG_HARDENING_TIER2_DESIGN_2026-05-17.md

---

## 1. Status

**DESIGN ONLY — NO LIVE QUOTE REQUESTS**

This document does **not** authorize any source modification, runtime
mutation, deploy/systemd change, IBKR connection, market-data
subscription, or order placement. It exclusively designs the offline
quote-check module that Tier 3 will implement in a follow-on prompt.

Live BAG execution remains **NOT AUTHORIZED** until at minimum Tier 3
(this design), Tier 4 (bracket/failsafe), Tier 5
(spread_id reconciliation), and Tier 6 (live fill harness) land.

---

## 2. Tier 1 / Tier 2 baseline

### 2.1 Tier 1 — typed OptionsSpreadSpec (committed 51166d4)

- `chad/options/spread_spec.py` — frozen dataclass `OptionsSpreadSpec`
  (symbol, expiry, long/short strike, long/short right, ratios,
  exchange/currency, spread_type, max_loss_per_contract,
  `net_debit_estimate`, spread_id, dte) with strict validators and
  `from_legacy_meta` / `as_dict` bridges.
- `chad/strategies/alpha_options.py:473` stamps the typed spec under
  `meta["spread_spec"]` alongside the legacy string-keyed fields.
- `chad/execution/ibkr_adapter.py:1057-1130` consumes the typed spec
  with a `Mapping → from_legacy_meta` fallback when serialized through
  IPC.
- `chad/execution/paper_exec_evidence_writer.py:987-1004` hydrates the
  spec from either the typed object or legacy meta when stamping
  attribution records.
- `scripts/preview_bag_intent.py` — dry-run CLI that constructs an
  OptionsSpreadSpec, projects to legacy meta, and prints the
  contract+order summary. No IBKR import on the dry-run path.

### 2.2 Tier 2 — BAG LMT discipline (committed 2ab03e7)

`chad/execution/ibkr_adapter._resolve_bag_lmt_discipline` (line 1260):

| Condition | Action | Marker |
|---|---|---|
| `sec_type == "BAG"` and `order_type != "LMT"` | coerce to LMT | `BAG_MKT_COERCED_TO_LMT` |
| `limit_price` missing or non-positive | hydrate from `meta["net_debit_estimate"]`, then `meta["spread_spec"].net_debit_estimate` | `BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE` |
| No positive limit price derivable | return `None` → caller skips submission | `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE` |
| `sec_type != "BAG"` | pass-through unchanged | — |

Test coverage: `chad/tests/test_bag_lmt_discipline.py` (10+ cases over
coercion, hydration, skip, and non-BAG no-op). Full suite: **2074 passed**.

### 2.3 What Tier 1 + Tier 2 do *not* do

- They do **not** validate the LMT price against any broker mid, leg
  bid/ask, or theo price.
- They do **not** request any market data, options or otherwise.
- They do **not** authorize live BAG execution.

The hydrated limit price is `actual_width × 100 × 0.50` — a strike-only
heuristic (`chad/options/strike_selector.py:218`) with no market
corroboration.

---

## 3. Current quote reality

### 3.1 Equity / FX / futures path (exists)

- `chad/market_data/ibkr_price_provider.py:121` `IBKRPriceProvider` —
  thread-safe single-symbol snapshot.
  - Supports `STK`, `FX`, `FUT` via `_make_contract` (line 165).
  - Calls `self._ib.reqMktData(contract, "", False, False)` and reads
    `ticker.{last, close, bid, ask}` (line 220-234).
  - Returns `PriceSnapshot(symbol, last, close, bid, ask, ts_utc,
    source, delayed)` — already carries `bid`/`ask` fields.
  - Sets `reqMarketDataType(3)` (delayed OK) at init.
  - Falls back to `runtime/price_cache.json` when all ticks NaN.
- `chad/market_data/service.py:113` `IBKRMarketDataService` — thin
  facade. Currently discards `bid`/`ask` and returns a one-field price
  (`snap.last if snap.last > 0 else snap.close`, line 140) into
  `service.PriceSnapshot(symbol, asset_class, price, …)`. Bid/ask are
  available in the underlying provider but **not exposed** at the
  service layer.

### 3.2 Options path (does not exist for execution)

- `chad/market_data/options_chain_refresh.py` writes
  `runtime/options_chains_cache.json` via `reqContractDetails` — it
  records **expirations + strikes only** (line 329). No per-strike
  bid/ask is persisted. The `(bid+ask)/2` reference at line 176-181 is
  used only for the underlying spot price, **not** for any option
  contract.
- `chad/options/chain_provider.py` — grep for `bid|ask|reqMktData`
  returns **zero hits**. The chain provider is metadata-only.
- `runtime/options_chains_cache.json` confirmed shape (snapshot
  2026-05-15T12:30:28Z):
  ```
  chains.SPY = {symbol, exchange, expirations[...], strikes[...],
                ts_utc, ttl_seconds}
  ```
  No `bids`, no `asks`, no per-strike pricing.
- `runtime/options_greeks.json` (`options_greeks.v1`, 2026-05-16,
  `provider_status: approximated`):
  per `symbol.expirations.<exp>.strikes.<strike>`:
  `{call_delta, put_delta, call_gamma=None, put_gamma=None,
    call_theta=None, put_theta=None, call_theo_price, put_theo_price,
    moneyness, near_atm, source: synthetic}`.
  - `call_theo_price` / `put_theo_price` are Black-Scholes synthetics
    derived from `spot, strike, T, iv (from VIX), r` in
    `chad/market_data/options_greeks_publisher.py:152-217`.
  - These are **theoretical**, not broker mid. They are explicitly
    labelled `source: synthetic`, `provider_status: approximated`.

### 3.3 BAG / combo quote (does not exist)

- `chad/execution/ibkr_adapter.py` — no `reqMktData` call on a BAG
  contract anywhere. The BAG is constructed (line 1213-1234) and either
  qualified leg-by-leg or fed into `placeOrder`. Combo bid/ask is never
  requested.
- Repository-wide grep for `combo.*quote|bag.*quote|combo_mid|
  bag_mid|combo_bid|combo_ask|get_bag_mid` returns **zero production
  hits** — only the `test_quarantine_20260511_alpha_options.py`
  attribution reason string `synthetic_bag_close_credit_30pct_not_
  market_quote`, which is a paper-fill record explicitly tagged "NOT a
  market quote".

### 3.4 Summary table — quote reality

| Asset | Bid/ask source | Currently consumed by execution? |
|---|---|---|
| Equity / ETF underlying | `IBKRPriceProvider.get_snapshot` → ticker.bid/ask (delayed OK) | No — service discards bid/ask; `last`/`close` only |
| FX | Same provider, Forex contract | last/close/mid fallback (`get_fx_rate`) |
| Future | Same provider with `FUT` + tradingClass | last/close only |
| Option leg (per strike) | None — chain cache stores no quotes; greeks are synthetic | No |
| BAG / combo | None — no `reqMktData` ever called on BAG | No |

---

## 4. Existing code reusable for Tier 3

| Component | Reusable for offline Tier 3? | Notes |
|---|---|---|
| `OptionsSpreadSpec` | **Yes** — direct input to quote checker | Provides validated symbol/expiry/strike/right pairs |
| `chad/options/strike_selector.py::SpreadSpec.net_debit_estimate` | **Yes** — value to be checked | Pure heuristic; explicitly the thing the quote check guards |
| `chad/utils/options_greeks_gate.get_option_greeks` | **Yes (advisory only)** | Provides synthetic `theo_price` — usable as *fallback* mid, NEVER as broker mid |
| `IBKRPriceProvider` | **No** — wrong contract type, STK/FX/FUT only | Out of scope for offline Tier 3 |
| `IBKRMarketDataService` | **No** — discards bid/ask | Out of scope for offline Tier 3 |
| `options_chains_cache.json` | **No** — no quotes stored | Out of scope until refresher learns to write quotes |

No production code path needs modification to land Tier 3 as designed
below. The offline module is purely additive.

---

## 5. Recommended architecture (Option A — offline first)

### 5.1 New file

`chad/options/quote_check.py` — pure offline quote validation.

**Hard import rules:**
- ❌ No `from ib_async import …`
- ❌ No `import chad.execution.…`
- ❌ No `import chad.strategies.…`
- ❌ No `chad.market_data.ibkr_*` import (those modules import ib_async)
- ✅ May import `chad.options.spread_spec` (pure)
- ✅ May import `chad.utils.options_greeks_gate` (pure, read-only JSON)
- ✅ Stdlib only otherwise (dataclasses, math, typing, optional logging)

### 5.2 Data shapes

```python
@dataclass(frozen=True)
class OptionLegQuote:
    """A single leg's broker-side quote, or its synthetic fallback."""
    symbol: str
    expiry: str          # YYYYMMDD
    strike: float
    right: str           # "C" | "P"
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]     # caller-supplied mid; if None, derived from bid/ask
    theo_price: Optional[float]  # synthetic black-scholes fallback only
    ts_utc: Optional[str]    # quote freshness
    source: str              # "broker_leg" | "broker_combo" | "synthetic_theo" | "fake"

@dataclass(frozen=True)
class BagComboQuote:
    """Optional direct BAG combo quote."""
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    ts_utc: Optional[str]
    source: str              # "broker_combo" | "fake"

@dataclass(frozen=True)
class SpreadQuoteCheckInput:
    spec: OptionsSpreadSpec
    proposed_limit_price: float
    combo_quote: Optional[BagComboQuote]      # preferred when present
    long_leg_quote: Optional[OptionLegQuote]
    short_leg_quote: Optional[OptionLegQuote]
    max_deviation_abs: float = 0.05           # absolute $ floor
    max_deviation_pct: float = 0.10           # 10% of computed mid
    quote_max_age_seconds: float = 30.0       # freshness window

@dataclass(frozen=True)
class SpreadQuoteCheckResult:
    ok: bool
    reason: str                        # short marker: see §5.4
    mid_used: Optional[float]
    mid_source: str                    # "combo" | "leg_diff" | "theo" | "none"
    deviation_abs: Optional[float]     # proposed_limit_price - mid_used
    deviation_pct: Optional[float]     # deviation_abs / mid_used
    threshold_abs: float
    threshold_pct: float
    stale: bool                        # any consulted quote older than ttl
    notes: List[str]
```

### 5.3 Core function

```python
def evaluate_spread_quote(
    input: SpreadQuoteCheckInput,
    *,
    now_utc: Optional[datetime] = None,
) -> SpreadQuoteCheckResult: ...
```

Decision tree (in order):

1. **Combo quote preferred.** If `combo_quote.mid` is positive & finite
   AND its `ts_utc` is within `quote_max_age_seconds`, use it.
   `mid_source="combo"`.
2. **Leg fallback.** Else, if both leg `mid` values (or both
   `(bid+ask)/2` derivations) are positive & finite AND fresh:
   compute `debit_mid = long_mid − short_mid` for debit verticals
   (BULL_CALL / BEAR_PUT). `mid_source="leg_diff"`.
3. **Synthetic theo fallback.** Else, if both `theo_price` values are
   positive & finite: `theo_debit = long_theo − short_theo`.
   `mid_source="theo"`, `notes` records `"synthetic_theo_fallback"`.
4. **No mid available.** Return `ok=False, reason=
   "BAG_QUOTE_CHECK_NO_MID_AVAILABLE", mid_used=None,
   mid_source="none"`.

Once a mid is selected:

- Compute `deviation_abs = proposed_limit_price − mid_used` and
  `deviation_pct = deviation_abs / mid_used` (mid_used > 0 guard).
- For **debit** verticals, the trader buys the spread; an above-mid
  limit is acceptable, a far-below-mid limit is suspicious (we may
  underpay and never fill, but more importantly a stale heuristic
  could overpay). Tolerance is **two-sided**:
  - `|deviation_abs| ≤ max(max_deviation_abs, max_deviation_pct × mid_used)`
- If within tolerance: `ok=True, reason=
  "BAG_QUOTE_CHECK_OK_<source>"`.
- If outside tolerance: `ok=False, reason=
  "BAG_QUOTE_CHECK_LIMIT_OUT_OF_TOLERANCE"`.
- If any consulted quote is stale beyond `quote_max_age_seconds`:
  `stale=True`, and the result is forced `ok=False, reason=
  "BAG_QUOTE_CHECK_STALE_QUOTE"`.
- All branches are pure / deterministic / no IO.

### 5.4 Result markers

Exact strings (chosen to mirror Tier 2's `BAG_…` prefix so log scrapes
unify cleanly):

- `BAG_QUOTE_CHECK_OK_COMBO`
- `BAG_QUOTE_CHECK_OK_LEG_DIFF`
- `BAG_QUOTE_CHECK_OK_THEO_FALLBACK_ADVISORY`
- `BAG_QUOTE_CHECK_NO_MID_AVAILABLE`
- `BAG_QUOTE_CHECK_LIMIT_OUT_OF_TOLERANCE`
- `BAG_QUOTE_CHECK_STALE_QUOTE`

`THEO_FALLBACK_ADVISORY` is **never** acceptable for live submission —
in Tier 3 it returns `ok=True` because Tier 3 does not gate the adapter
yet, but downstream wiring in Tier 3b (adapter enforcement) must reject
this result for live mode.

### 5.5 Tests (no IBKR, no live data)

`chad/tests/test_bag_quote_check.py` covers:

1. Combo mid within tolerance → `OK_COMBO`.
2. Combo mid outside tolerance → `LIMIT_OUT_OF_TOLERANCE`.
3. Combo missing, leg mids present, leg-diff within tolerance →
   `OK_LEG_DIFF`.
4. Combo missing, one leg bid/ask missing → falls through to theo.
5. All broker mids missing, theo prices present → `OK_THEO_FALLBACK_ADVISORY`
   + `notes` records `"synthetic_theo_fallback"`.
6. All sources missing → `NO_MID_AVAILABLE`.
7. Stale combo quote → `STALE_QUOTE` even when arithmetic is in
   tolerance.
8. Stale leg quote w/ fresh combo → ignores leg quote (combo wins),
   still `OK_COMBO`.
9. Same-strike spread input → reject at OptionsSpreadSpec level (not
   quote_check's job; tested indirectly).
10. `proposed_limit_price` non-positive → `LIMIT_OUT_OF_TOLERANCE` with
    deviation_abs >= mid_used.
11. Tolerance picks the larger of `(abs floor, pct × mid)` — verified
    with a tight-mid case where pct floor dominates and a wide-mid
    case where abs floor dominates.

A `FakeQuoteProvider` test helper constructs `OptionLegQuote` /
`BagComboQuote` directly — there is no `IB` shim, no `Ticker`, no
async, no networking.

### 5.6 What Tier 3 does NOT do

- Does **not** call `reqMktData`.
- Does **not** call `ibkr_price_provider`, `ibkr_bar_provider`, or any
  `ib_async` symbol.
- Does **not** modify `ibkr_adapter._resolve_bag_lmt_discipline`.
- Does **not** modify `alpha_options` or any strategy.
- Does **not** mutate `runtime/`.
- Does **not** install systemd units.
- Does **not** authorize the adapter to enforce the check yet.

---

## 6. Future live quote path (Tier 3b and beyond)

Out of scope for Tier 3 implementation — captured here so the offline
module's interface accommodates it.

### 6.1 Direct BAG quote (preferred)

Once Tier 3 lands and an offline gate exists, a follow-on tier may add
`chad/market_data/options_quote_provider.py`:

- Builds the same BAG contract as `ibkr_adapter._resolve_bag_contract`
  (reuse, do not duplicate the assembly logic).
- Calls `ib.reqMktData(bag, "", snapshot=False, regulatorySnapshot=False)`
  with `reqMarketDataType(3)` (delayed OK).
- Reads `ticker.{bid, ask, modelGreeks}` for the combo.
- Returns `BagComboQuote(bid, ask, mid=(bid+ask)/2, ts_utc=..., source="broker_combo")`.

### 6.2 Leg fallback

- For each leg, build the standalone `Option` contract (already
  qualified during BAG resolution — reuse `long_opt`, `short_opt` from
  `_resolve_bag_contract` line 1135-1150).
- `reqMktData` on each leg, await `ticker.bid` + `ticker.ask` with the
  same 5s timeout used by `IBKRPriceProvider.SNAPSHOT_TIMEOUT_S`.
- Compute `mid = (bid+ask)/2` per leg; emit two `OptionLegQuote`
  objects.

### 6.3 Freshness

`ticker.time` (when present) → ISO `ts_utc`. Default
`quote_max_age_seconds = 30.0` for a marketable LMT use case. Delayed
data (`reqMarketDataType(3)`) is typically ~15-minute lag — the
freshness check is the gate that distinguishes "delayed is fine for
mid bracketing" from "no live ticker at all".

### 6.4 Tolerance policy

Implementation can start with the offline-module defaults
(`max_deviation_abs=$0.05`, `max_deviation_pct=10%`). Operator-tunable
once we see real combo mids vs heuristic limit prices on the
`scripts/preview_bag_intent.py` corpus. **Tier 3 does not commit to a
numeric policy at the source level** — the result struct returns the
deviation; the operator-visible threshold lives in
`SpreadQuoteCheckInput`.

### 6.5 No order placement

Quote requests are read-only. `placeOrder` is still gated by the
existing LiveGate + execution-mode posture. The Tier 3b provider
returns a `BagComboQuote` and exits — it never calls `placeOrder`.

---

## 7. Safety rules (Tier 3 implementation)

The Tier 3 implementation prompt must enforce:

- No live quote request anywhere in `chad/options/quote_check.py`.
- No order placement anywhere in the new module.
- No adapter enforcement of the quote check yet — the offline module
  is callable but unwired.
- No strategy behavior change — `alpha_options` is untouched.
- No runtime/ mutation in tests (tests use ephemeral fixtures or
  `tmp_path`).
- No systemd / deploy file changes.
- No commit until tests pass full suite (currently 2074 passing).

---

## 8. Risk / gap table

| Quote-check concern | Current evidence | Risk | Recommended design |
|---|---|---|---|
| No real bid/ask in options_chains_cache | `options_chain_refresh.py:329` writes only expirations+strikes; grep `bid\|ask` on `chain_provider.py` → 0 hits | High — no broker corroboration possible from cache | Do not depend on chain cache for quotes; future Tier 3b live probe is the only path to real bid/ask |
| Synthetic greeks/theo prices only | `options_greeks.json` `provider: synthetic_black_scholes_vix`, `source: synthetic`, no gamma/theta | Med — theo can drift far from market in skew or earnings events | Theo permitted as **advisory** fallback only; result marker `OK_THEO_FALLBACK_ADVISORY` must be rejected by live-mode adapter gate (Tier 3b) |
| BAG combo quote unavailable today | adapter never calls `reqMktData` on BAG; no `combo_mid` helper anywhere in `chad/` | High — heuristic `0.50 × width × 100` can be far off true combo mid in volatile or skewed conditions | Tier 3b live probe with combo-preferred path; Tier 3 offline module accepts `BagComboQuote=None` and falls through |
| Leg quote fallback needed | `IBKRPriceProvider` is STK/FX/FUT only; no OPT path | Med — when combo quote is illiquid IBKR may not return a combo bid/ask even if legs are quoted | Tier 3 module accepts independent leg quotes; live probe (Tier 3b) calls `reqMktData` on `long_opt`/`short_opt` directly |
| Limit price tolerance policy | Tier 2 hydration is unconditional; no tolerance enforcement | Med — a stale heuristic can submit at a price the market won't fill (passive miss) or far through the mid (slippage) | Two-sided tolerance `|dev_abs| ≤ max(abs_floor, pct × mid)`; operator-tunable via input struct |
| Quote freshness | `IBKRPriceProvider` has 5s snapshot timeout but no caller-side staleness check; option chains TTL=3600 | Med-high — delayed-data path can return ticks tens of minutes old | `quote_max_age_seconds=30.0` default; stale quote returns `STALE_QUOTE` regardless of arithmetic |
| Live broker subscription requirements | `reqMarketDataType(3)` (delayed) used everywhere; options_chain_refresh hits Error 10089 path (`options_chain_refresh.py:195`) | Low for delayed/free path; Unknown for combo quotes (may require OPRA L1 subscription) | Tier 3 is offline → no subscription needed; Tier 3b documents the subscription requirement before any live probe |
| Dry-run fake provider tests | `scripts/preview_bag_intent.py` already operates with `ib=None`; no live BAG quote test exists | Low | Tier 3 tests use synthetic `OptionLegQuote` / `BagComboQuote`; no `IB` shim required |
| Adapter integration boundary | `ibkr_adapter._resolve_bag_lmt_discipline` (line 1260) is the single BAG limit-price chokepoint; the natural place to call a quote check | Med — coupling Tier 3 to the adapter prematurely re-introduces the live-quote dependency | Tier 3 module is unwired. Tier 3b adds the adapter call after the offline module is proven |

---

## 9. Tier 3 classification

**A. READY FOR IMPLEMENTATION AS OFFLINE QUOTE-CHECK MODULE ONLY.**

Evidence:

- (A) is supported: `OptionsSpreadSpec`, synthetic theo prices, and
  `net_debit_estimate` provide enough fixture data for a unit-tested
  offline module.
- (B) is not supported today: there is no combo-quote helper, no
  options leg quote in the chain cache, and the price provider is
  STK/FX/FUT only.
- (C) is plausible but unconfirmed: combo / OPT L1 quotes may require
  a paid OPRA subscription; resolving that is a deploy-side question
  for Tier 3b, not Tier 3.
- (D) is rejected: deferring the quote check until bracket /
  reconciliation lands leaves Tier 4+ without a quote-validation
  primitive to compose with. The offline module is the dependency
  Tier 4-6 will rest on.

---

## 10. Recommended next implementation task

**Option A — Build offline quote-check module first.**

Justification:

1. Zero broker-coupling risk. No `reqMktData`, no `placeOrder`, no
   `ib_async` import — cannot accidentally hit a live socket.
2. Composable. Once landed, Tier 3b (live combo probe) only adds an
   `OptionsQuoteProvider` whose output flows directly into the
   already-tested `evaluate_spread_quote`.
3. Provides a unit-tested gate available to `scripts/preview_bag_intent.py`
   immediately — operator can run the check against fake quotes today
   and against real quotes once Tier 3b lands.
4. Matches Tier 1 / Tier 2 cadence (additive module, no behavior
   change, no runtime mutation, no live coupling) and the established
   safety rules (no full rewrites, one change at a time).
5. The risk/gap table maps cleanly to a single small module with a
   pure function and a typed result.

Option B (live IBKR quote probe first) is rejected — it requires a
broker connection, a subscription investigation, and a quote-result
schema that is only fully constrained once the offline checker exists.
Option C (adapter enforcement first) is rejected by Step 7's stop
condition. Option D (defer) is rejected per §9.

---

## 11. Recommended next implementation prompt (outline)

**Title:** Phase D Item 2 — BAG Hardening Tier 3 Step 1: offline
quote-check module.

**Scope:**
- New file: `chad/options/quote_check.py` (only).
- New test file: `chad/tests/test_bag_quote_check.py` (only).
- No modification to `chad/execution/`, `chad/strategies/`,
  `chad/market_data/`, `chad/options/spread_spec.py`, `runtime/`,
  `deploy/`, `ops/`, `config/`, `shared/`, or existing docs.

**Acceptance:**
- `python3 -m py_compile chad/options/quote_check.py`
- `python3 -m py_compile chad/tests/test_bag_quote_check.py`
- `python3 -m pytest chad/tests/test_bag_quote_check.py -x -q`
- `python3 -m pytest chad/tests/ -x -q` → unchanged from 2074 passing
  baseline (≥ 2074 + new tests).
- `python3 chad/core/full_cycle_preview.py --dry-run` exits clean.
- No `from ib_async` import in either new file.
- No `import chad.execution`, `import chad.strategies`,
  `import chad.market_data.ibkr_*` in either new file.
- All 11 test cases in §5.5 present and passing.

**Stop conditions:**
- Working tree not clean at start.
- `chad/options/quote_check.py` already exists.
- Module imports `ib_async` or any `chad.market_data.ibkr_*`.
- Module mutates anything under `runtime/`.
- Tests require an `IB` shim or live socket.

---

## 12. Remaining live BAG blockers after Tier 3

Tier 3 closes one of five blockers identified in
`PHASE_D_ITEM2_BAG_HARDENING_TIER2_DESIGN_2026-05-17.md` §2 — and only
**partially**: it adds the offline gate but not the live probe.

Open blockers after Tier 3:

1. **Tier 3b — live quote probe.** A
   `chad/market_data/options_quote_provider.py` that requests combo
   and per-leg `reqMktData` and returns `BagComboQuote` /
   `OptionLegQuote` objects consumable by `evaluate_spread_quote`.
   Documents OPRA subscription requirement.
2. **Tier 4 — bracket / OCA or fail-safe exit.** No `parentId`,
   `transmit`, or OCA wiring anywhere in execution. Sole exit today
   is `alpha_options.max_hold_seconds=3600` driven by `live_loop`. A
   live BAG with `live_loop` paused is unmanaged.
3. **Tier 5 — spread_id-aware reconciliation / position tracking.**
   `paper_exec_evidence_writer` already stamps `spread_id`; live
   `trade_closer` must learn to FIFO-match the BAG pair as a single
   unit instead of two independent leg fills.
4. **Tier 6 — live BAG fill harness.** Pure offline test fixture
   exercising `placeOrder → Trade → orderStatus` for a BAG contract
   with a fake `IB` shim. Meaningful only after Tiers 3b/4/5.

Live BAG execution stays **NOT AUTHORIZED** until all four ship.

---

## 13. References

- `chad/options/spread_spec.py` — typed BAG contract
- `chad/options/strike_selector.py:218` — net_debit_estimate heuristic
- `chad/strategies/alpha_options.py:473` — spread_spec emission
- `chad/execution/ibkr_adapter.py:1057-1234` — BAG contract assembly
- `chad/execution/ibkr_adapter.py:1260-1330` — Tier 2 LMT discipline
- `chad/execution/paper_exec_evidence_writer.py:987-1004` — paper-fill
  spread_spec hydration
- `chad/market_data/ibkr_price_provider.py:201-277` — STK/FX/FUT
  snapshot reference for Tier 3b live probe
- `chad/market_data/options_chain_refresh.py:329` — chain cache shape
  (no quotes)
- `chad/market_data/options_greeks_publisher.py:152-217` — synthetic
  theo prices
- `chad/utils/options_greeks_gate.py` — read-only greeks lookup
- `scripts/preview_bag_intent.py` — dry-run preview CLI
- `chad/tests/test_bag_lmt_discipline.py` — Tier 2 coverage
- `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER1.md`
- `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER2_DESIGN_2026-05-17.md`
