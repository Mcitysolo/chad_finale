# Box 052 — BAG adapter quote-enforcement policy

**Status:** Pending Action — DOCUMENTATION ONLY. No production code change.
No runtime mutation. No order placement. No live authorization.

**Scope:** Locks the quote-enforcement contract for BAG (combo / vertical-spread)
intents at the IBKR adapter boundary (`chad/execution/ibkr_adapter.py`).
Pairs with Box 051 (per-share vs contract-dollar unit normalization).

---

## 1. Canonical enforcement policy

The adapter rejects unsafe BAG prices and consumes only broker-facing
**per-share / per-unit** values. Enforcement is **structural**, not
midpoint-based, and is implemented as two defense layers:

### Layer 1 — `_resolve_bag_lmt_discipline`
File: `chad/execution/ibkr_adapter.py:1623-1700`

Hydration source priority (only positive finite values accepted):

1. `intent.limit_price`  (when positive finite)
2. `meta["net_debit_estimate"]`
3. `meta["spread_spec"].net_debit_estimate`  (typed `OptionsSpreadSpec`)
4. `meta["spread_spec"]["net_debit_estimate"]`  (Mapping)

If none of the above yields a positive finite value, the function emits
`BAG_INTENT_SKIPPED_NO_LIMIT_PRICE` and returns `None`. The submit path
in `_submit_intent` (line 2277) consumes the `None` and exits **without
building or placing any order**.

`_positive_float` (line 1654) is the rejection filter:
- `None` → reject
- `TypeError`/`ValueError` from `float(value)` → reject (non-numeric string)
- `math.isfinite(out)` false → reject (NaN, +inf, -inf)
- `out <= 0.0` → reject (zero, negative)

### Layer 2 — `_OrderFactory.build` defense-in-depth
File: `chad/execution/ibkr_adapter.py:1729`

```python
if order_type == "LMT" and (math.isnan(limit_price) or limit_price <= 0.0):
    raise ValidationError("Limit order requires positive limit_price")
```

This gate is **sec_type-agnostic**: it protects ALL LMT order paths
(STK / FUT / OPT / CASH / BAG). Any future regression that bypasses
Layer 1 is caught here before any order reaches IBKR.

---

## 2. Hard rules

1. **Per-share contract is verbatim.** When a valid positive per-share
   value is derivable, the adapter forwards it to `intent.limit_price`
   and on to `order.lmtPrice` **without unit conversion** (no ×100, no
   ÷100). Unit responsibility lives with the producer
   (`alpha_options.py`).
2. **No silent contract-dollar conversion.** If a producer mistakenly
   passes a contract-dollar value (e.g. 350.0 for a $7-wide vertical),
   the adapter does NOT auto-divide; the unsafe value reaches IBKR
   and IBKR's tick / range validation will reject. The downstream
   IBKR error log is the canonical signal — the adapter's job is to
   trust the per-share contract, not police producer arithmetic.
3. **Skip beats partial.** When no positive finite per-share value is
   derivable, the BAG intent is SKIPPED (no order built, no preview,
   no broker call). The `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE` marker is
   the canonical signal for ops/replay.
4. **No live broker-quote dependency at adapter.** The offline quote
   engine in `chad/options/quote_check.py` is **unwired** at the
   adapter (per its module docstring, lines 19-22). The adapter does
   not consume broker bid/ask/last for BAG; its enforcement is
   structural (producer-derived per-share `net_debit_estimate`).
5. **Non-BAG passthrough is preserved.** `_resolve_bag_lmt_discipline`
   returns non-BAG intents unchanged. The BAG quote-enforcement
   rules MUST NOT impose any transform on STK / FUT / OPT / CASH
   limit prices.

---

## 3. Rejection-path summary

| Input variant                                         | Layer | Outcome                                                 |
| ----------------------------------------------------- | ----- | ------------------------------------------------------- |
| `limit_price=None` + no `meta`                         | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| `limit_price=NaN` + no usable meta                     | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| `limit_price=+inf` / `-inf` + no usable meta            | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| `limit_price=0` / negative + no usable meta             | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| `meta["net_debit_estimate"]=NaN` / inf / 0 / negative   | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| `meta["net_debit_estimate"]="not-a-number"`             | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| `OptionsSpreadSpec.net_debit_estimate=None` / NaN / inf | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| Dict `spread_spec.net_debit_estimate=inf`               | 1     | Skip + `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE`             |
| Bypass Layer 1 with NaN / 0 / negative LMT              | 2     | `ValidationError("Limit order requires positive limit_price")` |
| Bypass Layer 1 with contract-dollar misuse (350.0)      | —     | Accepted as per-share by design; IBKR will reject; producer responsibility |
| Valid positive per-share value (e.g. 3.75)              | 1     | Forwarded verbatim to `intent.limit_price`; no unit transform |
| Non-BAG intent (STK / FUT / OPT / CASH)                 | 1     | Returned unchanged                                      |

---

## 4. Quote-staleness / midpoint wiring

Wiring `chad.options.quote_check` (offline midpoint engine) into the
adapter is **out of scope for Box 052** by design. The wiring is
referenced in the engine's own module docstring as "Tier 3c" and is
tracked as a separate live-readiness item. When it lands:

- Box 052 evidence MUST be updated to reflect the new midpoint gate.
- `test_box052_official_bag_adapter_quote_enforcement.py::test_quote_check_engine_is_unwired_at_adapter_documented_contract`
  will fail by design — that failure is the canonical signal to refresh
  this policy doc and the Box 052 evidence file.

Until then, BAG enforcement remains structural (positive per-share
`net_debit_estimate` derivable, else skip).

---

## 5. Acceptance criteria — Box 052

| Criterion                                                              | Status |
| ---------------------------------------------------------------------- | ------ |
| Adapter rejects NaN / inf / -inf / 0 / negative BAG limit_price         | PASS   |
| Adapter rejects NaN / inf / non-numeric / 0 / negative `net_debit_estimate` | PASS |
| Adapter consumes only broker-facing per-share / per-unit values         | PASS   |
| No silent contract-dollar → per-share conversion at adapter             | PASS (documented; producer responsibility) |
| Defense-in-depth: `_OrderFactory.build` raises on unsafe LMT            | PASS   |
| Non-BAG (STK / FUT / OPT / CASH) limit_price behavior unchanged          | PASS   |
| Quote-engine wiring policy explicit (currently unwired)                 | PASS   |
| Tests cover all rejection paths                                         | PASS (30 new + existing 17 in `test_bag_lmt_discipline.py`) |

---

## 6. Runtime / live invariants

- No `runtime/*.json` mutation.
- No SQLite mutation.
- No order placement.
- No live authorization.
- No `systemctl daemon-reload` / restart / start / stop.
- `chad-live-loop.service` remains `active (running)`.
- `runtime/live_readiness.json` `ready_for_live` remains `false`.
