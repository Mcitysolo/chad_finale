# BOX-051 (Official Matrix) — BAG limit-price unit normalization policy

- **Box number (Official Matrix):** 051
- **Box title (Official Matrix):** BAG Tier 3C — limit-price unit normalization — contract-dollar vs per-share BAG price split is built and tested
- **Stage:** Stage 4 — Full live-readiness path (first official Stage-4 box)
- **Cut timestamp (UTC):** 2026-05-20T22:45:43Z
- **HEAD at cut:** `bbe7525`
- **Branch:** `main`

---

## 0. Scope and safety statement

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This policy does not flip
  `ready_for_live`, does not place any order, does not submit any BAG
  order, and does not authorize live trading.
- **No runtime mutation.** No ledgers, fills, trades, broker events,
  systemd, or runtime JSON modified.
- **No production code change.** The per-share-vs-contract-dollar
  split is **already built and tested** in the codebase; this box
  documents the policy and adds one explicit unit-invariant test
  file to lock the contract.

---

## 1. The canonical unit policy (per-share vs contract-dollar)

| Concept                                                                          | Unit                                                                                       |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `OptionsSpreadSpec.net_debit_estimate`                                            | **per-share** debit (e.g. `$3.50` for a $7-wide bull-call vertical)                          |
| `meta["net_debit_estimate"]` (legacy-dict shadow of the typed spec)              | **per-share** debit                                                                         |
| `NormalizedIntent.limit_price` for `sec_type="BAG"`                               | **per-share** debit / credit; passed directly as IBKR BAG order `lmtPrice`                 |
| IBKR BAG order `lmtPrice` field                                                  | **per-share** debit / credit (IBKR API convention for BAG combo orders)                    |
| `OptionsSpreadSpec.max_loss_per_contract`                                         | **contract-dollar** (= per-share max-loss × `_OPTIONS_MULTIPLIER` = 100)                    |
| `meta["max_loss_per_contract"]` (legacy-dict shadow)                              | **contract-dollar**                                                                         |
| `paper_exec_evidence_writer` BAG fill `notional`                                  | **contract-dollar** (= `quantity × per_share_fill × _OPTIONS_MULTIPLIER`)                  |
| `_OPTIONS_MULTIPLIER` (constant in `chad/execution/paper_exec_evidence_writer.py:979`) | **100** — standard equity-option contract multiplier                                  |

**The boundary between per-share and contract-dollar is at the
`paper_exec_evidence_writer` / IBKR-order-build layer.** Everything
upstream (strategy emit, spec stamp, meta dict, adapter hydration,
broker-facing LMT) is in **per-share** units; only `notional` (used
for PnL and reporting) is in contract-dollar units, computed by
explicit `× _OPTIONS_MULTIPLIER` multiplication at one well-known
site.

### 1.1 Producer (strategy) side — per-share

`chad/strategies/alpha_options.py` builds an `OptionsSpreadSpec` and
stamps the per-share `net_debit_estimate` into both:

- the typed spec (`meta["spread_spec"].net_debit_estimate`),
- and the legacy-dict shadow (`meta["net_debit_estimate"]`),

both in per-share units. Sizing decisions use the contract-dollar
form `max_loss_per_contract` (per-share × 100); routing/hydration
uses the per-share form.

### 1.2 Adapter (IBKR) side — per-share preserved exactly

`chad/execution/ibkr_adapter.py::_resolve_bag_lmt_discipline`:

- If `intent.limit_price` is missing or non-positive, hydrate from
  `meta["net_debit_estimate"]` → `meta["spread_spec"].net_debit_estimate`
  (typed) → `meta["spread_spec"]["net_debit_estimate"]` (Mapping).
- **No unit conversion** is performed on the hydrated value — it
  becomes the IBKR BAG order `lmtPrice` as-is, in per-share units.
- If no positive per-share value can be derived, emit
  `BAG_INTENT_SKIPPED_NO_LIMIT_PRICE` and return `None` (caller skips
  submission).

### 1.3 Paper-fill side — contract-dollar boundary

`chad/execution/paper_exec_evidence_writer.py` enforces:

- `fill_price` ← `net_debit_estimate` (per-share)
- `notional` ← `quantity × fill_price × _OPTIONS_MULTIPLIER` (contract-dollar; constant=100)

This is the **only** place a `× 100` is legitimately applied in the
BAG pipeline. Anywhere else, applying a `× 100` would be a unit bug.

---

## 2. Hard rules (anti-misuse)

1. **Adapter trusts the producer.** `_resolve_bag_lmt_discipline` does
   NOT auto-divide contract-dollar values. If a producer mistakenly
   stamps a contract-dollar value (e.g. `350.0` for a $3.50 per-share
   debit) into `meta["net_debit_estimate"]`, the adapter will set
   `intent.limit_price=350.0` per-share and the IBKR BAG order will
   be 100× too high — an obvious bug that IBKR will likely reject but
   the adapter does not pre-validate.
2. **Unit responsibility lies with the producer** (currently
   `alpha_options.py`). Any new BAG-emitting strategy MUST stamp
   `net_debit_estimate` in per-share units.
3. **No silent `× 100` / `÷ 100` outside the paper-fill notional**.
   Adding one would break the per-share LMT contract.
4. **Use `max_loss_per_contract`** for contract-dollar accounting
   (sizing, capital allocation); use `net_debit_estimate` for
   per-share LMT.
5. **STK / FUT / OPT intents are NOT subject to BAG per-share
   semantics.** `_resolve_bag_lmt_discipline` returns non-BAG intents
   unchanged.

---

## 3. Test coverage (proves the split is built)

| Test file                                                                         | Coverage                                                                                    | Pass count |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------- |
| `chad/tests/test_bag_lmt_discipline.py`                                            | 17 tests on `_resolve_bag_lmt_discipline`: non-BAG passthrough, MKT→LMT coercion, per-share hydration from 3 meta sources in priority order, skip on no derivable LMT, no mutation. | 17/17 |
| `chad/tests/test_alpha_options_bag_paper_fill.py`                                  | BAG paper fill: `fill_price ← net_debit_estimate` (per-share); `notional = qty × per_share × 100` (contract-dollar). Multiple tests on the multiplier boundary. | 12/12 |
| `chad/tests/test_alpha_options_bag_no_etf_downgrade.py`                            | `asset_class="options"` preserved for BAG (no STK/ETF downgrade); cross-cutting with per-share semantics. | 22/22 |
| `chad/tests/test_bag_preview.py`                                                  | Preview CLI: spread → preview output preserves per-share `net_debit_estimate`.              | 5/5 |
| `chad/tests/test_options_spread_spec.py`                                          | `OptionsSpreadSpec` round-trip: typed → `to_legacy_meta()` → `from_legacy_meta()` preserves per-share `net_debit_estimate`. | 9/9 |
| `chad/tests/test_alpha_options.py`                                                | Multiple tests, including: `test_bag_limit_price_is_net_debit` (asserts `meta["net_debit_estimate"]` is set per-share), `test_options_routing_sets_limit_price_from_net_debit` (asserts `intent.meta["limit_price"] == 3.25` per-share, NOT 325). | (subset of 51 alpha_options tests) |
| `chad/tests/test_alpha_options_meta_preservation.py`                              | Meta preservation through emit→intent→trade-write boundary.                                  | 12/12 |
| **`chad/tests/test_box051_official_bag_lmt_unit_normalization.py`** (new — added by this box) | 7 explicit unit-invariant tests: per-share-no-unit-conversion (3 hydration paths + explicit limit-price), contract-dollar misuse documented (no auto-÷100), spec typed→legacy_meta per-share preservation, notional × 100 boundary, non-BAG passthrough. | 7/7 |

Total: **154 / 154 passed** after adding the new Box-051 invariant
file. No regression in existing BAG tests.

---

## 4. Why this closes Box 051 as `PASS_ALREADY_NORMALIZED`

Per the Box-051 prompt acceptance criterion: "contract-dollar vs
per-share BAG price split is built and tested."

- **Built:** confirmed — see §1 (per-share producer / per-share
  adapter / contract-dollar notional at the paper-fill boundary
  only; constant `_OPTIONS_MULTIPLIER = 100`).
- **Tested:** confirmed — 147 existing BAG tests across 7 files
  enforce the split implicitly; the new 7-test
  `test_box051_official_bag_lmt_unit_normalization.py` file makes
  the invariant explicit and traceable to this box.
- **Code patched:** **NO production code change.** The existing code
  already correctly implements the policy. Only one test file added.

---

## 5. Patches summary

| Patch class            | Action                                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Production code        | **None** — `alpha_options.py`, `ibkr_adapter.py`, `paper_exec_evidence_writer.py`, `OptionsSpreadSpec` unchanged.        |
| Live config            | **None** — no `runtime/*.json`, SQLite, ledgers, fills, fees, trades, broker events.                                     |
| Documentation          | **Added** `ops/pending_actions/BOX-051_BAG_limit_price_unit_normalization_policy.md` (this file).                        |
| Evidence               | **Added** `runtime/completion_matrix_evidence/BOX-051_OFFICIAL_BAG_Tier_3C_limit_price_unit_normalization.md`.            |
| Tests                  | **Added** `chad/tests/test_box051_official_bag_lmt_unit_normalization.py` (7 unit-invariant tests; 7/7 pass).            |
| Frozen historical SSOT | **Unchanged.**                                                                                                            |
| Staged / committed     | **None.** HEAD invariant `bbe7525`.                                                                                       |

---

## 6. False-closure guardrails

- Does NOT place / submit any BAG order.
- Does NOT modify production code in `alpha_options.py`,
  `ibkr_adapter.py`, `paper_exec_evidence_writer.py`, or `OptionsSpreadSpec`.
- Does NOT change `_OPTIONS_MULTIPLIER`.
- Does NOT add any auto-conversion (`× 100` / `÷ 100`) in the adapter.
- Does NOT change live/paper mode.
- Does NOT authorize live trading.

**live trading not authorized. CHAD remains PAPER. `ready_for_live=false`.**
