# PENDING ACTION — MYM `invalid_quantity` Classified EXPECTED_SAFETY_SKIP

- date: 2026-05-26
- prepared_by: audit (read-only; documentation-only Pending Action)
- target_branch: main
- governance: documentation only — no code change, no test change, no config edit, no runtime JSON edit, no service restart, no broker action, no live posture change
- status: PROPOSED (recommendation: keep current behavior, no patch)
- linked evidence: `reports/parity_audit/MYM_INVALID_QUANTITY_AUDIT_20260526T211447Z.md`
- related: `PR-M2K-MYM_bar_provider_futures_mapping_2026-05-26.md` (§8 noted the MYM `invalid_quantity` whole-units rounding as out-of-scope for that patch; this document is the dedicated classification)

---

## 1. Claim under review

> MYM is being rejected every cycle with `intent_skipped_invalid_quantity`. Is this a local code bug that must be fixed before paper-complete?

**Answer: No.** MYM `invalid_quantity` is an **EXPECTED_SAFETY_SKIP**, not a local code defect. The pipeline is doing exactly what the SIZING_FLOOR contract requires for whole-unit instruments at minimum base size.

---

## 2. Mechanical chain (forensic walk-through)

The skip is the deterministic product of three correct subsystems chained together:

1. **Raw strategy size = 1 contract.** MYM is emitted by `alpha_futures` / `gamma_futures` as a single-contract intent at the lowest sizing tier permitted by the strategy. (alpha_futures `_EQUITY_INDEX_FUTURES = frozenset({"MES","MNQ","MYM","M2K"})`; minimum emitted base size = 1 contract.)

2. **Confidence multiplier attenuates 1.0 → 0.9869330069874857.** The dynamic-risk-allocator applies the current paper-epoch confidence factor (≈ 0.9869) as a multiplicative scalar on the raw size. At raw=1.0, the post-attenuation real-valued size is `1.0 × 0.9869330069874857 = 0.9869330069874857`.

3. **Quantity normalizer floors fractional futures to 0 and rejects.** Futures are whole-unit instruments by IBKR contract definition; partial contracts cannot be transmitted. The normalizer floors `0.9869` to `0`, then rejects the intent as `invalid_quantity` because qty<1. The rejection is logged as `intent_skipped_invalid_quantity` and the intent does NOT reach the broker.

Net effect: MYM is filtered out at the sizing-normalization stage, never enters order construction, and produces no fill, no partial fill, no placeholder, no $100-paper-open, no broker round-trip.

---

## 3. Why this is `EXPECTED_SAFETY_SKIP`, not a defect

The behavior is **fail-closed by construction**:

- A fractional futures contract is physically untransmittable. Either it must round to ≥1 (which would silently *increase* exposure beyond the allocator's stated intent) or it must round to 0 (which suppresses the intent). The normalizer chose 0. That is the conservative, safety-preserving direction.
- The reject is captured, classified, and counted. It is observable in `intent_skipped_invalid_quantity` and is therefore not silently dropped.
- No P0-class invariants are violated: no fake fill is written (P0-1), no placeholder pricing escapes (P0-2), no broker truth divergence (P0-3), no live posture drift (P0-4).
- No A-gate is bypassed; this skip occurs at the sizing-normalization stage which sits between gate A-pass and adapter dispatch.
- `ready_for_live = false` and `allow_ibkr_live = false` are preserved across the skip.

This is the same behavior pattern as `same_side_position_open` suppression: an emitted intent that is correctly filtered before contact with the broker, on a deterministic, audit-trail-preserving path.

---

## 4. Secondary observation — documentation gap in SIZING_FLOOR

The SIZING_FLOOR contract (`docs/SIZING_FLOOR.md` and embedded comment lines in the allocator) states (paraphrased): *"the confidence multiplier attenuates position size but never zeroes a position."*

That statement is **true for real-valued (equity-share or dollar-notional) sizing**, where any positive multiplier preserves a positive size. It is **not true for whole-unit futures at base size = 1**, where any multiplier < 1.0 floors to zero by the quantity normalizer's contract.

This is a documentation precision gap, not a code defect. The allocator's invariant ("attenuates but never zeroes") is mechanically correct at the real-number layer; it is the *downstream* whole-unit normalizer that introduces the zero. The composition of the two correct subsystems produces a zero at the integer floor when raw=1 and confidence<1.0.

---

## 5. Why this does NOT block paper-complete

- Paper-complete is defined against the SCR=CONFIDENT band gate, which requires {`sharpe_like`, `effective_trades`, `win_rate`, `total_pnl`} to clear thresholds across the realized fill set. The MYM `invalid_quantity` rejects are excluded from the fill set (they never become fills), so they do not contaminate or degrade the realized-fill statistics.
- The reject count is finite and bounded by emission cadence (one reject per cycle when MYM is emitted with raw=1 and confidence<1.0). It does not pollute the paper P&L, does not affect equity, does not impact ledger truth.
- Other strategies continue to fill normally; 14/15 intents per cycle survive the sizing-normalization stage (see §6 evidence).
- The safety posture (`ready_for_live=false`, `allow_ibkr_live=false`) is preserved by the skip itself — the skip is precisely the mechanism that keeps an untransmittable order out of the broker path.

There is therefore no paper-complete blocker. The skip is, in operational terms, free.

---

## 6. Evidence (read-only audit)

Linked report: `reports/parity_audit/MYM_INVALID_QUANTITY_AUDIT_20260526T211447Z.md`

Headline counters:
- **153 / 153** `intent_skipped_invalid_quantity` rejects in the trailing 24-hour window are MYM. No other symbol contributed any reject in that classifier bucket.
- **14 / 15** intents per cycle survive sizing normalization (the 1 lost per cycle is MYM under the chain in §2; the other 14 reach adapter dispatch where they are evaluated against gates, position conflicts, duplicate-order suppression, etc.).
- Safety posture invariants during the window:
  - `ready_for_live = false`
  - `allow_ibkr_live = false`
  - `allow_ibkr_paper = true`
  - `stop_bus.active = false`
  - `reconciliation_state.status = GREEN`

No new fills attributable to the MYM skip. No P0-1/P0-2/P0-3 violations. No live-posture drift.

---

## 7. Policy options (for record; no action proposed)

Listed for completeness. Each is a forward design choice, NOT a recommended patch under this Pending Action.

1. **Keep current behavior.** MYM continues to skip every cycle while raw=1 and confidence<1.0. Reject count remains the dominant entry in `intent_skipped_invalid_quantity` until the strategy grows base size or confidence returns to 1.0. Zero risk, zero code change, zero test change.

2. **Exempt 1-contract futures from confidence attenuation.** Bypass the confidence multiplier when raw_size==1 and instrument is whole-unit, forcing the floor to remain at 1. *Trade-off:* breaks the SIZING_FLOOR invariant ("confidence attenuates everything uniformly"), introduces a special-case branch in the allocator. Not recommended without an explicit policy decision; would need its own Pending Action with full impact analysis.

3. **Require confidence multiplier before strategy emits base size.** Move the multiplier earlier: have the strategy ask for `max(1, round(target_dollar / contract_value × confidence))` rather than emit raw=1 unconditionally. *Trade-off:* couples the strategy module to the allocator's confidence factor and reverses the current separation of concerns. Significant architectural change.

4. **Allow minimum-one override only when risk cap permits.** Add a `min_qty_floor_one` flag on whole-unit instruments; if the post-attenuation size ∈ (0, 1) AND the per-symbol risk cap permits 1 contract, round up to 1 instead of down to 0. *Trade-off:* increases realized exposure beyond the allocator's stated size; would need risk-cap re-verification per symbol. Asymmetric error: rounding up is the riskier direction.

5. **Strategy-level suppression so MYM does not emit if adjusted size would fall below 1.** Have the strategy peek at the current confidence factor and refuse to emit when `raw × confidence < 1` for whole-unit instruments. *Trade-off:* the cleanest of the "change something" options because it preserves the SIZING_FLOOR invariant downstream and just stops emitting unfillable intents at the source. Minor concern: introduces a strategy-side dependency on the allocator's confidence factor.

---

## 8. Recommendation

**Keep current behavior. Do not patch.**

- The skip is fail-closed and audit-traced; it costs nothing in fills, P&L, or safety posture.
- No paper-complete blocker is introduced.
- Patching to round up (Option 4) is the wrong direction because it asymmetrically increases exposure; patching to bypass attenuation (Option 2) breaks an invariant for a single edge case; the strategy-side suppression (Option 5) is cleaner but still optional.
- Documentation refinement (§4) MAY be added in a separate, small, doc-only PR if desired — clarifying that the "attenuates but never zeroes" invariant applies at the real-number layer, with whole-unit normalization composing a zero floor when raw=1 and confidence<1.0. Not required for paper-complete.
- Test coverage MAY be added to assert the explicit chain (raw=1 × confidence<1.0 ⇒ floored to 0 ⇒ classified `invalid_quantity`), to lock in the behavior and detect any silent rounding regression. Optional; not required for paper-complete.

---

## 9. Out of scope (do NOT bundle into this Pending Action)

- M2K bar-provider futures mapping fix — already addressed in `PR-M2K-MYM_bar_provider_futures_mapping_2026-05-26.md` (VERIFIED).
- alpha_futures / alpha_crypto / delta / gamma_futures edge_decay halts — separate governance Pending Action under `scripts/clear_edge_decay.py`.
- BOX-034 canonical equity divergence (`pnl_state` vs `portfolio_snapshot`) — separate Pending Action.
- Any change to the confidence multiplier value, sizing_factor, or SCR band gating logic.
- Any change to live posture, allow_ibkr_live, or live promotion sequence.

---

## 10. Sign-off

This Pending Action is **documentation-only**. No operator GO required to apply, because nothing is being applied. The artifact's purpose is to record the classification (`EXPECTED_SAFETY_SKIP`), the mechanical chain (§2), the SIZING_FLOOR doc gap (§4), the policy options (§7), and the recommendation to keep current behavior (§8) — so that a future audit reading `intent_skipped_invalid_quantity` for MYM does not re-open this as a defect.

CHAD remains PAPER. `allow_ibkr_live = false`. `ready_for_live = false`. No service restart. No runtime edits. No config edits.
