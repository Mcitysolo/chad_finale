# BOX-055 — BAG live fill harness policy (offline)

**Status:** Pending Action (paper / governance — no operator-applied change required).
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 055.
**Title:** BAG live fill harness
**Acceptance criterion (verbatim):**
> "offline harness proves placeOrder → Trade → status → fill round trip."

## Purpose

Codify the contract that any future change to the BAG (combo / vertical spread)
order lifecycle must continue to satisfy. This is a paper / governance policy
only — it does NOT authorize live trading and does NOT change live mode. The
offline harness lives at `chad/tests/test_box055_official_bag_live_fill_harness.py`
and proves the round trip without touching IBKR.

## Scope

| Component | Path | Box-055 obligation |
|---|---|---|
| Mock IBKR client | test-local `_FakeIB` | exposes `placeOrder`, `openTrades`, `trades`, `fills`, `sleep` |
| BAG contract | test-local `_FakeContract` | secType="BAG", `comboLegs=[long, short]` with non-zero conIds |
| BAG order | test-local `_FakeOrder` | orderType="LMT", non-null `lmtPrice` (Box-051 normalized) |
| Trade lifecycle | test-local `_FakeTrade` + adapter `_install_trade_status_handler` | PendingSubmit → Filled writes SQLite idempotency row |
| Fill harvest | adapter `_ib_probe` | executions in `ib.fills()` for the matching `orderId` resolve to TERMINAL_AT_BROKER:Filled |
| spread_id preservation | `simulate_bag_paper_fill` + `normalize_paper_fill_evidence` | `spread_id` from intent meta survives into `ev.extra["spread_id"]` on the evidence record |

## Invariants (must hold for every future BAG lifecycle change)

1. **placeOrder is invoked exactly once per BAG submission.** Identity of the
   contract and order is preserved on the returned Trade object.
2. **secType="BAG" survives placeOrder.** No silent downgrade to STK / ETF;
   both ComboLeg objects (long + short) remain attached with their `action`,
   `conId`, and `ratio` intact.
3. **Status transitions are recorded in the SQLite idempotency store.**
   PendingSubmit → Filled writes status=filled and captures
   `broker_order_id`.
4. **A Filled trade is detected by `_ib_probe` via `ib.fills()`.** The probe
   returns `TERMINAL_AT_BROKER:Filled`, which causes
   `_SQLiteIdempotencyStore.claim_or_reclaim` to refuse a second submission
   for the same key (GAP-036 invariant).
5. **`spread_id` survives intent → simulator → evidence.** The strategy stamps
   `spread_id` in meta (alpha_options:511) and the simulator copies it into
   `ev.extra["spread_id"]`. SCR / trade_closer / portfolio engine use this
   key to match BUY opener and SELL closer.
6. **Cancelled status does NOT synthesize a paper fill.** The idempotency row
   goes terminal-negative ("cancelled"); no execution is harvested; the
   record's status is not promoted to `paper_fill`.
7. **Missing BAG meta is rejected loudly.** `simulate_bag_paper_fill` returns
   `False` and annotates `ev.extra["bag_simulator_skipped_reason"]` so the
   rest of `normalize_paper_fill_evidence` can refuse to persist an
   untrusted record.
8. **No real broker connection is used.** The harness builds the adapter
   with `ib_factory=lambda: None`. `adapter._ib` is `None`. No network I/O
   may occur during the test.

## Out of scope for Box 55

- Live order placement against IBKR (gated by Box 56+ and live readiness publisher).
- Multi-leg ratio spreads beyond 1:1 (vertical debit only — Box 053 covers
  bracket / OCA exit policy; Box 054 covers spread_id-aware reconciliation).
- DOM / Level 2 quote feed (Box 056 external blocker).

## Verification command

```
source venv/bin/activate
python3 -m pytest \
  chad/tests/test_box055_official_bag_live_fill_harness.py \
  chad/tests/test_alpha_options_bag_paper_fill.py \
  chad/tests/test_bag_lmt_discipline.py \
  chad/tests/test_box051_official_bag_lmt_unit_normalization.py \
  chad/tests/test_box052_official_bag_adapter_quote_enforcement.py \
  chad/tests/test_box053_official_bag_bracket_oca_or_fail_safe_exit.py \
  chad/tests/test_box054_official_bag_spread_id_aware_reconciliation.py \
  chad/tests/test_gap_036_submit_confirm_lifecycle.py \
  chad/tests/test_ibkr_open_order_guard.py \
  chad/tests/test_paper_fill_unconfirmed_status_gate.py \
  -q
```

Last run: 158 passed in 26.41s (2026-05-21).

## Pending Action

This document is a **policy** only. There is **no runtime config change** to
apply, no service restart required, and no operator approval needed beyond
review. CHAD remains PAPER. Live trading remains NOT authorized.
`ready_for_live=false`. Box 56+ remain open.
