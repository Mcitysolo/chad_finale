# CHAD Margin/Buying-Power BLOCK — Phase C WIRED (SHADOW) — 2026-07-10

Status record for the margin-block build ladder (design SSOT:
`docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md`, Part 7 shadow→enforce ladder / Part 8 phases).
This doc records the Phase C landing and pre-registers the G3b enforce-flip criteria. It does
**not** change any threshold or the config `mode` (that remains `shadow` in
`config/margin_block.json`); flipping `mode` is a separate, deliberate authorization commit.

## Phase ladder

- **Phase A — buying-power providers.** DONE (51b370d). `buying_power_provider` (IBKR) +
  `kraken_bp_provider` (crypto): read-only parse/cache/report, TTL + fail-closed sentinels.
- **Phase B — ledger + gate.** DONE (fb46fa6 B1 ledger, 3c80d40 B2 ALLOW/BLOCK gate).
  `pending_exposure_ledger` (reserve/release/expire/rebuild_from_broker) + `margin_block.decide`
  (pure, per-order, fail-closed on increasing orders, reduce-only allowed, modify banned).
- **Phase C — Shadow wiring. WIRED (this record), shadow-only, NOT enforcing.**
  Commits: `G3C U1` (chokepoint wiring + `margin_shadow_gate`), `G3C U2` (integration test).
  - Chokepoint: `chad/execution/ibkr_adapter.py::_submit_intent`, the single narrowest point
    both public submit paths (`submit_routed_signals`, `submit_strategy_trade_intents`)
    converge on. The gate runs AFTER the idempotency key is computed and BEFORE the claim, so
    an enforce BLOCK writes no idempotency row. It is caller-side and synchronous — not inside
    the L1-CLD connection-owner loop, so it never blocks broker I/O.
  - `chad/execution/margin_shadow_gate.py`: evaluates every intent through the pure
    `margin_block.decide`, emits a grep-able `MARGIN_SHADOW` marker (verdict/reason/symbol/
    strategy/est_exposure/headroom/staleness/mode) and appends a verdict row to a dedicated
    evidence ndjson (`data/margin_shadow/margin_shadow_YYYYMMDD.ndjson`, **never** `runtime/`).
  - **Shadow contract — what IS honored:** cached-data-only (no synchronous broker calls in
    the submit path — CI-invariant + latency-budget tests); the ledger *reserve* path is
    exercised each call (fresh per-evaluation ledger, discarded after — no cross-call phantom
    accumulation at a pre-claim chokepoint that has no broker orderId yet); **blocks nothing**
    while `mode==shadow`; **provably side-effect-free on the submission outcome** (tests).
    **Fails OPEN in shadow, loudly** (`MARGIN_GATE_ERROR` + proceed) — shadow can never stop
    trading. The enforce mirror fails **closed**. The evidence append is a synchronous
    best-effort write (sub-ms, latency-budget-tested); a failure is swallowed, never failing
    the order.
  - **Shadow contract — what is NOT yet honored (deferred to G3b, honestly):** the
    *reconcile*-from-broker path and the cross-call *reserve→release* lifecycle do NOT run in
    the default wiring — `open_orders_source` is not yet wired, so `rebuild_from_broker` never
    runs and the ledger `decide()` sees is always EMPTY (flagged `staleness`). Consequences:
    (a) the in-flight/burst accounting that stops a 17.9×-style same-cycle burst is **inert**
    here; (b) the design's "a leaked-reservation bug surfaces in shadow" goal is not delivered
    (a per-evaluation ledger cannot surface a cross-call leak). Both are G3b prerequisites, not
    shadow bugs.
  - Wired at boot in `chad/core/live_loop.py` via `build_default_shadow_gate()` (FAIL-OPEN:
    returns `None` on any config problem → no gate → byte-identical order flow).

## Data-gap finding (blocks meaningful shadow verdicts, and therefore G3b)

Verified 2026-07-10: the account **`ExcessLiquidity`** and **`FullInitMarginReq`** fields the
gate needs are **not published to any read-only runtime snapshot** (`runtime/positions_truth.json`
carries `NetLiquidation` only). So the DEFAULT production snapshot source returns a fail-closed
`BuyingPowerSnapshot`, and shadow will honestly emit **`STALE_OR_MISSING_MARGIN_DATA`** with
`staleness=true` for real IBKR orders. This is shadow working as designed — it surfaces the
missing pipeline rather than fabricating a verdict.

**G3b prerequisite P0:** a read-only runtime account-summary publisher that publishes
`NetLiquidation`, `ExcessLiquidity`, `AvailableFunds`, `FullInitMarginReq`, `FullMaintMarginReq`
(and, for the crypto lane, Kraken free-margin) on a ≤30s TTL, so `margin_shadow_gate`'s default
source can build a *usable* snapshot. Until then, shadow evidence will be dominated by
staleness blocks, which is the honest signal.

Scope note: this wiring covers the **IBKR lane** (equity/etf/futures/forex/options). The crypto
(Kraken) lane's own pre-submit chokepoint is a separate future wiring.

## G3b — enforce-flip criteria (pre-registered NOW, before any flip)

Flipping `config/margin_block.json` `mode` `shadow` → `enforce_paper` requires ALL of:

1. **Data pipeline live.** The G3b P0 account-summary publisher is running and
   `margin_shadow_gate` produces `usable=true` snapshots (no `MARGIN_FIELDS_NOT_PUBLISHED`).
2. **≥ 5 trading days of clean shadow.** ≥5 distinct trading days of shadow evidence in
   `data/margin_shadow/` with **zero erroneous BLOCK verdicts** — i.e., no `would_block=true`
   row for an order that then filled fine at the broker (cross-checked against `data/fills/`),
   and no `staleness=true` on a BLOCK that governed. (A stale/missing-data BLOCK does not count
   toward "clean" — it counts toward criterion 1 not being met.)
3. **Zero `MARGIN_GATE_ERROR` in the window** (or every occurrence root-caused and fixed) — the
   gate never fails open on a real order during the proving window.
4. **No missed over-size.** Manual/synthetic over-exposure probes in shadow produce the
   expected BLOCK verdict (the gate would have caught a real over-size).
5. **Ledger lifecycle + in-flight truth for enforce.** Before enforce, ALL of:
   - a real `open_orders_source` is wired so `rebuild_from_broker` runs and `decide()` counts
     genuine in-flight exposure — the same-cycle **burst protection is inert until this lands**;
   - the persistent-ledger reserve→release-on-fill + rebuild-on-restart lifecycle is wired and
     tested (the shadow per-evaluation ledger is sufficient for shadow observability but enforce
     needs the maintained ledger so a leaked reservation cannot false-block a live order);
   - the ledger reservation key is reconciled: the pre-claim chokepoint keys reservations by the
     idempotency key, while `rebuild_from_broker`/`expire` key by broker orderId — so the
     modify-ban branch (`is_reserved(order_id)`) stays inert until these key spaces agree.
6. **Enforce exposure accuracy.** The gate is placed pre-claim (so a BLOCK writes no idempotency
   row), which is BEFORE `_resolve_bag_lmt_discipline`; for a BAG/combo whose limit price is
   hydrated post-claim, the gate's `price`/`notional` inputs can differ from what reaches the
   broker. Confirm enforce accuracy for combos (or gate combos on a resolved price) before flip.
7. **`/local-review` GO** on the enforce diff, and the flip is a **single deliberate commit**
   (git history is the enforcement-authorization log, per the config `_mode_note`).

Caveat (design Part 7): paper margin/whatIf ≠ live margin. A clean paper shadow proves
**plumbing**, not thresholds; live enforcement remains far downstream and gated separately.

## Deploy status

NOT DEPLOYED. Repo-only. The shadow gate activates at the **next `chad-live-loop` restart**,
shadow-only (blocks nothing). No systemctl, no runtime writes, no config `mode` change.
