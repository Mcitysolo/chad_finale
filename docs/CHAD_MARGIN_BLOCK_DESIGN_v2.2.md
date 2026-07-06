# CHAD Margin / Buying-Power BLOCK — Design Document v2.2

**Status:** DESIGN / REVIEW-COMPLETE (3 passes) — ready to commit as build SSOT. Supersedes v2.1.
**Closes:** GAP-003 / RISK-V9-7-04 (July-2 audit: no buying-power/margin pre-trade check)
**Repo:** `chad_finale` @ `09b46c2` | **Date:** 2026-07-04 | **Author:** TEAM CHAD
**Pairs with:** the edge-validation harness (proves *edge*; this prevents *ruin*).

---

## CHANGELOG v2.1 → v2.2 (third-pass fix — load-bearing, no architecture change)

- **Reduce-only ALLOW must register in the ledger (or the remainder is defeated).** v2.1's flow diagram let the reduce-only path ALLOW *without* reserving, while the reducible remainder depends on `pending_reducing_qty`. With reducers unregistered, `pending_reducing_qty` stays 0, the remainder is always the full position, and a reduce-burst (three SELL-100s vs long 100) all pass → over-flatten into a short — the exact bug the remainder was meant to close. **Fix:** the reduce-only ALLOW now **`reserve(order_id, side, symbol, reducing=true, ...)`**, released on fill/cancel/reject like any order, so `pending_reducing_qty` accumulates and the remainder shrinks across a burst. Applied to the 2.2 diagram, §1.3, §4, and the named Phase-B test ("reduce-burst cannot flip position" **must fail if reducers skip the ledger**).
- **Partial-fill reservation semantics stated deliberately (minor, non-blocking).** `release` on fill is ambiguous for partials. Target: **partial fills shrink the reservation by filled qty.** Until that is implemented, the **full reservation persists on partial fill (conservative — may over-block further reduces, never under-blocks).** Stated so a blocked-flatten in shadow is understood, not mis-debugged.

**Three-pass invariant, intact:** no path expands what gets through; exits are never trapped (up to what's actually reducible); both directions of the 17.9× mechanism are structurally closed.

---

## PART 0 — PLAIN ENGLISH

### What this machine is
A **hard stop before every order**: *if this fills, could it over-leverage the account?* If yes, block it — regardless of strategy, sizer, or signal. The last wall before a blown-up account.

### The 17.9× lesson — and its two mirror-images
The audit found CHAD once built a book **17.9× too large with every risk layer in place**. The real mechanism: a **burst of orders each passing an individual check against a stale snapshot**, with nothing counting orders *already in flight*. v2 fixed the entry side with an in-flight ledger; v2.1 found the same bug mirrored on the exit side (a burst of reducers over-flattening into a new opposite position); v2.2 makes the exit fix actually work by **registering reducers in the ledger too**. Every order, increasing or reducing, is judged **net of what's already working**, and a reduce can never carry you past flat.

### Two rules that govern everything
1. **When in doubt, block — EXCEPT to reduce risk, and only up to what's actually reducible.** An *increasing* order we can't confirm safe → block. A *reducing* order is allowed even on stale data — but **only up to the reducible remainder** (last-known position minus reducing orders *already in flight*, which requires reducers to be ledgered). Beyond that it's increasing in disguise → full checks.
2. **Always trust the scarier number.** Margin is computed two independent ways (broker whatIf + our own, on the same basis); the **larger** governs. Never trust a single path — the runaway trusted one path.

### The aggregate cap is the ruin-preventer
A hard ceiling on total book size vs equity (start **2.0×**, tight on purpose), counting filled positions **and** in-flight orders. Ergodicity as a number.

### Rolled out safely
Built + tested **offline** this weekend. Then **SHADOW** (log-only, cached-data, full ledger exercised, zero submit-latency, blocks nothing) until proven. Then ENFORCE in paper. Then, far downstream, live. **This weekend stops at offline build.**

---

## PART 1 — SCOPE & BOUNDARIES

### 1.1 What it IS
A per-order pre-trade gate — every order, every lane — that projects post-fill margin/exposure (counting in-flight orders, both directions) and returns **ALLOW** or **BLOCK(reason)** before the order reaches the broker.

### 1.2 What it is NOT
- Does NOT size, resize, or auto-adjust — blocks the whole order or allows it.
- Does NOT replace the futures gate, loss-guard, or `composite_size_cap` — independent, additive, final backstop; all layers must pass; none trusts another.
- Does NOT modify orders — **order modifications are blocked entirely.** Because a TWS/ib_insync modify is `placeOrder` re-called with a live `orderId`, the gate **detects `orderId`-already-live and BLOCKs it (MODIFY_NOT_ALLOWED)** — enforced in the gate, not just asserted by the chokepoint invariant. Cancels are always allowed (they reduce exposure).
- Does NOT flip `ready_for_live` or change execution mode.
- In SHADOW: blocks nothing (outcome suppressed), but runs the full pipeline including the ledger.

### 1.3 Independence, the pending ledger, and reduce-only bounding
The gate owns a **PendingExposureLedger** (in-memory, gate-authoritative): **every ALLOW reserves — increasing and reducing alike** — via `reserve(order_id, side, symbol, reducing:bool, margin, notional)`, released on fill/cancel/reject. **Reducing orders MUST be reserved (`reducing=true`)**; otherwise `pending_reducing_qty` never accumulates and the reducible-remainder bound is defeated. TTL-expiry is **reconciled against the broker's actual open-orders truth** (never blind-release). On **process restart the ledger rebuilds from broker open-orders before any verdict is served.** The aggregate check counts filled positions + all live reservations + the candidate. **Reduce-only classification is net of pending reducing orders on that symbol** (§4). Margin always uses `max(whatIf_margin, independent_margin_on_same_basis)`.

**Partial fills (deliberate, stated):** target semantics — a partial fill shrinks the reservation by the filled qty. Until implemented, the **full reservation persists on partial fill (conservative — may over-block further reduces, never under-blocks)**; a blocked-flatten from this cause is expected behavior, not a bug.

---

## PART 2 — ARCHITECTURE

```
chad/risk/margin_block.py            # the gate: ALLOW/BLOCK + reason + bounded reduce-only + modify-detect
chad/risk/pending_exposure_ledger.py # reserve(reducing) / release / expiry-reconcile; restart-rebuild; net-of-pending
chad/risk/buying_power_provider.py   # IBKR account fields, cached, freshness TTL, fail-closed sentinels
chad/risk/kraken_bp_provider.py      # minimal crypto notional/balance check (own lane, no cross-margin)
tests/risk/test_margin_block.py
tests/risk/test_pending_exposure_ledger.py
tests/risk/test_buying_power_provider.py
config/margin_block.json             # frozen, schema-validated thresholds (pre-registered)
```

### 2.1 Data needed (IBKR — currently only NetLiquidation is collected)
From `accountSummary()`: `NetLiquidation`, `BuyingPower`, `ExcessLiquidity`, `AvailableFunds`, `FullInitMarginReq`, `FullMaintMarginReq`. Per candidate: whatIf `InitMarginChange`/`MaintMarginChange` **when fresh & available**, always plus the independent margin estimate. USD exposure is converted to the **account base currency (CAD)** using `USDCAD_CONVERSION_CONSTANT` applied **conservatively — biased to overstate USD exposure in CAD terms** (FX fails-closed). The constant is the single FX source; if FX is ever upgraded to a fetched rate, that rate carries its own TTL and the stale-FX-blocks-increasing rule (out of scope here).

### 2.2 Decision flow (per order, before submit)
```
order + cached account fields + open positions + pending ledger
        │
        ├─ orderId already live?  ── YES ──► BLOCK (MODIFY_NOT_ALLOWED)      [enforces the modify ban, sits first]
        │        │ NO
        ├─ REDUCE-ONLY within reducible remainder?
        │     (remainder = last_known_position − pending_reducing_qty on this symbol;
        │      order reduces AND qty ≤ remainder — robust to stale equity, bounded vs burst & stale-flat)
        │        └─ YES ──► ALLOW + ledger.reserve(order_id, side, symbol, reducing=true, margin, notional)
        │                          (relaxed checks, but MUST reserve so pending_reducing_qty accumulates)
        │        │ NO (increasing, or reduce beyond remainder — full checks, fail-closed)
        ├─ account data fresh & complete?      ── NO ──► BLOCK (STALE_OR_MISSING_MARGIN_DATA)
        ├─ margin = max(whatIf, independent-on-same-basis) computable?  ── NO ──► BLOCK (MARGIN_UNCOMPUTABLE)
        ├─ (filled + reserved + this) gross ≤ max(N×equity, current)?   ── NO ──► BLOCK (AGGREGATE_EXPOSURE)
        ├─ projected ExcessLiquidity ≥ floor?                           ── NO ──► BLOCK (EXCESS_LIQUIDITY_FLOOR)
        ├─ projected InitMargin ≤ ceiling% NetLiq?                      ── NO ──► BLOCK (INIT_MARGIN_CEILING)
        ├─ single-order notional ≤ cap% equity?                         ── NO ──► BLOCK (SINGLE_ORDER_CAP)
        │        │ all YES
        └──────► ALLOW + ledger.reserve(order_id, side, symbol, reducing=false, margin, notional)
```
Crypto orders route to `kraken_bp_provider` (notional-vs-balance, no leverage, own lane). Internal exception → ENFORCE: BLOCK (INTERNAL_ERROR); SHADOW: log SHADOW_ERROR, allow (but ledger reserve/release still runs).

### 2.3 Chokepoint (tested, not asserted) + modify enforcement
One function every submit path calls immediately before `placeOrder`. A **CI-invariant test** asserts `placeOrder` is unreachable except through the gate. Because a modify re-uses `placeOrder` with a live `orderId`, the chokepoint alone can't distinguish it — so the **gate's own `orderId`-already-live → BLOCK** rule (2.2, first in the flow so a modify can never masquerade as a reduce) enforces the modify ban; a named test ("same-orderId re-place is BLOCKED") covers it.

---

## PART 3 — PRE-REGISTERED THRESHOLDS (frozen in `config/margin_block.json`)

Conservative starts, committed before enforcement, schema-validated, hash-logged.
- **Aggregate gross exposure ≤ [N]× NetLiq** — ruin cap, counts in-flight. Start **N = 2.0×**.
- **ExcessLiquidity floor ≥ [F]% NetLiq** — start **F = 25%**.
- **InitMargin ceiling ≤ [C]% NetLiq** — start **C = 50%**.
- **Single-order notional ≤ [S]% equity** — start **S = 10%**.
- **Per-asset-class margin rates** (independent estimate): equities Reg-T **~50%**, futures per-contract **[per symbol]**, crypto **100% (unlevered)**. Conservative.
- **Account-data freshness TTL** — start **30s** (stale → treat as missing).
- **FX conservative bias** — USD→CAD overstatement factor applied to the constant. Pre-registered.
- **whatIf pacing** — respect IBKR pacing; cap whatIf calls/sec; on pacing/timeout → independent estimate governs.
- **whatIf-vs-independent log-gap tolerance** — start **20%** (logged only; conservative number always used).

Config schema-validated; **invalid → refuse to start** (fail-closed at startup). **No env-var overrides.**

---

## PART 4 — FAIL-CLOSED vs BOUNDED REDUCE-ONLY (the corrected core)

- **Increasing orders** (or reduce beyond the reducible remainder): fail-closed everywhere — missing/stale data, uncomputable margin, unreadable positions/ledger, internal error → BLOCK.
- **Reducing orders, within the reducible remainder** (`remainder = last_known_position − pending_reducing_qty`): **allowed even on stale data**, AND **reserved in the ledger (`reducing=true`)** so `pending_reducing_qty` accumulates and a burst cannot over-flatten past flat. Reduce-detection is a pure function of last-known position sign + order side + qty, net of pending reducers — robust when equity/margin figures are unavailable; a stale-flat position can't spawn a short.
- **Aggregate cap** is `post-fill ≤ max(N×equity, current_exposure)` so an already-breached book can still be *reduced*, never *increased*.

Adversarially tested: over-leveraged + stale feed + flatten → ALLOW (up to remainder); **reduce-burst → the portion past the remainder BLOCKs (cannot flip position); the test fails if reducers skip the ledger.**

---

## PART 5 — COMPOSITION WITH EXISTING LAYERS

Independent defense-in-depth, downstream of the sizer, orthogonal to `composite_size_cap` / futures gate / loss-guard / SCR. Can only *reduce* what passes, never expand. Never a reason to relax any existing gate. `CHAD_DISABLE_FUTURES_EXECUTION` unchanged.

---

## PART 6 — BLOCK UPSTREAM CONTRACT

On BLOCK: **log the structured marker, consume the signal, apply a cooldown, do NOT hot-retry** the identical order (the `duplicate_blocked` deadlock lesson). A blocked increasing-order does not re-emit until inputs change or cooldown elapses.

---

## PART 7 — OBSERVE → ENFORCE ROLLOUT

1. **Offline build + test** (this weekend) — pure logic + ledger + providers + fixtures; no live path touched.
2. **SHADOW in paper** (later, market-open) — evaluates every real order using **cached data only** (no synchronous broker calls in the submit path), **runs the full pipeline including reserve/release/reconcile** (only the ALLOW/BLOCK outcome suppressed, so a leaked-reservation bug surfaces in shadow logs, not as a false block later), logs `MARGIN_BLOCK_SHADOW` / `SHADOW_TIMEOUT` / `SHADOW_ERROR`, blocks nothing, provably adds no submit latency (tested). Run until it neither false-blocks legitimate orders nor misses over-sized ones. **Caveat: paper margin/whatIf ≠ live — a clean paper shadow proves plumbing, not thresholds.**
3. **ENFORCE in paper** — flip to blocking, paper only, after shadow proves clean.
4. **ENFORCE in live** — only as part of the live-money decision, never before the edge harness passes.

**On any process restart (shadow or enforce), the ledger rebuilds from the broker's open-orders truth before the gate serves a verdict.** This weekend stops at step 1.

---

## PART 8 — BUILD PLAN (phases for `/goal`; each `/local-review`-gated, paused before commit)

- **Phase A — Providers.** `buying_power_provider.py` (IBKR fields, cached, TTL, fail-closed sentinels, additive to the existing collector — does not modify it) + `kraken_bp_provider.py` (crypto balance/notional) + tests. Fully offline (fixtures).
- **Phase B — Ledger + gate (pure logic — the heart).** `pending_exposure_ledger.py` (reserve **with reducing flag** / release / expiry-reconcile, **restart-rebuild from broker truth**, **net-of-pending reducible remainder**) + `margin_block.py` (ALLOW/BLOCK, bounded reduce-only that reserves, modify-detect, max-conservative margin, conservative FX, all thresholds from frozen schema-validated config). No I/O, no live import. **Named adversarial tests:** burst-of-orders cannot exceed the aggregate cap; **"reduce-burst cannot flip position" (fails if reducers skip the ledger)**; over-leveraged+stale+flatten ALLOWs up to remainder; **"same-orderId re-place is BLOCKED"**; every fail-closed path; leaked-reservation reconciliation; **ledger rebuilds from broker truth on restart**.
- **Phase C — Shadow wiring (inert but ledger-live).** Wire the gate as the pre-`placeOrder` chokepoint in **SHADOW only**, cached-data-only, latency-budgeted, exception-isolated, **but exercising reserve/release/reconcile fully** (outcome suppressed). The CI-invariant test (placeOrder only reachable via the gate) lands here, plus a test that SHADOW never changes an order's fate or blows the latency budget. **Enforce-flip is a separate future step, not in this build.**

Phases A-B fully offline. Phase C touches the path but only inert shadow. Each: explicit-path, additive, strict typing, `/local-review` before GO.

---

## PART 9 — WHAT THIS DOES FOR CHAD

CHAD gains the **ruin-prevention** half of the live-money prerequisites. A hard, independent, fail-closed (but exit-safe, and exit-*bounded*) gate that makes the 17.9× runaway **structurally impossible on both the entry and exit sides** — because every order (increasing *and* reducing) is reserved and judged net of what's already in flight, the account's live buying-power reality gets the final word on every increasing order, the scarier of two independent margin estimates always governs, de-risking is never blocked (up to what's actually reducible), and the modify back-door is shut. It composes as pure defense-in-depth, exercises its most stateful component in shadow before it ever blocks, rebuilds its ledger from broker truth on restart, and never touches live money until far downstream. With this and the edge harness — ruin prevented, edge proven — `ready_for_live`'s lock is finally backed by two real machines rather than an assertion.

---

*End of design v2.2. Review-complete (3 passes). Ready to commit as the build SSOT. Phase A begins on operator GO.*
