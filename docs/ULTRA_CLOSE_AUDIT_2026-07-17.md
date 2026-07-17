# ULTRA-CLOSE — Pre-Flip Surprise Hunt (Exit Overlay ACTIVE)

- **Filed:** 2026-07-17
- **Type:** Adversarial pre-flip audit (U1, read-only) — "find the Monday surprise on Thursday"
- **Subject:** `chad/risk/position_exit_overlay.py` SHADOW → ACTIVE flip, gated on
  `ops/pending_actions/EXIT_AUDIT_equity_roundtrip_close_2026-07-13.md` §4
- **HEAD at audit:** `45a4ec2` · **Evidence window:** 2026-07-13 → 2026-07-17T13:47Z
- **Method:** 4 independent agents (criteria / evidence-chain / adversarial / recon) + lead
  verification. Every claim below is cited to code, live runtime state, or an executed probe.

---

## VERDICT: **FLIP_BLOCKED**

Five independent blockers. Any ONE of them is sufficient; they do not share a root cause, so
fixing one does not retire the others.

| # | Blocker | Class | One-line |
|---|---|---|---|
| **B-1** | **SCR = WARMUP** | Criterion 2, hard | `effective_trades=67` (<100), `sizing_factor=0.1`, `sharpe_like=−15.09`. Objective, current, unarguable. |
| **B-2** | **Seed-lot untrust does not propagate** | Evidence integrity | The flip banks **fabricated PnL as clean evidence** into SCR *and* Stage-2. The first close will almost certainly be a seed lot. |
| **B-3** | **Anchor wipe (live-proven ×2)** | Correctness | `entry_price`/`peak`/`age` are silently destroyed and re-seeded at spot. The stop is computed off a price that was never an entry. |
| **B-4** | **Regime-reduction double-sell** | Safety | Reproduces the INCIDENT-0713 TLT flip/oversell mechanism exactly. No broker clamp on that leg. |
| **B-5** | **Unbounded rejection storm** | Safety | A rejected close re-fires every ~72s **forever**; a filled-but-unconfirmed close re-fires every 900s. The overlay has no feedback channel. |

**The headline correction: the premise of the hunt is inverted.** The feared surprise was
"the flip produces closes but NO evidence." That is **refuted** — every link of the chain
works, end to end, proven by execution. The real surprise is worse: **the flip produces closes
WITH evidence, and the evidence is FALSE.**

---

## A. CRITERIA AUDIT — §4, all seven

**Tally: 3 PASS · 2 FAIL · 1 PROXY · 1 PARTIAL → BLOCKED.**

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | paper + `ibkr_dry_run` | **PASS** | `runtime/execution_environment.json` ts 13:41:52Z ttl 60 (fresh): `exec_mode=paper`, `ibkr_dry_run=true`, `live_enabled=false`. Live-loop PID 688202 env `CHAD_EXECUTION_MODE=paper`; drop-in `chad-live-loop.service.d/20-execution-mode.conf:2`. |
| 2 | SCR ∈ {CONFIDENT, CAUTIOUS} | **FAIL — HARD** | `runtime/scr_state.json` ts 13:34:12Z ttl 180 (**fresh, not stale**): `state=WARMUP`, `effective_trades=67`, `sizing_factor=0.1`, `sharpe_like=−15.09`, `win_rate=0.015`. |
| 3 | Reconciliation GREEN | **PASS** (literal) — 3 caveats | `position_guard_drift.json` v3 ts 13:33:37Z: `qty_mismatch=0`, `drift_count=0`; `reconciliation_state.json status=GREEN`, `worst_diff=0.0`. See caveats below. |
| 4 | ≥5 sessions shadow proof | **FAIL** (purposive) / PASS (literal) | 744 WOULD_CLOSE, **all UNH**, all `close_qty=open_qty=broker_confirmed_qty=273.0`, zero exceptions. But only **2** sessions proposed anything. |
| 5 | Confirmation proof | **SATISFIED BY PROXY** — unsatisfiable in shadow | Bootstrap paradox is real; a proxy exists. See §A.5. |
| 6 | Verification green | **PASS** (known-5 + 1 flake) | 5 failed / 3644 passed / 1 skipped. `full_cycle_preview` exit 0, "No broker calls were made". |
| 7 | Kill-switch + rollback | **PARTIAL FAIL** | Kill-switch verified; **rollback procedure does not exist**. |

### A.3 — Criterion 3 reads GREEN, but three caveats hollow it out

1. **The zero is partly by construction.** `chad/core/position_guard.py:683-704,742-747`:
   operator-excluded symbols are relabelled `mixed_ownership_info` and **excluded from
   `drift_count`**. Live right now: **AAPL guard 14 vs broker 7** — a genuine
   guard-exceeds-broker divergence, reported as "info". Criterion 3's *literal text*
   ("0 qty_mismatch") passes; its *stated intent* ("Fault C's split-brain is resolved — no
   phantom longs") does not.
2. **Single-snapshot self-comparison.** `detect_guard_vs_broker_drift_v2` (`position_guard.py:556`)
   reads ONE mapping — the "broker leg" is the `broker_sync|*` mirror *inside*
   `position_guard.json`. It compares the guard to itself (confirms the EXT1 finding).
   *Mitigation:* UNH=273 is independently triangulated via `positions_snapshot.json`
   (clientId 99, conId 13272, avgCost 425.0066) and reconciliation (clientId 83).
3. **GREEN is ~2 days old, and the window disagrees.** The overlay's own evidence shows real
   guard>broker divergence on **non-excluded** symbols during the shadow window: 07-13 UNH
   399/207, **TLT 1340/700** (the INCIDENT-0713 oversell, visible in shadow), IWM 392/200,
   SVXY 296/156, V 301/154; 07-15 UNH 339/273. **7,626 of 21,670 rows (35%) diverge.**

### A.4 — The 744 rows are ONE decision, not a body of proof

The brief cited "UNH's 142 WOULD_CLOSE rows". **The real count is 744** (385 on 07-16, 359 on
07-17). Verified per-row:

- **Every** WOULD_CLOSE maps to a broker-confirmed position. `close_qty == open_qty ==
  broker_confirmed_qty == 273.0` on all 744, zero exceptions. **Criterion 4's clamp
  requirement is met.**
- But the count is misleading in three ways:
  - **744 rows ≈ ONE decision.** One symbol, one reason (`atr_trailing_stop`), one position.
    ACTIVE is first-fire: the first close (07-16T14:38:34Z @ 434.53) ends it. The other 743
    are moot.
  - **559/744 (75%) are outside RTH** → gated in ACTIVE at `ibkr_adapter.py:2699`. Shadow
    overstates ACTIVE by ~4×.
  - **The phantom defense has NEVER fired.** Zero `SKIP_UNCONFIRMED` in 21,670 rows — and
    evidence writing is unconditional (`position_exit_overlay.py:671-674`), so that is a true
    zero, not a logging gap. **The reduce-only clamp and the phantom guard are unexercised in
    production; both are proven by unit test only.**

**Reading taken: purposive → FAIL.** Literally, "every proposed close maps to broker truth" is
*vacuously true* over the empty set, so the three HOLD-only sessions (07-13/14/15) would count
and the criterion passes. But the clause is headed **"Shadow proof"**: a session that proposes
nothing *tests* nothing. Under the literal reading, five days of an **empty book** would
satisfy the bar — an absurdity, in a PA whose entire thesis is not flipping on thin evidence.
**2 qualifying sessions, not 5.**

### A.5 — Criterion 5: the bootstrap paradox is REAL; a proxy exists

**Is a "demonstrably confirmable close" satisfiable in shadow? No — structurally.** Shadow
records a **mark**, not a fill: schema `exit_overlay.v1` carries `price`/`atr`/`atr_stop`
(decision inputs from `price_cache`) and **no fill price, no submit, no broker round-trip**.
Submission is gated on `MODE_ACTIVE` (`position_exit_overlay.py:660`), so the overlay can never
obtain a fill price in shadow. Read strictly as *"the overlay must demonstrate it"*, **C5
requires the first ACTIVE close as its own proof — unsatisfiable.**

**But the operative text is narrower:** *"Fault C no longer **universally** rejects equity
closes."* *Universally* falls to a single counterexample from **any** path. It exists, in-window
and in Epoch-3 (`data/fills/FILLS_20260713.ndjson`):

- `c1f1242…` TLT SELL **1340 @ 84.03**, `status=Filled`, `reject=false`, 17:30:14Z
- `e6d8242…` TLT SELL **700 @ 83.99**, `status=paper_fill`, `extra.ibkr_exec_id` present
- `4694fbf…` TLT SELL **640 @ 83.99**, broker-confirmed

Real, non-placeholder, non-`error` fill prices against a genuinely broker-held 700 TLT long.
**Fault C is not universal.**

**The PA's Fault-C narrative is also partly wrong — the real mechanism is different.** All 30
July rejected equity SELLs are IWM with
`pnl_untrusted_reason: "placeholder_no_broker_confirmed_fill_price (deviation=66-67%;
placeholder_fill_price=100.0; price_cache=299.99)"`. **Every one is outside RTH** (11:13Z
pre-open; 20:05/20:20/20:24Z post-close); the confirmed TLT sell was 17:30Z **inside** RTH. IWM
held **200** at the broker while selling only 10 — **no phantom involved**. These are
*quote-availability* rejects, not *split-brain* rejects. Both defenses already landed:
**PR-02b `5c5507e`** (2026-05-25) removed the $100 fallback (`position_reconciler.py:461-476` —
abstain if price ≤ 0), and the **RTH gate `8b39730`** (2026-07-11, deployed 07-12), which the
overlay inherits via `apply_close_intents`. Result: **zero rejected SELLs 07-13..07-17.**

**Decisive answer: not satisfiable in shadow; SATISFIED BY PROXY.** The operator must either
accept the TLT proxy explicitly, or rewrite C5. It cannot be met as literally written.

### A.7 — Criterion 7 is self-referential

- **Kill-switch: PASS.** `KILL_SWITCH_ENV="CHAD_POSITION_EXIT_OVERLAY"` (`:96`), `resolve_mode`
  (`:207`). Verified live: env unset → config `shadow`; heartbeat `mode=shadow` fresh.
- **Rollback: FAIL.** grep for rollback/revert across the PA matches **only line 167 — the
  criterion restating itself.** No procedure, no revert commit, no undo path beyond naming the
  env var. The doc cites its own requirement as its satisfaction.
- **Manual oversight: FAIL.** "first 3 live cycles under manual oversight" — no owner, no abort
  threshold, no watch-list defined anywhere.

---

## B. THE FULL EVIDENCE CHAIN — traced by execution, not by reading

**Headline: NO link is broken. The chain works. That is the bad news.**

| # | Link | Status | Citation |
|---|---|---|---|
| 1 | close intent | **OK** | `position_exit_overlay.py:546-556` — real `evaluate_positions` on real `position_guard.json` → `WOULD_CLOSE/hard_stop_loss` qty 273 |
| 2 | → `apply_close_intents` | **OK** | `position_reconciler.py:400`; wired `position_exit_overlay.py:671-678` |
| 3 | → `_close_intent_to_ibkr` | **OK** | `position_reconciler.py:375-390` |
| 4 | → adapter `_submit_intent` | **OK** | `ibkr_adapter.py:2489,2676,3511` — real key `b45f7c38…` |
| 5 | → broker Filled → evidence | **OK (defective)** | `paper_exec_evidence_writer.py:2096` → FILLS + FEES + EXECUTION_METRICS |
| 6 | → FIFO lot consumption | **OK** | `trade_closer.py:688-766` — really consumed the `RECON_ADOPT_UNH` seed lot |
| 7 | → realized PnL | **OK** | `trade_closer.py:192-256` → −11,485.11 (420.71 → 378.64 × 273) |
| 8 | → `FILLS_<date>.ndjson` | **OK** | `paper_exec_evidence_writer.py:854`, `strategy=gamma` |
| 9 | → trade_history | **OK** | `chad-trade-closer.timer` live, every 60s, exit 0. The Jul-13 gap = no closes since the TLT incident, **not** a dead writer |
| 10 | → SCR `effective_trades` | **OK — and that is the bug** | `trade_stats_engine.py:111-115,438,651-703` — `_is_untrusted → False` ⇒ **COUNTS** |
| 11 | → Stage-2 admission | **OK — and that is the bug** | `trade_log_adapter.py:261-294` — `trust_exclusion → None` ⇒ **ADMITTED** |

*Nothing was faked: no real broker order was placed (adapter stubbed to return `Filled`). That
is the single UNPROVEN step, and it is unprovable without the flip.*

### B-2 (BLOCKER) — seed-lot untrust does not propagate

**`seed_lot_scoring` = SCOREABLE. The round-trip counts — and it must not.**

This answers the brief's question in the branch it did not anticipate. The seed lot carries
`pnl_untrusted: true`, `scoring_excluded: true`, `provenance: UNATTRIBUTED_EPOCH3_ACCUMULATION`.
But `to_payload()` (`trade_closer.py:230-256`) emits `tags: ["paper","closed","gamma"]`, **no
`extra` key**, and buries the meta under `payload["meta"]`. SCR reads
`extra = payload.get("extra")` (`trade_stats_engine.py:438`) and **never reads `meta`**. The
`fill_ids` quarantine backstop (`:376-396`) cannot help either: `RECON_ADOPT_*` appears in
**zero** fills files — `get_exclusion_sets()` returned 4,782 ids, **zero** matching. Both the
flag path and the backstop miss.

`excluded_untrusted=321` bites *fill-tagged* rows; it **cannot see meta-tagged FIFO lots**.

**Exposure — 4 seed lots, 810 shares of fabricated cost basis**, all of which inject fake alpha
into SCR and Stage-2 the moment they close: SVXY 156@57.62, UNH 273@420.71, V 181@358.65,
IWM 200@292.95.

> **Plainly, for the operator:** the current positions **will** bank evidence when closed —
> that is precisely the problem. The evidence will be **fabricated**, because the cost basis it
> is scored against was invented by the Epoch-3 rebuild, not paid in the market. The
> "won't-bank-evidence" fear was the *safe* branch. We are in the unsafe one.

### B-3 (BLOCKER) — the anchor wipe, live-proven twice

`_save_anchors` (`:693`) **replaces the state file wholesale** with `result.updated_anchors`,
but `updated_anchors[key]` is only assigned at `:501` — *after* all three skip paths `continue`
out (`SKIP_NON_EQUITY:461`, `SKIP_UNCONFIRMED:468`, `SKIP_NO_DATA:482`). **One cycle where a
position skips → its anchor is erased from disk.** The next cycle re-seeds
`entry_price = peak = trough = current price` (`:488-492`).

`peak` is a monotonic `max()` ratchet (`:490`) — an arithmetic decrease is **impossible** unless
the anchor was destroyed. Two decreases are in the live evidence:

```
WIPE @ 2026-07-14T03:29:05Z: peak 429.82 -> 428.76 | entry reseeded 428.76 | age -> 0.00
WIPE @ 2026-07-15T16:25:17Z: peak 431.67 -> 422.09 | entry reseeded 422.09 | age -> 0.00
```

Both immediately follow a **23:45 evidence gap** (07-13 last row 23:45:24Z → next 03:29:05Z;
07-14 last 23:45:12Z → next 16:25:17Z) — the XOV-2345 daily false-flat signature (fix XOV2-1..4
committed, **undeployed**). Reproduced from a clean fixture:

```
BEFORE: {'gamma|UNH': {'entry_price': 422.09, 'peak': 459.45, ...}}
AFTER : {}          # ONE empty-guard cycle
```

**Consequences, all live right now:**

- `runtime/position_exit_overlay_state.json` currently claims `gamma|UNH entry_price=422.09,
  first_seen_utc=2026-07-15T16:25:17Z`. **Neither is real** — both are artifacts of the 07-15
  wipe. The 8% hard stop (388.32) is computed off a **fabricated** anchor.
- **The guard has no cost basis at all.** `gamma|UNH` = `{open, opened_at, strategy, symbol,
  side, quantity, last_state, source: paper_ledger_rebuild}` — **no price field exists**. So
  `entry_price` is *structurally* unknowable to this overlay; `:488-489` falls back to
  `entry_price = price` (first sight). The hard stop is not a stop-loss in any economic sense:
  a position already deep underwater re-anchors at the depressed price, and the stop sits 8%
  below *that*. Live proof: `gamma|AAPL` anchored today 13:32:42Z at `entry=333.35 =
  peak = trough = spot`.
- **B-4-adjacent: `max_hold` can never fire.** `_age_days` (`:401`) prefers the guard's
  `opened_at` — which `paper_ledger_rebuild` **rewrites**. `gamma|UNH` age climbed to 1.845d
  then reset to **0.0009d** at 13:41. IWM reset on 07-14 and again 07-15. A 20/30-day timer
  cannot survive a ~1–2 day reset cadence. **The ATR trail is the only exit that can fire.**

### B-4 (BLOCKER) — regime-reduction reproduces INCIDENT-0713

Authoritative call order in `run_once()`:

| line | action |
|---|---|
| 1643/1647 | `_rebuild_guard_from_broker()` — **the only writer of `broker_sync\|*`** |
| 2253 | reconciler `apply_close_intents` |
| **2276** | **`_exit_overlay.run_cycle()`** |
| 2332 | regime-reduction `apply_close_intents` |

`broker_sync` is a snapshot taken **~600 lines and three close-paths earlier**, never refreshed
mid-cycle. So `_reduce_only_reclamp` (`:793`) re-reads a guard whose broker leg is **pre-sell**.

**Kill chain:** overlay submits SELL 273 at 2276. Guard mutation requires positive fill
confirmation (`position_reconciler.py:557-577`; `_CONFIRMED_FILL_STATUSES={"filled","paper_fill"}`
at `position_guard.py:33`) — so on `PendingSubmit`, or on the documented
`RECONCILER_CLOSE_ABSTAIN_NO_PRICE` path (`position_reconciler.py:468-476`, *"evidence write
skipped; guard remains open"*), **`gamma|UNH` stays `open=True` though the broker really
filled**. At 2332 `handle_regime_transition` calls `load_open_positions()` fresh, sees 273 still
open, and `generate_partial_close_intents` (`regime_reduction.py:134`) proposes
`int(273*0.5)=136` SELL — **with no broker-truth clamp of any kind**. → **409 sold vs 273 held
= net −136 short.** This is the TLT mechanism exactly.

What saves the *reconciler* leg is luck, not design: it derives `strategy` the same way and
proposes the same 273, so it collides on one idempotency key and is `duplicate_blocked`.
Regime-reduction's 136 is a **different key** — the collision does not save it.

### B-5 (BLOCKER) — no cooldown, no feedback, unbounded storm

`grep -in "cooldown|backoff|dedupe|attempted|last_close|retry|throttle|rate_limit"` over the
overlay → **zero matches**. Measured re-proposal rate: **385/session, ~50/hour, one per 72s
cycle.** The only thing between that and 385 broker orders is the adapter's idempotency store,
which has two holes:

- **S3 terminal-negative** (`ibkr_adapter.py:1047-1049`): a `rejected`/`cancelled`/`error` row is
  **DELETEd and re-INSERTed → retry permitted**. A rejected close re-submits **every cycle,
  forever, unbounded.**
- **S2 terminal-positive TTL** (`:1030-1044`, `terminal_positive_ttl_s` default **900s** at
  `:295`): after a **fill**, the same key reclaims after 15 min. Combined with B-4's
  confirmation gap: **another 273 sold every 15 minutes.**

**The overlay is structurally incapable of backing off**: `apply_close_intents` returns `None`
(`position_reconciler.py:400`) and `_submit_active` (`:665`) discards everything. It cannot
distinguish "blocked as duplicate" from "filled" from "rejected". **It has no feedback channel.**

---

## C. ADVERSARIAL SWEEP — the ACTIVE path, item by item

| Item | Status | Finding |
|---|---|---|
| **Partial fills** | **VULNERABLE** | No partial-fill handling exists anywhere in `ibkr_adapter.py` (`grep partial_filled\|cumQty\|filled_qty` → nothing). Guard mutation (`position_reconciler.py:577-598`) sets `open=False` with **no quantity arithmetic** — all-or-nothing, never writes back 123. The overlay re-proposes **273, not 123**. Safety rests **entirely** on `_rebuild_guard_from_broker` refreshing `broker_sync` before the next evaluation. |
| **Close rejected/timeout** | **VULNERABLE** | Storm, unbounded. See B-5. |
| **RTH gate vs standing WOULD_CLOSE** | **SAFE (gate) / VULNERABLE (burst)** | Gate holds: `_evaluate_rth_gate` (`ibkr_adapter.py:2699`) returns `market_closed` **before** the idempotency claim → no row, and `market_closed` ∈ `_UNCONFIRMED_BROKER_STATUSES` (`live_loop.py:1491`), not in `_STATUS_CANON` → **can never canonicalize to a fill. No phantom close.** But 559/744 WOULD_CLOSE are outside RTH, and blocked submits write **no** idempotency row, so nothing accumulates to dedupe against. At 13:30Z the gate opens and the first cycle fires a live 273 SELL; whether cycle 2 (72s later) fires again depends on B-5's TTL/status race, not on overlay logic. |
| **Idempotency collisions** | **SAFE — do not "fix"** | `_stable_idempotency_payload` (`ibkr_adapter.py:3535-3569`) hashes `{strategy,symbol,sec_type,exchange,currency,side,order_type,quantity,asset_class,limit_price}` and **explicitly excludes timestamps**. It collides **by design**, and the collision is the **only** thing turning 385 proposals into ~1 order. A timestamped key would be catastrophically worse: 385 distinct keys = 385 real orders/session. The hole is expiry (B-5), not collision. |
| **Flip-executor / reconciler collision** | **VULNERABLE** | See B-4. Reconciler leg saved by key collision; **regime-reduction leg is not**. |
| **03:15 restart mid-position** | **SAFE** — *my prior hypothesis was wrong* | Anchors are **RELOADED, not reset**: `_load_anchors` (`:683`) reads persisted state and `first_seen_utc` survives across the 03:15 restarts (live file shows `gamma|UNH` first_seen 07-15T16:25Z persisting through 07-17). A standing close survives the restart. **The real anchor destroyer is B-3's skip-path wipe, which fires at any hour.** |
| **Broker-truth-stale in ACTIVE** | **VULNERABLE — the design claim is half-true** | **The test you would want does not exist.** `_broker_signed_by_symbol` (`:362-385`) reads `quantity`, *deliberately ignores the `open` flag* (`:369-371`), and validates **no timestamp, no TTL, no staleness**. Live: `broker_sync\|TLT open=False qty=0.0 updated=2026-07-13T19:13Z` — **3.7 days stale** (harmless only because qty=0). And `_rebuild_guard_from_broker` returns early on `BrokerTruthUnavailable` (`live_loop.py:1284-1291`) — that `return` exits *the rebuild*, and `run_once` **continues to the overlay at 2276 with a frozen `broker_sync`**. **Precisely:** the `None`/0 case IS safe — `.get(symbol, 0.0)` → `broker_held <= 0` → `SKIP_UNCONFIRMED` (`:466-469`), and the reclamp drops it (`:806-811`). **There is no "unconstrained" branch; missing truth never yields an unclamped close.** The danger is a **stale non-zero**, which the clamp accepts as authoritative. It fails **closed on absent** truth and **open on stale** truth. Existing tests (`test_position_exit_overlay.py:150,159`) cover absent/opposite truth only. |

### C-extra — the WOULD_CLOSE signal is chatter, not a decision

On 07-17 UNH logged **359 WOULD_CLOSE _and_ 323 HOLD** — it oscillates.

```
peak=459.45  atr=14.3297  atr_stop = 459.45 - 2.5*14.3297 = 423.6256
price oscillating 423.62 (WOULD_CLOSE) <-> 427.50 (HOLD)
```

`atr_stop` jumped **434.70 → 423.63** between two consecutive cycles (07-17T02:44→02:45) as ATR
was revised **9.90 → 14.33 (+45%)** when the 07-16 bar (range 420.37–461.62, 41 points) entered
the window. **The trailing stop LOOSENED by $11 as volatility rose, un-firing 744 standing
closes.** Margins as tight as **−0.026**; 9 verdict flips on 07-17; median margin −0.25.
Measured across the window: the stop moved **down** on 4 occasions; `peak` is *not* monotonic
across the file **only** because of the B-3 wipes.

The `peak=459.45` anchor itself is **legitimate** — corroborated by 7 ticks 13:47–14:04Z and the
daily high 461.62, ratcheting in during **pre-market** 10:11–13:47Z on 07-16 (`peak=max(prior,
price)` at `:490` accepts every cached price with no RTH filter, no sanity band vs the daily bar
range, no decay). And the would-be exit (@434.53 vs entry 422.09) was *profitable*.

**But the same position reads WOULD_CLOSE and HOLD hours apart on data revision alone, and
ACTIVE is first-fire. The flip's outcome is decided by which cycle wins the race, not by the
rule.** That is not an exit strategy.

### C-extra — the overlay has no exclusion awareness (contained, but single-layer)

`grep` for any exclusion term in `position_exit_overlay.py` → **empty**. It evaluated
`gamma|AAPL` 4× on 07-17 and `gamma|MSFT` 653× on 07-13/14/15 — both in
`_EFFECTIVE_NON_CHAD_SYMBOLS` (= AAPL, BAC, CVX, LLY, MSFT, NVDA, PEP, QQQ, SPY), each an
**operator-owned pre-existing broker position** (`reconciliation_state.exclusion_policy`:
`{"reason": "pre-existing broker position", "owner": "operator"}`).

**Outcome is SAFE:** in ACTIVE it would log `EXIT_OVERLAY_ACTIVE_CLOSE` for AAPL and then be
silently dropped by the GAP-001 chokepoint (`position_reconciler.py:425`, *"operator-excluded
symbols never receive close intents from ANY caller"*), because `_submit_active` routes through
`apply_close_intents` (*"lazy: reuse, don't fork"*).

**But note what this means:** the phantom-guard **confirms against operator-owned broker
shares** — treating the operator's inventory as evidence that CHAD's position is real (AAPL:
guard says gamma owns 14, broker shows 7, all the operator's → `broker_confirmed_qty=7`,
"confirmed"). Defense is **single-layer and accidental**: refactor or bypass
`apply_close_intents` and CHAD sells the operator's stock. **UNH is not excluded**, so this does
not gate the flip — but the PA's §3 claim that the overlay "honors `_EFFECTIVE_NON_CHAD_SYMBOLS`"
is **false as written**.

### C-extra — the Friday-flip weekend hazard

An RTH-blocked close still writes a FILLS row (`status=market_closed`) and does **not** confirm
the guard → re-fires every ~72s. **Fri 20:00Z → Mon 13:30Z ≈ 3,250 junk rows per tripped
symbol.** Fails safe (trade_closer skips the status; no phantom PnL) but pollutes the very
evidence the criteria are graded on. *A Friday flip is the worst-timed flip available.*

---

## D. FIRST SCOREABLE ROUND-TRIP — arithmetic

Live at 13:42Z, gap to ATR stop:

| position | lot | gap |
|---|---|---|
| **SVXY** | **SEED** | **0.06 ATR / 0.11%** |
| **UNH** | **SEED** | 0.32 ATR / 1.07% |
| **SPY** | **FRESH** | 0.92 ATR / 0.94% |
| IWM | SEED | 1.27 ATR / 1.79% |
| BAC / V / AAPL | FRESH / mixed | 2.16–2.27 ATR |

- **First close: SVXY, ~2026-07-17 (today).** 0.11% adverse against a 1.86% daily ATR range —
  essentially *at* the stop. It is a **SEED lot** ⇒ **the very first banked evidence is
  contaminated** (B-2).
- **First CLEAN round-trip: ~2026-07-20 → 2026-07-23.** SPY needs 0.92 ATR ≈ 0.94% against a
  ~1.02% daily ATR — roughly one daily range unit, expected within 1–4 sessions. Closing SPY's
  30 consumes two clean 15-lots → 2 clean round-trips. (`gamma|V` also yields one clean 9-share
  trip, but only *after* FIFO eats the 181 seed first.) **This is stochastic — a probabilistic
  estimate, not a proven date.**
- **NOT max-hold-driven.** Nominally 08-06 (UNH) / 08-15 (SPY) — but B-3 resets the clock every
  ~1–2 days, so **those dates will never arrive**.

**Answer to the brief's question:** post-flip entry/exit pairs are **not** the only scoreable
round-trips — seed-lot closes score too. That is the defect, not the feature. Until B-2 is
fixed, **the first ~810 shares of "evidence" are fabricated**, and they arrive *first*, ahead of
any clean pair.

---

## E. OTHER DEFECTS (non-blocking, but real)

- **B1 — PA-EP8 misses the actual literal.** `_STATUS_CANON` keys are `"FILLED"`/`"filled"`;
  IBKR emits **`"Filled"`**. 198 of 431 production fills are uncanonicalized, `status_raw` is
  `None` on **all 431**, and `STATUS_CANON_UNMAPPED` is firing live. Not chain-breaking
  (trade_closer lowercases; SCR/Stage-2 read closed trades, which carry no status) — but
  **PA-EP8 is not doing its job**.
- **B2 — PA-EP3 join not threaded on this path.** PA-EP3 exists only at `live_loop.py:3072`.
  `_ReconcilerIntent` has no `idempotency_key`, and `apply_close_intents` builds
  `PaperExecEvidence` without `execution_id` → falls back to a derived hash (`b23f8e2d…`) ≠ the
  adapter's key (`b45f7c38…`). **Slippage / signal_decay join is broken for overlay closes.**
- **Cleared** (attacked, held): attribution resolves to `gamma`, **not** `manual`/`config_default`;
  `_broker_signed_by_symbol` reads only `broker_sync|` keys and the overlay skips those keys
  (`:431`), so the **dual-booked guard does not inflate broker truth**; guard-close accepts
  `filled` → no oversell storm from that path.

---

## F. PA DOC DRIFT (governance-relevant — the PA is the flip contract)

1. **§6 "NOT DEPLOYED" is STALE.** The overlay **is** deployed and running in shadow (heartbeat
   fresh 13:46:24Z; evidence through 13:47:40Z).
2. **§3:131 "honors `_EFFECTIVE_NON_CHAD_SYMBOLS`" — FALSE.** No exclusion term appears in the
   module. Inherited only at the chokepoint (`position_reconciler.py:425`). Mitigated, but
   **single-layer, not the documented double**.
3. **§3:131 says "default OFF"; shipped config is `mode: "shadow"`** (§6 and `:34` say default
   SHADOW). Contradiction.
4. **§3:132 says evidence → `data/exit_overlay_shadow/`; actual is `data/exit_overlay/`.**
5. **§6 cites wiring at `live_loop.py:1933-1953`; actual is `live_loop.py:2264-2278`.**

---

## G. WHAT WOULD MAKE THE FLIP SAFE

Ordered; each is independently necessary.

1. **Fix B-2 first — it is a prerequisite, not a follow-up.** Propagate lot meta into
   `extra`/`tags` in `trade_closer.to_payload()`, **or** pin `RECON_ADOPT_*` into the quarantine
   manifest. Without this the flip **actively corrupts** the evidence base the criteria are
   graded on — the one harm that is not undone by flipping back.
2. **Fix B-3 (anchor wipe).** `_save_anchors` must MERGE, not replace; prune only on a
   *confirmed* close. Independently: the guard carries **no cost basis**, so `entry_price` is
   structurally unknowable — either source a real basis (`positions_snapshot.avgCost` exists:
   UNH 425.0066) or **delete the hard-stop condition** rather than compute it from fiction.
3. **Fix B-4.** Clamp the regime-reduction leg against broker truth, or refresh `broker_sync`
   after each close path.
4. **Fix B-5.** Give the overlay a feedback channel (`apply_close_intents` must return outcomes)
   and a cooldown; stop S3's unconditional retry-on-reject.
5. **Re-grade after SCR exits WARMUP**, with ≥5 sessions carrying proposed closes on **more than
   one symbol**, and with the phantom defense and reduce-only clamp exercised **at least once in
   production**.
6. **Operator ruling on C5** (accept the TLT proxy, or rewrite it) and **write a real rollback
   procedure** for C7.
7. **Do not flip on a Friday** — see the weekend hazard.

---

## H. SURPRISES CAUGHT (the point of the exercise)

1. **The feared failure was the safe branch.** "Closes but no evidence" is refuted; "closes with
   FALSE evidence" is real, and it lands **first** (SVXY, a seed lot, today).
2. **The anchor is fiction, live, right now** — wiped twice, re-seeded at spot; the guard has no
   cost basis to anchor to at all.
3. **`max_hold` can never fire** — the clock resets every 1–2 days. Two of the three exit
   conditions are effectively dead; only the ATR trail can fire, and it coin-flips.
4. **The 744 rows are one decision, 75% of them outside RTH** — the shadow proof is ~4× thinner
   than the count suggests, and the phantom defense has never once fired.
5. **The idempotency "collision" is load-bearing safety, not a bug** — the intuitive fix would
   have been catastrophic.
6. **Criterion 3 reads GREEN by reclassification** — AAPL guard 14 vs broker 7 is relabelled
   "info" and excluded from the count.
7. **The overlay treats the operator's own shares as confirmation** — contained by one
   accidental layer.
8. **A Friday flip is the worst-timed flip available** (~3,250 junk rows per tripped symbol over
   the weekend).

---

*Read-only audit. No tracked file was modified, no runtime state mutated, no order placed. The
one simulated close (§B) ran against stubbed adapters and temp copies.*
