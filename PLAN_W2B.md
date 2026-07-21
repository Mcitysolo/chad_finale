# PLAN_W2B — Wave 2 Lane B: ContextBuilder Position Awareness (PHASE 1, PLAN ONLY)

**Branch:** `goal/wave2-contextbuilder` (forked from `main` @ `059cfe2`, which already
contains the merged W2A books-cleanup: CAD tiers, UNH re-attribution, ghost-scrub).
**Worktree:** `/home/ubuntu/chad_w2b` — ALL Wave-2B build work happens here.
**Live checkout:** `/home/ubuntu/chad_finale` is on `main` (canonical); the box's
5-min timers execute *that* tree. **No edits to the live tree, and no runtime
mutation, in Phase 1–2.** The build lands DEFAULT-OFF and SHADOW-first; the ON flip
is a Phase-3 operator-gated step and is *conditioned* on D4 below.

## The disease (root cause of three documented defects)

Strategies are position-blind: `ctx.portfolio.positions` is **always `{}`**.
`ContextBuilder.build_async` already accepts `current_positions` (`chad/utils/context_builder.py:367`,
sync wrapper `:550`) and defaults it to `{}` (`:388`); the portfolio the strategies read
is built from it at `_build_portfolio_for_context` (`:840`, `positions=dict(positions)` `:859`).
**No caller anywhere in `chad/` passes `current_positions`** — grep-confirmed. The overlay's
own module docstring states the defect verbatim (`chad/risk/position_exit_overlay.py:8-10`).

**Premise correction (record this): there are FIVE call sites, not one.** The task named
`live_execution_router.py:50` as "the sole caller." It is the live hot path, but four others
also build an empty-book context:
- `chad/core/live_execution_router.py:50` — `ContextBuilder().build().context` (hot path → `_build_available_signals` → the strategy registry)
- `chad/core/full_execution_cycle.py:166`
- `chad/core/routed_execution_runner.py:76`
- `chad/core/ibkr_execution_runner.py:296` and `:402`

Which of these is live must be settled before wiring (D5), or strategies see a populated book
in one runner and an empty book in another.

The three defects this feeds:
- **(a) native strategy exits never ran** — gamma/delta/omega exit branches are gated on `qty>0`, so an empty book makes them structurally unreachable.
- **(b) gamma re-buys what the overlay just sold** (today's UNH churn; receipts in `docs/PFF1_churn_case_for_contextbuilder.md`).
- **(c) beta's 13F logic reads an empty book → perpetual underweight-BUY.**

**Load-bearing finding up front:** the *same single injection* produces both the SAFE
behaviors (gamma/beta stop over-buying) and a DANGEROUS one (gamma/delta/omega start emitting
unclamped exit SELLs that race the ACTIVE exit overlay → oversell/short-flip). You cannot
cherry-pick with the injection alone. This is what makes D4 the crux of the lane.

---

## Baseline — worktree-15 (this branch, measured 2026-07-20) + D8 methodology

Full suite in the worktree (`cd /home/ubuntu/chad_w2b && source /home/ubuntu/chad_finale/venv/bin/activate
&& python3 -m pytest chad/tests/ -q`): **15 failed / 3867 passed / 5 skipped**. This is
exactly the W2A worktree-18 baseline **minus the 3 `test_tier_manager` IDs** that `main`
@ `059cfe2` already fixed. The failing set (record it — it is the D8 baseline):
```
test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes
test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage
test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop
test_kraken_execution.py::test_intent_builder_btc_basic
test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified
test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only
test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only
test_repo_write_guard.py::  (6 tests — hardcodes REPO=/home/ubuntu/chad_finale)
test_routing_gates.py::test_e4_kraken_passive_order_params
test_routing_gates.py::test_e4_kraken_aggressive_order_params
```
All 15 are environment-coupling artifacts (absolute live-tree paths / live posture reads).
**None touch `context_builder.py` or `live_execution_router.py`** (grep-confirmed: those two
files have ZERO dedicated tests today), so none can mask a regression in our area.

**D8 gate (per commit):** re-run the full worktree suite and require
`new_failing_set ⊆ worktree-15` — **no new failing test ID**, no green-by-count. Fast
pre-check: the affected files (new W2B tests + the 5 existing `MarketContext`-constructing
strategy tests: `test_core_pipeline.py`, `test_gamma_asset_class.py`,
`test_delta_abstains_when_no_live_price.py`, `test_phase_a_item2_session_zones.py`,
`test_phase_a_item5_liquidity.py`).

---

## (1) WHERE POSITIONS COME FROM — the authoritative read path + netting rule

There are **three** position legs, and only **one** is independent:

| leg | source | what it is |
|-----|--------|-----------|
| strategy attribution | `position_guard.json` — `gamma\|UNH`, `beta\|MSFT`, … | per-strategy claim, *derived* from the FIFO ledger in paper mode |
| guard broker mirror | `position_guard.json` — `broker_sync\|UNH`, … | a *mirror* of IBKR truth, written into the same file |
| independent broker | `positions_snapshot.json` | a *separate* collector/clientId reading IBKR directly |

**The two guard legs are the SAME shares booked twice.** `gamma|UNH 228` and
`broker_sync|UNH 228` are one 228-share position, not 456. Summing invents a 2× phantom —
stated verbatim in code (`position_guard.py:817`; `position_exit_overlay.py:858`). Live right
now, all three legs agree at UNH=228 and reconciliation is GREEN (`worst_diff=0.0`). The guard
is **multi-booked** in general (MSFT carries `gamma|MSFT` closed + `beta|MSFT` 16 + `broker_sync|MSFT` 34).

**Recommended read path (D1):** inject **broker truth, keyed by SYMBOL, netted to ONE
`Position` per symbol**, sourced primarily from `runtime/positions_snapshot.json`:
- it is the only leg not sharing a source with the strategy legs (so it can't false-flat the way the two guard legs can — the XOV-2345 trap, `position_guard.py:855-858`);
- it natively carries **signed `position`**, **`avgCost`** (→ `avg_price`), `currency`, `secType`;
- freshness-gate on `ts_utc`/`ttl_seconds=300` (mirror `position_guard.py:907-911`);
- cross-check / fallback to the guard `broker_sync|<sym>` mirror (via `_agg_guard_broker_mirror` / `_broker_signed_by_symbol`, `position_guard.py:798` / `position_exit_overlay.py:403`) only when reconciliation is GREEN or the mirror agrees within tolerance; if the snapshot is stale/missing and the mirror disagrees, resolve to **UNKNOWN** (see §2), never to empty.

Netting rule (one signed qty + avg_price + asset_class per symbol):
```
signed_qty  = snapshot.row["position"]     # ALREADY signed — do NOT re-sign
avg_price   = snapshot.row["avgCost"]       # else FIFO wsum/qsum; else None
asset_class = classify(symbol)              # -> chad.types.AssetClass enum  (see D6)
Position(symbol, asset_class, signed_qty, avg_price)
```
Invariants (all load-bearing): **NEVER** `strategy_leg + broker_sync_leg` (2× phantom);
**NEVER** sum the two guard legs against each other; if the signed qty is derived from the
guard mirror instead of the snapshot, sign from `side` via `_signed_qty` (`position_guard.py:546`),
never from the unsigned `quantity`. **Do NOT** read the FIFO's own `broker_sync` queues for
quantity — they diverge (FIFO `broker_sync|UNH=143` vs guard/snapshot 228); use the FIFO
(`_fifo_truth_from_state`, `position_exit_overlay.py:1318`) only for `avg_price`/`opened_at`
enrichment when the snapshot lacks `avgCost`.

There is **no existing helper** that returns a per-symbol strategy-consumable `Position`. The
reconciler/overlay helpers (`load_open_positions`, `_agg_*`, `_broker_signed_by_symbol`) all
serve drift-detection or reduce-only clamping. Wave-2B adds the first one:
`chad/core/context_positions.py :: load_context_positions() -> Mapping[str, Position]` (new,
stdlib + `chad.types`, no side effects, fail-closed).

**The dual-booking is why aggregation is the top plumbing hazard:** the qty each strategy
sees — and therefore whether gamma's 50-unit cap freezes a leg, whether beta thinks it is at
target, whether an exit fires — is entirely determined by this netting. A naïve per-symbol sum
over guard keys double-counts (`gamma|SVXY 156 + broker_sync|SVXY 156 = 312`). D7 (below)
requires a no-over-count proof before the loader is trusted.

### The attribution-vs-broker-total fork (D2 — resolve before wiring)
On `_EFFECTIVE_NON_CHAD_SYMBOLS` = **{AAPL, BAC, CVX, LLY, MSFT, NVDA, PEP, QQQ, SPY}**
(`position_reconciler.py:36-43`, live-resolved) the broker total blends **operator inventory
with CHAD lots**: BAC broker 202 vs CHAD-attributed 26; LLY broker 182 vs CHAD 0; AAPL CHAD
claims 14 but broker holds only 7. Injecting the **full broker total** would make strategies
believe they own operator shares — and a position-aware *exit* (gamma/delta/omega) could then
try to **sell operator inventory**. Two honest options:
- **(b, recommended)** inject the **CHAD-attributed portion only** — `_agg_guard_strategy`
  netted per symbol, reduce-only clamped to broker (`min(|strat|,|broker|)`, broker sign),
  the exit overlay's exact discipline. Operator-only inventory (LLY 182, CHAD 0) is invisible;
  mixed names show only the CHAD share (BAC → 26). Non-excluded symbols are identical either
  way (all three legs agree). This is the option that cannot make a strategy act on operator shares.
- **(a)** inject broker truth but **tag** excluded symbols so strategies treat them as
  do-not-trade — larger, requires a new context surface every strategy must consult.

---

## (2) STALENESS / FAIL-OPEN — UNKNOWN-not-empty (the house rule)

The original disease is *empty book presented as truth*. The fix must never re-introduce it.
`load_context_positions()` therefore distinguishes three outcomes, not two:
- **KNOWN** — snapshot fresh (age ≤ 300 s) and (mirror agrees OR reconciliation GREEN): inject the netted positions.
- **UNKNOWN** — snapshot stale/missing/malformed, or snapshot vs mirror disagree beyond tolerance: **do not fabricate an empty book.**
- **OFF** — the flag is off: today's behavior (empty), which is the *acknowledged* baseline, not a silent regression.

**UNKNOWN semantics (D3 — recommended: idle the cycle).** In ON mode, if positions resolve
to UNKNOWN, `_build_available_signals` returns an **empty available-signals set for that
cycle** (the system-idle path already exists: `live_execution_router.py:83-85` "No strategies
available. System idle."), and writes a loud `CTX_POSITIONS_UNKNOWN` marker + heartbeat. It
does **not** run strategies on a false-empty book. Rationale: feeding empty-as-truth when the
read failed is exactly the disease, now sanctioned; idling one cycle is the fail-closed that
honours the house rule. (Rejected: fall back to empty. Larger alternative: add
`ctx.portfolio.positions_status = "unknown"` and make each of the 5 strategies abstain on it —
correct but a bigger blast radius; deferred.)

Every read path in the reconciler/overlay already fails closed to `{}`/UNKNOWN and never
raises; `load_context_positions()` mirrors that (never raises; returns a typed
`PositionsView(status, positions)`).

---

## (3) BLAST RADIUS — who changes, and the interaction with the ACTIVE overlay

**Exactly 5 of 17 registered strategies read `ctx.portfolio.positions`** (registry
`chad/strategies/__init__.py:141`, iterated `:252`). The other 12 never touch it (ALPHA uses
an in-process state dict; ALPHA_FUTURES/ALPHA_OPTIONS are position-aware but via
`trade_closer_state.json`/guard files, **not** `ctx` — already firing, NOT in this blast radius).

| strategy | reads positions | direction | behavior change when positions appear |
|----------|:---:|-----------|----------------------------------------|
| **BETA** | yes | BUY-only | stops over-buying names at/above target; **never sells** (no SELL leg) — benign |
| **BETA_TREND** | yes | BUY-only | fewer entry BUYs on held names; no sell leg — benign |
| **GAMMA** | yes | BUY + SELL(exits) | large legs (≥50) frozen by cap; re-buy stops (churn fix); **small legs can emit exit SELLs** |
| **DELTA** | yes | BUY + SELL(exits) | exits become reachable; **also adds to held positions (no `qty≤0` guard)** — danger #1 |
| **OMEGA** | yes | BUY + SELL(hedge-off) | can now actually sell real hedge inventory (SH) once it de-hedges |

**Defect (b) — gamma churn — is FIXED, safely (firsthand-confirmed).** `gamma.py:302-305`:
`pos = portfolio.positions.get(symbol); qty = ...; if abs(qty) >= max_position_units (50): return []`;
BUY branch is `if qty <= 0:` (`:386`). Today empty → `qty=0` → cap never bites, BUY branch
always active → re-buy/stack every cycle (the PFF1 UNH re-buy of 228). Inject UNH=228 → cap
fires → gamma returns `[]` before it can buy *or* sell. Residual caveat (danger #4): this stops
*stacking on a live position*, but if a symbol is flattened to 0 by the overlay and gamma's
signal is still long, gamma re-enters at qty=0 — the flatten→rebuy ping-pong persists unless
the signal flips or the injected book still shows residual. Closing that fully is the
"overlay-owns-symbol cooldown" (PFF1 option 1), a D4 follow-on, not the injection itself.

**Defect (c) — beta perpetual underweight-BUY — is FIXED, safely (firsthand-confirmed).**
`_current_position_weights` (`beta.py:181-201`) returns `{}` on an empty book → `current=0` →
`gap = target − 0` always ≥ `underweight_gap` (0.5%) → perpetual BUY. With real weights, names
at/above target fail `gap < underweight_gap` (`:339-340`) → beta stops buying them. Beta has
**no SELL path** (grep-confirmed), so overweight names are merely skipped, never trimmed — no
mass-sell risk. (Note: beta reads `pos.avg_price` as a price fallback at `:197`, so the
injected `avg_price` must be real, not a placeholder — this is why the netting rule carries `avgCost`.)

**Defect (a) — native exits fire — is the DANGEROUS part.** gamma (`:314`), delta (`:610`),
omega hedge-off (`:381`) exits are all `if qty>0`-gated and become reachable the instant
positions appear. `vol_spike_exit` (atr_pct>0.055) and `trend_break_exit` (fast EMA ≤ slow EMA)
fire on the FIRST populated cycle (they don't need warmed in-memory state). Sizing is a small
partial `min(base_size≈5, qty)` — **not full-size, and `TradeSignal` has NO `reduce_only`
field** (`chad/types.py:101-122`).

**The double-exit / oversell interaction (Agent-C confirmed, the crux):**
- The overlay is **ACTIVE** (`chad-live-loop.service.d/97-exit-overlay-active.conf`; heartbeat `mode=active`; today's one WOULD_CLOSE = the UNH-273 event). It runs at `live_loop.py:2276` and submits reduce-only, broker-clamped closes through `apply_close_intents`.
- Strategy-native exit SELLs are ordinary `TradeSignal`s on a **disjoint path**: routing → netting → `_paper_adapter.submit_strategy_trade_intents` (`live_loop.py:3008`), with **no reduce-only clamp**.
- The two paths converge **only** at the adapter's SQLite idempotency key — which **includes `quantity`** (`ibkr_adapter.py:3534-3569`). So overlay-full-close vs strategy-full-close dedup *by coincidence of equal qty*, but **overlay-full (228) vs strategy-partial (5) do NOT dedup** → both fill → 233 sold against a 228 lot → **oversell → FIFO flip to short** (the INCIDENT-0713 shape). The overlay's `_reduce_only_reclamp` reads `broker_sync` truth at `:2276`, *before* the strategy's `:3008` sell exists, and the strategy path has no clamp at all.
- Exposed names: the **small** gamma legs where the exit block is reached (BAC, SPY, AAPL, SH, LLY, QQQ, NVDA, MSFT); large legs are frozen by gamma's cap; DELTA can hit any symbol it scans (SPY/QQQ) and can *also add*; OMEGA can sell SH.

**Conclusion:** turning strategies position-aware is safe for BUY-suppression (defects b, c) but
opens an oversell surface for native exits (defect a) against the live overlay. The rollout
must not flip native exits on until that surface is closed — this is D4.

---

## (4) ROLLOUT DESIGN — tri-state flag, shadow-compare, gated flip

**Flag `CHAD_CTX_POSITIONS`, tri-state `off | shadow | on`, default OFF.** Copy the exit
overlay's `resolve_mode` idiom exactly (`position_exit_overlay.py:222-238`): env overrides
config, a garbage value falls through to the config default (does not fail loud). OFF is
**byte-identical to today** — the injection is implemented at the decision point, config is
never mutated, "the rollback IS the default" (the CHAD_CRYPTO_EXPLORATION zero-residue
doctrine, `regime_activation.py:38-41`).

**Wiring chokepoint (D5):** inside `_build_available_signals` (`live_execution_router.py:43-60`),
the single hot-path funnel to the registry. `current_positions` already exists on `build()`,
so the change is: resolve mode → build the context with (or without) injected positions → run
the registry. Do **not** mutate `ContextBuilder` internals, so its default stays exact when OFF.

**Shadow-compare (mirror the exit-overlay / margin-shadow contract):** in `shadow` mode,
build the context **twice** per cycle — `ctx_off` (empty, today's behavior) and `ctx_on`
(injected) — run the registry over each, and emit **one evidence record per cycle** to
`data/ctx_positions_shadow/ctx_positions_YYYYMMDD.ndjson` (schema `ctx_positions_shadow.v1`,
under `data/` never `runtime/`), containing the signal-set diff:
`{signals_removed[], signals_added[], unchanged_count, by_strategy{}}` where a signal key is
`(strategy, symbol, side)`. **Shadow acts on `ctx_off`** — zero behavior change — exactly the
overlay's `if mode == ACTIVE:` gate applied to *which context feeds downstream*. A heartbeat
(`runtime/ctx_positions_heartbeat.json`) is written in **every** mode incl. OFF so "disabled"
stays distinguishable from "dead."

**The ON-flip go/no-go criterion (empirical, from the shadow corpus):** flip to `on` only when
the shadow diff is **dominated by removed BUYs** (safe suppression) and **`signals_added`
contains no equity/ETF SELL** on the unclamped path — OR every such SELL is neutralized by the
D4 guardrail. This is the whole reason for shadow: it lets us *measure* the oversell surface
before arming it.

**D4 — the double-exit guardrail / ON scope (the crux decision).** Options for the ON flip:
- **(recommended, this lane)** ON uses the injection for **BUY-suppression + beta weight-awareness ONLY**, and a companion filter drops/neutralizes newly-enabled **strategy equity/ETF exit SELLs** at the same chokepoint, leaving the ACTIVE overlay as the **sole equity/ETF exit authority** ("overlay owns the symbol"). Surgical; kills churn (b) and the underweight pump (c) with **zero** new SELL surface. Native strategy exits stay dormant.
- **(follow-on lane)** route strategy exits through `apply_close_intents` with a **lot-keyed** idempotency identity (`position_key+side`, dropping `quantity` from `ibkr_adapter.py:3558-3569`) so full-vs-partial collapse to one clamped close — this is what would *safely* unlock defect (a). Bigger; its own lane.
- **(rejected)** allow native exits to fire as-is → oversell/short-flip vs the ACTIVE overlay.

So Wave-2B *delivers* defects (b) and (c) at the ON flip and *defers* defect (a) to the
exit-routing follow-on. The build (Phase 2) is entirely OFF/shadow and safe; the ON flip is
Phase-3, operator-gated, and blocked on D4's guardrail landing.

---

## (5) TEST PLAN — D8 gate + D7 in-test set-diff + the UNH churn regression

New tests (add the FIRST coverage for both target files):

1. **`test_ctx_positions_loader.py`** — `load_context_positions()`: netting proof (**D7 /
   no-over-count**, the W2A idiom): given a dual-booked guard/snapshot fixture
   (`gamma|SVXY 156` + `broker_sync|SVXY 156` + snapshot 156), assert the loader returns
   `{"SVXY": Position(qty=156)}` — **exactly one leg, never 312**; assert the two aggregates are
   compared, never summed (copy `test_w2a_reattribution_guard_dedup.py:102-107`). Freshness gate:
   stale snapshot → `status=UNKNOWN`, positions `{}`-but-flagged (not KNOWN-empty). asset_class
   classifier: STK→EQUITY vs ETF allow-list (D6).

2. **`test_ctx_positions_flag.py`** — tri-state parser (copy `test_w1a_drift_v4_publisher.py:93-99`
   + `test_position_exit_overlay.py:258-260`): OFF (unset), OFF (`"0"`), `shadow`, `on`, garbage
   → config default. **OFF inertness (D7 set-equality):** run the registry over a context built
   through `_build_available_signals` with the flag OFF and assert the emitted signal-key set is
   **identical** to the pre-change baseline set (the injection is provably inert when OFF).

3. **`test_ctx_positions_shadow.py`** — shadow writes the diff ndjson, acts on `ctx_off`
   (asserts the downstream signal set equals the OFF set → shadow changes nothing), heartbeat in
   all modes, evidence under `data/` not `runtime/`.

4. **`test_ctx_positions_unh_churn.py`** — the **regression fixture** for defect (b). Inline the
   one WOULD_CLOSE record from `docs/PFF1_churn_case_for_contextbuilder.md` (do NOT read the live
   ndjson — that is the non-hermetic coupling D8 warns about). Build the context with the
   `_make_ctx` factory (copy `test_gamma_asset_class.py:60-88`) and a **pinned**
   `max_position_units=50`:
   ```python
   UNH = Position("UNH", AssetClass.EQUITY, 228.0, 424.97)
   off = keyset(gamma_handler(_make_ctx(symbol="UNH", price=424.97, positions={})))
   on  = keyset(gamma_handler(_make_ctx(symbol="UNH", price=424.97, positions={"UNH": UNH})))
   assert (StrategyName.GAMMA, "UNH", SignalSide.BUY) in off      # today: the churn re-buy
   assert (StrategyName.GAMMA, "UNH", SignalSide.BUY) not in on   # fix: cap gate at gamma.py:304 fires
   assert off - on == {(StrategyName.GAMMA, "UNH", SignalSide.BUY)}  # D7 characterized diff
   ```
   (Pin `max_position_units` so the proof is about the injection, not the threshold.)

5. **`test_ctx_positions_blast_radius.py`** — the interaction guardrail. With the D4 filter on:
   inject a **small** held leg (e.g. `AAPL 14`) in a downtrend and assert gamma/delta emit **no
   equity/ETF exit SELL** on the strategy path (overlay stays sole authority); with beta, inject a
   name at target and assert beta stops buying it but emits no SELL.

**Gate for every W2B commit (D8):** full worktree suite `new_failing_set ⊆ worktree-15`; the
new tests use `tmp_path`/inline fixtures (coupling-free). `full_cycle_preview` clean.

---

## DECISION POINTS (please answer before Phase-2 build)

- **D1 — Position source & netting.** Inject **broker truth, netted to ONE `Position` per
  symbol**, snapshot-primary (`positions_snapshot.json`, freshness-gated) with the guard
  `broker_sync` mirror as GREEN-gated cross-check; **never sum legs**. (Recommended.) Alt:
  strategy-attributed `gamma|X` leg as the primary — rejected as primary (shares a source with
  the mirror; XOV-2345 false-flat risk), but see D2 for its role on excluded symbols.
- **D2 — Excluded-symbol policy.** On `{AAPL,BAC,CVX,LLY,MSFT,NVDA,PEP,QQQ,SPY}` inject the
  **CHAD-attributed portion only** (clamped to broker), so operator inventory is invisible and
  no strategy can act on operator shares (recommended) — vs inject the full broker total, vs
  inject-and-tag do-not-trade.
- **D3 — UNKNOWN-not-empty semantics.** In ON mode, resolve unreadable/stale positions to
  **UNKNOWN and idle the cycle** (recommended) — never feed an empty book as truth. Alt (bigger):
  add `ctx.portfolio.positions_status` and make the 5 strategies abstain on UNKNOWN.
- **D4 — Double-exit guardrail / ON scope (the crux).** ON = **BUY-suppression + beta weight
  only**, with a companion filter dropping newly-enabled strategy equity/ETF exit SELLs so the
  ACTIVE overlay stays the sole equity/ETF exit authority (recommended). This ships defects (b)+(c)
  and **defers defect (a)** (native exits) to a follow-on lane that routes strategy exits through
  the shared reduce-only, lot-keyed close path. Confirm this scoping (vs building the exit-routing
  now, vs allowing native exits — rejected: oversell/short-flip vs the live overlay).
- **D5 — Wiring scope.** Wire the flag at `_build_available_signals` (`live_execution_router.py:50`,
  the live hot path) via a shared `load_context_positions()` helper (recommended). Confirm which of
  the 5 call sites are live and whether the other 4 (`full_execution_cycle`, `routed_execution_runner`,
  `ibkr_execution_runner`×2) must be wired in the same commit for consistency, or documented as a follow-on.
- **D6 — asset_class classifier.** Reuse the overlay's `_ETF_SYMBOLS` allow-list + `_asset_class`
  (`position_exit_overlay.py:131-140`, `:340`), mapped from its strings to the `chad.types.AssetClass`
  enum; treat the ETF allow-list as a maintenance surface (any unlisted ETF classifies EQUITY).
  (Recommended.) `secType="STK"` alone cannot distinguish EQUITY from ETF, so a classifier is required.
- **D7 — No-over-count proof.** Require a D7-style set-equality proof (mirroring W2A item-2) that
  `load_context_positions()` returns exactly one netted `Position` per symbol and never sums
  `strategy|X + broker_sync|X` (the SVXY 156+156 → 312 hazard) — as the FIRST build step, before
  the loader is wired. (Recommended.)
- **D8 — Verification methodology.** Adopt the failing-test-ID **SET-diff** gate against the
  recorded **worktree-15** baseline (`⊆`, no new ID), plus D7-style in-test set-equality (OFF
  inert / ON characterized-diff). (Recommended; mirrors W2A.)

## Phase-2 commit order (all `W2B`-prefixed, set-diff green-vs-baseline each)

1. `W2B: context_positions loader + no-over-count proof (D1/D2/D6/D7)` — new `chad/core/context_positions.py`, `test_ctx_positions_loader.py`. No wiring yet.
2. `W2B: CHAD_CTX_POSITIONS tri-state flag + OFF-inert wiring at _build_available_signals (D5)` — resolver + injection branch; OFF proven byte-identical.
3. `W2B: positions shadow-compare + evidence ndjson + heartbeat (D3 UNKNOWN idle)` — `test_ctx_positions_shadow.py`.
4. `W2B: UNH churn regression fixture (D7 characterized diff)` — `test_ctx_positions_unh_churn.py`.
5. `W2B: double-exit guardrail — overlay stays sole equity/ETF exit authority (D4)` + `test_ctx_positions_blast_radius.py`.

The ON flip (`CHAD_CTX_POSITIONS=on` via a unit drop-in) and the defect-(a) exit-routing
follow-on are **Phase-3, operator-gated**, and blocked on the shadow-corpus go/no-go and D4.
Phase 2 is entirely OFF/shadow, worktree-only, no runtime mutation.

---

## STATUS: Phase 1 (plan) complete — awaiting decisions D1–D8 before any build.
Grounded by four parallel read-only forensic sweeps (position-truth, blast-radius, overlay
interaction, test methodology) + a measured worktree-15 baseline. Live tree untouched; nothing
in `runtime/` mutated.

---

## PHASE 2 — CLOSURE RECORD (built 2026-07-21)

All eight decisions locked by the operator ("GO with all recommendations — locked") and built
to the finish line (W2B-1..5). Worktree-only (`/home/ubuntu/chad_w2b`, branch
`goal/wave2-contextbuilder`); the live tree `/home/ubuntu/chad_finale` was untouched and nothing
in `runtime/` was mutated. Everything lands **DEFAULT-OFF**: `CHAD_CTX_POSITIONS` is unset, so
all five commits are byte-identical to today until an operator flips the flag. The rollback IS
the default.

### Decisions → landing commits

| Decision | Resolution (locked) | Landed in |
|----------|---------------------|-----------|
| **D1** position source & netting | broker truth, netted to ONE `Position` per symbol, snapshot-primary (`positions_snapshot.json`, freshness-gated), guard `broker_sync` mirror as GREEN-gated cross-check; **never sum** the dual-booked legs. | W2B-1 (`load_context_positions`) |
| **D2** excluded-symbol policy | inject the **CHAD-attributed portion only**, clamped to broker (`sign(broker)·min(\|strat\|,\|broker\|)`); operator inventory invisible, no strategy can act on operator shares. | W2B-1 (`_clamp_to_broker`) |
| **D3** UNKNOWN-not-empty | unreadable/stale/mirror-conflicting broker truth → UNKNOWN → **idle the cycle**; never feed empty-as-truth. | W2B-2 (`build_cycle_context`→`None`→router/runner idle) + W2B-3 (shadow never reached on a false-empty book) |
| **D4** double-exit guardrail (crux) | ON = **BUY-suppression + beta-weighting ONLY**; the ACTIVE overlay stays the **sole equity/ETF exit authority**; native strategy exits quarantined behind the filter, **deferred** to the exit-routing-unification follow-on lane (their oversell collision is why they don't ride along). | W2B-5 (`filter_overlay_owned_exits` at router + pipeline + full_cycle) |
| **D5** wiring scope | wire the hot path via the shared loader; audit the other 4 sites, wire live, document dead → **3 live wired** (router, `ibkr_execution_runner:407`, `full_execution_cycle`); **2 dead documented** (`ibkr_execution_runner:296` no-caller, `routed_execution_runner` `__main__`-only). | W2B-2 |
| **D6** asset_class classifier | reuse the overlay ETF allow-list (`_asset_class`) → `AssetClass` enum; the allow-list is a **maintenance surface** (unlisted ETF → EQUITY). Sentinel candidate (unknown symbol → warn, not guess) documented in the `context_positions` module docstring, deferred as hardening. | W2B-1 (`_asset_class_enum` + docstring) |
| **D7** no-over-count proof (build-step 1) | set-equality proof that the loader returns exactly one netted `Position` per symbol and never sums `strategy\|X + broker_sync\|X` (the SVXY 156+156→312 hazard); plus OFF-inert / ON-characterized-diff proofs. | W2B-1 (loader proof) + W2B-4 (UNH characterized diff) |
| **D8** verification methodology | failing-test-ID **SET-diff** gate against worktree-15 (⊆, no new ID) + in-test set-equality (OFF inert / ON characterized-diff). | every commit |

### Commits
`9f7d6c9` W2B-1 loader + no-over-count proof · `8cf6932` W2B-2 tri-state flag + OFF-inert wiring
(3 live sites) · `710570d` W2B-3 shadow-compare + evidence ndjson + heartbeat · `bffe157` W2B-4
UNH churn regression fixture · `8b632b3` W2B-5 D4 double-exit guardrail + blast-radius test.

### Gate result (definitive, `chad/tests/`)
**15 failed / 3918 passed / 5 skipped.** The failing set is **exactly** the recorded worktree-15
baseline (9 env-coupling IDs + the 6 `test_repo_write_guard` IDs) — **zero new failing IDs**
across all five commits. Targeted W2B suite **51/51** (loader 14 / flag 18 / shadow 8 / churn 3
/ blast-radius 8). `full_cycle_preview` clean in OFF (default), exit 0. None of the 15 failures
touch any W2B file.

### Load-bearing finding recorded during the build
The **executed** strategy SELLs come from the **pipeline** (`_build_plan_and_intents`,
`live_loop:2503`), NOT the router — the router's `routed_signals` feed `routed_signal_map`
(attribution). So the D4 guardrail is wired at **all three** ON-injection chokepoints, not just
the router the plan named; a router-only filter would have been cosmetic. The filter classifies
from each signal's **own** `asset_class` (D6 symbol classifier as fallback) so a futures/crypto/
options SELL whose *symbol* the equity classifier would misread as EQUITY is never mis-dropped.

### Phase-3 ON-flip go/no-go (gated on the shadow corpus + operator GO)
The ON flip (`CHAD_CTX_POSITIONS=on` via a unit drop-in) and the defect-(a) exit-routing
follow-on remain **Phase-3, operator-gated**. Run `CHAD_CTX_POSITIONS=shadow` first to build the
corpus (`data/ctx_positions_shadow/ctx_positions_YYYYMMDD.ndjson`, schema `ctx_positions_shadow.v1`;
heartbeat `runtime/ctx_positions_heartbeat.json`). Flip to `on` only when the corpus shows:
1. the diff is **dominated by removed BUYs** (safe suppression), and
2. **`added_sells_count == 0`** on the unclamped path across the corpus — OR every such SELL is
   neutralized by the D4 guardrail (which, once ON, drops them and records `exits_filtered` in
   the heartbeat). The guardrail makes (2) hold by construction at ON; the shadow *measures* the
   surface in advance so the flip is judged on evidence.

Until then: `goal/wave2-contextbuilder` is **NOT pushed** — stopped at the push decision; the box
runs `main` unchanged.

### Phase-2 status: COMPLETE.
