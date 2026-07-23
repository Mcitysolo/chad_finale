# PLAN_W4B — Wave 4 Lane B: Flatten-All (LC6) + Exit-Routing Unification (J16)

**Phase 1 — PLAN ONLY.** No code in this commit. Worktree `chad_w4b`, branch
`goal/wave4-flatten`, base `main@6979b99`. Live tree untouched. Phase 2 begins
only after operator decisions D1–D8.

Method: three parallel read-only audits (flatten/cancel machinery; the ctx-filter
→ overlay seam; live-reality/drill conventions), layered on the six Lane-A audits
(pipeline, overlay internals, conventions, XOV2, DQ). Lane separation honored: this
plan touches NO Lane-A surface (no fuse/loss-control/drawdown/DQ files); the only
signal-path seam named here is the ctx-positions filter, per the lane constraint.

---

## 1. Audited ground truth (premise verification)

| # | Premise / claim | Verdict | Evidence |
|---|---|---|---|
| P1 | "~5 native sell-urges/cycle" | **STALE — real number is ~2/cycle** (mean 2.07, peak 6): two *standing* urges, `gamma\|MSFT` (operator-excluded) + `gamma\|PSQ` (actionable), firing essentially every cycle ≈ 2,400 urges/day; 100% at the router site; `site=pipeline` fired 0× in retained journal; site 3 (`full_execution_cycle.py:227`) drops **silently** (result assigned, never used) | journal 299 lines/6.1h; shadow corpus 1,864 records, 6 distinct tuples ever |
| P2 | Information discarded today | CONFIRMED total: dropped urges are discarded objects; only the integer `exits_filtered` reaches `ctx_positions_heartbeat.json`. The shadow corpus (`data/ctx_positions_shadow/`, schema `ctx_positions_shadow.v1`) stopped at the ON-flip (2026-07-22 14:07Z) and captured only `(strategy,symbol,side)` — no qty/confidence/reason | `live_execution_router.py:76-105`; `ctx_positions_shadow.py:213` |
| P3 | Filter scope | Cruder than "native exit urges": `is_overlay_owned_exit` drops **every** equity/ETF SELL unconditionally — long exits, would-be short entries, flips, protectives, SELLs on symbols not even held. Never reads `meta`. BUYs and futures/crypto/options/forex SELLs pass | `context_positions.py:394-441` |
| P4 | Live modes | `CHAD_CTX_POSITIONS=on` (flip landed 2026-07-22 14:08Z, drop-in misleadingly named `98-ctx-positions-shadow.conf`); overlay `active`; crypto overlay `shadow` (env unset); RTH gate default-ON (env unset); futures-disable trio `=1/=1/=0` **on chad-live-loop only** (per-process env — a standalone CLI does NOT inherit it); Kraken `paper_kraken`; loss-guard enforce **=1** | systemd drop-ins + `/proc/<pid>/environ` |
| P5 | Overlay plug-in point | `evaluate_positions` takes injectable side-channel maps (fifo_truth, excluded_symbols, price_meta) — an advice map is the established shape; a new fired reason auto-propagates into markers, `exit_overlay.v2` evidence, and close intents, inheriting the B5 retry governor, `_reduce_only_reclamp`, and all exclusion backstops | `position_exit_overlay.py:602-811,958-1027` |
| P6 | Operator exclusions | SSOT = `config/reconciliation_exclusions.json` → `_EFFECTIVE_NON_CHAD_SYMBOLS` = {AAPL, MSFT, NVDA, BAC, CVX, LLY, PEP, QQQ, SPY}; honored at reconciler generation, the `apply_close_intents` chokepoint (GAP-001 P48 — "never receive close intents from ANY caller"), and overlay `SKIP_EXCLUDED` (first gate). Ctx filter itself does NOT consult exclusions | `position_reconciler.py:31-53,420-435` |
| P7 | "quarterly per SSOT" drill | **NOT FOUND — unverifiable.** No SSOT text mandates a quarterly flatten drill. What exists: chaos-drill re-drill carried with *no cadence* (SSOT v9.8:446), and a **30-day STOP-drill** in `docs/PHASE9_PROMOTION_CHECKLIST.md:24`. Plan cites honestly; cadence is a governance proposal (D8), not a fact | grep both trees |
| P8 | Cancel machinery | **No adapter-level cancel primitive exists** (no cancelOrder/reqGlobalCancel anywhere in production). Only bulk precedent: `paper_position_closer.py:359-367` — clientId-scoped `openOrders()` loop. The W1A "reaper" cancels nothing at the broker (idempotency-ledger rows only). Cross-client enumeration MUST be `reqAllOpenOrdersAsync` (clientId-scoped variant returns false-negative 0/0 — 2026-05-27 lore); SSOT v9.7:203,332 prescribes a read-only re-probe **immediately before any actual flatten** | flatten audit §1 |
| P9 | Flatten precedents | `scripts/flatten_futures_oneshot.py` = the real broker-flatten precedent: preview table → typed `FLATTEN` token → broker-truth-derived closing orders (side from position sign, never hardcoded) → bounded terminal-status wait → post-verify FLAT/INCOMPLETE; direct ib_async, clientId 7715, deliberately bypasses all CHAD gates. The micro EOD flatten writes `eod_flatten_intents.json` that **nothing consumes** (dormant, and guard-sized not broker-sized — an anti-pattern for LC6) | flatten audit §2 |
| P10 | Book reality (sizing the problem) | 11 IBKR positions, all STK (no FUT/crypto on IBKR book); 5 of 11 symbols are operator-excluded; guard has 10 strategy legs + 11 `broker_sync` mirrors and is **dual-booked with splits** (broker AAPL=7 vs guard gamma AAPL=14 — never sum legs, always clamp). Adapter exec-state holds 26 non-terminal rows that are demonstrably stale (Submitted IEMG/VWO Jul-20, never became positions) — cancel-phase truth is the broker probe, never the DB. Kraken paper: FIFO lot book in `exec_state_paper.sqlite3` IS the lane's truth; **no resting orders exist in paper** — "cancel all" is a structural no-op there | live runtime reads |
| P11 | Gates on an out-of-band CLI | Adapter-level only: margin gate (None unless wired), RTH gate (per-process env, default ON, NO exit carve-out → after-hours closes block `market_closed`), idempotency (shared sqlite with live-loop — free dedupe, benign `duplicate_blocked`), futures hard gate (per-process env — unset in a fresh CLI), open-order guard (clientId-scoped). **NOT consulted in the adapter: stop_bus, SCR, LiveGate, operator intent** — an emergency CLI can flatten during a stop_bus halt | flatten audit §6 |
| P12 | Post-flatten hold | stop_bus and STOP-state are the WRONG hold tools — both freeze the close machinery too (stop_bus early-returns the whole cycle incl. both overlays; STOP ⇒ LiveGate DENY_ALL with `allow_exits_only=False`). The right tool exists: `operator_intent EXIT_ONLY` (entries frozen at LiveGate, exits/overlays alive, TTL-bounded). Also: stop_bus auto-clears on clean streak — not durable anyway | flatten audit §7 |
| P13 | Typed-confirm patterns | Six tokens in production. Strongest stack = reaper (dry-run default + `--execute` + exact `--confirm TOKEN` + gates imported not re-implemented + archive-before-mutate) + flatten_oneshot (interactive preview → anomaly warning → typed token). Anti-patterns to avoid: epoch_reset's silent degrade-to-dry-run on bad token; DRIFT-RECON's `WARMUP` acceptance (the code-shaped hole INCIDENT-0713 went through) | flatten audit §5 |
| P14 | Channel policy | `.claude/hooks/chad-order-guard.sh` blocks agent shells from even *naming* flatten entrypoints — flatten tools are operator-terminal-only by standing policy. The new CLI must be added to that blocklist | live tree hook |
| P15 | Report conventions | Drill proof → `reports/ratification/PROOF_<DOMAIN>_<YYYYMMDD>.json` (chaos-drill proof lives there); operational reports → `reports/<topic>_<UTCstamp>.json` with explicit `REPORT_SCHEMA`; archive manifest + SHA256SUMS posture from epoch_reset. Coach narration: no staged-op precedent exists; burst sends need distinct dedupe keys (shared key eats messages 2-5 for 15 min); coach cards max 6 lines, unknown kinds fall through safely to the generic template | live-reality audit §5-6 |

## 2. Item 1 — LC6 EMERGENCY FLATTEN-ALL

### 2.1 Shape

`scripts/flatten_all.py` — standalone operator-terminal CLI (added to the
order-guard blocklist, P14). One command, five phases, both lanes. Dedicated
clientId registered in `chad/execution/ibkr_client_ids.py` (new `FLATTEN_ALL`
constant; 7715 stays reserved for the futures oneshot). Dry-run (**drill**) is the
default; execution requires `--execute --confirm FLATTEN-ALL` **and** an
interactive re-typed token; a wrong token is a **hard error exit 2** — never the
epoch_reset silent degrade (P13).

**Emergency gate set (D4):** minimal and deliberate — `CHAD_EXECUTION_MODE ∈
{paper, dry_run}` (fail-closed, read from the environment + `live_readiness`
posture cross-check) + typed confirm. Deliberately NOT gated on SCR state or
reconciliation status: emergencies are precisely when those are broken, and the
DRIFT-RECON WARMUP-acceptance hole (P13) shows "green gates" are not what makes a
mutation safe — the typed token and broker-truth clamps are. A future live-mode
flatten is a separate authorization with its own token (out of W4B scope).

### 2.2 Phases

**Phase 0 — PROBE (read-only, fail-closed).**
- IBKR: fresh single-threaded MainThread connect (own clientId, own event loop —
  L1-CLD constraint: never mix threads/loops); `reqAllOpenOrdersAsync()`
  (cross-client — the SSOT v9.7-prescribed pre-flatten re-probe, P8) +
  `ib.positions()` behind `broker_position_sync` semantics: unreachable/dead ⇒
  `BrokerTruthUnavailable` ⇒ **ABORT LOUD** — unknown is never flat (XOV2).
- Cross-check vs `runtime/positions_snapshot.json` (two-source confirm,
  reattribute-UNH precedent); divergence ⇒ warn in preview, live probe wins.
- Kraken paper: open lots from `RoundTripBook` (`exec_state_paper.sqlite3`) —
  the paper lane's broker truth (P10). Resting orders: assert none (structural).
- Preview table (flatten_oneshot pattern): every open order to cancel, every
  position with derived close order, scope/exclusion annotations, expected-count
  anomaly warning. Preview is printed in drill AND execute modes.

**Phase 1 — CANCEL (IBKR only).** Per D3: `reqGlobalCancel()` (the emergency
semantics — kills every working order across all clientIds, including manual
ones; that is the point and is documented in the preview), then poll
`reqAllOpenOrdersAsync()` to zero with bounded wait; residual open orders ⇒
report `CANCEL_INCOMPLETE` with the survivors. The 26 stale adapter-DB rows are
ignored as truth (P10) — broker probe is the only cancel inventory.

**Phase 2 — CLOSE (reduce-only, broker-clamped — INCIDENT-0713 defenses).**
- Scope per D1 (default `--scope chad`): close CHAD-attributed strategy legs,
  each **clamped to the broker net for that symbol** — `close_qty =
  min(guard_leg_qty_sum_never… )` — concretely: `min(chad_attributed_qty,
  broker_held_same_side)`, side derived from the broker position sign, skip if
  broker flat (no order minted from ledger belief, ever). `--scope broker-all`
  (everything incl. operator-excluded symbols) requires the second token
  `INCLUDE-EXCLUDED` (D1) — without it, excluded symbols are untouchable here
  exactly as everywhere else.
- Submission path per D2 (recommend **through `IbkrAdapter`**): keeps the paper
  books coherent — fills flow to evidence → harvester → trade_closer → guard, so
  flatten does not manufacture a guard/broker split-brain that would need manual
  `close_guard_entry` cleanup afterward. The CLI process env is set explicitly:
  `CHAD_RTH_GATE=0` for this process only (emergency close must work after hours;
  paper fills permitting), futures trio left unset (per-process, P4/P11 — so a
  future futures position IS closable here, unlike via the live loop). Adapter
  constructed with `margin_gate=None`, `dry_run` bound to drill-vs-execute.
  `duplicate_blocked` results are classified benign-idempotent in the report.
- Kraken paper: reduce-only close intents through the crypto-overlay reclamp
  pattern (`reduce_only_reclamp_crypto` against a fresh book read immediately
  before dispatch — the book FLIPS residuals, so clamp-at-dispatch is mandatory)
  → `TrustedFillEngine.process_intent`; deterministic no-timestamp idempotency
  keys (`flatten|<pair>|<side>|<qty>` mirroring `crypto_exit|` keys).

**Phase 3 — CONFIRM (broker-verified + SLA).**
- Per-order timestamped trace (paper_position_closer pattern): `placed_at` →
  first `orderStatus` (**ack**) → terminal status; emits `ack_ms` and `fill_ms`
  per order + p50/p95 rollups — the first true per-order SLA measurement in the
  system (execution_quality's `ack_latency_ms_p95` is a healthcheck mirror, not
  per-order). Bounded terminal wait (default 60s/order, config).
- **Residual check**: re-probe positions (fail-closed again) → per-symbol verdict
  `FLAT` / `RESIDUAL(qty)` / `EXCLUDED_UNTOUCHED`; overall `FLAT | INCOMPLETE`.
  Kraken: re-read book, open lots must be 0 (scoped).
- Idempotency of the whole command: re-running re-probes and derives closes only
  for residuals; already-flat symbols skip; adapter fingerprints dedupe
  accidental double-fires within TTL.

**Phase 4 — REPORT + NARRATION.**
- Report: `reports/flatten_all_<YYYYMMDDTHHMMSSZ>.json`, `REPORT_SCHEMA =
  "flatten_all_report.v1"` — probe snapshot (+hash), cancel inventory/outcomes,
  close orders with clamps (requested vs clamped vs filled), SLA table, residual
  verdicts, gate/token evidence, scope + excluded symbols untouched list.
  Evidence ndjson: `data/flatten_all/flatten_all_<date>.ndjson` (append-only,
  per-event). No `runtime/` writes by the tool itself.
- Coach narration (staged, distinct dedupe keys per stage — P15):
  `flatten_started` → `flatten_cancel_done` → `flatten_closes_submitted` →
  `flatten_confirmed` (or `flatten_residuals`), each via `format_alert` generic
  kinds (≤6-line cards), severity critical, value-free titles, numbers in facts.
  Registered `_tpl_flatten_*` templates are a cosmetic follow-up.
- Optional post-flatten hold (D5): `--hold` (default ON) sets
  `operator_intent EXIT_ONLY` with TTL 24h — entries frozen at LiveGate, exits
  and overlays stay alive (P12). Explicitly NOT stop_bus/STOP.

### 2.3 The DRILL

Drill = the default no-`--execute` invocation, hardened into a provable artifact:
- Runs Phase 0 fully (real read-only probes), builds the complete cancel + close
  inventories, then pushes every IBKR close through the adapter with
  `dry_run=True` — exercising the full chain (intent build → idempotency payload
  → gate traversal → `dry_run` short-circuit at the placeOrder boundary), and
  builds (never dispatches) the Kraken close intents.
- Proves per stage: connectivity, cross-client order visibility (count vs
  clientId-scoped count — catches the 0/0 false-negative class), position parity
  vs snapshot, clamp math on the real book, adapter chain reachability
  (`status=dry_run` per intent), narration dry-send (formatted, not sent, or sent
  to a drill topic per operator preference).
- Artifact: `reports/ratification/PROOF_FLATTEN_DRILL_<YYYYMMDD>.json`
  (`flatten_drill_proof.v1`) — house home for drill proofs (P15) — including a
  `drill_gaps[]` list (anything unreachable/blocked) and SLA of the probes.
- The operator fires the real drill from the terminal (order-guard channel, P14).
  Cadence: **no SSOT mandate exists** (P7) — D8 proposes filing a backlog SSOT
  amendment (quarterly flatten-drill + alignment with the 30-day STOP drill)
  rather than inventing a mandate in code.

## 3. Item 2 — J16 EXIT-ROUTING UNIFICATION

### 3.1 Principle

The D4 filter stays — one sell authority (the overlay) is preserved. What changes:
a dropped urge stops being *discarded* and becomes *advice* — a recorded, typed,
TTL-bounded input the overlay evaluates under its own rules. Zero information
discarded; zero new sell authority created.

### 3.2 Capture (the named filter seam — the only signal-path change)

New module `chad/core/exit_advice.py`:
- `record_dropped_urges(dropped, *, site, view, paths…)` called at the three
  filter call sites. Router site (where 100% of live drops occur, P1) records the
  full `TradeSignal`: strategy, symbol, side, size, confidence, asset_class,
  `meta.reason` (`trend_break_exit`/`atr_trail_exit`/…), plus enrichment
  available in scope: `position_open` (from the injected `PositionsView`),
  `excluded` (from the exclusion SSOT). Pipeline site records the collapsed
  `RoutedSignal` shape. Site 3 additionally gets the missing parity log line.
- Rows: `exit_advice.v1` → `data/exit_advice/exit_advice_YYYYMMDD.ndjson`
  (append-only, under `data/` never `runtime/` — write-guard + house rule);
  queue summary + heartbeat → `runtime/exit_advice_state.json`
  (`exit_advice_state.v1`, ts/ttl, written every cycle incl. OFF — heartbeat
  doctrine). All paths injectable; factory raises under `PYTEST_CURRENT_TEST`
  without explicit paths (overlay leak-guard pattern).
- Flag `CHAD_EXIT_ADVICE`: `off | record | consume`, default **off**, garbage→off.
  `off` = today's behavior byte-identical (blast-radius tests extend the existing
  OFF/shadow list-equality pins).

### 3.3 Consumption (overlay side, under overlay law)

- Aggregator builds `advice_by_symbol` from today's rows: advising strategies,
  freshness, max confidence, reasons — **excluding** excluded-symbol advice
  (recorded with `excluded=true` for observability, never consumable; overlay
  `SKIP_EXCLUDED` remains the structural backstop above it, and
  `apply_close_intents` the chokepoint backstop below — three layers unchanged).
- Injected into `evaluate_positions` as a side-channel map (P5). New rule
  evaluated AFTER the existing kernel (hard_stop → atr_trail → max_hold keep
  precedence): fired reason `strategy_advice` per the D6 policy. Recommended v1
  policy: advice counts only when (a) fresh (TTL: 2 overlay cycles), (b) from the
  strategy that owns the position leg, (c) the position is broker-confirmed open.
  Rationale: this restores exactly the pre-W2B native-exit behavior — but now
  clamped reduce-only, exclusion-guarded, retry-governed, and evidenced.
  Alternatives (trail-tightening only; N-strategy quorum) in D6.
- **Shadow-comparable first (mandatory):** in `record` mode the overlay computes
  advice verdicts as `ADVICE_WOULD_CLOSE` evidence (new marker + `exit_overlay.v2`
  rows with `verdict=WOULD_CLOSE, reason=strategy_advice`) without acting — the
  overlay is ACTIVE live, so the advice rule carries its own would-only stage
  inside the active overlay. Flip to `consume` is a separate operator GO judged
  on a pre-registered criterion (W2B methodology): e.g. N days of corpus, every
  `ADVICE_WOULD_CLOSE` reviewed, zero excluded-symbol consumptions, clamp always
  ≤ broker. gamma|PSQ is the expected first live case; gamma|MSFT must appear
  ONLY as `excluded=true` records.
- Advice-fired closes inherit automatically (P5): B5 submit governor,
  `_reduce_only_reclamp`, RTH/idempotency at the adapter, evidence, heartbeat.
- Config: `config/exit_advice.json` (D7) — TTL, policy knobs, per-strategy
  enable. The frozen `position_exit_overlay.json` is NOT amended (its header
  declares thresholds frozen, sole expected change `mode`).

### 3.4 What J16 deliberately does not do

No second sell path (advice never reaches the adapter except through the overlay);
no BUY-side advice (out of scope; recorded rows keep side for later analysis); no
short-entry resurrection (a dropped SELL that was a would-be short entry records
`position_open=false` and can never fire the advice rule — closes require an open
broker-confirmed leg); no change to the filter's drop behavior in any mode.

## 4. Interaction analysis

| Surface | LC6 flatten | J16 advice |
|---|---|---|
| Exit overlay | Runs independently; overlay may submit closes concurrently — adapter idempotency + reduce-only clamps make double-close benign (second close clamps to residual/flat ⇒ skip). Post-flatten, overlay sees flat book ⇒ HOLD/skips | Advice is an overlay input; overlay precedence order unchanged; B5 governor prevents advice hammering |
| stop_bus / SCR / LiveGate | Deliberately NOT consulted (adapter never reads them, P11) — flatten works during halts. Post-hold uses operator_intent EXIT_ONLY, not stop_bus (P12) | Unchanged — advice consumption lives inside the overlay which already runs under cycle-level gates |
| Operator exclusions | `--scope chad` never touches them; `--scope broker-all` requires second token; report lists `EXCLUDED_UNTOUCHED` explicitly | Untouchable at every layer: capture tags, aggregator filters, overlay SKIP_EXCLUDED, apply_close_intents backstop |
| Futures trio / RTH | Per-process envs (P4): CLI sets its own (`CHAD_RTH_GATE=0`; trio unset ⇒ futures closable) — documented in preview + report | N/A (equity/ETF SELLs only reach the filter) |
| Idempotency store | Shared sqlite with live-loop: flatten submits are visible to (and deduped against) live-loop history — a feature; `duplicate_blocked` classified benign | Advice closes use the overlay's existing keys — no new key shape |
| Lane A (fuse box) | No shared files. Future composition note: a fuse-box "enforce" would gate entries only, never flatten closes (flatten bypasses stage-3 entirely; overlay path exempt by Lane-A design) | Advice-fired closes are overlay closes — Lane A's invariant (never block overlay closes) covers them by construction |
| INCIDENT-0713 | Clamps from live broker probe at submit time; side from broker sign; skip-if-flat; crypto clamp-at-dispatch vs flipping book; ledger used for attribution only, never sizing | Advice can only close broker-confirmed open legs, clamped reduce-only — a stale/false urge cannot flip anything |
| Micro EOD flatten | Untouched (its intents file remains unconsumed — flagged as dormant machinery for a future disposition, NOT silently adopted) | — |

## 5. Test plan

- LC6: all phases against fakes (fake IB with canned open-orders/positions incl.
  cross-client visibility split; fake adapter capturing dry_run intents; sqlite
  tmp book for Kraken) — probe fail-closed (dead socket ⇒ abort, never flat),
  clamp math (guard>broker, broker>guard, flat, sign flip), scope/exclusion
  (chad default never touches the 9; broker-all requires second token), cancel
  verify loop, residual verdicts, SLA trace math, report schema, token gates
  (wrong token = exit 2, no silent degrade), idempotent re-run, drill proof
  content. Conftest write-guard: all outputs via injectable tmp paths.
- J16: byte-identical OFF (extend blast-radius list-equality pins); record-mode
  row content per site incl. `excluded`/`position_open` enrichment; site-3 parity
  log; aggregator TTL/ownership/exclusion filters; `ADVICE_WOULD_CLOSE` evidence
  in record mode with zero submissions (fake adapter never called — W2BS receipt
  pattern); consume-mode close inherits clamp+governor; gamma|MSFT-shaped receipt
  regression (excluded ⇒ recorded, never consumed); heartbeat every cycle.
- Set-diff methodology: W4B-0 captures the worktree baseline failing set; every
  commit must keep failures ⊆ baseline (named-test diff). `py_compile` per file;
  `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview` smoke at the wiring commits.

## 6. Phase-2 commit plan (prefix `W4B`)

| # | Commit | Contents |
|---|---|---|
| W4B-0 | baseline capture | full-suite baseline set → `audits/W4B_BASELINE.md` |
| W4B-1 | advice recorder | `exit_advice.py` + 3-site seam wiring (record mode) + site-3 parity log + heartbeat + tests |
| W4B-2 | advice aggregation + would-only | aggregator + overlay `strategy_advice` rule emitting `ADVICE_WOULD_CLOSE` in record mode + `config/exit_advice.json` + tests |
| W4B-3 | consume mode | consume gating (flip = operator GO later) + shadow-compare report tooling + receipt regressions |
| W4B-4 | flatten core | CLI skeleton: gates, tokens, clientId registration, Phase-0 probe (fail-closed) + preview + scope resolution + tests |
| W4B-5 | flatten act phases | cancel + close (both lanes, clamps) + confirm/SLA + report artifact + tests |
| W4B-6 | drill + narration + governance | drill proof artifact, coach narration kinds, order-guard hook addition, docs (incl. honest drill-cadence note), closure record |

Stops exactly as prior waves: no push, no merge, no flag flips, no live restarts.

## 7. Decision points (operator input required before Phase 2)

- **D1 — Flatten scope default.** `--scope chad` (CHAD-attributed legs, clamped
  to broker net; operator-excluded symbols untouchable) as default; `--scope
  broker-all` requires second token `INCLUDE-EXCLUDED`. *Recommended: as stated.*
  Alternative: broker-all default (true "everything flat") — rejected by default
  because 5 of 11 current positions are operator pre-existing holdings.
- **D2 — Close submission path.** Through `IbkrAdapter` (book coherence:
  fills→closer→guard; idempotency + evidence free; per-process env neutralizes
  RTH/futures gates) vs direct ib_async (flatten_oneshot style — no ledger
  coherence, needs manual guard cleanup after). *Recommended: adapter.*
- **D3 — Cancel primitive.** `reqGlobalCancel` (cross-client, kills manual
  orders too — emergency semantics, verified by re-probe) vs per-order
  `cancelOrder` loop over `reqAllOpenOrdersAsync` results (cross-client cancel
  needs master-client caveats). *Recommended: reqGlobalCancel + verify loop.*
- **D4 — Emergency gate set.** Minimal (paper-mode fail-closed + typed tokens;
  NOT SCR/reconciliation-gated — emergencies are when those are broken; wrong
  token = hard error) vs full DRIFT-RECON-style gates. *Recommended: minimal.*
- **D5 — Post-flatten hold.** `--hold` default ON → `operator_intent EXIT_ONLY`
  TTL 24h (entries frozen, exits alive) vs print-instructions-only. *Recommended:
  default ON.* (Explicitly not stop_bus/STOP — both freeze exits, P12.)
- **D6 — Advice consumption policy v1.** Owning-strategy + fresh (TTL 2 cycles)
  + broker-confirmed-open ⇒ close (`strategy_advice`) vs trail-tightening-only
  vs N-strategy quorum. *Recommended: owning-strategy close — restores pre-W2B
  native-exit parity under overlay law; judged on the would-close corpus first.*
- **D7 — Advice config home.** New `config/exit_advice.json` (frozen overlay
  config untouched) vs amending `position_exit_overlay.json`. *Recommended: new
  file.*
- **D8 — Drill cadence governance.** Record "no SSOT quarterly mandate found"
  (P7) in the drill proof + file a backlog SSOT-v9.9 amendment proposing
  quarterly flatten-drill cadence (aligned with the 30-day STOP drill) vs
  hard-coding a cadence check now. *Recommended: honest note + backlog PA.*

— END PLAN (Phase 1). STOP here; Phase 2 requires D1–D8.
