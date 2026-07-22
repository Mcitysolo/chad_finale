# PLAN_W3A — Wave 3 Lane A: The Harness Trade-Log Adapter (PHASE 1 — PLAN ONLY)

**Status:** DESIGN / FOR REVIEW — no adapter code written. This document + the operator
checklist are the only artifacts on branch `goal/wave3-harness-adapter` (worktree
`/home/ubuntu/chad_w3a`, based on `main` @ `f978f82`).
**Gate it controls:** `ready_for_live = false`. Stage-2 is the seam by which REAL paper
trades reach the identical Stage-1 verdict engine (Phases 0–5). Until that engine renders a
`PASS (candidate)` on genuine, sealed, cost-adjusted out-of-sample real trades, the gate
stays locked (SSOT `docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md` §1.3 / Part 6 / Part 7).
**Constraints (binding):** worktree only, no `runtime/` mutation, no `live_loop` changes,
set-diff test methodology, commits prefixed `W3A`.

> **D1 accepted (2026-07-22) — framing correction:** the v4 checklist's "E6-B: build the
> adapter" is amended to **"E6-B: *unstaple* the adapter"** — the adapter already exists; the
> checklist bends to reality, not the reverse. Decisions D1–D8 answered GO (all as
> recommended, with the confirmations recorded in the closure section). Build proceeds
> W3A-1..7, set-diff green vs the 371-test baseline at each step, stopping at the push decision.

---

## HOUSEKEEPING — operator checklist

The prompt asked to commit the operator's reconciled checklist as
`docs/CHAD_ELITE_CHECKLIST_v4_2026-07-22.md`. **That file is NOT present** in the tree
(neither `docs/CHAD_ELITE_CHECKLIST*` in the live tree nor the worktree). Per the standing
instruction ("if absent, note it and proceed — do not block"), it is noted here and NOT
fabricated. The nearest existing artifact is `docs/PHASE9_PROMOTION_CHECKLIST.md` (a
different document). When the operator pastes/uploads the v4 checklist it will be committed
as a standalone `W3A` housekeeping commit — it does not gate this plan.

---

## THE REFRAME (the premise is stale — the seam already exists)

The task brief states *"Phase 6 — trade_log_adapter.py … does not exist."* **It does.**

`chad/validation/trade_log_adapter.py` is a 1001-line, tested module, built across five
commits and evolved since Phase-6 landing:

| commit | what it added |
|---|---|
| `d60c105` | G4-U1: Stage-2 trade-log adapter — fail-closed trust gate + manifest + tests |
| `49cdfc0` | G4-U4: local-review hardening (adapter + CLI) |
| `5f7aa3a` | CRYPTO-TRUST: trusted-fill provenance separation (`admitted_by_provenance`) |
| `217a3dc` | FU1-B2: seed-lot trust exclusion un-missable (meta+extra+fill_ids) |
| `b4f0eaf` | W2A-1: honour the operator quarantine manifest |

It is re-exported nowhere from `chad/validation/__init__.py` (deliberately — it carries a
`__main__`), is wired into `cli.py` (`--stage stage2`), and has 459 lines of tests in
`tests/validation/test_trade_log_adapter.py` (36 test functions). The design doc itself
(Part 5) already records it as *"Adapter built + tested as a stub now."*

**Therefore W3A is a HARDEN-AND-EXTEND of an existing seam, not a greenfield build.** Every
section below is written as *audit-of-existing → identified delta → planned change*. This is
the single most important framing decision and is surfaced as **D1**.

### What the existing adapter already does (verified, not assumed)
- **Source:** reads `data/trades/trade_history_YYYYMMDD.ndjson`, backups excluded by strict
  regex (`_LEDGER_RE`, line 134) so a `.scr_reset_bak` / `.pre_*_bak` copy can never
  double-count (`iter_ledger_files`, 686-698).
- **Trust gate (fail-closed):** `trust_exclusion` (287-336) with fixed precedence
  `placeholder_100 → broker_rejected → non_fill_status → validate_only → pnl_untrusted →
  scoring_excluded`, checking all three carriers (`tags` / `extra` / `meta`, 296-334); plus
  `is_quarantined` (339-370) checked *before* the trust gate for operator-manifest pins
  (`load_quarantine_pins`, 704-753). Structural guarantee: a row reaches `admitted` only on
  the `trust_exclusion(...) is None` branch (the sole append site, 663), with a post-loop
  re-run tripwire (672-673).
- **Mapping:** `AdmittedTrade` (376-425) → `to_fill_mapping()` (397-412) sets
  `entry_price = exit_price = fill_price` and carries `pnl = gross_pnl` authoritatively;
  consumed by `chad.validation.cost_model.Trade.from_fill`.
- **Output:** deterministic `stage2_trades_<win>.ndjson` + `stage2_manifest_<win>.json`
  under `edge_reports/stage2/`; manifest counts rows_read/admitted/`excluded_by_reason`
  (fixed schema)/date-range/strategies/`admitted_by_provenance`/`admitted_by_instrument_class`.
- **Isolation:** stdlib + `cost_model` enums only; reads the evidence writer as TEXT, imports
  no broker/runtime module; writes nothing under `runtime/` and never `ready_for_live`.

### The one fact that makes the whole seam inert today
`cli.py::run_stage2_trade_log` (855-1027) **hard-codes `n_walk_forward_windows = 0`**
(`cli.py:843`) and `n_regimes_in_oos` is never derived. With `W_MIN = 6`
(`verdict.py:74`), the verdict precedence `… → INSUFFICIENT_DATA → FAIL → PASS`
(`verdict.py:273-274`) short-circuits **every** Stage-2 head to `INSUFFICIENT_DATA` by
construction (`verdict.py:335-336`; documented at `cli.py:1008-1010`). The adapter can admit
trusted fills all day; the machine is *structurally incapable of ever saying more than
INSUFFICIENT_DATA on real trades* until W3A supplies walk-forward windows and regimes and
either a Stage-2 lockbox or a documented reason it stays lockbox-free.

**The good news:** the missing machinery is already built and tested in Phases 0–5 —
`splits.generate_walk_forward` / `splits.partition` (label-horizon-aware purge+embargo) and
`regime_labeler.label_series` (the independent, non-circular labeler, §3.4). W3A is
overwhelmingly *wiring + honesty-stamping*, not new statistics.

---

## BASELINE + SET-DIFF METHODOLOGY

- **Harness suite baseline (measured on this branch, 2026-07-22):** `tests/validation/` =
  **371 passed** (3.0s). This is the green anchor; every `W3A` commit must show
  **set-diff green vs this baseline** (no previously-passing validation test regresses; new
  tests are added deltas).
- **Full-suite baseline:** the known-5 pre-existing `chad/tests/` failures documented across
  W1B/W2A/W2B (heartbeat two-writer schema race etc.) are *not* touched by W3A — W3A edits
  only `chad/validation/` + `tests/validation/`. Any full-suite delta must be `⊆ known-5`.
- **No runtime mutation, no `live_loop`:** W3A writes only `chad/validation/trade_log_adapter.py`,
  `chad/validation/cli.py` (the Stage-2 handler — offline harness, NOT `live_loop`; see **D8**),
  and `tests/validation/*`. It never writes `runtime/`, never imports a broker/runtime module
  (the transitive-import-closure test `tests/validation/test_isolation.py` stays the guard).

---

## (1) SOURCE OF TRUTH

**Decision (recommended, already the built choice): the canonical closed-lap record is
`data/trades/trade_history_YYYYMMDD.ndjson`, filtered to `payload.schema_version ==
"closed_trade.v1"`.** The three candidates, adjudicated:

| store | what it is | verdict |
|---|---|---|
| `data/trades/trade_history_*.ndjson` | one **closed round-trip** per line (`closed_trade.v1`), realized `pnl` + `entry_time_utc` + `exit_time_utc`, **hash-chained** (`prev_hash`→`record_hash`, genesis=`"GENESIS"`), written by `trade_closer.py:1002-1054` | **CANONICAL** |
| `runtime/trade_closer_state.json` | **open-lot FIFO queues only** (`processed_fill_ids` dedupe set of 5,963 + `queues` of 12 open lots + `saved_at_utc`); no closed trades, no realized pnl, no exit (`trade_closer.py:740-753`) | **REJECT** — not a closed-lap record |
| `runtime/scr_state.json` | derived **aggregate** (`stats.effective_trades=67`, WARMUP), not a per-trade store | **REJECT as the trade source**, but see the cross-check below |

Rejecting the FIFO state and SCR aggregate is not a close call — neither carries per-lap
realized pnl. The ledger is also the only store with a **tamper-evident hash chain**, which
the harness should *verify* (today the adapter reads `record_hash` for dedup only, 657-662;
it does not walk `prev_hash`→`record_hash` to detect an edited/inserted lap — a W3A
hardening item, see **D6**).

### How trust-exclusions flow (only honest laps enter) — PFF1 / W2A mechanics
The ledger interleaves genuine laps with reconciliation artifacts. The exclusion markers
`pnl_untrusted` / `scoring_excluded` live **redundantly on `payload.tags` (string tags),
`payload.meta`, and `payload.extra`** — never as a single top-level key (verified on real
rows). The adapter already mirrors SCR's predicate across all three carriers
(`trust_exclusion`, 318-334), which is the correct defense-in-depth. The authoritative live
mechanics it mirrors:
- **SCR effective-set predicate** — `trade_stats_engine.py:_is_untrusted` (120-141): markers
  `("pnl_untrusted","scoring_excluded")` over `tags ∪ extra ∪ meta`; unreadable trust block
  ⇒ untrusted (fail-closed, 138-140). Effective loop 688-722; final `pnl != 0` filter
  747-749.
- **Quarantine union** — `chad.utils.quarantine.get_exclusion_sets` (268-311): union of four
  sources — runtime `quarantine_manifest_*.json`, `data/fills/FILLS_*` untrusted fill-ids,
  **FIFO seed lots** (`RECON_ADOPT_*`), and `data/*/quarantine_*.json` sidecars; a derived
  closed trade is dropped if **any** of its `payload.fill_ids` is pinned
  (`is_record_quarantined`, 314-331). The adapter reads the **manifest subset only** (by
  design — the fills-scan / FIFO / sidecar branches require importing runtime readers that
  would break harness isolation), because the ghost-scrub pins by `record_hash` regardless
  (`load_quarantine_pins` docstring, 704-722).
- **Seed-lot (FU1-B2):** an Epoch-3 `RECON_ADOPT_*` lot has a **fabricated cost basis**
  (`provenance: UNATTRIBUTED_EPOCH3_ACCUMULATION`, tagged `pnl_untrusted`+`scoring_excluded`
  on tags **and** meta **and** extra — real example verified). Arithmetically well-formed,
  economically fictional → `scoring_excluded` (`trust_exclusion`, 328-334).

### The divergence problem (the load-bearing finding of section 1) — **D2**
The adapter **re-derives its own gate** (for isolation) rather than consuming SCR's exact
effective set. So `admitted` is NOT guaranteed equal to SCR's `effective_trades=67`. Known
divergences, each needing a stated policy:

| case | SCR effective set | current adapter | proposed W3A policy |
|---|---|---|---|
| `manual` trades | **excluded** (`_is_manual`, `:696-698`) | **admitted** unless separately `pnl_untrusted` | **exclude** (add a `manual` reason) — hand-trades are not strategy edge |
| **futures** rows | **excluded** (`is_futures_row`, `:715-717`; Bug-B contamination) | **admitted** as `InstrumentClass.FUT` | **quarantine as a separate untrusted evidence class** pending the Bug-B book disposition (Day-0 gating remainder, CLAUDE.md); never pool with equity edge |
| `warmup_sim` | **excluded** (`_is_warmup_sim`) | not explicitly caught | add a `warmup_sim` reason (align) |
| unfilled IBKR-paper | **excluded** | caught only via `$100` placeholder | verify parity; extend if a non-`$100` unfilled shape exists |
| `pnl == 0` scratch lap | **dropped** (`:747-749`) | **admitted** (0-return) | **admit** as a legitimate 0-return sample, but **report the count** so the delta vs SCR's 67 is explained, not silent |

**W3A requirement:** the Stage-2 manifest must carry a **reconciliation block** —
`{adapter_admitted, scr_effective_trades, delta, delta_explained_by:{manual,futures,pnl_zero,…}}`
— read from `runtime/scr_state.json` as TEXT (no import; isolation preserved) — so an
unexplained divergence between the two scorekeepers is *loud*, never invisible. (Reading
`scr_state.json` as text for a **reported cross-check** does not violate the isolation
contract, which forbids *importing* runtime readers and *mutating* runtime; a one-way text
read used only in the manifest is the same shape as the quarantine-manifest read the adapter
already does. Flagged for confirmation in **D2** since it is the first runtime-file read.)

---

## (2) SCHEMA MAPPING (harness contract vs what real fills carry)

Harness input contract = the eight fields `AdmittedTrade` supplies to
`cost_model.Trade.from_fill` (`trade_log_adapter.py:387-395`): `instrument_class, quantity,
fill_price, notional, gross_pnl, liquidity_tier, multiplier, strategy` (+ `provenance`).

| harness field | real `closed_trade.v1` source | status / policy |
|---|---|---|
| `quantity` | `payload.quantity` (fallback `size`) | present; require finite > 0 (else malformed) |
| `fill_price` | `payload.fill_price` | present; require finite > 0 |
| `notional` | `payload.notional` (fallback `|qty·price|`) | present; require finite > 0 |
| `gross_pnl` | **currently `payload.pnl`** | see cost note + **D3** |
| `instrument_class` | derived: `asset_class`→broker→symbol heuristic (`classify_instrument`, 227-252) | `asset_class` absent on real rows; nearest is `meta.raw_asset_class` (`"equity"`) — **add `meta.raw_asset_class` to the precedence** so equities aren't symbol-guessed |
| `multiplier` | `extra.multiplier` / `payload.multiplier` (default 1.0); real rows carry `contract_multiplier` | **add `payload.contract_multiplier`** to the lookup (currently missed) |
| `strategy` | `payload.strategy` (`gamma`,`manual`,…) | present; the real segmentation key |
| `liquidity_tier` | **hard-coded `MID`** (489) | acceptable default; document as a conservative assumption |
| `regime` (provenance) | `payload.regime` (e.g. `risk_on`, present 07-13, absent 07-20) | **carry as provenance ONLY** — it is CHAD's own (proven-buggy) classifier; the harness slices with its **own** `regime_labeler` (§3.4). Never use the ledger tag as the slicing authority |
| `setup_family` | **ABSENT** — `grep -rc setup_family data/trades/` = **0** | **mark-unknown.** The field is a `meta.setup_family` concept (`trade_closer.py:246,315,530,875`) that live gamma/manual/reconciled fills never set; consumer buckets absent→`UNKNOWN`. **Segment by `strategy` (± harness regime), not `setup_family`** |
| costs (`commission`/`slippage`/`fees`) | present but **all `0.0`** on real rows today | see the cost note — the harness is the sole cost authority |

### Cost note (S4) — verified against the real ledger, and a forward-looking trap → **D3**
On every real closed lap today, `pnl == gross_pnl == net_pnl` and `commission == slippage ==
0.0`. IBKR paper fills carry **no modeled costs** — precisely the "trading is free" lie the
design doc §3.5 exists to defeat. So the harness charging its own conservative haircut on
`payload.pnl` is **correct today** (no double-charge). **But** PA-EP1 fee-modeling
(`fee_model=ibkr_fixed_v1`) is *forward-only* (CLAUDE.md / EVIDENCE_PIPELINE_WAVE1): once
modeled commissions land in the ledger, `net_pnl` will diverge from `gross_pnl`, and an
adapter feeding `pnl` (tracking net) on top of the harness haircut would **double-charge
costs on post-EP1 rows only** — a silent, date-dependent bias. **Proposed W3A fix:** feed
`payload.gross_pnl` explicitly (fall back to `pnl` when `gross_pnl` absent), making the
harness cost model the single, temporally-consistent cost authority. Stamp the manifest with
which pnl field each row used.

### Timestamp reliability (flagged during the real-row read)
Real rows mix `…Z` and `…+00:00` offset forms, and at least one verified lap has
`exit_time_utc` (`13:50:56Z`) **before** `entry_time_utc` (`13:51:10`) — a netting/clock
artifact. The walk-forward time axis (§4) must (a) normalize both tz forms, and (b) apply a
**pessimistic sort key** and an `exit < entry` sanity policy (order by `exit_time_utc`, count
and quarantine inverted-duration rows rather than trusting a negative hold).

---

## (3) SAMPLE-SIZE HONESTY (n=67 today; two eras that may not be one population)

**What the harness can legitimately say, per sample regime:**
- **n = 67 (today, WARMUP):** the honest verdict is **`INSUFFICIENT_DATA`**, and it stays so
  even after W3A wires windows/regimes. 67 is a *pooled* count; fragmented into per-strategy
  heads (the Stage-2 grouping, `cli.py:895-897`) almost every head is `< N_MIN=30`, and
  `W_MIN=6` walk-forward windows / `R_MIN=3` regimes are near-certainly unmet on a corpus
  dominated by a single day. The machine working correctly = it refuses to grade this.
- **n = 100 (SCR's WARMUP→CONFIDENT threshold):** still below a *strict* per-head bar once
  fragmented; possibly a first pooled-portfolio `INSUFFICIENT_DATA`→borderline signal, never
  a `PASS (candidate)` on a corpus this thin (design doc Part 5: "expect frequent honest
  INSUFFICIENT_DATA").
- **n = 300+:** the first regime where per-head `N_min` + `W_min` + `R_min` can *plausibly*
  be met and a real `FAIL` / `PASS (candidate)` becomes possible — still conditional on the
  universe-bias flag and cost haircut.

### The two-population argument (must NOT silently pool) — **D4**
There is no `blindfolded`/`pre-sight`/`post-sight` token in the repo — that is operator
vocabulary. The real regime split it names is the **position-aware exit overlay going
SHADOW→ACTIVE** (`chad/risk/position_exit_overlay.py`; `runtime/exit_overlay_heartbeat.json`
now `mode=active, evaluated=10, would_close=0`; SHADOW→ACTIVE audit
`docs/ULTRA_CLOSE_AUDIT_2026-07-17.md`, first CLEAN round-trip expected ~2026-07-20→07-23).
- **Pre-overlay (≤ ~07-20):** "closed" laps are dominated by reconciliation/broker-truth
  **adopts** of pre-existing accumulation (fabricated cost basis, `pnl_untrusted` +
  `scoring_excluded`) plus the 07-13 gamma-UNH ledger-watcher fills. A **different
  data-generating process** — and mostly excluded as untrusted anyway.
- **Post-overlay (≳ 07-20):** genuine engine-driven round-trip closes (the 6 clean 07-20
  laps).

These are **different populations**. **Pooling them into one verdict is illegitimate by
default** and would be exactly the "getting lucky in one regime / grading a mixed sample"
failure the harness exists to block. **W3A requirement:** the adapter stamps every manifest
**and** every downstream report with a **sample-regime block** — `{epoch, overlay_boundary,
era: PRE_OVERLAY|POST_OVERLAY, n_per_era, pooled: false}` — keyed off a **frozen, declared
boundary date** (the overlay activation, ~`2026-07-20`, confirmed in **D4**). Default =
**report eras SEPARATELY**; pooling is permitted only behind an explicit
`--allow-era-pooling` flag that stamps the report `POOLED_HETEROGENEOUS_POPULATION` and
requires a documented same-population justification. The epoch (`CHAD_v8.9_Paper_Epoch_3`,
started `2026-06-30`; 83 earlier day-files fenced) is a second, coarser partition already on
disk (`runtime/epoch_state.json`) — the adapter must refuse any lap outside the active epoch.

---

## (4) LOCKBOX INTEGRITY (walk-forward for a stream that grows daily)

**Current state:** Stage-2 **bypasses the lockbox entirely** — `run_stage2_trade_log` never
seals or opens an OOS box; the report's `oos_section` is a hard stub (`access_count:0,
sealed:false, contaminated:false`, `cli.py:955-970`) and it hard-codes
`n_walk_forward_windows=0` / `oos_access_count=0` (`cli.py:840-843`). It leans wholly on the
adapter's fail-closed trust gate.

**Why the Phase-5 lockbox can't just be pointed at a growing log:** `oos_lockbox.returns_hash`
seals the **exact series content** (`{"n":len,"values":[…]}`, order- and count-sensitive).
Appending even one new daily lap changes `n` and the hash, so the next `seal()` hits the
"different `oos_hash`" branch and **raises `OOSSealError` — it treats a grown series as a
*swapped test set*** (`oos_lockbox.py:347-355`). The lockbox has no concept of an append-only
/ rolling frontier.

**W3A design — a date-cutoff walk-forward boundary (the correct abstraction for a growing
stream):**
1. **Freeze an OOS cutoff *date*, not a series.** All laps with `exit_time_utc < cutoff` form
   the sealed evaluation corpus; laps `≥ cutoff` are the *forward frontier* and are never in
   the sealed box. New daily laps extend the frontier — they do **not** mutate the sealed
   series, so the content hash is stable and `OOSSealError` never fires spuriously.
2. **Wire the already-built walk-forward** — replace the hard-coded `0`s by ordering admitted
   laps on the normalized `exit_time_utc` axis (§2 timestamp policy) and calling
   `splits.generate_walk_forward` / `splits.partition` with a **declared per-strategy
   `label_horizon`** (the holding period — required kwarg, SSOT §3.7; see **D5**). This yields
   real `n_walk_forward_windows` and purge/embargo boundaries.
3. **Wire the independent regime labeler** — `regime_labeler.label_series` over the same axis
   to produce `n_regimes_in_oos` (never CHAD's ledger `regime` tag, §3.4).
4. **Advancing the frozen cutoff = a new logged trial.** Moving the cutoff forward to admit
   more history is a config change that **increments the deflation trial count**
   (`config_freeze.deflation_trials`) exactly as a threshold change does (§3.2) — so "wait for
   more data, then re-seal" cannot be a free re-roll.
5. **Hash-chain verification** (from §1) is the tamper-integrity complement: a broken
   `prev_hash`→`record_hash` chain fails the run loudly rather than silently scoring an
   injected lap.

**Honesty invariant:** even fully wired, at n=67 across a heterogeneous, single-day-dominated
corpus the result is `INSUFFICIENT_DATA`. W3A makes the machine **capable** of a stricter
verdict when data justifies it; it never manufactures one.

---

## (5) OUTPUT (where verdicts land, cadence, and the wording contract)

- **Where.** Adapter artifacts stay under `edge_reports/stage2/`
  (`stage2_trades_<win>.ndjson` + `stage2_manifest_<win>.json`). The signed verdict lands
  under `edge_reports/` (`edge_report_<ts>.json` + human `.md`, via `report_writer`). **Never
  `runtime/`. Never `ready_for_live`.** (SSOT §1.2; enforced by the isolation test.)
- **Cadence.** **On-demand CLI first** — `python -m chad.validation.cli --stage stage2
  [--since D --until D]`. A timer is explicitly **out of W3A scope** (noted for a later wave;
  a Stage-2 verdict must be an operator-initiated act, not a background publisher that could
  normalize a `PASS` into ambient "it's ready" pressure).
- **Wording contract (verified, plus one hardening).** The harness emits exactly five
  verdicts — `PASS`, `FAIL`, `INSUFFICIENT_DATA`, `NOT_REPLAYABLE`, `CONTAMINATED`
  (`verdict.py:81-88`). **The word "GO" appears nowhere.** A pass is only ever surfaced as
  the constant `PASS_LABEL = "PASS (candidate)"` (`verdict.py:77-78`), with the reason string
  "Still only evidence for a human decision; the machine never flips ready_for_live"
  (`verdict.py:387-391`); every report carries the universe-provenance disclaimer
  "Results are evidence for a human decision only — the harness never flips ready_for_live"
  (`report_writer.py:54-59`); a non-sealed run is stamped `(dry run … NOT evidence)`
  (`report_writer.py:226-227`).
  - **Hardening (D7):** `VerdictResult.to_dict()` still emits the **raw enum string
    `"PASS"`** under the `verdict` key alongside the safe `label` (`verdict.py:241-242`). A
    machine consumer reading `verdict` sees a bare `"PASS"`. W3A adds, to the **Stage-2**
    report path only, (a) an unconditional top banner `EVIDENCE — NOT AN AUTHORIZATION TO
    TRADE LIVE`, (b) the sample-regime + evidence-class blocks (§2/§3) so no `PASS` can be
    read without its era and its `SIMULATED_AGAINST_LIVE_TICKS`-vs-broker-confirmed split, and
    (c) a Stage-2 assertion that the raw `verdict` value is never rendered without its
    `label`. The operator alone decides; the harness reports evidence.

---

## (6) TEST PLAN (set-diff; exclusions bind; poisoned laps cannot enter)

**Existing coverage to preserve (already proves the core of requirement 6)** — in
`tests/validation/test_trade_log_adapter.py`: `test_excludes_pnl_untrusted`,
`test_excludes_validate_only`, `test_excludes_broker_rejected_by_status|_by_tag`,
`test_excludes_non_fill_status`, `test_excludes_100_placeholder_equity|_by_trust_state_marker`,
`test_self_check_holds_on_admitted` (the tripwire), `test_trust_constants_match_live_write_path`
(live-parity), `test_run_adapter_byte_identical` (determinism),
`test_duplicate_record_hash_deduped`. All in-memory synthetic fixtures — **nothing reads the
real ledger** (the house rule). W3A **extends**, never rebuilds these.

**New tests W3A adds (each a set-diff delta on the 371 baseline):**
1. **Synthetic-laps fixture (exclusions bind + capability):** a multi-strategy, multi-day,
   multi-regime closed_trade.v1 corpus with known honest laps → asserts the wired
   walk-forward yields the expected `n_walk_forward_windows`, `regime_labeler` yields the
   expected `n_regimes_in_oos`, and (below minimums) the verdict is `INSUFFICIENT_DATA`.
2. **Poisoned-lap fixtures (each proves it CANNOT enter):** one per new/aligned exclusion —
   `manual` (D2), `futures`/Bug-B (D2), `warmup_sim`, `scoring_excluded` seed lot with the
   marker on **meta only** (the FU1-B2 pre-mirror case), quarantine-manifest-pinned by
   `fill_ids`, and a **broken hash-chain** row (D6). Assert admitted-set membership via
   **set-diff**: `admitted(clean ∪ poison) == admitted(clean)`.
3. **Era-stamping test:** a fixture straddling the frozen overlay boundary → asserts the
   manifest/report carry `era` counts and that pooling is **refused** without
   `--allow-era-pooling`, and stamped `POOLED_HETEROGENEOUS_POPULATION` when forced.
4. **Cost-authority test (D3):** a lap with `gross_pnl != net_pnl` (post-EP1 shape) → asserts
   the harness charges its haircut on **gross**, not net (no double-charge), and stamps which
   field was used.
5. **Reconciliation cross-check test (D2):** a fixture whose SCR-predicate exclusions differ
   from the adapter's → asserts the manifest reconciliation block reports the delta and its
   explanation.
6. **Timestamp-sanity test:** mixed `Z`/`+00:00` and an `exit < entry` lap → asserts
   normalization + inverted-duration quarantine/count.
7. **Wording-contract test (D7):** asserts no Stage-2 report surface can render a bare `PASS`
   without `label`, and the `EVIDENCE — NOT AN AUTHORIZATION` banner is present.

---

## DECISION POINTS (please answer before any Phase-2 build)

- **D1 — Reframe.** Confirm W3A is a **harden-and-extend of the existing
  `trade_log_adapter.py` + Stage-2 CLI wiring**, not a greenfield rebuild. (Recommended: yes.)
- **D2 — Trust-gate alignment.** Keep the adapter re-deriving its own gate (isolation) **plus**
  a text-read reconciliation cross-check against `scr_state.json`? And the per-case policy:
  **exclude `manual`**, **quarantine futures as a separate untrusted class** pending Bug-B,
  **admit `pnl==0`** as 0-return (reported), align `warmup_sim`. (Recommended: yes to all.)
  Sub-question: is a one-way TEXT read of `runtime/scr_state.json` for a *reported* cross-check
  acceptable under the isolation contract? (I read it as yes — same shape as the existing
  quarantine-manifest text read — but it is the first runtime-file read, so it needs your OK.)
- **D3 — Gross vs net pnl.** Switch the cost-model input from `payload.pnl` to
  `payload.gross_pnl` (fallback `pnl`) so the harness is the single, temporally-consistent
  cost authority once PA-EP1 fee-modeling lands? (Recommended: yes.)
- **D4 — Era boundary + pooling.** Freeze the overlay-activation cutoff at **`2026-07-20`**
  (or a date you specify) and **report pre/post eras separately by default, never pooled**?
  (Recommended: yes; pooling only behind an explicit stamped flag.)
- **D5 — Walk-forward `label_horizon`.** How is the per-strategy holding period declared for
  purge/embargo — a static per-strategy map, or derived from observed `entry→exit` spans?
  (Recommended: static conservative map now; derived is a later refinement.)
- **D6 — Ledger integrity gates.** Add an explicit `schema_version == "closed_trade.v1"`
  admit-gate **and** `prev_hash`→`record_hash` chain verification (fail-loud on a break)?
  (Recommended: yes to both.)
- **D7 — Output hardening + cadence.** On-demand CLI only for W3A (timer deferred), plus the
  Stage-2 "EVIDENCE — NOT AN AUTHORIZATION" banner and the no-bare-`PASS` assertion?
  (Recommended: yes.)
- **D8 — Scope of `cli.py` edits.** The core fix (`n_walk_forward_windows`, regimes, lockbox
  wiring, banners) lives in `cli.py::run_stage2_trade_log`, **not** in `trade_log_adapter.py`.
  `cli.py` is offline harness, not `live_loop`, so it is within the "no live_loop" constraint
  — but confirm you accept W3A touching `cli.py`'s Stage-2 handler (and only that handler).

---

## PHASE-2 COMMIT ORDER (all `W3A`-prefixed; set-diff green vs the 371 baseline each)

1. **W3A-0** — housekeeping: commit the operator v4 checklist when supplied (standalone).
2. **W3A-1** — schema-mapping completeness: `meta.raw_asset_class` + `contract_multiplier`
   precedence, gross-pnl input (D3), timestamp normalization + `exit<entry` policy (+ tests).
3. **W3A-2** — trust-gate alignment: `manual` + `warmup_sim` reasons, futures separate class,
   `schema_version`/hash-chain gates (D2/D6) (+ poisoned-lap tests).
4. **W3A-3** — reconciliation cross-check block vs `scr_state.json` (D2) (+ test).
5. **W3A-4** — Stage-2 walk-forward + regime wiring in `cli.py` (D5/D8): kills the hard-coded
   `0`s via `generate_walk_forward` / `label_series` (+ synthetic-laps capability test).
6. **W3A-5** — date-cutoff OOS boundary for the growing stream (D4 era + lockbox integrity)
   (+ era-stamping test).
7. **W3A-6** — output hardening: EVIDENCE banner, sample-regime/evidence-class blocks,
   no-bare-`PASS` assertion (D7) (+ wording-contract test).
8. **W3A-7** — PLAN_W3A Phase-2 closure record (decisions → commits + Phase-3 go/no-go).

---

## STATUS: Phase 1 (plan) complete — awaiting decisions D1–D8 before any build.

No adapter/CLI/test code has been written. The only change on this branch is this document
(and, when supplied, the operator checklist). The live tree is untouched.
