# PLAN_W5A — Wave 5 Lane A: The Measurement Layer (E2/E2-B TCA · E3 MAE/MFE · DQ2 Clock Gate · Harness Handoff)

**Phase 1 — PLAN ONLY.** No code in this commit. Worktree `chad_w5a`, branch
`goal/wave5-measurement`, base `main@e86eaaf` (the W4A fuse box was merged to
main since the wave started — `e86eaaf` is the merge of the W4A-0..9 stack;
verified). Live tree untouched. Phase 2 begins only after operator decisions
D1–D10.

Method: four parallel read-only audits over the worktree (TCA substrate,
exit-overlay excursion, clock inventory, harness/adapter contract) before any
design. **Every one of the four items carried a stale or under-/over-stated
premise** — §1 records the verified ground truth this plan is built on. The
task's own hints were half-right: the hash-chained fill envelopes ARE the
friend (they already carry the decision reference + slippage + fee materials),
but the W3B-6 SIM-mark stamps are NOT (they describe exit-decision mark
freshness, not an entry decision price), and the W4B-2 close stamps do NOT
deterministically reach the closed_trade today (§1-P9).

---

## 0. Scope and non-negotiables

- Four items, ALL observer-class. The two that add a payload field are
  flag-gated default-OFF (behavior-neutral until an operator flips the flag);
  the sentinel finding is warn-first and read-only.
- One change at a time; set-diff test methodology (§8); commits prefixed `W5A`.
- **No `chad/validation/` edits this wave** (Lane-B-adjacent frozen territory).
  Item 4 is a CONTRACT DOCUMENT only — it names the fields Lane B will later
  wire, and the import-isolation test (`tests/validation/test_isolation.py`)
  is the reason the adapter must keep reading the ledger as text.
- No runtime mutation, no systemd changes, no config mutation outside the
  worktree. The closed_trade ledger is **hash-chained** — §1-P7 makes the
  schema disposition (D6) a correctness constraint, not a style choice.
- **Prime invariant: nothing in W5A changes a P&L number, a fill, a match, or
  a trade decision.** Every field is additive and descriptive; every reader is
  fail-open. The measurement layer OBSERVES the exam; it does not sit it.

## 1. Audited ground truth (premise verification)

| # | Premise / claim | Verdict | Evidence |
|---|---|---|---|
| P1 | "Decision price = the signal's own stamp; W3B-6 SIM-mark staleness stamps matter here" | **STALE (misdirected).** W3B-6 is `exit_overlay.v2` mark provenance (`mark_ts_utc/mark_age_s/mark_source`) — the freshness of the mark the EXIT overlay used, not an entry decision price. `TradeSignal` carries **no** price field at all | `chad/risk/position_exit_overlay.py:275-279`; `chad/types.py:101-122` |
| P2 | "compute IS = slippage + fees + opportunity cost, decision-benchmarked" | **UNDERSTATED — the materials already exist, scattered, keyed by `fill_id`.** Decision reference: PA-EP2a `extra.submit_quote.ref_price` (on EVERY executor fill) + `extra.expected_price`. Realized slippage: `data/slippage/SLIPPAGE_*.ndjson` + `EXECUTION_METRICS_*.ndjson.slippage_bps`. E2 is a JOIN, not a from-scratch compute | writer `paper_exec_evidence_writer.py:1036-1064` (submit_quote), `:750-777` (exec-metric), `chad/analytics/slippage_tracker.py:96-150` |
| P3 | closed_trade already has cost slots | VERIFIED but **unpopulated**: `to_payload` writes `commission=0.0`, `slippage=0.0`, `fees=None`, `pnl_breakdown.cost_basis_status="unavailable"` (all three cost inputs passed `None`). `net_pnl == gross`. Slippage is **never** computed in this path | `chad/execution/trade_closer.py:302-383`; `chad/analytics/pnl_breakdown.py:114-182` |
| P4 | Fees are modeled | VERIFIED, in a SIBLING file: IBKR `ibkr_fixed_v1` writes `fee_amount` to `FEES_*.ndjson` (fill only carries the marker `extra.fee_model`); Kraken `kraken_paper_v1` puts `fee` on the trusted fill AND `extra.entry_fee/exit_fee` on the Kraken round-trip row. **Neither dollar value reaches `closed_trade.v1`** — `trade_closer._extract_fill` drops it | `paper_exec_evidence_writer.py:1628-1766`; `chad/core/kraken_trusted_fill_engine.py:205-225,561-614`; `trade_closer.py:465-623` |
| P5 | "opportunity cost on unfilled qty" | **OVERSTATED — no data source.** No requested-vs-filled split is persisted in the closer lane; only a boolean `partial_fill`. `pending_exposure_ledger.release(filled_qty=)` is a documented no-op stub; `paper_shadow_runner` tracks unfilled qty but never persists it to the fill/closed_trade evidence. For the PAPER lane the executor fills fully, so opp-cost ≈ 0 by construction | `paper_exec_evidence_writer.py:704`; `chad/risk/pending_exposure_ledger.py:47-49,330-351` |
| P6 | funding/borrow cost | **DOES NOT EXIST** in any accounting path. All "funding" hits are Kraken-futures funding-*rate* intel feeds, never a cost applied to a lap. Kraken spot has no funding. Greenfield for E2-B | repo-wide grep: only `*_intel_publisher.py` / `crypto_signal_filter.py` |
| P7 | "schema bump per the D6/W4A precedent" | **The two W4A precedents apply to DIFFERENT artifact classes.** `drawdown_state` (D6) bumped v1→v2 because it is NOT hash-chained. `closed_trade` gained `regime` (W4A-2) **additively with NO bump** because it IS hash-chained: `verify_ledger_chain` recomputes the byte-exact `record_hash` **only for rows with `schema_version == "closed_trade.v1"` exactly** — a `.v2` bump silently SKIPS hash verification. So the correct precedent for THIS artifact is additive-no-bump (D6) | `trade_closer.py:374-376` (regime, no bump); adapter `verify_ledger_chain` exact-match `chad/validation/trade_log_adapter.py:1070` |
| P8 | "the overlay already walks every position with price+peak+trough — extend its evidence" | **HALF-TRUE.** The walk + persisted per-position `peak`/`trough` are real, BUT they are a **close-to-close point-price sample, not a true high/low watermark** (intracycle spikes between ~72s cycles are missed), and the excursion is **discarded at close** (anchor pruned, absent from `closed_trade.v1`). No MAE/MFE code exists today | overlay update `position_exit_overlay.py:753-755`; prune `:1214-1215`; crypto twin `crypto_exit_overlay.py:276-278` |
| P9 | W4B-2 close stamps reach the closed_trade | **NO (nuance).** `_close_intent_to_ibkr` stamps `action/close_origin/reason/position_key/open_side/close_side` onto `intent.meta`; but the FIFO matcher populates `ClosedTrade.meta` from the **OPENING** lot's meta only — close-side provenance does not deterministically appear on `closed_trade.v1` | `chad/core/position_reconciler.py:407-412`; `trade_closer._match_fill` opening-meta only |
| P10 | overlay excursion is already in evidence | VERIFIED and USEFUL: every cycle's verdict (HOLD + WOULD_CLOSE) is written to `data/exit_overlay/exit_overlay_YYYYMMDD.ndjson` (`exit_overlay.v2`) and `to_dict` **already emits `peak`/`trough`** — so a SAMPLED MAE/MFE over a lap's open life is reconstructable with no new walker | `position_exit_overlay.py:281-307,1292-1310` |
| P11 | join keys | closed_trade↔fill = `fill_id` (and `payload.fill_ids` for the leg list); closed_trade↔harness = `record_hash` (strongest, 1:1); overlay↔closed_trade = `strategy\|symbol` (temporal — anchor live now = lap closing now). `fill_ids` is NOT surfaced into the adapter's provenance today | `trade_closer.py:359`; adapter `:597-619`; overlay `:675,790` |
| P12 | DQ2 clock skew detection | **GREENFIELD, but the raw material is already on disk.** IBKR `server_time_iso` (reqCurrentTime) sits in `runtime/ibkr_status.json` beside box `ts_utc` and is **never diffed** by any consumer. No fill-time-vs-sequence monotonicity check exists; `verify_ledger_chain` checks hash+sequence but NOT time; the `FILLS_*` chain has no verifier at all | `chad/core/ibkr_healthcheck.py:181,195,231`; `verify_ledger_chain` `:1021-1084` |
| P13 | fill timestamps are single-source | **NO.** `fill_time_utc` = broker `execution.time` on the harvester path, box-time fallback in the evidence writer; every envelope ALSO carries `timestamp_utc` = box write time. `sequence_id` is monotonic by WRITE order, so a later `sequence_id` can carry an earlier broker `fill_time_utc` (batch harvest / reconnect replay) — exactly the DQ2 anomaly | harvester `ibkr_paper_fill_harvester.py:328-374,252`; writer `:700,730,761` |
| P14 | sentinel is extensible warn-first | VERIFIED: add a `check_clock_health()` method + one `_safe("clock_health", …)` line in `run()`; warn-first is free (`maybe_notify` pages ONLY on `STATUS_FAIL`). EXS9 `check_stale_processes` is the config-driven, warn-capped template; adopt the `ttl_verified=false ⇒ warn-cap` honesty rule for operator-proposed skew thresholds. The sentinel test pins the check count — must update | `chad/ops/exterminator_sentinel.py:124-149,1243-1363,1392-1396` |
| P15 | R-unit denominator | `stop_width_usd` lives in `payload.meta.stop_width_usd` but only for the 4 setup_family-stamped strategies (`alpha`, `alpha_futures`, `alpha_intraday`, `alpha_intraday_micro`). IS-in-R is computable for those laps; R=null (loud) otherwise | `trade_closer.py:298,373`; `setup_family_expectancy_updater.py:379-381` |
| P16 | adapter consumes costs today | **NO — and there is a double-charge hazard.** The adapter reads `gross_pnl`/`pnl` only; it ignores `pnl_breakdown`/`commission`/`slippage`/`fees`/`net_pnl`/`cost_basis_status`. `cost_model` ESTIMATES costs (participation/volatility default to 0 → a fixed haircut). If a realized-cost field is carried AND the harness also haircuts, rows are charged twice — the contract (Item 4) must resolve inform-vs-replace | adapter `trade_log_adapter.py:546-630,557-562`; `cost_model.py:373-457` |

## 2. Architecture — one observer layer, four instruments

Nothing new runs on a timer that isn't already running. The measurement layer
attaches to two existing chokepoints and one existing scanner:

```
chad/analytics/implementation_shortfall.py   # E2/E2-B: join fill_ids -> decision px / slippage / fee -> IS in $/bps/R
chad/analytics/excursion_recorder.py         # E3: close-time MAE/MFE snapshot from overlay anchors -> sidecar
data/exit_overlay/excursion_YYYYMMDD.ndjson  # E3 evidence: excursion.v1, one row per closed lap
config/exterminator.json  (+ clock_health)   # DQ2: EXS10 thresholds (ttl_verified:false -> warn-cap)
docs/CONTRACT_W5A_harness_handoff.md          # Item 4: the field contract (no chad/validation edits)
```

- **E2/E2-B (Item 1)** stamps `payload["implementation_shortfall"]` at the ONE
  place a lap becomes a payload — `trade_closer.ClosedTrade.to_payload`
  (`:302`) — because the row is hash-chained and can only grow BEFORE its hash
  is computed (P7). The IS module is a pure joiner: given a lap's `fill_ids` +
  prices + qty + `stop_width_usd`, it reads the sibling evidence
  (`FILLS/FEES/SLIPPAGE/EXECUTION_METRICS` for IBKR; `TrustedFill` /
  Kraken round-trip `extra.entry_fee/exit_fee` for Kraken) and returns the
  breakdown. Flag `CHAD_TCA_STAMP` (off|on, default off) gates the stamp —
  OFF ⇒ byte-identical closed_trade.
- **E3 (Item 2)** adds NO second walker. A close-detection hook in the overlay
  (`_save_anchors`, where a position leaves `live_keys` = a real close, before
  the prune at `:1214`) snapshots the anchor's final `peak`/`trough`/`entry`
  into `excursion_YYYYMMDD.ndjson`. That sidecar is authoritative (survives the
  prune race). A flag-gated best-effort read stamps `mae`/`mfe` onto the
  closed_trade at mint when the anchor is still live (the common case). Sampled
  by default; a true-watermark upgrade folds bar H/L (D7).
- **DQ2 (Item 3)** is a new warn-first `EXS10 check_clock_health` in the
  exterminator sentinel — pure-read, zero new broker calls: it diffs
  `ibkr_status.json` `server_time_iso − ts_utc` (broker-vs-box skew) and scans
  today's `FILLS_*`/`trade_history_*` for `fill_time_utc`-vs-`sequence_id`
  non-monotonicity (P12/P13). No behavior change; a finding is a report row.
- **Item 4** is a document. It names the fields E2/E3 add and the rules Lane B
  must honor to consume them (additive-nullable, `record_hash` join,
  admitted-laps-only, the inform-vs-replace double-charge resolution, P16).

**Modes:**

| Flag / config | Values | Default | Gates |
|---|---|---|---|
| `CHAD_TCA_STAMP` | off \| on | off | the `implementation_shortfall` payload field |
| `CHAD_E3_EXCURSION` | off \| sidecar \| stamp | off | `excursion.v1` sidecar (`sidecar`) + closed_trade `mae/mfe` stamp (`stamp`) |
| `config.clock_health` | present/absent | absent ⇒ EXS10 warns "unconfigured" (EXS9 idiom) | DQ2 thresholds |

Garbage flag values → off. The sidecar-vs-stamp split lets an operator get the
authoritative excursion evidence (`sidecar`) without touching the hash-chained
closed_trade until they're ready (`stamp`).

## 3. Item 1 — E2/E2-B Implementation Shortfall (TCA)

### 3.1 The join (the real work, P2/P4)
`chad/analytics/implementation_shortfall.py::compute_lap_is(lap, evidence)`:
- **Inputs** from the lap: `fill_ids=[open_id, close_id]`, `entry_price`,
  `exit_price`, `quantity`, `contract_multiplier`, `side`, `strategy`,
  `symbol`, `meta.stop_width_usd`, `broker`.
- **Per-fill lookup** (keyed on `fill_id`, cheap per-day index):
  - decision reference = `extra.submit_quote.ref_price` (PA-EP2a, always
    present) with `extra.expected_price` fallback (D2);
  - realized slippage $ = from `SLIPPAGE_*.ndjson` (per-fill, already
    signed BUY/SELL) or recomputed `(fill − ref)·qty·mult·side` if absent;
  - fee $ = `FEES_*.ndjson.fee_amount` by `fill_id` (IBKR) / `TrustedFill.fee`
    / Kraken round-trip `extra.entry_fee+exit_fee` (Kraken).
- **Aggregate** IS_$ = realized_slippage_$ (open+close legs) + fees_$
  + funding_$ (null/0 for paper, D4) + opportunity_cost_$ (null/0 for paper,
  D3).
- **Units:** IS_bps = IS_$ / (decision_price·qty·mult) · 1e4; IS_R = IS_$ /
  `stop_width_usd` when present, else null + loud note (P15/D5).
- **cost_basis_status:** `real` when every leg resolved a fee+decision price;
  `partial` when some legs missing; `unavailable` when none (mirrors the
  pnl_breakdown vocabulary).

### 3.2 The stamp (additive, hash-safe, flag-gated)
`ClosedTrade.to_payload` (`:302`) calls the joiner under `CHAD_TCA_STAMP=on`
and adds ONE nullable block:
```
"implementation_shortfall": {
  "schema_version": "implementation_shortfall.v1",
  "is_usd": ..., "is_bps": ..., "is_r": ...|null,
  "slippage_usd": ..., "fees_usd": ..., "funding_usd": null,
  "opportunity_cost_usd": null,
  "decision_price": ..., "decision_price_source": "submit_quote.ref_price"|"expected_price",
  "cost_basis_status": "real"|"partial"|"unavailable",
  "legs": [{"fill_id":..., "role":"open"|"close", "slippage_usd":..., "fee_usd":..., "ref_price":...}]
}
```
OFF ⇒ the key is absent (byte-identical, absent==null per the regime idiom).
Both lanes: IBKR reads the sibling evidence; the Kraken lane is RICHER (the
round-trip row already carries entry/exit fee + expected_price), so the joiner
prefers the Kraken-native fields when `broker` is Kraken/crypto.

### 3.3 Also fill the existing slots (optional, D6-scoped)
When `CHAD_TCA_STAMP=on` and costs resolve `real`, populate the pre-existing
`pnl_breakdown` commission/slippage/fees (currently None) so the decomposition
is internally consistent — WITHOUT changing `pnl`/`net_pnl`/`gross_pnl`
(the accounting numbers stay exactly as matched; only the *descriptive*
breakdown gains the real cost figures, `cost_basis_status` flips
`unavailable`→`real`). This is still observer-class (no P&L change).

## 4. Item 2 — E3 MAE/MFE

### 4.1 Fidelity (D7)
- **Default: SAMPLED** — reuse the overlay's existing point-price `peak`/
  `trough` (P8/P10). Zero change to the walk; MAE = worst adverse excursion
  from entry across the lap's cycles, MFE = best favorable. Honest about its
  cadence: `excursion_source: "sampled_point_price"`.
- **Upgrade (gated): TRUE WATERMARK** — fold the latest bar's `high`/`low`
  (already read by `_atr`, `:362-398`) into the `peak`/`trough` update
  (`:753-755` / crypto `:276-278`). Additive, but it changes the walk's
  accumulation and is still bounded by daily-bar granularity + cycle cadence —
  so it is a decision, not a default.

### 4.2 Capture at close (the missing persistence, P8)
`chad/analytics/excursion_recorder.py`, invoked from the overlay's
`_save_anchors` at the live_keys→closed transition (BEFORE prune):
- snapshot `{strategy, symbol, position_key, entry_price, peak, trough,
  opened_at_utc, closed_detect_utc, cycles_observed, excursion_source}` →
  `excursion_YYYYMMDD.ndjson` (`excursion.v1`), both lanes.
- MAE_pct = (trough − entry)/entry (long) / mirrored (short); MFE_pct
  similarly; MAE_usd/MFE_usd = pct · entry · qty · mult when qty known.
- Keyed for the harness join on `strategy|symbol` + `opened_at_utc`
  (temporal disambiguation for partial-close / fast-reopen, P11 caveat noted
  loudly in the row).

### 4.3 Closed_trade stamp (flag-gated, best-effort)
Under `CHAD_E3_EXCURSION=stamp`, `to_payload` does a best-effort read of the
live overlay anchor (or the just-written excursion row) keyed `strategy|symbol`
and adds `"mae": {...}`, `"mfe": {...}` additively (absent when the anchor was
already pruned — the sidecar remains authoritative for downstream join). Never
blocks the mint; fail-open.

## 5. Item 3 — DQ2 Clock-health gate

### 5.1 EXS10 `check_clock_health` (warn-first, read-only, P12/P14)
Two sub-checks, each `STATUS_WARN`-capped (operator-proposed thresholds,
`ttl_verified:false`):
1. **Broker-vs-box skew:** read `runtime/ibkr_status.json`; skew_ms =
   |`server_time_iso` − `ts_utc`|. Warn above `warn_skew_ms` (default proposal
   e.g. 2000ms). Missing `server_time_iso` ⇒ warn "broker_time_absent"
   (a silently-unmeasured clock is itself a finding).
2. **Fill-time vs sequence monotonicity:** scan today's `FILLS_*.ndjson`
   (bounded to the current day); flag any row whose `fill_time_utc` regresses
   vs the prior `sequence_id`'s beyond `warn_regression_ms`. Evidence carries
   the offending `sequence_id` pair + delta; numbers live in `evidence`, never
   the title (CTF-T2).
Config block `clock_health` in `config/exterminator.json`; absent ⇒ EXS10
warns "unconfigured" (EXS9 idiom). Registered via one `_safe("clock_health",
self.check_clock_health)` line; `test_exterminator_sentinel.py` check-count pin
updated in the same commit.

### 5.2 Honest scope
DQ2 is a **detector**, not a clock-unification refactor. "All timestamps
traceable to one source" is the aspiration; today there are ≥3 clocks (box,
broker exec, bar — P13). W5A raises findings on divergence/sequencing (warn);
actually collapsing to one authoritative clock (e.g. broker-time-stamped fills
end to end) is a larger follow-up, explicitly out of scope and named as such.

## 6. Item 4 — Harness handoff contract (doc only, no `chad/validation/` edits)

`docs/CONTRACT_W5A_harness_handoff.md` states, for Lane B to wire later:
- **New fields** (all additive-nullable on `closed_trade.v1`, ride only on
  ADMITTED laps): `implementation_shortfall.{is_usd,is_bps,is_r,slippage_usd,
  fees_usd,funding_usd,opportunity_cost_usd,decision_price,decision_price_source,
  cost_basis_status,legs}`; `mae/mfe.{pct,usd,source}`.
- **Join key:** `record_hash` (lap granularity, P11). Per-leg TCA (the `legs`
  array) needs `payload.fill_ids` surfaced into the adapter's `provenance` —
  named as the ONE adapter change Lane B must make (not done here).
- **Double-charge resolution (P16), the key contract clause:** the closed_trade
  now carries REALIZED costs. Lane B routes them so the cost_model does NOT
  also estimate — either (a) feed realized slippage/fee as the haircut and turn
  the estimated haircut OFF for rows with `cost_basis_status=real`
  (RECOMMENDED, D10), or (b) feed real `volatility`/`participation`/`bar_volume`
  so the model is data-driven. The contract MUST pick one; recommending (a).
- **Stability rules:** schema stays `closed_trade.v1` (a `.v2` bump skips the
  hash recompute, P7); no new REQUIRED keys (the adapter's required-field gate
  must still pass old rows); no `chad.execution.*`/runtime import from the
  adapter (read as text); bump the adapter's OUTPUT `stage2_trade_log` version
  when Lane B surfaces the new fields.

## 7. Interaction analysis (REQUIRED)

### 7.1 Nothing changes a number
| Path | Why it is observer-class |
|---|---|
| closed_trade P&L | `pnl`/`gross_pnl`/`net_pnl` are matched by FIFO and untouched; TCA adds a DESCRIPTIVE block + (optionally) fills the already-present breakdown slots. No re-pricing | 
| Hash chain | TCA/MAE-MFE fields are added at mint, BEFORE `record_hash` is computed over the payload — chain-safe by construction (P7); schema stays `closed_trade.v1` so `verify_ledger_chain`'s exact-match recompute still fires |
| Overlay decisions | E3 reads the anchor's existing peak/trough and (sampled default) changes NOTHING in the walk; the watermark upgrade (D7) only widens peak/trough, never a reduce/hold decision |
| Sentinel | EXS10 is warn-only; `maybe_notify` pages only on FAIL — no new page, no behavior change |
| Adapter / Stage-2 | UNTOUCHED this wave (frozen); the contract is a doc. Until Lane B wires it, the new fields are invisible to scoring (the adapter reads gross only) — so W5A cannot double-charge or mis-score anything on its own |

### 7.2 Flags default off ⇒ byte-identical
`CHAD_TCA_STAMP=off` and `CHAD_E3_EXCURSION=off` produce byte-identical
`closed_trade.v1` rows and no new evidence — proven per commit by a
shadow-identical test (§8). DQ2 has no flag because it changes no behavior; its
config-absent path warns rather than acting.

### 7.3 Composition with existing measurement
- `slippage_tracker` / `EXECUTION_METRICS` keep writing their per-fill records
  as today; E2 READS them, never replaces them (no duplicate writer).
- `pnl_breakdown` gains real values only when TCA resolves `real`; its schema
  and the `net_pnl` it reports are unchanged.
- The overlay's `exit_overlay.v2` evidence is unchanged; E3 adds a SEPARATE
  `excursion.v1` sidecar (no schema break to the heartbeat or verdict rows).
- DQ2 reads `ibkr_status.json` + `FILLS_*` read-only; it does not touch the
  watchdog, the latency tracker, or the hash chain.

## 8. Phase-2 commit plan (each: build → set-diff green → commit, prefix `W5A`)

| # | Commit | Contents |
|---|---|---|
| W5A-0 | baseline capture | full-suite run in worktree; record failing-test set + count as `audits/W5A_BASELINE.md` (W4A methodology) |
| W5A-1 | IS joiner core | `implementation_shortfall.py` (per-fill decision/slippage/fee lookup + $/bps/R aggregation, both lanes), stdlib, unit tests over synthetic evidence; no wiring |
| W5A-2 | TCA stamp | `CHAD_TCA_STAMP` gate + additive `implementation_shortfall` block at `to_payload` + (D6-scoped) breakdown-slot fill; byte-identical-when-off proof; hash-chain-intact proof |
| W5A-3 | E3 excursion sidecar | `excursion_recorder.py` + overlay close-detection hook (both lanes) + `excursion.v1`; sampled default; anchor-prune-race test |
| W5A-4 | E3 closed_trade stamp | `CHAD_E3_EXCURSION=stamp` best-effort `mae/mfe` at mint; watermark upgrade behind D7 if approved; tests |
| W5A-5 | DQ2 EXS10 | `check_clock_health` + `clock_health` config + sentinel registration + `test_exterminator_sentinel.py` count pin; skew + monotonicity fault-injection tests |
| W5A-6 | harness contract + closure | `docs/CONTRACT_W5A_harness_handoff.md` + PLAN_W5A closure record (decisions → commits). NO `chad/validation/` edit |

Set-diff methodology: after every commit, `python3 -m pytest chad/tests/ -q`
in the worktree; failure set must be ⊆ W5A-0 baseline (named-test diff).
`py_compile` per changed file; `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview`
smoke at W5A-2/-3/-4 (the payload-touching commits). No pushes, no merges —
same STOP as W4A/W4B.

## 9. What W5A deliberately does NOT do
- Does not build the live-lane opportunity-cost capture (requested-vs-filled at
  submit) — no data source today (P5); paper fills fully. Named as a follow-up.
- Does not model funding/borrow (P6) — greenfield, null for paper.
- Does not unify the clock sources (P13) — DQ2 detects divergence only.
- Does not edit `chad/validation/` — the adapter wiring + the `fill_ids`
  provenance surfacing are Lane B's, per the contract.
- Does not bump `closed_trade` to `.v2` (P7 hash constraint).

## 10. Decision points (operator input required before Phase 2)

- **D1 — TCA compute location.** Stamp at mint in `trade_closer.to_payload`
  (hash-safe; the only place a hash-chained row can grow) vs a post-hoc
  sidecar. *Recommended: mint-time stamp* (P7 forces it — a post-hoc re-stamp
  would break the chain).
- **D2 — Decision price source.** `submit_quote.ref_price` (PA-EP2a, present on
  every fill) with `expected_price` fallback, vs `expected_price` primary.
  *Recommended: submit_quote.ref_price → expected_price fallback* (the honest
  decision-time reference; expected_price is the strategy's hope, not the
  market).
- **D3 — Opportunity cost.** Null/0 for the paper lane (full-fill, no data
  source — P5) + a named live-lane follow-up, vs build a submit-time
  requested-vs-filled capture now. *Recommended: null for paper + follow-up*
  (no source exists; paper fills fully).
- **D4 — Funding/borrow.** Null for paper (P6, no carry concept; Kraken spot),
  vs model. *Recommended: null + note.*
- **D5 — R-unit denominator.** `stop_width_usd` when present (4 stamped
  strategies), R=null + loud otherwise (P15), vs a universal R proxy. *Recommended:
  stop_width_usd, null-when-absent* (no honest universal R exists).
- **D6 — Schema disposition.** Additive-nullable fields, `schema_version`
  stays `closed_trade.v1` (the regime precedent + hash-chain constraint, P7),
  vs bump to `.v2` AND update `verify_ledger_chain:1070` in the same commit.
  *Recommended: additive, NO bump* (a `.v2` bump silently disables hash
  verification for those rows — the drawdown_state.v2 precedent does NOT
  transfer to a hash-chained artifact). Sub-question: also populate the
  existing `pnl_breakdown` cost slots (§3.3) — *recommended yes, values only,
  no P&L change*.
- **D7 — MAE/MFE fidelity.** Sampled point-price (reuse the walk, zero change,
  P8/P10) vs true bar-H/L watermark (fold high/low into the peak/trough
  update). *Recommended: sampled default; watermark as an explicit, separately
  approved upgrade* (the watermark changes the walk and is still daily-bar
  bounded).
- **D8 — MAE/MFE persistence.** Overlay close-time `excursion.v1` sidecar
  (authoritative, survives the prune race) + flag-gated best-effort `mae/mfe`
  stamp onto the closed_trade, vs sidecar-only (harness joins it). *Recommended:
  sidecar + best-effort stamp* (the sidecar is the source of truth; the stamp
  is convenience for the common case).
- **D9 — DQ2 scope.** Warn-first EXS10 detector (broker-vs-box skew + fill-seq
  monotonicity), read-only, no clock unification, vs a broader clock-integrity
  build. *Recommended: warn-first detector; unification deferred and named*
  (P12/P13).
- **D10 — Harness double-charge resolution (contract clause).** Instruct Lane B
  to feed realized cost as the haircut and disable the estimated haircut for
  `cost_basis_status=real` rows (replace), vs feed real vol/participation so
  the model is data-driven (inform). *Recommended: replace* (realized truth
  beats a re-estimate; the estimated haircut exists only because real costs
  were unavailable — P16).

— END PLAN (Phase 1). STOP here; Phase 2 requires D1–D10.
