# PLAN_W6A — Wave 6 Lane A: Futures Contract Resolution

**Branch:** `goal/wave6-futures` (worktree `/home/ubuntu/chad_w6a`, from `main@c76fbea`)
**Date:** 2026-07-23
**Phase:** 1 — PLAN ONLY. No code changed. This document is the sole commit.
**Commit prefix:** `W6A`

---

## 0. Scope

**In scope:** futures *contract resolution* for market data, bars backfill, per-symbol
error isolation, roll-calendar coverage, signal-path readiness proof, EXS5 disposition.

**Explicitly OUT OF SCOPE and UNTOUCHED — futures EXECUTION gate.**
`CHAD_DISABLE_FUTURES_EXECUTION=1` is a permanent Bug-B-legacy safety guard, set at
`/etc/systemd/system/chad-live-loop.service.d/91-disable-futures-exec.conf` (also sets
`CHAD_DISABLE_FUTURES=1` and `CHAD_FUTURES_EXECUTION_ENABLED=0`), enforced at
`chad/execution/futures_gate.py:45-47`. **This lane does not read, modify, weaken,
bypass, or propose changing that gate, its drop-in, or `futures_gate.py`.** Every
deliverable below is data-path or signal-path only. No futures order can be submitted
as a result of this lane, by design and by unchanged configuration.

Also untouched: no runtime mutation (`runtime/` is read-only to this lane), no systemd
unit edits, no service restarts, no config mutation (any config change ships as a
Pending Action only).

---

## 1. PREMISE AUDIT — the June evidence is stale, and the real defect is different

The brief was written against June evidence. I re-measured current state first, as
instructed. **Four of the five stated premises are stale.** The lane is still
justified, but for different reasons, and one stated goal is not achievable here.

| # | Stated premise | Measured 2026-07-23 | Verdict |
|---|---|---|---|
| P1 | 5 of 10 futures fail resolution (M2K/ZB/M6E/SIL/MYM) | All 10 resolve and write bars. `data/bars/1d/{M2K,ZB,M6E,SIL,MYM}.json` all present, all last bar `2026-07-22` | **STALE** |
| P2 | Nightly bars-refresh unit stuck failed | `chad-ibkr-daily-bars-refresh.service`: `inactive (dead)`, last run 2026-07-23 02:46:15, **`status=0/SUCCESS`** | **STALE** |
| P3 | EXS5 standing fail is caused by bars-refresh | EXS5's only failed CHAD unit is **`chad-xgb-train.service`** (failed since 2026-07-19, `status=1/FAILURE`). Bars-refresh is not failed | **STALE — and blocks goal (5)** |
| P4 | gamma_futures + alpha_futures starved of data | Both need `min_bars=40` (`alpha_futures.py:74`, `gamma_futures.py:166`). Universes resolve to symbols with 209–251 bars. Neither strategy consumes M6E or SIL at all | **STALE** |
| P5 | Per-symbol failure fails the whole refresh unit | Per-symbol *work* is already isolated (`nightly_bars_refresh.py:245-257`); per-symbol *exit status* is **not** (`:284-285`) | **HALF-TRUE — real** |

### Why it self-healed, and why it will re-break

No futures-resolution fix has landed since June — `git log --since=2026-06-01` on
`ibkr_historical_provider.py`, `nightly_bars_refresh.py`, `futures_contract_resolver.py`
shows only `e17ea24` (Kraken freshness marker) and `ee3e5f1` (UTC co-writer). **The
system self-healed by calendar drift**: the hardcoded expiry schedules simply advanced
past the months that were expired in June. Nothing was repaired. The same class of
failure returns on a fixed, known schedule — see D4 below.

### What is actually broken RIGHT NOW

Measured, not inferred:

```
SYM   bars  last bar     median 20d volume   resolved front month
MES    250  2026-07-22        810,410              202609
MNQ    250  2026-07-22      1,974,468              202609
MGC    251  2026-07-22        107,654              202608
MCL    249  2026-07-21 <--        42,552           202608  <-- EXPIRED CONTRACT
MYM    209  2026-07-22        107,195              202609
M2K    250  2026-07-22         85,454              202609
ZN     145  2026-07-22        976,393              202609
ZB     145  2026-07-22        193,538              202609
M6E     88  2026-07-22          6,433              202609
SIL     37  2026-07-22            179  <-- 600x thinner than MGC   202608
```

Two live defects fall out immediately: **MCL is pointed at a contract that expired two
days ago** (it is the only symbol a day behind), and **SIL is pointed at an illiquid
serial month** (179 contracts/day). Details in §2.

---

## 2. ROOT CAUSE — per symbol, with evidence

### The resolution machinery (three independent paths, two duplicated tables)

| Path | Table | Consumer |
|---|---|---|
| A | `IBKRHistoricalProvider._EXPIRY_SCHEDULE` (`ibkr_historical_provider.py:166-181`) | nightly daily-bars refresh |
| B | `futures_contract_resolver.EXPIRY_SCHEDULE` (`futures_contract_resolver.py:31-46`) | hot-path contract resolution |
| C | `runtime/futures_roll_state.json` via `futures_expiry_gate` / `roll_gate` | live 1m bar provider + `alpha_futures.py:567` |

A and B are **byte-identical copies**, self-documented as a "mirror copy"
(`futures_contract_resolver.py:19`). I verified they agree today for all 10 symbols
(no divergence yet) — but nothing enforces that. Two hand-maintained calendars is the
structural defect underneath every symbol-level defect below.

### D1 — MCL: resolver returns an EXPIRED contract *(live, today)*

`_resolve_front_month` (`ibkr_historical_provider.py:201-226`) compares at **YYYYMM
granularity** with a crude day-of-month heuristic (`cutoff_day <= 20`, line 224).
For MCL the schedule entries are **delivery months**, but the routine treats them as
**expiry months**. The code's own comment (`:169-172`) documents that NYMEX crude
expires in the month *before* delivery — MCLM6 (June delivery) last traded 2026-05-18 —
yet the comparison logic cannot express that offset.

Today: `cutoff_str="202607"`, `cutoff_day=28 > 20` → skip 202607 → return **202608**.
The Aug-delivery contract last traded ~2026-07-21. **Evidence: MCL is the only symbol
whose last bar is 2026-07-21 while all nine others are 2026-07-22.** MCL has already
begun to go stale and will start generating Error 162 / empty responses.

MCL is `gamma_futures`' **primary** symbol (`gamma_futures.py:86`).

### D2 — SIL: resolves to an illiquid serial month, and has no roll-state entry at all

Two compounding faults:

1. Resolution picks **202608** (August silver), a low-volume serial month. Evidence:
   median 20-day volume **179 contracts** vs MGC's 107,654; the file holds only 37 bars
   from 2026-05-29. `min_bars=40` → **SIL is the one symbol genuinely below threshold.**
2. `SIL` is **absent from the roll publisher's publish universe**
   (`futures_roll_publisher.py:41-44` lists 9 symbols, no SIL). So `futures_expiry_gate`
   takes the `no_roll_mapping` branch with `skip=False` (`futures_expiry_gate.py:115-124`)
   and polls SIL unconditionally. **This is precisely the failure the gate was built to
   prevent** — the module docstring (`:5-8`) cites ~588 Error 162/day after SILK6 expired.
   The guard cannot cover the symbol it was written for.

Also to verify against secdef (flagged, not asserted): `SIL` maps to
`ibkr_symbol="SI", multiplier=1000` (`:196`) while `tradingClass` is set to `SIL`.
SI is the 5,000oz contract; SIL is the 1,000oz micro. The mapping may be resolving
correctly via tradingClass, but the spec row is internally inconsistent.

### D3 — M6E / ZB / ZN / MYM: short history is inherent, not a bug

`M6E n=88` (from 2026-03-17), `ZB/ZN n=145` (from 2025-12-22), `MYM n=209`
(from 2025-09-22). These are **per-contract** histories: each series begins when that
specific contract was listed. The fetch requests `days=400`
(`nightly_bars_refresh.py:248`); IBKR returns only what that contract has.

This is correct front-month behaviour, **not a resolution failure**. All four are far
above `min_bars=40`. No fix required for signal readiness. It becomes a real constraint
only if a future consumer needs >145 bars — noted as a deferred item (§7 D5), not built
here.

### D4 — Schedule exhaustion cliff: silent, dated, and guaranteed

Both hardcoded tables end, and both fall back to `return schedule[-1]`
(`ibkr_historical_provider.py:226`, `futures_contract_resolver.py:73`) — i.e. **on
exhaustion they silently return the last known, by-then-expired month.** No exception,
no log, no alarm. Dates:

| Symbol | Last scheduled month | Silent-failure begins |
|---|---|---|
| SIL | 202612 | **~Dec 2026** |
| MCL | 202701 | ~Jan 2027 |
| ZN, ZB, M6E | 202703 | ~Mar 2027 |
| MGC | 202704 | ~Apr 2027 |
| MES, MNQ, MYM, M2K | 202706 | ~Jun 2027 |

This is the June failure, pre-scheduled to recur. Fixing it is the substance of this lane.

### D5 — Exit-code coupling keeps the unit red on any single bad symbol

`nightly_bars_refresh.py:245-257` isolates per-symbol *work* correctly (try/except per
future, one bad symbol cannot abort the loop). But `main()` aggregates:

```python
if failure > 0:   # :284
    return 1
```

One failed symbol out of 52 → unit exits 1 → systemd `failed` → EXS5 red. **Work
isolation without status isolation.** This is the mechanism behind the June standing
fail, and it is still present.

### D6 — `test_futures_expiry_gate` failure is a wall-clock time bomb, not a product bug

Current known-fail: `test_bar_provider_skips_expired_in_polling_loop`
(`chad/tests/test_futures_expiry_gate.py:182`). It stubs `MES current_expiry=2026-06-19`
and asserts MES is still polled. That date is now in the past, so the gate correctly
skips MES too → `assert 'MES' in ['M2K']` fails.

Root cause: `poll_once` → `_filter_expired_futures()` (`ibkr_bar_provider.py:553`)
threads **no `now` parameter**, so the gate always reads the real wall clock. The gate
itself already accepts `now` (`futures_expiry_gate.evaluate_symbol`) — only the
provider call path drops it.

**Correction to the brief:** `test_gap037_reconciler_futures` does not exist; the file
is `test_gap037_reconciler_futures_contract.py`, and it **passes** — 35 passed / 1 failed
across both files. Only the one test above needs converting.

---

## 3. FIX DESIGN

Design goals: one source of truth, fail **loud** instead of silent, per-symbol
isolation at every layer, and no new network dependency in the hot path.

### W6A-1 — Single expiry source of truth (kill the mirror)

Make `futures_contract_resolver` the sole owner. `IBKRHistoricalProvider._EXPIRY_SCHEDULE`
is deleted and `_resolve_front_month` delegates to `resolve_contract_month()`.
A named test asserts the provider has no private schedule table, so the mirror cannot
silently reappear.

### W6A-2 — Roll-aware resolution with an explicit expiry rule per family

Replace month-string comparison with a **last-trade-date rule per contract family**,
so MCL's delivery-vs-expiry offset is expressed in code rather than in a comment:

| Family | Rule |
|---|---|
| Equity-index micros (MES MNQ MYM M2K) | quarterly, 3rd Friday of contract month |
| Rates (ZN ZB) | quarterly, ~3rd-Friday convention |
| Energy (MCL) | last-trade ≈ 3 business days before the 25th of the month **prior** to delivery |
| Metals (MGC SIL) | last-trade ≈ 3rd-last business day of the month prior to delivery |
| FX micros (M6E) | quarterly, 2 business days before 3rd Wednesday |

Resolution then means: *first contract whose computed last-trade-date is ≥ today + roll
buffer*. This fixes D1 directly and makes the `cutoff_day <= 20` heuristic unnecessary.

**Reuse of B4, and its honest limit.** `futures_roll_publisher` (B4) already computes
third-Friday quarterly expiries — `third_friday()`, `next_quarterly_expiry()`,
`_quarterly_expiry_after()` (`:48-70`). W6A-2 **imports and reuses these** for the
equity-index family rather than reimplementing. But B4 covers only 4 of 10 symbols:
`_SUPPORTED_QUARTERLY_EQUITY_INDEX = {MES, MNQ, MYM, M2K}` (`:31-33`); MCL/MGC/ZN/ZB/M6E
are emitted as `unsupported_v1` and SIL is not published at all. **So B4 can be reused,
but cannot today be the source of truth for the 6 symbols that need it most.** W6A-3
closes that gap.

### W6A-3 — Extend B4 coverage; add SIL to the publish universe

Add the non-equity-index families to `futures_roll_publisher` using the W6A-2 rules, and
add `SIL` to `_DEFAULT_SYMBOLS`. This gives `futures_expiry_gate` a real entry for every
symbol, closing the `no_roll_mapping` hole that leaves SIL unguarded (D2.2).

**Risk to control:** `roll_gate.check_roll_gate` is consumed by `alpha_futures.py:567`,
and `block_new_entries=True` only bites when `roll_supported=True`
(`roll_gate.py:141`). Flipping symbols from unsupported→supported can therefore begin
**blocking entries** that previously passed. Mitigation: land W6A-3 with the new families
publishing `roll_supported=true` for observability but `block_new_entries` computed and
**recorded** while the blocking arm stays behind a default-off flag until one full roll
cycle has been observed. Decision point §7 D2.

### W6A-4 — Liquidity-aware month selection for SIL

Restrict SIL (and MGC) to the exchange's liquid cycle rather than every listed serial
month, so resolution lands on Sep silver rather than Aug. Implemented as an explicit
per-symbol allowed-months set in the resolver, verified against secdef in W6A-8. Also
resolves the `SI`/`SIL` spec inconsistency (D2 note) or documents why it is correct.

### W6A-5 — Fail loud on schedule exhaustion

Delete both `return schedule[-1]` fallbacks. On exhaustion, return `None` and raise a
named, catchable error at the call site so the symbol fails **visibly** (isolated per
W6A-6) instead of silently fetching an expired contract. Add a horizon check that warns
when any symbol has fewer than N future months remaining, so the D4 cliff is announced
months ahead rather than discovered by outage. With W6A-2's computed rules the tables
become generative, so exhaustion should stop being a recurring class at all — the check
is the backstop.

### W6A-6 — Per-symbol exit-status isolation (the actual EXS5 mechanism)

Change `nightly_bars_refresh.main()` so **non-fatal per-symbol failures do not fail the
unit**:

- exit 0 — all symbols succeeded, or failures are within a declared tolerance
- exit 1 — a *systemic* failure (IBKR connection dead, import error, all symbols failed)
- exit 2 — retained for connection failure

Per-symbol failures are written to a machine-readable status artifact
(`bars_refresh_state.v1`) with `ok`/`failed` lists and timestamps, so a single bad
contract is **visible** without turning the whole unit red. Freshness/coverage then
becomes a data-quality signal, not a systemd-failure signal. Tolerance policy is
decision point §7 D1.

---

## 4. BARS BACKFILL — bounded and idempotent

Only two symbols need anything: **SIL** (37 bars, below `min_bars=40`, wrong month) and
**MCL** (stale since 2026-07-21, expired contract). Backfill is a consequence of
correct resolution, not a separate data migration.

- **Tool:** `ops/futures_bars_backfill.py` — dry-run by default, `--execute` required,
  `--symbols` explicit (no implicit all-symbol sweep).
- **Bounded:** max 400 calendar days per symbol (matches existing `days=400`),
  hard cap on IBKR requests per run, sequential with existing pacing delays.
- **Idempotent:** writes via the existing `_write_bars_file` atomic path; re-running
  with unchanged inputs produces a byte-identical file. Merge is by `ts_utc` key —
  existing bars are never duplicated.
- **Archive-before-mutate:** copies the current `data/bars/1d/<SYM>.json` to a timestamped
  archive before any write, consistent with the epoch-reset bootstrap tool's convention.
- **Non-destructive on failure:** partial fetch leaves the existing file untouched.
- **Continuity note:** backfill does **not** create a back-adjusted continuous series.
  Each symbol remains a single-contract history. Stitching is deferred (§7 D5).

Runs against the worktree copy first; touching `data/bars/1d/` in the live tree is a
separate operator GO (§7 D3).

---

## 5. WAKE VERIFICATION — signal-path only, no execution

**Already measured (read-only probe, this session, against today's real bars):**

- Universes resolve correctly with live data: `alpha_futures` → MES/MNQ/MGC;
  `gamma_futures` → MCL/MYM/M2K. All six have 209–251 bars, well above `min_bars=40`.
- Both strategies **execute their full evaluation path and produce 0 signals** — not
  because of missing data, but because no setup triggered and the session gate fired
  (`OVERNIGHT_GATE_SKIP symbol=MGC hour=23`, `alpha_futures.py:523-548`; probe ran
  23:25 UTC, outside the 13:30–20:00 UTC RTH window).

**So the strategies are not data-starved. They are gated.** Two gates, both correct and
both out of this lane's control:

1. **Regime roster.** `runtime/regime_state.json` reports `regime="ranging"`, and
   `config/regime_activation_matrix.json` excludes both `alpha_futures` and
   `gamma_futures` from the `ranging` roster. They are on-roster in
   trending_bull / trending_bear / volatile / unknown. (`omega_macro` — the actual
   consumer of ZN/ZB/M6E — **is** on the ranging roster and is live now.)
2. **Operator intent.** `live_gate.operator_mode=EXIT_ONLY` per INCIDENT-0723
   ("entries frozen pending book repair") denies all entry lanes regardless.

**Deliverable W6A-7 — fixture proof.** A test that pins a deterministic fixture
(frozen clock inside RTH, synthetic regime `trending_bull`, real bar shapes) and asserts
both strategies emit a well-formed `TradeSignal` for at least one symbol. This proves
the signal path end-to-end **without touching regime state, operator intent, or the
execution gate.** No live behaviour changes; no order can be produced.

The honest statement for the closure record: *waking these strategies in production is a
regime and operator-intent matter, not a data matter. This lane can prove the data and
signal path are sound; it cannot and should not un-bench them.*

---

## 6. EXS5 CLOSURE — cannot be achieved by this lane alone

`check_failed_services` (`exterminator_sentinel.py:961-986`) returns FAIL if **any**
`chad-`-prefixed unit is in the failed state. Current state:

```
● chad-xgb-train.service   loaded failed failed   CHAD XGBoost Model Retraining
                           (failed since 2026-07-19, status=1/FAILURE)
```

**That is the only failed CHAD unit.** `chad-ibkr-daily-bars-refresh.service` is not
failed — it exited 0 this morning.

Therefore: **W6A-6 removes the bars-refresh unit's ability to go red on a single bad
contract, which is the durable half of the EXS5 goal. But EXS5 will remain FAIL until
`chad-xgb-train.service` is separately repaired**, which is a different subsystem
(XGB retraining — see the dormant-feature census's `~99% MES loss-prob keep-OFF` note)
and is **not** in this lane. Flagging rather than scope-creeping. Decision point §7 D4.

Deliverable here: `W6A-9` records the EXS5 dependency explicitly in the closure record
so the sentinel row's continued red is attributed correctly and not mistaken for a
futures-data regression.

---

## 7. DECISION POINTS — operator input requested

**D1 — Bars-refresh failure tolerance (W6A-6).** When should the unit exit non-zero?
 (a) only on systemic failure — any number of per-symbol failures exits 0, surfaced via
 the status artifact; (b) tolerance threshold — e.g. >20% of symbols failed exits 1;
 (c) futures-only tolerance — equities/crypto failures still fail the unit.
 *Recommendation: (a).* It is the cleanest separation of "unit broken" from "one contract
 unavailable", and the status artifact plus a data-freshness check carries the signal.

**D2 — Roll-gate blocking arm (W6A-3).** Extending B4 to MCL/MGC/ZN/ZB/M6E/SIL flips
them `roll_supported=true`, which activates `block_new_entries` in `alpha_futures`.
 (a) observe-only for one roll cycle, blocking behind a default-off flag;
 (b) enable blocking immediately.
 *Recommendation: (a).* Consistent with shadow-first practice across W4/W5, and a
 miscomputed expiry rule that blocks entries is worse than one that merely reports.

**D3 — Backfill target (W6A / §4).** Worktree-only proof, or operator GO to write
`data/bars/1d/{SIL,MCL}.json` in the live tree? *Recommendation: worktree first,
then a separate explicit GO.* MCL is `gamma_futures`' primary symbol and is already
stale, so the live write has real urgency once the resolver is verified.

**D4 — `chad-xgb-train.service` (§6).** Out of scope as written. Options: (a) leave for
a separate lane — EXS5 stays red; (b) add a bounded diagnosis-only sub-item to W6A
(no fix, just root-cause + finding); (c) fold the fix into this lane.
 *Recommendation: (b).* Cheap, and it lets the closure record state exactly what stands
 between EXS5 and green without expanding this lane's change surface.

**D5 — Continuous back-adjusted series (D3/§4).** M6E/ZB/ZN histories are 88–145 bars
because they are single-contract. Building a stitched continuous series is a
substantially larger piece of work with its own correctness burden (back-adjustment
method, split points, and a real risk of manufacturing PnL-looking artifacts in the
harness). *Recommendation: DEFER, file as a W6B candidate.* Nothing today needs >145 bars.

**D6 — Scope confirmation.** Given P1–P4 are stale, does the operator want the lane
retargeted as scoped above (MCL expiry + SIL month/coverage + exhaustion cliff +
exit-status isolation + fixture proof), or narrowed?
 *Recommendation: proceed as scoped.* D1/D2/D4/D5 are real, dated, and currently
 unguarded — the lane's value survives the stale premises intact.

---

## 8. WORK BREAKDOWN & VERIFICATION

| ID | Item | Type |
|---|---|---|
| W6A-0 | This plan | doc (committed alone) |
| W6A-1 | Single expiry source of truth; delete mirror table | refactor |
| W6A-2 | Family-based last-trade-date resolution; reuse B4 helpers | fix (D1) |
| W6A-3 | Extend roll publisher coverage; add SIL to universe | fix (D2.2) |
| W6A-4 | Liquidity-aware month selection for SIL/MGC | fix (D2.1) |
| W6A-5 | Fail loud on exhaustion + horizon warning | fix (D4) |
| W6A-6 | Per-symbol exit-status isolation + `bars_refresh_state.v1` | fix (D5) |
| W6A-7 | Fixture signal-path proof for gamma/alpha_futures | test |
| W6A-8 | secdef verification of resolved contracts (read-only) | verification |
| W6A-9 | Closure record incl. EXS5 attribution | doc |

**Named tests to convert (currently failing):**
- `test_bar_provider_skips_expired_in_polling_loop` — thread an optional `now` through
  `poll_once` → `_filter_expired_futures` → `filter_universe` (the gate already accepts
  it), then freeze the clock in the test. Removes the wall-clock time bomb.

**Set-diff methodology.** Baseline captured on `goal/wave6-futures` at `main@c76fbea`
before any change: **16 failed, 4498 passed, 5 skipped** (156s) — consistent with the
standing 16-fail baseline carried since W5A. Every subsequent verification compares the
*set* of failing test IDs against that baseline — pass/fail counts alone are not
accepted as evidence. Target: baseline fail-set minus
`test_bar_provider_skips_expired_in_polling_loop` (→15), no additions.

Recorded baseline fail-set (Appendix A) — the lane converts exactly one row:

```
test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes
test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage
test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop   <-- W6A converts
test_kraken_execution.py::test_intent_builder_btc_basic
test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified
test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only
test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only
test_repo_write_guard.py::test_guard_blocks_direct_write_under_data
test_repo_write_guard.py::test_guard_blocks_write_under_runtime
test_repo_write_guard.py::test_guard_blocks_mkdir_under_data
test_repo_write_guard.py::test_guard_reraises_even_when_caller_swallows
test_repo_write_guard.py::test_baseline_blocks_all_but_records_only_new_sinks
test_repo_write_guard.py::test_grandfathered_write_is_blocked_but_not_recorded
test_routing_gates.py::test_e4_kraken_passive_order_params
test_routing_gates.py::test_e4_kraken_aggressive_order_params
test_w4b8_flatten_bare_terminal.py::test_bare_terminal_drill_reaches_drill_complete
```

`test_gap037_reconciler_futures_contract.py` does **not** appear — it passes in full,
confirming the §2 D6 correction to the brief.

**Per-change verification sequence** (governance §4), run in the worktree:
```
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q
CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview
```

**Standing constraints:** one change at a time; surgical edits only; no runtime
mutation; no config mutation (Pending Actions only); no systemd edits; no service
restarts; futures execution gate untouched.

---

## 9. EVIDENCE INDEX

| Claim | Source |
|---|---|
| Unit exited 0 today | `systemctl status chad-ibkr-daily-bars-refresh.service` — `status=0/SUCCESS`, 2026-07-23 02:46:15 |
| EXS5 red = xgb-train | `systemctl list-units --state=failed` — sole `chad-` unit |
| All 10 futures have bars | `data/bars/1d/*.json` inventory, §1 table |
| MCL on expired contract | last bar 2026-07-21 vs 2026-07-22 for all others; `_resolve_front_month('MCL')='202608'`; `ibkr_historical_provider.py:169-172` |
| SIL illiquid month | median 20d volume 179 vs MGC 107,654; 37 bars from 2026-05-29 |
| SIL absent from roll calendar | `futures_roll_publisher.py:41-44` |
| Mirror schedules identical | both paths resolve identically for all 10 symbols, verified 2026-07-23 |
| Exhaustion returns expired month | `ibkr_historical_provider.py:226`, `futures_contract_resolver.py:73` |
| Exit-status coupling | `nightly_bars_refresh.py:284-285` |
| min_bars=40 | `alpha_futures.py:74`, `gamma_futures.py:166` |
| Strategies run, emit 0, session-gated | read-only probe; `OVERNIGHT_GATE_SKIP symbol=MGC hour=23` |
| Regime bench | `runtime/regime_state.json` `regime="ranging"`; `config/regime_activation_matrix.json` ranging roster |
| Execution gate location | `/etc/systemd/system/chad-live-loop.service.d/91-disable-futures-exec.conf`; `chad/execution/futures_gate.py:45-47` |
| No futures fix since June | `git log --since=2026-06-01` on the three resolution files |
