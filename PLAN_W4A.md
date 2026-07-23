# PLAN_W4A — Wave 4 Lane A: The Fuse Box (LC2 / LC3 / LC5-ENFORCE + DQ Policies)

**Phase 1 — PLAN ONLY.** No code in this commit. Worktree `chad_w4a`, branch
`goal/wave4-fusebox`, base `main@6979b99` (all prior wave branches merged; verified).
Live tree untouched. Phase 2 begins only after operator decisions D1–D8.

Method: six parallel read-only audits over the worktree (loss-guard/edge-decay,
drawdown machinery, order pipeline/exit overlay, ops-event conventions, LC2/LC3
data sources, feed-freshness/DQ) before any design. Several tasked premises were
stale or under-stated; §1 records the verified ground truth this plan is built on.

---

## 0. Scope and non-negotiables

- Four items, ALL flag-gated default-OFF. Flag flips are separate operator GOs —
  nothing in W4A changes live behavior at merge time.
- One change at a time; set-diff test methodology (§8); commits prefixed `W4A`.
- No runtime mutation, no systemd changes, no config mutation outside the worktree.
  (Live-tree caveat: `config/edge_decay_config.json` carries an APPLIED-but-uncommitted
  v2 — `clear_on_recovery`, `halt_ttl_days:14` — the W1B PA. W4A must not touch that
  file; the fuse box gets its own config.)
- **Prime invariant: a fuse must never block a reduce-only close.** Enforced
  structurally (§5.1), not by convention.

## 1. Audited ground truth (premise verification)

| # | Premise / claim | Verdict | Evidence |
|---|---|---|---|
| P1 | "LC4-adjacent machinery exists (loss-guard enforce, edge-decay halts w/ self-clear)" | VERIFIED, and richer: LC2's $-leg **already exists** as `PerStrategyLossGuard` (GAP-026) — per-strategy session $-loss limits, `config/per_strategy_loss_limits.json`, enforce env `CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE` (default report_only), protective-signal carve-out, self-clear at UTC-midnight/epoch | `chad/risk/per_strategy_loss_guard.py`; wired `chad/core/live_loop.py:2009-2048` |
| P2 | LC3 has a **dormant prototype**: `symbol_performance_blocker` — 3 consecutive losing closes per symbol → 2h block, exit/flip bypass, enforced in-loop | VERIFIED dormant: enforcement hook live (`live_loop.py:2759-2767`) but writer is a CLI nobody schedules; **unsafe to copy** — no pnl_untrusted/quarantine filtering (would count the +625.17 seed close and PFF1 phantoms), hardcoded `/home/ubuntu/chad_finale` REPO_ROOT | `chad/risk/symbol_performance_blocker.py:41,84-122` |
| P3 | "drawdown_state publishes fresh on a 120s timer" | VERIFIED: `chad-drawdown-publisher.timer` OnUnitActiveSec=120, file TTL 300 | `ops/drawdown_publisher.py:40-87`, `ops/systemd/chad-drawdown-publisher.timer:5-6` |
| P4 | "−15% HWM machinery currently reports only" | VERIFIED: zero trading-side readers of `drawdown_state.json`; `halt` boolean computed (`chad/risk/drawdown_guard.py:245`) and consumed by nothing; `enforcement_active` hardcoded false | repo-wide reader census: metrics/watchdog/sentinel/epoch-reset only |
| P5 | "A4 staleness guard mandatory" | A4 guard EXISTS but metrics-layer only (W3B-2: `metrics_server.py:530-549,593-613`, 900s). LC5 needs its **own** decision-time staleness policy; `_compute_state_staleness` + `runtime_json.read_runtime_state_json` are the reusable patterns | — |
| P6 | 5d/20d drawdown budgets | **DO NOT EXIST.** Only the 60d rolling HWM. `equity_history.ndjson` has ~22-23 daily CAD rows — enough for 5d, marginal for 20d (enforcement must gate on sample depth) | `drawdown_guard.py:46,205-245` |
| P7 | Drawdown trust | Publisher has NO contamination defense; `hwm_cad` is the H3 phantom-HWM carrier; equity history is Bug-B volatile. LC5 must not shrink off a phantom peak unguarded | `ops/epoch_reset_bootstrap.py:106-111` |
| P8 | Enforcement placement | **Two-pipeline gap**: signal-level filters (pipeline-A) are bypassable — submitted intents are rebuilt by `_build_plan_and_intents` (pipeline-B) which consults neither loss guard nor halts. Airtight enforcement = per-intent layer (stage-3 gate cluster `live_loop.py:2665+`) or adapter | `chad/core/ibkr_execution_runner.py:392-472` |
| P9 | Exit-overlay interaction | Overlay closes **never pass through the stage-3 intent gates** — `run_cycle → apply_close_intents → adapter` directly. A stage-3 fuse gate is entry-scoped *by construction*. At the adapter, IBKR overlay closes are **indistinguishable** (close stamp dropped in `_close_intent_to_ibkr`); Kraken closes carry `reduce_only=True` + `CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE` markers | `chad/core/position_reconciler.py:324-397`; `chad/risk/crypto_exit_overlay.py:373-424` |
| P10 | Existing exit-blocking traps (do not replicate) | Edge-decay's signal filter drops a halted strategy's exits (no protective carve-out); regime activation matrix has no exit exemption; futures off-switch blocks both sides deliberately | `live_loop.py:1828-1839,2548-2605,2935-2948` |
| P11 | Wiring template | Margin shadow gate (Phase C) is the exact pattern: injectable gate, ctor-default None, pre-claim hook, `should_block()` always-False in shadow, marker + dated ndjson evidence under `data/<name>/`, fail-open shadow / fail-closed enforce, pytest evidence-dir guard, Kraken mirror | `chad/execution/margin_shadow_gate.py`; hook `ibkr_adapter.py:2692-2694`; `kraken_executor.py:366-376` |
| P12 | Trusted loss substrate | `closed_trade.v1` ledger (`data/trades/trade_history_*.ndjson`) is THE source, **with predicates** (§4.1). `equity_history` confirmed unusable. Fills are per-leg (no PnL). Kraken TrustedFillEngine rows are trusted but **zero exist yet**, and its writer omits `schema_version` (D6-gate structural exclusion — noted, not W4A's to fix) | data-source audit §2 |
| P13 | setup_family | Already plumbed end-to-end for 4 strategies (`alpha`, `alpha_futures`, `alpha_intraday`, `alpha_intraday_micro`): signal meta → position_guard → trade_closer → `closed_trade.v1 meta.setup_family`. No strategy-level "family" concept exists anywhere — needs a config map | `chad/execution/trade_closer.py:314-320` |
| P14 | Regime in ledger | `closed_trade.v1` has **no regime field** (live rows: `regime=None`). Live classifier vocabulary: `trending_bull/trending_bear/ranging/volatile/unknown`, `runtime/regime_state.json` TTL 360 | `chad/analytics/regime_classifier.py:62-64` |
| P15 | Symbol→sector | **No map exists anywhere.** Universe: 38 equities/ETFs + 10 futures roots + 3 crypto pairs | `config/universe.json` |
| P16 | trade_closer topology | Closes mint in a SEPARATE process (oneshot timer, every 60s). Fuse state must be a published runtime artifact read fail-safe by live_loop — the edge-decay/symbol-blocker pattern | `deploy/chad-trade-closer.timer` |
| P17 | DQ precedent | No config anywhere declares a stale *policy* — every block/reduce/degrade is hard-coded per reader. `config/exterminator.json feeds{}` is ~90% of the needed schema (lacks a `policy` field + reader-side hook). All three policy archetypes already exist in production: **block** (`positions_truth` futures caps), **reduce** (`margin_block` stale ⇒ reduce-only-allowed), **degrade** (`price_cache` ⇒ bar-close fallback) | DQ audit §§2-4 |
| P18 | DQ highest-value gaps | SCR hard-block gate reads `scr_state.json` **raw, no ts/ttl check, fail-open** (frozen PAUSED blocks forever; corrupt file silently disables the governor). Stop-bus inputs: dead `ibkr_status.json` silently disables the latency halt | `live_loop.py:2665-2698,609-661` |
| P19 | Conventions | Coach `format_alert` registry + sanctioned generic-kind fallback; dedupe = `runtime/dedupe/`, stable identity (rule-id + entity, digits stripped, values never in titles); sentinel visibility = `ts_utc`+`ttl_seconds` in file + feeds row in `config/exterminator.json` (`ttl_verified:false` ⇒ warn-cap); state canon = `schema_version:"<stem>.vN"` via `chad/utils/runtime_json.py` | conventions audit |

## 2. Architecture — one spine, four fuses

New module family under `chad/risk/`:

```
chad/risk/fuse_box.py            # core: FuseBoxState, trusted-loss counting, trip/clear engine
chad/risk/fuse_gate.py           # per-intent gate object (margin-gate pattern), IBKR + Kraken views
config/fuse_box.json             # families, per-bucket thresholds, LC5 ladder, modes
config/symbol_sectors.json       # LC3 static sector/theme map (D3)
config/feed_policies.json        # DQ1/DQ3 per-feed class + stale policy (D7)
runtime/fuse_box_state.json      # fuse_box_state.v1 — published every cycle incl. OFF (heartbeat doctrine)
data/fuse_box/fuse_box_YYYYMMDD.ndjson   # evidence: trips, clears, would_block (shadow), staleness events
```

**Evaluator placement (P16):** inside `live_loop.run_once`, immediately after the
existing loss-guard block (~`live_loop.py:2048`) — re-derives per-bucket session
stats from the trusted ledger each cycle (the stateless GAP-026 pattern), diffs
against prior `fuse_box_state.json` for edge-triggered trip/clear events, writes
state atomically. No new systemd units. Failure-soft: an evaluator exception never
kills the cycle (loss-guard precedent `live_loop.py:2045-2048`).

**Enforcement placement (P8+P9):** stage-3 per-intent gate, inserted beside the
symbol blocker (~`live_loop.py:2759`), sharing its bypass predicate
(`is_flip_signal` / side ∈ {EXIT, CLOSE}) **plus** the loss-guard's
`is_protective_signal` tags (reduce/hedge/stop_loss/liquidation). Kraken mirror in
`KrakenExecutor.execute_with_risk` before `check_risk` (margin-gate slot),
exempting `reduce_only` intents and `crypto_exit|`-keyed / overlay-marked intents.
Overlay closes never reach either point (P9) — the invariant holds structurally,
and the predicate bypass covers strategy-emitted exits on top.

**Modes:** per-item env flags, house tri-state, garbage → off:

| Flag | Values | Default | Gates |
|---|---|---|---|
| `CHAD_FUSE_LC2` | off \| shadow \| enforce | off | setup-family/strategy-family fuses |
| `CHAD_FUSE_LC3` | off \| shadow \| enforce | off | symbol + sector fuses |
| `CHAD_FUSE_LC5` | off \| shadow \| enforce | off | progressive drawdown sizing |
| `CHAD_DQ_POLICIES` | off \| shadow \| enforce | off | per-feed stale policies |

Shadow = full evaluation, state + would_block/would_shrink evidence, zero behavior
change (`should_block()` returns False — margin-gate contract). Enforce is a
separate operator GO per flag. Factories replicate the pytest leak guards
(explicit evidence/state dirs under `PYTEST_CURRENT_TEST`; conftest write-guard
compatibility).

## 3. LC2 — per-setup daily stop

**Buckets** (config-defined in `config/fuse_box.json`):
- `family:<name>` — named groups over `strategy_id`s (new concept, config map over
  the 18 registry ids; registry-perimeter warn-check like invariant 7 does for
  loss limits).
- `setup:<strategy_id>:<setup_family>` — sub-strategy grain for the 4 stamped
  strategies (P13), free to enable per-family since the ledger already carries it.

**Trip condition** (per bucket, per session window = `max(UTC midnight,
epoch_started_at_utc)` — the GAP-026 window): `consecutive_losers >= N` (default 3,
symbol-blocker precedent) OR `session_net_pnl <= -$X` (defaults seeded from
`per_strategy_loss_limits.json` scale). Optional regime scoping per bucket
(`regimes: ["ranging", ...]`): losses only count toward a bucket when the close's
regime matches — mechanics per D2.

**Counting substrate** — trusted predicates (P12), all already precedented:
1. `schema_version` startswith `closed_trade.` (extend to `paper_trade_result.`
   when the crypto lane's schema gap is resolved — out of W4A scope);
2. not quarantined (`chad/utils/quarantine.py:268-335`, operator manifests +
   sidecars + RECON_ADOPT seed lots);
3. no `pnl_untrusted` / `scoring_excluded` / `validate_only` in extra/tags/meta;
4. strategy ∉ {broker_sync, manual, paper_exec, unknown, ""}; no manual/warmup_sim tag;
5. futures rows excluded until Bug-B disposition (`futures_bug_b`, adapter precedent);
6. PnL = `net_pnl` (fallback `pnl`), keyed on `exit_time_utc`.
A "loser" = trusted close with pnl < 0 (pnl == 0 scratches don't extend or reset
streaks — edge-decay precedent).

**Clear:** self-clear at next session window (counter derivation makes this
automatic); tripped→cleared transition is edge-detected against prior state and
emits the clear event. No manual step required; an operator CLI
(`scripts/clear_fuse.py`, mirroring `clear_edge_decay.py` provenance rules) ships
for early manual clears, preserving `previous_*` counters (GAP-018 idiom).

## 4. LC3 — per-symbol / per-sector stop (anti-revenge)

Same engine, two more bucket kinds:
- `symbol:<SYM>` — consecutive trusted losing closes on one symbol (default N=3,
  $X optional) within the session window.
- `sector:<name>` — sum over the symbols mapped to a sector/theme in
  `config/symbol_sectors.json` (new static map; scope = the 38-symbol universe +
  10 futures roots grouped as equity_index/energy/metals/rates/fx + crypto).
  Unmapped symbol ⇒ `sector:unmapped` bucket, count-only, never trips (loud in
  evidence) — a missing map row must not create a blockable bucket nobody ratified.

**Relationship to the dormant `symbol_performance_blocker` (D3):** recommend
SUBSUME — the fuse gate sits beside the blocker's hook; the blocker is left
untouched (its enforcement no-ops while its state file stays empty). Retirement is
a separate cleanup after LC3 proves out in shadow. Rationale: the blocker's counting
is untrusted (P2) and its 2h-TTL clear conflicts with session-scoped semantics.

**Clear:** next session, same as LC2. (The blocker's 2h re-arm is NOT copied;
anti-revenge means the ticker stays off the menu for the session.)

## 5. LC5 — drawdown ENFORCEMENT (progressive, not cliff)

### 5.1 New budgets (publisher side, additive)
Extend `drawdown_guard.compute_drawdown` + publisher to emit, alongside v1 fields:
`dd_5d_pct`, `dd_20d_pct` (drawdown vs the max of the trailing 5/20 dated
equity rows), `sample_count_5d`, `sample_count_20d`. Schema disposition per D6
(recommend bump to `drawdown_state.v2` + update the EXS7 pin and EXS1 row in the
same commit). Trust guards (P7):
- enforcement ignores any window whose `sample_count < window` (insufficient rows
  ⇒ that budget reports but never enforces);
- HWM sanity: rows preceding `epoch_started_at_utc` are excluded from the 5d/20d
  windows (60d HWM/−15% legacy fields unchanged — report-only as today).

### 5.2 Progressive ladder (config, in `fuse_box.json`)
```
"lc5_ladder": {
  "dd_5d":  [ {"at_pct": -3.0, "factor": 0.75}, {"at_pct": -5.0, "factor": 0.50}, {"at_pct": -8.0, "factor": 0.25} ],
  "dd_20d": [ {"at_pct": -8.0, "factor": 0.50}, {"at_pct": -12.0, "factor": 0.25} ],
  "emergency_halt_pct": -15.0
}
```
Effective factor = min(rung factors triggered across both windows), clamped (0,1].
At `emergency_halt_pct` (the existing, currently-unwired −15% boolean): **block new
entries** via the fuse gate — deliberately NOT stop_bus (stop_bus halts the whole
cycle including both exit overlays, P10/D5). Exits, flips, protectives, and the
overlay remain fully free at every rung including emergency.

### 5.3 Enforcement point (D4)
Recommend **per-intent quantity multiplier** in live_loop immediately after the SCR
CAUTIOUS block (~`live_loop.py:2729`), copying its mechanics exactly: multiplicative
on `intent.quantity`, FUT min-1-contract rounding, equity min-1-share floor,
entries only (same bypass predicates as §2). Kraken twin: factor parameter into
`execution_pipeline`'s crypto sizing beside `_read_crypto_scr_sizing_factor`
(:1481-1497). Composition: LC5 × SCR × existing sizers — all multiplicative,
order-independent, never > 1.0. Alternative (cap-level, profit-lock pattern
`dynamic_risk_allocator.py:337-353`) documented in D4.

### 5.4 Staleness (the mandatory A4-grade guard)
Read `drawdown_state.json` via `runtime_json.read_runtime_state_json` (fail-closed
`(obj, Freshness)` canon). If stale/missing/corrupt:
- **no enforcement change** — the last-applied factor (persisted in
  `fuse_box_state.json`) is retained, never tightened or loosened on unknown data;
- loud: `FUSE_LC5_DRAWDOWN_STALE` marker + coach `feed_stale`-kind alert (dedupe
  key `fuse_lc5_drawdown_stale`, value-free) + `lc5.staleness` field in state;
- paired `FUSE_LC5_DRAWDOWN_RESTORED` on recovery (XOV2 pattern, P19/§1-P17.3);
- if staleness persists > `stale_max_seconds` (config, default 3600), degrade
  factor to the most conservative rung already reached this session — never
  upward — and say so. (Prewritten policy, not improvisation: DQ row for this feed.)

## 6. DQ1/DQ3 — per-feed dependency classes + prewritten stale policies

**Config** (`config/feed_policies.json`, D7 — schema borrows the exterminator
feeds-row shape plus the missing column):
```
{ "schema_version": "feed_policies.v1",
  "feeds": {
    "runtime/scr_state.json":    { "class": "execution_critical", "ttl_seconds": 180,
                                   "on_stale": "block_entries", "on_corrupt": "block_entries",
                                   "on_missing": "block_entries", "ttl_source": "artifact_self_declared" },
    "runtime/ibkr_status.json":  { "class": "execution_critical", "ttl_seconds": 120,
                                   "on_stale": "degrade_loud", ... },
    "runtime/drawdown_state.json": { "class": "advisory_until_lc5", "ttl_seconds": 300,
                                   "on_stale": "hold_last_loud", ... }
  } }
```
Policy vocabulary (all with existing production archetypes, P17): `block_entries`
(reduce-only still allowed — margin_block precedent, the XOV2 generalization),
`reduce` (factor clamp), `degrade_loud` (documented fallback + marker + alert),
`ignore` (advisory feeds — current fail-open behavior, now *declared* instead of
implicit). Artifact's own `ttl_seconds` wins over config (house rule).

**Enforcement hook:** new `chad/utils/feed_policy.py` — `read_with_policy(path)`
wrapping `runtime_json.read_runtime_state_json`, returning `(obj, verdict)`;
verdict carries class/policy/action + XOV2 semantics (typed unknown ≠ default;
never a value indistinguishable from a legitimate reading).

**Initial wired sites (D7): exactly two**, the P18 gaps — one change at a time:
1. SCR gate read (`live_loop.py:2665-2698`): stale/corrupt `scr_state.json` under
   enforce ⇒ block new entries (fail-CLOSED replaces today's fail-open), exits free;
   frozen-PAUSED also becomes visible (stale ⇒ policy, not eternal block).
2. Stop-bus input snapshot (`live_loop.py:609-661`): dead `ibkr_status.json` /
   `pnl_state.json` ⇒ `DQ_INPUT_DEAD` marker + coach alert + evidence (a silently
   disarmed latency halt becomes loud). Behavior change beyond loudness: none in W4A.
All other feeds get config rows + shadow evidence only. The 4 `_is_stale`
cut-pastes and the live_gate reader twin are noted as a later consolidation, NOT
touched in W4A.

**Fault-injection tests:** per wired site + per policy verb: back-dated `ts_utc`,
corrupt bytes, missing file, future-dated, env-TTL override — the
`test_xov2345_*` / `test_bugb_fixa_position_cap.py` pattern, plus
shadow-is-byte-identical proofs.

## 7. Trip events, state, observability (all fuses)

Per P19 conventions, every trip/clear/staleness event gets all four surfaces:
1. **Marker**: `FUSE_TRIP kind=<k> bucket=<id>` / `FUSE_CLEAR ...` /
   `FUSE_WOULD_BLOCK ...` (shadow) — grep-able log lines + evidence ndjson row.
2. **Coach NOTIFY**: `format_alert("fuse_trip", facts)` via the sanctioned
   generic-kind path first (edge_decay_cleared precedent); a registered
   `_tpl_fuse_trip` template is a cosmetic follow-up, not W4A-blocking. Severity:
   warning (trip/clear), critical (LC5 emergency, DQ block_entries firing).
3. **Dedupe-stable identity**: `fuse_<kind>_<bucket-identity>` — rule/bucket id +
   entity only, digits stripped (`stable_identity`), values live in evidence/facts,
   never in titles.
4. **Sentinel-visible state**: `runtime/fuse_box_state.json` written every cycle
   including OFF (heartbeat doctrine — `evaluated=0` must be distinguishable from
   dead), `schema_version: fuse_box_state.v1`, `ts_utc`, `ttl_seconds: 180`
   (cycle cadence + grace). Phase-2 adds the `config/exterminator.json` feeds row
   (`ttl_verified: true`, `ttl_source: "publisher cadence 60s cycle × 3"`) and an
   EXS7 `schema_contracts.enforced` pin.

State schema sketch:
```
{ "schema_version": "fuse_box_state.v1", "ts_utc": "...", "ttl_seconds": 180,
  "modes": {"lc2": "shadow", "lc3": "off", "lc5": "off", "dq": "off"},
  "session_date": "2026-07-23", "epoch_started_at_utc": "2026-06-30T12:17:42Z",
  "fuses": [ { "fuse_id": "symbol:TLT", "kind": "symbol", "tripped": true,
               "tripped_at_utc": "...", "trip_rule": "consecutive_losers",
               "consecutive_losers": 3, "session_net_pnl": -412.55,
               "clears_at": "next_session", "regime_scope": null } ],
  "lc5": { "factor": 1.0, "dd_5d_pct": null, "dd_20d_pct": null,
           "staleness": "fresh", "emergency": false },
  "dq": { "verdicts": { "runtime/scr_state.json": "fresh" } } }
```
(Values in rows; identities value-free — CTF-T2.)

## 8. Interaction analysis (REQUIRED)

### 8.1 The prime invariant — closes always pass
| Path | Why a fuse can never block it |
|---|---|
| Equity exit overlay closes | Never traverse stage-3 gates or the Kraken mirror (P9: `apply_close_intents` → adapter direct). Structural. |
| Crypto exit overlay closes | `reduce_only=True` + `CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE` marker + `crypto_exit\|` idempotency prefix — explicit exemption in the Kraken mirror, tested. |
| Strategy-emitted exits/flips/protectives | Stage-3 bypass predicate = symbol-blocker's (flip / EXIT / CLOSE) ∪ loss-guard's protective tags (reduce/hedge/stop_loss/liquidation). Tested per tag. |
| Reconciler closes | Same path as overlay closes (bypasses stage-3). |
| DQ `block_entries` | Verb is defined as entries-only (margin_block stale precedent: reduce-only allowed). Fault-injection asserts a close passes under every policy verb. |
| LC5 all rungs incl. emergency | Multiplier and emergency block apply after the bypass predicate — exits untouched at −15% too. Emergency is deliberately NOT stop_bus (stop_bus kills the overlays with the rest of the cycle — P10). |
What W4A does NOT fix (pre-existing, documented so nobody blames the fuse box):
edge-decay's exit-dropping signal filter, regime matrix's no-exit-exemption,
futures gate both-sides block (P10). Candidate follow-ups, out of scope.

### 8.2 Composition with existing breakers
| Existing mechanism | Grain / basis | LC2/LC3/LC5 relationship | Conflict? |
|---|---|---|---|
| PerStrategyLossGuard (GAP-026) | per-strategy $/session, signal-level | LC2 = same substrate + window, different grain (family/setup) and different layer (per-intent — closes the P8 pipeline-B hole the guard has). COEXIST in W4A (D1); subsumption is a later decision once LC2-enforce proves out | No — both drop entries only; double-drop is idempotent |
| Edge-decay halt | per-strategy trailing streak (trusted), 14d TTL | Different horizon (session vs trailing). A strategy can be halted AND its family tripped — both block entries; distinct dedupe keys (`edge_decay_<s>` vs `fuse_family_<f>`) so alerts don't collide | No |
| Strategy throttle gate | per-strategy today win-rate tiers | Orthogonal; throttle's untrusted-counting defect (counts validate_only pnl=0 rows) is pre-existing, NOT inherited (LC2 counts via §3 predicates) | No |
| symbol_performance_blocker | per-symbol, 2h TTL, dormant | Subsumed (D3); both hooks coexist harmlessly (blocker state empty) | No |
| SCR gate (PAUSED/CAUTIOUS) | global state machine | LC5 factor composes multiplicatively with CAUTIOUS scaling; PAUSED hard-block precedes fuse gate — fuse evaluation still runs (state stays fresh) | No |
| stop_bus | global cycle halt | Cycle halt preempts everything incl. fuse evaluator ⇒ `fuse_box_state.json` may age during a halt; TTL 180 means sentinel goes warn — correct behavior (loud), documented | No |
| Profit lock | realized-PnL tiers → cap factor / stop_new_entries | LC5 is mark-to-market drawdown; both multiplicative on different layers (cap vs intent qty under D4-A) — combined effect is the product, intended | No |
| Margin gate (shadow) | per-order BP | Independent pre-claim adapter hook; fuse gate is upstream (stage-3). Ordering: fuse first (cheaper), margin last (adapter) | No |
| Regime matrix / futures gate / weekend gate | asset/regime scoping | Upstream/downstream filters, orthogonal; fuse evaluation reads closes, not signals — no feedback loop | No |
| Exit overlay | close authority | Fuses never touch it (8.1). LC5 shrink reduces future entry sizes ⇒ smaller future closes — no loop (overlay reads broker truth, not fuse state) | No |
Feedback-loop check: fuses consume *closed-trade* outcomes and *equity history*;
they gate *future entries*. No fuse output feeds any counter input within a
session (a tripped bucket produces no new trusted closes ⇒ counters freeze, clear
at session roll — monotone, no oscillation). LC5 hysteresis: factor changes only
on fresh publisher data (120s cadence) and rungs are step functions with
session-scoped worst-rung memory (§5.4) — no flapping.

## 9. Phase-2 commit plan (each: build → set-diff green → commit, prefix `W4A`)

| # | Commit | Contents |
|---|---|---|
| W4A-0 | baseline capture | full-suite run in worktree; record failing-test set + count as `audits/W4A_BASELINE.md` (W2A/W3B methodology) |
| W4A-1 | fuse core + state | `fuse_box.py` (trusted counting predicates, trip/clear engine, state publisher), `config/fuse_box.json`, state heartbeat incl. OFF; unit tests |
| W4A-2 | regime + family plumbing | D2 outcome (regime stamp in `trade_closer.to_payload` if approved — additive, forward-only), family/sector config maps + registry warn-parity; tests |
| W4A-3 | LC2 buckets | family + setup_family fuses, coach/dedupe/marker eventing; tests incl. eventing identity |
| W4A-4 | LC3 buckets | symbol + sector fuses, `symbol_sectors.json`, unmapped-bucket guard; tests |
| W4A-5 | gate wiring | `fuse_gate.py` + stage-3 hook + Kraken mirror, shadow default, bypass predicates; blast-radius tests (byte-identical when off; overlay/close exemption proofs) |
| W4A-6 | LC5 budgets | 5d/20d windows in drawdown_guard/publisher (D6 schema disposition), epoch-scoped windows, sample-depth gates; tests |
| W4A-7 | LC5 enforcement | ladder + factor application at D4 point + staleness guard + emergency entry-block; tests incl. stale-no-change-and-loud |
| W4A-8 | DQ policies | `feed_policies.json` + `feed_policy.py` + 2 wired sites; fault-injection suite |
| W4A-9 | observability + closure | exterminator feeds row + EXS7 pin, `scripts/clear_fuse.py`, docs, PLAN_W4A closure record (decisions → commits) |

Set-diff methodology: after every commit, `python3 -m pytest chad/tests/ -q` in the
worktree; failure set must be ⊆ W4A-0 baseline set (named-test diff, not count
diff). `py_compile` per changed file; `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview`
smoke at W4A-5, -7, -8 (the wiring commits). No pushes, no merges — same STOP as
W3B (merge = live for oneshot timers).

## 10. Decision points (operator input required before Phase 2)

- **D1 — LC2 grain & GAP-026 relationship.** Families as config groups over
  strategy_ids + optional setup_family sub-fuses (4 stamped strategies); COEXIST
  with PerStrategyLossGuard (different grain+layer). *Recommended: yes to both
  grains, coexist.* Alternative: strategy-only grain, or subsume the guard now.
- **D2 — Regime bucketing mechanics.** (a) Stamp live regime into
  `closed_trade` payload at `trade_closer.to_payload` (one additive field,
  forward-only, mirrors the crypto lane) — accurate, survives replay; or
  (b) join-at-count from `runtime/regime_state.json` (no closer change, ±60s
  skew, unavailable for backfill). *Recommended: (a).*
- **D3 — LC3 vs dormant symbol_performance_blocker.** Subsume (new fuse, blocker
  untouched/no-op) vs extend the blocker in place. *Recommended: subsume (P2:
  untrusted counting, hardcoded path, 2h-TTL semantics mismatch).*
- **D4 — LC5 enforcement point.** (A) per-intent quantity multiplier post-SCR
  (live_loop ~:2729 + Kraken twin) — per-intent precision, exact SCR mechanics; or
  (B) allocator cap multiplier (profit-lock pattern) — single site, both lanes,
  but cap-level and orchestrator-latency. *Recommended: (A).*
- **D5 — Emergency −15% wiring.** Wire the existing halt boolean as fuse-gate
  entry-block (exits/overlay free) vs leave −15% unwired (ladder only).
  *Recommended: wire as entry-block; explicitly NOT stop_bus.*
- **D6 — drawdown_state schema.** Bump to `.v2` + update EXS7 pin/EXS1 row in the
  same commit vs additive fields under `.v1`. *Recommended: v2 bump (honest
  versioning; sentinel pins updated atomically).*
- **D7 — DQ config home + initial scope.** New `config/feed_policies.json`
  (sentinel keeps sole ownership of exterminator.json) + exactly the two P18 wired
  sites vs broader wiring. *Recommended: new file, two sites; everything else
  config-rows + shadow evidence only.*
- **D8 — IBKR close-stamp / adapter backstop.** Thread `action/reason` into
  intent meta at `_close_intent_to_ibkr` + adapter-level fuse backstop — in W4A or
  deferred? *Recommended: DEFER (stage-3 placement + structural bypass make it
  unnecessary for correctness now; smaller Wave-4 lane-B candidate).*

— END PLAN (Phase 1). STOP here; Phase 2 requires D1–D8.
