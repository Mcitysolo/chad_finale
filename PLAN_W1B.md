# PLAN_W1B — Wave-1 Forensics (PLAN ONLY)

**Branch:** `goal/wave1-forensics` (from `main` @ `e17ea24`)
**Status:** Phase-1 plan. **No build in this commit.** Phase 2 builds only after an explicit GO.
**Author aid:** grounded in a 4-stream read-only forensic survey (heartbeat writers, strategy logging config, edge-decay monitor, lifecycle-replay engine), each cross-checked against live runtime + a full clean-suite baseline run on 2026-07-19. Every fix target below was grep-verified **not** to be a Lane-A file.

---

## 0. Binding constraints (both phases)

- **repo-only** — no host mutation. `runtime/*` and `data/*` are gitignored and the repo-write leak-guard (`conftest.py` → `chad/testing/repo_write_guard.py`) *fails any test that writes under working-tree `runtime/`/`data/`*. New tests write to `tmp_path` only, and **read** only `tmp_path` (the Item-1 tests are the one legacy exception — they read live `runtime/`, which is exactly why they flake; see §3).
- **no systemd calls** — never `systemctl`. This wave adds **no** new units; the four fixes are all in-process code + tests. (The heartbeat *timer* and *orchestrator* are live services; we do **not** touch, restart, or decommission them.)
- **no runtime mutation** — never run a publisher/engine `--execute`/write against real `runtime/`. Item 4's replay engine is exercised only against a `tmp_path` repo-root in tests; the live timer keeps running unchanged.
- **do NOT touch Lane A (W1A) files** — `chad/ops/exterminator_sentinel.py`, `chad/core/position_guard.py`, `ops/var_publisher.py`, `chad/risk/portfolio_var.py`, `scripts/reap_ibkr_exec_state.py`, `chad/ops/reconciliation_publisher.py`, and `PLAN_W1A.md`. **This branch is cut from `main` (pre-W1A), so those files structurally do not carry W1A's edits here — a merge later is conflict-free as long as W1B never edits them.** Verified: none of the four fix targets is on that list.
- **suite green vs known-5 after every commit** — see §1.
- **commits `W1B`** — one concern per commit, subject `W1B-<n>: …` (matches the `W1A-<n>` / `CBAR-S1` convention). Local commits only; **push is not assumed** (recent waves land LOCAL / push-DENIED).

### Verification sequence (run after every commit)
```
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -q 2>&1 | tail -25        # FULL run (no -x) — needed to confirm "green vs known-5"
CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview 2>&1 | tail -30
```
> `-x` (as CLAUDE.md shows) stops at the first known failure and cannot confirm "green vs known-5". Use the full run for the gate; `-x` only for fast local iteration.

---

## 1. Test baseline (freeze BEFORE building) — and a suite-level non-determinism that blocks Item 1's acceptance

**Baseline captured on this branch tip (one full `pytest chad/tests/ -q` run, 2026-07-19): `8 failed, 3770 passed, 1 skipped` in ~183s.**

| # | Failing node | Nature | Deterministic? |
|---|--------------|--------|----------------|
| 1–3 | `test_tier_manager.py::{test_1_scale_tier_at_182k, test_2_starter_caps_at_2600, test_7_promotion_immediate}` | CAD re-pricing, driven by the **uncommitted** working-tree currency-audit config (`config/tiers.json` + `config/withdrawal_policy.json` are `M` in the tree) — not a code regression | ✅ yes |
| 4 | `test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop` | MES bar-provider dependency | ✅ yes |
| 5 | `test_quarantine_sidecar.py::test_per_strategy_loss_guard_excludes_quarantined_delta_trades` | delta-loss-guard | ✅ yes |
| — | `test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only` | **Item 1 flake** (heartbeat schema alternation) | ❌ timing-gated |
| — | `test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only` | **Item 1 flake** (heartbeat schema alternation) | ❌ timing-gated |
| — | `test_xgb_promotion_workflow.py::test_trainer_candidate_dir_named_with_utc_timestamp` | **NOT a real failure** — SIGALRM collateral (see §1a); passes in isolation (`1 passed in 5.62s`) | ❌ order-gated |

The **deterministic known-5** = the 5 in rows 1–5 (identical to Lane A's W1A baseline). "Green vs known-5" = the post-commit failing set is a **subset of** those 5 (the 2 Item-1 flakes go away once Item 1 lands; the xgb row goes away once §1a lands). Any *other* new red blocks the commit.

### 1a. Cross-cutting finding — a leaked `SIGALRM` makes the full suite non-deterministic (this reframes Item 1's acceptance test)

Item 1's stated acceptance is **"the two live_posture flaky tests must pass 5 consecutive full-suite runs."** That is currently **unachievable for a reason unrelated to the heartbeat**, and the plan must fix it first:

- **Root cause (proven):** `chad/portfolio/ibkr_portfolio_collector_v2.py::main()` (`:384`) arms a 60 s wall-clock guard via `install_wall_clock_guard()` (`:392` → `signal.alarm(60)` at `:84`) and **never disarms** on return (the code comment at `:89-90` assumes the process exits — true in production, false under pytest). Empirically confirmed: after `install_wall_clock_guard()`, `signal.alarm(0)` reports `60` s pending, and `main()` contains no `disarm_wall_clock_guard()`.
- **The leak vector:** `chad/tests/test_ibkr_portfolio_collector_v2_positions.py` calls the real `collector_module.main(...)` **three times** (`:196, :224, :247`) and has **no** alarm-clearing fixture (unlike `test_gap043_collector_wall_clock_guard.py`, which owns an autouse `_clear_alarm`). So a 60 s alarm survives into whatever tests run next.
- **The two observed manifestations, one cause (`_signum = 14` = SIGALRM):**
  - *Run 1:* the alarm fired during an unrelated test's **teardown** → `INTERNALERROR … CollectorWallClockTimeout: 124` → **the whole session aborted** at 171 s (only 3283 of ~3646 tests ran).
  - *Run 2:* the alarm fired inside `xgb.train` during `test_trainer_candidate_dir_named_with_utc_timestamp` → that test "failed" (it **passes in isolation**).
- **Consequence:** ~1-in-2 full-suite runs is polluted by this alarm. You cannot demonstrate *5 consecutive clean full-suite runs* while it exists — regardless of how well Item 1's heartbeat race is fixed. **Therefore the SIGALRM de-flake is a prerequisite of Item 1, sequenced as `W1B-0` (see §3 + §7 + decision D0).**

---

## 2. Headline: like W1A, 3 of the 4 briefs are stale or understated; the 4th (logging) is an accurate one-liner

| Item | Brief premise | Reality (this survey) |
|------|---------------|-----------------------|
| 1 | "two-writer **schema race**; one writer owns the file **or** schemas merge" | Real, but **not a torn/partial write** — both writers do atomic `tmp+os.replace`, so it's whole-file **last-writer-wins schema alternation**. "One writer owns" is **BLOCKED** (the timer writer is the *sole* `decision_trace_livegate_down` alert source — cannot be decommissioned). The viable fix is **schema merge** (additively make the timer writer emit the two posture keys) — plus a **prerequisite SIGALRM de-flake** (§1a) without which the acceptance test can't pass. |
| 2 | "`chad.strategies.*` INFO doesn't reach journald; diagnose the propagation/level gap; **may be a one-liner**" | **Accurate — it is a one-liner.** `run_loop()` binds the handler+INFO to `chad.live_loop` (a *sibling* of `chad.strategies`); root stays at default WARNING, so strategy INFO is dropped **at source**. Not `propagate=False`, not debug-vs-info. One caveat (volume) + one carve-out (`gamma_reversion:411` is `DEBUG`, stays hidden). |
| 3 | "`check_all` re-evaluates strategies in the halt store even when absent from the trusted ledger; self-clear criteria in config — stop the roach motel" | Premise **confirmed but understated**: there are **two** roach-motel mechanisms. (a) ledger-absence (the brief's), **and** (b) the monitor has **no self-clear path at all** — even a *ledger-resident* strategy that recovers stays halted forever (the `ok`/`insufficient` branches set `halted:False` **in memory only**, never writing the store). The fix must address both. |
| 4 | "replay honors Epoch-3; **phantom 12-position** report eliminated" | Bug **confirmed** (engine never filters by epoch). But the numbers are **stale**: the Epoch-3 boundary is **`2026-06-30T12:17:42Z`** (not 2026-05-27), and the live phantom count is **14 → 10** (drops **4**: `MCL, MES, MGC, QQQ`), not "12 → 0". |

---

## 3. Item 1 — heartbeat two-writer schema alternation (+ SIGALRM prerequisite)

### Premise check vs brief
- **Exactly two writers**, both atomic (`tmp + os.replace`), **neither emits `schema_version`**:
  - **Writer A (timer, 5-min):** `chad/core/decision_trace_heartbeat.py::main()` payload `:154-172`. Keys: `ts_utc, ok, live_gate_url, latency_ms, error, live_gate, alert_sent`. `live_gate` = the **verbatim `/live-gate` response** (which *nests* `allow_ibkr_live`/`allow_ibkr_paper`). Driven by `chad-decision-trace-heartbeat.timer` (enabled+active, fires every 5 min — verified today).
  - **Writer B (orchestrator, ~60s):** `chad/core/orchestrator.py::_write_decision_trace_heartbeat` `:410-471` (called `:971`). Emits **top-level** `allow_ibkr_live` / `allow_ibkr_paper` (+ `source, mode, scr_state, live_readiness, …`) **and** the nested `live_gate`.
- **The flake:** both tests read the **live** file and assert **top-level** keys:
  - `test_pr03_ib_async_phase2_migration.py:297-298`: `assert hb.get("allow_ibkr_live") is False` / `assert hb.get("allow_ibkr_paper") is True`.
  - `test_pr04_options_chain_refresh_remediation.py:545-546`: identical.
  - Those top-level keys exist **only in Writer B**. When the timer (Writer A) wrote most recently (~15-20 % of each 300 s window), `hb.get("allow_ibkr_live")` is `None` → `None is False` → `AssertionError`. **Live-reproduced**: passes in a Writer-B window, fails in a Writer-A window (sampled across timer fires).
- **Corrections to the brief:** (i) it is *not* a torn-write race — a reader always sees one complete JSON doc; it's whole-file schema alternation. (ii) "one writer owns" is **BLOCKED**: `decision_trace_livegate_down` is emitted **only** by the timer writer (`decision_trace_heartbeat.py:169`), so the timer can't be retired (corroborated by memory `weekend_maint_20260613`). (iii) The orchestrator comment (~`:274-285`) claiming the timer "stopped running since 2026-04-03" is **stale** — the timer is active.
- **Both live readers are already schema-tolerant** (`backend/operator_surface.py::perf_snapshot` reads a guarded subset; `operator_surface_v2` passes the whole blob through) — so *adding* keys to Writer A breaks nothing.

### The fix — schema merge (recommended), + SIGALRM prerequisite, + endpoint-down guard
1. **`W1B-0` (prerequisite, §1a):** kill the leaked SIGALRM so the full suite is deterministic. **Recommended:** a surgical autouse teardown fixture in `chad/tests/test_ibkr_portfolio_collector_v2_positions.py` that calls `signal.alarm(0)` after each test — mirroring the existing `_clear_alarm` in `test_gap043_collector_wall_clock_guard.py:35-42`. (Alternatives in D0: a global `conftest.py` guard, or a production `finally: disarm_wall_clock_guard()` in `main()`.)
2. **`W1B-1` core (schema merge):** in **Writer A** (`chad/core/decision_trace_heartbeat.py`, payload `:154-172`), **additively** emit top-level `allow_ibkr_live` and `allow_ibkr_paper`, sourced from the `/live-gate` payload it already fetches (`res.payload.get("allow_ibkr_live")`, etc.). Now both writers agree on the two keys the tests + any consumer need, so the file is schema-consistent regardless of which service wrote last. This is the brief's "schemas merge", it's a single peripheral file (a oneshot script, **not** the hot path), purely additive, and it fixes the **live artifact** (an observability win), not just the tests' perception.
3. **`W1B-1` defensive (endpoint-down guard):** residual caveat — if `/live-gate` is unreachable at read time, **both** writers store `live_gate=null` and top-level posture `None`, so the tests would still red on a transient outage. Add a guard to both tests: `pytest.skip("live-gate unreachable")` when `hb.get("live_gate")` is falsy, keeping the posture assertion only when the endpoint is up. This is what makes "5 consecutive full-suite runs" robust to a `/live-gate` blip.

### Files to touch
- **EDIT** `chad/core/decision_trace_heartbeat.py` — additive top-level `allow_ibkr_live`/`allow_ibkr_paper` in the payload (`:154-172`). No schema_version bump (keep additive; readers are tolerant).
- **EDIT** `chad/tests/test_pr03_ib_async_phase2_migration.py` + `chad/tests/test_pr04_options_chain_refresh_remediation.py` — add the endpoint-down `skip` guard around the two assertions (assertions themselves unchanged).
- **EDIT** `chad/tests/test_ibkr_portfolio_collector_v2_positions.py` — autouse `signal.alarm(0)` teardown fixture (the SIGALRM de-flake).
- **NEW** `chad/tests/test_w1b_heartbeat_schema_merge.py` — drive Writer A's payload builder against a stubbed `/live-gate` response and assert both top-level keys are present and equal the nested values.
- **DO NOT TOUCH:** `chad/core/orchestrator.py` (Writer B already emits the keys; its heartbeat writer sits next to the hot loop — leave it), and the two live services/timers.

### Test plan
- **Merge unit (hermetic):** Writer A payload built from a fake live-gate dict `{allow_ibkr_live:False, allow_ibkr_paper:True, …}` → top-level `allow_ibkr_live is False`, `allow_ibkr_paper is True`, and they mirror `payload["live_gate"]`'s nested values.
- **Merge unit — endpoint down:** live-gate `None` → top-level keys `None` (matches Writer B's down-behavior; no crash).
- **SIGALRM isolation:** after the positions-test module runs, `signal.alarm(0)` returns `0` (no pending alarm). Add a tiny guard test that arms then relies on the fixture to clear.
- **Flake gate (acceptance):** with `W1B-0`+`W1B-1` in, run the full suite 5× consecutively (`/live-gate` up on this host); the two live_posture tests pass every run and no SIGALRM abort/collateral occurs. Failing set ⊆ known-5.

### Risks
- **Touching a live-adjacent producer:** Writer A is a standalone oneshot (safe); we deliberately avoid Writer B (orchestrator hot-loop neighbor).
- **Endpoint-down skip could mask a real regression** (a test that always skips is dead). Mitigate: only skip when `live_gate` is falsy (rare, transient); assert on presence otherwise; log the skip reason.
- **`schema_version` absence** — neither writer versions the file; adding one now would ripple to every reader. Keep additive; do **not** introduce `schema_version` in this wave.
- **The two tests remain non-hermetic** (they read live `runtime/`). We are hardening, not converting them to `tmp_path`, because their *purpose* is to assert the live system's paper posture. Full hermeticity is a larger, separate refactor (noted, not done here).

---

## 4. Item 2 — strategy-module INFO logging never reaches journald

### Premise check vs brief — accurate; it is a one-liner
- `CRYPTO_EXPLORATION_HANDLER_PASS` is logged at `chad/strategies/alpha_crypto.py:678` via `logger.info(...)`, logger = `getLogger(__name__)` = **`chad.strategies.alpha_crypto`** (`:66`). No-signal reason lines: `alpha_crypto.py:680,:816` (INFO). (Carve-outs: `gamma_reversion.py:411` is `LOG.debug` — stays hidden even after this fix; `beta.py:300,:311` are `WARNING` — already reach journald.)
- **Root cause:** the live-loop process's *only* logging setup is `run_loop()` at `chad/core/live_loop.py:3387-3396`, which attaches a `StreamHandler` and `setLevel(INFO)` to **`chad.live_loop`** — a *sibling* of `chad.strategies`. There is **no** `basicConfig`/`dictConfig`/root handler in this process. So root = default **WARNING**; `chad` and `chad.strategies` are **NOTSET** → effective level inherits **WARNING** → `chad.strategies.* .info()` fails `isEnabledFor(INFO)` and is **dropped at the source**. (Empirically: `getLogger("chad.strategies.alpha_crypto").isEnabledFor(logging.INFO)` is `False` before the fix.) It is **not** `propagate=False` (none exists) and **not** debug-vs-info (the WARNING lines beside them do reach journald).
- journald wiring is fine: `chad-live-loop.service` sets `StandardOutput=journal` / `StandardError=journal`, and the `StreamHandler` writes to stderr.

### The fix (one line) + a volume decision
- **Recommended (D2 option a):** at `chad/core/live_loop.py:3387`, change `logging.getLogger("chad.live_loop")` → `logging.getLogger("chad")`. Every `chad.*` logger (including `chad.strategies.*` and `chad.live_loop`) then inherits INFO and reaches the single handler by propagation. Verified: post-change `chad.strategies.alpha_crypto` effective level = INFO, `isEnabledFor(INFO)` = True, reachable handler = True. The `run_loop` format string has no `%(name)s`, so output is visually unchanged. **Do not** add a second handler on `chad` while leaving one on `chad.live_loop` (that double-emits every live-loop line).
- **Trade-off / D2 option b (lower volume):** the one-liner promotes **all** `chad.*` INFO to journald — some modules may be chatty. If volume is a concern, instead set the handler on `chad` (so it's reachable) but keep `chad` at WARNING and raise only `chad.strategies` to INFO (`getLogger("chad.strategies").setLevel(logging.INFO)` alongside the handler placement). More lines, narrower blast radius.

### Files to touch
- **EDIT** `chad/core/live_loop.py` — the single line at `:3387` (inside `run_loop()`'s logging bootstrap — **not** the `run_once` signal path). *Note:* Lane A self-imposed "no `live_loop.py` edits"; that was W1A's stance, not a W1B constraint, and this line is the process logging setup, not the hot path. It is not a Lane-A-modified file (W1A never edits `live_loop.py`), so no merge conflict.
- **NEW/EDIT** `chad/tests/test_w1b_strategy_logging_visibility.py` — assert the **effective configuration** (see below), because `caplog` masks the bug (it attaches its own root handler and forces levels). Optionally refactor `:3387-3396` into a tiny `_configure_live_loop_logging()` helper so the test can invoke exactly the production setup.

### Test plan
- After invoking the production logging setup: `getLogger("chad.strategies.alpha_crypto").getEffectiveLevel() == logging.INFO`; `isEnabledFor(logging.INFO) is True`; `getLogger("chad").handlers` non-empty (handler reachable by propagation); negative guard: no logger on the `chad` tree has `propagate is False`.
- Optional `caplog` smoke: `caplog.set_level(INFO, logger="chad.strategies.alpha_crypto")`, drive the exploration-handler branch, assert `"CRYPTO_EXPLORATION_HANDLER_PASS" in caplog.text`.

### Risks
- **journald volume** (the main one) — see D2. Recommend accepting it (this is an observability wave) but flag it for the operator.
- **Refactor-for-testability** slightly enlarges the diff; keep it a pure extract (no behavior change) if done.
- `gamma_reversion.py:411` (DEBUG) stays hidden — call it out; promoting DEBUG is out of scope.

---

## 5. Item 3 — edge-decay "roach motel": halts never self-clear

### Premise check vs brief — confirmed, and worse than stated
- Module `chad/risk/edge_decay_monitor.py`; `check_all` `:448-454` iterates **only** `collect_recent_trades_by_strategy(...)` keys (`:452`) — i.e. the **trusted ledger** (`data/trades/trade_history_*.ndjson`, excluding `pnl_untrusted`/`validate_only`/quarantined). Halt store = `runtime/strategy_allocations.json`. Called from `chad/core/live_loop.py:1686`.
- **Mechanism 1 (brief's):** a strategy absent from the ledger is never passed to `check_strategy`, so its persisted halt is never re-evaluated → permanent. **alpha_crypto chain verified live:** 2148 records, all `pnl:0.0` + `pnl_untrusted`/`validate_only` tags → excluded → `collect_recent_trades_by_strategy` returns only `{gamma, manual}` → alpha_crypto absent → its 2026-07-02 halt could never clear (operator cleared it manually 2026-07-15).
- **Mechanism 2 (NOT in the brief — the fix is bigger):** `check_all`/`check_strategy` contain **zero** calls to `clear_strategy_halt`. The non-halt branches `insufficient_trades` (`:420-425`) and `ok` (`:441-446`) return `halted:False` **in memory only** — no file write. So even a *ledger-resident* strategy that recovers stays `halted:true` in the store forever. The only clear path today is the operator CLI `scripts/clear_edge_decay.py:76`.
- Config: `config/edge_decay_config.json` (`schema_version:"edge_decay_config.v1"`, `consecutive_threshold:5`, `min_trades:20`), loaded by `_load_config_values` (`:74-95`) with in-code fallbacks `DEFAULT_CONSECUTIVE_THRESHOLD=5` / `DEFAULT_MIN_TRADES=20` (`:70-71`).

### The fix (in `check_all`/`check_strategy` + config)
1. **Union the iteration set** in `check_all`: evaluate `set(ledger) | set(halted strategies from read_allocations())` (`read_allocations` at `:255`). Strategies halted-but-absent-from-ledger (alpha_crypto) are now re-considered each cycle.
2. **Pre-registered self-clear criteria** (register in `config/edge_decay_config.json`, mirroring the existing threshold keys, with in-code safe defaults so behavior is code-driven):
   - `halt_ttl_days` (e.g. 14): auto-clear a halt when `now − halted_at > ttl` **and** there is no *fresh trusted losing evidence* (strategy absent from ledger, or present with `streak < consecutive_threshold`). This unsticks alpha_crypto (no trusted losing fills ⇒ no evidence to keep the halt) without un-halting a genuinely-decaying strategy.
   - `clear_on_recovery: true` (fixes Mechanism 2): when a ledger-resident halted strategy shows `streak < consecutive_threshold`, actually **call `clear_strategy_halt`** (write the store) instead of the current silent in-memory `halted:False`.
3. **Clear via the existing `clear_strategy_halt`** with `cleared_by="edge_decay_monitor_auto"` + a distinct `clear_reason`, reusing GAP-018 semantics (`previous_consecutive_negative` preserved, counter reset). Plumb new keys through `_load_config_values` (`:74-95`) + `default_factory` dataclass fields alongside `consecutive_threshold`/`min_trades` (`:399-400`).
- `live_loop.py` already re-reads the persisted file each cycle (`:1691-1699`) and prunes alert-tracking on clear (`:1734-1742`), so **no `live_loop` change** is required.

### Files to touch
- **EDIT** `chad/risk/edge_decay_monitor.py` — `check_all` (union), `check_strategy`/branch (recovery clear + TTL clear), `_load_config_values` + dataclass fields.
- **EDIT** `config/edge_decay_config.json` — additive `halt_ttl_days` + `clear_on_recovery` (bump `schema_version` → `.v2` since keys change). **Governance note (D3):** this is a *self-clear/hygiene* config with conservative in-code defaults, not a risk cap / live-mode / sizing change; the enabling logic lives in code. If the operator classifies `edge_decay_config.json` as "strategy config" under governance rule #3, the *specific TTL value* becomes a Pending Action while the code defaults ship.
- **NEW** `chad/tests/test_w1b_edge_decay_self_clear.py`.

### Test plan (mirror `test_edge_decay_monitor.py` / `test_gap018_halt_clear_semantics.py` fixtures; tmp store)
- **Ledger-absent + TTL elapsed:** seed `strategy_allocations.json` with `halted:true`, `halted_at` past `halt_ttl_days`, strategy has **no** ledger fills → `check_all` → store shows `halted:false`, `cleared_by="edge_decay_monitor_auto"`.
- **Ledger-absent + TTL not elapsed:** stays halted (no premature clear).
- **Recovery (Mechanism 2):** ledger-resident, `streak < threshold`, currently `halted:true` → `check_all` → store persisted `halted:false` (today it wrongly stays true).
- **Still-decaying:** ledger-resident, `streak ≥ threshold` → stays halted (no false clear).
- **Idempotent rerun** → no thrash; counter-preservation intact.

### Risks
- **Direction of change is *toward* trading** — auto-un-halting re-enables a strategy. Criteria must be conservative/evidence-based ("no fresh trusted losing evidence" + TTL). Posture is paper, so blast radius is low, but state it plainly.
- **`halted_at` availability** — TTL needs a timestamp on the halt record. If `set_strategy_halted` doesn't already stamp one, add it additively (and treat a missing `halted_at` as "not yet eligible" to avoid clearing on unknown age). Confirm during build.
- **Schema bump** — `edge_decay_config.v2` must keep old keys; `_load_config_values` must tolerate a `.v1` file (fall back to defaults for new keys).

---

## 6. Item 4 — lifecycle-replay ignores the Epoch-3 reset (phantom positions)

### Premise check vs brief — bug confirmed; the numbers are stale
- Engine `chad/ops/lifecycle_replay_engine.py` reconstructs positions by summing signed fill qty from `data/fills/FILLS_*.ndjson` (+ `fees`, + diagnostic `broker_events`). **No epoch filter anywhere** — the fills loop `:61` and fees loop `:98` accumulate **all** history; `load_epoch_state`/`is_pre_epoch` are never imported.
- **Canonical epoch source:** `runtime/epoch_state.json` (`epoch_state.v1`) → `epoch_started_at_utc = "2026-06-30T12:17:42Z"` (the 06-30 reset-bootstrap boundary — **not** the "2026-05-27" in the brief), read via `chad/utils/epoch.py::load_epoch_state()` + per-record `is_pre_epoch(record, cutoff)` (`:172`).
- **Live evidence (ran the engine read-only):** `positions_count = 14` (**not** "12"), fills eligible split pre=2425 / post=198. Post-epoch-only replay → **10** positions; the **4** dropped phantoms are `MCL, MES, MGC, QQQ` (3 futures + QQQ, pre-reset residuals with zero net post-epoch qty). Downstream `runtime/lifecycle_replay_coverage.json` currently reads `REPLAY_MISMATCH` (replay 14 vs snapshot 9) — the census "RED".
- **Independent of PA-EP8v2** (the replay *netting* rebuild): this is purely an input timestamp filter; the net-qty math is untouched. `build_replay_state(repo_root)` **already takes a `repo_root`** (`:142`), so the fix and its test are cleanly rootable to `tmp_path`.

### The fix — mirror the one epoch-aware engine
Mirror `chad/analytics/trade_stats_engine.py` (the "1/5 that honors the epoch"; cutoff resolution `:651-660`, per-record skip `:428-432`):
1. In `build_replay_state` (`:142`), resolve `epoch_cutoff = load_epoch_state(runtime_dir=repo_root/"runtime")`'s `epoch_started_at` (from the passed `repo_root`, so tests point it at `tmp_path`).
2. Thread `epoch_cutoff` into `replay_positions(evidence, epoch_cutoff=…)`; at the top of the fills loop (`:61-62`) and fees loop (`:98`), `if epoch_cutoff and is_pre_epoch(row, epoch_cutoff): continue`.
3. **Fail-safe to legacy:** `load_epoch_state` returns `None` when `epoch_state.json` is absent/corrupt → `epoch_cutoff=None` → no filter (byte-for-byte legacy behavior preserved). Result on live data: `positions_count` 14 → 10.
- *(Optional follow-on, not this wave)* thread the same cutoff through `lifecycle_replay_coverage.py` + `lifecycle_replay_drift_audit.py` for a consistent post-epoch comparison; the engine fix alone corrects the phantom count.

### Files to touch
- **EDIT** `chad/ops/lifecycle_replay_engine.py` — `replay_positions` (add `epoch_cutoff` param + two skips), `build_replay_state` (resolve + pass the cutoff). Additive imports from `chad.utils.epoch`.
- **NEW** `chad/tests/test_w1b_lifecycle_replay_epoch.py` — mirror `chad/tests/test_epoch_filter.py` fixture style.

### Test plan (fully hermetic — `tmp_path` repo-root, no live runtime read)
- **Fixture:** tmp `repo_root` with `data/fills/FILLS_test.ndjson` = one pre-epoch fill (`payload.symbol=MES, side=BUY, status="Filled", quantity=1, fill_price=100, fill_time_utc="2026-06-01T00:00:00Z"`) + one post-epoch fill (`SPY`, `fill_time_utc="2026-07-01T00:00:00Z"`); `runtime/epoch_state.json` with `epoch_started_at_utc="2026-06-30T12:17:42Z"`. (Use title-case `"Filled"`; the engine uppercases at `:66`.)
- **Assert:** `build_replay_state(tmp_root)["positions"]` → `"MES" not in`, `"SPY" in`, `positions_count == 1`.
- **Fail-safe:** no `epoch_state.json` → both symbols present (legacy preserved), mirroring `test_epoch_missing_state_preserves_legacy_behavior`.

### Risks
- **Timestamp field resolution:** `is_pre_epoch` → `record_realized_time` prefers `exit_time_utc → fill_time_utc → entry_time_utc → timestamp_utc`. A fill with a missing/oddly-named time field would be treated as **not** pre-epoch (kept) — safe/legacy direction, but note it (verify FILLS rows carry `fill_time_utc`).
- **Fees vs positions:** `positions_count` derives from **fills only**; filtering fees is for cost-consistency and does not change the phantom count. Keep both filters but don't over-index on fees.
- **Live artifact only updates after the next timer fire** — the repo fix corrects the *code*; the live `runtime/lifecycle_replay_state.json` (14) refreshes when `chad-lifecycle-replay-engine.timer` next runs (an operator/runtime event, out of this wave's scope).

---

## 7. Proposed commit sequence (Phase 2, each verified green-vs-known-5)

| Commit | Concern |
|--------|---------|
| `W1B-0` | Prereq — de-flake the leaked collector SIGALRM (autouse `signal.alarm(0)` teardown) so the full suite is deterministic; freeze the baseline node IDs. *(gates Item 1's "5 consecutive runs")* |
| `W1B-1` | Item 1 — schema merge: Writer A (`decision_trace_heartbeat.py`) emits top-level `allow_ibkr_live`/`allow_ibkr_paper`; add endpoint-down `skip` guard to the two live_posture tests; new merge unit test |
| `W1B-2` | Item 2 — one-line logger fix in `run_loop()` (`chad.live_loop` → `chad`) + effective-config test |
| `W1B-3` | Item 3 — edge-decay self-clear: `check_all` union + recovery/TTL clear via `clear_strategy_halt`; additive `edge_decay_config.v2` keys; new self-clear test |
| `W1B-4` | Item 4 — lifecycle-replay epoch filter (mirror `trade_stats_engine`) + hermetic regression test |

One concern per commit; `py_compile` + full `pytest chad/tests/ -q` + `full_cycle_preview` after each. **The plan itself is committed alone now** (`W1B: Phase-1 plan`), before any of the above.

---

## 8. Decisions required before/within Phase 2 (GO gates)

- **D0 (Item 1 prerequisite):** how to kill the SIGALRM leak — **(a)** surgical autouse teardown fixture in `test_ibkr_portfolio_collector_v2_positions.py` *(recommended — mirrors the existing `test_gap043` pattern, zero production risk)*; **(b)** a global `conftest.py` autouse alarm-clear *(most robust; broader blast radius)*; **(c)** production `finally: disarm_wall_clock_guard()` in `collector.main()` *(fixes the true root but changes production for a test-only symptom, against the code's stated "process-exit clears it" assumption)*. *Recommend (a).*
- **D1 (Item 1):** **schema merge** (make Writer A emit the two posture keys — production, brief-aligned, fixes the live artifact) *(recommended)* vs. **test-only** (read the nested `live_gate` block both writers already share in `pr03`/`pr04` — no production change, but the live file keeps alternating shape). Both need D0 + the endpoint-down guard to hit "5 consecutive".
- **D2 (Item 2):** the one-liner promotes **all** `chad.*` INFO to journald *(recommended for an observability wave)* vs. a narrower variant that raises only `chad.strategies` to INFO (more lines, less journald volume).
- **D3 (Item 3):** whether adding `halt_ttl_days`/`clear_on_recovery` to `config/edge_decay_config.json` counts as governed "strategy config". Recommend: ship the **logic + conservative in-code defaults** as code; if the operator treats the config values as governed, the *TTL value* lands as a Pending Action while defaults hold.

## 9. Explicit non-goals / out of scope
- No `systemctl`; no install/enable/restart of any unit or timer; no decommission of the heartbeat timer (it is the sole `livegate_down` alert source).
- No writes to `runtime/`/`data/`; the replay engine is never run `--execute`/for-real against live runtime — only `tmp_path` in tests.
- No edits to any **Lane-A (W1A)** file (§0 list) or `PLAN_W1A.md`; no edits to `chad/core/orchestrator.py` (Writer B) — Item 1 fixes the *timer* writer only.
- No conversion of the two live_posture tests to fully hermetic `tmp_path` (larger refactor; we harden, not rewrite, their purpose being a live-posture check).
- No promotion of `gamma_reversion.py:411` DEBUG line (out of scope for the INFO fix).
- No change to the uncommitted working-tree currency-audit config (`config/tiers.json`, `config/withdrawal_policy.json`) — those drive 3 of the known-5 and are not this wave's concern.
- Push is not assumed; commits land LOCAL pending explicit GO.
