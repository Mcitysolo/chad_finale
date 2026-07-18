# PLAN_W1A — Wave-1 Observability (PLAN ONLY)

**Branch:** `goal/wave1-observability` (from `main` @ `e17ea24`)
**Status:** Phase-1 plan. **No build in this commit.** Phase 2 builds only after an explicit GO.
**Author aid:** grounded in a 5-stream read-only code survey (VaR publisher, EXT1 sentinel, drift comparator, reaper target, test baseline), each cross-checked against live runtime + on-disk anchors on 2026-07-18.

---

## 0. Binding constraints (both phases)

- **repo-only** — no host mutation. All `runtime/*` and `data/*` are gitignored (`.gitignore:7`) and a repo-write leak-guard (`conftest.py` → `chad/testing/repo_write_guard.py`) *fails any test* that writes under working-tree `runtime/`/`data/`. New tests write to `tmp_path` only.
- **no systemd calls** — never invoke `systemctl`. New `.service`/`.timer` units are added **as repo files** (the established pattern: units live in `ops/systemd/` and `deploy/`; OS install/enable is a separate operator step, out of scope). Honor the GAP-032 wants-lint (`scripts/lint_systemd_wants_symlinks.sh`, test `chad/tests/test_gap_032_systemd_wants_lint.py`) — adding plain unit files creates no wants symlinks, so it stays exit 0.
- **no runtime mutation** — never run a publisher/reaper/`--execute` against real `runtime/`. Detection is read-only; purge is exercised only against `tmp_path` sqlite in tests.
- **no `live_loop.py` signal-path changes** — treat `chad/core/live_loop.py` as untouchable. `run_once(logger)` (`chad/core/live_loop.py:1540`–~3325) is the hot path and peripheral helpers share the module. **None of the four items require touching it.**
- **suite green vs known-5 after every commit** — see §1.
- **commits `W1A`** — one concern per commit, subject `W1A-<n>: …` (matches the `CEW1-W1a` / `FU1-B2` / `CBAR-S1` convention). Local commits only; **push is not assumed** (recent waves land LOCAL / push-DENIED).

### Verification sequence (run after every commit)
```
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -q 2>&1 | tail -25        # FULL run (no -x) to see the whole baseline
CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview 2>&1 | tail -30
```
> CLAUDE.md shows `pytest -x`; `-x` stops at the first known failure, so it cannot confirm "green vs known-5". Use the full run for the gate, `-x` only for fast local iteration.

---

## 1. Test baseline (freeze BEFORE building)

**Action W1A-0 (Phase 2, first step):** one full `pytest chad/tests/ -q` run on this branch tip; capture the failing node IDs as the frozen baseline.

**Expected deterministic known-5:**
| # | Test | Nature |
|---|------|--------|
| 1–3 | `chad/tests/test_tier_manager.py` (×3) | CAD re-pricing, driven by the **uncommitted** working-tree CAD-relabel config (`config/tiers.json`, `config/withdrawal_policy.json` are modified in the tree) — not a code regression |
| 4 | `chad/tests/test_futures_expiry_gate.py` | MES bar-provider dependency |
| 5 | `chad/tests/test_quarantine_sidecar.py` | delta-loss-guard |

**Plus up to 2 non-deterministic flakes** (heartbeat two-writer race, not part of the 5): `test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only` and `test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only`. They only *read* live `runtime/decision_trace_heartbeat.json`; a rerun flips pass↔fail. **Do not add new tests that read live runtime files.** If a post-commit run shows a 6th/7th red, re-run once to confirm it is one of these two before proceeding.

"Green vs known-5" = the failing set after a commit is a **subset of** {the 3 tier_manager + futures_expiry_gate + quarantine_sidecar (+ the 2 known flakes)}. Any *new* red blocks the commit.

---

## 2. Headline: Wave-1 substantially overlaps the already-shipped Exterminator Sentinel

The single most important finding for this plan: **much of what the four-item brief asks for already exists**, mostly inside `chad/ops/exterminator_sentinel.py` (EXT1, commit `a11eeaa`, and its units are **installed + active** on the host — contradicting the memory note that said "units NOT installed"). Each item's section has a "Premise check vs brief" up front. Summary:

| Item | Brief premise | Reality |
|------|---------------|---------|
| 1 | "publisher **fix** kills EXS1" | Publisher already writes a valid `var_state.v1`; it froze because **no timer runs it**. The kill is the **timer**, not a timestamp fix. The real code fix is **input-freshness gating** (a different bug). |
| 2 | "add sentinel **overall rollup** fail/warn/ok" | **Already implemented**: `run()` computes `overall_status = worst_status(...)`, writes it to both report files, CLI prints it. Tested. The genuine gap is *driver-attribution + a downstream consumer*. |
| 3 | "drift comparator **v4** with `positions_snapshot.json` independent leg" | The independent-leg comparator **already exists and runs live as sentinel check EXS4** (`check_reconciliation_drift`). Item 3 = lift that logic into the `position_guard_drift.json` emitter behind a default-off flag. |
| 4 | "store status-**reaper**, dry-run default" | An incumbent reaper exists (`ops/sqlite_retention.py`, weekly, live timer) but is **age-only, un-gated, real-delete-by-default** — the opposite of the brief. Item 4 is a *new status-aware, gated, dry-run-default* sibling. |

**Three decisions this changes** (flagged inline, listed in §8): the Item-2 re-scope, the Item-3 RED-gate coupling, and the Item-4 harden-incumbent-vs-new-script fork.

---

## 3. Item 1 — VaR publisher timer + publisher fix (clears EXS1's `var_state` row)

### Premise check vs brief
- The publisher `ops/var_publisher.py::publish_var` (`:62`) already renders a schema-correct `var_state.v1` (`chad/risk/portfolio_var.py::report_to_state_dict:385`, `schema_version` pinned `:388`) and writes `runtime/var_state.json` atomically with `ts_utc = utc_now_iso()` (write-time) and `ttl_seconds = 3600`.
- **There is no unit that runs it** — grep of `ops/systemd/`, `deploy/`, host `systemctl list-unit-files` → zero VaR units; no importer/caller of `publish_var`. The artifact froze at `ts_utc=2026-05-07T13:39:30Z` (~72d stale). That is why EXS1's `var_state` feed row fails.
- **EXS1 is not a VaR check** — it is the generic `check_stale_feeds` (`chad/ops/exterminator_sentinel.py:312`); `var_state` is one config row (`config/exterminator.json:24`, `ttl_verified=true`, `fail_after_seconds=7200`). EXS1 keys purely on `ts_utc` age > `fail_after`. **A timer at ≤3600s cadence clears the row.** No VaR math change does that.
- The historical "stale-as-fresh" bug lived in the **consumer** (`metrics_server` exporting `status_ok=1` ignoring age) and was **already fixed** by A4/P0A (`chad/ops/metrics_server.py:550`, `var_ok = status=="ok" and not var_stale`). A "fix" that stamps `ts_utc` from data-time would make the artifact *older* and trip EXS1 *harder* — do not do that.

### The genuine code fix (what "publisher fix" should mean)
The publisher never validates **input** freshness (`data/bars/1d/*`, `runtime/price_cache.json`, `runtime/positions_truth.json`). With a timer, it would recompute hourly and stamp `ts_utc=now` **even against stale bars** — the stale-as-fresh failure re-emerging one layer down, invisible to EXS1 and A4 (both inspect only the artifact's own `ts_utc`). Fix: **gate on input freshness** and reflect it in the artifact.

### Files to touch
- **NEW** `ops/systemd/chad-var-publisher.service` — oneshot, `ExecStart=…/venv/bin/python3 -m ops.var_publisher`, `WorkingDirectory`+`PYTHONPATH=/home/ubuntu/chad_finale`, `ProtectSystem=full`, `ReadWritePaths=…/runtime`, `OnFailure=chad-service-alert@%N.service`, `TimeoutStartSec=120` (template: `ops/systemd/chad-exterminator-sentinel.service`).
- **NEW** `ops/systemd/chad-var-publisher.timer` — `OnBootSec=180`, `OnUnitActiveSec=1800` (30 min; comfortably < the 3600s TTL and the 7200s EXS1 floor, with margin), `Persistent=true`.
- **EDIT** `ops/var_publisher.py` — add input-freshness gating: before stamping, check the max input age; if any required input exceeds a staleness bound, downgrade the published `status` (e.g. `insufficient_data`/`stale_inputs`) instead of `ok`, so the A4 consumer (`var_ok = status=="ok" and not stale`) correctly reports not-ok. Add an `--allow-stale-inputs` escape hatch (default off) for manual diagnostic runs.
- **EDIT** `chad/risk/portfolio_var.py::report_to_state_dict` (`:385`) — additively surface `inputs_fresh: bool` + `oldest_input_age_seconds` + `oldest_input` in `var_state.v1` (additive keys; keep `schema_version` unless a bump is warranted — prefer additive to avoid touching every reader). Keep `enforcement_active=False` (VaR gates no trading — this is observability hygiene only).
- **NEW** `docs/VAR_PUBLISHER_INSTALL.md` — operator install/enable steps (mirrors `docs/EXTERMINATOR_SENTINEL_INSTALL.md`); makes explicit that **the repo change does not activate the timer** — until an operator installs+enables it, EXS1's `var_state` row stays red.
- **NEW** `chad/tests/test_w1a_var_publisher_freshness.py`.
- **UNCHANGED / re-run only:** `chad/tests/test_portfolio_var_drawdown_report_only.py`, `test_p0a_var_staleness_guard.py`, `test_exterminator_sentinel.py::test_stale_feeds_*` — confirm they still pass (the additive schema keys must not break `test_var_publisher_writes_schema`).

### Approach
1. Add the two unit files + install doc (pure file writes).
2. Add input-freshness gating in `var_publisher.py`, reading input mtimes/timestamps via a small helper; thresholds configurable via constant + env override (`CHAD_VAR_INPUT_MAX_AGE_SECONDS`, default e.g. 172800s/48h to tolerate weekends).
3. Additive `var_state.v1` fields via `report_to_state_dict`.
4. Never execute the publisher against live `runtime/` — tests drive `publish_var(..., runtime_dir=tmp_path)` with synthetic inputs.

### Test plan
- Fresh inputs → `status="ok"`, `inputs_fresh=true`.
- One stale input (old mtime) → `status != "ok"`, `inputs_fresh=false`, `oldest_input` named; A4 metrics path would export `chad_var_status_ok=0`.
- `--allow-stale-inputs` overrides the downgrade (diagnostic).
- Schema-parity: additive keys present; existing required keys (`ts_utc`, `ttl_seconds`, `status`, `method`) intact.
- Unit-file lint: `scripts/lint_systemd_wants_symlinks.sh` exit 0 unchanged.

### Risks
- **Cadence mismatch:** consumers disagree — EXS1 = 3600/7200s, R02+metrics = 86400s. Timer must clear the strictest (EXS1). 30-min cadence does. **Do not pick a >1h cadence** or EXS1 still fails.
- **Additive schema vs bump:** additive keys avoid a `var_state.v2` migration across `metrics_server`/`health_monitor_rules`/`coach_voice`/sentinel-config; prefer additive. If a bump is needed, that ripples to `config/exterminator.json:30` (`ttl_source` cites `:388`) and every reader — treat as a separate item.
- **Activation gap:** the timer is inert until an operator installs it (no `systemctl` here). Plan output = repo change complete + tested; runtime EXS1 clears only post-deploy. State this plainly in the PA/install doc.
- **False downgrade on legit weekend staleness:** the 48h threshold must tolerate normal market-closed gaps; make it env-tunable and document the choice.

---

## 4. Item 2 — Sentinel overall rollup (fail/warn/ok)

### Premise check vs brief — **the rollup already exists**
`chad/ops/exterminator_sentinel.py::run()` already computes and writes it:
- `counts = {ok,warn,fail}` (`:1111`), `overall = worst_status(c.status for c in checks)` (`:1115`, aggregator `:73`).
- Written to `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` as `overall_status` (`:1122`) + `counts` (`:1123`), and to the history ndjson (`:1150`). CLI prints `overall=… ok=.. warn=.. fail=..` (`:1328`).
- Vocabulary is already exactly `ok`/`warn`/`fail` (constants `:61-63`). Tested: `test_worst_status_ordering`, `test_report_schema_and_eight_checks`, `test_failing_check_does_not_trigger_any_repair`. Live report currently reads `overall_status="fail"`.

**So "add a fail/warn/ok rollup" is a no-op.** The genuine, still-missing pieces:
1. **No driver attribution** — the rollup says "fail" but not *which* checks drove it. An operator must open all 8.
2. **No downstream consumer** — no health rule reads `EXTERMINATOR_SENTINEL_LATEST.json`; `overall_status` feeds nothing. The only outward signals are `maybe_notify` (Telegram on any `fail`, `:1158`) and systemd `OnFailure`.
3. **Structural quirks** any rollup logic must respect: EXS6 (`dirty_git`) and EXS8 (`ml_anomalies`) **can never emit `fail`** (top out at warn); the EXS999 self-check is always `warn`; "stale/blind/unknown" are folded into `warn` (not distinct statuses).

### Recommended W1A scope (additive, in-module, low-risk) — **DECISION D1**
Deliver **driver attribution** on the existing rollup, and (optionally, pending D1) a **read-only consumer**:
- **Primary:** enrich the report with an additive `rollup` block naming the drivers — `{overall, first_fail, driver_check_ids: [...], driver_reason}` — computed from the same `checks` list. Purely additive to `run()`'s return dict; no vocabulary change, no new status values.
- **Optional (D1):** a read-only health-monitor rule `R-XX` in `chad/ops/health_monitor_rules.py` that reads the LATEST report and emits a **NOTIFY_ONLY** finding mirroring `overall_status` (deduped against the sentinel's own `maybe_notify` to avoid double-alerting). This is peripheral ops code, **not** the live_loop signal path.

Alternative (if the user wants strictly minimal): **verify + document only** — assert the existing rollup with a couple of tests and a docs note; ship no new behavior.

### Files to touch
- **EDIT** `chad/ops/exterminator_sentinel.py` — additive `rollup` block in `run()` (near `:1117`). No change to `worst_status`, `_STATUS_RANK`, `CheckResult`, or the status vocabulary.
- *(D1 optional)* **EDIT** `chad/ops/health_monitor_rules.py` — add read-only `R-XX` reading `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json`; NOTIFY_ONLY; dedupe-safe.
- **EDIT/EXTEND** `chad/tests/test_exterminator_sentinel.py` (or **NEW** `chad/tests/test_w1a_sentinel_rollup.py`) — assert the `rollup` block; assert that EXS6/EXS8-only-warn worlds roll up to `warn`, all-ok → `ok`, any-fail → `fail`, and `driver_check_ids` name the failing checks.

### Test plan
- all-ok fixture → `rollup.overall="ok"`, empty drivers.
- one EXS-fail fixture → `overall="fail"`, `first_fail` + `driver_check_ids` = that check.
- warn-only (e.g. EXS6 dirty_git) → `overall="warn"`, drivers = warners.
- schema stability: existing `overall_status`/`counts`/`checks` keys unchanged; `read_only_confirmed=True`, `runtime_files_modified=[]` still hold (anti-auto-heal locks).

### Risks
- **Re-implementing an existing feature.** The whole risk of Item 2 is doing redundant work; hence the re-scope to attribution + consumer.
- **Double-alerting** (if D1) — a health rule + `maybe_notify` both firing on the same fail. Keep the rule NOTIFY_ONLY and dedupe on a stable identity (title, not fluctuating values — the CTF T2 lesson).
- **CLI exit code is intentionally always 0** (`:1332`, to avoid EXS5 self-trip). Do **not** "fix" it to reflect severity — that is a deliberate non-feature; changing it would make the sentinel's own unit fail and trip EXS5.

---

## 5. Item 3 — Drift comparator v4 (independent `positions_snapshot.json` leg, `CHAD_DRIFT_V4` default off)

### Premise check vs brief — self-comparison CONFIRMED; independent leg already exists as EXS4
- **Confirmed self-comparison:** the `position_guard_drift.json` emitter runs `detect_guard_vs_broker_drift_v2` (`chad/core/position_guard.py:556`; the emitter is `chad/ops/reconciliation_publisher.py::_emit_position_guard_drift:244`, stamps `position_guard_drift.v3` `:307`). Both legs derive from the **single** `position_guard.json`: the `broker_sync|…` rows vs the strategy rows. There is **no v3 detector function** — "v3" is only the output schema label. The genuinely independent broker read (clientId=83) exists in the same publisher but feeds a *different* file (`reconciliation_state.json`), not the drift file.
- **The independent-leg comparator already exists and runs live** as sentinel **EXS4** `check_reconciliation_drift` (`chad/ops/exterminator_sentinel.py:657` + helpers `:1239-1299`): reads `runtime/positions_snapshot.json` (clientId=99, TTL 300s, refreshed ~5min, live-verified fresh) **and** `position_guard.json`; **freshness-gates** the independent leg (`age > ttl*3` → WARN "blind", never a false OK); aggregates **three legs separately** (independent broker / guard broker-mirror / guard strategy) and **never sums mirror+strategy** (the guard dual-books the same shares — live-proven for UNH/SVXY/V/IWM); emits `mirror_vs_independent_broker` etc.
- **`CHAD_DRIFT*` is greenfield** — grep confirms zero existing `CHAD_DRIFT*` env in code/units.

**So Item 3 = lift EXS4's independent-leg comparison into the drift emitter behind `CHAD_DRIFT_V4` (default off).**

### Approach
1. **NEW pure detector** `detect_guard_vs_independent_snapshot_drift(guard_state, snapshot, *, excluded_symbols, now)` in `chad/core/position_guard.py` — three-leg, dual-booking-aware, freshness-gated, no disk I/O (mirrors the `test_position_guard_same_side_hardening` purity invariant). Reuse `_position_key`/`_signed_qty`. Adding a pure function to `position_guard.py` is safe — it is **not** `live_loop.py`, and the function is **not** called from the signal path.
2. **Wire behind the flag** in `_emit_position_guard_drift` (`reconciliation_publisher.py:244`): when `os.environ.get("CHAD_DRIFT_V4")` is truthy, load `runtime/positions_snapshot.json` and emit the v4 result; **default off ⇒ the v2 → v3 path is byte-for-byte unchanged** ⇒ suite stays green and live behavior is untouched.
3. **Observability-only, sibling artifact — DECISION D2.** Recommend emitting a **separate** file `runtime/position_guard_drift_v4.json` (schema `position_guard_drift.v4`) rather than mutating `position_guard_drift.json`. This guarantees **zero** risk to the existing v3 consumers (`ops/live_readiness_publish.py:194` RED gate, `service_failure_alert.py`, EXS4, `coach_intents`). **v4 does NOT feed the live-readiness RED gate in Wave-1** — a new comparator must not be able to flip live-readiness until it has soaked. (D2 = confirm sibling-file + observability-only vs. later promoting `mirror_vs_independent_broker` into `drift_count`.)
4. **Fail-closed** on a stale/blind/missing snapshot: mark the v4 result `independent_leg="blind"` and do **not** zero-out or falsely indict — never let a stale snapshot read as "all agree."

### Files to touch
- **EDIT** `chad/core/position_guard.py` — new pure detector (additive; signal path untouched).
- **EDIT** `chad/ops/reconciliation_publisher.py` — flag-gated v4 emit to the sibling file; default-off no-op.
- **NEW** `chad/tests/test_w1a_drift_v4.py` — detector unit tests + publisher flag on/off tests (tmp_path).
- **UNCHANGED:** `chad/ops/exterminator_sentinel.py` (leave EXS4 alone this wave — note the intentional short-term duplication + a follow-up to converge on a shared helper, to avoid destabilizing a live check).

### Test plan
- Detector: agreeing legs → no drift; guard mirror shows +N but snapshot shows −N (side flip) → `mirror_vs_independent_broker`; guard holds, snapshot empty → `phantom_guard_entry`; **dual-book fixture** (strategy + mirror both +273, snapshot +273) → **no** spurious 2× drift; excluded symbol → info-only.
- Freshness: snapshot older than `ttl*3` → `independent_leg="blind"`, WARN-shaped, never OK/false-indict; missing snapshot → blind, no crash.
- Publisher: `CHAD_DRIFT_V4` unset → only `position_guard_drift.json` written, identical to today; set → sibling `position_guard_drift_v4.json` also written; existing v3 file and `drift_count` semantics unchanged either way.

### Risks
- **Dual-booking (highest):** never sum mirror + strategy. Compare each leg to the snapshot separately (EXS4's model). A naive single-file aggregation invents 2× phantoms.
- **Futures symbol ambiguity:** guard keys carry no `secType`/contract-month; snapshot aggregates by bare symbol. Latent only (book is 100% STK now); v4 can exploit the snapshot's `conId`/`secType` to disambiguate, but the guard side lacks them — keep futures out of v4's actionable set initially.
- **Crypto invisible:** `positions_snapshot` is IBKR-only; Kraken lots absent. v4's independent leg covers equities/futures only — document the blind spot.
- **clientId scoping / transient divergence:** snapshot=99 vs reconciliation read=83; both hit the same paper account but can diverge during gateway resets. Trust the snapshot's **own freshness**, never assumed liveness.
- **Divergence from EXS4:** two copies of the logic can drift apart. Mitigate by a follow-up that factors a shared helper; for Wave-1, keep v4's semantics identical to EXS4 and cross-reference in comments.

---

## 6. Item 4 — Store status-reaper (read-only detection + gated purge, dry-run default)

### Premise check vs brief — target identified; incumbent contradicts "dry-run default"
- **Target store:** `runtime/ibkr_adapter_state.sqlite3`, table `ibkr_exec_state` — CHAD's **own** persisted idempotency/dedup ledger (`chad/execution/ibkr_adapter.py`: path `:342`, store `:830`, schema `:846`). Columns `idempotency_key, status, created_at_utc, updated_at_utc, broker_order_id, payload_json, result_json`. **Not live broker truth** — `status` is CHAD's last-observed snapshot; live order truth comes from `reqAllOpenOrdersAsync()` separately.
- **Stale rows accumulate:** non-terminal statuses — `PendingSubmit`/`PreSubmitted`/`Submitted`/`claimed`/`duplicate_open_order`/`ValidationError` (classifier `:790-812`: terminal-positive=`{filled}`, terminal-negative=`{cancelled,apicancelled,rejected,inactive,error}`, else non-terminal). Existing cleanup is **lazy** (per-key `claim_or_reclaim` only when the *same key* re-submits) — an orphaned key never re-submitted lingers forever. That is the P1-8 / GAP-036 root cause (`docs/CHAD_GAPS_TO_CLOSE.md:82`).
- **Incumbent reaper exists and is the opposite of the brief:** `ops/sqlite_retention.py` deletes `ibkr_exec_state` rows by **age only** (`DELETE … WHERE updated_at_utc < cutoff` `:143`), **status-blind (also deletes `Filled` evidence)**, **no SCR/exec_mode/LiveGate gate**, **no archive**, and **dry-run is opt-in not default** — wired to a **live weekly timer** (`chad-sqlite-retention.timer`, enabled/active).

### Approach — new status-aware sibling, reusing the gate module — **DECISION D3**
Ship a **new** script; **do not** modify `ops/sqlite_retention.py` or its live timer this wave (that would change a live-scheduled unit's semantics — out of the surgical, additive lane). Note "harden the incumbent (gates + dry-run default)" as a follow-up PA.

- **NEW** `scripts/reap_ibkr_exec_state.py`:
  - **Detection (default, read-only):** open the sqlite **read-only**, classify rows (terminal vs non-terminal via the adapter's own vocabulary), report counts + the full delete-candidate list + retain list; **mutates nothing**.
  - **Purge (`--execute` + typed `--confirm REAP-IBKR-EXEC-STATE`):** reuse the gold-standard gating — import `run_gates` / gate helpers from `scripts/reconcile_ledger_to_broker.py` (`_gate_exec_mode_paper:520`, `_gate_scr:531`, `_gate_reconciliation_not_red`, aggregated `run_gates:558`), the **repo-root bootstrap (F1)** so lazy `chad` imports resolve from any CWD, **archive-before-mutate** (copy the sqlite to `.bak_reap_<UTC>` with sha256, mirroring `_backup:615` and the on-disk `.bak_pre_reap_*` precedent), and atomic replace. `--execute` without the exact token → refuse (rc 2), matching `reconcile_ledger_to_broker` semantics.
  - **Purge criteria (BOX-008 pattern):** delete **non-terminal rows older than a threshold**; **preserve** terminal `Filled`/`Cancelled` evidence and recent rows. Threshold via `--older-than-days` (no default that could surprise; require it, or default to a conservative long margin).
  - **Safety against double-trade:** deleting a dedup row whose order is *still working at the broker* would drop the idempotency guard and permit a duplicate submission. Wave-1 default = **age-margin only, no broker connect** (repo-only/no-live posture); document that `--execute` must only be run by an operator after the standard checklist. An optional `--broker-probe` (calls `reqAllOpenOrders`, purge only `ABSENT`) is noted as a **follow-up enhancement**, not built this wave (it requires a live connection).
- **NEW** `chad/tests/test_w1a_reap_ibkr_exec_state.py` — tmp_path sqlite (never touch real runtime; the write-guard enforces this).
- **NEW** `ops/pending_actions/W1A_reaper_purge_authorization.md` — the operator PA to actually run `--execute` against runtime (out of build scope), plus the "harden incumbent" follow-up note.

### Test plan (templates: `test_reconcile_ledger_to_broker.py`, `test_epoch_reset_bootstrap.py`)
- **Detection read-only:** run default mode over a seeded tmp sqlite → correct candidate/retain counts, file byte-identical after (`sha256` unchanged).
- **Dry-run default:** no `--execute` → mutates nothing.
- **Token discipline:** `--execute` without `--confirm` → rc 2; wrong token → rc 2; correct token → applies.
- **Selectivity:** correct token deletes only stale non-terminal rows; **preserves** `Filled`, `Cancelled`, and recent non-terminal rows; a `.bak_reap_*` archive exists post-run.
- **Idempotent rerun** → rc 0, no further deletes.
- **Gate refusals:** exec_mode=live / SCR unsafe / reconciliation RED → refuse before any write (drive via the imported gates with a temp runtime dir).

### Risks
- **Double-trade if a still-live order's guard is dropped** — the one real hazard (split-brain in `position_guard`/`trade_closer` does **not** apply; this store only governs duplicate-submission protection). Mitigate with a conservative age margin (an order older than N days is not still "working") and operator confirm; the broker-probe cross-check is the stronger control, deferred.
- **Wrong-store hazard:** `runtime/exec_state_paper.sqlite3` (Kraken trusted-lot FIFO + 2148 Stage-2 evidence rows) is **NOT safe to reap** and is explicitly out of scope; `runtime/ibkr_exec_state.db` is a dead 0-byte legacy file. The script must hard-target `ibkr_adapter_state.sqlite3::ibkr_exec_state` and refuse any other path.
- **Overlap with the incumbent weekly delete** — the reaper and `sqlite_retention` both touch the same table; keep them non-conflicting (reaper is status-aware/gated/manual; incumbent stays age-only/auto). Document the relationship.
- **`--execute` against real runtime is forbidden in build/test** — every test uses `tmp_path`; there is no code path in this wave that runs `--execute` on `runtime/`.

---

## 7. Proposed commit sequence (Phase 2, each verified green-vs-known-5)

| Commit | Concern |
|--------|---------|
| `W1A-0` | (no code) freeze the test baseline via one full run; record failing node IDs |
| `W1A-1` | Item 1a — input-freshness gating in `ops/var_publisher.py` + additive `var_state.v1` fields + tests |
| `W1A-2` | Item 1b — repo-side `chad-var-publisher.service` + `.timer` + `docs/VAR_PUBLISHER_INSTALL.md` |
| `W1A-3` | Item 2 — additive `rollup` driver-attribution block in the sentinel + tests *(consumer rule pending D1)* |
| `W1A-4` | Item 3a — new pure `detect_guard_vs_independent_snapshot_drift` in `position_guard.py` + detector tests |
| `W1A-5` | Item 3b — `CHAD_DRIFT_V4` (default off) wiring in `reconciliation_publisher.py` → sibling `position_guard_drift_v4.json` + tests |
| `W1A-6` | Item 4a — `scripts/reap_ibkr_exec_state.py` (detect + gated purge, dry-run default) + tests |
| `W1A-7` | Item 4b — `ops/pending_actions/W1A_reaper_purge_authorization.md` + incumbent-hardening follow-up note |

One concern per commit; `python3 -m py_compile` + full `pytest chad/tests/ -q` + `full_cycle_preview` after each. Tag only if the user designates a Wave milestone.

## 8. Decisions required before/within Phase 2 (GO gates)

- **D1 (Item 2):** the literal "add rollup" is already done. Confirm the re-scope: **(a)** driver-attribution only, **(b)** attribution + a read-only NOTIFY_ONLY health rule consumer, or **(c)** verify+document only. *Recommend (a) now, (b) as a fast-follow.*
- **D2 (Item 3):** v4 output as a **sibling observability file** that does **not** touch the live-readiness RED gate (recommended), vs. later promoting `mirror_vs_independent_broker` into `drift_count`. *Recommend sibling/observability-only for Wave-1.*
- **D3 (Item 4):** **new gated sibling script** (recommended) vs. hardening the incumbent `ops/sqlite_retention.py` in place (touches a live timer's semantics — deferred to a PA).

## 9. Explicit non-goals / out of scope
- No `systemctl`, no install/enable of any unit, no service restart.
- No writes to `runtime/`/`data/`; no running the VaR publisher or the reaper `--execute` against real runtime.
- No `chad/core/live_loop.py` changes (whole file off-limits).
- No change to the live v3 drift path when `CHAD_DRIFT_V4` is off; no change to the incumbent `sqlite_retention` behavior/timer; no change to EXS4.
- No config mutation applied directly (risk caps / live mode / strategy config remain Pending-Actions-only); the currency-audit working-tree config edits are not part of this wave.
- Push is not assumed; commits land local pending explicit GO.
