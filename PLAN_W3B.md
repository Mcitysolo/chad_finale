# PLAN_W3B — Wave 3 Lane B: Small-Fixes + Observability Bundle

- **Status**: PHASE 1 — PLAN. No code written. Awaiting GO + decision-point answers.
- **Worktree**: `/home/ubuntu/chad_w3b`, branch `goal/wave3-smallfixes`, base `488460e` (main).
- **Lane boundary**: Lane A owns `chad/validation/` and `PLAN_W3A.md` — this lane will not touch either. (Item 3 *reads* `chad/validation/trade_log_adapter.py` read-only to confirm additive-schema tolerance; it will not modify it.)
- **Constraints**: worktree only; no runtime mutation; no systemd installs (repo units + install docs only); set-diff methodology; one item per commit, `W3B-N` prefixes; verification sequence after every change.
- **Method**: six parallel read-only investigations against the live tree (2026-07-22 ~20:25–20:30Z evidence reads). Every brief was tested against ground truth before planning.

---

## 0. Premise scorecard (what the brief said vs. what is true)

| # | Brief premise | Verdict | Ground truth |
|---|---|---|---|
| 1 | EXS9 stale-process check; ":9618 month-old-math" | **PARTLY WRONG** | `:9618` is a TCP **port** (chad-backend uvicorn serving SCR shadow math in-process), not a PID. That instance already remediated (backend restarted 2026-07-20 per W2A runbook Step 5). EXS9 = the generalization BOX-034A:58 recorded as an unbuilt "candidate ops item". Sentinel is **live** on a 5-min oneshot timer, not shelf-ware. |
| 2 | qty_mismatch / broker_untracked gate readiness, page nobody | **CONFIRMED, narrower** | No alert fires on drift-file *content* (drift-only RED is silent), but reconciliation RED already pages. Real kind names: `qty_mismatch` / `broker_untracked_position` / `phantom_guard_entry` + info-only `mixed_ownership_info`. Current live state: `drift_count=0` (the old "5 @1.9×" was fixed by PFF1 `31f5517` and reclassified by A5 `b27890a`). |
| 3 | Overlay evidence priced 55s stale vs broker fill | **MISATTRIBUTED** | The 55.3s observation is the **paper executor's SIM fill quote stamp** (PA_SIM_MARK_freshness_2026-07-20.md: UNH ref 423.00 vs fill 424.88). Overlay evidence records carry **no mark timestamp at all** — staleness is invisible, not measured. Both lanes discard the mark's ts at the loader boundary. |
| 4 | drawdown_state (P1-3) + ibkr_watchdog (P1-4) orphaned — revive | **PREMISE DEAD** | Both **alive and fresh** (21s / 20s old at check; 120s timers, enabled). P1-3/4 CLOSED 2026-06-17 (DEFENSE_BOARD_RECONCILIATION) and re-confirmed in SSOT v9.8:428. Earlier audit was misled by the timer-triggered oneshot service showing `disabled`. Residual gaps are real but different (see §4). |
| 5 | Registry 3-way mismatch (P1-1): build single registry + tripwire | **PREMISE DEAD** | Already **shipped 2026-06-17** (`ff58803` canonical `chad/strategy_registry.py` 18/16/2 + `1ff9a62` startup tripwire; SSOT v9.8:426 marks P1-1 CLOSED). "18/16/2" is the shape of the *fix*, not the mismatch. Residual: 7 hand-maintained perimeter lists the tripwire does not cover (see §5). |
| 6 | P3 hygiene: dedupe files, logrotate, dynamic_caps archive | **MIXED** | Dedupe: unbounded-growth half stale (BOX-042 + stable identity fixed it; 65 files / 2.8 KB steady state); loose-in-runtime half correct. Logs: mostly journald-bounded; genuinely unbounded = `claude_client.log` (12.4 MB, active), `gpt_client.log`, cron consensus log. dynamic_caps: **NOT hygiene-safe as briefed** — the family is live infrastructure; `dynamic_caps_quarantine.json` (mtime March) is a **live-read input every orchestrator cycle** (correlation overlay raises if absent). Only a narrow module slice is archivable. |

**Cross-cutting activation fact** (affects every item): several consumers are `Type=oneshot` timer services re-exec'd every run (exterminator-sentinel 5-min, reconciliation-publisher 5-min, live-readiness 10-min). **Merging this branch into the live tree's `main` IS deployment for those items** — no restart step exists to gate them. Long-running services (live-loop, metrics, telegram-bot, backend) only pick changes up at their next gated restart. The merge itself therefore needs explicit GO, and the plan tags each item ONESHOT-LIVE-AT-MERGE vs RESTART-GATED.

---

## 1. EXS9 — stale-process sentinel check

**Goal**: detect long-running services whose process started *before* their code last changed (fresh artifact timestamps hiding month-old math — the chad-backend/:9618 lesson; the BOX-034A ~15-day orchestrator precedent).

**Design** (copies the sentinel's existing check pattern exactly):
- New method `check_stale_processes()` → `CheckResult(check_id="EXS9", name="stale_processes", ...)` in `chad/ops/exterminator_sentinel.py`, registered via one `_safe("stale_processes", ...)` line after `:1137`. Rollup (`_build_rollup`) picks it up automatically — no rollup changes.
- New **injectable provider** `default_service_uptime_provider` mirroring `default_systemctl_provider` (:157-171): `systemctl show <unit> -p ActiveEnterTimestamp,MainPID --no-pager` — pure read, already permitted verbatim by the mutation-token lock tests (`show` is whitelisted; no test weakening).
- Code-side timestamp: `git log -1 --format=%ct -- <path> [<path>...]` per service (read-only; permitted by lock tests as written). **Import-graph-scoped per BOX-034A §3** ("a Python service reloads ONLY its own import graph") — naive HEAD-time comparison over-alerts on unrelated commits (proven today: live-loop restarted 14:08, HEAD `488460e` 19:48 is validation-only).
- New config section `stale_processes` in `config/exterminator.json`: per-unit rows `{unit, code_paths[], grace_seconds, _note}` + `excluded_units[]` with justification notes.
- **Covered units** (long-running, repo-code): chad-backend (`backend/`, `chad/analytics/`), chad-dashboard, chad-kraken-ws, chad-live-loop, chad-metrics, chad-orchestrator, chad-shadow-status, chad-telegram-bot, chad-ibkr-bar-provider.
- **Justified exclusions**: chad-ibgateway (non-repo Java + nightly 03:15 restart), chad-xvfb / chad-x11vnc (X infra), chad-paper-shadow-runner (masked, PR-06 Path A), all `Type=oneshot` timer services (fresh code by construction every fire).
- **Severity: `warn` maximum, never `fail`.** Code-newer-than-process is frequently the *governed* deploy-pending state (CLAUDE.md rules #6/#7 forbid restarts without instruction; BOX-059 gate 1 deliberately rewards long MainPID stability). A `fail` would page on correct behavior and poison EXS5/overall. Since `maybe_notify` pages only on `fail`, a warn-capped EXS9 **never pages** — it surfaces in the report, history, and rollup only. This is intentional: an operator-facing dashboard fact, not an alarm.
- **Tests**: move the two hard pins (`test_exterminator_sentinel.py:242` and `:262` pin exactly EXS1..EXS8 → EXS1..EXS9; rename `test_report_schema_and_eight_checks`); add EXS9 unit tests with injected uptime/git providers (stale, fresh, excluded, provider-error → EXS999 path). Do NOT weaken `test_no_mutation_tokens_in_source` / `test_subprocess_calls_are_read_only_queries` (deploy PA forbids it) — design requires no weakening.
- **Activation**: ONESHOT-LIVE-AT-MERGE (sentinel timer re-execs every 5 min). Zero paging risk by design (warn-cap).

## 2. Drift-content transition alerting

**Goal**: a drift-only condition (reconciliation GREEN + `drift_count>0`) currently flips live-readiness RED with zero notification. Alert on drift-set **transitions** (appeared / resolved), coach-voiced, dedupe-stable.

**Design** (write-site pattern — the emitter already holds the payload; no polling lag/double-read race):
- Hook: `chad/ops/reconciliation_publisher.py::_emit_position_guard_drift()` (:249), after the atomic `tmp.replace(...)` at :326 (or at the `main()`:421 call site — function already returns `drift_count`). Publisher already imports `telegram_notify.notify` (:478, :542).
- Transition state: `runtime/position_guard_drift_last_alerted.json` (atomic tmp+replace), storing the last-alerted identity set — same precedent as `scr_last_notified_state.json` / `phase8_alert_state.json`. (New runtime file is written by the *deployed service* post-merge, not by this lane — no runtime mutation in Phase 2.)
- **Stable identity** (CTF-T2 rule: values in evidence, never in identity): per-record identity = `(drift_kind, symbol)` only. Quantities, `snapshot_generation`, `guard_keys` excluded. `appeared = current − last`, `resolved = last − current`; alert when either non-empty.
- **`mixed_ownership_info` records are excluded** — info-only by design; alerting on them re-introduces the false-positive class A5 (`b27890a`) killed.
- Formatter: new coach kind `position_drift` (appeared) / `position_drift_resolved` — `_tpl_position_drift*` builders + `_TEMPLATES` entries in `chad/utils/coach_voice.py` (:663); rendered via the existing `_coach_format()` presentation-only contract (fail → plain-text fallback).
- Send: `telegram_notify.notify(msg, severity="warning", dedupe_key=f"position_drift_{stable_identity(...)}")` — a changed drift set mints a new key (never TTL-suppressed); the transition file remains the primary gate; 900s dedupe TTL is belt-and-braces. NOT `critical` — that tier stays reserved for the existing `reconciliation_red` page.
- Whole block wrapped fail-soft (`try/except`) — alerting must never block reconciliation (default_notifier doctrine).
- **Tests**: transition matrix (empty→drift, drift→empty, drift→different-drift, unchanged→silent, info-only→silent), identity stability under qty changes, fail-soft on notify exception.
- **Activation**: ONESHOT-LIVE-AT-MERGE (publisher timer, 5-min). First run after merge treats current set (empty) as baseline — no alert storm possible at `drift_count=0`.

## 3. SIM-mark staleness — stamp the mark, then (optionally) upgrade the source

**Goal**: overlay evidence must carry the mark's provenance and age; divergence like the UNH 55.3s/$513 case must be measurable in every record, and the mark source should move toward freshest broker-truth.

**Re-scoped design** (two tiers — see Decision D4 for how far to go):

**Tier 1 — stamp (zero behavior change, both lanes):**
- Equity: `_default_price_loader` (`chad/risk/position_exit_overlay.py:1262-1290`) already parses `price_cache.json`'s `ts_utc` for the TTL check and then discards it — return it. Add `mark_ts_utc`, `mark_age_s`, `mark_source` (`price_cache` | `bar_close_fallback` + bar date) to `ExitOverlayVerdict` (:244-268) / `to_dict` (:270-293). Bump `EVIDENCE_SCHEMA_VERSION` → `exit_overlay.v2` (additive fields only).
- Crypto: `_default_marks_loader` (`chad/risk/crypto_exit_overlay.py:700-718`) holds `touch.ts_utc` in hand at :711-716 and discards it — same stamp, same fields.
- Loader-boundary change: `PriceLoader`/`MarksLoader` return a parallel meta map (or richer type) so injected test loaders stay simple.
- Lane-A safety: `chad/validation/trade_log_adapter.py` references exit_overlay evidence — verify **read-only** that it tolerates additive fields + a version bump (its fail-closed schema gate admits no-schema Kraken laps per W3A-2, but v2 tolerance must be confirmed, not assumed). If it hard-pins `exit_overlay.v1`, keep `schema_version` at v1 and add fields only (additive-under-v1), and record that choice — we do not touch Lane-A files.
- The stamp makes the 55s-class divergence *visible*; it does not shrink it.

**Tier 2 — freshest broker-truth mark (flag-gated, RESTART-GATED):**
- Best candidate (ranked #1): shared truth `ib` (clientId 99) portfolio marks — `updatePortfolio` events already stream in-process (StartupFetchALL); zero new broker requests. Requires an `ib.updatePortfolioEvent` handler keeping `{symbol: (marketPrice, received_at)}` (ib_async doesn't timestamp PortfolioItems), gated on `api_connected()` exactly like `broker_position_sync.py:69-73` (XOV-2345-class hazard: wrapper caches reset on socket drop).
- Wiring: `PositionExitOverlay.__init__` already accepts injected `price_loader` (:810); `build_default_overlay` (:1380) hardcodes the default — live_loop passes a portfolio-marks-first loader closure behind env flag `CHAD_OVERLAY_PORTFOLIO_MARKS` (default OFF), freshest-stamped-age-wins vs price_cache, `mark_source` recording which won.
- Explicitly NOT doing: new `reqMktData`/`reqTickers` on clientId 99 (governance: read-only-hardened connection; price refresh owns that job on its own clientId).
- Out-of-scope pointer: PA_SIM_MARK remediations (a) fill-mark freshness bound in `paper_exec_evidence_writer.py` (distinct from `_SUBMIT_QUOTE_TTL_S=300` — other readers depend on 300s) and (b) broker fill-price back-fill remain that PA's items; W3B absorbs only the stamp+source scope unless D4 says otherwise.
- **Tests**: stamp presence both lanes, fallback-source labeling, age computation vs injected clock, flag-off byte-path unchanged, (Tier 2) dead-connection → fall back to price_cache with source recorded.

## 4. Publishers — re-scoped: repo-parity + TTL rows + drawdown staleness guard

Premise dead (both fresh); the three genuinely missing pieces, all copied from the VaR pattern:

- **4a — repo-track the installed units**: `chad-drawdown-publisher.{service,timer}` and `chad-ibkr-watchdog.{service,timer}` exist ONLY in `/etc/systemd/system` — zero repo copies (config-drift-only units). Check byte-faithful copies into `ops/systemd/` + an install/parity doc mirroring `docs/VAR_PUBLISHER_INSTALL.md` (repo-only artifacts; operator installs per governance #6/#7). Both installed units lack the VaR service's `OnFailure=chad-service-alert@%N.service` (P0A-A2: publisher deaths page) — see Decision D5 for byte-parity vs. proposed-delta handling.
- **4b — sentinel TTL rows** (`config/exterminator.json` `feeds`, the EXS1 table): neither artifact has a row today. Add `drawdown_state` (`ttl_seconds=300` — `ttl_verified=true`, `ttl_source=ops/drawdown_publisher.py:41`; warn 300 / fail 900) and `ibkr_watchdog_last` (`ttl_seconds=120`, verified, `ts_field=ts_unix`; warn 240 / fail 720). Both artifacts self-declare `ttl_seconds`, and the artifact's own field wins at runtime per config `_doc` governance. Optionally mirror into `feed_watchdog.WATCHED_FEEDS` (:17-27) — the secondary net also lacks both.
- **4c — drawdown gauges staleness guard**: `metrics_server._var_drawdown_lines` (:567-577) trusts `dd_status=="ok"` with **no age check** — the exact stale-as-fresh class A4 fixed for VaR, unfixed for drawdown. Extend the A4 pattern (`_compute_state_staleness`, fail-closed on missing/unparseable `ts_utc`): `chad_drawdown_stale`, `chad_drawdown_age_seconds`, `chad_drawdown_status_ok=0` beyond TTL (default 900s = 3× publisher TTL; env-overridable like `CHAD_VAR_STATE_TTL_SECONDS`).
- **Activation**: 4b ONESHOT-LIVE-AT-MERGE (sentinel); 4c RESTART-GATED (chad-metrics long-running, up since 07-14); 4a inert by construction (docs + unit files, operator-applied).

## 5. Registry — re-scoped: guard the uncovered perimeter

P1-1 closed; the canonical registry + 6-invariant tripwire run at every engine build. Remaining exposure = hand-maintained lists outside the tripwire:

- **5a — delete dead `REAL_STRATEGIES`** (`chad/execution/paper_exec_evidence_writer.py:92-100`, 7 names, zero references) — the single most misleading artifact for future auditors (attribution is denylist-based via `PLACEHOLDER_STRATEGIES`/`_is_real_strategy`, not whitelist-based). Pure deletion.
- **5b — perimeter guards** for the seven uncovered sources (enforcement split per Decision D6):
  1. `KNOWN_STRATEGIES` hand tuple (`strategy_routing_diagnostics.py:54-71`; 17, missing `alpha_intraday_micro`) — derive from registry or pin-test.
  2. `config/per_strategy_loss_limits.json` keys (17: ACTIVE + dormant `alpha_forex`; `alpha_intraday_micro` falls to default) — validate keys ⊆ declared, warn on dormant, flag unknown names.
  3. Chassis sleeve frozensets (`dynamic_risk_allocator.py:484-491`; union currently == ACTIVE exactly, unguarded) — invariant: union == ACTIVE, pairwise disjoint.
  4. `config/regime_activation_matrix.json` names ⊆ declared (dormant `alpha_forex` in trending_bull/bear = warn, not error).
  5. `dominance_allocator.DEFAULT_BASE_WEIGHTS` ⊆ declared.
  6.-7. Display maps (`dashboard/api.py:59-74`, `daily_chad_report.py:157-172`; each missing names, one legacy `crypto` alias) — coverage pin-tests only (cosmetic; `.get(k,k)` fallbacks make gaps non-breaking).
- Placement: extend `assert_registry_consistency()` (`chad/strategy_registry.py:245-378`) for config-vs-registry invariants (lazy imports per its existing style, no cycles) + a new `chad/tests/test_strategy_registry_perimeter.py` for pins. Follow the `WEIGHT_DEFERRED_ALLOWLIST` convention (every allowlisted divergence requires an `ops/pending_actions/*policy.md` artifact).
- **Known intentional divergences to allowlist, not "fix"**: dormant-in-tiers UserWarnings (invariant 6, pinned), loss-limit `alpha_forex` row (side-engine track), regime-matrix `alpha_forex`.
- **Activation**: tripwire changes are import-time/engine-build behavior — RESTART-GATED for live-loop, but new *hard* invariants could brick the engine build on a config typo; see D6 (default recommendation: warn-tier at runtime, hard only in tests).

## 6. P3 hygiene — re-scoped per findings

- **6a — dedupe relocation + policy**: move the per-key family (65 files, 2.8 KB) into `runtime/dedupe/`. Exactly two path sites: `telegram_notify.py:169-173` (`_dedupe_path` → `RUNTIME_DIR/"dedupe"/...` + `mkdir(parents=True, exist_ok=True)` in `_dedupe_mark`) and `ops/cleanup_telegram_dedupe.py` defaults (scan both dirs during a migration window). No caller passes paths — keys only. Policy: ship repo-side `chad-dedupe-cleanup.{service,timer}` in `ops/systemd/` + install doc (BOX-042 left scheduling manual). `telegram_bot_dedupe.json` (separate single-file mechanism) stays put. **Split-brain window flagged**: oneshot senders use the new path at merge; long-running senders keep the old path until restart → dedupe suppression not shared across the two paths → worst case one duplicate alert per key per 900s TTL. Negligible, accepted, documented. Old loose files left in place for the cleanup tool's migration-window scan (no runtime mutation by this lane).
- **6b — log rotation**: the only material unbounded logs are `logs/claude/claude_client.log` (12.4 MB, active; plain `FileHandler` at `chad/intel/claude_client.py:195`), `logs/gpt/gpt_client.log` (`gpt_client.py:166` — docstring falsely claims "rotating"), cron `consensus_update.log`, and daily `calls_*.ndjson` accumulation. journald is already bounded (drop-ins + disk-guard vacuum); `telegram_bot.log` is bounded but **double-written** (RotatingFileHandler + systemd `append:` fd) — leave it alone, document the hazard, never rename-rotate it. Preferred fix per D7: code-side `RotatingFileHandler` for the two Python logs (matches the in-repo `telegram_bot.py:249` precedent, self-contained, no operator install; fixes the false docstring) + a small repo-side logrotate conf (`ops/logrotate/chad-logs.conf`, `copytruncate` + `su ubuntu ubuntu` style matching `/etc/logrotate.d/chad-backend`) with install doc covering the cron log + `calls_*.ndjson` age rule. `runtime/disk_guard_audit.ndjson` (1.6 MB, root-owned, root-written) noted but left to the disk-guard owner — not ours to rotate.
- **6c — dynamic_caps: NARROWED to the caller-less slice**. Archivable (zero importers verified across chad/, backend/, ops/, scripts/, all ExecStart lines): `chad/risk/dominance_caps_bridge.py`, `dominance_risk_layer.py`, `dominance_strategy.py`, `risk_governed_strategy.py` — via **git-committed** move to `_archive/bak_purge_<date>/chad/risk/` (established convention; MUST be git-committed because `/usr/local/bin/chad_disk_guard.sh` deletes `_archive` files >30 days old — `_archive/` is purgatory, not storage). Pre-move: sweep `chad/tests/` for imports of the archived modules. **Explicitly NOT archived**: `chad/risk/dynamic_caps.py` (imported by `kraken_executor.py:37` — live), `quarantine_strategy.py`/`quarantine_layer.py` (caller-less BUT sole producer of `dynamic_caps_quarantine.json`, which `correlation_layer.py:11,64-66` **reads every cycle and raises if absent** — archiving makes the frozen March weights permanently unregenerable; flagged as a refactor candidate, not hygiene). Dead runtime files (`dynamic_caps_dominance_overlay.json`, `dynamic_caps_risk_governed.json`, `dominance_allocator.json`, the two static cleanup-audit NDJSONs, May's dedupe-cleanup audits) are **runtime mutation → not this lane**; listed in a short operator disposition note instead.

---

## 7. Phase-2 build order (one commit per line, each followed by the full verification sequence)

| Commit | Item | Content | Activation class |
|---|---|---|---|
| W3B-1 | 4b | Sentinel TTL rows (drawdown_state, ibkr_watchdog_last) (+ feed_watchdog mirror per D5) | ONESHOT-LIVE-AT-MERGE |
| W3B-2 | 4c | Drawdown gauges staleness guard (A4 pattern) | RESTART-GATED (metrics) |
| W3B-3 | 4a | Repo-track drawdown/watchdog units + install/parity doc | Inert (operator-applied) |
| W3B-4 | 2 | Drift-content transition alerting (publisher write-site + coach kinds + tests) | ONESHOT-LIVE-AT-MERGE |
| W3B-5 | 1 | EXS9 stale-process check (provider + config section + test-pin moves) | ONESHOT-LIVE-AT-MERGE (warn-only, never pages) |
| W3B-6 | 3-T1 | Mark stamps both lanes (`exit_overlay.v2` additive) | RESTART-GATED (live-loop) |
| W3B-7 | 3-T2 | Portfolio-marks-first loader, `CHAD_OVERLAY_PORTFOLIO_MARKS` default OFF (only if D4 approves) | RESTART-GATED + flag |
| W3B-8 | 5a | Delete dead `REAL_STRATEGIES` | Inert |
| W3B-9 | 5b | Registry perimeter guards + pin-tests (enforcement per D6) | RESTART-GATED |
| W3B-10 | 6a | Dedupe dir move + repo-side cleanup timer + install doc | Mixed (split-brain window documented) |
| W3B-11 | 6b | RotatingFileHandler ×2 + repo logrotate conf + install doc (per D7) | RESTART-GATED |
| W3B-12 | 6c | Archive 4 caller-less dominance/risk-governed modules (git-committed) + operator disposition note | Inert |

Ordering rationale: pure-observability first (smallest blast radius), alerting second, the two behavior-adjacent items (EXS9 test-pin move, mark stamps) mid, hygiene last. Each commit stands alone and is individually revertable.

Verification per commit: `py_compile` on changed files → `pytest chad/tests/ -x -q` (baseline first; W2A-era known-fail set must be re-baselined in the worktree before W3B-1) → `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview` tail. Plus per-item targeted tests listed above.

## 8. Risks

- **Merge = deploy for oneshot timers** (sentinel, reconciliation publisher, live-readiness): W3B-1/-4/-5 go live at the next 5-min fire after merge. Mitigations: EXS9 is warn-capped (structurally cannot page); drift alerting baselines on the current empty set (no storm at drift_count=0) and is fail-soft; TTL rows use artifact-verified TTLs with warn-first thresholds. Merge still requires explicit GO.
- **Schema readers** (Item 3): `trade_log_adapter` tolerance verified read-only before W3B-6; if v2 bump is unsafe, fields ship additive-under-v1 (documented in the commit).
- **Hard perimeter invariants** (Item 5) could brick engine build on config drift — default recommendation is warn-at-runtime / hard-in-tests (D6).
- **Dedupe split-brain** (6a): bounded to one duplicate alert per key per 900s until restarts complete.
- **Test-pin churn**: EXS9 moves two hard pins; the deploy PA forbids weakening the read-only lock tests — design requires none.
- **Lane A adjacency**: Item 3 touches evidence *producers* only; `chad/validation/` stays untouched. If Lane A lands adapter changes mid-wave, re-verify tolerance before W3B-6.

## 9. Decision points (Phase-1 output — answers gate Phase 2)

- **D1 (EXS9 comparator)**: per-service curated `code_paths[]` + `git log -1 --format=%ct -- <paths>` (RECOMMENDED — BOX-034A-correct, small curation burden, false-negatives acceptable at warn tier) vs whole-HEAD comparison (zero curation, over-alerts on every commit) vs file-mtime (fragile under checkout/touch).
- **D2 (EXS9 severity)**: warn-only forever (RECOMMENDED — deploy-pending is governed state; report-surface, never pages) vs warn→fail escalation after N days (pages eventually; requires ratifying N).
- **D3 (drift alert severity + resolved-transitions)**: `warning` + alert on both appeared and resolved (RECOMMENDED) vs `critical` (competes with reconciliation_red) vs appeared-only.
- **D4 (Item 3 scope)**: Tier 1 stamps only (minimum honest fix) vs Tier 1 + Tier 2 flag-gated portfolio-marks loader (RECOMMENDED — the brief asked for "freshest broker-truth mark", and the flag+default-OFF keeps it observer-class) vs also absorbing PA_SIM_MARK items (a)/(b) (fill-side bound + back-fill — NOT recommended here; separate PA, different chokepoint owner).
- **D5 (unit repo-parity)**: byte-identical copies of installed units + install doc listing `OnFailure=chad-service-alert@%N` as a *proposed delta* for a future PA (RECOMMENDED — repo mirrors reality; improvements go through PA) vs repo copies that already include OnFailure (repo≠installed until operator syncs — drift by construction). Also: mirror TTL rows into `feed_watchdog.WATCHED_FEEDS` too? (RECOMMENDED yes — one-line each, second net.)
- **D6 (registry perimeter enforcement)**: hybrid — runtime checks emit UserWarning (invariant-6 style), hard assertions live in pin-tests only (RECOMMENDED — config drift can't brick the engine; CI still fails loud) vs full hard runtime invariants (maximal protection, brick risk) vs tests-only (no runtime signal).
- **D7 (log rotation mechanism)**: RotatingFileHandler code fix for claude/gpt logs + small logrotate conf for cron log (RECOMMENDED — self-contained, matches telegram_bot precedent, no operator dependency for the main offenders) vs logrotate-only for everything (one mechanism, but operator-install-dependent and adds a second writer-vs-rotator interaction class).
- **D8 (merge/activation GO)**: acknowledge that merging `goal/wave3-smallfixes` into the live tree's `main` activates W3B-1/-4/-5 at the next timer fires with no restart gate. Phase-3 merge therefore needs its own explicit GO, separate from the Phase-2 build GO.

— END PLAN_W3B (Phase 1) —
