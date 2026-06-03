# BOX-034A Inc 3 — Pending Action: Restart chad-orchestrator.service to activate Step 1a currency tagging live

**Status:** COMPLETED 2026-06-03 01:27:37 UTC — see section 10 Completion Record
**Date:** 2026-06-02
**Author:** TEAM CHAD
**Related:** BOX-034A (canonical equity currency unification); Inc 3 Step 1a (commit b6d333f); Step 1b (commit 40a9c55)

## 1. Purpose
Step 1a (orchestrator.py:919-922, commit b6d333f) injects `total_equity_currency`/`total_equity_currency_ok` into `dynamic_caps.json`. It is committed and unit-verified (full suite 2782 green). It is NOT live: the writer daemon `chad-orchestrator.service` has run un-restarted since 2026-05-18 12:02:42 UTC (~15 days) and executes pre-1a source, so the tags are absent in the live file. The earlier `chad-live-loop` restart did not help — `chad-live-loop` consumes `dynamic_caps` (Redis) and does not write it. This PA restarts the correct daemon to ship 1a live.

## 2. Forensic basis (read-only, complete)
- Live writer: `chad-orchestrator.service` (`python -m chad.core.orchestrator`, `run_forever` 60s) -> `refresh_dynamic_caps` -> `_atomic_write_json(dynamic_caps.json)` at orchestrator.py:922. Sole non-test caller; 60s cadence (file mtime matches `cycle_start`). Allocator CLI is the only other file-writer and is wired to no unit (manual). `state_bus.publish_dynamic_caps` is Redis-only (no file I/O). No third writer.
- 1a injection (lines 919-921) sits between `build_payload()` and the atomic write — correctly on the live path.
- Daemon boot: ExecMainStartTimestamp 2026-05-18 12:02:42 UTC. BOOT commit 6abf16b (last HEAD before boot); HEAD d80f44b.

## 3. Ships-on-restart blast radius (critical)
A Python service reloads ONLY its own import graph on restart. The orchestrator caps-compute import graph (`dynamic_risk_allocator`, `savage_allocator`, `allocator_v3`, `correlation_strategy`/`correlation_layer`, `state_bus`, `execution_config`) is BYTE-IDENTICAL since BOOT. The only changed module in that graph is `orchestrator.py`, whose entire BOOT..HEAD diff (+55/-4, one logical change) is BOX-034A Step 1a — additive currency tagging, nothing else.
- The ~45 other changed non-test modules (`live_loop`, `position_reconciler`, `paper_*`, `market_data/*`, `ops/*` publishers, `profit_lock`, `stop_bus_*`, strategies) are NOT imported by this daemon. They ship into their own services on their own restarts, independently. This restart does NOT activate them.
- Net behavioral delta of this restart: two additive keys appended to `dynamic_caps.json` after `build_payload()`.

## 4. Risk assessment — SAFE
- No change to `dynamic_caps` numeric values (`total_equity`, `daily_risk_fraction`, `portfolio_risk_cap`, `strategy_caps`) — produced by a byte-identical import graph; tags inserted after `build_payload`.
- No risk-cap sizing change — sizing inputs untouched; `_ok` flag is non-enforcing (warn-before-enforce; reader assertions are a later step).
- No order-adjacent code — orchestrator submits/adjusts nothing (`submitted_orders=[]` hard-coded); execution is `chad-live-loop`'s domain.
- No new side-effects — heartbeat / live-gate fetch / fast-loop all predate BOOT.
- Brief refresh gap: during the ~few-second restart `dynamic_caps.json` does not refresh; consumers use the last value. Acceptable for a <=10s window on a 60s-cadence artifact.

## 5. Pre-restart gate (all must hold)
- [x] 1a committed (b6d333f), on HEAD.
- [x] Full suite green (2782).
- [x] Diff review complete; verdict SAFE (sec 3-4).
- [ ] `portfolio_snapshot.json` currently shows `ibkr_equity_currency_ok=true` AND `kraken_equity_currency_ok=true` (so derive yields ok=true) — VERIFY immediately before restart (procedure Step A).
- [ ] Daemon expected to take a clean SIGTERM (confirm in shutdown log, as `chad-live-loop` did earlier this session).

## 6. Procedure (Channel 1 — operator terminal; requires explicit GO)
Step A - capture pre-state:
    jq '{total_equity, total_equity_currency, total_equity_currency_ok}' runtime/dynamic_caps.json
    jq '{ibkr_equity_currency_ok, kraken_equity_currency_ok}' runtime/portfolio_snapshot.json
Step B - restart + health:
    sudo systemctl restart chad-orchestrator.service
    sleep 6
    systemctl is-active chad-orchestrator.service
    systemctl show chad-orchestrator.service -p MainPID -p ActiveEnterTimestamp --no-pager
    sudo journalctl -u chad-orchestrator.service -n 30 --no-pager
Step C - wait one full 60s cycle (confirm dynamic_caps.json mtime advanced past restart), then verify per sec 7.

## 7. Post-restart verification (success criteria)
- `dynamic_caps.json`: `total_equity_currency="CAD"`; `total_equity_currency_ok=TRUE` (given both snapshot legs true per Step A). If FALSE despite true legs -> STOP and investigate `_derive_total_equity_currency_ok`. `total_equity` numeric unchanged from Step A pre-state.
- After the next `chad-profit-lock` oneshot fire (<=60s later): `pnl_state.json` `account_equity_currency_ok` flips to TRUE automatically (1b reads now-tagged dynamic_caps); `account_equity` numeric unchanged.
- Orchestrator journal: no traceback/error since restart; 60s `cycle_start` resumes.

## 8. Rollback
Restart ships only additive keys via committed code; expected rollback need is nil. `_ok` is non-enforcing (no sizing/order impact), so no urgent rollback pressure. If behavior is unexpected, investigate live; if necessary, revert orchestrator.py to BOOT state and restart.

## 9. Notes / expectations
- `total_equity_currency` defaults to "CAD" via code default (unit sets no `CHAD_BASE_CURRENCY`); CAD is the intended base. Optional future hardening: set `CHAD_BASE_CURRENCY=CAD` explicitly in the Environment of all currency-aware units (collector/publisher/orchestrator) so the tag never relies on a code default — a separate unit-change PA, not required here.
- This PA activates Step 1a (writers) only. Step 2 reader assertions (warn-before-enforce) and Inc 4 (reconciliation test rewrite) remain separate.
- Finding: the ~15-day-stale orchestrator means orchestrator-side changes are not live until restarted. A periodic-restart or restart-on-change policy for `chad-orchestrator.service` is a candidate ops item (cf. BOX-034C watchdog family).

## 10. Completion Record
- Restart executed 2026-06-03 01:27:37 UTC; new MainPID 2970790 (was 2874768); clean systemd deactivation of old daemon, clean init of new, zero tracebacks.
- SAFE verdict confirmed empirically: total_equity numerically continuous across the boundary (313656.93 old daemon 01:26:53 to new daemon 01:27:38); sizing math unchanged.
- VERIFIED 2026-06-03 02:00Z: dynamic_caps total_equity_currency=CAD, total_equity_currency_ok=true; pnl_state account_equity_currency=CAD, account_equity_currency_ok=true (auto-flipped via profit-lock oneshot); account_equity == dynamic_caps.total_equity (single-sourced); error scan empty.
- Step 1a (b6d333f) + Step 1b (40a9c55) now LIVE. Remaining for BOX-034A: Step 2 reader assertions (warn-before-enforce), Inc 4 reconciliation-test rewrite.
