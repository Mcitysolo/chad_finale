# CHAD Overhaul 2026-04-19/21 — Post-Overhaul State

## What This Document Is

Written at overhaul completion. Captures system state, completed work, known issues, and the Phase-8/9/10 roadmap. Supersedes SSOT v8.0 as the current architecture reference.

## System State at Overhaul Close

- **Trading mode:** PAPER (`CHAD_EXECUTION_MODE=paper`)
- **SCR:** WARMUP, `sizing_factor=0.1`, `sharpe_like=+0.8065`, `effective_trades=37` (as of 2026-04-21T22:35Z)
- **Reconciliation:** GREEN, `worst_diff=0`, `mismatches=[]`, `drifts=[GLD, SPY]`
- **Paper equity:** ~$994k, broker account DUK902770
- **IBKR gateway:** paper port 4002 listening; `chad-ibgateway.service` active
- **Dashboard:** https://chadtrades.com — HTTP 200, TLS valid `notBefore=2026-04-20 13:04 GMT`, `notAfter=2026-07-19 13:04 GMT` (Let's Encrypt, auto-renewal via `certbot.timer`)
- **Kernel:** `6.17.0-1009-aws` (current; no kernel update pending as of 2026-04-21)

### Active services

| Service | Active | Enabled |
|---|---|---|
| chad-live-loop.service | active | enabled |
| chad-trade-closer.timer | active | enabled |
| chad-scr-sync.timer | active | enabled |
| chad-reconciliation-publisher.timer | active | enabled |
| chad-ibgateway.service | active | enabled |
| chad-ibkr-bar-provider.service | active | enabled |
| chad-dashboard.service | active | enabled |
| chad-backend.service | active | enabled |
| chad-orchestrator.service | active | enabled |
| chad-kraken-ws.service | active | enabled |
| chad-metrics.service | active | enabled |
| chad-shadow-status.service | active | enabled |
| chad-telegram-bot.service | active | enabled |
| chad-xvfb.service / chad-x11vnc.service | active | enabled |
| nginx.service | active | enabled |
| certbot.timer | active | enabled |

### Contained / disabled (intentional)

- `chad-reconciliation.timer` — kept disabled; reconciliation-publisher.timer is the authoritative writer. Restoring it would create a dual-writer risk.
- `chad-reconciliation-publisher.service` — inactive at steady state; triggered by its timer.
- `chad-options-chain-refresh.service` — inactive; timer still firing but wrapped ahead of Phase-8 timeout fix.

### Retired (masked)

- `chad-polygon-stocks.service`
- `chad-bars-validate.service`
- `chad-daily-bars-backfill.service`

### Runtime SHA snapshot (overhaul close)

- `trade_closer_state.json`: `550591df3ad66f20c64e73bddaeec441e99bfdb95e14311d02f84ed5438d7c60`
- `position_guard.json`: `83c8936da64938cb3c5bd0b4208bc5aa5cb0b6004945a572feac7050882c377a`
- `scr_state.json`: `2da36df81298820e0b8d8e0464e089d3c71fa5001642ab0df326a36d2c57b246`
- `shadow_state.json`: `721976154c584620053361a53176feb8c6bed1b0e5731a114d55a650fff53eef`

## Overhaul Completed Work

- **Phase 1 — Revert point.** Git tag `REVERT_PRE_OVERHAUL_20260419` at commit `45f3728`. Tarball `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz` (SHA `2be81cbae94fb86c266f1c519f208f058ac3a99b99032f2fec7a2eefbcc10a53`). Systemd snapshot captured. Recovery steps in `/home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt`.
- **Phase 2 — Reconciliation unblock (Steps 1–4).** Live-loop quiesced, trade-closer writer identified and rebuilt, position_guard cleared + directly repopulated, reconciliation containment applied, reconciliation verified GREEN.
- **Phase 3 — Data cleanup (Steps 5, 5.1).** 396 trades retagged. Bad-data quarantine under `_archive/bak_quarantine_20260402/`.
- **Phase 4 — SCR verification (Steps 6–7).** WARMUP confirmed; SCR computation traces validated against shadow data.
- **Phase 5 — Hardening (Steps 9–10).** Dashboard v2 with polished fintech UI; fail-closed auth; TLS via Let's Encrypt; nginx reverse proxy hardened.
- **Phase 6 — Hygiene (Steps 11–12).** Three obsolete services retired (polygon-stocks, bars-validate, daily-bars-backfill). Git state tidied. `chad-options-chain-refresh` hung-process cleanup.
- **Phase 7 — Restart (Steps 13–15).** Five retries against ISSUE-56 / ISSUE-74 / ISSUE-29; six code fixes landed; 24 GREEN cycles observed over 25 minutes in Step 14; Step 15 formally certified 5+ consecutive GREEN cycles.
- **Phase 8 — Reboot (Step 16).** Skipped — no pending kernel update. `reboot-required` flag absent.
- **Phase 9 — Smoke test + close (Step 17).** This document.

## Code Fixes Committed During Overhaul

| SHA | Subject |
|---|---|
| `b672042` | FIX: ISSUE-29 — reconciler respects partial_attribution_residual |
| `0e3e007` | ISSUE-56 fix v2: reduce-not-close for partial broker_sync attribution |
| `e5db43a` | ISSUE-74 fix: synthesize fill_id for broker_sync lots |
| `709be69` | ISSUE-56 fix extension: cover rebuild path at live_loop.py:284 |
| `1b850c6` | FIX: ISSUE-56 — broker_sync anchor yields to strategy ownership |
| `92b48df` | HARDEN: Dashboard TLS + fail-closed auth + retire 3 obsolete services |
| `8c82580` | FIX: Crypto signals — use live Kraken price for entry gates |
| `2dadcd7` | BUILD: Daily loss limit + per-symbol performance blocker |
| `0238292` | FIX: Alpha daily signal limit 3 per symbol per day |
| `c38675f` | BUILD: Dashboard v2 — polished fintech UI + auth |
| `bdac7a6` | FIX: SCR excludes zero-PnL trades from win rate calculation |
| `4653d51` | FIX: reqExecutionsAsync monkey-patch must be async coroutine |
| `43779e6` | FIX: Skip reqExecutionsAsync on IB connect to prevent cold-start hang |
| `4254026` | FIX: SCR thresholds calibrated for early paper trading |

## SSOT v8.1 Issue Backlog

**ISSUE-22 [P2] [OPEN]:** Legacy placeholder audit item — revisit during Phase-8 intent schema work.

**ISSUE-29 [P1] [PARTIAL]:** `apply_close_intents` mutates guard before broker confirms. Fix applied at the reconciler (b672042) so partial_attribution_residual entries are not wiped. Root cause in `apply_close_intents` still untreated — produces spurious close intents each cycle, no state corruption because of per-cycle rebuild. Proper fix is mutate-after-confirm.

**ISSUE-50 [P1] [OPEN]:** `chad-options-chain-refresh` hangs when IBKR ushmds farm is down. Timeout wrapper pending — Phase-8 item.

**ISSUE-54 [P2] [OPEN]:** `runtime/pnl_state.json` still tracked in git. Needs `git rm --cached`; belongs in gitignore.

**ISSUE-56 [P0] [CLOSED]:** `broker_sync` anchor stayed open=true while strategy already closed the position; caused reconciliation RED. Fixed by v2 reduce-not-close path (1b850c6 → 709be69 → 0e3e007) and reconciler skip for `partial_attribution_residual` entries (b672042).

**ISSUE-58 [P2] [OPEN]:** `chad-trade-closer.timer` uses `OnBootSec=45` + `OnUnitActiveSec=60`; when manually restarted mid-session, first fire can be absent until a manual service kick seeds the active anchor. Refactor to `OnCalendar` or document the seed step.

**ISSUE-74 [P0] [CLOSED]:** `broker_sync` lots lacked `fill_id`, raising `KeyError` in trade-closer. Synthesized fill_id at lot creation time (e5db43a).

**ISSUE-75 [P1] [OPEN]:** Multiple call sites write `position_guard.json` directly. Unify through a single setter to prevent drift and simplify testing.

**ISSUE-78 [P2] [OPEN]:** Two code paths read `CHAD_EXECUTION_MODE`; results can diverge under rapid mode transitions. Unify into a single accessor.

## Phase-8 Sprint Items (immediate post-overhaul)

- Halt-on-reconciliation-mismatch (not just log RED)
- Structured intent object schema (replace dict-based intents)
- Strategy router module with regime × strategy activation matrix
- Data freshness gate at routing layer
- Graduated circuit breaker (reduce → suppress → flatten → stop)
- Per-trade slippage tracking vs model
- STOP bus trigger expansion
- Stale intent expiry
- ISSUE-29 proper fix: `apply_close_intents` mutate-before-confirm → mutate-after-confirm
- ISSUE-50: options-chain-refresh timeout wrapper
- ISSUE-75: unify guard writes through single setter
- ISSUE-74 schema: formalize `broker_sync` lot shape with required `fill_id`
- ISSUE-54: `git rm --cached runtime/pnl_state.json` + gitignore
- ISSUE-58: trade-closer timer boot trigger pattern (OnCalendar or documented seed)
- ISSUE-78: unify the two `CHAD_EXECUTION_MODE` code paths

## Phase-9 Items (pre-live-capital checklist)

- Regime classifier as hard prerequisite for strategy activation
- Regime × strategy activation matrix (codified, not ad-hoc)
- Net EV gate on every signal
- Signal stacking minimum vote threshold (2–3 families)
- Multi-timeframe confirmation gate
- Volatility-adjusted position sizing + fractional Kelly
- Correlation cluster exposure cap
- Event-risk suppression calendar
- Backtest/paper/live identical interfaces
- OMS/EMS separation
- Adaptive entry thresholds by regime

## Phase-10 Items (live-capital feedback loop)

- Per-setup rolling expectancy scorecard
- Signal decay measurement at T+1 / T+5 / T+15 / T+30
- Strategy health score (composite)
- Edge decay detection with auto-allocation reduction
- Live feature distribution monitoring (drift alerts)
- Implementation shortfall decomposition (modeled vs realized)

## Known Operational Notes

- **`chad-options-chain-refresh`:** hung-process cleared 2026-04-20. Timer will fire next at 2026-04-21 12:30 UTC. If the IBKR ushmds farm is still down the service will hang again; the timeout wrapper (Phase-8 / ISSUE-50) is the durable fix.
- **GLD and SPY in reconciliation drifts:** broker-side positions not fully attributed to strategies. Classified as `BROKER_DRIFT` rather than mismatches by the b672042 classifier. Not errors — expected during WARMUP while attribution backfills.
- **ISSUE-29:** active defect. Does not corrupt state thanks to per-cycle guard rebuild, but produces spurious close intents per cycle. Phase-8 target.
- **`chad-reconciliation.timer`:** intentionally disabled to avoid dual-writer conflict with `chad-reconciliation-publisher.timer`. Restore only if explicitly needed and only after retiring the publisher.
- **Trade-closer timer:** after a fresh manual restart, the `OnUnitActiveSec` anchor is missing; the first fire must be seeded via `sudo systemctl start chad-trade-closer.service`. See ISSUE-58.

## Revert Point

- **Git tag:** `REVERT_PRE_OVERHAUL_20260419`
- **Commit:** `45f3728fa512c27d77453489c11d95ca0e075cb9`
- **Tarball:** `/home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz`
- **Tarball SHA:** `2be81cbae94fb86c266f1c519f208f058ac3a99b99032f2fec7a2eefbcc10a53`
- **Instructions:** `/home/ubuntu/chad_revert_points/HOW_TO_REVERT.txt`

## SCR Progression Path

- **Current:** WARMUP — `sizing_factor=0.1`, `effective_trades=37`, `sharpe_like=+0.8065`
- **To CAUTIOUS:** ~50 effective trades with `sharpe > threshold` sustained
- **To CONFIDENT:** ~200 effective trades with sustained positive performance
- **ETA at current paper cadence:** CAUTIOUS expected in 2–4 weeks
