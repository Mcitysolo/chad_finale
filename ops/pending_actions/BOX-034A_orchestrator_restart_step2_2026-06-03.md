# BOX-034A Inc 3 — Pending Action: Restart chad-orchestrator.service to activate Step 2 warn-mode (Consumers 3/4) live

**Status:** PENDING — awaiting explicit operator GO (governance rule #7, live service restart)
**Date:** 2026-06-03
**Related:** BOX-034A; Inc 3 Step 2 (commit d0f7a78); prior restart PA (step1a, commit fc51872)

## 1. Purpose
Step 2 (d0f7a78) added warn-mode CURRENCY_WARN_* assertions for 5 equity consumers. Consumers 1/2/5 (profit_lock) auto-activate via the profit-lock oneshot (reloads code each fire). Consumers 3/4 (orchestrator: _load_portfolio_snapshot leg checks + post-build_payload risk-cap guard) run in the chad-orchestrator daemon, which loads code at boot — last booted 2026-06-03 01:27:37 UTC at fcc9986, predating d0f7a78. This PA restarts it to activate Consumers 3/4 live.

## 2. Ships-on-restart (verified)
Daemon boot: 2026-06-03 01:27:37 UTC (fcc9986). orchestrator.py diff fcc9986..d0f7a78 = ONLY the Step 2 Consumer 3/4 warn blocks. No other orchestrator change. (1a tag-writing already live from the prior boot.)

## 3. Risk — SAFE (warn-only)
- Every Step 2 check is logger.warning(); no raise, no return, no equity-value change, no control-flow change (verified at commit: grep raise/return over added lines = 0; 10 tests; suite green).
- No sizing/cap/order change — warn blocks emit log lines only on the bad condition; silent on clean state.
- Lower risk than the Step 1a restart (which added tag-writing); this adds logging only.
- Brief refresh gap during restart (<=10s on 60s cadence) — acceptable.

## 4. Pre-restart gate
- [x] Step 2 committed (d0f7a78), on HEAD.
- [x] Warn-only verified (grep raise=0; 10 tests; suite green).
- [x] orchestrator.py BOOT..HEAD = only Step 2 warn blocks (§2).
- [ ] Live logs currently show NO CURRENCY_WARN_* (already-live profit_lock-side warns silent = clean currency state) — VERIFY at Step A.

## 5. Procedure (Channel 1; requires explicit GO)
Step A - pre-state (confirm warn-mode currently silent + 1a still tagged):
    sudo journalctl -u chad-orchestrator.service -u chad-profit-lock.service --since "2026-06-03 01:27:37 UTC" --no-pager | grep -c "CURRENCY_WARN_"
    jq '{total_equity_currency, total_equity_currency_ok}' runtime/dynamic_caps.json
Step B - restart + health:
    sudo systemctl restart chad-orchestrator.service
    sleep 6
    systemctl is-active chad-orchestrator.service
    systemctl show chad-orchestrator.service -p MainPID -p ActiveEnterTimestamp --no-pager
    sudo journalctl -u chad-orchestrator.service -n 25 --no-pager
Step C - watch >=2 cycles, confirm NO CURRENCY_WARN_*:
    sleep 130
    sudo journalctl -u chad-orchestrator.service --since "<restart time>" --no-pager | grep "CURRENCY_WARN_" | tail -20
    jq '{total_equity, total_equity_currency, total_equity_currency_ok}' runtime/dynamic_caps.json

## 6. Post-restart verification (success criteria)
- Orchestrator healthy (new PID, clean init, no traceback).
- NO CURRENCY_WARN_* lines over >=2 cycles (tripwires armed + silent = clean currency state under live warn checks).
- dynamic_caps still total_equity_currency=CAD/ok=true (1a regression check); total_equity numerically continuous.

## 7. Rollback
Warn-only; nil rollback need. A firing CURRENCY_WARN_* is diagnostic (a currency-state signal), not a code failure — investigate the currency state, not the code; warns never block.

## 8. Notes
- Activates Step 2 readers in WARN mode only. The enforce flip (warn -> fail-closed) is a SEPARATE later PA, gated on a clean warn-silence window.
- Inc 4 (reconciliation-test rewrite) remains for BOX-034A closeout.
