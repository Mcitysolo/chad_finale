# Pending Action — chad-live-loop restart to activate Bug B Fix A
Date: 2026-06-03  •  Author: TEAM CHAD (issued) / SOLO (executes)  •  Status: PENDING operator GO (rule #7)

## 1. Objective
Restart chad-live-loop.service to load HEAD e4986ad, activating Bug B Fix A — the cumulative broker-truth per-symbol futures position cap. This restart has no other purpose.

## 2. Scope
- Restarts: chad-live-loop.service (running PID 2788851, booted 2026-06-02 20:15:42 UTC -> new PID).
- Deploys: exactly ONE runtime behavior change — Fix A. Proven by pre-restart audit: live_loop import-graph closure (empirical sys.modules + no lazy imports) excludes all other in-window changes; `git log d80f44b..e4986ad -- chad/core/live_loop.py` = one commit (e4986ad); range-diff byte-identical to Fix A (+212/0). orchestrator/trade_closer/harvester/profit_lock changes are already live in their own services.

## 3. Pre-conditions (verified 2026-06-03)
- HEAD e4986ad; suite 2799 green; live_loop.py parses clean.
- Deploy = Fix A only (SAFE).
- Env gate armed and SURVIVES restart (flags in 91-disable-futures-exec.conf drop-in; new process inherits): CHAD_DISABLE_FUTURES_EXECUTION=1, CHAD_DISABLE_FUTURES=1, CHAD_FUTURES_EXECUTION_ENABLED=0.
- Working tree clean (only known _archive/PR-05 state); unit file + 9 drop-ins unchanged on disk.

## 4. Risk profile — LOW / warn-side
Fix A can only REFUSE a FUT open or fail-closed; it never places an order. Exits/flips pass untouched. The env gate remains armed as a redundant backup through and after the restart, so futures stay blocked regardless of Fix A's behavior. Worst realistic case: a startup traceback — caught immediately in logs, reverted per §6.

## 5. Procedure (Channel 1, operator — on explicit GO)
sudo systemctl restart chad-live-loop
systemctl status chad-live-loop --no-pager
(then tail the live-loop log for the verification markers in §6)

## 6. Acceptance / verification (within ~2 live-loop cycles)
- FUTURES_POSITION_CAP_BLOCK symbol=M6E strategy=omega_macro side=BUY qty=2.0 net=217.0 projected=219.0 cap=3 appears AHEAD of FUTURES_EXECUTION_DISABLED_SKIP (cap now fires first).
- FUTURES_POSITION_CAP_UNVERIFIED absent (positions_truth GREEN).
- M2K / MCL behavior unchanged; zero tracebacks.
- The three env-gate flags still present in the new PID's environment.

## 7. Rollback
If tracebacks or anomalous behavior: revert with `git revert e4986ad` (or restore prior live_loop.py) and restart. The env gate independently blocks all futures opens in the meantime — no runaway exposure during rollback.

## 8. Status log
- 2026-06-03: authored, PENDING operator GO.
