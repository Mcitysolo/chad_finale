# CHAD Engineering Instructions for Claude Code

## System Identity
- Name: CHAD (Compounding Hedge-Fund Algorithmic Desk)
- Canonical root: /home/ubuntu/chad_finale
- Python binary: python3 (never python)
- Virtual environment: /home/ubuntu/chad_finale/venv
- Current trading posture: PAPER — CHAD_EXECUTION_MODE=paper; SCR=WARMUP, sizing_factor=0.1, sharpe_like=+1.057, effective_trades=9, win_rate=0.778, total_pnl=+$129.86 (as of 2026-05-06; SCR values are Paper Epoch 2 in-progress — will grow as trades accumulate)
- Account equity: ~$183,264 USD paper
- Hardening status: P0-P2 complete, GAP-1 through GAP-25 complete; 2026-04-19/21 Overhaul complete

## Governance Rules — Non-Negotiable
1. One change at a time. Baseline, change, verify, proceed.
2. No full rewrites. Surgical changes to existing files only.
3. No direct config mutation. Risk caps, live mode, and strategy config must be prepared as Pending Actions only — never applied directly.
4. After every code change run the verification sequence below.
5. Never modify anything inside runtime_FREEZE_* or data_FREEZE_*
6. Never modify systemd service files without explicit instruction.
7. Never restart live services without explicit instruction.
8. Commit and tag git after each completed P0, P1, or P2 item.

## Verification Sequence After Every Change
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30

## Live Promotion Checklist

### Completed Hardening (for reference)
- P0-1 through P0-4: execution hardening — DONE
- P1-1 through P1-3: observability and broker truth — DONE
- P2-1 through P2-3: hygiene and test surface — DONE
- GAP-1 through GAP-25: governance audit items — DONE

### Pre-Live Operator Tasks
1. OS reboot — no pending kernel update as of 2026-04-21 (kernel 6.17.0-1009-aws current; reboot deferred to next actual kernel update)
2. Disk cleanup — prune backup archives to below 75% usage
3. IB Gateway latency — investigate and resolve dangerous (>750ms) classification
4. Verify all 1360 tests pass after reboot
5. Run full_cycle_preview.py --dry-run clean
6. Confirm live_readiness.json flips to ready_for_live: true
7. Review open paper positions (MES short) before mode switch

### Live Activation Sequence (requires explicit GO)
1. Set trading posture from DRY_RUN to LIVE in config (Pending Action)
2. Validate LiveGate accepts the posture change
3. Monitor first 3 execution cycles with manual oversight
4. Confirm broker truth reconciliation on first fill

## Key Files Reference
- Hot path entry: chad/core/orchestrator.py
- Execution: chad/execution/ibkr_adapter.py
- LiveGate: chad/core/live_gate.py
- Risk and allocation: chad/risk/dynamic_risk_allocator.py
- Profit Lock: chad/risk/profit_lock.py
- Attribution: chad/execution/paper_exec_evidence_writer.py
- Full preview: chad/core/full_cycle_preview.py

## Active Git Tags
- STABILITY_FREEZE_20260307_GREEN — original stable baseline
- PRE_HARDENING_20260402 — snapshot before P0 hardening began
- RATIFICATION_MASTER_20260402 — all hardening and GAP items complete
- REVERT_PRE_OVERHAUL_20260419 — snapshot before 2026-04-19/21 overhaul (commit 45f3728; tarball /home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz)

## Rollback Command
git checkout RATIFICATION_MASTER_20260402
