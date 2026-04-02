# CHAD Engineering Instructions for Claude Code

## System Identity
- Name: CHAD (Compounding Hedge-Fund Algorithmic Desk)
- Canonical root: /home/ubuntu/chad_finale
- Python binary: python3 (never python)
- Virtual environment: /home/ubuntu/chad_finale/venv
- Current trading posture: DRY_RUN / PAUSED — no live broker execution
- Account equity: ~$994k paper

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

## Priority Work Order

### P0 — Execute in strict order, one at a time

P0-1: Remove reqContractDetails from CHAD hot paths
- Target files (never touch venv):
  chad/execution/ibkr_adapter.py line 646
  chad/core/paper_shadow_runner.py line 695
  chad/core/paper_position_closer.py line 186
- Replacement: use explicit contract metadata from intent.meta
  using contract_month and multiplier when present
  use cached resolution otherwise
  never perform live network lookup in the hot path
- Success check: grep -rn "reqContractDetails" chad/ returns zero results

P0-2: Enforce 10-second execution liveness contract
- Target: chad/execution/ibkr_adapter.py and ibkr_executor.py
- Add hard 10s timeout to every broker submission call
- On timeout: log TIMEOUT classification, continue loop safely
- Failure classes: TIMEOUT, BLOCKED, REJECTED, FAILED, UNKNOWN
- Success check: every broker submit wrapped in asyncio.wait_for timeout=10.0

P0-3: Enforce single IB() session discipline
- Hot path target: chad/execution/ibkr_adapter.py line 509
- Refactor to accept injected IB session, not create new IB()
- Non-hot-path files such as ledger watcher and health check
  may retain own sessions but must be documented as non-execution
- Success check: grep -rn "IB()" chad/execution/ returns zero results

P0-4: Fix paper_exec attribution flattening
- Target: chad/execution/paper_exec_evidence_writer.py
- primary_strategy must never fall back to paper_exec
  when a real strategy name exists upstream in the intent or plan
- Raise attribution error if no real strategy can be resolved
  rather than silently writing paper_exec
- Success check: paper_exec appears only as schema version constant
  never as a fallback for primary_strategy

### P1 — After all P0 items verified
P1-1: Complete Profit Lock LiveGate wiring
P1-2: Add uniform EXECUTION RESULT logs to all execution lanes
P1-3: Complete loop-level broker truth rebuild

### P2 — After all P1 items verified
P2-1: Quarantine all .bak files from canonical paths
P2-2: Fix price-cache timer cadence overlap
P2-3: Repair pytest collection surface

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
- PRE_HARDENING_20260402 — all work captured before P0 fixes

## Rollback Command
git checkout PRE_HARDENING_20260402
