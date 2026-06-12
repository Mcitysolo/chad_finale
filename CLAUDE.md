# CHAD Engineering Instructions for Claude Code

## System Identity
- Name: CHAD (Compounding Hedge-Fund Algorithmic Desk)
- Canonical root: /home/ubuntu/chad_finale
- Python binary: python3 (never python)
- Virtual environment: /home/ubuntu/chad_finale/venv
- Current trading posture: PAPER — CHAD_EXECUTION_MODE=paper. SCR snapshot (runtime/scr_state.json @ 2026-06-12T23:23:35Z; ttl_seconds=180 — a moving target, re-read before relying on it): state=CONFIDENT, sizing_factor=1.0, paper_only=false, sharpe_like=+3.55, effective_trades=216, win_rate=0.731, total_pnl=+$27,494 (paper_trades=4171; excluded_untrusted=321, excluded_manual=17). CAVEAT: these are in-sample, paper-fill, partially-contaminated stats per SSOT v9.7 Q3 — NOT validated edge evidence. Live activation still gated by the live_readiness publisher (ready_for_live=false @ 2026-06-12T23:19:20Z).
- Account equity: ~$219,865 USD paper total (IBKR paper leg ~$219,607) — latest dated equity_history.v1 value (runtime/equity_history.ndjson, date_utc=2026-06-11). CAVEAT: Bug-B futures contamination makes this figure volatile and untrusted as a P&L signal — it swung $240,331 (06-09) → $268,193 (06-10) → $219,865 (06-11) across three days; treat as a contaminated balance, not realized equity.
- Hardening status: P0-P2 complete, GAP-1 through GAP-25 complete; 2026-04-19/21 Overhaul complete
- ib_async migration: COMPLETE (PHASE1_MIGRATED=23, PHASE2_DEFERRED=0; zero production ib_insync importers). Pinned by chad/tests/test_ib_async_import_parity.py (PHASE2_DEFERRED_FILES==()) and chad/tests/test_pr03_ib_async_phase2_migration.py (verified 2026-06-12).

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
CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview 2>&1 | tail -30

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
4. Verify all 1465 tests pass after reboot
5. Run full_cycle_preview.py clean
6. Confirm live_readiness.json flips to ready_for_live: true
7. Run scripts/lint_systemd_wants_symlinks.sh — require exit 0 (GAP-032 preventive guard; chad-scoped regular-file count must be 0)
8. Review open paper positions (MES short) before mode switch

### Live Activation Sequence (requires explicit GO)
1. Set trading posture from DRY_RUN to LIVE in config (Pending Action)
2. Validate LiveGate accepts the posture change
3. Monitor first 3 execution cycles with manual oversight
4. Confirm broker truth reconciliation on first fill

## Position-Guard Rebuilder Policy (GAP-028 Option B — PERMISSIVE)
- `_rebuild_guard_from_paper_ledger` (chad/core/live_loop.py:383) intentionally does NOT consult
  `reconciliation_state.exclusion_policy`. Strategies may continue to attribute against excluded
  symbols; the rebuilder mirrors `trade_closer_state.queues` faithfully.
- Maintenance surface: the drift detector
  (`chad.core.position_guard.detect_guard_vs_broker_truth_drift`) is wired into
  `chad-reconciliation-publisher` and emits `runtime/position_guard_drift.json`
  (schema_version=position_guard_drift.v1) every cycle. Operators close stale entries with
  `scripts/close_guard_entry.py` (atomic guard-close + trade_closer FIFO clear; honours the
  §4 invariant that closing the guard alone lets the next live-loop cycle rebuild it).
- Fail-closed gates on the CLI: refuses when SCR ∉ {CONFIDENT, CAUTIOUS}, when exec_mode is
  not paper/dry_run, when LiveGate operator intent is ALLOW_LIVE, or on `broker_sync|*` keys.

## Key Files Reference
- Hot path entry: chad/core/orchestrator.py
- Execution: chad/execution/ibkr_adapter.py
- LiveGate: chad/core/live_gate.py
- Risk and allocation: chad/risk/dynamic_risk_allocator.py
- Allocator reality (verified 2026-06-12): allocator_v3 / Kelly ceiling is wired but never instantiated — AllocV3Strategy (chad/core/orchestrator.py:592) has zero construction sites; _allocator_factory() returns CorrelationOverlayStrategy(), so the LIVE overlay is correlation_overlay (chad/risk/correlation_strategy.py). The CHAD_ALLOCATOR_MODE=V3 unit env (chad-orchestrator.service.d/40-allocator-v3.conf) is therefore inert — it is read nowhere in code and the V3/Kelly path never executes. This doc does NOT imply V3 is the live allocator.
- Profit Lock: chad/risk/profit_lock.py
- Attribution: chad/execution/paper_exec_evidence_writer.py
- Full preview: chad/core/full_cycle_preview.py
- Guard drift detector: chad/core/position_guard.py (detect_guard_vs_broker_truth_drift)
- Guard close CLI: scripts/close_guard_entry.py

## Active Git Tags
- STABILITY_FREEZE_20260307_GREEN — original stable baseline
- PRE_HARDENING_20260402 — snapshot before P0 hardening began
- RATIFICATION_MASTER_20260402 — all hardening and GAP items complete
- REVERT_PRE_OVERHAUL_20260419 — snapshot before 2026-04-19/21 overhaul (commit 45f3728; tarball /home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz)
- EVIDENCE_PIPELINE_WAVE1_2026-06-12 — Evidence Pipeline Wave-1 closure record (annotated, on 3185502/PA-EP8)

## Evidence Pipeline (2026-06-12)
Wave 1 landed (repo-side; the changes activate at the Wave-2 gated restart) — tagged EVIDENCE_PIPELINE_WAVE1_2026-06-12 on 3185502:
- PA-EP3 (f9454e1): thread intent idempotency_key into evidence execution_id (slippage join fix).
- PA-EP1 (4ccb0f6): modeled IBKR Fixed commissions at the evidence chokepoint (fee_model=ibkr_fixed_v1, forward-only).
- PA-EP7-T (fc99c06): containment tripwire for placeholder records (fix deferred to PA-EP7v2).
- PA-EP8 (3185502): status canonicalization at the evidence chokepoint (FILLED/filled → paper_fill; status_raw provenance; harvester shares the map; replay files intentionally untouched — observability-only per PR-09, pending PA-EP8v2 netting rebuild).
Day-0 gating remainder (not yet landed): EP6, EP4 (after EP6), EP2a, Bug-B book disposition.
Backlog: PA-EP7v2 — spec filed at docs/PA_EP7v2_spec.md (commit 7991b26); PA-EP8v2 — replay netting rebuild.

## Rollback Command
git checkout RATIFICATION_MASTER_20260402
