# CHAD Defense-Board Reconciliation — 2026-06-17

Status: COMPLETE. The P0–P3 defense checklist was reconciled against live system reality. Verdict: materially stale across all tiers — nearly every item was already handled, already fine, or misdiagnosed. No item is an active live-trading exposure. ready_for_live remains FALSE (gated on out-of-sample edge validation, which is NOT on this checklist).

## Code landed this session (origin/main)
- 94eefae — preserve unknown keys across all portfolio_snapshot.json writers (fixed authoritative-USD clobber by the multi-writer relay)
- 32e8cd6 — tier_manager passes real US-equity market_open (deferral was dead; demotions now defer to close); market-hours helper de-duped into chad/utils/market_hours.py
- ff58803 — single canonical strategy registry (18 declared / 16 active / 2 dormant) + startup drift assertion
- 1ff9a62 — registry tripwire validates per-tier enabled_strategies lists
- (Channel-1) chad-backend 99-root-fix.conf host 0.0.0.0 -> 127.0.0.1 (loopback defense-in-depth; CHAD_ROOT preserved)

## P0 — both CLOSED
- P0-1 delta $100 placeholder fills: defense present; 0 such fills in data.
- P0-2 Prometheus vs SCR conflation: split into chad_paper_* (raw) vs chad_scr_* (canonical) with divergence HELP text.

## P1
- P1-1 strategy registry 3-way drift: FIXED this session (ff58803, 1ff9a62). CLOSED.
- P1-2 ib_insync -> ib_async migration: done. CLOSED.
- P1-3 drawdown_state freshness: fresh. CLOSED.
- P1-4 ibkr_watchdog freshness: fresh. CLOSED.
- P1-5 phantom drawdown -36.7%: forward code fixed; residual is contaminated old history (CAD-in-USD + futures-runaway peaks); enforcement_active=false (gauge only, not braking). Self-clears at Day-0. No code needed.
- P1-6 position_guard_drift.json: exists, written by reconciliation_publisher. CLOSED.
- P1-7 live-loop on fixed build: confirmed. CLOSED.
- P1-8 stale sqlite order rows: lifecycle works; a few orphaned rows remain. Hygiene, non-gating.

## P2 — no active live exposure
- P2-1 same_side guard: misread; fine. CLOSED.
- P2-2 operator_intent "ALLOW_LIVE": by-design (allows paper entries; live blocked downstream). Doc clarification only.
- P2-3 HALT_BOOST_SUPPRESSED log noise: cosmetic read-path clamp. Open/benign.
- P2-4 XGB MES anomaly: moot, veto disabled 3 ways. CLOSED.
- P2-5 port binding: all ports loopback; backend competing 0.0.0.0 drop-in hardened this session; dir pristine. CLOSED.
- P2-6 legend freshness: fresh per weekly schedule. CLOSED.
- P2-7 Telegram urllib3 churn: recoverable; needs PTB 20.x upgrade. Open/benign.
- P2-8 apscheduler deprecation: dead dependency (unimported). Drop from requirements. Open/benign.
- P2-9 options-chain-refresh: healthy. CLOSED.
- P2-10 readiness gate: PROVEN fail-closed (ready_for_live=false off drift_count=2). CLOSED.
- P2-11 dual ledger: doc-only ambiguity (config vs canonical). Doc clarification.

## P3
- P3-1 telegram dedupe clutter: stale — 29 files (not 1573); runtime/ 167M healthy. Optional tidy.
- P3-2 "legacy" dynamic_caps backups: MISDIAGNOSED — the 3 Mar-26 files (dominance_overlay/quarantine/risk_governed) are LIVE config referenced in 3–4 modules each. Do NOT archive. CLOSED.
- P3-3 log rotation: already configured (chad-backend/feeds/polygon); 26M logs; disk 62%. CLOSED.
- P3-4 stale docs/memory: addressed by THIS record.
- P3-5 zero-fill strategy classification: pipeline healthy (delta 6, alpha 1 filled today; reconciler/broker_sync are system sources). Remaining strategies quiet; dormant-vs-regime-silent needs a multi-day window -> edge-validation, not a defense fix. Deferred.

## The real frontier (NOT on this checklist)
The defense board has no edge-proof line. Finishing every item = clean plumbing + soak, NOT proven-profitable. The single gate to any live-money conversation is the out-of-sample, cost-adjusted edge-validation harness. Until it passes, ready_for_live stays FALSE.

Open non-gating cleanups (optional): P1-8 stale rows, P2-3 log noise, P2-7 PTB upgrade, P2-8 dead dep, P2-11/P2-2 doc clarifications, P3-1 tidy, market_hours DST 1hr tidy.
