### 1. Title
CHAD Soak Mechanical Evidence-Gated Rule for Transient broker_authority_RED Windows

### 2. Status
PROPOSED — pending operator authorization before Session 1 opens at 2026-05-28T00:00:00Z

### 3. Source authority
- reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md
- CLAUDE.md governance rule #1 (one change at a time)
- CLAUDE.md governance rule #3 (operator-domain decisions)
- ops/pending_actions/SESSION_1_tracker_amendment_evidence_gated_2026-05-27.md (superseded by this document; cite explicitly)

### 4. Scope
Defines how a transient runtime/positions_truth.json::broker_authority_status=RED
window shall be classified during any active paper-soak session
(Paper Epoch 3 Session 1 through Session 5, and any future epoch).

### 5. The rule (verbatim — do not paraphrase)

A broker_authority_RED window may be classified as EXPLAINED during a
soak session only when ALL of the following are mechanically
verifiable from artifacts written during the window:

1. Duration: the RED window duration is ≤ 1× ledger publish cadence
   (15 minutes wall-clock).
2. No fresh signals: zero RoutedSignal emissions during the window,
   verified from the signal_router audit log.
3. No fresh entries: zero intent_type=ENTRY items in the execution
   pipeline, verified from execution_plan_audit.ndjson (or equivalent
   audit artifact).
4. No sizing decisions: runtime/dynamic_caps.json ts_utc lies outside
   the RED window OR the per-strategy sizing_factor values are
   unchanged across the window.
5. No SCR transitions: runtime/scr_state.json state band is unchanged
   across the window.
6. No risk-state mutation: ts_utc on runtime/profit_lock_state.json,
   runtime/stop_bus.json, and runtime/tier_state.json are all
   unchanged across the window.
7. No PnL realization: zero new closed-trade rows in
   data/trades/trade_history_<date>.ndjson during the window.
8. Reconciler self-resolved: the next chad-reconciliation-publisher
   cycle after the window returned broker_authority_status=GREEN
   without operator intervention or manual JSON edit.

If any of (1)–(8) cannot be verified or fails, the window is FAIL /
NOT COUNTABLE for soak purposes. No exception.

EXPLAINED classifications are recorded in
runtime/session_explanations.json with:

- window_open_utc, window_close_utc, duration_seconds
- the artifact paths used to prove each criterion
- the evaluator's sha256 of each artifact at the time of evaluation
- the evaluator process identity and ts_utc

Operators cannot mark a window EXPLAINED. Only the soak evaluator can,
mechanically. If the evaluator is not available, the default
classification is FAIL / NOT COUNTABLE.

### 6. Why this rule
- Strict any-RED-fails burns soak days on transient races that did
  not affect any decision.
- Loose operator-judgment EXPLAINED violates auditability and
  CLAUDE.md rule #3.
- Mechanical evidence-gated is the institutional middle ground: the
  machine grades itself; the operator cannot override.

### 7. What this rule does NOT do
- Does NOT silently fix R1 (canonical position-truth authority).
- Does NOT lower the soak bar — every criterion is stricter than
  current implicit behavior.
- Does NOT auto-pass any session.
- Does NOT authorize any live trading.

### 8. Evaluator implementation status
- Evaluator code: NOT YET BUILT.
- This document locks the rule in writing.
- Evaluator will be implemented in a separate Pending Action that
  must reference this rule by sha256.
- Until the evaluator is built, ALL RED windows default to FAIL /
  NOT COUNTABLE — there is no manual override path.

### 9. Acceptance criteria
- This file exists at the declared path.
- File is committed.
- The Session 1 tracker (PAPER_EPOCH_3_SESSION_TRACKER_2026-05-28.md
  if present) is updated to reference this rule for any RED
  classification decision.
- ops/pending_actions/SESSION_1_tracker_amendment_evidence_gated_2026-05-27.md
  is marked SUPERSEDED-BY this document.

### 10. Operator approvals required
- Operator GO to adopt this rule for Paper Epoch 3 Session 1.
- Operator GO required separately for the evaluator implementation PA.

### 11. No-live confirmation

This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.

### 12. Session 1 impact
This rule, if adopted, defines how the Session 1 evaluator will
classify any RED windows that occur during the
2026-05-28T00:00:00Z → 2026-05-29T00:00:00Z window. Because the
evaluator is not yet implemented, Session 1 will default to FAIL /
NOT COUNTABLE on any RED occurrence. The rule is locked now so the
evaluator (when built) has unambiguous criteria to enforce.
