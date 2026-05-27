# Session 1 tracker amendment — evidence-gated treatment of broker_authority_RED windows

# HIGH_ID: (cross-cuts TRUTH-RECONCILE-1 / Session 1 evaluation)
# Status: PROPOSED
# Severity: HIGH (Session-1 gating rule; not auto-applied)
# Effort: S (documentation only)

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §17 R1-A (canonical-authority gap; daily broker_authority_RED windows)
  - §19 closeout items 1 (TRUTH-RECONCILE-1) and 2 (MGC-DRIFT-1)
- `ops/pending_actions/R1_canonical_position_authority_gap_2026-05-27.md`
  - "Session 1 impact" section: today (Day-0) observed ≥ 96 min of
    broker_authority_RED on a CONFIDENT day with no operator intervention.

# Problem statement
Session 1 evaluation opens at 2026-05-28T00:00:00Z. Pass criterion §3.7 of
the Session 1 tracker requires `runtime/reconciliation_state.json::status =
GREEN` and `runtime/positions_truth.json::broker_authority_status = GREEN`
for the full window. The R1 canonical-authority gap means a 5–15 min
`broker_authority_RED` window driven solely by publisher cadence-lag is
**plausible** during Session 1 — Day-0 already observed two such windows.

A blanket FAIL on any RED interval would penalise a known-benign transient
that has no decision-path impact. A blanket PASS on any RED interval would
mask real broker-truth divergence.

# Proposed amendment — strict evidence gate (no auto-pass)
A `broker_authority_RED` window of ≤ 15 minutes may be classified
**EXPLAINED** (counted as PASS) ONLY when, for every minute of the window,
fresh evidence proves that no decision was affected:

1. **Ledger evidence** — `runtime/ibkr_paper_ledger_state.json` ts_utc is
   within its 15-min cadence; no entry was added, removed, or had its
   `qty` changed during the window.
2. **Snapshot evidence** — `runtime/positions_snapshot.json` ts_utc is
   within its 5-min cadence; the `positions_count` delta within the window
   is explained by a single completed cycle of the snapshot writer.
3. **Broker evidence** — `runtime/reconciliation_state.json` shows
   `broker_source` continuously available; no `chad_state_source` flips.
4. **Timestamp evidence** — every per-symbol record in `position_guard.json`
   has `updated_at_utc` consistent with the cadence.
5. **Routing evidence** — `runtime/last_route_decision.json` was not
   issued for any symbol whose truth status was disputed during the
   window.
6. **Sizing evidence** — `runtime/dynamic_caps.json` did not flip
   `total_equity` based on a stale snapshot.
7. **PnL evidence** — `runtime/pnl_state.json` aggregation did not drop
   any symbol from the trusted set due to the disputed status.
8. **SCR evidence** — `runtime/scr_state.json::state` remained CONFIDENT
   (or stable at its prior band).
9. **Risk-state evidence** — `runtime/stop_bus.json::active` did not flip
   to true during the window.

Without ALL nine evidence items honestly populated for every minute of the
window, the window remains **FAIL / NOT COUNTABLE**.

A RED window of > 15 minutes is **never EXPLAINED**. It is FAIL / NOT
COUNTABLE regardless of evidence.

# Explicit non-rules
- No rule that auto-passes a RED window.
- No rule that classifies based on duration alone.
- No rule that classifies based on absence of broker orders alone.
- No rule that treats "publisher running on its normal cadence" as
  evidence of no decision impact — cadence-lag is the *cause* of these
  windows, not exculpatory of them.

# Affected artefacts
- `ops/pending_actions/SESSION_1_*` tracker (if it exists; the amendment
  rule lives in operator-domain). The actual tracker amendment is
  operator-applied; this Pending Action only proposes the rule.

# Required closeout
- Operator reviews and approves (or rejects) the amendment.
- If approved, operator updates the Session 1 tracker with the
  evidence-gated rule above and the explicit non-rules.
- This PA is closed when the operator records the decision (approve /
  reject / defer) in this file's Status field.

# Required tests
- No code change in this PA.
- A future PR that codifies the evidence-gate (a "session pass calculator")
  must include tests that:
  - Reject a RED window of any length without all 9 evidence items.
  - Reject a RED window > 15 min unconditionally.
  - Classify EXPLAINED only when all 9 items pass independently.

# Operator approvals required
- **Approval 1 (amendment text):** operator approves the exact evidence-gate
  wording before it lands in any tracker.
- **Approval 2 (no auto-pass rule):** explicit confirmation that no rule
  will be added anywhere that auto-passes a RED window.

# Session 1 impact
- **Does this item block Session 1?** PARTIALLY YES.
  - Without the amendment, the existing tracker rule § 3.7 will fail Session 1
    if any RED interval occurs.
  - With the amendment, a benign cadence-lag RED interval ≤ 15 min may
    pass — **only with full evidence**.
- The amendment trades off "strict full-window GREEN" for "evidence-gated
  benign cadence-lag tolerance" — operator's choice.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.
