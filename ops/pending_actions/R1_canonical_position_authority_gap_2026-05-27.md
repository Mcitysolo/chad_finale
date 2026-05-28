> **STATUS UPDATE 2026-05-27:**
> Decision 1 institutional response landed in commit <will-be-filled-after-commit>.
> Per-surface authority designation is REPLACED by the `position_truth_v2`
> engine — see `docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md`.
> Code stub at `chad/core/position_truth_engine.py` and schema at
> `chad/schemas/position_truth_v2.py` implement the 5 merge rules in
> §9 of the design.
>
> Production wiring (`positions_truth.json` repointing) is deferred to
> a separate operator-authorized phase per the migration plan in §12
> of the design.

# R1 — Canonical Position-Authority Gap (positions_snapshot vs ibkr_paper_ledger_state)

# Status: PROPOSED

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §3.B (Tier 2 position-truth table)
  - §8 (Tier 7 reconciliation deep dive, MGC drift)
  - §17 R1-A and R1-B (root-cause grouping)
  - §19 closeout item 1 (TRUTH-RECONCILE-1)

# Severity
**HIGH** (Effort M) per audit §18.

# Problem statement
There is **no code-enforced single canonical writer** for the paper-position truth surface. Two surfaces co-exist with different schemas, different cadences, and intermittently different counts:

- `runtime/positions_snapshot.json` — symbol-keyed list of position rows. Source `ibkr_portfolio_collector_v2`. Published by `chad-positions-snapshot.service` via `chad-positions-snapshot.timer` (5 min cadence). **No `schema_version` field.**
- `runtime/ibkr_paper_ledger_state.json` — hash-keyed dict of position records (SHA256-style keys → contract metadata). Published by `chad/portfolio/ibkr_paper_ledger_watcher.py` via `chad-ibkr-paper-ledger-watcher.timer` (15 min cadence). **No `schema_version` field.**

The classifier `chad/ops/lifecycle_truth_publisher.py` (and adjacent `lifecycle_replay_*` modules) emits `runtime/positions_truth.json` (`positions_truth.v1`). When the two surfaces disagree, it sets `broker_authority_status=RED` with reasons like:

- `BROKER_AUTHORITY_RED: count_mismatch_ledger=18_vs_snapshot=19`
- `truth_source=FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN`
- `replay_diagnostic_blocks_truth=true`

The mismatch is **directionally inconsistent** within a single day:
- Morning audit (13:50Z): snapshot=18, ledger=19 → RED
- Afternoon audit (14:30Z): snapshot=19, ledger=18 → RED
- Post-audit re-check (15:24Z): GREEN, both publishers caught up

Each flip is explained by a transient cadence-lag (one publisher tick lands before the other), but the structural pattern is permanent until a single authority is designated.

# Evidence
1. **`positions_truth.json` at audit time (14:30Z):**
   ```
   broker_authority_status = RED
   broker_authority_reason = BROKER_AUTHORITY_RED: count_mismatch_ledger=18_vs_snapshot=19
   truth_ok = false
   truth_source = FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN
   snapshot_positions_count = 19
   ledger_state_positions_count = 18
   replay_positions_count = 7
   missing_from_replay = ['BAC','CVX','GLD','GOOGL','IEMG','KO','M2K','M6E','NVDA','PEP','TLT','UNH','VWO']
   replay_only = ['MES', 'MGC']
   ```
2. **MGC drift case study:**
   - `position_guard.json::alpha_futures|MGC` opened 2026-05-27T14:34:50Z (SELL qty=1.0).
   - `position_guard.json::broker_sync|MGC` closed 2026-05-27T13:43:29Z by `broker_truth_rebuild`.
   - `positions_snapshot.json` at 14:32:44Z (10 s before the alpha_futures trade) did not contain MGC.
   - `position_guard_drift.json` emitted `drift_count=1` `broker_truth_missing` for `alpha_futures|MGC`.
   - At 15:22:49Z, after subsequent snapshot ticks, `drift_count=0` and `positions_truth` flipped GREEN.
3. **Existing acknowledgement:** `ops/pending_actions/BOX-047_dual_ledger_authority_policy.md` documents the dual-authority concern (Pending Action only; no code-enforced resolution).
4. **PR-09 contract** (`chad/ops/lifecycle_truth_publisher.py` and `positions_truth.v1` schema header): explicitly separates `broker_authority_status` from `replay_diagnostic_status`. The contract is in place; the **single canonical writer** between snapshot and ledger is not.

# Affected files / services / artifacts
- `runtime/positions_snapshot.json` (no schema_version)
- `runtime/ibkr_paper_ledger_state.json` (no schema_version)
- `runtime/positions_truth.json` (`positions_truth.v1`)
- `runtime/position_guard.json` (no schema_version, 69 keys today)
- `runtime/position_guard_drift.json` (`position_guard_drift.v1`)
- `runtime/reconciliation_state.json` (no schema_version)
- `chad/portfolio/ibkr_portfolio_collector_v2.py` (snapshot writer)
- `chad/portfolio/ibkr_paper_ledger_watcher.py` (ledger writer)
- `chad/ops/lifecycle_truth_publisher.py` (truth classifier)
- `chad/ops/lifecycle_replay_coverage.py`, `chad/ops/lifecycle_replay_drift_audit.py`
- `chad-positions-snapshot.service` + `.timer` (5 min cadence)
- `chad-ibkr-paper-ledger-watcher.service` + `.timer` (15 min cadence)
- `chad-reconciliation-publisher.service` + `.timer`
- `chad-lifecycle-truth-publisher.service` + `.timer`

# Root-cause hypothesis
The two writers run on **different timers with no coordination barrier**, so any trade that lands between a snapshot tick (5 min) and the next ledger tick (15 min) produces an apparent count mismatch. The truth classifier correctly flags this as `broker_authority_RED`, but downstream consumers cannot distinguish "transient cadence lag" from "real divergence." The schemas (symbol-keyed list vs hash-keyed dict) also force the classifier to do a contract→symbol resolution that is not externally verifiable from a `view`-level audit.

Underlying cause: **no governance document specifies which file is the source of truth when they disagree, and no test enforces it.**

# Why this matters
1. Today's intraday validation observed `broker_authority_RED` for at least 96 min (13:50Z → 15:22Z) on a CONFIDENT paper day with no operator intervention.
2. Session 1 pass criterion §3.7 requires **reconciliation GREEN** for the full window. A single 5–15 min RED interval during Session 1 (2026-05-28T00:00Z → 23:59Z) would force a FAIL or NOT COUNTABLE verdict.
3. Every audit so far has surfaced a "new" surprise in this surface (the audit's recurring-surprise hypothesis traces directly to this gap).
4. Promoting to live is blocked while truth_ok can flip false on a transient timing artifact.

# Current safety posture
- All live gates held: `ready_for_live=false`, `allow_ibkr_live=false`, `allow_ibkr_paper=true`.
- `chad_mode=paper`, `live_enabled=false`.
- `truth_source=FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN` is **fail-closed by design** when truth is uncertain.
- No broker orders have been placed under uncertain truth state today.
- Defense-in-depth (PR-09 separation, position_guard, position_guard_drift detector) is functioning.

# Scope for future remediation
1. **Designate the single canonical paper-position writer.** Two options:
   a. **Snapshot-authoritative**: `positions_snapshot.json` becomes the source of truth; `ibkr_paper_ledger_state.json` becomes a derived audit trail with no role in `broker_authority_status`.
   b. **Ledger-authoritative**: `ibkr_paper_ledger_state.json` becomes the source of truth; `positions_snapshot.json` becomes a snapshot view for dashboards only.
   The decision is operator-domain; the PR is code+doc.
2. **Add `schema_version` to both files** (and to `position_guard.json`, `reconciliation_state.json`, `portfolio_snapshot.json`, `decision_trace_heartbeat.json` — see also SCHEMA-VERSION-1 / MED).
3. **Coordinate cadences**: either run the chosen authority writer at the higher cadence (per-cycle or 1 min) and the derived view less often, or add a strict "wait for both publishers before classifying" barrier in `lifecycle_truth_publisher.py`.
4. **Add a regression test** that fails if `positions_truth.broker_authority_status=RED` for more than 2× the canonical-writer cadence on a regular trading day with `reconciliation_state.status=GREEN`.
5. **Update PR-09 contract** to make the authority direction explicit (currently the contract separates broker_authority from replay_diagnostic but does not specify snapshot-vs-ledger precedence).

# Explicitly out of scope
- Live-mode enablement.
- Any change to runtime config or systemd unit files.
- Any change to broker order routing.
- Deletion of either file (both remain on disk; one becomes "derived").
- Modification of `runtime/epoch_state.json`.

# Required tests
- New: regression test that mocks a snapshot/ledger count mismatch and asserts the truth classifier resolves to a deterministic GREEN/RED per the canonical-writer rule (not based on timing).
- New: regression test that fails if either canonical surface lacks `schema_version`.
- Existing must continue to pass: `chad/tests/test_positions_truth_classifier.py`, `chad/tests/test_pr09_position_truth_contract.py`, `chad/tests/test_portfolio_surface.py`, `chad/tests/test_ibkr_portfolio_collector_v2_positions.py`.
- Full pytest baseline must remain ≥ current minus the known persistent `test_phase_a_item4_setup_tagging.py::test_02_alpha_intraday_momentum_surge_setup_family` failure.

# Required runtime verification
After remediation lands and a single clean cycle has executed:
- `runtime/positions_truth.json::broker_authority_status = GREEN` continuously for ≥ 60 min on a regular trading day.
- `runtime/positions_truth.json::truth_ok = true`.
- `runtime/positions_truth.json::truth_source` not equal to `FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN`.
- `runtime/position_guard_drift.json::drift_count = 0` continuously across at least 3 consecutive cadence ticks of the chosen canonical writer.
- `runtime/reconciliation_state.json::status = GREEN` continuously for ≥ 60 min.
- Both canonical and derived files carry `schema_version`.
- Test suite passes with the new regression tests.

# Operator approvals required
- **Approval 1 (architecture):** which surface becomes canonical (snapshot vs ledger). Operator-domain.
- **Approval 2 (publisher cadence change):** if the canonical writer cadence is changed from 5 min / 15 min to something else.
- **Approval 3 (no live coupling):** explicit confirmation that this remediation does NOT unblock live transition by itself — live gating remains governed by Pre-Live Operator Tasks 1–8 in `CLAUDE.md`.

# Definition of done
1. A merged PR designating the single canonical paper-position writer with code-enforced rules in `chad/ops/lifecycle_truth_publisher.py`.
2. `schema_version` present on both surfaces and on `position_guard.json`, `reconciliation_state.json`, `portfolio_snapshot.json`.
3. New regression tests merged and passing.
4. Runtime verification observed for ≥ 60 min on a regular trading day (paper).
5. SSOT / forward-erratum doc updated to reflect the designated authority.
6. BOX-047 dual-ledger-authority Pending Action closed with link to the PR.
7. CLAUDE.md "Position-Guard Rebuilder Policy (GAP-028 Option B — PERMISSIVE)" section reviewed for compatibility with the new authority rule.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.

# Session 1 impact
- **Does this item block Session 1 evaluation (window opens 2026-05-28T00:00:00Z)?** PARTIALLY YES.
- A recurrence of `broker_authority_status=RED` during the Session 1 window (2026-05-28T00:00Z → 23:59Z) **would fail criterion §3.7** of the Session 1 tracker and force a FAIL or NOT COUNTABLE verdict. Today (Day-0) already observed at least 96 min of RED state on a CONFIDENT day with no broker outage and no operator action — a recurrence tomorrow is plausible.
- **Minimum remediation needed before Session 1 opens** (≈ 9 h from this Pending Action creation): the full architectural remediation is **not achievable** in 9 h as a controlled Channel-2 task. The PR requires design, review, test, and runtime soak.
- **Mitigation for Session 1**: amend the Session 1 tracker so that a transient `broker_authority_RED` window of ≤ 15 min (≤ 1× ledger cadence) is treated as `EXPLAINED` rather than FAIL, with the explanation linked to this Pending Action. This is a tracker-amendment Pending Action (separate from this remediation), not authorized here.
- **Latest acceptable closure date for the architectural remediation**: before Session 2 if Session 1 closes PASS via the mitigation; before Session 1 re-base if Session 1 closes FAIL/NOT COUNTABLE due to this item.
