# HISTORICAL-PLACEHOLDER — Quarantine of 14 trusted-fake fills in FILLS_20260503.ndjson

# HIGH_ID: HISTORICAL-PLACEHOLDER-1
# Status: PROPOSED
# Severity: HIGH (per audit §18 Findings Priority Matrix Effort S / HIGH cell; §19 closeout list classifies as MED — this PA treats it as HIGH consistent with the priority matrix and with the defense-in-depth standard already applied to live placeholder emission)
# Effort: S

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §6.3 (Historical fingerprint scan — 14 trusted-fake rows)
  - §6.4 (Trusted-fake timeline — all 14 in FILLS_20260503.ndjson lines 1–14)
  - §6.5 (Verdict: source/executor hardening VERIFIED; historical contamination PARTIAL)
  - §17 R4-A (root cause)
  - §18 Findings Priority Matrix — Effort S / Severity HIGH cell ("Quarantine 14 trusted-fake rows in `FILLS_20260503.ndjson` via exclusion tag")
  - §19 closeout item 7 (HISTORICAL-PLACEHOLDER-1)

# Problem statement
The historical fills file `data/fills/FILLS_20260503.ndjson` retains **14 trusted-fake placeholder rows** on lines 1–14, all `sym=SPY strat=delta`, all with `fill_price=100.0`, all timestamped 2026-05-03T11:48:55Z – 12:48:37Z. They are pre-hardening artifacts of the documented P0-1 incident.

SCR currently excludes these rows via `excluded_untrusted=321` in the live SCR stats — the exclusion is contract-enforced and visible. However, **the rows remain on disk**, and any new aggregator, replay path, backtest, or report that does not explicitly call the exclusion predicate would silently consume `100.0` as a real fill price. This is the textbook R4 pattern: "placeholder/fake-value source not removed (only blocked downstream)."

The audit's §18 priority matrix lists "Quarantine 14 trusted-fake rows in FILLS_20260503.ndjson via exclusion tag (no data deletion)" in the Effort S / Severity HIGH cell. The audit's §19 closeout list classifies the same item as MED. This PA treats it as **HIGH** for defense-in-depth parity with the executor/strategy-side hardening that has already shipped (PR-02 / PR-02b / paper_exec_evidence_writer / trade_closer).

# Evidence
1. **Historical fingerprint scan** (audit §6.3, full 18,098-row scan):
   - Rows with `fill_price=100.0`: **836**
   - Trusted-fake placeholders (`fill_price=100.0 AND reject != true AND pnl_untrusted != true`): **14**
   - Rows with explicit `placeholder` tag: 97
2. **Timeline concentration** (audit §6.4):
   - All 14 trusted-fake rows are in `data/fills/FILLS_20260503.ndjson` lines 1–14
   - All `sym=SPY strat=delta`
   - Span 2026-05-03T11:48:55Z – 12:48:37Z
3. **Since 2026-05-03**: zero trusted-fake placeholders (defense-in-depth hardening is effective for new fills).
4. **Current exclusion**: SCR `excluded_untrusted=321` includes these 14 rows.
5. **Prohibited action** (audit §20): "DO NOT delete or modify the 14 trusted-fake placeholder rows in `data/fills/FILLS_20260503.ndjson`." — the rows must remain on disk; quarantine is via tag/sidecar, not deletion.

# Affected files / readers
- `data/fills/FILLS_20260503.ndjson` (DO NOT EDIT — retain as-is for forensic preservation)
- Any reader/aggregator of `data/fills/FILLS_*.ndjson`, in particular:
  - `chad/execution/paper_exec_evidence_writer.py` (executor-side placeholder catcher, already hardened)
  - `chad/execution/trade_closer.py` (close-side defense)
  - SCR effective-trades calculator (already excludes 321 rows)
  - Any replay / backtest / report harness
  - Any new aggregator that may be written in future PRs

# Root-cause hypothesis
1. The 14 rows were emitted by the strategy-side delta code path before PR-02 (delta abstain on missing price) shipped.
2. Defense-in-depth (executor / trade_closer / paper_exec_evidence_writer) catches all subsequent placeholder emissions; the historical rows pre-date that defense.
3. The exclusion contract lives in SCR's effective-trades calculator and is honoured today. It is **not** centralised in a shared validation function, so any new reader that re-implements aggregation could silently re-include them.

# Required closeout
1. **Centralise the placeholder rejection predicate** in a shared module: `chad/execution/fill_validation.py::is_trusted_fake_placeholder(row)`. Detection MUST use multiple signals (defense-in-depth):
   - explicit trusted-fake marker (if present)
   - placeholder marker in `tags` / `extra.placeholder_fill_price`
   - delta SPY @ $100 pattern (`strat=delta AND sym=SPY AND fill_price=100.0`)
   - source/origin marker
   - `pnl_untrusted=true` flag
   - known audit exclusion fields
2. **Update fill readers** to call the shared predicate and exclude trusted-fake rows from any "trusted" code path. **Do not** delete or rewrite the historical rows.
3. **Read-only quarantine tool** at `chad/tools/quarantine_placeholder_fills.py`:
   - CLI `--check` is read-only and emits an evidence report.
   - CLI `--apply` requires `--operator-approve "<reason>"` with a non-empty reason; writes a timestamped backup and a sidecar quarantine file; produces an evidence report. **No `--apply` is invoked in this PA's commit.**
4. **Tests**:
   - Known 14-row fixture rejected by the shared predicate.
   - Legitimate SPY fill not rejected.
   - SCR / PnL / replay / backtest / report code paths all exclude trusted-fake rows.
   - Quarantine tool `--check` is read-only.

# Acceptance criteria
- `chad/execution/fill_validation.py` exists and is the single source of truth for placeholder rejection.
- All known readers of historical fill files call `is_trusted_fake_placeholder()` before treating a row as a trusted fill.
- A pytest fixture containing the 14 known rows verifies 14/14 are rejected.
- The historical file `data/fills/FILLS_20260503.ndjson` is unmodified on disk (sha256 unchanged).
- Quarantine tool `--check` exits cleanly with an evidence report; `--apply` requires explicit operator reason.

# Required tests
- `chad/tests/test_fill_validation.py`:
  - Fixture: 14-row synthetic copy of the known placeholder pattern → all rejected.
  - Negative fixture: legitimate SPY fill at non-$100 price → not rejected.
  - Each reader code path (SCR / PnL / replay / backtest / report) is exercised via dependency injection or test seam.
  - Quarantine tool `--check` mode produces a report without mutating any file.

# Operator approvals required
- **Approval 1 (shared-validator placement):** operator approves `chad/execution/fill_validation.py` as the canonical home of the predicate.
- **Approval 2 (sidecar quarantine path):** if a sidecar exclusion file is created (`data/fills/FILLS_20260503.quarantine.json` or similar), operator approves the path and schema.
- **Approval 3 (no historical-data deletion):** explicit confirmation that the historical `.ndjson` file is **never** mutated by this closeout; quarantine is via tag/sidecar.

# Session 1 impact
- **Does this item block Session 1 evaluation (window opens 2026-05-28T00:00:00Z)?** NO.
- SCR already excludes the 14 rows; Session 1 SCR/PnL evaluation is not affected by their on-disk presence.
- This PA hardens against a hypothetical future aggregator that does not honour the exclusion contract; it does not change any current Session 1 input.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.
