# Soak-policy reconciliation — BOX-059 is canonical (clean-soak grading)

**Status:** RATIFIED — governance decision, doc-only. Does NOT pass any soak,
does NOT change runtime, config, or live posture. CHAD remains PAPER.
`ready_for_live=false`, `allow_ibkr_live=false`.
**Date:** 2026-06-20
**Type:** Decision record / conflict reconciliation (no code, no restart, no
runtime write, no git commit — operator commits).
**Decided by:** operator ratification, recorded here.

## 1. Why this document exists

Three soak documents specify overlapping but conflicting "clean session"
criteria — most visibly an SCR-band conflict (CONFIDENT-only vs.
CONFIDENT-or-CAUTIOUS) and a session-boundary conflict (RTH session vs.
UTC calendar day). This record ratifies which document governs soak
PASS / FAIL / NOT-COUNTABLE, so there is a single source of truth for
grading. It resolves a documentation conflict only; it grades no session.

## 2. Ratified decision

1. **BOX-059 is THE canonical full-session clean-soak policy.**
   `ops/pending_actions/BOX-059_sustained_clean_paper_soak_policy.md` —
   its eight gates ("Eight gates", lines 38–89), its RTH-session boundary
   and five-trading-day window ("Soak window — minimum duration",
   lines 20–36) govern PASS / FAIL / NOT-COUNTABLE.

2. **SCR criterion (band, not level).** A session requires SCR ∈
   {CONFIDENT, CAUTIOUS}, never PAUSED — BOX-059 **gate 7** (lines 76–82).
   **CAUTIOUS does NOT fail a session.** This supersedes, **FOR SOAK
   GRADING ONLY**, the CONFIDENT-only framing in:
   - `PAPER_EPOCH_3_SESSION_TRACKER_2026-05-28.md` §3 criterion 6
     ("SCR remains CONFIDENT", line 42); and
   - `PAPER_EPOCH_3_START_2026-05-27.md` §7 criterion 3
     ("SCR stability … remains CONFIDENT", line 120).

   (Both of those criteria already carry an "or any transition is
   explained" escape hatch; this decision makes the CAUTIOUS band an
   explicitly accepted steady state for grading, not a downgrade that must
   be justified per session.)

3. **Session boundary = BOX-059 RTH session**, not the Tracker's UTC
   calendar day (`PAPER_EPOCH_3_SESSION_TRACKER_2026-05-28.md` line 26,
   "US-equity calendar session"; `PAPER_EPOCH_3_START_2026-05-27.md` §8
   line 137, "calendar US-equity session"). BOX-059 defines a trading day
   as "any RTH session in which `feed_state` is non-stale and
   `chad-live-loop.service` has been up the entire session" (lines 24–27).

4. **Fill-productivity is NOT a clean-session requirement.** The
   `PAPER_EPOCH_3_START_2026-05-27.md` §8 "≥1 real broker-confirmed fill
   produced (per … ACTIVE_PRODUCTIVE strategy)" criterion (line 139) is
   **not adopted** as a clean-soak gate. BOX-059's gate 4 ("Fills clean",
   lines 58–63) grades fill *integrity*, not fill *count*.

5. **RED-window classification authority is unchanged.** The locked
   `SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md`
   (sha256 `800480a570e8b2e50d115b9a62d52d6b7022cd5dc0a4421a740528061ef60723`)
   remains the **sole** authority for classifying
   `broker_authority_status=RED` windows WITHIN a BOX-059 session. Its
   eight mechanical criteria, EXPLAINED/FAIL semantics, and "no operator
   override" rule are untouched by this reconciliation.

## 3. Rationale (one line)

The soak grades **cleanliness / stability**, not profitability; edge
validation is a separate gate. The CAUTIOUS band is a deliberate, explained
de-contamination posture (futures exclusion, 2026-06-19) — not a
performance collapse.

## 4. Explicit non-effects

This decision:
- does **NOT** flip `ready_for_live` or `allow_ibkr_live`;
- does **NOT** advance live promotion (L1–L16);
- does **NOT** remove, weaken, or add any gate;
- does **NOT** pass, fail, or count any soak session.

The out-of-sample, cost-adjusted **edge-validation harness remains the gate
to live money.** A clean BOX-059 soak proves stability, not edge.

## 5. Known pre-soak blockers (must clear before any session counts clean)

(a) **`position_guard_drift.drift_count` must be 0** with each drift
    resolved + explained. The ratified intent records an open
    `alpha|CVX` `side_mismatch` (drift_count=1) as the blocker to clear.
    *Runtime observation 2026-06-20:* `runtime/position_guard_drift.json`
    currently reads `drift_count=0, drifts=[]` — the CVX drift appears
    cleared or is flapping. **Re-verify at soak start**; BOX-059 gate 6
    (lines 71–74) requires a continuously-green drift signal across the
    full window, so a flapping drift still breaks the streak.

(b) **`runtime/epoch_state.json` still reads
    `active_epoch=CHAD_v8.9_Paper_Epoch_2`** (verified 2026-06-20). The
    flip to Paper_Epoch_3 is a **separate operator action, still not
    taken** (`PAPER_EPOCH_3_START_2026-05-27.md` §11, lines 161–165). This
    record does not authorize it.

## 6. Source documents reconciled (cited by path)

| Role in this reconciliation | Path |
|---|---|
| Canonical clean-soak policy (governs grading) | `ops/pending_actions/BOX-059_sustained_clean_paper_soak_policy.md` |
| Superseded for SCR-band + boundary (grading only) | `ops/pending_actions/PAPER_EPOCH_3_SESSION_TRACKER_2026-05-28.md` |
| Superseded for SCR-band + boundary + fill-productivity (grading only) | `ops/pending_actions/PAPER_EPOCH_3_START_2026-05-27.md` |
| RED-window classification authority (unchanged) | `ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md` (sha256 `800480a5…`) |

"Superseded … (grading only)" means: where these documents conflict with
BOX-059 **on soak grading**, BOX-059 wins. Every other purpose those
documents serve (observation ledgers, evidence-field templates, day-0
record, non-live statements) is unaffected.

## 7. SSOT pointer (operator action — not applied here)

`docs/CHAD_UNIFIED_SSOT_v9.7_2026-06-04.md` is **LOCKED** (tag
`SSOT_v9_7_2026-06-04`, commit `4bac3a8`) and has no dedicated decision-log
section. It is **not edited** by this record (no SSOT surgery). At the next
SSOT cut (v9.8), the operator may add this one line to its decision log:

> Soak grading: BOX-059 is canonical; SCR band {CONFIDENT,CAUTIOUS} (gate 7,
> CAUTIOUS does not fail), RTH-session boundary, fill-count not a gate;
> RED-window rule (sha256 800480a5…) unchanged. Ref:
> `ops/pending_actions/SOAK_POLICY_RECONCILIATION_BOX-059_CANONICAL_2026-06-20.md`.

## 8. Constraints honored

Doc-only. No code change, no config mutation, no service restart, no runtime
write, no git commit (operator commits). No gate added or removed. CHAD
remains PAPER; live trading remains NOT authorized.
