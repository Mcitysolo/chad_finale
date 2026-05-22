# BOX-046 (Official Matrix) — Stale positions_snapshot memory retracted

- **Box number (Official Matrix):** 046
- **Box title (Official Matrix):** Stale memory retracted — old positions_snapshot "stale since Apr 3" claim is removed/superseded
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T19:07:14Z
- **HEAD at cut:** `bbe7525` (short) — "GAP-039 (Phase-58/59): relocate stop-bus evaluate before early-return"
- **Branch:** `main`

> **Note on box numbering.** This file closes the **Official Matrix
> Box 046** ("Stale memory retracted"). It is distinct from the
> Supplemental Annex Box 046 ("DoD checklist updated to v1.0")
> closed earlier in the run; the Official Matrix and Supplemental
> Annex use independent numbering. The Supplemental Box 046
> evidence lives at
> `runtime/completion_matrix_evidence/BOX-046_DOD_checklist_updated_to_v1_0.md`;
> the Official Matrix Box 046 evidence (created together with this
> policy doc) lives at
> `runtime/completion_matrix_evidence/BOX-046_OFFICIAL_stale_memory_retracted.md`.

---

## 0. Scope and safety statement

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This retraction does not flip
  `ready_for_live`, does not change any runtime state, and does not
  authorize live trading.
- **`ready_for_live` remains `false`** (`runtime/live_readiness.json`).
- This is a documentation / auto-memory supersession only — no
  runtime file is mutated.

---

## 1. The stale claim being retracted

**Source:** Claude Code auto-memory at
`/home/ubuntu/.claude/projects/-home-ubuntu-chad-finale/memory/audit_2026_05_09_positions_snapshot_stale.md`
(authored 2026-05-09; indexed by the session-bound `MEMORY.md`).

**Original wording (verbatim from the memory description):**

> "positions_snapshot.json has been stale since 2026-04-03 because no
> active systemd unit writes it; positions_truth.json reads it and
> reports phantom-stale state. broker_sync|* guard entries are
> CORRECT."

**Original body claim:**

> "What IS stale: `runtime/positions_snapshot.json` (last write
> 2026-04-03 23:55:24 UTC, 36 days old)."

This wording reflected filesystem state at 2026-05-09. It is
**superseded** as of 2026-05-20 — see §2.

---

## 2. Why it is superseded (filesystem evidence at audit time)

Read-only `ls -la` on `runtime/positions_snapshot.json` at audit time
(2026-05-20T19:07:14Z) shows:

```
-rw-r--r-- 1 ubuntu ubuntu 3714 May 20 12:45 /home/ubuntu/chad_finale/runtime/positions_snapshot.json
```

That is **mtime ≈ 2026-05-20 12:45 UTC** — i.e. the snapshot was
written today, ≈ 6.5 hours before this audit. The "stale since Apr 3"
claim is therefore factually false at the filesystem level: the file
is fresh today.

Related read-only observations at the same audit moment:

| File / unit                                       | Observation                                                                                                |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `runtime/positions_snapshot.json`                 | mtime 2026-05-20 12:45 UTC — **refreshed today**                                                            |
| `runtime/positions_truth.json`                    | mtime 2026-05-20 19:08 UTC — current                                                                        |
| `runtime/position_guard.json`                     | mtime 2026-05-20 04:19 UTC — current cycle of the live-loop-rebuilt guard                                    |
| `runtime/position_guard_drift.json`               | mtime 2026-05-20 19:05 UTC; `drift_count: 2` (alpha\|MSFT, alpha_intraday\|MSFT — `side_mismatch`)            |
| `runtime/reconciliation_state.json`               | mtime 2026-05-20 19:06 UTC; **`status: RED`**, **`broker_source: "unavailable:"`**                          |
| `chad-positions-snapshot.service`                 | `failed` (Result=exit-code, ExecMainStatus=1, ActiveEnterTimestamp empty)                                    |
| `runtime/live_readiness.json`                     | `ready_for_live: false` (publisher-driven; unchanged)                                                       |

**Interpretation:**

- The snapshot file IS being written (today's mtime is proof).
- The dedicated `chad-positions-snapshot.service` is still in
  `failed` state, so either an earlier-today success was followed by
  a later failure, or another path is writing the snapshot (e.g.
  manual / orchestrator-embedded invocation of
  `python -m chad.portfolio.ibkr_portfolio_collector_v2 collect`).
  The service-health follow-up remains a Stage-3 maintenance item
  (Box 045 §7 / Box 052 G-26 / Box 054 D-06).
- The reconciliation status is `RED` with `broker_source: "unavailable:"`
  — meaning even though the snapshot file is fresh, the canonical
  reconciliation pipeline currently cannot resolve broker truth.
  Under these conditions the right answer to "what positions does
  CHAD have right now?" is **UNKNOWN / requires audit** — not
  "we know the positions because the May-9 memory told us".

---

## 3. Canonical current position-truth source policy (replaces the May-9 memory's guidance)

When reasoning about current positions:

1. **Read `runtime/position_guard.json`** — `broker_sync|*` entries
   are the primary live-loop-rebuilt broker view. Rebuilt every cycle
   by the orchestrator's broker-sync path.
2. **Cross-check `runtime/position_guard_drift.json`** — any
   `drift_count > 0` indicates a per-strategy guard-vs-broker
   divergence; treat the affected key as **operator-review pending**
   (G-01 / D-01 in the Stage-3 trackers).
3. **Verify `runtime/reconciliation_state.json`** — if `status: RED`
   with `broker_source` unavailable or any drift advisory, the
   canonical position truth is **UNKNOWN / requires audit**. Do NOT
   substitute the May-9 memory's symbol list or any other older
   snapshot.
4. **Read `runtime/positions_snapshot.json`** — supplementary; check
   mtime first; only use as supporting evidence, not the sole truth.
5. **Read `runtime/positions_truth.json`** — derived view; OK to
   read but cross-check against #1 and #3.

**Hard rule:**

> Do NOT quote the May-9 memory's symbol list (TLT, JNJ, CVX, AAPL,
> MES, GOOGL, NVDA, QQQ, UNH, VWO, IEMG, GLD, BAC, PEP) as a current
> claim. That list is a 2026-05-09 snapshot and may have drifted.
> Always re-read the canonical runtime files at the moment of the
> question.

---

## 4. Patches applied by Box 046 retraction

| Target                                                                                                              | Action                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/home/ubuntu/.claude/projects/-home-ubuntu-chad-finale/memory/audit_2026_05_09_positions_snapshot_stale.md`          | **Patched** — added "SUPERSEDED 2026-05-20" header at top, marked frontmatter `name`/`description` with SUPERSEDED tag, prepended canonical current-truth-source rule, preserved original body verbatim under "ORIGINAL FINDING (FROZEN)" section. |
| `/home/ubuntu/.claude/projects/-home-ubuntu-chad-finale/memory/MEMORY.md`                                            | **Patched** — index entry rewritten to mark SUPERSEDED 2026-05-20 with link to this Box-046 retraction and a one-line summary of the canonical current-truth-source rule.                                            |
| `ops/pending_actions/BOX-046_stale_positions_snapshot_memory_retracted.md`                                            | **Added** — this policy doc (the formal retraction record).                                                                                                                                                       |
| `runtime/completion_matrix_evidence/BOX-046_OFFICIAL_stale_memory_retracted.md`                                       | **Added** — the Box-046 (Official) closure evidence (paired with this doc).                                                                                                                                        |
| Repository docs / code referencing `positions_snapshot`                                                              | **Unchanged** — repo grep for "stale since Apr" / "Apr 3" / "April 3" returned **0 hits** in the repo (excluding venv/.git/etc.). The stale claim lived only in auto-memory; no repo doc carried it forward. The repo references to `positions_snapshot` are code paths (file-name references), not stale-claim text. |
| Frozen historical SSOT (v8.x, v9.0 – v9.5 errata)                                                                    | **Unchanged** — forward-only discipline preserved; no retroactive edits.                                                                                                                                          |

**`files_patched_count` = 2** (the auto-memory body + MEMORY.md index;
new policy doc + new evidence file are *added*, not *patched*).

---

## 5. Static-check intent (verified at the evidence file)

- This policy doc and the paired evidence file contain the word
  **`superseded`** explicitly.
- This policy doc and the paired evidence file contain the canonical
  safety phrase **`live trading not authorized`**.
- The original "stale since Apr 3" wording remains traceable only
  inside the SUPERSEDED auto-memory body (preserved under "ORIGINAL
  FINDING (FROZEN)") and within quoted/refutation context here in §1
  — not as a positive current claim.

---

## 6. False-closure guardrails

This retraction does NOT:

- claim CHAD is complete.
- claim CHAD is live-ready.
- authorize live trading.
- mutate any `runtime/*.json`, SQLite, ledger, fill, fee, trade,
  broker event, or order.
- restart / start / stop / daemon-reload any service.
- assert what CHAD's current positions are. (At audit time
  `reconciliation_state.status: RED` and `broker_source: "unavailable:"`
  — so the only safe current-position answer is **UNKNOWN / requires
  audit**.)

---

## 7. Anti-speculation footer

- The retraction is grounded in the read-only filesystem mtime
  observation (`runtime/positions_snapshot.json` mtime 2026-05-20
  12:45 UTC); no inference, no guess.
- The original May-9 memory body is preserved verbatim inside the
  patched memory file under "ORIGINAL FINDING (FROZEN)" so the
  retraction is traceable.
- No symbol list or current-position claim is asserted by this
  retraction.
- `runtime/live_readiness.json` `ready_for_live` remained `false`
  before, during, and after this retraction.

**live trading not authorized. CHAD remains PAPER. `ready_for_live=false`.**
