# BOX-041 — dynamic_caps backup retention policy (Pending Action)

- Status: **PENDING — not applied to runtime**
- Owner: CHAD ops
- Authored: 2026-05-20
- Closes Box 41 (GAP-011 dynamic_caps clutter bounded)
- Live trading authorization required: **NO**
- Restart required: **NO**

---

## 1. Background

`runtime/dynamic_caps.json` is the authoritative per-strategy dollar-cap file
written by `chad/core/orchestrator.py::Orchestrator.refresh_dynamic_caps()`.
The orchestrator writes it via `_atomic_write_json(...)` which:

1. Writes a `.tmp.<pid>` sidecar
2. Fsyncs the bytes
3. `os.replace(tmp, path)` — atomic rename
4. Fsyncs the parent directory (best-effort)

→ **The atomic-write path creates NO backup files** in steady state. There is
no `*.bak`, `*.~`, `*.old`, `.pre_fix_*`, `.post_fix_*`, or `.stale_*` file
emitted by any CHAD code path. Repo-wide search confirms zero code matches
for those suffix patterns against `dynamic_caps*`.

## 2. Current `runtime/` inventory (2026-05-20T03:30:01Z)

Active runtime state (5 files, 18.4 KB total):

| file | size | mtime |
|------|-----:|-------|
| runtime/dynamic_caps.json | 9254 | 2026-05-20T03:29Z |
| runtime/dynamic_caps_correlation.json | 1584 | 2026-05-20T03:29Z |
| runtime/dynamic_caps_dominance_overlay.json | 1210 | 2026-03-26T18:37Z |
| runtime/dynamic_caps_risk_governed.json | 3756 | 2026-03-26T18:49Z |
| runtime/dynamic_caps_quarantine.json | 2555 | 2026-03-26T20:12Z |

Operator forensic backup-tag artifacts (3 files, 11.1 KB total):

| file | size | created | purpose |
|------|-----:|---------|---------|
| runtime/dynamic_caps_correlation.json.stale_20260326 | 1572 | 2026-03-26T20:23Z | snapshot prior to staleness fix |
| runtime/dynamic_caps.json.pre_fix_20260424 | 5165 | 2026-04-24T19:47Z | before-state of 2026-04-24 fix |
| runtime/dynamic_caps.json.post_fix_20260424T215400Z | 4348 | 2026-04-24T21:54Z | after-state of 2026-04-24 fix |

**Total dynamic_caps disk footprint: ~29 KB** (all 8 files combined).

## 3. Retention policy (proposed; not applied)

Because the writer never produces backups itself, the only growth source is
**operator-initiated forensic snapshots** taken with name suffixes like
`*.pre_fix_*`, `*.post_fix_*`, `*.stale_*`. The following rule formalizes
boundedness:

### 3.1 Pattern definitions

Operator-tag backup pattern (regex on basename):

    ^(dynamic_caps(_[a-z_]+)?\.json)\.(pre_fix|post_fix|stale)_[0-9TZ]+$

Active state pattern (never matches the above):

    ^dynamic_caps(_[a-z_]+)?\.json$

### 3.2 Retention rule

- **Active files** (`runtime/dynamic_caps*.json` with no further suffix):
  **NEVER deleted, NEVER archived** by any retention tool. Protected by the
  fact that the operator-tag pattern requires a 3rd dotted segment.
- **Operator-tag backups**: keep the **newest 10** per base-file, OR all
  whose mtime is **within 90 days**, whichever is the larger set. Older
  artifacts are moved to `_archive/dynamic_caps_backups/YYYY/MM/`, not
  deleted (so forensic history is preserved off the hot runtime path).
- **Crash-residue `.tmp.<pid>`** files: already pruned by `chad-disk-guard`
  every cycle via `find $RUNTIME -maxdepth 2 -name "*.tmp*" -delete`. No
  change needed.

### 3.3 Why archive rather than delete

`pre_fix_*` / `post_fix_*` snapshots are audit evidence for past incidents
(e.g., 2026-04-24 hotfix). Deleting them would destroy the forensic record.
Archiving moves them to `_archive/dynamic_caps_backups/` where they're
naturally bounded by the existing `chad-disk-guard` 30-day archive sweep
(`find $ROOT/_archive -mtime +30 -delete`) — but only AFTER they've already
been preserved for the active-archive window.

### 3.4 Apply procedure (operator, not automated)

When an operator decides to enforce this policy:

1. Run with `--dry-run` first; tool prints exact files that would move.
2. Verify nothing in the "to move" list matches the active-state pattern.
3. Run with `--apply` to perform `mv` into `_archive/dynamic_caps_backups/`.
4. The disk-guard 30-day archive sweep then bounds the archive long-term.

Until that operator action is taken, the 3 currently-present forensic files
(~11 KB total) are explicitly grandfathered and remain in `runtime/`. Their
existence is bounded by **operator infrequency** — they are only ever
created by hand during incident response.

## 4. Test / dry-run guidance (when tool is added)

A future cleanup utility (deferred — not added in this box) must satisfy:

- **Default behavior is `--dry-run`**; `--apply` must be explicit.
- Active-file match → tool MUST refuse and exit non-zero (fail-safe).
- Tool must log the exact list of moved files to NDJSON for audit.
- Unit tests must cover:
  - Active file never selected for move/delete.
  - Newest N forensic backups preserved.
  - Old forensic backups selected.
  - Dry-run performs no filesystem mutation.
  - Apply mode moves only the selected files into a temp archive dir
    (test harness uses `tmp_path`, never `/home/ubuntu/chad_finale/runtime`).

## 5. Why no code change is required for Box-41 closure now

- Writer path is atomic; no code creates `dynamic_caps*` backups.
- Existing forensic backup count: **3** (manual, dated 2026-03-26 and
  2026-04-24); no new ones in 26 days.
- Total dynamic_caps disk usage: **29 KB** — three orders of magnitude
  below any operationally meaningful clutter threshold.
- `chad-disk-guard` already removes any `.tmp*` residue.
- This Pending Action documents the retention rule that prevents future
  unbounded growth and protects active state. The rule is enforceable by
  any future cleanup tool (to be added under a separate change ticket).

## 6. Sign-off checklist (operator)

- [ ] Verified active file pattern excludes operator-tag pattern.
- [ ] Reviewed and accepted "archive, do not delete" forensic policy.
- [ ] (Future) Approved deferred change ticket for cleanup utility.

No live trading authorization required. No restart required. No SQLite or
runtime JSON mutation required. This document is a policy artifact only.
