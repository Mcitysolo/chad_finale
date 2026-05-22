# BOX-043 — Runtime cleanup / logrotate retention policy (Pending Action)

- Status: **POLICY DOC + DRY-RUN-PROVEN TOOL READY — apply NOT executed**
- Owner: CHAD ops
- Authored: 2026-05-20
- Closes Box 43 (Logrotate / runtime cleanup coverage complete)
- Live trading authorization required: **NO**
- Restart required: **NO**

---

## 1. Purpose

Provide a single authoritative coverage matrix for every growing
runtime / log / data / report / backup artifact family in the repo, and
fill the remaining unbounded-growth gaps with a safe, tested,
dry-run-first cleanup utility (`ops/cleanup_runtime_artifacts.py`).

Box 041 closed the `dynamic_caps*` operator backups by policy.
Box 042 closed `runtime/telegram_dedupe_*.json` with a cleanup tool +
policy. This box covers everything else.

## 2. Coverage matrix

Legend for "verdict" column:
- **COVERED** — already bounded by an existing system tool (logrotate,
  chad-disk-guard, sqlite_retention, telegram_dedupe cleanup, dynamic_caps
  policy).
- **INTENTIONAL RETAIN** — audit-critical or operator-forensic artifact
  that must NOT be auto-deleted. Policy is "retain; never auto-prune".
- **NEW COVERAGE (this box)** — bounded by
  `ops/cleanup_runtime_artifacts.py` and the rules below.

| # | Category | Files | Size | Growth | Coverage | Verdict |
|---|---------|------:|-----:|--------|----------|---------|
| 1 | `runtime/*.tmp`, `runtime/*.tmp.*` | n/a | n/a | per-cycle | `chad_disk_guard.sh` | COVERED |
| 2 | `data/feeds/*` | varies | 8 KB | per-day | `chad_disk_guard.sh` (3-day) | COVERED |
| 3 | `data/feeds/polygon_stocks/POLYGON_STOCKS_*.ndjson` | varies | varies | per-day | `/etc/logrotate.d/chad-feeds` (5-day, maxsize 200M) | COVERED |
| 4 | `logs/backend-uvicorn.log` | 1 + rotated | <1 MB | append | `/etc/logrotate.d/chad-backend` (14-day) | COVERED |
| 5 | `logs/polygon_stocks.log` + 14 rotated `.gz` | 15 | <100 KB | append | `/etc/logrotate.d/chad-polygon` (14-day) | COVERED |
| 6 | `runtime/ibkr_adapter_state.sqlite3` rows (`ibkr_exec_state`) | n/a | 156 KB | live | `ops/sqlite_retention.py` (default 14-day) | COVERED |
| 7 | `runtime/telegram_dedupe_*.json` | 1581 | 66 KB | per-key | `ops/cleanup_telegram_dedupe.py` + Box 042 policy | COVERED |
| 8 | `_archive/*` | varies | 4.5 MB | varies | `chad_disk_guard.sh` (30-day) | COVERED |
| 9 | `runtime/dynamic_caps*` operator backups (`.pre_fix_*`, `.post_fix_*`, `.stale_*`) | 3 | 11 KB | manual | Box 041 policy | INTENTIONAL RETAIN |
| 10 | `runtime/proofs/BURNIN_CHECK_*`, `RUNTIME_REBUILD_PROOF_*`, etc. | 14,688 | 60 MB | 1 per 10 min | **`ops/cleanup_runtime_artifacts.py` (30-day archive)** | NEW COVERAGE |
| 11 | `data/config_snapshots/snapshot_*.json` | 65,323 | 334 MB | 1 per min | **`ops/cleanup_runtime_artifacts.py` (30-day archive)** | NEW COVERAGE |
| 12 | `backups/chad_backup_*.tar.gz` + `.sha256` + `.manifest.json` | 30 | 4.6 GB | 1 per day | **`ops/cleanup_runtime_artifacts.py` (keep latest 14)** | NEW COVERAGE |
| 13 | `logs/claude/calls_*.ndjson` | 47 daily NDJSON | 15 MB | 1 per day | **`ops/cleanup_runtime_artifacts.py` (30-day gzip-in-place)** | NEW COVERAGE |
| 14 | `data/fills/*` | 65 | 23 MB | per-fill | audit-critical | INTENTIONAL RETAIN |
| 15 | `data/trades/*` | 74 | 13 MB | per-trade | audit-critical | INTENTIONAL RETAIN |
| 16 | `data/broker_events/*` | 91 | 11 MB | per-event | audit-critical | INTENTIONAL RETAIN |
| 17 | `data/fees/*` | 52 | 16 MB | per-fee | audit-critical | INTENTIONAL RETAIN |
| 18 | `data/traces/decision_trace_YYYYMMDD.ndjson` | 93 | 239 MB | 1 per day | audit-critical (operator forensic) | INTENTIONAL RETAIN |
| 19 | `runtime/completion_matrix_evidence/*.md` | 58 | 964 KB | per-box | audit-critical | INTENTIONAL RETAIN |
| 20 | `runtime/intel_cache/*`, `runtime/sec_13f_cache/*` | 19 | 26 MB | overwrite-in-place | live cache (each filename overwrites) | INTENTIONAL RETAIN |
| 21 | `runtime/*.bak`, `*.pre_*` operator forensic backups (ibkr_adapter_state, trade_closer_state, position_guard) | ~15 | ~27 MB | manual | inherits Box 041 forensic-backup policy | INTENTIONAL RETAIN |
| 22 | `runtime/disk_guard_audit.ndjson` | 1 | 918 KB | append | logrotate stanza staged below (size-based rotation) | INTENTIONAL RETAIN (rotation Channel 1) |
| 23 | `reports/*` | thousands | 213 MB | varies | dated reports; operator forensic | INTENTIONAL RETAIN |

Result: **23 artifact categories total. 8 already COVERED, 11
INTENTIONALLY RETAINED, 4 newly covered by this box's tool. Zero
remaining unbounded-growth gaps.**

## 3. New coverage delivered (this box)

### 3.1 Tool: `ops/cleanup_runtime_artifacts.py`

- Default mode: **dry-run**.
- `--apply` required for any mutation.
- Per-category retention:

| category | rule | action |
|----------|------|--------|
| `runtime_proofs` | mtime newer than `--keep-proof-days` (default 30) | archive to `_archive/runtime_proofs/YYYY/MM/` |
| `config_snapshots` | mtime newer than `--keep-snapshot-days` (default 30) | archive to `_archive/config_snapshots/YYYY/MM/` |
| `backups` | keep latest `--keep-backups` (default 14) `chad_backup_*.tar.gz` triplets (primary + `.sha256` + `.manifest.json`) | delete older triplets |
| `claude_logs` | mtime newer than `--keep-claude-log-days` (default 30) | gzip in place to `calls_YYYYMMDD.ndjson.gz` |

- Strict regex scoping per category — files that don't match the
  category's `filename_re` are never touched.
- Audit-critical roots (`data/fills`, `data/trades`, `data/fees`,
  `data/broker_events`, `data/traces`,
  `runtime/completion_matrix_evidence`) are listed as
  `_FORBIDDEN_ROOTS`. The factory `default_categories()` refuses to
  produce a Category whose source resolves under any of those roots.
- Non-positive retention values are refused at construction time.
- NDJSON audit log of every action (real or dry-run) at
  `runtime/cleanup_runtime_artifacts.audit.ndjson` (overridable).

### 3.2 Tests proving safety

`chad/tests/test_box043_runtime_artifacts_cleanup.py` — 12 tests, all
passing:

| test | invariant |
|------|-----------|
| `test_runtime_proofs_active_files_preserved_dry_run` | fresh proofs survive dry-run |
| `test_runtime_proofs_stale_files_archived_to_year_month` | 60-day-old proof moves to `_archive/runtime_proofs/YYYY/MM/` |
| `test_runtime_proofs_unrelated_files_not_touched` | filename-regex scoping holds |
| `test_config_snapshots_active_preserved` | fresh snapshot is preserved |
| `test_config_snapshots_stale_archived` | 45-day-old snapshot archives |
| `test_backups_keep_latest_k_with_sidecars` | keep-latest 2 of 5 backups; older triplets including sidecars deleted |
| `test_backups_dry_run_preserves_everything` | dry-run leaves all 3 backups in place; audit log records 2 stale candidates |
| `test_claude_logs_old_files_gzipped_in_place` | 35-day-old `calls_*.ndjson` gzipped; `.gz` content is verified to round-trip |
| `test_default_categories_never_point_at_audit_critical_roots` | factory hard-fails on forbidden roots |
| `test_non_positive_retention_is_refused` | bad config rejected |
| `test_empty_source_dir_is_handled` | clean exit on missing/empty dir |
| `test_run_cleanup_end_to_end_empty` | all 4 categories run cleanly together |

### 3.3 Live dry-run summary (2026-05-20T11:25:48Z, no mutation)

```
$ python3 ops/cleanup_runtime_artifacts.py --json-summary \
    --audit-log runtime/cleanup_runtime_artifacts.box043_dryrun.audit.ndjson
{
  "mode": "DRY-RUN",
  "categories": [
    {"name": "runtime_proofs",   "scanned": 13137, "active_preserved":  4306, "stale_in_scope":  8831, ...},
    {"name": "config_snapshots", "scanned": 65333, "active_preserved": 41618, "stale_in_scope": 23715, ...},
    {"name": "backups",          "scanned":    10, "active_preserved":    10, "stale_in_scope":     0, ...},
    {"name": "claude_logs",      "scanned":    47, "active_preserved":    31, "stale_in_scope":    16, ...}
  ]
}
```

Pre-run vs. post-run file counts (live):

|              | before | after |
|--------------|-------:|------:|
| runtime/proofs/                              | 14689 | 14689 |
| data/config_snapshots/                       | 65333 | 65333 |
| backups/chad_backup_*.tar.gz                 |    10 |    10 |
| logs/claude/calls_*.ndjson                   |    47 |    47 |

(Run timestamp differs slightly from BOX-042 inventory because a new
proof and a new snapshot fire each cycle. The dry-run made zero file
mutations across all 4 categories.)

Notes on the dry-run numbers:
- `backups`: only 10 of the 30 visible files are primaries
  (`chad_backup_*.tar.gz`); the other 20 are `.sha256` and
  `.manifest.json` sidecars. With `--keep-backups 14` and 10 primaries
  present, **nothing** is stale yet — the rule activates once 15+
  primaries accumulate.
- `runtime_proofs` and `config_snapshots` both show large stale counts
  because >30-day-old files exist. The dry-run did not move any of
  them; the operator triggers `--apply` when ready.

## 4. Channel 1 / logrotate addendum (staged, NOT installed)

`runtime/disk_guard_audit.ndjson` is an append-only NDJSON file written
by `chad-disk-guard.timer` (every 30 min). It is currently ~1 MB and
grows slowly, but should be rotated for long-term safety. The
following logrotate stanza is staged here for an operator to install
into `/etc/logrotate.d/chad-runtime-ndjson` (Channel 1; daemon-reload
deferred and outside this box's scope):

```
/home/ubuntu/chad_finale/runtime/disk_guard_audit.ndjson {
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su ubuntu ubuntu
}

/home/ubuntu/chad_finale/runtime/cleanup_telegram_dedupe.audit.ndjson
/home/ubuntu/chad_finale/runtime/cleanup_runtime_artifacts.audit.ndjson
{
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su ubuntu ubuntu
}
```

This addendum is **policy only**; nothing under `/etc` was touched and
no daemon-reload was performed.

## 5. Operator runbook

```bash
# 1) Always dry-run first; review per-category numbers
python3 ops/cleanup_runtime_artifacts.py --json-summary

# 2) Apply (archives proofs+snapshots; deletes old backup triplets; gzips old claude logs)
python3 ops/cleanup_runtime_artifacts.py --apply --json-summary

# 3) Run a single category (e.g. proofs only)
python3 ops/cleanup_runtime_artifacts.py --apply --categories runtime_proofs

# 4) Tighter retention (e.g. keep last 14 days of proofs)
python3 ops/cleanup_runtime_artifacts.py --apply --keep-proof-days 14
```

After `--apply`, the existing `chad-disk-guard.sh` 30-day rolling sweep
of `_archive/` bounds the archived data long-term. No service restart,
no daemon-reload, no SQLite mutation is involved at any stage.

## 6. What is NOT in scope

- **Live cleanup application.** Box 43 closes on policy + tool +
  tests + dry-run proof. No `--apply` was executed.
- **Installing the logrotate stanza** into `/etc/logrotate.d/`. The
  stanza is staged in §4 for an operator to install at their
  discretion (Channel 1 action).
- **Scheduling the cleanup tool via cron / systemd timer.** That is a
  follow-up operator decision.
- **`data/traces` and `reports/*`** auto-pruning. These are operator
  forensic / audit-evidence categories and remain INTENTIONAL RETAIN.
  Manual pruning by an operator is allowed; automated deletion is not.

## 7. Box 43 closure rationale

| acceptable closure | satisfied? | how |
|--------------------|------------|-----|
| every growing artifact category covered or intentionally retained | **YES** | §2 matrix; 23/23 categories classified, 0 uncovered |
| cleanup tool exists with dry-run proof | **YES** | `ops/cleanup_runtime_artifacts.py` + §3.3 |
| safe retention policy exists | **YES** | this file |
| tests proving safety | **YES** | `chad/tests/test_box043_runtime_artifacts_cleanup.py` (12/12 pass) |

No live runtime JSON was mutated. No service was restarted. No SQLite
was mutated. No order/broker action was taken. No live trading
authorization was granted or implied.
