# BOX-042 — Telegram dedupe retention policy (Pending Action)

- Status: **POLICY DOC + DRY-RUN-PROVEN TOOL READY — apply NOT executed**
- Owner: CHAD ops
- Authored: 2026-05-20
- Closes Box 42 (GAP-012 / NEW-GAP-049 — Telegram dedupe bounded)
- Live trading authorization required: **NO**
- Restart required: **NO**

---

## 1. Background

`chad/utils/telegram_notify.py` ships a small Telegram notifier with a
TTL-based dedupe mechanism. On every successful send for a given
`dedupe_key`, the notifier writes:

    runtime/telegram_dedupe_<sanitized_key>.json

with payload `{"last_sent_unix": <float>}`. The write is atomic:

```python
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps({"last_sent_unix": time.time()}, indent=2) + "\n", ...)
tmp.replace(p)
```

→ **There is exactly one file per unique `dedupe_key`**, and that file is
overwritten on every subsequent send. The notifier never creates a second
file, never creates a `.bak`/`.old`/`.archive` variant, and emits no
backups. The total file count is therefore equal to the **cardinality of
distinct dedupe_keys ever used**, not the number of alerts sent.

`_dedupe_allows()` (chad/utils/telegram_notify.py:127) treats a file whose
age `>= dedupe_ttl_s` as if it does not exist. Default TTL is **900s** via
`TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS`. Files older than that contribute
nothing to live suppression — they are **dead state**.

## 2. Growth source

Most call sites pass a **fixed-cardinality** dedupe_key (e.g.
`ibkr_down`, `stop_bus_triggered`, `daily_exec_report`, `salary_authorized`,
`weekly_investor_report`, `feed_stale_<rule>`, `health_claude_analysis`,
`scr_milestone_<STATE>`, `decision_trace_livegate_down`, `live_loop_sigterm`,
`live_loop_exception`, `reconciliation_red`, `ibkr_stale`, `ibkr_failures`,
`drawdown_threshold`, `edge_decay_<strategy>`, `trade_<sym>_<strat>_<side>`).

One caller, `chad/ops/health_monitor.py:311`, builds keys as:

```python
dedupe_key=f"health_{finding.rule_id}_{finding.title[:30]}"
```

The 30-char title prefix contains rule-specific variable values (e.g.,
the SCR-gap raw/effective vote counts, "967rawvs194effect",
"955rawvs194effect", …). Every distinct combination becomes a new dedupe
file. This is the sole source of file-count growth.

## 3. Current inventory (2026-05-20T11:22Z)

- `telegram_dedupe_files_found`: **true**
- `telegram_dedupe_total_files`: **1581**
- `telegram_dedupe_backup_files`: **0** (no `.bak`/`.old`/`.archive` files)
- `telegram_dedupe_total_bytes`: **67,569** (~66 KB)
- `oldest_dedupe_mtime_utc`: **2026-03-24T04:43:42Z**
  (`telegram_dedupe_ibkr_paper_ledger_watcher_error.json`)
- `newest_dedupe_mtime_utc`: **2026-05-20T11:10:46Z**
  (`telegram_dedupe_feed_stale_regime_state.json`)
- Active files (within TTL × safety_multiplier = 3600s of `now`): **10**
- Stale files (mtime older than 3600s): **1571**

Each file is ~43 bytes. Disk pressure is negligible; the concern is the
unbounded file count over months/years.

## 4. Retention policy

### 4.1 Pattern definitions

Telegram dedupe sentinel pattern (regex on basename):

    ^telegram_dedupe_[A-Za-z0-9._-]+\.json$

Excluded suffixes (never in scope):

- `.tmp` and `.tmp.<pid>` — atomic-write residue, already bounded by
  `chad-disk-guard`'s `find $RUNTIME -maxdepth 2 -name "*.tmp*" -delete`.
- Any other compound suffix.

### 4.2 Retention rule

1. **Active state** — files whose mtime is within
   `TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS × safety_multiplier` seconds of
   now. With default TTL=900s and safety_multiplier=4 this is the
   last 3600s (one hour). Active files are **NEVER deleted, NEVER
   archived** by any tool.
2. **Stale state** — files older than the safety window. These contribute
   nothing to live suppression. They are **archived** to
   `_archive/telegram_dedupe/YYYY/MM/<original_basename>` (where `YYYY`
   and `MM` come from the file's mtime). The existing
   `chad-disk-guard.sh` rolling-30-day archive sweep then provides the
   long-term bound on `_archive/`.
3. **Archive over delete.** The default mode of the cleanup tool is
   `archive`, not `delete`. The `--delete-instead-of-archive` flag is
   available but discouraged.

### 4.3 Why safety_multiplier=4

A single TTL of headroom is conservative for normal operation. The 4×
multiplier (one hour at default TTL) hedges against:
- Clock skew between writer/reader.
- TTL having been raised in a service-unit env override.
- Active alerts that have just barely crossed the TTL boundary at the
  exact moment the cleanup runs.

A user can override with `--safety-multiplier N` (N ≥ 1.0). The tool
**refuses** to operate with N < 1.0.

## 5. Cleanup utility

**`ops/cleanup_telegram_dedupe.py`** implements the policy:

- Default mode: **dry-run**. `--apply` required to perform any
  filesystem mutation.
- Honors `TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS` from env unless
  `--ttl-seconds N` is passed.
- Writes an NDJSON audit log of every move/delete (or, in dry-run, every
  intended action) to `runtime/cleanup_telegram_dedupe.audit.ndjson`
  (or a caller-specified path).
- Refuses to operate when `safety_multiplier < 1.0` or `ttl_seconds <= 0`.
- Only files matching `runtime/telegram_dedupe_*.json` are even
  considered; `.tmp` sidecars and unrelated runtime artifacts are
  outside scope by construction.

### 5.1 Tests proving safety

`chad/tests/test_box042_telegram_dedupe_cleanup.py` — 9 tests, all
passing:

| test | invariant proved |
|------|------------------|
| `test_active_dedupe_state_is_never_touched_dry_run` | active mtime files are preserved in dry-run |
| `test_active_dedupe_state_is_never_touched_apply` | active mtime files are preserved even with `--apply` |
| `test_stale_files_are_archived_to_year_month_tree` | stale files move to `_archive/telegram_dedupe/YYYY/MM/` |
| `test_dry_run_performs_no_mutation_but_logs_targets` | dry-run writes audit NDJSON but does NOT touch files |
| `test_apply_only_touches_telegram_dedupe_files` | unrelated runtime JSON / `.tmp` / non-dedupe files are not in scope |
| `test_delete_mode_unlinks_instead_of_archiving` | delete mode unlinks files but never creates archive tree |
| `test_safety_multiplier_below_one_is_refused` | safety_multiplier < 1 raises ValueError |
| `test_zero_or_negative_ttl_is_refused` | ttl_seconds <= 0 raises ValueError |
| `test_empty_runtime_dir_is_handled` | clean exit on empty dir |

### 5.2 Live dry-run (2026-05-20T11:22:58Z, NO mutation performed)

    $ python3 ops/cleanup_telegram_dedupe.py --json-summary \
        --audit-log runtime/cleanup_telegram_dedupe.box042_dryrun.audit.ndjson
    {
      "mode": "DRY-RUN",
      "runtime_dir": "/home/ubuntu/chad_finale/runtime",
      "archive_dir": "/home/ubuntu/chad_finale/_archive/telegram_dedupe",
      "ttl_seconds": 900,
      "safety_multiplier": 4.0,
      "safety_window_seconds": 3600.0,
      "scanned": 1581,
      "active_preserved": 10,
      "stale_in_scope": 1571,
      "archived": 0,
      "deleted": 0,
      "skipped": 0,
      "delete_instead_of_archive": false
    }

Pre-run vs. post-run file count: **1581 → 1581** (zero mutation).

The 10 files protected as "active" included `telegram_dedupe_stop_bus_triggered.json`
(167s old), `telegram_dedupe_feed_stale_regime_state.json` (744s old),
and 8 `health_R*_*` rule sentinels (203-206s old). All represent
suppression windows that are still live and must not be disturbed.

## 6. Operator runbook (NOT EXECUTED HERE)

When an operator decides to apply, the recommended sequence is:

```bash
# 1) Always dry-run first; review the count and the audit log
python3 ops/cleanup_telegram_dedupe.py --json-summary

# 2) Apply (archive to _archive/telegram_dedupe/YYYY/MM/)
python3 ops/cleanup_telegram_dedupe.py --apply --json-summary

# 3) Verify only stale files moved
ls -la _archive/telegram_dedupe/2026/*/ | tail
ls runtime/telegram_dedupe_*.json | wc -l   # ≈ active count
```

The `_archive/` 30-day rolling sweep in `chad_disk_guard.sh` provides the
long-term bound on archived sentinels.

## 7. Why a policy + tool is the right closure

- Disk pressure today is negligible (~66 KB), so closure does not require
  immediate live cleanup.
- The unbounded-growth risk is real (rule-title cardinality is potentially
  unbounded) and the tool plus retention rule **prevents** unbounded
  growth in a deterministic, replayable way.
- Active dedupe state is provably protected by:
  1. Pattern-based scope limit (only `telegram_dedupe_*.json`).
  2. mtime-based safety window with 4× TTL buffer.
  3. Default dry-run mode.
  4. Unit tests against the exact patterns used in production.
- Optional: an operator may schedule the tool from cron/systemd-timer
  with `--apply` once they accept the policy. That scheduling decision
  is intentionally **deferred** (it is a runtime-mutation action) and
  is out of scope for Box-042 closure.

---

## 8. Box-042 closure rationale

Box-042 ("Telegram dedupe bounded") is closed by this Pending Action plus
the BOX-042 evidence file. The four acceptable closure paths from the
task spec are satisfied as follows:

| acceptable closure | satisfied? | how |
|--------------------|------------|-----|
| dedupe file count already bounded by existing policy | n/a | not yet — cardinality grows with rule-title variance |
| cleanup/retention policy exists | **YES** | this file |
| cleanup/archive tool exists with dry-run proof | **YES** | `ops/cleanup_telegram_dedupe.py` + §5.2 |
| code/docs/tests added to prevent unbounded growth | **YES** | `chad/tests/test_box042_telegram_dedupe_cleanup.py` + this doc |

No live runtime JSON was mutated. No service was restarted. No SQLite was
mutated. No order/broker action was taken. No live trading authorization
was granted or implied.
