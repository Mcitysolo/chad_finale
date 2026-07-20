# PENDING ACTION — ibkr_exec_state reaper purge authorization (W1A-6)

- **Filed:** 2026-07-18
- **Type:** Guarded maintenance tool. Tool BUILT + tested; runs read-only by default.
  **`--execute` NOT run against runtime; the destructive purge awaits typed operator GO.**
- **Author:** W1A (Wave-1 observability)
- **Status:** PROPOSED. Detection is safe to run now; `--execute` awaits this PA's GO.
- **Governance:** dry-run default · typed confirm token `REAP-IBKR-EXEC-STATE` ·
  archive-before-mutate (`.bak_reap_<UTC>` + sha256) · fail-closed gates
  (exec_mode paper/dry_run · SCR ∈ {CONFIDENT,CAUTIOUS,WARMUP} · reconciliation not RED) ·
  hard-targets `ibkr_adapter_state.sqlite3::ibkr_exec_state` and refuses any other DB · NO broker I/O.

---

## 1. What the tool is

`scripts/reap_ibkr_exec_state.py` — a status-aware reaper for CHAD's own IBKR
idempotency/dedup ledger `runtime/ibkr_adapter_state.sqlite3` (table
`ibkr_exec_state`, `chad/execution/ibkr_adapter.py`). That store is CHAD's
last-observed order snapshot — the duplicate-submission guard — **not** live
broker truth (live order truth comes from `reqAllOpenOrdersAsync()` separately).

**Root cause it addresses (P1-8 / GAP-036, `docs/CHAD_GAPS_TO_CLOSE.md:82`):** the
adapter only cleans a key lazily, when the *same* key re-submits
(`claim_or_reclaim`). An orphaned non-terminal key that is never re-submitted
lingers forever, accumulating stale `PendingSubmit`/`PreSubmitted`/`Submitted`/
`claimed`/`duplicate_open_order`/`ValidationError` rows.

## 2. Why a NEW tool and not the incumbent

The incumbent `ops/sqlite_retention.py` (wired to the live weekly
`chad-sqlite-retention.timer`) already prunes this table — but it is the OPPOSITE
of what governance wants:

| Property | Incumbent `sqlite_retention.py` | This reaper |
|---|---|---|
| Selection | **age-only** (`DELETE … WHERE updated_at_utc < cutoff`) | **status-aware** (stale AND non-terminal) |
| Terminal evidence | **deleted** (`Filled`/`Cancelled` fall in the age window) | **preserved** |
| Gates | none (no exec_mode/SCR/reconciliation check) | fail-closed, 3 gates |
| Archive | none | `.bak_reap_<UTC>` + sha256 before any delete |
| Default | real-delete (dry-run opt-in) | **read-only detection** (`--execute` opt-in) |
| Schedule | live weekly timer | manual, operator-run |

Per GO decision **D3**, this wave ships the new sibling reaper and **does not
touch** `sqlite_retention.py` or its live timer (changing a live-scheduled unit's
semantics is out of the surgical lane). Hardening the incumbent is a follow-up —
see §5.

## 3. Detection (safe to run now, read-only)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 scripts/reap_ibkr_exec_state.py            # default: read-only detection over live DB
# optional: widen/narrow the age margin (default 30 days)
python3 scripts/reap_ibkr_exec_state.py --older-than-days 30
```

Detection opens the DB **read-only** (`file:…?mode=ro`), classifies every row, and
prints: total rows, counts by class, the stale non-terminal delete-candidates,
and the retain set — mutating nothing. It also prints the **incumbent-diff** (§4).

## 4. The incumbent-diff (evidence for §5)

The detection report includes, for the same age cutoff:

- `incumbent_age_only_would_delete_count` — every row older than the cutoff, the
  set the age-only reaper deletes **status-blind**.
- `preserved_by_status_awareness_count` + `preserved_by_status_awareness[]` — the
  subset of those old rows that are **terminal** (`Filled`/`Cancelled`/…): order
  evidence the incumbent destroys but status-aware logic keeps.

That second list is the concrete, live evidence for the harden-the-incumbent PA
below: it names exactly which terminal execution records the weekly age-only
timer is quietly erasing.

## 5. Follow-up PA (proposed, not this wave) — harden the incumbent

Harden `ops/sqlite_retention.py` in place so the live weekly timer stops deleting
terminal evidence and gains the same governance:

1. Make it **status-aware** — exclude terminal `Filled`/`Cancelled`/… from the
   age-based `DELETE` (or restrict deletion to non-terminal rows), so the weekly
   run cannot erase execution evidence.
2. Add the **fail-closed gates** (exec_mode/SCR/reconciliation) and
   **archive-before-mutate**, matching this reaper.
3. Make **dry-run the default** (real-delete opt-in), and cite the incumbent-diff
   from §4 as the justification.

Because it edits a live-scheduled unit's behavior, that change is its own gated
PA (baseline → change → verify), not a Wave-1 item.

## 6. Purge — operator, gated (requires explicit GO)

**Only run this after the standard pre-live checklist and reading a fresh
detection report.** The one real hazard is dropping the dedup row of an order that
is **still working at the broker**, which would permit a duplicate submission; the
conservative age margin (default 30d — well past any working order) is the Wave-1
mitigation. A stronger `--broker-probe` cross-check (keep only broker-`ABSENT`
keys) is a noted follow-up that needs a live connection and is **not** built here.

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Read a fresh detection report first.
python3 scripts/reap_ibkr_exec_state.py --older-than-days 30

# 2. Purge — typed token REQUIRED; refuses (no write) if any gate fails.
python3 scripts/reap_ibkr_exec_state.py --older-than-days 30 \
    --execute --confirm REAP-IBKR-EXEC-STATE

# 3. The run archives the DB to runtime/ibkr_adapter_state.sqlite3.bak_reap_<UTC>
#    (sha256 logged) before any DELETE, and prints rows_before/after/deleted.
```

Refusal exit codes: `2` (missing/wrong `--confirm`), `3` (bad `--older-than-days`),
`4` (wrong/absent DB — including the forbidden wrong-stores
`exec_state_paper.sqlite3` and `ibkr_exec_state.db`), `5` (a gate failed).

## 7. Rollback

The pre-purge archive is a full copy of the DB. To restore:

```bash
sudo systemctl stop chad-orchestrator     # only if the adapter has the DB open
cp runtime/ibkr_adapter_state.sqlite3.bak_reap_<UTC> runtime/ibkr_adapter_state.sqlite3
```

Deleted rows are only stale **non-terminal** dedup entries; restoring the archive
reinstates them exactly.

## 8. Scope / non-goals

- No `systemctl`; no change to `sqlite_retention.py` or its timer this wave.
- No broker I/O; Wave-1 purge is age-margin only.
- `--execute` is **never** run against real runtime in build/test — every test
  uses a `tmp_path` sqlite (`chad/tests/test_w1a_reap_ibkr_exec_state.py`).
- Hard-targets `ibkr_adapter_state.sqlite3::ibkr_exec_state`; refuses the Kraken
  trusted-lot store `exec_state_paper.sqlite3` and the dead `ibkr_exec_state.db`.
