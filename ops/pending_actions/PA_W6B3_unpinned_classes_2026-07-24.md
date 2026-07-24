# PENDING ACTION — W6B-3: declare unpinned exception classes

**Status:** PREPARED, NOT APPLIED. Requires operator GO.
**Type:** config-only (`config/exterminator.json` → `schema_contracts.unpinned_classes`).
**Risk:** none — the field is additive and inert until declared.
**Prepared:** 2026-07-24 (W6B-3)

---

## The premise correction this rests on

The plan said "expand `unpinned_known` to cover all remaining genuinely-unpinned runtime
files", estimating 57 undocumented. **The real number is 122**, plus 4 files that are not
parseable as JSON at all:

```
runtime/*.json                     205
  pinned (schema_version present)   74
  unpinned                         127   <- plan said 62
  unreadable                         4   <- plan did not report this class
```

**65 of the 127 are `telegram_dedupe_*.json`** — per-alert markers whose filenames are
generated from alert text, e.g.
`telegram_dedupe_health_R13_SCRgap214rawvs67effecti.json`. They accreted 2026-05-29 →
2026-07-22, so the plan's 62 was a miscount rather than drift. (For scale: the June defense
board recorded 29 of these and called it "stale — optional tidy". It has more than doubled.)

Enumerating that namespace is not viable. It is **unbounded** — one new file per new alert
string — so a path list would be stale on arrival and need a commit per alert.

## What changes

`unpinned_classes` declares coverage by **glob pattern with a stated reason**, and the WARN
is driven by what remains **undocumented**.

The old rule was `if unpinned_known: WARN` — permanent by construction, since that dict
never shrinks on its own. A permanent warn trains operators to skim past it, which is the
same failure mode as P2-3's per-cycle log noise.

The new rule fires on files that are unpinned **and** unclassified: a genuinely new
artifact nobody has ruled on. That is the actionable event.

## The classes to declare

```json
"unpinned_classes": {
  "telegram_dedupe_*.json": {
    "kind": "ephemeral_alert_marker",
    "reason": "Per-alert dedupe markers written by the Telegram send path. Filenames are generated from alert text, so the namespace is unbounded and cannot be enumerated. They carry no consumer contract -- existence and mtime are the entire payload. Retention is governed by BOX-042."
  },
  "broker_truth_snapshot_*.json": {
    "kind": "dated_forensic_snapshot",
    "reason": "Point-in-time broker snapshots captured during an incident (2026-04-19/20). Historical evidence, never re-read by a live consumer, safe to archive."
  },
  "quarantine_manifest_*.json": {
    "kind": "incident_scoped_manifest",
    "reason": "Written once per quarantine incident. Read by the exclusion chain by explicit path, not discovered by scan; a schema pin would add no validation the readers do not already do."
  }
}
```

## What is deliberately LEFT undocumented

This is the honest part, and it is the reason this PA does not clear EXS7 to green.

After the three classes above, **~54 files remain undocumented**, and they are left that
way on purpose. Documenting them would clear the warn while changing nothing — the warn is
supposed to be pointing at them.

They fall into three actionable groups:

1. **Genuine unpinned publishers (~50)** — real, load-bearing runtime state with no
   `schema_version`: `dynamic_caps.json`, `feed_state.json`, `expectancy_state.json`,
   `operator_intent.json`, `paper_exec_state.json`, `kraken_prices.json`,
   `signal_guard.json`, `tier_enforcement_state.json`, and others. **The fix for these is
   to pin them**, publisher by publisher, the way W6B-2 pinned three. That is wave work,
   not a config declaration.

2. **Test-probe leakage (2)** — `__guard_probe_should_not_exist__.json` and
   `__guard_swallow_probe__.json` (1 byte, contents `x`) leaked into live `runtime/`. The
   fix is deletion, not documentation. This lane does not mutate runtime files, so they are
   flagged and left.

3. **Unreadable (4)** — reported as their own class:

   | File | Defect |
   |---|---|
   | `flip_executor_audit.json` | **NDJSON under a `.json` name** — `json.load()` raises `Extra data: line 2`. 6765 bytes, rows from 2026-05-04 |
   | `signal_throttle_audit.json` | same defect, 752 bytes, rows from 2026-05-03 |
   | `claude_usage.json` | zero bytes |
   | `__guard_swallow_probe__.json` | 1 byte |

   The first two are the interesting pair: any consumer calling `json.load()` on them
   fails. They were **completely invisible** to EXS7 before this change — not enforced so
   never opened, not schema-bearing so never counted.

**Expected post-apply state:** EXS7 stays WARN, reporting roughly *"54 unpinned and
unclassified, 4 unreadable"* instead of a curated 5. The number shrinks as publishers get
pinned, so the warn becomes a progress bar rather than a permanent fixture.

## How to apply

Add the `unpinned_classes` block to `config/exterminator.json` under `schema_contracts`.
No other change; the field is read defensively (absent → `{}` → today's behaviour).

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m pytest chad/tests/test_w6b_unpinned_coverage.py chad/tests/test_exterminator_sentinel.py -q
```

## Rollback

Remove the `unpinned_classes` key. The classifier then documents nothing, and the warn
reports the full 127 as undocumented — noisier than today, but not wrong.

## Evidence

| Claim | Source |
|---|---|
| 205 scanned / 74 pinned / 127 unpinned / 4 unreadable | full enumeration of live `runtime/*.json`, 2026-07-24 |
| 65 dedupe markers, dates 05-29 → 07-22 | `ls -la runtime/telegram_dedupe_*.json` |
| June recorded 29 of them | `docs/DEFENSE_BOARD_RECONCILIATION_2026-06-17.md` §P3-1 |
| Unbounded naming | `telegram_dedupe_health_R13_SCRgap214rawvs67effecti.json` — name derived from alert text |
| Classifier behaviour | `chad/ops/exterminator_sentinel.py::_classify_unpinned` |
| Tests | `chad/tests/test_w6b_unpinned_coverage.py` (16 tests) |
