# dynamic_caps Legacy Slice + _archive Purgatory Disposition (W3B-12)

## Premise correction, on the record

The W3B brief said "legacy dynamic_caps archived". The 2026-07-22
investigation found **dynamic_caps is live, load-bearing infrastructure** —
`runtime/dynamic_caps.json` is rewritten every orchestrator cycle and read by
14+ live modules (kraken_executor, daily_throttle, profit_lock,
ibkr_execution_runner, health_monitor, feed_watchdog TTL 180s, dashboards,
state_bus, live_loop redis subscriber, ...). Only a narrow caller-less slice
was archivable, and one file that LOOKS dead is anything but:

**`runtime/dynamic_caps_quarantine.json` (mtime March) is read EVERY cycle**
by the live correlation overlay (`chad/risk/correlation_layer.py` `Q_PATH`;
`compute_overlay()` raises `"quarantine_weights missing"` if absent — today's
fresh `dynamic_caps_correlation.json` proves consumption). It stays exactly
where it is. `chad/risk/quarantine_layer.py` / `quarantine_strategy.py` are
caller-less BUT the only producer of that file — archiving them would make
the frozen March weights permanently unregenerable. **Kept**; flagged as a
refactor candidate (make the correlation overlay's quarantine input optional
or regenerate it), which is a behavior change needing its own PA.

## What W3B-12 archived (git-committed move, history preserved)

Four modules with ZERO importers (verified across chad/, ops/, backend/,
scripts/, tests, and all installed ExecStart lines) →
`_archive/bak_purge_20260722/chad/risk/`:

- dominance_caps_bridge.py
- dominance_risk_layer.py
- dominance_strategy.py
- risk_governed_strategy.py

(`dominance_allocator.py` STAYS — it is the side-engine track that holds
alpha_forex and is pinned by the W3B-9 perimeter tests.)

## The _archive purgatory disposition (the EXS6 haunting)

`/usr/local/bin/chad_disk_guard.sh` deletes `_archive/` files older than 30
days every 30 minutes — **`_archive/` is a purgatory, not storage**. It had
already disk-deleted 20 git-tracked files (bak_purge_20260506/,
bak_quarantine_20260402/), leaving 20 permanent ` D` entries in the live
tree's git status. The EXS6 dirty_git allowlist entry for `_archive/`
(config/exterminator.json) carried its own expiry note: *"REMOVE this entry
once the deletions are committed."*

W3B-12 completes that lifecycle:
1. the 20 deletions are committed (files remain in git history);
2. the allowlist entry is REMOVED per its expiry note. The allowlist was
   belt-and-braces: `_archive/` is outside EXS6's `production_paths` scope,
   so purged-archive deletions are production-filtered — they were never
   "drift", but the entry itself was standing documentation debt and 20
   permanent status lines of noise;
3. `test_archive_deletions_are_allowlisted` re-pinned as
   `test_archive_deletions_production_filtered_not_allowlisted`.

Lifecycle going forward (by CONVENTION, not paged — EXS6 deliberately does
not watch `_archive/`): archive by git-committed move → ~30 days later
disk-guard purges the disk copies → next hygiene pass commits the deletions.
History keeps the bytes either way. The four modules archived above will
themselves be disk-purged in ~August; commit those deletions then.

## Operator disposition list (runtime files — NOT touched by this lane)

Dead runtime artifacts an operator may delete at will (no readers verified
2026-07-22; deletion is runtime mutation and therefore not this lane's):

- `runtime/dynamic_caps_dominance_overlay.json` (Mar 26, no readers)
- `runtime/dynamic_caps_risk_governed.json` (Mar 26, no readers)
- `runtime/dominance_allocator.json` (Mar 26, no readers)
- `runtime/cleanup_runtime_artifacts.box043_dryrun.audit.ndjson` (11.2MB, static since May 20)
- `runtime/cleanup_telegram_dedupe.audit.ndjson` + `.box042_dryrun.audit.ndjson` (1.9MB, static May)

Do NOT delete: `dynamic_caps.json`, `dynamic_caps_correlation.json` (live,
rewritten every cycle), `dynamic_caps_quarantine.json` (live-read input, see
above).
