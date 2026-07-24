# PENDING ACTION — W6B-1: EXS7 enforcement-coverage adoption

**Status:** PREPARED, NOT APPLIED. Requires operator GO.
**Type:** config-only (`config/exterminator.json` → `schema_contracts.enforced`).
**Risk:** low, and measured — see the pre-flight below.
**Prepared:** 2026-07-24 (W6B-1)

---

## What this changes

Adds **59 already-pinned runtime contracts** to `schema_contracts.enforced`, taking EXS7
from **10 validated contracts to 69**.

No publisher is touched. No code path changes. The only behavioural effect is that EXS7
begins validating 59 contracts it currently ignores.

## Why

EXS7 today validates 10 contracts and warns about 5 unpinned ones. Meanwhile **64 runtime
files already carry a `schema_version` that nothing checks** — including load-bearing ones:
`regime_state.v1`, `strategy_health.v1`, `winner_scaling.v1`, `tier_state.v3`,
`portfolio_state.v1`, `pnl_state.v1`, `positions_truth.v1`, `governor_state.v1`,
`lifecycle_replay_state.v3`, `options_chain_cache.v2`, `futures_roll_state.v1`.

The publishers already did the work of declaring a contract. Nothing was reading it back.

## Pre-flight: this is inert on day one

```
$ python3 -m ops.exs7_adoption --runtime-dir /home/ubuntu/chad_finale/runtime --verify
eligible=59 breaks=0

Adoption is inert today: every eligible contract already holds.
```

Every one of the 59 was validated against the real check's logic before this PA was
written. **Applying it turns nothing red.** EXS7 stays WARN (from the unrelated
`unpinned_known` list) and gains 59 contracts under active validation.

That is the whole reason this ships as a reviewed config diff rather than a surprise: the
answer to "what does this break?" is measured, not asserted.

## What was excluded, and why

5 of the 64 are deliberately **not** adopted. The filter is not about freshness — EXS7
breaks on `missing | unreadable | version-absent/unrecognised | required_keys_missing`,
and **staleness is not one of them** (that is EXS1's job). A dormant but permanent file is
safe to enforce: `savage_alloc_state.json` has not been written in 136 days and would pass
every cycle.

The disqualifying property is instead: **can this file be legitimately ABSENT during
normal operation?**

| Class | File | Reason |
|---|---|---|
| dated_one_off | `quarantine_manifest_20260511.json` | written once for one incident; prunable, so absence is not a defect |
| dated_one_off | `quarantine_manifest_pff1_ghost_scrub.json` | same |
| event_conditional | `epoch_reset_state.json` | exists only after an epoch reset is executed |
| event_conditional | `stop_bus_recovery_state.json` | exists only after the stop bus trips and recovers — absent is the healthy state |
| event_conditional | `lifecycle_replay_drift_audit.json` | on-demand audit artifact, not a cadence publisher |
| publisher_not_yet_fired | `bars_refresh_state.json` | W6A landed the publisher; the nightly run has not fired, so the artifact does not exist yet |

The last one is worth calling out: it arrived from Lane A **one merge after** the filter
was specified, and is the exact case the filter exists for. Adopting it today would
manufacture a red sentinel for a publisher that is working correctly and simply has not run
yet. It becomes eligible automatically once the first nightly run lands — the exclusion is
conditional on absence, not permanent, and there is a named test for that.

## Why the contracts are minimal

Each adopted entry pins `accepts: [<version observed today>]` and
`required_keys: ["schema_version"]` — **nothing else**.

Inferring a richer key contract from a single observation is how an optional field becomes
a permanent false FAIL: the publisher omits it on some branch six weeks from now and the
sentinel goes red over a key that was never promised. The minimal contract still catches
every break mode that matters — the file vanishes, becomes corrupt, or silently changes
schema version.

Richer per-file key contracts are a follow-on for someone who has read the specific
publisher. The 10 existing hand-written entries already model that, and are untouched here.

---

## How to apply

The block is regenerated from the live tree rather than pasted from this document, so it
cannot drift from reality between preparation and application:

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Re-run the pre-flight. Do NOT proceed if this reports any break.
python3 -m ops.exs7_adoption --verify

# 2. Regenerate the block and merge it into config/exterminator.json
python3 -m ops.exs7_adoption --emit > /tmp/exs7_block.json
python3 - <<'EOF'
import json, pathlib
cfg_path = pathlib.Path("config/exterminator.json")
cfg = json.loads(cfg_path.read_text())
block = json.loads(pathlib.Path("/tmp/exs7_block.json").read_text())
enforced = cfg["schema_contracts"]["enforced"]
added = [k for k in block if k not in enforced]   # never overwrite a hand-written entry
enforced.update({k: block[k] for k in added})
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"added {len(added)} contracts; enforced is now {len(enforced)}")
EOF

# 3. Confirm EXS7 still holds
python3 -m pytest chad/tests/test_w6b_exs7_adoption.py -q
```

Note step 2 **never overwrites an existing key**. The 10 hand-written contracts carry
richer `required_keys` and consumer notes (e.g. the `position_guard_drift` entry documents
that `live_readiness_publish.py:195` accepts v1|v2|v3, so the check must too). Those must
survive.

## Rollback

Remove the added keys — every one is identifiable by
`"pinned_at": "W6B-1 liveness-filtered adoption (ops/exs7_adoption.py)"`. No publisher
changed, so there is nothing else to undo.

## Verification after apply

- `python3 -m ops.exs7_adoption --verify` → `breaks=0`
- Next sentinel cycle: EXS7 `evidence.enforced_contracts` goes 10 → 69,
  `evidence.break_count` stays 0, status stays WARN (driven by `unpinned_known`, which this
  PA does not touch).

## Evidence

| Claim | Source |
|---|---|
| 74 pinned / 10 enforced / 64 unvalidated | `python3 -m ops.exs7_adoption` |
| 59 eligible, 0 breaks | `python3 -m ops.exs7_adoption --verify` |
| Break vocabulary (staleness not among them) | `chad/ops/exterminator_sentinel.py::check_schema_breaks` (:1059-1151) |
| Exclusion rules and their reasons | `ops/exs7_adoption.py` — `DATED_ONE_OFF_PATTERNS`, `EVENT_CONDITIONAL`, `PUBLISHER_NOT_YET_FIRED` |
| bars_refresh_state publisher landed, artifact absent | `chad/market_data/nightly_bars_refresh.py:196,278`; file not present on live tree |
| Tests | `chad/tests/test_w6b_exs7_adoption.py` (20 tests) |
