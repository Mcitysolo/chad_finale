# PENDING ACTION — W6B-2: envelope pins for the two high-risk unpinned contracts

**Status:** PREPARED, NOT APPLIED. Requires operator GO.
**Type:** config-only (`config/exterminator.json` → `schema_contracts.enforced`).
**Risk:** low. **No publisher is touched** — that is the design, not a happy accident.
**Prepared:** 2026-07-24 (W6B-2, resolving decision point D1)

---

## Background

`unpinned_known` lists 5 runtime contracts that carry no `schema_version`. Three are
low-risk and were pinned directly in W6B-2's code change (their publishers now emit
`positions_snapshot.v1`, `reconciliation_state.v1`, `profit_lock_state.v1`).

The other two are **not** safe to pin that way, and this PA covers them:

- `runtime/position_guard.json` — its top-level **key space is position identity**
  (`"<strategy>|<SYMBOL>"`). Adding a `schema_version` key inserts a pseudo-position into
  a namespace that readers iterate.
- `runtime/trade_closer_state.json` — hot-path FIFO state (`queues`, `processed_fill_ids`).

## The D1 ruling and what it bought

> pin the ENVELOPE — top-level contract + per-entry value shape — never enumerate
> identity keys; keyspace stays free.

Implemented as a new optional `envelope` block in an enforced contract, plus one change to
`check_schema_breaks`: **`schema_version` validation is now opt-in.** A contract declaring
neither `schema_version` nor `accepts` is validated on its envelope alone.

The consequence is the important part: **`position_guard.json` is never modified.** No new
key enters the identity namespace, so the reader hazard the plan flagged simply does not
arise — there is nothing to prove safe, because nothing changed on the write side. The
plan's fallback option (a) was "document as a justified exception, do not pin"; the
envelope gets real enforcement at that same zero risk.

## What the envelope asserts

Deliberately **mirrors the writer's own already-declared invariant**, at
`chad/core/position_guard.py::_validate_position_guard_schema`:

> "Entries that claim to be open must carry the minimum identification the rest of CHAD
> relies on. **Closed entries can be sparse.**"

```json
"runtime/position_guard.json": {
  "envelope": {
    "required_keys": ["_version", "_written_by"],
    "reserved_key_prefix": "_",
    "entry_value_type": "object",
    "entry_required_keys_when": {
      "field": "open", "equals": true,
      "required_keys": ["strategy", "symbol", "side"]
    }
  },
  "pinned_at": "W6B-2 (D1) envelope pin — mirrors chad/core/position_guard.py::_validate_position_guard_schema",
  "_note": "Key space is position identity and is intentionally unconstrained. Never add schema_version to this file."
},
"runtime/trade_closer_state.json": {
  "envelope": {
    "required_keys": ["queues", "processed_fill_ids", "saved_at_utc"]
  },
  "pinned_at": "W6B-2 (D1) envelope pin",
  "_note": "Fixed top level, no free key space. Deliberately does not constrain the shape of `queues` — FIFO internals are trade_closer's business."
}
```

### The conditional rule is the point, and it was nearly got wrong

The obvious envelope — "every entry must carry the 7 fields every live entry has" — is
**wrong**, and the live data actively hides that. All 39 current entries carry
`opened_at`, so an envelope inferred from observation would look correct and pass the
pre-flight. But `position_guard.py:459` (`reset_from_broker_truth`) constructs closed
entries **without** `opened_at`. That envelope would go red the first time an operator
reset a position from broker truth — turning a recovery tool into a sentinel failure.

Deriving the rule from the writer's stated contract rather than from a data sample is what
avoids that. It also makes the check genuinely additive: the writer enforces this invariant
on write and refuses bad writes, but **nothing ever checked the file afterwards**. This
adds an independent read-side check that catches corruption from a writer bypass, a manual
edit, or a partial write.

## Pre-flight

Both envelopes were validated against the live artifacts:

```
LIVE position_guard.json    breaks: []
LIVE trade_closer_state.json breaks: []
```

Adoption is inert on day one. Pinned as `test_live_artifact_satisfies_its_proposed_envelope`.

## How to apply

Merge the two blocks above into `config/exterminator.json` →
`schema_contracts.enforced`, and remove the two corresponding entries from
`unpinned_known` (they are no longer unvalidated). The other three `unpinned_known`
entries are removed by the W6B-2 code change once their publishers have written once.

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m pytest chad/tests/test_w6b_envelope_contracts.py -q   # 16 tests
```

## Ordering note

The three code-side pins take effect only when each publisher next writes. Do not remove
their `unpinned_known` entries until `runtime/positions_snapshot.json`,
`runtime/reconciliation_state.json`, and `runtime/profit_lock_state.json` each show a
`schema_version` on disk — otherwise EXS7 loses sight of them in the gap.

The envelope pins have no such ordering constraint: they validate the files exactly as they
are today.

## Rollback

Remove the two `enforced` entries. No publisher changed, so there is nothing else to undo.
The `envelope` support in `check_schema_breaks` is inert when no contract declares one.

## Evidence

| Claim | Source |
|---|---|
| Key space is position identity | `runtime/position_guard.json` — 41 top-level keys = 2 meta + 39 `strategy\|SYMBOL` |
| Writer's declared invariant | `chad/core/position_guard.py:70-104` |
| Closed entries may be sparse | `chad/core/position_guard.py:452-465` (`reset_from_broker_truth`) |
| Reserved-prefix convention already honoured by readers | `position_guard.py:511,653,824` — `key.startswith("_")` skips |
| Both envelopes clean on live data | `chad/tests/test_w6b_envelope_contracts.py::test_live_artifact_satisfies_its_proposed_envelope` |
| Version checking still works for versioned contracts | `...::test_versioned_contract_still_validates_version` |
