# W6B Phase-2 — premise re-verification after merging main (W6A)

**Date:** 2026-07-24
**Branch:** `goal/wave6-hardening`, merged `main@e24f82c` (W6A futures fix) into `640fb4f`.
**Purpose:** the plan (`PLAN_W6B.md`, W6B-0) was written against `main@c76fbea`, which
predates W6A. Before executing, every load-bearing premise was re-measured. This file
records what held, what moved, and what was wrong — before any code changed.

The merge was clean (no conflicts): 12 files, all Lane A territory, none touched by this lane.

---

## Summary

| # | Plan premise | Status today | Consequence |
|---|---|---|---|
| P1 | EXS7: 74 pinned / 10 enforced / 64 unvalidated | **HOLDS exactly** | §2 dominant-win argument stands |
| P2 | EXS7: 62 unpinned, 57 undocumented | **WRONG — it is 127 unpinned + 4 unreadable, 122 undocumented** | W6B-3 redesigned: pattern classes, not enumeration |
| P3 | `deploy/chad-options-chain-refresh.service` missing from repo | **WRONG — the repo copy exists under a legacy name and drifts by one line** | W6B-11/N1 shrinks to a one-line fix |
| P4 | Sentinel is EXS1–EXS10; W6A may have shifted enumeration | **HOLDS — still 10 checks, no renumbering** | no rework |
| P5 | `bars_refresh_state.v1` now exists as a runtime artifact | **PARTLY — the publisher exists; the file does not yet** | must NOT be adopted into `enforced` |
| P6 | EXS5 red is `chad-xgb-train.service` alone, refusing at 72 < 100 | **HOLDS** | D3 design unchanged |
| P7 | EXS8 has no baseline; `no_baseline` warn | **HOLDS** | W6B-4/5/6 chain unchanged |
| P8 | Standing test baseline 16 failed | **Expected 15** — W6A converted one | set-diff by membership, not count |

---

## P2 — the unpinned count was materially understated

Full enumeration of `runtime/*.json` on the live tree, 2026-07-24:

```
dict, schema_version present : 74     <- matches the plan
dict, no schema_version      : 127    <- plan said 62
unreadable as JSON           : 4      <- plan did not report this class
                               ----
total                          205
```

Of the 127 unpinned, **65 are `telegram_dedupe_*.json`** — per-alert dedupe markers whose
filenames are generated from alert text (e.g.
`telegram_dedupe_health_R13_SCRgap214rawvs67effecti.json`). They accreted from 2026-05-29
through 2026-07-22; they are not new, so the plan's 62 was simply a miscount, not drift.

**This breaks W6B-3 as planned.** "Expand `unpinned_known` to cover all remaining
genuinely-unpinned runtime files" would enumerate 122 paths, 65 of them from an *unbounded*
namespace that grows every time a new alert string appears. The list would be stale on
arrival and would need a commit per new alert. W6B-3 is therefore redesigned around
**pattern classes** with a documented reason per class, so coverage is complete and stays
complete without maintenance.

### New sub-finding: 4 runtime `.json` files are not valid JSON

Invisible to EXS7 today — they are neither enforced nor listed, so nothing reports them.

| File | Defect |
|---|---|
| `runtime/flip_executor_audit.json` | **NDJSON in a `.json` filename** — `json.load()` raises `Extra data: line 2`. 6765 bytes, first row 2026-05-04 |
| `runtime/signal_throttle_audit.json` | Same defect. 752 bytes, first row 2026-05-03 |
| `runtime/claude_usage.json` | Zero bytes |
| `runtime/__guard_swallow_probe__.json` | 1 byte (`x`) — test-probe leakage, sibling of the plan's `__guard_probe_should_not_exist__.json` finding |

The first two are the interesting pair: any consumer calling `json.load()` on them fails.
Recorded for disposition; this lane does not delete or rewrite runtime files.

---

## P3 — the options-chain repo unit exists; it drifts by exactly one line

The plan reported `deploy/chad-options-chain-refresh.service` absent. It is absent *under
that name*. The repo carries **`deploy/options_chain_refresh.service`** — a legacy naming
form (no `chad-` prefix, underscores) that predates the current convention.

Diffed against the live unit, the entire delta is:

```
4a5
> OnFailure=chad-service-alert@%N.service
```

So the governance gap is real but far smaller than reported: the live alert wiring is
unversioned, and a redeploy from repo would drop it — but everything else about the unit
is already tracked. The fix is one line in an existing file, not a unit capture.

**A full sweep of all 26 repo units against live found this as the *only* drift.** The
other 25 are byte-identical.

**Separate, wider observation (recorded, not fixed here):** there are **126** live
`chad-*.service` units and **26** in `deploy/`. ~100 live units have no repo copy at all —
including `chad-xgb-train.service`, which this lane touches in W6B-7. Closing that gap
wholesale is its own lane; W6B commits the repo copy only for the units it modifies
(W3B parity pattern).

---

## P5 — `bars_refresh_state.v1` publisher exists, artifact does not

W6A added `_write_bars_refresh_state` (`chad/market_data/nightly_bars_refresh.py:196`,
pinned `bars_refresh_state.v1` at `:278`), but the nightly refresh has not run since the
merge, so `runtime/bars_refresh_state.json` **does not exist on disk**.

Adopting a contract into `enforced` makes EXS7 **FAIL** when the file is missing. Adopting
`bars_refresh_state.v1` today would manufacture a red sentinel for a publisher that is
working correctly and simply has not fired yet. It is therefore **excluded by the liveness
filter** (W6B-1) with "publisher landed, artifact not yet produced" as the recorded reason
— a candidate for a later wave once the first nightly run lands.

This is the concrete case the liveness filter was designed for, and it arrived from Lane A
one merge after the filter was specified.

---

## P8 — baseline is 15, not 16, and membership is what matters

W6A-5 converted `test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop`
(the wall-clock time-bomb test). The plan's recorded 16-row fail-set therefore loses exactly
that row. Verification for this lane is **set-diff on membership**: the target is the plan's
16 rows minus that one, and W6B's obligation is to add nothing and remove nothing further.

---

## Method note

All measurements above are reads. The sentinel was **not** invoked during this
re-verification — the timer's own artifact
(`runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json`, generated 2026-07-24T00:37:25Z) was
read instead, avoiding the extra runtime row the plan had to disclose in its §0.
