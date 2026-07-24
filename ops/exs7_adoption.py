#!/usr/bin/env python3
"""
ops/exs7_adoption.py — EXS7 enforcement-coverage tooling (W6B-1).

WHY THIS EXISTS
---------------
EXS7 pins 10 runtime contracts into ``schema_contracts.enforced`` and validates
them every sentinel cycle. Another 64 runtime files already CARRY a
``schema_version`` that nothing checks. The sentinel's WARN, meanwhile, is
scoped to a hand-curated list of 5 files in ``unpinned_known`` — so the warn
describes a curated sample while the larger gap (pinned-yet-unvalidated) is
invisible.

The dominant, near-zero-risk win is adoption into ``enforced``: pure config,
no publisher touched.

WHAT "LIVENESS" ACTUALLY MEANS HERE
-----------------------------------
The plan framed the filter as freshness ("live publisher + known cadence").
Reading what ``check_schema_breaks`` actually does narrows that materially.
It breaks on exactly four conditions:

    missing | unreadable_or_not_an_object |
    schema_version_absent/unrecognised | required_keys_missing

**Staleness is not one of them** — feed freshness is EXS1's job. So a dormant
but permanent file is perfectly safe to enforce: it sits there, valid, and
never breaks. ``savage_alloc_state.json`` has not been rewritten in 136 days
and would pass enforcement every cycle.

The real disqualifier is therefore not "is it stale?" but:

    **can this file be legitimately ABSENT during normal operation?**

Three classes can, and they are the exclusions:

  1. DATED ONE-OFFS — ``quarantine_manifest_20260511.json`` and friends. Written
     once for one incident, safe to archive or prune. Enforcing them converts
     ordinary housekeeping into a red sentinel.
  2. EVENT-CONDITIONAL artifacts — ``epoch_reset_state.json``,
     ``stop_bus_recovery_state.json``. These exist only *after* the event that
     writes them. Absence is the normal state, not a defect.
  3. PUBLISHER-NOT-YET-FIRED — a contract whose writer has landed in code but
     has never produced its artifact. ``bars_refresh_state.v1`` (W6A) is the
     live example: the publisher exists, the nightly run has not fired, so the
     file does not exist. Enforcing it would manufacture a red for a publisher
     that is working correctly.

Everything else is eligible.

WHAT GETS PINNED
----------------
Deliberately minimal: ``accepts`` = the version observed today, and
``required_keys`` = ``["schema_version"]`` only.

Inferring a richer key contract from a single observation is how you turn an
optional field into a permanent false FAIL — the publisher omits it on some
branch six weeks from now and the sentinel goes red over a key that was never
promised. The minimal contract still catches every break mode that actually
matters: the file vanishes, becomes corrupt, or silently changes schema
version. Richer per-file key contracts are a follow-on for someone who has
read the publisher, and the 10 existing hand-written entries already model
that.

USAGE
-----
    python3 -m ops.exs7_adoption                 # report (default)
    python3 -m ops.exs7_adoption --verify        # would adoption break EXS7 today?
    python3 -m ops.exs7_adoption --emit          # config block for the Pending Action

Read-only. Writes nothing, ever.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "exterminator.json"

# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------

# Class 1 — dated / one-off artifacts. Matched by NAME, because the property
# that disqualifies them (written once for one incident, prunable) is encoded
# in the name and does not depend on today's mtime.
DATED_ONE_OFF_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"_\d{8}(_|\.)"),          # ..._20260511.json, ..._20260420_step.json
    re.compile(r"^quarantine_manifest"),  # incident-scoped quarantine records
    re.compile(r"^broker_truth_snapshot"),
    re.compile(r"^__.*__\.json$"),        # test-probe leakage
)

# Class 2 — event-conditional. Absence is normal: these exist only after the
# event that writes them has occurred. Listed explicitly (not pattern-matched)
# because this is a semantic judgment about each publisher, and a reader
# deserves to see the reason rather than reverse-engineer a regex.
EVENT_CONDITIONAL: Dict[str, str] = {
    "runtime/epoch_reset_state.json":
        "written only when an epoch reset is executed; absent on a system that "
        "has never reset (and after archival)",
    "runtime/stop_bus_recovery_state.json":
        "written only when the stop bus trips and recovers; absent is the "
        "healthy steady state",
    "runtime/lifecycle_replay_drift_audit.json":
        "audit artifact produced on demand by lifecycle_replay_drift_audit.py, "
        "not on a cadence",
}

# Class 3 — publisher has landed but has never produced its artifact. Adopting
# these is correct LATER, once the first run lands; adopting now manufactures a
# red for a working publisher.
PUBLISHER_NOT_YET_FIRED: Dict[str, str] = {
    "runtime/bars_refresh_state.json":
        "W6A landed _write_bars_refresh_state (nightly_bars_refresh.py:196, pins "
        "bars_refresh_state.v1 at :278) but the nightly run has not fired since "
        "the merge, so the artifact does not exist yet. Re-evaluate after the "
        "first nightly run.",
}


def _is_dated_one_off(basename: str) -> bool:
    return any(p.search(basename) for p in DATED_ONE_OFF_PATTERNS)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _load_enforced(config_path: Path) -> Dict[str, Any]:
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    sc = cfg.get("schema_contracts") or {}
    return sc.get("enforced") or {}


def classify(runtime_dir: Path, config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Partition runtime/*.json into already-enforced / eligible / excluded.

    Pure read. Returns a dict shaped for both the report and the emitter.
    """
    enforced = _load_enforced(config_path)

    already: List[str] = []
    eligible: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    unpinned: List[str] = []
    unreadable: List[Dict[str, str]] = []

    for path in sorted(runtime_dir.glob("*.json")):
        rel = f"runtime/{path.name}"

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            unreadable.append({"file": rel, "error": str(exc)[:120]})
            continue

        if not isinstance(payload, dict):
            unreadable.append({"file": rel, "error": "not a JSON object"})
            continue

        version = payload.get("schema_version")
        if not version:
            unpinned.append(rel)
            continue

        if rel in enforced:
            already.append(rel)
            continue

        if _is_dated_one_off(path.name):
            excluded.append({
                "file": rel,
                "class": "dated_one_off",
                "reason": "written once for one incident; prunable, so absence is not a defect",
            })
            continue

        if rel in EVENT_CONDITIONAL:
            excluded.append({
                "file": rel, "class": "event_conditional",
                "reason": EVENT_CONDITIONAL[rel],
            })
            continue

        eligible.append({"file": rel, "schema_version": str(version)})

    # Class 3 is keyed on files that do NOT exist, so it is folded in separately.
    # Resolved against the runtime dir under audit, not REPO_ROOT — the two
    # differ whenever this runs from a worktree against the live tree.
    for rel, reason in PUBLISHER_NOT_YET_FIRED.items():
        if not (runtime_dir.parent / rel).exists():
            excluded.append({
                "file": rel, "class": "publisher_not_yet_fired", "reason": reason,
            })

    return {
        "already_enforced": already,
        "eligible": eligible,
        "excluded": excluded,
        "unpinned": unpinned,
        "unreadable": unreadable,
    }


def build_contract(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The minimal contract for an auto-adopted file. See module docstring for
    why required_keys is schema_version alone."""
    return {
        "schema_version": entry["schema_version"],
        "accepts": [entry["schema_version"]],
        "required_keys": ["schema_version"],
        "pinned_at": "W6B-1 liveness-filtered adoption (ops/exs7_adoption.py)",
    }


def verify(runtime_dir: Path, eligible: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pre-flight: would adopting these break EXS7 right now?

    Mirrors chad/ops/exterminator_sentinel.py::check_schema_breaks so the answer
    is about the real check, not an approximation of it. An empty list means
    adoption is inert on day one — which is the whole point of shipping the
    config change as a reviewed Pending Action rather than a surprise.
    """
    breaks: List[Dict[str, Any]] = []
    for entry in eligible:
        contract = build_contract(entry)
        path = runtime_dir.parent / entry["file"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            breaks.append({"file": entry["file"], "break": "missing"})
            continue
        except Exception as exc:
            breaks.append({
                "file": entry["file"], "break": "unreadable_or_not_an_object",
                "error": str(exc)[:120],
            })
            continue
        if not isinstance(payload, dict):
            breaks.append({
                "file": entry["file"], "break": "unreadable_or_not_an_object",
            })
            continue
        actual = payload.get("schema_version")
        if actual is None:
            breaks.append({"file": entry["file"], "break": "schema_version_absent"})
        elif str(actual) not in contract["accepts"]:
            breaks.append({
                "file": entry["file"], "break": "schema_version_unrecognised",
                "actual": actual, "expected_one_of": contract["accepts"],
            })
        missing = [k for k in contract["required_keys"] if k not in payload]
        if missing:
            breaks.append({
                "file": entry["file"], "break": "required_keys_missing",
                "missing_keys": missing,
            })
    return breaks


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EXS7 enforcement-coverage tooling (read-only)")
    ap.add_argument("--runtime-dir", default=str(REPO_ROOT / "runtime"))
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--verify", action="store_true",
                    help="check whether adopting the eligible set would break EXS7 today")
    ap.add_argument("--emit", action="store_true",
                    help="print the config block for the Pending Action")
    args = ap.parse_args(argv)

    runtime_dir = Path(args.runtime_dir)
    result = classify(runtime_dir, Path(args.config))

    if args.emit:
        block = {
            e["file"]: build_contract(e) for e in result["eligible"]
        }
        print(json.dumps(block, indent=2, sort_keys=True))
        return 0

    if args.verify:
        breaks = verify(runtime_dir, result["eligible"])
        print(f"eligible={len(result['eligible'])} breaks={len(breaks)}")
        for b in breaks:
            print(f"  BREAK {b['file']}: {b['break']}")
        if breaks:
            print("\nAdoption would turn EXS7 RED. Do NOT apply the PA as-is.")
            return 1
        print("\nAdoption is inert today: every eligible contract already holds.")
        return 0

    print(f"already enforced : {len(result['already_enforced'])}")
    print(f"eligible         : {len(result['eligible'])}")
    print(f"excluded         : {len(result['excluded'])}")
    print(f"unpinned         : {len(result['unpinned'])}")
    print(f"unreadable       : {len(result['unreadable'])}")
    print("\n-- EXCLUDED (with reason) --")
    for e in sorted(result["excluded"], key=lambda x: (x["class"], x["file"])):
        print(f"  [{e['class']}] {e['file']}\n      {e['reason']}")
    print("\n-- UNREADABLE (invisible to EXS7 today) --")
    for u in result["unreadable"]:
        print(f"  {u['file']}: {u['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
