"""Read-only validator for ``position_truth_v2.v1`` documents.

Validates a JSON file (default: ``runtime/position_truth_v2.json``,
not present in this phase) and reports schema / consistency
violations. Never mutates any file.

CLI:
    python -m chad.validators.position_truth_v2 --check [--path X]

Exit codes:
    0 valid
    2 invalid (one or more failures listed in JSON output)
    3 file missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chad.schemas.position_truth_v2 import (
    HEALTH_SEVERITY,
    RULE_TO_HEALTH,
    SCHEMA_VERSION,
    VALID_AUTHORITY_DECISIONS,
    VALID_MERGE_RULES,
    VALID_SIDES,
    VALID_VALUE_SOURCES,
    VS_DISAGREEMENT,
    VS_FAIL_CLOSED,
    health_from_rules,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "runtime" / "position_truth_v2.json"

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_MISSING = 3

REQUIRED_TOP_KEYS = (
    "schema_version",
    "ts_utc",
    "engine_version",
    "authority_mode",
    "ttl_seconds",
    "source_artifacts",
    "positions",
    "global_authority_health",
    "fail_closed_symbols",
    "warnings",
    "errors",
)

REQUIRED_POSITION_KEYS = (
    "qty",
    "side",
    "value_source",
    "snapshot_value",
    "ledger_value",
    "agreement",
    "delta",
    "delta_reason",
    "merge_rule",
    "authority_decision",
    "fail_closed",
    "last_reconciled_utc",
    "provenance_chain",
)


def validate(doc: dict) -> list[str]:
    """Return a list of validation failures. Empty list ⇔ valid."""
    failures: list[str] = []

    # ---- top-level required keys ----------------------------------------
    for k in REQUIRED_TOP_KEYS:
        if k not in doc:
            failures.append(f"missing_top_key:{k}")

    sv = doc.get("schema_version")
    if sv != SCHEMA_VERSION:
        failures.append(f"schema_version_mismatch:got={sv!r}:expected={SCHEMA_VERSION!r}")

    # ---- per-symbol validation ------------------------------------------
    positions = doc.get("positions") or {}
    if not isinstance(positions, dict):
        failures.append("positions_not_a_dict")
        positions = {}

    rules_seen: list[str] = []
    for sym, p in positions.items():
        if not isinstance(p, dict):
            failures.append(f"position_not_dict:{sym}")
            continue
        for k in REQUIRED_POSITION_KEYS:
            if k not in p:
                failures.append(f"position_missing_key:{sym}:{k}")

        vs = p.get("value_source")
        if vs not in VALID_VALUE_SOURCES:
            failures.append(f"position_bad_value_source:{sym}:{vs}")
        side = p.get("side")
        if side not in VALID_SIDES:
            failures.append(f"position_bad_side:{sym}:{side}")
        ad = p.get("authority_decision")
        if ad not in VALID_AUTHORITY_DECISIONS:
            failures.append(f"position_bad_authority_decision:{sym}:{ad}")
        mr = p.get("merge_rule")
        if mr not in VALID_MERGE_RULES:
            failures.append(f"position_bad_merge_rule:{sym}:{mr}")
        else:
            rules_seen.append(mr)

        # Provenance must have at least one entry.
        chain = p.get("provenance_chain")
        if not isinstance(chain, list) or len(chain) == 0:
            failures.append(f"position_empty_provenance_chain:{sym}")

        # DISAGREEMENT must be fail_closed.
        if vs == VS_DISAGREEMENT and not p.get("fail_closed"):
            failures.append(f"position_disagreement_without_fail_closed:{sym}")
        # FAIL_CLOSED value_source must have fail_closed=true and qty=None.
        if vs == VS_FAIL_CLOSED:
            if not p.get("fail_closed"):
                failures.append(f"position_fail_closed_value_source_without_fail_closed_flag:{sym}")
            if p.get("qty") is not None:
                failures.append(f"position_fail_closed_with_non_null_qty:{sym}")

    # ---- global health consistency --------------------------------------
    declared = doc.get("global_authority_health")
    expected = health_from_rules(rules_seen)
    # If engine recorded errors, RED is allowed regardless of per-symbol rules.
    if doc.get("errors"):
        if declared != "RED":
            failures.append(
                f"global_health_with_errors_must_be_RED:got={declared}"
            )
    else:
        if declared not in HEALTH_SEVERITY:
            failures.append(f"global_health_unknown:{declared}")
        elif declared != expected:
            failures.append(
                f"global_health_mismatch:declared={declared}:expected={expected}"
            )

    # ---- fail_closed_symbols consistency --------------------------------
    declared_fc = set(doc.get("fail_closed_symbols") or [])
    expected_fc = {s for s, p in positions.items() if isinstance(p, dict) and p.get("fail_closed")}
    if declared_fc != expected_fc:
        failures.append(
            f"fail_closed_symbols_mismatch:declared={sorted(declared_fc)}:expected={sorted(expected_fc)}"
        )

    return failures


def validate_path(path: Path) -> tuple[int, dict]:
    if not path.is_file():
        return EXIT_MISSING, {
            "validator": "position_truth_v2",
            "path": str(path),
            "error": "file_missing",
        }
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return EXIT_INVALID, {
            "validator": "position_truth_v2",
            "path": str(path),
            "failures": [f"json_parse_error:{exc}"],
        }
    failures = validate(doc)
    report = {
        "validator": "position_truth_v2",
        "path": str(path),
        "schema_version": doc.get("schema_version"),
        "global_authority_health": doc.get("global_authority_health"),
        "positions_count": len(doc.get("positions") or {}),
        "failures": failures,
        "valid": not failures,
    }
    return (EXIT_OK if not failures else EXIT_INVALID), report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="chad.validators.position_truth_v2",
        description="Validate a position_truth_v2.json document.",
    )
    p.add_argument("--check", action="store_true", help="Run validator (default).")
    p.add_argument("--path", default=str(DEFAULT_PATH), help="Path to position_truth_v2.json")
    args = p.parse_args(argv)
    code, report = validate_path(Path(args.path))
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
