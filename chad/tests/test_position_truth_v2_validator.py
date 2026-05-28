"""Tests for chad/validators/position_truth_v2.py."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chad.validators import position_truth_v2 as v
from chad.schemas import position_truth_v2 as schema


def _valid_doc() -> dict:
    """Hand-crafted minimal-but-valid position_truth_v2 document."""
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "ts_utc": "2026-05-28T01:00:00Z",
        "engine_version": schema.ENGINE_VERSION,
        "authority_mode": schema.AUTHORITY_MODE,
        "ttl_seconds": schema.DEFAULT_TTL_SECONDS,
        "source_artifacts": {
            "snapshot": {"path": "x", "ts_utc": "2026-05-28T00:59:50Z", "sha256": "deadbeef", "age_seconds": 10.0},
            "ledger":   {"path": "y", "ts_utc": "2026-05-28T00:59:55Z", "sha256": "cafebabe", "age_seconds": 5.0},
        },
        "positions": {
            "AAPL": {
                "qty": 100.0,
                "side": "LONG",
                "value_source": "both",
                "snapshot_value": 100.0,
                "ledger_value": 100.0,
                "agreement": True,
                "delta": 0,
                "delta_reason": "in_agreement",
                "merge_rule": "M1",
                "authority_decision": "ledger",
                "fail_closed": False,
                "last_reconciled_utc": "2026-05-28T01:00:00Z",
                "provenance_chain": [
                    {"surface": "snapshot", "ref": "265598", "ts_utc": "2026-05-28T00:59:50Z"},
                    {"surface": "ledger",   "ref": "aaaa1111", "ts_utc": "2026-05-28T00:59:55Z"},
                ],
            }
        },
        "global_authority_health": "GREEN",
        "fail_closed_symbols": [],
        "warnings": [],
        "errors": [],
    }


def test_valid_truth_v2_passes():
    failures = v.validate(_valid_doc())
    assert failures == [], failures


def test_missing_schema_version_fails():
    doc = _valid_doc()
    del doc["schema_version"]
    failures = v.validate(doc)
    assert any(f.startswith("missing_top_key:schema_version") for f in failures)
    assert any(f.startswith("schema_version_mismatch") for f in failures)


def test_disagreement_without_fail_closed_is_invalid():
    doc = _valid_doc()
    doc["positions"]["AAPL"]["value_source"] = "DISAGREEMENT"
    doc["positions"]["AAPL"]["fail_closed"] = False
    doc["positions"]["AAPL"]["merge_rule"] = "M4"
    # Adjust global health so the only failure surfaced is the one we test.
    doc["global_authority_health"] = "RED"
    failures = v.validate(doc)
    assert any("position_disagreement_without_fail_closed:AAPL" == f for f in failures), failures


def test_empty_provenance_chain_is_invalid():
    doc = _valid_doc()
    doc["positions"]["AAPL"]["provenance_chain"] = []
    failures = v.validate(doc)
    assert any("position_empty_provenance_chain:AAPL" == f for f in failures)


def test_global_health_mismatch_detected():
    doc = _valid_doc()
    # All symbols are M1 → expected GREEN. Declare YELLOW → mismatch.
    doc["global_authority_health"] = "YELLOW"
    failures = v.validate(doc)
    assert any(f.startswith("global_health_mismatch:") for f in failures), failures


def test_missing_file_exits_3(tmp_path):
    rc, report = v.validate_path(tmp_path / "does-not-exist.json")
    assert rc == v.EXIT_MISSING
    assert report["error"] == "file_missing"


# ---------------------------------------------------------------------------
# Extra integration: validator accepts engine output verbatim
# ---------------------------------------------------------------------------

def test_validator_accepts_engine_output_for_m1_fixture(tmp_path):
    """Run the engine over the m1_both_agree fixture, write it, and feed
    the file through the validator. Must pass."""
    from chad.core import position_truth_engine as pte

    base = Path(__file__).parent / "fixtures" / "position_truth_v2" / "m1_both_agree"
    eng = pte.PositionTruthEngine(
        snapshot_path=base / "snapshot.json",
        ledger_path=base / "ledger.json",
        snapshot_ttl=10**9,
        ledger_ttl=10**9,
    )
    out = tmp_path / "ptv2.json"
    eng.write_truth_v2(out)
    rc, report = v.validate_path(out)
    assert rc == v.EXIT_OK, report["failures"]
    assert report["valid"] is True
    assert report["global_authority_health"] == "GREEN"


def test_fail_closed_value_source_must_have_null_qty():
    doc = _valid_doc()
    doc["positions"]["AAPL"]["value_source"] = "FAIL_CLOSED"
    doc["positions"]["AAPL"]["qty"] = 100.0  # invalid combo
    doc["positions"]["AAPL"]["fail_closed"] = True
    doc["positions"]["AAPL"]["merge_rule"] = "M5"
    doc["global_authority_health"] = "RED"
    doc["fail_closed_symbols"] = ["AAPL"]
    failures = v.validate(doc)
    assert any(f == "position_fail_closed_with_non_null_qty:AAPL" for f in failures), failures
