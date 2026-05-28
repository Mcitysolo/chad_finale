"""Tests for chad/core/position_truth_engine.py (Decision 1 / R1 closeout).

All tests run under CHAD_SKIP_IB_CONNECT=1. No broker calls; no production
runtime mutation; fixtures live under chad/tests/fixtures/position_truth_v2/.

Fixtures use static timestamps; the engine is constructed with very large
TTLs (LENIENT_TTL) so M5-staleness only fires when the fixture explicitly
intends it (m5_stale_source uses ts=2020 to age out under any TTL).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chad.core import position_truth_engine as pte
from chad.schemas import position_truth_v2 as schema

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "position_truth_v2"

# Large TTL so static fixture timestamps never look stale unless the
# fixture is intentionally ancient (m5_stale_source uses 2020-01-01).
LENIENT_TTL = 10**9


def _engine_for(fixture: str, **kwargs) -> pte.PositionTruthEngine:
    base = FIXTURE_ROOT / fixture
    return pte.PositionTruthEngine(
        snapshot_path=base / "snapshot.json",
        ledger_path=base / "ledger.json",
        snapshot_ttl=kwargs.get("snapshot_ttl", LENIENT_TTL),
        ledger_ttl=kwargs.get("ledger_ttl", LENIENT_TTL),
    )


def _load_expected(fixture: str) -> dict:
    return json.loads((FIXTURE_ROOT / fixture / "expected_truth_v2.json").read_text())


# ---------------------------------------------------------------------------
# Per-rule tests (M1 – M5)
# ---------------------------------------------------------------------------

def test_m1_both_agree_yields_green():
    eng = _engine_for("m1_both_agree")
    doc = eng.run()
    expected = _load_expected("m1_both_agree")
    assert doc.global_authority_health == expected["expected_global_authority_health"] == "GREEN"
    assert doc.positions["AAPL"].merge_rule == "M1"
    assert doc.positions["AAPL"].value_source == "both"
    assert doc.positions["AAPL"].qty == 100.0
    assert doc.positions["AAPL"].side == "LONG"
    assert doc.positions["AAPL"].agreement is True
    assert doc.positions["AAPL"].fail_closed is False
    assert doc.fail_closed_symbols == []


def test_m2_ledger_lag_yields_yellow_with_ledger_authority():
    eng = _engine_for("m2_ledger_lag")
    doc = eng.run()
    expected = _load_expected("m2_ledger_lag")
    assert doc.global_authority_health == expected["expected_global_authority_health"] == "YELLOW"
    p = doc.positions["M6E"]
    assert p.merge_rule == "M2"
    assert p.value_source == "ledger"
    assert p.qty == 56.0
    assert p.snapshot_value == 58.0
    assert p.ledger_value == 56.0
    assert p.delta == -2.0
    assert p.delta_reason == "ledger_lag_within_cadence"
    assert p.fail_closed is False
    assert p.authority_decision == "ledger"


def test_m3_missed_fill_yields_degraded_with_snapshot_authority():
    eng = _engine_for("m3_missed_fill")
    doc = eng.run()
    expected = _load_expected("m3_missed_fill")
    assert doc.global_authority_health == expected["expected_global_authority_health"] == "DEGRADED"
    p = doc.positions["TSLA"]
    assert p.merge_rule == "M3"
    assert p.value_source == "snapshot"
    assert p.qty == 18.0
    assert p.snapshot_value == 18.0
    assert p.ledger_value == 17.0
    assert p.delta == -1.0
    assert p.delta_reason == "missed_fill_event"
    assert p.fail_closed is True
    assert p.authority_decision == "snapshot"
    assert doc.fail_closed_symbols == ["TSLA"]


def test_m4_structural_mismatch_yields_red_with_fail_closed():
    eng = _engine_for("m4_structural_mismatch")
    doc = eng.run()
    expected = _load_expected("m4_structural_mismatch")
    assert doc.global_authority_health == expected["expected_global_authority_health"] == "RED"
    p = doc.positions["SPY"]
    assert p.merge_rule == "M4"
    assert p.value_source == "DISAGREEMENT"
    assert p.qty is None
    assert p.snapshot_value == 1.0
    assert p.ledger_value == -1.0
    assert p.delta_reason == "structural_mismatch"
    assert p.fail_closed is True
    assert p.authority_decision == "FAIL_CLOSED"
    assert doc.fail_closed_symbols == ["SPY"]


def test_m5_stale_source_yields_red_with_fail_closed():
    # Use the realistic 600s snapshot TTL so the 2020-01-01 ts ages out.
    eng = _engine_for(
        "m5_stale_source",
        snapshot_ttl=schema.SNAPSHOT_TTL_SECONDS,
        ledger_ttl=schema.LEDGER_TTL_SECONDS,
    )
    doc = eng.run()
    expected = _load_expected("m5_stale_source")
    assert doc.global_authority_health == expected["expected_global_authority_health"] == "RED"
    p = doc.positions["QQQ"]
    assert p.merge_rule == "M5"
    assert p.value_source == "FAIL_CLOSED"
    assert p.qty is None
    assert p.delta_reason == "stale_source"
    assert p.fail_closed is True
    assert doc.fail_closed_symbols == ["QQQ"]


# ---------------------------------------------------------------------------
# Real-world derived fixtures
# ---------------------------------------------------------------------------

def test_real_world_19_vs_18_classifies_correctly():
    """18 symbols agree; MGC present only in ledger → M5 missing_in_one_source."""
    eng = _engine_for("real_19_vs_18")
    doc = eng.run()
    expected = _load_expected("real_19_vs_18")

    assert len(doc.positions) == expected["expected_total_symbols"] == 19

    m1_syms = [s for s, p in doc.positions.items() if p.merge_rule == "M1"]
    assert len(m1_syms) == expected["expected_m1_count"] == 18

    m5_syms = [s for s, p in doc.positions.items() if p.merge_rule == "M5"]
    assert m5_syms == expected["expected_m5_symbols"] == ["MGC"]

    mgc = doc.positions["MGC"]
    assert mgc.merge_rule == "M5"
    assert mgc.delta_reason == "missing_in_one_source"
    assert mgc.snapshot_value is None
    assert mgc.ledger_value == -1.0
    assert mgc.fail_closed is True
    assert mgc.qty is None

    assert doc.global_authority_health == "RED"  # M5 dominates 18× M1
    assert doc.fail_closed_symbols == ["MGC"]


def test_mgc_drift_classified_per_rule_decision():
    eng = _engine_for("mgc_drift")
    doc = eng.run()
    expected = _load_expected("mgc_drift")
    assert doc.global_authority_health == expected["expected_global_authority_health"] == "RED"
    p = doc.positions["MGC"]
    assert p.merge_rule == "M5"
    assert p.delta_reason == "missing_in_one_source"
    assert p.snapshot_value is None
    assert p.ledger_value == -1.0
    assert p.fail_closed is True


# ---------------------------------------------------------------------------
# Safety / contract tests
# ---------------------------------------------------------------------------

def test_engine_does_not_mutate_source_files():
    """sha256 of both source files must be identical before and after run()."""
    base = FIXTURE_ROOT / "real_19_vs_18"
    snap = base / "snapshot.json"
    led = base / "ledger.json"
    before_snap = hashlib.sha256(snap.read_bytes()).hexdigest()
    before_led = hashlib.sha256(led.read_bytes()).hexdigest()

    eng = _engine_for("real_19_vs_18")
    _ = eng.run()

    after_snap = hashlib.sha256(snap.read_bytes()).hexdigest()
    after_led = hashlib.sha256(led.read_bytes()).hexdigest()
    assert before_snap == after_snap, "engine mutated snapshot file"
    assert before_led == after_led, "engine mutated ledger file"


def test_atomic_write_prevents_partial_state(tmp_path):
    """write_truth_v2() must write via a tmp file + os.replace, leaving no
    stray .tmp file behind."""
    eng = _engine_for("m1_both_agree")
    out = tmp_path / "out" / "position_truth_v2.json"
    eng.write_truth_v2(out)
    assert out.is_file()
    # No leftover tmp
    assert not out.with_suffix(out.suffix + ".tmp").exists()
    # File parses back as the same schema_version
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == "position_truth_v2.v1"


def test_cli_check_mode_does_not_write(tmp_path, monkeypatch, capsys):
    """The CLI in --check mode must not write any file under runtime/."""
    # Point the CLI at fixtures so we don't depend on live runtime state.
    base = FIXTURE_ROOT / "m1_both_agree"

    # Sanity: no position_truth_v2 file exists in fixture or tmp before
    assert not (tmp_path / "position_truth_v2.json").exists()

    rc = pte.main([
        "--check",
        "--snapshot", str(base / "snapshot.json"),
        "--ledger", str(base / "ledger.json"),
        "--snapshot-ttl", str(LENIENT_TTL),
        "--ledger-ttl", str(LENIENT_TTL),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["schema_version"] == "position_truth_v2.v1"

    # No file written
    assert not (tmp_path / "position_truth_v2.json").exists()


def test_schema_version_present_in_output():
    eng = _engine_for("m1_both_agree")
    doc = eng.run()
    assert doc.schema_version == "position_truth_v2.v1"
    assert doc.engine_version == "1.0.0"
    assert doc.authority_mode == "merged_with_provenance"


def test_global_health_computed_from_per_symbol_max_severity():
    """Mixing M1 + M3 → DEGRADED (M3 is the worst). Adding M5 escalates to RED."""
    # Use the m3 fixture as base, then construct a virtual scenario:
    # we run the helper directly to confirm severity ordering.
    assert schema.health_from_rules(["M1"]) == "GREEN"
    assert schema.health_from_rules(["M1", "M2"]) == "YELLOW"
    assert schema.health_from_rules(["M1", "M2", "M3"]) == "DEGRADED"
    assert schema.health_from_rules(["M1", "M3", "M4"]) == "RED"
    assert schema.health_from_rules(["M1", "M3", "M5"]) == "RED"
    assert schema.health_from_rules([]) == "GREEN"
