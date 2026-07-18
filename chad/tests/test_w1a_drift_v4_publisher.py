"""
W1A-5 — CHAD_DRIFT_V4 publisher wiring (sibling observability file, default off).

Per GO decision D2: the v4 independent-leg view is emitted to a SIBLING file
(runtime/position_guard_drift_v4.json) that does NOT touch the v3 file and does
NOT feed the live-readiness RED gate. Default off ⇒ pure no-op; the v3 path is
untouched. All writes go to tmp_path.
"""

from __future__ import annotations

import json

from chad.ops import reconciliation_publisher as recon

# Reuse the guard/snapshot fixtures + fixed clock from the detector test module.
from chad.tests.test_w1a_drift_v4 import _guard_dualbook, _snap, NOW  # noqa: E402


def _write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _neutralize_exclusions(monkeypatch) -> None:
    # Isolate the test from prod config so excluded_symbols is deterministic.
    monkeypatch.setattr(recon, "EXCLUSION_POLICY", {})
    monkeypatch.setattr(recon, "_BROKER_PREEXISTING", frozenset())


def test_publisher_flag_off_is_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CHAD_DRIFT_V4", raising=False)
    out = tmp_path / "position_guard_drift_v4.json"
    g = tmp_path / "position_guard.json"
    s = tmp_path / "positions_snapshot.json"
    _write(g, _guard_dualbook())
    _write(s, _snap([{"symbol": "UNH", "position": 273.0}]))

    r = recon._emit_position_guard_drift_v4(guard_path=g, snapshot_path=s, out_path=out, now=NOW)

    assert r is None            # no-op when flag unset
    assert not out.exists()     # no sibling file written


def test_publisher_flag_zero_is_off(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHAD_DRIFT_V4", "0")  # strict parse: "0" is OFF
    out = tmp_path / "v4.json"
    r = recon._emit_position_guard_drift_v4(out_path=out, now=NOW)
    assert r is None
    assert not out.exists()


def test_publisher_flag_on_writes_sibling_with_actionable_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHAD_DRIFT_V4", "1")
    _neutralize_exclusions(monkeypatch)
    out = tmp_path / "position_guard_drift_v4.json"
    g = tmp_path / "position_guard.json"
    s = tmp_path / "positions_snapshot.json"
    _write(g, _guard_dualbook("UNH", 273.0))
    _write(s, _snap([{"symbol": "UNH", "position": -273.0}]))  # mirror flip

    r = recon._emit_position_guard_drift_v4(guard_path=g, snapshot_path=s, out_path=out, now=NOW)

    assert r is not None
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == "position_guard_drift.v4"
    assert payload["independent_leg"] == "fresh"
    assert payload["drift_count"] == 1
    assert "ts_utc" in payload and payload["ttl_seconds"] == recon.TTL_SECONDS
    # D2: the v3 file is NOT touched by the v4 emit.
    assert not (tmp_path / "position_guard_drift.json").exists()


def test_publisher_flag_on_blind_snapshot_still_writes_sibling(tmp_path, monkeypatch) -> None:
    # A missing/stale independent leg must produce an honest "blind" sibling, not
    # silence and not a false "all agree".
    monkeypatch.setenv("CHAD_DRIFT_V4", "1")
    _neutralize_exclusions(monkeypatch)
    out = tmp_path / "position_guard_drift_v4.json"
    g = tmp_path / "position_guard.json"
    _write(g, _guard_dualbook())

    r = recon._emit_position_guard_drift_v4(
        guard_path=g, snapshot_path=tmp_path / "does_not_exist.json", out_path=out, now=NOW,
    )

    assert r["independent_leg"] == "blind"
    assert out.exists()
    assert json.loads(out.read_text())["drift_count"] == 0


def test_flag_parser_strictness(monkeypatch) -> None:
    for on in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("CHAD_DRIFT_V4_TESTPARSE", on)
        assert recon._flag_on("CHAD_DRIFT_V4_TESTPARSE") is True
    for off in ("0", "false", "no", "off", "", "banana"):
        monkeypatch.setenv("CHAD_DRIFT_V4_TESTPARSE", off)
        assert recon._flag_on("CHAD_DRIFT_V4_TESTPARSE") is False
