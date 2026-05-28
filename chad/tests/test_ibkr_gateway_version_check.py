"""Tests for chad/tools/ibkr_gateway_version_check.py + R20 health rule (Fix C).

All tests use pytest tmp_path fixtures — they NEVER touch /home/ubuntu/Jts.
Run under CHAD_SKIP_IB_CONNECT=1; no broker calls, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.tools import ibkr_gateway_version_check as vc


# --- helpers ---------------------------------------------------------------


def _make_install(jts_root: Path, build_dirname: str, *, jar: str | None = None,
                  version_txt: str | None = None) -> Path:
    """Create <jts_root>/ibgateway/<build_dirname>/ with optional jar/version.txt."""
    build_dir = jts_root / "ibgateway" / build_dirname
    build_dir.mkdir(parents=True)
    if jar is not None:
        jars = build_dir / "jars"
        jars.mkdir()
        (jars / jar).write_text("", encoding="utf-8")
    if version_txt is not None:
        (build_dir / "version.txt").write_text(version_txt, encoding="utf-8")
    return build_dir


# --- tool: detection strategies --------------------------------------------


def test_directory_name_strategy_parses_build_1037(tmp_path: Path) -> None:
    _make_install(tmp_path, "1037")
    inst = vc.detect_installed(tmp_path)
    assert inst["build"] == 1037
    assert inst["display"] == "10.37"
    assert inst["detection_source"] == "directory_name"


def test_directory_name_strategy_parses_build_1045(tmp_path: Path) -> None:
    _make_install(tmp_path, "1045")
    inst = vc.detect_installed(tmp_path)
    assert inst["build"] == 1045
    assert inst["display"] == "10.45"


def test_multiple_versions_picks_highest(tmp_path: Path) -> None:
    _make_install(tmp_path, "1037")
    _make_install(tmp_path, "1045")
    inst = vc.detect_installed(tmp_path)
    assert inst["build"] == 1045
    assert inst["detection_source"] == "directory_name"


def test_jar_filename_fallback_when_dir_name_ambiguous(tmp_path: Path) -> None:
    _make_install(tmp_path, "current", jar="jts4launch-1037.jar")
    inst = vc.detect_installed(tmp_path)
    assert inst["build"] == 1037
    assert inst["detection_source"] == "jar_filename"


def test_returns_unknown_when_no_install_found(tmp_path: Path) -> None:
    # Empty tmp_path — no ibgateway dir at all.
    report = vc.build_report(jts_root=tmp_path, target_build=1045)
    assert report["installed"]["build"] is None
    assert report["comparison"]["severity"] == "unknown"
    assert vc._exit_code_for(report) == vc.EXIT_UNKNOWN


# --- tool: severity classification -----------------------------------------


def test_severity_info_when_at_or_above_target() -> None:
    at = vc.classify(1045, 1045)
    assert at["comparison"]["severity"] == "info"
    assert at["exit_code"] == vc.EXIT_INFO
    above = vc.classify(1050, 1045)
    assert above["comparison"]["severity"] == "info"
    assert above["comparison"]["is_current"] is True


def test_severity_warning_when_within_5_builds() -> None:
    res = vc.classify(1042, 1045)
    assert res["comparison"]["severity"] == "warning"
    assert res["comparison"]["build_delta"] == 3
    assert res["exit_code"] == vc.EXIT_STALE_OR_WARNING


def test_severity_stale_when_more_than_5_builds_behind() -> None:
    res = vc.classify(1037, 1045)  # today's situation, delta=8
    assert res["comparison"]["severity"] == "stale"
    assert res["comparison"]["build_delta"] == 8
    assert res["exit_code"] == vc.EXIT_STALE_OR_WARNING


# --- tool: schema + output -------------------------------------------------


def test_json_output_schema_v1(tmp_path: Path) -> None:
    _make_install(tmp_path, "1037")
    report = vc.build_report(jts_root=tmp_path, target_build=1045)
    assert report["schema_version"] == "ibkr_gateway_version_check.v1"
    for key in ("schema_version", "ts_utc", "installed", "target", "comparison",
                "recommendation"):
        assert key in report
    for key in ("build", "display", "install_path", "detection_source",
                "detection_error"):
        assert key in report["installed"]
    for key in ("build", "display", "source"):
        assert key in report["target"]
    for key in ("is_current", "build_delta", "severity"):
        assert key in report["comparison"]


def test_output_flag_writes_atomically(tmp_path: Path) -> None:
    _make_install(tmp_path, "1037")
    out = tmp_path / "out.json"
    report = vc.build_report(jts_root=tmp_path, target_build=1045)
    vc.write_output_atomic(report, out)
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "ibkr_gateway_version_check.v1"
    # No tmp files left behind.
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_no_filesystem_mutation_without_output_flag(tmp_path: Path) -> None:
    _make_install(tmp_path, "1037")

    def _snapshot() -> dict[str, float]:
        return {str(p): p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}

    before = _snapshot()
    # build_report performs no writes; only detection reads.
    vc.build_report(jts_root=tmp_path, target_build=1045)
    after = _snapshot()
    assert after == before


def test_no_network_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    _make_install(tmp_path, "1037")
    # Must complete without touching the network.
    report = vc.build_report(jts_root=tmp_path, target_build=1045)
    assert report["installed"]["build"] == 1037


# --- health-monitor rule R20 (mirrors R19 monkeypatch idiom) ---------------


def _write_cache(runtime: Path, severity: str, build: int, delta: int) -> None:
    payload = {
        "schema_version": "ibkr_gateway_version_check.v1",
        "ts_utc": "2026-05-28T12:00:00+00:00",
        "installed": {
            "build": build,
            "display": vc.build_to_display(build) if build else None,
            "install_path": "/home/ubuntu/Jts/ibgateway/x",
            "detection_source": "directory_name" if build else "none",
            "detection_error": None if build else "detection_failed",
        },
        "target": {"build": 1045, "display": "10.45", "source": "test"},
        "comparison": {"is_current": severity == "info", "build_delta": delta,
                       "severity": severity},
        "recommendation": "test",
    }
    (runtime / "ibkr_gateway_version.json").write_text(json.dumps(payload),
                                                       encoding="utf-8")


def test_rule_ibkr_gateway_version_info_when_current(monkeypatch, tmp_path) -> None:
    from chad.ops import health_monitor_rules as hmr
    _write_cache(tmp_path, "info", 1045, -0)
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    findings: list = []
    hmr.rule_ibkr_gateway_version(findings)
    assert [f for f in findings if f.rule_id == "R20"] == []


def test_rule_ibkr_gateway_version_warning_when_warning(monkeypatch, tmp_path) -> None:
    from chad.ops import health_monitor_rules as hmr
    _write_cache(tmp_path, "warning", 1042, 3)
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    findings: list = []
    hmr.rule_ibkr_gateway_version(findings)
    r20 = [f for f in findings if f.rule_id == "R20"]
    assert r20 and r20[0].severity == "WARNING"


def test_rule_ibkr_gateway_version_critical_when_stale(monkeypatch, tmp_path) -> None:
    from chad.ops import health_monitor_rules as hmr
    _write_cache(tmp_path, "stale", 1037, 8)
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    findings: list = []
    hmr.rule_ibkr_gateway_version(findings)
    r20 = [f for f in findings if f.rule_id == "R20"]
    assert r20 and r20[0].severity == "CRITICAL"


def test_rule_ibkr_gateway_version_warning_when_detection_unknown(monkeypatch, tmp_path) -> None:
    from chad.ops import health_monitor_rules as hmr
    _write_cache(tmp_path, "unknown", 0, None)
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    findings: list = []
    hmr.rule_ibkr_gateway_version(findings)
    r20 = [f for f in findings if f.rule_id == "R20"]
    assert r20 and r20[0].severity == "WARNING"
