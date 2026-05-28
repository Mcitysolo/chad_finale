"""Unit tests for chad/validators/port_binding.py (PORT-BINDING-1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chad.validators import port_binding as pb


def test_status_server_default_is_localhost():
    src = (pb.REPO_ROOT / "chad/web/status_server.py").read_text(encoding="utf-8")
    m = re.search(r'CHAD_STATUS_HOST"\s*,\s*"([^"]+)"', src)
    assert m, "could not find CHAD_STATUS_HOST default in status_server.py"
    assert m.group(1) == "127.0.0.1"


def test_metrics_server_default_is_localhost():
    src = (pb.REPO_ROOT / "chad/ops/metrics_server.py").read_text(encoding="utf-8")
    m = re.search(r'CHAD_METRICS_HOST"\s*,\s*"([^"]+)"', src)
    assert m, "could not find CHAD_METRICS_HOST default in metrics_server.py"
    assert m.group(1) == "127.0.0.1"


def test_audit_covers_9619_and_9620_and_9618():
    audits = pb.run_audit(live_check=False)
    ports = {a.port for a in audits}
    assert ports == {9618, 9619, 9620}


def test_validator_exits_0_when_all_defaults_localhost(monkeypatch):
    audits = pb.run_audit(live_check=False)
    code, report = pb.evaluate(audits)
    assert code == 0, report["failures"]
    # 9618 should appear in warnings (operator-domain systemd arg).
    assert any(w["port"] == 9618 for w in report["warnings"])


def test_validator_exits_2_when_default_flipped_to_zero(monkeypatch, tmp_path):
    # Inject a fake target with a non-localhost default by patching TARGETS
    # and pointing the source path at a tmpdir file we author.
    fake_src = tmp_path / "fake_server.py"
    fake_src.write_text('CHAD_FAKE_HOST", "0.0.0.0")\n', encoding="utf-8")

    original_repo_root = pb.REPO_ROOT
    monkeypatch.setattr(pb, "REPO_ROOT", tmp_path)

    fake_targets = [(7777, "fake_owner", "fake_server.py",
                     r'CHAD_FAKE_HOST"\s*,\s*"([^"]+)"', "CHAD_FAKE_HOST")]
    monkeypatch.setattr(pb, "TARGETS", fake_targets)

    audits = pb.run_audit(live_check=False)
    assert audits[0].port == 7777
    assert audits[0].code_default_host == "0.0.0.0"
    assert audits[0].is_localhost is False

    code, report = pb.evaluate(audits)
    assert code == 2
    assert any(f["port"] == 7777 for f in report["failures"])

    # Restore for sanity
    monkeypatch.setattr(pb, "REPO_ROOT", original_repo_root)


def test_allowlist_passes_with_warning(monkeypatch, tmp_path):
    fake_src = tmp_path / "fake_server.py"
    fake_src.write_text('CHAD_FAKE_HOST", "0.0.0.0")\n', encoding="utf-8")

    monkeypatch.setattr(pb, "REPO_ROOT", tmp_path)
    fake_targets = [(7778, "fake_with_reason", "fake_server.py",
                     r'CHAD_FAKE_HOST"\s*,\s*"([^"]+)"', "CHAD_FAKE_HOST")]
    monkeypatch.setattr(pb, "TARGETS", fake_targets)
    monkeypatch.setattr(pb, "ALLOWLIST", {7778: "audit-approved external Prom scrape"})

    audits = pb.run_audit(live_check=False)
    code, report = pb.evaluate(audits)
    assert code == 0  # allowlisted
    assert any(w.get("port") == 7778 for w in report["warnings"])


def test_main_exit_code_0_when_localhost(capsys):
    rc = pb.main(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "port_binding.v1" in out


# ---------------------------------------------------------------------------
# Allowlist-file overlay tests (Decision 3 / PORT-BINDING-1 hardening)
# ---------------------------------------------------------------------------

def _write_allowlist(tmp_path: Path, *, port: int, service: str = "chad-dashboard",
                     reason: str = "nginx-fronted",
                     schema_version: str = "port_binding_allowlist.v1") -> Path:
    doc = {
        "schema_version": schema_version,
        "comment": "test fixture",
        "allowlist": {
            service: {
                "port": port,
                "bind": "0.0.0.0",
                "reason": reason,
                "reviewed_utc": "2026-05-27",
                "reviewed_by": "test",
            }
        },
    }
    fp = tmp_path / "port_binding_allowlist.json"
    fp.write_text(json.dumps(doc), encoding="utf-8")
    return fp


def test_allowlist_schema_version_present():
    src = (pb.REPO_ROOT / "config" / "port_binding_allowlist.json").read_text(encoding="utf-8")
    doc = json.loads(src)
    assert doc.get("schema_version") == pb.ALLOWLIST_SCHEMA_VERSION


def test_allowlist_chad_dashboard_passes_with_warning(monkeypatch, tmp_path):
    fake_src = tmp_path / "fake_dashboard.py"
    fake_src.write_text('CHAD_DASH_HOST", "0.0.0.0")\n', encoding="utf-8")

    monkeypatch.setattr(pb, "REPO_ROOT", tmp_path)
    fake_targets = [(8765, "chad-dashboard (fake)", "fake_dashboard.py",
                     r'CHAD_DASH_HOST"\s*,\s*"([^"]+)"', "CHAD_DASH_HOST")]
    monkeypatch.setattr(pb, "TARGETS", fake_targets)

    allowlist_fp = _write_allowlist(tmp_path, port=8765, service="chad-dashboard")
    file_allowlist = pb.load_allowlist_file(allowlist_fp)
    audits = pb.run_audit(live_check=False)
    code, report = pb.evaluate(audits, file_allowlist=file_allowlist)
    assert code == 0
    matches = [w for w in report["warnings"] if w.get("port") == 8765]
    assert matches, report
    assert matches[0]["allowlist_source"] == "file"
    assert "chad-dashboard" in matches[0]["allowlist_reason"]


def test_unknown_service_with_zero_bind_fails(monkeypatch, tmp_path):
    fake_src = tmp_path / "fake_unknown.py"
    fake_src.write_text('CHAD_UNK_HOST", "0.0.0.0")\n', encoding="utf-8")

    monkeypatch.setattr(pb, "REPO_ROOT", tmp_path)
    fake_targets = [(7799, "unlisted-service", "fake_unknown.py",
                     r'CHAD_UNK_HOST"\s*,\s*"([^"]+)"', "CHAD_UNK_HOST")]
    monkeypatch.setattr(pb, "TARGETS", fake_targets)

    # Allowlist exists but covers a different port — unknown service must still fail.
    allowlist_fp = _write_allowlist(tmp_path, port=8765, service="chad-dashboard")
    file_allowlist = pb.load_allowlist_file(allowlist_fp)
    audits = pb.run_audit(live_check=False)
    code, report = pb.evaluate(audits, file_allowlist=file_allowlist)
    assert code == pb.EXIT_NONLOCAL_DEFAULT
    assert any(f["port"] == 7799 for f in report["failures"])


def test_chad_backend_with_zero_bind_fails_until_d3_patched(monkeypatch, tmp_path):
    """Until the Channel-1 install runbook flips uvicorn --host to 127.0.0.1,
    chad-backend (port 9618) remains an operator-domain warning (code-side
    cannot enforce) — but if a Python-side default is ever introduced that
    binds 9618 to 0.0.0.0 without an allowlist entry, the validator must
    refuse the commit.
    """
    fake_src = tmp_path / "fake_backend.py"
    fake_src.write_text('CHAD_BACKEND_HOST", "0.0.0.0")\n', encoding="utf-8")

    monkeypatch.setattr(pb, "REPO_ROOT", tmp_path)
    fake_targets = [(9618, "chad-backend (fake python default)", "fake_backend.py",
                     r'CHAD_BACKEND_HOST"\s*,\s*"([^"]+)"', "CHAD_BACKEND_HOST")]
    monkeypatch.setattr(pb, "TARGETS", fake_targets)

    audits = pb.run_audit(live_check=False)
    # No allowlist entry for 9618 — must fail.
    code, report = pb.evaluate(audits, file_allowlist={})
    assert code == pb.EXIT_NONLOCAL_DEFAULT
    assert any(f["port"] == 9618 for f in report["failures"])


def test_validator_returns_structured_json(capsys):
    rc = pb.main(["--check"])
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["validator"] == "port_binding.v1"
    assert isinstance(doc["audits"], list)
    assert isinstance(doc["failures"], list)
    assert isinstance(doc["warnings"], list)
    # Allowlist surface is exposed in the report
    assert "allowlist_file_loaded" in doc
    assert "allowlist_file_entries" in doc
    # Real repo: no failures expected
    assert rc == 0


def test_allowlist_file_missing_returns_empty():
    bogus = Path("/tmp/does-not-exist-port-allowlist.json")
    assert pb.load_allowlist_file(bogus) == {}


def test_allowlist_file_schema_mismatch_raises(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text(json.dumps({"schema_version": "wrong", "allowlist": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        pb.load_allowlist_file(fp)
