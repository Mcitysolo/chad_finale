"""Unit tests for chad/validators/port_binding.py (PORT-BINDING-1)."""

from __future__ import annotations

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
