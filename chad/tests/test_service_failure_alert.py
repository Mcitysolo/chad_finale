"""Tests for chad/ops/service_failure_alert.py (Decision 3).

All tests run with dry_run=True / via the library entry point — no real
Telegram message is sent and no real journalctl read is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.ops import service_failure_alert as sfa


# ---------------------------------------------------------------------------
# Library-level run() tests (no subprocess calls reach the network)
# ---------------------------------------------------------------------------

def _stub_journal_ok(monkeypatch, lines: list[str] | None = None) -> None:
    monkeypatch.setattr(
        sfa, "_read_journal_tail",
        lambda unit, n: (lines if lines is not None else ["line1", "line2"], None),
    )


def _stub_systemctl(monkeypatch, value: str = "failed") -> None:
    monkeypatch.setattr(sfa, "_systemctl_active", lambda unit: value)


def test_payload_schema_version_v2(monkeypatch, tmp_path):
    # P0A-A1: schema bumped v1 -> v2 (adds telegram_sent / telegram_delivery_status /
    # delivery_error so per-incident delivery is auditable).
    _stub_journal_ok(monkeypatch)
    _stub_systemctl(monkeypatch)
    res = sfa.run(
        failed_unit="chad-options-chain-refresh.service",
        severity="HIGH",
        journal_tail_n=10,
        include_runtime_snapshot=False,
        dry_run=True,
        artifact_dir=tmp_path,
    )
    assert res.payload["schema_version"] == "service_failure_alert.v2"
    assert res.exit_code == sfa.EXIT_OK


def test_non_chad_unit_refused_with_exit_2(tmp_path):
    res = sfa.run(
        failed_unit="systemd-resolved.service",
        dry_run=True,
        artifact_dir=tmp_path,
    )
    assert res.exit_code == sfa.EXIT_INVALID_UNIT
    assert "refused" in res.payload.get("error", "")


def test_journal_read_failure_emits_artifact_with_exit_3(monkeypatch, tmp_path):
    monkeypatch.setattr(sfa, "_read_journal_tail", lambda u, n: ([], "journalctl_timeout"))
    _stub_systemctl(monkeypatch)
    res = sfa.run(
        failed_unit="chad-backend.service",
        dry_run=True,
        artifact_dir=tmp_path,
    )
    assert res.exit_code == sfa.EXIT_JOURNAL_FAILED
    # Artifact still written
    assert res.artifact_path.exists()
    doc = json.loads(res.artifact_path.read_text(encoding="utf-8"))
    assert doc["journal_error"] == "journalctl_timeout"
    assert doc["journal_tail"] == []


def test_runtime_snapshot_includes_only_metadata_not_contents(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_systemctl(monkeypatch)
    res = sfa.run(
        failed_unit="chad-options-chain-refresh.service",
        include_runtime_snapshot=True,
        dry_run=True,
        artifact_dir=tmp_path,
    )
    snap = res.payload.get("runtime_snapshot")
    assert isinstance(snap, dict) and snap, "expected runtime_snapshot to be populated"
    for fname, meta in snap.items():
        # The schema permits only these keys — never a "contents"/"data" key.
        permitted = {"present", "mtime_utc", "size_bytes"}
        assert set(meta.keys()).issubset(permitted), f"{fname} leaked: {meta}"
        assert "contents" not in meta and "data" not in meta and "body" not in meta


def test_dry_run_does_not_send_telegram(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_systemctl(monkeypatch)

    sentinel = {"called": False}

    def _boom(*a, **kw):
        sentinel["called"] = True
        raise AssertionError("telegram should NOT be called during --dry-run")

    monkeypatch.setattr(
        "chad.utils.telegram_notify.notify", _boom, raising=False,
    )

    res = sfa.run(
        failed_unit="chad-backend.service",
        dry_run=True,
        artifact_dir=tmp_path,
    )
    assert res.exit_code == sfa.EXIT_OK
    assert sentinel["called"] is False


def test_artifact_path_under_reports_service_failures(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_systemctl(monkeypatch)

    # Override artifact_dir but verify the canonical default sits under reports/.
    expected_default = sfa.REPO_ROOT / "reports" / "service_failures"
    assert sfa.ARTIFACT_DIR == expected_default

    res = sfa.run(
        failed_unit="chad-options-chain-refresh.service",
        dry_run=True,
        artifact_dir=tmp_path,
    )
    assert res.artifact_path.parent == tmp_path
    assert res.artifact_path.name.endswith("__chad-options-chain-refresh.service.json")


def test_unit_name_template_expansion(monkeypatch, tmp_path):
    """The failed_unit value is captured verbatim into the payload so the
    template expansion %i → unit-name in the systemd template lands cleanly."""
    _stub_journal_ok(monkeypatch)
    _stub_systemctl(monkeypatch)
    for unit in (
        "chad-options-chain-refresh.service",
        "chad-backend.service",
        "chad-reconciliation-publisher.service",
        "chad-live-loop.service",
        "chad-shadow-status.timer",
    ):
        res = sfa.run(
            failed_unit=unit,
            dry_run=True,
            artifact_dir=tmp_path,
        )
        assert res.exit_code == sfa.EXIT_OK
        assert res.payload["failed_unit"] == unit
        assert unit in res.artifact_path.name
