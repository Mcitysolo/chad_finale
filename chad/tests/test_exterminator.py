"""Regression locks for the Exterminator Read-Only Sentinel (SSOT v9.0 §7)."""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

ext = importlib.import_module("chad.ops.exterminator")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixed_clock():
    fixed = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    return lambda: fixed


@pytest.fixture()
def quiet_providers():
    return {
        "git_provider": lambda: {
            "head": "abc123", "branch": "main", "clean": True,
            "dirty_files": [], "tags": [], "error": None,
        },
        "systemctl_provider": lambda: {"failed_units": [], "error": None},
        "fills_provider": lambda: {"available": False, "source": None, "fills": []},
        "bar_freshness_provider": lambda: {"matches": 0, "files_scanned": 0, "samples": []},
        "ml_shadow_provider": lambda: {"matches": 0, "files_scanned": 0, "samples": []},
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _healthy_runtime(runtime_dir: Path, ts_utc: str = "2026-05-05T11:59:00Z") -> None:
    _write_json(runtime_dir / "scr_state.json", {
        "schema_version": "scr_state.v1",
        "state": "WARMUP",
        "sizing_factor": 0.10,
        "ts_utc": ts_utc, "ttl_seconds": 600,
        "stats": {"total_trades": 18, "effective_trades": 7,
                  "excluded_untrusted": 6, "excluded_nonfinite": 0, "excluded_manual": 0,
                  "win_rate": 0.71, "sharpe_like": 0.5, "max_drawdown": -6.2, "total_pnl": 4.88},
    })
    _write_json(runtime_dir / "live_readiness.json", {
        "schema_version": "live_readiness_state.v1",
        "ready_for_live": False,
        "requirements_remaining": ["epoch_2_warmup_in_progress", "operator_GO"],
        "epoch_metadata": {"epoch": "CHAD_v8.9_Paper_Epoch_2"},
    })
    _write_json(runtime_dir / "epoch_state.json", {
        "schema_version": "epoch_state.v1",
        "active_epoch": "CHAD_v8.9_Paper_Epoch_2", "paper_only": True, "ready_for_live": False,
    })
    _write_json(runtime_dir / "profit_lock_state.json", {
        "schema_version": "profit_lock_state.v1",
        "mode": "NORMAL", "sizing_factor": 1.0, "profit_lock_active": False,
        "daily_loss_today": 0.0, "daily_loss_limit_hit": False, "stop_new_entries": False,
        "ts_utc": ts_utc, "ttl_seconds": 600,
    })
    _write_json(runtime_dir / "stop_bus.json", {
        "schema_version": "stop_bus.v1", "active": False,
    })
    _write_json(runtime_dir / "reconciliation_state.json", {
        "schema_version": "reconciliation_state.v1",
        "status": "GREEN", "worst_diff": 0.0, "mismatches": [], "drifts": [],
        "ts_utc": ts_utc, "ttl_seconds": 600,
    })
    _write_json(runtime_dir / "regime_state.json", {
        "schema_version": "regime_state.v1", "regime": "trending_bull", "confidence": 0.79,
        "ts_utc": ts_utc, "ttl_seconds": 600,
    })
    _write_json(runtime_dir / "portfolio_snapshot.json", {
        "schema_version": "portfolio_snapshot.v1",
        "ibkr": 182763.28, "kraken": 184.58, "coinbase": 0.0,
    })
    _write_json(runtime_dir / "strategy_health.json", {
        "schema_version": "strategy_health.v1",
        "strategies": {"alpha": {"samples": 88, "score": 0.7}, "beta": {"samples": 4, "score": 0.5}},
    })
    _write_json(runtime_dir / "winner_scaling.json", {
        "schema_version": "winner_scaling.v1",
        "ts_utc": ts_utc,
        "strategies": {"alpha": {"multiplier": 1.0}, "beta": {"multiplier": 1.0}},
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exterminator_generates_reports_without_runtime_mutation(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    reports_dir = tmp_path / "reports" / "exterminator"
    _healthy_runtime(runtime_dir)

    runtime_snapshot = {p.name: p.read_bytes() for p in runtime_dir.iterdir() if p.is_file()}

    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=reports_dir,
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    json_path, md_path = sentinel.write_reports(report)

    assert json_path.exists() and md_path.exists()
    # Runtime files must be byte-identical post-run.
    for name, blob in runtime_snapshot.items():
        assert (runtime_dir / name).read_bytes() == blob, f"runtime mutated: {name}"
    assert report["runtime_files_modified"] == []
    assert report["services_restarted"] is False


def test_exterminator_handles_missing_runtime_file(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True)  # empty
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    missing_findings = [f for f in report["findings"] if f["id"] == "EX004"]
    assert missing_findings, "EX004 missing-file findings expected"
    assert all(f["severity"] == "WARNING" for f in missing_findings)


def test_exterminator_handles_invalid_json(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "scr_state.json").write_text("{not json", encoding="utf-8")
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    invalid = [f for f in report["findings"] if f["id"] == "EX005"]
    assert invalid, "EX005 invalid-JSON finding expected"
    assert any(f["severity"] == "CRITICAL" for f in invalid)


def test_exterminator_detects_dirty_git_from_injected_provider(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    _healthy_runtime(runtime_dir)
    providers = dict(quiet_providers)
    providers["git_provider"] = lambda: {
        "head": "deadbee", "branch": "main", "clean": False,
        "dirty_files": [" M chad/ops/x.py", "?? new_file.py"],
        "tags": [], "error": None,
    }
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **providers,
    )
    report = sentinel.run()
    dirty = [f for f in report["findings"] if f["id"] == "EX002"]
    assert dirty, "EX002 dirty-git finding expected"
    assert dirty[0]["severity"] == "WARNING"
    assert report["repo"]["git_clean"] is False
    assert report["repo"]["dirty_files"]


def test_exterminator_classifies_scr_warmup_as_info(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    _healthy_runtime(runtime_dir)  # SCR=WARMUP by default
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    warmup_findings = [f for f in report["findings"] if f["id"] == "EX020"]
    assert warmup_findings, "EX020 SCR-WARMUP finding expected"
    assert warmup_findings[0]["severity"] == "INFO"
    # And critical count from this scenario must be zero.
    assert report["counts"]["critical"] == 0


def test_exterminator_detects_halt_boost_contradiction(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    _healthy_runtime(runtime_dir)
    _write_json(runtime_dir / "strategy_health.json", {
        "schema_version": "strategy_health.v1",
        "strategies": {
            "delta": {"samples": 82, "score": 0.4, "edge_decay_halt": True},
            "alpha": {"samples": 88, "score": 0.7},
        },
    })
    _write_json(runtime_dir / "winner_scaling.json", {
        "schema_version": "winner_scaling.v1",
        "ts_utc": "2026-05-05T11:59:00Z",
        "strategies": {
            "delta": {"multiplier": 1.5},
            "alpha": {"multiplier": 1.0},
        },
    })
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    contradiction = [f for f in report["findings"] if f["id"] == "EX010"]
    assert contradiction, "EX010 halt+boost contradiction expected"
    assert contradiction[0]["severity"] == "WARNING"
    assert "delta" in contradiction[0]["evidence"]["boosts_for_halted"]


def test_exterminator_classifies_reconciliation_green_with_drifts_as_warning(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    _healthy_runtime(runtime_dir)
    _write_json(runtime_dir / "reconciliation_state.json", {
        "schema_version": "reconciliation_state.v1",
        "status": "GREEN", "worst_diff": 0.0, "mismatches": [],
        "drifts": [{"symbol": "BAC"}, {"symbol": "CVX"}],
        "ts_utc": "2026-05-05T11:59:00Z", "ttl_seconds": 600,
    })
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    drift = [f for f in report["findings"] if f["id"] == "EX009"]
    assert drift, "EX009 GREEN-with-drifts finding expected"
    assert drift[0]["severity"] == "WARNING"


def test_exterminator_detects_placeholder_fill_price(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    _healthy_runtime(runtime_dir)
    providers = dict(quiet_providers)
    providers["fills_provider"] = lambda: {
        "available": True, "source": "test_fixture",
        "fills": [
            {"symbol": "SPY", "fill_price": 100.0, "ts_utc": "2026-05-05T10:00:00Z", "trade_id": "t1"},
            {"symbol": "QQQ", "fill_price": 425.5},
        ],
    }
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **providers,
    )
    report = sentinel.run()
    placeholders = [f for f in report["findings"] if f["id"] == "EX006" and f["severity"] == "WARNING"]
    assert placeholders, "EX006 placeholder-fill finding expected"
    assert placeholders[0]["evidence"]["placeholder_fills"][0]["symbol"] == "SPY"


def test_exterminator_does_not_restart_services_or_patch_files():
    src = Path(ext.__file__).read_text(encoding="utf-8")
    forbidden = (
        "ib_insync",
        "ibapi",
        "systemctl restart",
        "systemctl stop",
        "systemctl start ",
        "systemctl enable",
        "systemctl disable",
        "systemctl reload",
        "shutil.rmtree",
        "git commit",
        "git push",
        "git add ",
        "git tag ",
        "openai",
        "anthropic",
    )
    for token in forbidden:
        assert token not in src, f"forbidden token in exterminator: {token!r}"


def test_exterminator_report_schema_required_fields(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    _healthy_runtime(runtime_dir)
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=tmp_path / "reports",
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    required_top = {"schema_version", "generated_at_utc", "mode", "repo", "runtime_posture",
                    "findings", "counts", "read_only_confirmed", "runtime_files_modified", "services_restarted"}
    assert required_top <= set(report.keys())
    assert report["schema_version"] == "exterminator_report.v1"
    assert report["mode"] == "read_only"
    assert report["read_only_confirmed"] is True
    assert report["services_restarted"] is False
    assert report["runtime_files_modified"] == []
    repo_keys = {"head", "branch", "git_clean", "dirty_files", "tags"}
    assert repo_keys <= set(report["repo"].keys())
    posture_keys = {"epoch", "paper_only", "live_readiness", "scr_state", "scr_sizing_factor",
                    "profit_lock_mode", "stop_bus_active", "reconciliation_status"}
    assert posture_keys <= set(report["runtime_posture"].keys())
    counts_keys = {"critical", "warning", "info"}
    assert counts_keys <= set(report["counts"].keys())
    finding_keys = {"id", "severity", "category", "title", "summary", "evidence",
                    "auto_fix_allowed", "recommended_next_action", "requires_operator"}
    for f in report["findings"]:
        assert finding_keys <= set(f.keys())
        assert f["severity"] in ("INFO", "WARNING", "CRITICAL")
        assert f["id"].startswith("EX")


def test_exterminator_outputs_markdown_and_json(tmp_path, fixed_clock, quiet_providers):
    runtime_dir = tmp_path / "runtime"
    reports_dir = tmp_path / "reports" / "exterminator"
    _healthy_runtime(runtime_dir)
    sentinel = ext.Exterminator(
        repo_root=tmp_path, runtime_dir=runtime_dir, reports_dir=reports_dir,
        clock=fixed_clock, **quiet_providers,
    )
    report = sentinel.run()
    json_path, md_path = sentinel.write_reports(report)
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"
    assert json_path.parent == md_path.parent == reports_dir
    written_json = json.loads(json_path.read_text(encoding="utf-8"))
    md_text = md_path.read_text(encoding="utf-8")
    # Both contain the same finding IDs.
    json_ids = {f["id"] for f in written_json["findings"]}
    for fid in json_ids:
        assert fid in md_text, f"finding {fid} missing from markdown"
    # Markdown must contain required sections.
    assert "CHAD Exterminator Stage 1" in md_text
    assert "Posture Summary" in md_text
    assert "Findings" in md_text
    assert "Next Actions" in md_text
    assert "read-only" in md_text.lower() and "no" in md_text.lower()
