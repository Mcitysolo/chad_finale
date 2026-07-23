"""Regression locks for the Exterminator Sentinel — Stage 1.

Two families of test live here:

1. ANTI-AUTO-HEALING LOCKS. Stage 1 observes and never repairs. These tests
   pin that contract by construction, not by convention: the sentinel may write
   exactly two paths (its own report + history) and every other byte under
   runtime/ must be identical after a run. If someone later adds a "helpful"
   auto-fix, these fail.

2. CHECK SEMANTICS. Each of the 8 checks is exercised for its ok/warn/fail
   verdicts, with the XOV-2345 independent-leg rule (check 4) locked hardest —
   including the specific false-flat scenario that blinded the same-source legs.
"""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sentinel_mod = importlib.import_module("chad.ops.exterminator_sentinel")

NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
REAL_CONFIG = Path(__file__).resolve().parents[2] / "config" / "exterminator.json"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ndjson(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture()
def clock():
    return lambda: NOW


@pytest.fixture()
def quiet_providers():
    return {
        "systemctl_provider": lambda query: {"failed_units": [], "error": None, "query": list(query)},
        "git_provider": lambda: {"head": "abc123", "branch": "main", "entries": [], "error": None},
        "notifier": lambda message, dedupe_key: False,
        # W3B-5 (EXS9): every unit healthy — started 1h ago, newest relevant
        # commit 2 days ago, so processes are current.
        "service_uptime_provider": lambda unit: {
            "unit": unit, "active_state": "active",
            "active_enter_unix": (NOW - timedelta(hours=1)).timestamp(),
            "main_pid": 4242, "error": None,
        },
        "code_timestamp_provider": lambda paths: {
            "paths": list(paths),
            "commit_unix": (NOW - timedelta(days=2)).timestamp(),
            "commit_hash": "abc123def456", "error": None,
        },
    }


def _fresh_runtime(runtime: Path, *, ts: datetime | None = None) -> None:
    """A healthy runtime: guard agrees with the independent collector."""
    ts = ts or (NOW - timedelta(seconds=30))
    _write_json(runtime / "positions_snapshot.json", {
        "source": "ibkr_portfolio_collector_v2", "ts_utc": _iso(ts), "ttl_seconds": 300,
        "positions_count": 2, "cash": 1000.0,
        "positions": [
            {"symbol": "UNH", "position": 273.0, "secType": "STK", "currency": "USD", "avgCost": 420.7},
            {"symbol": "V", "position": 181.0, "secType": "STK", "currency": "USD", "avgCost": 347.8},
        ],
    })
    # Dual-booked exactly as the live guard is: gamma|SYM AND broker_sync|SYM
    # describe the SAME shares. Summing them would invent a 2.0x phantom.
    _write_json(runtime / "position_guard.json", {
        "_version": 1,
        "_written_by": "test",
        "gamma|UNH": {"symbol": "UNH", "quantity": 273.0, "side": "BUY", "open": True, "strategy": "gamma"},
        "broker_sync|UNH": {"symbol": "UNH", "quantity": 273.0, "side": "BUY", "open": True, "strategy": "broker_sync"},
        "gamma|V": {"symbol": "V", "quantity": 181.0, "side": "BUY", "open": True, "strategy": "gamma"},
        "broker_sync|V": {"symbol": "V", "quantity": 181.0, "side": "BUY", "open": True, "strategy": "broker_sync"},
    })
    _write_json(runtime / "position_guard_drift.json", {
        "schema_version": "position_guard_drift.v3", "ts_utc": _iso(ts), "ttl_seconds": 360,
        "drift_count": 0, "info_count": 0, "excluded_symbols": [], "snapshot_generation": 1,
        "counts_by_kind": {}, "drifts": [],
    })
    _write_json(runtime / "trade_closer_state.json", {
        "processed_fill_ids": [], "saved_at_utc": _iso(ts),
        "queues": [
            {"strategy": "gamma", "symbol": "UNH", "lots": [{"fill_id": "a", "side": "BUY", "quantity": 273.0}]},
            {"strategy": "gamma", "symbol": "V", "lots": [{"fill_id": "b", "side": "BUY", "quantity": 181.0}]},
        ],
    })
    _write_json(runtime / "scr_state.json", {
        "schema_version": "scr_state.v1", "ts_utc": _iso(ts), "ttl_seconds": 180,
        "state": "WARMUP", "sizing_factor": 0.1, "stats": {"effective_trades": 67},
    })
    _write_json(runtime / "var_state.json", {
        "schema_version": "var_state.v1", "ts_utc": _iso(ts), "ttl_seconds": 3600,
        "status": "OK", "method": "historical",
    })
    _write_json(runtime / "exit_overlay_heartbeat.json", {
        "schema_version": "exit_overlay_heartbeat.v1", "ts_utc": _iso(ts), "ttl_seconds": 900,
        "mode": "shadow", "evaluated": 2, "would_close": 0, "healthy": True,
    })
    _write_json(runtime / "kraken_prices.json", {"ts_utc": _iso(ts), "ttl_seconds": 30, "prices": {}, "connected": True})
    _write_json(runtime / "ibkr_bars_cache.json", {"ts_utc": _iso(ts), "bar_count": 10, "symbols": []})
    _write_ndjson(runtime / "equity_history.ndjson", [
        {"date_utc": "2026-07-15", "ts_utc": _iso(ts), "total_equity_cad": 994381.63, "schema_version": "equity_history.v2"},
    ])
    _write_json(runtime / "live_readiness.json", {
        "schema_version": "live_readiness_state.v1", "ts_utc": _iso(ts), "ttl_seconds": 604800,
        "ready_for_live": False, "latest_report_path": "x", "latest_report_sha256": "y",
    })
    _write_json(runtime / "stop_bus.json", {"schema_version": "stop_bus.v1", "active": False})
    _write_json(runtime / "epoch_state.json", {"schema_version": "epoch_state.v1", "active_epoch": "Epoch_3"})
    # W3B-1: the two 120s-cadence publisher artifacts gained EXS1 rows.
    _write_json(runtime / "drawdown_state.json", {
        "schema_version": "drawdown_state.v1", "ts_utc": _iso(ts), "ttl_seconds": 300,
        "status": "ok", "drawdown_pct": -0.6, "halt": False, "enforcement_active": False,
    })
    _write_json(runtime / "ibkr_watchdog_last.json", {
        "ts_unix": ts.timestamp(), "ok": True, "ttl_seconds": 120,
        "consecutive_failures": 0,
    })
    # W5A-5 (DQ2/EXS10): a healthy runtime has broker time ≈ box time (no skew).
    _write_json(runtime / "ibkr_status.json", {
        "ts_utc": _iso(ts), "server_time_iso": _iso(ts), "api_ms": 40.0,
    })
    # W4A-1 (INCIDENT-0723 inheritance b): the fuse-box heartbeat gained an
    # EXS1 row — a healthy runtime includes it fresh, all-modes-off.
    _write_json(runtime / "fuse_box_state.json", {
        "schema_version": "fuse_box_state.v1", "ts_utc": _iso(ts), "ttl_seconds": 180,
        "modes": {"lc2": "off", "lc3": "off", "lc5": "off", "dq": "off"},
        "fuses": [],
    })
    # W5B-5: the allocator heartbeat gained an EXS1 row — a healthy runtime
    # includes it fresh, flag-off (book/limits null; standing_findings always
    # present, since they bound what the evidence may claim either way).
    _write_json(runtime / "portfolio_allocator_state.json", {
        "schema_version": "allocator_state.v1", "ts_utc": _iso(ts), "ttl_seconds": 180,
        "mode": "off", "active": False,
        "cycle": {"intents_evaluated": 0, "bypassed": 0, "would_approve": 0,
                  "would_resize": 0, "would_reject": 0, "errors": 0, "by_limit": {}},
        "book": None, "limits": None,
        "standing_findings": [{"id": "W5B-SF1"}],
    })


def _make(tmp_path: Path, clock, quiet_providers, **overrides):
    runtime = tmp_path / "runtime"
    return sentinel_mod.ExterminatorSentinel(
        repo_root=tmp_path,
        runtime_dir=runtime,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "runtime" / "reports",
        config_path=REAL_CONFIG,
        clock=clock,
        **{**quiet_providers, **overrides},
    )


# ---------------------------------------------------------------------------
# 1. Anti-auto-healing locks
# ---------------------------------------------------------------------------


def test_run_does_not_mutate_any_runtime_byte(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    before = {p.name: p.read_bytes() for p in runtime.iterdir() if p.is_file()}

    s = _make(tmp_path, clock, quiet_providers)
    report = s.run()
    s.write_reports(report)

    for name, blob in before.items():
        assert (runtime / name).read_bytes() == blob, f"sentinel mutated runtime file: {name}"
    assert report["runtime_files_modified"] == []
    assert report["services_restarted"] is False
    assert report["read_only_confirmed"] is True
    assert report["mode"] == "read_only"


def test_sentinel_writes_exactly_two_paths_and_nothing_else(tmp_path, clock, quiet_providers):
    """The write allowlist IS the read-only contract."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    before = {p for p in runtime.rglob("*") if p.is_file()}

    s = _make(tmp_path, clock, quiet_providers)
    latest, history = s.report_paths()
    s.write_reports(s.run())

    after = {p for p in runtime.rglob("*") if p.is_file()}
    created = after - before
    assert created == {latest, history}, f"unexpected writes: {created - {latest, history}}"


def test_no_mutation_tokens_in_source():
    """Anti-auto-healing: the module must not contain a repair verb."""
    src = Path(sentinel_mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "systemctl restart", "systemctl stop", "systemctl start", "systemctl enable",
        "systemctl disable", "systemctl reload", "systemctl reset-failed",
        "git commit", "git push", "git add ", "git tag ", "git checkout", "git reset",
        "shutil.rmtree", "os.remove", "os.unlink", "Path.unlink", ".unlink(",
        "ib_insync", "ibapi", "openai", "anthropic",
    )
    for token in forbidden:
        assert token not in src, f"forbidden mutation token in sentinel: {token!r}"


def test_subprocess_calls_are_read_only_queries():
    """Every subprocess argv in the module must be a read-only verb."""
    src = Path(sentinel_mod.__file__).read_text(encoding="utf-8")
    read_only_git = ("rev-parse", "status")
    for line in src.splitlines():
        if '"git"' in line and "subprocess" not in line:
            continue
    # The git provider may only ever run rev-parse/status.
    assert '"--porcelain"' in src
    for verb in ("commit", "push", "merge", "rebase", "clean"):
        assert f'"{verb}"' not in src, f"git {verb} must never appear"
    # systemd access is a list-units query only.
    assert "list-units" in src
    assert "--state=failed" in src
    for verb in ("start", "stop", "restart", "reset-failed"):
        assert f'"{verb}"' not in src, f"systemctl {verb} must never appear"
    assert all(v in src for v in read_only_git)


def test_defaults_refuse_real_runtime_under_pytest(tmp_path):
    """Test-write leak guard: never append to the real runtime history."""
    with pytest.raises(RuntimeError, match="reports_dir must be explicit"):
        sentinel_mod.ExterminatorSentinel(repo_root=tmp_path)


def test_failing_check_does_not_trigger_any_repair(tmp_path, clock, quiet_providers):
    """A fail is reported and notified — never fixed."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "var_state.json", {
        "schema_version": "var_state.v1", "ts_utc": _iso(NOW - timedelta(days=69)),
        "ttl_seconds": 3600, "status": "OK", "method": "historical",
    })
    before = {p.name: p.read_bytes() for p in runtime.iterdir() if p.is_file()}

    s = _make(tmp_path, clock, quiet_providers)
    report = s.run()
    assert report["overall_status"] == "fail"
    assert report["remedy_type"] == "NOTIFY_ONLY"
    for check in report["checks"]:
        assert check["remedy_type"] == "NOTIFY_ONLY"
    for name, blob in before.items():
        assert (runtime / name).read_bytes() == blob, f"sentinel repaired {name} instead of reporting it"


# ---------------------------------------------------------------------------
# 2. Report contract
# ---------------------------------------------------------------------------


def test_report_schema_and_nine_checks(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    report = _make(tmp_path, clock, quiet_providers).run()

    required = {"schema_version", "generated_at_utc", "mode", "stage", "overall_status",
                "counts", "checks", "read_only_confirmed", "runtime_files_modified",
                "services_restarted", "remedy_type"}
    assert required <= set(report)
    assert report["schema_version"] == "exterminator_sentinel.v1"
    assert report["stage"] == 1
    assert len(report["checks"]) == 10
    assert [c["check_id"] for c in report["checks"]] == [f"EXS{i}" for i in range(1, 11)]
    for check in report["checks"]:
        assert {"check_id", "name", "status", "title", "summary", "evidence", "remedy_type"} <= set(check)
        assert check["status"] in ("ok", "warn", "fail")
        assert check["evidence"] != {} or check["status"] == "ok"


def test_history_appends_one_line_per_run(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    s = _make(tmp_path, clock, quiet_providers)
    _, history = s.report_paths()

    s.write_reports(s.run())
    s.write_reports(s.run())

    lines = [ln for ln in history.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["schema_version"] == "exterminator_sentinel.v1"
    assert set(row["checks"]) == {f"EXS{i}" for i in range(1, 11)}


def test_latest_is_rewritten_not_appended(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    s = _make(tmp_path, clock, quiet_providers)
    latest, _ = s.report_paths()
    s.write_reports(s.run())
    s.write_reports(s.run())
    assert json.loads(latest.read_text(encoding="utf-8"))["schema_version"] == "exterminator_sentinel.v1"


def test_check_exception_is_contained(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    s = _make(tmp_path, clock, quiet_providers)
    s.check_failed_services = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    report = s.run()
    assert len(report["checks"]) == 10
    self_checks = [c for c in report["checks"] if c["check_id"] == "EXS999"]
    assert self_checks and self_checks[0]["status"] == "warn"


# ---------------------------------------------------------------------------
# 3. Check 1 — stale feeds
# ---------------------------------------------------------------------------


def test_stale_feeds_ok_when_fresh(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    assert _make(tmp_path, clock, quiet_providers).check_stale_feeds().status == "ok"


def test_stale_feeds_fails_on_var_state_breach(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "var_state.json", {
        "schema_version": "var_state.v1", "ts_utc": _iso(NOW - timedelta(days=69)),
        "ttl_seconds": 3600, "status": "OK", "method": "historical",
    })
    result = _make(tmp_path, clock, quiet_providers).check_stale_feeds()
    assert result.status == "fail"
    row = next(r for r in result.evidence["feeds"] if r["feed"] == "var_state")
    assert row["status"] == "fail"
    assert row["breach_ratio"] > 1000
    assert row["ttl_source"].startswith("artifact:")


def test_unverified_ttl_is_capped_at_warn(tmp_path, clock, quiet_providers):
    """An unratified TTL may never fail a gate — operator_verify is honest."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "ibkr_bars_cache.json", {"ts_utc": _iso(NOW - timedelta(days=30)), "bar_count": 0})
    result = _make(tmp_path, clock, quiet_providers).check_stale_feeds()
    row = next(r for r in result.evidence["feeds"] if r["feed"] == "bars")
    assert row["status"] == "warn"
    assert row["ttl_source"] == "operator_verify"
    assert "bars" in result.evidence["unverified_ttls"]


def test_missing_feed_warns_rather_than_fails(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    (runtime / "kraken_prices.json").unlink()
    result = _make(tmp_path, clock, quiet_providers).check_stale_feeds()
    row = next(r for r in result.evidence["feeds"] if r["feed"] == "kraken_ticks")
    assert row == {"feed": "kraken_ticks", "path": "runtime/kraken_prices.json",
                   "ttl_verified": True, "ttl_source": "artifact:ttl_seconds (kraken_prices.json declares ttl_seconds=30)",
                   "status": "warn", "reason": "missing"}


def test_artifact_ttl_overrides_config_ttl(tmp_path, clock, quiet_providers):
    """The publisher's declared TTL wins so config can never silently drift."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "scr_state.json", {
        "schema_version": "scr_state.v1", "ts_utc": _iso(NOW - timedelta(seconds=400)),
        "ttl_seconds": 100000, "state": "WARMUP", "sizing_factor": 0.1, "stats": {},
    })
    result = _make(tmp_path, clock, quiet_providers).check_stale_feeds()
    row = next(r for r in result.evidence["feeds"] if r["feed"] == "scr_state")
    assert row["ttl_seconds"] == 100000.0
    assert row["status"] == "ok"


# ---------------------------------------------------------------------------
# 4. Check 1 — R14 blind pattern
# ---------------------------------------------------------------------------


def test_blind_check_fails_when_overlay_watches_nothing(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "exit_overlay_heartbeat.json", {
        "schema_version": "exit_overlay_heartbeat.v1", "ts_utc": _iso(NOW - timedelta(seconds=30)),
        "ttl_seconds": 900, "mode": "shadow", "evaluated": 0, "would_close": 0, "healthy": True,
    })
    result = _make(tmp_path, clock, quiet_providers).check_stale_feeds()
    row = next(r for r in result.evidence["feeds"] if r["feed"] == "exit_overlay_blind_check")
    assert row["status"] == "fail"
    assert row["reason"] == "overlay_blind"
    assert row["broker_positions_held"] == 2
    assert result.status == "fail"


def test_blind_check_silent_when_truth_feed_stale(tmp_path, clock, quiet_providers):
    """Unproven input is silence, not an alert — do not cry wolf."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "exit_overlay_heartbeat.json", {
        "schema_version": "exit_overlay_heartbeat.v1", "ts_utc": _iso(NOW - timedelta(seconds=30)),
        "ttl_seconds": 900, "mode": "shadow", "evaluated": 0, "would_close": 0, "healthy": True,
    })
    _write_json(runtime / "positions_snapshot.json", {
        "source": "ibkr_portfolio_collector_v2", "ts_utc": _iso(NOW - timedelta(hours=4)),
        "ttl_seconds": 300, "positions": [{"symbol": "UNH", "position": 273.0, "secType": "STK"}],
    })
    row = next(r for r in _make(tmp_path, clock, quiet_providers).check_stale_feeds().evidence["feeds"]
               if r["feed"] == "exit_overlay_blind_check")
    assert row["status"] == "ok"
    assert row["reason"] == "truth_feed_stale_proves_nothing"


def test_blind_check_silent_when_broker_genuinely_flat(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "exit_overlay_heartbeat.json", {
        "schema_version": "exit_overlay_heartbeat.v1", "ts_utc": _iso(NOW - timedelta(seconds=30)),
        "ttl_seconds": 900, "mode": "shadow", "evaluated": 0, "would_close": 0, "healthy": True,
    })
    _write_json(runtime / "positions_snapshot.json", {
        "source": "ibkr_portfolio_collector_v2", "ts_utc": _iso(NOW - timedelta(seconds=30)),
        "ttl_seconds": 300, "positions": [],
    })
    row = next(r for r in _make(tmp_path, clock, quiet_providers).check_stale_feeds().evidence["feeds"]
               if r["feed"] == "exit_overlay_blind_check")
    assert row["status"] == "ok"
    assert row["reason"] == "broker_genuinely_flat"


def test_blind_check_silent_when_overlay_mode_off(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "exit_overlay_heartbeat.json", {
        "schema_version": "exit_overlay_heartbeat.v1", "ts_utc": _iso(NOW - timedelta(seconds=30)),
        "ttl_seconds": 900, "mode": "off", "evaluated": 0, "would_close": 0, "healthy": True,
    })
    row = next(r for r in _make(tmp_path, clock, quiet_providers).check_stale_feeds().evidence["feeds"]
               if r["feed"] == "exit_overlay_blind_check")
    assert row["status"] == "ok"
    assert row["reason"] == "overlay_mode_off"


# ---------------------------------------------------------------------------
# 5. Check 2 — placeholder fills
# ---------------------------------------------------------------------------


def _fills(tmp_path: Path, rows) -> None:
    _write_ndjson(tmp_path / "data" / "fills" / "FILLS_20260715.ndjson", rows)


def test_placeholder_contained_is_ok(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _fills(tmp_path, [{"payload": {
        "symbol": "IWM", "status": "rejected", "fill_price": 295.87, "strategy": "delta",
        "fill_id": "abc", "tags": ["paper", "rejected", "pnl_untrusted", "placeholder"],
        "extra": {"pnl_untrusted": True, "trust_state": "PLACEHOLDER",
                  "placeholder_fill_price": 100.0, "placeholder_price_cache": 295.87},
    }}])
    result = _make(tmp_path, clock, quiet_providers).check_placeholder_fills()
    assert result.status == "ok"
    assert result.evidence["contained_count"] == 1


def test_uncontained_placeholder_fails(tmp_path, clock, quiet_providers):
    """The defense failing to demote a placeholder is the actual bug."""
    _fresh_runtime(tmp_path / "runtime")
    _fills(tmp_path, [{"payload": {
        "symbol": "SPY", "status": "paper_fill", "fill_price": 100.0, "strategy": "gamma",
        "fill_id": "leak1", "tags": ["paper", "filled", "placeholder"],
        "extra": {"placeholder_fill_price": 100.0, "trust_state": "PLACEHOLDER"},
    }}])
    result = _make(tmp_path, clock, quiet_providers).check_placeholder_fills()
    assert result.status == "fail"
    assert result.evidence["leaked_count"] == 1
    assert result.evidence["leaked"][0]["symbol"] == "SPY"


def test_placeholder_detected_despite_zeroed_top_level_price(tmp_path, clock, quiet_providers):
    """The writer zeroes fill_price, so equality-on-100 finds nothing.

    This is the exact bug in the old scanner (exterminator.py:395): it tested
    fill_price == 100.0, but a caught placeholder never carries that value.
    """
    _fresh_runtime(tmp_path / "runtime")
    _fills(tmp_path, [{"payload": {
        "symbol": "QQQ", "status": "rejected", "fill_price": 0.0, "notional": 0.0,
        "strategy": "delta", "fill_id": "z", "tags": ["rejected", "pnl_untrusted"],
        "extra": {"pnl_untrusted": True, "placeholder_fill_price": 100.0, "trust_state": "PLACEHOLDER"},
    }}])
    result = _make(tmp_path, clock, quiet_providers).check_placeholder_fills()
    assert result.evidence["contained_count"] == 1, "must key off markers, not fill_price == 100.0"


def test_clean_fills_are_ok(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _fills(tmp_path, [{"payload": {
        "symbol": "SPY", "status": "paper_fill", "fill_price": 751.06, "strategy": "gamma",
        "fill_id": "ok1", "tags": ["paper", "filled"], "extra": {},
    }}])
    result = _make(tmp_path, clock, quiet_providers).check_placeholder_fills()
    assert result.status == "ok"
    assert result.evidence["rows_scanned"] == 1


def test_no_fill_ledger_warns_rather_than_passing(tmp_path, clock, quiet_providers):
    """A scan that proved nothing must not report ok — the old EX006's sin."""
    _fresh_runtime(tmp_path / "runtime")
    result = _make(tmp_path, clock, quiet_providers).check_placeholder_fills()
    assert result.status == "warn"
    assert "proved nothing" in result.summary


# ---------------------------------------------------------------------------
# 6. Check 3 — untrusted fills outside permitted stores
# ---------------------------------------------------------------------------


def _trades(tmp_path: Path, rows) -> None:
    _write_ndjson(tmp_path / "data" / "trades" / "trade_history_20260715.ndjson", rows)


def test_untrusted_row_in_scored_store_fails(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _trades(tmp_path, [
        {"symbol": "SPY", "strategy": "gamma", "trade_id": "t1", "tags": ["filled"]},
        {"symbol": "MES", "strategy": "omega", "trade_id": "t2", "pnl_untrusted": True, "tags": []},
    ])
    result = _make(tmp_path, clock, quiet_providers).check_untrusted_fills()
    assert result.status == "fail"
    assert result.evidence["leak_count"] == 1
    assert result.evidence["leaks"][0]["marker"] == "pnl_untrusted"


def test_validate_only_row_in_scored_store_fails(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _trades(tmp_path, [{"symbol": "BTC", "strategy": "alpha_crypto", "trade_id": "k1",
                        "extra": {"validate_only": True}, "tags": []}])
    result = _make(tmp_path, clock, quiet_providers).check_untrusted_fills()
    assert result.status == "fail"
    assert result.evidence["leaks"][0]["marker"] == "validate_only"


def test_untrusted_rows_in_permitted_stores_are_not_leaks(tmp_path, clock, quiet_providers):
    """data/fills and trade_closer legitimately hold marked rows."""
    _fresh_runtime(tmp_path / "runtime")
    _fills(tmp_path, [{"payload": {"symbol": "IWM", "status": "rejected", "fill_id": "p",
                                   "tags": ["pnl_untrusted"], "extra": {"pnl_untrusted": True}}}])
    _trades(tmp_path, [{"symbol": "SPY", "strategy": "gamma", "trade_id": "clean", "tags": []}])
    result = _make(tmp_path, clock, quiet_providers).check_untrusted_fills()
    assert result.status == "ok"
    assert "data/fills/FILLS_*.ndjson" in result.evidence["permitted_stores"]


def test_untrusted_tag_variant_is_caught(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _trades(tmp_path, [{"symbol": "SPY", "strategy": "gamma", "trade_id": "t3", "tags": ["PNL_UNTRUSTED"]}])
    assert _make(tmp_path, clock, quiet_providers).check_untrusted_fills().status == "fail"


def test_incident_0713_row_shape_is_caught_and_identified(tmp_path, clock, quiet_providers):
    """Pins the real row this check found in the live scored store.

    The INCIDENT-0713 TLT close is hash-chain-enveloped, carries NO trade_id,
    marks pnl_untrusted only under `extra`, and is attributed to "manual" via a
    config_default fallback. SCR currently drops it through the *manual* bucket
    (excluded_untrusted=0, excluded_manual=1), so its untrusted marker is doing
    no work -- a re-attribution away from "manual" would score it. The check
    must still flag it, and must produce a usable handle from close_key.
    """
    _fresh_runtime(tmp_path / "runtime")
    _trades(tmp_path, [{
        "payload": {
            "symbol": "TLT", "strategy": "manual", "side": "SELL", "quantity": 640.0,
            "fill_price": 83.98, "pnl": 0.0, "tags": ["ibkr_paper", "manual"],
            "extra": {
                "pnl_untrusted": True,
                "pnl_untrusted_reason": "symbol_close_detected_without_fill_matcher",
                "attribution_source": "config_default",
                "close_key": "27f21da5c20f01d25bf04771da982ae2b10ac759893792bd47db5a31299c09c2",
                "source_strategies": ["manual"],
            },
        },
        "record_hash": "deadbeef", "sequence_id": 1,
    }])
    result = _make(tmp_path, clock, quiet_providers).check_untrusted_fills()
    assert result.status == "fail"
    leak = result.evidence["leaks"][0]
    assert leak["symbol"] == "TLT"
    assert leak["strategy"] == "manual"
    assert leak["marker"] == "pnl_untrusted"
    assert leak["reason"] == "symbol_close_detected_without_fill_matcher"
    assert leak["row_id"] == "27f21da5c20f01d2", "close_key must supply a handle when trade_id is absent"
    # The summary must not overclaim that SCR is scoring the row.
    assert "Presence is not proof they are being scored" in result.summary
    assert result.evidence["verify_with"].startswith("runtime/scr_state.json")


# ---------------------------------------------------------------------------
# 7. Check 4 — INDEPENDENT-LEG RULE (the XOV-2345 lesson)
# ---------------------------------------------------------------------------


def test_drift_ok_when_guard_matches_independent_leg(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "ok"
    assert result.evidence["actionable_count"] == 0


def test_dual_booked_guard_is_not_double_counted(tmp_path, clock, quiet_providers):
    """gamma|UNH 273 + broker_sync|UNH 273 is ONE 273-share position, not 546."""
    _fresh_runtime(tmp_path / "runtime")
    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "ok", "summing the dual-booked legs would invent a 2.0x phantom"


def test_xov2345_false_flat_is_caught_when_same_source_legs_say_green(tmp_path, clock, quiet_providers):
    """THE regression that matters.

    Reproduces XOV-2345: the guard is false-flatted while position_guard_drift
    (whose legs both come from that same guard file) reports drift_count=0. The
    same-source legs agree with each other and see nothing. Only the independent
    collector still shows the real position.
    """
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "position_guard.json", {"_version": 2, "_written_by": "false_flat"})
    _write_json(runtime / "trade_closer_state.json", {
        "processed_fill_ids": [], "saved_at_utc": _iso(NOW), "queues": [],
    })

    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "fail"
    assert "leg_disagreement" in result.evidence
    assert result.evidence["leg_disagreement"]["independent_leg_finds"] == 2
    assert "position_guard_drift.json" in result.evidence["leg_disagreement"]["same_source_legs_claim_clean"]
    kinds = {d["drift_kind"] for d in result.evidence["actionable_drifts"]}
    assert kinds == {"mirror_vs_independent_broker"}


def test_phantom_guard_entry_is_caught(tmp_path, clock, quiet_providers):
    """Guard holds TLT, the independent collector says flat."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    guard = json.loads((runtime / "position_guard.json").read_text())
    guard["gamma|TLT"] = {"symbol": "TLT", "quantity": 640.0, "side": "BUY", "open": True, "strategy": "gamma"}
    _write_json(runtime / "position_guard.json", guard)
    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "fail"
    tlt = next(d for d in result.evidence["actionable_drifts"] if d["symbol"] == "TLT")
    assert tlt["drift_kind"] == "phantom_guard_entry"
    assert tlt["independent_broker_qty"] == 0.0


def test_independent_leg_stale_degrades_to_blind_not_ok(tmp_path, clock, quiet_providers):
    """No independent truth ⇒ warn. Never a false ok."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "positions_snapshot.json", {
        "source": "ibkr_portfolio_collector_v2", "ts_utc": _iso(NOW - timedelta(hours=6)),
        "ttl_seconds": 300, "positions": [],
    })
    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "warn"
    assert result.title == "Independent leg blind"


def test_independent_leg_missing_degrades_to_blind(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    (runtime / "positions_snapshot.json").unlink()
    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "warn"
    assert result.title == "Independent leg blind"


def test_mixed_ownership_symbols_are_info_not_fail(tmp_path, clock, quiet_providers):
    """Operator-owned symbols blend broker shares with CHAD lots.

    Comparing a strategy total against the combined broker total is not
    like-with-like; inflating drift here re-introduces the false-RED class that
    WKF U3 fixed (chad/core/position_guard.py:686).
    """
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    guard = json.loads((runtime / "position_guard.json").read_text())
    guard["gamma|BAC"] = {"symbol": "BAC", "quantity": 22.0, "side": "BUY", "open": True, "strategy": "gamma"}
    guard["broker_sync|BAC"] = {"symbol": "BAC", "quantity": 176.0, "side": "BUY", "open": True, "strategy": "broker_sync"}
    _write_json(runtime / "position_guard.json", guard)
    snap = json.loads((runtime / "positions_snapshot.json").read_text())
    snap["positions"].append({"symbol": "BAC", "position": 176.0, "secType": "STK", "currency": "USD"})
    _write_json(runtime / "positions_snapshot.json", snap)
    drift = json.loads((runtime / "position_guard_drift.json").read_text())
    drift["excluded_symbols"] = ["BAC"]
    _write_json(runtime / "position_guard_drift.json", drift)

    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    bac = next(d for d in result.evidence["info_drifts"] if d["symbol"] == "BAC")
    assert bac["drift_kind"] == "mixed_ownership_info"
    assert bac["is_excluded"] is True
    assert not any(d["symbol"] == "BAC" for d in result.evidence["actionable_drifts"])


def test_guard_vs_fifo_split_warns(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "trade_closer_state.json", {
        "processed_fill_ids": [], "saved_at_utc": _iso(NOW),
        "queues": [
            {"strategy": "gamma", "symbol": "UNH", "lots": [{"fill_id": "a", "side": "BUY", "quantity": 100.0}]},
            {"strategy": "gamma", "symbol": "V", "lots": [{"fill_id": "b", "side": "BUY", "quantity": 181.0}]},
        ],
    })
    result = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift()
    assert result.status == "warn"
    assert result.evidence["split_count"] == 1
    split = result.evidence["guard_vs_fifo_splits"][0]
    assert (split["symbol"], split["guard_qty"], split["fifo_qty"]) == ("UNH", 273.0, 100.0)


def test_check4_never_reads_broker_sync_as_sole_truth(tmp_path, clock, quiet_providers):
    """The rejected leg is named in evidence so the rule stays legible."""
    _fresh_runtime(tmp_path / "runtime")
    ev = _make(tmp_path, clock, quiet_providers).check_reconciliation_drift().evidence
    assert "positions_snapshot.json" in ev["independent_leg"]
    assert "clientId=99" in ev["independent_leg"]
    assert "broker_sync" in ev["rejected_leg"]


# ---------------------------------------------------------------------------
# 8. Check 5 — failed services
# ---------------------------------------------------------------------------


def test_failed_chad_unit_fails(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers,
                 "systemctl_provider": lambda q: {"failed_units": ["chad-orchestrator.service"], "error": None}}
    result = _make(tmp_path, clock, providers).check_failed_services()
    assert result.status == "fail"
    assert result.evidence["failed_units"] == ["chad-orchestrator.service"]
    assert "never restarts" in result.evidence["operator_action"]


def test_non_chad_failed_unit_is_not_a_fail(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers,
                 "systemctl_provider": lambda q: {"failed_units": ["apt-daily.service"], "error": None}}
    result = _make(tmp_path, clock, providers).check_failed_services()
    assert result.status == "ok"
    assert result.evidence["non_chad_failed_units"] == ["apt-daily.service"]


def test_systemctl_query_is_read_only(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    seen: list[list[str]] = []

    def spy(query):
        seen.append(list(query))
        return {"failed_units": [], "error": None}

    _make(tmp_path, clock, {**quiet_providers, "systemctl_provider": spy}).check_failed_services()
    assert seen and seen[0][:3] == ["systemctl", "list-units", "--state=failed"]
    for verb in ("start", "stop", "restart", "reset-failed", "enable", "disable"):
        assert verb not in seen[0]


def test_systemctl_probe_error_warns(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers, "systemctl_provider": lambda q: {"failed_units": [], "error": "boom"}}
    assert _make(tmp_path, clock, providers).check_failed_services().status == "warn"


# ---------------------------------------------------------------------------
# 9. Check 6 — dirty git
# ---------------------------------------------------------------------------


def test_dirty_production_path_warns(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers, "git_provider": lambda: {
        "head": "a", "branch": "main", "error": None,
        "entries": [" M chad/core/live_loop.py", " M config/tiers.json"],
    }}
    result = _make(tmp_path, clock, providers).check_dirty_git()
    assert result.status == "warn"
    assert result.evidence["dirty_count"] == 2


def test_archive_deletions_production_filtered_not_allowlisted(tmp_path, clock, quiet_providers):
    """W3B-12 disposition pin: the 20 disk-guard-purged _archive deletions
    were committed and the allowlist entry removed per its own expiry note.
    The allowlist is now EMPTY — _archive/ paths fall to the production_paths
    filter (outside scope, not drift), so a future disk-guard purge of
    tracked archives is neither dirty nor 'allowlisted noise' in every
    report. The disposition lifecycle (archive by git-committed move → purge
    → commit the deletions at the next hygiene pass) is convention, recorded
    in docs/DYNAMIC_CAPS_DISPOSITION_2026-07-22.md; history keeps the bytes."""
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers, "git_provider": lambda: {
        "head": "a", "branch": "main", "error": None,
        "entries": ["D  _archive/bak_purge_20260722/chad/risk/dominance_strategy.py",
                    " D _archive/bak_quarantine_20260402/chad/utils/telegram_bot.py.bak"],
    }}
    result = _make(tmp_path, clock, providers).check_dirty_git()
    assert result.status == "ok"
    assert result.evidence["allowlisted_count"] == 0  # the entry is GONE
    assert result.evidence["dirty_count"] == 0  # production-filtered, not drift
    # and a production-path deletion still flags exactly as before
    providers2 = {**quiet_providers, "git_provider": lambda: {
        "head": "a", "branch": "main", "error": None,
        "entries": [" D chad/risk/somefile.py"],
    }}
    result2 = _make(tmp_path, clock, providers2).check_dirty_git()
    assert result2.status == "warn"
    assert result2.evidence["dirty_count"] == 1


def test_untracked_files_are_not_dirty(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers, "git_provider": lambda: {
        "head": "a", "branch": "main", "error": None, "entries": ["?? audits/new_thing.md"],
    }}
    assert _make(tmp_path, clock, providers).check_dirty_git().status == "ok"


def test_non_production_dirty_path_ignored(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    providers = {**quiet_providers, "git_provider": lambda: {
        "head": "a", "branch": "main", "error": None, "entries": [" M docs/notes.md"],
    }}
    assert _make(tmp_path, clock, providers).check_dirty_git().status == "ok"


# ---------------------------------------------------------------------------
# 10. Check 7 — schema breaks
# ---------------------------------------------------------------------------


def test_schema_warns_on_known_unpinned_contracts(tmp_path, clock, quiet_providers):
    """Enforced contracts hold, but unpinned ones are reported not hidden."""
    _fresh_runtime(tmp_path / "runtime")
    result = _make(tmp_path, clock, quiet_providers).check_schema_breaks()
    assert result.status == "warn"
    assert result.evidence["break_count"] == 0
    assert "runtime/reconciliation_state.json" in result.evidence["unpinned_known"]


def test_schema_version_mismatch_fails(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "scr_state.json", {
        "schema_version": "scr_state.v99", "ts_utc": _iso(NOW), "ttl_seconds": 180,
        "state": "WARMUP", "sizing_factor": 0.1, "stats": {},
    })
    result = _make(tmp_path, clock, quiet_providers).check_schema_breaks()
    assert result.status == "fail"
    brk = next(b for b in result.evidence["breaks"] if b["file"] == "runtime/scr_state.json")
    assert brk["break"] == "schema_version_unrecognised"
    assert brk["actual"] == "scr_state.v99"


def test_missing_required_key_fails(tmp_path, clock, quiet_providers):
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    _write_json(runtime / "scr_state.json", {
        "schema_version": "scr_state.v1", "ts_utc": _iso(NOW), "ttl_seconds": 180, "state": "WARMUP",
    })
    result = _make(tmp_path, clock, quiet_providers).check_schema_breaks()
    assert result.status == "fail"
    brk = next(b for b in result.evidence["breaks"] if b["break"] == "required_keys_missing")
    assert set(brk["missing_keys"]) == {"sizing_factor", "stats"}


def test_drift_schema_accepts_v1_v2_and_v3(tmp_path, clock, quiet_providers):
    """live_readiness_publish.py:195 accepts v1|v2|v3, so this check must too."""
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    for version in ("position_guard_drift.v1", "position_guard_drift.v2", "position_guard_drift.v3"):
        _write_json(runtime / "position_guard_drift.json", {
            "schema_version": version, "ts_utc": _iso(NOW), "ttl_seconds": 360,
            "drift_count": 0, "info_count": 0, "excluded_symbols": [], "snapshot_generation": 1,
            "counts_by_kind": {}, "drifts": [],
        })
        result = _make(tmp_path, clock, quiet_providers).check_schema_breaks()
        breaks = [b for b in result.evidence["breaks"] if b["file"] == "runtime/position_guard_drift.json"]
        assert not breaks, f"{version} must be accepted"


# ---------------------------------------------------------------------------
# 11. Check 8 — ML anomalies
# ---------------------------------------------------------------------------


def test_ml_reports_no_baseline_honestly(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _write_json(tmp_path / "shared" / "models" / "xgb_veto_manifest.json", {
        "model_version": "xgb_veto_20260714_020007",
        "metrics": {"val_veto_rate_at_0.65": 0.7123},
    })
    result = _make(tmp_path, clock, quiet_providers).check_ml_anomalies()
    assert result.status == "warn"
    assert result.title == "no_baseline"
    assert result.evidence["model_version_present"] is True
    assert result.evidence["baseline_veto_rate"] is None
    assert "training-time validation statistic" in result.evidence["no_baseline_detail"]


def test_ml_flags_stale_manifest_alongside_no_baseline(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _write_json(tmp_path / "shared" / "models" / "xgb_veto_manifest.json", {
        "model_version": "xgb_veto_20260510_020007",
        "metrics": {"val_veto_rate_at_0.65": 0.7123},
    })
    result = _make(tmp_path, clock, quiet_providers).check_ml_anomalies()
    assert result.status == "warn"
    assert result.evidence["manifest_stale"] is True
    assert result.evidence["model_version_age_days"] > 30


def test_ml_missing_manifest_warns(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    result = _make(tmp_path, clock, quiet_providers).check_ml_anomalies()
    assert result.status == "warn"
    assert result.title == "ML manifest absent"


def test_ml_missing_model_version_warns(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    _write_json(tmp_path / "shared" / "models" / "xgb_veto_manifest.json", {"metrics": {}})
    result = _make(tmp_path, clock, quiet_providers).check_ml_anomalies()
    assert result.status == "warn"
    assert result.evidence["model_version_present"] is False


# ---------------------------------------------------------------------------
# 12. Notification
# ---------------------------------------------------------------------------


def test_no_telegram_when_nothing_failed(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    sent: list[tuple[str, str]] = []
    s = _make(tmp_path, clock, {**quiet_providers,
                                "notifier": lambda m, k: sent.append((m, k)) or True})
    report = s.run()
    report["checks"] = [c for c in report["checks"] if c["status"] != "fail"]
    assert s.maybe_notify(report) is False
    assert sent == []


def test_telegram_sent_on_fail_with_stable_dedupe_key(tmp_path, clock, quiet_providers):
    _fresh_runtime(tmp_path / "runtime")
    sent: list[tuple[str, str]] = []

    def notifier(message, dedupe_key):
        sent.append((message, dedupe_key))
        return True

    s = _make(tmp_path, clock, {**quiet_providers, "notifier": notifier})
    report = {"checks": [
        {"check_id": "EXS4", "status": "fail", "title": "Guard disagrees with independent broker truth",
         "summary": "x", "evidence": {}, "name": "reconciliation_drift", "remedy_type": "NOTIFY_ONLY"},
    ]}
    assert s.maybe_notify(report) is True
    assert len(sent) == 1
    assert sent[0][1] == "exterminator_sentinel_EXS4"


def test_dedupe_key_is_identical_across_cycles(tmp_path, clock, quiet_providers):
    """CTF-T2: the same failure must not mint a new key each cycle."""
    _fresh_runtime(tmp_path / "runtime")
    keys: list[str] = []
    s = _make(tmp_path, clock, {**quiet_providers, "notifier": lambda m, k: keys.append(k) or True})
    for evaluated in (0, 0):
        report = {"checks": [{"check_id": "EXS1", "status": "fail", "title": "Stale feed detected",
                              "summary": f"var_state is {evaluated + 5967360}s old", "evidence": {},
                              "name": "stale_feeds", "remedy_type": "NOTIFY_ONLY"}]}
        s.maybe_notify(report)
    assert len(set(keys)) == 1, f"dedupe key fluctuated across cycles: {keys}"


def test_stable_identity_strips_values_but_keeps_identifiers():
    """Mirrors chad/ops/health_monitor.py:249 — values go, identifiers stay."""
    assert sentinel_mod.stable_identity("SCR gap 214 raw vs 67 effective") == "SCR gap raw vs effective"
    assert sentinel_mod.stable_identity("Disk 87.3% full") == "Disk % full"
    assert sentinel_mod.stable_identity("Feed STALE: kraken (12345s old)") == "Feed STALE: kraken (s old)"
    for identifier in ("EXS1", "M6E", "alpha_crypto", "chad-live-loop"):
        assert identifier in sentinel_mod.stable_identity(f"unit {identifier} down")


def test_stable_identity_truncates_multi_digit_ids_but_stays_deterministic():
    """A known quirk of the upstream regex, pinned so it can't surprise anyone.

    The lookbehind only spares a digit preceded by a LETTER, so the second digit
    of "R14" is preceded by "1" and gets stripped -> "R1". That is lossy but
    perfectly deterministic, which is all a dedupe key needs. It matters here
    only because every EXS check_id is single-digit and therefore survives
    intact -- EXS1..EXS8 never collide with each other.
    """
    assert sentinel_mod.stable_identity("R14") == "R1"
    assert sentinel_mod.stable_identity("R14") == sentinel_mod.stable_identity("R14")
    # EXS1..EXS9 are single-digit and survive stable_identity intact.
    ids = [sentinel_mod.stable_identity(f"EXS{i}") for i in range(1, 10)]
    assert ids == [f"EXS{i}" for i in range(1, 10)]
    assert len(set(ids)) == 9, "single-digit check_id identities must never collide"
    # EXS10 (W5A DQ2) is the FIRST two-digit id: stable_identity strips the
    # trailing "0" (preceded by "1") -> "EXS1", colliding with EXS1. This is
    # harmless because EXS10 is WARN-ONLY (rider R3) and maybe_notify keys the
    # dedupe ONLY on FAILED checks, so EXS10 can never enter the dedupe path
    # where the collision would matter.
    assert sentinel_mod.stable_identity("EXS10") == "EXS1"


def test_notifier_failure_does_not_raise(tmp_path, clock, quiet_providers):
    """Alerting is supplementary; a dead Telegram must never break the scan."""
    _fresh_runtime(tmp_path / "runtime")

    def boom(message, dedupe_key):
        raise RuntimeError("telegram down")

    s = _make(tmp_path, clock, {**quiet_providers, "notifier": boom})
    report = s.run()
    with pytest.raises(RuntimeError):
        s.maybe_notify({"checks": [{"check_id": "EXS4", "status": "fail", "title": "t",
                                    "summary": "s", "evidence": {}, "name": "n",
                                    "remedy_type": "NOTIFY_ONLY"}]})
    # run() itself is unaffected by notifier health
    assert report["schema_version"] == "exterminator_sentinel.v1"


# ---------------------------------------------------------------------------
# 13. Config contract
# ---------------------------------------------------------------------------


def test_shipped_config_is_valid_and_covers_the_spec_feeds():
    cfg = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["schema_version"] == "exterminator_config.v1"
    required = {"var_state", "scr_state", "positions_snapshot", "kraken_ticks", "bars",
                "exit_overlay_heartbeat", "equity_history"}
    assert required <= set(cfg["feeds"])


def test_every_feed_cites_a_ttl_source_or_flags_operator_verify():
    cfg = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    for name, feed in cfg["feeds"].items():
        assert "ttl_source" in feed, f"{name} declares no ttl_source"
        if feed["ttl_verified"]:
            assert feed["ttl_source"] != "operator_verify", f"{name} claims verified with no citation"
            assert len(feed["ttl_source"]) > 10, f"{name} citation is not specific"
        else:
            assert feed["ttl_source"] == "operator_verify", f"{name} must flag operator_verify"
            assert feed["fail_after_seconds"] is None, f"{name} unverified TTL must not be able to fail"


def test_worst_status_ordering():
    assert sentinel_mod.worst_status(["ok", "ok"]) == "ok"
    assert sentinel_mod.worst_status(["ok", "warn"]) == "warn"
    assert sentinel_mod.worst_status(["warn", "fail", "ok"]) == "fail"
    assert sentinel_mod.worst_status([]) == "ok"
