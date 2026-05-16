"""Read-only consumer tests for runtime/earnings_intel.json.

Covers the module-level fail-open loader
``chad.intel.strategy_intelligence._load_earnings_intel_context`` and its
dashboard surface in ``chad.dashboard.api`` (read-only,
intelligence-only — never wired into any strategy, execution, or risk
gate during the one-week observation period).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from chad.intel.strategy_intelligence import (
    EARNINGS_INTEL_DEFAULT_TTL_SEC,
    EARNINGS_INTEL_UPCOMING_LIMIT,
    _load_earnings_intel_context,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _baseline_payload(
    *,
    age_seconds: int = 60,
    ttl: int = EARNINGS_INTEL_DEFAULT_TTL_SEC,
    status: str = "partial",
    symbols: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ts = _now() - timedelta(seconds=age_seconds)
    payload: Dict[str, Any] = {
        "schema_version": "earnings_intel.v1",
        "status": status,
        "ts_utc": _iso(ts),
        "ttl_seconds": ttl,
        "source": {
            "provider": "fmp_stable",
            "provider_status": "partial",
            "endpoints": [
                "earnings-calendar",
                "price-target-consensus",
                "analyst-estimates annual",
                "sec-filings-search",
            ],
        },
        "window": {
            "date_from": "2026-05-02",
            "date_to": "2026-06-30",
            "forward_days": 45,
            "lookback_days": 14,
        },
        "summary": summary if summary is not None else {
            "symbols_requested": 25,
            "symbols_processed": 25,
            "symbols_with_next_earnings": 1,
            "symbols_with_price_targets": 10,
            "symbols_with_analyst_estimates": 10,
            "symbols_with_sec_filings": 21,
        },
        "symbols": symbols if symbols is not None else {
            "NVDA": {
                "data_available": True,
                "next_earnings_date": "2026-05-20",
                "days_to_next_earnings": 4,
                "price_target_consensus": 276.75,
                "annual_eps_avg_estimate": 8.31929,
                "sec_filings_count": 8,
                "provider_errors": [],
            },
            "AAPL": {
                "data_available": True,
                "next_earnings_date": None,
                "days_to_next_earnings": None,
                "price_target_consensus": 324.21,
                "annual_eps_avg_estimate": 8.71903,
                "sec_filings_count": 3,
                "provider_errors": [],
            },
        },
    }
    return payload


@pytest.fixture()
def runtime_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "runtime"
    rd.mkdir()
    return rd


def _write_intel(runtime_dir: Path, payload: Dict[str, Any]) -> Path:
    path = runtime_dir / "earnings_intel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests 1–7: helper behavior
# ---------------------------------------------------------------------------


def test_fresh_payload_returns_freshness_fresh(runtime_dir: Path) -> None:
    """1. Fresh payload (within TTL) yields freshness='fresh'."""
    _write_intel(runtime_dir, _baseline_payload(age_seconds=60))
    ctx = _load_earnings_intel_context(runtime_dir)
    assert ctx["freshness"] == "fresh"
    assert ctx["status"] == "partial"
    assert ctx["source_provider"] == "fmp_stable"
    assert ctx["ttl_seconds"] == EARNINGS_INTEL_DEFAULT_TTL_SEC
    assert ctx["summary"]["symbols_requested"] == 25
    assert len(ctx["upcoming"]) == 1
    assert ctx["upcoming"][0]["symbol"] == "NVDA"


def test_stale_ttl_returns_stale_but_preserves_summary(runtime_dir: Path) -> None:
    """2. Past ts_utc + ttl_seconds yields freshness='stale' but keeps summary/upcoming."""
    _write_intel(
        runtime_dir,
        _baseline_payload(age_seconds=EARNINGS_INTEL_DEFAULT_TTL_SEC * 3, ttl=EARNINGS_INTEL_DEFAULT_TTL_SEC),
    )
    ctx = _load_earnings_intel_context(runtime_dir)
    assert ctx["freshness"] == "stale"
    assert ctx["status"] == "partial"
    assert ctx["summary"]["symbols_requested"] == 25
    assert ctx["summary"]["symbols_with_next_earnings"] == 1
    assert len(ctx["upcoming"]) == 1
    assert ctx["upcoming"][0]["symbol"] == "NVDA"


def test_missing_file_returns_missing(runtime_dir: Path) -> None:
    """3. Absent earnings_intel.json yields freshness='missing' and empty summary/upcoming."""
    ctx = _load_earnings_intel_context(runtime_dir)
    assert ctx["freshness"] == "missing"
    assert ctx["status"] == "unknown"
    assert ctx["summary"]["symbols_requested"] is None
    assert ctx["summary"]["symbols_with_next_earnings"] is None
    assert ctx["upcoming"] == []
    assert ctx["ts_utc"] is None
    assert ctx["ttl_seconds"] is None
    assert ctx["source_provider"] is None


def test_malformed_json_returns_malformed(runtime_dir: Path) -> None:
    """4. Unparseable JSON yields freshness='malformed' with empty summary/upcoming."""
    (runtime_dir / "earnings_intel.json").write_text("{not valid json", encoding="utf-8")
    ctx = _load_earnings_intel_context(runtime_dir)
    assert ctx["freshness"] == "malformed"
    assert ctx["status"] == "unknown"
    assert ctx["summary"]["symbols_requested"] is None
    assert ctx["upcoming"] == []


def test_partial_status_is_not_treated_as_error(runtime_dir: Path) -> None:
    """5. status='partial' is a by-design publisher state — never error."""
    _write_intel(runtime_dir, _baseline_payload(status="partial"))
    ctx = _load_earnings_intel_context(runtime_dir)
    assert ctx["status"] == "partial"
    assert ctx["freshness"] == "fresh"


def test_upcoming_list_sorted_by_days_ascending(runtime_dir: Path) -> None:
    """6. Upcoming entries sort by days_to_next_earnings ascending, None last."""
    symbols = {
        "FAR": {
            "data_available": True,
            "next_earnings_date": "2026-07-15",
            "days_to_next_earnings": 60,
        },
        "SOON": {
            "data_available": True,
            "next_earnings_date": "2026-05-20",
            "days_to_next_earnings": 4,
        },
        "MID": {
            "data_available": True,
            "next_earnings_date": "2026-06-10",
            "days_to_next_earnings": 25,
        },
        "NODAYS": {
            "data_available": True,
            "next_earnings_date": "2026-06-30",
            "days_to_next_earnings": None,
        },
        "NO_DATE": {
            "data_available": True,
            "next_earnings_date": None,
            "days_to_next_earnings": None,
        },
    }
    _write_intel(runtime_dir, _baseline_payload(symbols=symbols))
    ctx = _load_earnings_intel_context(runtime_dir)
    ordered = [row["symbol"] for row in ctx["upcoming"]]
    assert ordered == ["SOON", "MID", "FAR", "NODAYS"]
    # NO_DATE is filtered out entirely because next_earnings_date is None.
    assert "NO_DATE" not in ordered


def test_upcoming_list_capped_at_limit(runtime_dir: Path) -> None:
    """7. Upcoming list is capped at EARNINGS_INTEL_UPCOMING_LIMIT entries."""
    symbols = {
        f"SYM{i:02d}": {
            "data_available": True,
            "next_earnings_date": "2026-06-01",
            "days_to_next_earnings": i,
        }
        for i in range(20)
    }
    _write_intel(runtime_dir, _baseline_payload(symbols=symbols))
    ctx = _load_earnings_intel_context(runtime_dir)
    assert len(ctx["upcoming"]) == EARNINGS_INTEL_UPCOMING_LIMIT
    # The 10 nearest (smallest days_to_next_earnings) must be retained.
    assert [row["symbol"] for row in ctx["upcoming"]] == [
        f"SYM{i:02d}" for i in range(EARNINGS_INTEL_UPCOMING_LIMIT)
    ]


# ---------------------------------------------------------------------------
# Test 8: dashboard surface
# ---------------------------------------------------------------------------


def test_dashboard_intelligence_payload_includes_earnings_intel(
    runtime_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8. dashboard._intelligence() includes the earnings_intel block keyed via the helper."""
    # The dashboard module requires CHAD_DASHBOARD_PASSWORD at import time.
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test-password-not-used")

    # Force a fresh import so the module reads the env var we just set and
    # also picks up the runtime path from the test fixture.
    if "chad.dashboard.api" in sys.modules:
        del sys.modules["chad.dashboard.api"]
    api = importlib.import_module("chad.dashboard.api")

    # Redirect the dashboard's RUNTIME constant to our tmp runtime dir so we
    # can plant a controlled earnings_intel.json without touching the real
    # repo runtime/ files.
    _write_intel(runtime_dir, _baseline_payload(age_seconds=60))
    monkeypatch.setattr(api, "RUNTIME", runtime_dir)

    sb = api.StateBuilder()
    intel = sb._intelligence()

    assert "earnings_intel" in intel
    block = intel["earnings_intel"]
    assert block["freshness"] == "fresh"
    assert block["status"] == "partial"
    assert block["source_provider"] == "fmp_stable"
    assert block["ttl_seconds"] == EARNINGS_INTEL_DEFAULT_TTL_SEC
    assert isinstance(block["summary"], dict)
    assert isinstance(block["upcoming"], list)
    assert block["summary"]["symbols_requested"] == 25
    assert len(block["upcoming"]) == 1
    assert block["upcoming"][0]["symbol"] == "NVDA"


# ---------------------------------------------------------------------------
# Tests 9–10: governance — no strategy/execution/risk imports of the helper
# ---------------------------------------------------------------------------


_HELPER_TOKENS = ("_load_earnings_intel_context", "earnings_intel.json")


def _scan_for_helper_imports(roots: list[Path]) -> list[str]:
    offenders: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(tok in text for tok in _HELPER_TOKENS):
                offenders.append(str(path))
    return offenders


def test_no_strategy_files_import_earnings_intel_helper() -> None:
    """9. No file under chad/strategies/ may reference the helper or its file path."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _scan_for_helper_imports([repo_root / "chad" / "strategies"])
    assert offenders == [], (
        "Strategy files must not consume earnings_intel during the "
        f"one-week observation period. Offenders: {offenders}"
    )


def test_no_execution_or_risk_files_import_earnings_intel_helper() -> None:
    """10. No file under chad/execution/ or chad/risk/ may reference the helper or its file path."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _scan_for_helper_imports(
        [repo_root / "chad" / "execution", repo_root / "chad" / "risk"]
    )
    assert offenders == [], (
        "Execution/risk files must not consume earnings_intel "
        f"(intel-only). Offenders: {offenders}"
    )
