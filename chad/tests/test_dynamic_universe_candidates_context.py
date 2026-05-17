"""Read-only consumer tests for runtime/dynamic_universe_candidates.json.

Covers the module-level fail-open loader
``chad.intel.strategy_intelligence._load_dynamic_universe_candidates_context``
and its dashboard surface in ``chad.dashboard.api`` (read-only,
intelligence-only — never wired into any strategy, execution, or risk
gate during the Phase D observation period).
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.intel.strategy_intelligence import (
    DYNAMIC_UNIVERSE_CANDIDATES_DEFAULT_TTL_SEC,
    DYNAMIC_UNIVERSE_CANDIDATES_FILENAME,
    DYNAMIC_UNIVERSE_CANDIDATES_TOP_LIMIT,
    _load_dynamic_universe_candidates_context,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _candidate(
    *,
    rank: int,
    symbol: str,
    score: float = 0.5,
    reasons: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    rs_class: str = "neutral",
    rvol_class: str = "low",
    catalyst_strength: str = "none",
    earnings_days: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "rank": rank,
        "symbol": symbol,
        "score": score,
        "reasons": list(reasons or []),
        "warnings": list(warnings or []),
        "rs_class": rs_class,
        "rvol_class": rvol_class,
        "catalyst_strength": catalyst_strength,
        "catalyst_direction": "neutral",
        "confirmed_gate_relevant": False,
        "data_available": True,
        "earnings_days": earnings_days,
        "rs_excess_vs_spy_5d": 0.0,
        "rvol": 0.0,
        "liquidity_class": "unknown",
    }


def _baseline_payload(
    *,
    age_seconds: int = 60,
    ttl: int = DYNAMIC_UNIVERSE_CANDIDATES_DEFAULT_TTL_SEC,
    status: str = "ok",
    candidates: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
    provider_status: str = "real",
) -> Dict[str, Any]:
    ts = _now() - timedelta(seconds=age_seconds)
    if candidates is None:
        candidates = [
            _candidate(
                rank=1,
                symbol="LLY",
                score=0.63,
                reasons=["rs_strong", "confirmed_high_catalyst"],
                warnings=["rvol_low"],
                rs_class="strong",
                rvol_class="low",
                catalyst_strength="high",
            ),
            _candidate(
                rank=2,
                symbol="NVDA",
                score=0.63,
                reasons=["rs_strong", "confirmed_high_catalyst"],
                warnings=["rvol_low", "earnings_within_7d"],
                rs_class="strong",
                rvol_class="low",
                catalyst_strength="high",
                earnings_days=4,
            ),
            _candidate(
                rank=3,
                symbol="MA",
                score=0.43,
                reasons=["rs_neutral", "confirmed_high_catalyst"],
                warnings=["rvol_low"],
                rs_class="neutral",
                rvol_class="low",
                catalyst_strength="high",
            ),
        ]
    if summary is None:
        summary = {
            "symbols_considered": 25,
            "candidates_published": len(candidates),
            "strong_rs_count": sum(1 for c in candidates if c.get("rs_class") == "strong"),
            "high_rvol_count": 0,
            "confirmed_catalyst_count": 2,
            "earnings_warning_count": 1,
        }
    payload: Dict[str, Any] = {
        "schema_version": "dynamic_universe_candidates.v1",
        "ts_utc": _iso(ts),
        "ttl_seconds": ttl,
        "status": status,
        "source": {
            "provider": "local_runtime_intelligence",
            "provider_status": provider_status,
            "inputs": {
                "active_universe": {"available": True, "source_type": "runtime", "stale": False},
                "earnings_intel": {"available": True, "fresh": True},
                "event_risk": {"available": True, "fresh": True},
                "news_intel": {"available": True, "fresh": True},
                "relative_strength": {"available": True, "fresh": True},
                "volume_scan": {"available": True, "fresh": True},
            },
        },
        "candidates": candidates,
        "summary": summary,
    }
    return payload


@pytest.fixture()
def runtime_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "runtime"
    rd.mkdir()
    return rd


def _write_payload(runtime_dir: Path, payload: Dict[str, Any]) -> Path:
    path = runtime_dir / DYNAMIC_UNIVERSE_CANDIDATES_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests 1–7: helper behavior
# ---------------------------------------------------------------------------


def test_fresh_payload_returns_freshness_fresh(runtime_dir: Path) -> None:
    """1. Fresh payload (within TTL) yields freshness='fresh'."""
    _write_payload(runtime_dir, _baseline_payload(age_seconds=60))
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    assert ctx["freshness"] == "fresh"
    assert ctx["status"] == "ok"
    assert ctx["source_provider"] == "local_runtime_intelligence"
    assert ctx["ttl_seconds"] == DYNAMIC_UNIVERSE_CANDIDATES_DEFAULT_TTL_SEC
    assert ctx["summary"]["symbols_considered"] == 25
    assert len(ctx["top_candidates"]) == 3
    assert ctx["top_candidates"][0]["symbol"] == "LLY"


def test_stale_ttl_returns_stale_but_preserves_top_candidates(runtime_dir: Path) -> None:
    """2. Past ts_utc + ttl_seconds yields freshness='stale' but keeps summary/top_candidates."""
    _write_payload(
        runtime_dir,
        _baseline_payload(
            age_seconds=DYNAMIC_UNIVERSE_CANDIDATES_DEFAULT_TTL_SEC * 5,
            ttl=DYNAMIC_UNIVERSE_CANDIDATES_DEFAULT_TTL_SEC,
        ),
    )
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    assert ctx["freshness"] == "stale"
    assert ctx["status"] == "ok"
    assert ctx["summary"]["symbols_considered"] == 25
    assert len(ctx["top_candidates"]) == 3
    assert ctx["top_candidates"][0]["symbol"] == "LLY"


def test_missing_file_returns_missing(runtime_dir: Path) -> None:
    """3. Absent dynamic_universe_candidates.json yields freshness='missing'."""
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    assert ctx["freshness"] == "missing"
    assert ctx["status"] == "unknown"
    assert ctx["summary"]["symbols_considered"] is None
    assert ctx["summary"]["candidates_published"] is None
    assert ctx["top_candidates"] == []
    assert ctx["ts_utc"] is None
    assert ctx["ttl_seconds"] is None
    assert ctx["source_provider"] is None


def test_malformed_json_returns_malformed(runtime_dir: Path) -> None:
    """4. Unparseable JSON yields freshness='malformed' with empty summary/top_candidates."""
    (runtime_dir / DYNAMIC_UNIVERSE_CANDIDATES_FILENAME).write_text(
        "{not valid json", encoding="utf-8"
    )
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    assert ctx["freshness"] == "malformed"
    assert ctx["status"] == "unknown"
    assert ctx["summary"]["symbols_considered"] is None
    assert ctx["top_candidates"] == []


def test_partial_status_is_not_treated_as_error(runtime_dir: Path) -> None:
    """5. status='partial' is a by-design publisher state — never error."""
    _write_payload(runtime_dir, _baseline_payload(status="partial", provider_status="partial"))
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    assert ctx["status"] == "partial"
    assert ctx["freshness"] == "fresh"
    assert isinstance(ctx["top_candidates"], list)


def test_top_candidates_limited_to_ten(runtime_dir: Path) -> None:
    """6. top_candidates list is capped at DYNAMIC_UNIVERSE_CANDIDATES_TOP_LIMIT (10) entries."""
    big = [
        _candidate(rank=i + 1, symbol=f"SYM{i:02d}", score=1.0 - i * 0.01)
        for i in range(25)
    ]
    summary = {
        "symbols_considered": 25,
        "candidates_published": 25,
        "strong_rs_count": 0,
        "high_rvol_count": 0,
        "confirmed_catalyst_count": 0,
        "earnings_warning_count": 0,
    }
    _write_payload(runtime_dir, _baseline_payload(candidates=big, summary=summary))
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    assert len(ctx["top_candidates"]) == DYNAMIC_UNIVERSE_CANDIDATES_TOP_LIMIT
    # First 10 ranked candidates are preserved in publisher order.
    expected = [f"SYM{i:02d}" for i in range(DYNAMIC_UNIVERSE_CANDIDATES_TOP_LIMIT)]
    assert [r["symbol"] for r in ctx["top_candidates"]] == expected


def test_top_candidates_preserve_rank_symbol_score_reasons_warnings(runtime_dir: Path) -> None:
    """7. top_candidates rows preserve rank, symbol, score, reasons, warnings, and class fields."""
    cands = [
        _candidate(
            rank=1,
            symbol="AAA",
            score=0.91,
            reasons=["rs_strong", "confirmed_high_catalyst"],
            warnings=["rvol_low", "earnings_within_7d"],
            rs_class="strong",
            rvol_class="low",
            catalyst_strength="high",
            earnings_days=2,
        ),
    ]
    summary = {
        "symbols_considered": 1,
        "candidates_published": 1,
        "strong_rs_count": 1,
        "high_rvol_count": 0,
        "confirmed_catalyst_count": 1,
        "earnings_warning_count": 1,
    }
    _write_payload(runtime_dir, _baseline_payload(candidates=cands, summary=summary))
    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    row = ctx["top_candidates"][0]
    assert row["rank"] == 1
    assert row["symbol"] == "AAA"
    assert row["score"] == pytest.approx(0.91)
    assert row["reasons"] == ["rs_strong", "confirmed_high_catalyst"]
    assert row["warnings"] == ["rvol_low", "earnings_within_7d"]
    assert row["rs_class"] == "strong"
    assert row["rvol_class"] == "low"
    assert row["catalyst_strength"] == "high"
    assert row["earnings_days"] == 2


# ---------------------------------------------------------------------------
# Test 8: dashboard surface
# ---------------------------------------------------------------------------


def test_dashboard_intelligence_payload_includes_dynamic_universe_candidates(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8. dashboard._intelligence() includes the dynamic_universe_candidates block."""
    # The dashboard module requires CHAD_DASHBOARD_PASSWORD at import time.
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test-password-not-used")

    # Force a fresh import so the module reads the env var we just set and
    # picks up the runtime path from the test fixture.
    if "chad.dashboard.api" in sys.modules:
        del sys.modules["chad.dashboard.api"]
    api = importlib.import_module("chad.dashboard.api")

    _write_payload(runtime_dir, _baseline_payload(age_seconds=60))
    monkeypatch.setattr(api, "RUNTIME", runtime_dir)

    sb = api.StateBuilder()
    intel = sb._intelligence()

    assert "dynamic_universe_candidates" in intel
    block = intel["dynamic_universe_candidates"]
    assert block["freshness"] == "fresh"
    assert block["status"] == "ok"
    assert block["source_provider"] == "local_runtime_intelligence"
    assert block["ttl_seconds"] == DYNAMIC_UNIVERSE_CANDIDATES_DEFAULT_TTL_SEC
    assert isinstance(block["summary"], dict)
    assert isinstance(block["top_candidates"], list)
    assert block["summary"]["symbols_considered"] == 25
    assert block["top_candidates"][0]["symbol"] == "LLY"


# ---------------------------------------------------------------------------
# Tests 9–10: governance — no strategy/execution/risk imports of the helper
# ---------------------------------------------------------------------------


_HELPER_TOKENS = (
    "_load_dynamic_universe_candidates_context",
    "dynamic_universe_candidates.json",
)


def _scan_for_helper_imports(roots: List[Path]) -> List[str]:
    offenders: List[str] = []
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


def test_no_strategy_files_import_dynamic_universe_candidates_helper() -> None:
    """9. No file under chad/strategies/ may reference the helper or its file path."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _scan_for_helper_imports([repo_root / "chad" / "strategies"])
    assert offenders == [], (
        "Strategy files must not consume dynamic_universe_candidates during "
        f"the Phase D observation period. Offenders: {offenders}"
    )


def test_no_execution_or_risk_files_import_dynamic_universe_candidates_helper() -> None:
    """10. No file under chad/execution/ or chad/risk/ may reference the helper or its file path."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _scan_for_helper_imports(
        [repo_root / "chad" / "execution", repo_root / "chad" / "risk"]
    )
    assert offenders == [], (
        "Execution/risk files must not consume dynamic_universe_candidates "
        f"(intel-only). Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# Tests 11–12: helper performs no writes / no mutation
# ---------------------------------------------------------------------------


def test_helper_does_not_write_runtime_universe_json(runtime_dir: Path) -> None:
    """11. Helper must not create runtime/universe.json under any path (universe replacement is not authorized)."""
    _write_payload(runtime_dir, _baseline_payload(age_seconds=60))
    universe_path = runtime_dir / "universe.json"
    assert not universe_path.exists()
    _load_dynamic_universe_candidates_context(runtime_dir)
    assert not universe_path.exists()
    # The only artifact in the runtime dir is what the test planted.
    children = sorted(p.name for p in runtime_dir.iterdir())
    assert children == [DYNAMIC_UNIVERSE_CANDIDATES_FILENAME]


def test_helper_does_not_mutate_dynamic_universe_candidates_json(runtime_dir: Path) -> None:
    """12. Helper must not modify dynamic_universe_candidates.json on disk."""
    payload = _baseline_payload(age_seconds=60)
    path = _write_payload(runtime_dir, payload)
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    ctx = _load_dynamic_universe_candidates_context(runtime_dir)
    # Returned context is normalized — confirm we don't hand back the
    # raw publisher payload (this guards against accidental aliasing).
    assert "schema_version" not in ctx
    assert "candidates" not in ctx

    after_bytes = path.read_bytes()
    after_mtime = path.stat().st_mtime_ns
    assert before_bytes == after_bytes
    assert before_mtime == after_mtime
