"""
GAP-015A / GAP-016A — report-only VaR + drawdown tests.

These tests verify ONLY that the report-only artifacts and pure-function
modules behave correctly. They explicitly assert that no live_loop
suppression has been wired in this batch.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.risk import drawdown_guard, portfolio_var  # noqa: E402

PYTHON = "/home/ubuntu/chad_finale/venv/bin/python3"
VAR_PUBLISHER = REPO_ROOT / "ops" / "var_publisher.py"
DRAWDOWN_PUBLISHER = REPO_ROOT / "ops" / "drawdown_publisher.py"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_ndjson(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_bars(symbol: str, closes: List[float], start_date: str = "2025-05-07") -> Dict[str, Any]:
    bars = []
    for i, c in enumerate(closes):
        bars.append({"close": float(c), "open": float(c), "high": float(c), "low": float(c),
                     "volume": 1000.0, "ts_utc": f"2026-01-{(i % 28) + 1:02d}"})
    return {
        "bars": bars,
        "source": "test",
        "symbol": symbol,
        "timeframe": "1d",
        "ts_utc": start_date,
        "ttl_seconds": 86400,
    }


def _build_synthetic_bars(symbol: str, n: int, start: float, drift: float, vol: float) -> List[float]:
    """Deterministic synthetic close series with controlled volatility."""
    closes: List[float] = [float(start)]
    # Use a simple sin-based oscillation so stdev > 0 without RNG.
    for i in range(1, n):
        phase = (i % 7) - 3.0  # -3..3
        step = drift + vol * phase
        closes.append(max(1.0, closes[-1] * (1.0 + step)))
    return closes


# ---------------------------------------------------------------------------
# GAP-015A — portfolio_var pure-function tests
# ---------------------------------------------------------------------------


def test_portfolio_var_empty_portfolio_returns_zero_report(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_json(rt / "positions_truth.json", {"positions": []})
    _write_json(rt / "price_cache.json", {"prices": {}})
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 100000.0})

    report = portfolio_var.compute_portfolio_var(
        positions_truth_path=rt / "positions_truth.json",
        price_cache_path=rt / "price_cache.json",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        bars_dir=tmp_path / "bars",
    )

    assert report.status == "insufficient_data"
    assert report.var_95_1day_usd == 0.0
    assert report.var_99_1day_usd == 0.0
    assert report.symbol_count == 0
    assert report.symbols_used == []
    assert report.portfolio_equity_usd == 100000.0


def test_portfolio_var_non_empty_portfolio_returns_positive_var(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    rt.mkdir()
    bars.mkdir()

    _write_json(
        rt / "positions_truth.json",
        {"positions": [{"symbol": "AAA", "secType": "STK", "position": 100.0}]},
    )
    _write_json(rt / "price_cache.json", {"prices": {"AAA": 50.0}})
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 100000.0})

    closes = _build_synthetic_bars("AAA", n=60, start=50.0, drift=0.0001, vol=0.005)
    _write_json(bars / "AAA.json", _make_bars("AAA", closes))

    report = portfolio_var.compute_portfolio_var(
        positions_truth_path=rt / "positions_truth.json",
        price_cache_path=rt / "price_cache.json",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        bars_dir=bars,
    )

    assert report.status == "ok"
    assert report.symbol_count == 1
    assert "AAA" in report.symbols_used
    assert report.var_95_1day_usd > 0.0
    assert report.var_99_1day_usd > report.var_95_1day_usd
    assert report.portfolio_equity_usd == 100000.0
    assert math.isfinite(report.var_pct_of_equity)


def test_portfolio_var_insufficient_bars_marks_insufficient_data(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    rt.mkdir()
    bars.mkdir()

    _write_json(
        rt / "positions_truth.json",
        {"positions": [{"symbol": "BBB", "secType": "STK", "position": 10.0}]},
    )
    _write_json(rt / "price_cache.json", {"prices": {"BBB": 100.0}})
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 50000.0})

    # Only 5 closes — far below MIN_BARS_FOR_VAR.
    _write_json(bars / "BBB.json", _make_bars("BBB", [100.0, 101.0, 99.0, 100.5, 102.0]))

    report = portfolio_var.compute_portfolio_var(
        positions_truth_path=rt / "positions_truth.json",
        price_cache_path=rt / "price_cache.json",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        bars_dir=bars,
    )

    assert report.status == "insufficient_data"
    assert report.var_95_1day_usd == 0.0
    assert report.symbol_count == 0
    assert any("BBB" in m and "insufficient_bars" in m for m in report.symbols_missing_data)


# ---------------------------------------------------------------------------
# GAP-015A — var_publisher schema test
# ---------------------------------------------------------------------------


def _run_publisher(script: Path, runtime_dir: Path, bars_dir: Path = None, env_extra: Dict[str, str] = None) -> Dict[str, Any]:
    env = os.environ.copy()
    env["CHAD_RUNTIME_DIR"] = str(runtime_dir)
    env["CHAD_SKIP_IB_CONNECT"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"publisher failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return json.loads(proc.stdout)


def test_var_publisher_writes_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_json(rt / "positions_truth.json", {"positions": []})
    _write_json(rt / "price_cache.json", {"prices": {}})
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 12345.67})

    summary = _run_publisher(VAR_PUBLISHER, rt)
    assert summary["ok"] is True
    assert summary["schema_version"] == "var_state.v1"

    state_path = rt / "var_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())

    required_keys = {
        "schema_version", "ts_utc", "ttl_seconds", "status", "method",
        "confidence_levels", "var_95_1day_usd", "var_99_1day_usd",
        "portfolio_equity_usd", "var_pct_of_equity",
        "symbol_count", "symbols_used", "symbols_missing_data",
        "enforcement_active", "notes",
    }
    assert required_keys.issubset(state.keys())
    assert state["schema_version"] == "var_state.v1"
    assert state["ttl_seconds"] == 3600
    assert state["confidence_levels"] == [0.95, 0.99]
    assert state["enforcement_active"] is False
    assert state["status"] == "insufficient_data"
    assert state["var_95_1day_usd"] == 0.0
    assert state["var_99_1day_usd"] == 0.0


# ---------------------------------------------------------------------------
# GAP-016A — drawdown_guard pure-function tests
# ---------------------------------------------------------------------------


def test_drawdown_guard_identifies_hwm_from_history(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_ndjson(
        rt / "equity_history.ndjson",
        [
            {"date_utc": "2026-04-01", "total_equity_usd": 100000.0},
            {"date_utc": "2026-04-02", "total_equity_usd": 110000.0},  # HWM here
            {"date_utc": "2026-04-03", "total_equity_usd": 105000.0},
            {"date_utc": "2026-04-04", "total_equity_usd": 102000.0},
        ],
    )
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 102000.0})
    _write_json(rt / "pnl_state.json", {"account_equity": 102000.0})

    report = drawdown_guard.compute_drawdown(
        equity_history_path=rt / "equity_history.ndjson",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        pnl_state_path=rt / "pnl_state.json",
    )

    assert report.status == "ok"
    assert report.hwm_usd == 110000.0
    assert report.current_equity_usd == 102000.0
    assert report.drawdown_pct == pytest.approx((102000.0 - 110000.0) / 110000.0 * 100.0, abs=1e-6)
    assert report.enforcement_active is False


def test_drawdown_guard_report_only_halt_when_threshold_exceeded(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_ndjson(
        rt / "equity_history.ndjson",
        [
            {"date_utc": "2026-04-01", "total_equity_usd": 200000.0},
            {"date_utc": "2026-04-02", "total_equity_usd": 150000.0},
        ],
    )
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 150000.0})

    # 25% drawdown from 200k HWM. Default threshold is -15% → should halt (report-only).
    report = drawdown_guard.compute_drawdown(
        equity_history_path=rt / "equity_history.ndjson",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
    )

    assert report.halt is True
    assert report.enforcement_active is False
    assert report.drawdown_pct < report.halt_threshold_pct  # more negative


def test_drawdown_guard_does_not_enforce_by_default(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_ndjson(
        rt / "equity_history.ndjson",
        [
            {"date_utc": "2026-04-01", "total_equity_usd": 100000.0},
            {"date_utc": "2026-04-02", "total_equity_usd": 50000.0},
        ],
    )
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 50000.0})

    # Massive 50% drawdown. Even so the report must keep enforcement_active=False.
    report = drawdown_guard.compute_drawdown(
        equity_history_path=rt / "equity_history.ndjson",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
    )
    assert report.enforcement_active is False
    assert report.halt is True  # observability flag still flips

    state = drawdown_guard.report_to_state_dict(report, ts_utc="2026-05-07T10:00:00Z")
    assert state["enforcement_active"] is False


def test_drawdown_publisher_writes_schema(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_ndjson(
        rt / "equity_history.ndjson",
        [
            {"date_utc": "2026-04-01", "total_equity_usd": 100000.0},
            {"date_utc": "2026-04-02", "total_equity_usd": 95000.0},
        ],
    )
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 95000.0})

    summary = _run_publisher(DRAWDOWN_PUBLISHER, rt)
    assert summary["ok"] is True
    assert summary["schema_version"] == "drawdown_state.v1"

    state_path = rt / "drawdown_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())

    required_keys = {
        "schema_version", "ts_utc", "ttl_seconds", "status",
        "current_equity_usd", "hwm_usd", "drawdown_pct",
        "halt_threshold_pct", "halt", "enforcement_active",
        "sample_count", "lookback_days", "notes",
    }
    assert required_keys.issubset(state.keys())
    assert state["schema_version"] == "drawdown_state.v1"
    assert state["ttl_seconds"] == 300
    assert state["enforcement_active"] is False
    assert state["hwm_usd"] == 100000.0
    assert state["current_equity_usd"] == 95000.0


# ---------------------------------------------------------------------------
# Cross-cutting: report-only batch must NOT have live_loop suppression
# ---------------------------------------------------------------------------


def test_no_live_loop_suppression_in_report_only_batch() -> None:
    """
    GAP-015A/016A explicitly forbid wiring suppression into live_loop in this
    batch. We verify by reading live_loop.py and asserting that it does NOT
    import/reference our new modules nor the report-only artifacts.
    """
    candidates = [
        REPO_ROOT / "live_loop.py",
        REPO_ROOT / "chad" / "core" / "live_loop.py",
        REPO_ROOT / "chad" / "live_loop.py",
    ]
    forbidden_tokens = (
        "portfolio_var",
        "drawdown_guard",
        "var_state.json",
        "drawdown_state.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden_tokens:
            assert token not in text, (
                f"GAP-015A/016A batch must not wire {token} into {path}; "
                f"suppression is deferred to a later batch."
            )

    # Sanity: ensure the new modules themselves do not import or write stop_bus.
    # We strip docstrings/comments so disclaimers like "does not write stop_bus"
    # do not falsely trip the guard — we only flag executable references.
    import re

    def _strip_docs_and_comments(src: str) -> str:
        no_triple_d = re.sub(r'"""[\s\S]*?"""', "", src)
        no_triple_s = re.sub(r"'''[\s\S]*?'''", "", no_triple_d)
        no_line_comments = re.sub(r"(?m)#.*$", "", no_triple_s)
        return no_line_comments

    new_files = [
        REPO_ROOT / "chad" / "risk" / "portfolio_var.py",
        REPO_ROOT / "chad" / "risk" / "drawdown_guard.py",
        REPO_ROOT / "ops" / "var_publisher.py",
        REPO_ROOT / "ops" / "drawdown_publisher.py",
    ]
    for path in new_files:
        code_only = _strip_docs_and_comments(path.read_text(encoding="utf-8"))
        assert "stop_bus" not in code_only.lower(), (
            f"report-only batch must not touch stop_bus (executable code in {path})"
        )
