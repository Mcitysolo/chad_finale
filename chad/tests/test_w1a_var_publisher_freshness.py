"""
W1A-1 — VaR publisher input-freshness gating.

A timer that recomputes VaR on a cadence would, without an input-freshness
gate, stamp ts_utc=now onto var_state.json even when the underlying bars /
prices are stale — the "stale-as-fresh" failure one layer below the artifact's
own ts_utc (which EXS1 and the A4 metrics guard already inspect). These tests
verify the gate: stale market-data inputs downgrade status ok -> stale_inputs
and surface additive observability fields, while --allow-stale-inputs suppresses
only the downgrade (never the recorded fields). All writes go to tmp_path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.risk import portfolio_var
from ops import var_publisher


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _oscillating_closes(n: int, start: float = 150.0, vol: float = 0.01) -> List[float]:
    """Deterministic close series with non-zero population stdev (no RNG)."""
    closes = [float(start)]
    for i in range(1, n):
        phase = (i % 7) - 3.0  # -3..3
        closes.append(max(1.0, closes[-1] * (1.0 + vol * phase)))
    return closes


def _seed_ok_book(rt: Path, bars: Path, symbol: str = "AAPL") -> None:
    """Seed a runtime dir + bars dir that produce a status='ok' VaR report."""
    _write_json(rt / "positions_truth.json",
                {"positions": [{"symbol": symbol, "position": 100, "secType": "STK"}]})
    _write_json(rt / "price_cache.json", {"prices": {symbol: 150.0}})
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 100000.0})
    _write_json(bars / f"{symbol}.json",
                {"bars": [{"close": c} for c in _oscillating_closes(40)],
                 "symbol": symbol, "timeframe": "1d"})


def _age_file(path: Path, seconds_old: float) -> None:
    t = time.time() - seconds_old
    os.utime(path, (t, t))


# ---------------------------------------------------------------------------
# Fresh path
# ---------------------------------------------------------------------------


def test_fresh_inputs_status_ok_and_inputs_fresh(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    _seed_ok_book(rt, bars)

    state = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars)

    assert state["status"] == "ok"
    assert state["inputs_fresh"] is True
    assert state["oldest_input_age_seconds"] is not None
    assert state["oldest_input"] is not None
    assert state["input_max_age_seconds"] == var_publisher.DEFAULT_INPUT_MAX_AGE_SECONDS
    assert state["schema_version"] == "var_state.v1"


# ---------------------------------------------------------------------------
# Stale path -> downgrade
# ---------------------------------------------------------------------------


def test_stale_bar_downgrades_status_and_names_oldest_input(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    _seed_ok_book(rt, bars)
    # Age the bar file 3 days (> the 48h default bound); positions/price stay fresh.
    _age_file(bars / "AAPL.json", 3 * 86400)

    state = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars)

    assert state["status"] == "stale_inputs"
    assert state["inputs_fresh"] is False
    assert state["oldest_input"] == "bars/1d/AAPL.json"
    assert state["oldest_input_age_seconds"] > 48 * 3600
    # The A4 consumer keys var_ok on status=="ok"; stale_inputs correctly reads not-ok.
    assert state["status"] != "ok"


def test_allow_stale_inputs_suppresses_downgrade_but_not_fields(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    _seed_ok_book(rt, bars)
    _age_file(bars / "AAPL.json", 3 * 86400)

    state = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars, allow_stale_inputs=True)

    # Escape hatch keeps status ok for diagnostic runs...
    assert state["status"] == "ok"
    # ...but never masks the recorded freshness fields.
    assert state["inputs_fresh"] is False
    assert state["oldest_input"] == "bars/1d/AAPL.json"


def test_env_override_tightens_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    _seed_ok_book(rt, bars)
    _age_file(bars / "AAPL.json", 2 * 3600)  # 2h old

    # Default 48h -> fresh.
    fresh = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars)
    assert fresh["inputs_fresh"] is True
    assert fresh["status"] == "ok"

    # 1h bound via env -> the 2h-old bar is now stale.
    monkeypatch.setenv("CHAD_VAR_INPUT_MAX_AGE_SECONDS", "3600")
    stale = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars)
    assert stale["inputs_fresh"] is False
    assert stale["status"] == "stale_inputs"
    assert stale["input_max_age_seconds"] == 3600


# ---------------------------------------------------------------------------
# Schema parity
# ---------------------------------------------------------------------------


def test_schema_additive_keys_present_and_required_intact(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    _seed_ok_book(rt, bars)

    state = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars)

    required = {
        "schema_version", "ts_utc", "ttl_seconds", "status", "method",
        "confidence_levels", "var_95_1day_usd", "var_99_1day_usd",
        "portfolio_equity_cad", "var_pct_of_equity",
        "symbol_count", "symbols_used", "symbols_missing_data",
        "enforcement_active", "notes",
    }
    additive = {"inputs_fresh", "oldest_input_age_seconds", "oldest_input", "input_max_age_seconds"}
    assert required.issubset(state.keys())
    assert additive.issubset(state.keys())
    assert state["schema_version"] == "var_state.v1"  # NOT bumped — additive only
    assert state["enforcement_active"] is False


def test_empty_book_does_not_false_downgrade(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    bars = tmp_path / "bars"
    rt.mkdir()
    bars.mkdir()
    _write_json(rt / "positions_truth.json", {"positions": []})
    _write_json(rt / "price_cache.json", {"prices": {}})

    state = var_publisher.publish_var(runtime_dir=rt, bars_dir=bars)

    # No positions -> insufficient_data. The freshness gate must NOT turn that
    # into stale_inputs (there is no market data to be stale about).
    assert state["status"] == "insufficient_data"
    assert state["inputs_fresh"] is True


# ---------------------------------------------------------------------------
# compute_input_freshness unit behavior
# ---------------------------------------------------------------------------


def test_compute_input_freshness_no_inputs_is_fresh_null(tmp_path: Path) -> None:
    out = var_publisher.compute_input_freshness(
        runtime_dir=tmp_path / "runtime",
        bars_dir=tmp_path / "bars",
        symbols_used=[],
        max_age_seconds=172800,
    )
    assert out["inputs_fresh"] is True
    assert out["oldest_input_age_seconds"] is None
    assert out["oldest_input"] is None


def test_compute_input_freshness_reports_report_to_state_status_override(tmp_path: Path) -> None:
    # report_to_state_dict must honor status_override without mutating the report.
    from chad.risk.portfolio_var import VarReport, report_to_state_dict

    report = VarReport(
        status="ok", method="parametric", portfolio_equity_cad=1.0,
        var_95_1day_usd=1.0, var_99_1day_usd=1.0, symbols_used=["X"],
        symbols_missing_data=[], notes=[], symbol_count=1, var_pct_of_equity=0.1,
    )
    d = report_to_state_dict(report, ts_utc="2026-07-18T00:00:00Z",
                             status_override="stale_inputs", inputs_fresh=False)
    assert d["status"] == "stale_inputs"
    assert report.status == "ok"  # frozen report untouched
    assert d["inputs_fresh"] is False
