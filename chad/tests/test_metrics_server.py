from __future__ import annotations

import json
import math
from pathlib import Path

from chad.ops.metrics_server import MetricLine, _escape_label, _normalize_trade_record, _paper_rollup_metrics


def test_escape_label_prometheus_safe() -> None:
    assert _escape_label('a"b') == 'a\\"b'
    assert _escape_label("a\\b") == "a\\\\b"
    assert _escape_label("a\nb") == "a\\nb"


def test_metricline_render_with_labels_sorted_and_escaped() -> None:
    ln = MetricLine(
        name="chad_test_metric",
        labels={"z": 'x"y', "a": "1"},
        value=1.5,
    )
    rendered = ln.render()
    assert rendered.startswith("chad_test_metric{")
    assert 'a="1"' in rendered
    assert 'z="x\\"y"' in rendered
    assert rendered.endswith(" 1.5")


def test_normalize_trade_record_excludes_paper_sim() -> None:
    rec = {"payload": {"tags": ["paper_sim"], "strategy": "x", "symbol": "y", "pnl": 1, "notional": 1, "is_live": False}}
    assert _normalize_trade_record(rec) is None


def test_rollup_metrics_eliminate_nan_and_inf(tmp_path: Path, monkeypatch) -> None:
    # Create a fake trades dir and write an NDJSON file with:
    # - one NaN pnl
    # - one inf pnl
    # - one valid pnl
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    f = trades_dir / "trade_history_20990101.ndjson"

    def env(payload):
        return {"timestamp_utc": "2099-01-01T00:00:00+00:00", "sequence_id": 1, "payload": payload, "prev_hash": "GENESIS", "record_hash": "x"}

    rows = [
        env({"strategy": "manual", "symbol": "EUR", "side": "BUY", "notional": 1.0, "pnl": float("nan"), "exit_time_utc": "2099-01-01T00:00:01+00:00", "entry_time_utc": "2099-01-01T00:00:00+00:00", "is_live": False, "tags": [], "extra": {}}),
        env({"strategy": "manual", "symbol": "EUR", "side": "BUY", "notional": 1.0, "pnl": float("inf"), "exit_time_utc": "2099-01-01T00:00:02+00:00", "entry_time_utc": "2099-01-01T00:00:00+00:00", "is_live": False, "tags": [], "extra": {}}),
        env({"strategy": "manual", "symbol": "EUR", "side": "BUY", "notional": 1.0, "pnl": -0.5, "exit_time_utc": "2099-01-01T00:00:03+00:00", "entry_time_utc": "2099-01-01T00:00:00+00:00", "is_live": False, "tags": [], "extra": {}}),
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Monkeypatch cwd-based discovery by calling rollup directly
    lines = _paper_rollup_metrics(trades_dir)

    m = {ln.name: ln.value for ln in lines if not ln.labels}
    # RAW includes NaN/Inf rows; LEAN excludes them (aligns with SCR/trade_stats_engine).
    assert m["chad_paper_trades_total_raw"] == 3.0
    assert m["chad_paper_trades_total"] == 1.0
    assert m["chad_paper_pnl_nonfinite_count"] == 2.0
    assert m["chad_paper_total_pnl"] == -0.5
    assert m["chad_paper_avg_pnl"] == -0.5
    assert m["chad_paper_pnl_nonfinite_count"] == 2.0
    # totals/avg must be finite
    assert math.isfinite(m["chad_paper_total_pnl"])
    assert math.isfinite(m["chad_paper_avg_pnl"])
