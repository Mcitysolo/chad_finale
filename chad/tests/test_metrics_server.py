from __future__ import annotations

import json
from pathlib import Path

from chad.ops.metrics_server import MetricLine, _escape_label, _render_prometheus, _safe_read_json


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
    # Labels are sorted by key, and values are escaped.
    assert rendered.startswith('chad_test_metric{')
    assert 'a="1"' in rendered
    assert 'z="x\\"y"' in rendered
    assert rendered.endswith(" 1.5")


def test_render_prometheus_includes_help_type_and_metrics() -> None:
    lines = [
        MetricLine("chad_metrics_server_up", {}, 1.0),
        MetricLine("chad_systemd_unit_active", {"unit": "x"}, 1.0),
    ]
    body = _render_prometheus(lines)
    assert "# HELP chad_metrics_server_up" in body
    assert "# TYPE chad_metrics_server_up gauge" in body
    assert 'chad_systemd_unit_active{unit="x"} 1.0' in body
    assert "chad_metrics_server_up 1.0" in body


def test_safe_read_json_missing_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    assert _safe_read_json(p) is None


def test_safe_read_json_invalid_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert _safe_read_json(p) is None


def test_safe_read_json_valid_returns_dict(tmp_path: Path) -> None:
    p = tmp_path / "ok.json"
    payload = {"a": 1, "b": {"c": True}}
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = _safe_read_json(p)
    assert isinstance(out, dict)
    assert out == payload
