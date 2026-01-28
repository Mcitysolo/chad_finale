from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture()
def ops_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Import chad.ops.daily_ops_report and sandbox all filesystem roots into tmp_path
    so tests never touch /home/ubuntu/CHAD FINALE.
    """
    import chad.ops.daily_ops_report as m

    # Build a fake CHAD repo layout inside tmp_path
    root = tmp_path / "CHAD FINALE"
    runtime = root / "runtime"
    data = root / "data"
    trades = data / "trades"
    shadow = data / "shadow"
    reports = root / "reports" / "ops"
    config = root / "config"

    for p in (runtime, trades, shadow, reports, config):
        p.mkdir(parents=True, exist_ok=True)

    # Patch module-level paths
    monkeypatch.setattr(m, "ROOT", root, raising=True)
    monkeypatch.setattr(m, "RUNTIME", runtime, raising=True)
    monkeypatch.setattr(m, "DATA", data, raising=True)
    monkeypatch.setattr(m, "TRADES_DIR", trades, raising=True)
    monkeypatch.setattr(m, "REPORTS_DIR", reports, raising=True)
    monkeypatch.setattr(m, "CONFIG_DIR", config, raising=True)
    monkeypatch.setattr(m, "SYMBOL_CAPS_CONFIG", config / "symbol_caps.json", raising=True)

    # Avoid network access for metrics; force deterministic values
    def _fake_fetch_metrics_text(timeout_s: float = 2.0):
        metrics_text = "\n".join(
            [
                '# HELP chad_scr_state One-hot SCR state label.',
                '# TYPE chad_scr_state gauge',
                'chad_scr_state{state="WARMUP"} 1.0',
                'chad_scr_state{state="CAUTIOUS"} 0.0',
                'chad_scr_state{state="CONFIDENT"} 0.0',
                'chad_paper_trades_total 10.0',
                'chad_paper_win_rate 0.5',
                'chad_paper_total_pnl -1.25',
            ]
        )
        return metrics_text, None

    monkeypatch.setattr(m, "fetch_metrics_text", _fake_fetch_metrics_text, raising=True)

    # Provide basic runtime files the report expects
    (runtime / "portfolio_snapshot.json").write_text("{}", encoding="utf-8")
    (runtime / "ibkr_status.json").write_text(
        json.dumps({"ok": True, "latency_ms": 123.4, "ttl_seconds": 120, "ts_utc": "2026-01-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    (runtime / "full_execution_cycle_last.json").write_text(json.dumps({"counts": {"raw_signals": 1}}) + "\n", encoding="utf-8")
    (shadow / "shadow_state.json").write_text("{}", encoding="utf-8")

    return m


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_ledger_fallback_uses_latest_when_today_missing(ops_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    m = ops_module

    # Force "today" to a date that doesn't exist
    monkeypatch.setattr(m, "today_ymd_utc", lambda: "20990101", raising=True)

    # Create two ledgers; latest is the second one by lexicographic date
    trades_dir = m.TRADES_DIR
    older = trades_dir / "trade_history_20260115.ndjson"
    newer = trades_dir / "trade_history_20260116.ndjson"

    _write_ledger(
        older,
        [
            {"payload": {"strategy": "beta", "symbol": "AAPL", "pnl": 1.0, "extra": {}}},
        ],
    )
    _write_ledger(
        newer,
        [
            {"payload": {"strategy": "beta", "symbol": "AAPL", "pnl": 2.0, "extra": {}}},
        ],
    )

    chosen = m.ledger_path_today_or_latest()
    assert chosen == newer


def test_summarize_ledger_payload_wrapped_and_untrusted_detection(ops_module, monkeypatch: pytest.MonkeyPatch):
    m = ops_module
    trades_dir = m.TRADES_DIR

    ledger = trades_dir / "trade_history_20260116.ndjson"
    _write_ledger(
        ledger,
        [
            # Trusted realized (pnl counts)
            {"timestamp_utc": "2026-01-16T00:00:01Z", "payload": {"strategy": "beta", "symbol": "AAPL", "pnl": 1.25, "extra": {}}},
            # Untrusted entry-only (counts as record, counts as untrusted, pnl still included if present; our summarizer includes pnl if finite)
            {"timestamp_utc": "2026-01-16T00:00:02Z", "payload": {"strategy": "beta", "symbol": "AAPL", "pnl": 0.0, "extra": {"pnl_untrusted": True}}},
            # Another trusted negative
            {"timestamp_utc": "2026-01-16T00:00:03Z", "payload": {"strategy": "alpha", "symbol": "MSFT", "pnl": -2.0, "extra": {}}},
        ],
    )

    summ, err = m.summarize_ledger(ledger)
    assert err is None
    assert summ is not None
    assert summ.total_records == 3
    assert summ.beta_records == 2
    assert summ.alpha_records == 1
    assert summ.untrusted_records == 1
    assert summ.total_pnl == pytest.approx(1.25 + 0.0 - 2.0, rel=1e-12)


def test_symbol_caps_config_drives_snapshot(ops_module, monkeypatch: pytest.MonkeyPatch):
    m = ops_module

    # Write config enabling AAPL with custom thresholds
    m.SYMBOL_CAPS_CONFIG.write_text(
        json.dumps(
            {
                "enabled": True,
                "symbols": {"AAPL": {"max_trades_per_day": 111, "max_consecutive_losses": 7}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Intercept evaluator call and assert it receives the configured thresholds
    calls: list[tuple[str, int, int]] = []

    def _fake_eval(symbol: str, max_trades_per_day: int, max_consecutive_losses: int) -> Dict[str, Any]:
        calls.append((symbol, max_trades_per_day, max_consecutive_losses))
        return {"ok": True, "decision": {"allowed": True, "reason_code": "OK", "symbol": symbol, "max_trades_per_day": max_trades_per_day, "max_consecutive_losses": max_consecutive_losses}}

    monkeypatch.setattr(m, "run_symbol_cap_evaluator", _fake_eval, raising=True)

    snap = m.build_symbol_caps_snapshot()
    assert snap["enabled"] is True
    assert "AAPL" in snap["symbols"]
    assert calls == [("AAPL", 111, 7)]


def test_build_report_includes_symbol_caps_and_markdown_pretty_line(ops_module, monkeypatch: pytest.MonkeyPatch):
    m = ops_module

    # Ensure a ledger exists so ledger summary is OK
    monkeypatch.setattr(m, "today_ymd_utc", lambda: "20260116", raising=True)
    ledger = m.TRADES_DIR / "trade_history_20260116.ndjson"
    _write_ledger(
        ledger,
        [
            {"timestamp_utc": "2026-01-16T00:00:01Z", "payload": {"strategy": "beta", "symbol": "AAPL", "pnl": -1.0, "extra": {}}},
        ],
    )

    # Config enable AAPL
    m.SYMBOL_CAPS_CONFIG.write_text(
        json.dumps({"enabled": True, "symbols": {"AAPL": {"max_trades_per_day": 200, "max_consecutive_losses": 8}}}) + "\n",
        encoding="utf-8",
    )

    # Fake evaluator output to keep report deterministic
    def _fake_eval(symbol: str, max_trades_per_day: int, max_consecutive_losses: int) -> Dict[str, Any]:
        return {
            "ok": True,
            "decision": {
                "allowed": True,
                "reason_code": "OK",
                "trades_counted": 154,
                "consecutive_losses": 1,
                "max_trades_per_day": max_trades_per_day,
                "max_consecutive_losses": max_consecutive_losses,
                "ledger_path": str(ledger),
            },
        }

    monkeypatch.setattr(m, "run_symbol_cap_evaluator", _fake_eval, raising=True)

    report, md = m.build_report()

    assert report["symbol_caps"]["enabled"] is True
    assert "AAPL" in report["symbol_caps"]["symbols"]
    assert "## Symbol Caps (policy snapshot)" in md
    assert "AAPL: allowed=`True`" in md
    assert "caps=`200/8`" in md
