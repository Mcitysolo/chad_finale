"""P0A-A3 — hot-path alert delivery must be loud, never silently swallowed.

live_loop previously wrapped send_stop_bus_alert / send_drawdown_alert /
send_edge_decay_alert in bare ``except Exception: pass``. A raising send
(telegram config/transport error) was swallowed with no trace — so a lost
alert (e.g. the 2026-07-13 mis-sized TLT flip) was indistinguishable from a
delivered one. ``_fire_alert_safe`` now logs loudly with the greppable marker
``ALERT_DELIVERY_FAILED kind=<kind>`` and stays fail-soft (the loop continues).
"""

from __future__ import annotations

import logging

from chad.core import live_loop


def test_fire_alert_safe_success_returns_true():
    calls = []
    ok = live_loop._fire_alert_safe("stop_bus", lambda r: calls.append(r), "halt")
    assert ok is True
    assert calls == ["halt"]


def test_fire_alert_safe_raising_send_logs_marker_and_continues(caplog):
    def _raises(*a, **k):
        raise RuntimeError("telegram config_error: Missing TELEGRAM_BOT_TOKEN")

    with caplog.at_level(logging.ERROR, logger="chad.live_loop"):
        ok = live_loop._fire_alert_safe("stop_bus", _raises, "reason")

    # Fail-soft: returns False instead of raising — the caller (the live loop)
    # continues past the alert block.
    assert ok is False
    markers = [r.getMessage() for r in caplog.records]
    assert any("ALERT_DELIVERY_FAILED" in m and "kind=stop_bus" in m for m in markers)


def test_fire_alert_safe_never_raises_into_the_loop():
    def _raises(*a, **k):
        raise ValueError("boom")

    # The whole point: no exception propagates. If this raised, the live loop
    # cycle would crash. Reaching the assert proves fail-soft.
    result = live_loop._fire_alert_safe("drawdown", _raises, -7.5, 5.0)
    assert result is False


def test_fire_alert_safe_marks_each_kind(caplog):
    def _raises(*a, **k):
        raise RuntimeError("x")

    for kind in ("stop_bus", "drawdown", "edge_decay"):
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="chad.live_loop"):
            live_loop._fire_alert_safe(kind, _raises)
        assert any(
            f"kind={kind}" in r.getMessage() and "ALERT_DELIVERY_FAILED" in r.getMessage()
            for r in caplog.records
        ), f"missing marker for kind={kind}"


def test_fire_alert_safe_uses_provided_logger(caplog):
    log = logging.getLogger("chad.live_loop.test_custom")

    def _raises(*a, **k):
        raise RuntimeError("x")

    with caplog.at_level(logging.ERROR, logger="chad.live_loop.test_custom"):
        live_loop._fire_alert_safe("edge_decay", _raises, logger=log)
    assert any("ALERT_DELIVERY_FAILED" in r.getMessage() for r in caplog.records)
