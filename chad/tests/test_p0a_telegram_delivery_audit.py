"""P0A-A1 — telegram delivery audit + dedupe-suppression-is-not-failure.

Regression coverage for the defect class that latched
``chad-service-alert@chad-ibkr-bar-provider`` into systemd ``failed`` on
2026-06-27: ``notify()`` returned ``False`` for BOTH a dedupe-suppressed
duplicate (intentional, not an error) and a genuine send failure, and the
service-failure handler mapped any ``False`` -> ``EXIT_TELEGRAM_FAILED`` (4).
A flapping service therefore latched its own alert handler FAILED via
*successfully suppressed* duplicate alerts.

These tests assert:
  * ``notify_detailed`` distinguishes sent / suppressed / config-error /
    transport-error.
  * ``notify`` preserves its historical bool contract.
  * the service-failure alert artifact carries v2 delivery-audit fields.
  * a dedupe-suppressed duplicate is exit 0 (delivered), NOT exit 4.
  * a genuine transport/config failure records the error and exits 4.
"""

from __future__ import annotations

import json

import pytest

from chad.ops import service_failure_alert as sfa
from chad.utils import telegram_notify as tn
from chad.utils.telegram_notify import DeliveryStatus, NotifyOutcome


# ---------------------------------------------------------------------------
# notify_detailed — the four dispositions are distinguishable
# ---------------------------------------------------------------------------

def _stub_cfg(monkeypatch, *, dedupe_ttl_s: int = 900) -> None:
    cfg = tn.NotifyConfig(
        token="t", chat_id=1, timeout_s=1.0, max_retries=0, dedupe_ttl_s=dedupe_ttl_s
    )
    monkeypatch.setattr(tn, "load_config", lambda: cfg)


def test_notify_detailed_sent(monkeypatch):
    _stub_cfg(monkeypatch)
    monkeypatch.setattr(tn, "_post_send_message", lambda cfg, *, text: (True, ""))
    out = tn.notify_detailed("hello")
    assert out.status is DeliveryStatus.SENT
    assert out.sent and not out.suppressed and not out.failed
    assert out.error is None


def test_notify_detailed_suppressed_is_not_a_failure(monkeypatch):
    _stub_cfg(monkeypatch)
    # Force dedupe to suppress; _post_send_message must NOT be reached.
    monkeypatch.setattr(tn, "_dedupe_allows", lambda cfg, key: False)

    def _boom(cfg, *, text):  # pragma: no cover - must not be called
        raise AssertionError("suppressed send must not hit the wire")

    monkeypatch.setattr(tn, "_post_send_message", _boom)
    out = tn.notify_detailed("hello", dedupe_key="k")
    assert out.status is DeliveryStatus.SUPPRESSED_DEDUPE
    assert out.suppressed and not out.failed and not out.sent


def test_notify_detailed_config_error_does_not_raise(monkeypatch):
    def _raise():
        raise tn.NotifyError("Missing TELEGRAM_BOT_TOKEN")

    monkeypatch.setattr(tn, "load_config", _raise)
    out = tn.notify_detailed("hello")
    assert out.status is DeliveryStatus.CONFIG_ERROR
    assert out.failed and "TELEGRAM_BOT_TOKEN" in (out.error or "")


def test_notify_detailed_transport_error(monkeypatch):
    _stub_cfg(monkeypatch)
    monkeypatch.setattr(
        tn, "_post_send_message", lambda cfg, *, text: (False, "HTTPError 401")
    )
    out = tn.notify_detailed("hello")
    assert out.status is DeliveryStatus.TRANSPORT_ERROR
    assert out.failed and "401" in (out.error or "")


# ---------------------------------------------------------------------------
# notify() wrapper preserves its historical bool contract
# ---------------------------------------------------------------------------

def test_notify_wrapper_config_error_still_raises(monkeypatch):
    def _raise():
        raise tn.NotifyError("Missing TELEGRAM_BOT_TOKEN")

    monkeypatch.setattr(tn, "load_config", _raise)
    with pytest.raises(tn.NotifyError):
        tn.notify("hello")


def test_notify_wrapper_suppressed_returns_false(monkeypatch):
    _stub_cfg(monkeypatch)
    monkeypatch.setattr(tn, "_dedupe_allows", lambda cfg, key: False)
    assert tn.notify("hello", dedupe_key="k") is False


def test_notify_wrapper_transport_raise_on_fail(monkeypatch):
    _stub_cfg(monkeypatch)
    monkeypatch.setattr(tn, "_post_send_message", lambda cfg, *, text: (False, "boom"))
    assert tn.notify("hello") is False
    with pytest.raises(tn.NotifyError):
        tn.notify("hello", raise_on_fail=True)


# ---------------------------------------------------------------------------
# service_failure_alert — v2 delivery-audit fields + exit-code semantics
# ---------------------------------------------------------------------------

def _stub_journal_ok(monkeypatch) -> None:
    monkeypatch.setattr(sfa, "_read_journal_tail", lambda unit, n: (["l1"], None))
    monkeypatch.setattr(sfa, "_systemctl_active", lambda unit: "activating")


def _stub_notify_detailed(monkeypatch, outcome: NotifyOutcome) -> None:
    monkeypatch.setattr(
        "chad.utils.telegram_notify.notify_detailed",
        lambda *a, **k: outcome,
        raising=False,
    )


def test_schema_is_v2(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_notify_detailed(monkeypatch, NotifyOutcome(DeliveryStatus.SENT))
    res = sfa.run(failed_unit="chad-ibkr-bar-provider.service", artifact_dir=tmp_path)
    assert res.payload["schema_version"] == "service_failure_alert.v2"


def test_dry_run_records_delivery_fields(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    res = sfa.run(
        failed_unit="chad-ibkr-bar-provider.service",
        dry_run=True,
        artifact_dir=tmp_path,
    )
    doc = json.loads(res.artifact_path.read_text(encoding="utf-8"))
    assert doc["telegram_delivery_status"] == "dry_run"
    assert doc["telegram_sent"] is False
    assert doc["delivery_error"] is None
    assert res.exit_code == sfa.EXIT_OK


def test_sent_incident_records_true_and_exit_ok(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_notify_detailed(monkeypatch, NotifyOutcome(DeliveryStatus.SENT))
    res = sfa.run(failed_unit="chad-ibkr-bar-provider.service", artifact_dir=tmp_path)
    doc = json.loads(res.artifact_path.read_text(encoding="utf-8"))
    assert doc["telegram_sent"] is True
    assert doc["telegram_delivery_status"] == "sent"
    assert doc["delivery_error"] is None
    assert res.exit_code == sfa.EXIT_OK


def test_dedupe_suppression_is_exit_ok_not_4(monkeypatch, tmp_path):
    """The core regression: a suppressed duplicate must NOT latch exit 4."""
    _stub_journal_ok(monkeypatch)
    _stub_notify_detailed(monkeypatch, NotifyOutcome(DeliveryStatus.SUPPRESSED_DEDUPE))
    res = sfa.run(failed_unit="chad-ibkr-bar-provider.service", artifact_dir=tmp_path)
    doc = json.loads(res.artifact_path.read_text(encoding="utf-8"))
    assert res.exit_code == sfa.EXIT_OK
    assert doc["telegram_delivery_status"] == "suppressed_dedupe"
    assert doc["telegram_sent"] is False       # no message THIS invocation
    assert doc["delivery_error"] is None        # but NOT an error
    assert "telegram_error" not in doc          # never annotated as a failure


def test_transport_failure_records_error_and_exit_4(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_notify_detailed(
        monkeypatch, NotifyOutcome(DeliveryStatus.TRANSPORT_ERROR, "HTTPError 401")
    )
    res = sfa.run(failed_unit="chad-ibkr-bar-provider.service", artifact_dir=tmp_path)
    doc = json.loads(res.artifact_path.read_text(encoding="utf-8"))
    assert res.exit_code == sfa.EXIT_TELEGRAM_FAILED
    assert doc["telegram_sent"] is False
    assert doc["telegram_delivery_status"] == "transport_error"
    assert doc["delivery_error"] == "HTTPError 401"
    assert doc["telegram_error"] == "HTTPError 401"


def test_config_error_records_error_and_exit_4(monkeypatch, tmp_path):
    _stub_journal_ok(monkeypatch)
    _stub_notify_detailed(
        monkeypatch,
        NotifyOutcome(DeliveryStatus.CONFIG_ERROR, "Missing TELEGRAM_BOT_TOKEN"),
    )
    res = sfa.run(failed_unit="chad-ibkr-bar-provider.service", artifact_dir=tmp_path)
    doc = json.loads(res.artifact_path.read_text(encoding="utf-8"))
    assert res.exit_code == sfa.EXIT_TELEGRAM_FAILED
    assert doc["telegram_delivery_status"] == "config_error"
    assert "TELEGRAM_BOT_TOKEN" in doc["delivery_error"]
