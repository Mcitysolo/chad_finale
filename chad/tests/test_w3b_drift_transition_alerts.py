"""W3B-4 — drift-content transition alerting (appeared/resolved, dedupe-stable).

Before W3B-4 a drift-only condition (reconciliation GREEN + drift_count>0)
flipped live-readiness RED silently: the publisher pages on
reconciliation_state RED, but nothing read the drift file's CONTENT.

Covered here:
- identity extraction: (drift_kind, symbol) only — values never in identity
  (CTF-T2); mixed_ownership_info hard-excluded (A5);
- transition matrix: empty->drift (appeared), drift->empty (resolved),
  drift->same (silent), drift->different (both), qty-change-only (silent);
- the D3-locked proof that mixed_ownership_info records can NEVER alert;
- transport-failure semantics: state not advanced -> next cycle retries;
- fail-soft: a raising notifier never propagates;
- pytest injection guard: un-injected calls no-op (no real runtime write,
  no real Telegram send).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from chad.ops import reconciliation_publisher as rp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _payload(drifts, drift_count=None, info_count=0):
    actionable = [
        d for d in drifts if d.get("drift_kind") in rp._DRIFT_ALERT_KINDS
    ]
    return {
        "schema_version": "position_guard_drift.v3",
        "drift_count": drift_count if drift_count is not None else len(actionable),
        "info_count": info_count,
        "counts_by_kind": {},
        "drifts": drifts,
    }


def _rec(kind, symbol, qty_delta=1.0):
    return {"drift_kind": kind, "symbol": symbol, "qty_delta": qty_delta}


class _Notifier:
    """Records calls; returns a stubbed NotifyOutcome-alike."""

    def __init__(self, status="sent"):
        self.status = status
        self.calls = []

    def __call__(self, message, *, severity, dedupe_key):
        self.calls.append(
            {"message": message, "severity": severity, "dedupe_key": dedupe_key}
        )
        return SimpleNamespace(status=SimpleNamespace(value=self.status))


def _run(tmp_path, payload, notifier=None, state_name="state.json"):
    notifier = notifier or _Notifier()
    disp = rp._maybe_alert_drift_transitions(
        payload, state_path=tmp_path / state_name, notify_fn=notifier
    )
    return disp, notifier


def _seed_state(tmp_path, identities, state_name="state.json"):
    (tmp_path / state_name).write_text(
        json.dumps(
            {
                "schema_version": "position_guard_drift_alert_state.v1",
                "identities": sorted(identities),
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# identity extraction
# ---------------------------------------------------------------------------


def test_identity_is_kind_and_symbol_only():
    p = _payload([_rec("qty_mismatch", "aapl", qty_delta=5.0)])
    assert rp._drift_identity_set(p) == {"qty_mismatch|AAPL"}
    # a changed delta yields the SAME identity
    p2 = _payload([_rec("qty_mismatch", "AAPL", qty_delta=7.0)])
    assert rp._drift_identity_set(p2) == {"qty_mismatch|AAPL"}


def test_identity_excludes_mixed_ownership_info():
    p = _payload(
        [
            _rec("mixed_ownership_info", "BAC"),
            _rec("broker_untracked_position", "LLY"),
        ]
    )
    assert rp._drift_identity_set(p) == {"broker_untracked_position|LLY"}


# ---------------------------------------------------------------------------
# transition matrix
# ---------------------------------------------------------------------------


def test_empty_to_empty_is_silent(tmp_path):
    disp, notifier = _run(tmp_path, _payload([]))
    assert notifier.calls == []
    assert disp["appeared"] == [] and disp["resolved"] == []
    assert disp["state_advanced"] is True


def test_empty_to_drift_alerts_appeared_warning(tmp_path):
    disp, notifier = _run(tmp_path, _payload([_rec("qty_mismatch", "AAPL")]))
    assert disp["appeared"] == ["qty_mismatch|AAPL"]
    assert len(notifier.calls) == 1
    assert notifier.calls[0]["severity"] == "warning"
    assert notifier.calls[0]["dedupe_key"].startswith("position_drift_appeared_")
    # state advanced -> replay of the same payload is silent
    disp2, notifier2 = _run(tmp_path, _payload([_rec("qty_mismatch", "AAPL")]))
    assert notifier2.calls == []
    assert disp2["appeared"] == []


def test_drift_to_empty_alerts_resolved_info(tmp_path):
    _seed_state(tmp_path, {"qty_mismatch|AAPL"})
    disp, notifier = _run(tmp_path, _payload([]))
    assert disp["resolved"] == ["qty_mismatch|AAPL"]
    assert len(notifier.calls) == 1
    assert notifier.calls[0]["severity"] == "info"
    assert notifier.calls[0]["dedupe_key"].startswith("position_drift_resolved_")


def test_drift_to_different_drift_alerts_both(tmp_path):
    _seed_state(tmp_path, {"qty_mismatch|AAPL"})
    disp, notifier = _run(
        tmp_path, _payload([_rec("broker_untracked_position", "LLY")])
    )
    assert disp["appeared"] == ["broker_untracked_position|LLY"]
    assert disp["resolved"] == ["qty_mismatch|AAPL"]
    severities = sorted(c["severity"] for c in notifier.calls)
    assert severities == ["info", "warning"]


def test_qty_change_only_is_silent(tmp_path):
    _seed_state(tmp_path, {"qty_mismatch|AAPL"})
    disp, notifier = _run(
        tmp_path, _payload([_rec("qty_mismatch", "AAPL", qty_delta=99.0)])
    )
    assert notifier.calls == []
    assert disp["appeared"] == [] and disp["resolved"] == []


def test_changed_set_mints_new_dedupe_key(tmp_path):
    _, n1 = _run(tmp_path, _payload([_rec("qty_mismatch", "AAPL")]), state_name="s1.json")
    _, n2 = _run(tmp_path, _payload([_rec("qty_mismatch", "MSFT")]), state_name="s2.json")
    assert n1.calls[0]["dedupe_key"] != n2.calls[0]["dedupe_key"]


# ---------------------------------------------------------------------------
# D3-locked proof: mixed_ownership_info can NEVER alert
# ---------------------------------------------------------------------------


def test_mixed_ownership_info_can_never_alert(tmp_path):
    """A5 (b27890a) reclassified operator-owned symbols out of drift_count so
    they never page. The live 2026-07-22 state is exactly this: 5 info
    records, drift_count=0. This payload must produce zero notifications in
    every direction — appearing, persisting, and resolving."""
    live_like = _payload(
        [
            _rec("mixed_ownership_info", "AAPL"),
            _rec("mixed_ownership_info", "BAC"),
            _rec("mixed_ownership_info", "LLY"),
            _rec("mixed_ownership_info", "MSFT"),
            _rec("mixed_ownership_info", "SPY"),
        ],
        drift_count=0,
        info_count=5,
    )
    # appearing from empty
    disp, notifier = _run(tmp_path, live_like)
    assert notifier.calls == [] and disp["appeared"] == []
    # persisting
    disp, notifier = _run(tmp_path, live_like)
    assert notifier.calls == []
    # resolving to empty
    disp, notifier = _run(tmp_path, _payload([]))
    assert notifier.calls == [] and disp["resolved"] == []
    # and the identity extractor is structurally blind to the kind
    assert rp._drift_identity_set(live_like) == set()
    assert "mixed_ownership_info" not in rp._DRIFT_ALERT_KINDS


# ---------------------------------------------------------------------------
# delivery semantics
# ---------------------------------------------------------------------------


def test_transport_failure_leaves_state_for_retry(tmp_path):
    failing = _Notifier(status="transport_error")
    disp, _ = _run(tmp_path, _payload([_rec("qty_mismatch", "AAPL")]), notifier=failing)
    assert disp["state_advanced"] is False
    assert not (tmp_path / "state.json").exists()
    # next cycle retries the same appeared set
    disp2, retry = _run(tmp_path, _payload([_rec("qty_mismatch", "AAPL")]))
    assert disp2["appeared"] == ["qty_mismatch|AAPL"]
    assert len(retry.calls) == 1
    assert disp2["state_advanced"] is True


def test_dedupe_suppression_counts_as_delivered(tmp_path):
    suppressed = _Notifier(status="suppressed_dedupe")
    disp, _ = _run(
        tmp_path, _payload([_rec("qty_mismatch", "AAPL")]), notifier=suppressed
    )
    assert disp["state_advanced"] is True


def test_raising_notifier_never_propagates(tmp_path):
    def boom(message, *, severity, dedupe_key):
        raise RuntimeError("telegram down")

    disp = rp._maybe_alert_drift_transitions(
        _payload([_rec("qty_mismatch", "AAPL")]),
        state_path=tmp_path / "state.json",
        notify_fn=boom,
    )
    # fail-soft: helper swallows, reconciliation continues
    assert disp["state_advanced"] is False


def test_corrupt_state_file_treated_as_first_run(tmp_path):
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")
    disp, notifier = _run(tmp_path, _payload([_rec("qty_mismatch", "AAPL")]))
    assert disp["appeared"] == ["qty_mismatch|AAPL"]
    assert len(notifier.calls) == 1


# ---------------------------------------------------------------------------
# pytest injection guard
# ---------------------------------------------------------------------------


def test_uninjected_call_noops_under_pytest(tmp_path):
    disp = rp._maybe_alert_drift_transitions(_payload([_rec("qty_mismatch", "AAPL")]))
    assert disp.get("skipped") == "pytest_requires_explicit_injection"
    assert disp["state_advanced"] is False

    disp = rp._maybe_alert_drift_transitions(
        _payload([]), state_path=tmp_path / "s.json"
    )  # notifier still missing
    assert disp.get("skipped") == "pytest_requires_explicit_injection"


# ---------------------------------------------------------------------------
# coach templates
# ---------------------------------------------------------------------------


def test_coach_templates_registered_and_render():
    from chad.utils.coach_voice import format_alert

    appeared = format_alert(
        "position_drift",
        {"appeared": ["qty_mismatch|AAPL"], "counts_by_kind": {"qty_mismatch": 1}},
        mode="PRO",
    )
    assert "AAPL" in appeared
    resolved = format_alert(
        "position_drift_resolved",
        {"resolved": ["broker_untracked_position|LLY"], "still_active": 0},
        mode="PRO",
    )
    assert "LLY" in resolved


def test_wiring_call_site_is_fail_soft():
    """The _emit_position_guard_drift call site must wrap the helper in its
    own try/except (belt-and-braces on top of the helper's internal guard)."""
    import inspect

    src = inspect.getsource(rp._emit_position_guard_drift)
    assert "_maybe_alert_drift_transitions" in src
    idx = src.index("_maybe_alert_drift_transitions")
    assert "try:" in src[:idx], "helper call must be inside a try block"
