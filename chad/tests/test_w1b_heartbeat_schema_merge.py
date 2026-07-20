"""W1B-1 — decision_trace_heartbeat (Writer A) schema merge.

runtime/decision_trace_heartbeat.json has two writers:
  * Writer A — chad/core/decision_trace_heartbeat.py (the 5-min timer oneshot),
  * Writer B — chad/core/orchestrator.py::_write_decision_trace_heartbeat (~60s).

Writer B has always emitted top-level allow_ibkr_live / allow_ibkr_paper;
Writer A historically emitted only the nested live_gate block. Whichever
service wrote last defined the file's shape, so the two live_posture tests
(pr03/pr04), which assert the *top-level* keys, went red ~15-20% of runs
whenever the timer had written most recently (whole-file last-writer-wins
schema alternation -- not a torn write).

W1B-1 makes Writer A additively surface the two top-level posture keys, so the
file is schema-consistent regardless of writer. This module pins that merge:
  * healthy /live-gate  -> top-level keys present and mirror the nested block;
  * live values pass through (not hard-coded);
  * endpoint unreachable -> both top-level keys null (matches Writer B's
    endpoint-down behavior) and nothing crashes.

Hermetic: /live-gate fetch is stubbed, the output path is redirected to
tmp_path, and the telegram alert path is stubbed to a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.core import decision_trace_heartbeat as hb


def _run_main_to(tmp_path: Path, monkeypatch, res: "hb.LiveGateResult") -> dict:
    out = tmp_path / "decision_trace_heartbeat.json"
    monkeypatch.setenv("DECISION_TRACE_HEARTBEAT_OUT", str(out))
    monkeypatch.setattr(hb, "_fetch_live_gate", lambda url, timeout_s: res)
    # A unit test must never reach telegram; the alert path only runs on the
    # endpoint-down case but stub it unconditionally to stay hermetic.
    monkeypatch.setattr(hb, "_alert", lambda *a, **k: False)
    rc = hb.main()
    assert rc == 0
    assert out.is_file()
    return json.loads(out.read_text(encoding="utf-8"))


def test_schema_merge_healthy_livegate(tmp_path: Path, monkeypatch) -> None:
    payload = {"allow_ibkr_live": False, "allow_ibkr_paper": True, "mode": "paper"}
    res = hb.LiveGateResult(ok=True, payload=payload, error=None, latency_ms=1.0)
    doc = _run_main_to(tmp_path, monkeypatch, res)

    # Top-level posture keys are present (the two live_posture tests read these).
    assert doc["allow_ibkr_live"] is False
    assert doc["allow_ibkr_paper"] is True
    # And they mirror the nested live_gate block both writers already share.
    assert doc["live_gate"]["allow_ibkr_live"] == doc["allow_ibkr_live"]
    assert doc["live_gate"]["allow_ibkr_paper"] == doc["allow_ibkr_paper"]


def test_top_level_posture_passes_live_values_through(tmp_path: Path, monkeypatch) -> None:
    # Guard against a hard-coded False/True: feed the opposite posture and
    # confirm the top-level keys reflect the live-gate response verbatim.
    payload = {"allow_ibkr_live": True, "allow_ibkr_paper": False}
    res = hb.LiveGateResult(ok=True, payload=payload, error=None, latency_ms=1.0)
    doc = _run_main_to(tmp_path, monkeypatch, res)

    assert doc["allow_ibkr_live"] is True
    assert doc["allow_ibkr_paper"] is False


def test_schema_merge_endpoint_down_is_null_not_crash(tmp_path: Path, monkeypatch) -> None:
    res = hb.LiveGateResult(ok=False, payload=None, error="ConnectionRefused", latency_ms=0.0)
    doc = _run_main_to(tmp_path, monkeypatch, res)

    # Keys are present but null (matches Writer B's down-behavior); no crash.
    assert "allow_ibkr_live" in doc and doc["allow_ibkr_live"] is None
    assert "allow_ibkr_paper" in doc and doc["allow_ibkr_paper"] is None
    assert doc["live_gate"] is None
    # This is precisely the state the pr03/pr04 endpoint-down skip guard keys on.
    assert not doc["live_gate"]
