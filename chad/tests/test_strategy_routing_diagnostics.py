"""Tests for GAP-025 strategy routing diagnostics writer.

The writer must:
  * produce a runtime/strategy_routing_diagnostics.json artifact
  * be a pure observer — running it must not mutate routing state
  * include zero_fill flags and block-reason counts when supplied
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from chad.ops import strategy_routing_diagnostics as rd


@pytest.fixture()
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect runtime + data paths to a tmp dir."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    fills_dir = tmp_path / "data" / "fills"
    fills_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rd, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(rd, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        rd, "DIAGNOSTICS_PATH",
        runtime_dir / "strategy_routing_diagnostics.json",
    )
    monkeypatch.setattr(
        rd, "EPOCH_STATE_PATH", runtime_dir / "epoch_state.json"
    )
    monkeypatch.setattr(
        rd, "DYNAMIC_CAPS_PATH", runtime_dir / "dynamic_caps.json"
    )
    monkeypatch.setattr(
        rd, "ALLOCATIONS_PATH", runtime_dir / "strategy_allocations.json"
    )

    # Realistic fixtures — Epoch 2 is active.
    (runtime_dir / "epoch_state.json").write_text(
        json.dumps(
            {
                "schema_version": "epoch_state.v1",
                "active_epoch": "CHAD_v8.9_Paper_Epoch_2",
                "epoch_started_at_utc": "2026-05-04T00:54:30Z",
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "dynamic_caps.json").write_text(
        json.dumps(
            {
                "strategies": {
                    "alpha": {"dollar_cap": 733.88},
                    "delta": {"dollar_cap": 100.00},
                    "gamma_reversion": {"dollar_cap": 250.00},
                }
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "strategy_allocations.json").write_text(
        json.dumps(
            {
                "schema_version": "strategy_allocations.v1",
                "allocations": {
                    "delta": {
                        "halted": True,
                        "halt_reason": "consecutive_negative_15",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    # alpha has fills in epoch; gamma_reversion has none.
    (fills_dir / "FILLS_20260504.ndjson").write_text(
        json.dumps(
            {
                "payload": {
                    "strategy": "alpha",
                    "fill_time_utc": "2026-05-04T15:00:00+00:00",
                    "fill_price": 100.0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (trades_dir / "trade_history_20260504.ndjson").write_text(
        json.dumps(
            {
                "payload": {
                    "strategy": "alpha",
                    "exit_time_utc": "2026-05-04T16:00:00+00:00",
                    "net_pnl": 12.50,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return runtime_dir


def _signal(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(strategy=strategy, symbol="SPY", side="BUY")


def test_strategy_routing_diagnostics_written_without_affecting_routing(
    isolated_runtime,
):
    """
    Writing the diagnostic artifact must produce the file at the
    expected path and must not mutate routing state. We assert the
    artifact appears, the input signal list is unchanged, and that
    routing-state files (strategy_allocations / dynamic_caps) are
    byte-identical before and after the write.
    """
    tracker = rd.RoutingDiagnostics()
    incoming = [_signal("alpha"), _signal("alpha"), _signal("delta")]
    tracker.observe_signals("signals_generated_this_cycle", incoming)
    tracker.observe_signals(
        "signals_after_regime_gate", [_signal("alpha"), _signal("delta")]
    )

    alloc_before = (
        isolated_runtime / "strategy_allocations.json"
    ).read_bytes()
    caps_before = (isolated_runtime / "dynamic_caps.json").read_bytes()
    incoming_snapshot = list(incoming)

    written = rd.write_diagnostics(tracker)
    assert written is not None
    assert written.exists()

    # No mutation of inputs.
    assert incoming == incoming_snapshot
    # No mutation of routing state files.
    assert (
        isolated_runtime / "strategy_allocations.json"
    ).read_bytes() == alloc_before
    assert (isolated_runtime / "dynamic_caps.json").read_bytes() == (
        caps_before
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "strategy_routing_diagnostics.v1"
    assert payload["tracker_attached"] is True
    strategies = payload["strategies"]
    assert (
        strategies["alpha"]["signals_generated_this_cycle"] == 2
    ), strategies["alpha"]
    assert strategies["alpha"]["signals_after_regime_gate"] == 1
    assert strategies["delta"]["halted"] is True
    assert strategies["delta"]["halt_reason"] == "consecutive_negative_15"


def test_zero_fill_strategy_diagnostics_include_block_reasons(
    isolated_runtime,
):
    """
    A strategy with zero fills in Epoch 2 must surface that fact and
    any per-cycle block reasons recorded by the tracker.
    """
    tracker = rd.RoutingDiagnostics()
    tracker.observe_signals(
        "signals_generated_this_cycle", [_signal("gamma_reversion")]
    )
    tracker.observe_block(
        "gamma_reversion", "regime_gate_drop", count=1
    )
    tracker.observe_block(
        "gamma_reversion", "net_exposure_block", count=2
    )

    written = rd.write_diagnostics(tracker)
    assert written is not None
    payload = json.loads(written.read_text(encoding="utf-8"))
    info = payload["strategies"]["gamma_reversion"]
    assert info["zero_fill_epoch2"] is True
    assert info["last_fill_at"] is None
    assert info["blocked_reasons"] == {
        "regime_gate_drop": 1,
        "net_exposure_block": 2,
    }
    assert info["current_cap"] == pytest.approx(250.00)
    # alpha did fill in epoch — must be False.
    assert payload["strategies"]["alpha"]["zero_fill_epoch2"] is False


def test_routing_diagnostics_tracks_generated_signals(isolated_runtime):
    """
    observe_signals('signals_generated_this_cycle', ...) records a
    per-strategy count of newly-emitted signals. The artifact must
    surface those counts and mark the stage as observed.
    """
    tracker = rd.RoutingDiagnostics()
    tracker.observe_signals(
        "signals_generated_this_cycle",
        [
            _signal("alpha"),
            _signal("alpha"),
            _signal("alpha"),
            _signal("delta"),
        ],
    )

    written = rd.write_diagnostics(tracker)
    assert written is not None
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert (
        "signals_generated_this_cycle" in payload["stages_observed"]
    )
    assert payload["null_reasons"].get(
        "signals_generated_this_cycle"
    ) is None
    strategies = payload["strategies"]
    assert strategies["alpha"]["signals_generated_this_cycle"] == 3
    assert strategies["delta"]["signals_generated_this_cycle"] == 1


def test_routing_diagnostics_tracks_regime_gate_drops(isolated_runtime):
    """
    observe_drop on the regime gate attributes the per-strategy delta
    to 'regime_inactive', and the surviving count is recorded under
    signals_after_regime_gate.
    """
    tracker = rd.RoutingDiagnostics()
    pre = [_signal("delta"), _signal("delta"), _signal("alpha")]
    post = [_signal("alpha")]
    tracker.observe_signals("signals_generated_this_cycle", pre)
    tracker.observe_drop(pre, post, "regime_inactive")
    tracker.observe_signals("signals_after_regime_gate", post)

    written = rd.write_diagnostics(tracker)
    assert written is not None
    payload = json.loads(written.read_text(encoding="utf-8"))
    delta_info = payload["strategies"]["delta"]
    assert delta_info["blocked_reasons"].get("regime_inactive") == 2
    assert delta_info["signals_after_regime_gate"] == 0
    assert payload["strategies"]["alpha"][
        "signals_after_regime_gate"
    ] == 1


def test_routing_diagnostics_tracks_net_exposure_blocks(
    isolated_runtime,
):
    """
    A net-exposure-blocked signal must be attributed to net_exposure
    and reflected in the surviving count.
    """
    tracker = rd.RoutingDiagnostics()
    pre = [_signal("alpha"), _signal("alpha"), _signal("gamma_reversion")]
    post = [_signal("alpha")]
    tracker.observe_signals("signals_generated_this_cycle", pre)
    tracker.observe_drop(pre, post, "net_exposure")
    tracker.observe_signals("signals_after_net_exposure_gate", post)

    written = rd.write_diagnostics(tracker)
    payload = json.loads(written.read_text(encoding="utf-8"))
    alpha_info = payload["strategies"]["alpha"]
    gamma_info = payload["strategies"]["gamma_reversion"]
    assert alpha_info["blocked_reasons"].get("net_exposure") == 1
    assert gamma_info["blocked_reasons"].get("net_exposure") == 1
    assert alpha_info["signals_after_net_exposure_gate"] == 1
    assert gamma_info["signals_after_net_exposure_gate"] == 0


def test_routing_diagnostics_tracks_halted_or_edge_decay_drops(
    isolated_runtime,
):
    """
    The edge-decay/halt filter is attributed to 'edge_decay'. A halted
    strategy's signals must show up as drops at the
    signals_after_edge_decay_or_halt_filter stage.
    """
    tracker = rd.RoutingDiagnostics()
    pre = [_signal("delta"), _signal("delta"), _signal("alpha")]
    # Simulate halt filter — only alpha survives.
    post = [_signal("alpha")]
    tracker.observe_signals("signals_generated_this_cycle", pre)
    tracker.observe_drop(pre, post, "edge_decay")
    tracker.observe_signals(
        "signals_after_edge_decay_or_halt_filter", post
    )

    written = rd.write_diagnostics(tracker)
    payload = json.loads(written.read_text(encoding="utf-8"))
    delta_info = payload["strategies"]["delta"]
    assert delta_info["blocked_reasons"].get("edge_decay") == 2
    assert (
        delta_info["signals_after_edge_decay_or_halt_filter"] == 0
    )
    assert (
        payload["strategies"]["alpha"][
            "signals_after_edge_decay_or_halt_filter"
        ]
        == 1
    )
    # Disk-derived halt info is preserved.
    assert delta_info["halted"] is True


def test_routing_diagnostics_no_tracker_not_attached_when_instrumented(
    isolated_runtime,
):
    """
    With every wired stage observed, the artifact must NOT contain a
    null_reason of 'tracker_not_attached_for_this_cycle' for any
    stage. Stages that are deliberately not present (e.g. the spam
    governor) are marked 'stage_not_present'.
    """
    tracker = rd.RoutingDiagnostics()
    tracker.mark_stage_not_present("signals_after_spam_governor")
    sig = [_signal("alpha")]
    for stage in (
        "signals_generated_this_cycle",
        "signals_after_edge_decay_or_halt_filter",
        "signals_after_loss_guard_report_only_or_enforced",
        "signals_after_net_exposure_gate",
        "signals_after_strategy_throttle",
        "signals_after_regime_gate",
    ):
        tracker.observe_signals(stage, sig)

    written = rd.write_diagnostics(tracker)
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["tracker_attached"] is True
    null_reasons = payload["null_reasons"]
    # No tracker_not_attached anywhere.
    assert "tracker_not_attached_for_this_cycle" not in (
        null_reasons.values()
    )
    # spam_governor explicitly marked not present.
    assert (
        null_reasons.get("signals_after_spam_governor")
        == "stage_not_present"
    )
    # All other instrumented stages have no null_reason at all.
    for stage in (
        "signals_generated_this_cycle",
        "signals_after_edge_decay_or_halt_filter",
        "signals_after_loss_guard_report_only_or_enforced",
        "signals_after_net_exposure_gate",
        "signals_after_strategy_throttle",
        "signals_after_regime_gate",
    ):
        assert null_reasons.get(stage) is None, (stage, null_reasons)


def test_diagnostics_failure_does_not_break_live_loop(
    isolated_runtime, monkeypatch: pytest.MonkeyPatch
):
    """
    Any internal failure inside the diagnostics writer must NOT raise
    out to the live loop. Force build_diagnostics to throw, then call
    write_diagnostics — it must return None silently.

    Additionally verify the live_loop wire-in wraps the call in a
    try/except so a propagating exception still couldn't break it.
    """

    def boom(*a, **kw):
        raise RuntimeError("synthetic_diag_failure")

    monkeypatch.setattr(rd, "build_diagnostics", boom)
    result = rd.write_diagnostics(rd.RoutingDiagnostics())
    assert result is None

    # Force atomic_write_json to fail too — same guarantee.
    monkeypatch.undo()

    def disk_boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(rd, "atomic_write_json", disk_boom)
    assert rd.write_diagnostics(rd.RoutingDiagnostics()) is None

    # And: the live_loop wire-in is wrapped in try/except. Inspect the
    # source so future refactors can't accidentally remove the guard.
    import inspect
    from chad.core import live_loop

    src = inspect.getsource(live_loop.run_once)
    assert "_write_routing_diagnostics(_routing_diag)" in src
    # Find the call site and verify it sits inside a try block whose
    # except clause logs at debug level rather than re-raising.
    idx = src.index("_write_routing_diagnostics(_routing_diag)")
    preceding = src[:idx]
    # Last 'try:' before the call should be paired with an except.
    assert preceding.rfind("try:") > preceding.rfind("except"), (
        "diagnostics call must be wrapped in try/except in live_loop"
    )
