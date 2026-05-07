"""Tests for GAP-026 per-strategy daily loss guard.

The guard must be report-only by default and only suppress fresh
entries when CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1. It must never
suppress exits, reductions, hedges, or other protective actions, and
it must anchor its "today" PnL window to the active epoch so that
contaminated prior-epoch PnL cannot trigger current-day suppression.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chad.risk import per_strategy_loss_guard as psg


@pytest.fixture()
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect runtime + data + config paths into tmp_path."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config" / "per_strategy_loss_limits.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(psg, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(psg, "EPOCH_STATE_PATH",
                        runtime_dir / "epoch_state.json")
    monkeypatch.setattr(psg, "TRADES_DIR", trades_dir)
    monkeypatch.setattr(psg, "CONFIG_PATH", config_path)

    config_path.write_text(
        json.dumps(
            {
                "schema_version": "per_strategy_loss_limits.v1",
                "default_mode": "report_only",
                "enforcement_env": psg.ENFORCEMENT_ENV,
                "default_limit_usd": -250.0,
                "limits_usd": {
                    "alpha": -300.0,
                    "delta": -200.0,
                    "alpha_options": -200.0,
                },
            }
        ),
        encoding="utf-8",
    )
    # Default fixture: epoch started just before "today" so today's
    # window is (UTC midnight today, now).
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
    monkeypatch.delenv(psg.ENFORCEMENT_ENV, raising=False)
    return SimpleNamespace(
        runtime=runtime_dir,
        trades=trades_dir,
        config=config_path,
    )


def _entry_signal(strategy: str, side: str = "BUY") -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol="SPY",
        side=side,
        intent="entry",
        meta={},
        tags=("entry",),
    )


def _exit_signal(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol="SPY",
        side="EXIT",
        intent="exit",
        meta={"exit": True, "intent": "exit"},
        tags=("exit", "close"),
    )


def _hedge_signal(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol="SPY",
        side="SELL",
        intent="hedge",
        meta={"intent": "hedge"},
        tags=("hedge",),
    )


def _reduce_signal(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol="SPY",
        side="SELL",
        intent="reduce",
        meta={"intent": "reduce"},
        tags=("reduce",),
    )


def _stoploss_signal(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol="SPY",
        side="SELL",
        intent="entry",  # not protective by intent alone
        meta={"reason": "stop_loss"},
        tags=("stop_loss",),
    )


def _write_trade(
    trades_dir: Path,
    *,
    strategy: str,
    pnl: float,
    exit_ts: datetime,
    fname: str = "trade_history_today.ndjson",
) -> None:
    line = json.dumps(
        {
            "payload": {
                "strategy": strategy,
                "exit_time_utc": exit_ts.isoformat(),
                "net_pnl": pnl,
            }
        }
    )
    p = trades_dir / fname
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def test_per_strategy_loss_guard_warns_report_only_by_default(
    isolated_paths, caplog: pytest.LogCaptureFixture
):
    """
    With no env flag set, a breached strategy is reported via WARNING
    on report() but no signal is suppressed by the guard.
    """
    now = datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc)
    today_noon = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    _write_trade(
        isolated_paths.trades,
        strategy="alpha",
        pnl=-450.0,  # well below -300 limit
        exit_ts=today_noon,
    )

    guard = psg.PerStrategyLossGuard(now=now)
    assert "alpha" in guard.breached
    assert guard.enforce is False

    caplog.set_level(logging.WARNING)
    guard.report()
    assert any(
        "PER_STRATEGY_LOSS_GUARD REPORT_ONLY" in rec.getMessage()
        and "strategy=alpha" in rec.getMessage()
        for rec in caplog.records
    )

    sig = _entry_signal("alpha")
    decision = guard.should_suppress(sig)
    assert decision.suppressed is False
    assert decision.reason == "report_only_mode"


def test_per_strategy_loss_guard_does_not_suppress_when_enforce_off(
    isolated_paths,
):
    """
    Even with a deeply breached strategy, an OFF env flag must keep
    every routed signal in the kept list.
    """
    now = datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc)
    today_noon = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    for _ in range(3):
        _write_trade(
            isolated_paths.trades,
            strategy="delta",
            pnl=-200.0,  # cumulative -600 vs -200 limit
            exit_ts=today_noon,
        )

    guard = psg.PerStrategyLossGuard(now=now)  # enforce off
    signals = [
        _entry_signal("delta"),
        _entry_signal("alpha"),
        _entry_signal("delta", side="SELL"),
    ]
    kept, decisions = guard.filter_signals(signals)
    assert kept == signals
    assert all(d.suppressed is False for d in decisions)


def test_per_strategy_loss_guard_suppresses_entries_when_enforce_on(
    isolated_paths, monkeypatch: pytest.MonkeyPatch
):
    """
    With ENFORCE=1, fresh entry signals from a breached strategy are
    dropped from the kept list while non-breached strategies pass.
    """
    monkeypatch.setenv(psg.ENFORCEMENT_ENV, "1")
    now = datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc)
    today_noon = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    _write_trade(
        isolated_paths.trades,
        strategy="delta",
        pnl=-275.0,  # below -200 limit
        exit_ts=today_noon,
    )

    guard = psg.PerStrategyLossGuard(now=now)
    assert guard.enforce is True
    assert "delta" in guard.breached

    delta_entry = _entry_signal("delta")
    alpha_entry = _entry_signal("alpha")
    kept, decisions = guard.filter_signals([delta_entry, alpha_entry])
    assert kept == [alpha_entry]
    delta_decision = next(d for d in decisions if d.strategy == "delta")
    assert delta_decision.suppressed is True
    assert delta_decision.reason == "loss_limit_breached_enforced"
    assert delta_decision.enforce_mode is True


def test_per_strategy_loss_guard_never_suppresses_exits_or_protective_actions(
    isolated_paths, monkeypatch: pytest.MonkeyPatch
):
    """
    With ENFORCE=1 and a breached strategy, exits, closes, reductions,
    hedges, and stop-loss-tagged signals must still pass.
    """
    monkeypatch.setenv(psg.ENFORCEMENT_ENV, "1")
    now = datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc)
    today_noon = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    _write_trade(
        isolated_paths.trades,
        strategy="alpha",
        pnl=-700.0,
        exit_ts=today_noon,
    )

    guard = psg.PerStrategyLossGuard(now=now)
    assert guard.enforce is True
    assert "alpha" in guard.breached

    protective = [
        _exit_signal("alpha"),
        _hedge_signal("alpha"),
        _reduce_signal("alpha"),
        _stoploss_signal("alpha"),
    ]
    kept, decisions = guard.filter_signals(protective)
    assert kept == protective, [d.reason for d in decisions]
    assert all(
        d.is_protective and d.suppressed is False for d in decisions
    )

    # Sanity: a fresh entry from the same strategy IS suppressed.
    entry = _entry_signal("alpha")
    decision = guard.should_suppress(entry)
    assert decision.suppressed is True


def test_epoch1_delta_loss_does_not_trigger_epoch2_current_day_suppression(
    isolated_paths, monkeypatch: pytest.MonkeyPatch
):
    """
    A massive delta loss recorded in Epoch 1 (before the active epoch
    started AND on a prior UTC day) must NOT count toward today's
    realized PnL window. Even with ENFORCE=1, no suppression should
    fire from those records.
    """
    monkeypatch.setenv(psg.ENFORCEMENT_ENV, "1")
    epoch1_loss_ts = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    _write_trade(
        isolated_paths.trades,
        strategy="delta",
        pnl=-9999.99,
        exit_ts=epoch1_loss_ts,
        fname="trade_history_20260415.ndjson",
    )

    now = datetime(2026, 5, 7, 18, 0, tzinfo=timezone.utc)
    guard = psg.PerStrategyLossGuard(now=now)
    # Today's window is bounded by max(midnight today, epoch start) so
    # 2026-04-15 records are excluded.
    assert guard.realized_pnl.get("delta", 0.0) == 0.0
    assert "delta" not in guard.breached

    delta_entry = _entry_signal("delta")
    decision = guard.should_suppress(delta_entry)
    assert decision.suppressed is False
    assert decision.reason == "within_loss_limit"
