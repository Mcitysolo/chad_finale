"""W1B-3 — edge-decay "roach motel": halts now self-clear (with guardrails).

Two roach-motel mechanisms are fixed:
  * Mechanism 1 — check_all() iterated only the trusted-ledger keys, so a
    strategy halted-but-absent from the trusted ledger (e.g. alpha_crypto,
    whose fills are all pnl_untrusted/validate_only and excluded) was never
    re-evaluated and its halt could never clear. check_all() now unions the
    halted-store keys in.
  * Mechanism 2 — the non-halt branches set halted:False in memory only, never
    writing the store, so even a ledger-resident strategy that recovered stayed
    halted forever. The recovery/TTL clear now calls clear_strategy_halt (writes
    the store).

Binding rider verified here:
  (i)   self-clear decisions use TRUSTED-ledger data only — pnl_untrusted /
        validate_only rows can never clear a halt;
  (ii)  halted:False is WRITTEN to disk (both mechanisms), not just returned;
  (iii) every self-clear emits a marker AND a coach-voiced NOTIFY;
  (iv)  operator-imposed and operator-cleared halts are untouched by the
        self-clear path.

All state is tmp_path; the notification path is spied so tests never reach
telegram or write under the working-tree runtime/.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import chad.risk.edge_decay_monitor as edm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_store(
    path: Path,
    strategy: str,
    *,
    halted: bool = True,
    halted_at: str = "",
    halted_by: str | None = "edge_decay_monitor",
    halt_reason: str = "consecutive_negative_6",
    consecutive_negative: int = 6,
) -> None:
    entry = {
        "halted": halted,
        "halt_reason": halt_reason,
        "halted_at": halted_at,
        "cleared_at": None,
        "cleared_by": "",
        "consecutive_negative": consecutive_negative,
    }
    if halted_by is not None:
        entry["halted_by"] = halted_by
    payload = {
        "schema_version": "strategy_allocations.v1",
        "updated_at": "2026-07-01T00:00:00+00:00",
        "allocations": {strategy: entry},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _empty_glob(tmp_path: Path) -> str:
    """A trades glob that matches no files -> the strategy is ledger-absent."""
    return str(tmp_path / "no_trades" / "trade_history_*.ndjson")


def _write_trades(tmp_path: Path, strategy: str, pnls, *, untrusted: bool = False) -> str:
    d = tmp_path / "data" / "trades"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "trade_history_test.ndjson"
    with f.open("w", encoding="utf-8") as fh:
        for i, pnl in enumerate(pnls):
            rec = {"strategy": strategy, "pnl": pnl}
            if untrusted:
                rec["pnl_untrusted"] = True
            fh.write(json.dumps(rec) + "\n")
    return str(d / "trade_history_*.ndjson")


def _entry(path: Path, strategy: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["allocations"][strategy]


@pytest.fixture
def spy_notify(monkeypatch):
    """Replace the whole notification helper with a recording spy so clear
    tests stay fully hermetic (no telegram, no runtime write)."""
    calls = []
    monkeypatch.setattr(
        edm, "_emit_auto_clear_notification",
        lambda strategy, **kw: calls.append({"strategy": strategy, **kw}),
    )
    return calls


# ---------------------------------------------------------------------------
# Mechanism 1 — ledger-absent + TTL
# ---------------------------------------------------------------------------


def test_ledger_absent_ttl_elapsed_clears_and_persists(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    old = (_now() - timedelta(days=30)).isoformat()
    _seed_store(store, "alpha_crypto", halted_at=old)

    edm.EdgeDecayMonitor(
        halt_ttl_days=14, clear_on_recovery=True,
        allocations_path=store, trades_glob=_empty_glob(tmp_path),
    ).check_all()

    e = _entry(store, "alpha_crypto")  # rider (ii): re-read from disk
    assert e["halted"] is False
    assert e["cleared_by"] == "edge_decay_monitor_auto"
    assert e["clear_reason"] == "auto_ttl_no_trusted_losing_evidence"
    # GAP-018 counter semantics preserved on the auto path.
    assert e["consecutive_negative"] == 0
    assert e["previous_consecutive_negative"] == 6
    # rider (iii): a NOTIFY was emitted for the release.
    assert [c["strategy"] for c in spy_notify] == ["alpha_crypto"]


def test_ledger_absent_ttl_not_elapsed_stays_halted(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    recent = (_now() - timedelta(days=1)).isoformat()
    _seed_store(store, "alpha_crypto", halted_at=recent)

    edm.EdgeDecayMonitor(
        halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
    ).check_all()

    assert _entry(store, "alpha_crypto")["halted"] is True
    assert spy_notify == []


def test_missing_halted_at_is_never_ttl_cleared(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "alpha_crypto", halted_at="")  # no timestamp -> age unknown

    edm.EdgeDecayMonitor(
        halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
    ).check_all()

    assert _entry(store, "alpha_crypto")["halted"] is True
    assert spy_notify == []


# ---------------------------------------------------------------------------
# Mechanism 2 — ledger-resident recovery
# ---------------------------------------------------------------------------


def test_recovery_clears_and_persists(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "gamma", halted_at=(_now() - timedelta(days=1)).isoformat())
    glob = _write_trades(tmp_path, "gamma", [1.0] * 6)  # resident, streak 0

    edm.EdgeDecayMonitor(
        min_trades=5, consecutive_threshold=5, clear_on_recovery=True,
        halt_ttl_days=14, allocations_path=store, trades_glob=glob,
    ).check_all()

    e = _entry(store, "gamma")
    assert e["halted"] is False
    assert e["cleared_by"] == "edge_decay_monitor_auto"
    assert e["clear_reason"] == "auto_recovery_streak_below_threshold"
    assert [c["strategy"] for c in spy_notify] == ["gamma"]


def test_clear_on_recovery_false_keeps_resident_halt(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "gamma", halted_at=(_now() - timedelta(days=1)).isoformat())
    glob = _write_trades(tmp_path, "gamma", [1.0] * 6)

    edm.EdgeDecayMonitor(
        min_trades=5, consecutive_threshold=5, clear_on_recovery=False,
        halt_ttl_days=14, allocations_path=store, trades_glob=glob,
    ).check_all()

    assert _entry(store, "gamma")["halted"] is True
    assert spy_notify == []


def test_still_decaying_stays_halted(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "gamma", halted_at=(_now() - timedelta(days=30)).isoformat())
    glob = _write_trades(tmp_path, "gamma", [-1.0] * 6)  # streak 6 >= threshold

    edm.EdgeDecayMonitor(
        min_trades=5, consecutive_threshold=5, clear_on_recovery=True,
        halt_ttl_days=14, allocations_path=store, trades_glob=glob,
    ).check_all()

    # Active decay: never cleared (even though TTL elapsed) — re-halted instead.
    assert _entry(store, "gamma")["halted"] is True
    assert spy_notify == []


# ---------------------------------------------------------------------------
# rider (i) — untrusted rows can never clear a halt
# ---------------------------------------------------------------------------


def test_untrusted_wins_cannot_manufacture_a_recovery(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    # Recent halt (< TTL) so ONLY a (wrongful) recovery could clear it.
    _seed_store(store, "alpha_crypto", halted_at=(_now() - timedelta(days=1)).isoformat())
    # 100 winning fills, all tagged pnl_untrusted -> excluded from trusted ledger.
    glob = _write_trades(tmp_path, "alpha_crypto", [5.0] * 100, untrusted=True)

    edm.EdgeDecayMonitor(
        min_trades=5, consecutive_threshold=5, clear_on_recovery=True,
        halt_ttl_days=14, allocations_path=store, trades_glob=glob,
    ).check_all()

    # The untrusted "wins" never made the strategy resident, so no recovery
    # clear; TTL not elapsed, so no TTL clear -> the halt holds.
    assert _entry(store, "alpha_crypto")["halted"] is True
    assert spy_notify == []


# ---------------------------------------------------------------------------
# rider (iv) — operator halts untouched
# ---------------------------------------------------------------------------


def test_operator_imposed_halt_explicit_provenance_untouched(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    old = (_now() - timedelta(days=30)).isoformat()
    _seed_store(
        store, "beta", halted_at=old,
        halted_by="operator", halt_reason="operator_manual_freeze",
    )

    edm.EdgeDecayMonitor(
        halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
    ).check_all()

    assert _entry(store, "beta")["halted"] is True  # protected
    assert spy_notify == []


def test_legacy_non_monitor_reason_untouched(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    old = (_now() - timedelta(days=30)).isoformat()
    # Legacy record: no halted_by, and a reason that is NOT the monitor's
    # consecutive_negative_* signature -> inferred operator/other provenance.
    _seed_store(
        store, "beta", halted_at=old,
        halted_by=None, halt_reason="manual_risk_freeze",
    )

    edm.EdgeDecayMonitor(
        halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
    ).check_all()

    assert _entry(store, "beta")["halted"] is True  # protected
    assert spy_notify == []


def test_legacy_monitor_reason_without_provenance_is_cleared(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    old = (_now() - timedelta(days=30)).isoformat()
    # Legacy monitor halt (pre-W1B-3): no halted_by but the monitor's own
    # reason signature -> eligible.
    _seed_store(
        store, "alpha_crypto", halted_at=old,
        halted_by=None, halt_reason="consecutive_negative_6",
    )

    edm.EdgeDecayMonitor(
        halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
    ).check_all()

    assert _entry(store, "alpha_crypto")["halted"] is False
    assert [c["strategy"] for c in spy_notify] == ["alpha_crypto"]


# ---------------------------------------------------------------------------
# rider (iii) — the REAL marker + coach-voiced NOTIFY (telegram intercepted)
# ---------------------------------------------------------------------------


def test_real_notification_emits_marker_and_notify(tmp_path, monkeypatch, caplog):
    sent = []
    monkeypatch.setattr(
        "chad.utils.telegram_notify.notify",
        lambda message, **kw: sent.append({"message": message, **kw}) or True,
    )
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "alpha_crypto", halted_at=(_now() - timedelta(days=30)).isoformat())

    with caplog.at_level("WARNING", logger="chad.risk.edge_decay_monitor"):
        edm.EdgeDecayMonitor(
            halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
        ).check_all()

    # marker
    assert "EDGE_DECAY_AUTO_CLEARED strategy=alpha_crypto" in caplog.text
    # coach-voiced NOTIFY with a per-strategy dedupe key
    assert len(sent) == 1
    assert sent[0]["dedupe_key"] == "edge_decay_auto_clear:alpha_crypto"
    assert "alpha_crypto" in sent[0]["message"]


# ---------------------------------------------------------------------------
# idempotency — no thrash on rerun
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# W1B-5 (review): scratch / manual / warmup_sim rows must not drive a clear
# ---------------------------------------------------------------------------


def _write_raw_trades(tmp_path: Path, records: list[dict]) -> str:
    d = tmp_path / "data" / "trades"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "trade_history_test.ndjson"
    with f.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(d / "trade_history_*.ndjson")


def test_pnl_zero_tail_does_not_manufacture_recovery(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "gamma", halted_at=(_now() - timedelta(days=1)).isoformat())
    # 5 trusted losses then a scratch (pnl==0). The scratch must be EXCLUDED so
    # it cannot reset the decay streak into a wrongful recovery clear.
    rows = [{"strategy": "gamma", "pnl": -1.0} for _ in range(5)]
    rows.append({"strategy": "gamma", "pnl": 0.0})
    glob = _write_raw_trades(tmp_path, rows)

    edm.EdgeDecayMonitor(
        min_trades=5, consecutive_threshold=5, clear_on_recovery=True,
        halt_ttl_days=14, allocations_path=store, trades_glob=glob,
    ).check_all()

    assert _entry(store, "gamma")["halted"] is True  # still decaying, not cleared
    assert spy_notify == []


def test_manual_tagged_win_tail_does_not_clear(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "gamma", halted_at=(_now() - timedelta(days=1)).isoformat())
    # A "manual"-tagged win is not strategy performance -> excluded -> the
    # trusted streak stays at 5 and the halt holds.
    rows = [{"strategy": "gamma", "pnl": -1.0} for _ in range(5)]
    rows.append({"strategy": "gamma", "pnl": 5.0, "tags": ["manual"]})
    glob = _write_raw_trades(tmp_path, rows)

    edm.EdgeDecayMonitor(
        min_trades=5, consecutive_threshold=5, clear_on_recovery=True,
        halt_ttl_days=14, allocations_path=store, trades_glob=glob,
    ).check_all()

    assert _entry(store, "gamma")["halted"] is True
    assert spy_notify == []


def test_idempotent_rerun_no_second_clear(tmp_path, spy_notify):
    store = tmp_path / "strategy_allocations.json"
    _seed_store(store, "alpha_crypto", halted_at=(_now() - timedelta(days=30)).isoformat())
    mon = edm.EdgeDecayMonitor(
        halt_ttl_days=14, allocations_path=store, trades_glob=_empty_glob(tmp_path),
    )
    mon.check_all()
    mon.check_all()  # second pass: entry is already halted:False

    assert _entry(store, "alpha_crypto")["halted"] is False
    # Only ONE release notification across both passes (edge-triggered).
    assert len(spy_notify) == 1
