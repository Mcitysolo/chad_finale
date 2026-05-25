"""ISSUE-29 / ISSUE-75 regression tests.

Covers:
- mark_position_closed must require confirmed fill evidence (fill_id +
  status filled/paper_fill, not PendingSubmit, not pnl_untrusted) before
  flipping a position_guard entry to closed.
- write_position_guard is the single atomic writer; every guard write
  site routes through it.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from chad.core import position_guard


@dataclass
class _Intent:
    strategy: str
    symbol: str
    side: str
    quantity: float


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    state_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", state_path)
    return state_path


def _seed_open(path: Path, key: str = "alpha|SPY") -> None:
    payload = {
        key: {
            "open": True,
            "strategy": key.split("|")[0],
            "symbol": key.split("|")[1],
            "side": "BUY",
            "quantity": 30.0,
            "last_state": "OPEN",
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ISSUE-29
# ---------------------------------------------------------------------------

def test_position_guard_not_written_before_fill_confirmed(tmp_state):
    """mark_position_closed without fill evidence must NOT mutate the guard."""
    _seed_open(tmp_state)
    intent = _Intent("alpha", "SPY", "SELL", 30.0)

    # No fill evidence at all → must refuse and return False.
    result = position_guard.mark_position_closed(intent, fill_evidence=None)

    assert result is False
    state = _load(tmp_state)
    assert state["alpha|SPY"]["open"] is True, (
        "guard must remain open when no fill confirmation is supplied"
    )
    assert state["alpha|SPY"].get("last_state") == "OPEN"


def test_position_guard_written_after_fill_confirmed(tmp_state):
    """mark_position_closed with confirmed fill evidence must mutate the guard."""
    _seed_open(tmp_state)
    intent = _Intent("alpha", "SPY", "SELL", 30.0)

    fill_evidence = {
        "fill_id": "deadbeef-1",
        "status": "paper_fill",
        "pnl_untrusted": False,
        "reject": False,
    }
    result = position_guard.mark_position_closed(intent, fill_evidence=fill_evidence)

    assert result is True
    state = _load(tmp_state)
    assert state["alpha|SPY"]["open"] is False
    assert state["alpha|SPY"]["last_state"] == "CLOSED"
    assert state["alpha|SPY"]["closed_by"] == "fill_confirmed"
    assert state["alpha|SPY"]["closed_fill_id"] == "deadbeef-1"


def test_position_guard_pending_submit_does_not_close_guard(tmp_state):
    """PendingSubmit (and other pending statuses) must NOT close the guard."""
    _seed_open(tmp_state)
    intent = _Intent("alpha", "SPY", "SELL", 30.0)

    for pending_status in (
        "PendingSubmit", "PreSubmitted", "Submitted",
        "ApiPending", "Inactive", "Unknown", "",
    ):
        fill_evidence = {
            "fill_id": "deadbeef-pending",
            "status": pending_status,
        }
        result = position_guard.mark_position_closed(
            intent, fill_evidence=fill_evidence,
        )
        assert result is False, (
            f"pending status {pending_status!r} must not confirm fill"
        )

    state = _load(tmp_state)
    assert state["alpha|SPY"]["open"] is True


def test_position_guard_untrusted_fill_does_not_close_guard(tmp_state):
    """pnl_untrusted (or rejected/tag-marked) fills must NOT close the guard."""
    _seed_open(tmp_state)
    intent = _Intent("alpha", "SPY", "SELL", 30.0)

    untrusted_variants = [
        {"fill_id": "x1", "status": "paper_fill", "pnl_untrusted": True},
        {"fill_id": "x2", "status": "paper_fill", "reject": True},
        {"fill_id": "x3", "status": "rejected"},
        {"fill_id": "x4", "status": "paper_fill",
         "tags": ["paper", "pnl_untrusted"]},
        {"fill_id": "x5", "status": "paper_fill",
         "extra": {"pnl_untrusted": True}},
        # Missing fill_id → cannot confirm.
        {"fill_id": "", "status": "paper_fill"},
        {"status": "paper_fill"},
    ]
    for ev in untrusted_variants:
        result = position_guard.mark_position_closed(intent, fill_evidence=ev)
        assert result is False, f"untrusted evidence accepted: {ev!r}"

    state = _load(tmp_state)
    assert state["alpha|SPY"]["open"] is True


# ---------------------------------------------------------------------------
# ISSUE-75
# ---------------------------------------------------------------------------

def test_all_position_guard_write_sites_use_atomic_writer(tmp_state, monkeypatch):
    """Every external position_guard write must route through write_position_guard.

    Static-style assertion: the modules that write guard state import
    `save_state` (the backward-compatible alias) or `write_position_guard`
    from chad.core.position_guard. Any other writer would have to grep
    for `position_guard.json` and write_text/json.dump it directly —
    none should exist.

    Runtime assertion: spy on write_position_guard and confirm
    save_state delegates to it for all the public mutator entrypoints.
    """
    # 1. Static check — no module under chad/ writes position_guard.json
    #    without going through the position_guard module.
    repo_root = Path(__file__).resolve().parents[2]
    forbidden_patterns = (
        'position_guard.json".write_text',
        "position_guard.json'.write_text",
    )
    offenders: list[str] = []
    for py_file in repo_root.glob("chad/**/*.py"):
        if py_file.name.startswith("test_") or "/__pycache__/" in str(py_file):
            continue
        if py_file.resolve() == Path(position_guard.__file__).resolve():
            continue  # the canonical writer module is allowed
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for pat in forbidden_patterns:
            if pat in text:
                offenders.append(f"{py_file}: {pat}")
        # Also catch the shape `open(<...>position_guard.json<...>, "w")`.
        if 'position_guard.json' in text and (
            'open(' in text and ', "w"' in text or ", 'w'" in text
        ):
            # Crude check — but flag only if 'position_guard.json' appears
            # near an open(..., "w") pattern in the same file.
            for line in text.splitlines():
                low = line.strip()
                if (
                    "position_guard.json" in low
                    and "open(" in low
                    and ('"w"' in low or "'w'" in low)
                ):
                    offenders.append(f"{py_file}: {low}")
    assert not offenders, (
        "Direct write sites bypass the atomic writer:\n"
        + "\n".join(offenders)
    )

    # 2. Runtime check — every public mutator delegates to write_position_guard.
    calls: list[dict] = []
    real_writer = position_guard.write_position_guard

    def _spy_writer(state, path=None):
        calls.append(dict(state))
        return real_writer(state, path)

    monkeypatch.setattr(position_guard, "write_position_guard", _spy_writer)
    # save_state must continue to work; rebind it to the alias that
    # references the patched writer.
    monkeypatch.setattr(
        position_guard, "save_state",
        lambda s: position_guard.write_position_guard(s),
    )

    intent = _Intent("alpha", "SPY", "BUY", 30.0)
    position_guard.mark_position_open(intent)
    position_guard.replace_position(_Intent("alpha", "SPY", "SELL", 30.0))
    position_guard.mark_position_closed(
        _Intent("alpha", "SPY", "SELL", 30.0),
        fill_evidence={
            "fill_id": "f-confirmed",
            "status": "paper_fill",
        },
    )
    position_guard.reset_from_broker("alpha", "SPY")
    position_guard.reset_all_positions()

    assert len(calls) >= 5, (
        f"expected each mutator to route through write_position_guard, "
        f"observed {len(calls)} calls"
    )

    # 3. The writer must produce a recoverable on-disk state — no partial
    #    writes. Reload the file and assert it is valid JSON.
    final_state = _load(tmp_state)
    assert isinstance(final_state, dict)
    assert "_version" in final_state
    assert final_state.get("_written_by") == "position_guard"


def test_write_position_guard_rejects_invalid_schema(tmp_state):
    """write_position_guard must raise on schema violation and not write."""
    # Non-dict input.
    with pytest.raises(ValueError):
        position_guard.write_position_guard([1, 2, 3])  # type: ignore[arg-type]

    # Open entry missing required fields.
    with pytest.raises(ValueError):
        position_guard.write_position_guard({
            "alpha|SPY": {"open": True},  # missing strategy/symbol/side
        })

    # File must remain absent (or unchanged) on rejection.
    if tmp_state.exists():
        existing = _load(tmp_state)
        # If it does exist (from another test), it must not contain the
        # rejected payload.
        assert "alpha|SPY" not in existing or existing["alpha|SPY"].get("open") is not True


# ---------------------------------------------------------------------------
# is_fill_confirmed pure-function coverage
# ---------------------------------------------------------------------------

def test_is_fill_confirmed_truth_table():
    confirmed = position_guard.is_fill_confirmed
    assert confirmed({"fill_id": "x", "status": "paper_fill"}) is True
    assert confirmed({"fill_id": "x", "status": "filled"}) is True
    # Case-insensitive on status.
    assert confirmed({"fill_id": "x", "status": "Paper_Fill"}) is True

    # Negative cases.
    assert confirmed(None) is False
    assert confirmed({}) is False
    assert confirmed({"status": "paper_fill"}) is False  # no fill_id
    assert confirmed({"fill_id": "x"}) is False  # no status
    assert confirmed({"fill_id": "x", "status": "PendingSubmit"}) is False
    assert confirmed({"fill_id": "x", "status": "rejected"}) is False
    assert confirmed({"fill_id": "x", "status": "error"}) is False
    assert confirmed({"fill_id": "x", "status": "paper_fill",
                      "pnl_untrusted": True}) is False
    assert confirmed({"fill_id": "x", "status": "paper_fill",
                      "tags": ["pnl_untrusted"]}) is False
    assert confirmed({"fill_id": "x", "status": "paper_fill",
                      "extra": {"pnl_untrusted": True}}) is False


# ---------------------------------------------------------------------------
# reconciler-side ISSUE-29 fix preserved
# ---------------------------------------------------------------------------

def test_reconciler_preserves_existing_reject_skip(monkeypatch, tmp_state):
    """The original reconciler-side reject skip (the existing fix) must
    remain intact alongside the new positive-confirmation gate."""
    from chad.core import position_reconciler

    # Seed an open position the reconciler will try to close.
    _seed_open(tmp_state, key="reconciler|SPY")

    class _FakeOrder:
        def __init__(self, status: str) -> None:
            self.symbol = "SPY"
            self.side = "SELL"
            self.quantity = 30.0
            self.status = status
            self.submitted_at = None
            self.asset_class = "EQUITY"

    class _RejectAdapter:
        def submit_strategy_trade_intents(self, intents):
            return [_FakeOrder("error")]

    close_intents = [{
        "symbol": "SPY",
        "action": "CLOSE",
        "open_side": "BUY",
        "close_side": "SELL",
        "quantity": 30.0,
        "reason": "reconciler_flip_test",
        "position_key": "reconciler|SPY",
        "strategy": "reconciler",
    }]

    # Make the PR-02b close-fill price resolver return a positive value
    # so the evidence path can at least try to normalize. The legacy
    # _load_price helper is no longer the production resolver after
    # PR-02b — the cascade _resolve_close_fill_price is.
    monkeypatch.setattr(
        position_reconciler,
        "_resolve_close_fill_price",
        lambda sym: 100.0,
    )

    position_reconciler.apply_close_intents(close_intents, _RejectAdapter())

    state = _load(tmp_state)
    assert state["reconciler|SPY"]["open"] is True, (
        "broker-rejected close must NOT mutate the guard (existing fix)"
    )
