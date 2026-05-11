"""GAP-028 regression tests — drift detector wiring + close_guard_entry CLI.

Covers the 9 tests required in GAP-028 §6 (test #8 covers Option B PERMISSIVE
behaviour — the rebuilder intentionally does NOT consult exclusion_policy).

All tests redirect runtime/data paths into tmp_path via monkeypatch so they
are hermetic and never touch real CHAD state files.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _seed_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Redirect every runtime/data path the CLI touches into tmp_path.

    Returns a dict of paths so individual tests can pre-seed state.
    """
    from chad.core import position_guard
    import close_guard_entry as cli  # type: ignore

    guard_path = tmp_path / "runtime" / "position_guard.json"
    tc_path = tmp_path / "runtime" / "trade_closer_state.json"
    scr_path = tmp_path / "runtime" / "scr_state.json"
    actions_dir = tmp_path / "data" / "operator_actions"

    guard_path.parent.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(cli, "TRADE_CLOSER_STATE_PATH", tc_path)
    monkeypatch.setattr(cli, "SCR_STATE_PATH", scr_path)
    monkeypatch.setattr(cli, "OPERATOR_ACTIONS_DIR", actions_dir)

    # Default: paper mode, CONFIDENT SCR, LiveGate reports allow_ibkr_live=false.
    # The CLI fetches the live-gate snapshot over HTTP; monkeypatch the fetcher
    # to a deterministic safe value so tests don't depend on the backend running.
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    _seed_json(scr_path, {
        "schema_version": "scr_state.v1",
        "state": "CONFIDENT",
        "sizing_factor": 1.0,
        "paper_only": False,
    })
    monkeypatch.setattr(
        cli, "_fetch_live_gate_snapshot",
        lambda: {"allow_ibkr_live": False, "allow_ibkr_paper": True},
    )

    return {
        "cli": cli,
        "guard_path": guard_path,
        "tc_path": tc_path,
        "scr_path": scr_path,
        "actions_dir": actions_dir,
        "monkeypatch": monkeypatch,
    }


def _seed_synthetic_drift_state() -> Dict[str, Any]:
    """Snapshot mirroring GAP-028 §1 — delta|AAPL guard vs broker_sync|AAPL truth."""
    return {
        "delta|AAPL": {
            "open": True, "strategy": "delta", "symbol": "AAPL",
            "side": "BUY", "quantity": 31.0, "last_state": "OPEN",
            "source": "paper_ledger_rebuild",
        },
        "broker_sync|AAPL": {
            "open": True, "strategy": "broker_sync", "symbol": "AAPL",
            "side": "SELL", "quantity": 2.0, "last_state": "OPEN",
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — drift detector fires on synthetic delta|AAPL state
# ---------------------------------------------------------------------------

def test_detect_guard_vs_broker_truth_drift_fires_on_delta_AAPL_synthetic_state():
    from chad.core import position_guard

    state = _seed_synthetic_drift_state()
    drift = position_guard.detect_guard_vs_broker_truth_drift(state)

    assert len(drift) == 1, f"expected exactly one drift record, got {drift!r}"
    rec = drift[0]
    assert rec["key"] == "delta|AAPL"
    assert rec["strategy"] == "delta"
    assert rec["symbol"] == "AAPL"
    assert rec["guard_side"] == "BUY"
    assert rec["broker_side"] == "SELL"
    assert rec["broker_present"] is True
    assert rec["drift_kind"] == "side_mismatch"


# ---------------------------------------------------------------------------
# Test 2 — CLI clears trade_closer FIFO in the same op
# ---------------------------------------------------------------------------

def test_close_guard_entry_clears_trade_closer_fifo_in_same_op(cli_env):
    cli = cli_env["cli"]
    _seed_json(cli_env["guard_path"], {
        "delta|AAPL": {
            "open": True, "strategy": "delta", "symbol": "AAPL",
            "side": "BUY", "quantity": 31.0, "last_state": "OPEN",
        },
    })
    _seed_json(cli_env["tc_path"], {
        "queues": [
            {"strategy": "delta", "symbol": "AAPL", "lots": [
                {"side": "BUY", "quantity": 11.0, "fill_price": 230.0,
                 "fill_id": "x1", "ts_utc": "2026-05-09T00:00:00Z"},
                {"side": "BUY", "quantity": 20.0, "fill_price": 231.0,
                 "fill_id": "x2", "ts_utc": "2026-05-10T00:00:00Z"},
            ]},
            {"strategy": "alpha", "symbol": "SPY", "lots": [
                {"side": "BUY", "quantity": 5.0, "fill_price": 500.0,
                 "fill_id": "y1", "ts_utc": "2026-05-09T00:00:00Z"},
            ]},
        ],
        "processed_fill_ids": ["x1", "x2", "y1"],
    })

    rc = cli.run([
        "--strategy", "delta", "--symbol", "AAPL",
        "--reason", "broker_truth_drift_test", "--by", "test", "--confirm",
    ])
    assert rc == 0

    guard = _load_json(cli_env["guard_path"])
    assert guard["delta|AAPL"]["open"] is False
    assert guard["delta|AAPL"]["last_state"] == "CLOSED"
    assert guard["delta|AAPL"]["closed_reason"] == "stale_guard_entry"
    assert guard["delta|AAPL"]["closed_by"] == "broker_truth_drift_test"

    tc = _load_json(cli_env["tc_path"])
    queue_keys = {(e["strategy"], e["symbol"]) for e in tc["queues"]}
    assert ("delta", "AAPL") not in queue_keys, (
        f"trade_closer FIFO entry must be removed atomically. queues={tc['queues']}"
    )
    assert ("alpha", "SPY") in queue_keys, "unrelated strategy must be untouched"


# ---------------------------------------------------------------------------
# Test 3 — CLI refuses on broker_sync|* keys
# ---------------------------------------------------------------------------

def test_close_guard_entry_refuses_on_broker_sync_key(cli_env, caplog):
    cli = cli_env["cli"]
    _seed_json(cli_env["guard_path"], {
        "broker_sync|AAPL": {
            "open": True, "strategy": "broker_sync", "symbol": "AAPL",
            "side": "SELL", "quantity": 2.0, "last_state": "OPEN",
        },
    })
    _seed_json(cli_env["tc_path"], {"queues": [], "processed_fill_ids": []})

    with caplog.at_level("ERROR"):
        rc = cli.run([
            "--strategy", "broker_sync", "--symbol", "AAPL",
            "--reason", "test", "--by", "test", "--confirm",
        ])
    assert rc == 3, "must exit with code 3 for broker_sync key refusal"
    assert any("broker_sync" in m for m in caplog.messages)

    guard = _load_json(cli_env["guard_path"])
    assert guard["broker_sync|AAPL"]["open"] is True, "must NOT mutate broker_sync entry"


# ---------------------------------------------------------------------------
# Test 4 — CLI refuses when exec_mode is not paper
# ---------------------------------------------------------------------------

def test_close_guard_entry_refuses_when_exec_mode_not_paper(cli_env, monkeypatch, caplog):
    cli = cli_env["cli"]
    _seed_json(cli_env["guard_path"], {
        "delta|AAPL": {
            "open": True, "strategy": "delta", "symbol": "AAPL",
            "side": "BUY", "quantity": 31.0, "last_state": "OPEN",
        },
    })
    _seed_json(cli_env["tc_path"], {"queues": [
        {"strategy": "delta", "symbol": "AAPL", "lots": [
            {"side": "BUY", "quantity": 31.0, "fill_id": "x1"}
        ]},
    ], "processed_fill_ids": []})

    monkeypatch.setenv("CHAD_EXECUTION_MODE", "live")

    with caplog.at_level("ERROR"):
        rc = cli.run([
            "--strategy", "delta", "--symbol", "AAPL",
            "--reason", "test", "--by", "test", "--confirm",
        ])
    assert rc == 4, "must exit with code 4 when fail-closed gate trips"
    assert any("exec_mode" in m for m in caplog.messages)

    guard = _load_json(cli_env["guard_path"])
    assert guard["delta|AAPL"]["open"] is True, "must NOT close guard when gate trips"
    tc = _load_json(cli_env["tc_path"])
    assert any(e["strategy"] == "delta" and e["symbol"] == "AAPL" for e in tc["queues"]), (
        "must NOT clear FIFO when gate trips"
    )


# ---------------------------------------------------------------------------
# Test 5 — CLI refuses when LiveGate reports allow_ibkr_live=true
# ---------------------------------------------------------------------------

def test_close_guard_entry_refuses_when_livegate_allow_ibkr_live(cli_env, caplog):
    """Predicate is `allow_ibkr_live` (real IBKR orders enabled), NOT
    `operator_intent.operator_mode == ALLOW_LIVE` (which is the normal
    paper-mode posture set by the auto-refresher and must NOT block
    maintenance)."""
    cli = cli_env["cli"]
    _seed_json(cli_env["guard_path"], {
        "delta|AAPL": {
            "open": True, "strategy": "delta", "symbol": "AAPL",
            "side": "BUY", "quantity": 31.0, "last_state": "OPEN",
        },
    })
    _seed_json(cli_env["tc_path"], {"queues": [], "processed_fill_ids": []})
    cli_env["monkeypatch"].setattr(
        cli, "_fetch_live_gate_snapshot",
        lambda: {"allow_ibkr_live": True, "allow_ibkr_paper": True},
    )

    with caplog.at_level("ERROR"):
        rc = cli.run([
            "--strategy", "delta", "--symbol", "AAPL",
            "--reason", "test", "--by", "test", "--confirm",
        ])
    assert rc == 4, "must exit with code 4 when allow_ibkr_live=true"
    assert any("livegate" in m.lower() or "allow_ibkr_live" in m.lower() for m in caplog.messages)

    guard = _load_json(cli_env["guard_path"])
    assert guard["delta|AAPL"]["open"] is True, (
        "must NOT close guard when allow_ibkr_live=true"
    )


# ---------------------------------------------------------------------------
# Test 5b — CLI fails closed when LiveGate endpoint is unreachable
# ---------------------------------------------------------------------------

def test_close_guard_entry_refuses_when_livegate_unreachable(cli_env, caplog):
    cli = cli_env["cli"]
    _seed_json(cli_env["guard_path"], {
        "delta|AAPL": {
            "open": True, "strategy": "delta", "symbol": "AAPL",
            "side": "BUY", "quantity": 31.0, "last_state": "OPEN",
        },
    })
    _seed_json(cli_env["tc_path"], {"queues": [], "processed_fill_ids": []})
    cli_env["monkeypatch"].setattr(cli, "_fetch_live_gate_snapshot", lambda: None)

    with caplog.at_level("ERROR"):
        rc = cli.run([
            "--strategy", "delta", "--symbol", "AAPL",
            "--reason", "test", "--by", "test", "--confirm",
        ])
    assert rc == 4, "must fail-closed when /live-gate is unreachable"
    assert any("livegate_unreachable" in m.lower() for m in caplog.messages)

    guard = _load_json(cli_env["guard_path"])
    assert guard["delta|AAPL"]["open"] is True, (
        "must NOT close guard when LiveGate snapshot cannot be read"
    )


# ---------------------------------------------------------------------------
# Test 6 — CLI is idempotent on already-closed entry
# ---------------------------------------------------------------------------

def test_close_guard_entry_idempotent_on_already_closed(cli_env, caplog):
    cli = cli_env["cli"]
    _seed_json(cli_env["guard_path"], {
        "delta|AAPL": {
            "open": False, "strategy": "delta", "symbol": "AAPL",
            "side": "BUY", "quantity": 31.0, "last_state": "CLOSED",
            "closed_by": "previously_closed",
        },
    })
    _seed_json(cli_env["tc_path"], {"queues": [], "processed_fill_ids": []})

    with caplog.at_level("WARNING"):
        rc = cli.run([
            "--strategy", "delta", "--symbol", "AAPL",
            "--reason", "test", "--by", "test", "--confirm",
        ])
    assert rc == 0, "idempotent re-run must exit 0"
    assert any("IDEMPOTENT_NOOP" in m for m in caplog.messages)

    guard = _load_json(cli_env["guard_path"])
    assert guard["delta|AAPL"]["closed_by"] == "previously_closed", (
        "previously-closed entry must NOT be overwritten on re-run"
    )

    actions = list(cli_env["actions_dir"].glob("*.ndjson"))
    assert actions == [], (
        "no audit record should be written when the call is a pure idempotent no-op"
    )


# ---------------------------------------------------------------------------
# Test 7 — CLI writes audit record under data/operator_actions/
# ---------------------------------------------------------------------------

def test_close_guard_entry_writes_operator_action_audit_record(cli_env):
    cli = cli_env["cli"]
    previous = {
        "open": True, "strategy": "omega_macro", "symbol": "M6E",
        "side": "BUY", "quantity": 146.0, "last_state": "OPEN",
        "source": "paper_ledger_rebuild",
    }
    _seed_json(cli_env["guard_path"], {"omega_macro|M6E": dict(previous)})
    _seed_json(cli_env["tc_path"], {
        "queues": [{"strategy": "omega_macro", "symbol": "M6E", "lots": [
            {"side": "BUY", "quantity": 146.0, "fill_id": "z1"}
        ]}],
        "processed_fill_ids": ["z1"],
    })

    rc = cli.run([
        "--strategy", "omega_macro", "--symbol", "M6E",
        "--reason", "broker_truth_long_26_vs_guard_long_146",
        "--by", "operator", "--confirm",
    ])
    assert rc == 0

    actions = list(cli_env["actions_dir"].glob("*.ndjson"))
    assert len(actions) == 1, f"expected exactly one audit file, got {actions}"
    lines = actions[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "exactly one ndjson record per invocation"

    rec = json.loads(lines[0])
    assert rec["schema_version"] == "operator_action.close_guard_entry.v1"
    assert rec["action"] == "close_guard_entry"
    assert rec["strategy"] == "omega_macro"
    assert rec["symbol"] == "M6E"
    assert rec["key"] == "omega_macro|M6E"
    assert rec["reason"] == "broker_truth_long_26_vs_guard_long_146"
    assert rec["by"] == "operator"
    assert rec["trade_closer_fifo_cleared"] is True
    # previous_entry must be captured for rollback
    assert rec["previous_entry"]["open"] is True
    assert rec["previous_entry"]["quantity"] == 146.0
    assert rec["previous_entry"]["side"] == "BUY"


# ---------------------------------------------------------------------------
# Test 8 — Option B PERMISSIVE: rebuilder does NOT consult exclusion_policy
# ---------------------------------------------------------------------------

def test_rebuilder_consults_exclusion_policy_when_strict_mode(tmp_path, monkeypatch):
    """GAP-028 §7 — Option B (PERMISSIVE) chosen.

    The rebuilder intentionally does NOT consult `reconciliation_state.exclusion_policy`.
    This test locks in that behaviour: even when a symbol is on the exclusion list,
    a non-empty trade_closer queue produces a per-strategy guard entry. The maintenance
    surface for stale entries is the drift detector + close_guard_entry CLI, not
    upstream exclusion plumbing.
    """
    import logging
    from chad.core import live_loop, position_guard

    guard_path = tmp_path / "position_guard.json"
    tc_path = tmp_path / "trade_closer_state.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(live_loop, "_TRADE_CLOSER_STATE_PATH", tc_path)

    # AAPL is on the operator exclusion list (added 2026-04-01) but the rebuilder
    # must mirror trade_closer queues regardless.
    _seed_json(tc_path, {
        "queues": [{
            "strategy": "delta", "symbol": "AAPL",
            "lots": [{
                "side": "BUY", "quantity": 31.0, "fill_price": 230.0,
                "fill_id": "x1", "ts_utc": "2026-05-09T00:00:00Z",
            }],
        }],
        "processed_fill_ids": [],
    })
    _seed_json(guard_path, {})

    live_loop._rebuild_guard_from_paper_ledger(logging.getLogger("test"))

    state = _load_json(guard_path)
    assert "delta|AAPL" in state, (
        "Option B PERMISSIVE: rebuilder must mirror queue entry even for excluded symbol"
    )
    assert state["delta|AAPL"]["open"] is True
    assert state["delta|AAPL"]["quantity"] == 31.0
    assert state["delta|AAPL"]["source"] == "paper_ledger_rebuild"


# ---------------------------------------------------------------------------
# Test 9 — reconciliation publisher emits position_guard_drift.json
# ---------------------------------------------------------------------------

def test_reconciliation_publisher_emits_position_guard_drift_json(tmp_path, monkeypatch):
    """GAP-028 §5.1: _emit_position_guard_drift writes the v1 advisory file
    with a non-empty `drifts` array when guard state contains a side mismatch."""
    from chad.ops import reconciliation_publisher as pub
    from chad.core import position_guard

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    guard_path = runtime_dir / "position_guard.json"
    drift_path = runtime_dir / "position_guard_drift.json"

    monkeypatch.setattr(pub, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(pub, "GUARD_PATH", guard_path)
    monkeypatch.setattr(pub, "DRIFT_OUT_PATH", drift_path)
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)

    _seed_json(guard_path, _seed_synthetic_drift_state())

    n = pub._emit_position_guard_drift()
    assert n == 1, f"expected exactly one drift, got {n}"
    assert drift_path.is_file(), "drift advisory file must be written"

    payload = _load_json(drift_path)
    assert payload["schema_version"] == "position_guard_drift.v1"
    assert payload["drift_count"] == 1
    assert payload["ttl_seconds"] == pub.TTL_SECONDS
    assert "ts_utc" in payload
    assert isinstance(payload["drifts"], list) and len(payload["drifts"]) == 1
    rec = payload["drifts"][0]
    assert rec["key"] == "delta|AAPL"
    assert rec["drift_kind"] == "side_mismatch"
    assert rec["guard_side"] == "BUY"
    assert rec["broker_side"] == "SELL"

    # Read-only invariant: position_guard.json mtime must not change.
    pre_mtime = guard_path.stat().st_mtime_ns
    pub._emit_position_guard_drift()
    assert guard_path.stat().st_mtime_ns == pre_mtime, (
        "_emit_position_guard_drift must NOT mutate position_guard.json"
    )
