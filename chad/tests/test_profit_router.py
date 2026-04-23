"""Tests for chad.risk.profit_router (50/30/20 advisory routing)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chad.risk.profit_router import ProfitRouter, RoutingDecision, SCHEMA_VERSION


@pytest.fixture()
def router(tmp_path: Path) -> ProfitRouter:
    return ProfitRouter(routing_path=tmp_path / "profit_routing.json")


# ---------------------------------------------------------------------------
# 50/30/20 split
# ---------------------------------------------------------------------------

def test_50_30_20_split_correct(router: ProfitRouter) -> None:
    decision = router.route_profit(realized_pnl=1000.0, closing_strategy="alpha")
    assert "no_routing" not in decision
    assert math.isclose(decision["trading_capital"], 500.0, rel_tol=1e-9)
    assert math.isclose(decision["beta_allocation"], 300.0, rel_tol=1e-9)
    assert math.isclose(decision["amplifier_allocation"], 200.0, rel_tol=1e-9)
    # The three slices must equal the full realized_pnl
    total = (
        decision["trading_capital"]
        + decision["beta_allocation"]
        + decision["amplifier_allocation"]
    )
    assert math.isclose(total, 1000.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# No routing on losses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pnl", [0.0, -0.01, -250.0])
def test_no_routing_on_loss(router: ProfitRouter, pnl: float) -> None:
    out = router.route_profit(realized_pnl=pnl, closing_strategy="alpha")
    assert out == {"no_routing": True, "reason": "not_profitable"}
    # No file should be written for a non-routing decision
    assert not router.routing_path.exists()


def test_no_routing_on_nan(router: ProfitRouter) -> None:
    out = router.route_profit(realized_pnl=float("nan"), closing_strategy="alpha")
    assert out.get("no_routing") is True


def test_no_routing_on_invalid_type(router: ProfitRouter) -> None:
    out = router.route_profit(realized_pnl="not-a-number", closing_strategy="alpha")
    assert out == {"no_routing": True, "reason": "invalid_pnl"}


# ---------------------------------------------------------------------------
# Accumulation across routings
# ---------------------------------------------------------------------------

def test_accumulates_beta_allocation(router: ProfitRouter) -> None:
    router.route_profit(100.0, "alpha")
    router.route_profit(250.0, "gamma")
    router.route_profit(50.0, "omega")
    # Loss: ignored
    router.route_profit(-1000.0, "alpha")

    # 30% of 400 = 120
    assert math.isclose(router.get_beta_accumulated(), 120.0, rel_tol=1e-9)
    assert math.isclose(router.get_amplifier_accumulated(), 80.0, rel_tol=1e-9)

    totals = router.get_totals()
    assert math.isclose(totals["trading_capital"], 200.0, rel_tol=1e-9)
    assert math.isclose(totals["beta_allocation"], 120.0, rel_tol=1e-9)
    assert math.isclose(totals["amplifier_allocation"], 80.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_writes_schema_versioned_file(router: ProfitRouter) -> None:
    router.route_profit(100.0, "alpha")
    assert router.routing_path.is_file()
    payload = json.loads(router.routing_path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert isinstance(payload["decisions"], list)
    assert len(payload["decisions"]) == 1
    entry = payload["decisions"][0]
    assert entry["source_strategy"] == "alpha"
    assert entry["realized_pnl"] == 100.0
    assert "routing_timestamp" in entry


def test_reload_router_preserves_accumulation(tmp_path: Path) -> None:
    path = tmp_path / "profit_routing.json"
    r1 = ProfitRouter(routing_path=path)
    r1.route_profit(1000.0, "alpha")

    # Simulate process restart
    r2 = ProfitRouter(routing_path=path)
    assert math.isclose(r2.get_beta_accumulated(), 300.0, rel_tol=1e-9)


def test_corrupt_file_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "profit_routing.json"
    path.write_text("{not valid json")
    r = ProfitRouter(routing_path=path)
    # Corrupt file shouldn't crash; treated as empty
    assert r.get_beta_accumulated() == 0.0
    # And a subsequent route still works and overwrites cleanly
    decision = r.route_profit(100.0, "alpha")
    assert "no_routing" not in decision
    payload = json.loads(path.read_text())
    assert payload["decisions"][0]["realized_pnl"] == 100.0


def test_records_account_equity_when_provided(router: ProfitRouter) -> None:
    router.route_profit(100.0, "alpha", account_equity=1_000_000.0)
    payload = json.loads(router.routing_path.read_text())
    assert payload["decisions"][0]["account_equity_at_decision"] == 1_000_000.0


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------

def test_routing_decision_to_dict() -> None:
    d = RoutingDecision(
        realized_pnl=100.0,
        source_strategy="alpha",
        trading_capital=50.0,
        beta_allocation=30.0,
        amplifier_allocation=20.0,
        routing_timestamp="2026-04-23T00:00:00Z",
    )
    assert d.to_dict() == {
        "realized_pnl": 100.0,
        "source_strategy": "alpha",
        "trading_capital": 50.0,
        "beta_allocation": 30.0,
        "amplifier_allocation": 20.0,
        "routing_timestamp": "2026-04-23T00:00:00Z",
    }
