# chad/phase4/tests/test_phase4_smoke.py
"""Phase-4 smoke tests (typed, minimal).

These confirm the scaffolds import and perform the simplest expected behavior.
They’re intentionally tiny so CI stays fast while we wire the real Phase-4 code.
"""

from chad.phase4.router import route
from chad.phase4.ledger import record, snapshot
from chad.phase4.risk import preflight
from chad.phase4.alloc import size


def test_router_smoke() -> None:
    out = route({"symbol": "AAPL"}, dry_run=True)
    assert out["routed"] is True
    assert out["dry_run"] is True
    assert out["symbol"] == "AAPL"


def test_ledger_smoke() -> None:
    record({"id": 1, "event": "TEST"})
    snap = snapshot()
    assert any(e.get("id") == 1 for e in snap)


def test_risk_smoke() -> None:
    res = preflight({"symbol": "AAPL"})
    assert res["ok"] is True
    assert isinstance(res.get("checks"), list)


def test_alloc_smoke() -> None:
    s = size({"symbol": "AAPL"}, equity=10_000.0)
    assert isinstance(s["qty"], int)
    assert s["qty"] >= 1
