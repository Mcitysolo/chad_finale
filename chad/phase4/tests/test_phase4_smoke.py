def test_router_smoke():
    from chad.phase4.router import route
    out = route({"symbol":"AAPL"}, dry_run=True)
    assert out["routed"] is True and out["dry_run"] is True and out["symbol"] == "AAPL"

def test_ledger_smoke():
    from chad.phase4.ledger import record, snapshot
    record({"id": 1, "event": "TEST"})
    snap = snapshot()
    assert any(e.get("id") == 1 for e in snap)

def test_risk_smoke():
    from chad.phase4.risk import preflight
    res = preflight({"symbol":"AAPL"})
    assert res["ok"] is True

def test_alloc_smoke():
    from chad.phase4.alloc import size
    s = size({"symbol":"AAPL"}, equity=10000.0)
    assert s["qty"] >= 1
