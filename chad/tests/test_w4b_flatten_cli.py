"""W4B-5 (LC6): flatten-all core — gates, tokens, probe, scope, clamps, phases.

Pins (PLAN_W4B §2 + §5, D1-D8 GO + riders):
  GATES (D4 minimal):
    - mode gate fail-closed: only paper|dry_run pass; missing/garbage/live
      REFUSE; no invented mode variants;
    - live_readiness cross-check: ready_for_live=true REFUSES; unreadable file
      warns but never blocks (env gate is the hard wall);
    - token walls hard-error (FlattenAbort, exit-2 semantics at the CLI):
      wrong --confirm, broker-all without INCLUDE-EXCLUDED — never the
      epoch_reset silent degrade.
  PROBE (Phase 0, fail-closed):
    - dead/unanswerable connection => FlattenAbort (unknown is never flat);
    - missing cross-client enumeration => FlattenAbort;
    - positions + cross-client orders parsed; clientId-scoped count captured
      (the 0/0 false-negative visibility-split proof);
    - snapshot crosscheck lists divergences; live probe wins.
  SCOPE (D1 + rider):
    - chad default: legs clamped to broker net via per-symbol budget; excluded
      symbols untouchable AND NAMED; non-chad remainder NAMED; a ledger-only
      leg on a broker-flat symbol NAMED (never an order);
    - broker-all: operator/unattributed remainder becomes operator_flatten
      legs (origin operator|unattributed);
    - clamp math: guard>broker, broker>guard, flat, sign flip.
  CANCEL (D3 + rider):
    - drill enumerates + classifies, cancels NOTHING;
    - execute: global cancel + filtered verify loop; non-CHAD collateral by
      name; survivors reported, terminal statuses never counted as survivors.
  CONFIRM/REPORT:
    - SLA math (ack/fill ms + percentiles); duplicate_blocked benign;
    - residual verdicts FLAT|RESIDUAL, untouched restated by name, overall
      INCOMPLETE on any residual;
    - drill -> reports/ratification/PROOF_FLATTEN_DRILL_<date>.json
      (flatten_drill_proof.v1); execute -> reports/flatten_all_<stamp>.json
      (flatten_all_report.v1).
  REGISTRY: FLATTEN_ALL registered, distinct from the futures oneshot id.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "flatten_all_mod",
    Path(__file__).resolve().parents[2] / "scripts" / "flatten_all.py")
fa = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fa  # dataclass field resolution needs the module registered
_SPEC.loader.exec_module(fa)

NOW = datetime(2026, 7, 23, 16, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

def _pos(symbol, qty, sec_type="STK", avg_cost=100.0):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol, secType=sec_type,
                                 localSymbol=symbol,
                                 lastTradeDateOrContractMonth=""),
        position=qty, avgCost=avg_cost)


def _trade(order_id, client_id, symbol, action="BUY", qty=1.0, status="Submitted"):
    return SimpleNamespace(
        order=SimpleNamespace(orderId=order_id, permId=order_id * 10,
                              clientId=client_id, action=action,
                              totalQuantity=qty, orderType="LMT"),
        contract=SimpleNamespace(symbol=symbol),
        orderStatus=SimpleNamespace(status=status))


class FakeIB:
    """Canned broker truth: positions + cross-client open orders. Global cancel
    flips every order to Cancelled after `cancel_lag` re-probes (0 = at once)."""

    def __init__(self, positions=(), trades=(), own_orders=0, connected=True,
                 cancel_lag=0, stubborn_ids=()):
        self._positions = list(positions)
        self._trades = list(trades)
        self._own = own_orders
        self._connected = connected
        self._cancel_requested = False
        self._cancel_lag = cancel_lag
        self._stubborn = set(stubborn_ids)
        self.global_cancel_calls = 0
        self.client = SimpleNamespace(isConnected=lambda: connected)

    def isConnected(self):
        return self._connected

    def positions(self):
        return list(self._positions)

    def reqAllOpenOrders(self):
        if self._cancel_requested:
            if self._cancel_lag > 0:
                self._cancel_lag -= 1
            else:
                for t in self._trades:
                    if t.order.orderId not in self._stubborn:
                        t.orderStatus.status = "Cancelled"
        return list(self._trades)

    def openOrders(self):
        return [None] * self._own

    def reqGlobalCancel(self):
        self.global_cancel_calls += 1
        self._cancel_requested = True


# --------------------------------------------------------------------------- #
# gates + tokens (D4)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode,ok", [
    ("paper", True), ("dry_run", True), ("live", False), ("", False),
    ("ibkr_paper", False), ("PAPER", True), ("garbage", False),
])
def test_mode_gate_fail_closed(mode, ok):
    gates = fa.check_gates({"CHAD_EXECUTION_MODE": mode})
    assert gates["mode_gate_ok"] is ok
    assert gates["ok"] is ok


def test_mode_gate_missing_env_refuses():
    assert fa.check_gates({})["ok"] is False


def test_readiness_true_refuses_even_in_paper(tmp_path):
    lr = tmp_path / "live_readiness.json"
    lr.write_text(json.dumps({"ready_for_live": True}))
    gates = fa.check_gates({"CHAD_EXECUTION_MODE": "paper"}, lr)
    assert gates["ready_for_live"] is True
    assert gates["ok"] is False


def test_readiness_false_passes(tmp_path):
    lr = tmp_path / "live_readiness.json"
    lr.write_text(json.dumps({"ready_for_live": False}))
    gates = fa.check_gates({"CHAD_EXECUTION_MODE": "paper"}, lr)
    assert gates["ok"] is True and gates["readiness_warn"] is None


def test_readiness_unreadable_warns_never_blocks(tmp_path):
    gates = fa.check_gates({"CHAD_EXECUTION_MODE": "paper"},
                           tmp_path / "missing.json")
    assert gates["ok"] is True
    assert "unreadable" in gates["readiness_warn"]


def test_wrong_confirm_token_hard_error():
    with pytest.raises(fa.FlattenAbort):
        fa.verify_tokens(execute=True, confirm="FLATTEN", scope="chad",
                         scope_confirm=None)


def test_broker_all_requires_second_token():
    with pytest.raises(fa.FlattenAbort):
        fa.verify_tokens(execute=False, confirm=None, scope="broker-all",
                         scope_confirm=None)
    # correct double token passes
    fa.verify_tokens(execute=True, confirm=fa.CONFIRM_TOKEN,
                     scope="broker-all",
                     scope_confirm=fa.INCLUDE_EXCLUDED_TOKEN)


def test_drill_needs_no_tokens():
    fa.verify_tokens(execute=False, confirm=None, scope="chad",
                     scope_confirm=None)


# --------------------------------------------------------------------------- #
# probe (Phase 0, fail-closed)
# --------------------------------------------------------------------------- #

def test_probe_dead_socket_aborts_never_flat():
    with pytest.raises(fa.FlattenAbort, match="BROKER_TRUTH_UNAVAILABLE"):
        fa.probe_broker(FakeIB(connected=False))


def test_probe_missing_enumeration_aborts():
    ib = FakeIB()
    del FakeIB.reqAllOpenOrders  # simulate a facade without cross-client enum
    try:
        with pytest.raises(fa.FlattenAbort, match="enumeration unavailable"):
            fa.probe_broker(ib)
    finally:
        FakeIB.reqAllOpenOrders = _REQ_ALL  # restore


_REQ_ALL = FakeIB.reqAllOpenOrders


def test_probe_parses_positions_and_cross_client_orders():
    ib = FakeIB(
        positions=[_pos("V", 190.0), _pos("TLT", -640.0), _pos("ZERO", 0.0)],
        trades=[_trade(1, 99, "IEMG"), _trade(2, 0, "VWO", status="PreSubmitted")],
        own_orders=0)
    probe = fa.probe_broker(ib)
    assert probe.positions["V"]["qty"] == 190.0
    assert probe.positions["TLT"]["qty"] == -640.0
    assert "ZERO" not in probe.positions          # flat rows dropped
    assert len(probe.all_open_orders) == 2
    assert {o["client_id"] for o in probe.all_open_orders} == {99, 0}
    # visibility-split proof: scoped count captured alongside the real one
    assert probe.own_open_orders_count == 0


def test_snapshot_crosscheck_lists_divergence_live_wins(tmp_path):
    probe = fa.BrokerProbe(positions={"V": {"qty": 190.0}})
    snap = tmp_path / "positions_snapshot.json"
    snap.write_text(json.dumps({"positions": [
        {"symbol": "V", "position": 200.0},
        {"symbol": "GONE", "position": 5.0},
    ]}))
    diffs = fa.crosscheck_snapshot(probe, snap)
    by_sym = {d.get("symbol"): d for d in diffs}
    assert by_sym["V"]["live_qty"] == 190.0 and by_sym["V"]["snapshot_qty"] == 200.0
    assert by_sym["GONE"]["live_qty"] == 0.0
    # unreadable snapshot is advisory, never fatal
    diffs2 = fa.crosscheck_snapshot(probe, tmp_path / "missing.json")
    assert any("error" in d for d in diffs2)


# --------------------------------------------------------------------------- #
# clamp + scope resolution (D1 + rider)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strat,broker,expected", [
    (8.0, 5.0, 5.0),      # guard bloated -> broker wins (INCIDENT-0713)
    (3.0, 5.0, 3.0),      # guard smaller -> guard wins
    (5.0, 0.0, 0.0),      # broker flat -> nothing
    (5.0, -5.0, 0.0),     # sign flip -> nothing
    (-4.0, -6.0, -4.0),   # short side clamps by magnitude
])
def test_clamp_to_broker(strat, broker, expected):
    assert fa._clamp_to_broker(strat, broker) == expected


def _probe(**symqty):
    return fa.BrokerProbe(positions={
        s: {"qty": q, "sec_type": "STK"} for s, q in symqty.items()})


def _guard(**legs):
    """legs: key=(symbol, side, qty)"""
    out = {"_version": 1}
    for key, (sym, side, qty) in legs.items():
        out[key.replace("_", "|", 1)] = {
            "open": True, "symbol": sym, "side": side, "quantity": qty}
    out["broker_sync|X"] = {"open": True, "symbol": "X", "side": "BUY",
                            "quantity": 1.0}  # mirrors are never leg inputs
    return out


def test_chad_scope_clamps_and_names_everything():
    res = fa.resolve_targets(
        probe=_probe(PSQ=5.0, MSFT=34.0, UNH=7.0, GHOST=0.0),
        guard_state=_guard(gamma_PSQ=("PSQ", "BUY", 8.0),
                           beta_MSFT=("MSFT", "BUY", 14.0),
                           omega_GHOST=("GHOST", "BUY", 3.0)),
        scope="chad", excluded=["MSFT", "AAPL"])
    targets = {t["position_key"]: t for t in res["targets"]}
    # PSQ: guard 8 clamped to broker 5, reduce-only SELL
    assert targets["gamma|PSQ"]["quantity"] == 5.0
    assert targets["gamma|PSQ"]["close_side"] == "SELL"
    assert targets["gamma|PSQ"]["origin"] == "chad"
    # MSFT excluded: never a target, NAMED untouched (D1 rider)
    assert "beta|MSFT" not in targets
    untouched = {u["symbol"]: u for u in res["untouched"]}
    assert untouched["MSFT"]["reason"] == "operator_excluded"
    assert untouched["MSFT"]["broker_qty"] == 34.0
    # UNH: broker-held but no chad leg -> NAMED not_chad_attributed
    assert untouched["UNH"]["reason"] == "not_chad_attributed"
    assert untouched["UNH"]["unclosed_qty"] == 7.0
    # GHOST: ledger leg on broker-flat symbol -> NAMED, no order ever
    assert untouched["GHOST"]["reason"] == "broker_flat_ledger_only"
    assert untouched["GHOST"]["ledger_qty"] == 3.0
    assert all(t["symbol"] != "GHOST" for t in res["targets"])


def test_chad_scope_multi_leg_budget_stays_reduce_only():
    res = fa.resolve_targets(
        probe=_probe(TLT=6.0),
        guard_state=_guard(gamma_TLT=("TLT", "BUY", 4.0),
                           omega_TLT=("TLT", "BUY", 4.0),
                           short_TLT=("TLT", "SELL", 2.0)),
        scope="chad", excluded=[])
    tlt = [t for t in res["targets"] if t["symbol"] == "TLT"]
    # net 4+4-2=6 clamped to broker 6; opposing SELL leg never a close target
    assert sum(t["quantity"] for t in tlt) <= 6.0
    assert all(t["close_side"] == "SELL" for t in tlt)
    assert all(not t["position_key"].startswith("short|") for t in tlt)


def test_broker_all_adds_operator_legs_with_origin():
    res = fa.resolve_targets(
        probe=_probe(MSFT=34.0, UNH=7.0),
        guard_state=_guard(),
        scope="broker-all", excluded=["MSFT"])
    by_key = {t["position_key"]: t for t in res["targets"]}
    assert by_key["operator|MSFT"]["origin"] == "operator"
    assert by_key["operator|MSFT"]["quantity"] == 34.0
    assert by_key["operator|UNH"]["origin"] == "unattributed"
    assert by_key["operator|UNH"]["strategy"] == "operator_flatten"


def test_short_position_closes_with_buy():
    res = fa.resolve_targets(
        probe=_probe(TLT=-640.0),
        guard_state=_guard(gamma_TLT=("TLT", "SELL", 640.0)),
        scope="chad", excluded=[])
    (t,) = res["targets"]
    assert t["open_side"] == "SELL" and t["close_side"] == "BUY"
    assert t["quantity"] == 640.0


def test_close_dicts_split_chad_vs_operator():
    chad, oper = fa.close_dicts_from_targets([
        {"symbol": "PSQ", "position_key": "gamma|PSQ", "strategy": "gamma",
         "open_side": "BUY", "close_side": "SELL", "quantity": 5.0,
         "sec_type": "STK", "origin": "chad"},
        {"symbol": "MSFT", "position_key": "operator|MSFT",
         "strategy": "operator_flatten", "open_side": "BUY",
         "close_side": "SELL", "quantity": 34.0, "sec_type": "STK",
         "origin": "operator"},
    ])
    assert [c["symbol"] for c in chad] == ["PSQ"]
    assert chad[0]["action"] == "CLOSE" and chad[0]["reason"] == "flatten_all_chad"
    assert [o["symbol"] for o in oper] == ["MSFT"]
    assert oper[0]["reason"] == "flatten_all_operator"


def test_excluded_config_loader_never_empty(tmp_path):
    cfg = tmp_path / "exclusions.json"
    cfg.write_text(json.dumps({
        "reconciler_non_chad_symbols": ["AAPL", "MSFT"],
        "broker_preexisting_symbols": ["SPY", "QQQ"],
        "exclusion_policy": {"BAC": {}},
    }))
    assert fa.load_excluded_symbols(cfg) == ["AAPL", "BAC", "MSFT", "QQQ", "SPY"]
    # unreadable -> the reconciler's local floor, never an empty set
    assert fa.load_excluded_symbols(tmp_path / "missing.json") == ["AAPL", "MSFT"]


# --------------------------------------------------------------------------- #
# cancel (D3 + rider)
# --------------------------------------------------------------------------- #

def test_drill_cancels_nothing_but_names_collateral():
    ib = FakeIB(trades=[_trade(1, 99, "IEMG"), _trade(2, 1234, "VWO"),
                        _trade(3, 0, "MANUAL")])
    res = fa.cancel_phase(ib, chad_client_ids=[99, 7716], execute=False)
    assert ib.global_cancel_calls == 0
    assert res["executed"] is False and res["verified_zero"] is None
    assert {o["order_id"] for o in res["collateral_non_chad"]} == {2, 3}


def test_execute_cancels_and_verifies_zero():
    ib = FakeIB(trades=[_trade(1, 99, "IEMG"), _trade(2, 0, "MANUAL")],
                cancel_lag=2)
    res = fa.cancel_phase(ib, chad_client_ids=[99], execute=True,
                          sleep=lambda s: None)
    assert ib.global_cancel_calls == 1
    assert res["verified_zero"] is True and res["survivors"] == []
    assert [o["order_id"] for o in res["collateral_non_chad"]] == [2]


def test_cancel_survivors_reported_terminal_never_counted():
    ib = FakeIB(trades=[_trade(1, 99, "A"), _trade(2, 99, "B", status="Filled")],
                stubborn_ids={1})
    res = fa.cancel_phase(ib, chad_client_ids=[99], execute=True,
                          sleep=lambda s: None, timeout_s=0.0)
    assert res["verified_zero"] is False
    assert [o["order_id"] for o in res["survivors"]] == [1]  # Filled is terminal


# --------------------------------------------------------------------------- #
# confirm: SLA + residual (Phase 3)
# --------------------------------------------------------------------------- #

def _submitted(symbol, status="Filled", with_trade=True, ack_s=0.2, fill_s=1.0):
    t0 = NOW
    log = []
    if with_trade:
        log = [SimpleNamespace(status="PendingSubmit", time=t0),
               SimpleNamespace(status="Submitted", time=t0 + timedelta(seconds=ack_s)),
               SimpleNamespace(status="Filled", time=t0 + timedelta(seconds=fill_s))]
    trade = SimpleNamespace(orderStatus=SimpleNamespace(status="Filled"), log=log) \
        if with_trade else None
    return SimpleNamespace(
        symbol=symbol, side="SELL", quantity=5.0, status=status,
        idempotency_key=f"k|{symbol}", ib_order_id=7,
        submitted_at=t0, raw={"trade": trade} if with_trade else {})


def test_sla_math_ack_and_fill_ms():
    out = fa.measure_orders([_submitted("PSQ", ack_s=0.2, fill_s=1.0),
                             _submitted("UNH", ack_s=0.4, fill_s=2.0)],
                            poll=lambda: None, timeout_s=0.0)
    rows = {r["symbol"]: r for r in out["orders"]}
    assert rows["PSQ"]["ack_ms"] == pytest.approx(200.0)
    assert rows["PSQ"]["fill_ms"] == pytest.approx(1000.0)
    assert out["sla"]["measured_fills"] == 2
    assert out["sla"]["ack_ms_p50"] in (200.0, 400.0)
    assert out["sla"]["fill_ms_p95"] == pytest.approx(2000.0)


def test_duplicate_blocked_classified_benign_idempotent():
    so = _submitted("PSQ", status="duplicate_blocked", with_trade=False)
    out = fa.measure_orders([so], poll=lambda: None, timeout_s=0.0)
    (row,) = out["orders"]
    assert row["benign_duplicate"] is True
    assert row["ack_ms"] is None            # no broker leg, no SLA sample
    assert out["sla"]["measured_acks"] == 0


def test_residual_flat_and_incomplete_verdicts():
    resolution = {
        "targets": [{"symbol": "PSQ", "quantity": 5.0},
                    {"symbol": "UNH", "quantity": 7.0}],
        "untouched": [{"symbol": "MSFT", "broker_qty": 34.0,
                       "reason": "operator_excluded"}],
    }
    # PSQ fully closed; UNH left 3 -> RESIDUAL; MSFT restated by name
    ib = FakeIB(positions=[_pos("UNH", 3.0), _pos("MSFT", 34.0)])
    res = fa.residual_check(ib, resolution)
    per = {r["symbol"]: r for r in res["per_symbol"]}
    assert per["PSQ"]["verdict"] == "FLAT"
    assert per["UNH"]["verdict"] == "RESIDUAL"
    assert per["UNH"]["residual_qty"] == 3.0
    assert res["overall"] == "INCOMPLETE"
    (named,) = res["untouched_named"]
    assert named["symbol"] == "MSFT" and named["verdict"] == "EXCLUDED_UNTOUCHED"


def test_residual_check_reprobes_fail_closed():
    with pytest.raises(fa.FlattenAbort):
        fa.residual_check(FakeIB(connected=False),
                          {"targets": [], "untouched": []})


# --------------------------------------------------------------------------- #
# report artifacts (Phase 4 / drill)
# --------------------------------------------------------------------------- #

def test_drill_writes_ratification_proof(tmp_path):
    out = fa.write_report(tmp_path, {"probe": {}}, drill=True, now=NOW)
    assert out == tmp_path / "ratification" / "PROOF_FLATTEN_DRILL_20260723.json"
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == "flatten_drill_proof.v1"


def test_execute_writes_stamped_report(tmp_path):
    out = fa.write_report(tmp_path, {"probe": {}}, drill=False, now=NOW)
    assert out == tmp_path / "flatten_all_20260723T160000Z.json"
    assert json.loads(out.read_text())["schema_version"] == "flatten_all_report.v1"


# --------------------------------------------------------------------------- #
# clientId registry (plan §2.1)
# --------------------------------------------------------------------------- #

def test_flatten_all_client_id_registered_and_distinct():
    from chad.execution import ibkr_client_ids as reg
    cmap = reg.client_id_map()
    assert cmap["FLATTEN_ALL"] == reg.FLATTEN_ALL == 7716
    assert cmap["FUTURES_FLATTEN_ONESHOT"] == 7715
    assert reg.FLATTEN_ALL != reg.FUTURES_FLATTEN_ONESHOT
    reg.assert_no_collisions()


# --------------------------------------------------------------------------- #
# W4B-6: act-phase orchestration (drill/execute on the same code path)
# --------------------------------------------------------------------------- #

class ClosingFakeAdapter:
    """Adapter double that behaves like the paper broker: a submitted close
    reduces the FakeIB book and returns a filled SubmittedOrder-like. dry_run
    short-circuits at the placeOrder boundary exactly like the real adapter."""

    def __init__(self, ib, dry_run):
        self._ib = ib
        self._dry_run = dry_run
        self.intents = []

    def submit_strategy_trade_intents(self, intents):
        out = []
        for it in intents:
            self.intents.append(it)
            if self._dry_run:
                out.append(SimpleNamespace(
                    symbol=it.symbol, side=it.side, quantity=it.quantity,
                    status="dry_run", idempotency_key="k", ib_order_id=None,
                    submitted_at=NOW, raw={}))
                continue
            for p in self._ib._positions:
                if p.contract.symbol == it.symbol:
                    delta = it.quantity if str(it.side).upper() == "BUY" else -it.quantity
                    p.position += delta
            out.append(SimpleNamespace(
                symbol=it.symbol, side=it.side, quantity=it.quantity,
                status="paper_fill", idempotency_key="k", ib_order_id=11,
                submitted_at=NOW, raw={}))
        return out


def _kraken_db(tmp_path, rows=()):
    import sqlite3
    db = tmp_path / "exec_state_paper.sqlite3"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE kraken_trusted_lots (strategy TEXT, symbol TEXT, "
                "direction TEXT, qty_remaining REAL, entry_price REAL, "
                "entry_fee_per_unit REAL, opened_at_utc TEXT, regime TEXT DEFAULT '')")
    for r in rows:
        con.execute("INSERT INTO kraken_trusted_lots (strategy, symbol, direction, "
                    "qty_remaining, entry_price, entry_fee_per_unit, opened_at_utc) "
                    "VALUES (?,?,?,?,?,?,?)", r)
    con.commit()
    con.close()
    return db


def _kraken_deps(tmp_path, rows=(), held=None, dispatched=None):
    db = _kraken_db(tmp_path, rows)
    held = held if held is not None else {}
    dispatched = dispatched if dispatched is not None else []

    def process(intent):
        dispatched.append(intent)
        return {"trusted": True}

    return {
        "db_path": db,
        "pair_of": lambda sym: {"BTC-USD": "XBTUSD", "SOL-USD": "SOLUSD"}.get(sym, sym),
        "open_qty_reader": lambda strat, sym: held.get(f"{strat}|{sym}", 0.0),
        "process_intent": process,
    }


def test_kraken_probe_groups_lots_and_names_unreadable(tmp_path):
    db = _kraken_db(tmp_path, [
        ("alpha_crypto", "BTC-USD", "long", 0.3, 50000.0, 10.0, "t"),
        ("alpha_crypto", "BTC-USD", "long", 0.2, 51000.0, 10.0, "t"),
        ("alpha_crypto", "SOL-USD", "short", 5.0, 150.0, 0.1, "t"),
    ])
    probe = fa.kraken_probe(db)
    assert probe["available"] is True and probe["resting_orders"] == 0
    lots = {(l["position_key"], l["direction"]): l["qty"] for l in probe["open_lots"]}
    assert lots[("alpha_crypto|BTC-USD", "long")] == pytest.approx(0.5)
    assert lots[("alpha_crypto|SOL-USD", "short")] == 5.0
    bad = fa.kraken_probe(tmp_path / "nope" / "missing.sqlite3")
    assert bad["available"] is False and "unreadable" in bad["error"]


def test_kraken_close_intents_side_pair_and_deterministic_key():
    intents = fa.kraken_close_intents(
        [{"position_key": "a|BTC-USD", "strategy": "a", "symbol": "BTC-USD",
          "direction": "long", "qty": 0.5},
         {"position_key": "a|SOL-USD", "strategy": "a", "symbol": "SOL-USD",
          "direction": "short", "qty": 5.0}],
        lambda sym: {"BTC-USD": "XBTUSD", "SOL-USD": "SOLUSD"}[sym])
    btc, sol = intents
    assert btc["side"] == "sell" and btc["pair"] == "XBTUSD"   # long -> sell
    assert sol["side"] == "buy" and sol["pair"] == "SOLUSD"    # short -> buy
    assert btc["idempotency_key"] == "flatten|XBTUSD|sell|0.5000000000"


def test_kraken_execute_clamps_at_dispatch_and_drill_never_dispatches():
    dispatched = []
    intents = [{"position_key": "a|BTC-USD", "strategy": "a", "symbol": "BTC-USD",
                "pair": "XBTUSD", "side": "sell", "quantity": 0.5,
                "reason": "flatten_all_kraken", "idempotency_key": "k1"},
               {"position_key": "a|SOL-USD", "strategy": "a", "symbol": "SOL-USD",
                "pair": "SOLUSD", "side": "buy", "quantity": 5.0,
                "reason": "flatten_all_kraken", "idempotency_key": "k2"}]

    def process(intent):
        dispatched.append(intent)
        return {"trusted": True}

    # book moved: BTC now 0.3 (clamp), SOL flat (drop, NAMED)
    res = fa.kraken_execute(
        intents, open_qty_reader=lambda s, sym: {"BTC-USD": 0.3}.get(sym, 0.0),
        process_intent=process, execute=True)
    rows = {r.get("position_key"): r for r in res if "position_key" in r}
    assert rows["a|BTC-USD"]["quantity"] == pytest.approx(0.3)
    assert rows["a|BTC-USD"]["result"] == {"trusted": True}
    assert [r for r in res if "reclamp_dropped" in r] == [
        {"reclamp_dropped": ["a|SOL-USD"]}]
    assert len(dispatched) == 1 and dispatched[0].reduce_only is True

    # drill: builds, clamps, NEVER dispatches
    dispatched.clear()
    res2 = fa.kraken_execute(
        intents, open_qty_reader=lambda s, sym: 1.0,
        process_intent=process, execute=False)
    assert dispatched == []
    assert all(r["result"] == {"status": "dry_run"}
               for r in res2 if "position_key" in r)


def test_submit_closes_chokepoint_vs_authorized_bypass():
    ib = FakeIB(positions=[_pos("PSQ", 5.0), _pos("MSFT", 34.0)])
    adapter = ClosingFakeAdapter(ib, dry_run=True)
    # chad leg on an EXCLUDED symbol: apply_close_intents refuses (wall 3)
    res = fa.submit_closes(
        chad_closes=[{"symbol": "MSFT", "action": "CLOSE", "open_side": "BUY",
                      "close_side": "SELL", "quantity": 34.0,
                      "reason": "flatten_all_chad", "position_key": "x|MSFT",
                      "strategy": "x"}],
        operator_closes=[], adapter=adapter)
    assert res["submitted"] == [] and adapter.intents == []
    # operator leg (broker-all double-token): designed bypass, same adapter
    res2 = fa.submit_closes(
        chad_closes=[{"symbol": "PSQ", "action": "CLOSE", "open_side": "BUY",
                      "close_side": "SELL", "quantity": 5.0,
                      "reason": "flatten_all_chad", "position_key": "g|PSQ",
                      "strategy": "g"}],
        operator_closes=[{"symbol": "MSFT", "action": "CLOSE", "open_side": "BUY",
                          "close_side": "SELL", "quantity": 34.0,
                          "reason": "flatten_all_operator",
                          "position_key": "operator|MSFT",
                          "strategy": "operator_flatten"}],
        adapter=adapter)
    assert [s.symbol for s in res2["submitted"]] == ["PSQ", "MSFT"]
    assert res2["submit_errors"] == []
    # the W4B-2 provenance stamps ride every leg to the adapter boundary
    assert all(i.meta["close_origin"] == "apply_close_intents"
               for i in adapter.intents)


def test_run_flatten_drill_end_to_end(tmp_path):
    ib = FakeIB(positions=[_pos("PSQ", 5.0), _pos("MSFT", 34.0)],
                trades=[_trade(1, 1234, "IEMG")])
    adapter = ClosingFakeAdapter(ib, dry_run=True)
    guard = {"gamma|PSQ": {"open": True, "symbol": "PSQ", "side": "BUY",
                           "quantity": 8.0}}
    dispatched = []
    kraken = _kraken_deps(tmp_path,
                          rows=[("a", "BTC-USD", "long", 0.5, 5e4, 1.0, "t")],
                          held={"a|BTC-USD": 0.5}, dispatched=dispatched)
    payload = fa.run_flatten(
        execute=False, scope="chad", ib=ib, adapter=adapter,
        guard_state=guard, excluded=["MSFT"], chad_client_ids=[99],
        kraken=kraken, now=NOW, log=lambda s: None)
    # drill: nothing cancelled, nothing dispatched, chain proven
    assert ib.global_cancel_calls == 0
    assert dispatched == []
    assert payload["overall"] == "DRILL_COMPLETE"
    assert payload["mode"] == "drill" and payload["drill_gaps"] == []
    # the chad close went through the REAL chokepoint into the adapter dry_run
    assert [i.symbol for i in adapter.intents] == ["PSQ"]
    assert payload["sla"]["orders"][0]["status"] == "dry_run"
    assert payload["closes"]["chad"][0]["quantity"] == 5.0    # broker-clamped
    # named untouched + visibility split + kraken inventory all present
    assert [u["symbol"] for u in payload["resolution"]["untouched"]] == ["MSFT"]
    assert payload["probe"]["visibility_split"]["all_open_orders"] == 1
    assert len(payload["kraken_probe"]["open_lots"]) == 1
    assert [r["quantity"] for r in payload["closes"]["kraken"]
            if "position_key" in r] == [0.5]


def test_run_flatten_execute_end_to_end_flat(tmp_path):
    ib = FakeIB(positions=[_pos("PSQ", 5.0), _pos("MSFT", 34.0)],
                trades=[_trade(1, 99, "IEMG")])
    adapter = ClosingFakeAdapter(ib, dry_run=False)
    guard = {"gamma|PSQ": {"open": True, "symbol": "PSQ", "side": "BUY",
                           "quantity": 5.0}}
    dispatched = []
    kraken = _kraken_deps(tmp_path,
                          rows=[("a", "BTC-USD", "long", 0.5, 5e4, 1.0, "t")],
                          held={"a|BTC-USD": 0.5}, dispatched=dispatched)
    payload = fa.run_flatten(
        execute=True, scope="chad", ib=ib, adapter=adapter,
        guard_state=guard, excluded=["MSFT"], chad_client_ids=[99],
        kraken=kraken, now=NOW, sleep=lambda s: None, log=lambda s: None)
    assert ib.global_cancel_calls == 1
    assert payload["cancel"]["verified_zero"] is True
    # PSQ closed on the fake broker; MSFT untouched and restated by name
    per = {r["symbol"]: r for r in payload["residuals"]["per_symbol"]}
    assert per["PSQ"]["verdict"] == "FLAT"
    (named,) = payload["residuals"]["untouched_named"]
    assert named["symbol"] == "MSFT" and named["verdict"] == "EXCLUDED_UNTOUCHED"
    assert len(dispatched) == 1                     # kraken lane dispatched
    # kraken residual read is from the (unchanged) tmp book — flagged not flat,
    # which drives the honest INCOMPLETE overall (test book has no engine)
    assert payload["kraken_residual"]["checked"] is True
    assert payload["overall"] == "INCOMPLETE"


def test_run_flatten_probe_failure_aborts_loud(tmp_path):
    with pytest.raises(fa.FlattenAbort):
        fa.run_flatten(execute=False, scope="chad", ib=FakeIB(connected=False),
                       adapter=None, guard_state={}, excluded=[],
                       chad_client_ids=[], now=NOW, log=lambda s: None)


def test_missing_kraken_lane_is_a_named_drill_gap():
    ib = FakeIB(positions=[_pos("PSQ", 5.0)])
    adapter = ClosingFakeAdapter(ib, dry_run=True)
    payload = fa.run_flatten(
        execute=False, scope="chad", ib=ib, adapter=adapter, guard_state={},
        excluded=[], chad_client_ids=[99], kraken=None, now=NOW,
        log=lambda s: None)
    assert payload["overall"] == "DRILL_GAPS"
    assert any(g["stage"] == "kraken_probe" for g in payload["drill_gaps"])


def test_append_events_writes_ndjson(tmp_path):
    out = fa.append_events(tmp_path / "flatten_all", {
        "mode": "drill", "scope": "chad", "overall": "DRILL_COMPLETE",
        "resolution": {"targets": [1, 2]}, "cancel": {"orders_before_count": 3,
                                                      "collateral_non_chad": [1]},
        "sla": {"orders": [1]},
    }, now=NOW)
    (row,) = [json.loads(l) for l in out.read_text().splitlines()]
    assert row["schema_version"] == "flatten_all_event.v1"
    assert row["targets"] == 2 and row["collateral_non_chad"] == 1
