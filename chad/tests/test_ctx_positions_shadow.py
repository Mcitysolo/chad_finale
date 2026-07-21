"""W2B-3 — positions shadow recorder: signal-set diff + state isolation + heartbeat.

Hermetic: evidence/heartbeat go to tmp_path; ctx is faked. Proves the shadow
acts on nothing, records the OFF-vs-ON diff, flags added SELLs (the go/no-go
danger signal), and leaves strategy in-memory state untouched.
"""
from __future__ import annotations

import json
from collections import namedtuple
from types import SimpleNamespace

from chad.types import AssetClass, Position
from chad.core import context_positions as cp
from chad.core import ctx_positions_shadow as sh

_Sig = namedtuple("Sig", "symbol side")
_UNH = Position(symbol="UNH", asset_class=AssetClass.EQUITY, quantity=228.0, avg_price=424.98)


def _ctx(positions=None, prices=None):
    return SimpleNamespace(
        portfolio=SimpleNamespace(positions=dict(positions or {}), cash=0.0,
                                  total_equity=1e6, equity=1e6, net_liq=1e6, extra={}),
        prices=dict(prices or {"UNH": 424.98}), bars={}, now=None,
    )


def _reg(name, handler):
    return SimpleNamespace(name=SimpleNamespace(value=name), handler=handler)


def _rh(name, handler, ctx, logger):
    return handler(ctx) or []


def _view_known():
    return cp.PositionsView(cp.STATUS_KNOWN, {"UNH": _UNH}, "fresh", {"n_injected": 1})


def _paths(tmp_path):
    return {"evidence_dir": tmp_path / "data" / "ctx_positions_shadow",
            "heartbeat_path": tmp_path / "runtime" / "ctx_positions_heartbeat.json"}


# --------------------------------------------------------------------------- #
# build_ctx_on — faithful clone (same market data, swapped book)
# --------------------------------------------------------------------------- #

def test_build_ctx_on_swaps_only_positions():
    ctx_off = _ctx(positions={}, prices={"UNH": 424.98})
    ctx_on = sh.build_ctx_on(ctx_off, _view_known())
    assert ctx_on.portfolio.positions == {"UNH": _UNH}
    assert ctx_off.portfolio.positions == {}          # original untouched
    assert ctx_on.prices == ctx_off.prices            # same market data


# --------------------------------------------------------------------------- #
# shadow diff — removed BUY (safe) and added SELL (go/no-go danger)
# --------------------------------------------------------------------------- #

def _gamma_buy(ctx):
    pos = ctx.portfolio.positions.get("UNH")
    if pos is not None and float(getattr(pos, "quantity", 0.0)) > 0.0:
        return []
    return [_Sig("UNH", "BUY")]


def _gamma_sell_on_hold(ctx):
    pos = ctx.portfolio.positions.get("UNH")
    if pos is not None and float(getattr(pos, "quantity", 0.0)) > 0.0:
        return [_Sig("UNH", "SELL")]      # native exit fires only when it sees the book
    return []


def test_shadow_records_removed_buy(tmp_path):
    regs = [_reg("gamma", _gamma_buy)]
    available_off = {"gamma": _gamma_buy(_ctx())}     # flat book -> BUY present
    rec = sh.record_cycle(mode="shadow", regs=regs, run_handler=_rh,
                          ctx_off=_ctx(), available_off=available_off, view=_view_known(),
                          **_paths(tmp_path))
    assert rec["removed"] == [["gamma", "UNH", "BUY"]]
    assert rec["added"] == []
    assert rec["added_sells_count"] == 0

    files = list((tmp_path / "data" / "ctx_positions_shadow").glob("ctx_positions_*.ndjson"))
    assert len(files) == 1
    line = json.loads(files[0].read_text().strip().splitlines()[0])
    assert line["schema_version"] == "ctx_positions_shadow.v1"
    assert line["removed"] == [["gamma", "UNH", "BUY"]]


def test_shadow_flags_added_sell(tmp_path):
    regs = [_reg("gamma", _gamma_sell_on_hold)]
    available_off = {"gamma": _gamma_sell_on_hold(_ctx())}   # flat -> nothing
    rec = sh.record_cycle(mode="shadow", regs=regs, run_handler=_rh,
                          ctx_off=_ctx(), available_off=available_off, view=_view_known(),
                          **_paths(tmp_path))
    assert rec["added_sells_count"] == 1                      # the danger signal
    assert rec["added_sells"] == [["gamma", "UNH", "SELL"]]


def test_evidence_under_data_not_runtime(tmp_path):
    p = _paths(tmp_path)
    sh.record_cycle(mode="shadow", regs=[_reg("gamma", _gamma_buy)], run_handler=_rh,
                    ctx_off=_ctx(), available_off={"gamma": _gamma_buy(_ctx())},
                    view=_view_known(), **p)
    assert (tmp_path / "data" / "ctx_positions_shadow").exists()
    assert "data" in str(p["evidence_dir"]) and "runtime" not in str(p["evidence_dir"])


# --------------------------------------------------------------------------- #
# heartbeat — written in shadow and on
# --------------------------------------------------------------------------- #

def test_heartbeat_written_shadow(tmp_path):
    p = _paths(tmp_path)
    sh.record_cycle(mode="shadow", regs=[_reg("gamma", _gamma_buy)], run_handler=_rh,
                    ctx_off=_ctx(), available_off={"gamma": _gamma_buy(_ctx())},
                    view=_view_known(), **p)
    hb = json.loads(p["heartbeat_path"].read_text())
    assert hb["mode"] == "shadow"
    assert hb["schema_version"] == "ctx_positions_heartbeat.v1"


def test_on_mode_heartbeat_only_no_evidence(tmp_path):
    p = _paths(tmp_path)
    rec = sh.record_cycle(mode="on", regs=[_reg("gamma", _gamma_buy)], run_handler=_rh,
                          ctx_off=_ctx(positions={"UNH": _UNH}),
                          available_off={"gamma": []}, view=_view_known(), **p)
    assert rec is None                                        # no diff record in on
    assert not (tmp_path / "data" / "ctx_positions_shadow").exists()   # no evidence ndjson
    assert json.loads(p["heartbeat_path"].read_text())["mode"] == "on"


# --------------------------------------------------------------------------- #
# state isolation — the counterfactual run leaves _STATE untouched
# --------------------------------------------------------------------------- #

def test_snapshot_restore_isolates_gamma_state():
    import chad.strategies.gamma as gmod
    gmod._STATE.set("SENTINEL", {"entry": 1.0, "held": 1, "peak": 2.0})
    snap = sh._snapshot_strategy_state()
    gmod._STATE.set("SENTINEL", {"entry": 9.9, "held": 9, "peak": 9.9})
    gmod._STATE.set("EXTRA", {"entry": 5.0, "held": 5, "peak": 5.0})
    sh._restore_strategy_state(snap)
    assert gmod._STATE.get("SENTINEL") == {"entry": 1.0, "held": 1, "peak": 2.0}
    assert gmod._STATE.get("EXTRA") == {}                    # observation residue erased


def test_record_cycle_does_not_perturb_state(tmp_path):
    import chad.strategies.gamma as gmod
    gmod._STATE.set("UNH", {"entry": 3.0, "held": 2, "peak": 4.0})
    before = gmod._STATE.get("UNH")

    def _mutating_handler(ctx):
        gmod._STATE.set("UNH", {"entry": 99.0, "held": 99, "peak": 99.0})   # observation-run mutation
        return [_Sig("UNH", "SELL")] if ctx.portfolio.positions.get("UNH") else []

    sh.record_cycle(mode="shadow", regs=[_reg("gamma", _mutating_handler)], run_handler=_rh,
                    ctx_off=_ctx(), available_off={"gamma": []}, view=_view_known(),
                    **_paths(tmp_path))
    assert gmod._STATE.get("UNH") == before                  # restored after the observation run
