"""G3C U1 — unit tests for the margin/BP shadow gate + its IBKR chokepoint wiring.

Covers the shadow contract (design v2.2 Part 7/8): fail-open on error (MARGIN_GATE_ERROR),
shadow NEVER blocks (even on a BLOCK verdict), the enforce path DOES block behind an injected
enforce config, grep-able marker, evidence ndjson, and — at the adapter chokepoint — that a
shadow gate does not alter the submission outcome while an enforce BLOCK returns before the
idempotency claim (no idempotency row). No broker, no network, no runtime writes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig
from chad.execution.ibkr_executor import StrategyTradeIntent
from chad.execution.margin_shadow_gate import (
    MARKER_ERROR,
    MARKER_SHADOW,
    MarginShadowGate,
    build_default_shadow_gate,
    order_view_from_intent,
)
from chad.risk.buying_power_provider import BuyingPowerSnapshot
from chad.risk.margin_block import load_frozen_config

REPO = Path("/home/ubuntu/chad_finale")
NOW = 1_800_000_000.0


# --------------------------------------------------------------------------- #
# Fixtures / builders.
# --------------------------------------------------------------------------- #
def _shadow_cfg():
    return load_frozen_config(REPO / "config" / "margin_block.json")  # mode=shadow (frozen)


def _enforce_cfg():
    # Inject an enforce config WITHOUT touching the frozen file (constraint 5).
    return dataclasses.replace(_shadow_cfg(), mode="enforce_paper")


def _usable_snap(net=1_000_000.0, excess=800_000.0, init=50_000.0):
    return BuyingPowerSnapshot(
        usable=True, reason="OK", detail="", currency="CAD",
        captured_at_epoch=NOW, age_seconds=1.0, ttl_seconds=30.0,
        net_liquidation=net, buying_power=net * 2, excess_liquidity=excess,
        available_funds=excess, full_init_margin_req=init, full_maint_margin_req=init * 0.8,
    )


def _stale_snap():
    return BuyingPowerSnapshot.fail_closed("MISSING", "no margin data", ttl_seconds=30.0)


def _src(snap, positions=None):
    return lambda ov: (snap, positions or {})


def _gate(cfg, snap, *, positions=None, evidence_path=None, open_orders=lambda: []):
    return MarginShadowGate(
        cfg, snapshot_source=_src(snap, positions),
        open_orders_source=open_orders, evidence_path=evidence_path,
    )


class _OV:
    """A minimal intent-like object for order_view_from_intent."""
    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "AAPL")
        self.side = kw.get("side", "BUY")
        self.asset_class = kw.get("asset_class", "equity")
        self.quantity = kw.get("quantity", 10.0)
        self.currency = kw.get("currency", "USD")
        self.notional_estimate = kw.get("notional_estimate", 1900.0)
        self.limit_price = kw.get("limit_price", 190.0)
        self.meta = kw.get("meta", {})
        self.strategy = kw.get("strategy", "alpha")
        self.source_strategies = kw.get("source_strategies", ("alpha",))


def _view(**kw):
    return order_view_from_intent(_OV(**kw), order_id=kw.get("order_id", "k1"))


# --------------------------------------------------------------------------- #
# 1. order_view mapping.
# --------------------------------------------------------------------------- #
def test_order_view_maps_fields_and_etf_to_equity():
    v = order_view_from_intent(_OV(asset_class="etf", symbol="spy", currency=""), order_id="oid")
    assert v["order_id"] == "oid"
    assert v["symbol"] == "SPY"
    assert v["asset_class"] == "equity"     # etf normalized → equity (RegT)
    assert v["currency"] == "USD"           # empty defaults to USD
    assert v["qty"] == 10.0
    assert v["whatif_init_margin"] is None   # shadow: no whatIf broker call


def test_order_view_nonpositive_qty_and_price_become_none():
    v = order_view_from_intent(_OV(quantity=0.0, limit_price=-5.0, notional_estimate=0.0), order_id="x")
    assert v["qty"] is None and v["price"] is None and v["notional"] is None


# --------------------------------------------------------------------------- #
# 2. Shadow NEVER blocks (the core safety property).
# --------------------------------------------------------------------------- #
def test_shadow_allow_proceeds():
    g = _gate(_shadow_cfg(), _usable_snap())
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.verdict == "ALLOW" and v.would_block is False
    assert g.should_block(v) is False


def test_shadow_block_verdict_still_never_blocks():
    """An over-sized order is a TRUE BLOCK, but shadow must not act on it."""
    g = _gate(_shadow_cfg(), _usable_snap(net=1_000_000.0))
    v = g.evaluate(_view(quantity=100_000.0, limit_price=50.0, notional_estimate=5_000_000.0), now_epoch=NOW)
    assert v.verdict == "BLOCK" and v.would_block is True
    assert g.should_block(v) is False       # shadow blocks nothing
    assert v.reason in {"AGGREGATE_EXPOSURE", "SINGLE_ORDER_CAP", "EXCESS_LIQUIDITY_FLOOR", "INIT_MARGIN_CEILING"}


def test_shadow_stale_snapshot_flags_staleness_but_proceeds():
    g = _gate(_shadow_cfg(), _stale_snap())
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.verdict == "BLOCK" and v.reason == "STALE_OR_MISSING_MARGIN_DATA"
    assert v.staleness is True and "account_snapshot" in v.staleness_detail
    assert g.should_block(v) is False


# --------------------------------------------------------------------------- #
# 3. Fail-open on internal error (shadow) — MARGIN_GATE_ERROR.
# --------------------------------------------------------------------------- #
def test_gate_error_fails_open_in_shadow(caplog):
    def boom(_ov):
        raise RuntimeError("boom")
    g = MarginShadowGate(_shadow_cfg(), snapshot_source=boom, evidence_path=None)
    with caplog.at_level("ERROR"):
        v = g.evaluate(_view(), now_epoch=NOW)
    assert v.evaluated is False and v.verdict == "ERROR" and v.reason == MARKER_ERROR
    assert v.would_block is False and g.should_block(v) is False
    assert any(MARKER_ERROR in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 4. Enforce path (injected config) — DOES block; fails CLOSED on error.
# --------------------------------------------------------------------------- #
def test_enforce_blocks_on_block_verdict():
    g = _gate(_enforce_cfg(), _stale_snap())
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.would_block is True
    assert g.should_block(v) is True        # enforce acts on the BLOCK


def test_enforce_allows_on_allow_verdict():
    g = _gate(_enforce_cfg(), _usable_snap())
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.verdict == "ALLOW"
    assert g.should_block(v) is False


def test_enforce_fails_closed_on_wiring_error():
    def boom(_ov):
        raise RuntimeError("boom")
    g = MarginShadowGate(_enforce_cfg(), snapshot_source=boom, evidence_path=None)
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.evaluated is False
    assert g.should_block(v) is True        # enforce fails CLOSED on wiring error


# --------------------------------------------------------------------------- #
# 5. Marker + evidence.
# --------------------------------------------------------------------------- #
def test_marker_line_is_single_grepable_line():
    g = _gate(_shadow_cfg(), _usable_snap())
    v = g.evaluate(_view(strategy="omega", symbol="TLT"), now_epoch=NOW)
    line = v.marker_line()
    assert "\n" not in line
    assert line.startswith(MARKER_SHADOW)
    for token in ("verdict=", "reason=", "symbol=TLT", "strategy=omega", "est_exposure=", "headroom=", "mode=shadow"):
        assert token in line


def test_evidence_ndjson_written_with_fields(tmp_path):
    ev = tmp_path / "margin_shadow"
    g = _gate(_shadow_cfg(), _usable_snap(), evidence_path=ev)
    g.evaluate(_view(), now_epoch=NOW)
    files = list(ev.glob("margin_shadow_*.ndjson"))
    assert len(files) == 1
    row = json.loads(files[0].read_text().strip())
    assert row["verdict"] == "ALLOW" and row["mode"] == "shadow"
    assert row["schema_version"] == "margin_shadow.v1"
    assert set(["ts_utc", "reason", "symbol", "est_exposure_cad", "staleness", "would_block"]) <= set(row)


def test_evidence_write_is_deterministic_in_now(tmp_path):
    ev1, ev2 = tmp_path / "a", tmp_path / "b"
    _gate(_shadow_cfg(), _usable_snap(), evidence_path=ev1).evaluate(_view(), now_epoch=NOW)
    _gate(_shadow_cfg(), _usable_snap(), evidence_path=ev2).evaluate(_view(), now_epoch=NOW)
    f1 = next(ev1.glob("*.ndjson")).read_bytes()
    f2 = next(ev2.glob("*.ndjson")).read_bytes()
    assert f1 == f2


def test_evidence_write_failure_never_raises(tmp_path):
    # Point evidence at a path whose parent is a FILE → mkdir fails; evaluate must still return.
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    g = _gate(_shadow_cfg(), _usable_snap(), evidence_path=blocker / "sub")
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.verdict == "ALLOW"      # no exception propagated


# --------------------------------------------------------------------------- #
# 6. Default factory — fail-open + honest stale default source.
# --------------------------------------------------------------------------- #
def test_build_default_gate_loads_shadow_config():
    g = build_default_shadow_gate(repo_root=REPO)
    assert g is not None and g.mode == "shadow"


def test_build_default_gate_failopen_on_bad_config(tmp_path, caplog):
    bad = tmp_path / "nope.json"
    with caplog.at_level("ERROR"):
        g = build_default_shadow_gate(repo_root=REPO, config_path=bad)
    assert g is None
    assert any(MARKER_ERROR in r.getMessage() for r in caplog.records)


def test_default_source_returns_failclosed_snapshot(tmp_path):
    """The real runtime source honestly reports margin fields are not published → stale block."""
    g = build_default_shadow_gate(repo_root=REPO)
    assert g is not None
    v = g.evaluate(_view(), now_epoch=NOW)
    assert v.reason == "STALE_OR_MISSING_MARGIN_DATA" and v.staleness is True


# --------------------------------------------------------------------------- #
# 7. Adapter chokepoint — shadow does NOT alter submission; enforce blocks pre-claim.
# --------------------------------------------------------------------------- #
def _intent(symbol="AAPL", qty=10.0, price=190.0):
    return StrategyTradeIntent(
        strategy="alpha", symbol=symbol, sec_type="STK", exchange="SMART", currency="USD",
        side="BUY", order_type="LMT", quantity=qty, notional_estimate=qty * price, limit_price=price,
    )


def test_adapter_shadow_gate_does_not_alter_dry_run_submission(tmp_path):
    gate = _gate(_shadow_cfg(), _usable_snap(net=1_000_000.0),
                 evidence_path=tmp_path / "ev")
    cfg = IbkrConfig(dry_run=True, state_db_path=tmp_path / "s.db")
    adapter = IbkrAdapter(config=cfg, margin_gate=gate)
    out = adapter.submit_strategy_trade_intents([_intent()])
    assert len(out) == 1 and out[0].status == "dry_run"      # submission outcome unchanged
    assert out[0].symbol == "AAPL"
    # And the gate DID run (evidence written).
    assert list((tmp_path / "ev").glob("*.ndjson"))


def test_adapter_no_gate_matches_gated_shadow_submission(tmp_path):
    """Byte-for-byte the same SubmittedOrder status/qty with the shadow gate present vs absent."""
    cfg_a = IbkrConfig(dry_run=True, state_db_path=tmp_path / "a.db")
    cfg_b = IbkrConfig(dry_run=True, state_db_path=tmp_path / "b.db")
    plain = IbkrAdapter(config=cfg_a).submit_strategy_trade_intents([_intent()])[0]
    gated = IbkrAdapter(
        config=cfg_b,
        margin_gate=_gate(_shadow_cfg(), _stale_snap(), evidence_path=tmp_path / "ev"),
    ).submit_strategy_trade_intents([_intent()])[0]
    assert plain.status == gated.status == "dry_run"
    assert plain.side == gated.side and plain.quantity == gated.quantity


def test_adapter_enforce_gate_blocks_before_claim_no_idempotency_row(tmp_path):
    db = tmp_path / "s.db"
    gate = _gate(_enforce_cfg(), _stale_snap(), evidence_path=tmp_path / "ev")  # enforce → STALE blocks
    cfg = IbkrConfig(dry_run=True, state_db_path=db, enable_idempotency=True)
    adapter = IbkrAdapter(config=cfg, margin_gate=gate)

    raw = _intent()
    out = adapter.submit_strategy_trade_intents([raw])
    assert len(out) == 1 and out[0].status == "margin_blocked"
    assert out[0].ib_order_id is None

    # No idempotency row was written for this intent (blocked BEFORE the claim).
    key = adapter._compute_idempotency_key(adapter._intent_from_trade_intent(raw))
    assert adapter._idempotency is not None
    assert adapter._idempotency.get(key) is None


def test_adapter_enforce_gate_allows_good_order_through(tmp_path):
    gate = _gate(_enforce_cfg(), _usable_snap(net=1_000_000.0), evidence_path=tmp_path / "ev")
    cfg = IbkrConfig(dry_run=True, state_db_path=tmp_path / "s.db", enable_idempotency=True)
    adapter = IbkrAdapter(config=cfg, margin_gate=gate)
    out = adapter.submit_strategy_trade_intents([_intent(qty=1.0, price=100.0)])
    assert out[0].status == "dry_run"       # small order passes the enforce gate
