"""G3C U2 — integration: N intents through the FULL submit path with the margin gate.

Drives ``IbkrAdapter.submit_strategy_trade_intents`` end-to-end (dry-run, no live broker —
the "scripted fake broker" is the injected account snapshot + the intent sizing that scripts
ALLOW vs BLOCK):

  * SHADOW: every intent submits (none blocked), the gate runs for EVERY intent (N evidence
    rows + N markers), the idempotency claim is written for all (proving shadow never
    short-circuits the claim), and the would-be-blocked over-sized intent is only *logged*.
  * ENFORCE (injected config, frozen file untouched): the over-exposure intent is blocked
    BEFORE the idempotency claim (status margin_blocked, no idempotency row), while the
    in-budget intents submit normally.

No live broker, no network, no runtime writes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig
from chad.execution.ibkr_executor import StrategyTradeIntent
from chad.execution.margin_shadow_gate import MARKER_SHADOW, MarginShadowGate
from chad.risk.buying_power_provider import BuyingPowerSnapshot
from chad.risk.margin_block import load_frozen_config

REPO = Path("/home/ubuntu/chad_finale")
NOW = 1_800_000_000.0


def _cfg(mode: str):
    base = load_frozen_config(REPO / "config" / "margin_block.json")
    return base if mode == "shadow" else dataclasses.replace(base, mode=mode)


def _snap():
    # NetLiq 1M CAD → single-order cap = 10% = 100k CAD; excess 800k, init 50k.
    return BuyingPowerSnapshot(
        usable=True, reason="OK", detail="", currency="CAD",
        captured_at_epoch=NOW, age_seconds=1.0, ttl_seconds=30.0,
        net_liquidation=1_000_000.0, buying_power=2_000_000.0, excess_liquidity=800_000.0,
        available_funds=800_000.0, full_init_margin_req=50_000.0, full_maint_margin_req=40_000.0,
    )


def _gate(mode: str, evidence_dir: Path):
    return MarginShadowGate(
        _cfg(mode),
        snapshot_source=lambda ov: (_snap(), {}),
        open_orders_source=lambda: [],
        evidence_path=evidence_dir,
    )


def _intent(symbol: str, notional_usd: float):
    # notional_usd is the order notional the gate reads; qty/price cosmetic.
    qty = 10.0
    return StrategyTradeIntent(
        strategy=f"strat_{symbol.lower()}", symbol=symbol, sec_type="STK", exchange="SMART",
        currency="USD", side="BUY", order_type="LMT", quantity=qty,
        notional_estimate=notional_usd, limit_price=notional_usd / qty,
    )


# Two in-budget intents ($1k USD ≈ $1.5k CAD < 100k cap) + one over-exposure
# ($100k USD ≈ $148.7k CAD > 100k cap → SINGLE_ORDER_CAP).
def _three_intents():
    return [
        _intent("SPY", 1_000.0),
        _intent("BIGCO", 100_000.0),   # over-exposure
        _intent("TLT", 2_000.0),
    ]


def _evidence_rows(evidence_dir: Path):
    rows = []
    for f in sorted(evidence_dir.glob("margin_shadow_*.ndjson")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_shadow_all_intents_submit_and_are_all_evaluated(tmp_path, caplog):
    ev = tmp_path / "ev"
    adapter = IbkrAdapter(
        config=IbkrConfig(dry_run=True, state_db_path=tmp_path / "s.db", enable_idempotency=True),
        margin_gate=_gate("shadow", ev),
    )
    intents = _three_intents()
    with caplog.at_level("INFO"):
        out = adapter.submit_strategy_trade_intents(intents)

    # Every intent SUBMITTED — shadow blocks nothing, including the over-exposure one.
    assert len(out) == 3
    assert all(so.status == "dry_run" for so in out)
    assert [so.symbol for so in out] == ["SPY", "BIGCO", "TLT"]

    # The gate ran for EVERY intent → one evidence row each.
    rows = _evidence_rows(ev)
    assert len(rows) == 3
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["SPY"]["verdict"] == "ALLOW"
    assert by_symbol["TLT"]["verdict"] == "ALLOW"
    assert by_symbol["BIGCO"]["verdict"] == "BLOCK"           # would block...
    assert by_symbol["BIGCO"]["would_block"] is True
    assert by_symbol["BIGCO"]["reason"] == "SINGLE_ORDER_CAP"
    assert all(r["mode"] == "shadow" for r in rows)

    # A MARGIN_SHADOW marker was emitted for each evaluation.
    markers = [r for r in caplog.records if r.getMessage().startswith(MARKER_SHADOW)]
    assert len(markers) == 3

    # Shadow never short-circuits the claim: all three have an idempotency row.
    for raw in intents:
        key = adapter._compute_idempotency_key(adapter._intent_from_trade_intent(raw))
        assert adapter._idempotency.get(key) is not None


def test_enforce_blocks_over_exposure_before_claim_only(tmp_path):
    ev = tmp_path / "ev"
    adapter = IbkrAdapter(
        config=IbkrConfig(dry_run=True, state_db_path=tmp_path / "s.db", enable_idempotency=True),
        margin_gate=_gate("enforce_paper", ev),
    )
    intents = _three_intents()
    out = adapter.submit_strategy_trade_intents(intents)
    status = {so.symbol: so.status for so in out}

    # In-budget intents submit; the over-exposure one is BLOCKED.
    assert status["SPY"] == "dry_run"
    assert status["TLT"] == "dry_run"
    assert status["BIGCO"] == "margin_blocked"

    # The blocked intent wrote NO idempotency row (blocked before the claim);
    # the allowed intents DID claim.
    keys = {
        raw.symbol: adapter._compute_idempotency_key(adapter._intent_from_trade_intent(raw))
        for raw in intents
    }
    assert adapter._idempotency.get(keys["BIGCO"]) is None
    assert adapter._idempotency.get(keys["SPY"]) is not None
    assert adapter._idempotency.get(keys["TLT"]) is not None


def test_enforce_blocked_order_can_be_resubmitted_after_headroom(tmp_path):
    """Because the block wrote no idempotency row, a later (smaller) resubmit is not treated
    as a duplicate — proving the pre-claim block leaves no residue."""
    ev = tmp_path / "ev"
    adapter = IbkrAdapter(
        config=IbkrConfig(dry_run=True, state_db_path=tmp_path / "s.db", enable_idempotency=True),
        margin_gate=_gate("enforce_paper", ev),
    )
    blocked = adapter.submit_strategy_trade_intents([_intent("BIGCO", 100_000.0)])
    assert blocked[0].status == "margin_blocked"
    # Same symbol, now in-budget → submits cleanly (not duplicate_blocked).
    ok = adapter.submit_strategy_trade_intents([_intent("BIGCO", 1_000.0)])
    assert ok[0].status == "dry_run"
