"""U2 CRYPTO-2 — Kraken margin/BP shadow gate at the submit chokepoint.

Mirrors test_margin_shadow_gate.py:325-358: (a) behavioral spy-gate proving
every execute_with_risk evaluates the gate; (b) source-inspection proving the
gate precedes store.claim + router.execute (no submit path bypasses it); plus
an end-to-end shadow run over a real KrakenBuyingPowerSnapshot that blocks
nothing and writes evidence to tmp_path.
"""

from __future__ import annotations

import inspect
import json
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

from chad.execution import kraken_executor as ke
from chad.execution.kraken_executor import (
    ExecStateClaim,
    KrakenExecutor,
    RiskGateResult,
    StrategyTradeIntent,
)
from chad.execution.kraken_margin_gate import (
    build_default_kraken_shadow_gate,
    kraken_order_view_from_intent,
)
from chad.execution.kraken_trade_router import TradeResponse

_NOW = 1_800_000_000.0


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _intent(side="buy", volume=0.1, pair="SOLUSD"):
    return StrategyTradeIntent(
        strategy="alpha_crypto", pair=pair, side=side, ordertype="market",
        volume=volume, notional_estimate=volume * 170.0, price=170.0,
    )


class _SpyGate:
    def __init__(self):
        self.calls = []
        self.config = types.SimpleNamespace(is_enforce=False)

    def evaluate(self, order_view, *, now_epoch):
        self.calls.append(order_view)
        return {"evaluated": True}

    def should_block(self, _verdict):
        return False


def _executor_with_spy(spy):
    router = MagicMock()
    router.execute.return_value = TradeResponse(txids=[], raw={})
    ex = KrakenExecutor(router=router, margin_gate=spy)
    # Neutralize downstream caps/store so we isolate the gate invariant.
    ex._caps_path = None
    store = MagicMock()
    store.claim.return_value = ExecStateClaim(
        inserted=True, status="NEW", broker_order_id=None,
        submit_attempts=0, claim_attempts=1,
    )
    store.bump_submit_attempt.return_value = 1
    ex._store = store
    return ex, router


def test_ci_invariant_every_execute_with_risk_evaluates_gate(monkeypatch):
    spy = _SpyGate()
    ex, router = _executor_with_spy(spy)
    monkeypatch.setattr(ke, "load_dynamic_caps", lambda _p: {})
    monkeypatch.setattr(
        ke, "check_risk",
        lambda *, caps_data, intent: types.SimpleNamespace(
            allowed=True, reason="ok", adjusted_notional=intent.notional_estimate
        ),
    )
    intents = [_intent("buy"), _intent("sell"), _intent("buy", pair="ETHUSD")]
    for it in intents:                 # validate path
        ex.execute_with_risk(it, live=False)
    ex.execute_with_risk(intents[0], live=True)  # live path (store mocked)
    assert len(spy.calls) == len(intents) + 1
    assert {c["asset_class"] for c in spy.calls} == {"crypto"}


def test_ci_invariant_gate_evaluated_even_when_risk_denies(monkeypatch):
    spy = _SpyGate()
    ex, router = _executor_with_spy(spy)
    monkeypatch.setattr(ke, "load_dynamic_caps", lambda _p: {})
    monkeypatch.setattr(
        ke, "check_risk",
        lambda *, caps_data, intent: types.SimpleNamespace(
            allowed=False, reason="cap", adjusted_notional=0.0
        ),
    )
    rr, resp = ex.execute_with_risk(_intent("buy"), live=False)
    assert rr.allowed is False          # risk denied downstream
    assert len(spy.calls) == 1          # ...but the gate STILL ran first


def test_ci_invariant_gate_precedes_claim_and_router_in_source():
    src = inspect.getsource(KrakenExecutor.execute_with_risk)
    gate_pos = src.index("_evaluate_margin_gate(")
    claim_pos = src.index("self._store.claim(")
    router_pos = src.index("self._router.execute(")
    assert gate_pos < claim_pos, "margin gate must precede the idempotency claim"
    assert gate_pos < router_pos, "margin gate must precede router.execute"


def test_shadow_never_blocks_and_writes_evidence(tmp_path):
    balances = tmp_path / "kraken_balances.json"
    balances.write_text(json.dumps({
        "ts_utc": _iso(_NOW), "ok": True,
        "balances": {"CAD": 252.85}, "usd_equivalent": 184.58,
        "cad_equivalent": 252.85, "error": None,
    }), encoding="utf-8")
    ev_dir = tmp_path / "margin_shadow"

    gate = build_default_kraken_shadow_gate(
        balances_path=balances, evidence_dir=ev_dir,
        now_fn=lambda: _NOW,
    )
    assert gate is not None
    ov = kraken_order_view_from_intent(_intent("buy", volume=0.1), order_id="X")
    verdict = gate.evaluate(ov, now_epoch=_NOW)
    # SHADOW: must never block regardless of the verdict.
    assert gate.should_block(verdict) is False
    # Evidence written to tmp (data/margin_shadow is write-guarded; we used tmp).
    files = list(ev_dir.glob("margin_shadow_*.ndjson"))
    assert files, "shadow gate must write an evidence row"
    rows = [json.loads(ln) for ln in files[0].read_text().splitlines() if ln.strip()]
    assert rows and rows[-1].get("symbol") == "SOL-USD"


def test_build_default_kraken_gate_fail_open_on_bad_config(tmp_path):
    bad = tmp_path / "margin_block.json"
    bad.write_text("{ not json", encoding="utf-8")
    gate = build_default_kraken_shadow_gate(config_path=bad, evidence_dir=tmp_path)
    assert gate is None  # fail-open: broken config never wires a blocker
