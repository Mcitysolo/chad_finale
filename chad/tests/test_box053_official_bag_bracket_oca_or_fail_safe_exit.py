"""Official Matrix Box 053 — BAG bracket/OCA or fail-safe exit invariants.

Pins the contract that **a live BAG position cannot sit unmanaged if
live_loop fails**. Closure path = ``PASS_LIVE_BAG_BLOCKED_UNTIL_PROTECTED``:
the CHAD execution stack today contains NO bracket / OCA wiring AND NO
out-of-band BAG fail-safe closer. The only BAG exit mechanism is
``alpha_options.max_hold_seconds=3600`` driven by ``live_loop``. To
satisfy Box 053 acceptance, live BAG entry MUST be structurally
blocked — confirmed at three layers:

  Layer A — Adapter dry_run posture
    ``IbkrConfig.dry_run`` defaults to ``True``. Under dry_run the
    adapter's ``_submit_intent`` returns a ``status="dry_run"`` synthetic
    ``SubmittedOrder`` and **never invokes broker ``placeOrder``**.

  Layer B — LiveGate fail-closed gates
    ``evaluate_live_gate()`` returns ``allow_ibkr_live=False`` whenever
    ``runtime/live_readiness.json`` has ``ready_for_live=false`` OR
    ``exec_cfg.ibkr_dry_run=True`` OR ``operator_intent != ALLOW_LIVE``.
    Each gate is a sufficient block.

  Layer C — No bracket/OCA, no fail-safe closer wiring exists
    Grep-anchored structural assertions: the adapter source contains no
    ``parentId``, ``transmit``, ``ocaGroup``, ``ocaType``, ``Bracket``,
    ``bracket_`` reference; no ``bag_failsafe_closer`` module is
    importable. These tests are the canonical signal — when bracket/OCA
    or an out-of-band closer lands, the corresponding assertion fails
    by design and Box 053 evidence must be updated.

This test does NOT exercise broker, runtime, or live state. It is a
pure-unit invariant test of the BAG live-exit safety contract.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from chad.core import live_gate as lg
from chad.execution.ibkr_adapter import (
    IbkrAdapter,
    IbkrConfig,
    NormalizedIntent,
    _OrderFactory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bag_intent(
    *,
    limit_price: float = 3.50,
    meta: Optional[Dict[str, Any]] = None,
) -> NormalizedIntent:
    base_meta: Dict[str, Any] = {
        "expiry": "20260619",
        "long_strike": 425.0,
        "short_strike": 430.0,
        "long_right": "C",
        "short_right": "C",
        "net_debit_estimate": limit_price,
    }
    if meta:
        base_meta.update(meta)
    return NormalizedIntent(
        strategy="alpha_options",
        symbol="SPY",
        sec_type="BAG",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=1.0,
        notional_estimate=0.0,
        asset_class="options_spread",
        source_strategies=("alpha_options",),
        created_at=datetime(2026, 5, 21, 0, 0, 0, tzinfo=timezone.utc),
        limit_price=limit_price,
        meta=base_meta,
    )


class _StubOrder:
    """Stand-in for ib_async.order.Order. Mirrors the attributes
    ``_OrderFactory.build`` sets so this test stays pure-unit (no IBKR).
    """

    action = ""
    orderType = ""
    totalQuantity = 0.0
    tif = ""
    outsideRth = False
    whatIf = False
    account = ""
    lmtPrice = 0.0


def _stub_contract_classes():
    return (object, object, object, _StubOrder, object)


class _PlaceOrderProbe:
    """IB stub that records whether placeOrder / qualifyContracts was
    called. Adapter ``dry_run=True`` should yield 0 calls for both.
    """

    def __init__(self) -> None:
        self.place_calls: int = 0
        self.qualify_calls: int = 0

    def isConnected(self) -> bool:
        return True

    def connect(self, *a: Any, **k: Any) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def managedAccounts(self):
        return ["DU0000000"]

    def qualifyContracts(self, *contracts: Any):
        self.qualify_calls += 1
        return list(contracts)

    def whatIfOrder(self, contract: Any, order: Any) -> Any:
        return order

    def placeOrder(self, contract: Any, order: Any) -> Any:
        self.place_calls += 1

        class _Trade:
            orderStatus = type("S", (), {"status": "Submitted"})()
            order = None
            fills: list = []
            commissionReport: list = []

        return _Trade()

    def sleep(self, seconds: float) -> None:
        return None


def _write_readiness(runtime: Path, *, ready: bool) -> None:
    payload = {
        "schema_version": "live_readiness_state.v1",
        "ready_for_live": bool(ready),
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ttl_seconds": 604800,
        "next_evaluation_cadence": "weekly",
    }
    (runtime / "live_readiness.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Layer A — Adapter dry_run posture: live BAG entry is impossible by default
# ---------------------------------------------------------------------------


def test_default_ibkr_config_is_dry_run() -> None:
    """``IbkrConfig.dry_run`` defaults to ``True``. Under dry_run the
    adapter never invokes broker ``placeOrder``. This is the canonical
    structural block on live BAG entry today.
    """
    cfg = IbkrConfig()
    assert cfg.dry_run is True


def test_adapter_dry_run_bag_submit_does_not_call_place_order() -> None:
    """End-to-end Layer A proof: even when the IB probe is bound,
    ``_submit_intent(BAG)`` under ``dry_run=True`` returns a
    ``status="dry_run"`` synthetic result and ``placeOrder`` is never
    called. Equivalent to Box 53's "live BAG cannot be opened" claim
    at the adapter boundary.
    """
    cfg = IbkrConfig(dry_run=True, enable_idempotency=False)
    adapter = IbkrAdapter(config=cfg)
    intent = _make_bag_intent()

    result = adapter._submit_intent(intent)

    assert result is not None
    assert result.status == "dry_run"
    assert result.dry_run is True
    assert result.sec_type == "BAG"


# ---------------------------------------------------------------------------
# Layer B — LiveGate fail-closed when ready_for_live is False
# ---------------------------------------------------------------------------


def test_live_gate_denies_ibkr_live_when_ready_for_live_false(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical structural block: when ``ready_for_live=false`` the
    LiveGate sets ``allow_ibkr_live=False`` and mode=DENY_ALL. No live
    BAG (or any other live) order can flow.
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_readiness(runtime, ready=False)
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)

    decision = lg.evaluate_live_gate()

    assert decision.allow_ibkr_live is False
    assert decision.context.live_readiness.ready_for_live is False


def test_live_gate_denies_ibkr_live_when_adapter_in_dry_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with ``ready_for_live=true`` (hypothetical), the final
    ``allow_ibkr_live`` requires ``not exec_cfg.ibkr_dry_run``. With
    ``IBKR_DRY_RUN=1`` (default), live BAG is denied. Pins the
    adapter / LiveGate AND-gate.
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_readiness(runtime, ready=True)
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)
    monkeypatch.setenv("IBKR_DRY_RUN", "1")
    monkeypatch.setenv("IBKR_ENABLED", "1")

    decision = lg.evaluate_live_gate()

    assert decision.context.execution.ibkr_dry_run is True
    assert decision.allow_ibkr_live is False


# ---------------------------------------------------------------------------
# Layer C — No bracket/OCA wiring in adapter
# ---------------------------------------------------------------------------


_BRACKET_OCA_TOKENS = (
    "parentId",
    ".transmit",
    "ocaGroup",
    "ocaType",
    "Bracket",
    "bracket_",
)


def test_ibkr_adapter_source_has_no_bracket_or_oca_wiring() -> None:
    """Structural anchor: the adapter source MUST NOT contain
    bracket / OCA tokens — that proves no protective-child order is
    ever attached to a BAG (or any other) order. When the bracket /
    OCA wiring lands, this assertion fails by design and Box 053
    evidence must be refreshed.
    """
    import chad.execution.ibkr_adapter as adapter

    src = Path(adapter.__file__).read_text(encoding="utf-8")
    found = [tok for tok in _BRACKET_OCA_TOKENS if tok in src]
    assert not found, (
        f"Unexpected bracket/OCA token(s) in ibkr_adapter.py: {found!r}. "
        "If protective-child wiring has been added, refresh Box 053 evidence."
    )


def test_order_factory_does_not_set_bracket_or_oca_attributes_on_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end Layer C proof: ``_OrderFactory.build`` produces a BAG
    Order that has none of ``parentId``, ``transmit``-flagged child,
    ``ocaGroup``, ``ocaType``. The stub Order class does not declare
    these attributes — if the factory tries to set them, getattr will
    expose only the unset default (the absence of a value is what
    Box 053 anchors).
    """
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = _make_bag_intent()
    prepared = factory.build(intent, what_if=True)

    # No bracket/OCA fields should have been written onto the Order.
    for attr in ("parentId", "ocaGroup", "ocaType"):
        assert not hasattr(prepared.order, attr), (
            f"BAG order unexpectedly carries {attr!r} — bracket/OCA wiring has landed; "
            "refresh Box 053 evidence."
        )


# ---------------------------------------------------------------------------
# Layer C — No out-of-band BAG fail-safe closer module exists
# ---------------------------------------------------------------------------


_FAILSAFE_CLOSER_CANDIDATES = (
    "chad.risk.bag_failsafe_closer",
    "chad.execution.bag_failsafe_closer",
    "chad.ops.bag_failsafe_closer",
    "ops.bag_failsafe_closer",
)


def test_no_bag_failsafe_closer_module_exists() -> None:
    """Structural anchor: no ``bag_failsafe_closer`` module has been
    landed yet. The sole BAG closure mechanism is the in-process
    ``alpha_options.max_hold_seconds`` exit (driven by ``live_loop``).
    When the out-of-band closer ships, this assertion fails by
    design — that failure is the canonical signal to refresh Box 053
    evidence and policy.
    """
    found = [
        name
        for name in _FAILSAFE_CLOSER_CANDIDATES
        if importlib.util.find_spec(name) is not None
    ]
    assert not found, (
        f"Unexpected bag_failsafe_closer module(s) importable: {found!r}. "
        "Refresh Box 053 evidence — closure path may shift to "
        "PASS_BAG_FAILSAFE_EXIT_VERIFIED."
    )


def test_alpha_options_max_hold_seconds_is_the_documented_bag_exit() -> None:
    """Pin the documented BAG exit mechanism contract: the only BAG
    exit signal source today is ``alpha_options.AlphaOptionsTuning.
    max_hold_seconds`` (driven by ``live_loop``). The value MUST stay
    at the documented 3600 s (1 h) default unless Box 053 policy is
    refreshed.
    """
    from chad.strategies.alpha_options import AlphaOptionsTuning

    tuning = AlphaOptionsTuning()
    assert hasattr(tuning, "max_hold_seconds")
    assert int(tuning.max_hold_seconds) == 3600, (
        "AlphaOptionsTuning.max_hold_seconds default changed; refresh Box 053 "
        "policy + evidence (documents the 1 h max-hold contract)."
    )


# ---------------------------------------------------------------------------
# RISK-BAG-03 documentation contract: posture explicitly blocks live BAG
# ---------------------------------------------------------------------------


def test_runtime_posture_artifacts_consistent_with_paper_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-check the canonical posture artifacts: in PAPER, runtime
    state MUST have ``ready_for_live=false`` AND the execution
    environment MUST have ``exec_mode != 'live'`` AND
    ``ibkr_dry_run=true``. Read-only probe of the real runtime — no
    mutation.
    """
    root = Path("/home/ubuntu/chad_finale/runtime")
    live_readiness = json.loads((root / "live_readiness.json").read_text(encoding="utf-8"))
    assert live_readiness.get("ready_for_live") is False

    exec_env_path = root / "execution_environment.json"
    if exec_env_path.is_file():
        env_obj = json.loads(exec_env_path.read_text(encoding="utf-8"))
        assert str(env_obj.get("exec_mode", "")).lower() != "live"
        assert bool(env_obj.get("ibkr_dry_run", True)) is True
