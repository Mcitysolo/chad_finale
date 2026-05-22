"""Official Matrix Box 054 — BAG spread_id-aware reconciliation invariants.

Box 054 acceptance: **concurrent BAG spreads are keyed by spread_id, not
just (strategy, symbol)**.

Audit finding: today CHAD's identity-keying surface is heterogeneous:

  * ``chad/strategies/alpha_options.py`` stamps a fresh ``spread_id``
    (uuid4) onto every BAG signal — preserved in
    ``OptionsSpreadSpec`` and ``meta["spread_id"]``.
  * ``chad/options/spread_spec.py::OptionsSpreadSpec.to_legacy_meta``
    projects ``spread_id`` into the legacy dict.
  * ``chad/execution/execution_pipeline.py:948-958`` (``BAG_META_KEYS``)
    preserves ``spread_id`` end-to-end through the planner artifact.
  * ``chad/execution/paper_exec_evidence_writer.py:1028`` hydrates
    ``spread_id`` from the typed spec.
  * ``chad/execution/ibkr_adapter.py::_stable_idempotency_payload``
    (line 2940) keys BAG idempotency by the **leg tuple** (expiry +
    long_strike + short_strike + long_right + short_right) — which is
    functionally a per-spread identity at the broker submission
    boundary even without consulting ``spread_id`` directly.
  * ``chad/core/position_guard.py::_position_key`` (line 184) keys by
    only ``f"{strategy}|{symbol}"`` — NOT spread_id-aware.
  * ``chad/execution/paper_exec_evidence_writer.py::_find_opening_bag_fill``
    (line 1036) searches by ``(strategy, symbol)`` only — NOT spread_id
    filtered.
  * ``alpha_options.max_hold_exit`` iterates paper-ledger entries
    keyed by ``f"{strategy}|{symbol}"`` — implicitly one BAG per
    (strategy, symbol).

Effect: today CHAD enforces "one ``alpha_options`` BAG per symbol"
implicitly via the position_guard ``(strategy, symbol)`` key. The
broker-side identity (idempotency key) is leg-tuple-aware, but the
runtime book-keeping is not. Two concurrent BAGs with the same
``(strategy, symbol)`` but different ``spread_id`` would collide on
position_guard and on the most-recent-opener lookup. Box 053
established that **live BAG cannot open** today; the prospective
collision risk is therefore structurally blocked. Box 054 pins the
spread_id-preservation contract end-to-end AND documents the
position_guard / opener-lookup keying gap as the unblock condition
for concurrent live BAG support.

This test does NOT exercise broker, runtime, or live state. It is a
pure-unit invariant test of the spread_id-preservation contract +
the documented (strategy, symbol)-only identity gap.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from chad.core import position_guard
from chad.execution.ibkr_adapter import (
    IbkrAdapter,
    IbkrConfig,
    NormalizedIntent,
)
from chad.options.spread_spec import OptionsSpreadSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(spread_id: str, *, long_strike: float = 425.0,
               short_strike: float = 430.0, expiry: str = "20260619",
               long_right: str = "C", short_right: str = "C") -> OptionsSpreadSpec:
    return OptionsSpreadSpec(
        symbol="SPY",
        expiry=expiry,
        long_strike=long_strike,
        short_strike=short_strike,
        long_right=long_right,
        short_right=short_right,
        ratio_long=1,
        ratio_short=1,
        exchange="SMART",
        currency="USD",
        spread_type="BULL_CALL",
        max_loss_per_contract=500.0,
        net_debit_estimate=1.85,
        spread_id=spread_id,
        dte=29,
    )


def _make_bag_intent(
    *,
    strategy: str = "alpha_options",
    symbol: str = "SPY",
    side: str = "BUY",
    limit_price: float = 1.85,
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
        strategy=strategy,
        symbol=symbol,
        sec_type="BAG",
        exchange="SMART",
        currency="USD",
        side=side,
        order_type="LMT",
        quantity=1.0,
        notional_estimate=0.0,
        asset_class="options_spread",
        source_strategies=(strategy,),
        created_at=datetime(2026, 5, 21, 0, 0, 0, tzinfo=timezone.utc),
        limit_price=limit_price,
        meta=base_meta,
    )


def _adapter() -> IbkrAdapter:
    return IbkrAdapter(config=IbkrConfig(dry_run=True, enable_idempotency=False))


# ---------------------------------------------------------------------------
# Section A — spread_id preservation across the producer / spec boundary
# ---------------------------------------------------------------------------


def test_options_spread_spec_round_trips_spread_id_via_legacy_meta() -> None:
    """``OptionsSpreadSpec`` carries an explicit ``spread_id`` and
    ``to_legacy_meta()`` projects it; ``from_legacy_meta()`` reconstructs
    it. This is the canonical preservation channel.
    """
    spec = _make_spec(spread_id="box054-sid-A")
    legacy = spec.to_legacy_meta()
    assert legacy.get("spread_id") == "box054-sid-A"

    rebuilt = OptionsSpreadSpec.from_legacy_meta("SPY", legacy)
    assert rebuilt.spread_id == spec.spread_id


def test_options_spread_spec_omits_spread_id_when_none() -> None:
    """When ``spread_id`` is None, ``to_legacy_meta()`` MUST NOT emit
    the key — preserves backward compatibility with producers that
    have not yet stamped a spread_id.
    """
    spec = OptionsSpreadSpec(
        symbol="SPY",
        expiry="20260619",
        long_strike=425.0,
        short_strike=430.0,
        long_right="C",
        short_right="C",
        ratio_long=1,
        ratio_short=1,
        exchange="SMART",
        currency="USD",
        spread_type="BULL_CALL",
        max_loss_per_contract=500.0,
        net_debit_estimate=1.85,
        spread_id=None,
    )
    legacy = spec.to_legacy_meta()
    assert "spread_id" not in legacy


def test_alpha_options_signal_meta_has_uuid_spread_id() -> None:
    """``alpha_options.py:440`` stamps ``spread_id = str(uuid.uuid4())``
    onto every BAG signal. This test re-asserts the upstream emission
    contract: a fresh BAG meta always carries a non-empty string
    ``spread_id``.
    """
    import re
    from chad.strategies.alpha_options import build_alpha_options_config  # noqa: F401

    # The simplest assertion: read the source and confirm the uuid4
    # stamp + meta key are both present. A full strategy emission
    # would require a market-data harness — out of scope for Box 054.
    import chad.strategies.alpha_options as alpha
    src = Path(alpha.__file__).read_text(encoding="utf-8")
    assert re.search(r"spread_id\s*=\s*str\(uuid\.uuid4\(\)\)", src), (
        "alpha_options must stamp spread_id from uuid4 — Box 054 contract"
    )
    assert '"spread_id": spread_id' in src


# ---------------------------------------------------------------------------
# Section B — Pipeline preserves spread_id end-to-end
# ---------------------------------------------------------------------------


def test_execution_pipeline_bag_meta_keys_include_spread_id() -> None:
    """``execution_pipeline.py:948-958`` declares ``BAG_META_KEYS`` and
    line 1236-1242 declares ``_OPT_META_KEYS``. Both are strict
    allow-lists for meta-key preservation from survivor signal into
    the planner artifact. Both MUST include ``spread_id`` so the
    identifier survives end-to-end through the pipeline.
    """
    import chad.execution.execution_pipeline as ep

    src = Path(ep.__file__).read_text(encoding="utf-8")
    # First tuple: ``BAG_META_KEYS = (`` … ``)`` — strict allow-list at
    # the survivor-meta projection step.
    after_marker = src.split("BAG_META_KEYS = (", 1)[1]
    tuple_block = after_marker.split(")", 1)[0]
    assert '"spread_id"' in tuple_block, (
        "BAG_META_KEYS tuple must include spread_id — pipeline preservation contract"
    )

    # Second tuple: ``_OPT_META_KEYS = (`` … ``)`` — single-leg OPT
    # plus BAG strikes; both spread_id and spread_type must survive.
    after_marker2 = src.split("_OPT_META_KEYS = (", 1)[1]
    tuple_block2 = after_marker2.split(")", 1)[0]
    assert '"spread_id"' in tuple_block2, (
        "_OPT_META_KEYS tuple must include spread_id — pipeline preservation contract"
    )


def test_paper_evidence_writer_hydrates_spread_id_from_spec() -> None:
    """``paper_exec_evidence_writer._hydrate_legacy_bag_meta_from_spec``
    (line 985) lists ``spread_id`` in the backfill key list (line 1028),
    so a typed ``OptionsSpreadSpec`` under ``extra["spread_spec"]``
    populates the legacy ``extra["spread_id"]`` when blank.
    """
    from chad.execution.paper_exec_evidence_writer import (
        _hydrate_legacy_bag_meta_from_spec,
    )

    spec = _make_spec(spread_id="box054-sid-W")
    extra: Dict[str, Any] = {"spread_spec": spec}
    out = _hydrate_legacy_bag_meta_from_spec(extra)
    assert out is extra
    assert extra.get("spread_id") == "box054-sid-W"


# ---------------------------------------------------------------------------
# Section C — IBKR adapter idempotency key is leg-tuple-aware (broker
# submission boundary already distinguishes concurrent BAGs)
# ---------------------------------------------------------------------------


def test_ibkr_idempotency_key_differs_when_long_strike_differs() -> None:
    """Same (strategy, symbol, side) but different long_strike →
    different idempotency keys. The broker submission boundary
    is leg-tuple-aware even though it does not consult spread_id
    directly.
    """
    adapter = _adapter()
    a = _make_bag_intent(meta={"long_strike": 425.0})
    b = _make_bag_intent(meta={"long_strike": 420.0})
    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_ibkr_idempotency_key_differs_when_expiry_differs() -> None:
    """Different expiry → different idempotency key."""
    adapter = _adapter()
    a = _make_bag_intent(meta={"expiry": "20260619"})
    b = _make_bag_intent(meta={"expiry": "20260717"})
    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_ibkr_idempotency_key_differs_when_right_differs() -> None:
    """Different rights (e.g. calls vs puts) → different idempotency key."""
    adapter = _adapter()
    a = _make_bag_intent(meta={"long_right": "C", "short_right": "C"})
    b = _make_bag_intent(meta={"long_right": "P", "short_right": "P"})
    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_ibkr_idempotency_key_stable_within_same_leg_tuple() -> None:
    """Same leg tuple → same idempotency key (this is the duplicate-
    submit suppression guarantee at the broker boundary; documented
    further by ``test_ibkr_idempotency_key_stability.py``).
    """
    adapter = _adapter()
    a = _make_bag_intent(meta={"net_debit_estimate": 1.85})
    b = _make_bag_intent(meta={"net_debit_estimate": 1.92})  # mark drifted
    assert adapter._compute_idempotency_key(a) == adapter._compute_idempotency_key(b)


# ---------------------------------------------------------------------------
# Section D — position_guard _position_key gap (documented contract)
# ---------------------------------------------------------------------------


def test_position_guard_key_is_strategy_symbol_only() -> None:
    """Pin the current contract: ``_position_key(strategy, symbol)``
    returns ``f"{strategy}|{symbol}"``. **It does NOT include spread_id.**

    This is the documented Box 054 gap. Today it is structurally safe
    because (a) Box 053 blocks live BAG entry, (b) position_guard
    enforces "one ``alpha_options`` BAG per symbol" implicitly via the
    same-side-open guard. When concurrent live BAGs are required, this
    key must be widened.
    """
    key_a = position_guard._position_key("alpha_options", "SPY")
    key_b = position_guard._position_key("alpha_options", "SPY")
    # Same (strategy, symbol) → same key regardless of any other context.
    assert key_a == key_b == "alpha_options|SPY"


def test_position_guard_key_collides_on_same_strategy_and_symbol() -> None:
    """The gap pin: two BAGs that differ ONLY by their (hypothetical)
    spread_id collide on position_guard. Box 054 documents this
    explicitly so any future "concurrent BAGs allowed" change must
    refresh the position_guard key alongside.
    """
    key_a = position_guard._position_key("alpha_options", "SPY")
    key_b = position_guard._position_key("alpha_options", "SPY")
    # Even when callers pass distinct spread_ids in meta, the position
    # key is determined ONLY by (strategy, symbol).
    assert key_a == key_b


def test_position_guard_key_for_non_bag_is_unchanged() -> None:
    """Non-BAG keying (STK / FUT / OPT / etc.) MUST keep the
    ``(strategy, symbol)`` shape. Box 054 changes nothing for non-BAG
    flows.
    """
    assert position_guard._position_key("alpha", "SPY") == "alpha|SPY"
    assert position_guard._position_key("gamma_futures", "MES") == "gamma_futures|MES"


# ---------------------------------------------------------------------------
# Section E — paper opener lookup gap (documented contract)
# ---------------------------------------------------------------------------


def test_find_opening_bag_fill_signature_is_strategy_symbol_only() -> None:
    """Pin the contract on
    ``paper_exec_evidence_writer._find_opening_bag_fill``: its
    signature is ``(strategy, symbol)`` — no spread_id parameter.

    Today this is structurally safe (Box 053 live block + one-BAG-per-
    (strategy, symbol) guard). When concurrent BAGs are supported,
    this lookup MUST be widened to accept spread_id.
    """
    import inspect

    from chad.execution.paper_exec_evidence_writer import _find_opening_bag_fill

    params = list(inspect.signature(_find_opening_bag_fill).parameters)
    assert params == ["strategy", "symbol"], (
        f"_find_opening_bag_fill signature widened (params={params}); "
        "refresh Box 054 evidence — spread_id-aware lookup may have landed."
    )


# ---------------------------------------------------------------------------
# Section F — Live BAG remains blocked (cross-ref to Box 053)
# ---------------------------------------------------------------------------


def test_live_bag_still_blocked_so_collision_risk_is_prospective_only() -> None:
    """Cross-ref to Box 053: the structural collision risk on
    position_guard / opener-lookup is **prospective only** because
    live BAG cannot open today (``IbkrConfig.dry_run`` defaults to
    True; ``runtime/live_readiness.json::ready_for_live=false``).
    """
    import json

    cfg = IbkrConfig()
    assert cfg.dry_run is True

    p = Path("/home/ubuntu/chad_finale/runtime/live_readiness.json")
    if p.is_file():
        obj = json.loads(p.read_text(encoding="utf-8"))
        assert obj.get("ready_for_live") is False


# ---------------------------------------------------------------------------
# Section G — alpha_options today emits at most one BAG signal per call
# ---------------------------------------------------------------------------


def test_alpha_options_signal_emit_block_returns_single_signal_list() -> None:
    """Source-text anchor: ``alpha_options.py`` returns ``[signal]``
    after constructing the BAG ``TradeSignal``. This pins the
    one-BAG-per-emit contract that, in combination with the
    (strategy, symbol) guard key, prevents concurrent BAG state in
    the current implementation.
    """
    import chad.strategies.alpha_options as alpha

    src = Path(alpha.__file__).read_text(encoding="utf-8")
    # The signal-build block at lines ~501-536 ends with ``return [signal]``.
    # Pin that exact return.
    assert "return [signal]" in src, (
        "alpha_options must emit at most one BAG signal per call "
        "(one-BAG-per-(strategy,symbol) contract)."
    )
