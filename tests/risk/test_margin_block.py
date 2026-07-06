"""Tests for chad/risk/margin_block.py (Margin BLOCK Phase B, the ALLOW/BLOCK gate).

Fixtures only — no network, no broker, injected time (deterministic). Each named
adversarial case from the design (docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md §2.2 /
Part 4 / Part 8) is explicit:

  * burst-of-increasing cannot exceed the aggregate cap (each order counts prior
    ledger reservations);
  * "reduce-burst cannot flip position" (3× SELL-100 vs long 100) — #1 allows +
    reserves(reducing), #2/#3 hit remainder=0 → full checks → BLOCK, and the
    test FAILS IF THE REDUCER DID NOT RESERVE (an explicit un-reserved contrast);
  * over-leveraged + stale feed + flatten → ALLOW up to the remainder, BLOCK
    beyond it (an exit is never trapped, but is bounded);
  * "same-orderId re-place is BLOCKED" (the modify ban, branch (a), FIRST — it
    beats the reduce branch);
  * every fail-closed path (stale / missing / incomplete / uncomputable /
    unreadable) → BLOCK for an increasing order;
  * aggregate uses max(N×NetLiq, current): an already-breached book can still
    REDUCE, never INCREASE;
  * max(whatIf, independent) — the conservative (larger) governs, each direction;
  * strict config: a typo'd / missing / wrong-type key → refuse to start;
  * SHADOW: an internal exception logs + allows but STILL reserves; ENFORCE:
    blocks (INTERNAL_ERROR);
  * crypto orders route to their own lane (kraken notional-vs-balance);
  * isolation: the module imports NO live/execution/broker path and never
    submits an order.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT
from chad.risk.kraken_bp_provider import parse_kraken_balances
from chad.risk.margin_block import (
    AllowOrBlock,
    MarginBlockConfig,
    MarginBlockConfigError,
    MarginBlockReason,
    decide,
    load_frozen_config,
)
from chad.risk.pending_exposure_ledger import PendingExposureLedger

_T0 = 1_000.0  # injected base epoch
_FROZEN_CONFIG_PATH = "config/margin_block.json"

# The frozen thresholds (Part 3) as test defaults, so decision fixtures are
# hermetic (no temp files) and each case overrides only what it exercises.
_DEFAULTS = dict(
    account_size_basis="NetLiquidation",
    schema_version="1.0.0",
    aggregate_gross_exposure_max_mult_netliq=2.0,
    excess_liquidity_floor_pct_netliq=25.0,
    init_margin_ceiling_pct_netliq=50.0,
    single_order_notional_max_pct_netliq=10.0,
    whatif_vs_independent_log_gap_tolerance_pct=20.0,
    margin_rates_by_asset_class={
        "equity_regt": 0.50,
        "crypto_unlevered": 1.00,
        "futures_blanket_conservative": 0.25,
    },
    account_data_ttl_seconds=30.0,
    fx_conservative_bias_mult=1.05,
)


def _cfg(mode: str = "shadow", **overrides: Any) -> MarginBlockConfig:
    kw = dict(_DEFAULTS)
    kw.update(overrides)
    return MarginBlockConfig(mode=mode, **kw)


@dataclass
class _Snap:
    """A duck-typed BuyingPowerSnapshot stand-in (IBKR lane)."""

    usable: bool = True
    net_liquidation: Optional[float] = 100_000.0
    excess_liquidity: Optional[float] = 100_000.0
    full_init_margin_req: Optional[float] = 0.0
    reason: str = "OK"


@dataclass
class _Kraken:
    """A duck-typed KrakenBuyingPowerSnapshot stand-in (crypto lane)."""

    usable: bool = True
    available_cad: Optional[float] = 5_000.0
    reason: str = "OK"


class _BoomPositions:
    """A positions object whose lookup raises — an injected INTERNAL error."""

    def get(self, _symbol: str) -> Any:
        raise RuntimeError("positions backend unavailable")


def _order(order_id: str, symbol: str, side: str, qty: float, *, price: float = 100.0,
           currency: str = "CAD", asset_class: str = "equity",
           whatif_init_margin: Optional[float] = None, notional: Optional[float] = None,
           multiplier: Optional[float] = None) -> dict:
    o = {
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "currency": currency,
        "asset_class": asset_class,
    }
    if whatif_init_margin is not None:
        o["whatif_init_margin"] = whatif_init_margin
    if notional is not None:
        o["notional"] = notional
    if multiplier is not None:
        o["multiplier"] = multiplier
    return o


def _pos(qty: float, notional_cad: Optional[float] = None) -> dict:
    d: dict = {"qty": qty}
    if notional_cad is not None:
        d["notional_cad"] = notional_cad
    return d


_FAIL_CLOSED_BLOCKS = {
    MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA,
    MarginBlockReason.MARGIN_UNCOMPUTABLE,
    MarginBlockReason.AGGREGATE_EXPOSURE,
    MarginBlockReason.EXCESS_LIQUIDITY_FLOOR,
    MarginBlockReason.INIT_MARGIN_CEILING,
    MarginBlockReason.SINGLE_ORDER_CAP,
    MarginBlockReason.CRYPTO_INSUFFICIENT_BALANCE,
    MarginBlockReason.INTERNAL_ERROR,
}


# ===========================================================================
# (a) modify ban — FIRST, beats the reduce branch
# ===========================================================================
def test_same_order_id_re_place_is_blocked():
    """An orderId already live in the ledger → BLOCK MODIFY_NOT_ALLOWED."""
    led = PendingExposureLedger()
    led.reserve("o1", "BUY", "AAPL", False, 500.0, 1000.0, qty=10.0, now=_T0)
    cfg = _cfg()
    order = _order("o1", "AAPL", "BUY", 10.0)
    v = decide(order, _Snap(), {}, led, cfg, now=_T0)
    assert isinstance(v, AllowOrBlock)
    assert v.blocked
    assert v.reason is MarginBlockReason.MODIFY_NOT_ALLOWED


def test_modify_ban_is_first_even_for_a_reduce():
    """A live orderId on a REDUCE order is still a modify → MODIFY_NOT_ALLOWED,
    not an ALLOW via the reduce branch. Branch (a) sits before branch (b)."""
    led = PendingExposureLedger()
    # 'o1' is already working; the same id now arrives as a would-be reduce.
    led.reserve("o1", "SELL", "AAPL", True, 0.0, 1000.0, qty=100.0, now=_T0)
    cfg = _cfg()
    positions = {"AAPL": _pos(+100.0, notional_cad=10_000.0)}
    reduce_order = _order("o1", "AAPL", "SELL", 100.0)
    v = decide(reduce_order, _Snap(), positions, led, cfg, now=_T0)
    assert v.blocked
    assert v.reason is MarginBlockReason.MODIFY_NOT_ALLOWED


# ===========================================================================
# burst-of-increasing cannot exceed the aggregate cap (counts prior reservations)
# ===========================================================================
def test_burst_of_increasing_cannot_exceed_aggregate_cap():
    """Each order is judged net of what is already reserved; a burst tips over
    the cap because prior orders reserved. If reservations were NOT counted, #3
    would wrongly pass — so #3 blocking proves the accumulation."""
    led = PendingExposureLedger()
    cfg = _cfg()  # N=2.0, single cap 10% → 10_000 on netliq 100_000
    snap = _Snap(net_liquidation=100_000.0, excess_liquidity=100_000.0, full_init_margin_req=0.0)
    positions = {"BOOK": _pos(1000.0, notional_cad=180_000.0)}  # filled gross 180k

    # cap = max(2*100k, 180k) = 200k. Each order notional 10k (CAD).
    v1 = decide(_order("o1", "AAPL", "BUY", 100.0), snap, positions, led, cfg, now=_T0)
    v2 = decide(_order("o2", "AAPL", "BUY", 100.0), snap, positions, led, cfg, now=_T0)
    v3 = decide(_order("o3", "AAPL", "BUY", 100.0), snap, positions, led, cfg, now=_T0)

    assert v1.allowed and v1.reason is MarginBlockReason.ALLOWED           # 180k+0+10k=190k ≤ 200k
    assert v2.allowed and v2.reason is MarginBlockReason.ALLOWED           # 180k+10k+10k=200k ≤ 200k
    assert v3.blocked and v3.reason is MarginBlockReason.AGGREGATE_EXPOSURE  # 180k+20k+10k=210k > 200k
    assert led.total_reserved().notional == pytest.approx(20_000.0)         # only o1,o2 reserved

    # Control: with NO prior reservations a single 10k order passes — proving it
    # is the accumulated in-flight reservations, not the order size, that blocks.
    fresh = PendingExposureLedger()
    solo = decide(_order("o1", "AAPL", "BUY", 100.0), snap, positions, fresh, cfg, now=_T0)
    assert solo.allowed


# ===========================================================================
# "reduce-burst cannot flip position" — MUST FAIL IF THE REDUCER DID NOT RESERVE
# ===========================================================================
def test_reduce_burst_cannot_flip_position():
    """3× SELL-100 vs long 100. #1 is within the remainder → ALLOW + reserve
    (reducing). #2/#3 see remainder 0 (BECAUSE #1 reserved) → full checks →
    BLOCK. A constrained account (order notional > single-order cap) makes the
    flip-order a hard BLOCK, so the demonstration does not rely on staleness."""
    led = PendingExposureLedger()
    cfg = _cfg()  # netliq 50k → single-order cap 5k; order notional 10k
    snap = _Snap(net_liquidation=50_000.0, excess_liquidity=50_000.0, full_init_margin_req=0.0)
    positions = {"AAPL": _pos(+100.0, notional_cad=10_000.0)}

    v1 = decide(_order("o1", "AAPL", "SELL", 100.0), snap, positions, led, cfg, now=_T0)
    assert v1.allowed and v1.reason is MarginBlockReason.ALLOWED_REDUCE_ONLY
    # The load-bearing assertion: the reducer RESERVED, so the remainder shrank.
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)

    v2 = decide(_order("o2", "AAPL", "SELL", 100.0), snap, positions, led, cfg, now=_T0)
    v3 = decide(_order("o3", "AAPL", "SELL", 100.0), snap, positions, led, cfg, now=_T0)
    assert v2.blocked and v2.reason is MarginBlockReason.SINGLE_ORDER_CAP
    assert v3.blocked and v3.reason is MarginBlockReason.SINGLE_ORDER_CAP
    # The blocked flip-orders did NOT reserve; the reducing qty stays at 100.
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)


def test_reduce_burst_would_over_flatten_if_reducer_skipped_the_ledger():
    """The explicit contrast that makes the invariant real: if the first reducer
    is NOT registered, the remainder stays at the full position and a second
    SELL-100 is wrongly classified as reduce-only → ALLOW (over-flatten past
    flat into a short). This is exactly the 2.1→2.2 bug the reservation closes."""
    led = PendingExposureLedger()  # #1 never reserved (simulating the bug)
    cfg = _cfg()
    snap = _Snap(net_liquidation=50_000.0, excess_liquidity=50_000.0, full_init_margin_req=0.0)
    positions = {"AAPL": _pos(+100.0, notional_cad=10_000.0)}
    # With pending_reducing_qty == 0, remainder == 100, so a SELL-100 is treated
    # as reduce-only and ALLOWED — the over-flatten the real gate prevents.
    v = decide(_order("o2", "AAPL", "SELL", 100.0), snap, positions, led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.ALLOWED_REDUCE_ONLY


# ===========================================================================
# over-leveraged + stale feed + flatten → ALLOW up to remainder, BLOCK beyond
# ===========================================================================
@pytest.mark.parametrize("stale_snapshot", [None, _Snap(usable=False, reason="STALE")])
def test_over_leveraged_stale_flatten_allows_up_to_remainder(stale_snapshot):
    """A hugely over-leveraged book with a dead/missing feed can still flatten up
    to the reducible remainder (exit never trapped), but a reduce BEYOND the
    remainder falls to the fail-closed checks and BLOCKs on the stale feed."""
    led = PendingExposureLedger()
    cfg = _cfg()
    positions = {"AAPL": _pos(+100.0, notional_cad=999_999.0)}  # over-leveraged

    flatten = decide(_order("o1", "AAPL", "SELL", 100.0), stale_snapshot, positions, led, cfg, now=_T0)
    assert flatten.allowed and flatten.reason is MarginBlockReason.ALLOWED_REDUCE_ONLY
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)

    # Beyond the remainder (remainder now 0) → full checks → stale feed → BLOCK.
    beyond = decide(_order("o2", "AAPL", "SELL", 1.0), stale_snapshot, positions, led, cfg, now=_T0)
    assert beyond.blocked and beyond.reason is MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA


def test_reduce_beyond_remainder_in_one_order_falls_to_full_checks():
    """A single over-reduce (SELL 150 vs long 100, no pending) is not reduce-only
    — it would flip — so the WHOLE order goes to the fail-closed checks."""
    led = PendingExposureLedger()
    cfg = _cfg()
    positions = {"AAPL": _pos(+100.0, notional_cad=10_000.0)}
    v = decide(_order("o1", "AAPL", "SELL", 150.0), _Snap(usable=False, reason="STALE"),
               positions, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA


# ===========================================================================
# every fail-closed path → BLOCK for an increasing order
# ===========================================================================
def test_increasing_blocks_on_stale_snapshot():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    v = decide(_order("o1", "AAPL", "BUY", 10.0), _Snap(usable=False, reason="STALE"), {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA


def test_increasing_blocks_on_missing_snapshot():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    v = decide(_order("o1", "AAPL", "BUY", 10.0), None, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA


def test_increasing_blocks_on_incomplete_snapshot_fields():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    snap = _Snap(usable=True, net_liquidation=None)  # usable flag set but a field missing
    v = decide(_order("o1", "AAPL", "BUY", 10.0), snap, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA


def test_increasing_blocks_on_uncomputable_margin_no_notional_no_whatif():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # No price and no notional and no whatIf → neither margin estimate computable.
    bad = {"order_id": "o1", "symbol": "AAPL", "side": "BUY", "qty": 10.0,
           "currency": "CAD", "asset_class": "equity"}
    v = decide(bad, _Snap(), {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.MARGIN_UNCOMPUTABLE


def test_increasing_blocks_on_uncomputable_margin_unknown_asset_class():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # Notional computable, but unknown asset class → no independent rate, and no
    # whatIf → margin uncomputable.
    order = _order("o1", "AAPL", "BUY", 10.0, asset_class="bond")
    v = decide(order, _Snap(), {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.MARGIN_UNCOMPUTABLE


def test_increasing_blocks_when_notional_uncomputable_but_whatif_present():
    """whatIf gives a margin, but the notional itself is uncomputable (no price /
    no notional) → the aggregate/cap checks cannot run → MARGIN_UNCOMPUTABLE."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    order = {"order_id": "o1", "symbol": "AAPL", "side": "BUY", "qty": 10.0,
             "currency": "CAD", "asset_class": "equity", "whatif_init_margin": 500.0}
    v = decide(order, _Snap(), {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.MARGIN_UNCOMPUTABLE


def test_increasing_blocks_on_unreadable_positions_enforce_internal_error():
    """Unreadable positions (an internal exception) → ENFORCE fail-closed BLOCK
    (INTERNAL_ERROR)."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    v = decide(_order("o1", "AAPL", "BUY", 10.0), _Snap(), _BoomPositions(), led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.INTERNAL_ERROR


def test_increasing_blocks_on_excess_liquidity_floor():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # netliq 100k → floor 25k. excess 30k, margin 10k → projected 20k < 25k.
    snap = _Snap(net_liquidation=100_000.0, excess_liquidity=30_000.0, full_init_margin_req=0.0)
    order = _order("o1", "AAPL", "BUY", 200.0, price=100.0)  # notional 20k, indep margin 10k
    v = decide(order, snap, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.EXCESS_LIQUIDITY_FLOOR


def test_increasing_blocks_on_init_margin_ceiling():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # netliq 100k → ceiling 50k. current init 46k, margin 10k → 56k > 50k.
    snap = _Snap(net_liquidation=100_000.0, excess_liquidity=100_000.0, full_init_margin_req=46_000.0)
    order = _order("o1", "AAPL", "BUY", 200.0, price=100.0)  # notional 20k → indep 10k
    v = decide(order, snap, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.INIT_MARGIN_CEILING


def test_increasing_blocks_on_single_order_cap():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # netliq 100k → single cap 10k. notional 12k > 10k. Keep margin low so init /
    # excess pass and single-order is the binding check.
    snap = _Snap(net_liquidation=100_000.0, excess_liquidity=100_000.0, full_init_margin_req=0.0)
    order = _order("o1", "AAPL", "BUY", 120.0, price=100.0)  # notional 12k
    v = decide(order, snap, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.SINGLE_ORDER_CAP


def test_every_fail_closed_reason_is_a_block():
    """Sanity: no fail-closed path accidentally returns allowed."""
    for reason in _FAIL_CLOSED_BLOCKS:
        assert reason is not MarginBlockReason.ALLOWED
        assert reason is not MarginBlockReason.ALLOWED_REDUCE_ONLY


# ===========================================================================
# aggregate uses max(N×NetLiq, current): breached book can REDUCE not INCREASE
# ===========================================================================
def test_breached_book_can_reduce_but_not_increase():
    cfg = _cfg("enforce_paper")
    snap = _Snap(net_liquidation=100_000.0, excess_liquidity=100_000.0, full_init_margin_req=0.0)
    # filled gross 5,000,000 ≫ N×netliq (200,000): the book is already breached.
    positions = {"AAPL": _pos(+100.0, notional_cad=5_000_000.0)}

    # INCREASE (BUY on a long) → aggregate cap = max(200k, 5.0M) = 5.0M, and
    # 5.0M + 0 + 10k > 5.0M → BLOCK. A breached book cannot grow.
    led_inc = PendingExposureLedger()
    inc = decide(_order("o1", "AAPL", "BUY", 100.0, price=100.0), snap, positions, led_inc, cfg, now=_T0)
    assert inc.blocked and inc.reason is MarginBlockReason.AGGREGATE_EXPOSURE

    # REDUCE (SELL within remainder) → branch (b), never reaches the aggregate.
    led_red = PendingExposureLedger()
    red = decide(_order("o2", "AAPL", "SELL", 50.0, price=100.0), snap, positions, led_red, cfg, now=_T0)
    assert red.allowed and red.reason is MarginBlockReason.ALLOWED_REDUCE_ONLY


# ===========================================================================
# max(whatIf, independent) — the conservative (larger) governs, each direction
# ===========================================================================
def _governs_snap() -> _Snap:
    # netliq 1M → init ceiling 500k; current init 495k → BLOCK iff margin > 5k.
    return _Snap(net_liquidation=1_000_000.0, excess_liquidity=600_000.0, full_init_margin_req=495_000.0)


def test_conservative_margin_independent_governs():
    """independent (6k) > whatIf (1k): the larger governs → 495k+6k > 500k →
    BLOCK. Had whatIf (1k) been used, the order would have passed."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # notional 12k CAD → independent = 12k*0.5 = 6k; whatIf 1k. notional < single
    # cap (100k) so INIT is the binding check.
    order = _order("o1", "AAPL", "BUY", 120.0, price=100.0, whatif_init_margin=1_000.0)
    v = decide(order, _governs_snap(), {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.INIT_MARGIN_CEILING


def test_conservative_margin_whatif_governs():
    """whatIf (6k) > independent (4k): the larger governs → BLOCK. Had
    independent (4k) been used, the order would have passed."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # notional 8k CAD → independent = 4k; whatIf 6k.
    order = _order("o1", "AAPL", "BUY", 80.0, price=100.0, whatif_init_margin=6_000.0)
    v = decide(order, _governs_snap(), {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.INIT_MARGIN_CEILING


def test_conservative_margin_control_both_small_allows():
    """Both estimates below the threshold → the max is still below → ALLOW.
    Proves the two BLOCKs above are driven specifically by the larger margin."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    # notional 8k → independent 4k; whatIf 4k → max 4k → 495k+4k=499k ≤ 500k.
    order = _order("o1", "AAPL", "BUY", 80.0, price=100.0, whatif_init_margin=4_000.0)
    v = decide(order, _governs_snap(), {}, led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.ALLOWED


# ===========================================================================
# conservative FX bias — USD exposure overstated in CAD
# ===========================================================================
def test_usd_notional_converted_to_cad_with_conservative_bias():
    """A USD order's reserved CAD notional == native × constant × bias — the
    pre-registered overstatement (FX fails closed, §2.1)."""
    led = PendingExposureLedger()
    cfg = _cfg()  # bias 1.05
    # netliq large so nothing else binds; native 1000 USD.
    snap = _Snap(net_liquidation=10_000_000.0, excess_liquidity=10_000_000.0, full_init_margin_req=0.0)
    order = _order("o1", "AAPL", "BUY", 10.0, price=100.0, currency="USD")  # native 1000 USD
    v = decide(order, snap, {}, led, cfg, now=_T0)
    assert v.allowed
    expected_cad = 1000.0 * USDCAD_CONVERSION_CONSTANT * 1.05
    assert led.total_reserved().notional == pytest.approx(expected_cad)


# ===========================================================================
# crypto lane routes to the kraken own-lane notional-vs-balance check
# ===========================================================================
def test_crypto_increasing_within_balance_allows():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    kraken = _Kraken(usable=True, available_cad=5_000.0)
    order = _order("c1", "BTC", "BUY", 1.0, price=1_000.0, currency="CAD", asset_class="crypto")
    v = decide(order, kraken, {}, led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.ALLOWED
    assert led.total_reserved().notional == pytest.approx(1_000.0)


def test_crypto_increasing_beyond_balance_blocks():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    kraken = _Kraken(usable=True, available_cad=5_000.0)
    order = _order("c1", "BTC", "BUY", 1.0, price=10_000.0, currency="CAD", asset_class="crypto")
    v = decide(order, kraken, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.CRYPTO_INSUFFICIENT_BALANCE


def test_crypto_blocks_on_unusable_kraken_snapshot():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    kraken = _Kraken(usable=False, available_cad=None, reason="STALE")
    order = _order("c1", "BTC", "BUY", 1.0, price=1_000.0, currency="CAD", asset_class="crypto")
    v = decide(order, kraken, {}, led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA


def test_crypto_decision_depends_on_kraken_available_not_ibkr():
    """Own-lane routing: the SAME order allows at available 5k and blocks at 500 —
    the verdict is driven by the kraken balance, not any IBKR margin field."""
    cfg = _cfg("enforce_paper")
    order = _order("c1", "BTC", "BUY", 1.0, price=1_000.0, currency="CAD", asset_class="crypto")
    ok = decide(order, _Kraken(available_cad=5_000.0), {}, PendingExposureLedger(), cfg, now=_T0)
    no = decide(order, _Kraken(available_cad=500.0), {}, PendingExposureLedger(), cfg, now=_T0)
    assert ok.allowed
    assert no.blocked and no.reason is MarginBlockReason.CRYPTO_INSUFFICIENT_BALANCE


def test_crypto_uses_real_kraken_provider_snapshot():
    """Routes against a REAL KrakenBuyingPowerSnapshot from the committed
    provider (parse_kraken_balances), not just a stand-in."""
    captured = datetime(2026, 7, 6, tzinfo=timezone.utc).timestamp()
    snap = parse_kraken_balances(
        {"ts_utc": "2026-07-06T00:00:00Z", "ok": True, "balances": {"CAD": 5_000.0}, "error": None},
        now_epoch=captured + 1.0,
    )
    assert snap.usable and snap.available_cad == pytest.approx(5_000.0)
    cfg = _cfg("enforce_paper")
    led = PendingExposureLedger()
    order = _order("c1", "BTC", "BUY", 1.0, price=1_000.0, currency="CAD", asset_class="crypto")
    v = decide(order, snap, {}, led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.ALLOWED


def test_crypto_burst_cannot_exceed_available_balance():
    """The crypto lane, like the IBKR lane, counts in-flight reservations: a
    same-cycle burst of crypto BUYs cannot each pass against the same static
    balance and over-commit. available 5_000; two BUYs of 3_000 → #1 allows +
    reserves, #2 sees 3_000 reserved and BLOCKs (3_000 + 3_000 > 5_000). Without
    counting in-flight, both would wrongly pass (the 17.9× burst hole)."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    kraken = _Kraken(usable=True, available_cad=5_000.0)
    o1 = _order("c1", "BTC", "BUY", 3.0, price=1_000.0, currency="CAD", asset_class="crypto")
    o2 = _order("c2", "ETH", "BUY", 3.0, price=1_000.0, currency="CAD", asset_class="crypto")
    v1 = decide(o1, kraken, {}, led, cfg, now=_T0)
    v2 = decide(o2, kraken, {}, led, cfg, now=_T0)
    assert v1.allowed and v1.reason is MarginBlockReason.ALLOWED
    assert led.total_reserved().notional == pytest.approx(3_000.0)
    assert v2.blocked and v2.reason is MarginBlockReason.CRYPTO_INSUFFICIENT_BALANCE
    # The blocked order did not reserve; only o1's 3_000 stands.
    assert led.total_reserved().notional == pytest.approx(3_000.0)


def test_crypto_reduce_within_remainder_allowed_even_if_kraken_unusable():
    """Branch (b) precedes the crypto lane: a crypto flatten within the remainder
    is allowed even with a dead kraken feed (exit never trapped)."""
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    positions = {"BTC": _pos(+2.0, notional_cad=20_000.0)}
    order = _order("c1", "BTC", "SELL", 1.0, price=1_000.0, currency="CAD", asset_class="crypto")
    v = decide(order, _Kraken(usable=False, available_cad=None), positions, led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.ALLOWED_REDUCE_ONLY


# ===========================================================================
# SHADOW vs ENFORCE internal-error handling (logs + allows + reserves vs blocks)
# ===========================================================================
def test_shadow_internal_error_logs_allows_and_still_reserves(caplog):
    led = PendingExposureLedger()
    cfg = _cfg("shadow")
    order = _order("o1", "AAPL", "BUY", 10.0, price=100.0, currency="CAD")
    with caplog.at_level("ERROR"):
        v = decide(order, _Snap(), _BoomPositions(), led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.SHADOW_ERROR
    assert v.reserved is True
    # The order was still counted in the ledger (no silent under-count in shadow).
    assert led.is_reserved("o1") is True
    assert any("SHADOW_ERROR" in rec.message for rec in caplog.records)


def test_enforce_internal_error_blocks_and_does_not_reserve():
    led = PendingExposureLedger()
    cfg = _cfg("enforce_paper")
    order = _order("o1", "AAPL", "BUY", 10.0, price=100.0, currency="CAD")
    v = decide(order, _Snap(), _BoomPositions(), led, cfg, now=_T0)
    assert v.blocked and v.reason is MarginBlockReason.INTERNAL_ERROR
    assert v.reserved is False
    assert led.is_reserved("o1") is False


# ===========================================================================
# ALLOW paths reserve with the correct reducing flag
# ===========================================================================
def test_increasing_allow_reserves_increasing_not_reducing():
    led = PendingExposureLedger()
    cfg = _cfg()
    order = _order("o1", "AAPL", "BUY", 50.0, price=100.0, currency="CAD")
    v = decide(order, _Snap(), {}, led, cfg, now=_T0)
    assert v.allowed and v.reserved
    assert led.total_reserved().count == 1
    assert led.pending_reducing_qty("AAPL") == 0.0  # increasing, not a reducer


def test_reduce_only_allow_reserves_reducing():
    led = PendingExposureLedger()
    cfg = _cfg()
    positions = {"AAPL": _pos(+100.0, notional_cad=10_000.0)}
    order = _order("o1", "AAPL", "SELL", 40.0, price=100.0, currency="CAD")
    v = decide(order, _Snap(), positions, led, cfg, now=_T0)
    assert v.allowed and v.reason is MarginBlockReason.ALLOWED_REDUCE_ONLY
    assert led.pending_reducing_qty("AAPL") == pytest.approx(40.0)


# ===========================================================================
# strict config: refuse to start on typo / missing / wrong type
# ===========================================================================
def test_load_frozen_config_real_file_ok():
    cfg = load_frozen_config(_FROZEN_CONFIG_PATH)
    assert cfg.mode == "shadow"
    assert cfg.aggregate_gross_exposure_max_mult_netliq == 2.0
    assert cfg.excess_liquidity_floor_pct_netliq == 25.0
    assert cfg.init_margin_ceiling_pct_netliq == 50.0
    assert cfg.single_order_notional_max_pct_netliq == 10.0
    assert cfg.fx_conservative_bias_mult == 1.05
    assert cfg.margin_rate("equity") == 0.50
    assert cfg.margin_rate("crypto") == 1.00
    assert cfg.margin_rate("futures") == 0.25
    assert cfg.margin_rate("bond") is None


def _write_cfg(tmp_path, mutate) -> str:
    raw = json.loads(open(_FROZEN_CONFIG_PATH, "r", encoding="utf-8").read())
    mutate(raw)
    p = tmp_path / "margin_block.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return str(p)


def test_config_unknown_top_level_key_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r.__setitem__("evil_key", 1))
    with pytest.raises(MarginBlockConfigError, match="unknown key"):
        load_frozen_config(path)


def test_config_typod_threshold_key_refuses(tmp_path):
    def mutate(r):
        r["thresholds"]["aggregate_gross_exposure_max_mult_netliq_TYPO"] = r["thresholds"].pop(
            "aggregate_gross_exposure_max_mult_netliq"
        )
    path = _write_cfg(tmp_path, mutate)
    # Both an unknown key AND a missing required key — either message is fine.
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(path)


def test_config_missing_required_key_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r.pop("mode"))
    with pytest.raises(MarginBlockConfigError, match="missing required key"):
        load_frozen_config(path)


def test_config_missing_nested_required_key_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r["thresholds"].pop("single_order_notional_max_pct_netliq"))
    with pytest.raises(MarginBlockConfigError, match="missing required key"):
        load_frozen_config(path)


def test_config_wrong_type_mode_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r.__setitem__("mode", 5))
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(path)


def test_config_wrong_type_threshold_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r["thresholds"].__setitem__(
        "single_order_notional_max_pct_netliq", "ten-percent"))
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(path)


def test_config_bool_where_number_expected_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r["thresholds"].__setitem__(
        "aggregate_gross_exposure_max_mult_netliq", True))
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(path)


def test_config_unknown_mode_value_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r.__setitem__("mode", "enforce_everything"))
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(path)


def test_config_safety_flag_false_refuses(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r["safety"].__setitem__("no_env_overrides", False))
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(path)


def test_config_underscore_key_is_allowed(tmp_path):
    path = _write_cfg(tmp_path, lambda r: r.__setitem__("_extra_note", "harmless doc"))
    cfg = load_frozen_config(path)  # underscore keys are documentation, not a violation
    assert cfg.mode == "shadow"


def test_config_malformed_json_refuses(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json ", encoding="utf-8")
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(str(p))


def test_config_unreadable_path_refuses(tmp_path):
    with pytest.raises(MarginBlockConfigError):
        load_frozen_config(str(tmp_path / "does_not_exist.json"))


def test_config_no_env_override(monkeypatch):
    """No env-var overrides (Part 3): setting an env var does not change the
    loaded mode — the frozen file is the sole source."""
    monkeypatch.setenv("MARGIN_BLOCK_MODE", "enforce_live")
    monkeypatch.setenv("CHAD_MARGIN_BLOCK_MODE", "enforce_live")
    cfg = load_frozen_config(_FROZEN_CONFIG_PATH)
    assert cfg.mode == "shadow"  # env ignored


# ===========================================================================
# isolation: NO live/execution/broker import, NO order submission
# ===========================================================================
def test_module_imports_no_live_execution_or_broker_path():
    import chad.risk.margin_block as mod

    with open(mod.__file__, "r", encoding="utf-8") as fh:
        text = fh.read()
    tree = ast.parse(text)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    banned = ("execution", "live_loop", "live_gate", "orchestrator", "ibkr",
              "broker", "ib_async", "ib_insync")
    for module_name in imported:
        low = module_name.lower()
        for token in banned:
            assert token not in low, f"forbidden import {module_name!r} (token {token!r})"


def test_module_never_submits_an_order():
    import chad.risk.margin_block as mod

    with open(mod.__file__, "r", encoding="utf-8") as fh:
        text = fh.read()
    # No order-submission token anywhere in the source (docstrings included).
    for token in ("placeOrder", "place_order", "reqGlobalCancel"):
        assert token not in text, token
    # And no call/attribute named like a submit path (AST-level).
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr.lower() not in ("placeorder", "place_order", "submitorder"), node.attr


def test_gate_exposes_no_order_or_submit_symbol():
    import chad.risk.margin_block as mod

    for attr in ("placeOrder", "place_order", "submit", "cancel", "ib", "IB"):
        assert not hasattr(mod, attr), attr
