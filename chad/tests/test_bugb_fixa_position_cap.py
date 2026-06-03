"""Bug B Fix A — cumulative broker-truth per-symbol futures position cap.

Hermetic tests for the three production pieces in chad/core/live_loop.py:

- _futures_net_broker_position / _read_broker_truth_for_caps /
  _futures_net_from_truth: ttl-aware, broker-authority-GREEN-gated,
  fail-closed (None) broker net reader.
- _FUTURES_POSITION_CAPS: per-symbol caps derived from the strategy spec
  tables (DEFAULT_SPECS ∪ OMEGA_MACRO_SPECS, min() on conflict).
- _futures_cap_check: the cap decision used at the run_once() FUT-open
  chokepoint, including within-cycle pending-adds accumulation.

The cap consults no env flag — it binds with the futures env gate ON or OFF.
Exits/flips never reach _futures_cap_check (classified upstream, same as the
env gate); these tests cover the open-intent decision itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.core.live_loop import (
    _FUTURES_POSITION_CAPS,
    _futures_cap_check,
    _futures_net_broker_position,
)


def _truth_payload(
    positions: list,
    *,
    status: str = "GREEN",
    truth_ok: bool = True,
    age_s: float = 0.0,
    ttl_s: int = 60,
) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    return {
        "schema_version": "positions_truth.v1",
        "broker_authority_status": status,
        "truth_ok": truth_ok,
        "ts_utc": ts.isoformat().replace("+00:00", "Z"),
        "ttl_seconds": ttl_s,
        "positions": positions,
    }


def _write_truth(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "positions_truth.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


_LIVE_BOOK = [
    {"symbol": "M6E", "secType": "FUT", "position": 217.0, "currency": "USD"},
    {"symbol": "M2K", "secType": "FUT", "position": -25.0, "currency": "USD"},
    {"symbol": "MCL", "secType": "FUT", "position": 50.0, "currency": "USD"},
    {"symbol": "SPY", "secType": "STK", "position": 31.0, "currency": "USD"},
]


# ---------------------------------------------------------------------------
# helper: broker-truth net reader
# ---------------------------------------------------------------------------

def test_helper_fresh_green_truth_returns_net(tmp_path) -> None:
    p = _write_truth(tmp_path, _truth_payload(_LIVE_BOOK))
    assert _futures_net_broker_position("M6E", path=p) == 217.0
    assert _futures_net_broker_position("M2K", path=p) == -25.0
    assert _futures_net_broker_position("MES", path=p) == 0.0  # flat = 0, not None
    # STK entries never count toward a futures net.
    assert _futures_net_broker_position("SPY", path=p) == 0.0


def test_helper_stale_truth_returns_none(tmp_path) -> None:
    p = _write_truth(tmp_path, _truth_payload(_LIVE_BOOK, age_s=3600.0, ttl_s=60))
    assert _futures_net_broker_position("M6E", path=p) is None


def test_helper_not_green_returns_none(tmp_path) -> None:
    p = _write_truth(tmp_path, _truth_payload(_LIVE_BOOK, status="RED"))
    assert _futures_net_broker_position("M6E", path=p) is None
    p2 = _write_truth(tmp_path, _truth_payload(_LIVE_BOOK, truth_ok=False))
    assert _futures_net_broker_position("M6E", path=p2) is None


def test_helper_missing_or_malformed_returns_none(tmp_path) -> None:
    assert _futures_net_broker_position("M6E", path=tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _futures_net_broker_position("M6E", path=bad) is None
    # Malformed position value on a matching entry → fail-closed None.
    p = _write_truth(
        tmp_path,
        _truth_payload([{"symbol": "M6E", "secType": "FUT", "position": "garbage"}]),
    )
    assert _futures_net_broker_position("M6E", path=p) is None


# ---------------------------------------------------------------------------
# cap map derivation
# ---------------------------------------------------------------------------

def test_cap_map_values_and_min_on_conflict() -> None:
    assert _FUTURES_POSITION_CAPS["M6E"] == 3   # omega_macro-only
    assert _FUTURES_POSITION_CAPS["MCL"] == 2   # alpha/gamma table
    assert _FUTURES_POSITION_CAPS["ZB"] == 2    # min(alpha 3, omega 2)
    assert _FUTURES_POSITION_CAPS["MES"] == 5
    assert "FAKE_SYMBOL" not in _FUTURES_POSITION_CAPS


# ---------------------------------------------------------------------------
# cap decision math
# ---------------------------------------------------------------------------

def test_add_under_cap_allowed() -> None:
    pending: dict = {}
    verdict, net, projected, cap = _futures_cap_check("M6E", "BUY", 2.0, 0.0, pending)
    assert verdict == "allow"
    assert (net, projected, cap) == (0.0, 2.0, 3)
    assert pending["M6E"] == 2.0


def test_add_over_cap_blocked() -> None:
    pending: dict = {}
    verdict, net, projected, cap = _futures_cap_check("M6E", "BUY", 4.0, 0.0, pending)
    assert verdict == "block"
    assert (projected, cap) == (4.0, 3)
    assert pending == {}, "blocked intents must not consume headroom"


def test_reduce_against_long_allowed_even_untagged() -> None:
    # SELL 10 against long 217 shrinks |net| — passes even though it reaches
    # the cap check as an untagged "open"-classified intent.
    pending: dict = {}
    verdict, net, projected, _cap = _futures_cap_check("M6E", "SELL", 10.0, 217.0, pending)
    assert verdict == "allow"
    assert projected == 207.0
    assert pending["M6E"] == -10.0


def test_short_side_add_over_cap_blocked() -> None:
    # Shorts are capped by magnitude too: M2K cap 5, net -5, SELL 1 → |−6| > 5.
    verdict, _n, projected, cap = _futures_cap_check("M2K", "SELL", 1.0, -5.0, {})
    assert verdict == "block"
    assert (projected, cap) == (-6.0, 5)


def test_unknown_symbol_refused() -> None:
    verdict, _n, _p, cap = _futures_cap_check("FAKE_SYMBOL", "BUY", 1.0, 0.0, {})
    assert verdict == "block"
    assert cap == 0


def test_none_net_unverified() -> None:
    pending: dict = {}
    verdict, _n, _p, _c = _futures_cap_check("M6E", "BUY", 2.0, None, pending)
    assert verdict == "unverified"
    assert pending == {}


# ---------------------------------------------------------------------------
# within-cycle cumulative projection (Reinforcement B)
# ---------------------------------------------------------------------------

def test_within_cycle_two_adds_accumulate_second_blocked() -> None:
    # MCL cap 2, broker flat. alpha buys 2 (fills the cap), gamma buys 1 in
    # the SAME cycle — must be blocked because the projection includes the
    # first approved add, not just broker truth.
    pending: dict = {}
    v1, _n1, p1, _c = _futures_cap_check("MCL", "BUY", 2.0, 0.0, pending)
    assert v1 == "allow" and p1 == 2.0 and pending["MCL"] == 2.0
    v2, n2, p2, cap = _futures_cap_check("MCL", "BUY", 1.0, 0.0, pending)
    assert v2 == "block"
    assert (n2, p2, cap) == (2.0, 3.0, 2)
    assert pending["MCL"] == 2.0, "blocked second add must not consume headroom"


def test_within_cycle_reduce_frees_headroom() -> None:
    # M6E cap 3, broker +3 (at cap). A SELL 2 reduce frees headroom; a later
    # BUY 1 in the same cycle then projects 3-2+1=2 <= cap → allowed.
    pending: dict = {}
    v1, _n, p1, _c = _futures_cap_check("M6E", "SELL", 2.0, 3.0, pending)
    assert v1 == "allow" and p1 == 1.0
    v2, _n2, p2, _c2 = _futures_cap_check("M6E", "BUY", 1.0, 3.0, pending)
    assert v2 == "allow" and p2 == 2.0


# ---------------------------------------------------------------------------
# M6E +217 live shape
# ---------------------------------------------------------------------------

def test_m6e_live_shape_buy_refused_sell_reduce_allowed(tmp_path) -> None:
    p = _write_truth(tmp_path, _truth_payload(_LIVE_BOOK))
    net = _futures_net_broker_position("M6E", path=p)
    assert net == 217.0

    # The runaway signature: omega_macro BUY 2 with broker net +217 → block.
    verdict, n, projected, cap = _futures_cap_check("M6E", "BUY", 2.0, net, {})
    assert verdict == "block"
    assert (n, projected, cap) == (217.0, 219.0, 3)

    # Disposition direction: SELL (reduce) is allowed.
    verdict_s, _n, projected_s, _c = _futures_cap_check("M6E", "SELL", 50.0, net, {})
    assert verdict_s == "allow"
    assert projected_s == 167.0
