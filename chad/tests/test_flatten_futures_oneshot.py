"""Tests for scripts/flatten_futures_oneshot.py — deterministic, no network.

Covers:
- closing-order derivation (long->SELL, short->BUY, flat->None)
- preview formatting for a fake 3-position set
- the typed-confirmation gate, incl. the invariant that a non-"FLATTEN"
  input places NOTHING (the order-placement call is mocked and asserted unused)
- non-FUT positions are ignored; empty set -> NOTHING

No ib_async or broker connection is required — flatten_futures takes injectable
I/O, and the script's ib_async imports are lazy (inside connect/place), so the
module imports cleanly offline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Load the standalone script by path (it lives in scripts/, not a package).
# Register it in sys.modules BEFORE exec so @dataclass can resolve its module.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "flatten_futures_oneshot.py"
_spec = importlib.util.spec_from_file_location("flatten_futures_oneshot", _SCRIPT)
ff = importlib.util.module_from_spec(_spec)
sys.modules["flatten_futures_oneshot"] = ff
_spec.loader.exec_module(ff)  # type: ignore[union-attr]


# --- fakes -----------------------------------------------------------------

class _FakeContract:
    def __init__(self, symbol, sec_type="FUT", con_id=0, local_symbol="", month="",
                 exchange=""):
        self.symbol = symbol
        self.secType = sec_type
        self.conId = con_id
        self.localSymbol = local_symbol
        self.lastTradeDateOrContractMonth = month
        self.exchange = exchange


class _FakePos:
    def __init__(self, symbol, position, **kw):
        self.contract = _FakeContract(symbol, **kw)
        self.position = position


class _FakeIB:
    """Returns `first` on the first positions() call, then `after` (re-verify).

    qualifyContracts(*contracts) mirrors ib_async: it mutates each contract in
    place. When `qualify_to` is a string it stamps that exchange on every
    contract (the broker resolved it from the conId); when None it leaves the
    exchange untouched (the broker could NOT resolve it — exercises the map /
    skip fallbacks).
    """
    def __init__(self, first, after=None, qualify_to="CME"):
        self._first = list(first)
        self._after = list(after) if after is not None else []
        self._qualify_to = qualify_to
        self._calls = 0

    def positions(self):
        self._calls += 1
        return self._first if self._calls == 1 else self._after

    def qualifyContracts(self, *contracts):
        if self._qualify_to is not None:
            for c in contracts:
                c.exchange = self._qualify_to
        return list(contracts)


def _fut(symbol, position, con_id, month="20260918", exchange=""):
    return _FakePos(symbol, position, sec_type="FUT", con_id=con_id,
                    local_symbol=f"{symbol}{month[-2:]}", month=month,
                    exchange=exchange)


# --- 1. closing-order derivation -------------------------------------------

def test_derive_long_to_sell():
    co = ff.derive_closing_order(_fut("MES", 142, 1001))
    assert (co.action, co.quantity, co.position) == ("SELL", 142, 142)
    assert co.symbol == "MES" and co.con_id == 1001


def test_derive_short_to_buy():
    co = ff.derive_closing_order(_fut("MNQ", -19, 1002))
    assert (co.action, co.quantity, co.position) == ("BUY", 19, -19)


def test_derive_another_long_to_sell():
    co = ff.derive_closing_order(_fut("MGC", 36, 1003))
    assert (co.action, co.quantity) == ("SELL", 36)


def test_derive_flat_is_none():
    assert ff.derive_closing_order(_fut("MES", 0, 1004)) is None


def test_derive_uses_broker_contract_identity():
    co = ff.derive_closing_order(_fut("M6E", -2, 5555, month="20260615"))
    assert co.con_id == 5555
    assert co.contract_month == "20260615"
    assert co.local_symbol == "M6E15"


# --- 2. preview formatting -------------------------------------------------

def test_preview_lists_all_three_positions():
    orders = [
        ff.derive_closing_order(_fut("MES", 142, 1001)),
        ff.derive_closing_order(_fut("MNQ", -19, 1002)),
        ff.derive_closing_order(_fut("MGC", 36, 1003)),
    ]
    table = ff.format_preview_table(orders)
    for token in ("MES", "MNQ", "MGC", "SELL", "BUY", "142", "19", "36",
                  "1001", "1002", "1003"):
        assert token in table
    assert "3 closing order(s)" in table


# --- 3. typed-confirmation gate --------------------------------------------

@pytest.mark.parametrize("text,ok", [
    ("FLATTEN", True),
    ("FLATTEN ", True),     # surrounding whitespace stripped
    (" FLATTEN", True),
    ("flatten", False),     # case-sensitive
    ("FLATTE", False),
    ("yes", False),
    ("", False),
    ("FLATTEN now", False),
])
def test_confirm_gate(text, ok):
    assert ff.confirm_gate(input_fn=lambda _p: text) is ok


def test_confirm_gate_eof_is_refusal():
    def _eof(_p):
        raise EOFError
    assert ff.confirm_gate(input_fn=_eof) is False


def test_non_flatten_input_places_nothing():
    """The critical gate: a non-confirm input must place ZERO orders."""
    ib = _FakeIB(first=[_fut("MES", 142, 1001), _fut("MNQ", -19, 1002),
                        _fut("MGC", 36, 1003)])
    placer = MagicMock()
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "no", out=out_lines.append, place_order_fn=placer,
    )
    assert status == "ABORTED"
    placer.assert_not_called()                      # <-- nothing placed
    assert any("Aborted" in ln for ln in out_lines)


def test_off_count_warns_but_still_lists_and_can_abort():
    ib = _FakeIB(first=[_fut("MES", 142, 1001), _fut("MNQ", -19, 1002)])  # only 2
    placer = MagicMock()
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "no", out=out_lines.append, place_order_fn=placer,
    )
    assert status == "ABORTED"
    placer.assert_not_called()
    assert any("WARNING" in ln and "found 2" in ln for ln in out_lines)


# --- 4. confirmed path + filtering (still offline; placement mocked) -------

def test_confirmed_places_each_order_and_reverifies_flat():
    futs = [_fut("MES", 142, 1001), _fut("MNQ", -19, 1002), _fut("MGC", 36, 1003)]
    ib = _FakeIB(first=futs, after=[])  # re-verify: nothing left
    placer = MagicMock(return_value=MagicMock(orderStatus=MagicMock(
        status="Filled", filled=1, avgFillPrice=1.0)))
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "FLATTEN", out=out_lines.append, place_order_fn=placer,
    )
    assert status == "FLAT"
    assert placer.call_count == 3
    # each placement got the live contract + derived action/qty
    actions = {c.args[2] for c in placer.call_args_list}
    assert actions == {"SELL", "BUY"}


def test_confirmed_but_position_remains_is_incomplete():
    futs = [_fut("MES", 142, 1001)]
    ib = _FakeIB(first=futs, after=futs)  # re-verify: still there
    placer = MagicMock(return_value=MagicMock(orderStatus=MagicMock(
        status="Submitted", filled=0, avgFillPrice=0.0)))
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "FLATTEN", out=lambda _l: None,
        place_order_fn=placer, expected_count=1,
    )
    assert status == "INCOMPLETE"


def test_non_futures_ignored_and_empty_is_nothing():
    # STK position present but no FUT -> nothing to flatten, no placement.
    stk = _FakePos("AAPL", 100, sec_type="STK", con_id=9, month="")
    ib = _FakeIB(first=[stk])
    placer = MagicMock()
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "FLATTEN", out=out_lines.append, place_order_fn=placer,
    )
    assert status == "NOTHING"
    placer.assert_not_called()
    assert any("nothing to flatten" in ln.lower() for ln in out_lines)


# --- 5. exchange resolution (Warning 321 fix) ------------------------------

def test_resolve_exchange_qualified_used_as_is():
    # qualifyContracts fills the exchange from the conId -> used as-is.
    c = _FakeContract("MES", con_id=1001)          # starts blank
    ib = _FakeIB(first=[], qualify_to="CME")
    assert ff.resolve_exchange(ib, c, out=lambda _l: None) == "CME"
    assert c.exchange == "CME"


def test_resolve_exchange_blank_mcl_mapped_to_nymex():
    # qualify can't resolve (qualify_to=None) -> MCL maps to NYMEX.
    c = _FakeContract("MCL", con_id=2002)
    ib = _FakeIB(first=[], qualify_to=None)
    assert ff.resolve_exchange(ib, c, out=lambda _l: None) == "NYMEX"
    assert c.exchange == "NYMEX"


def test_resolve_exchange_blank_m6e_mapped_to_cme():
    c = _FakeContract("M6E", con_id=2003)
    ib = _FakeIB(first=[], qualify_to=None)
    assert ff.resolve_exchange(ib, c, out=lambda _l: None) == "CME"
    assert c.exchange == "CME"


def test_resolve_exchange_unknown_symbol_returns_none_never_guesses():
    c = _FakeContract("ZZZ", con_id=3003)
    ib = _FakeIB(first=[], qualify_to=None)
    assert ff.resolve_exchange(ib, c, out=lambda _l: None) is None
    assert not c.exchange                            # left blank — never guessed


def test_mapped_symbol_blank_qualify_is_still_placed_on_resolved_exchange():
    # MCL with a blank exchange resolves via the map and IS placed (on NYMEX).
    ib = _FakeIB(first=[_fut("MCL", 3, 5005)], after=[], qualify_to=None)
    placer = MagicMock(return_value=MagicMock(orderStatus=MagicMock(
        status="Filled", filled=3, avgFillPrice=1.0)))
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "FLATTEN", out=out_lines.append,
        place_order_fn=placer, expected_count=1,
    )
    assert status == "FLAT"
    placer.assert_called_once()
    assert placer.call_args.args[1].exchange == "NYMEX"   # placed on resolved exch


def test_unknown_symbol_skipped_placeorder_not_called_and_incomplete():
    # Unknown symbol, qualify can't resolve, not in map -> SKIP, place nothing.
    # `after=[]` (broker reports flat) proves the skip flag forces INCOMPLETE.
    ib = _FakeIB(first=[_fut("ZZZ", 5, 4004)], after=[], qualify_to=None)
    placer = MagicMock()
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "FLATTEN", out=out_lines.append,
        place_order_fn=placer, expected_count=1,
    )
    assert status == "INCOMPLETE"
    placer.assert_not_called()                       # <-- never submitted blank
    assert any("SKIPPED" in ln and "ZZZ" in ln for ln in out_lines)


def test_mixed_one_resolvable_one_unknown_places_only_resolvable():
    # qualify resolves nothing (qualify_to=None); M6E resolves via the map,
    # ZZZ is unknown -> only M6E is placed, ZZZ is skipped.
    ib = _FakeIB(
        first=[_fut("M6E", 2, 1001), _fut("ZZZ", 5, 4004)],
        after=[_fut("ZZZ", 5, 4004)],   # the skipped one remains
        qualify_to=None,
    )
    placer = MagicMock(return_value=MagicMock(orderStatus=MagicMock(
        status="Filled", filled=2, avgFillPrice=1.0)))
    out_lines = []
    status = ff.flatten_futures(
        ib, input_fn=lambda _p: "FLATTEN", out=out_lines.append,
        place_order_fn=placer, expected_count=2,
    )
    assert status == "INCOMPLETE"
    placer.assert_called_once()
    assert placer.call_args.args[1].symbol == "M6E"
