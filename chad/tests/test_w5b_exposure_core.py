"""
W5B-1 — exposure core tests.

Covers: delta_usd math and signing, the honest-null discipline (unmapped
futures root and missing price never become a silent zero), asset-class and
venue derivation, sector reuse, the provisional-book aggregates, and a parity
pin binding the allocator's local futures reference to the three real tables
it was consolidated from.
"""

from __future__ import annotations

import pytest

from chad.risk.portfolio_allocator import (
    FUTURES_POINT_VALUES,
    UNMAPPED_SECTOR,
    VENUE_IBKR,
    VENUE_KRAKEN,
    ExposureVector,
    ProvisionalBook,
    build_base_book,
    classify_asset_class,
    derive_venue,
    futures_root,
    load_book_positions,
    load_price_cache,
    multiplier_for,
    vector_from_intent,
    vector_from_position,
)

# --------------------------------------------------------------------------- #
# The real book (PLAN_W5B §12.1), frozen as a fixture.
# Sourced from live runtime/positions_truth.json x price_cache.json on
# 2026-07-23. Embedded rather than read live because the worktree has no
# runtime/ and because a test that moves with the market is not a test.
# --------------------------------------------------------------------------- #

REAL_BOOK = [
    {"symbol": "V", "position": 195.0, "secType": "STK", "currency": "USD", "avgCost": 348.28},
    {"symbol": "LLY", "position": 182.0, "secType": "STK", "currency": "USD", "avgCost": 1189.47},
    {"symbol": "SVXY", "position": 163.0, "secType": "STK", "currency": "USD", "avgCost": 58.11},
    {"symbol": "PSQ", "position": 10.0, "secType": "STK", "currency": "USD", "avgCost": 26.59},
    {"symbol": "SPY", "position": 247.0, "secType": "STK", "currency": "USD", "avgCost": 751.67},
    {"symbol": "IWM", "position": 200.0, "secType": "STK", "currency": "USD", "avgCost": 295.89},
    {"symbol": "MA", "position": 10.0, "secType": "STK", "currency": "USD", "avgCost": 544.28},
    {"symbol": "BAC", "position": 213.0, "secType": "STK", "currency": "USD", "avgCost": 59.65},
    {"symbol": "MSFT", "position": 34.0, "secType": "STK", "currency": "USD", "avgCost": 385.41},
    {"symbol": "UNH", "position": 240.0, "secType": "STK", "currency": "USD", "avgCost": 425.03},
    {"symbol": "AAPL", "position": 12.0, "secType": "STK", "currency": "USD", "avgCost": 328.25},
]

REAL_PRICES = {
    "V": 350.76, "LLY": 1184.00, "SVXY": 56.60, "PSQ": 26.52, "SPY": 739.42,
    "IWM": 291.97, "MA": 530.06, "BAC": 61.28, "MSFT": 381.65, "UNH": 422.70,
    "AAPL": 321.01,
}

REAL_GROSS = 671_037.40


@pytest.fixture()
def sectors():
    from chad.risk.fuse_box import load_sector_map, make_sector_lookup

    return make_sector_lookup(load_sector_map())


# --------------------------------------------------------------------------- #
# Asset class / venue / multiplier
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "sec_type,symbol,expected",
    [
        ("STK", "AAPL", "equity"),
        ("STK", "SPY", "etf"),
        ("FUT", "MES", "futures"),
        ("OPT", "SPY", "options"),
        ("CRYPTO", "BTC-USD", "crypto"),
        ("CASH", "EUR", "forex"),
        ("", "BTC-USD", "crypto"),
        ("WEIRD", "XYZ", "unknown"),
    ],
)
def test_asset_class_from_sec_type(sec_type, symbol, expected):
    assert classify_asset_class(sec_type, symbol) == expected


def test_mym_is_futures_not_equity():
    """P7 caveat A: position_exit_overlay._asset_class omits MYM from its
    futures set and misclassifies it as equity. The allocator reads sec_type
    directly and must not inherit that defect."""
    assert classify_asset_class("FUT", "MYM") == "futures"
    assert multiplier_for("futures", "MYM") == (0.5, None)


def test_venue_split():
    assert derive_venue("crypto") == VENUE_KRAKEN
    for ac in ("equity", "etf", "futures", "options", "forex", "unknown"):
        assert derive_venue(ac) == VENUE_IBKR


def test_futures_root_strips_contract_month():
    assert futures_root("MES") == "MES"
    assert futures_root("MESZ5") == "MES"
    assert futures_root("MES-20261218") == "MES"
    assert futures_root("M6E") == "M6E"


def test_unmapped_futures_root_is_null_not_zero():
    """D2's binding rule. A full-size ES has no point value anywhere in the
    repo; it must yield a loud null, never a silent 0 contribution to gross."""
    mult, reason = multiplier_for("futures", "ES")
    assert mult is None
    assert reason == "unmapped_futures_root"

    v = vector_from_intent(
        {"symbol": "ES", "side": "BUY", "quantity": 1, "sec_type": "FUT",
         "limit_price": 6000.0},
        prices={}, sector_lookup=lambda s: "equity_index_fut",
    )
    assert v.delta_usd is None
    assert v.null_reason == "unmapped_futures_root"
    assert not v.computable


def test_futures_multiplier_parity_with_source_tables():
    """The allocator's local reference is a READ-ONLY consolidation of three
    unreconciled tables. This pins it to them: if any source table changes a
    point value, this fails rather than letting the allocator drift into
    valuing a contract differently from the sizer that trades it."""
    from chad.risk.futures_position_sizer import FUTURES_SPECS
    from chad.strategies.alpha_futures import DEFAULT_SPECS
    from chad.strategies.omega_macro import OMEGA_MACRO_SPECS

    for root, spec in DEFAULT_SPECS.items():
        assert FUTURES_POINT_VALUES[root] == spec.point_value, root
    for root, spec in OMEGA_MACRO_SPECS.items():
        assert FUTURES_POINT_VALUES[root] == spec.point_value, root
    for root, spec in FUTURES_SPECS.items():
        assert FUTURES_POINT_VALUES[root] == spec.point_value, root

    union = set(DEFAULT_SPECS) | set(OMEGA_MACRO_SPECS) | set(FUTURES_SPECS)
    assert set(FUTURES_POINT_VALUES) == union
    assert len(FUTURES_POINT_VALUES) == 9


# --------------------------------------------------------------------------- #
# delta_usd math
# --------------------------------------------------------------------------- #

def test_delta_usd_equity_long(sectors):
    v = vector_from_position(
        {"symbol": "LLY", "position": 182.0, "secType": "STK", "currency": "USD"},
        prices=REAL_PRICES, sector_lookup=sectors,
    )
    assert v.delta_usd == pytest.approx(215_488.0)
    assert v.beta == 1.0 and v.beta_source == "default_1.0"
    assert v.beta_weighted_usd == pytest.approx(v.delta_usd)
    assert v.sector == "healthcare"
    assert v.asset_class == "equity"
    assert v.venue == VENUE_IBKR
    assert v.price_source == "price_cache"


def test_short_position_is_negative(sectors):
    v = vector_from_position(
        {"symbol": "SPY", "position": -100.0, "secType": "STK", "currency": "USD"},
        prices=REAL_PRICES, sector_lookup=sectors,
    )
    assert v.delta_usd == pytest.approx(-73_942.0)
    assert v.side == "SHORT"


def test_sell_intent_is_negative(sectors):
    """positions_truth signs its own quantities; an intent does not, so SELL
    must flip the sign here or a short entry would add to gross as a long."""
    v = vector_from_intent(
        {"symbol": "SPY", "side": "SELL", "quantity": 100, "sec_type": "STK",
         "limit_price": 739.42},
        prices=REAL_PRICES, sector_lookup=sectors,
    )
    assert v.delta_usd == pytest.approx(-73_942.0)


def test_futures_intent_uses_point_value(sectors):
    v = vector_from_intent(
        {"symbol": "MES", "side": "BUY", "quantity": 2, "sec_type": "FUT",
         "limit_price": 6000.0},
        prices={}, sector_lookup=sectors,
    )
    assert v.multiplier == 5.0
    assert v.delta_usd == pytest.approx(60_000.0)
    assert v.asset_class == "futures"


def test_intent_price_ladder(sectors):
    """limit_price → expected_price → price_cache, in that order."""
    common = {"symbol": "AAPL", "side": "BUY", "quantity": 10, "sec_type": "STK"}

    v = vector_from_intent({**common, "limit_price": 300.0, "expected_price": 310.0},
                           prices=REAL_PRICES, sector_lookup=sectors)
    assert (v.price, v.price_source) == (300.0, "intent_limit_price")

    v = vector_from_intent({**common, "expected_price": 310.0},
                           prices=REAL_PRICES, sector_lookup=sectors)
    assert (v.price, v.price_source) == (310.0, "intent_expected_price")

    v = vector_from_intent(common, prices=REAL_PRICES, sector_lookup=sectors)
    assert (v.price, v.price_source) == (321.01, "price_cache")


def test_no_price_anywhere_is_null_not_zero(sectors):
    v = vector_from_intent(
        {"symbol": "ZZZZ", "side": "BUY", "quantity": 10, "sec_type": "STK"},
        prices={}, sector_lookup=sectors,
    )
    assert v.delta_usd is None
    assert v.null_reason == "no_price_available"


def test_position_falls_back_to_avg_cost_and_labels_it(sectors):
    """positions_truth carries no inline mark (P8). avgCost is entry cost, not
    a mark — usable as a fallback only if the evidence says which it is."""
    v = vector_from_position(
        {"symbol": "LLY", "position": 182.0, "secType": "STK", "avgCost": 1189.47},
        prices={}, sector_lookup=sectors,
    )
    assert v.price_source == "position_avg_cost"
    assert v.delta_usd == pytest.approx(182 * 1189.47)


def test_unmapped_symbol_lands_in_unmapped_bucket(sectors):
    v = vector_from_intent(
        {"symbol": "ZZZZ", "side": "BUY", "quantity": 1, "sec_type": "STK",
         "limit_price": 10.0},
        prices={}, sector_lookup=sectors,
    )
    assert v.sector == UNMAPPED_SECTOR


# --------------------------------------------------------------------------- #
# The provisional book over the real 11-symbol book
# --------------------------------------------------------------------------- #

def test_real_book_gross_and_net(sectors):
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    assert book.gross_usd == pytest.approx(REAL_GROSS, abs=0.5)
    # The finding: gross == net, because every single position is long.
    assert book.net_usd == pytest.approx(REAL_GROSS, abs=0.5)
    assert len(book.base_vectors) == 11
    assert book.uncomputable == []


def test_real_book_sector_concentration(sectors):
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    by_sector = book.by_sector()
    assert by_sector["healthcare"] == pytest.approx(316_936.0, abs=0.5)
    assert by_sector["index_etf"] == pytest.approx(241_030.74, abs=0.5)
    assert by_sector["financials"] == pytest.approx(86_751.44, abs=0.5)
    # PSQ, the only genuinely short-beta ticket, is rounding error.
    assert by_sector["inverse_etf"] == pytest.approx(265.20, abs=0.5)
    assert by_sector["inverse_etf"] / book.gross_usd < 0.001


def test_real_book_is_single_currency(sectors):
    """One currency key ⇒ the sums are unit-clean. More than one would mean
    gross is adding currencies with no FX rate to bridge them (§12.2)."""
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    assert list(book.currency_mix()) == ["USD"]


def test_provisional_book_accumulates_intents(sectors):
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    before = book.gross_usd
    v = vector_from_intent(
        {"symbol": "LLY", "side": "BUY", "quantity": 10, "sec_type": "STK",
         "limit_price": 1184.0},
        prices=REAL_PRICES, sector_lookup=sectors,
    )
    book.add_intent(v)
    assert book.gross_usd == pytest.approx(before + 11_840.0)
    assert len(book.intent_vectors) == 1


def test_per_symbol_combines_position_and_intent(sectors):
    """The whole point: a position and a pending add on one symbol combine into
    one ticket. The per-ORDER cap cannot see this; that is why LLY is already
    $215k against an enforced $150k per-symbol cap."""
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    book.add_intent(vector_from_intent(
        {"symbol": "LLY", "side": "BUY", "quantity": 10, "sec_type": "STK",
         "limit_price": 1184.0},
        prices=REAL_PRICES, sector_lookup=sectors,
    ))
    assert book.by_symbol()["LLY"] == pytest.approx(215_488.0 + 11_840.0)


def test_uncomputable_excluded_from_sums_and_counted(sectors):
    """An unknown must not be summed as zero — it must be excluded and named."""
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    before = book.gross_usd
    book.add_intent(vector_from_intent(
        {"symbol": "ES", "side": "BUY", "quantity": 1, "sec_type": "FUT",
         "limit_price": 6000.0},
        prices={}, sector_lookup=sectors,
    ))
    assert book.gross_usd == pytest.approx(before)
    assert len(book.uncomputable) == 1
    assert book.summary()["uncomputable_reasons"] == ["unmapped_futures_root"]


def test_empty_book_is_zero_not_error():
    book = ProvisionalBook()
    assert book.gross_usd == 0.0
    assert book.net_usd == 0.0
    assert book.summary()["symbols"] == 0


def test_summary_shape(sectors):
    book = build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)
    s = book.summary()
    assert set(s) == {
        "symbols", "gross_usd", "net_usd", "by_sector", "by_venue",
        "currency_mix", "uncomputable", "uncomputable_reasons",
    }
    assert s["symbols"] == 11
    assert s["by_venue"] == {"IBKR": pytest.approx(REAL_GROSS, abs=0.5)}


# --------------------------------------------------------------------------- #
# Loaders degrade rather than raise
# --------------------------------------------------------------------------- #

def test_loaders_tolerate_missing_files(tmp_path):
    assert load_price_cache(tmp_path / "nope.json") == {}
    assert load_book_positions(tmp_path / "nope.json") == []


def test_loaders_tolerate_corrupt_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_price_cache(bad) == {}
    assert load_book_positions(bad) == []


def test_zero_quantity_row_dropped_by_loader(tmp_path):
    import json

    p = tmp_path / "pt.json"
    p.write_text(json.dumps({"positions": [
        {"symbol": "AAPL", "position": 0.0, "secType": "STK"},
        {"symbol": "MSFT", "position": 5.0, "secType": "STK"},
    ]}), encoding="utf-8")
    rows = load_book_positions(p)
    assert [r["symbol"] for r in rows] == ["MSFT"]


def test_vector_is_frozen(sectors):
    v = vector_from_position(
        {"symbol": "AAPL", "position": 1.0, "secType": "STK"},
        prices=REAL_PRICES, sector_lookup=sectors,
    )
    assert isinstance(v, ExposureVector)
    with pytest.raises(Exception):
        v.delta_usd = 1.0  # type: ignore[misc]
