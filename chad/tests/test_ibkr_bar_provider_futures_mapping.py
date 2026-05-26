"""PR-M2K-MYM — bar-provider futures mapping regression guards.

Covers chad/market_data/ibkr_bar_provider.py:
- FUTURES_SYMBOLS must include the active futures roots (incl. M2K, MYM)
- _make_ib_contract() must build ib_async.Future for those roots, not Stock
- exchange_map must place each root on its correct IBKR exchange (MYM→CBOT,
  M2K→CME, plus existing MES/MNQ/M6E/MCL/MGC/ZN/ZB/SIL→SI mappings)
- equities still resolve to Stock(SMART) — futures-set additions cannot
  silently regress equity path

Tests pass with ib=None so no IBKR connection is required.
"""

from __future__ import annotations

from chad.market_data import ibkr_bar_provider as bp


def _ib_async_classes():
    from ib_async import Future, Stock  # imported lazily, same as bp._make_ib_contract
    return Future, Stock


# ---------------------------------------------------------------------------
# Set membership
# ---------------------------------------------------------------------------

def test_futures_symbols_set_includes_m2k_and_mym() -> None:
    """Module-level FUTURES_SYMBOLS must include M2K and MYM."""
    assert {"M2K", "MYM"} <= bp.FUTURES_SYMBOLS, (
        f"FUTURES_SYMBOLS missing M2K/MYM; got {sorted(bp.FUTURES_SYMBOLS)}"
    )


def test_futures_symbols_set_retains_existing_members() -> None:
    """Existing futures roots must remain in FUTURES_SYMBOLS."""
    expected_existing = {"MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL"}
    assert expected_existing <= bp.FUTURES_SYMBOLS, (
        f"FUTURES_SYMBOLS regressed; missing {expected_existing - bp.FUTURES_SYMBOLS}"
    )


def test_default_universe_includes_m2k_and_mym() -> None:
    """Defensive fallback DEFAULT_UNIVERSE should also list M2K/MYM."""
    assert "M2K" in bp.DEFAULT_UNIVERSE
    assert "MYM" in bp.DEFAULT_UNIVERSE


# ---------------------------------------------------------------------------
# Contract dispatch — M2K / MYM
# ---------------------------------------------------------------------------

def test_m2k_resolves_to_futures_not_stock_smart() -> None:
    """_make_ib_contract('M2K') must return Future, not Stock/SMART."""
    Future, Stock = _ib_async_classes()
    c = bp._make_ib_contract("M2K", ib=None)
    assert isinstance(c, Future), f"M2K resolved to {type(c).__name__}, expected Future"
    assert not isinstance(c, Stock)
    assert c.symbol == "M2K"
    assert c.exchange == "CME"
    assert c.tradingClass == "M2K"
    assert c.currency == "USD"


def test_mym_resolves_to_futures_not_stock_smart() -> None:
    """_make_ib_contract('MYM') must return Future, not Stock/SMART."""
    Future, Stock = _ib_async_classes()
    c = bp._make_ib_contract("MYM", ib=None)
    assert isinstance(c, Future), f"MYM resolved to {type(c).__name__}, expected Future"
    assert not isinstance(c, Stock)
    assert c.symbol == "MYM"
    assert c.exchange == "CBOT"
    assert c.tradingClass == "MYM"
    assert c.currency == "USD"


def test_mym_exchange_is_cbot_not_default_cme() -> None:
    """Explicit guard: MYM must resolve to CBOT, not silently fall through to the CME default."""
    c = bp._make_ib_contract("MYM", ib=None)
    assert c.exchange == "CBOT", (
        f"MYM exchange={c.exchange} — must be CBOT (config/universe.json) "
        f"not the CME default fallback"
    )


# ---------------------------------------------------------------------------
# Regression — existing futures mappings unchanged
# ---------------------------------------------------------------------------

def test_existing_futures_mappings_unchanged() -> None:
    """MES/MNQ/M6E remain on CME, MCL→NYMEX, MGC→COMEX, ZN/ZB→CBOT."""
    Future, _ = _ib_async_classes()
    expected = {
        "MES": ("MES", "CME", "MES"),
        "MNQ": ("MNQ", "CME", "MNQ"),
        "MCL": ("MCL", "NYMEX", "MCL"),
        "MGC": ("MGC", "COMEX", "MGC"),
        "M6E": ("M6E", "CME", "M6E"),
        "ZN":  ("ZN",  "CBOT", "ZN"),
        "ZB":  ("ZB",  "CBOT", "ZB"),
    }
    for root, (ibkr_sym, exchange, trading_class) in expected.items():
        c = bp._make_ib_contract(root, ib=None)
        assert isinstance(c, Future), f"{root} not Future"
        assert c.symbol == ibkr_sym, f"{root}.symbol={c.symbol} expected {ibkr_sym}"
        assert c.exchange == exchange, f"{root}.exchange={c.exchange} expected {exchange}"
        assert c.tradingClass == trading_class, (
            f"{root}.tradingClass={c.tradingClass} expected {trading_class}"
        )
        assert c.currency == "USD"


def test_sil_retains_root_symbol_remap_to_si() -> None:
    """SIL's special-case ibkr root-symbol remap (SI) and COMEX exchange must persist."""
    Future, _ = _ib_async_classes()
    c = bp._make_ib_contract("SIL", ib=None)
    assert isinstance(c, Future)
    assert c.symbol == "SI", f"SIL ibkr_sym={c.symbol} expected SI"
    assert c.exchange == "COMEX"
    assert c.tradingClass == "SIL"


# ---------------------------------------------------------------------------
# Equity path unaffected
# ---------------------------------------------------------------------------

def test_equity_symbol_still_returns_stock_smart() -> None:
    """SPY must still resolve to Stock(SMART) — futures additions cannot regress equities."""
    _, Stock = _ib_async_classes()
    c = bp._make_ib_contract("SPY", ib=None)
    assert isinstance(c, Stock), f"SPY resolved to {type(c).__name__}, expected Stock"
    assert c.symbol == "SPY"
    assert c.exchange == "SMART"
    assert c.currency == "USD"


def test_unknown_symbol_falls_back_to_stock_smart() -> None:
    """Unknown symbols still take the Stock(SMART) branch — no spurious futures classification."""
    _, Stock = _ib_async_classes()
    c = bp._make_ib_contract("ZZZNOTAFUTURE", ib=None)
    assert isinstance(c, Stock)
    assert c.exchange == "SMART"
