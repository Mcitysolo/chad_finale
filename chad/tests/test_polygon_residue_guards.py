"""
GAP-012 / GAP-013 closure tests: Polygon residue guards.

The Polygon subscription was cancelled; IBKR is the sole authoritative
market data source. `chad/market_data/polygon_daily_bars_backfill.py` has
been removed entirely. The legacy `backend/polygon_stocks_stream.py`
remains as a gated/optional tool. These tests assert:

1. Importing `backend/polygon_stocks_stream.py` is safe (no network, no
   SystemExit, no RuntimeError) regardless of CHAD_BAR_PROVIDER.
2. Executing its main() without CHAD_BAR_PROVIDER=polygon raises
   SystemExit with a clear message.
3. The IBKR bar provider docstring no longer claims Polygon is the default
   for equities.
4. The active provider used by the running CHAD daemon does not require
   the polygon-api-client package to be importable.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _without_polygon_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove CHAD_BAR_PROVIDER (or set it to anything non-polygon)."""
    monkeypatch.delenv("CHAD_BAR_PROVIDER", raising=False)


def _purge_module(name: str) -> None:
    """Drop a previously-imported module so importlib re-runs top-level code."""
    sys.modules.pop(name, None)


# ----------------------------------------------------------------------
# backend.polygon_stocks_stream
# ----------------------------------------------------------------------


def test_polygon_stocks_stream_import_safe_without_polygon_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the streamer module must succeed even with no override."""
    _without_polygon_provider(monkeypatch)

    # Make backend/ importable without modifying sys.path globally.
    backend_dir = str(REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        monkeypatch.syspath_prepend(backend_dir)

    _purge_module("polygon_stocks_stream")
    mod = importlib.import_module("polygon_stocks_stream")

    assert callable(getattr(mod, "_require_polygon_provider"))
    assert callable(getattr(mod, "main"))


def test_polygon_stocks_stream_execution_requires_polygon_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling main() without CHAD_BAR_PROVIDER=polygon must SystemExit."""
    _without_polygon_provider(monkeypatch)

    backend_dir = str(REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        monkeypatch.syspath_prepend(backend_dir)

    _purge_module("polygon_stocks_stream")
    mod = importlib.import_module("polygon_stocks_stream")

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    msg = str(excinfo.value)
    assert "CHAD_BAR_PROVIDER=polygon" in msg, (
        f"guard SystemExit message did not name the override: {msg!r}"
    )


# ----------------------------------------------------------------------
# polygon_daily_bars_backfill removal
# ----------------------------------------------------------------------


def test_polygon_daily_bars_backfill_removed() -> None:
    """The legacy backfill module must no longer be present in the tree."""
    legacy = REPO_ROOT / "chad" / "market_data" / "polygon_daily_bars_backfill.py"
    assert not legacy.exists(), (
        f"polygon_daily_bars_backfill.py should be deleted but still exists at {legacy}"
    )

    with pytest.raises(ImportError):
        importlib.import_module("chad.market_data.polygon_daily_bars_backfill")


# ----------------------------------------------------------------------
# ibkr_bar_provider docstring drift
# ----------------------------------------------------------------------


def test_ibkr_bar_provider_docstring_no_longer_claims_polygon_default() -> None:
    """The provider must not advertise Polygon as the default for equities."""
    src_path = REPO_ROOT / "chad" / "market_data" / "ibkr_bar_provider.py"
    src = src_path.read_text(encoding="utf-8")

    # The exact stale phrasing must be gone from comments / docstrings.
    assert "IBKR for futures, Polygon for equities (default)" not in src, (
        "Stale docstring still advertises Polygon as the default for equities."
    )

    # The current operational behavior must be stated explicitly.
    assert "IBKR historical bars are AUTHORITATIVE" in src, (
        "Docstring must declare IBKR as authoritative for active symbols."
    )


# ----------------------------------------------------------------------
# Active path import does not require polygon-api-client
# ----------------------------------------------------------------------


def test_active_provider_does_not_require_polygon_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Block the `polygon` package and verify the active IBKR-side market-data
    modules still import. This proves the operational path has no hidden
    dependency on polygon-api-client.
    """
    # Remove any cached references first.
    for name in list(sys.modules):
        if name == "polygon" or name.startswith("polygon."):
            sys.modules.pop(name, None)

    blocker = _ImportBlocker({"polygon"})
    sys.meta_path.insert(0, blocker)
    try:
        # Confirm the blocker is effective.
        with pytest.raises(ImportError):
            importlib.import_module("polygon")

        # Active modules must still import cleanly with no provider override.
        _without_polygon_provider(monkeypatch)
        for active_mod in (
            "chad.market_data.ibkr_bar_provider",
            "chad.market_data.ibkr_historical_provider",
            "chad.market_data.ibkr_price_provider",
            "chad.market_data.nightly_bars_refresh",
            "chad.market_data.price_cache_refresh",
            "chad.market_data.build_bars_cache",
            "chad.market_data.service",
        ):
            _purge_module(active_mod)
            importlib.import_module(active_mod)
    finally:
        sys.meta_path.remove(blocker)


class _ImportBlocker:
    """A meta-path finder that raises ImportError for the named top-level pkgs."""

    def __init__(self, blocked: set[str]) -> None:
        self._blocked = set(blocked)

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: D401
        top = fullname.split(".", 1)[0]
        if top in self._blocked:
            raise ImportError(f"blocked by test: {fullname}")
        return None
