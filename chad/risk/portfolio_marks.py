"""W3B-7 — receipt-time-stamped broker portfolio marks for the exit overlay.

Flag-gated: ``CHAD_OVERLAY_PORTFOLIO_MARKS`` (default OFF → the overlay keeps
its price_cache loader byte-identically).

Why: the equity overlay prices decisions and evidence from
``runtime/price_cache.json`` — a 60s-cadence *delayed-frozen* snapshot whose
worst case (~55s stale, PA_SIM_MARK_freshness_2026-07-20) diverged $1.88/sh
from the broker fill. The shared truth ``ib`` (clientId 99) already streams
``updatePortfolio`` events (ib_async's connectAsync subscribes to account
updates by default), so the broker's own mark is available in-process with
ZERO new broker requests — this module only listens, it never issues any
req* call (pinned by test_module_never_issues_broker_requests). ib_async
does not timestamp PortfolioItems, so this cache stamps receipt time itself.

Safety:
- XOV-2345 class hazard: ib_async wrapper caches reset on socket drop, so
  every read is gated on ``isConnected()`` — no connection, no marks
  (mirrors broker_position_sync.fetch_positions' fail-closed contract).
- The event handler swallows everything: a mark cache must never raise into
  ib_async's event dispatch.
- Composition is freshest-stamped-age-wins per symbol, and ``mark_source``
  records which source won, so evidence stays auditable either way.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

FLAG_ENV = "CHAD_OVERLAY_PORTFOLIO_MARKS"
MARK_SOURCE = "ib_portfolio_mark"

_TRUTHY = {"1", "true", "yes", "on"}


def _flag_on(env: Optional[Mapping[str, str]] = None) -> bool:
    if env is None:
        import os

        env = os.environ
    return str(env.get(FLAG_ENV, "")).strip().lower() in _TRUTHY


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class PortfolioMarkCache:
    """Receipt-time-stamped ``{symbol: (marketPrice, received_at)}`` cache.

    Attach once per ``ib`` instance; the handler lives on the instance's
    ``updatePortfolioEvent`` and keeps accumulating across reconnects of the
    same object (ib_async re-issues the account subscription on reconnect).
    """

    def __init__(self, ib: Any, clock: Optional[Callable[[], datetime]] = None) -> None:
        self.ib = ib
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._marks: Dict[str, Tuple[float, datetime]] = {}
        # Event attach is best-effort: a missing event attribute (stub/test ib)
        # leaves an empty cache that fails closed on read.
        try:
            ib.updatePortfolioEvent += self._on_update
        except Exception:  # noqa: BLE001
            pass

    # -- event side -----------------------------------------------------

    def _on_update(self, item: Any, *_: Any) -> None:
        """updatePortfolio handler. MUST never raise into ib_async dispatch."""
        try:
            contract = getattr(item, "contract", None)
            symbol = str(
                getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or ""
            ).strip().upper()
            price = float(getattr(item, "marketPrice", 0.0) or 0.0)
            if symbol and price > 0.0 and math.isfinite(price):
                self._marks[symbol] = (price, self._clock())
        except Exception:  # noqa: BLE001
            pass

    # -- read side ------------------------------------------------------

    def connected(self) -> bool:
        try:
            return bool(self.ib.isConnected())
        except Exception:  # noqa: BLE001
            return False

    def get_marks(
        self, symbols: Sequence[str]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
        """(prices, meta) for the requested symbols. Fail-closed on a dead
        connection: the cached values would be broker truth from BEFORE the
        drop, and XOV-2345 proved stale-cache-as-truth false-flats books."""
        if not self.connected():
            return {}, {}
        prices: Dict[str, float] = {}
        meta: Dict[str, Dict[str, str]] = {}
        for sym in symbols:
            key = str(sym or "").strip().upper()
            got = self._marks.get(key)
            if got is None:
                continue
            price, received_at = got
            prices[sym] = price
            meta[sym] = {"ts_utc": _iso(received_at), "source": MARK_SOURCE}
        return prices, meta


def compose_freshest_loader(
    base_loader: Callable[[Sequence[str]], Any],
    cache: PortfolioMarkCache,
    clock: Optional[Callable[[], datetime]] = None,
) -> Callable[[Sequence[str]], Tuple[Dict[str, float], Dict[str, Dict[str, str]]]]:
    """Per-symbol freshest-stamped-age-wins over (base_loader, portfolio marks).

    - base_loader may return a plain dict (unstamped → the portfolio mark,
      which IS stamped, wins whenever present) or a (prices, meta) tuple;
    - a symbol only one source knows keeps that source;
    - ``mark_source`` in the returned meta records which source won.
    """
    now_fn = clock or (lambda: datetime.now(timezone.utc))

    def _age(meta_row: Optional[Mapping[str, Any]], now: datetime) -> Optional[float]:
        dt = _parse_iso_ts((meta_row or {}).get("ts_utc"))
        return (now - dt).total_seconds() if dt is not None else None

    def _load(symbols: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
        loaded = base_loader(symbols)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            prices, meta = dict(loaded[0] or {}), {k: dict(v) for k, v in (loaded[1] or {}).items()}
        else:
            prices, meta = dict(loaded or {}), {}
        pm_prices, pm_meta = cache.get_marks(symbols)
        now = now_fn()
        for sym, price in pm_prices.items():
            base_age = _age(meta.get(sym), now)
            pm_age = _age(pm_meta.get(sym), now)
            missing_from_base = sym not in prices
            base_unstamped = base_age is None
            pm_fresher = pm_age is not None and base_age is not None and pm_age < base_age
            if missing_from_base or base_unstamped or pm_fresher:
                prices[sym] = price
                meta[sym] = pm_meta[sym]
        return prices, meta

    return _load


# Live-loop calls the overlay block every cycle; the cache must be a per-ib
# singleton so (a) marks accumulate across cycles and (b) the event handler
# attaches exactly once instead of leaking one handler per cycle.
_ACTIVE_CACHE: Optional[PortfolioMarkCache] = None


def _cache_for(ib: Any) -> PortfolioMarkCache:
    global _ACTIVE_CACHE
    if _ACTIVE_CACHE is None or _ACTIVE_CACHE.ib is not ib:
        _ACTIVE_CACHE = PortfolioMarkCache(ib)
    return _ACTIVE_CACHE


def maybe_build_overlay_price_loader(
    repo_root: Path,
    ib: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    cache: Optional[PortfolioMarkCache] = None,
) -> Optional[Callable[[Sequence[str]], Any]]:
    """Flag gate + composition. Returns None (→ overlay default loader,
    byte-identical behavior) when the flag is off, ``ib`` is absent, or
    anything fails — the loader upgrade is strictly opt-in and fail-open."""
    if not _flag_on(env):
        return None
    if ib is None:
        return None
    try:
        from chad.risk.position_exit_overlay import _default_price_loader

        live_cache = cache or _cache_for(ib)
        return compose_freshest_loader(_default_price_loader(repo_root), live_cache)
    except Exception:  # noqa: BLE001
        return None
