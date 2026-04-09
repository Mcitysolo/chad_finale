"""
context_builder.py
===================

This module provides a production‑grade implementation of a ``ContextBuilder``
for the CHAD trading system. It replaces simplistic or brittle context loading
logic with a modular, configurable, and highly testable design built using
modern Python best practices. The builder assembles a *market context* by
loading recent price bars and ticks, inferring missing data when necessary,
and computing useful aggregates (such as the latest price and current
notional exposure) for downstream strategy engines.

Key design objectives addressed by this implementation include:

* **Single Responsibility** – separate classes for loading bars and ticks,
  assembling context objects, and computing derived metrics. Each class does
  one thing well.
* **Dependency Injection** – providers for bars and ticks can be supplied to
  ``ContextBuilder`` externally, enabling easy swapping of data sources in
  tests or different environments.
* **Configurability** – default behaviour is controlled by environment
  variables, but all settings can be overridden explicitly without code
  changes.
* **Asynchronous I/O** – bar and tick loading use ``asyncio`` to parallelise
  file reads or future network calls without blocking the event loop. A
  synchronous convenience wrapper is also provided.
* **LRU Caching** – expensive per‑symbol operations are cached with
  ``functools.lru_cache`` to minimise redundant work when rebuilding
  contexts repeatedly.
* **Robustness** – missing symbols and malformed data are handled gracefully
  and logged with clear warnings. The builder never returns partially
  initialised objects.
* **Observability** – an ``evidence`` structure records which symbols were
  requested, which data files were found or missing, and any fallback logic
  applied during the build. This aids debugging and auditability without
  polluting the primary API.

Even with these improvements, please note that this module does not create
historical data where none exists. If the underlying ``bars`` directory lacks
files for your desired futures symbols, the resulting context will still
contain empty bar series for those symbols. Additional infrastructure (such
as a data ingestion pipeline) is required to populate the bar cache before
running CHAD in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# These imports are intentionally broad so that the builder remains compatible
# with different versions of the CHAD codebase. If these types are not
# available in your environment, you can safely define lightweight stubs in
# your local type modules or adjust the imports accordingly.
try:
    from chad.types import Position, PortfolioSnapshot  # type: ignore
except Exception:
    # Provide fallback stubs if the CHAD type module is unavailable.
    @dataclass(frozen=True)
    class Position:
        symbol: str
        quantity: float

    @dataclass(frozen=True)
    class PortfolioSnapshot:
        timestamp: datetime
        positions: Mapping[str, Position]
        cash: float

# Legend consensus loader — wired into MarketContext.legend so that
# legend-aware strategies (alpha, beta, delta) can resolve their universe
# and weights from data/legend_top_stocks.json instead of falling through to
# unconditional silence. Import is fail-soft so a missing legend file does
# not break context construction; load_legend() raises LegendLoaderError
# which is handled at the call site.
try:
    from chad.utils.legend_loader import load_legend, LegendLoaderError  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    load_legend = None  # type: ignore[assignment]
    class LegendLoaderError(Exception):  # type: ignore[no-redef]
        pass

__all__ = [
    "MarketContext",
    "ContextResult",
    "BarsProvider",
    "TicksProvider",
    "ContextBuilder",
]


DEFAULT_BARS_PATH = os.getenv(
    "CHAD_BARS_PATH",
    os.path.join(os.getcwd(), "data", "bars", "1d"),
)
"""
Default location of daily bar JSON files. This can be overridden by setting
the ``CHAD_BARS_PATH`` environment variable.
"""

DEFAULT_FUTURES_SYMBOLS: Tuple[str, ...] = tuple(
    (os.getenv("CHAD_FUTURES_SYMBOLS", "MES,MNQ,MCL,MGC").replace(" ", "").split(","))
)
"""
Canonical list of futures symbols to be included in every context build.
May be customised via the ``CHAD_FUTURES_SYMBOLS`` environment variable.
"""

CRYPTO_SYMBOLS: Tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD")
"""
Canonical list of crypto symbols sourced from runtime/kraken_prices.json.
Crypto trades 24/7 and is wired in via _merge_kraken_prices().
"""

KRAKEN_PRICES_PATH = os.getenv(
    "CHAD_KRAKEN_PRICES_PATH",
    os.path.join(os.getcwd(), "runtime", "kraken_prices.json"),
)
KRAKEN_MAX_AGE_SECONDS = 300


@dataclass
class MarketContext:
    """
    Data container for the assembled market state. It includes ticks,
    historical bars, and the latest observed prices for a set of symbols.

    Attributes
    ----------
    ticks : Dict[str, Dict[str, float]]
        The most recent tick per symbol. Each value is a mapping with at
        least a ``price`` key; additional metadata keys may be present.
    bars : Dict[str, List[Mapping[str, float]]]
        Historical daily bar series per symbol. Each bar should include
        ``time``, ``open``, ``high``, ``low``, and ``close`` fields. Empty
        lists represent missing history.
    prices : Dict[str, float]
        A flat mapping of symbol to last price, derived from ``ticks`` or
        inferred from the latest bar. Only symbols that appear in either
        ``ticks`` or ``bars`` will be represented.
    """

    ticks: Dict[str, Mapping[str, float]]
    bars: Dict[str, List[Mapping[str, float]]]
    prices: Dict[str, float]
    now: Optional[datetime] = None
    portfolio: Optional[object] = None
    legend: Optional[object] = None


@dataclass
class ContextResult:
    """
    Wrapper for the result of a context build. Besides the assembled
    ``MarketContext``, it includes the current symbol notionals, overall
    notional exposure, and an evidence map describing the build process.
    """

    context: MarketContext
    prices: Dict[str, float]
    current_symbol_notional: Dict[str, float]
    current_total_notional: float
    evidence: Mapping[str, object] = field(default_factory=dict)


class BarsProvider:
    """
    Load daily bar data from the filesystem. The provider assumes each
    symbol's history lives in a JSON file named ``<SYMBOL>.json`` within a
    specified directory. Each file should contain an object with a ``bars``
    key mapping to a list of bar dictionaries.

    Parameters
    ----------
    bars_path : str | os.PathLike
        The directory containing bar JSON files. Missing files are logged
        and yield empty histories, but do not raise exceptions. See
        :func:`load_bars` for details.
    """

    def __init__(self, bars_path: str | os.PathLike = DEFAULT_BARS_PATH) -> None:
        self.bars_path = Path(bars_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def available_symbols(self) -> List[str]:
        """Return a sorted list of all symbols with bar files present."""
        if not self.bars_path.is_dir():
            self.logger.warning("Bars directory %s does not exist", self.bars_path)
            return []
        return sorted([p.stem for p in self.bars_path.glob("*.json")])

    @lru_cache(maxsize=512)
    def _load_file(self, symbol: str) -> Optional[List[Mapping[str, float]]]:
        """Load a single symbol's bar series from disk."""
        path = self.bars_path / f"{symbol}.json"
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bars: List[Mapping[str, float]] = data.get("bars", [])
            return bars
        except Exception as exc:
            self.logger.error("Failed to load bars for %s: %s", symbol, exc)
            return None

    async def load_bars(self, symbols: Sequence[str]) -> Dict[str, List[Mapping[str, float]]]:
        """
        Asynchronously load bar histories for a list of symbols. Missing files
        yield empty lists. This function can be awaited within an asyncio
        event loop or used synchronously via :func:`load_bars_sync`.
        """
        loop = asyncio.get_running_loop()
        results: Dict[str, List[Mapping[str, float]]] = {}

        async def _load(sym: str) -> None:
            bars = await loop.run_in_executor(None, self._load_file, sym)
            results[sym] = bars or []

        await asyncio.gather(*[_load(s) for s in symbols])
        return results

    def load_bars_sync(self, symbols: Sequence[str]) -> Dict[str, List[Mapping[str, float]]]:
        """
        Synchronous convenience wrapper around :func:`load_bars`. If no event
        loop exists, it will be created temporarily.
        """
        try:
            return asyncio.run(self.load_bars(symbols))
        except RuntimeError:
            # Already in a running loop; create a new loop just for this call.
            return asyncio.get_event_loop().run_until_complete(self.load_bars(symbols))


class TicksProvider:
    """
    Provide the most recent tick (price) for each symbol. In this simple
    implementation, ticks are derived from the latest bar close. In a live
    system, this could pull from a real‑time feed or database. Symbols
    missing both ticks and bars will not appear in the output.

    Parameters
    ----------
    now : datetime
        The current timestamp used for fallback tick generation when only
        bar data is available.
    """

    def __init__(self, now: Optional[datetime] = None) -> None:
        self.now = now or datetime.now(timezone.utc)
        self.logger = logging.getLogger(self.__class__.__name__)

    def derive_ticks_from_bars(self, bars: Dict[str, List[Mapping[str, float]]]) -> Dict[str, Mapping[str, float]]:
        """
        Derive a tick from the last bar close for each symbol. This fallback
        ensures that ``prices`` can be computed even without real‑time ticks.
        """
        ticks: Dict[str, Mapping[str, float]] = {}
        for symbol, series in bars.items():
            if not series:
                continue
            last_bar = series[-1]
            price = last_bar.get("close")
            if price is not None:
                ticks[symbol] = {
                    "symbol": symbol,
                    "price": float(price),
                    "timestamp": self.now.isoformat(),
                    "source": "bars_fallback",
                }
        return ticks

    async def load_ticks(self, symbols: Sequence[str], bars: Dict[str, List[Mapping[str, float]]]) -> Dict[str, Mapping[str, float]]:
        """
        Asynchronously load ticks for each symbol. By default this method
        simply derives ticks from bar history. Override this method to
        integrate with a real‑time price feed.
        """
        return self.derive_ticks_from_bars(bars)

    def load_ticks_sync(self, symbols: Sequence[str], bars: Dict[str, List[Mapping[str, float]]]) -> Dict[str, Mapping[str, float]]:
        """
        Synchronous wrapper around :func:`load_ticks`. See its documentation
        for behaviour. If overriding ``load_ticks``, prefer to implement
        asynchronous logic and call this wrapper for convenience.
        """
        # Use asyncio.run even if load_ticks isn't a coroutine; asyncio will
        # detect and handle the synchronous call gracefully.
        return asyncio.run(self.load_ticks(symbols, bars))


class ContextBuilder:
    """
    Assemble a ``MarketContext`` and associated metadata from bars and ticks.
    This builder relies on ``BarsProvider`` and ``TicksProvider`` instances
    to supply raw data. Additional metadata such as current symbol notionals
    or account cash can be provided externally when integrating with a live
    trading engine.

    Parameters
    ----------
    bars_provider : BarsProvider, optional
        Provider responsible for loading historical bar data. If omitted, a
        default instance using ``DEFAULT_BARS_PATH`` is created.
    ticks_provider : TicksProvider, optional
        Provider responsible for loading real‑time ticks. If omitted, a
        default provider deriving ticks from bar data is used.
    futures_symbols : Iterable[str] | None, optional
        Additional symbols (e.g. futures) that should always be included
        in the universe, regardless of whether they appear in ticks. If
        ``None``, defaults to ``DEFAULT_FUTURES_SYMBOLS``.
    now : datetime | None, optional
        Timestamp used for derived tick generation. If omitted, uses
        ``datetime.now(timezone.utc)``.
    logger : logging.Logger, optional
        Logger for diagnostic output. If omitted, a module‑level logger is
        obtained.
    """

    def __init__(
        self,
        bars_provider: Optional[BarsProvider] = None,
        ticks_provider: Optional[TicksProvider] = None,
        futures_symbols: Optional[Iterable[str]] = None,
        now: Optional[datetime] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.bars_provider = bars_provider or BarsProvider()
        self.ticks_provider = ticks_provider or TicksProvider(now=now)
        self.futures_symbols = tuple(futures_symbols) if futures_symbols is not None else DEFAULT_FUTURES_SYMBOLS
        self.now = now or datetime.now(timezone.utc)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def build_async(
        self,
        *,
        current_positions: Optional[Mapping[str, Position]] = None,
        current_cash: float = 0.0,
    ) -> ContextResult:
        """
        Build a market context asynchronously. Bar and tick loading are
        executed concurrently.

        Parameters
        ----------
        current_positions : Mapping[str, Position], optional
            Existing portfolio positions. Used to compute current notional
            exposure per symbol. If omitted, an empty dict is assumed.
        current_cash : float, default 0.0
            Cash balance used when constructing ``PortfolioSnapshot``.

        Returns
        -------
        ContextResult
            The assembled market context along with aggregated metrics and
            evidence of the build.
        """
        current_positions = current_positions or {}

        # Determine the universe of symbols to load: existing positions,
        # available bars, and explicit futures
        symbols_from_positions = set(current_positions.keys())
        available_bar_symbols = set(self.bars_provider.available_symbols())
        universe: List[str] = []
        # Existing positions first to preserve order
        for s in symbols_from_positions:
            if s not in universe:
                universe.append(s)
        # Then bars symbols
        for s in available_bar_symbols:
            if s not in universe:
                universe.append(s)
        # Then explicit futures
        for s in self.futures_symbols:
            if s not in universe:
                universe.append(s)
        # Then crypto symbols (24/7 — sourced from runtime/kraken_prices.json)
        for s in CRYPTO_SYMBOLS:
            if s not in universe:
                universe.append(s)

        # Asynchronously load bars and ticks
        bars = await self.bars_provider.load_bars(universe)
        ticks = await self.ticks_provider.load_ticks(universe, bars)

        # Compute last prices from ticks; fallback to last bar close
        prices: Dict[str, float] = {}
        for symbol in universe:
            if symbol in ticks:
                prices[symbol] = float(ticks[symbol]["price"])
            else:
                series = bars.get(symbol) or []
                if series:
                    price = series[-1].get("close")
                    if price is not None:
                        prices[symbol] = float(price)

        # Merge Kraken crypto spot prices (24/7 feed) — fail-closed if stale/missing
        kraken_evidence = self._merge_kraken_prices(prices)

        # Compute notionals
        current_symbol_notional: Dict[str, float] = {}
        for symbol, pos in current_positions.items():
            price = prices.get(symbol)
            if price is not None:
                current_symbol_notional[symbol] = pos.quantity * price

        current_total_notional = sum(current_symbol_notional.values())

        # Load portfolio equity from dynamic_caps.json (authoritative source)
        total_equity = self._load_equity_from_dynamic_caps()

        # Wrap ticks as attribute-accessible objects for strategy compatibility
        # (strategies access tick.price, not tick["price"])
        wrapped_ticks = self._wrap_ticks(ticks)

        # Build portfolio object with required fields for strategy handlers
        portfolio = self._build_portfolio_for_context(
            total_equity=total_equity,
            positions=current_positions or {},
        )

        # Load legend consensus (top-stocks weights). Beta short-circuits to
        # an empty signal list when ctx.legend is None; alpha and delta also
        # consult ctx.legend.weights for universe selection. Hardcoding None
        # here was the root cause of the equity-strategy "no_signal" pandemic.
        # Fail-closed to None on any loader error so context construction
        # remains robust if the legend file is missing or malformed.
        legend_consensus = None
        legend_load_error: Optional[str] = None
        if load_legend is not None:
            try:
                legend_consensus = load_legend()
            except LegendLoaderError as exc:
                legend_load_error = str(exc)
                self.logger.warning("legend_load_failed: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive
                legend_load_error = f"unexpected:{exc}"
                self.logger.warning("legend_load_unexpected_error: %s", exc)

        # Build context
        context = MarketContext(
            ticks=wrapped_ticks,
            bars=bars,
            prices=prices,
            now=self.now,
            portfolio=portfolio,
            legend=legend_consensus,
        )

        # Evidence for auditability
        evidence = {
            "requested_symbols": universe,
            "bars_found": {s: len(bars.get(s, [])) for s in universe},
            "ticks_found": {s: (s in ticks) for s in universe},
            "kraken_merge": kraken_evidence,
            "legend_loaded": legend_consensus is not None,
            "legend_symbols_count": (
                len(legend_consensus.weights) if legend_consensus is not None else 0
            ),
            "legend_load_error": legend_load_error,
        }

        return ContextResult(
            context=context,
            prices=prices,
            current_symbol_notional=current_symbol_notional,
            current_total_notional=current_total_notional,
            evidence=evidence,
        )

    def build(
        self,
        *,
        current_positions: Optional[Mapping[str, Position]] = None,
        current_cash: float = 0.0,
    ) -> ContextResult:
        """
        Synchronous wrapper around :func:`build_async`. If running inside an
        existing event loop, it will create a new one temporarily.
        """
        try:
            return asyncio.run(self.build_async(current_positions=current_positions, current_cash=current_cash))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(
                self.build_async(current_positions=current_positions, current_cash=current_cash)
            )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _merge_kraken_prices(self, prices: Dict[str, float]) -> Dict[str, object]:
        """
        Merge Kraken crypto spot prices into the supplied prices dict in-place.

        Reads ``runtime/kraken_prices.json``, validates freshness against
        ``KRAKEN_MAX_AGE_SECONDS``, and merges the canonical
        :data:`CRYPTO_SYMBOLS` into ``prices``.

        Fail-closed: any missing file, malformed JSON, stale feed, or unparseable
        timestamp results in *no* mutation to ``prices`` and an evidence record
        explaining why. Never raises.
        """
        evidence: Dict[str, object] = {
            "merged": False,
            "merged_symbols": [],
            "reason": None,
            "feed_age_seconds": None,
            "feed_ts_utc": None,
        }
        try:
            path = Path(KRAKEN_PRICES_PATH)
            if not path.is_file():
                evidence["reason"] = "file_missing"
                self.logger.warning("Kraken prices file missing at %s", path)
                return evidence
            with path.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            evidence["reason"] = f"read_error: {type(exc).__name__}"
            self.logger.warning("Failed to read kraken_prices.json: %s", exc)
            return evidence

        ts_raw = doc.get("ts_utc")
        if not isinstance(ts_raw, str):
            evidence["reason"] = "missing_ts_utc"
            self.logger.warning("kraken_prices.json missing ts_utc")
            return evidence
        try:
            # tolerate trailing 'Z'
            ts_clean = ts_raw[:-1] if ts_raw.endswith("Z") else ts_raw
            feed_ts = datetime.fromisoformat(ts_clean).replace(tzinfo=timezone.utc)
        except ValueError:
            evidence["reason"] = "bad_ts_format"
            self.logger.warning("kraken_prices.json bad ts_utc: %s", ts_raw)
            return evidence

        age_seconds = (datetime.now(timezone.utc) - feed_ts).total_seconds()
        evidence["feed_ts_utc"] = ts_raw
        evidence["feed_age_seconds"] = age_seconds
        if age_seconds > KRAKEN_MAX_AGE_SECONDS:
            evidence["reason"] = "stale"
            self.logger.warning(
                "kraken_prices.json stale: age=%.1fs > %ds",
                age_seconds, KRAKEN_MAX_AGE_SECONDS,
            )
            return evidence

        kraken_prices = doc.get("prices")
        if not isinstance(kraken_prices, dict):
            evidence["reason"] = "missing_prices"
            self.logger.warning("kraken_prices.json missing prices map")
            return evidence

        merged: List[str] = []
        for sym in CRYPTO_SYMBOLS:
            v = kraken_prices.get(sym)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv <= 0.0 or fv != fv:  # reject zero/NaN
                continue
            prices[sym] = fv
            merged.append(sym)

        evidence["merged"] = bool(merged)
        evidence["merged_symbols"] = merged
        if not merged:
            evidence["reason"] = "no_valid_symbols"
        return evidence

    def _construct_portfolio_snapshot(
        self,
        *,
        positions: Mapping[str, Position],
        cash: float,
        total_equity: float,
        now: datetime,
    ) -> PortfolioSnapshot:
        """
        Attempt to build a ``PortfolioSnapshot`` instance in a way that
        accommodates variations in the local class constructor. The base
        version of CHAD defines a dataclass that accepts ``timestamp``,
        ``positions``, and ``cash``. Some forks include ``total_equity`` or
        ``equity`` fields. This method uses introspection to detect
        supported fields and only passes values that are accepted.
        """
        try:
            import inspect
            sig = inspect.signature(PortfolioSnapshot)  # type: ignore
            params = sig.parameters
            kwargs: Dict[str, object] = {}
            if "timestamp" in params:
                kwargs["timestamp"] = now
            if "positions" in params:
                kwargs["positions"] = positions
            if "cash" in params:
                kwargs["cash"] = float(cash)
            # Pass equity if supported
            if "equity" in params:
                kwargs["equity"] = float(total_equity)
            if "total_equity" in params:
                kwargs["total_equity"] = float(total_equity)
            return PortfolioSnapshot(**kwargs)  # type: ignore
        except Exception:
            # Fallback: assume minimal dataclass signature
            return PortfolioSnapshot(
                timestamp=now, positions=positions, cash=float(cash)
            )  # type: ignore

    _DYNAMIC_CAPS_PATH = Path("/home/ubuntu/chad_finale/runtime/dynamic_caps.json")

    def _load_equity_from_dynamic_caps(self) -> float:
        """
        Read total_equity from runtime/dynamic_caps.json.

        This is the authoritative equity source maintained by the dynamic
        risk allocator.  Returns 0.0 on any failure (caller should treat
        as missing, not as zero equity).
        """
        try:
            if not self._DYNAMIC_CAPS_PATH.is_file():
                self.logger.warning("dynamic_caps.json not found — equity unknown")
                return 0.0
            data = json.loads(self._DYNAMIC_CAPS_PATH.read_text(encoding="utf-8"))
            equity = float(data.get("total_equity", 0.0))
            if equity > 0:
                return equity
            self.logger.warning("dynamic_caps.json total_equity is %s", equity)
            return 0.0
        except Exception as exc:
            self.logger.warning("Failed to read dynamic_caps.json: %s", exc)
            return 0.0

    @staticmethod
    def _wrap_ticks(
        ticks: Dict[str, Mapping[str, float]],
    ) -> Dict[str, object]:
        """
        Wrap tick dicts as SimpleNamespace objects so that strategy
        handlers can access tick.price (attribute access) instead of
        tick["price"] (dict access).
        """
        from types import SimpleNamespace

        wrapped: Dict[str, object] = {}
        for symbol, tick_data in ticks.items():
            wrapped[symbol] = SimpleNamespace(**dict(tick_data))
        return wrapped

    def _build_portfolio_for_context(
        self,
        *,
        total_equity: float,
        positions: Mapping[str, object],
    ) -> object:
        """
        Build a portfolio object compatible with strategy handler expectations.

        Strategies access:
          - portfolio.cash
          - portfolio.positions
          - portfolio.extra["equity"], portfolio.extra["equity_peak"]
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            timestamp=self.now,
            cash=total_equity,
            positions=dict(positions),
            total_equity=total_equity,
            equity=total_equity,
            net_liq=total_equity,
            extra={"equity": total_equity, "equity_peak": total_equity},
        )
