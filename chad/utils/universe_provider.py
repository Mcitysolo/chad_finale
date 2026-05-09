#!/usr/bin/env python3
"""
CHAD — Universe Provider (Production)

Single source of truth for "what symbols a strategy should consider".

Design goals:
- Config-driven (no strategy hardcoding)
- Safe fallback when config/legend is missing
- Deterministic ordering
- Minimal surface area (no network I/O)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chad.utils.legend_loader import load_legend, LegendLoaderError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "universe.json"
DEFAULT_RUNTIME_PATH = ROOT / "runtime" / "universe.json"
DEFAULT_FALLBACK: List[str] = ["SPY", "QQQ"]

# Live-screened runtime/universe.json is preferred when it has been refreshed
# within this freshness window. Beyond it, we fall back to the operator-
# controlled config/universe.json. 5400s = 90min covers an intraday gap
# (long enough that a single missed builder run does not flap the universe,
# short enough that a stuck builder cannot pin a stale list overnight).
RUNTIME_FRESHNESS_SECONDS: int = 5400

_LOG = logging.getLogger("chad.utils.universe_provider")


@dataclass(frozen=True)
class UniverseConfig:
    mode: str  # "STATIC" | "LEGEND_TOP_N"
    top_n: int
    static_symbols: List[str]
    fallback: List[str]

    @staticmethod
    def default() -> "UniverseConfig":
        return UniverseConfig(
            mode="STATIC",
            top_n=0,
            static_symbols=list(DEFAULT_FALLBACK),
            fallback=list(DEFAULT_FALLBACK),
        )


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in symbols:
        sym = str(s).strip().upper()
        if not sym:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def load_universe_config(path: Path = DEFAULT_CONFIG_PATH) -> UniverseConfig:
    """
    Load config/universe.json if present. If missing or invalid, return defaults.

    Schema:
      {
        "mode": "LEGEND_TOP_N" | "STATIC",
        "top_n": 25,
        "static_symbols": ["SPY","QQQ"],
        "fallback": ["SPY","QQQ"]
      }
    """
    try:
        if not path.is_file():
            return UniverseConfig.default()

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return UniverseConfig.default()

        mode = str(raw.get("mode", "STATIC")).strip().upper()
        top_n = int(raw.get("top_n", 0) or 0)

        static_symbols = _normalize_symbols(raw.get("static_symbols") or [])
        fallback = _normalize_symbols(raw.get("fallback") or DEFAULT_FALLBACK)

        if mode not in {"STATIC", "LEGEND_TOP_N"}:
            mode = "STATIC"
        if top_n < 0:
            top_n = 0

        if not static_symbols:
            static_symbols = list(fallback) if fallback else list(DEFAULT_FALLBACK)
        if not fallback:
            fallback = list(DEFAULT_FALLBACK)

        return UniverseConfig(
            mode=mode,
            top_n=top_n,
            static_symbols=static_symbols,
            fallback=fallback,
        )
    except Exception:
        return UniverseConfig.default()


_RUNTIME_SHAPE_KEYS: Tuple[str, ...] = ("symbols", "universe", "tickers")


def _extract_symbol_list(raw: object) -> Optional[List[str]]:
    """Pull the symbol list out of a parsed JSON document.

    Supported shapes:
      - {"symbols":  [...]}
      - {"universe": [...]}
      - {"tickers":  [...]}
      - [...]  (a bare list)

    Returns the normalized list, or None when no usable list is present.
    """
    if isinstance(raw, list):
        normalized = _normalize_symbols(raw)
        return normalized if normalized else None
    if isinstance(raw, dict):
        for key in _RUNTIME_SHAPE_KEYS:
            v = raw.get(key)
            if isinstance(v, list):
                normalized = _normalize_symbols(v)
                if normalized:
                    return normalized
    return None


def _load_runtime_universe_with_meta(
    path: Optional[Path] = None,
    max_age_s: int = RUNTIME_FRESHNESS_SECONDS,
    now: Optional[float] = None,
) -> Tuple[Optional[List[str]], str, float]:
    """Read runtime/universe.json with reason classification.

    Returns ``(symbols_or_none, reason, age_s)`` where reason is one of:
      - "fresh"     — file present, parses cleanly, within freshness window
      - "missing"   — file absent
      - "stale"     — file present but mtime older than freshness window
      - "malformed" — file present but unparseable / wrong shape / no symbols

    A None ``symbols`` always pairs with a non-"fresh" reason. Pure: no logging.
    """
    if path is None:
        path = DEFAULT_RUNTIME_PATH
    try:
        if not path.is_file():
            return None, "missing", -1.0
    except Exception:
        return None, "missing", -1.0

    try:
        ref_now = float(now) if now is not None else time.time()
        mtime = path.stat().st_mtime
        age_s = max(0.0, ref_now - mtime)
    except Exception:
        return None, "malformed", -1.0

    if age_s >= max_age_s:
        return None, "stale", age_s

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "malformed", age_s

    symbols = _extract_symbol_list(raw)
    if not symbols:
        return None, "malformed", age_s
    return symbols, "fresh", age_s


def _load_fresh_runtime_symbols(
    path: Optional[Path] = None,
    max_age_s: int = RUNTIME_FRESHNESS_SECONDS,
    now: Optional[float] = None,
) -> Optional[List[str]]:
    """Return the live-screened symbol list from runtime/universe.json when fresh.

    "Fresh" means the file exists, parses cleanly, has a non-empty symbol list,
    and its mtime is within `max_age_s` of `now` (defaulting to wall clock).
    Returns None when the file is missing, stale, malformed, or empty —
    callers fall back to the static config in that case.

    Thin wrapper around :func:`_load_runtime_universe_with_meta` for callers
    that only need the list, not the reason metadata.
    """
    symbols, _reason, _age = _load_runtime_universe_with_meta(
        path=path, max_age_s=max_age_s, now=now,
    )
    return symbols


def _load_static_config_symbols(
    path: Optional[Path] = None,
) -> Optional[List[str]]:
    """Read config/universe.json directly as a symbol-list document.

    The legacy ``UniverseConfig`` schema (``mode``/``static_symbols``) is the
    primary way to drive this module, but the operator-controlled
    config/universe.json on disk uses the simpler ``{"symbols":[...]}`` shape.
    This helper lets the fallback path return that list verbatim instead of
    collapsing to the ``["SPY","QQQ"]`` ``DEFAULT_FALLBACK`` when no
    ``static_symbols`` key is present.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    try:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _extract_symbol_list(raw)


@dataclass(frozen=True)
class UniverseLoad:
    """Result of resolving the active universe with full source provenance."""
    symbols: List[str]
    source_path: str
    source_type: str  # "runtime" | "config" | "legend" | "default_fallback"
    symbol_count: int
    stale: bool
    reason: str       # "fresh" | "missing" | "stale" | "malformed"

    def as_dict(self) -> Dict[str, object]:
        return {
            "source_path": self.source_path,
            "source_type": self.source_type,
            "symbol_count": self.symbol_count,
            "stale": self.stale,
            "reason": self.reason,
        }


def load_active_universe(
    cfg: UniverseConfig | None = None,
    *,
    runtime_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    freshness_s: int = RUNTIME_FRESHNESS_SECONDS,
    now: Optional[float] = None,
) -> UniverseLoad:
    """Resolve the active universe with full source provenance.

    Preference order:
      1. runtime/universe.json (live-screened) when fresh.
      2. cfg.static_symbols (STATIC mode) or legend top-N (LEGEND_TOP_N).
      3. config/universe.json read as a bare symbol-list document.
      4. cfg.fallback / DEFAULT_FALLBACK as last resort.

    When runtime preference cannot be applied, a single structured warning
    ``universe_provider.using_static_fallback`` is logged with the reason
    (missing | stale | malformed) so operators can correlate strategy
    universe drift with a stalled universe_builder timer.
    """
    if runtime_path is None:
        runtime_path = DEFAULT_RUNTIME_PATH
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    runtime_symbols, runtime_reason, runtime_age = _load_runtime_universe_with_meta(
        path=runtime_path, max_age_s=freshness_s, now=now,
    )
    if runtime_symbols and runtime_reason == "fresh":
        return UniverseLoad(
            symbols=list(runtime_symbols),
            source_path=str(runtime_path),
            source_type="runtime",
            symbol_count=len(runtime_symbols),
            stale=False,
            reason="fresh",
        )

    _LOG.warning(
        "universe_provider.using_static_fallback runtime_path=%s reason=%s age_s=%.1f freshness_s=%d",
        str(runtime_path),
        runtime_reason,
        runtime_age,
        int(freshness_s),
    )

    c = cfg or load_universe_config(config_path)
    fallback_reason = runtime_reason  # carries why we fell back

    if c.mode == "LEGEND_TOP_N":
        try:
            legend = load_legend()
            items = sorted(
                legend.weights.items(),
                key=lambda kv: (-float(kv[1]), str(kv[0])),
            )
            syms = _normalize_symbols(
                [sym for sym, _w in items[: max(0, int(c.top_n))]]
            )
            if syms:
                return UniverseLoad(
                    symbols=syms,
                    source_path="legend",
                    source_type="legend",
                    symbol_count=len(syms),
                    stale=True,
                    reason=fallback_reason,
                )
        except (LegendLoaderError, Exception):
            pass

    # STATIC (or LEGEND_TOP_N that produced nothing): prefer the loaded config's
    # static_symbols, then re-read config/universe.json as a bare symbol-list,
    # then the hard-coded fallback.
    if c.static_symbols and c.static_symbols != list(DEFAULT_FALLBACK):
        return UniverseLoad(
            symbols=list(c.static_symbols),
            source_path=str(config_path),
            source_type="config",
            symbol_count=len(c.static_symbols),
            stale=True,
            reason=fallback_reason,
        )

    config_symbols = _load_static_config_symbols(config_path)
    if config_symbols:
        return UniverseLoad(
            symbols=list(config_symbols),
            source_path=str(config_path),
            source_type="config",
            symbol_count=len(config_symbols),
            stale=True,
            reason=fallback_reason,
        )

    fb = list(c.fallback) if c.fallback else list(DEFAULT_FALLBACK)
    return UniverseLoad(
        symbols=fb,
        source_path="default_fallback",
        source_type="default_fallback",
        symbol_count=len(fb),
        stale=True,
        reason=fallback_reason,
    )


def get_trade_universe(
    cfg: UniverseConfig | None = None,
    *,
    runtime_path: Optional[Path] = None,
    freshness_s: int = RUNTIME_FRESHNESS_SECONDS,
    now: Optional[float] = None,
) -> List[str]:
    """
    Return the configured universe (just the symbol list).

    Thin compatibility wrapper around :func:`load_active_universe` — preserved
    so existing strategy callers keep their ``List[str]`` return shape.
    """
    return list(
        load_active_universe(
            cfg=cfg,
            runtime_path=runtime_path,
            freshness_s=freshness_s,
            now=now,
        ).symbols
    )
