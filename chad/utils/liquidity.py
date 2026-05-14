"""CHAD shared liquidity classifier — float-aware ADDV gating (Phase A Item 5).

Classifies an equity/ETF symbol into LARGE / STANDARD / THIN / UNKNOWN based
on the most recent average daily dollar volume (ADDV = mean(close * volume))
computed from NDJSON daily-bar files.

Design contract
---------------
* Pure utility — no broker, adapter, execution, or strategy imports.
* Stdlib-only IO. Files are read from
  ``<repo_root>/data/daily_bars/<SYMBOL>.ndjson`` (with fallbacks).
* Fail-open on every error path: missing files, unreadable files,
  malformed NDJSON, and insufficient data all resolve to
  :class:`LiquidityClass.UNKNOWN`, which is treated as "pass" by the
  strategy gate.
* Strategy gates consume :func:`blocks_thin_entry` for **entry** signals
  only. Exits, stop-losses, and position reductions must not consult
  this module.
* The classification cache is keyed by ``(normalized_symbol, repo_root)``
  so tests using temp paths cannot contaminate production lookups.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("chad.utils.liquidity")


class LiquidityClass(str, Enum):
    LARGE = "large"
    STANDARD = "standard"
    THIN = "thin"
    UNKNOWN = "unknown"


_LARGE_THRESHOLD_USD: float = 500_000_000.0
_STANDARD_THRESHOLD_USD: float = 50_000_000.0
_ADV_LOOKBACK_DAYS: int = 20
_MIN_BARS_REQUIRED: int = 5
THIN_MIN_CONFIDENCE: float = 0.80


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _resolve_root(repo_root: Optional[Path]) -> Path:
    if repo_root is None:
        return _project_root()
    return Path(repo_root)


def _bars_path(symbol: str, repo_root: Optional[Path] = None) -> Path:
    """Return the first existing bar-file candidate, else the primary."""
    root = _resolve_root(repo_root)
    sym = _normalize_symbol(symbol)
    candidates = [
        root / "data" / "bars" / "1d" / f"{sym}.json",
        root / "data" / "daily_bars" / f"{sym}.ndjson",
        root / "data" / "daily_bars" / f"{sym}_daily.ndjson",
        root / "data" / f"{sym}_daily_bars.ndjson",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _parse_bar_record(rec: object) -> Optional[Tuple[float, float]]:
    if not isinstance(rec, dict):
        return None
    close_val = rec.get("close", rec.get("c"))
    volume_val = rec.get("volume", rec.get("v", rec.get("vol")))
    try:
        c = float(close_val)
        v = float(volume_val)
    except (TypeError, ValueError):
        return None
    if c != c or v != v:
        return None
    if c <= 0 or v <= 0:
        return None
    return c, v


def _compute_addv(
    symbol: str,
    repo_root: Optional[Path] = None,
) -> Optional[float]:
    """Read up to the last ``_ADV_LOOKBACK_DAYS`` valid bars and return the
    average daily dollar volume. Returns ``None`` on any failure or when
    fewer than ``_MIN_BARS_REQUIRED`` valid bars are found.
    """
    path = _bars_path(symbol, repo_root=repo_root)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOG.debug("liquidity: unreadable bars path=%s err=%s", path, exc)
        return None

    pairs: List[Tuple[float, float]] = []
    bars_iter: Optional[List[object]] = None
    try:
        whole = json.loads(text)
    except (ValueError, TypeError):
        whole = None
    if isinstance(whole, dict):
        candidate = whole.get("bars", whole.get("data"))
        if isinstance(candidate, list):
            bars_iter = list(candidate)
    elif isinstance(whole, list):
        bars_iter = list(whole)

    if bars_iter is not None:
        for rec in bars_iter:
            parsed = _parse_bar_record(rec)
            if parsed is None:
                continue
            pairs.append(parsed)
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except (ValueError, TypeError) as exc:
                LOG.debug(
                    "liquidity: skipped malformed NDJSON line path=%s err=%s",
                    path, exc,
                )
                continue
            parsed = _parse_bar_record(rec)
            if parsed is None:
                continue
            pairs.append(parsed)

    if len(pairs) < _MIN_BARS_REQUIRED:
        return None

    window = pairs[-_ADV_LOOKBACK_DAYS:]
    total = 0.0
    for c, v in window:
        total += c * v
    return total / float(len(window))


_CACHE: Dict[Tuple[str, str], LiquidityClass] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _cache_key(symbol: str, repo_root: Optional[Path]) -> Tuple[str, str]:
    sym = _normalize_symbol(symbol)
    root = _resolve_root(repo_root)
    try:
        root_str = str(root.resolve())
    except OSError:
        root_str = str(root)
    return (sym, root_str)


def classify(
    symbol: str,
    *,
    repo_root: Optional[Path] = None,
    refresh: bool = False,
) -> LiquidityClass:
    """Classify ``symbol`` by ADDV. Fail-open: missing/unreadable -> UNKNOWN."""
    key = _cache_key(symbol, repo_root)
    if not refresh:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    addv = _compute_addv(symbol, repo_root=repo_root)
    if addv is None:
        cls = LiquidityClass.UNKNOWN
    elif addv >= _LARGE_THRESHOLD_USD:
        cls = LiquidityClass.LARGE
    elif addv >= _STANDARD_THRESHOLD_USD:
        cls = LiquidityClass.STANDARD
    else:
        cls = LiquidityClass.THIN

    _CACHE[key] = cls
    return cls


def blocks_thin_entry(
    symbol: str,
    confidence: float,
    *,
    repo_root: Optional[Path] = None,
    thin_min_confidence: float = THIN_MIN_CONFIDENCE,
) -> Tuple[bool, LiquidityClass, float]:
    """Return ``(blocked, liquidity_class, required_confidence)``.

    A THIN symbol is blocked only when the signal confidence falls below
    the absolute ``thin_min_confidence`` floor (default 0.80). LARGE /
    STANDARD / UNKNOWN classifications never block — UNKNOWN is the
    fail-open path for symbols with no daily-bar data.
    """
    cls = classify(symbol, repo_root=repo_root)
    if cls is LiquidityClass.THIN:
        floor = float(thin_min_confidence)
        if float(confidence) < floor:
            return True, LiquidityClass.THIN, floor
        return False, LiquidityClass.THIN, floor
    return False, cls, float(confidence)


__all__ = [
    "LiquidityClass",
    "THIN_MIN_CONFIDENCE",
    "classify",
    "clear_cache",
    "blocks_thin_entry",
]
