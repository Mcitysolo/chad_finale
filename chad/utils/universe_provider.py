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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from chad.utils.legend_loader import load_legend, LegendLoaderError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "universe.json"
DEFAULT_FALLBACK: List[str] = ["SPY", "QQQ"]


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


def get_trade_universe(cfg: UniverseConfig | None = None) -> List[str]:
    """
    Return the configured universe.

    - STATIC: returns cfg.static_symbols
    - LEGEND_TOP_N: returns top N symbols by legend weights, falling back safely.

    Deterministic ordering:
    - LEGEND_TOP_N uses sorted legend weights descending (ties stable by symbol).
    """
    c = cfg or load_universe_config()

    if c.mode == "STATIC":
        return list(c.static_symbols) if c.static_symbols else list(c.fallback)

    if c.mode == "LEGEND_TOP_N":
        try:
            legend = load_legend()
            items = sorted(
                legend.weights.items(),
                key=lambda kv: (-float(kv[1]), str(kv[0])),
            )
            syms = [sym for sym, _w in items[: max(0, int(c.top_n))]]
            syms = _normalize_symbols(syms)
            return syms if syms else list(c.fallback)
        except (LegendLoaderError, Exception):
            return list(c.fallback)

    return list(c.fallback)
