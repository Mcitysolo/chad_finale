"""
Kraken trusted-fill + lunch-money-sizing config loader (STRICT / fail-closed).

Backs:
  * the trusted paper-fill engine  (taker fee + slippage model)
  * the min-size bump/skip sizing decision (per-pair Kraken minima)

Contract (mirrors chad/risk/margin_block.load_frozen_config):
  * unknown non-underscore key         -> KrakenTradingConfigError
  * missing required (non-underscore) key -> KrakenTradingConfigError
  * wrong leaf type                    -> KrakenTradingConfigError
  * keys beginning "_" are documentation and ignored
No env overrides. A typo'd threshold must NOT silently fall back to a code
default — that silent drift is exactly what strict validation prevents.

Source file: config/kraken_trading.json (schema_version "kraken_trading.v1").
Rows marked operator_verify:true carry documented public values that could not
be checked offline against the live Kraken account and must be operator-verified
before any enforce/live promotion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_REPO_ROOT = Path("/home/ubuntu/chad_finale")
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "kraken_trading.json"
_SCHEMA_VERSION = "kraken_trading.v1"


class KrakenTradingConfigError(ValueError):
    """Raised on any invalid/missing/unreadable Kraken trading config."""


# Type-token schema. "num" = int|float (finite); bool is NOT accepted as num.
# A nested dict value is a sub-schema; "map:num" = an open mapping of str->num.
_SCHEMA: Dict[str, Any] = {
    "schema_version": "str",
    "frozen_utc": "str",
    "taker_fee": {
        "operator_verify": "bool",
        "default_taker_bps": "num",
        "taker_bps_by_pair": "map:num",
    },
    "slippage_model": {
        "operator_verify": "bool",
        "slippage_impact_floor_bps": "num",
        "max_tick_age_seconds": "num",
    },
    "min_order_size_by_pair": {
        "operator_verify": "bool",
        "default_min_volume": "num",
        "min_volume_by_pair": "map:num",
    },
}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v)


def _finite(v: Any) -> bool:
    try:
        f = float(v)
        return f == f and f not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False


def _check_leaf(token: str, value: Any, path: str) -> None:
    if token == "str":
        if not isinstance(value, str):
            raise KrakenTradingConfigError(f"{path}: expected str, got {type(value).__name__}")
    elif token == "bool":
        if not isinstance(value, bool):
            raise KrakenTradingConfigError(f"{path}: expected bool, got {type(value).__name__}")
    elif token == "num":
        if not _is_num(value):
            raise KrakenTradingConfigError(f"{path}: expected finite number, got {value!r}")
    elif token == "map:num":
        if not isinstance(value, Mapping):
            raise KrakenTradingConfigError(f"{path}: expected mapping, got {type(value).__name__}")
        for k, v in value.items():
            if str(k).startswith("_"):
                continue
            if not _is_num(v):
                raise KrakenTradingConfigError(f"{path}.{k}: expected finite number, got {v!r}")
    else:  # pragma: no cover - defensive
        raise KrakenTradingConfigError(f"{path}: unknown schema token {token!r}")


def _validate(raw: Any, schema: Any, path: str) -> None:
    if not isinstance(raw, Mapping):
        raise KrakenTradingConfigError(f"{path or '<root>'}: expected object, got {type(raw).__name__}")
    if not isinstance(schema, Mapping):  # pragma: no cover - defensive
        raise KrakenTradingConfigError(f"{path}: bad schema")
    # unknown non-underscore keys -> reject
    for key in raw.keys():
        if str(key).startswith("_"):
            continue
        if key not in schema:
            raise KrakenTradingConfigError(f"{path}.{key}: unknown key (strict validation)")
    # missing required keys / recurse
    for key, sub in schema.items():
        if key not in raw:
            raise KrakenTradingConfigError(f"{path}.{key}: missing required key")
        child = raw[key]
        child_path = f"{path}.{key}" if path else key
        if isinstance(sub, Mapping):
            _validate(child, sub, child_path)
        else:
            _check_leaf(sub, child, child_path)


@dataclass(frozen=True)
class KrakenTradingConfig:
    raw: Mapping[str, Any]

    # -- taker fee --
    def taker_bps(self, pair: str) -> float:
        fee = self.raw["taker_fee"]
        by_pair = fee.get("taker_bps_by_pair", {}) or {}
        v = by_pair.get(pair)
        if _is_num(v):
            return float(v)
        return float(fee["default_taker_bps"])

    def taker_fee(self, notional: float, pair: str) -> float:
        """Taker fee on a fill of |notional| (quote ccy). Always >= 0."""
        try:
            n = abs(float(notional))
        except (TypeError, ValueError):
            return 0.0
        return n * self.taker_bps(pair) / 1e4

    # -- slippage model --
    @property
    def slippage_impact_floor_bps(self) -> float:
        return float(self.raw["slippage_model"]["slippage_impact_floor_bps"])

    @property
    def max_tick_age_seconds(self) -> float:
        return float(self.raw["slippage_model"]["max_tick_age_seconds"])

    # -- min order size --
    def min_volume(self, pair: str) -> float:
        section = self.raw["min_order_size_by_pair"]
        by_pair = section.get("min_volume_by_pair", {}) or {}
        v = by_pair.get(pair)
        if _is_num(v):
            return float(v)
        return float(section["default_min_volume"])


def load_kraken_trading_config(path: Optional[Path] = None) -> KrakenTradingConfig:
    """Load + strictly validate the config. Raises KrakenTradingConfigError on any problem."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KrakenTradingConfigError(f"unreadable config {cfg_path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KrakenTradingConfigError(f"invalid JSON in {cfg_path}: {exc}") from exc
    _validate(raw, _SCHEMA, "")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise KrakenTradingConfigError(
            f"schema_version mismatch: expected {_SCHEMA_VERSION!r}, got {raw.get('schema_version')!r}"
        )
    return KrakenTradingConfig(raw=raw)


_CACHED: Optional[KrakenTradingConfig] = None


def get_kraken_trading_config(path: Optional[Path] = None) -> KrakenTradingConfig:
    """Process-cached accessor (default path only). Tests pass an explicit path to bypass the cache."""
    global _CACHED
    if path is not None:
        return load_kraken_trading_config(path)
    if _CACHED is None:
        _CACHED = load_kraken_trading_config()
    return _CACHED
