#!/usr/bin/env python3
"""
CHAD Paper Execution Evidence Writer
====================================

Compatibility-first, production-grade replacement for:
    chad/execution/paper_exec_evidence_writer.py

Primary goals
-------------
1. Preserve import contract expected by:
       /usr/local/bin/chad_paper_trade_executor.py

   Specifically:
       from chad.execution.paper_exec_evidence_writer import (
           PaperExecEvidence,
           write_paper_exec_evidence,
       )

2. Accept legacy/future keyword arguments without crashing.

3. Write append-only, hash-chained evidence to:
   - data/fills/FILLS_YYYYMMDD.ndjson
   - data/fees/FEES_YYYYMMDD.ndjson
   - data/execution_metrics/EXECUTION_METRICS_YYYYMMDD.ndjson

4. Resolve true strategy attribution using this precedence:
   A. explicit strategy, if real
   B. explicit source_strategies, if real
   C. planner artifact by symbol (runtime/full_execution_cycle_last.json)
   D. tags fallback
   E. raise StrategyAttributionError — never silently flatten to paper_exec

5. Never allow placeholder values like:
   - "unknown"
   - "manual"
   - "paper_exec"
   to outrank a real strategy when one is available.
   If no real strategy can be resolved at all, raise rather than
   silently writing paper_exec as primary_strategy.

Python
------
3.10+
No third-party dependencies.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class StrategyAttributionError(RuntimeError):
    """Raised when no real strategy can be resolved for a paper execution record."""


# =============================================================================
# Paths / constants
# =============================================================================

ROOT = Path("/home/ubuntu/chad_finale")
DATA_DIR = ROOT / "data"
FILLS_DIR = DATA_DIR / "fills"
FEES_DIR = DATA_DIR / "fees"
EXEC_METRICS_DIR = DATA_DIR / "execution_metrics"
RUNTIME_DIR = ROOT / "runtime"
LOCKS_DIR = RUNTIME_DIR / "locks"

PLAN_ARTIFACT_DEFAULT = RUNTIME_DIR / "full_execution_cycle_last.json"
GENESIS = "GENESIS"

FILL_SCHEMA_VERSION = "paper_exec_fill.v4"
FEE_SCHEMA_VERSION = "paper_exec_fee.v4"
EXEC_METRIC_SCHEMA_VERSION = "paper_exec_execution_metric.v4"

PLACEHOLDER_STRATEGIES = {"", "unknown", "manual", "paper_exec"}
REAL_STRATEGIES = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "omega",
    "alpha_crypto",
    "alpha_forex",
}


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(ts: Optional[datetime] = None) -> str:
    value = ts or _utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ymd(ts: Optional[datetime] = None) -> str:
    value = ts or _utc_now()
    return value.astimezone(timezone.utc).strftime("%Y%m%d")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        return s if s else default
    s = str(value).strip()
    return s if s else default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_strategy(value: Any, default: str = "") -> str:
    return _safe_str(value, default).lower()


def _is_real_strategy(value: Any) -> bool:
    s = _normalize_strategy(value, "")
    return bool(s) and s not in PLACEHOLDER_STRATEGIES


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        s = _safe_str(item).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _filter_real_strategies(values: Iterable[Any]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        s = _normalize_strategy(value, "")
        if _is_real_strategy(s):
            cleaned.append(s)
    return _dedupe_keep_order(cleaned)


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_nonempty_line(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    last: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                last = line.rstrip("\n")
    return last


def _append_hash_chained_record(path: Path, payload: Mapping[str, Any]) -> Dict[str, str]:
    _ensure_dir(path.parent)
    _ensure_dir(LOCKS_DIR)

    lock_path = LOCKS_DIR / f"{path.name}.lock"

    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

        last_line = _last_nonempty_line(path)
        if last_line:
            try:
                last_obj = json.loads(last_line)
                prev_hash = _safe_str(last_obj.get("record_hash"), GENESIS)
                prev_seq = int(last_obj.get("sequence_id", 0))
            except Exception:
                prev_hash = GENESIS
                prev_seq = 0
        else:
            prev_hash = GENESIS
            prev_seq = 0

        record: Dict[str, Any] = {
            "payload": dict(payload),
            "prev_hash": prev_hash,
            "sequence_id": prev_seq + 1,
            "timestamp_utc": _iso_z(),
        }
        record["record_hash"] = _hash_text(_canonical_json(record))

        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            os.fsync(out.fileno())

        return {
            "path": str(path),
            "sequence_id": str(record["sequence_id"]),
            "record_hash": record["record_hash"],
        }


# =============================================================================
# Planner resolver
# =============================================================================

class PlannerAttributionResolver:
    """
    Resolve planner attribution from runtime/full_execution_cycle_last.json.

    Important rule:
    - If order["strategy"] is missing or placeholder, DO NOT promote "unknown".
    - Instead choose the first real contributor.
    """

    def __init__(self, plan_path: Path = PLAN_ARTIFACT_DEFAULT) -> None:
        self._plan_path = plan_path
        self._cache_mtime_ns: Optional[int] = None
        self._by_symbol: Dict[str, Dict[str, Any]] = {}

    def resolve(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._refresh_if_needed()
        return self._by_symbol.get(symbol.upper())

    def _refresh_if_needed(self) -> None:
        if not self._plan_path.exists():
            self._cache_mtime_ns = None
            self._by_symbol = {}
            return

        stat = self._plan_path.stat()
        if self._cache_mtime_ns == stat.st_mtime_ns:
            return

        raw = _read_json(self._plan_path, {})
        orders = raw.get("orders", []) if isinstance(raw, dict) else []
        plan_now_iso = _safe_str(raw.get("now"), "") if isinstance(raw, dict) else ""

        by_symbol: Dict[str, Dict[str, Any]] = {}

        if isinstance(orders, list):
            for order in orders:
                if not isinstance(order, dict):
                    continue

                symbol = _safe_str(order.get("symbol")).upper()
                if not symbol:
                    continue

                raw_strategy = _normalize_strategy(order.get("strategy"), "")
                contributors_raw = order.get("contributors") or order.get("source_strategies") or []
                if not isinstance(contributors_raw, list):
                    contributors_raw = [contributors_raw]

                contributors = _filter_real_strategies(contributors_raw)

                # only keep explicit strategy if it is real
                strategy = raw_strategy if _is_real_strategy(raw_strategy) else ""

                # if explicit strategy missing/placeholder, promote first real contributor
                if not strategy and contributors:
                    strategy = contributors[0]

                # P0-4: pass through empty if no real strategy found —
                # the attribution engine decides whether to raise.

                if _is_real_strategy(strategy) and strategy not in contributors:
                    contributors.insert(0, strategy)

                # if still empty, leave contributors empty
                if not contributors and _is_real_strategy(strategy):
                    contributors = [strategy]

                by_symbol[symbol] = {
                    "strategy": strategy,
                    "contributors": contributors,
                    "asset_class": _safe_str(order.get("asset_class"), "").lower(),
                    "plan_now_iso": plan_now_iso,
                    "plan_path": str(self._plan_path),
                }

        self._cache_mtime_ns = stat.st_mtime_ns
        self._by_symbol = by_symbol


# =============================================================================
# Compatibility-first event object
# =============================================================================

class PaperExecEvidence:
    """
    Compatibility-first payload object.

    Accepts arbitrary keyword args from legacy/current executor code, including
    fields like slippage_bps, latency_ms, expected_price, planned_hash, etc.
    """

    __slots__ = (
        "symbol",
        "side",
        "quantity",
        "fill_price",
        "notional",
        "strategy",
        "source_strategies",
        "broker",
        "venue",
        "account_id",
        "is_live",
        "reject",
        "partial_fill",
        "asset_class",
        "order_type",
        "status",
        "entry_time_utc",
        "exit_time_utc",
        "fill_time_utc",
        "regime",
        "pnl",
        "fee_amount",
        "fee_currency",
        "plan_now_iso",
        "plan_path",
        "tags",
        "extra",
        "source",
        "slippage_bps",
        "latency_ms",
        "expected_price",
        "execution_id",
        "planned_hash",
        "extra_fields",
    )

    def __init__(self, **kwargs: Any) -> None:
        self.symbol = _safe_str(kwargs.pop("symbol", "UNKNOWN"), "UNKNOWN")
        self.side = _safe_str(kwargs.pop("side", "BUY"), "BUY")
        self.quantity = _safe_float(kwargs.pop("quantity", kwargs.pop("qty", 0.0)), 0.0)
        self.fill_price = _safe_float(kwargs.pop("fill_price", kwargs.pop("price", 0.0)), 0.0)
        self.notional = _safe_float(kwargs.pop("notional", 0.0), 0.0)

        self.strategy = _safe_str(kwargs.pop("strategy", ""), "")
        raw_sources = kwargs.pop("source_strategies", ())
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        elif not isinstance(raw_sources, (list, tuple)):
            raw_sources = []
        self.source_strategies = tuple(_safe_str(v) for v in raw_sources if _safe_str(v))

        self.broker = _safe_str(kwargs.pop("broker", "paper_exec"), "paper_exec")
        self.venue = _safe_str(kwargs.pop("venue", "paper_exec"), "paper_exec")
        self.account_id = _safe_str(kwargs.pop("account_id", "PAPER_EXEC"), "PAPER_EXEC")

        self.is_live = _safe_bool(kwargs.pop("is_live", False), False)
        self.reject = _safe_bool(kwargs.pop("reject", False), False)
        self.partial_fill = _safe_bool(kwargs.pop("partial_fill", False), False)

        self.asset_class = _safe_str(kwargs.pop("asset_class", ""), "")
        self.order_type = _safe_str(kwargs.pop("order_type", "SIM"), "SIM")
        self.status = _safe_str(kwargs.pop("status", "FILLED"), "FILLED")

        self.entry_time_utc = _safe_str(kwargs.pop("entry_time_utc", ""), "")
        self.exit_time_utc = _safe_str(kwargs.pop("exit_time_utc", ""), "")
        self.fill_time_utc = _safe_str(kwargs.pop("fill_time_utc", ""), "")

        self.regime = _safe_str(kwargs.pop("regime", "paper"), "paper")

        self.pnl = _safe_float(kwargs.pop("pnl", 0.0), 0.0)
        self.fee_amount = _safe_float(kwargs.pop("fee_amount", 0.0), 0.0)
        self.fee_currency = _safe_str(kwargs.pop("fee_currency", "USD"), "USD")

        self.plan_now_iso = _safe_str(kwargs.pop("plan_now_iso", ""), "")
        self.plan_path = _safe_str(kwargs.pop("plan_path", ""), "")

        raw_tags = kwargs.pop("tags", ())
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        elif not isinstance(raw_tags, (list, tuple)):
            raw_tags = []
        self.tags = tuple(_safe_str(v) for v in raw_tags if _safe_str(v))

        extra = kwargs.pop("extra", {})
        self.extra = dict(extra) if isinstance(extra, Mapping) else {}

        self.source = _safe_str(kwargs.pop("source", "paper_trade_executor"), "paper_trade_executor")

        self.slippage_bps = _safe_float(kwargs.pop("slippage_bps", 0.0), 0.0)
        self.latency_ms = _safe_float(kwargs.pop("latency_ms", 0.0), 0.0)
        self.expected_price = _safe_float(kwargs.pop("expected_price", 0.0), 0.0)
        self.execution_id = _safe_str(kwargs.pop("execution_id", ""), "")
        self.planned_hash = _safe_str(kwargs.pop("planned_hash", ""), "")

        self.extra_fields = dict(kwargs)

    @classmethod
    def from_any(cls, value: Any) -> "PaperExecEvidence":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if hasattr(value, "__dict__"):
            return cls(**dict(value.__dict__))
        raise TypeError(f"Unsupported PaperExecEvidence input type: {type(value).__name__}")


# =============================================================================
# Attribution engine
# =============================================================================

class StrategyAttributionEngine:
    """
    Resolve strategy attribution for paper execution evidence records.

    Contributor merge contract (SSOT v6.4):
    ────────────────────────────────────────
    1. One primary_strategy — always a real strategy name, never a placeholder.
    2. One source_strategies list — always contains primary_strategy; preserves
       all real contributors in order, no deduplication loss.
    3. No flattening — placeholder values (unknown, manual, paper_exec) never
       override a real strategy when one is available at any precedence level.
    4. No lineage loss — if no real strategy can be resolved from any source,
       StrategyAttributionError is raised rather than silently writing a
       placeholder as primary_strategy.

    Attribution precedence (highest to lowest):
      A. Explicit strategy kwarg, if real
      B. Explicit source_strategies kwarg, if any are real
      C. Planner artifact by symbol (runtime/full_execution_cycle_last.json)
      D. Tags fallback
      E. Raise StrategyAttributionError — never silently degrade
    """

    def __init__(self, planner: Optional[PlannerAttributionResolver] = None) -> None:
        self._planner = planner or PlannerAttributionResolver()

    def resolve(self, ev: PaperExecEvidence) -> Dict[str, Any]:
        explicit = _normalize_strategy(ev.strategy, "")
        explicit_sources = _filter_real_strategies(ev.source_strategies)
        tag_candidates = _filter_real_strategies(ev.tags)

        # 1) explicit strategy if real
        if _is_real_strategy(explicit):
            sources = explicit_sources[:] if explicit_sources else [explicit]
            if explicit not in sources:
                sources.insert(0, explicit)
            return {
                "primary_strategy": explicit,
                "source_strategies": sources,
                "asset_class": _safe_str(ev.asset_class, ""),
                "plan_now_iso": _safe_str(ev.plan_now_iso, ""),
                "plan_path": _safe_str(ev.plan_path, ""),
            }

        # 2) explicit source_strategies if real
        if explicit_sources:
            return {
                "primary_strategy": explicit_sources[0],
                "source_strategies": explicit_sources,
                "asset_class": _safe_str(ev.asset_class, ""),
                "plan_now_iso": _safe_str(ev.plan_now_iso, ""),
                "plan_path": _safe_str(ev.plan_path, ""),
            }

        # 3) planner artifact by symbol
        planned = self._planner.resolve(ev.symbol)
        if planned:
            strategy = _normalize_strategy(planned.get("strategy"), "")
            contributors = planned.get("contributors") or []
            if not isinstance(contributors, list):
                contributors = [contributors]

            real_contributors = _filter_real_strategies(contributors)

            if _is_real_strategy(strategy):
                primary = strategy
            elif real_contributors:
                primary = real_contributors[0]
            else:
                # P0-4: planner artifact exists but has no real strategy —
                # fall through to tags / final raise rather than flatten.
                planned = None  # force fall-through

            if planned is not None:
                source_strategies = real_contributors[:] if real_contributors else [primary]
                if _is_real_strategy(primary) and primary not in source_strategies:
                    source_strategies.insert(0, primary)

                return {
                    "primary_strategy": primary,
                    "source_strategies": source_strategies,
                    "asset_class": _safe_str(ev.asset_class, _safe_str(planned.get("asset_class"), "")),
                    "plan_now_iso": _safe_str(ev.plan_now_iso, _safe_str(planned.get("plan_now_iso"), "")),
                    "plan_path": _safe_str(ev.plan_path, _safe_str(planned.get("plan_path"), "")),
                }

        # 4) tags fallback
        if tag_candidates:
            return {
                "primary_strategy": tag_candidates[0],
                "source_strategies": tag_candidates,
                "asset_class": _safe_str(ev.asset_class, ""),
                "plan_now_iso": _safe_str(ev.plan_now_iso, ""),
                "plan_path": _safe_str(ev.plan_path, ""),
            }

        # 5) P0-4: raise — never silently flatten to paper_exec
        raise StrategyAttributionError(
            f"No real strategy resolved for {ev.symbol} — "
            f"all attribution sources exhausted (explicit, source_strategies, planner, tags)"
        )


# =============================================================================
# Payload factory
# =============================================================================

class EvidencePayloadFactory:
    def __init__(self, attribution_engine: Optional[StrategyAttributionEngine] = None) -> None:
        self._attr = attribution_engine or StrategyAttributionEngine()

    def build_fill_payload(self, ev: PaperExecEvidence) -> Dict[str, Any]:
        attr = self._attr.resolve(ev)

        symbol = _safe_str(ev.symbol, "UNKNOWN").upper()
        side = _safe_str(ev.side, "BUY").upper()
        quantity = _safe_float(ev.quantity, 0.0)
        fill_price = _safe_float(ev.fill_price, 0.0)
        notional = _safe_float(ev.notional, 0.0)
        if notional <= 0.0 and quantity > 0.0 and fill_price > 0.0:
            notional = quantity * fill_price

        tags = _dedupe_keep_order(
            [
                "paper",
                "filled" if not ev.reject else "rejected",
                attr["primary_strategy"],
                _safe_str(attr["asset_class"]).lower(),
                *[_safe_str(t).lower() for t in ev.tags],
            ]
        )

        return {
            "schema_version": FILL_SCHEMA_VERSION,
            "account_id": _safe_str(ev.account_id, "PAPER_EXEC"),
            "asset_class": _safe_str(attr["asset_class"], _safe_str(ev.asset_class, "")).lower(),
            "broker": _safe_str(ev.broker, "paper_exec"),
            "entry_time_utc": _safe_str(ev.entry_time_utc, _safe_str(ev.fill_time_utc, _iso_z())),
            "exit_time_utc": _safe_str(ev.exit_time_utc, _safe_str(ev.fill_time_utc, _iso_z())),
            "extra": self._build_extra(ev, attr),
            "fill_id": self._make_fill_id(ev, attr),
            "fill_price": fill_price,
            "fill_time_utc": _safe_str(ev.fill_time_utc, _iso_z()),
            "is_live": _safe_bool(ev.is_live, False),
            "notional": notional,
            "order_type": _safe_str(ev.order_type, "SIM"),
            "partial_fill": _safe_bool(ev.partial_fill, False),
            "plan_now_iso": _safe_str(attr["plan_now_iso"], _safe_str(ev.plan_now_iso, "")),
            "plan_path": _safe_str(attr["plan_path"], _safe_str(ev.plan_path, "")),
            "quantity": quantity,
            "reject": _safe_bool(ev.reject, False),
            "side": side,
            "source": _safe_str(ev.source, "paper_trade_executor"),
            "status": _safe_str(ev.status, "FILLED"),
            "strategy": attr["primary_strategy"],
            "source_strategies": list(attr["source_strategies"]),
            "symbol": symbol,
            "tags": tags,
            "venue": _safe_str(ev.venue, "paper_exec"),
        }

    def build_fee_payload(self, ev: PaperExecEvidence) -> Dict[str, Any]:
        attr = self._attr.resolve(ev)

        return {
            "schema_version": FEE_SCHEMA_VERSION,
            "account_id": _safe_str(ev.account_id, "PAPER_EXEC"),
            "broker": _safe_str(ev.broker, "paper_exec"),
            "extra": self._build_extra(ev, attr),
            "fee_amount": _safe_float(ev.fee_amount, 0.0),
            "fee_currency": _safe_str(ev.fee_currency, "USD"),
            "fee_id": self._make_fee_id(ev, attr),
            "fill_time_utc": _safe_str(ev.fill_time_utc, _iso_z()),
            "is_live": _safe_bool(ev.is_live, False),
            "plan_path": _safe_str(attr["plan_path"], _safe_str(ev.plan_path, "")),
            "side": _safe_str(ev.side, "BUY").upper(),
            "source": _safe_str(ev.source, "paper_trade_executor"),
            "strategy": attr["primary_strategy"],
            "source_strategies": list(attr["source_strategies"]),
            "symbol": _safe_str(ev.symbol, "UNKNOWN").upper(),
            "tags": _dedupe_keep_order(
                [
                    "paper",
                    "filled" if not ev.reject else "rejected",
                    attr["primary_strategy"],
                    _safe_str(attr["asset_class"]).lower(),
                    *[_safe_str(t).lower() for t in ev.tags],
                ]
            ),
            "venue": _safe_str(ev.venue, "paper_exec"),
        }

    def build_execution_metric_payload(self, ev: PaperExecEvidence) -> Dict[str, Any]:
        attr = self._attr.resolve(ev)

        return {
            "schema_version": EXEC_METRIC_SCHEMA_VERSION,
            "account_id": _safe_str(ev.account_id, "PAPER_EXEC"),
            "broker": _safe_str(ev.broker, "paper_exec"),
            "extra": self._build_extra(ev, attr),
            "execution_id": _safe_str(ev.execution_id, self._make_exec_id(ev, attr)),
            "expected_price": _safe_float(ev.expected_price, 0.0),
            "fill_price": _safe_float(ev.fill_price, 0.0),
            "fill_time_utc": _safe_str(ev.fill_time_utc, _iso_z()),
            "is_live": _safe_bool(ev.is_live, False),
            "latency_ms": _safe_float(ev.latency_ms, 0.0),
            "notional": _safe_float(ev.notional, 0.0),
            "partial_fill": _safe_bool(ev.partial_fill, False),
            "planned_hash": _safe_str(ev.planned_hash, ""),
            "quantity": _safe_float(ev.quantity, 0.0),
            "reject": _safe_bool(ev.reject, False),
            "side": _safe_str(ev.side, "BUY").upper(),
            "slippage_bps": _safe_float(ev.slippage_bps, 0.0),
            "source": _safe_str(ev.source, "paper_trade_executor"),
            "status": _safe_str(ev.status, "FILLED"),
            "strategy": attr["primary_strategy"],
            "source_strategies": list(attr["source_strategies"]),
            "symbol": _safe_str(ev.symbol, "UNKNOWN").upper(),
            "venue": _safe_str(ev.venue, "paper_exec"),
        }

    def _build_extra(self, ev: PaperExecEvidence, attr: Mapping[str, Any]) -> Dict[str, Any]:
        extra = dict(ev.extra or {})
        extra.setdefault("source_strategies", list(attr["source_strategies"]))
        if _safe_str(attr.get("plan_now_iso"), ""):
            extra.setdefault("plan_now_iso", _safe_str(attr.get("plan_now_iso"), ""))
        if _safe_str(attr.get("plan_path"), ""):
            extra.setdefault("plan_path", _safe_str(attr.get("plan_path"), ""))
        extra.setdefault("slippage_bps", _safe_float(ev.slippage_bps, 0.0))
        extra.setdefault("latency_ms", _safe_float(ev.latency_ms, 0.0))
        if _safe_float(ev.expected_price, 0.0):
            extra.setdefault("expected_price", _safe_float(ev.expected_price, 0.0))
        if _safe_str(ev.execution_id, ""):
            extra.setdefault("execution_id", _safe_str(ev.execution_id, ""))
        if _safe_str(ev.planned_hash, ""):
            extra.setdefault("planned_hash", _safe_str(ev.planned_hash, ""))
        if ev.extra_fields:
            extra.setdefault("unmapped_fields", dict(ev.extra_fields))
        return extra

    def _make_fill_id(self, ev: PaperExecEvidence, attr: Mapping[str, Any]) -> str:
        raw = "|".join(
            [
                _safe_str(ev.account_id, "PAPER_EXEC"),
                _safe_str(ev.symbol, "UNKNOWN").upper(),
                _safe_str(ev.side, "BUY").upper(),
                f"{_safe_float(ev.quantity, 0.0):.12f}",
                f"{_safe_float(ev.fill_price, 0.0):.12f}",
                _safe_str(ev.fill_time_utc, ""),
                _safe_str(attr.get("primary_strategy"), ""),
            ]
        )
        return _hash_text(raw)

    def _make_fee_id(self, ev: PaperExecEvidence, attr: Mapping[str, Any]) -> str:
        raw = "|".join(
            [
                _safe_str(ev.account_id, "PAPER_EXEC"),
                _safe_str(ev.symbol, "UNKNOWN").upper(),
                _safe_str(ev.side, "BUY").upper(),
                _safe_str(ev.fill_time_utc, ""),
                f"{_safe_float(ev.fee_amount, 0.0):.12f}",
                _safe_str(attr.get("primary_strategy"), ""),
            ]
        )
        return _hash_text(raw)

    def _make_exec_id(self, ev: PaperExecEvidence, attr: Mapping[str, Any]) -> str:
        raw = "|".join(
            [
                _safe_str(ev.account_id, "PAPER_EXEC"),
                _safe_str(ev.symbol, "UNKNOWN").upper(),
                _safe_str(ev.side, "BUY").upper(),
                _safe_str(ev.fill_time_utc, ""),
                _safe_str(attr.get("primary_strategy"), ""),
                f"{_safe_float(ev.quantity, 0.0):.12f}",
            ]
        )
        return _hash_text(raw)


# =============================================================================
# Writer service
# =============================================================================

class PaperExecutionEvidenceWriter:
    def __init__(self, payload_factory: Optional[EvidencePayloadFactory] = None) -> None:
        self._factory = payload_factory or EvidencePayloadFactory()

    def write(self, ev: PaperExecEvidence) -> Dict[str, str]:
        day = _ymd()

        fills_path = FILLS_DIR / f"FILLS_{day}.ndjson"
        fees_path = FEES_DIR / f"FEES_{day}.ndjson"
        exec_metrics_path = EXEC_METRICS_DIR / f"EXECUTION_METRICS_{day}.ndjson"

        fill_payload = self._factory.build_fill_payload(ev)
        fee_payload = self._factory.build_fee_payload(ev)
        metric_payload = self._factory.build_execution_metric_payload(ev)

        fill_meta = _append_hash_chained_record(fills_path, fill_payload)
        fee_meta = _append_hash_chained_record(fees_path, fee_payload)
        metric_meta = _append_hash_chained_record(exec_metrics_path, metric_payload)

        # Phase-8 Session 3 (E3): append slippage record + update rolling stats.
        # Failure is non-fatal — fills must always succeed even if tracker I/O
        # breaks, so we swallow any exception here.
        try:
            from chad.analytics.slippage_tracker import get_default_tracker

            tracker = get_default_tracker()
            tracker.record_fill(
                symbol=_safe_str(ev.symbol, "UNKNOWN"),
                strategy=_safe_str(ev.strategy, ""),
                side=_safe_str(ev.side, "BUY"),
                expected_price=_safe_float(ev.expected_price, 0.0),
                fill_price=_safe_float(ev.fill_price, 0.0),
                quantity=_safe_float(ev.quantity, 0.0),
                intent_id=_safe_str(ev.execution_id, ""),
                fill_timestamp=_safe_str(ev.fill_time_utc, ""),
            )
        except Exception:
            pass

        # Phase-8 Session 6 (F2): record a pending entry for retrospective
        # signal-decay measurement. Alpha at T+1/5/15/30 days is filled in
        # later by SignalDecayRecorder.compute_decay_for_pending() once the
        # necessary daily bars are on disk. Non-fatal.
        try:
            from chad.analytics.signal_decay import get_default_recorder as _get_decay_recorder

            _get_decay_recorder().record_entry(
                strategy=_safe_str(ev.strategy, ""),
                symbol=_safe_str(ev.symbol, "UNKNOWN"),
                side=_safe_str(ev.side, "BUY"),
                entry_price=_safe_float(ev.fill_price, 0.0),
                entry_time=_safe_str(ev.fill_time_utc, ""),
                intent_id=_safe_str(ev.execution_id, ""),
            )
        except Exception:
            pass

        return {
            "fills_path": fill_meta["path"],
            "fees_path": fee_meta["path"],
            "execution_metrics_path": metric_meta["path"],
        }


# =============================================================================
# Public compatibility API
# =============================================================================

_DEFAULT_WRITER = PaperExecutionEvidenceWriter()


def write_paper_exec_evidence(ev: Any) -> Dict[str, str]:
    normalized = PaperExecEvidence.from_any(ev)
    return _DEFAULT_WRITER.write(normalized)
