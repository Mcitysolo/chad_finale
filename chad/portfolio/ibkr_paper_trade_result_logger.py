#!/usr/bin/env python3
"""
chad/portfolio/ibkr_paper_trade_result_logger.py

Production-grade paper trade result logger for CHAD.

Purpose
-------
Write append-only trade-history records to:

    data/trades/trade_history_YYYYMMDD.ndjson

while preserving true strategy attribution from the upstream execution plan.

This script is designed to replace older behavior that flattened strategy identity
to "paper_exec", even when upstream plans already contained real strategy names
(e.g. beta, gamma, delta).

Design goals
------------
- Preserve true strategy identity from:
  1) explicit event.strategy
  2) source_strategies / contributing strategies
  3) raw_intent / execution plan metadata
  4) semantic fallbacks from tags/extra
- Append-only ledger with:
  - sequence_id
  - prev_hash
  - record_hash
- Atomic state persistence
- Single-responsibility, testable helpers
- Safe defaults, no silent data corruption
- File locking for concurrent writers
- Backward-compatible payload shape for CHAD reporting jobs

Python: 3.10+
No third-party dependencies required.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from chad.execution.paper_exec_evidence_writer import StrategyAttributionError

LOGGER = logging.getLogger("chad.ibkr_paper_trade_result_logger")


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_TRADES_DIR = DATA_DIR / "trades"
RUNTIME_DIR = ROOT / "runtime"
STATE_DIR = RUNTIME_DIR / "ledger_state"
TRADE_LEDGER_STATE_PATH = STATE_DIR / "trade_history_state.json"

GENESIS = "GENESIS"

STRATEGY_ALIASES: Dict[str, str] = {
    "alpha": "alpha",
    "beta": "beta",
    "beta_trend": "beta_trend",
    "gamma": "gamma",
    "omega": "omega",
    "delta": "delta",
    "alpha_crypto": "alpha_crypto",
    "crypto": "alpha_crypto",
    "alpha_forex": "alpha_forex",
    "forex": "alpha_forex",
    "paper_exec": "paper_exec",
    "unknown": "unknown",
}

TAG_STRATEGY_CANDIDATES = {
    "alpha",
    "beta",
    "beta_trend",
    "gamma",
    "omega",
    "delta",
    "alpha_crypto",
    "alpha_forex",
    "crypto",
    "forex",
}

SCHEMA_VERSION = "paper_trade_result.v2"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_ymd(d: Optional[dt.datetime] = None) -> str:
    base = d or _utc_now()
    return base.astimezone(dt.timezone.utc).strftime("%Y%m%d")


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        out = value.strip()
        return out if out else default
    out = str(value).strip()
    return out if out else default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), prefix=path.name + ".tmp.") as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _json_dumps_canonical(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ledger_path(day_ymd: Optional[str] = None) -> Path:
    _ensure_dir(DATA_TRADES_DIR)
    return DATA_TRADES_DIR / f"trade_history_{day_ymd or _utc_ymd()}.ndjson"


def _lock_path_for_ledger(ledger_path: Path) -> Path:
    return STATE_DIR / f"{ledger_path.stem}.lock"


def _normalize_strategy_name(raw: str) -> str:
    s = _safe_str(raw).lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9_,\-]+", "", s)
    if s in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[s]
    return s


def _normalize_strategy_list(values: Iterable[Any]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        name = _normalize_strategy_name(_safe_str(v))
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _coerce_sequence(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _extract_tags(value: Any) -> Tuple[str, ...]:
    raw = _coerce_sequence(value)
    tags: List[str] = []
    seen: set[str] = set()
    for item in raw:
        t = _safe_str(item).lower()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            tags.append(t)
    return tuple(tags)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


# ---------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------

@dataclass(slots=True)
class PaperTradeResultEvent:
    """
    Canonical event model for writing one paper trade result row.

    This class is intentionally permissive because upstream CHAD producers may
    provide slightly different shapes over time.
    """

    symbol: str
    side: str

    quantity: Optional[float] = None
    fill_price: Optional[float] = None
    notional: Optional[float] = None
    realized_pnl: Optional[float] = None

    strategy: Optional[str] = None
    source_strategies: Tuple[str, ...] = field(default_factory=tuple)

    entry_time_utc: Optional[str] = None
    exit_time_utc: Optional[str] = None
    ts_utc: Optional[str] = None

    account_id: str = "PAPER_EXEC"
    broker: str = "paper_exec"
    venue: str = "paper_exec"
    regime: str = "paper"
    is_live: bool = False
    source: str = "paper_trade_executor"

    original_notional: Optional[float] = None
    cap_notional_usd: Optional[float] = None
    plan_now_iso: Optional[str] = None
    plan_path: Optional[str] = None

    tags: Tuple[str, ...] = field(default_factory=tuple)
    extra: Dict[str, Any] = field(default_factory=dict)
    raw_intent: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "PaperTradeResultEvent":
        if isinstance(value, cls):
            return value
        if dataclasses.is_dataclass(value):
            return cls.from_mapping(dataclasses.asdict(value))
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        raise TypeError(f"Unsupported event type for PaperTradeResultEvent: {type(value).__name__}")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "PaperTradeResultEvent":
        raw_intent = _first_present(m, ["raw_intent", "intent", "order", "execution_intent"])
        raw_intent = raw_intent if isinstance(raw_intent, Mapping) else {}

        extra = _first_present(m, ["extra", "meta", "metadata"])
        extra = dict(extra) if isinstance(extra, Mapping) else {}

        symbol = _safe_str(_first_present(m, ["symbol", "ticker"]), "")
        side = _safe_str(_first_present(m, ["side", "action"]), "")

        source_strategies = _normalize_strategy_list(
            _coerce_sequence(
                _first_present(
                    m,
                    ["source_strategies", "contributing_strategies", "contributors", "strategies"],
                )
            )
        )

        tags = _extract_tags(_first_present(m, ["tags", "labels"]))

        return cls(
            symbol=symbol,
            side=side,
            quantity=_safe_float(_first_present(m, ["quantity", "qty", "size"])),
            fill_price=_safe_float(_first_present(m, ["fill_price", "price", "avg_fill_price"])),
            notional=_safe_float(_first_present(m, ["notional", "notional_usd"])),
            realized_pnl=_safe_float(_first_present(m, ["realized_pnl", "pnl", "pnl_usd"])),
            strategy=_safe_str(_first_present(m, ["strategy", "strategy_name", "brain"]), None),
            source_strategies=source_strategies,
            entry_time_utc=_safe_str(_first_present(m, ["entry_time_utc", "opened_at_utc"]), None),
            exit_time_utc=_safe_str(_first_present(m, ["exit_time_utc", "closed_at_utc"]), None),
            ts_utc=_safe_str(_first_present(m, ["ts_utc", "timestamp_utc", "created_at_utc"]), None),
            account_id=_safe_str(_first_present(m, ["account_id"]), "PAPER_EXEC"),
            broker=_safe_str(_first_present(m, ["broker"]), "paper_exec"),
            venue=_safe_str(_first_present(m, ["venue"]), "paper_exec"),
            regime=_safe_str(_first_present(m, ["regime"]), "paper"),
            is_live=_safe_bool(_first_present(m, ["is_live"]), False),
            source=_safe_str(_first_present(m, ["source"]), "paper_trade_executor"),
            original_notional=_safe_float(_first_present(m, ["original_notional"])),
            cap_notional_usd=_safe_float(_first_present(m, ["cap_notional_usd"])),
            plan_now_iso=_safe_str(_first_present(m, ["plan_now_iso"]), None),
            plan_path=_safe_str(_first_present(m, ["plan_path"]), None),
            tags=tags,
            extra=extra,
            raw_intent=dict(raw_intent),
        )


# ---------------------------------------------------------------------
# Strategy attribution
# ---------------------------------------------------------------------

class StrategyResolver:
    """
    Resolve primary strategy + contributor set from the richest available source.
    """

    def resolve(self, event: PaperTradeResultEvent) -> Tuple[str, Tuple[str, ...]]:
        explicit = _normalize_strategy_name(event.strategy or "")
        contributors = list(_normalize_strategy_list(event.source_strategies))

        raw_intent_strategies = self._extract_from_raw_intent(event.raw_intent)
        for s in raw_intent_strategies:
            if s not in contributors:
                contributors.append(s)

        tag_strategies = self._extract_from_tags(event.tags)
        for s in tag_strategies:
            if s not in contributors:
                contributors.append(s)

        extra_strategies = self._extract_from_extra(event.extra)
        for s in extra_strategies:
            if s not in contributors:
                contributors.append(s)

        # Primary strategy precedence
        if explicit and explicit not in {"paper_exec", "unknown"}:
            primary = explicit
        elif contributors:
            primary = contributors[0]
        elif explicit and explicit not in {"paper_exec", "unknown"}:
            primary = explicit
        else:
            # P0-4: raise — never silently flatten to paper_exec
            raise StrategyAttributionError(
                f"No real strategy resolved for {event.symbol} — "
                f"all attribution sources exhausted (explicit, source_strategies, raw_intent, tags, extra)"
            )

        # Ensure primary is included
        if primary not in contributors:
            contributors.insert(0, primary)

        # De-dup while preserving order
        normalized = _normalize_strategy_list(contributors)

        if not normalized:
            raise StrategyAttributionError(
                f"No real strategy resolved for {event.symbol} — "
                f"contributors list empty after normalization"
            )

        return primary, normalized

    def _extract_from_raw_intent(self, m: Mapping[str, Any]) -> Tuple[str, ...]:
        if not m:
            return ()
        vals: List[Any] = []
        vals.extend(_coerce_sequence(_first_present(m, ["strategy", "strategy_name", "brain"])))
        vals.extend(_coerce_sequence(_first_present(m, ["source_strategies", "contributors", "strategies"])))
        return _normalize_strategy_list(vals)

    def _extract_from_extra(self, m: Mapping[str, Any]) -> Tuple[str, ...]:
        if not m:
            return ()
        vals: List[Any] = []
        vals.extend(_coerce_sequence(_first_present(m, ["strategy", "strategy_name", "brain"])))
        vals.extend(_coerce_sequence(_first_present(m, ["source_strategies", "contributors", "strategies"])))
        return _normalize_strategy_list(vals)

    def _extract_from_tags(self, tags: Sequence[str]) -> Tuple[str, ...]:
        vals = [t for t in tags if t in TAG_STRATEGY_CANDIDATES]
        return _normalize_strategy_list(vals)


# ---------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------

class TradePayloadBuilder:
    def __init__(self, strategy_resolver: Optional[StrategyResolver] = None) -> None:
        self._strategy_resolver = strategy_resolver or StrategyResolver()

    def build(self, event: PaperTradeResultEvent) -> Dict[str, Any]:
        primary_strategy, contributors = self._strategy_resolver.resolve(event)

        symbol = _safe_str(event.symbol).upper()
        side = _safe_str(event.side).upper()

        if not symbol:
            raise ValueError("Trade event missing symbol")
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Trade event has invalid side={side!r}; expected BUY/SELL")

        fill_price = _safe_float(event.fill_price)
        quantity = _safe_float(event.quantity)
        notional = _safe_float(event.notional)
        pnl = _safe_float(event.realized_pnl, 0.0)

        if notional is None and fill_price is not None and quantity is not None:
            notional = abs(fill_price * quantity)

        tags = list(_extract_tags(event.tags))
        if "paper" not in tags:
            tags.append("paper")
        if "filled" not in tags:
            tags.append("filled")
        if primary_strategy not in {"unknown", ""} and primary_strategy not in tags:
            tags.append(primary_strategy)

        asset_hint = self._infer_asset_hint(symbol, event.raw_intent, event.extra)
        if asset_hint and asset_hint not in tags:
            tags.append(asset_hint)

        extra: Dict[str, Any] = dict(event.extra or {})
        if event.cap_notional_usd is not None:
            extra.setdefault("cap_notional_usd", event.cap_notional_usd)
        if event.original_notional is not None:
            extra.setdefault("original_notional", event.original_notional)
        if event.plan_now_iso:
            extra.setdefault("plan_now_iso", event.plan_now_iso)
        if event.plan_path:
            extra.setdefault("plan_path", event.plan_path)
        if event.source:
            extra.setdefault("source", event.source)

        # Preserve upstream identity inside extra for downstream analytics/debugging
        extra.setdefault("primary_strategy", primary_strategy)
        extra.setdefault("source_strategies", list(contributors))

        entry_time_utc = event.entry_time_utc or event.ts_utc or _utc_now_iso()
        exit_time_utc = event.exit_time_utc or event.ts_utc or entry_time_utc

        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "account_id": _safe_str(event.account_id, "PAPER_EXEC"),
            "broker": _safe_str(event.broker, "paper_exec"),
            "venue": _safe_str(event.venue, "paper_exec"),
            "entry_time_utc": entry_time_utc,
            "exit_time_utc": exit_time_utc,
            "fill_price": fill_price,
            "is_live": bool(event.is_live),
            "notional": notional,
            "pnl": pnl,
            "quantity": quantity,
            "regime": _safe_str(event.regime, "paper"),
            "side": side,
            "strategy": primary_strategy,
            "source_strategies": list(contributors),
            "symbol": symbol,
            "tags": tags,
            "extra": extra,
        }

        return payload

    def _infer_asset_hint(
        self,
        symbol: str,
        raw_intent: Mapping[str, Any],
        extra: Mapping[str, Any],
    ) -> str:
        sec_type = _safe_str(_first_present(raw_intent, ["sec_type", "asset_class", "instrument_type"]), "").lower()
        if sec_type in {"etf"}:
            return "etf"
        if sec_type in {"stk", "stock", "equity"}:
            return "equity"
        if sec_type in {"fut", "future", "futures"}:
            return "futures"
        if sec_type in {"opt", "option", "options"}:
            return "options"
        # Cheap fallback from symbol heuristics
        if symbol in {"SPY", "QQQ", "IWM", "DIA"}:
            return "etf"
        return ""


# ---------------------------------------------------------------------
# Ledger writer
# ---------------------------------------------------------------------

@dataclass(slots=True)
class LedgerState:
    date_ymd: str
    sequence_id: int
    last_record_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "trade_history_state.v1",
            "date_ymd": self.date_ymd,
            "sequence_id": self.sequence_id,
            "last_record_hash": self.last_record_hash,
            "ts_utc": _utc_now_iso(),
        }

    @classmethod
    def genesis(cls, date_ymd: str) -> "LedgerState":
        return cls(date_ymd=date_ymd, sequence_id=0, last_record_hash=GENESIS)


class TradeLedgerWriter:
    def __init__(self, state_path: Path = TRADE_LEDGER_STATE_PATH) -> None:
        self._state_path = state_path

    def append(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        day = _utc_ymd()
        ledger_path = _ledger_path(day)
        lock_path = _lock_path_for_ledger(ledger_path)

        _ensure_dir(ledger_path.parent)
        _ensure_dir(lock_path.parent)

        with open(lock_path, "a+", encoding="utf-8") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)

            state = self._load_state(expected_day=day)
            state = self._reconcile_state_with_file(state=state, ledger_path=ledger_path)

            prev_hash = state.last_record_hash
            next_seq = state.sequence_id + 1

            record: Dict[str, Any] = {
                "payload": dict(payload),
                "prev_hash": prev_hash,
                "sequence_id": next_seq,
                "timestamp_utc": _utc_now_iso(),
            }

            record_hash = _sha256_hex(_json_dumps_canonical(record))
            record["record_hash"] = record_hash

            with open(ledger_path, "a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                os.fsync(out.fileno())

            new_state = LedgerState(
                date_ymd=day,
                sequence_id=next_seq,
                last_record_hash=record_hash,
            )
            _atomic_write_json(self._state_path, new_state.to_dict())

            return {
                "ok": True,
                "ledger_path": str(ledger_path),
                "sequence_id": next_seq,
                "record_hash": record_hash,
                "strategy": payload.get("strategy"),
                "source_strategies": payload.get("source_strategies"),
            }

    def _load_state(self, expected_day: str) -> LedgerState:
        if not self._state_path.exists():
            return LedgerState.genesis(expected_day)
        try:
            obj = json.loads(self._state_path.read_text(encoding="utf-8"))
            day = _safe_str(obj.get("date_ymd"), expected_day)
            seq = int(obj.get("sequence_id", 0))
            last_hash = _safe_str(obj.get("last_record_hash"), GENESIS)
            if day != expected_day:
                return LedgerState.genesis(expected_day)
            if seq < 0:
                return LedgerState.genesis(expected_day)
            return LedgerState(day, seq, last_hash or GENESIS)
        except Exception:
            LOGGER.exception("Failed reading trade ledger state; resetting to genesis")
            return LedgerState.genesis(expected_day)

    def _reconcile_state_with_file(self, state: LedgerState, ledger_path: Path) -> LedgerState:
        """
        Reconcile sidecar state with actual file contents.
        This protects against crashes after write-before-state or state corruption.
        """
        if not ledger_path.exists():
            return LedgerState.genesis(state.date_ymd)

        try:
            last_nonempty = None
            with open(ledger_path, "rb") as f:
                for line in f:
                    if line.strip():
                        last_nonempty = line
            if last_nonempty is None:
                return LedgerState.genesis(state.date_ymd)
            obj = json.loads(last_nonempty.decode("utf-8"))
            seq = int(obj.get("sequence_id", 0))
            rh = _safe_str(obj.get("record_hash"), GENESIS)
            return LedgerState(state.date_ymd, seq, rh or GENESIS)
        except Exception:
            LOGGER.exception("Failed reconciling trade ledger file; preserving in-memory state")
            return state


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

class IBKRPaperTradeResultLogger:
    def __init__(
        self,
        payload_builder: Optional[TradePayloadBuilder] = None,
        ledger_writer: Optional[TradeLedgerWriter] = None,
    ) -> None:
        self._payload_builder = payload_builder or TradePayloadBuilder()
        self._ledger_writer = ledger_writer or TradeLedgerWriter()

    def log(self, event: Any) -> Dict[str, Any]:
        normalized = PaperTradeResultEvent.from_any(event)
        payload = self._payload_builder.build(normalized)
        result = self._ledger_writer.append(payload)
        LOGGER.info(
            "Paper trade result logged strategy=%s symbol=%s side=%s qty=%s pnl=%s seq=%s",
            payload.get("strategy"),
            payload.get("symbol"),
            payload.get("side"),
            payload.get("quantity"),
            payload.get("pnl"),
            result.get("sequence_id"),
        )
        return result


# ---------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------

_DEFAULT_LOGGER = IBKRPaperTradeResultLogger()


def log_trade_result(event: Any) -> Dict[str, Any]:
    """
    Backward-compatible convenience function for existing CHAD call sites.
    """
    return _DEFAULT_LOGGER.log(event)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _load_json_input(path: Optional[str]) -> Any:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if not sys.stdin.isatty():
        raw = sys.stdin.read()
        if raw.strip():
            return json.loads(raw)
    raise SystemExit("No JSON input provided. Use --event-json PATH or pipe JSON via stdin.")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHAD IBKR paper trade result logger")
    p.add_argument("--event-json", default="", help="Path to JSON event payload")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        payload = _load_json_input(args.event_json or None)
        result = log_trade_result(payload)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        LOGGER.exception("Failed to log paper trade result")
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
