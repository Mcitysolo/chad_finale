#!/usr/bin/env python3
"""
CHAD IBKR Paper Ledger Watcher
--------------------------------

Production-grade replacement for:
    chad/portfolio/ibkr_paper_ledger_watcher.py

Mission
-------
Track paper-account open/close state from IBKR, preserve true strategy attribution,
and write trade results with correct strategy identity instead of flattening to
generic labels.

Key improvements
----------------
1. Strategy attribution is no longer taken only from stale open_state.
2. The watcher resolves strategy from multiple sources, in priority order:
   - existing open-state attribution
   - latest full_execution_cycle plan artifact
   - tags / prior metadata
   - configured default strategy
3. open_state stores:
   - primary strategy
   - source_strategies
   - tags
   - plan reference metadata
4. State persistence is atomic.
5. File/report writes are structured and resilient.
6. The script is organized using SRP, typed dataclasses, dependency inversion,
   and strategy resolution as its own component.
7. The code is designed to be service-safe, restart-safe, and timer-safe.

Notes
-----
- This module keeps compatibility with:
    from chad.analytics.trade_result_logger import TradeResult, log_trade_result
- This module expects a runtime config file, by default:
    /home/ubuntu/chad_finale/runtime/ibkr_paper_ledger.json
- This module expects a planner artifact, by default:
    /home/ubuntu/chad_finale/runtime/full_execution_cycle_last.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from chad.analytics.trade_result_logger import TradeResult, log_trade_result
from chad.core.regime_tag import resolve_regime_label
from chad.utils.telegram_notify import NotifyError, notify

try:
    # Existing CHAD installs typically already have ib_insync.
    from ib_insync import IB, Fill  # type: ignore
except Exception:  # pragma: no cover - safe runtime fallback
    IB = None  # type: ignore
    Fill = Any  # type: ignore


# =============================================================================
# Constants / paths
# =============================================================================

ROOT = Path("/home/ubuntu/chad_finale")
CONFIG_PATH_DEFAULT = ROOT / "runtime" / "ibkr_paper_ledger.json"
PLAN_ARTIFACT_DEFAULT = ROOT / "runtime" / "full_execution_cycle_last.json"
ENV_FILE = Path("/etc/chad/telegram.env")
DEFAULT_STATE_PATH = ROOT / "runtime" / "ibkr_paper_ledger_state.json"
DEFAULT_REPORTS_DIR = ROOT / "reports" / "ledger"
DEFAULT_TRADES_DIR = ROOT / "data" / "trades"
DEFAULT_FILLS_DIR = ROOT / "data" / "fills"

# GAP-A003: forward-only matcher for snapshot-diff closes.
SNAPSHOT_DIFF_MATCHER_NAME = "snapshot_diff_fifo_v1"
SNAPSHOT_DIFF_MATCHER_LOOKBACK_DAYS = 7

LOGGER = logging.getLogger("chad.ibkr_paper_ledger_watcher")


# =============================================================================
# Utility helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(ts: Optional[datetime] = None) -> str:
    value = ts or utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else default
    out = str(value).strip()
    return out if out else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return iso_z(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(raw)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed reading JSON from %s", path)
        return default


def hash_key(parts: Sequence[Any]) -> str:
    raw = "|".join(safe_str(p, "") for p in parts)
    return __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()


def normalize_strategy(value: Any, default: str = "manual") -> str:
    s = safe_str(value, default).strip().lower()
    return s or default.lower()


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        v = safe_str(item).strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    s = safe_str(value)
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _extract_payload(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    payload = envelope.get("payload") if isinstance(envelope, Mapping) else None
    if isinstance(payload, dict):
        return payload
    return dict(envelope) if isinstance(envelope, Mapping) else {}


# =============================================================================
# GAP-A003: Snapshot-diff close fill matcher
# =============================================================================

@dataclass(slots=True, frozen=True)
class MatchedOpeningFill:
    fill_id: str
    fill_price: float
    quantity: float
    fill_time_utc: str
    source_path: str
    contract_multiplier: float


def collect_consumed_open_fill_ids(
    trades_dir: Path,
    *,
    lookback_days: int = SNAPSHOT_DIFF_MATCHER_LOOKBACK_DAYS,
    reference_time: Optional[datetime] = None,
) -> set[str]:
    """
    Walk closed_trade.v1 records under trades_dir and return the set of opening
    fill IDs already consumed by a prior close. Used by the snapshot-diff
    matcher to prevent double-matching the same opening fill twice.
    """
    consumed: set[str] = set()
    if not trades_dir.exists():
        return consumed

    ref = reference_time or utc_now()
    cutoff = ref - timedelta(days=int(lookback_days) + 1)

    for path in sorted(trades_dir.glob("trade_history_*.ndjson")):
        try:
            stem_date = path.stem.rsplit("_", 1)[-1]
            file_dt = datetime.strptime(stem_date, "%Y%m%d").replace(tzinfo=timezone.utc)
            if file_dt < cutoff - timedelta(days=1):
                continue
            if file_dt > ref + timedelta(days=2):
                continue
        except Exception:
            pass

        for envelope in _iter_ndjson(path):
            payload = _extract_payload(envelope)
            if not payload:
                continue

            fids = payload.get("fill_ids")
            if isinstance(fids, list):
                for fid in fids:
                    s = safe_str(fid)
                    if s:
                        consumed.add(s)

            entry_fid = safe_str(payload.get("entry_fill_id"))
            if entry_fid:
                consumed.add(entry_fid)

            extra = payload.get("extra")
            if isinstance(extra, dict):
                m = safe_str(extra.get("matched_open_fill_id"))
                if m:
                    consumed.add(m)
                ex_fids = extra.get("fill_ids")
                if isinstance(ex_fids, list):
                    for fid in ex_fids:
                        s = safe_str(fid)
                        if s:
                            consumed.add(s)
                ex_entry = safe_str(extra.get("entry_fill_id"))
                if ex_entry:
                    consumed.add(ex_entry)
    return consumed


def _fill_qualifies_as_open(
    fill: Mapping[str, Any],
    *,
    strategy: str,
    symbol: str,
    expected_open_side: str,
    close_time: datetime,
    consumed_ids: set[str],
) -> Tuple[bool, Optional[datetime]]:
    fid = safe_str(fill.get("fill_id"))
    if not fid or fid in consumed_ids:
        return False, None

    if normalize_strategy(fill.get("strategy"), "") != normalize_strategy(strategy, ""):
        return False, None
    if safe_str(fill.get("symbol")).upper() != safe_str(symbol).upper():
        return False, None
    if safe_str(fill.get("side")).upper() != safe_str(expected_open_side).upper():
        return False, None

    status = safe_str(fill.get("status")).lower()
    if status not in {"paper_fill", "filled"}:
        return False, None

    if bool(fill.get("reject")):
        return False, None

    if bool(fill.get("pnl_untrusted")):
        return False, None
    extra = fill.get("extra")
    if isinstance(extra, dict) and bool(extra.get("pnl_untrusted")):
        return False, None

    tags = fill.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if safe_str(tag).strip().lower() == "pnl_untrusted":
                return False, None

    fill_ts = _parse_iso_utc(fill.get("fill_time_utc"))
    if fill_ts is None or fill_ts > close_time:
        return False, None

    return True, fill_ts


def find_matched_opening_fill_for_snapshot_close(
    *,
    strategy: str,
    symbol: str,
    expected_open_side: str,
    close_time: datetime,
    consumed_ids: set[str],
    fills_dir: Path,
    lookback_days: int = SNAPSHOT_DIFF_MATCHER_LOOKBACK_DAYS,
) -> Optional[MatchedOpeningFill]:
    """
    Forward-only deterministic matcher for snapshot-diff closes.

    Scans data/fills/FILLS_*.ndjson within lookback window and returns the most
    recent unmatched opening fill that matches strategy + symbol + side + status
    constraints. Returns None when no deterministic match exists.
    """
    if not fills_dir.exists():
        return None

    cutoff = close_time - timedelta(days=int(lookback_days))
    candidates: List[Tuple[datetime, Mapping[str, Any], Path]] = []

    for path in sorted(fills_dir.glob("FILLS_*.ndjson")):
        try:
            date_str = path.stem.split("_", 1)[1]
            file_dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            if file_dt < cutoff - timedelta(days=1):
                continue
            if file_dt > close_time + timedelta(days=1):
                continue
        except Exception:
            pass

        for envelope in _iter_ndjson(path):
            payload = _extract_payload(envelope)
            if not payload:
                continue
            ok, ts = _fill_qualifies_as_open(
                payload,
                strategy=strategy,
                symbol=symbol,
                expected_open_side=expected_open_side,
                close_time=close_time,
                consumed_ids=consumed_ids,
            )
            if not ok or ts is None or ts < cutoff:
                continue
            candidates.append((ts, payload, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    ts, fill, src_path = candidates[0]

    fill_price = safe_float(fill.get("fill_price"), 0.0)
    quantity = safe_float(fill.get("quantity"), 0.0)
    if fill_price <= 0.0 or quantity <= 0.0:
        return None

    multiplier = safe_float(fill.get("contract_multiplier"), 0.0)
    if multiplier <= 0.0:
        ex = fill.get("extra")
        if isinstance(ex, dict):
            multiplier = safe_float(ex.get("contract_multiplier"), 0.0)
    if multiplier <= 0.0:
        multiplier = 1.0

    return MatchedOpeningFill(
        fill_id=safe_str(fill.get("fill_id")),
        fill_price=fill_price,
        quantity=quantity,
        fill_time_utc=safe_str(fill.get("fill_time_utc")),
        source_path=str(src_path),
        contract_multiplier=multiplier,
    )


# =============================================================================
# Interfaces / protocols
# =============================================================================

class BrokerGateway(Protocol):
    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def current_positions(self) -> List[Any]:
        ...

    def recent_fills(self) -> List[Any]:
        ...


# =============================================================================
# Config
# =============================================================================

@dataclass(slots=True)
class LedgerConfig:
    enabled: bool = False
    state_path: Path = DEFAULT_STATE_PATH
    reports_dir: Path = DEFAULT_REPORTS_DIR

    default_strategy: str = "manual"

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002
    ibkr_client_id: int = 0

    trades_dir: Path = DEFAULT_TRADES_DIR
    fills_dir: Path = DEFAULT_FILLS_DIR
    exec_window_seconds: float = 90.0
    matcher_lookback_days: int = SNAPSHOT_DIFF_MATCHER_LOOKBACK_DAYS

    plan_artifact_path: Path = PLAN_ARTIFACT_DEFAULT
    notify_on_write: bool = False
    notify_on_error: bool = True

    @classmethod
    def load(cls, path: Path) -> "LedgerConfig":
        raw = read_json(path, {})
        if not isinstance(raw, dict):
            return cls(enabled=False)

        ibkr = raw.get("ibkr") or {}
        if not isinstance(ibkr, dict):
            ibkr = {}

        return cls(
            enabled=bool(raw.get("enabled", False)),
            state_path=Path(str(raw.get("state_path") or DEFAULT_STATE_PATH)),
            reports_dir=Path(str(raw.get("reports_dir") or DEFAULT_REPORTS_DIR)),
            default_strategy=normalize_strategy(raw.get("default_strategy"), "manual"),
            ibkr_host=safe_str(ibkr.get("host"), "127.0.0.1"),
            ibkr_port=safe_int(ibkr.get("port"), 4002),
            ibkr_client_id=safe_int(ibkr.get("client_id"), 0),
            trades_dir=Path(str(raw.get("trades_dir") or DEFAULT_TRADES_DIR)),
            fills_dir=Path(str(raw.get("fills_dir") or DEFAULT_FILLS_DIR)),
            exec_window_seconds=max(1.0, safe_float(raw.get("exec_window_seconds"), 90.0)),
            matcher_lookback_days=max(
                1, safe_int(raw.get("matcher_lookback_days"), SNAPSHOT_DIFF_MATCHER_LOOKBACK_DAYS)
            ),
            plan_artifact_path=Path(str(raw.get("plan_artifact_path") or PLAN_ARTIFACT_DEFAULT)),
            notify_on_write=bool(raw.get("notify_on_write", False)),
            notify_on_error=bool(raw.get("notify_on_error", True)),
        )


# =============================================================================
# Report model
# =============================================================================

@dataclass(slots=True)
class RunReport:
    ts_utc: str
    ok: bool = True
    config_enabled: bool = False
    writes_trade_results: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_detail(self, **kwargs: Any) -> None:
        self.details.append(kwargs)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Open-state model
# =============================================================================

@dataclass(slots=True)
class OpenPositionRecord:
    symbol: str
    con_id: int
    sec_type: str
    qty: float
    avg_cost: float
    account_id: Optional[str]
    currency: str

    opened_at_utc: str

    strategy: str
    source_strategies: List[str]
    tags: List[str]

    plan_now_iso: Optional[str] = None
    plan_path: Optional[str] = None
    attribution_source: str = "unknown"

    def key(self) -> str:
        return hash_key([self.symbol, self.con_id, self.sec_type])


# =============================================================================
# Plan artifact resolver
# =============================================================================

@dataclass(slots=True)
class PlannedOrderAttribution:
    symbol: str
    strategy: str
    source_strategies: List[str]
    tags: List[str]
    plan_now_iso: Optional[str]
    plan_path: str
    attribution_source: str


class PlanAttributionResolver:
    """
    Reads the latest full_execution_cycle artifact and resolves symbol->strategy attribution.
    """

    def __init__(self, plan_artifact_path: Path) -> None:
        self._plan_artifact_path = plan_artifact_path
        self._cache_mtime_ns: Optional[int] = None
        self._cache_by_symbol: Dict[str, PlannedOrderAttribution] = {}

    def resolve_for_symbol(self, symbol: str) -> Optional[PlannedOrderAttribution]:
        self._refresh_if_needed()
        return self._cache_by_symbol.get(symbol.upper())

    def _refresh_if_needed(self) -> None:
        try:
            if not self._plan_artifact_path.exists():
                self._cache_by_symbol = {}
                self._cache_mtime_ns = None
                return

            stat = self._plan_artifact_path.stat()
            if self._cache_mtime_ns == stat.st_mtime_ns:
                return

            raw = read_json(self._plan_artifact_path, {})
            orders = raw.get("orders", [])
            if not isinstance(orders, list):
                orders = []

            plan_now_iso = safe_str(raw.get("now"), None) if isinstance(raw, dict) else None
            by_symbol: Dict[str, PlannedOrderAttribution] = {}

            for order in orders:
                if not isinstance(order, dict):
                    continue

                symbol = safe_str(order.get("symbol")).upper()
                if not symbol:
                    continue

                strategy = normalize_strategy(order.get("strategy"), "manual")

                contributors_raw = order.get("contributors") or order.get("source_strategies") or []
                if not isinstance(contributors_raw, list):
                    contributors_raw = [contributors_raw]

                contributors = dedupe_keep_order(
                    [normalize_strategy(v, strategy) for v in contributors_raw if safe_str(v)]
                )
                if strategy not in contributors:
                    contributors.insert(0, strategy)

                tags = dedupe_keep_order(
                    [
                        "planner",
                        strategy,
                        *contributors,
                        safe_str(order.get("asset_class")).lower(),
                    ]
                )

                by_symbol[symbol] = PlannedOrderAttribution(
                    symbol=symbol,
                    strategy=strategy,
                    source_strategies=contributors,
                    tags=tags,
                    plan_now_iso=plan_now_iso,
                    plan_path=str(self._plan_artifact_path),
                    attribution_source="full_execution_cycle_plan",
                )

            self._cache_by_symbol = by_symbol
            self._cache_mtime_ns = stat.st_mtime_ns

        except Exception:
            LOGGER.exception("Failed refreshing planner attribution cache from %s", self._plan_artifact_path)


# =============================================================================
# State store
# =============================================================================

class OpenStateStore:
    def __init__(self, state_path: Path) -> None:
        self._state_path = state_path

    def load(self) -> Dict[str, Dict[str, Any]]:
        raw = read_json(self._state_path, {})
        return raw if isinstance(raw, dict) else {}

    def save(self, state: Mapping[str, Any]) -> None:
        atomic_write_json(self._state_path, dict(state))


# =============================================================================
# IBKR adapter
# =============================================================================

class IBSyncBrokerGateway:
    def __init__(self, host: str, port: int, client_id: int) -> None:
        if IB is None:
            raise RuntimeError("ib_insync is not available in this environment")
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = IB()

    def connect(self) -> None:
        if self._ib.isConnected():
            return
        self._ib.connect(self._host, self._port, clientId=self._client_id, readonly=True, timeout=15)

    def disconnect(self) -> None:
        try:
            if self._ib.isConnected():
                self._ib.disconnect()
        except Exception:
            LOGGER.exception("Failed disconnecting IBKR client")

    def current_positions(self) -> List[Any]:
        return list(self._ib.positions())

    def recent_fills(self) -> List[Any]:
        return list(self._ib.fills())


# =============================================================================
# Attribution service
# =============================================================================

class StrategyAttributionService:
    def __init__(self, cfg: LedgerConfig, resolver: PlanAttributionResolver) -> None:
        self._cfg = cfg
        self._resolver = resolver

    def build_open_record_from_position(self, pos: Any, now: datetime) -> OpenPositionRecord:
        contract = getattr(pos, "contract", None)
        account = safe_str(getattr(pos, "account", None), None)
        symbol = safe_str(getattr(contract, "symbol", None)).upper()
        con_id = safe_int(getattr(contract, "conId", None), 0)
        sec_type = safe_str(getattr(contract, "secType", None))
        qty = safe_float(getattr(pos, "position", None), 0.0)
        avg_cost = safe_float(getattr(pos, "avgCost", None), 0.0)
        currency = safe_str(getattr(contract, "currency", None), "USD")

        planned = self._resolver.resolve_for_symbol(symbol)

        if planned:
            strategy = planned.strategy
            source_strategies = planned.source_strategies
            tags = dedupe_keep_order(["ibkr_paper", *planned.tags])
            plan_now_iso = planned.plan_now_iso
            plan_path = planned.plan_path
            attribution_source = planned.attribution_source
        else:
            strategy = self._cfg.default_strategy
            source_strategies = [self._cfg.default_strategy]
            tags = dedupe_keep_order(["ibkr_paper", self._cfg.default_strategy])
            plan_now_iso = None
            plan_path = None
            attribution_source = "config_default"

        return OpenPositionRecord(
            symbol=symbol,
            con_id=con_id,
            sec_type=sec_type,
            qty=qty,
            avg_cost=avg_cost,
            account_id=account,
            currency=currency,
            opened_at_utc=iso_z(now),
            strategy=strategy,
            source_strategies=source_strategies,
            tags=tags,
            plan_now_iso=plan_now_iso,
            plan_path=plan_path,
            attribution_source=attribution_source,
        )

    def resolve_close_strategy(self, rec: Mapping[str, Any], symbol: str) -> Tuple[str, List[str], str]:
        # Priority 1: persisted open-state attribution
        persisted_strategy = normalize_strategy(rec.get("strategy"), "")
        persisted_sources_raw = rec.get("source_strategies") or []
        if not isinstance(persisted_sources_raw, list):
            persisted_sources_raw = [persisted_sources_raw]
        persisted_sources = dedupe_keep_order(
            [normalize_strategy(v, "") for v in persisted_sources_raw if safe_str(v)]
        )

        if persisted_strategy and persisted_strategy != "manual":
            if persisted_strategy not in persisted_sources:
                persisted_sources.insert(0, persisted_strategy)
            return persisted_strategy, persisted_sources, "open_state"

        # Priority 2: latest plan artifact
        planned = self._resolver.resolve_for_symbol(symbol)
        if planned:
            return planned.strategy, planned.source_strategies, planned.attribution_source

        # Priority 3: tags
        tags = rec.get("tags") or []
        if not isinstance(tags, list):
            tags = [tags]
        tag_candidates = [safe_str(t).strip().lower() for t in tags if safe_str(t)]
        for candidate in tag_candidates:
            if candidate in {"alpha", "beta", "beta_trend", "gamma", "delta", "omega", "alpha_crypto", "alpha_forex"}:
                return candidate, [candidate], "tags"

        # Final fallback
        return self._cfg.default_strategy, [self._cfg.default_strategy], "config_default"


# =============================================================================
# Watcher core
# =============================================================================

class PaperLedgerWatcher:
    def __init__(
        self,
        cfg: LedgerConfig,
        gateway: BrokerGateway,
        state_store: OpenStateStore,
        attribution: StrategyAttributionService,
    ) -> None:
        self._cfg = cfg
        self._gateway = gateway
        self._state_store = state_store
        self._attribution = attribution

    def run_once(self) -> RunReport:
        now = utc_now()
        report = RunReport(ts_utc=iso_z(now), config_enabled=self._cfg.enabled)

        if not self._cfg.enabled:
            report.add_warning("ledger watcher disabled by config")
            self._write_report(report)
            return report

        open_state = self._state_store.load()
        current_open: Dict[str, OpenPositionRecord] = {}

        try:
            self._gateway.connect()
            positions = self._gateway.current_positions()
            fills_recent = self._recent_fills(now)
            report.add_detail(event="positions_loaded", count=len(positions))
            report.add_detail(event="fills_window", count=len(fills_recent), window_seconds=self._cfg.exec_window_seconds)
        except Exception as exc:
            msg = f"IBKR connect/read failure: {type(exc).__name__}: {exc}"
            report.add_error(msg)
            LOGGER.exception(msg)
            self._safe_notify_error(msg)
            self._write_report(report)
            return report
        finally:
            self._gateway.disconnect()

        # Build current open snapshot with planner attribution
        for pos in positions:
            try:
                rec = self._attribution.build_open_record_from_position(pos, now)
                current_open[rec.key()] = rec
            except Exception:
                LOGGER.exception("Failed building OpenPositionRecord from position %r", pos)
                report.add_warning("position skipped due to parse failure")

        # Detect closes: previously open but no longer open now
        close_candidates = [k for k in open_state.keys() if k not in current_open]
        report.add_detail(event="close_candidates", count=len(close_candidates))

        for key in close_candidates:
            rec = open_state.get(key) or {}
            try:
                self._log_close_candidate(rec=rec, key=key, now=now, report=report)
            except Exception as exc:
                msg = f"close candidate logging failed key={key}: {type(exc).__name__}: {exc}"
                LOGGER.exception(msg)
                report.add_error(msg)

        # Persist fresh current open state
        persisted_state = {
            key: self._serialize_open_record(value)
            for key, value in current_open.items()
        }
        self._state_store.save(persisted_state)

        self._write_report(report)
        return report

    def _recent_fills(self, now: datetime) -> List[Any]:
        cutoff = now - timedelta(seconds=float(self._cfg.exec_window_seconds))
        out: List[Any] = []
        for fill in self._gateway.recent_fills():
            ts = getattr(fill, "time", None)
            if ts is None or not hasattr(ts, "astimezone"):
                continue
            try:
                ts_utc = ts.astimezone(timezone.utc)
            except Exception:
                continue
            if ts_utc >= cutoff:
                out.append(fill)
        return out

    def _log_close_candidate(
        self,
        rec: Mapping[str, Any],
        key: str,
        now: datetime,
        report: RunReport,
    ) -> None:
        symbol = safe_str(rec.get("symbol")).upper()
        con_id = safe_int(rec.get("conId"), 0)
        sec_type = safe_str(rec.get("secType"))

        if not symbol or con_id <= 0 or not sec_type:
            report.add_warning(f"close candidate missing required fields key={key}")
            return

        qty_open = safe_float(rec.get("qty"), 0.0)
        avg_cost = safe_float(rec.get("avg_cost"), 0.0)
        notional = abs(qty_open) * avg_cost
        # `side` reflects the OPEN side of the position (BUY for long open, SELL for short open).
        side = "BUY" if qty_open > 0 else "SELL"

        strategy, source_strategies, attribution_source = self._attribution.resolve_close_strategy(rec, symbol)

        tags_raw = rec.get("tags") or ["ibkr_paper", strategy]
        if not isinstance(tags_raw, list):
            tags_raw = [tags_raw]
        tags = dedupe_keep_order([safe_str(t).lower() for t in tags_raw if safe_str(t)])
        if strategy not in tags:
            tags.append(strategy)

        extra: Dict[str, Any] = {
            "source": "ibkr_paper_ledger_watcher",
            "state_key_hash": hash_key([key]),
            "close_key": key,
            "conId": con_id,
            "secType": sec_type,
            "currency": safe_str(rec.get("currency"), "USD"),
            "plan_path": rec.get("plan_path"),
            "plan_now_iso": rec.get("plan_now_iso"),
            "source_strategies": source_strategies,
            "attribution_source": attribution_source,
        }

        # GAP-A003: try forward-only deterministic matcher to upgrade close trust.
        match_result = self._try_match_snapshot_diff_close(
            strategy=strategy,
            symbol=symbol,
            expected_open_side=side,
            close_time=now,
        )

        quantity = abs(qty_open)
        fill_price = avg_cost
        pnl = 0.0

        if match_result is not None:
            matched = match_result
            entry_price = matched.fill_price
            exit_price = avg_cost
            matched_qty = min(matched.quantity, abs(qty_open)) if matched.quantity > 0 else abs(qty_open)
            multiplier = matched.contract_multiplier
            if side == "BUY":
                gross_pnl = (exit_price - entry_price) * matched_qty * multiplier
            else:
                gross_pnl = (entry_price - exit_price) * matched_qty * multiplier
            quantity = matched_qty
            fill_price = exit_price
            pnl = gross_pnl
            notional = matched_qty * exit_price
            extra.update({
                "pnl_untrusted": False,
                "matched_open_fill_id": matched.fill_id,
                "matched_open_fill_time_utc": matched.fill_time_utc,
                "matched_open_fill_source": matched.source_path,
                "matcher": SNAPSHOT_DIFF_MATCHER_NAME,
                "pnl_trusted_reason": "matched_opening_fill_for_snapshot_diff_close",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "contract_multiplier": multiplier,
                "gross_pnl": gross_pnl,
            })
            report.add_detail(
                event="snapshot_diff_close_trusted",
                symbol=symbol,
                strategy=strategy,
                matched_open_fill_id=matched.fill_id,
                matcher=SNAPSHOT_DIFF_MATCHER_NAME,
            )
        else:
            extra.update({
                "pnl_untrusted": True,
                "pnl_untrusted_reason": "symbol_close_detected_without_fill_matcher",
            })

        tr = TradeResult(
            strategy=strategy,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            notional=notional,
            pnl=pnl,
            entry_time_utc=safe_str(rec.get("opened_at_utc"), iso_z(now)),
            exit_time_utc=iso_z(now),
            is_live=False,
            broker="ibkr",
            account_id=safe_str(rec.get("account_id"), None) or None,
            regime=resolve_regime_label(now_utc=now),
            tags=tags,
            extra=extra,
        )

        log_path = log_trade_result(tr)
        report.writes_trade_results += 1
        report.add_detail(
            event="trade_result_written",
            symbol=symbol,
            conId=con_id,
            strategy=strategy,
            source_strategies=source_strategies,
            attribution_source=attribution_source,
            log_path=str(log_path),
            pnl_untrusted=bool(extra.get("pnl_untrusted")),
        )

        if self._cfg.notify_on_write:
            self._safe_notify_write(symbol, strategy, log_path)

    def _try_match_snapshot_diff_close(
        self,
        *,
        strategy: str,
        symbol: str,
        expected_open_side: str,
        close_time: datetime,
    ) -> Optional[MatchedOpeningFill]:
        """
        Failure-soft wrapper around the GAP-A003 snapshot-diff matcher.

        Any error (corrupt fill record, unreadable trade history, parse failure)
        falls back to None — the caller then preserves the legacy untrusted
        close behavior.
        """
        try:
            consumed = collect_consumed_open_fill_ids(
                self._cfg.trades_dir,
                lookback_days=self._cfg.matcher_lookback_days,
                reference_time=close_time,
            )
            return find_matched_opening_fill_for_snapshot_close(
                strategy=strategy,
                symbol=symbol,
                expected_open_side=expected_open_side,
                close_time=close_time,
                consumed_ids=consumed,
                fills_dir=self._cfg.fills_dir,
                lookback_days=self._cfg.matcher_lookback_days,
            )
        except Exception:
            LOGGER.exception(
                "snapshot_diff_fifo_v1 matcher errored for symbol=%s strategy=%s — falling back to untrusted",
                symbol,
                strategy,
            )
            return None

    def _serialize_open_record(self, rec: OpenPositionRecord) -> Dict[str, Any]:
        return {
            "symbol": rec.symbol,
            "conId": rec.con_id,
            "secType": rec.sec_type,
            "qty": rec.qty,
            "avg_cost": rec.avg_cost,
            "account_id": rec.account_id,
            "currency": rec.currency,
            "opened_at_utc": rec.opened_at_utc,
            "strategy": rec.strategy,
            "source_strategies": rec.source_strategies,
            "tags": rec.tags,
            "plan_now_iso": rec.plan_now_iso,
            "plan_path": rec.plan_path,
            "attribution_source": rec.attribution_source,
        }

    def _write_report(self, report: RunReport) -> None:
        self._cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self._cfg.reports_dir / f"ibkr_paper_ledger_report_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_write_json(path, report.to_dict())
        LOGGER.info("ledger report written: %s", path)

    def _safe_notify_write(self, symbol: str, strategy: str, log_path: Any) -> None:
        try:
            notify(f"CHAD paper ledger wrote trade result: {symbol} strategy={strategy} path={log_path}")
        except NotifyError:
            LOGGER.exception("Telegram notify_on_write failed")

    def _safe_notify_error(self, msg: str) -> None:
        if not self._cfg.notify_on_error:
            return
        try:
            notify(f"CHAD paper ledger watcher error: {msg}")
        except NotifyError:
            LOGGER.exception("Telegram notify_on_error failed")


# =============================================================================
# Factory
# =============================================================================

class WatcherFactory:
    @staticmethod
    def build(config_path: Path) -> PaperLedgerWatcher:
        cfg = LedgerConfig.load(config_path)
        resolver = PlanAttributionResolver(cfg.plan_artifact_path)
        attribution = StrategyAttributionService(cfg, resolver)
        state_store = OpenStateStore(cfg.state_path)
        gateway = IBSyncBrokerGateway(cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id)
        return PaperLedgerWatcher(
            cfg=cfg,
            gateway=gateway,
            state_store=state_store,
            attribution=attribution,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHAD IBKR Paper Ledger Watcher")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH_DEFAULT,
        help=f"Path to config JSON (default: {CONFIG_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Python logging level",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, safe_str(level, "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        watcher = WatcherFactory.build(args.config)
        report = watcher.run_once()
        print(json.dumps(report.to_dict(), indent=2, default=json_default))
        return 0 if report.ok else 1
    except Exception as exc:
        LOGGER.exception("Unhandled fatal error in ibkr_paper_ledger_watcher")
        print(
            json.dumps(
                {
                    "ok": False,
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                    "ts_utc": iso_z(),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
