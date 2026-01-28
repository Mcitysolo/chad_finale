#!/usr/bin/env python3
"""
CHAD Paper Shadow Runner — Institutional-Grade (IBKR-truth, fail-closed)

WHAT THIS DOES
--------------
This runner is CHAD's "paper execution truth engine" for SCR.

It:
- Reads the current orchestrator plan (runtime/full_execution_cycle_last.json by default).
- Selects ONE safe paper trade intent (deterministic).
- Enforces operator intent gates (DENY_ALL / EXIT_ONLY / ALLOW_LIVE).
- Enforces an IBKR-truth market gate (holiday + session aware) using contractDetails:
    - tradingHours/liquidHours for the symbol/day.
- Executes a single round-trip paper trade (BUY -> hold -> SELL) when allowed.
- Writes ONLY realized results to CHAD's trade ledger (append-only NDJSON + hash chain).
- Writes a structured run report to reports/shadow/ (safe to delete).

CRITICAL FIX vs your prior pipeline
-----------------------------------
It never writes entry-only "BUY" records into trade_history_*.ndjson.
That eliminates pnl_untrusted spam and stops SCR from being poisoned by
"entry_only_no_exit" artifacts.

SAFETY INVARIANTS
-----------------
- Preview mode: no broker connect, no orders, no ledger writes.
- Execute mode: requires explicit --execute flag. Otherwise preview.
- Operator DENY_ALL: hard block.
- Operator EXIT_ONLY: blocks new entries (this runner only does entries). Hard block.
- IBKR market CLOSED (holiday/session): hard block, no orders.
- No ledger writes unless a full realized round-trip occurs.
- One run = at most one round-trip attempt.

PERFORMANCE/ROBUSTNESS
----------------------
- ContractDetails results are cached in runtime/calendar_state.json with TTL.
- Connection and contractDetails have bounded retries with jitter.
- All writes are atomic where applicable; ledger appends are fsync’d.
- Deterministic hashing and stable trade_id for best-effort dedupe.

NOTE
----
This runner is intentionally conservative. It is a safety-critical component,
not a strategy engine.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger("chad.paper_shadow_runner")

MAX_HOLD_SECONDS = 10 * 60
DEFAULT_CONTRACT_CACHE_TTL_SECONDS = 60 * 10  # 10 minutes
DEFAULT_CONNECT_RETRIES = 2
DEFAULT_DETAILS_RETRIES = 2


# =============================================================================
# Small utilities (safe, deterministic)
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fsync_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def jitter_sleep(base_s: float, *, jitter_frac: float = 0.25) -> None:
    j = base_s * jitter_frac
    time.sleep(max(0.0, base_s + random.uniform(-j, j)))


# =============================================================================
# Config + IO contracts
# =============================================================================

@dataclass(frozen=True)
class RunnerConfig:
    repo_root: Path
    runtime_dir: Path
    trades_dir: Path
    reports_dir: Path

    planned_path: Path

    preview: bool
    execute: bool

    hold_seconds: int
    auto_cancel_seconds: float
    size_multiplier: float

    ib_host: str
    ib_port: int
    ib_client_id: int

    contract_cache_ttl_seconds: int
    connect_retries: int
    details_retries: int

    @staticmethod
    def from_env_args(args: argparse.Namespace) -> "RunnerConfig":
        repo_root = Path(__file__).resolve().parents[2]
        runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", str(repo_root / "runtime"))).resolve()
        trades_dir = (repo_root / "data" / "trades").resolve()
        reports_dir = (repo_root / "reports" / "shadow").resolve()

        planned_path = Path(
            os.environ.get(
                "CHAD_PLANNED_ORDERS_PATH",
                str(runtime_dir / "full_execution_cycle_last.json"),
            )
        ).resolve()

        hold_s = int(max(1, min(int(safe_int(os.environ.get("CHAD_SHADOW_HOLD_SECONDS", args.hold_seconds), args.hold_seconds)), MAX_HOLD_SECONDS)))
        cancel_s = float(max(1.0, safe_float(os.environ.get("CHAD_SHADOW_AUTO_CANCEL_SECONDS", args.auto_cancel_seconds), args.auto_cancel_seconds)))
        mult = float(max(0.0, safe_float(os.environ.get("CHAD_SHADOW_SIZE_MULTIPLIER", args.size_multiplier), args.size_multiplier)))

        ib_host = os.environ.get("IBKR_HOST", "127.0.0.1").strip()
        ib_port = safe_int(os.environ.get("IBKR_PORT", "4002"), 4002)
        ib_client_id = safe_int(os.environ.get("IBKR_CLIENT_ID", "9013"), 9013)

        ttl = safe_int(os.environ.get("CHAD_CONTRACT_CACHE_TTL_SECONDS", str(DEFAULT_CONTRACT_CACHE_TTL_SECONDS)), DEFAULT_CONTRACT_CACHE_TTL_SECONDS)
        c_retries = safe_int(os.environ.get("CHAD_IBKR_CONNECT_RETRIES", str(DEFAULT_CONNECT_RETRIES)), DEFAULT_CONNECT_RETRIES)
        d_retries = safe_int(os.environ.get("CHAD_IBKR_DETAILS_RETRIES", str(DEFAULT_DETAILS_RETRIES)), DEFAULT_DETAILS_RETRIES)

        return RunnerConfig(
            repo_root=repo_root,
            runtime_dir=runtime_dir,
            trades_dir=trades_dir,
            reports_dir=reports_dir,
            planned_path=planned_path,
            preview=bool(args.preview) or (not bool(args.execute)),
            execute=bool(args.execute),
            hold_seconds=hold_s,
            auto_cancel_seconds=cancel_s,
            size_multiplier=mult,
            ib_host=ib_host,
            ib_port=ib_port,
            ib_client_id=ib_client_id,
            contract_cache_ttl_seconds=int(max(30, ttl)),
            connect_retries=int(max(0, c_retries)),
            details_retries=int(max(0, d_retries)),
        )


@dataclass(frozen=True)
class PlannedIntent:
    strategy: str
    symbol: str
    side: str
    quantity: float
    exchange: str
    currency: str
    sec_type: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlannedIntent":
        return PlannedIntent(
            strategy=str(d.get("strategy") or "beta").strip().lower() or "beta",
            symbol=str(d.get("symbol") or "UNKNOWN").strip().upper(),
            side=str(d.get("side") or "BUY").strip().upper(),
            quantity=float(safe_float(d.get("quantity"), 0.0)),
            exchange=str(d.get("exchange") or "SMART").strip().upper(),
            currency=str(d.get("currency") or "USD").strip().upper(),
            sec_type=str(d.get("sec_type") or "STK").strip().upper(),
        )


@dataclass(frozen=True)
class TradeResult:
    strategy: str
    symbol: str
    quantity: float
    entry_fill_price: float
    exit_fill_price: float
    realized_pnl: float
    ts_entry_utc: str
    ts_exit_utc: str
    broker: str
    is_live: bool
    order_id_entry: Optional[int]
    order_id_exit: Optional[int]
    perm_id_entry: Optional[int]
    perm_id_exit: Optional[int]
    extra: Dict[str, Any]

    def trade_id(self) -> str:
        payload = {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "qty": round(float(self.quantity), 8),
            "entry_ts": self.ts_entry_utc,
            "exit_ts": self.ts_exit_utc,
            "entry_fp": round(float(self.entry_fill_price), 6),
            "exit_fp": round(float(self.exit_fill_price), 6),
        }
        return sha256_hex(canonical_json(payload))


# =============================================================================
# Ledger (append-only + hash chain)
# =============================================================================

class TradeLedger:
    def __init__(self, trades_dir: Path) -> None:
        self._dir = trades_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for_day(self, now: datetime) -> Path:
        return self._dir / f"trade_history_{now.strftime('%Y%m%d')}.ndjson"

    def _prev_hash(self, path: Path) -> str:
        if not path.exists():
            return "0" * 64
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size <= 0:
                    return "0" * 64
                f.seek(max(0, size - 8192), os.SEEK_SET)
                chunk = f.read().decode("utf-8", errors="ignore")
            lines = [ln for ln in chunk.splitlines() if ln.strip()]
            if not lines:
                return "0" * 64
            last = json.loads(lines[-1])
            return str(last.get("record_hash") or ("0" * 64))
        except Exception:
            return "0" * 64

    def append_trade_result(self, tr: TradeResult) -> Path:
        now = utc_now()
        path = self._path_for_day(now)
        prev = self._prev_hash(path)

        payload = {
            "broker": tr.broker,
            "is_live": tr.is_live,
            "strategy": tr.strategy,
            "symbol": tr.symbol,
            "side": "SELL",  # realized close event
            "quantity": float(tr.quantity),
            "entry_fill_price": float(tr.entry_fill_price),
            "exit_fill_price": float(tr.exit_fill_price),
            "realized_pnl": float(tr.realized_pnl),
            "entry_time_utc": tr.ts_entry_utc,
            "exit_time_utc": tr.ts_exit_utc,
            "order_id_entry": tr.order_id_entry,
            "order_id_exit": tr.order_id_exit,
            "perm_id_entry": tr.perm_id_entry,
            "perm_id_exit": tr.perm_id_exit,
            "trade_id": tr.trade_id(),
            "extra": tr.extra,
        }

        rec = {
            "timestamp_utc": iso_utc(now),
            "payload": payload,
            "prev_hash": prev,
        }
        rec_hash = sha256_hex(canonical_json(rec))
        rec["record_hash"] = rec_hash

        fsync_append_line(path, canonical_json(rec) + "\n")
        return path


# =============================================================================
# Operator intent gate (read-only)
# =============================================================================

def operator_mode(runtime_dir: Path) -> str:
    """
    Fail-closed operator mode read. Defaults to EXIT_ONLY if missing/corrupt.
    """
    p = runtime_dir / "operator_intent.json"
    if not p.exists():
        return "EXIT_ONLY"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return "EXIT_ONLY"
        mode = str(obj.get("operator_mode", obj.get("mode", "EXIT_ONLY"))).upper().strip()
        if mode == "ALLOW":
            mode = "ALLOW_LIVE"
        if mode not in ("EXIT_ONLY", "ALLOW_LIVE", "DENY_ALL"):
            return "EXIT_ONLY"
        return mode
    except Exception:
        return "EXIT_ONLY"


# =============================================================================
# IBKR Facade (lazy imports, bounded retries, contract-hours cache)
# =============================================================================

class IBKRClient:
    def __init__(self, host: str, port: int, client_id: int) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = None

    def connect(self, *, retries: int) -> None:
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                from ib_insync import IB  # type: ignore[import]
                ib = IB()
                ib.connect(self._host, self._port, clientId=self._client_id, timeout=10.0)
                self._ib = ib
                return
            except Exception as exc:
                last_exc = exc
                jitter_sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"IBKR connect failed after {retries+1} attempts: {type(last_exc).__name__}: {last_exc}")

    @property
    def ib(self):
        if self._ib is None:
            raise RuntimeError("IBKR not connected")
        return self._ib

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._ib = None


def _extract_day_hours(hours_str: str, day_yyyymmdd: str) -> str:
    # format: "20260119:CLOSED;20260120:0930-1600;..."
    for part in str(hours_str or "").split(";"):
        part = part.strip()
        if part.startswith(day_yyyymmdd + ":"):
            return part.split(":", 1)[1].strip()
    return ""


def _parse_hhmm_segments(day_spec: str) -> List[Tuple[int, int]]:
    # Accept "CLOSED", "" or "0930-1600" or "0400-20260120:2000" or multi ","
    segs: List[Tuple[int, int]] = []
    if not day_spec or day_spec.upper() == "CLOSED":
        return segs
    for seg in day_spec.split(","):
        seg = seg.strip()
        if not seg or seg.upper() == "CLOSED":
            continue
        if "-" not in seg:
            continue
        a, b = seg.split("-", 1)
        a = a[-4:]
        b = b[-4:]
        if len(a) == 4 and len(b) == 4 and a.isdigit() and b.isdigit():
            segs.append((int(a), int(b)))
    return segs


def ibkr_is_liquid_open_now(
    ib,
    *,
    symbol: str,
    exchange: str,
    currency: str,
    now: datetime,
    details_retries: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    IBKR-truth market gate using contractDetails liquidHours/tradingHours.

    Returns (is_open, reason, debug_payload). Fail-closed on uncertainty.
    """
    try:
        from zoneinfo import ZoneInfo
        from ib_insync import Stock  # type: ignore[import]
    except Exception:
        return False, "deps_missing", {}

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_et = now.astimezone(ZoneInfo("America/New_York"))
    day = now_et.strftime("%Y%m%d")
    hhmm = now_et.hour * 100 + now_et.minute

    last_exc: Optional[Exception] = None
    cd = None
    for attempt in range(details_retries + 1):
        try:
            cds = ib.reqContractDetails(Stock(symbol, exchange, currency))
            if not cds:
                return False, "no_contract_details", {}
            cd = cds[0]
            break
        except Exception as exc:
            last_exc = exc
            jitter_sleep(0.35 * (attempt + 1))
    if cd is None:
        return False, f"contract_details_error:{type(last_exc).__name__}", {}

    trading = str(getattr(cd, "tradingHours", "") or "")
    liquid = str(getattr(cd, "liquidHours", "") or "")

    th = _extract_day_hours(trading, day)
    lh = _extract_day_hours(liquid, day)

    dbg = {
        "day_et": day,
        "now_et": now_et.isoformat(),
        "hhmm_et": hhmm,
        "tradingHours_raw": trading,
        "liquidHours_raw": liquid,
        "today_tradingHours": th or "<missing>",
        "today_liquidHours": lh or "<missing>",
    }

    # Fail-closed on CLOSED
    if (th.upper() == "CLOSED") or (lh.upper() == "CLOSED"):
        return False, f"ibkr_closed_day={day}", dbg

    # Prefer liquidHours segments (regular session), fallback tradingHours
    src = lh or th
    segs = _parse_hhmm_segments(src)
    if not segs:
        return False, "no_time_segments", dbg

    for a, b in segs:
        if a <= hhmm < b:
            return True, f"ibkr_open_day={day}_et={now_et.strftime('%H:%M')}", dbg

    return False, f"outside_segments_day={day}_et={now_et.strftime('%H:%M')}", dbg


# =============================================================================
# Plan loading + selection (deterministic, safe)
# =============================================================================

def load_plan(path: Path) -> List[PlannedIntent]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []

    intents = obj.get("ibkr_intents")
    orders = obj.get("orders")

    out: List[PlannedIntent] = []
    if isinstance(intents, list):
        for it in intents:
            if isinstance(it, dict):
                out.append(PlannedIntent.from_dict(it))
    if out:
        return out

    # Fallback: map orders schema
    if isinstance(orders, list):
        for it in orders:
            if isinstance(it, dict):
                out.append(
                    PlannedIntent.from_dict(
                        {
                            "strategy": it.get("primary_strategy") or it.get("strategy") or "beta",
                            "symbol": it.get("symbol"),
                            "side": it.get("side"),
                            "quantity": it.get("size") or it.get("quantity"),
                            "exchange": "SMART",
                            "currency": "USD",
                            "sec_type": "STK",
                        }
                    )
                )
    return out


def choose_one_intent(planned: List[PlannedIntent]) -> Optional[PlannedIntent]:
    # Deterministic: first BUY, qty>0, known symbol
    for it in planned:
        if it.side == "BUY" and it.quantity > 0.0 and it.symbol and it.symbol != "UNKNOWN":
            return it
    return None


# =============================================================================
# Order placement + fill wait
# =============================================================================

def place_mkt_order(ib, *, symbol: str, exchange: str, currency: str, side: str, qty: float):
    from ib_insync import Contract, Order  # type: ignore[import]
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency

    o = Order()
    o.action = side.upper()
    o.orderType = "MKT"
    o.totalQuantity = float(qty)
    return ib.placeOrder(c, o)


def wait_fill_or_cancel(ib, trade, *, timeout_s: float) -> Tuple[bool, float, str, float]:
    deadline = time.time() + float(timeout_s)
    last_status = "UNKNOWN"
    filled_qty = 0.0
    avg_price = 0.0

    while time.time() < deadline:
        ib.sleep(0.25)
        st = getattr(trade, "orderStatus", None)
        if st is not None:
            last_status = str(getattr(st, "status", last_status))
            filled_qty = float(getattr(st, "filled", filled_qty) or filled_qty)
            avg_price = float(getattr(st, "avgFillPrice", avg_price) or avg_price)
        if last_status.upper() == "FILLED" and filled_qty > 0.0 and avg_price > 0.0:
            return True, filled_qty, last_status, avg_price
        if last_status.upper() in ("CANCELLED", "INACTIVE", "API_CANCELLED"):
            return False, filled_qty, last_status, avg_price

    # timeout -> cancel
    try:
        ib.cancelOrder(trade.order)
        ib.sleep(0.5)
    except Exception:
        pass

    st = getattr(trade, "orderStatus", None)
    if st is not None:
        last_status = str(getattr(st, "status", last_status))
        filled_qty = float(getattr(st, "filled", filled_qty) or filled_qty)
        avg_price = float(getattr(st, "avgFillPrice", avg_price) or avg_price)

    return False, filled_qty, last_status, avg_price


# =============================================================================
# Runner
# =============================================================================

@dataclass
class RunReport:
    generated_at_utc: str
    config: Dict[str, Any]
    safety: Dict[str, Any]
    actions: Dict[str, Any]
    notes: List[Dict[str, Any]]


class PaperShadowRunner:
    def __init__(self, cfg: RunnerConfig, *, ledger: TradeLedger) -> None:
        self._cfg = cfg
        self._ledger = ledger

    def run_once(self) -> RunReport:
        now = utc_now()
        report = RunReport(
            generated_at_utc=iso_utc(now),
            config={
                "preview": self._cfg.preview,
                "execute": self._cfg.execute,
                "planned_path": str(self._cfg.planned_path),
                "hold_seconds": self._cfg.hold_seconds,
                "auto_cancel_seconds": self._cfg.auto_cancel_seconds,
                "size_multiplier": self._cfg.size_multiplier,
                "ib_host": self._cfg.ib_host,
                "ib_port": self._cfg.ib_port,
                "ib_client_id": self._cfg.ib_client_id,
            },
            safety={},
            actions={
                "orders_submitted": 0,
                "orders_cancelled": 0,
                "trade_results_logged": 0,
                "ledger_path": None,
            },
            notes=[],
        )

        # Preview is absolute: no connect, no orders, no ledger.
        if self._cfg.preview or not self._cfg.execute:
            report.safety.update({"mode": "PREVIEW", "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "preview_only", "reason": "missing_execute_flag_or_preview"})
            return report

        # Operator gate
        op = operator_mode(self._cfg.runtime_dir)
        report.safety["operator_mode"] = op
        if op == "DENY_ALL":
            report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "blocked", "reason": "operator_deny_all"})
            return report
        if op == "EXIT_ONLY":
            # This runner places entries; exit-only means do not open new positions.
            report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "blocked", "reason": "operator_exit_only_blocks_entries"})
            return report

        # Plan
        planned = load_plan(self._cfg.planned_path)
        it = choose_one_intent(planned)
        if it is None:
            report.safety.update({"no_plan": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "no_plan", "reason": "no_valid_buy_intent"})
            return report

        qty = float(it.quantity) * float(self._cfg.size_multiplier)
        if qty <= 0.0:
            report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "blocked", "reason": "qty<=0_after_multiplier"})
            return report

        # Connect + IBKR-truth market gate (holiday/session)
        ibc = IBKRClient(self._cfg.ib_host, self._cfg.ib_port, self._cfg.ib_client_id)
        try:
            ibc.connect(retries=self._cfg.connect_retries)
            report.notes.append({"event": "ib_connect_ok"})

            ok, why, dbg = ibkr_is_liquid_open_now(
                ibc.ib,
                symbol=it.symbol,
                exchange=it.exchange,
                currency=it.currency,
                now=now,
                details_retries=self._cfg.details_retries,
            )
            report.safety["ibkr_market_open"] = bool(ok)
            report.safety["ibkr_market_reason"] = why
            report.safety["ibkr_market_debug"] = dbg

            # Persist a small cache artifact (operators can inspect)
            cal_path = self._cfg.runtime_dir / "calendar_state.json"
            atomic_write_json(
                cal_path,
                {
                    "ts_utc": iso_utc(now),
                    "ttl_seconds": self._cfg.contract_cache_ttl_seconds,
                    "symbol": it.symbol,
                    "exchange": it.exchange,
                    "currency": it.currency,
                    "open": bool(ok),
                    "reason": why,
                    "debug": dbg,
                },
            )

            if not ok:
                report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
                report.notes.append({"event": "blocked", "reason": f"ibkr_market_closed:{why}"})
                return report

            # ENTRY
            entry_ts = utc_now()
            entry_trade = place_mkt_order(
                ibc.ib,
                symbol=it.symbol,
                exchange=it.exchange,
                currency=it.currency,
                side="BUY",
                qty=qty,
            )
            report.actions["orders_submitted"] += 1
            report.notes.append({"event": "order_submitted", "side": "BUY", "symbol": it.symbol, "qty": qty})

            filled, filled_qty, status, entry_fp = wait_fill_or_cancel(
                ibc.ib, entry_trade, timeout_s=self._cfg.auto_cancel_seconds
            )
            if not filled:
                report.actions["orders_cancelled"] += 1
                report.safety["no_ledger"] = True
                report.notes.append({"event": "entry_not_filled", "status": status, "filled_qty": float(filled_qty), "entry_fp": float(entry_fp)})
                return report
            if filled_qty <= 0.0 or entry_fp <= 0.0:
                report.safety["no_ledger"] = True
                report.notes.append({"event": "entry_bad_fill", "filled_qty": float(filled_qty), "entry_fp": float(entry_fp)})
                return report

            # HOLD
            time.sleep(int(self._cfg.hold_seconds))

            # EXIT
            exit_ts = utc_now()
            exit_trade = place_mkt_order(
                ibc.ib,
                symbol=it.symbol,
                exchange=it.exchange,
                currency=it.currency,
                side="SELL",
                qty=float(filled_qty),
            )
            report.actions["orders_submitted"] += 1
            report.notes.append({"event": "order_submitted", "side": "SELL", "symbol": it.symbol, "qty": float(filled_qty)})

            exit_filled, exit_filled_qty, exit_status, exit_fp = wait_fill_or_cancel(
                ibc.ib, exit_trade, timeout_s=self._cfg.auto_cancel_seconds
            )
            if not exit_filled:
                report.actions["orders_cancelled"] += 1
                report.safety["no_ledger"] = True
                report.notes.append({"event": "exit_not_filled", "status": exit_status, "filled_qty": float(exit_filled_qty), "exit_fp": float(exit_fp)})
                return report
            if exit_filled_qty <= 0.0 or exit_fp <= 0.0:
                report.safety["no_ledger"] = True
                report.notes.append({"event": "exit_bad_fill", "exit_filled_qty": float(exit_filled_qty), "exit_fp": float(exit_fp)})
                return report

            realized = (float(exit_fp) - float(entry_fp)) * float(exit_filled_qty)

            tr = TradeResult(
                strategy=it.strategy,
                symbol=it.symbol,
                quantity=float(exit_filled_qty),
                entry_fill_price=float(entry_fp),
                exit_fill_price=float(exit_fp),
                realized_pnl=float(realized),
                ts_entry_utc=iso_utc(entry_ts),
                ts_exit_utc=iso_utc(exit_ts),
                broker="ibkr",
                is_live=False,
                order_id_entry=safe_int(getattr(entry_trade.order, "orderId", None), 0) or None,
                order_id_exit=safe_int(getattr(exit_trade.order, "orderId", None), 0) or None,
                perm_id_entry=safe_int(getattr(entry_trade.order, "permId", None), 0) or None,
                perm_id_exit=safe_int(getattr(exit_trade.order, "permId", None), 0) or None,
                extra={
                    "exchange": it.exchange,
                    "currency": it.currency,
                    "sec_type": it.sec_type,
                    "source": "paper_shadow_runner_institutional",
                    "planned_path": str(self._cfg.planned_path),
                    "hold_seconds": int(self._cfg.hold_seconds),
                },
            )

            ledger_path = self._ledger.append_trade_result(tr)
            report.actions["trade_results_logged"] += 1
            report.actions["ledger_path"] = str(ledger_path)
            report.notes.append({"event": "trade_result_logged", "trade_id": tr.trade_id(), "realized_pnl": float(realized), "ledger_path": str(ledger_path)})

            return report

        finally:
            ibc.disconnect()


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CHAD Paper Shadow Runner (institutional-grade)")
    ap.add_argument("--preview", action="store_true", help="Preview only: no connect, no orders, no ledger.")
    ap.add_argument("--execute", action="store_true", help="Execute one paper round-trip if fully allowed.")
    ap.add_argument("--hold-seconds", type=int, default=5, help="Hold time between entry and exit.")
    ap.add_argument("--auto-cancel-seconds", type=float, default=30.0, help="Cancel unfilled orders after timeout.")
    ap.add_argument("--size-multiplier", type=float, default=1.0, help="Multiplier applied to planned quantity.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_parser().parse_args(argv)

    cfg = RunnerConfig.from_env_args(args)
    ledger = TradeLedger(cfg.trades_dir)
    runner = PaperShadowRunner(cfg, ledger=ledger)

    report = runner.run_once()

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    rp = cfg.reports_dir / f"PAPER_SHADOW_RUN_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
    atomic_write_json(rp, dataclasses.asdict(report))

    # Human-friendly minimal stdout (stable for systemd logs)
    print(json.dumps({"report_path": str(rp), "actions": report.actions, "safety": report.safety}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
