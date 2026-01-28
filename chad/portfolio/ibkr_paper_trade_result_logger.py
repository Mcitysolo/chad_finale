#!/usr/bin/env python3
"""
CHAD — IBKR Paper Trade Result Logger (Production)

Writes append-only, fsync-safe, hash-chained NDJSON records into:
  data/trades/trade_history_YYYYMMDD.ndjson

Key guarantees:
- concurrent-safe (fcntl lock)
- deterministic hash chaining (prev_hash + canonical JSON)
- no secrets written
- supports realized PnL override via IBKRPaperOrderEvent.realized_pnl
"""

from __future__ import annotations

import datetime as _dt
import fcntl  # Linux/Unix only; target host is Ubuntu
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
DATA_TRADES_DIR = ROOT / "data" / "trades"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _utc_ymd() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="strict")).hexdigest()


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except Exception:
        return default


def _ledger_path_today() -> Path:
    DATA_TRADES_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_TRADES_DIR / f"trade_history_{_utc_ymd()}.ndjson"


def _read_last_record_metadata(fp) -> tuple[int, str]:
    """
    Return (last_seq, last_hash). If empty/unreadable -> (0, genesis).
    """
    genesis = "0" * 64
    try:
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        if size <= 0:
            return 0, genesis

        # Read tail chunk(s) until we find at least one newline.
        chunk_size = 8192
        pos = size
        buf = b""
        while pos > 0:
            step = min(chunk_size, pos)
            pos -= step
            fp.seek(pos, os.SEEK_SET)
            buf = fp.read(step) + buf
            if b"\n" in buf:
                break

        lines = [ln for ln in buf.splitlines() if ln.strip()]
        if not lines:
            return 0, genesis

        last = lines[-1].decode("utf-8", errors="strict")
        rec = json.loads(last)
        seq = _safe_int(rec.get("sequence_id"), 0)
        rh = _safe_str(rec.get("record_hash"), genesis) or genesis
        if not _HASH_RE.fullmatch(rh):
            rh = genesis
        return seq, rh
    except Exception:
        return 0, genesis


@dataclass(frozen=True)
class IBKRPaperOrderEvent:
    """
    A single IBKR paper fill event.

    - realized_pnl: pass a float for exit/close fills to write trusted PnL.
      For entry-only fills, leave None and optionally tag pnl_untrusted in raw_intent.
    """
    strategy: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    fill_price: float
    notional_estimate: float

    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    order_type: str = "MKT"

    order_id: Optional[int] = None
    perm_id: Optional[int] = None

    raw_intent: Optional[Dict[str, Any]] = None
    realized_pnl: Optional[float] = None
    source: str = "paper_shadow_runner"


def _build_payload(event: IBKRPaperOrderEvent) -> Dict[str, Any]:
    strategy = _safe_str(event.strategy).strip().lower() or "unknown"
    symbol = _safe_str(event.symbol).strip().upper() or "UNKNOWN"
    side = _safe_str(event.side).strip().upper() or "BUY"
    order_type = _safe_str(event.order_type).strip().upper() or "MKT"

    qty = _safe_float(event.quantity, 0.0)
    filled_qty = _safe_float(event.filled_quantity, 0.0)
    fill_price = _safe_float(event.fill_price, 0.0)

    used_qty = filled_qty if filled_qty > 0 else qty
    notional_used = fill_price * used_qty if (fill_price > 0 and used_qty > 0) else _safe_float(event.notional_estimate, 0.0)

    realized = None if event.realized_pnl is None else _safe_float(event.realized_pnl, 0.0)

    tags = ["ibkr_paper", strategy, "filled", side.lower(), order_type.lower()]

    extra: Dict[str, Any] = {
        "currency": _safe_str(event.currency).strip().upper() or "USD",
        "exchange": _safe_str(event.exchange).strip().upper() or "SMART",
        "sec_type": _safe_str(event.sec_type).strip().upper() or "STK",
        "order_type": order_type,
        "source": _safe_str(event.source).strip() or "paper_shadow_runner",
        "fill_price_used": fill_price,
        "filled_quantity_used": used_qty,
        "notional_used": notional_used,
    }

    if event.order_id is not None:
        extra["order_id"] = _safe_int(event.order_id, 0)
    if event.perm_id is not None:
        extra["perm_id"] = _safe_int(event.perm_id, 0)

    if isinstance(event.raw_intent, dict):
        extra["raw_intent"] = event.raw_intent
        if bool(event.raw_intent.get("pnl_untrusted")):
            extra["pnl_untrusted"] = True
            reason = event.raw_intent.get("pnl_untrusted_reason")
            if reason:
                extra["pnl_untrusted_reason"] = _safe_str(reason)

    payload: Dict[str, Any] = {
        "account_id": None,
        "broker": "ibkr",
        "symbol": symbol,
        "strategy": strategy,
        "side": side,
        "quantity": float(used_qty),
        "fill_price": float(fill_price),
        "notional": float(notional_used),
        "pnl": float(realized) if realized is not None else 0.0,
        "is_live": False,
        "entry_time_utc": _utc_now_iso(),
        "exit_time_utc": _utc_now_iso(),
        "regime": None,
        "tags": tags,
        "extra": extra,
    }
    return payload


def log_ibkr_paper_order_event(event: IBKRPaperOrderEvent) -> Path:
    """
    Append one record to today's paper ledger.
    Returns the ledger path written to.
    """
    path = _ledger_path_today()
    payload = _build_payload(event)

    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o640)
    try:
        with os.fdopen(fd, "r+", encoding="utf-8", newline="\n") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            try:
                last_seq, last_hash = _read_last_record_metadata(fp)
                next_seq = int(last_seq) + 1
                ts = _utc_now_iso()

                core = {
                    "timestamp_utc": ts,
                    "sequence_id": next_seq,
                    "prev_hash": last_hash,
                    "payload": payload,
                }
                record_hash = _sha256_hex(last_hash + _canonical_json(core))
                record = dict(core)
                record["record_hash"] = record_hash

                fp.seek(0, os.SEEK_END)
                fp.write(_canonical_json(record) + "\n")
                fp.flush()
                os.fsync(fp.fileno())
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    finally:
        # closed by context manager
        pass

    return path
