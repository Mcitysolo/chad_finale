"""
Symbol Performance Blocker.

Blocks new entries on symbols that have lost money on the last three
consecutive closed trades. The block expires after two hours to give
the market time to change character.

Design principles:
- Fail-open: any I/O or parse error yields no block (never false-positive
  on a missing or malformed state file).
- Only blocks NEW entries. Exit/close intents must always be allowed
  through by callers.
- State is a pure read of closed-trade history; this module performs no
  broker calls.

State file schema (runtime/symbol_block_state.json):
    {
        "ts_utc": "...",
        "blocks": {
            "<SYMBOL>": {
                "blocked_until_utc": "...",
                "reason": "3_consecutive_losses",
                "last_3_pnl": [float, float, float]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path("/home/ubuntu/chad_finale")
TRADES_DIR = REPO_ROOT / "data" / "trades"
STATE_PATH = REPO_ROOT / "runtime" / "symbol_block_state.json"

CONSECUTIVE_LOSS_THRESHOLD = 3
LOOKBACK_TRADES_PER_SYMBOL = 5
BLOCK_DURATION_HOURS = 2
HISTORY_LOOKBACK_DAYS = 7


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_z(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        s = value.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _recent_trade_files() -> List[Path]:
    """Return the most recent N days of closed-trade NDJSON files."""
    if not TRADES_DIR.exists():
        return []
    today = _utc_now().date()
    files: List[Path] = []
    for offset in range(HISTORY_LOOKBACK_DAYS):
        dt = today - timedelta(days=offset)
        pattern = f"trade_history_{dt.strftime('%Y%m%d')}.ndjson"
        p = TRADES_DIR / pattern
        if p.exists():
            files.append(p)
    return files


def _load_closed_trades() -> List[Dict[str, Any]]:
    """Load closed trades from recent NDJSON history files. Fail-open."""
    trades: List[Dict[str, Any]] = []
    for path in _recent_trade_files():
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    payload = rec.get("payload") if isinstance(rec, dict) else None
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("schema_version") != "closed_trade.v1":
                        continue
                    sym = str(payload.get("symbol") or "").strip().upper()
                    if not sym:
                        continue
                    pnl_raw = payload.get("pnl")
                    try:
                        pnl = float(pnl_raw)
                    except Exception:
                        continue
                    exit_ts = _parse_iso(payload.get("exit_time_utc")) or _parse_iso(
                        rec.get("timestamp_utc")
                    )
                    if exit_ts is None:
                        continue
                    trades.append(
                        {"symbol": sym, "pnl": pnl, "exit_time_utc": exit_ts}
                    )
        except Exception as exc:
            logger.warning("symbol_blocker: failed to read %s: %s", path, exc)
            continue
    return trades


def _group_last_n_per_symbol(
    trades: List[Dict[str, Any]], n: int
) -> Dict[str, List[Dict[str, Any]]]:
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda r: r["exit_time_utc"], reverse=True)
        by_sym[sym] = by_sym[sym][:n]
    return by_sym


@dataclass(frozen=True)
class BlockRecord:
    symbol: str
    blocked_until_utc: datetime
    reason: str
    last_3_pnl: Tuple[float, float, float]


def _compute_blocks(
    by_sym: Dict[str, List[Dict[str, Any]]], now: datetime
) -> Dict[str, BlockRecord]:
    blocks: Dict[str, BlockRecord] = {}
    expiry = now + timedelta(hours=BLOCK_DURATION_HOURS)
    for sym, recs in by_sym.items():
        if len(recs) < CONSECUTIVE_LOSS_THRESHOLD:
            continue
        last_k = recs[:CONSECUTIVE_LOSS_THRESHOLD]
        if all(r["pnl"] < 0.0 for r in last_k):
            blocks[sym] = BlockRecord(
                symbol=sym,
                blocked_until_utc=expiry,
                reason="3_consecutive_losses",
                last_3_pnl=tuple(float(r["pnl"]) for r in last_k),  # type: ignore[arg-type]
            )
    return blocks


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _merge_with_existing(
    new_blocks: Dict[str, BlockRecord], now: datetime
) -> Dict[str, BlockRecord]:
    """
    Merge newly-computed blocks with any still-valid blocks from the
    existing state file. A block that was created earlier stays in force
    until its `blocked_until_utc` passes, even if the trade history no
    longer shows three consecutive losses.
    """
    merged: Dict[str, BlockRecord] = dict(new_blocks)
    if not STATE_PATH.exists():
        return merged
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        existing = raw.get("blocks") or {}
        for sym, rec in existing.items():
            if not isinstance(rec, dict):
                continue
            until = _parse_iso(rec.get("blocked_until_utc"))
            if until is None or until <= now:
                continue
            if sym in merged:
                continue
            pnl_list = rec.get("last_3_pnl") or []
            try:
                pnls = tuple(float(x) for x in pnl_list[:3])
                if len(pnls) != 3:
                    pnls = (0.0, 0.0, 0.0)
            except Exception:
                pnls = (0.0, 0.0, 0.0)
            merged[sym] = BlockRecord(
                symbol=sym,
                blocked_until_utc=until,
                reason=str(rec.get("reason") or "3_consecutive_losses"),
                last_3_pnl=pnls,  # type: ignore[arg-type]
            )
    except Exception as exc:
        logger.warning("symbol_blocker: failed to read existing state: %s", exc)
    return merged


def evaluate_symbol_blocks() -> Dict[str, bool]:
    """
    Scan recent closed trades and write the block state file.

    Returns a dict {symbol: is_blocked} for every symbol that had enough
    trades to be evaluated. Fail-open: on any exception the function
    returns an empty dict and leaves existing state untouched.
    """
    try:
        now = _utc_now()
        trades = _load_closed_trades()
        by_sym = _group_last_n_per_symbol(trades, LOOKBACK_TRADES_PER_SYMBOL)
        new_blocks = _compute_blocks(by_sym, now)
        merged = _merge_with_existing(new_blocks, now)

        state = {
            "ts_utc": _iso_z(now),
            "blocks": {
                sym: {
                    "blocked_until_utc": _iso_z(b.blocked_until_utc),
                    "reason": b.reason,
                    "last_3_pnl": list(b.last_3_pnl),
                }
                for sym, b in merged.items()
            },
        }
        _atomic_write_json(STATE_PATH, state)

        result: Dict[str, bool] = {}
        for sym, recs in by_sym.items():
            if len(recs) >= CONSECUTIVE_LOSS_THRESHOLD:
                result[sym] = sym in merged
        return result
    except Exception as exc:
        logger.warning("symbol_blocker: evaluate failed: %s", exc)
        return {}


def is_symbol_blocked(symbol: str) -> bool:
    """
    Return True iff the given symbol has an unexpired block record.
    Fail-open: missing / unreadable state returns False.
    """
    try:
        if not STATE_PATH.exists():
            return False
        sym = str(symbol or "").strip().upper()
        if not sym:
            return False
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        rec = (raw.get("blocks") or {}).get(sym)
        if not isinstance(rec, dict):
            return False
        until = _parse_iso(rec.get("blocked_until_utc"))
        if until is None:
            return False
        return until > _utc_now()
    except Exception:
        return False


def _cli() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = evaluate_symbol_blocks()
    blocked = sorted(sym for sym, is_b in result.items() if is_b)
    summary = {
        "blocked": blocked,
        "evaluated": len(result),
        "ts_utc": _iso_z(_utc_now()),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
