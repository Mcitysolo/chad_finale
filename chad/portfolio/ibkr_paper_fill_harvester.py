#!/usr/bin/env python3
"""
chad/portfolio/ibkr_paper_fill_harvester.py

Connects to IBKR paper gateway, reads confirmed fills via ib_insync,
and writes them to data/fills/FILLS_YYYYMMDD.ndjson using the same
hash-chained record format as paper_exec_evidence_writer.

This is additive — it does not replace the existing paper trade executor.

Usage:
    python -m chad.portfolio.ibkr_paper_fill_harvester
    python -m chad.portfolio.ibkr_paper_fill_harvester --dry-run

Python 3.10+
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("chad.ibkr_paper_fill_harvester")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FILLS_DIR = DATA_DIR / "fills"
TRADES_DIR = DATA_DIR / "trades"
RUNTIME_DIR = ROOT / "runtime"
LOCKS_DIR = RUNTIME_DIR / "locks"
POSITION_GUARD_PATH = RUNTIME_DIR / "position_guard.json"
HARVESTED_IDS_PATH = RUNTIME_DIR / "harvested_fill_ids.json"

FILL_SCHEMA_VERSION = "paper_exec_fill.v4"
TRADE_SCHEMA_VERSION = "closed_trade.v1"
GENESIS = "GENESIS"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 79


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
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


def _atomic_write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    raw = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=str(path.parent), prefix=path.name + ".tmp."
    ) as tmp:
        tmp.write(raw)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


# ---------------------------------------------------------------------------
# Dedup tracker
# ---------------------------------------------------------------------------

def _load_harvested_data() -> Dict[str, Any]:
    if not HARVESTED_IDS_PATH.exists():
        return {"exec_ids": [], "trade_ids": []}
    try:
        return json.loads(HARVESTED_IDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load harvested_fill_ids.json — starting fresh")
        return {"exec_ids": [], "trade_ids": []}


def load_harvested_ids() -> Set[str]:
    return set(_load_harvested_data().get("exec_ids", []))


def load_harvested_trade_ids() -> Set[str]:
    return set(_load_harvested_data().get("trade_ids", []))


def save_harvested_ids(
    fill_ids: Set[str],
    trade_ids: Optional[Set[str]] = None,
) -> None:
    existing = _load_harvested_data()
    merged_fills = sorted(fill_ids)
    merged_trades = sorted(trade_ids) if trade_ids is not None else existing.get("trade_ids", [])
    _atomic_write_json(HARVESTED_IDS_PATH, {
        "exec_ids": merged_fills,
        "count": len(merged_fills),
        "trade_ids": merged_trades,
        "trade_count": len(merged_trades),
        "updated_utc": _utc_now_iso(),
    })


# ---------------------------------------------------------------------------
# Position guard (strategy attribution)
# ---------------------------------------------------------------------------

def load_position_guard() -> Dict[str, Dict[str, Any]]:
    if not POSITION_GUARD_PATH.exists():
        logger.warning("position_guard.json not found")
        return {}
    try:
        return json.loads(POSITION_GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse position_guard.json")
        return {}


def resolve_strategy(symbol: str, guard: Dict[str, Dict[str, Any]]) -> str:
    """Look up strategy for a symbol from position_guard.json.

    Prefer entries with open=True. Keys are like 'alpha|AAPL'.
    """
    best: Optional[str] = None
    for key, entry in guard.items():
        if not isinstance(entry, dict):
            continue
        entry_symbol = entry.get("symbol", "")
        if entry_symbol.upper() != symbol.upper():
            continue
        strategy = entry.get("strategy", "")
        if not strategy:
            continue
        if entry.get("open", False):
            return strategy
        if best is None:
            best = strategy
    return best or "unknown"


# ---------------------------------------------------------------------------
# IBKR side mapping
# ---------------------------------------------------------------------------

def normalize_side(ibkr_side: str) -> str:
    s = ibkr_side.upper().strip()
    if s in ("BOT", "BUY"):
        return "BUY"
    if s in ("SLD", "SELL"):
        return "SELL"
    return s


# ---------------------------------------------------------------------------
# Fill ID generation (matches paper_exec_evidence_writer pattern)
# ---------------------------------------------------------------------------

def make_fill_id(
    account_id: str,
    symbol: str,
    side: str,
    quantity: float,
    fill_price: float,
    fill_time_utc: str,
    strategy: str,
) -> str:
    raw = "|".join([
        account_id,
        symbol.upper(),
        side.upper(),
        f"{quantity:.12f}",
        f"{fill_price:.12f}",
        fill_time_utc,
        strategy,
    ])
    return _sha256(raw)


# ---------------------------------------------------------------------------
# Hash-chained record writer
# ---------------------------------------------------------------------------

def append_hash_chained_record(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_dir(path.parent)
    _ensure_dir(LOCKS_DIR)

    lock_path = LOCKS_DIR / f"{path.name}.lock"

    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

        last_line = _last_nonempty_line(path)
        if last_line:
            try:
                last_obj = json.loads(last_line)
                prev_hash = last_obj.get("record_hash", GENESIS) or GENESIS
                prev_seq = int(last_obj.get("sequence_id", 0))
            except Exception:
                prev_hash = GENESIS
                prev_seq = 0
        else:
            prev_hash = GENESIS
            prev_seq = 0

        record: Dict[str, Any] = {
            "payload": payload,
            "prev_hash": prev_hash,
            "sequence_id": prev_seq + 1,
            "timestamp_utc": _utc_now_iso(),
        }
        record["record_hash"] = _sha256(_canonical_json(record))

        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            os.fsync(out.fileno())

        return {
            "path": str(path),
            "sequence_id": record["sequence_id"],
            "record_hash": record["record_hash"],
        }


# ---------------------------------------------------------------------------
# IBKR connection + harvest
# ---------------------------------------------------------------------------

def connect_ibkr():
    from ib_insync import IB, util
    util.patchAsyncio()
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
    return ib


def harvest(dry_run: bool = False) -> Dict[str, Any]:
    ib = connect_ibkr()

    try:
        fills = ib.fills()
        positions = ib.positions()
    finally:
        ib.disconnect()

    logger.info("IBKR returned %d fills, %d positions", len(fills), len(positions))

    guard = load_position_guard()
    already_harvested = load_harvested_ids()
    already_traded = load_harvested_trade_ids()

    day = _utc_ymd()
    fills_path = FILLS_DIR / f"FILLS_{day}.ndjson"
    now_iso = _utc_now_iso()

    fills_wrote = 0
    fills_skipped = 0
    trades_wrote = 0
    trades_skipped = 0
    new_fill_ids: Set[str] = set()
    new_trade_ids: Set[str] = set()

    # Collect trade history records grouped by date
    trade_records_by_date: Dict[str, List[Dict[str, Any]]] = {}

    for fill in fills:
        contract = fill.contract
        execution = fill.execution

        exec_id = execution.execId
        fill_is_new = exec_id not in already_harvested
        trade_is_new = exec_id not in already_traded

        if not fill_is_new and not trade_is_new:
            fills_skipped += 1
            trades_skipped += 1
            continue

        symbol = contract.symbol
        sec_type = contract.secType
        side = normalize_side(execution.side)
        quantity = float(execution.shares)
        fill_price = float(execution.price)
        notional = abs(fill_price * quantity)
        acct = execution.acctNumber

        # For futures, notional uses multiplier
        if sec_type == "FUT" and hasattr(contract, "multiplier") and contract.multiplier:
            try:
                notional = abs(fill_price * quantity * float(contract.multiplier))
            except (ValueError, TypeError):
                pass

        fill_time = execution.time
        if hasattr(fill_time, "isoformat"):
            fill_time_utc = fill_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            fill_time_utc = str(fill_time)

        strategy = resolve_strategy(symbol, guard)

        asset_class = "unknown"
        if sec_type == "STK":
            if symbol in {"SPY", "QQQ", "IWM", "DIA", "GLD", "SH", "SVXY", "IEMG", "VWO"}:
                asset_class = "etf"
            else:
                asset_class = "equity"
        elif sec_type == "FUT":
            asset_class = "futures"
        elif sec_type == "OPT":
            asset_class = "options"

        fill_id = make_fill_id(acct, symbol, side, quantity, fill_price, fill_time_utc, strategy)

        source_strategies = [strategy] if strategy != "unknown" else []

        tags = ["paper", "filled", "ibkr_harvest"]
        if strategy != "unknown":
            tags.append(strategy)
        if asset_class != "unknown":
            tags.append(asset_class)

        payload: Dict[str, Any] = {
            "schema_version": FILL_SCHEMA_VERSION,
            "account_id": acct,
            "asset_class": asset_class,
            "broker": "ibkr_paper",
            "entry_time_utc": fill_time_utc,
            "exit_time_utc": fill_time_utc,
            "extra": {
                "source_strategies": source_strategies,
                "slippage_bps": 0.0,
                "latency_ms": 0.0,
                "ibkr_exec_id": exec_id,
                "sec_type": sec_type,
                "source": "ibkr_paper_fill_harvester",
            },
            "fill_id": fill_id,
            "fill_price": fill_price,
            "fill_time_utc": fill_time_utc,
            "is_live": False,
            "notional": round(notional, 4),
            "order_type": "LMT",
            "partial_fill": False,
            "plan_now_iso": "",
            "plan_path": "",
            "quantity": quantity,
            "reject": False,
            "side": side,
            "source": "ibkr_paper_fill_harvester",
            "status": "paper_fill",
            "strategy": strategy,
            "source_strategies": source_strategies,
            "symbol": symbol.upper(),
            "tags": tags,
            "venue": "ibkr_paper",
        }

        # --- Write fill record (existing behavior) ---
        if not fill_is_new:
            fills_skipped += 1
        elif dry_run:
            logger.info("[DRY-RUN] Would write fill: %s %s %.0f @ %.2f strategy=%s",
                        symbol, side, quantity, fill_price, strategy)
            fills_wrote += 1
            new_fill_ids.add(exec_id)
        else:
            append_hash_chained_record(fills_path, payload)
            logger.info("Wrote fill: %s %s %.0f @ %.2f strategy=%s exec_id=%s",
                        symbol, side, quantity, fill_price, strategy, exec_id)
            fills_wrote += 1
            new_fill_ids.add(exec_id)

        # --- Build trade history record ---
        if not trade_is_new:
            trades_skipped += 1
        else:
            # Determine the fill date for file naming
            if hasattr(fill_time, "strftime"):
                fill_day = fill_time.astimezone(timezone.utc).strftime("%Y%m%d")
            else:
                fill_day = day

            # Contract multiplier
            contract_mult = 1.0
            if sec_type == "FUT" and hasattr(contract, "multiplier") and contract.multiplier:
                try:
                    contract_mult = float(contract.multiplier)
                except (ValueError, TypeError):
                    pass

            trade_payload: Dict[str, Any] = {
                "schema_version": TRADE_SCHEMA_VERSION,
                "strategy": strategy,
                "symbol": symbol.upper(),
                "side": side,
                "pnl": 0.0,
                "entry_time_utc": fill_time_utc,
                "exit_time_utc": fill_time_utc,
                "fill_price": fill_price,
                "entry_price": fill_price,
                "exit_price": fill_price,
                "quantity": quantity,
                "contract_multiplier": contract_mult,
                "notional": round(notional, 4),
                "fill_ids": [fill_id],
                "broker": "ibkr_paper",
                "account_id": acct,
                "is_live": False,
                "tags": ["paper", "filled", "ibkr_harvest"] + ([strategy] if strategy != "unknown" else []),
            }

            if not dry_run:
                trade_records_by_date.setdefault(fill_day, []).append(trade_payload)
                logger.info("Queued trade history: %s %s %.0f @ %.2f strategy=%s exec_id=%s",
                            symbol, side, quantity, fill_price, strategy, exec_id)
            else:
                logger.info("[DRY-RUN] Would write trade history: %s %s %.0f @ %.2f strategy=%s",
                            symbol, side, quantity, fill_price, strategy)
            trades_wrote += 1
            new_trade_ids.add(exec_id)

    # --- Write trade history records (atomic per date file) ---
    if not dry_run:
        for trade_day, records in trade_records_by_date.items():
            trade_path = TRADES_DIR / f"trade_history_{trade_day}.ndjson"
            for rec in records:
                append_hash_chained_record(trade_path, rec)
            logger.info("Wrote %d trade history records to %s", len(records), trade_path.name)

    # --- Update dedup tracker ---
    if not dry_run and (new_fill_ids or new_trade_ids):
        save_harvested_ids(
            fill_ids=already_harvested | new_fill_ids,
            trade_ids=already_traded | new_trade_ids,
        )

    return {
        "ok": True,
        "fills_wrote": fills_wrote,
        "fills_skipped": fills_skipped,
        "trades_wrote": trades_wrote,
        "trades_skipped": trades_skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="CHAD IBKR Paper Fill Harvester")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        result = harvest(dry_run=args.dry_run)
        print(json.dumps(result))
        return 0
    except Exception:
        logger.exception("Harvester failed")
        print(json.dumps({"ok": False, "wrote": 0, "error": "see logs"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
