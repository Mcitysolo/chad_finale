#!/usr/bin/env python3
"""
CHAD — Kraken PnL Watcher (Production, Correct Ledger Discipline)

Fix vs prior version
--------------------
The previous watcher incorrectly wrote enrichment records into the "today" daily
trade_history_YYYYMMDD.ndjson file by calling log_trade_result().

That breaks audit semantics because enrichments for 2025-12-20 ended up written
to 2025-12-21.

This corrected watcher writes enrichments into a dedicated, tamper-evident file:

  data/trades/trade_history_enriched.ndjson

This file has its own hash chain and is explicitly "derived/enriched" data.

What it does
------------
- Scans recent trade_history_YYYYMMDD.ndjson files for Kraken trades
- For each Kraken trade lacking extra.kraken_enriched == True:
    - calls Kraken QueryOrders(txid)
    - appends an enrichment record to trade_history_enriched.ndjson
    - updates runtime FIFO inventory to compute realized PnL on sells

Safety
------
- No orders are placed.
- No existing NDJSON lines are mutated.
- Enrichments are appended-only.

"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig, KrakenAPIError


ROOT = Path(__file__).resolve().parents[2]
DATA_TRADES = ROOT / "data" / "trades"
RUNTIME_DIR = ROOT / "runtime"
STATE_PATH = RUNTIME_DIR / "kraken_pnl_state.json"
ENRICH_LEDGER = DATA_TRADES / "trade_history_enriched.ndjson"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if not math.isfinite(f):
            return default
        return f
    except Exception:
        return default


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.is_file():
        return {
            "generated_at_utc": _utc_now_iso(),
            "inventory": {},  # symbol -> list of lots [{qty, price}]
            "enriched_txids": {},  # txid -> enriched_record_hash
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def _iter_day_trade_files(days_back: int) -> List[Path]:
    out: List[Path] = []
    now = _utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(max(0, int(days_back)) + 1):
        ymd = (now - timedelta(days=i)).strftime("%Y%m%d")
        p = DATA_TRADES / f"trade_history_{ymd}.ndjson"
        if p.is_file():
            out.append(p)
    if not out:
        out = sorted(DATA_TRADES.glob("trade_history_*.ndjson"))
    return out


@dataclass(frozen=True)
class KrakenOrderFacts:
    txid: str
    status: str
    pair: str
    side: str
    ordertype: str
    vol: float
    vol_exec: float
    cost: float
    fee: float
    price: float
    opentm: float
    closetm: float


def _fetch_order_facts(client: KrakenClient, txid: str) -> KrakenOrderFacts:
    res = client.query_orders(txid=txid)
    info = res.get(txid)
    if not isinstance(info, dict):
        raise KrakenAPIError(f"QueryOrders missing txid={txid}")

    descr = info.get("descr") or {}
    return KrakenOrderFacts(
        txid=txid,
        status=str(info.get("status") or ""),
        pair=str(descr.get("pair") or ""),
        side=str(descr.get("type") or ""),  # buy/sell
        ordertype=str(descr.get("ordertype") or ""),
        vol=_safe_float(info.get("vol"), 0.0),
        vol_exec=_safe_float(info.get("vol_exec"), 0.0),
        cost=_safe_float(info.get("cost"), 0.0),
        fee=_safe_float(info.get("fee"), 0.0),
        price=_safe_float(info.get("price"), 0.0),
        opentm=_safe_float(info.get("opentm"), 0.0),
        closetm=_safe_float(info.get("closetm"), 0.0),
    )


def _normalize_symbol(payload_symbol: str) -> str:
    s = (payload_symbol or "").strip().upper()
    return s or "UNKNOWN"


def _fifo_apply_sell(inventory: List[Dict[str, Any]], sell_qty: float, sell_price: float) -> Tuple[float, List[Dict[str, Any]]]:
    remaining = float(sell_qty)
    pnl = 0.0
    new_inv: List[Dict[str, Any]] = []
    inv = [dict(x) for x in inventory]

    for lot in inv:
        if remaining <= 1e-12:
            new_inv.append(lot)
            continue
        lot_qty = _safe_float(lot.get("qty"), 0.0)
        lot_price = _safe_float(lot.get("price"), 0.0)
        if lot_qty <= 1e-12:
            continue

        used = min(lot_qty, remaining)
        pnl += used * (sell_price - lot_price)
        lot_qty_after = lot_qty - used
        remaining -= used

        if lot_qty_after > 1e-12:
            lot["qty"] = lot_qty_after
            new_inv.append(lot)

    if remaining > 1e-12:
        pnl += remaining * sell_price

    return pnl, new_inv


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _append_enrichment_record(
    *,
    original: Dict[str, Any],
    facts: KrakenOrderFacts,
    realized_pnl: float,
) -> str:
    """
    Append an enrichment record into ENRICH_LEDGER with its own hash chain.

    Returns:
      record_hash
    """
    ENRICH_LEDGER.parent.mkdir(parents=True, exist_ok=True)

    prev_hash = "GENESIS"
    seq = 0
    if ENRICH_LEDGER.is_file():
        lines = ENRICH_LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines()
        if lines:
            last = json.loads(lines[-1])
            prev_hash = str(last.get("record_hash") or "GENESIS")
            seq = int(last.get("sequence_id") or 0)

    seq_next = seq + 1

    payload = original.get("payload") or {}
    extra0 = payload.get("extra") or {}

    record = {
        "sequence_id": seq_next,
        "timestamp_utc": _utc_now_iso(),
        "prev_hash": prev_hash,
        "payload": {
            "broker": "kraken",
            "strategy": str(payload.get("strategy") or "crypto"),
            "symbol": _normalize_symbol(str(payload.get("symbol") or "XXBT")),
            "side": str(facts.side or payload.get("side") or "BUY").upper(),
            "quantity": float(facts.vol_exec if facts.vol_exec > 0 else _safe_float(payload.get("quantity"), 0.0)),
            "fill_price": float(facts.price if facts.price > 0 else _safe_float(payload.get("fill_price"), 0.0)),
            "notional": float(facts.cost if facts.cost > 0 else _safe_float(payload.get("notional"), 0.0)),
            "pnl": float(realized_pnl),
            "is_live": True,
            "entry_time_utc": str(payload.get("entry_time_utc") or _utc_now_iso()),
            "exit_time_utc": str(payload.get("exit_time_utc") or _utc_now_iso()),
            "regime": None,
            "tags": list(payload.get("tags") or []) + ["kraken_enriched"],
            "extra": {
                "source": "kraken_pnl_watcher",
                "kraken_enriched": True,
                "txid": facts.txid,
                "status": facts.status,
                "pair": facts.pair,
                "ordertype": facts.ordertype,
                "cost": facts.cost,
                "fee": facts.fee,
                "price": facts.price,
                "vol_exec": facts.vol_exec,
                "original_sequence_id": original.get("sequence_id"),
                "original_record_hash": original.get("record_hash"),
                "original_extra": dict(extra0),
                "enriched_at_utc": _utc_now_iso(),
            },
        },
    }

    # Hash over (prev_hash + canonical json payload + seq) to make it tamper-evident.
    canonical = json.dumps(record["payload"], sort_keys=True, separators=(",", ":"))
    record_hash = _sha256_hex(prev_hash + "|" + str(seq_next) + "|" + canonical)
    record["record_hash"] = record_hash

    with ENRICH_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    return record_hash


def run_once(*, days_back: int, max_records: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "enrich_ledger": str(ENRICH_LEDGER),
        "state_path": str(STATE_PATH),
        "writes": {"enriched_records": 0, "details": []},
    }

    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    state = _load_state()
    inventory_by_symbol: Dict[str, List[Dict[str, Any]]] = state.get("inventory") or {}
    enriched_txids: Dict[str, str] = state.get("enriched_txids") or {}

    files = _iter_day_trade_files(days_back)
    scanned = 0

    for f in files:
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        # scan tail first for recency
        for line in lines[-max_records:]:
            scanned += 1
            if scanned > max_records:
                break

            try:
                rec = json.loads(line)
            except Exception:
                continue

            payload = rec.get("payload") or {}
            if payload.get("broker") != "kraken":
                continue

            extra = payload.get("extra") or {}
            txid = extra.get("txid")
            if not txid:
                continue

            if str(txid) in enriched_txids:
                continue

            # Only enrich if original fill_price is 0 or missing (our old logger behavior)
            if _safe_float(payload.get("fill_price"), 0.0) > 0:
                # already has a fill price; consider it enriched enough
                enriched_txids[str(txid)] = "already_has_fill_price"
                continue

            facts = _fetch_order_facts(client, str(txid))

            sym = _normalize_symbol(str(payload.get("symbol") or "XXBT"))
            inv = inventory_by_symbol.get(sym) or []

            realized = 0.0
            if str(facts.side).lower() == "buy":
                inv.append({"qty": float(facts.vol_exec), "price": float(facts.price)})
                inventory_by_symbol[sym] = inv
            elif str(facts.side).lower() == "sell":
                realized, inv2 = _fifo_apply_sell(inv, float(facts.vol_exec), float(facts.price))
                realized -= float(facts.fee)
                inventory_by_symbol[sym] = inv2

            record_hash = _append_enrichment_record(original=rec, facts=facts, realized_pnl=realized)
            enriched_txids[str(txid)] = record_hash

            report["writes"]["enriched_records"] += 1
            report["writes"]["details"].append(
                {"txid": txid, "symbol": sym, "side": facts.side, "record_hash": record_hash, "pnl": realized}
            )

    state["generated_at_utc"] = _utc_now_iso()
    state["inventory"] = inventory_by_symbol
    state["enriched_txids"] = enriched_txids
    _save_state(state)

    report["scanned"] = scanned
    report["files"] = [str(x) for x in files]
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CHAD Kraken PnL Watcher (enrichment ledger).")
    p.add_argument("--days-back", type=int, default=7)
    p.add_argument("--max-records", type=int, default=500)
    args = p.parse_args(argv)
    rep = run_once(days_back=int(args.days_back), max_records=int(args.max_records))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
