#!/usr/bin/env python3
"""
CHAD — Paper Shadow Runner (Phase 9 SCR evidence)

Problem this solves
-------------------
Your ledger proved you were stacking paper positions (AAPL +93, MSFT +69).
That creates tons of entry logs but too few realized outcomes, keeping SCR PAUSED.

This runner is designed to:
1) CLOSE existing open inventory first (FIFO lots from today's ledger),
2) Enforce FLAT-ONLY per symbol for new entries,
3) Produce realized outcomes fast (exit logs with realized_pnl),
4) Keep safety gates and paper-only discipline unchanged.

Safety (paper-only)
-------------------
To execute paper orders, ALL must be true:
- cfg.enabled == True
- cfg.armed == True
- env CHAD_PAPER_SHADOW_ARMED equals ARM_PHRASE
- LiveGate allows IBKR paper (allow_ibkr_paper == True)
Else -> preview only (no broker I/O)

No live trading is enabled by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from ib_insync import IB, MarketOrder, Stock  # type: ignore

from chad.portfolio.ibkr_paper_trade_result_logger import (
    IBKRPaperOrderEvent,
    log_ibkr_paper_order_event,
)

# ----------------------------
# Constants / Paths
# ----------------------------
ARM_ENV_VAR = "CHAD_PAPER_SHADOW_ARMED"
ARM_PHRASE = "I_UNDERSTAND_THIS_CAN_PLACE_PAPER_ORDERS"

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"
REPORTS_DIR = ROOT / "reports" / "shadow"
FULL_CYCLE_LAST = RUNTIME_DIR / "full_execution_cycle_last.json"
DEFAULT_CONFIG_PATH = RUNTIME_DIR / "paper_shadow.json"
LIVE_GATE_URL = "http://127.0.0.1:9618/live-gate"
DATA_TRADES_DIR = ROOT / "data" / "trades"

DEFAULT_HOLD_SECONDS = 90            # Phase 9: accelerate outcomes
MAX_HOLD_SECONDS = 1800              # hard ceiling
DEFAULT_CLOSE_FIRST = True           # close inventory before new entries
DEFAULT_FLAT_ONLY = True             # never enter a symbol already net-long
DEFAULT_MAX_CLOSE_QTY_PER_RUN = 3.0  # keep small, safe
DEFAULT_MAX_ORDERS_PER_RUN = 1


# ----------------------------
# Helpers
# ----------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _utc_ymd() -> str:
    return _utc_now().strftime("%Y%m%d")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _env_armed() -> bool:
    return (str(os.environ.get(ARM_ENV_VAR) or "").strip() == ARM_PHRASE)


def _fetch_live_gate(timeout_s: float = 2.0) -> Tuple[bool, List[str]]:
    try:
        with urllib.request.urlopen(LIVE_GATE_URL, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
            j = json.loads(raw)
            allow = bool(j.get("allow_ibkr_paper", False))
            reasons = list(j.get("reasons", []) or [])
            return allow, reasons
    except Exception as exc:
        return False, [f"live-gate fetch failed: {type(exc).__name__}: {exc}"]


def _load_full_cycle_intents() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not FULL_CYCLE_LAST.is_file():
        raise FileNotFoundError(f"Missing {FULL_CYCLE_LAST}. Run: python -m chad.core.full_execution_cycle")
    payload = json.loads(FULL_CYCLE_LAST.read_text(encoding="utf-8"))
    intents = payload.get("ibkr_intents") or []
    if not isinstance(intents, list):
        raise ValueError("runtime/full_execution_cycle_last.json: ibkr_intents is not a list")
    return payload, intents


@dataclass(frozen=True)
class IBKRConfig:
    host: str
    port: int
    client_id: int


@dataclass(frozen=True)
class PaperShadowConfig:
    enabled: bool
    armed: bool
    ibkr: IBKRConfig

    min_notional_usd: float
    max_orders_per_run: int
    auto_cancel_seconds: int
    size_multiplier: float
    strategy_allowlist: List[str]

    hold_seconds: int = DEFAULT_HOLD_SECONDS

    close_existing_first: bool = DEFAULT_CLOSE_FIRST
    flat_only_per_symbol: bool = DEFAULT_FLAT_ONLY
    max_close_qty_per_run: float = DEFAULT_MAX_CLOSE_QTY_PER_RUN


def load_config(path: Path) -> PaperShadowConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paper_shadow config JSON must be an object")

    ibkr_raw = raw.get("ibkr") or {}
    if not isinstance(ibkr_raw, dict):
        raise ValueError("paper_shadow.ibkr must be an object")

    host = _safe_str(ibkr_raw.get("host"), "127.0.0.1").strip() or "127.0.0.1"
    port = _safe_int(ibkr_raw.get("port"), 4002) or 4002
    client_id = _safe_int(ibkr_raw.get("client_id"), 9201) or 9201

    allow = raw.get("strategy_allowlist") or []
    if not isinstance(allow, list):
        raise ValueError("paper_shadow.strategy_allowlist must be a list")
    allowlist = [str(x).strip().upper() for x in allow if str(x).strip()]

    hold_seconds = _safe_int(raw.get("hold_seconds"), DEFAULT_HOLD_SECONDS) or DEFAULT_HOLD_SECONDS
    hold_seconds = max(1, min(int(hold_seconds), MAX_HOLD_SECONDS))

    max_orders_per_run = _safe_int(raw.get("max_orders_per_run"), DEFAULT_MAX_ORDERS_PER_RUN) or DEFAULT_MAX_ORDERS_PER_RUN
    max_orders_per_run = max(0, int(max_orders_per_run))

    close_existing_first = bool(raw.get("close_existing_first", DEFAULT_CLOSE_FIRST))
    flat_only = bool(raw.get("flat_only_per_symbol", DEFAULT_FLAT_ONLY))
    max_close_qty = float(_safe_float(raw.get("max_close_qty_per_run"), DEFAULT_MAX_CLOSE_QTY_PER_RUN))
    max_close_qty = max(0.0, max_close_qty)

    return PaperShadowConfig(
        enabled=bool(raw.get("enabled", False)),
        armed=bool(raw.get("armed", False)),
        ibkr=IBKRConfig(host=host, port=int(port), client_id=int(client_id)),
        min_notional_usd=float(_safe_float(raw.get("min_notional_usd"), 0.0)),
        max_orders_per_run=max_orders_per_run,
        auto_cancel_seconds=max(5, _safe_int(raw.get("auto_cancel_seconds"), 300)),
        size_multiplier=max(0.0, float(_safe_float(raw.get("size_multiplier"), 1.0))),
        strategy_allowlist=allowlist,
        hold_seconds=hold_seconds,
        close_existing_first=close_existing_first,
        flat_only_per_symbol=flat_only,
        max_close_qty_per_run=max_close_qty,
    )


def _filter_intents(
    intents_all: List[Dict[str, Any]],
    *,
    strategy_allowlist: List[str],
    min_notional_usd: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allow = {s.strip().upper() for s in strategy_allowlist if s.strip()}
    for it in intents_all:
        if not isinstance(it, dict):
            continue
        strat = str(it.get("strategy") or "").strip().upper()
        if allow and strat not in allow:
            continue
        notional = _safe_float(it.get("notional_estimate"), 0.0)
        if notional < float(min_notional_usd):
            continue
        out.append(it)
    return out


def _intent_to_preview(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strategy": _safe_str(it.get("strategy"), "").strip().lower(),
        "symbol": _safe_str(it.get("symbol"), "UNKNOWN").strip().upper(),
        "side": _safe_str(it.get("side"), "BUY").strip().upper(),
        "quantity": float(_safe_float(it.get("quantity"), 0.0)),
        "order_type": _safe_str(it.get("order_type"), "MKT").strip().upper(),
        "notional_estimate": float(_safe_float(it.get("notional_estimate"), 0.0)),
        "sec_type": _safe_str(it.get("sec_type"), "STK").strip().upper(),
        "exchange": _safe_str(it.get("exchange"), "SMART").strip().upper(),
        "currency": _safe_str(it.get("currency"), "USD").strip().upper(),
        "limit_price": it.get("limit_price"),
    }


def _today_ledger_path() -> Path:
    return DATA_TRADES_DIR / f"trade_history_{_utc_ymd()}.ndjson"


@dataclass
class _Lot:
    qty: float
    price: float


def _build_fifo_inventory_from_ledger() -> Dict[str, Deque[_Lot]]:
    """
    Build per-symbol FIFO lots from today's ledger.
    We treat BUY as adding lots, SELL as consuming lots (FIFO).
    This is deterministic and purely ledger-driven.
    """
    inv: Dict[str, Deque[_Lot]] = defaultdict(deque)
    p = _today_ledger_path()
    if not p.exists():
        return inv

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            payload = rec.get("payload", {}) or {}
            sym = str(payload.get("symbol") or "").strip().upper()
            if not sym:
                continue
            side = str(payload.get("side") or "").strip().upper()
            qty = float(payload.get("quantity") or 0.0)
            fp = float(payload.get("fill_price") or 0.0)
            if qty <= 0.0 or fp <= 0.0:
                continue

            if side == "BUY":
                inv[sym].append(_Lot(qty=qty, price=fp))
            elif side == "SELL":
                # consume FIFO lots
                remaining = qty
                q = inv[sym]
                while remaining > 0.0 and q:
                    lot = q[0]
                    take = min(lot.qty, remaining)
                    lot.qty -= take
                    remaining -= take
                    if lot.qty <= 1e-9:
                        q.popleft()
                # if sell exceeds inventory, we just exhaust (shouldn't happen in clean paper runs)
        except Exception:
            continue

    # prune empties
    return {sym: q for sym, q in inv.items() if q}


def _net_position_from_inventory(inv: Dict[str, Deque[_Lot]]) -> Dict[str, float]:
    return {sym: float(sum(lot.qty for lot in q)) for sym, q in inv.items()}


def _build_report_base(
    cfg: PaperShadowConfig,
    *,
    mode: str,
    reasons: List[str],
    full_cycle_payload: Dict[str, Any],
    intents_total: int,
    intents_filtered: int,
    planned_orders: int,
    inv_net: Dict[str, float],
) -> Dict[str, Any]:
    counts_obj = full_cycle_payload.get("counts")
    counts = counts_obj if isinstance(counts_obj, dict) else {}

    return {
        "generated_at_utc": _utc_now_iso(),
        "mode": mode,
        "reasons": list(reasons),
        "armed_env": _env_armed(),
        "config": {
            "enabled": bool(cfg.enabled),
            "armed": bool(cfg.armed),
            "auto_cancel_seconds": int(cfg.auto_cancel_seconds),
            "min_notional_usd": float(cfg.min_notional_usd),
            "max_orders_per_run": int(cfg.max_orders_per_run),
            "size_multiplier": float(cfg.size_multiplier),
            "hold_seconds": int(cfg.hold_seconds),
            "close_existing_first": bool(cfg.close_existing_first),
            "flat_only_per_symbol": bool(cfg.flat_only_per_symbol),
            "max_close_qty_per_run": float(cfg.max_close_qty_per_run),
            "reports_dir": str(REPORTS_DIR),
            "strategy_allowlist": list(cfg.strategy_allowlist),
            "ibkr": {"host": cfg.ibkr.host, "port": int(cfg.ibkr.port), "client_id": int(cfg.ibkr.client_id)},
        },
        "counts": {
            "raw_signals": int(_safe_int(counts.get("raw_signals"), 0)),
            "routed_signals": int(_safe_int(counts.get("routed_signals"), 0)),
            "evaluated_signals": int(_safe_int(counts.get("evaluated_signals"), 0)),
            "ibkr_intents_total": int(intents_total),
            "ibkr_intents_filtered": int(intents_filtered),
            "planned_orders": int(planned_orders),
        },
        "inventory_net": inv_net,
        "paper_intents_preview": [],
        "actions": {"orders_submitted": 0, "orders_cancelled": 0, "trade_results_logged": 0, "details": []},
        "post": {"openOrders_count": 0},
        "safety": {"no_orders_placed": True, "no_trade_ledgers_written": True},
    }


def _wait_for_fill_or_cancel(ib: IB, trade, timeout_s: float) -> Tuple[bool, float, str, float]:
    deadline = time.time() + max(1.0, float(timeout_s))
    last_status = ""
    while time.time() < deadline:
        try:
            status = str(getattr(trade.orderStatus, "status", "") or "")
            last_status = status
            filled_qty = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
            avg_fp = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
            if filled_qty > 0.0 and avg_fp > 0.0:
                return True, filled_qty, status, avg_fp
            if status.lower() in {"cancelled", "inactive"}:
                return False, filled_qty, status, avg_fp
        except Exception:
            pass
        ib.sleep(0.25)

    try:
        ib.cancelOrder(trade.order)
        ib.sleep(0.25)
        status = str(getattr(trade.orderStatus, "status", "") or last_status)
        filled_qty = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
        avg_fp = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
        if filled_qty > 0.0 and avg_fp > 0.0:
            return True, filled_qty, status, avg_fp
        return False, filled_qty, status, avg_fp
    except Exception:
        return False, 0.0, last_status or "unknown", 0.0


def _place_order_mkt(ib: IB, *, symbol: str, exchange: str, currency: str, side: str, qty: float):
    contract = Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)
    order = MarketOrder(side, qty)
    trade = ib.placeOrder(contract, order)
    return trade


def run_preview(
    cfg: PaperShadowConfig,
    full_cycle_payload: Dict[str, Any],
    intents_all: List[Dict[str, Any]],
    intents_filtered: List[Dict[str, Any]],
    reasons: List[str],
    inv_net: Dict[str, float],
) -> Dict[str, Any]:
    planned = intents_filtered[: max(0, cfg.max_orders_per_run)]
    report = _build_report_base(
        cfg,
        mode="preview",
        reasons=reasons,
        full_cycle_payload=full_cycle_payload,
        intents_total=len(intents_all),
        intents_filtered=len(intents_filtered),
        planned_orders=len(planned),
        inv_net=inv_net,
    )
    report["paper_intents_preview"] = [_intent_to_preview(it) for it in planned]
    report["safety"]["no_orders_placed"] = True
    report["safety"]["no_trade_ledgers_written"] = True
    return report


def _choose_close_symbol(inv: Dict[str, Deque[_Lot]]) -> Optional[str]:
    # close the symbol with largest remaining qty to reduce inventory fastest
    best_sym = None
    best_qty = 0.0
    for sym, q in inv.items():
        qty = float(sum(lot.qty for lot in q))
        if qty > best_qty + 1e-9:
            best_qty = qty
            best_sym = sym
    return best_sym


def run_execute(
    cfg: PaperShadowConfig,
    full_cycle_payload: Dict[str, Any],
    intents_all: List[Dict[str, Any]],
    intents_filtered: List[Dict[str, Any]],
    reasons: List[str],
) -> Dict[str, Any]:
    inv = _build_fifo_inventory_from_ledger()
    inv_net = _net_position_from_inventory(inv)

    # Apply flat-only gating to planned intents.
    # IMPORTANT: select up to max_orders_per_run *eligible* intents by scanning the
    # full filtered list (do not slice first). This prevents "AAPL-first" from
    # causing zero plans when already holding AAPL.
    planned: List[Dict[str, Any]] = []
    want = max(0, int(cfg.max_orders_per_run))
    for it in intents_filtered:
        if len(planned) >= want:
            break
        sym = str(it.get("symbol") or "").strip().upper()
        if not sym:
            continue
        if cfg.flat_only_per_symbol and inv_net.get(sym, 0.0) > 1e-9:
            continue
        planned.append(it)

    report = _build_report_base(
        cfg,
        mode="execute",
        reasons=reasons,
        full_cycle_payload=full_cycle_payload,
        intents_total=len(intents_all),
        intents_filtered=len(intents_filtered),
        planned_orders=len(planned),
        inv_net=inv_net,
    )
    report["paper_intents_preview"] = [_intent_to_preview(it) for it in planned]
    report["safety"]["no_orders_placed"] = False

    ib = IB()
    try:
        ib.connect(cfg.ibkr.host, cfg.ibkr.port, clientId=cfg.ibkr.client_id, timeout=10)

        # 1) CLOSE EXISTING FIRST (Phase 9 evidence acceleration)
        if cfg.close_existing_first and inv:
            sym = _choose_close_symbol(inv)
            if sym:
                # Find exchange/currency defaults from the most recent matching intent if available
                exch = "SMART"
                ccy = "USD"
                sec_type = "STK"
                strat = "beta"

                for it in intents_all:
                    if str(it.get("symbol") or "").strip().upper() == sym:
                        exch = str(it.get("exchange") or "SMART").strip().upper()
                        ccy = str(it.get("currency") or "USD").strip().upper()
                        sec_type = str(it.get("sec_type") or "STK").strip().upper()
                        strat = str(it.get("strategy") or "beta").strip().lower() or "beta"
                        break

                # Close at most max_close_qty_per_run, using FIFO lots for realized PnL.
                to_close = float(min(cfg.max_close_qty_per_run, inv_net.get(sym, 0.0)))
                if to_close > 0.0:
                    # Execute SELL and compute PnL against FIFO lot(s) we consume.
                    trade = _place_order_mkt(ib, symbol=sym, exchange=exch, currency=ccy, side="SELL", qty=to_close)
                    report["actions"]["orders_submitted"] += 1
                    report["actions"]["details"].append({"event": "exit_order_submitted", "symbol": sym, "qty": to_close, "exchange": exch, "currency": ccy})

                    filled, filled_qty, status, exit_fp = _wait_for_fill_or_cancel(ib, trade, timeout_s=float(cfg.auto_cancel_seconds))
                    if not filled:
                        report["actions"]["orders_cancelled"] += 1
                        report["actions"]["details"].append({"event": "exit_not_filled_cancelled", "symbol": sym, "status": status, "filled_qty": float(filled_qty)})
                    else:
                        # Consume FIFO lots to compute realized PnL for the closed qty.
                        q = inv.get(sym, deque())
                        remaining = float(filled_qty)
                        realized_total = 0.0
                        consumed = []

                        while remaining > 1e-9 and q:
                            lot = q[0]
                            take = min(lot.qty, remaining)
                            realized_total += (float(exit_fp) - float(lot.price)) * float(take)
                            consumed.append({"qty": float(take), "entry_price": float(lot.price)})
                            lot.qty -= take
                            remaining -= take
                            if lot.qty <= 1e-9:
                                q.popleft()

                        # Log trusted exit result
                        try:
                            log_path = log_ibkr_paper_order_event(
                                IBKRPaperOrderEvent(
                                    strategy=strat,
                                    symbol=sym,
                                    side="SELL",
                                    quantity=float(filled_qty),
                                    filled_quantity=float(filled_qty),
                                    fill_price=float(exit_fp),
                                    notional_estimate=float(exit_fp) * float(filled_qty),
                                    sec_type=sec_type,
                                    exchange=exch,
                                    currency=ccy,
                                    order_type="MKT",
                                    order_id=_safe_int(getattr(trade.order, "orderId", None), 0) or None,
                                    perm_id=_safe_int(getattr(trade.order, "permId", None), 0) or None,
                                    raw_intent={"exit": True, "close_existing_first": True, "fifo_consumed": consumed},
                                    realized_pnl=float(realized_total),
                                    source="paper_shadow_runner",
                                )
                            )
                            report["actions"]["trade_results_logged"] += 1
                            report["actions"]["details"].append({"event": "trade_result_logged", "log_path": str(log_path), "realized_pnl": float(realized_total)})
                        except Exception as exc:
                            report["actions"]["details"].append({"event": "exit_log_failed", "error": f"{type(exc).__name__}: {exc}"})

        # 2) If we closed inventory this run, stop here (one action per run keeps it safe)
        if report["actions"]["orders_submitted"] > 0 and cfg.close_existing_first:
            report["safety"]["no_trade_ledgers_written"] = (int(report["actions"]["trade_results_logged"]) == 0)
            return report

        # 3) Otherwise, place new entry+exit cycles (flat-only per symbol)
        if not planned:
            report["safety"]["no_trade_ledgers_written"] = True
            return report

        for it in planned:
            sym = str(it.get("symbol") or "UNKNOWN").strip().upper()
            exch = str(it.get("exchange") or "SMART").strip().upper()
            ccy = str(it.get("currency") or "USD").strip().upper()
            sec_type = str(it.get("sec_type") or "STK").strip().upper()
            strat = str(it.get("strategy") or "beta").strip().lower() or "beta"

            side = str(it.get("side") or "BUY").strip().upper()
            if side != "BUY":
                # Phase 9: only long entries in this runner
                continue

            qty = float(_safe_float(it.get("quantity"), 0.0)) * float(cfg.size_multiplier)
            if qty <= 0.0:
                continue

            entry_trade = _place_order_mkt(ib, symbol=sym, exchange=exch, currency=ccy, side="BUY", qty=qty)
            report["actions"]["orders_submitted"] += 1
            report["actions"]["details"].append({"event": "order_submitted", "symbol": sym, "side": "BUY", "qty": qty})

            filled, filled_qty, status, entry_fp = _wait_for_fill_or_cancel(ib, entry_trade, timeout_s=float(cfg.auto_cancel_seconds))
            if not filled:
                report["actions"]["orders_cancelled"] += 1
                report["actions"]["details"].append({"event": "order_not_filled_cancelled", "symbol": sym, "status": status, "filled_qty": float(filled_qty)})
                continue

            # Log entry as pnl_untrusted (excluded from SCR)
            try:
                log_ibkr_paper_order_event(
                    IBKRPaperOrderEvent(
                        strategy=strat,
                        symbol=sym,
                        side="BUY",
                        quantity=float(qty),
                        filled_quantity=float(filled_qty),
                        fill_price=float(entry_fp),
                        notional_estimate=float(entry_fp) * float(filled_qty),
                        sec_type=sec_type,
                        exchange=exch,
                        currency=ccy,
                        order_type="MKT",
                        order_id=_safe_int(getattr(entry_trade.order, "orderId", None), 0) or None,
                        perm_id=_safe_int(getattr(entry_trade.order, "permId", None), 0) or None,
                        raw_intent=dict(it) | {"pnl_untrusted": True, "pnl_untrusted_reason": "entry_only_no_exit"},
                        realized_pnl=None,
                        source="paper_shadow_runner",
                    )
                )
            except Exception as exc:
                report["actions"]["details"].append({"event": "entry_log_failed", "error": f"{type(exc).__name__}: {exc}"})

            # Hold then exit for realized outcome
            hold_s = max(1, min(int(cfg.hold_seconds), MAX_HOLD_SECONDS))
            time.sleep(hold_s)

            exit_trade = _place_order_mkt(ib, symbol=sym, exchange=exch, currency=ccy, side="SELL", qty=float(filled_qty))
            report["actions"]["details"].append({"event": "exit_order_submitted", "symbol": sym, "qty": float(filled_qty)})

            exit_filled, exit_filled_qty, exit_status, exit_fp = _wait_for_fill_or_cancel(ib, exit_trade, timeout_s=float(cfg.auto_cancel_seconds))
            if not exit_filled:
                report["actions"]["orders_cancelled"] += 1
                report["actions"]["details"].append({"event": "exit_not_filled_cancelled", "symbol": sym, "status": exit_status, "filled_qty": float(exit_filled_qty)})
                continue

            if float(exit_fp) <= 0.0 or float(entry_fp) <= 0.0:
                report["actions"]["details"].append({"event": "exit_missing_fill_price", "symbol": sym, "entry_fp": float(entry_fp), "exit_fp": float(exit_fp)})
                continue

            realized = (float(exit_fp) - float(entry_fp)) * float(exit_filled_qty)

            try:
                log_path = log_ibkr_paper_order_event(
                    IBKRPaperOrderEvent(
                        strategy=strat,
                        symbol=sym,
                        side="SELL",
                        quantity=float(exit_filled_qty),
                        filled_quantity=float(exit_filled_qty),
                        fill_price=float(exit_fp),
                        notional_estimate=float(exit_fp) * float(exit_filled_qty),
                        sec_type=sec_type,
                        exchange=exch,
                        currency=ccy,
                        order_type="MKT",
                        order_id=_safe_int(getattr(exit_trade.order, "orderId", None), 0) or None,
                        perm_id=_safe_int(getattr(exit_trade.order, "permId", None), 0) or None,
                        raw_intent=dict(it) | {"exit": True, "entry_fill_price": float(entry_fp), "exit_fill_price": float(exit_fp)},
                        realized_pnl=float(realized),
                        source="paper_shadow_runner",
                    )
                )
                report["actions"]["trade_results_logged"] += 1
                report["actions"]["details"].append({"event": "trade_result_logged", "log_path": str(log_path), "realized_pnl": float(realized)})
            except Exception as exc:
                report["actions"]["details"].append({"event": "exit_log_failed", "error": f"{type(exc).__name__}: {exc}"})

            # One cycle per run (safe and predictable)
            break

        try:
            opens = ib.reqOpenOrders()
            report["post"]["openOrders_count"] = int(len(opens))
        except Exception:
            report["post"]["openOrders_count"] = 0

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    report["safety"]["no_trade_ledgers_written"] = (int(report["actions"]["trade_results_logged"]) == 0)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHAD Paper Shadow Runner (paper-only, safe-by-default).")
    ap.add_argument("--preview", action="store_true", help="Force preview-only (no orders).")
    ap.add_argument("--execute", action="store_true", help="Execute paper orders (requires explicit arming).")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to runtime paper shadow config JSON.")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config).expanduser().resolve())
    allow_paper, lg_reasons = _fetch_live_gate()
    full_cycle_payload, intents_all = _load_full_cycle_intents()

    intents_filtered = _filter_intents(
        intents_all,
        strategy_allowlist=cfg.strategy_allowlist,
        min_notional_usd=cfg.min_notional_usd,
    )

    inv = _build_fifo_inventory_from_ledger()
    inv_net = _net_position_from_inventory(inv)

    reasons: List[str] = []
    if args.preview:
        reasons.append("--preview forced")
    if not cfg.enabled:
        reasons.append("paper_shadow.enabled is false")
    if not cfg.armed:
        reasons.append("paper_shadow.armed is false")
    if not _env_armed():
        reasons.append(f"{ARM_ENV_VAR} is not set to arm phrase")
    if allow_paper:
        reasons.append("live-gate allows IBKR paper")
    else:
        reasons.append("live-gate does NOT allow IBKR paper")
    reasons.extend(lg_reasons[:8])

    can_execute = bool(args.execute) and (not args.preview) and cfg.enabled and cfg.armed and _env_armed() and allow_paper

    if not can_execute:
        report = run_preview(cfg, full_cycle_payload, intents_all, intents_filtered, reasons, inv_net)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"PAPER_SHADOW_PREVIEW_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
        _atomic_write_json(out, report)
        print("[paper_shadow_runner] mode=preview")
        print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={cfg.armed}")
        for r in reasons:
            print(f"[paper_shadow_runner] reason: {r}")
        print(f"[paper_shadow_runner] wrote: {out}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    report = run_execute(cfg, full_cycle_payload, intents_all, intents_filtered, reasons)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"PAPER_SHADOW_EXECUTE_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
    _atomic_write_json(out, report)
    print("[paper_shadow_runner] mode=execute")
    print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={cfg.armed} env_armed={_env_armed()}")
    for r in reasons:
        print(f"[paper_shadow_runner] reason: {r}")
    print(f"[paper_shadow_runner] wrote: {out}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
