#!/usr/bin/env python3
"""
CHAD — Paper Shadow Runner (IBKR PAPER execution, safe-by-default)

What this does
--------------
- Builds a paper execution "intent" list from the authoritative runtime plan:
    runtime/full_execution_cycle_last.json
- Filters by allowlist, min_notional, and max_orders_per_run.
- In PREVIEW mode: writes a preview artifact only (no broker I/O).
- In EXECUTE mode: places *paper* orders on IBKR (when explicitly armed), and:
    - cancels unfilled orders after a bounded timeout
    - logs a CHAD TradeResult record ONLY if the order actually fills
      (so it counts as an effective trade, not a fake one)

Safety model
------------
To execute orders, ALL must be true:
- config.enabled == True
- config.armed == True
- env CHAD_PAPER_SHADOW_ARMED equals ARM_PHRASE
- LiveGate allows IBKR paper (allow_ibkr_paper == True)

Otherwise it falls back to PREVIEW.

Outputs
-------
Writes JSON artifacts under:
  reports/shadow/PAPER_SHADOW_PREVIEW_*.json
  reports/shadow/PAPER_SHADOW_EXECUTE_*.json

"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ib_insync import IB, MarketOrder, LimitOrder, Stock  # type: ignore

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


# ----------------------------
# Utilities
# ----------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _env_armed() -> bool:
    return (str((__import__("os").environ.get(ARM_ENV_VAR) or "")).strip() == ARM_PHRASE)


def _fetch_live_gate(timeout_s: float = 2.0) -> Tuple[bool, List[str]]:
    """
    Returns:
      (allow_ibkr_paper, reasons)
    """
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
    """
    Returns:
      (full_cycle_payload, ibkr_intents_list)
    """
    if not FULL_CYCLE_LAST.is_file():
        raise FileNotFoundError(f"Missing {FULL_CYCLE_LAST}. Run: python3 -m chad.core.full_execution_cycle")
    payload = json.loads(FULL_CYCLE_LAST.read_text(encoding="utf-8"))
    intents = payload.get("ibkr_intents") or []
    if not isinstance(intents, list):
        raise ValueError("runtime/full_execution_cycle_last.json: ibkr_intents is not a list")
    return payload, intents


def _norm_strategy(s: Any) -> str:
    return str(s or "").strip().lower()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class IBKRConn:
    host: str
    port: int
    client_id: int


@dataclass(frozen=True)
class PaperShadowConfig:
    enabled: bool
    armed: bool
    strategy_allowlist: List[str]
    max_orders_per_run: int
    size_multiplier: float
    min_notional_usd: float
    auto_cancel_seconds: int
    ibkr: IBKRConn


def load_config(path: Path) -> PaperShadowConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ibkr_raw = raw.get("ibkr") or {}
    return PaperShadowConfig(
        enabled=bool(raw.get("enabled", False)),
        armed=bool(raw.get("armed", False)),
        strategy_allowlist=[str(x) for x in (raw.get("strategy_allowlist") or [])],
        max_orders_per_run=max(0, int(raw.get("max_orders_per_run", 0))),
        size_multiplier=float(raw.get("size_multiplier", 1.0)),
        min_notional_usd=float(raw.get("min_notional_usd", 0.0)),
        auto_cancel_seconds=max(1, int(raw.get("auto_cancel_seconds", 15))),
        ibkr=IBKRConn(
            host=str(ibkr_raw.get("host", "127.0.0.1")),
            port=int(ibkr_raw.get("port", 4002)),
            client_id=int(ibkr_raw.get("client_id", 9201)),
        ),
    )


# ----------------------------
# Core logic
# ----------------------------

def _filter_intents(
    intents: List[Dict[str, Any]],
    *,
    strategy_allowlist: List[str],
    min_notional_usd: float,
) -> List[Dict[str, Any]]:
    allow = {s.strip().lower() for s in strategy_allowlist if str(s).strip()}
    out: List[Dict[str, Any]] = []
    for it in intents:
        strat = _norm_strategy(it.get("strategy"))
        notional = _safe_float(it.get("notional_estimate"), 0.0)
        if allow and strat.upper() not in {a.upper() for a in allow} and strat not in allow:
            continue
        if notional < float(min_notional_usd):
            continue
        out.append(it)
    return out


def _build_report_base(cfg: PaperShadowConfig, *, mode: str, reasons: List[str], full_cycle_payload: Dict[str, Any], intents_total: int, intents_filtered: int, planned_orders: int) -> Dict[str, Any]:
    return {
        "mode": mode,
        "generated_at_utc": _utc_now_iso(),
        "armed_env": _env_armed(),
        "config": {
            "enabled": bool(cfg.enabled),
            "armed": bool(cfg.armed),
            "strategy_allowlist": list(cfg.strategy_allowlist),
            "max_orders_per_run": int(cfg.max_orders_per_run),
            "size_multiplier": float(cfg.size_multiplier),
            "min_notional_usd": float(cfg.min_notional_usd),
            "auto_cancel_seconds": int(cfg.auto_cancel_seconds),
            "ibkr": {
                "host": cfg.ibkr.host,
                "port": int(cfg.ibkr.port),
                "client_id": int(cfg.ibkr.client_id),
            },
            "reports_dir": str(REPORTS_DIR),
        },
        "reasons": list(reasons),
        "counts": {
            # summary counts if present
            "raw_signals": int(((full_cycle_payload.get("summary") or {}).get("raw_signals") or 0)),
            "evaluated_signals": int(((full_cycle_payload.get("summary") or {}).get("evaluated_signals") or 0)),
            "routed_signals": int(((full_cycle_payload.get("summary") or {}).get("routed_signals") or 0)),
            "ibkr_intents_total": int(intents_total),
            "ibkr_intents_filtered": int(intents_filtered),
            "planned_orders": int(planned_orders),
        },
        "paper_intents_preview": [],
        "actions": {
            "orders_submitted": 0,
            "orders_cancelled": 0,
            "trade_results_logged": 0,
            "details": [],
        },
        "post": {
            "openOrders_count": 0,
        },
        "safety": {
            "no_orders_placed": True,
            "no_trade_ledgers_written": True,
        },
    }


def _intent_to_preview(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strategy": it.get("strategy"),
        "symbol": it.get("symbol"),
        "side": it.get("side"),
        "sec_type": it.get("sec_type"),
        "exchange": it.get("exchange"),
        "currency": it.get("currency"),
        "order_type": it.get("order_type"),
        "quantity": it.get("quantity"),
        "notional_estimate": it.get("notional_estimate"),
        "limit_price": it.get("limit_price"),
    }


def _place_one_order(ib: IB, it: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Places an order and returns (trade, detail_dict).
    """
    symbol = str(it.get("symbol") or "").strip().upper()
    side = str(it.get("side") or "BUY").strip().upper()
    qty = float(it.get("quantity") or 0.0)
    order_type = str(it.get("order_type") or "MKT").strip().upper()
    limit_price = it.get("limit_price", None)

    if qty <= 0:
        raise ValueError(f"Invalid quantity: {qty}")

    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    if order_type == "LMT":
        if limit_price is None:
            raise ValueError("Limit order requires limit_price")
        order = LimitOrder(side, qty, float(limit_price))
    else:
        # Default to market
        order = MarketOrder(side, qty)

    trade = ib.placeOrder(contract, order)
    detail = {
        "event": "order_submitted",
        "strategy": str(it.get("strategy")),
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "order_type": order_type,
        "notional_estimate": float(it.get("notional_estimate") or 0.0),
        "orderId": int(getattr(order, "orderId", 0) or 0) or None,
        "status": str(getattr(trade.orderStatus, "status", "") or ""),
    }
    return trade, detail


def _wait_for_fill_or_cancel(
    ib: IB,
    trade: Any,
    *,
    timeout_s: float,
) -> Tuple[bool, float, str, float]:
    """
    Returns: (filled, filled_qty, status, avg_fill_price)

    Hard rule:
      - Treat as FILLED only when:
          filled_qty > 0 AND avgFillPrice > 0

    Rationale:
      - Prevent SCR from counting placeholder paper events.
      - Ensure we only log TradeResult when we have concrete fill proof.
    """
    deadline = time.time() + float(timeout_s)
    filled_qty = 0.0
    status = ""
    avg_fp = 0.0

    while time.time() < deadline:
        ib.sleep(0.25)
        status = str(getattr(trade.orderStatus, "status", "") or "")
        filled_qty = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
        avg_fp = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)

        # Only accept "filled" if we have fill proof + price.
        if filled_qty > 0.0 and avg_fp > 0.0:
            return True, filled_qty, status, avg_fp

        if status.lower() in {"cancelled", "inactive"}:
            return False, filled_qty, status, avg_fp

    # Timeout: cancel, then re-check one last time.
    try:
        ib.cancelOrder(trade.order)
        ib.sleep(0.25)
        status = str(getattr(trade.orderStatus, "status", "") or "")
        filled_qty = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
        avg_fp = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
    except Exception:
        pass

    return False, filled_qty, status, avg_fp


def run_preview(cfg: PaperShadowConfig, full_cycle_payload: Dict[str, Any], intents_all: List[Dict[str, Any]], intents_filtered: List[Dict[str, Any]], reasons: List[str]) -> Dict[str, Any]:
    planned = intents_filtered[: max(0, cfg.max_orders_per_run)]
    report = _build_report_base(
        cfg,
        mode="preview",
        reasons=reasons,
        full_cycle_payload=full_cycle_payload,
        intents_total=len(intents_all),
        intents_filtered=len(intents_filtered),
        planned_orders=len(planned),
    )
    report["paper_intents_preview"] = [_intent_to_preview(it) for it in planned]
    report["safety"]["no_orders_placed"] = True
    report["safety"]["no_trade_ledgers_written"] = True
    return report


def run_execute(cfg: PaperShadowConfig, full_cycle_payload: Dict[str, Any], intents_all: List[Dict[str, Any]], intents_filtered: List[Dict[str, Any]], reasons: List[str]) -> Dict[str, Any]:
    planned = intents_filtered[: max(0, cfg.max_orders_per_run)]
    report = _build_report_base(
        cfg,
        mode="execute",
        reasons=reasons,
        full_cycle_payload=full_cycle_payload,
        intents_total=len(intents_all),
        intents_filtered=len(intents_filtered),
        planned_orders=len(planned),
    )
    report["paper_intents_preview"] = [_intent_to_preview(it) for it in planned]
    report["safety"]["no_orders_placed"] = False

    ib = IB()
    try:
        ib.connect(cfg.ibkr.host, cfg.ibkr.port, clientId=cfg.ibkr.client_id, timeout=10)

        for it in planned:
            trade, detail = _place_one_order(ib, it)
            report["actions"]["orders_submitted"] += 1
            report["actions"]["details"].append(detail)

            filled, filled_qty, status, avg_fp = _wait_for_fill_or_cancel(ib, trade, timeout_s=float(cfg.auto_cancel_seconds))
            if not filled:
                report["actions"]["orders_cancelled"] += 1
                report["actions"]["details"].append(
                    {"event": "order_not_filled_cancelled", "status": status, "filled_qty": float(filled_qty)}
                )
                continue

            # Filled: log effective strategy trade result
            try:
                log_path = log_ibkr_paper_order_event(
                    IBKRPaperOrderEvent(
                        strategy=str(it.get("strategy") or "").strip().lower() or "beta",
                        symbol=str(it.get("symbol") or "UNKNOWN"),
                        side=str(it.get("side") or "BUY"),
                        quantity=float(it.get("quantity") or 0.0),
                        filled_quantity=float(filled_qty or 0.0),
                        fill_price=float(avg_fp or 0.0),
                        notional_estimate=float(it.get("notional_estimate") or 0.0),
                        sec_type=str(it.get("sec_type") or "STK"),
                        exchange=str(it.get("exchange") or "SMART"),
                        currency=str(it.get("currency") or "USD"),
                        order_type=str(it.get("order_type") or "MKT"),
                        order_id=_safe_int(getattr(trade.order, "orderId", None), 0) or None,
                        perm_id=_safe_int(getattr(trade.order, "permId", None), 0) or None,
                        raw_intent=dict(it),
                    )
                )
                report["actions"]["trade_results_logged"] += 1
                report["actions"]["details"].append({"event": "trade_result_logged", "log_path": str(log_path)})
            except Exception as exc:
                report["actions"]["details"].append(
                    {"event": "trade_result_log_failed", "error": f"{type(exc).__name__}: {exc}"}
                )

        # Post: openOrders count
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
    p = argparse.ArgumentParser(description="CHAD Paper Shadow Runner (safe-by-default).")
    p.add_argument("--preview", action="store_true", help="Force preview-only (no orders).")
    p.add_argument("--execute", action="store_true", help="Execute paper orders (requires explicit arming).")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to runtime paper shadow config JSON.")
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config).expanduser().resolve())

    allow_paper, lg_reasons = _fetch_live_gate()

    full_cycle_payload, intents_all = _load_full_cycle_intents()
    intents_filtered = _filter_intents(
        intents_all,
        strategy_allowlist=cfg.strategy_allowlist,
        min_notional_usd=cfg.min_notional_usd,
    )

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
        reasons.extend(lg_reasons[:5])

    can_execute = bool(args.execute) and (not args.preview) and cfg.enabled and cfg.armed and _env_armed() and allow_paper

    if not can_execute:
        report = run_preview(cfg, full_cycle_payload, intents_all, intents_filtered, reasons)
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
