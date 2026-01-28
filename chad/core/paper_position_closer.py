"""
CHAD Paper Position Closer (IBKR PAPER ONLY)

This module is the missing "exit leg" for CHAD's paper warmup lifecycle.

It is intentionally conservative:
- PAPER ONLY (IBKR Gateway paper API :4002)
- Requires STOP=off
- Requires CHAD_EXECUTION_MODE=dry_run
- Requires operator intent EXIT_ONLY (unless explicitly overridden)
- Never opens new entries; only closes existing positions
- Default close allowlist is restrictive (AAPL only)

Key fix in this version:
- Works even WITHOUT IBKR market data subscription.
  If live/delayed market data is unavailable via API, it uses avgCost
  as a deterministic anchor for a LIMIT exit price.
- Sets outsideRth=True for LIMIT orders when RTH is closed, so the order
  can be eligible in extended hours (subject to IBKR rules).

All actions are written to an auditable JSON report in:
  /home/ubuntu/CHAD FINALE/reports/closer/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ib_insync import IB, MarketOrder, LimitOrder, Stock  # type: ignore


# ----------------------------- helpers ---------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"_error": "json_decode_error"}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


def _now_ts() -> float:
    return time.time()


@dataclass(frozen=True)
class CloseDecision:
    symbol: str
    qty: float
    side: str  # SELL closes long, BUY covers short (shorting not expected under current policy)
    order_type: str  # MKT/LMT/SKIP
    limit_price: Optional[float]
    outside_rth: bool
    reason: str


@dataclass
class CloserConfig:
    host: str
    port: int
    client_id: int

    require_exit_only: bool
    require_dry_run_mode: bool

    cancel_open_orders: bool
    allow_symbols: List[str]
    min_qty: float
    max_seconds_wait: int
    poll_seconds: float

    use_limit_orders: bool
    limit_offset_pct: float
    allow_market_outside_rth: bool

    reports_dir: Path
    stop_state_path: Path
    operator_intent_path: Path


def load_config(args: argparse.Namespace) -> CloserConfig:
    base = Path("/home/ubuntu/CHAD FINALE")
    allow_symbols = [s.strip().upper() for s in args.allow_symbols.split(",") if s.strip()]
    if not allow_symbols:
        allow_symbols = ["AAPL"]

    return CloserConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        require_exit_only=not args.allow_if_not_exit_only,
        require_dry_run_mode=True,
        cancel_open_orders=not args.no_cancel,
        allow_symbols=allow_symbols,
        min_qty=float(args.min_qty),
        max_seconds_wait=int(args.max_wait_seconds),
        poll_seconds=float(args.poll_seconds),
        use_limit_orders=not args.use_market_orders,
        limit_offset_pct=float(args.limit_offset_pct),
        allow_market_outside_rth=bool(args.allow_market_outside_rth),
        reports_dir=base / "reports" / "closer",
        stop_state_path=base / "runtime" / "stop_state.json",
        operator_intent_path=base / "runtime" / "operator_intent.json",
    )


# --------------------------- safety gates -------------------------------


def assert_safe_to_run(cfg: CloserConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    stop = _read_json(cfg.stop_state_path)
    if stop.get("_error"):
        reasons.append("STOP_STATE_INVALID_JSON => FAIL_CLOSED")
        return False, reasons
    if bool(stop.get("stop", False)):
        reasons.append(f"STOP_ENABLED reason={stop.get('reason', '')!r}")
        return False, reasons

    exec_mode = _env_str("CHAD_EXECUTION_MODE", "").lower()
    if cfg.require_dry_run_mode and exec_mode != "dry_run":
        reasons.append(f"CHAD_EXECUTION_MODE={exec_mode!r} (must be 'dry_run')")
        return False, reasons

    intent = _read_json(cfg.operator_intent_path)
    if intent.get("_error"):
        reasons.append("OPERATOR_INTENT_INVALID_JSON => FAIL_CLOSED")
        return False, reasons
    mode = str(intent.get("mode", "UNKNOWN")).upper()
    if cfg.require_exit_only and mode != "EXIT_ONLY":
        reasons.append(f"OPERATOR_INTENT={mode} (must be EXIT_ONLY)")
        return False, reasons

    reasons.append("SAFE_TO_RUN")
    return True, reasons


# --------------------------- IBKR helpers --------------------------------


def _is_rth_open_for_symbol(ib: IB, symbol: str) -> Tuple[bool, str]:
    """
    Conservative RTH check using IBKR liquidHours for the symbol.

    This is a safety gate only (avoid MKT orders outside RTH by default).
    """
    c = Stock(symbol, "SMART", "USD")
    cds = ib.reqContractDetails(c)
    lh = cds[0].liquidHours if cds else ""

    now_utc = datetime.now(timezone.utc)
    now_s = now_utc.strftime("%Y%m%d%H%M")
    open_now = False

    for part in (lh or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        date_s, window = part.split(":", 1)
        if window.upper() == "CLOSED":
            continue
        for seg in window.split(","):
            seg = seg.strip()
            if "-" not in seg:
                continue
            a, b = seg.split("-", 1)
            a = a.strip().replace(":", "")
            b = b.strip().replace(":", "")
            if len(a) == 4:
                a = f"{date_s}{a}"
            if len(b) == 4:
                b = f"{date_s}{b}"
            if len(a) != 12 or len(b) != 12:
                continue
            if a <= now_s <= b:
                open_now = True
                break
        if open_now:
            break

    return open_now, lh


def _snapshot_price_best_effort(ib: IB, symbol: str) -> Optional[float]:
    """
    Best-effort snapshot price.

    IMPORTANT:
    - If the account lacks market data subscription, IBKR may still offer delayed data,
      but API access can be restricted. In that case, this function may return None.
    - We must be able to operate without this. (We fallback to avgCost.)
    """
    c = Stock(symbol, "SMART", "USD")
    try:
        tickers = ib.reqTickers(c)
        if not tickers:
            return None
        t = tickers[0]
        bid = getattr(t, "bid", None)
        ask = getattr(t, "ask", None)
        last = getattr(t, "last", None)
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        if isinstance(last, (int, float)) and last > 0:
            return float(last)
        return None
    except Exception:
        return None


# --------------------------- closer core ---------------------------------


def build_close_decisions(ib: IB, cfg: CloserConfig) -> Tuple[List[CloseDecision], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    decisions: List[CloseDecision] = []

    positions = ib.positions()
    details.append({"event": "positions_snapshot", "count": len(positions), "ts_utc": _utc_now_iso()})

    for p in positions:
        c = p.contract
        sym = str(getattr(c, "symbol", "") or "").upper()
        sec = str(getattr(c, "secType", "") or "")
        qty = float(getattr(p, "position", 0.0) or 0.0)
        avg_cost = float(getattr(p, "avgCost", 0.0) or 0.0)

        if sec != "STK":
            continue
        if sym not in cfg.allow_symbols:
            continue
        if abs(qty) < cfg.min_qty:
            continue

        side = "SELL" if qty > 0 else "BUY"
        qty_abs = abs(qty)

        open_now, lh = _is_rth_open_for_symbol(ib, sym)
        details.append({"event": "rth_check", "symbol": sym, "open_now": open_now, "liquidHours": lh, "ts_utc": _utc_now_iso()})

        order_type = "LMT" if cfg.use_limit_orders else "MKT"
        outside_rth = bool(not open_now)

        if not open_now and order_type == "MKT" and not cfg.allow_market_outside_rth:
            decisions.append(
                CloseDecision(
                    symbol=sym,
                    qty=qty_abs,
                    side=side,
                    order_type="SKIP",
                    limit_price=None,
                    outside_rth=False,
                    reason="RTH_CLOSED_AND_MKT_NOT_ALLOWED",
                )
            )
            continue

        limit_price: Optional[float] = None

        if order_type == "LMT":
            px = _snapshot_price_best_effort(ib, sym)
            anchor = "snapshot" if (px is not None and px > 0) else "avgCost"

            if px is None or px <= 0:
                if avg_cost > 0:
                    px = float(avg_cost)
                else:
                    decisions.append(
                        CloseDecision(
                            symbol=sym,
                            qty=qty_abs,
                            side=side,
                            order_type="SKIP",
                            limit_price=None,
                            outside_rth=outside_rth,
                            reason="NO_PRICE_AND_NO_AVGCOST",
                        )
                    )
                    continue

            # SELL a bit below anchor; BUY a bit above anchor
            if side == "SELL":
                limit_price = float(px * (1.0 - cfg.limit_offset_pct))
            else:
                limit_price = float(px * (1.0 + cfg.limit_offset_pct))

            details.append(
                {
                    "event": "limit_price_anchor",
                    "symbol": sym,
                    "anchor_source": anchor,
                    "anchor_price": float(px),
                    "limit_price": float(limit_price),
                    "ts_utc": _utc_now_iso(),
                }
            )

        decisions.append(
            CloseDecision(
                symbol=sym,
                qty=qty_abs,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                outside_rth=outside_rth,
                reason="CLOSE_ALLOWED_SYMBOL",
            )
        )

    return decisions, details


def execute_decisions(ib: IB, cfg: CloserConfig, decisions: List[CloseDecision]) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]]]:
    details: List[Dict[str, Any]] = []
    errors: List[Tuple[int, int, str]] = []

    def on_error(reqId, errorCode, errorString, contract):
        errors.append((int(reqId), int(errorCode), str(errorString)))

    ib.errorEvent += on_error

    if cfg.cancel_open_orders:
        oos = ib.openOrders()
        details.append({"event": "open_orders_snapshot", "count": len(oos), "ts_utc": _utc_now_iso()})
        for o in oos:
            try:
                ib.cancelOrder(o)
            except Exception as e:
                details.append({"event": "cancel_order_error", "error": repr(e), "ts_utc": _utc_now_iso()})
        ib.sleep(1.0)

    for d in decisions:
        if d.order_type == "SKIP":
            details.append(
                {
                    "event": "decision_skipped",
                    "symbol": d.symbol,
                    "qty": d.qty,
                    "side": d.side,
                    "reason": d.reason,
                    "ts_utc": _utc_now_iso(),
                }
            )
            continue

        contract = Stock(d.symbol, "SMART", "USD")

        if d.order_type == "MKT":
            order = MarketOrder(d.side, int(d.qty))
            order.outsideRth = bool(d.outside_rth)
        else:
            assert d.limit_price is not None
            order = LimitOrder(d.side, int(d.qty), float(round(d.limit_price, 2)))
            order.outsideRth = bool(d.outside_rth)

        trade = ib.placeOrder(contract, order)

        details.append(
            {
                "event": "order_placed",
                "symbol": d.symbol,
                "qty": d.qty,
                "side": d.side,
                "order_type": d.order_type,
                "limit_price": d.limit_price,
                "outsideRth": bool(d.outside_rth),
                "orderId": int(trade.order.orderId),
                "ts_utc": _utc_now_iso(),
            }
        )

        start = _now_ts()
        last: Optional[Tuple[str, float, float]] = None
        while _now_ts() - start < float(cfg.max_seconds_wait):
            ib.sleep(cfg.poll_seconds)
            st = str(trade.orderStatus.status)
            filled = float(trade.orderStatus.filled or 0.0)
            remaining = float(trade.orderStatus.remaining or 0.0)
            cur = (st, filled, remaining)
            if cur != last:
                details.append({"event": "order_status", "symbol": d.symbol, "status": st, "filled": filled, "remaining": remaining, "ts_utc": _utc_now_iso()})
                last = cur
            if trade.isDone():
                break

        details.append(
            {
                "event": "order_final",
                "symbol": d.symbol,
                "final_status": str(trade.orderStatus.status),
                "filled": float(trade.orderStatus.filled or 0.0),
                "avgFillPrice": float(trade.orderStatus.avgFillPrice or 0.0),
                "ts_utc": _utc_now_iso(),
            }
        )

    return details, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="CHAD IBKR PAPER Position Closer (exit-only).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=9555)

    parser.add_argument("--allow-symbols", default="AAPL", help="Comma-separated symbols to close (default AAPL).")
    parser.add_argument("--min-qty", type=float, default=1.0)
    parser.add_argument("--max-wait-seconds", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=1.0)

    parser.add_argument("--use-market-orders", action="store_true", help="Use market orders (RTH only unless overridden).")
    parser.add_argument("--allow-market-outside-rth", action="store_true", help="Allow market orders outside RTH (NOT recommended).")
    parser.add_argument("--use-limit-offset-pct", dest="limit_offset_pct", type=float, default=0.002, help="Limit offset pct (default 0.2%).")

    parser.add_argument("--no-cancel", action="store_true", help="Do not cancel open orders before closing.")
    parser.add_argument("--allow-if-not-exit-only", action="store_true", help="Override operator intent gate (NOT recommended).")

    args = parser.parse_args()
    cfg = load_config(args)

    ok, safety_reasons = assert_safe_to_run(cfg)

    report: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "enabled": bool(ok),
        "safety_reasons": safety_reasons,
        "config": {
            "host": cfg.host,
            "port": cfg.port,
            "client_id": cfg.client_id,
            "allow_symbols": cfg.allow_symbols,
            "use_limit_orders": cfg.use_limit_orders,
            "limit_offset_pct": cfg.limit_offset_pct,
            "max_wait_seconds": cfg.max_seconds_wait,
            "poll_seconds": cfg.poll_seconds,
            "cancel_open_orders": cfg.cancel_open_orders,
        },
        "details": [],
        "errors": [],
    }

    if not ok:
        out = cfg.reports_dir / f"PAPER_CLOSER_ABORT_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        _write_json_atomic(out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    ib = IB()
    ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=10)

    decisions, d1 = build_close_decisions(ib, cfg)
    report["details"].extend(d1)
    report["decisions"] = [
        {
            "symbol": x.symbol,
            "qty": x.qty,
            "side": x.side,
            "order_type": x.order_type,
            "limit_price": x.limit_price,
            "outsideRth": bool(x.outside_rth),
            "reason": x.reason,
        }
        for x in decisions
    ]

    d2, errors = execute_decisions(ib, cfg, decisions)
    report["details"].extend(d2)
    report["errors"] = [{"reqId": a, "code": b, "msg": c} for (a, b, c) in errors]

    pos = ib.positions()
    report["post_positions"] = [
        {
            "symbol": str(getattr(p.contract, "symbol", "") or ""),
            "secType": str(getattr(p.contract, "secType", "") or ""),
            "qty": float(getattr(p, "position", 0.0) or 0.0),
            "avgCost": float(getattr(p, "avgCost", 0.0) or 0.0),
        }
        for p in pos
    ]

    ib.disconnect()

    out = cfg.reports_dir / f"PAPER_CLOSER_RUN_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _write_json_atomic(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
