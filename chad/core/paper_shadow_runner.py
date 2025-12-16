#!/usr/bin/env python3
"""
CHAD Paper Shadow Runner (Phase 8 – Paper Execution Runner, Production-Safe)

Purpose
-------
Provide a scheduled, audit-friendly “paper shadow” lane that can be toggled on/off
via runtime config. This lane can operate in two modes:

1) PREVIEW (default):
   - Builds signals → plans → IBKR intents
   - Writes an audit artifact
   - Places NO orders

2) EXECUTE (explicit, guarded):
   - Requires: --execute AND config enabled=true AND config armed=true
              AND CHAD_PAPER_SHADOW_ARMED env var set to ARM_PHRASE
              AND LiveGate allows IBKR paper
   - Places PAPER orders via IBKR API (ib_insync)
   - Enforces max orders, min notional, size multiplier
   - Cancels any still-open orders after auto_cancel_seconds
   - Writes an audit artifact with broker responses

Safety Guarantees
-----------------
- Default behavior is preview-only (no orders).
- Missing/invalid config => disabled => preview-only.
- --preview always forces preview-only.
- systemd timer is pinned to --preview (no orders from systemd).
- Execution requires explicit operator actions (config + env arming + CLI flag).
- Never enables LIVE trading. Never changes CHAD global mode. Never modifies caps.

Backwards-Compatible Test Contract
----------------------------------
The following symbols are REQUIRED by unit tests and must remain:
- ARM_ENV_NAME
- ARM_PHRASE
- PaperShadowConfig
- is_armed()
- should_place_paper_orders(cfg)

IMPORTANT: should_place_paper_orders() is intentionally “small” and only checks:
enabled + ENV arm phrase. It must not depend on cfg.armed (tests do not set it).

CLI
---
python -m chad.core.paper_shadow_runner --preview
python -m chad.core.paper_shadow_runner --config /path/to/paper_shadow.json --preview
python -m chad.core.paper_shadow_runner --config /path/to/paper_shadow.json --execute

Runtime Config
--------------
Default path: /home/ubuntu/CHAD FINALE/runtime/paper_shadow.json

Example:
{
  "enabled": false,
  "armed": false,
  "strategy_allowlist": ["BETA"],
  "max_orders_per_run": 5,
  "size_multiplier": 1.0,
  "min_notional_usd": 100.0,
  "auto_cancel_seconds": 15,
  "ibkr": {"host":"127.0.0.1","port":4002,"client_id":9201}
}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Backward-compatible test/ops contract constants
# ---------------------------------------------------------------------------

ARM_PHRASE = "I_UNDERSTAND_THIS_CAN_PLACE_PAPER_ORDERS"
ARM_ENV_NAME = "CHAD_PAPER_SHADOW_ARMED"

ROOT_DEFAULT = Path("/home/ubuntu/CHAD FINALE")
CONFIG_DEFAULT = ROOT_DEFAULT / "runtime" / "paper_shadow.json"
REPORTS_DIR_DEFAULT = ROOT_DEFAULT / "reports" / "shadow"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def is_armed() -> bool:
    """
    Test contract function: Return True only when the operator explicitly arms paper-order capability.
    Must remain pure and safe (no broker I/O).
    """
    return os.environ.get(ARM_ENV_NAME, "").strip() == ARM_PHRASE


def should_place_paper_orders(cfg: "PaperShadowConfig") -> Tuple[bool, List[str]]:
    """
    Test contract gating function (safe-by-default).

    NOTE:
    - Tests only expect enabled + env phrase checks.
    - Do NOT include cfg.armed in this function, or tests will fail.
    """
    reasons: List[str] = []
    if not bool(getattr(cfg, "enabled", False)):
        reasons.append("paper_shadow.enabled is false")
        return False, reasons
    if not is_armed():
        reasons.append("CHAD_PAPER_SHADOW_ARMED is not set to arm phrase")
        return False, reasons
    reasons.append("enabled=true and env-armed=true")
    return True, reasons


# ---------------------------------------------------------------------------
# Full runtime config schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperShadowConfig:
    enabled: bool = False
    # Additional config-level arming (separate from env arming).
    # This is NOT part of should_place_paper_orders() (tests do not set it).
    armed: bool = False

    strategy_allowlist: List[str] | None = None
    max_orders_per_run: int = 5
    size_multiplier: float = 1.0
    min_notional_usd: float = 100.0
    auto_cancel_seconds: int = 15

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002
    ibkr_client_id: int = 9201

    reports_dir: Path = REPORTS_DIR_DEFAULT

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_allowlist", self.strategy_allowlist or ["BETA"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "armed": bool(self.armed),
            "strategy_allowlist": list(self.strategy_allowlist or []),
            "max_orders_per_run": int(self.max_orders_per_run),
            "size_multiplier": float(self.size_multiplier),
            "min_notional_usd": float(self.min_notional_usd),
            "auto_cancel_seconds": int(self.auto_cancel_seconds),
            "ibkr": {"host": self.ibkr_host, "port": self.ibkr_port, "client_id": self.ibkr_client_id},
            "reports_dir": str(self.reports_dir),
        }


def load_config(path: Path) -> PaperShadowConfig:
    """
    Safe-by-default loader:
    - Missing config => enabled=False, armed=False
    - Invalid config => enabled=False, armed=False
    """
    if not path.is_file():
        return PaperShadowConfig(enabled=False, armed=False)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return PaperShadowConfig(enabled=False, armed=False)

    enabled = bool(raw.get("enabled", False))
    armed = bool(raw.get("armed", False))

    allow = raw.get("strategy_allowlist", ["BETA"])
    if not isinstance(allow, list):
        allow = ["BETA"]
    allow_norm = [str(x).strip().upper() for x in allow if str(x).strip()]

    ibkr = raw.get("ibkr", {}) if isinstance(raw.get("ibkr", {}), dict) else {}
    host = str(ibkr.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(ibkr.get("port", 4002))
    cid = int(ibkr.get("client_id", 9201))

    return PaperShadowConfig(
        enabled=enabled,
        armed=armed,
        strategy_allowlist=allow_norm or ["BETA"],
        max_orders_per_run=int(raw.get("max_orders_per_run", 5)),
        size_multiplier=float(raw.get("size_multiplier", 1.0)),
        min_notional_usd=float(raw.get("min_notional_usd", 100.0)),
        auto_cancel_seconds=int(raw.get("auto_cancel_seconds", 15)),
        ibkr_host=host,
        ibkr_port=port,
        ibkr_client_id=cid,
        reports_dir=REPORTS_DIR_DEFAULT,
    )


# ---------------------------------------------------------------------------
# Phase 8 execution gates (stronger than test gate)
# ---------------------------------------------------------------------------

def _should_execute_paper(cfg: PaperShadowConfig) -> Tuple[bool, List[str]]:
    """
    Strong execution gate for placing paper orders.
    Requires BOTH config arming and env arming.
    """
    reasons: List[str] = []
    if not cfg.enabled:
        reasons.append("paper_shadow.enabled is false")
        return False, reasons
    if not cfg.armed:
        reasons.append("paper_shadow.armed is false")
        return False, reasons
    if not is_armed():
        reasons.append("CHAD_PAPER_SHADOW_ARMED is not set to arm phrase")
        return False, reasons
    reasons.append("enabled=true, config-armed=true, env-armed=true")
    return True, reasons


def _live_gate_allows_paper() -> Tuple[bool, str]:
    """
    Read-only check against CHAD API gateway live-gate.
    If the endpoint is unavailable, we FAIL SAFE (no execution).
    """
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:9618/live-gate", timeout=2.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        allow_paper = bool(data.get("allow_ibkr_paper", False))
        allow_live = bool(data.get("allow_ibkr_live", False))
        if allow_live:
            return False, "live-gate indicates allow_ibkr_live=true (refuse paper runner)"
        if not allow_paper:
            return False, "live-gate indicates allow_ibkr_paper=false (refuse paper runner)"
        return True, "live-gate allows IBKR paper"
    except Exception as e:
        return False, f"live-gate check failed (fail-safe): {e}"


# ---------------------------------------------------------------------------
# Preview pipeline (build intents)
# ---------------------------------------------------------------------------

def _build_intents_preview(cfg: PaperShadowConfig) -> Dict[str, Any]:
    """
    Builds a preview payload with intent list.
    Imports heavy modules lazily; does NOT import ib_insync here.
    """
    # Lazy imports keep gating tests safe.
    from chad.engine import StrategyEngine
    from chad.strategies import register_core_strategies
    from chad.utils.context_builder import ContextBuilder
    from chad.utils.pipeline import DecisionPipeline, PipelineConfig
    from chad.utils.signal_router import SignalRouter
    from chad.execution.execution_pipeline import build_execution_plan
    from chad.execution.execution_pipeline import build_ibkr_intents_from_plan

    now = _utc_now()
    builder = ContextBuilder()
    result = builder.build()

    engine = StrategyEngine()
    register_core_strategies(engine)

    router = SignalRouter()
    pipeline = DecisionPipeline(engine=engine, router=router, config=PipelineConfig(use_policy=True))
    pr = pipeline.run(
        ctx=result.context,
        prices=result.prices,
        current_symbol_notional=result.current_symbol_notional,
        current_total_notional=result.current_total_notional,
    )

    routed = getattr(pr, "routed_signals", None) or []
    plan = build_execution_plan(routed_signals=routed, prices=result.prices)
    intents = build_ibkr_intents_from_plan(plan)

    allow = set((s or "").strip().upper() for s in (cfg.strategy_allowlist or []))
    filtered: List[Any] = []
    for intent in intents:
        strat = str(getattr(intent, "strategy", "")).strip().upper()
        if strat and strat in allow:
            filtered.append(intent)

    # Apply min_notional + max_orders and size multiplier.
    out: List[Dict[str, Any]] = []
    for intent in filtered:
        notional = float(getattr(intent, "notional_estimate", 0.0) or 0.0)
        if notional < float(cfg.min_notional_usd):
            continue
        qty = float(getattr(intent, "quantity", 0.0) or 0.0) * float(cfg.size_multiplier)
        out.append(
            {
                "strategy": str(getattr(intent, "strategy", "")),
                "symbol": str(getattr(intent, "symbol", "")),
                "sec_type": str(getattr(intent, "sec_type", "")),
                "exchange": str(getattr(intent, "exchange", "")),
                "currency": str(getattr(intent, "currency", "")),
                "side": str(getattr(intent, "side", "")),
                "order_type": str(getattr(intent, "order_type", "")),
                "quantity": float(qty),
                "notional_estimate": float(notional),
                "limit_price": getattr(intent, "limit_price", None),
            }
        )
        if len(out) >= int(cfg.max_orders_per_run):
            break

    payload: Dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "mode": "preview",
        "armed_env": is_armed(),
        "config": cfg.to_dict(),
        "counts": {
            "raw_signals": int(getattr(pr, "raw_signals_count", 0) or 0),
            "evaluated_signals": int(getattr(pr, "evaluated_signals_count", 0) or 0),
            "routed_signals": int(getattr(pr, "routed_signals_count", 0) or 0),
            "planned_orders": int(getattr(plan, "orders_count", 0) or 0),
            "ibkr_intents_total": len(intents),
            "ibkr_intents_filtered": len(out),
        },
        "paper_intents_preview": out,
        "safety": {
            "no_orders_placed": True,
            "no_trade_ledgers_written": True,
        },
    }
    return payload


def _write_artifact(cfg: PaperShadowConfig, *, prefix: str, payload: Dict[str, Any]) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    out = cfg.reports_dir / f"{prefix}_{ts}.json"
    _atomic_write_json(out, payload)
    return out


# ---------------------------------------------------------------------------
# Execute path (places paper orders)
# ---------------------------------------------------------------------------

def _ibkr_contract_and_order(intent: Dict[str, Any]):
    """
    Build ib_insync Contract + Order from intent dict.
    Limited to common STK use-case for Phase 8.
    """
    from ib_insync import Stock, MarketOrder, LimitOrder  # type: ignore[import]

    sec_type = str(intent.get("sec_type", "")).upper()
    symbol = str(intent.get("symbol", "")).upper()
    exchange = str(intent.get("exchange", "SMART")).upper() or "SMART"
    currency = str(intent.get("currency", "USD")).upper() or "USD"
    side = str(intent.get("side", "")).upper()
    order_type = str(intent.get("order_type", "")).upper()
    qty = float(intent.get("quantity", 0.0) or 0.0)

    if sec_type != "STK" or not symbol or qty <= 0.0 or side not in ("BUY", "SELL"):
        raise ValueError(f"unsupported intent: sec_type={sec_type} symbol={symbol} qty={qty} side={side}")

    contract = Stock(symbol, exchange, currency)
    if order_type == "LMT":
        lp = intent.get("limit_price", None)
        if lp is None:
            raise ValueError("limit order missing limit_price")
        order = LimitOrder(side, qty, float(lp))
    else:
        order = MarketOrder(side, qty)
    return contract, order


def _execute_paper_orders(cfg: PaperShadowConfig, intents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute paper orders via ib_insync. Cancels lingering orders after auto_cancel_seconds.
    """
    from ib_insync import IB  # type: ignore[import]

    now = _utc_now()
    report: Dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "mode": "execute",
        "armed_env": is_armed(),
        "config": cfg.to_dict(),
        "actions": {"orders_submitted": 0, "orders_cancelled": 0, "details": []},
        "safety": {
            "no_trade_ledgers_written": True,
        },
    }

    ib = IB()
    ib.connect(cfg.ibkr_host, cfg.ibkr_port, clientId=cfg.ibkr_client_id, timeout=10)

    try:
        submitted: List[Any] = []
        for intent in intents:
            try:
                contract, order = _ibkr_contract_and_order(intent)
            except Exception as e:
                report["actions"]["details"].append({"intent": intent, "status": "skipped", "reason": str(e)})
                continue

            trade = ib.placeOrder(contract, order)
            submitted.append(trade)
            report["actions"]["orders_submitted"] += 1
            report["actions"]["details"].append(
                {
                    "intent": intent,
                    "orderId": getattr(trade.order, "orderId", None),
                    "status": str(getattr(trade.orderStatus, "status", "")),
                }
            )

        # Allow fills to occur briefly, then cancel anything still open.
        ib.sleep(max(0.0, float(cfg.auto_cancel_seconds)))

        open_trades = list(ib.openTrades() or [])
        for t in open_trades:
            try:
                ib.cancelOrder(t.order)
                report["actions"]["orders_cancelled"] += 1
            except Exception as e:
                report["actions"]["details"].append({"orderId": getattr(t.order, "orderId", None), "cancel_error": str(e)})

        # Final sync
        ib.sleep(1.0)
        report["post"] = {
            "openOrders_count": len(list(ib.openOrders() or [])),
        }
        return report
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHAD Paper Shadow Runner (safe-by-default).")
    p.add_argument("--preview", action="store_true", help="Force preview-only (no orders).")
    p.add_argument("--execute", action="store_true", help="Execute paper orders (requires explicit arming).")
    p.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DEFAULT),
        help="Path to runtime paper shadow config JSON (safe-by-default).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(Path(args.config).expanduser().resolve())

    # Always build preview payload for auditability.
    reasons: List[str] = []

    # Preview forced?
    if bool(args.preview) or not bool(args.execute):
        reasons.append("--preview forced" if args.preview else "default preview (no --execute)")
        payload = _build_intents_preview(cfg)
        payload["reasons"] = reasons
        out = _write_artifact(cfg, prefix="PAPER_SHADOW_PREVIEW", payload=payload)
        print("[paper_shadow_runner] mode=preview")
        print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={cfg.armed}")
        for r in payload.get("reasons", []):
            print(f"[paper_shadow_runner] reason: {r}")
        print(f"[paper_shadow_runner] wrote: {out}")
        return 0

    # --execute requested: enforce strong gate + live-gate permission.
    ok, gate_reasons = _should_execute_paper(cfg)
    reasons.extend(gate_reasons)

    lg_ok, lg_reason = _live_gate_allows_paper()
    reasons.append(lg_reason)
    if not lg_ok:
        payload = _build_intents_preview(cfg)
        payload["reasons"] = reasons
        payload["mode"] = "preview"
        out = _write_artifact(cfg, prefix="PAPER_SHADOW_PREVIEW", payload=payload)
        print("[paper_shadow_runner] mode=preview")
        print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={cfg.armed}")
        for r in reasons:
            print(f"[paper_shadow_runner] reason: {r}")
        print(f"[paper_shadow_runner] wrote: {out}")
        return 0

    if not ok:
        payload = _build_intents_preview(cfg)
        payload["reasons"] = reasons
        payload["mode"] = "preview"
        out = _write_artifact(cfg, prefix="PAPER_SHADOW_PREVIEW", payload=payload)
        print("[paper_shadow_runner] mode=preview")
        print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={cfg.armed}")
        for r in reasons:
            print(f"[paper_shadow_runner] reason: {r}")
        print(f"[paper_shadow_runner] wrote: {out}")
        return 0

    # Build intents (preview builder) then execute.
    payload = _build_intents_preview(cfg)
    intents = list(payload.get("paper_intents_preview", []) or [])
    exec_report = _execute_paper_orders(cfg, intents)
    exec_report["reasons"] = reasons
    out = _write_artifact(cfg, prefix="PAPER_SHADOW_EXECUTE", payload=exec_report)

    print("[paper_shadow_runner] mode=execute")
    print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={cfg.armed} env_armed={is_armed()}")
    for r in reasons:
        print(f"[paper_shadow_runner] reason: {r}")
    print(f"[paper_shadow_runner] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
