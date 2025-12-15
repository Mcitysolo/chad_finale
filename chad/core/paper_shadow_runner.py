#!/usr/bin/env python3
"""
chad/core/paper_shadow_runner.py

Paper Shadow Lane Runner (Production-Safe, Disabled by Default)

Purpose
-------
This module scaffolds the *paper lane* portion of CHAD's shadow trading architecture
WITHOUT affecting CHAD's live lane and WITHOUT bypassing Phase-7 DRY_RUN hard-locks.

Key guarantees
--------------
1) Safe-by-default:
   - If runtime/paper_shadow.json is missing OR enabled=false → PREVIEW ONLY.
   - If enabled=true but CHAD_PAPER_SHADOW_ARM is not set to the exact arm phrase
     → PREVIEW ONLY.
2) No fake trades:
   - This module never writes TradeResult ledgers as "simulated" warmup.
   - When later enabled, it will only log real paper-order lifecycle outcomes.
3) No coupling to live:
   - Uses IB paper gateway (port 4002 typical) and never touches CHAD live execution.
   - CHAD core execution remains hard-locked DRY_RUN by execution_config.

Outputs
-------
Always writes a read-only artifact to:
  reports/shadow/PAPER_SHADOW_*.json

It does NOT write to:
  runtime/live_mode.json
  runtime/dynamic_caps.json
  data/trades/*

Arming model
------------
To actually place paper orders (NOT used in Phase-7 by default):

  1) runtime/paper_shadow.json: { "enabled": true, ... }
  2) Export the exact arm phrase:

     export CHAD_PAPER_SHADOW_ARM="I_UNDERSTAND_THIS_PLACES_PAPER_ORDERS"

If either condition is missing, it will only produce a preview artifact.

NOTE
----
In this Phase-7 build we keep this runner available, but we keep it "cold" unless
explicitly armed. This ensures production hygiene: reviewed, testable, and safe.

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ARM_ENV_NAME = "CHAD_PAPER_SHADOW_ARM"
ARM_PHRASE = "I_UNDERSTAND_THIS_PLACES_PAPER_ORDERS"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_dir() -> Path:
    return _repo_root() / "runtime"


def _reports_shadow_dir() -> Path:
    return _repo_root() / "reports" / "shadow"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class IbkrConn:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 9201


@dataclass(frozen=True)
class PaperShadowConfig:
    enabled: bool = False
    strategy_allowlist: Tuple[str, ...] = ("BETA",)
    max_orders_per_run: int = 5
    size_multiplier: float = 1.0
    min_notional_usd: float = 100.0
    auto_cancel_seconds: int = 15
    ibkr: IbkrConn = IbkrConn()

    @staticmethod
    def config_path() -> Path:
        return _runtime_dir() / "paper_shadow.json"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperShadowConfig":
        ibkr_d = d.get("ibkr", {}) if isinstance(d.get("ibkr", {}), dict) else {}
        ibkr = IbkrConn(
            host=str(ibkr_d.get("host", "127.0.0.1")),
            port=int(ibkr_d.get("port", 4002)),
            client_id=int(ibkr_d.get("client_id", 9201)),
        )
        allow = d.get("strategy_allowlist", ["BETA"])
        if not isinstance(allow, list):
            allow = ["BETA"]
        allow_norm = tuple(str(x).upper() for x in allow if str(x).strip())
        return cls(
            enabled=bool(d.get("enabled", False)),
            strategy_allowlist=allow_norm or ("BETA",),
            max_orders_per_run=max(0, int(d.get("max_orders_per_run", 5))),
            size_multiplier=float(d.get("size_multiplier", 1.0)),
            min_notional_usd=float(d.get("min_notional_usd", 100.0)),
            auto_cancel_seconds=max(1, int(d.get("auto_cancel_seconds", 15))),
            ibkr=ibkr,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strategy_allowlist": list(self.strategy_allowlist),
            "max_orders_per_run": self.max_orders_per_run,
            "size_multiplier": self.size_multiplier,
            "min_notional_usd": self.min_notional_usd,
            "auto_cancel_seconds": self.auto_cancel_seconds,
            "ibkr": {
                "host": self.ibkr.host,
                "port": self.ibkr.port,
                "client_id": self.ibkr.client_id,
            },
        }


def load_config_or_default() -> PaperShadowConfig:
    p = PaperShadowConfig.config_path()
    if not p.is_file():
        # Create a safe default config (disabled).
        cfg = PaperShadowConfig()
        _atomic_write_json(p, cfg.to_dict())
        return cfg

    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return PaperShadowConfig()
        return PaperShadowConfig.from_dict(d)
    except Exception:
        return PaperShadowConfig()


def is_armed() -> bool:
    return os.getenv(ARM_ENV_NAME, "").strip() == ARM_PHRASE


def should_place_paper_orders(cfg: PaperShadowConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not cfg.enabled:
        reasons.append("paper_shadow.enabled is false → preview only")
        return False, reasons
    if not is_armed():
        reasons.append(f"{ARM_ENV_NAME} not set to arm phrase → preview only")
        return False, reasons
    reasons.append("enabled=true and armed=true → paper order mode allowed")
    return True, reasons


def _build_preview_payload(cfg: PaperShadowConfig) -> Dict[str, Any]:
    """
    Preview-only payload. This intentionally does NOT hit IBKR.
    It uses CHAD's pipeline to compute intents and then emits a filtered,
    capped list of what would be submitted on the paper lane.
    """
    # Import heavy CHAD modules lazily so importing this file is always safe.
    from chad.execution.execution_pipeline import build_execution_plan, build_ibkr_intents_from_plan
    from chad.engine import StrategyEngine
    from chad.strategies import register_core_strategies
    from chad.utils.context_builder import ContextBuilder
    from chad.utils.pipeline import DecisionPipeline, PipelineConfig
    from chad.utils.signal_router import SignalRouter

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

    allow = set(s.upper() for s in cfg.strategy_allowlist)
    filtered = []
    for intent in intents:
        strat = str(getattr(intent, "strategy", "")).upper()
        if strat and strat in allow:
            filtered.append(intent)

    # Apply min_notional + max_orders
    trimmed = []
    for intent in filtered:
        notional = float(getattr(intent, "notional_estimate", 0.0) or 0.0)
        if notional < cfg.min_notional_usd:
            continue
        trimmed.append(intent)
        if len(trimmed) >= cfg.max_orders_per_run:
            break

    out_intents = []
    for intent in trimmed:
        qty = float(getattr(intent, "quantity", 0.0) or 0.0) * float(cfg.size_multiplier)
        out_intents.append(
            {
                "strategy": str(getattr(intent, "strategy", "")),
                "symbol": str(getattr(intent, "symbol", "")),
                "sec_type": str(getattr(intent, "sec_type", "")),
                "exchange": str(getattr(intent, "exchange", "")),
                "currency": str(getattr(intent, "currency", "")),
                "side": str(getattr(intent, "side", "")),
                "order_type": str(getattr(intent, "order_type", "")),
                "quantity": float(qty),
                "notional_estimate": float(getattr(intent, "notional_estimate", 0.0) or 0.0),
                "limit_price": getattr(intent, "limit_price", None),
            }
        )

    return {
        "generated_at_utc": now.isoformat(),
        "mode": "preview",
        "armed": is_armed(),
        "config": cfg.to_dict(),
        "counts": {
            "raw_signals": int(getattr(pr, "raw_signals_count", 0) or 0),
            "evaluated_signals": int(getattr(pr, "evaluated_signals_count", 0) or 0),
            "routed_signals": int(getattr(pr, "routed_signals_count", 0) or 0),
            "planned_orders": int(getattr(plan, "orders_count", 0) or 0),
            "ibkr_intents_total": len(intents),
            "ibkr_intents_filtered": len(out_intents),
        },
        "paper_intents_preview": out_intents,
        "safety": {
            "no_orders_placed": True,
            "no_trade_ledgers_written": True,
        },
    }


def write_preview(cfg: PaperShadowConfig) -> Path:
    payload = _build_preview_payload(cfg)
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    out = _reports_shadow_dir() / f"PAPER_SHADOW_PREVIEW_{ts}.json"
    _atomic_write_json(out, payload)
    return out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chad.core.paper_shadow_runner",
        description="Paper shadow lane runner (safe-by-default).",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Force preview mode (default behaviour unless enabled+armed).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config_or_default()

    allowed, reasons = should_place_paper_orders(cfg)
    # In Phase-7 we only ship preview mode; paper execution comes later when explicitly enabled.
    # Even if allowed==True, we STILL preview unless you later add a dedicated --execute flag.
    # This preserves the “no surprises” contract for this build.
    payload_path = write_preview(cfg)

    print("[paper_shadow_runner] mode=preview")
    print(f"[paper_shadow_runner] enabled={cfg.enabled} armed={is_armed()}")
    for r in reasons:
        print(f"[paper_shadow_runner] reason: {r}")
    print(f"[paper_shadow_runner] wrote: {payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
