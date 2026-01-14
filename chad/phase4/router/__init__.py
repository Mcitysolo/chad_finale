"""Phase-4 Execution Router (stub).
Routes validated TradeIntent objects to the execution engine with DRY_RUN default.
"""
__all__ = ["route"]
from typing import Dict, Any

def route(intent: Dict[str, Any], *, dry_run: bool = True) -> Dict[str, Any]:
    # TODO: replace with real wiring in Phase-4 PR
    return {"routed": True, "dry_run": dry_run, "symbol": intent.get("symbol")}
