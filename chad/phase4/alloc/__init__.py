"""Phase-4 Capital Allocation (stub)."""
from typing import Dict, Any

def size(intent: Dict[str, Any], equity: float = 10000.0) -> Dict[str, Any]:
    # Tiny sizing to keep DRY_RUN safe in early PRs
    return {"qty": max(1, int(equity * 0.0005 // 1)), "equity": equity}
