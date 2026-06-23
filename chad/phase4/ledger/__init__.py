"""Phase-4 Trade Ledger (stub)."""
from typing import Dict, Any, List
_LOG: List[Dict[str, Any]] = []

def record(entry: Dict[str, Any]) -> None:
    _LOG.append(entry)

def snapshot() -> List[Dict[str, Any]]:
    return list(_LOG)
