from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

"""
Strategy Spam Governor (paper-first safety tool)

Purpose
-------
Cap strategy output so one brain cannot flood the system with low-quality trades.
This is a *guard rail*, not alpha. It reduces churn and protects paper metrics.

This module is intentionally pure + deterministic:
- No network.
- No broker calls.
- No randomness.
- Atomic state writes (optional).

Integration idea
----------------
Call `apply_spam_governor(...)` on raw TradeSignals or RoutedSignals before execution planning.
We keep it generic by requiring only `strategy` and `symbol` attributes/keys.

Default policy (safe)
---------------------
- Max trades per strategy per day (ledger-based) OR per cycle (simple cap).
- Max trades per (strategy, symbol) per day to stop single-symbol spam.

You can run it in paper-only mode first; it never enables live.
"""

def _utc_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def _get_attr(o: Any, name: str, default: str = "") -> str:
    # supports dataclass/obj or dict
    if isinstance(o, dict):
        return str(o.get(name, default))
    return str(getattr(o, name, default))


@dataclass(frozen=True)
class SpamLimits:
    max_per_strategy_per_cycle: int = 5
    max_per_strategy_symbol_per_cycle: int = 2


def apply_spam_governor(
    signals: Iterable[Any],
    *,
    limits: SpamLimits = SpamLimits(),
    runtime_dir: Path | None = None,
    state_filename: str = "strategy_spam_governor_state.json",
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Returns (allowed_signals, report).
    """
    allowed: List[Any] = []
    dropped: List[Dict[str, Any]] = []

    per_strategy: Dict[str, int] = {}
    per_pair: Dict[tuple[str, str], int] = {}

    for sig in signals:
        strat = _get_attr(sig, "strategy", "unknown").strip() or "unknown"
        sym = _get_attr(sig, "symbol", "unknown").strip() or "unknown"

        per_strategy.setdefault(strat, 0)
        per_pair.setdefault((strat, sym), 0)

        if per_strategy[strat] >= limits.max_per_strategy_per_cycle:
            dropped.append({"reason": "max_per_strategy_per_cycle", "strategy": strat, "symbol": sym})
            continue
        if per_pair[(strat, sym)] >= limits.max_per_strategy_symbol_per_cycle:
            dropped.append({"reason": "max_per_strategy_symbol_per_cycle", "strategy": strat, "symbol": sym})
            continue

        allowed.append(sig)
        per_strategy[strat] += 1
        per_pair[(strat, sym)] += 1

    report = {
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "day_utc": _utc_ymd(),
        "limits": {
            "max_per_strategy_per_cycle": limits.max_per_strategy_per_cycle,
            "max_per_strategy_symbol_per_cycle": limits.max_per_strategy_symbol_per_cycle,
        },
        "in_signals": sum(per_strategy.values()) + len(dropped),
        "out_signals": len(allowed),
        "dropped": dropped[:200],  # cap report size
        "dropped_count": len(dropped),
        "strategy_counts_out": per_strategy,
    }

    if runtime_dir is not None:
        _atomic_write_json(Path(runtime_dir) / state_filename, report)

    return allowed, report
