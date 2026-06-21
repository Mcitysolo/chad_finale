"""
chad/risk/drawdown_guard.py

GAP-016A: report-only global drawdown state.

Computes a 60-day rolling high-water mark (HWM) over the local equity
history and compares it to the most recently observed total equity. The
result is consumed by ``ops/drawdown_publisher.py`` which writes
``runtime/drawdown_state.json`` (schema ``drawdown_state.v1``).

The guard is **report-only** in this batch:

* It does NOT write to ``runtime/stop_bus_state.json``.
* It does NOT suppress signals.
* It does NOT halt trading.
* It does NOT mutate ``live_loop.py``.
* It records a ``halt`` boolean for observability only.
* ``enforcement_active`` is always False in this batch.

Threshold
---------
``halt_threshold_pct`` defaults to ``-15.0`` (i.e., halt would activate when
drawdown is at least 15% below the rolling HWM). The default may be overridden
via the env var ``CHAD_DRAWDOWN_HALT_PCT`` (negative number expected; absolute
values are coerced to negative for safety).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"

EQUITY_HISTORY_PATH = RUNTIME_DIR / "equity_history.ndjson"
PORTFOLIO_SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
PNL_STATE_PATH = RUNTIME_DIR / "pnl_state.json"

DEFAULT_HALT_PCT = -15.0
DEFAULT_LOOKBACK_DAYS = 60
HALT_PCT_ENV = "CHAD_DRAWDOWN_HALT_PCT"


@dataclass(frozen=True)
class DrawdownReport:
    status: str
    # CAD-denominated, report-only. ``_current_equity`` sums the CAD snapshot
    # legs and ``_row_equity`` prefers ``total_equity_cad``; these figures have
    # always carried CAD, so the field names now say so. Renamed from the
    # historical ``current_equity_usd`` / ``hwm_usd`` (label fix only — values,
    # drawdown_pct math, and enforcement_active=False are all unchanged).
    current_equity_cad: float
    hwm_cad: float
    drawdown_pct: float
    halt_threshold_pct: float
    halt: bool
    enforcement_active: bool
    sample_count: int
    lookback_days: int
    notes: List[str]


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_equity_history(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or EQUITY_HISTORY_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _row_equity(row: Any) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    # Continuity-safe key priority. ``total_equity_cad`` is the honest CAD series
    # written by equity_history.v2; the legacy ``total_equity_usd`` key carried
    # the SAME CAD figure in v1 rows, so reading it as a fallback keeps the HWM
    # series continuous across the relabel. ``total_equity_cad`` MUST be tried
    # first: in v2 rows ``total_equity_usd`` now holds the true-USD figure (or
    # null), which must never feed the currency-agnostic drawdown math.
    for k in ("total_equity_cad", "total_equity_usd", "equity_usd", "total_equity", "equity"):
        v = row.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f) and f >= 0:
            return f
    return None


def _row_date(row: Dict[str, Any]) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for k in ("date_utc", "ts_utc", "date"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _current_equity(
    portfolio_snapshot_path: Optional[Path] = None,
    pnl_state_path: Optional[Path] = None,
    history_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, str]:
    snap = _safe_load_json(portfolio_snapshot_path or PORTFOLIO_SNAPSHOT_PATH)
    if isinstance(snap, dict):
        total = 0.0
        any_nonzero = False
        for k in ("ibkr_equity", "coinbase_equity", "kraken_equity"):
            try:
                v = float(snap.get(k, 0.0) or 0.0)
            except Exception:
                continue
            if v > 0:
                any_nonzero = True
            total += v
        if any_nonzero and total > 0:
            return total, "portfolio_snapshot.json"

    pnl = _safe_load_json(pnl_state_path or PNL_STATE_PATH)
    if isinstance(pnl, dict):
        try:
            v = float(pnl.get("account_equity", 0.0) or 0.0)
            if v > 0:
                return v, "pnl_state.json"
        except Exception:
            pass

    if history_rows:
        last = history_rows[-1]
        eq = _row_equity(last)
        if eq is not None and eq > 0:
            return eq, "equity_history.ndjson:last"

    return 0.0, "none"


def _resolve_halt_threshold(explicit: Optional[float]) -> Tuple[float, str]:
    if explicit is not None:
        try:
            v = float(explicit)
            if math.isfinite(v):
                return -abs(v) if v == 0 else v, "explicit_arg"
        except Exception:
            pass
    raw = os.environ.get(HALT_PCT_ENV)
    if raw is not None and raw.strip() != "":
        try:
            v = float(raw)
            if math.isfinite(v):
                # Coerce to negative so that operators passing 15 still mean "-15%".
                if v > 0:
                    v = -v
                return v, f"env:{HALT_PCT_ENV}"
        except Exception:
            pass
    return DEFAULT_HALT_PCT, "default"


def compute_drawdown(
    *,
    equity_history_path: Optional[Path] = None,
    portfolio_snapshot_path: Optional[Path] = None,
    pnl_state_path: Optional[Path] = None,
    halt_threshold_pct: Optional[float] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> DrawdownReport:
    """Compute report-only drawdown vs rolling HWM. Never raises."""
    notes: List[str] = []
    threshold, threshold_source = _resolve_halt_threshold(halt_threshold_pct)
    notes.append(f"halt_threshold_source={threshold_source}")
    notes.append(f"lookback_days={lookback_days}")

    history = _read_equity_history(equity_history_path)
    notes.append(f"equity_history_rows={len(history)}")

    if lookback_days > 0 and len(history) > lookback_days:
        history_window = history[-lookback_days:]
    else:
        history_window = list(history)

    current_eq, current_source = _current_equity(
        portfolio_snapshot_path=portfolio_snapshot_path,
        pnl_state_path=pnl_state_path,
        history_rows=history,
    )
    notes.append(f"current_equity_source={current_source}")

    equities: List[float] = []
    for row in history_window:
        eq = _row_equity(row)
        if eq is not None:
            equities.append(eq)
    if current_eq > 0:
        equities.append(current_eq)

    if not equities:
        return DrawdownReport(
            status="insufficient_data",
            current_equity_cad=current_eq,
            hwm_cad=0.0,
            drawdown_pct=0.0,
            halt_threshold_pct=threshold,
            halt=False,
            enforcement_active=False,
            sample_count=0,
            lookback_days=int(lookback_days),
            notes=notes + ["no_usable_equity_samples"],
        )

    hwm = max(equities)
    if hwm <= 0:
        dd_pct = 0.0
    else:
        dd_pct = (current_eq - hwm) / hwm * 100.0

    halt_flag = bool(dd_pct <= threshold) if current_eq > 0 else False

    status = "ok" if current_eq > 0 else "insufficient_data"
    if current_eq <= 0:
        notes.append("no_current_equity_observed")

    return DrawdownReport(
        status=status,
        current_equity_cad=current_eq,
        hwm_cad=hwm,
        drawdown_pct=dd_pct,
        halt_threshold_pct=threshold,
        halt=halt_flag,
        enforcement_active=False,
        sample_count=len(equities),
        lookback_days=int(lookback_days),
        notes=notes,
    )


def report_to_state_dict(report: DrawdownReport, ts_utc: str, ttl_seconds: int = 300) -> Dict[str, Any]:
    """Render a DrawdownReport into the drawdown_state.v1 schema."""
    return {
        "schema_version": "drawdown_state.v1",
        "ts_utc": ts_utc,
        "ttl_seconds": int(ttl_seconds),
        "status": report.status,
        "current_equity_cad": round(float(report.current_equity_cad), 2),
        "hwm_cad": round(float(report.hwm_cad), 2),
        "drawdown_pct": round(float(report.drawdown_pct), 4),
        "halt_threshold_pct": round(float(report.halt_threshold_pct), 4),
        "halt": bool(report.halt),
        "enforcement_active": False,
        "sample_count": int(report.sample_count),
        "lookback_days": int(report.lookback_days),
        "notes": list(report.notes),
    }


__all__ = [
    "compute_drawdown",
    "report_to_state_dict",
    "DrawdownReport",
    "DEFAULT_HALT_PCT",
    "DEFAULT_LOOKBACK_DAYS",
    "HALT_PCT_ENV",
]
