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
    # W4A-6 (LC5, drawdown_state.v2): short-horizon drawdown budgets vs the
    # trailing 5/20 DATED equity rows, epoch-scoped (rows before the active
    # epoch start are excluded — the H3 phantom-HWM guard, PLAN_W4A P7). dd_*
    # is None when the horizon has zero epoch-eligible rows; sample_count_* is
    # the number of rows the window actually used. LC5 enforcement (W4A-7)
    # ignores any window whose sample_count < the window length — these fields
    # REPORT always but only ENFORCE when deep enough. The 60d HWM fields above
    # are unchanged (report-only as today).
    dd_5d_pct: Optional[float] = None
    dd_20d_pct: Optional[float] = None
    sample_count_5d: int = 0
    sample_count_20d: int = 0


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


def _row_date_only(row: Dict[str, Any]) -> Optional[str]:
    """YYYY-MM-DD from a row's date/ts field (date portion only)."""
    raw = _row_date(row)
    if not raw:
        return None
    return raw[:10]


def _epoch_start_date(epoch_state_path: Optional[Path]) -> Optional[str]:
    """YYYY-MM-DD of the active epoch start, or None. Rows dated before this
    are excluded from the 5d/20d windows (H3 phantom-HWM guard, P7)."""
    obj = _safe_load_json(
        epoch_state_path or (RUNTIME_DIR / "epoch_state.json")
    )
    if not isinstance(obj, dict):
        return None
    v = obj.get("epoch_started_at_utc")
    if isinstance(v, str) and len(v) >= 10:
        return v[:10]
    return None


def _windowed_drawdown(
    history: List[Dict[str, Any]],
    current_eq: float,
    window: int,
    epoch_start_date: Optional[str],
) -> Tuple[Optional[float], int]:
    """Drawdown vs the max of the trailing *window* DATED, epoch-eligible
    equity rows. Returns (dd_pct or None, sample_count_used). dd is measured
    from current_eq against that window peak; can be positive (no drawdown).
    None when the window has zero eligible rows or no current equity."""
    dated: List[float] = []
    for row in history:
        d = _row_date_only(row)
        if d is None:
            continue
        if epoch_start_date is not None and d < epoch_start_date:
            continue
        eq = _row_equity(row)
        if eq is not None and eq > 0:
            dated.append(eq)
    trailing = dated[-window:] if window > 0 else list(dated)
    n = len(trailing)
    if n == 0 or current_eq <= 0:
        return None, n
    peak = max(trailing)
    if peak <= 0:
        return None, n
    return (current_eq - peak) / peak * 100.0, n


def compute_drawdown(
    *,
    equity_history_path: Optional[Path] = None,
    portfolio_snapshot_path: Optional[Path] = None,
    pnl_state_path: Optional[Path] = None,
    halt_threshold_pct: Optional[float] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    epoch_state_path: Optional[Path] = None,
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

    # W4A-6: short-horizon budgets (epoch-scoped). Computed from the FULL
    # dated history (not the 60d window slice) so the 5/20d peaks are honest.
    epoch_start_date = _epoch_start_date(epoch_state_path)
    dd_5d, n_5d = _windowed_drawdown(history, current_eq, 5, epoch_start_date)
    dd_20d, n_20d = _windowed_drawdown(history, current_eq, 20, epoch_start_date)
    notes.append(f"dd_5d_samples={n_5d} dd_20d_samples={n_20d}")
    if epoch_start_date:
        notes.append(f"epoch_start_date={epoch_start_date}")

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
            dd_5d_pct=dd_5d,
            dd_20d_pct=dd_20d,
            sample_count_5d=n_5d,
            sample_count_20d=n_20d,
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
        dd_5d_pct=dd_5d,
        dd_20d_pct=dd_20d,
        sample_count_5d=n_5d,
        sample_count_20d=n_20d,
    )


def report_to_state_dict(report: DrawdownReport, ts_utc: str, ttl_seconds: int = 300) -> Dict[str, Any]:
    """Render a DrawdownReport into the drawdown_state.v2 schema.

    W4A-6 (D6): bumped v1 → v2 for the additive dd_5d/dd_20d budget fields.
    Transition-safe: the sentinel EXS7 contract accepts v1|v2 and the
    exterminator feeds row is unchanged, so a still-v1 live file (before the
    drawdown-publisher restart that activates this) never trips a schema break
    (the position_guard_drift v1|v2|v3 precedent)."""
    return {
        "schema_version": "drawdown_state.v2",
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
        # W4A-6: additive short-horizon budgets (null when a horizon has no
        # epoch-eligible rows). Rounded like the 60d field; sample counts let
        # LC5 enforcement gate on depth.
        "dd_5d_pct": (
            round(float(report.dd_5d_pct), 4)
            if report.dd_5d_pct is not None else None
        ),
        "dd_20d_pct": (
            round(float(report.dd_20d_pct), 4)
            if report.dd_20d_pct is not None else None
        ),
        "sample_count_5d": int(report.sample_count_5d),
        "sample_count_20d": int(report.sample_count_20d),
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
