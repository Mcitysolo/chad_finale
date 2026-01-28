"""
CHAD Phase 10 — Model Doctor (Deterministic, Advisory-Only)

Purpose
-------
Generate a structured "model doctor" report that suggests ML/feature/rules improvements
based on CHAD's existing reporting artifacts.

This is NOT trading logic and does NOT mutate configs or runtime state.
It only reads reports and writes advisory outputs.

Inputs (read-only)
------------------
- reports/ops/REPORTS_INDEX_LATEST.json
- latest WEEKLY_INVESTOR_REPORT_*.json (preferred)
- fallback: latest DAILY_EXEC_REPORT_*.json + DAILY_PERF_REPORT_*.json

Outputs
-------
- reports/model_doctor/MODEL_DOCTOR_<ts>.json
- reports/model_doctor/MODEL_DOCTOR_<ts>.md

Contract
--------
- No broker calls
- No order execution
- No config writes
- No runtime writes (except the report files)
- Suggestions only

Run
---
python -m chad.ai.model_doctor
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path("/home/ubuntu/CHAD FINALE")
OPS_DIR = REPO_ROOT / "reports" / "ops"
OUT_DIR = REPO_ROOT / "reports" / "model_doctor"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def newest(glob_pat: str, base: Path) -> Optional[Path]:
    files = sorted(base.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_reports_index() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    idx = OPS_DIR / "REPORTS_INDEX_LATEST.json"
    if not idx.exists():
        return None, f"missing:{idx}"
    try:
        return read_json(idx), None
    except Exception as exc:
        return None, f"read_failed:{idx}:{type(exc).__name__}:{exc}"


@dataclass(frozen=True)
class InputsUsed:
    weekly_json: Optional[str]
    daily_exec_json: Optional[str]
    daily_perf_json: Optional[str]


def pick_inputs(index: Dict[str, Any]) -> InputsUsed:
    latest = index.get("latest") if isinstance(index, dict) else None
    if not isinstance(latest, dict):
        return InputsUsed(None, None, None)

    weekly_json = latest.get("weekly_investor_json")
    daily_exec_json = latest.get("daily_exec_json")
    daily_perf_json = latest.get("daily_perf_json")

    # If index doesn't have weekly yet, try to find newest directly
    if not isinstance(weekly_json, str) or not weekly_json.strip():
        w = newest("WEEKLY_INVESTOR_REPORT_*.json", OPS_DIR)
        weekly_json = str(w) if w else None

    if not isinstance(daily_exec_json, str) or not daily_exec_json.strip():
        d = newest("DAILY_EXEC_REPORT_*.json", OPS_DIR)
        daily_exec_json = str(d) if d else None

    if not isinstance(daily_perf_json, str) or not daily_perf_json.strip():
        p = newest("DAILY_PERF_REPORT_*.json", OPS_DIR)
        daily_perf_json = str(p) if p else None

    return InputsUsed(
        weekly_json=str(weekly_json) if weekly_json else None,
        daily_exec_json=str(daily_exec_json) if daily_exec_json else None,
        daily_perf_json=str(daily_perf_json) if daily_perf_json else None,
    )


def clamp_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except Exception:
        return None


def clamp_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def clamp_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def analyze_from_weekly(weekly: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the most useful signals from weekly report for advisory suggestions.
    """
    overall = weekly.get("overall") if isinstance(weekly.get("overall"), dict) else {}
    by_strategy = weekly.get("by_strategy") if isinstance(weekly.get("by_strategy"), dict) else {}
    top_symbols = weekly.get("top_symbols") if isinstance(weekly.get("top_symbols"), list) else []
    bottom_symbols = weekly.get("bottom_symbols") if isinstance(weekly.get("bottom_symbols"), list) else []

    total_trades = clamp_int(overall.get("trades")) or 0
    win_rate = clamp_float(overall.get("win_rate")) or 0.0
    total_pnl = clamp_float(overall.get("pnl_total")) or 0.0

    # Identify dominant losing symbol by absolute pnl and trade count if present
    worst_symbol = None
    if bottom_symbols:
        # bottom_symbols already sorted by pnl asc in generator
        worst_symbol = bottom_symbols[0]

    # Identify "concentration" heuristic
    concentration_note = None
    if worst_symbol and isinstance(worst_symbol, dict):
        wsym = clamp_str(worst_symbol.get("symbol"), "UNKNOWN")
        wpnl = clamp_float(worst_symbol.get("pnl_total"))
        wtrades = clamp_int(worst_symbol.get("trades"))
        if wtrades and total_trades and wtrades / max(1, total_trades) >= 0.5:
            concentration_note = {
                "type": "symbol_concentration",
                "symbol": wsym,
                "share_of_trades": wtrades / max(1, total_trades),
                "pnl_total": wpnl,
                "trades": wtrades,
            }

    return {
        "weekly_overall": {"trades": total_trades, "win_rate": win_rate, "pnl_total": total_pnl},
        "weekly_by_strategy": by_strategy,
        "weekly_top_symbols": top_symbols[:10],
        "weekly_bottom_symbols": bottom_symbols[:10],
        "concentration_note": concentration_note,
    }


def build_suggestions(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic suggestion generator (no ML, no GPT). Pure rules based on observed stats.
    """
    suggestions: Dict[str, Any] = {
        "new_feature_ideas": [],
        "symbol_blacklist_suggestions": [],
        "hyperparam_tuning_ideas": [],
        "policy_suggestions": [],
        "notes": [],
    }

    overall = analysis.get("weekly_overall") or {}
    pnl_total = clamp_float(overall.get("pnl_total"))
    win_rate = clamp_float(overall.get("win_rate"))

    if pnl_total is not None and pnl_total < 0:
        suggestions["notes"].append(
            f"Weekly PnL is negative ({pnl_total}). Focus on reducing loss magnitude and improving selectivity."
        )

    if win_rate is not None and win_rate < 0.45:
        suggestions["notes"].append(
            f"Win rate is low ({win_rate}). Consider tighter veto filters and avoid over-trading one symbol."
        )

    # Concentration
    cn = analysis.get("concentration_note")
    if isinstance(cn, dict):
        sym = clamp_str(cn.get("symbol"), "UNKNOWN")
        share = cn.get("share_of_trades")
        suggestions["notes"].append(
            f"High concentration detected: {sym} is a large share of trades. Consider per-symbol cooldown/caps."
        )
        # Not a hard blacklist; suggestion only
        suggestions["symbol_blacklist_suggestions"].append(sym)
        suggestions["policy_suggestions"].append(
            {"type": "symbol_trade_cap", "symbol": sym, "suggestion": "cap daily trades per symbol; add cooldown after N losses"}
        )

    # Feature ideas: generic, safe, useful
    suggestions["new_feature_ideas"].extend(
        [
            "realized_vol_change_5m_vs_60m",
            "overnight_gap_size",
            "distance_to_vwap",
            "distance_to_52w_high",
            "rolling_win_rate_by_symbol_20trades",
            "time_of_day_bucket",
        ]
    )

    # Hyperparam tuning ideas (generic; advisory)
    suggestions["hyperparam_tuning_ideas"].extend(
        [
            "Consider reducing max_depth if overfitting suspected (e.g., 8 -> 6).",
            "Consider increasing min_child_weight to reduce noisy splits.",
            "Re-evaluate class imbalance handling (scale_pos_weight) if win/loss labels skewed.",
        ]
    )

    return suggestions


def render_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# CHAD Model Doctor Report (Phase 10)")
    lines.append(f"- Generated: `{payload.get('generated_utc')}`")
    lines.append(f"- Host: `{payload.get('host')}`")
    lines.append("")
    lines.append("## Snapshot")
    snap = payload.get("snapshot") or {}
    if isinstance(snap, dict):
        o = snap.get("weekly_overall") or {}
        lines.append(f"- Weekly trades: `{o.get('trades')}`")
        lines.append(f"- Weekly win rate: `{o.get('win_rate')}`")
        lines.append(f"- Weekly total PnL: `{o.get('pnl_total')}`")
    lines.append("")

    lines.append("## Suggestions (advisory only)")
    sug = payload.get("suggestions") or {}
    if isinstance(sug, dict):
        lines.append("### Feature ideas")
        for x in sug.get("new_feature_ideas", []) or []:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("### Symbol watchlist / blacklist suggestions (review required)")
        for x in sug.get("symbol_blacklist_suggestions", []) or []:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("### Hyperparameter tuning ideas (review required)")
        for x in sug.get("hyperparam_tuning_ideas", []) or []:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("### Policy suggestions (review required)")
        for x in sug.get("policy_suggestions", []) or []:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("### Notes")
        for x in sug.get("notes", []) or []:
            lines.append(f"- {x}")
    lines.append("")

    lines.append("## Inputs used")
    inp = payload.get("inputs_used") or {}
    if isinstance(inp, dict):
        for k, v in inp.items():
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    return "\n".join(lines)


def run() -> Tuple[Path, Path]:
    safe_mkdir(OUT_DIR)

    idx, idx_err = load_reports_index()
    inputs_used = {"weekly_json": None, "daily_exec_json": None, "daily_perf_json": None, "reports_index_error": idx_err}

    weekly_obj: Optional[Dict[str, Any]] = None

    if idx:
        pick = pick_inputs(idx)
        inputs_used["weekly_json"] = pick.weekly_json
        inputs_used["daily_exec_json"] = pick.daily_exec_json
        inputs_used["daily_perf_json"] = pick.daily_perf_json

        if pick.weekly_json and Path(pick.weekly_json).exists():
            weekly_obj = read_json(Path(pick.weekly_json))

    # If no weekly report, do minimal output
    if weekly_obj is None:
        snapshot = {"note": "No weekly investor report found; generate one first."}
        suggestions = {
            "new_feature_ideas": [],
            "symbol_blacklist_suggestions": [],
            "hyperparam_tuning_ideas": [],
            "policy_suggestions": [],
            "notes": ["Weekly investor report missing; cannot compute weekly patterns."],
        }
    else:
        snapshot = analyze_from_weekly(weekly_obj)
        suggestions = build_suggestions(snapshot)

    payload: Dict[str, Any] = {
        "generated_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "phase": "10",
        "snapshot": snapshot,
        "suggestions": suggestions,
        "inputs_used": inputs_used,
    }

    ts = utc_now_compact()
    jpath = OUT_DIR / f"MODEL_DOCTOR_{ts}.json"
    mpath = OUT_DIR / f"MODEL_DOCTOR_{ts}.md"

    atomic_write_json(jpath, payload)
    atomic_write_md(mpath, render_md(payload))

    print(str(jpath))
    print(str(mpath))
    return jpath, mpath


if __name__ == "__main__":
    run()

