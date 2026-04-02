#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path("/home/ubuntu/chad_finale").resolve()


def runtime_dir() -> Path:
    return (repo_root() / "runtime").resolve()


def read_json_dict(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass(frozen=True)
class MacroInterpretation:
    generated_at_iso: str
    symbol: str
    macro_label: str
    event_risk_severity: str
    governor_mode: str
    ready_for_live: bool
    posture: str
    summary: str
    details: Dict[str, Any]


def _derive_posture(macro_label: str, event_severity: str, governor_mode: str) -> str:
    ml = macro_label.lower().strip()
    es = event_severity.lower().strip()
    gm = governor_mode.upper().strip()

    if gm == "DENY_ALL":
        return "hard_defensive"
    if gm == "PAUSE":
        return "paused"
    if es in {"high", "critical"}:
        return "defensive"
    if ml in {"risk_off", "tight", "defensive", "bearish"}:
        return "defensive"
    if gm == "TIGHTEN":
        return "cautious"
    if ml in {"risk_on", "constructive", "bullish"}:
        return "constructive"
    return "neutral"


def _build_summary(symbol: str, posture: str, macro_label: str, event_severity: str, governor_mode: str, ready_for_live: bool) -> str:
    return (
        f"Macro posture for {symbol}: {posture}. "
        f"macro_label={macro_label or 'unknown'}, "
        f"event_risk={event_severity or 'unknown'}, "
        f"governor_mode={governor_mode or 'unknown'}, "
        f"ready_for_live={ready_for_live}."
    )


def run_macro_interpreter(
    symbol: str,
    question: str = "",
    user_question: str = "",
    **_: Any,
) -> Dict[str, Any]:
    rt = runtime_dir()

    macro_state = read_json_dict(rt / "macro_state.json")
    event_risk = read_json_dict(rt / "event_risk.json")
    governor_state = read_json_dict(rt / "governor_state.json")
    live_readiness = read_json_dict(rt / "live_readiness.json")

    macro_label = str(macro_state.get("risk_label") or macro_state.get("macro_label") or "unknown").strip()
    event_severity = str(event_risk.get("severity") or "unknown").strip()
    governor_mode = str(governor_state.get("governor_mode") or "unknown").strip()
    ready_for_live = bool(live_readiness.get("ready_for_live", False))

    posture = _derive_posture(macro_label, event_severity, governor_mode)
    summary = _build_summary(symbol, posture, macro_label, event_severity, governor_mode, ready_for_live)

    result = MacroInterpretation(
        generated_at_iso=utc_now_iso(),
        symbol=str(symbol).strip().upper(),
        macro_label=macro_label,
        event_risk_severity=event_severity,
        governor_mode=governor_mode,
        ready_for_live=ready_for_live,
        posture=posture,
        summary=summary,
        details={
            "macro_state": macro_state,
            "event_risk": event_risk,
            "governor_state": governor_state,
            "live_readiness": live_readiness,
            "question": question or user_question,
        },
    )
    return asdict(result)


def run_macro_job(symbol: str, question: str = "", user_question: str = "", **kwargs: Any) -> Dict[str, Any]:
    return run_macro_interpreter(symbol=symbol, question=question, user_question=user_question, **kwargs)


def interpret_macro(symbol: str, question: str = "", user_question: str = "", **kwargs: Any) -> Dict[str, Any]:
    return run_macro_interpreter(symbol=symbol, question=question, user_question=user_question, **kwargs)


if __name__ == "__main__":
    print(json.dumps(run_macro_interpreter(symbol="AAPL"), indent=2, sort_keys=True))
