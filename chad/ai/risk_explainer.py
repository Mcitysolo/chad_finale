"""
CHAD Phase 10 — AI Risk Explainer (Deterministic, Advisory-Only)

Outputs (in reports/ops):
- AI_RISK_EXPLAIN_<ts>.json
- AI_RISK_EXPLAIN_<ts>.md

Reads (read-only):
- reports/ops/REPORTS_INDEX_LATEST.json
- latest Daily Ops report JSON referenced by the index

Contract:
- No broker calls
- No live-state mutation
- No secrets
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


def clamp_str(v: Any, default: str = "unknown") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def clamp_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


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


@dataclass(frozen=True)
class Finding:
    level: str  # info|warn|critical
    code: str
    message: str
    evidence: Dict[str, Any]


def _finding(level: str, code: str, message: str, **evidence: Any) -> Finding:
    lvl = level.strip().lower()
    if lvl not in {"info", "warn", "critical"}:
        lvl = "info"
    return Finding(level=lvl, code=code, message=message, evidence=dict(evidence))


def load_reports_index() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    idx = OPS_DIR / "REPORTS_INDEX_LATEST.json"
    if not idx.exists():
        return None, f"missing:{idx}"
    try:
        return read_json(idx), None
    except Exception as exc:
        return None, f"read_failed:{idx}:{type(exc).__name__}:{exc}"


def load_daily_ops_from_index(index: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    latest = index.get("latest")
    if not isinstance(latest, dict):
        return None, "reports_index_missing_latest"
    ops_path = latest.get("daily_ops_json")
    if not isinstance(ops_path, str) or not ops_path.strip():
        return None, "missing_daily_ops_json_path"
    p = Path(ops_path)
    if not p.exists():
        return None, f"missing:{p}"
    try:
        return read_json(p), None
    except Exception as exc:
        return None, f"read_failed:{p}:{type(exc).__name__}:{exc}"


def explain_scr(scr_state: str) -> Tuple[str, List[str]]:
    scr = scr_state.strip().upper()
    if scr == "WARMUP":
        return (
            "WARMUP means CHAD is still building confidence in current conditions (often paper-first).",
            [
                "This is normal after restarts or when confidence rules require more samples.",
                "Live trading should still be deny-by-default unless LiveGate and operator intent are green.",
            ],
        )
    if scr == "CAUTIOUS":
        return (
            "CAUTIOUS means CHAD sees risk/uncertainty and should behave conservatively per policy.",
            ["Check data freshness, reconciliation, and volatility regime flags."],
        )
    if scr == "CONFIDENT":
        return (
            "CONFIDENT means CHAD’s self-checks meet thresholds for higher confidence (still gated).",
            ["Live is still blocked unless all gates are green."],
        )
    if scr == "PAUSED":
        return (
            "PAUSED means CHAD intentionally stopped taking new risk (STOP/operator intent/stale data/etc.).",
            ["Check STOP, operator intent, and ops inputs/errors."],
        )
    return (
        f"{scr_state} is unknown to this explainer. Treat it as UNKNOWN and verify SCR source.",
        ["Check ops report scr_state and SCR component output."],
    )


def recommended_commands() -> List[str]:
    return [
        'cd "/home/ubuntu/CHAD FINALE"',
        'cat reports/ops/REPORTS_INDEX_LATEST.json',
        'ls -la reports/ops | tail -n 50',
        'cat runtime/ibkr_status.json',
        'sudo systemctl status chad-ibgateway.service --no-pager -l | sed -n "1,25p"',
        'sudo ss -lntp | egrep ":4001|:4002" || true',
        'sudo systemctl status chad-daily-exec-report.timer --no-pager -l | sed -n "1,25p"',
        'sudo systemctl status chad-weekly-investor-report.timer --no-pager -l | sed -n "1,25p"',
    ]


def analyze_ops(ops: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []

    scr = clamp_str(ops.get("scr_state"), "UNKNOWN")
    findings.append(_finding("info", "SCR_STATE", f"SCR State is {scr}.", scr_state=scr))

    ib = ops.get("ibkr_health")
    if isinstance(ib, dict):
        ok = clamp_bool(ib.get("ok"))
        lat = clamp_float(ib.get("latency_ms"))
        ttl = ib.get("ttl_seconds")
        if ok is True:
            findings.append(_finding("info", "IBKR_OK", "IBKR healthcheck is OK.", ok=ok, latency_ms=lat, ttl_seconds=ttl))
        elif ok is False:
            findings.append(_finding("critical", "IBKR_DOWN", "IBKR healthcheck is failing; CHAD should not go live.", ok=ok, latency_ms=lat, ttl_seconds=ttl))
        else:
            findings.append(_finding("warn", "IBKR_UNKNOWN", "IBKR healthcheck is unknown.", raw=ib))
    else:
        findings.append(_finding("warn", "IBKR_HEALTH_MISSING", "Ops report has no ibkr_health block.", raw_type=str(type(ib))))

    inputs = ops.get("inputs")
    if isinstance(inputs, dict):
        missing: List[str] = []
        for name, entry in inputs.items():
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            ok = clamp_bool(entry.get("ok"))
            if isinstance(path, str) and path and not Path(path).exists():
                missing.append(f"{name}:{path}")
            if ok is False:
                findings.append(_finding("warn", "INPUT_NOT_OK", f"Input '{name}' reported ok=false.", name=name, path=path, error=entry.get("error")))
        if missing:
            findings.append(_finding("critical", "INPUT_FILES_MISSING", "One or more required input files are missing on disk.", missing=missing))
        else:
            findings.append(_finding("info", "INPUT_FILES_PRESENT", "All referenced input files exist on disk.", count=len(inputs)))
    else:
        findings.append(_finding("warn", "INPUTS_BLOCK_MISSING", "Ops report has no 'inputs' block.", raw_type=str(type(inputs))))

    led = ops.get("ledger_summary")
    if isinstance(led, dict):
        findings.append(
            _finding(
                "info",
                "LEDGER_SUMMARY",
                "Ledger summary present.",
                total_records=led.get("total_records"),
                total_pnl=led.get("total_pnl"),
                alpha_records=led.get("alpha_records"),
                beta_records=led.get("beta_records"),
                untrusted_records=led.get("untrusted_records"),
            )
        )
    else:
        findings.append(_finding("warn", "LEDGER_SUMMARY_MISSING", "Ops report has no ledger_summary.", raw_type=str(type(led))))

    met = ops.get("metrics")
    if isinstance(met, dict) and clamp_bool(met.get("ok")) is True:
        findings.append(
            _finding(
                "info",
                "PAPER_METRICS",
                "Paper metrics present.",
                paper_trades_total=met.get("paper_trades_total"),
                paper_win_rate=met.get("paper_win_rate"),
                paper_total_pnl=met.get("paper_total_pnl"),
            )
        )
    else:
        findings.append(_finding("warn", "PAPER_METRICS_MISSING", "Paper metrics missing or not ok.", raw=met))

    return findings


def build_payload(index: Dict[str, Any], ops: Dict[str, Any], findings: List[Finding]) -> Dict[str, Any]:
    scr_state = clamp_str(ops.get("scr_state"), "UNKNOWN")
    scr_explain, scr_notes = explain_scr(scr_state)

    level_rank = {"info": 0, "warn": 1, "critical": 2}
    worst = "info"
    for f in findings:
        if level_rank.get(f.level, 0) > level_rank.get(worst, 0):
            worst = f.level

    return {
        "generated_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "phase": "10",
        "overall_level": worst,
        "scr_state": scr_state,
        "scr_explanation": scr_explain,
        "scr_notes": scr_notes,
        "findings": [{"level": f.level, "code": f.code, "message": f.message, "evidence": f.evidence} for f in findings],
        "artifacts_index": index.get("latest") if isinstance(index.get("latest"), dict) else {},
        "recommended_commands": recommended_commands(),
    }


def render_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# CHAD AI Risk Explainer (Phase 10)")
    lines.append(f"- Generated: `{payload.get('generated_utc')}`")
    lines.append(f"- Host: `{payload.get('host')}`")
    lines.append(f"- Overall: **{payload.get('overall_level')}**")
    lines.append("")
    lines.append("## SCR")
    lines.append(f"- State: **{payload.get('scr_state')}**")
    lines.append(payload.get("scr_explanation", ""))
    for n in payload.get("scr_notes", []) or []:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## Key Findings")
    for f in payload.get("findings", []) or []:
        lines.append(f"- **{str(f.get('level','info')).upper()}** `{f.get('code')}` — {f.get('message')}")
    lines.append("")
    lines.append("## Latest Report Artifacts")
    latest = payload.get("artifacts_index") or {}
    if isinstance(latest, dict):
        for k, v in latest.items():
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Next Commands (copy/paste)")
    for c in payload.get("recommended_commands", []) or []:
        lines.append(f"- `{c}`")
    lines.append("")
    return "\n".join(lines)


def run() -> Tuple[Path, Path]:
    safe_mkdir(OPS_DIR)

    idx, idx_err = load_reports_index()
    ts = utc_now_compact()
    jpath = OPS_DIR / f"AI_RISK_EXPLAIN_{ts}.json"
    mpath = OPS_DIR / f"AI_RISK_EXPLAIN_{ts}.md"

    if idx is None:
        payload = {
            "generated_utc": utc_now_iso(),
            "host": socket.gethostname(),
            "phase": "10",
            "overall_level": "critical",
            "error": idx_err,
            "recommended_commands": recommended_commands(),
        }
        atomic_write_json(jpath, payload)
        atomic_write_md(mpath, "# CHAD AI Risk Explainer (Phase 10)\n\nIndex missing.\n")
        print(str(jpath))
        print(str(mpath))
        return jpath, mpath

    ops, ops_err = load_daily_ops_from_index(idx)
    if ops is None:
        payload = {
            "generated_utc": utc_now_iso(),
            "host": socket.gethostname(),
            "phase": "10",
            "overall_level": "critical",
            "error": ops_err,
            "artifacts_index": idx.get("latest") if isinstance(idx.get("latest"), dict) else {},
            "recommended_commands": recommended_commands(),
        }
        atomic_write_json(jpath, payload)
        atomic_write_md(mpath, "# CHAD AI Risk Explainer (Phase 10)\n\nOps report missing.\n")
        print(str(jpath))
        print(str(mpath))
        return jpath, mpath

    findings = analyze_ops(ops)
    payload = build_payload(idx, ops, findings)

    atomic_write_json(jpath, payload)
    atomic_write_md(mpath, render_md(payload))

    print(str(jpath))
    print(str(mpath))
    return jpath, mpath


if __name__ == "__main__":
    run()
