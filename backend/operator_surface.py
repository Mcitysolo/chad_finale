"""
backend/operator_surface.py

CHAD Operator Surface (Phase 10 — Part 1)
Read-only, audit-first operator endpoints designed for:
- Zero ambiguity about system posture
- Zero side effects
- Fail-closed behavior
- High-quality observability

This module is intentionally standalone (router-only). It can be mounted into any
FastAPI app via:
    app.include_router(router)

Design goals:
- Never calls broker APIs directly
- Never mutates state
- Never throws unhandled exceptions to the operator
- Produces stable JSON shapes suitable for dashboards/Telegram summaries
- Avoids expensive work on hot path (caches git/systemd where safe)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["operator"])

# ----------------------------
# Constants / Paths (SSOT)
# ----------------------------

RUNTIME_DIR: Path = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
REPO_DIR: Path = Path(os.environ.get("CHAD_REPO_DIR", "/home/ubuntu/chad_finale"))

# Core runtime artifacts expected for operator clarity
RUNTIME_FILES: Dict[str, Path] = {
    "feed_state": Path("runtime/feed_state.json"),
    "positions_snapshot": Path("runtime/positions_snapshot.json"),
    "reconciliation_state": Path("runtime/reconciliation_state.json"),
    "dynamic_caps": Path("runtime/dynamic_caps.json"),
    "operator_intent": Path("runtime/operator_intent.json"),
    "portfolio_snapshot": Path("runtime/portfolio_snapshot.json"),
    # SSOT-alignment artifacts (absolute paths in your environment)
    "scr_state": Path("/home/ubuntu/CHAD FINALE/runtime/scr_state.json"),
    "tier_state": Path("/home/ubuntu/CHAD FINALE/runtime/tier_state.json"),
}

# Timers we care about (minimal list; avoids guessing)
OPERATOR_TIMERS: Tuple[str, ...] = (
    "chad-scr-sync.timer",
    "chad-tier-sync.timer",
)

# ----------------------------
# Utility: time / safe exec
# ----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: List[str], timeout_s: float = 2.0) -> Tuple[int, str, str]:
    """
    Safe subprocess runner.
    - Never raises
    - Returns (rc, stdout, stderr)
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(REPO_DIR) if REPO_DIR.exists() else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return int(p.returncode), p.stdout, p.stderr
    except Exception as exc:
        return 255, "", f"exec_error:{type(exc).__name__}:{exc}"


@lru_cache(maxsize=1)
def git_fingerprint() -> Dict[str, Any]:
    """
    Cached git fingerprint for operator visibility.
    Cache is per-process; good enough for service restarts.
    """
    out: Dict[str, Any] = {"repo_dir": str(REPO_DIR)}
    if not REPO_DIR.exists():
        out["ok"] = False
        out["error"] = "repo_dir_missing"
        return out

    rc, sha, err = _run(["git", "rev-parse", "HEAD"], timeout_s=1.5)
    out["head_sha"] = sha.strip() if rc == 0 else None
    out["git_rev_parse_rc"] = rc
    if rc != 0:
        out["ok"] = False
        out["error"] = err.strip()[:300]
        return out

    # Porcelain status (bounded)
    rc2, st, err2 = _run(["git", "status", "--porcelain"], timeout_s=2.0)
    out["dirty"] = bool(st.strip()) if rc2 == 0 else None
    out["status_rc"] = rc2
    if rc2 != 0:
        out["status_error"] = err2.strip()[:300]

    out["ok"] = True
    return out


def _safe_json_load(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    """
    Read and parse JSON dict from a file.
    Returns (data, error).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"read_error:{type(exc).__name__}"

    try:
        obj = json.loads(raw)
    except Exception as exc:
        return None, f"json_parse_error:{type(exc).__name__}"
    if not isinstance(obj, dict):
        return None, "invalid_shape:not_dict"
    return obj, None


def runtime_file_meta(path: Path) -> Dict[str, Any]:
    """
    Never raises. Returns metadata plus embedded ts_utc/ttl_seconds when present.
    """
    meta: Dict[str, Any] = {"path": str(path)}

    try:
        st = path.stat()
    except FileNotFoundError:
        meta["exists"] = False
        return meta
    except Exception as exc:
        meta["exists"] = False
        meta["error"] = f"stat_error:{type(exc).__name__}"
        return meta

    meta["exists"] = True
    meta["size_bytes"] = int(st.st_size)
    meta["mtime_epoch"] = float(st.st_mtime)

    obj, err = _safe_json_load(path)
    if err:
        meta["json"] = {"ok": False, "error": err}
        return meta

    meta["json"] = {"ok": True}
    if "ts_utc" in obj:
        meta["ts_utc"] = obj.get("ts_utc")
    if "ttl_seconds" in obj:
        meta["ttl_seconds"] = obj.get("ttl_seconds")
    return meta


def _systemd_is_active(unit: str) -> Dict[str, Any]:
    rc, out, err = _run(["systemctl", "is-active", unit], timeout_s=1.5)
    return {"unit": unit, "rc": rc, "state": out.strip() if out else None, "err": err.strip()[:200] if rc != 0 else None}


def _systemd_is_enabled(unit: str) -> Dict[str, Any]:
    rc, out, err = _run(["systemctl", "is-enabled", unit], timeout_s=1.5)
    return {"unit": unit, "rc": rc, "state": out.strip() if out else None, "err": err.strip()[:200] if rc != 0 else None}


def _systemctl_failed_chad() -> Dict[str, Any]:
    """
    Returns a normalized list of failed chad-* units. Never raises.
    """
    rc, out, err = _run(["systemctl", "--failed", "--no-pager"], timeout_s=2.0)
    if rc != 0:
        return {"ok": False, "rc": rc, "error": err.strip()[:300], "failed_units": []}

    failed: List[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # failed table contains the unit name as first column; keep only chad-*.
        if line.startswith("chad-"):
            failed.append(line.split()[0])
        elif " chad-" in line:
            # defensive parsing
            parts = line.split()
            for p in parts:
                if p.startswith("chad-"):
                    failed.append(p)
                    break

    # dedupe stable
    failed = list(dict.fromkeys(failed))
    return {"ok": True, "rc": rc, "failed_units": failed}


# ----------------------------
# Models (stable operator shapes)
# ----------------------------

class VersionResponse(BaseModel):
    ts_utc: str
    git: Dict[str, Any] = Field(default_factory=dict)
    runtime_dir: str
    repo_dir: str


class WhyBlockedResponse(BaseModel):
    ts_utc: str
    summary: str
    operator_intent: Dict[str, Any]
    live_gate: Dict[str, Any]
    shadow: Dict[str, Any]


class StatusResponse(BaseModel):
    ts_utc: str
    service: str = "CHAD Operator Surface"
    version: Dict[str, Any] = Field(default_factory=dict)
    timers: Dict[str, Any] = Field(default_factory=dict)
    failed_units: Dict[str, Any] = Field(default_factory=dict)
    runtime_files: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# Operator endpoints
# ----------------------------

@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        ts_utc=utc_now_iso(),
        git=git_fingerprint(),
        runtime_dir=str(RUNTIME_DIR),
        repo_dir=str(REPO_DIR),
    )


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    """
    Operator snapshot:
    - git fingerprint
    - timer health
    - failed units
    - runtime artifact presence (SSOT core)
    """
    files_meta: Dict[str, Any] = {}
    for k, rel in RUNTIME_FILES.items():
        p = rel
        # Resolve relative paths against repo dir for legacy runtime layout
        if not p.is_absolute():
            p = (REPO_DIR / p).resolve()
        files_meta[k] = runtime_file_meta(p)

    timers: Dict[str, Any] = {}
    for t in OPERATOR_TIMERS:
        timers[t] = {
            "is_enabled": _systemd_is_enabled(t),
            "is_active": _systemd_is_active(t),
        }

    failed = _systemctl_failed_chad()

    return StatusResponse(
        ts_utc=utc_now_iso(),
        version=git_fingerprint(),
        timers=timers,
        failed_units=failed,
        runtime_files=files_meta,
    )

@router.get("/risk_explain", tags=["risk"])
def risk_explain() -> Dict[str, Any]:
    """
    Human-readable risk + capital explanation (read-only).

    Sources (files only; no broker calls):
      - repo runtime/dynamic_caps.json (or /status runtime mapping)
      - SSOT runtime/scr_state.json

    Output is stable and bounded for dashboards/Telegram.
    """
    caps_path = (REPO_DIR / "runtime" / "dynamic_caps.json")
    scr_path = Path("/home/ubuntu/CHAD FINALE/runtime/scr_state.json")

    caps, caps_err = _safe_json_load(caps_path)
    scr, scr_err = _safe_json_load(scr_path)

    total_equity = float((caps or {}).get("total_equity") or 0.0)
    daily_risk_fraction = float((caps or {}).get("daily_risk_fraction") or 0.0)
    portfolio_risk_cap = float((caps or {}).get("portfolio_risk_cap") or 0.0)

    strategy_caps_raw = (caps or {}).get("strategy_caps") or {}
    if not isinstance(strategy_caps_raw, dict):
        strategy_caps_raw = {}

    rows: List[Dict[str, Any]] = []
    denom = portfolio_risk_cap if portfolio_risk_cap > 0 else 0.0
    for name in sorted(strategy_caps_raw.keys()):
        try:
            cap = float(strategy_caps_raw.get(name) or 0.0)
        except Exception:
            cap = 0.0
        pct = round((cap / denom) * 100.0, 4) if denom > 0 else None
        rows.append(
            {
                "strategy": str(name),
                "cap_usd": cap,
                "cap_pct_of_portfolio_risk_cap": pct,
            }
        )
@router.get("/perf_snapshot", tags=["perf"])
def perf_snapshot() -> Dict[str, Any]:
    """
    Performance + behavior snapshot (read-only).

    Sources (files only; no broker calls):
      - SSOT runtime/scr_state.json (performance stats + reasons)
      - repo runtime/portfolio_snapshot.json (equity breakdown)
      - optional repo runtime/decision_trace_heartbeat.json (last heartbeat proof)

    Output is stable + bounded for dashboards/Telegram.
    """
    scr_path = Path("/home/ubuntu/CHAD FINALE/runtime/scr_state.json")
    portfolio_path = REPO_DIR / "runtime" / "portfolio_snapshot.json"
    heartbeat_path = REPO_DIR / "runtime" / "decision_trace_heartbeat.json"

    scr, scr_err = _safe_json_load(scr_path)
    port, port_err = _safe_json_load(portfolio_path)
    hb, hb_err = _safe_json_load(heartbeat_path)

    # Pull SCR stats safely
    stats: Dict[str, Any] = {}
    reasons: List[str] = []
    state = None
    paper_only = None
    sizing_factor = None

    if scr_err is None and isinstance(scr, dict):
        state = scr.get("state")
        paper_only = scr.get("paper_only")
        sizing_factor = scr.get("sizing_factor")
        st = scr.get("stats")
        if isinstance(st, dict):
            stats = st
        rs = scr.get("reasons")
        if isinstance(rs, list):
            reasons = [str(x) for x in rs[:10]]

    # Pull portfolio snapshot safely
    portfolio_summary: Dict[str, Any] = {}
    if port_err is None and isinstance(port, dict):
        # Keep only key fields if present
        for k in ("ibkr_equity", "kraken_equity", "coinbase_equity", "total_equity", "ts_utc", "ttl_seconds"):
            if k in port:
                portfolio_summary[k] = port.get(k)

    # Heartbeat (optional)
    heartbeat_summary: Dict[str, Any] = {"ok": hb_err is None}
    if hb_err is None and isinstance(hb, dict):
        # keep a small subset
        for k in ("ts_utc", "trace_id", "live_gate", "notes"):
            if k in hb:
                heartbeat_summary[k] = hb.get(k)
    else:
        heartbeat_summary["error"] = hb_err

    # High-level operator narrative (short + factual)
    narrative: List[str] = []
    if state is not None:
        narrative.append(f"SCR.state={state}")
    if paper_only is not None:
        narrative.append(f"SCR.paper_only={bool(paper_only)}")
    if sizing_factor is not None:
        try:
            narrative.append(f"SCR.sizing_factor={float(sizing_factor):.4f}")
        except Exception:
            narrative.append("SCR.sizing_factor=UNKNOWN")

    # Key stats if present
    def _num(name: str) -> Optional[float]:
        v = stats.get(name)
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    win_rate = _num("win_rate")
    sharpe_like = _num("sharpe_like")
    max_dd = _num("max_drawdown")
    total_pnl = _num("total_pnl")
    total_trades = stats.get("total_trades")

    if win_rate is not None:
        narrative.append(f"win_rate={win_rate:.3f}")
    if sharpe_like is not None:
        narrative.append(f"sharpe_like={sharpe_like:.3f}")
    if max_dd is not None:
        narrative.append(f"max_drawdown={max_dd:.2f}")
    if total_pnl is not None:
        narrative.append(f"total_pnl={total_pnl:.2f}")
    if isinstance(total_trades, (int, float)):
        narrative.append(f"total_trades={int(total_trades)}")

    return {
        "ts_utc": utc_now_iso(),
        "schema_version": "perf_snapshot.v1",
        "scr": {
            "ok": scr_err is None,
            "error": scr_err,
            "state": state,
            "paper_only": paper_only,
            "sizing_factor": sizing_factor,
            "stats": stats,
            "reasons": reasons,
        },
        "portfolio_snapshot": {
            "ok": port_err is None,
            "error": port_err,
            "summary": portfolio_summary,
        },
        "decision_trace_heartbeat": heartbeat_summary,
        "operator_narrative": " | ".join(narrative),
    }

    # Posture summary (operator-friendly)
    posture: List[str] = []
    if caps_err is None:
        posture.append(f"total_equity=${total_equity:,.2f}")
        posture.append(f"daily_risk_fraction={daily_risk_fraction:.4f}")
        posture.append(f"portfolio_risk_cap≈${portfolio_risk_cap:,.2f}")
    else:
        posture.append(f"dynamic_caps=ERROR({caps_err})")

    if scr_err is None and isinstance(scr, dict):
        posture.append(f"SCR.state={scr.get('state')}")
        posture.append(f"SCR.paper_only={bool(scr.get('paper_only'))}")
        posture.append(f"SCR.sizing_factor={float(scr.get('sizing_factor') or 0.0):.4f}")
    else:
        posture.append(f"SCR=ERROR({scr_err})")

    # Why blocked summary (bounded)
    why: List[str] = []
    if scr_err is None and isinstance(scr, dict):
        if bool(scr.get("paper_only")):
            why.append("SCR requires paper-only (paper_only=true).")
        if str(scr.get("state") or "").upper() == "PAUSED":
            why.append("SCR state is PAUSED (live should be halted).")
        rs = scr.get("reasons")
        if isinstance(rs, list) and rs:
            why.extend([str(x) for x in rs[:6]])
    else:
        why.append("SCR missing → fail-closed posture assumed.")

    return {
        "ts_utc": utc_now_iso(),
        "schema_version": "risk_explain.v1",
        "dynamic_caps": {
            "ok": caps_err is None,
            "error": caps_err,
            "total_equity": total_equity,
            "daily_risk_fraction": daily_risk_fraction,
            "portfolio_risk_cap": portfolio_risk_cap,
        },
        "strategy_caps": rows,
        "posture_summary": " | ".join(posture),
        "why_blocked_summary": why,
    }

@router.get("/why_blocked", response_model=WhyBlockedResponse)
def why_blocked() -> WhyBlockedResponse:
    """
    High-signal operator explanation based on runtime artifacts only.
    Fail-closed if required files are missing.
    """
    # Prefer existing gateway if it already writes these to runtime.
    # We do not call broker APIs; we only read files.
    lg_path = REPO_DIR / "runtime" / "live_gate.json"
    op_path = REPO_DIR / "runtime" / "operator_intent.json"
    scr_path = Path("/home/ubuntu/CHAD FINALE/runtime/scr_state.json")

    lg, lg_err = _safe_json_load(lg_path)
    op, op_err = _safe_json_load(op_path)
    scr, scr_err = _safe_json_load(scr_path)

    # If live_gate.json is not present, we can still provide signal from scr_state/operator intent.
    live_gate_payload = {"ok": lg_err is None, "error": lg_err, "data": lg or {}}
    operator_payload = {"ok": op_err is None, "error": op_err, "data": op or {}}
    shadow_payload = {"ok": scr_err is None, "error": scr_err, "data": scr or {}}

    # Build a compact summary
    reasons: List[str] = []
    if operator_payload["ok"]:
        reasons.append(f"OperatorIntent={operator_payload['data'].get('mode')}")
    else:
        reasons.append("OperatorIntent=MISSING")

    if shadow_payload["ok"]:
        reasons.append(f"SCR.state={shadow_payload['data'].get('state')}")
        if shadow_payload["data"].get("paper_only") is True:
            reasons.append("SCR.paper_only=true")
    else:
        reasons.append("SCR=MISSING")

    if live_gate_payload["ok"]:
        lgd = live_gate_payload["data"]
        # common fields if present
        if isinstance(lgd, dict):
            if "mode" in lgd:
                reasons.append(f"CHAD_MODE={((lgd.get(mode) or {}).get(chad_mode))}")
            if "allow_ibkr_live" in lgd:
                reasons.append(f"allow_ibkr_live={lgd.get(allow_ibkr_live)}")
            if "allow_ibkr_paper" in lgd:
                reasons.append(f"allow_ibkr_paper={lgd.get(allow_ibkr_paper)}")
            # include reasons list if present
            rs = lgd.get("reasons")
            if isinstance(rs, list):
                reasons.extend([str(x) for x in rs[:6]])
    else:
        reasons.append("LiveGateSnapshot=MISSING")

    summary = " | ".join(reasons[:12])

    return WhyBlockedResponse(
        ts_utc=utc_now_iso(),
        summary=summary,
        operator_intent=operator_payload,
        live_gate=live_gate_payload,
        shadow=shadow_payload,
    )
