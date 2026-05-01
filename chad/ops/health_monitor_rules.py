"""
CHAD Health Monitor — Rule Engine (Tier 1)
Fast mechanical checks. No API call. Runs every cycle.
"""
from __future__ import annotations
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = REPO_ROOT / "runtime"
DATA = REPO_ROOT / "data"

@dataclass
class Finding:
    rule_id: str
    severity: str  # CRITICAL / WARNING / INFO
    title: str
    description: str
    remedy_type: str  # SAFE_AUTO / CODE_CHANGE / SERVICE_RESTART / NOTIFY_ONLY
    remedy_action: str  # what to do
    remedy_args: dict = field(default_factory=dict)
    evidence: str = ""

def _age(path: Path) -> Optional[float]:
    """Return file age in seconds, or None if missing."""
    try:
        return time.time() - path.stat().st_mtime
    except Exception:
        return None

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _service_active(name: str) -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() == "active"
    except Exception:
        return False

def _service_failed(name: str) -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-failed", name],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() == "failed"
    except Exception:
        return False


# ── Rule definitions ──────────────────────────────────────────────────────────

def rule_critical_services(findings: List[Finding]) -> None:
    """R01 — Critical services must be active."""
    critical = [
        "chad-live-loop.service",
        "chad-orchestrator.service",
        "chad-ibgateway.service",
        "chad-kraken-ws.service",
        "chad-telegram-bot.service",
        "chad-dashboard.service",
        "chad-backend.service",
    ]
    for svc in critical:
        if _service_failed(svc):
            findings.append(Finding(
                rule_id="R01",
                severity="CRITICAL",
                title=f"Service FAILED: {svc}",
                description=f"{svc} is in failed state and needs restart.",
                remedy_type="SERVICE_RESTART",
                remedy_action="restart_service",
                remedy_args={"service": svc},
                evidence=f"systemctl is-failed {svc} = failed",
            ))
        elif not _service_active(svc):
            findings.append(Finding(
                rule_id="R01",
                severity="CRITICAL",
                title=f"Service INACTIVE: {svc}",
                description=f"{svc} is not active.",
                remedy_type="SERVICE_RESTART",
                remedy_action="restart_service",
                remedy_args={"service": svc},
                evidence=f"systemctl is-active {svc} = inactive",
            ))


def rule_feed_freshness(findings: List[Finding]) -> None:
    """R02 — Critical feeds must be within TTL."""
    feeds = [
        ("price_cache.json", 180),
        ("regime_state.json", 180),
        ("dynamic_caps.json", 180),
        ("regime_booster.json", 240),
        ("kraken_prices.json", 120),
        ("reconciliation_state.json", 480),
    ]
    for fname, ttl in feeds:
        age = _age(RUNTIME / fname)
        if age is None:
            findings.append(Finding(
                rule_id="R02",
                severity="WARNING",
                title=f"Feed MISSING: {fname}",
                description=f"Runtime file {fname} does not exist.",
                remedy_type="NOTIFY_ONLY",
                remedy_action="notify",
                evidence=f"{fname} not found in {RUNTIME}",
            ))
        elif age > ttl * 2:
            findings.append(Finding(
                rule_id="R02",
                severity="CRITICAL",
                title=f"Feed STALE: {fname} ({int(age)}s old, TTL={ttl}s)",
                description=f"{fname} is {int(age)}s old — more than 2× TTL.",
                remedy_type="SERVICE_RESTART",
                remedy_action="restart_feed_publisher",
                remedy_args={"feed": fname, "age": age, "ttl": ttl},
                evidence=f"mtime age={int(age)}s TTL={ttl}s",
            ))


def rule_scr_state(findings: List[Finding]) -> None:
    """R03 — SCR must not be PAUSED."""
    d = _read_json(RUNTIME / "scr_state.json")
    state = str(d.get("state", "")).upper()
    if state == "PAUSED":
        findings.append(Finding(
            rule_id="R03",
            severity="CRITICAL",
            title="SCR is PAUSED — trading halted",
            description="SCR state machine is PAUSED. Sizing factor=0. No trades executing.",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"scr_state.json state={state}",
        ))


def rule_stop_bus(findings: List[Finding]) -> None:
    """R04 — Stop bus must not be active."""
    d = _read_json(RUNTIME / "stop_bus.json")
    if d.get("active"):
        findings.append(Finding(
            rule_id="R04",
            severity="CRITICAL",
            title="STOP BUS ACTIVE — all trading halted",
            description=f"Stop bus is active. Reason: {d.get('reason', 'unknown')}",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"stop_bus.json active=true reason={d.get('reason')}",
        ))


def rule_reconciliation(findings: List[Finding]) -> None:
    """R05 — Reconciliation must be GREEN."""
    d = _read_json(RUNTIME / "reconciliation_state.json")
    status = str(d.get("status", "")).upper()
    if status == "RED":
        findings.append(Finding(
            rule_id="R05",
            severity="CRITICAL",
            title="Reconciliation RED — position mismatch",
            description=f"Reconciliation is RED. worst_diff={d.get('worst_diff')} mismatches={d.get('mismatches')}",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"reconciliation_state.json status=RED",
        ))


def rule_profit_lock(findings: List[Finding]) -> None:
    """R06 — Alert on profit lock escalation."""
    d = _read_json(RUNTIME / "profit_lock_state.json")
    mode = str(d.get("mode", "NORMAL")).upper()
    if mode in ("LOCK2", "LOCK3", "HARD_STOP"):
        findings.append(Finding(
            rule_id="R06",
            severity="CRITICAL",
            title=f"Profit Lock: {mode}",
            description=f"Profit lock in {mode} — sizing heavily restricted.",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"profit_lock_state.json mode={mode}",
        ))


def rule_disk_usage(findings: List[Finding]) -> None:
    """R07 — Disk usage must stay below 75%."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        pct = used / total * 100
        if pct > 85:
            findings.append(Finding(
                rule_id="R07",
                severity="CRITICAL",
                title=f"Disk {pct:.1f}% full — CRITICAL",
                description="Disk usage above 85%. System may fail to write runtime files.",
                remedy_type="SAFE_AUTO",
                remedy_action="archive_old_fills",
                remedy_args={"current_pct": pct},
                evidence=f"disk usage={pct:.1f}%",
            ))
        elif pct > 75:
            findings.append(Finding(
                rule_id="R07",
                severity="WARNING",
                title=f"Disk {pct:.1f}% full — WARNING",
                description="Disk usage above 75%. Consider archiving old fill files.",
                remedy_type="SAFE_AUTO",
                remedy_action="archive_old_fills",
                remedy_args={"current_pct": pct},
                evidence=f"disk usage={pct:.1f}%",
            ))
    except Exception:
        pass


def rule_corrupt_runtime_files(findings: List[Finding]) -> None:
    """R08 — Runtime JSON files must not be zero-byte or corrupt."""
    critical_files = [
        "position_guard.json",
        "dynamic_caps.json",
        "scr_state.json",
        "profit_lock_state.json",
    ]
    for fname in critical_files:
        p = RUNTIME / fname
        if not p.exists():
            continue
        if p.stat().st_size == 0:
            findings.append(Finding(
                rule_id="R08",
                severity="CRITICAL",
                title=f"Zero-byte runtime file: {fname}",
                description=f"{fname} is 0 bytes — likely corrupted mid-write.",
                remedy_type="SAFE_AUTO",
                remedy_action="restore_from_backup",
                remedy_args={"file": fname},
                evidence=f"{fname} size=0",
            ))
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            findings.append(Finding(
                rule_id="R08",
                severity="CRITICAL",
                title=f"Corrupt runtime file: {fname}",
                description=f"{fname} failed JSON parse: {e}",
                remedy_type="SAFE_AUTO",
                remedy_action="restore_from_backup",
                remedy_args={"file": fname},
                evidence=str(e),
            ))


def rule_edge_decay_halts(findings: List[Finding]) -> None:
    """R09 — Notify on any active edge decay halts."""
    p = RUNTIME / "strategy_allocations.json"
    if not p.exists():
        return
    d = _read_json(p)
    for strategy, info in d.items():
        if isinstance(info, dict) and info.get("halted"):
            findings.append(Finding(
                rule_id="R09",
                severity="WARNING",
                title=f"Edge decay halt active: {strategy}",
                description=f"Strategy {strategy} is halted by edge decay monitor.",
                remedy_type="NOTIFY_ONLY",
                remedy_action="notify",
                evidence=f"strategy_allocations.json {strategy}.halted=true",
            ))


def run_all_rules() -> List[Finding]:
    """Run all rules and return list of findings."""
    findings: List[Finding] = []
    for fn in [
        rule_critical_services,
        rule_feed_freshness,
        rule_scr_state,
        rule_stop_bus,
        rule_reconciliation,
        rule_profit_lock,
        rule_disk_usage,
        rule_corrupt_runtime_files,
        rule_edge_decay_halts,
    ]:
        try:
            fn(findings)
        except Exception as e:
            pass  # rule failure never crashes the monitor
    return findings
