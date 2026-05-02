"""
CHAD Health Monitor — Remediation Engine (Tier 3)
Executes fixes. Safe actions run immediately.
Service restarts run immediately.
Code changes use 5-minute veto window.
"""
from __future__ import annotations
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = REPO_ROOT / "runtime"
VENV_PYTHON = REPO_ROOT / "venv/bin/python3"

# Feed → publisher service mapping
FEED_PUBLISHER_MAP = {
    "price_cache.json": "chad-ibkr-price-refresh.timer",
    "regime_state.json": "chad-live-loop.service",
    "dynamic_caps.json": "chad-orchestrator.service",
    "regime_booster.json": "chad-regime-booster.timer",
    "kraken_prices.json": "chad-kraken-ws.service",
    "reconciliation_state.json": "chad-reconciliation-publisher.timer",
    "event_risk.json": "chad-event-risk.timer",
}


def _run(cmd: list, timeout: int = 30) -> tuple[int, str, str]:
    """Run shell command, return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def restart_service(service: str, **kwargs) -> str:
    """Restart a systemd service."""
    rc, out, err = _run(["sudo", "-n", "systemctl", "restart", service])
    if rc == 0:
        return f"✅ Restarted {service}"
    return f"❌ Failed to restart {service}: {err}"


def restart_feed_publisher(feed: str, age: float, ttl: int, **kwargs) -> str:
    """Restart the publisher responsible for a stale feed."""
    svc = FEED_PUBLISHER_MAP.get(feed)
    if not svc:
        return f"⚠️ No publisher mapped for {feed} — manual investigation needed"
    rc, out, err = _run(["sudo", "-n", "systemctl", "restart", svc])
    if rc == 0:
        return f"✅ Restarted {svc} (feed {feed} was {int(age)}s old, TTL={ttl}s)"
    return f"❌ Failed to restart {svc}: {err}"


def archive_old_fills(current_pct: float, **kwargs) -> str:
    """Archive fill files older than 30 days to free disk space."""
    fills_dir = REPO_ROOT / "data/fills"
    cutoff = time.time() - (30 * 86400)
    archived = 0
    archive_dir = REPO_ROOT / "data/fills_archive"
    archive_dir.mkdir(exist_ok=True)
    for f in fills_dir.glob("*.ndjson"):
        if f.stat().st_mtime < cutoff:
            f.rename(archive_dir / f.name)
            archived += 1
    return f"✅ Archived {archived} fill files older than 30 days (disk was {current_pct:.1f}%)"


def restore_from_backup(file: str, **kwargs) -> str:
    """Restore a corrupt runtime file from the most recent backup."""
    target = RUNTIME / file
    # Find most recent .bak file for this runtime file
    bak_files = list(RUNTIME.glob(f"{file}*.bak")) + \
                list(REPO_ROOT.glob(f"**/{file}*.bak"))
    if not bak_files:
        return f"⚠️ No backup found for {file} — cannot auto-restore"
    latest = max(bak_files, key=lambda p: p.stat().st_mtime)
    import shutil
    shutil.copy2(latest, target)
    return f"✅ Restored {file} from {latest.name}"


def notify(**kwargs) -> str:
    """NOTIFY_ONLY — no auto-fix, just alert was sent."""
    return "📢 Alert sent — no auto-fix for this finding type"


def write_signal_throttle(trade_count: int, pnl: float, **kwargs) -> str:
    """Write a throttle flag to runtime/signal_throttle.json
    that live_loop reads to reduce signal frequency."""
    throttle = {
        "active": True,
        "reason": f"churn_detected_{trade_count}_trades",
        "max_signals_per_cycle": 3,
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "auto_expires_at_utc": (
            datetime.now(timezone.utc) + timedelta(hours=4)
        ).isoformat(),
        "trade_count": trade_count,
        "realized_pnl": pnl,
    }
    _path = RUNTIME / "signal_throttle.json"
    _tmp = _path.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(throttle, indent=2), encoding="utf-8")
    os.replace(str(_tmp), str(_path))
    return (
        f"✅ Signal throttle activated: max 3 signals/cycle "
        f"for 4 hours (churn: {trade_count} trades, "
        f"PnL=${pnl:.2f})"
    )


def clear_reconciliation_artifact(strategy: str, **kwargs) -> str:
    """Remove stale reconciliation strategy from winner_scaling."""
    try:
        ws_path = RUNTIME / "winner_scaling.json"
        if not ws_path.exists():
            return f"⚠️ winner_scaling.json not found"
        data = json.loads(ws_path.read_text(encoding="utf-8"))
        multipliers = data.get("multipliers", {})
        if strategy in multipliers:
            del multipliers[strategy]
            data["multipliers"] = multipliers
            _tmp = ws_path.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(str(_tmp), str(ws_path))
            return f"✅ Cleared stale artifact: {strategy}"
        return f"ℹ️ {strategy} not found in winner_scaling"
    except Exception as e:
        return f"❌ Failed to clear {strategy}: {e}"


# Remediation dispatch
REMEDY_DISPATCH = {
    "restart_service": restart_service,
    "restart_feed_publisher": restart_feed_publisher,
    "archive_old_fills": archive_old_fills,
    "restore_from_backup": restore_from_backup,
    "notify": notify,
    "write_signal_throttle": write_signal_throttle,
    "clear_reconciliation_artifact": clear_reconciliation_artifact,
}


def execute_remedy(remedy_action: str, remedy_args: dict) -> str:
    """Execute a remediation action. Returns result message."""
    fn = REMEDY_DISPATCH.get(remedy_action)
    if fn is None:
        return f"⚠️ Unknown remedy action: {remedy_action}"
    try:
        return fn(**remedy_args)
    except Exception as e:
        return f"❌ Remedy {remedy_action} failed: {e}"
