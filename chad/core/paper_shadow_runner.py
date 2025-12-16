#!/usr/bin/env python3
"""
CHAD Paper Shadow Runner (Production-Safe)

Purpose
- Provide a scheduled, audit-friendly "paper shadow" lane that can be toggled on/off via runtime config.
- In this Phase-7 build, it is SAFE-BY-DEFAULT and PREVIEW-ONLY unless explicitly enabled + armed.

Safety guarantees
- Default behaviour is preview-only (no orders).
- Missing/invalid config => disabled => preview-only (no orders).
- The --preview flag forces preview-only regardless of config.
- This module never flips CHAD live mode and does not modify caps.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Backward-compatible test/ops contract constants
# ---------------------------------------------------------------------------
# Backward-compatible test contract API (do not remove)
# These are intentionally small, pure, and safe-by-default:
# - No broker I/O
# - No ib_insync import on gating paths
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class PaperShadowConfig:
    enabled: bool = False

def is_armed() -> bool:
    """Return True only when the operator explicitly arms paper-order capability."""
    return os.environ.get(ARM_ENV_NAME, "").strip() == ARM_PHRASE

def should_place_paper_orders(cfg: PaperShadowConfig) -> Tuple[bool, List[str]]:
    """Gating logic for paper shadow lane. Safe-by-default."""
    reasons: List[str] = []
    if not bool(getattr(cfg, "enabled", False)):
        reasons.append("paper_shadow.enabled is false")
        return False, reasons
    if not is_armed():
        reasons.append("CHAD_PAPER_SHADOW_ARMED is not set to arm phrase")
        return False, reasons
    reasons.append("enabled=true and armed=true")
    return True, reasons

# ---------------------------------------------------------------------------
ARM_PHRASE = "I_UNDERSTAND_THIS_CAN_PLACE_PAPER_ORDERS"


# ---------------------------------------------------------------------------
# Backward-compatible test/ops contract constants
# ---------------------------------------------------------------------------
# Environment variable that can be used to 'arm' the runner in production.
# NOTE: The runner remains safe-by-default; tests expect this constant.
ARM_ENV_NAME = "CHAD_PAPER_SHADOW_ARMED"


import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DEFAULT = Path("/home/ubuntu/CHAD FINALE")
CONFIG_DEFAULT = ROOT_DEFAULT / "runtime" / "paper_shadow.json"
REPORTS_DIR_DEFAULT = ROOT_DEFAULT / "reports" / "shadow"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class PaperShadowConfig:
    enabled: bool = False
    armed: bool = False

    strategy_allowlist: List[str] | None = None
    max_orders_per_run: int = 5
    size_multiplier: float = 1.0
    min_notional_usd: float = 100.0
    auto_cancel_seconds: int = 15

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002
    ibkr_client_id: int = 9201

    reports_dir: Path = REPORTS_DIR_DEFAULT

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_allowlist", self.strategy_allowlist or ["BETA"])


def load_config(path: Path) -> PaperShadowConfig:
    """
    Safe-by-default loader:
    - Missing config => enabled=False, armed=False
    - Invalid config => enabled=False, armed=False
    """
    if not path.is_file():
        return PaperShadowConfig(enabled=False, armed=False)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return PaperShadowConfig(enabled=False, armed=False)

    enabled = bool(raw.get("enabled", False))
    armed = bool(raw.get("armed", False))

    allow = raw.get("strategy_allowlist", ["BETA"])
    if not isinstance(allow, list):
        allow = ["BETA"]
    allow = [str(x).strip().upper() for x in allow if str(x).strip()]

    ibkr = raw.get("ibkr", {}) if isinstance(raw.get("ibkr", {}), dict) else {}
    host = str(ibkr.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(ibkr.get("port", 4002))
    cid = int(ibkr.get("client_id", 9201))

    return PaperShadowConfig(
        enabled=enabled,
        armed=armed,
        strategy_allowlist=allow,
        max_orders_per_run=int(raw.get("max_orders_per_run", 5)),
        size_multiplier=float(raw.get("size_multiplier", 1.0)),
        min_notional_usd=float(raw.get("min_notional_usd", 100.0)),
        auto_cancel_seconds=int(raw.get("auto_cancel_seconds", 15)),
        ibkr_host=host,
        ibkr_port=port,
        ibkr_client_id=cid,
        reports_dir=REPORTS_DIR_DEFAULT,
    )


def run_once(cfg: PaperShadowConfig, *, preview_forced: bool) -> Dict[str, Any]:
    """
    Phase-7 stance:
    - Always write an audit artifact.
    - Only preview is implemented here (no orders).
    """
    now = _utc_now()

    mode = "preview"
    reason = ""
    if preview_forced:
        reason = "--preview forced"
    elif not cfg.enabled:
        reason = "paper_shadow.enabled is false → preview only"
    elif not cfg.armed:
        reason = "paper_shadow.armed is false → preview only"
    else:
        # We intentionally DO NOT place paper orders in this PR.
        reason = "armed mode intentionally not implemented yet → preview only"

    report: Dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "mode": mode,
        "enabled": bool(cfg.enabled),
        "armed": bool(cfg.armed),
        "reason": reason,
        "config": {
            "strategy_allowlist": cfg.strategy_allowlist,
            "max_orders_per_run": cfg.max_orders_per_run,
            "size_multiplier": cfg.size_multiplier,
            "min_notional_usd": cfg.min_notional_usd,
            "auto_cancel_seconds": cfg.auto_cancel_seconds,
            "ibkr": {"host": cfg.ibkr_host, "port": cfg.ibkr_port, "client_id": cfg.ibkr_client_id},
        },
        "actions": {"orders_submitted": 0, "details": []},
    }

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / f"PAPER_SHADOW_PREVIEW_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    _atomic_write_json(out, report)
    report["artifact_path"] = str(out)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CHAD Paper Shadow Runner (safe-by-default).")
    p.add_argument("--preview", action="store_true", help="Force preview-only (no orders).")
    p.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DEFAULT),
        help="Path to runtime paper shadow config JSON (safe-by-default).",
    )
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config).expanduser().resolve())
    report = run_once(cfg, preview_forced=bool(args.preview))

    print(f"[paper_shadow_runner] mode={report.get('mode')}")
    print(f"[paper_shadow_runner] enabled={report.get('enabled')} armed={report.get('armed')}")
    print(f"[paper_shadow_runner] reason: {report.get('reason')}")
    print(f"[paper_shadow_runner] wrote: {report.get('artifact_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
