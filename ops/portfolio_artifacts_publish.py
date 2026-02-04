#!/usr/bin/env python3
"""
CHAD Portfolio Artifact Publisher (Production, PROPOSE_ONLY)

File: ops/portfolio_artifacts_publish.py

Writes SSOT v4.2 portfolio artifacts:
- reports/portfolios/PORTFOLIO_<profile>_<ts>.json
- reports/rebalance/REBALANCE_<profile>_<ts>.json
- runtime/portfolio_state.json  (pointer + drift summary)

Engine integration:
- PortfolioEngine reads from EnginePaths.from_env() (CHAD_REPO_DIR, CHAD_RUNTIME_DIR, CHAD_CONFIG_DIR).
- We set those env vars explicitly to avoid ambiguity.

Hard guarantees:
- No broker calls, no orders (engine is read-only).
- Atomic writes with fsync(file)+rename+fsync(dir).
- Fail-closed: on error, write portfolio_state.json with UNKNOWN fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from chad.portfolio.portfolio_engine import EnginePaths, PortfolioEngine

# -----------------------------
# Paths / Config
# -----------------------------

REPO_DIR = Path(os.environ.get("CHAD_REPO_DIR", "/home/ubuntu/chad_finale")).resolve()
RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime")).resolve()
CONFIG_DIR = Path(os.environ.get("CHAD_CONFIG_DIR", "/home/ubuntu/CHAD FINALE/config")).resolve()

REPORTS_DIR = Path(os.environ.get("CHAD_REPORTS_DIR", "/home/ubuntu/CHAD FINALE/reports")).resolve()
PORTFOLIOS_DIR = REPORTS_DIR / "portfolios"
REBALANCE_DIR = REPORTS_DIR / "rebalance"

PORTFOLIO_STATE_PATH = RUNTIME_DIR / "portfolio_state.json"

PROFILE = os.environ.get("CHAD_PORTFOLIO_PROFILE", "BALANCED").strip().upper()
TTL_SECONDS = int(os.environ.get("CHAD_PORTFOLIO_STATE_TTL_SECONDS", "3600"))


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_hex(data)

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

    return digest


def fail_closed_portfolio_state(ts: str, reason: str) -> Dict[str, Any]:
    return {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "schema_version": "portfolio_state.v1",
        "active_profile": PROFILE,
        "portfolio_target_id": None,
        "portfolio_target_hash": None,
        "drift": {"max_position_weight_drift": None, "total_turnover_needed": None},
        "next_rebalance_check_ts_utc": None,
        "notes": reason,
    }


def _hash_obj(obj: Dict[str, Any]) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_hex(b)


# -----------------------------
# Publisher
# -----------------------------

def main() -> int:
    ts = utc_now_iso()
    ts_compact = utc_now_compact()

    PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
    REBALANCE_DIR.mkdir(parents=True, exist_ok=True)

    # Force engine paths deterministically
    os.environ["CHAD_REPO_DIR"] = str(REPO_DIR)
    os.environ["CHAD_RUNTIME_DIR"] = str(RUNTIME_DIR)
    os.environ["CHAD_CONFIG_DIR"] = str(CONFIG_DIR)

    try:
        paths = EnginePaths.from_env()
        eng = PortfolioEngine(paths=paths)

        # 1) targets artifact
        targets = eng.get_targets(profile=PROFILE)
        targets_payload: Dict[str, Any] = dict(targets)
        targets_payload["generated_ts_utc"] = ts

        target_hash = _hash_obj(targets_payload)
        target_id = f"pt_{PROFILE.lower()}_{ts_compact}_{target_hash[:8]}"
        targets_payload["portfolio_target_id"] = target_id
        targets_payload["portfolio_target_hash"] = f"sha256:{target_hash}"

        target_path = PORTFOLIOS_DIR / f"PORTFOLIO_{PROFILE}_{ts_compact}.json"
        atomic_write_json(target_path, targets_payload)

        # 2) rebalance artifact
        reb = eng.get_rebalance_latest(profile=PROFILE)
        reb_payload: Dict[str, Any] = dict(reb)
        reb_payload["generated_ts_utc"] = ts
        reb_payload["portfolio_target_id"] = target_id
        reb_payload["portfolio_target_hash"] = f"sha256:{target_hash}"

        reb_hash = _hash_obj(reb_payload)
        reb_payload["rebalance_plan_hash"] = f"sha256:{reb_hash}"

        reb_path = REBALANCE_DIR / f"REBALANCE_{PROFILE}_{ts_compact}.json"
        atomic_write_json(reb_path, reb_payload)

        # Drift summary
        diffs = reb_payload.get("diffs") or []
        max_abs = 0.0
        total_turnover = 0.0
        if isinstance(diffs, list):
            for d in diffs:
                if not isinstance(d, dict):
                    continue
                try:
                    v = float(d.get("delta_weight", 0.0))
                except Exception:
                    continue
                if abs(v) > max_abs:
                    max_abs = abs(v)
                total_turnover += abs(v)

        # 3) runtime pointer
        state = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "schema_version": "portfolio_state.v1",
            "active_profile": PROFILE,
            "portfolio_target_id": target_id,
            "portfolio_target_hash": f"sha256:{target_hash}",
            "drift": {
                "max_position_weight_drift": round(max_abs, 6),
                "total_turnover_needed": round(total_turnover, 6),
            },
            "next_rebalance_check_ts_utc": None,
            "artifacts": {
                "portfolio_target_path": str(target_path),
                "rebalance_plan_path": str(reb_path),
            },
            "notes": "propose_only_artifacts_written; no broker calls",
        }
        atomic_write_json(PORTFOLIO_STATE_PATH, state)

        print(json.dumps({
            "ok": True,
            "profile": PROFILE,
            "portfolio_target_path": str(target_path),
            "rebalance_plan_path": str(reb_path),
            "portfolio_state_path": str(PORTFOLIO_STATE_PATH),
            "ts_utc": ts,
        }, sort_keys=True))
        return 0

    except Exception as exc:
        state = fail_closed_portfolio_state(ts, reason=f"publish_error:{type(exc).__name__}")
        atomic_write_json(PORTFOLIO_STATE_PATH, state)
        print(json.dumps({
            "ok": False,
            "profile": PROFILE,
            "portfolio_state_path": str(PORTFOLIO_STATE_PATH),
            "ts_utc": ts,
            "error": str(exc),
        }, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
