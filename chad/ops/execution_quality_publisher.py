#!/usr/bin/env python3
"""
CHAD — Execution Quality + Latency State Publisher (SSOT v5.0 bootstrap)

What this does (read-only, safety-only):
- Publishes two SSOT v5.0 runtime artifacts (atomic writes, TTL-stamped):
  - runtime/latency_state.json
  - runtime/execution_quality.json

Why this exists:
- Your current active repo does NOT implement SSOT v5.0 artifacts like execution_quality.json
  or latency_state.json yet. This publisher is the minimal, production-safe bridge.

Design constraints:
- Never crashes if inputs are missing or malformed.
- Never makes network calls.
- Never enables trading or changes modes.
- Always writes outputs each run (even if values are UNKNOWN), with TTL + ts_utc.
- Atomic write: tmp -> fsync -> rename (delegated to runtime_json helper).

Inputs (best-effort, optional):
- runtime/ibkr_status.json (often has ok + latency_ms + ttl_seconds + ts_utc)
- runtime/ibkr_watchdog_last.json (freshness hints)
- runtime/full_execution_cycle_last.json (cycle timing hints, if present)
- runtime/positions_snapshot.json (symbol universe hints, if present)

Outputs (pinned shapes; conservative defaults):
- latency_state.json:
    {
      ts_utc, ttl_seconds,
      env_label: normal|thin|dangerous|unknown,
      broker: { name, ok, latency_ms, source_path },
      system: { loadavg_1m, loadavg_5m, loadavg_15m },
      thresholds_ms: { thin_ms, dangerous_ms },
      notes
    }

- execution_quality.json:
    {
      ts_utc, ttl_seconds,
      env_label: normal|thin|dangerous|unknown,
      by_venue_symbol: { "IBKR:SPY": {...}, ... }   # best-effort; may be empty
      worst_offenders: [],
      flags: [],
      hash_sha256: "sha256:..."
    }

This is a bootstrap. Later, you will extend this to compute true slippage/partial fills/rejects
from normalized broker_events + fills + execution_metrics once those ledgers exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chad.utils.runtime_json import read_runtime_state_json, stable_json_dumps, write_runtime_state_json

LOG = logging.getLogger("chad.ops.execution_quality_publisher")


# ----------------------------
# Env + path helpers
# ----------------------------

def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def _repo_root() -> Path:
    # Prefer CHAD_ROOT when set by systemd.
    root = str(os.environ.get("CHAD_ROOT", "")).strip()
    if root:
        p = Path(root).expanduser()
        if p.is_dir():
            return p.resolve()
    # Fallback: traverse up from this file: chad/ops/ -> chad/ -> repo
    return Path(__file__).resolve().parents[2]


def _runtime_dir(repo_root: Path) -> Path:
    # Allow override, else repo_root/runtime
    rd = str(os.environ.get("CHAD_RUNTIME_DIR", "")).strip()
    if rd:
        return Path(rd).expanduser().resolve()
    return (repo_root / "runtime").resolve()


def _safe_loadavg() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass
    return None, None, None


def _sha256_prefixed(payload: Dict[str, Any]) -> str:
    # Hash canonical JSON (stable order) without relying on json.dumps ordering.
    b = stable_json_dumps(payload).encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    return f"sha256:{h}"


# ----------------------------
# Classification logic
# ----------------------------

@dataclass(frozen=True)
class LatencyThresholds:
    thin_ms: int
    dangerous_ms: int


def _classify_latency_env(latency_ms: Optional[float], *, ok: Optional[bool], th: LatencyThresholds) -> str:
    """
    Conservative classifier:
    - If broker not ok or latency unknown -> unknown (caller may tighten elsewhere)
    - else:
        latency <= thin_ms -> normal
        thin_ms < latency <= dangerous_ms -> thin
        latency > dangerous_ms -> dangerous
    """
    if ok is False:
        return "dangerous"
    if latency_ms is None:
        return "unknown"
    try:
        x = float(latency_ms)
    except Exception:
        return "unknown"
    if x <= float(th.thin_ms):
        return "normal"
    if x <= float(th.dangerous_ms):
        return "thin"
    return "dangerous"


# ----------------------------
# Core read helpers (best effort)
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _extract_ibkr_latency(runtime_dir: Path) -> Tuple[Optional[bool], Optional[float], str]:
    """
    Returns: (ok, latency_ms, source_path)
    """
    p = runtime_dir / "ibkr_status.json"
    if not p.is_file():
        return None, None, str(p)

    obj = _read_json(p)
    ok = obj.get("ok")
    latency = obj.get("latency_ms")
    try:
        ok_b = bool(ok) if ok is not None else None
    except Exception:
        ok_b = None
    try:
        lat_f = float(latency) if latency is not None else None
    except Exception:
        lat_f = None
    return ok_b, lat_f, str(p)


def _extract_symbols(runtime_dir: Path, limit: int = 25) -> List[str]:
    """
    Best-effort symbol list for per-venue-symbol buckets.
    Uses positions_snapshot.json if present; otherwise empty.
    """
    p = runtime_dir / "positions_snapshot.json"
    if not p.is_file():
        return []
    obj = _read_json(p)
    positions = obj.get("positions")
    out: List[str] = []
    if isinstance(positions, list):
        for it in positions:
            if not isinstance(it, dict):
                continue
            sym = it.get("symbol")
            if isinstance(sym, str) and sym.strip():
                out.append(sym.strip())
            if len(out) >= limit:
                break
    # De-dup stable order
    seen = set()
    dedup: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


# ----------------------------
# Build payloads
# ----------------------------

def build_latency_state(*, runtime_dir: Path) -> Dict[str, Any]:
    th = LatencyThresholds(
        thin_ms=_env_int("CHAD_LATENCY_THIN_MS", 250),
        dangerous_ms=_env_int("CHAD_LATENCY_DANGEROUS_MS", 750),
    )

    ok, latency_ms, src = _extract_ibkr_latency(runtime_dir)
    env_label = _classify_latency_env(latency_ms, ok=ok, th=th)

    l1, l5, l15 = _safe_loadavg()

    payload: Dict[str, Any] = {
        "env_label": env_label,
        "broker": {
            "name": "IBKR",
            "ok": ok,
            "latency_ms": latency_ms,
            "source_path": src,
        },
        "system": {
            "loadavg_1m": l1,
            "loadavg_5m": l5,
            "loadavg_15m": l15,
        },
        "thresholds_ms": {"thin_ms": th.thin_ms, "dangerous_ms": th.dangerous_ms},
        "notes": (
            "Bootstrap latency classifier from runtime/ibkr_status.json when available. "
            "UNKNOWN means insufficient evidence; higher layers should tighten, not loosen."
        ),
    }
    return payload


def build_execution_quality(*, runtime_dir: Path, latency_env_label: str, latency_ms: Optional[float]) -> Dict[str, Any]:
    """
    Bootstrap execution quality:
    - env_label mirrors latency env_label until true execution metrics exist.
    - by_venue_symbol created for currently-held symbols (best effort) with unknown slippage/partials/rejects.
    """
    symbols = _extract_symbols(runtime_dir)
    by_vs: Dict[str, Any] = {}

    for sym in symbols:
        key = f"IBKR:{sym}"
        by_vs[key] = {
            "slippage_bps_p50": None,
            "ack_latency_ms_p95": latency_ms,
            "reject_rate": None,
            "partial_fill_rate": None,
            "flags": ["BOOTSTRAP_ONLY"],
        }

    payload: Dict[str, Any] = {
        "env_label": latency_env_label,
        "by_venue_symbol": by_vs,
        "worst_offenders": [],
        "flags": ([] if latency_env_label != "dangerous" else ["BROKER_ACK_LATENCY_HIGH"]),
        # hash_sha256 filled after we compute canonical hash
    }
    payload["hash_sha256"] = _sha256_prefixed({k: v for k, v in payload.items() if k != "hash_sha256"})
    return payload


# ----------------------------
# Write outputs
# ----------------------------

def publish_once(*, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Build latency first
    latency_state = build_latency_state(runtime_dir=runtime_dir)
    ttl_latency = _env_int("CHAD_LATENCY_STATE_TTL_SECONDS", 300)

    # Persist latency state
    latency_path = runtime_dir / "latency_state.json"
    write_runtime_state_json(latency_path, latency_state, ttl_seconds=int(ttl_latency), inject_ts=True)

    # Reload to ensure we capture the injected fields if needed
    latency_written = _read_json(latency_path)
    env_label = str(latency_written.get("env_label") or latency_state.get("env_label") or "unknown")
    broker_obj = latency_written.get("broker") if isinstance(latency_written.get("broker"), dict) else {}
    lat_ms = broker_obj.get("latency_ms")
    try:
        lat_ms_f = float(lat_ms) if lat_ms is not None else None
    except Exception:
        lat_ms_f = None

    # Build exec quality
    execq = build_execution_quality(runtime_dir=runtime_dir, latency_env_label=env_label, latency_ms=lat_ms_f)
    ttl_execq = _env_int("CHAD_EXECUTION_QUALITY_TTL_SECONDS", 300)

    execq_path = runtime_dir / "execution_quality.json"
    write_runtime_state_json(execq_path, execq, ttl_seconds=int(ttl_execq), inject_ts=True)

    LOG.info("published latency_state=%s ttl=%ss", latency_path, ttl_latency)
    LOG.info("published execution_quality=%s ttl=%ss", execq_path, ttl_execq)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Publish SSOT v5.0 execution_quality + latency_state runtime artifacts.")
    ap.add_argument("--once", action="store_true", help="Publish once and exit (default).")
    ap.add_argument("--loop", action="store_true", help="Loop forever, publishing at a fixed interval.")
    ap.add_argument("--interval-seconds", type=int, default=_env_int("CHAD_EXECQ_LOOP_INTERVAL_SECONDS", 60))
    ap.add_argument("--log-level", type=str, default=os.environ.get("CHAD_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    repo = _repo_root()
    runtime = _runtime_dir(repo)

    if args.loop:
        interval = int(max(10, args.interval_seconds))
        LOG.info("starting loop publisher: runtime_dir=%s interval=%ss", runtime, interval)
        while True:
            try:
                publish_once(runtime_dir=runtime)
            except Exception as e:
                # Fail-safe: never crash loop
                LOG.exception("publish failed (will retry): %s", e)
            time.sleep(interval)
        return 0

    # default: once
    try:
        publish_once(runtime_dir=runtime)
        return 0
    except Exception as e:
        LOG.exception("publish failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
