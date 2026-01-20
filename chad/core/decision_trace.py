"""
chad.core.decision_trace

World-class, audit-first DecisionTrace recorder for CHAD.

Why this exists (CSB alignment)
------------------------------
CHAD must be able to reconstruct any decision end-to-end via an append-only audit trail.
DecisionTrace is the "black box recorder" of the system and is required even when no trades occur.
It is designed to be:

- Append-only (NDJSON)
- Partitioned by UTC day
- Deterministic and machine-readable
- Safe to write concurrently (best-effort process lock + fsync)
- Schema-versioned
- Minimal but extensible

This module does NOT execute trades. It only records decisions.

Operational goals
-----------------
- Every "decision cycle" should write one DecisionTrace record.
- LiveGate evaluation should be traceable (inputs + gate results + final decision).
- Files must be human-inspectable and safe to rotate by day.

Security goals
--------------
- Never write secrets (API keys, tokens, passwords).
- Only write paths/hashes/IDs and non-sensitive numeric values.

"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# ----------------------------
# Constants / Types
# ----------------------------

SCHEMA_VERSION: int = 1
LiveGateMode = Literal["DENY_ALL", "EXIT_ONLY", "ALLOW_LIVE"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def utc_day_yyyymmdd(ts: Optional[datetime] = None) -> str:
    d = (ts or utc_now()).astimezone(timezone.utc).date()
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


def stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON: compact separators, sorted keys.
    """
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


# ----------------------------
# Data model (schema v1)
# ----------------------------

@dataclass(frozen=True)
class GateResult:
    """
    One gate evaluation result.
    """
    name: str
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveGateDecisionTrace:
    """
    Snapshot of LiveGate result.
    """
    mode: LiveGateMode
    reasons: List[str]
    allow_exits_only: bool
    allow_ibkr_live: bool
    allow_ibkr_paper: bool


@dataclass(frozen=True)
class DecisionTraceRecord:
    """
    Canonical DecisionTrace record (one per decision cycle/evaluation).
    """
    schema_version: int
    trace_id: str
    cycle_id: str
    ts_utc: str

    # Minimal references to inputs (no secrets). These should point to runtime files.
    inputs: Dict[str, Any]

    # Gate results in deterministic order.
    gates: List[GateResult]

    # What was decided at the end.
    livegate: LiveGateDecisionTrace

    # Optional payloads (kept small). Large payloads should be referenced by path/hash.
    signal_intents: List[Dict[str, Any]] = field(default_factory=list)
    execution_plans: List[Dict[str, Any]] = field(default_factory=list)

    # Artifact references for quick navigation
    artifacts: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Writer (append-only NDJSON)
# ----------------------------

class DecisionTraceWriter:
    """
    Append-only DecisionTrace writer with daily partitioning.

    File naming:
        data/traces/decision_trace_YYYYMMDD.ndjson

    Concurrency:
        - best-effort file lock using an atomic lockfile
        - bounded wait with timeout
        - append + flush + fsync for durability

    This is robust enough for single-host systemd services (CHAD's expected deployment).
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        filename_prefix: str = "decision_trace",
        lock_timeout_s: float = 2.0,
        lock_poll_s: float = 0.02,
    ) -> None:
        self._base_dir = base_dir
        self._prefix = filename_prefix
        self._lock_timeout_s = float(lock_timeout_s)
        self._lock_poll_s = float(lock_poll_s)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _target_path_for_day(self, yyyymmdd: str) -> Path:
        return self._base_dir / f"{self._prefix}_{yyyymmdd}.ndjson"

    def _lock_path_for_day(self, yyyymmdd: str) -> Path:
        return self._base_dir / f".{self._prefix}_{yyyymmdd}.lock"

    def _acquire_lock(self, lock_path: Path) -> Optional[int]:
        """
        Attempt to acquire a lock by creating a lock file atomically.
        Returns an open fd on success, None on timeout.
        """
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(fd, f"pid={os.getpid()} ts_utc={utc_now_iso()}\n".encode("utf-8"))
                return fd
            except FileExistsError:
                if (time.time() - start) >= self._lock_timeout_s:
                    return None
                time.sleep(self._lock_poll_s)

    def _release_lock(self, lock_path: Path, fd: int) -> None:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                # Lock cleanup failure should not crash trading; worst-case we time out next attempt.
                pass

    def append(self, record: DecisionTraceRecord) -> Path:
        """
        Append a single DecisionTraceRecord to today's file.
        Returns the target file path.
        """
        yyyymmdd = utc_day_yyyymmdd()
        target = self._target_path_for_day(yyyymmdd)
        lock_path = self._lock_path_for_day(yyyymmdd)

        line = stable_json_dumps(_record_to_primitive(record)) + "\n"

        fd = self._acquire_lock(lock_path)
        if fd is None:
            # Fail-safe: do not crash core loop; write to a fallback file with no lock.
            # Still append-only; better to record than drop.
            fallback = self._base_dir / f"{self._prefix}_{yyyymmdd}_nolock.ndjson"
            _append_fsync(fallback, line)
            return fallback

        try:
            _append_fsync(target, line)
            return target
        finally:
            self._release_lock(lock_path, fd)


def _append_fsync(path: Path, line: str) -> None:
    """
    Durable append:
      open in append mode, write line, flush, fsync.

    This is intentionally conservative: it trades a small performance hit
    for operational safety and audit integrity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _record_to_primitive(record: DecisionTraceRecord) -> Dict[str, Any]:
    """
    Convert dataclasses to JSON-serializable primitives with controlled structure.
    """
    d = asdict(record)
    # Convert GateResult list properly (asdict already did, but keep for clarity)
    d["gates"] = [asdict(g) for g in record.gates]
    d["livegate"] = asdict(record.livegate)
    return d


# ----------------------------
# Public helpers (record creation)
# ----------------------------

def new_trace_id() -> str:
    return str(uuid.uuid4())


def new_cycle_id(ts: Optional[datetime] = None) -> str:
    """
    Canonical cycle_id format: YYYYMMDDTHHMMSSZ
    """
    dt = (ts or utc_now()).astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def build_decision_trace_record(
    *,
    trace_id: str,
    cycle_id: str,
    inputs: Dict[str, Any],
    gates: List[GateResult],
    livegate: LiveGateDecisionTrace,
    signal_intents: Optional[List[Dict[str, Any]]] = None,
    execution_plans: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> DecisionTraceRecord:
    """
    Build a schema-consistent DecisionTraceRecord.
    """
    return DecisionTraceRecord(
        schema_version=SCHEMA_VERSION,
        trace_id=trace_id,
        cycle_id=cycle_id,
        ts_utc=utc_now_iso(),
        inputs=dict(inputs),
        gates=list(gates),
        livegate=livegate,
        signal_intents=list(signal_intents or []),
        execution_plans=list(execution_plans or []),
        artifacts=dict(artifacts or {}),
    )


def default_writer() -> DecisionTraceWriter:
    """
    Default writer location: <repo>/data/traces/
    """
    # repo root is /home/ubuntu/chad_finale by convention
    # resolve from this file's location: chad/core/decision_trace.py -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    return DecisionTraceWriter(base_dir=repo_root / "data" / "traces")


# ----------------------------
# Optional: lightweight self-test (manual run)
# ----------------------------

def _demo() -> None:
    """
    Manual smoke:
      python -m chad.core.decision_trace
    """
    w = default_writer()
    tid = new_trace_id()
    cid = new_cycle_id()
    rec = build_decision_trace_record(
        trace_id=tid,
        cycle_id=cid,
        inputs={"example": "runtime/ibkr_status.json"},
        gates=[GateResult(name="STOP", passed=True, reason="not_engaged")],
        livegate=LiveGateDecisionTrace(
            mode="DENY_ALL",
            reasons=["demo"],
            allow_exits_only=False,
            allow_ibkr_live=False,
            allow_ibkr_paper=False,
        ),
        artifacts={"trace_file_hint": "data/traces/decision_trace_YYYYMMDD.ndjson"},
    )
    path = w.append(rec)
    print(f"Wrote DecisionTrace: {path}")


if __name__ == "__main__":
    _demo()
