#!/usr/bin/env python3
"""
chad/ops/paper_shadow_runner_wrapper_v2.py

CHAD Paper Shadow Runner Wrapper V2 (Phase 11 — Crash-Safe Run Journal)

Purpose
-------
Crash-safe evidence for paper_shadow_runner without modifying its safety-critical logic.

This wrapper:
- Writes start/finish records to SQLite via PaperRunJournal (WAL, fail-soft)
- Computes planned_hash for runtime/full_execution_cycle_last.json (if present)
- Runs paper_shadow_runner in PREVIEW or EXECUTE mode
- Captures bounded stdout/stderr for operator troubleshooting
- Emits stable machine JSON output (systemd / logs / dashboards)

Non-negotiables
---------------
- No broker calls beyond what paper_shadow_runner does
- No state mutation besides run journal writes (exec_state_paper.sqlite3)
- Fail-soft: wrapper never raises unhandled exceptions
- Hard timeout to prevent hung runs
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from chad.execution.paper_run_journal import PaperRunJournal

# ----------------------------
# Constants (paste-safe)
# ----------------------------

RUN_ID_TIME_FMT = "%Y%m%dT%H%M%SZ"
UTC_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(UTC_ISO_FMT)


def utc_now_run_id_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(RUN_ID_TIME_FMT)


def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def safe_json_last(stdout: str, limit_lines: int = 50) -> Dict[str, Any]:
    """
    Fallback parser: scan last JSON-looking line from stdout.
    Fail-soft: returns {}.
    """
    if not stdout:
        return {}
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return {}
    tail = lines[-limit_lines:] if len(lines) > limit_lines else lines
    for ln in reversed(tail):
        if ln.startswith("{") and ln.endswith("}"):
            obj = safe_json_load(ln)
            if obj is not None:
                return obj
    return {}


def safe_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    Prefer parsing full stdout as JSON (multi-line JSON supported).
    Fallback: scan last JSON-looking line.
    Fail-soft: returns {}.
    """
    if not stdout:
        return {}
    obj = safe_json_load(stdout.strip())
    if isinstance(obj, dict):
        return obj
    return safe_json_last(stdout)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate(s: str, max_chars: int) -> str:
    if not s:
        return ""
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...(truncated)"


def derive_outcome(result_hint: Dict[str, Any], rc: int) -> str:
    """
    Deterministic outcome categorization.

    IMPORTANT:
    - preview/no_orders/no_ledger must classify as preview_or_blocked
    - do not rely on result_hint existing; wrapper parses multi-line JSON now
    """
    if rc != 0:
        return "error_rc"

    safety = result_hint.get("safety") if isinstance(result_hint.get("safety"), dict) else {}
    actions = result_hint.get("actions") if isinstance(result_hint.get("actions"), dict) else {}

    if bool(safety.get("no_plan")):
        return "blocked_no_plan"

    op_mode = safety.get("operator_mode")
    if op_mode == "EXIT_ONLY":
        return "blocked_exit_only"
    if op_mode == "DENY_ALL":
        return "blocked_deny_all"

    if bool(safety.get("no_orders")) and bool(safety.get("no_ledger")):
        return "preview_or_blocked"

    if bool(safety.get("no_ledger")):
        return "no_ledger"

    try:
        logged = int(actions.get("trade_results_logged") or 0)
        if logged > 0:
            return "executed_logged"
    except Exception:
        pass

    return "ok_unknown"


# ----------------------------
# Dependency Injection Interfaces
# ----------------------------

@dataclass(frozen=True)
class RunConfig:
    repo_root: Path
    db_path: Path
    planned_path: Path
    timeout_seconds: int
    max_stdout_chars: int
    max_stderr_chars: int


class Runner:
    def run(self, args: Sequence[str], cwd: Path, timeout_seconds: int) -> Tuple[int, str, str]:
        raise NotImplementedError


class SubprocessRunner(Runner):
    def run(self, args: Sequence[str], cwd: Path, timeout_seconds: int) -> Tuple[int, str, str]:
        p = subprocess.run(
            list(args),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
            check=False,
        )
        return int(p.returncode), (p.stdout or ""), (p.stderr or "")


class Journal:
    def start(self, *, run_id: str, mode: str, planned_hash: str, notes: Dict[str, Any]) -> None:
        raise NotImplementedError

    def finish(self, *, run_id: str, outcome: str, error: str, notes: Dict[str, Any]) -> None:
        raise NotImplementedError


class PaperJournalAdapter(Journal):
    def __init__(self, journal: PaperRunJournal) -> None:
        self._j = journal

    def start(self, *, run_id: str, mode: str, planned_hash: str, notes: Dict[str, Any]) -> None:
        # fail-soft inside PaperRunJournal
        self._j.start(run_id=run_id, mode=mode, planned_hash=planned_hash, notes=notes)

    def finish(self, *, run_id: str, outcome: str, error: str, notes: Dict[str, Any]) -> None:
        self._j.finish(run_id=run_id, outcome=outcome, error=error, notes=notes)


# ----------------------------
# Core wrapper logic
# ----------------------------

def build_default_config() -> RunConfig:
    repo_root = Path(__file__).resolve().parents[2]  # .../chad_finale
    db_path = repo_root / "runtime" / "exec_state_paper.sqlite3"
    planned_path = repo_root / "runtime" / "full_execution_cycle_last.json"

    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None:
            return int(default)
        try:
            return int(str(v).strip())
        except Exception:
            return int(default)

    return RunConfig(
        repo_root=repo_root,
        db_path=db_path,
        planned_path=planned_path,
        timeout_seconds=_env_int("CHAD_PAPER_SHADOW_TIMEOUT_SECONDS", 240),
        max_stdout_chars=_env_int("CHAD_PAPER_SHADOW_MAX_STDOUT_CHARS", 6000),
        max_stderr_chars=_env_int("CHAD_PAPER_SHADOW_MAX_STDERR_CHARS", 2000),
    )


def make_run_id(prefix: str) -> str:
    # No fragile f-strings; deterministic string build
    return prefix + "_" + utc_now_run_id_stamp() + "_" + str(os.getpid())


def compute_planned_hash(planned_path: Path) -> str:
    try:
        if planned_path.is_file():
            return sha256_file(planned_path)
    except Exception:
        return ""
    return ""


def run_wrapper(mode: str, cfg: RunConfig, runner: Runner, journal: Journal) -> Dict[str, Any]:
    run_id = make_run_id("paper_shadow")
    planned_hash = compute_planned_hash(cfg.planned_path)

    journal.start(
        run_id=run_id,
        mode=mode,
        planned_hash=planned_hash,
        notes={
            "ts_utc": utc_now_iso(),
            "planned_path": str(cfg.planned_path),
            "planned_present": cfg.planned_path.is_file(),
            "cwd": str(cfg.repo_root),
            "mode": mode,
        },
    )

    cmd = [sys.executable, "-m", "chad.core.paper_shadow_runner"]
    if mode == "EXECUTE":
        cmd.append("--execute")
    else:
        cmd.append("--preview")

    rc = 0
    stdout = ""
    stderr = ""
    err_hint = ""

    try:
        rc, stdout, stderr = runner.run(cmd, cwd=cfg.repo_root, timeout_seconds=cfg.timeout_seconds)
    except Exception as exc:
        rc = 1
        err_hint = "wrapper_exec_error:" + type(exc).__name__ + ":" + str(exc)

    # Parse full stdout as JSON (multi-line supported)
    result_hint = safe_json_from_stdout(stdout)
    outcome = derive_outcome(result_hint, rc)

    stdout_trunc = truncate(stdout, cfg.max_stdout_chars)
    stderr_trunc = truncate(stderr, cfg.max_stderr_chars)

    # Finish journal (best-effort)
    journal.finish(
        run_id=run_id,
        outcome=outcome,
        error=(err_hint or stderr_trunc),
        notes={
            "ts_utc": utc_now_iso(),
            "rc": rc,
            "outcome": outcome,
            "report_path": result_hint.get("report_path") if isinstance(result_hint, dict) else None,
            "safety": result_hint.get("safety") if isinstance(result_hint.get("safety"), dict) else {},
            "actions": result_hint.get("actions") if isinstance(result_hint.get("actions"), dict) else {},
            "stdout_tail": stdout_trunc,
            "stderr_tail": stderr_trunc,
        },
    )

    return {
        "ok": True,
        "ts_utc": utc_now_iso(),
        "run_id": run_id,
        "mode": mode,
        "planned_hash": planned_hash,
        "rc": rc,
        "outcome": outcome,
        "db_path": str(cfg.db_path),
        "stdout_tail": stdout_trunc,
        "stderr_tail": stderr_trunc,
        "result_hint": result_hint,
    }


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CHAD Paper Shadow Runner Wrapper V2 (journals runs, no behavior change).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true", help="Run paper_shadow_runner in preview mode.")
    g.add_argument("--execute", action="store_true", help="Run paper_shadow_runner in execute mode (paper).")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    mode = "EXECUTE" if args.execute else "PREVIEW"

    cfg = build_default_config()
    journal = PaperJournalAdapter(PaperRunJournal(cfg.db_path))
    runner = SubprocessRunner()

    out = run_wrapper(mode, cfg, runner, journal)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
