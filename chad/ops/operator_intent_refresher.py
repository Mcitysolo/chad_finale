#!/usr/bin/env python3
"""
CHAD — Operator Intent Authority Service
========================================

Purpose
-------
Canonical, production-grade authority for runtime/operator_intent.json.

This script replaces legacy refresh-only behavior with a hardened operator-intent
governance tool that can:

1) refresh
   Keep operator_intent.json fresh in non-live environments using canonical modes only.

2) set
   Explicitly set operator intent to:
      - ALLOW_LIVE
      - EXIT_ONLY
      - DENY_ALL

3) show
   Print current operator intent with freshness diagnostics.

4) verify
   Validate runtime/operator_intent.json against canonical contract and return strict
   exit codes suitable for systemd, CI, and forensic audit workflows.

Design goals
------------
- Canonical-only output (never emits legacy "ALLOW")
- Fail-closed semantics
- Deterministic behavior
- Strict validation
- Atomic writes delegated to authoritative store
- Clean audit logs
- Safe for systemd timer/service usage
- Future-proof for CHAD governance expansion

Operational rules
-----------------
- If execution mode is LIVE:
    * refresh refuses to widen permissions automatically
    * explicit set is allowed only when operator provides --allow-live-write
- In non-live mode:
    * refresh may write ALLOW_LIVE by policy because execution mode still blocks live
- All writes use backend.operator_intent_store.OperatorIntentStore
- Canonical mode names only:
    * ALLOW_LIVE
    * EXIT_ONLY
    * DENY_ALL

Exit codes
----------
0  = success / valid
2  = policy refusal / invalid inputs / unsafe operation
3  = verification failure
4  = runtime/state read failure
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final, Optional

from backend.operator_intent_store import OperatorIntentStore, OperatorMode

LOG = logging.getLogger("chad.ops.operator_intent_authority")

DEFAULT_RUNTIME_PATH: Final[Path] = Path("/home/ubuntu/chad_finale/runtime/operator_intent.json")
DEFAULT_TTL_SECONDS: Final[int] = 900
DEFAULT_REASON_NON_LIVE: Final[str] = "auto_refresh_allow_entries_non_live"
DEFAULT_REASON_EXIT_ONLY: Final[str] = "operator_exit_only"
DEFAULT_REASON_DENY_ALL: Final[str] = "operator_deny_all"
MAX_REASON_LEN: Final[int] = 240


class ExitCode(int, Enum):
    SUCCESS = 0
    INVALID = 2
    VERIFY_FAILED = 3
    STATE_ERROR = 4


class Command(str, Enum):
    REFRESH = "refresh"
    SET = "set"
    SHOW = "show"
    VERIFY = "verify"


@dataclass(frozen=True)
class RuntimeContext:
    repo_root: Path
    runtime_path: Path
    execution_mode: str
    hostname: str
    pid: int
    now_utc: str


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    mode: str
    reason: str
    ts_utc: str
    ttl_seconds: int
    freshness_ok: bool
    freshness_reason: str
    details: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


def normalize_reason(reason: str, default: str) -> str:
    text = str(reason or "").strip()
    if not text:
        text = default
    if len(text) > MAX_REASON_LEN:
        text = text[:MAX_REASON_LEN]
    return text


def resolve_repo_root() -> Path:
    candidates = []

    env_root = str(os.environ.get("CHAD_ROOT", "")).strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(here.parents)

    candidates.append(Path("/home/ubuntu/chad_finale"))

    for candidate in candidates:
        try:
            c = candidate.resolve()
        except Exception:
            continue
        if (c / "chad").is_dir() and (c / "runtime").is_dir():
            return c

    cwd = Path.cwd().resolve()
    if (cwd / "chad").is_dir() and (cwd / "runtime").is_dir():
        return cwd

    raise RuntimeError("repo_root_not_found")


def resolve_runtime_path(repo_root: Path) -> Path:
    env_path = str(os.environ.get("CHAD_OPERATOR_INTENT_PATH", "")).strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (repo_root / "runtime" / "operator_intent.json").resolve()


def normalize_exec_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"live", "paper", "dry_run"}:
        return mode
    return "dry_run"


def detect_execution_mode() -> str:
    from chad.execution.execution_config import get_execution_mode, ExecutionMode
    m = get_execution_mode()
    if m == ExecutionMode.IBKR_LIVE:
        return "live"
    if m == ExecutionMode.IBKR_PAPER:
        return "paper"
    return "dry_run"


def build_context() -> RuntimeContext:
    repo_root = resolve_repo_root()
    runtime_path = resolve_runtime_path(repo_root)
    return RuntimeContext(
        repo_root=repo_root,
        runtime_path=runtime_path,
        execution_mode=detect_execution_mode(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        now_utc=utc_now_iso(),
    )


def build_store(ctx: RuntimeContext) -> OperatorIntentStore:
    return OperatorIntentStore(
        path=ctx.runtime_path,
        default_ttl_seconds=env_int("CHAD_OPERATOR_INTENT_TTL_SECONDS", DEFAULT_TTL_SECONDS),
    )


def canonical_mode(raw: str) -> str:
    mode = str(raw or "").strip().upper()
    if mode == "ALLOW":
        mode = "ALLOW_LIVE"
    normalized = OperatorMode.normalize(mode)
    if not normalized:
        raise ValueError(f"invalid_operator_mode:{raw!r}")
    return normalized


def load_raw_json(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            raise FileNotFoundError(str(path))
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("json_not_object")
        return obj
    except Exception as exc:
        raise RuntimeError(f"raw_read_failed:{type(exc).__name__}:{exc}") from exc


def verify_state(ctx: RuntimeContext, *, require_canonical_mode: bool = True) -> VerificationResult:
    store = build_store(ctx)
    st = store.load_fail_closed()

    mode = str(st.mode or "")
    reason = str(st.reason or "")
    ts_utc = str(st.ts_utc or "")
    ttl_seconds = int(st.ttl_seconds or 0)
    freshness_ok = bool(st.freshness.ok)
    freshness_reason = str(st.freshness.reason or "")

    details: dict[str, Any] = {
        "repo_root": str(ctx.repo_root),
        "runtime_path": str(ctx.runtime_path),
        "execution_mode": ctx.execution_mode,
        "hostname": ctx.hostname,
        "pid": ctx.pid,
        "freshness": asdict(st.freshness),
    }

    ok = True

    if mode not in {OperatorMode.ALLOW_LIVE, OperatorMode.EXIT_ONLY, OperatorMode.DENY_ALL}:
        ok = False
        details["mode_error"] = f"non_canonical_mode:{mode!r}"

    if require_canonical_mode:
        raw = load_raw_json(ctx.runtime_path)
        raw_mode = str(raw.get("operator_mode", raw.get("mode", ""))).strip().upper()
        if raw_mode == "ALLOW":
            ok = False
            details["raw_mode_error"] = "legacy_mode_ALLOW_detected"

    if not reason.strip():
        ok = False
        details["reason_error"] = "empty_reason"

    if ttl_seconds <= 0:
        ok = False
        details["ttl_error"] = f"invalid_ttl:{ttl_seconds}"

    if not ts_utc.strip():
        ok = False
        details["ts_error"] = "missing_ts_utc"

    if not freshness_ok:
        ok = False
        details["freshness_error"] = freshness_reason

    return VerificationResult(
        ok=ok,
        mode=mode,
        reason=reason,
        ts_utc=ts_utc,
        ttl_seconds=ttl_seconds,
        freshness_ok=freshness_ok,
        freshness_reason=freshness_reason,
        details=details,
    )


def refresh_mode_for_execution(exec_mode: str) -> str:
    """
    Policy:
    - dry_run / paper => ALLOW_LIVE is acceptable because other gates still deny true live
    - live => refresh must refuse automatic widening
    """
    if exec_mode in {"dry_run", "paper"}:
        return OperatorMode.ALLOW_LIVE
    raise RuntimeError("refresh_refused_live_mode")


def write_intent(
    ctx: RuntimeContext,
    *,
    mode: str,
    reason: str,
    ttl_seconds: int,
    allow_live_write: bool,
) -> dict[str, Any]:
    mode_norm = canonical_mode(mode)
    reason_norm = normalize_reason(reason, default="operator_intent_update")
    ttl = int(ttl_seconds)

    if ttl <= 0:
        raise ValueError(f"invalid_ttl:{ttl}")

    if ctx.execution_mode == "live" and not allow_live_write:
        raise RuntimeError("write_refused_live_mode_without_allow_live_write")

    store = build_store(ctx)
    written = store.set_intent(mode=mode_norm, reason=reason_norm, ttl_seconds=ttl)

    payload = {
        "ok": True,
        "written": written.to_runtime_dict(),
        "context": asdict(ctx),
    }
    return payload


def _unexpired_hold(ctx: RuntimeContext) -> Optional[tuple[str, str, int]]:
    """Return (mode, reason, remaining_ttl_s) when the CURRENT state is an
    explicitly-set operator HOLD (EXIT_ONLY / DENY_ALL) whose own declared
    TTL has not yet elapsed; None otherwise.

    INCIDENT-0723 (D4a): the 10-minute auto-refresh timer used to rewrite
    ALLOW_LIVE unconditionally, stomping an operator-granted EXIT_ONLY hold
    within seconds of it being set. A hold is judged by the STATE's own
    ts_utc + ttl_seconds (e.g. a 24h hold), NOT the store's default
    freshness window (900s) — the whole point of a hold is to outlive it.
    Unreadable/absent/malformed state -> None (normal refresh proceeds)."""
    try:
        st = build_store(ctx).load_fail_closed()
        held = str(st.mode or "").strip().upper()
        if held not in (OperatorMode.EXIT_ONLY, OperatorMode.DENY_ALL):
            return None
        ts = datetime.fromisoformat(str(st.ts_utc).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        remaining = int(float(st.ttl_seconds or 0) - age_s)
        if remaining <= 0:
            return None
        return held, str(st.reason or ""), remaining
    except Exception:
        return None


def cmd_refresh(ctx: RuntimeContext, args: argparse.Namespace) -> int:
    try:
        mode = refresh_mode_for_execution(ctx.execution_mode)
    except Exception as exc:
        LOG.error(
            "refresh refused: execution_mode=%s reason=%s",
            ctx.execution_mode,
            str(exc),
        )
        return ExitCode.INVALID

    hold = _unexpired_hold(ctx)
    if hold is not None:
        held_mode, held_reason, remaining = hold
        payload = write_intent(
            ctx,
            mode=held_mode,
            reason=held_reason or DEFAULT_REASON_EXIT_ONLY,
            ttl_seconds=remaining,   # ts moves to now; expiry deadline holds
            allow_live_write=False,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        LOG.info(
            "operator_intent refresh PRESERVED operator hold mode=%s "
            "remaining_ttl_s=%s (INCIDENT-0723 D4a: auto-refresh must not "
            "widen an explicit hold) path=%s",
            held_mode,
            remaining,
            ctx.runtime_path,
        )
        return ExitCode.SUCCESS

    reason = normalize_reason(
        getattr(args, "reason", "") or os.environ.get("CHAD_OPERATOR_INTENT_REASON", ""),
        DEFAULT_REASON_NON_LIVE,
    )
    ttl = int(getattr(args, "ttl_seconds", 0) or env_int("CHAD_OPERATOR_INTENT_TTL_SECONDS", DEFAULT_TTL_SECONDS))

    payload = write_intent(
        ctx,
        mode=mode,
        reason=reason,
        ttl_seconds=ttl,
        allow_live_write=False,
    )

    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    LOG.info(
        "operator_intent refreshed mode=%s ttl_seconds=%s execution_mode=%s reason=%r path=%s",
        mode,
        ttl,
        ctx.execution_mode,
        reason,
        ctx.runtime_path,
    )
    return ExitCode.SUCCESS


def cmd_set(ctx: RuntimeContext, args: argparse.Namespace) -> int:
    mode = canonical_mode(args.mode)
    default_reason = {
        OperatorMode.ALLOW_LIVE: DEFAULT_REASON_NON_LIVE,
        OperatorMode.EXIT_ONLY: DEFAULT_REASON_EXIT_ONLY,
        OperatorMode.DENY_ALL: DEFAULT_REASON_DENY_ALL,
    }[mode]
    reason = normalize_reason(args.reason, default_reason)
    ttl = int(args.ttl_seconds)

    payload = write_intent(
        ctx,
        mode=mode,
        reason=reason,
        ttl_seconds=ttl,
        allow_live_write=bool(args.allow_live_write),
    )

    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    LOG.info(
        "operator_intent set mode=%s ttl_seconds=%s execution_mode=%s reason=%r path=%s allow_live_write=%s",
        mode,
        ttl,
        ctx.execution_mode,
        reason,
        ctx.runtime_path,
        bool(args.allow_live_write),
    )
    return ExitCode.SUCCESS


def cmd_show(ctx: RuntimeContext, _args: argparse.Namespace) -> int:
    store = build_store(ctx)
    st = store.load_fail_closed()

    payload = {
        "ok": True,
        "state": st.to_runtime_dict(),
        "freshness": asdict(st.freshness),
        "context": asdict(ctx),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return ExitCode.SUCCESS


def cmd_verify(ctx: RuntimeContext, args: argparse.Namespace) -> int:
    try:
        result = verify_state(ctx, require_canonical_mode=bool(args.require_canonical_mode))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"verify_exception:{type(exc).__name__}:{exc}",
                    "context": asdict(ctx),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.STATE_ERROR

    payload = {
        "ok": result.ok,
        "mode": result.mode,
        "reason": result.reason,
        "ts_utc": result.ts_utc,
        "ttl_seconds": result.ttl_seconds,
        "freshness_ok": result.freshness_ok,
        "freshness_reason": result.freshness_reason,
        "details": result.details,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return ExitCode.SUCCESS if result.ok else ExitCode.VERIFY_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CHAD Operator Intent Authority Service (canonical governance tool)."
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("CHAD_LOG_LEVEL", "INFO"),
        help="DEBUG | INFO | WARNING | ERROR",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_refresh = sub.add_parser(
        Command.REFRESH.value,
        help="Refresh operator_intent.json safely for non-live environments.",
    )
    p_refresh.add_argument(
        "--ttl-seconds",
        type=int,
        default=env_int("CHAD_OPERATOR_INTENT_TTL_SECONDS", DEFAULT_TTL_SECONDS),
        help="TTL seconds for refreshed state.",
    )
    p_refresh.add_argument(
        "--reason",
        default=os.environ.get("CHAD_OPERATOR_INTENT_REASON", DEFAULT_REASON_NON_LIVE),
        help="Audit reason for refreshed state.",
    )

    p_set = sub.add_parser(
        Command.SET.value,
        help="Explicitly set canonical operator intent.",
    )
    p_set.add_argument(
        "--mode",
        required=True,
        choices=[OperatorMode.ALLOW_LIVE, OperatorMode.EXIT_ONLY, OperatorMode.DENY_ALL],
        help="Canonical operator mode.",
    )
    p_set.add_argument(
        "--reason",
        default="",
        help="Audit reason.",
    )
    p_set.add_argument(
        "--ttl-seconds",
        type=int,
        default=env_int("CHAD_OPERATOR_INTENT_TTL_SECONDS", DEFAULT_TTL_SECONDS),
        help="TTL seconds for written state.",
    )
    p_set.add_argument(
        "--allow-live-write",
        action="store_true",
        help="Required to write operator intent while CHAD execution mode is live.",
    )

    p_show = sub.add_parser(
        Command.SHOW.value,
        help="Show current operator intent and freshness.",
    )

    p_verify = sub.add_parser(
        Command.VERIFY.value,
        help="Verify canonical operator_intent.json contract.",
    )
    p_verify.add_argument(
        "--require-canonical-mode",
        action="store_true",
        help="Fail verification if raw file still contains legacy operator_mode=ALLOW.",
    )

    return parser


def configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        ctx = build_context()
    except Exception as exc:
        LOG.error("failed to build runtime context: %s", exc)
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"context_build_failed:{type(exc).__name__}:{exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.STATE_ERROR

    try:
        command = Command(args.command)
    except Exception:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"unknown_command:{args.command!r}",
                    "context": asdict(ctx),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.INVALID

    try:
        if command is Command.REFRESH:
            return int(cmd_refresh(ctx, args))
        if command is Command.SET:
            return int(cmd_set(ctx, args))
        if command is Command.SHOW:
            return int(cmd_show(ctx, args))
        if command is Command.VERIFY:
            return int(cmd_verify(ctx, args))

        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"unhandled_command:{command.value}",
                    "context": asdict(ctx),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.INVALID

    except ValueError as exc:
        LOG.error("validation error: %s", exc)
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"validation_error:{exc}",
                    "context": asdict(ctx),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.INVALID

    except RuntimeError as exc:
        LOG.error("runtime policy refusal: %s", exc)
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "context": asdict(ctx),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ExitCode.INVALID

    except Exception as exc:
        LOG.exception("unexpected failure")
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"unexpected_exception:{type(exc).__name__}:{exc}",
                    "context": asdict(ctx),
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
        return ExitCode.STATE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
