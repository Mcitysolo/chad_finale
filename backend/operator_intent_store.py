"""
backend.operator_intent_store

Production-grade Operator Intent storage + TTL freshness enforcement.

Design goals:
- Single authoritative implementation for runtime/operator_intent.json
- Fail-closed safety: missing/corrupt/stale => DENY_ALL
- Concurrency-safe: shared lock for reads, exclusive lock for writes
- Atomic writes: write temp -> fsync -> replace
- Schema compatibility: supports legacy + canonical fields
- Strong validation and explicit freshness reasons (audit-grade)
- Minimal dependencies: stdlib + pydantic (already in CHAD)

Runtime file (canonical):
{
  "operator_mode": "EXIT_ONLY" | "ALLOW_LIVE" | "DENY_ALL",
  "operator_reason": "string",
  "ts_utc": "2026-02-10T18:52:07Z",
  "ttl_seconds": 3600
}

Legacy compatibility:
- mode / reason / updated_at_utc
- operator_mode / operator_reason / updated_at_utc
- operator_mode / operator_reason / ts_utc   (with ttl_seconds)
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

DEFAULT_TTL_SECONDS = 3600
MAX_TTL_SECONDS = 7 * 86400
MAX_REASON_LEN = 240
DEFAULT_PATH = Path("runtime/operator_intent.json")


class OperatorMode:
    ALLOW_LIVE = "ALLOW_LIVE"
    EXIT_ONLY = "EXIT_ONLY"
    DENY_ALL = "DENY_ALL"

    @classmethod
    def normalize(cls, raw: Any) -> str:
        s = str(raw or "").strip().upper()
        if s == "ALLOW":
            return cls.ALLOW_LIVE
        if s in (cls.ALLOW_LIVE, cls.EXIT_ONLY, cls.DENY_ALL):
            return s
        return ""


class OperatorIntentPayload(BaseModel):
    operator_mode: str = Field(..., description="ALLOW_LIVE | EXIT_ONLY | DENY_ALL")
    operator_reason: str = Field(..., description="Audit reason")
    ts_utc: str = Field(..., description="UTC ISO timestamp; must be parseable")
    ttl_seconds: int = Field(..., ge=1, description="TTL seconds; must be positive")

    @field_validator("operator_mode")
    @classmethod
    def _v_mode(cls, v: str) -> str:
        mode = OperatorMode.normalize(v)
        if not mode:
            raise ValueError(f"invalid operator_mode: {v!r}")
        return mode

    @field_validator("operator_reason")
    @classmethod
    def _v_reason(cls, v: str) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("operator_reason required")
        if len(s) > MAX_REASON_LEN:
            return s[:MAX_REASON_LEN]
        return s

    @field_validator("ts_utc")
    @classmethod
    def _v_ts(cls, v: str) -> str:
        _ = parse_ts_utc(v)
        return v

    @field_validator("ttl_seconds")
    @classmethod
    def _v_ttl(cls, v: int) -> int:
        iv = int(v)
        if iv <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if iv > MAX_TTL_SECONDS:
            raise ValueError(f"ttl_seconds too large (max {MAX_TTL_SECONDS})")
        return iv


@dataclass(frozen=True)
class FreshnessResult:
    ok: bool
    reason: str
    age_seconds: float
    ttl_seconds: float


@dataclass(frozen=True)
class OperatorIntentState:
    mode: str
    reason: str
    ts_utc: str
    ttl_seconds: int
    freshness: FreshnessResult

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "operator_mode": self.mode,
            "operator_reason": self.reason,
            "ts_utc": self.ts_utc,
            "ttl_seconds": int(self.ttl_seconds),
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_ts_utc(ts: Any) -> datetime:
    if not isinstance(ts, str) or not ts.strip():
        raise ValueError("missing_ts_utc")
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_freshness(*, ts_utc: str, ttl_seconds: int, now: Optional[datetime] = None) -> FreshnessResult:
    now_dt = now or datetime.now(timezone.utc)
    ts_dt = parse_ts_utc(ts_utc)
    age_s = (now_dt - ts_dt).total_seconds()
    ttl_s = float(int(ttl_seconds))

    if age_s < 0:
        return FreshnessResult(ok=False, reason="ts_in_future_clock_skew", age_seconds=age_s, ttl_seconds=ttl_s)
    if age_s > ttl_s:
        return FreshnessResult(ok=False, reason=f"expired age_s={age_s:.1f} ttl_s={ttl_s:.1f}", age_seconds=age_s, ttl_seconds=ttl_s)
    return FreshnessResult(ok=True, reason=f"fresh age_s={age_s:.1f} ttl_s={ttl_s:.1f}", age_seconds=age_s, ttl_seconds=ttl_s)


@contextmanager
def _locked_file(path: Path, *, exclusive: bool):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+", encoding="utf-8")
    try:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(f.fileno(), lock_type)
        yield f
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        if not path.is_file():
            return None, "missing"
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None, "not_a_dict"
        return obj, ""
    except json.JSONDecodeError:
        return None, "json_decode_error"
    except Exception as exc:
        return None, f"read_error:{type(exc).__name__}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _normalize_from_any_schema(obj: Dict[str, Any]) -> Tuple[Optional[OperatorIntentPayload], str]:
    mode_raw = obj.get("operator_mode", obj.get("mode"))
    reason_raw = obj.get("operator_reason", obj.get("reason"))
    ts_raw = obj.get("ts_utc", obj.get("updated_at_utc"))
    ttl_raw = obj.get("ttl_seconds")

    mode = OperatorMode.normalize(mode_raw)
    if not mode:
        return None, "invalid_or_missing_mode"

    reason = str(reason_raw or "").strip()[:MAX_REASON_LEN]
    if not reason:
        return None, "invalid_or_missing_reason"

    if ttl_raw is None:
        return None, "missing_ttl_seconds"
    try:
        ttl = int(ttl_raw)
    except Exception:
        return None, "invalid_ttl_seconds"

    ts = str(ts_raw or "").strip()
    if not ts:
        return None, "missing_ts_utc"

    try:
        payload = OperatorIntentPayload(
            operator_mode=mode,
            operator_reason=reason,
            ts_utc=ts,
            ttl_seconds=ttl,
        )
        return payload, ""
    except ValidationError as e:
        msg = e.errors()[0].get("msg", "unknown")
        return None, f"validation_error:{msg}"


class OperatorIntentStore:
    def __init__(
        self,
        *,
        path: Path = DEFAULT_PATH,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        env_ttl_var: str = "CHAD_OPERATOR_INTENT_TTL_SECONDS",
    ) -> None:
        self._path = Path(path)
        self._default_ttl_seconds = self._load_default_ttl(default_ttl_seconds, env_ttl_var)

    @staticmethod
    def _load_default_ttl(fallback: int, env_var: str) -> int:
        v = os.getenv(env_var)
        if not v:
            return int(fallback)
        try:
            iv = int(str(v).strip())
            if iv <= 0:
                return int(fallback)
            return min(iv, MAX_TTL_SECONDS)
        except Exception:
            return int(fallback)

    def load_fail_closed(self, *, now: Optional[datetime] = None) -> OperatorIntentState:
        now_dt = now or datetime.now(timezone.utc)

        with _locked_file(self._path, exclusive=False):
            obj, err = _safe_read_json(self._path)

        if obj is None:
            fres = FreshnessResult(ok=False, reason=f"operator_intent_{err}", age_seconds=float("inf"), ttl_seconds=0.0)
            return OperatorIntentState(
                mode=OperatorMode.DENY_ALL,
                reason=f"operator_intent_{err}",
                ts_utc="",
                ttl_seconds=0,
                freshness=fres,
            )

        payload, nerr = _normalize_from_any_schema(obj)
        if payload is None:
            fres = FreshnessResult(ok=False, reason=f"operator_intent_invalid:{nerr}", age_seconds=float("inf"), ttl_seconds=0.0)
            return OperatorIntentState(
                mode=OperatorMode.DENY_ALL,
                reason=f"operator_intent_invalid:{nerr}",
                ts_utc=str(obj.get("ts_utc") or obj.get("updated_at_utc") or ""),
                ttl_seconds=int(obj.get("ttl_seconds") or 0) if str(obj.get("ttl_seconds") or "").isdigit() else 0,
                freshness=fres,
            )

        fres = compute_freshness(ts_utc=payload.ts_utc, ttl_seconds=payload.ttl_seconds, now=now_dt)
        if not fres.ok:
            return OperatorIntentState(
                mode=OperatorMode.DENY_ALL,
                reason=f"operator_intent_stale_or_missing:{fres.reason}",
                ts_utc=payload.ts_utc,
                ttl_seconds=payload.ttl_seconds,
                freshness=fres,
            )

        return OperatorIntentState(
            mode=payload.operator_mode,
            reason=payload.operator_reason,
            ts_utc=payload.ts_utc,
            ttl_seconds=payload.ttl_seconds,
            freshness=fres,
        )

    def set_intent(
        self,
        *,
        mode: str,
        reason: str,
        ttl_seconds: Optional[int] = None,
        ts_utc: Optional[str] = None,
    ) -> OperatorIntentState:
        mode_norm = OperatorMode.normalize(mode)
        if not mode_norm:
            raise ValueError(f"invalid mode: {mode!r}")
        reason_s = str(reason or "").strip()
        if not reason_s:
            raise ValueError("reason required")

        ttl = int(ttl_seconds) if ttl_seconds is not None else int(self._default_ttl_seconds)
        ttl = max(1, min(ttl, MAX_TTL_SECONDS))
        ts = ts_utc or utc_now_iso()

        payload = OperatorIntentPayload(
            operator_mode=mode_norm,
            operator_reason=reason_s[:MAX_REASON_LEN],
            ts_utc=ts,
            ttl_seconds=ttl,
        )
        out = payload.model_dump()

        with _locked_file(self._path, exclusive=True):
            _atomic_write_json(self._path, out)

        fres = compute_freshness(ts_utc=payload.ts_utc, ttl_seconds=payload.ttl_seconds, now=datetime.now(timezone.utc))
        return OperatorIntentState(
            mode=payload.operator_mode,
            reason=payload.operator_reason,
            ts_utc=payload.ts_utc,
            ttl_seconds=payload.ttl_seconds,
            freshness=fres,
        )


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CHAD OperatorIntent store (fail-closed TTL enforcement).")
    ap.add_argument("--path", default=str(DEFAULT_PATH), help="Path to operator_intent.json (default: runtime/operator_intent.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="Show evaluated operator intent (fail-closed).")

    s_set = sub.add_parser("set", help="Set operator intent (atomic write).")
    s_set.add_argument("--mode", required=True, help="ALLOW_LIVE | EXIT_ONLY | DENY_ALL")
    s_set.add_argument("--reason", required=True, help="Audit reason")
    s_set.add_argument("--ttl-seconds", type=int, default=None, help="TTL seconds (default from env/DEFAULT_TTL_SECONDS)")

    args = ap.parse_args()
    store = OperatorIntentStore(path=Path(args.path))

    if args.cmd == "show":
        st = store.load_fail_closed()
        print(json.dumps(
            {
                "mode": st.mode,
                "reason": st.reason,
                "ts_utc": st.ts_utc,
                "ttl_seconds": st.ttl_seconds,
                "fresh_ok": st.freshness.ok,
                "fresh_reason": st.freshness.reason,
                "age_seconds": st.freshness.age_seconds,
            },
            indent=2,
            sort_keys=True,
        ))
        return 0

    if args.cmd == "set":
        st = store.set_intent(mode=args.mode, reason=args.reason, ttl_seconds=args.ttl_seconds)
        print(json.dumps(st.to_runtime_dict(), indent=2, sort_keys=True))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
