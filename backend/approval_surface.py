from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Constants / configuration
# =============================================================================

RUNTIME_DIR: Final[Path] = Path(
    os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/chad_finale/runtime")
).resolve()

APPROVALS_PATH: Final[Path] = Path(
    os.environ.get("CHAD_APPROVALS_PATH", str(RUNTIME_DIR / "pending_approvals.json"))
).resolve()

CONTROL_DIR: Final[Path] = Path(
    os.environ.get("CHAD_CONTROL_DIR", "/home/ubuntu/chad_finale/control")
).resolve()

PENDING_ACTIONS_DIR: Final[Path] = Path(
    os.environ.get("CHAD_PENDING_ACTIONS_DIR", str(CONTROL_DIR / "pending_actions"))
).resolve()

APPROVED_ACTIONS_DIR: Final[Path] = Path(
    os.environ.get("CHAD_APPROVED_ACTIONS_DIR", str(CONTROL_DIR / "approved_actions"))
).resolve()

REJECTED_ACTIONS_DIR: Final[Path] = Path(
    os.environ.get("CHAD_REJECTED_ACTIONS_DIR", str(CONTROL_DIR / "rejected_actions"))
).resolve()

DATA_ACTIONS_DIR: Final[Path] = Path(
    os.environ.get("CHAD_ACTIONS_LEDGER_DIR", "/home/ubuntu/chad_finale/data/actions")
).resolve()

DEFAULT_TTL_SECONDS: Final[int] = int(os.environ.get("CHAD_APPROVAL_TTL_SECONDS", "3600"))
MIN_TTL_SECONDS: Final[int] = 60
MAX_TTL_SECONDS: Final[int] = 86400
MAX_SUMMARY_LEN: Final[int] = 240
MAX_REASON_LEN: Final[int] = 240
SCHEMA_VERSION: Final[str] = "approvals.v2"

ApprovalKind = Literal["rebalance_execute", "rotation_execute", "custom"]
ApprovalStatus = Literal["pending", "approved", "denied", "expired", "cancelled"]

router = APIRouter(prefix="/approvals", tags=["approvals"])


# =============================================================================
# Time / hashing / JSON helpers
# =============================================================================

def _utc_now_epoch() -> int:
    return int(time.time())


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iso_from_epoch(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(epoch)))


def _canonical_bytes(obj: Any) -> bytes:
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload)
    digest = _sha256_hex(data)

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


def _append_ndjson(path: Path, record: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(record)
    digest = _sha256_hex(data)

    with path.open("ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

    return digest


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# =============================================================================
# Models
# =============================================================================

class CreateApprovalIn(BaseModel):
    kind: ApprovalKind
    summary: str = Field(..., min_length=3, max_length=MAX_SUMMARY_LEN)
    payload: Dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: int = Field(default=DEFAULT_TTL_SECONDS, ge=MIN_TTL_SECONDS, le=MAX_TTL_SECONDS)

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, v: str) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("summary_required")
        return s[:MAX_SUMMARY_LEN]


class DecideApprovalIn(BaseModel):
    decision: Literal["approve", "deny"] = Field(..., description="approve | deny")
    decided_by: str = Field(..., min_length=1, max_length=64)
    decision_reason: str = Field(default="", max_length=MAX_REASON_LEN)

    @field_validator("decided_by")
    @classmethod
    def _validate_decided_by(cls, v: str) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("decided_by_required")
        return s[:64]

    @field_validator("decision_reason")
    @classmethod
    def _validate_decision_reason(cls, v: str) -> str:
        return str(v or "").strip()[:MAX_REASON_LEN]


@dataclass(frozen=True)
class ApprovalStore:
    schema_version: str
    ts_utc: str
    ttl_seconds: int
    items: list[dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ts_utc": self.ts_utc,
            "ttl_seconds": self.ttl_seconds,
            "items": self.items,
        }


# =============================================================================
# Core store logic
# =============================================================================

def _ensure_dirs() -> None:
    APPROVALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_store(obj: Dict[str, Any]) -> ApprovalStore:
    items = obj.get("items")
    if not isinstance(items, list):
        items = []
    ttl = obj.get("ttl_seconds", DEFAULT_TTL_SECONDS)
    try:
        ttl_i = int(ttl)
    except Exception:
        ttl_i = DEFAULT_TTL_SECONDS
    ttl_i = min(max(ttl_i, MIN_TTL_SECONDS), MAX_TTL_SECONDS)
    ts_utc = str(obj.get("ts_utc") or _utc_now_iso()).strip() or _utc_now_iso()
    schema_version = str(obj.get("schema_version") or SCHEMA_VERSION).strip() or SCHEMA_VERSION
    return ApprovalStore(schema_version=schema_version, ts_utc=ts_utc, ttl_seconds=ttl_i, items=items)


def _read_store() -> ApprovalStore:
    return _normalize_store(_safe_read_json(APPROVALS_PATH))


def _request_id_for(kind: str, summary: str, payload: Dict[str, Any], ttl_seconds: int) -> str:
    basis = {
        "kind": kind,
        "summary": summary,
        "payload": payload,
        "ttl_seconds": int(ttl_seconds),
    }
    return _sha256_hex(_canonical_bytes(basis))[:24]


def _find(items: list[dict[str, Any]], request_id: str) -> Optional[dict[str, Any]]:
    rid = str(request_id or "").strip()
    for it in items:
        if str(it.get("request_id") or "") == rid:
            return it
    return None


def _item_expired(item: dict[str, Any], now_epoch: int) -> bool:
    try:
        exp = int(item.get("expires_epoch", 0))
    except Exception:
        return False
    return exp > 0 and now_epoch > exp


def _status_for_view(item: dict[str, Any], now_epoch: int) -> str:
    status = str(item.get("status") or "").strip().lower()
    if status == "pending" and _item_expired(item, now_epoch):
        return "expired"
    return status or "pending"


def _purged_items(items: list[dict[str, Any]], now_epoch: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if _item_expired(it, now_epoch):
            expired = dict(it)
            expired["status"] = "expired"
            out.append(expired)
        else:
            out.append(it)
    return out


def _write_store(store: ApprovalStore) -> str:
    payload = store.to_dict()
    payload["schema_version"] = SCHEMA_VERSION
    payload["ts_utc"] = _utc_now_iso()
    return _atomic_write_json(APPROVALS_PATH, payload)


# =============================================================================
# V5 bridge helpers
# =============================================================================

def _queue_path(status: str, request_id: str) -> Path:
    name = f"ACTION_{request_id}.json"
    if status == "pending":
        return PENDING_ACTIONS_DIR / name
    if status == "approved":
        return APPROVED_ACTIONS_DIR / name
    if status == "denied":
        return REJECTED_ACTIONS_DIR / name
    if status == "expired":
        return REJECTED_ACTIONS_DIR / name
    if status == "cancelled":
        return REJECTED_ACTIONS_DIR / name
    return PENDING_ACTIONS_DIR / name


def _remove_queue_mirrors(request_id: str) -> None:
    name = f"ACTION_{request_id}.json"
    for base in (PENDING_ACTIONS_DIR, APPROVED_ACTIONS_DIR, REJECTED_ACTIONS_DIR):
        p = base / name
        try:
            if p.exists():
                p.unlink()
        except Exception:
            continue


def _approval_to_action_doc(item: dict[str, Any]) -> Dict[str, Any]:
    request_id = str(item.get("request_id") or "").strip()
    status = str(item.get("status") or "").strip().lower()
    created_ts_utc = str(item.get("created_ts_utc") or _utc_now_iso())
    expires_ts_utc = str(item.get("expires_ts_utc") or "")
    decided_ts_utc = item.get("decided_ts_utc")
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}

    return {
        "action_id": request_id,
        "ts_utc": created_ts_utc,
        "expires_ts_utc": expires_ts_utc or None,
        "requested_by": "approvals_api",
        "permission_class": "live_operator",
        "type": "APPROVAL_REQUEST",
        "kind": str(item.get("kind") or "").strip(),
        "status": status,
        "summary": str(item.get("summary") or "").strip(),
        "payload": payload,
        "legacy_request_id": request_id,
        "approvals": ([] if status == "pending" else [{
            "decided_ts_utc": decided_ts_utc,
            "decided_by": item.get("decided_by"),
            "decision_reason": item.get("decision_reason"),
            "decision": status,
        }]),
        "expected_pre_state_hash": None,
        "expected_post_state_hash": None,
        "notes": "Bridge document generated from legacy /approvals API for SSOT v5 queue parity.",
    }


def _mirror_to_v5_queue(item: dict[str, Any]) -> str:
    request_id = str(item.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("missing_request_id_for_queue_mirror")

    _remove_queue_mirrors(request_id)
    doc = _approval_to_action_doc(item)
    path = _queue_path(str(item.get("status") or "").strip().lower(), request_id)
    return _atomic_write_json(path, doc)


def _ledger_path_for_today() -> Path:
    return DATA_ACTIONS_DIR / f"ACTIONS_{time.strftime('%Y%m%d', time.gmtime())}.ndjson"


def _append_action_ledger(*, event_type: str, item: dict[str, Any], actor: str, reason: str) -> str:
    request_id = str(item.get("request_id") or "").strip()
    doc = _approval_to_action_doc(item)

    record = {
        "ts_utc": _utc_now_iso(),
        "deployment_id": os.environ.get("DEPLOYMENT_ID", "primary"),
        "event_type": event_type,
        "actor": actor,
        "reason": reason,
        "request_id": request_id,
        "status": str(item.get("status") or "").strip().lower(),
        "kind": str(item.get("kind") or "").strip(),
        "summary": str(item.get("summary") or "").strip(),
        "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
        "decided_by": item.get("decided_by"),
        "decided_ts_utc": item.get("decided_ts_utc"),
        "decision_reason": item.get("decision_reason"),
        "action_doc": doc,
    }
    return _append_ndjson(_ledger_path_for_today(), record)


def _sync_bridge_for_item(*, event_type: str, item: dict[str, Any], actor: str, reason: str) -> Dict[str, str]:
    queue_sha = _mirror_to_v5_queue(item)
    ledger_sha = _append_action_ledger(event_type=event_type, item=item, actor=actor, reason=reason)
    return {
        "queue_sha256": queue_sha,
        "ledger_sha256": ledger_sha,
    }


# =============================================================================
# Public response helpers
# =============================================================================

def _list_response(store: ApprovalStore, items: list[dict[str, Any]], digest: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "path": str(APPROVALS_PATH),
        "sha256": digest,
        "count": len(items),
        "items": items,
    }


# =============================================================================
# Routes
# =============================================================================

@router.get("", response_model=Dict[str, Any])
def list_approvals() -> Dict[str, Any]:
    _ensure_dirs()
    now = _utc_now_epoch()
    store = _read_store()
    items = _purged_items(store.items, now)
    normalized = ApprovalStore(
        schema_version=SCHEMA_VERSION,
        ts_utc=_utc_now_iso(),
        ttl_seconds=store.ttl_seconds,
        items=items,
    )
    digest = _write_store(normalized)
    return _list_response(normalized, items, digest)


@router.get("/health", response_model=Dict[str, Any])
def approvals_health() -> Dict[str, Any]:
    return list_approvals()


@router.get("/pending", response_model=Dict[str, Any])
def approvals_pending() -> Dict[str, Any]:
    resp = list_approvals()
    items = resp.get("items") if isinstance(resp.get("items"), list) else []
    pending = [
        it for it in items
        if isinstance(it, dict) and _status_for_view(it, _utc_now_epoch()) == "pending"
    ]
    resp["count"] = len(pending)
    resp["items"] = pending
    return resp


@router.get("/latest", response_model=Dict[str, Any])
def approvals_latest() -> Dict[str, Any]:
    resp = list_approvals()
    items = resp.get("items") if isinstance(resp.get("items"), list) else []
    if not items:
        return {**resp, "latest": None}

    def _sort_key(it: dict[str, Any]) -> str:
        return str(it.get("created_ts_utc") or "")

    latest = sorted([it for it in items if isinstance(it, dict)], key=_sort_key)[-1]
    return {**resp, "latest": latest}


@router.post("", response_model=Dict[str, Any])
def create_approval(req: CreateApprovalIn) -> Dict[str, Any]:
    _ensure_dirs()
    now = _utc_now_epoch()
    store = _read_store()
    items = _purged_items(store.items, now)

    request_id = _request_id_for(req.kind, req.summary, req.payload, req.ttl_seconds)
    existing = _find(items, request_id)
    if existing is not None:
        return {
            "ok": True,
            "idempotent": True,
            "request_id": request_id,
            "status": _status_for_view(existing, now),
            "path": str(APPROVALS_PATH),
        }

    expires_epoch = now + int(req.ttl_seconds)
    item: dict[str, Any] = {
        "request_id": request_id,
        "kind": req.kind,
        "status": "pending",
        "summary": req.summary,
        "payload": req.payload,
        "created_ts_utc": _utc_now_iso(),
        "expires_ts_utc": _iso_from_epoch(expires_epoch),
        "expires_epoch": expires_epoch,
        "decided_ts_utc": None,
        "decided_by": None,
        "decision_reason": None,
    }
    items.append(item)

    normalized = ApprovalStore(
        schema_version=SCHEMA_VERSION,
        ts_utc=_utc_now_iso(),
        ttl_seconds=store.ttl_seconds,
        items=items,
    )
    digest = _write_store(normalized)
    bridge_meta = _sync_bridge_for_item(
        event_type="approval_created",
        item=item,
        actor="approvals_api",
        reason="create",
    )

    return {
        "ok": True,
        "idempotent": False,
        "request_id": request_id,
        "sha256": digest,
        "path": str(APPROVALS_PATH),
        "bridge": bridge_meta,
    }


@router.get("/{request_id}", response_model=Dict[str, Any])
def get_approval(request_id: str) -> Dict[str, Any]:
    _ensure_dirs()
    now = _utc_now_epoch()
    store = _read_store()
    items = _purged_items(store.items, now)

    normalized = ApprovalStore(
        schema_version=SCHEMA_VERSION,
        ts_utc=_utc_now_iso(),
        ttl_seconds=store.ttl_seconds,
        items=items,
    )
    digest = _write_store(normalized)

    item = _find(items, request_id)
    if item is None:
        raise HTTPException(status_code=404, detail="approval_not_found")

    item_view = dict(item)
    item_view["status"] = _status_for_view(item, now)

    return {
        "ok": True,
        "request_id": request_id,
        "sha256": digest,
        "path": str(APPROVALS_PATH),
        "item": item_view,
    }


@router.post("/{request_id}", response_model=Dict[str, Any])
def decide_approval(request_id: str, body: DecideApprovalIn) -> Dict[str, Any]:
    _ensure_dirs()
    now = _utc_now_epoch()
    store = _read_store()
    items = _purged_items(store.items, now)

    item = _find(items, request_id)
    if item is None:
        raise HTTPException(status_code=404, detail="approval_not_found")

    current_status = _status_for_view(item, now)
    if current_status != "pending":
        return {
            "ok": True,
            "idempotent": True,
            "request_id": request_id,
            "status": current_status,
            "path": str(APPROVALS_PATH),
        }

    decision = body.decision.strip().lower()
    new_status: ApprovalStatus = "approved" if decision == "approve" else "denied"

    item["status"] = new_status
    item["decided_ts_utc"] = _utc_now_iso()
    item["decided_by"] = body.decided_by
    item["decision_reason"] = body.decision_reason

    normalized = ApprovalStore(
        schema_version=SCHEMA_VERSION,
        ts_utc=_utc_now_iso(),
        ttl_seconds=store.ttl_seconds,
        items=items,
    )
    digest = _write_store(normalized)
    bridge_meta = _sync_bridge_for_item(
        event_type="approval_decided",
        item=item,
        actor=body.decided_by,
        reason=body.decision_reason or body.decision,
    )

    return {
        "ok": True,
        "idempotent": False,
        "request_id": request_id,
        "status": new_status,
        "sha256": digest,
        "path": str(APPROVALS_PATH),
        "bridge": bridge_meta,
    }
