from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# Config
# =============================================================================

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
APPROVALS_PATH = Path(os.environ.get("CHAD_APPROVALS_PATH", str(RUNTIME_DIR / "pending_approvals.json")))
DEFAULT_TTL_SECONDS = int(os.environ.get("CHAD_APPROVAL_TTL_SECONDS", "3600"))  # 1h

ApprovalKind = Literal["rebalance_execute", "rotation_execute", "custom"]
ApprovalStatus = Literal["pending", "approved", "denied", "expired", "cancelled"]

router = APIRouter(prefix="/approvals", tags=["approvals"])


# =============================================================================
# Helpers (never raise outward)
# =============================================================================

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write_store(store: Dict[str, Any]) -> str:
    APPROVALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(store)
    digest = _sha256_hex(data)

    tmp = APPROVALS_PATH.with_suffix(APPROVALS_PATH.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, APPROVALS_PATH)

    try:
        dfd = os.open(str(APPROVALS_PATH.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

    return digest


def _read_store() -> Dict[str, Any]:
    try:
        if not APPROVALS_PATH.is_file():
            return {}
        obj = json.loads(APPROVALS_PATH.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _ensure_store() -> Dict[str, Any]:
    store = _read_store()
    if store.get("schema_version") != "approvals.v1":
        store = {
            "schema_version": "approvals.v1",
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": DEFAULT_TTL_SECONDS,
            "items": [],
        }
    if not isinstance(store.get("items"), list):
        store["items"] = []
    return store


def _purge_expired(items: List[dict], now_epoch: int) -> List[dict]:
    out: List[dict] = []
    for it in items:
        try:
            exp = int(it.get("expires_epoch", 0))
            if exp > 0 and now_epoch > exp:
                continue
        except Exception:
            pass
        out.append(it)
    return out


def _find(items: List[dict], request_id: str) -> Optional[dict]:
    rid = request_id.strip()
    for it in items:
        if str(it.get("request_id") or "") == rid:
            return it
    return None


# =============================================================================
# Models
# =============================================================================

class CreateApprovalIn(BaseModel):
    kind: ApprovalKind
    summary: str = Field(..., min_length=3, max_length=240)
    payload: Dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: int = Field(default=DEFAULT_TTL_SECONDS, ge=60, le=86400)


class DecideApprovalIn(BaseModel):
    decision: Literal["approve", "deny"] = Field(..., description="approve | deny")
    decided_by: str = Field(..., min_length=1, max_length=64)
    decision_reason: str = Field(default="", max_length=240)


# =============================================================================
# Routes
# =============================================================================

@router.get("", response_model=Dict[str, Any])
def list_approvals() -> Dict[str, Any]:
    store = _ensure_store()
    now = int(time.time())
    items: List[dict] = _purge_expired(store.get("items", []), now)
    store["items"] = items
    store["ts_utc"] = _utc_now_iso()
    digest = _atomic_write_store(store)
    return {"ok": True, "path": str(APPROVALS_PATH), "sha256": digest, "count": len(items), "items": items}


@router.post("", response_model=Dict[str, Any])
def create_approval(req: CreateApprovalIn) -> Dict[str, Any]:
    store = _ensure_store()
    now = int(time.time())
    items: List[dict] = _purge_expired(store.get("items", []), now)
    store["items"] = items

    created = _utc_now_iso()
    expires_epoch = now + int(req.ttl_seconds)
    expires_ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_epoch))

    # idempotent request_id from content
    basis = {"kind": req.kind, "summary": req.summary, "payload": req.payload, "ttl_seconds": int(req.ttl_seconds)}
    request_id = _sha256_hex(_canonical_bytes(basis))[:24]

    existing = _find(items, request_id)
    if existing is not None:
        return {"ok": True, "idempotent": True, "request_id": request_id, "status": existing.get("status"), "path": str(APPROVALS_PATH)}

    it = {
        "request_id": request_id,
        "kind": req.kind,
        "status": "pending",
        "summary": req.summary,
        "payload": req.payload,
        "created_ts_utc": created,
        "expires_ts_utc": expires_ts_utc,
        "expires_epoch": expires_epoch,
        "decided_ts_utc": None,
        "decided_by": None,
        "decision_reason": None,
    }
    items.append(it)

    store["ts_utc"] = _utc_now_iso()
    digest = _atomic_write_store(store)
    return {"ok": True, "idempotent": False, "request_id": request_id, "sha256": digest, "path": str(APPROVALS_PATH)}


@router.get("/{request_id}", response_model=Dict[str, Any])
def get_approval(request_id: str) -> Dict[str, Any]:
    store = _ensure_store()
    now = int(time.time())
    items: List[dict] = _purge_expired(store.get("items", []), now)
    store["items"] = items

    it = _find(items, request_id)
    store["ts_utc"] = _utc_now_iso()
    digest = _atomic_write_store(store)

    if it is None:
        raise HTTPException(status_code=404, detail="approval_not_found")

    return {"ok": True, "request_id": request_id, "sha256": digest, "path": str(APPROVALS_PATH), "item": it}


@router.post("/{request_id}", response_model=Dict[str, Any])
def decide_approval(request_id: str, body: DecideApprovalIn) -> Dict[str, Any]:
    store = _ensure_store()
    now = int(time.time())
    items: List[dict] = _purge_expired(store.get("items", []), now)
    store["items"] = items

    it = _find(items, request_id)
    if it is None:
        raise HTTPException(status_code=404, detail="approval_not_found")

    status = str(it.get("status") or "")
    if status != "pending":
        # idempotent: do not re-decide
        return {"ok": True, "idempotent": True, "request_id": request_id, "status": status, "path": str(APPROVALS_PATH)}

    decision = body.decision.strip().lower()
    new_status: ApprovalStatus = "approved" if decision == "approve" else "denied"

    it["status"] = new_status
    it["decided_ts_utc"] = _utc_now_iso()
    it["decided_by"] = body.decided_by
    it["decision_reason"] = body.decision_reason

    store["ts_utc"] = _utc_now_iso()
    digest = _atomic_write_store(store)
    return {"ok": True, "idempotent": False, "request_id": request_id, "status": new_status, "sha256": digest, "path": str(APPROVALS_PATH)}
