#!/usr/bin/env python3
"""
CHAD — Soak evidence writers (observability only).

Implements the three append-only ndjson writers specified by
``ops/pending_actions/SOAK_STATUS_HISTORY_WRITER_2026-06-20.md``:

1. ``emit_status_history``       — PRIMARY, ``soak_status_history.v1``
2. ``emit_signal_router_emissions`` — COMPANION #2, ``soak_signal_router.v1``
3. ``emit_entry_intent_audit``   — COMPANION #3, ``soak_entry_intent.v1``

Hard contract (PA §5, §8):
- **Pure observers.** They READ the seven runtime authority/state files plus the
  in-memory RoutedSignal/intent objects. They WRITE *only* under ``data/soak/``.
  They MUST NOT mutate any ``runtime/`` gating state, ``ready_for_live``,
  ``allow_ibkr_live``, ``epoch_state.json``, the position guard, trade-closer
  queues, or any source file, and MUST NOT alter the host's return value.
- **Best-effort + isolated.** Every public writer wraps its entire body in a
  ``try/except`` that NEVER raises into the host. A writer failure (disk full,
  permission, serialization error) cannot corrupt, delay, or alter its host's
  primary job. The host attach points add a second, redundant ``try/except``.
- **Append-only.** Dated ndjson, one ``json.dumps(...) + "\\n"`` per row, append
  mode only. Never truncates, never opens a source file for write.
- ``data/soak/`` is created on first run (``mkdir(parents=True, exist_ok=True)``);
  it is the only directory these writers create and it is outside ``runtime/``.

This module is NOT the RED-window classifier (a separate, later PA). It produces
no verdict and counts no soak session.

ACTIVATION GATE (default OFF)
-----------------------------
All three writers are **inert unless** the environment flag
``CHAD_SOAK_EVIDENCE_WRITERS`` is truthy (``1``/``true``/``yes``/``on``). The hot-path
host files (lifecycle_truth_publisher, signal_router, live_loop) are executed in
place by the running services, so editing them would otherwise begin appending to
live ``data/soak/`` on the very next 60s tick — without any restart. The gate keeps
the in-tree code a pure no-op in production until the operator explicitly opts in,
which is exactly the PA §6 one-way ordering ("writer live → soak clock starts →
classifier can grade") and honours the governance posture of no unreviewed live
writes. The operator activates it at deploy time (e.g. a systemd drop-in
``Environment=CHAD_SOAK_EVIDENCE_WRITERS=1``, prepared as a Pending Action), then
confirms first-row sanity per PA §7. Tests set the flag explicitly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from chad.utils.runtime_json import stable_json_dumps, utc_now, utc_now_iso

LOG = logging.getLogger("chad.ops.soak.evidence_writers")

# ----------------------------------------------------------------------------
# Schema identities (the field names + schema_version strings are the contract)
# ----------------------------------------------------------------------------
SCHEMA_STATUS_HISTORY = "soak_status_history.v1"
SCHEMA_SIGNAL_ROUTER = "soak_signal_router.v1"
SCHEMA_ENTRY_INTENT = "soak_entry_intent.v1"

WRITER_IDENTITY_STATUS_HISTORY = "chad.ops.soak.status_history_writer"

# PRIMARY source files (PA §3.1 field-mapping table). Order is the canonical
# source_sha256 / read_errors order.
_PRIMARY_SOURCE_FILES = (
    "positions_truth.json",
    "reconciliation_state.json",
    "scr_state.json",
    "dynamic_caps.json",
    "profit_lock_state.json",
    "stop_bus.json",
    "tier_state.json",
)

_SOAK_SUBDIR = "soak"

# Activation gate — default OFF (see module docstring). The in-tree writers are
# no-ops in production until the operator sets this flag at deploy time.
_ACTIVATION_ENV = "CHAD_SOAK_EVIDENCE_WRITERS"
_TRUTHY = {"1", "true", "yes", "on"}


def writers_enabled() -> bool:
    """True iff the soak evidence writers are activated via env (default False)."""
    return str(os.environ.get(_ACTIVATION_ENV, "")).strip().lower() in _TRUTHY


# ----------------------------------------------------------------------------
# Small path / time helpers (mirror the established CHAD conventions; no new
# dependency on private helpers in other modules).
# ----------------------------------------------------------------------------
def _default_data_dir() -> Path:
    """Resolve ``data/`` the same way the codebase does elsewhere.

    Honors ``CHAD_DATA_DIR`` then ``CHAD_ROOT`` (so tests/operators can redirect
    writes), else falls back to the repo root inferred from this file's location
    (chad/ops/soak/evidence_writers.py -> parents[3] == repo root).
    """
    dd = str(os.environ.get("CHAD_DATA_DIR", "")).strip()
    if dd:
        return Path(dd).expanduser().resolve()
    root = str(os.environ.get("CHAD_ROOT", "")).strip()
    if root:
        return (Path(root).expanduser() / "data").resolve()
    return (Path(__file__).resolve().parents[3] / "data").resolve()


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    st = value.strip()
    if not st:
        return None
    if st.endswith("Z"):
        st = st[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(st).astimezone(timezone.utc)
    except Exception:
        return None


def _date_compact(ts_iso: str) -> str:
    """UTC ``YYYYMMDD`` for dated rotation, from the row clock."""
    dt = _parse_ts(ts_iso) or utc_now()
    return dt.strftime("%Y%m%d")


def _age_seconds(row_ts: Any, source_ts: Any) -> Optional[int]:
    a = _parse_ts(row_ts)
    b = _parse_ts(source_ts)
    if a is None or b is None:
        return None
    return int((a - b).total_seconds())


def _sha256_prefixed(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    """Append exactly one ndjson row (PA §5.6: one dumps + one write per row).

    This is the sole disk-write primitive for all three writers and the single
    point the isolation tests force to raise.
    """
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def _soak_out_path(data_dir: Path, prefix: str, ts_iso: str) -> Path:
    out_dir = Path(data_dir) / _SOAK_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)  # PA §5.5 — only dir we create
    return out_dir / f"{prefix}_{_date_compact(ts_iso)}.ndjson"


def _value(obj: Any) -> Any:
    """Best-effort scalar extraction for enum-or-str fields (e.g. SignalSide)."""
    if obj is None:
        return None
    v = getattr(obj, "value", None)
    if v is not None:
        return v
    return obj


# ----------------------------------------------------------------------------
# PRIMARY — status-history row builder + writer
# ----------------------------------------------------------------------------
def build_status_history_row(
    *,
    runtime_dir: Path,
    ts_utc: str,
    truth_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build one ``soak_status_history.v1`` row from the seven runtime sources.

    Pure: reads only; never writes. Per-file read failures are recorded in
    ``read_errors`` and the row is still produced (PA §5.2 — a partial row beats
    no row).
    """
    runtime_dir = Path(runtime_dir)
    paths: Dict[str, Path] = {name: runtime_dir / name for name in _PRIMARY_SOURCE_FILES}
    if truth_path is not None:
        # Read the just-written positions_truth.json exactly (PA §4.1).
        paths["positions_truth.json"] = Path(truth_path)

    objs: Dict[str, Dict[str, Any]] = {}
    source_sha256: Dict[str, str] = {}
    read_errors: List[str] = []
    for name in _PRIMARY_SOURCE_FILES:
        try:
            text = paths[name].read_text(encoding="utf-8")
            source_sha256[name] = _sha256_prefixed(text)
            parsed = json.loads(text)
            objs[name] = parsed if isinstance(parsed, dict) else {}
        except Exception:
            read_errors.append(name)
            source_sha256[name] = ""
            objs[name] = {}

    pt = objs["positions_truth.json"]
    rc = objs["reconciliation_state.json"]
    scr = objs["scr_state.json"]
    dc = objs["dynamic_caps.json"]
    pl = objs["profit_lock_state.json"]
    sb = objs["stop_bus.json"]
    tier = objs["tier_state.json"]

    # sizing_digest (criterion 4): sha256 of canonical-JSON(strategy_caps).
    strategy_caps = dc.get("strategy_caps")
    sizing_digest = _sha256_prefixed(stable_json_dumps(strategy_caps)) if isinstance(strategy_caps, dict) else ""

    broker_publish_ts = pt.get("ts_utc")

    return {
        "schema_version": SCHEMA_STATUS_HISTORY,
        "ts_utc": ts_utc,
        "writer_identity": WRITER_IDENTITY_STATUS_HISTORY,
        "cadence_source": "lifecycle_truth_publisher",

        "broker_authority_status": pt.get("broker_authority_status"),
        "broker_authority_as_of_utc": pt.get("as_of_event_ts_utc"),
        "broker_authority_publish_ts_utc": broker_publish_ts,
        "broker_authority_age_seconds": _age_seconds(ts_utc, broker_publish_ts),

        "reconciliation_status": rc.get("status"),
        "reconciliation_ts_utc": rc.get("ts_utc"),

        "scr_band": scr.get("state"),
        "scr_ts_utc": scr.get("ts_utc"),

        "sizing_digest": sizing_digest,
        "dynamic_caps_ts_utc": dc.get("ts_utc"),

        "profit_lock_mode": pl.get("mode"),
        "profit_lock_sizing_factor": pl.get("sizing_factor"),
        "profit_lock_ts_utc": pl.get("ts_utc"),

        # stop_bus.json has NO ts_utc (PA §5.4): faithful proxy is the
        # triggered_at / cleared_at pair plus active. None normalized to "".
        "stop_bus_active": sb.get("active"),
        "stop_bus_triggered_at": sb.get("triggered_at") or "",
        "stop_bus_cleared_at": sb.get("cleared_at") or "",

        "tier_name": tier.get("tier_name"),
        "tier_ts_utc": tier.get("ts_utc"),

        "source_sha256": source_sha256,
        "read_errors": read_errors,
    }


def emit_status_history(
    *,
    repo_root: Path,
    runtime_dir: Path,
    data_dir: Path,
    truth_path: Optional[Path] = None,
    ts_utc: Optional[str] = None,
) -> None:
    """PRIMARY writer. Best-effort + isolated — NEVER raises into the host.

    Appends one ``soak_status_history.v1`` row to
    ``data/soak/status_history_<YYYYMMDD>.ndjson``. ``repo_root`` is accepted for
    signature fidelity with the attach point (PA §4.1) though path resolution
    flows from ``runtime_dir`` / ``data_dir``.
    """
    try:
        if not writers_enabled():
            return
        row_ts = ts_utc or utc_now_iso()
        row = build_status_history_row(
            runtime_dir=runtime_dir,
            ts_utc=row_ts,
            truth_path=truth_path,
        )
        out_path = _soak_out_path(Path(data_dir), "status_history", row_ts)
        _append_row(out_path, row)
    except Exception as exc:  # noqa: BLE001 — best-effort, must not propagate
        LOG.warning("soak status_history emit failed (best-effort, ignored): %s", exc)


# ----------------------------------------------------------------------------
# COMPANION #2 — signal-router emission log
# ----------------------------------------------------------------------------
def build_signal_router_row(routed_signal: Any, ts_utc: str) -> Dict[str, Any]:
    rs = routed_signal
    srcs = getattr(rs, "source_strategies", ()) or ()
    return {
        "schema_version": SCHEMA_SIGNAL_ROUTER,
        "ts_utc": ts_utc,
        "symbol": getattr(rs, "symbol", None),
        "side": _value(getattr(rs, "side", None)),
        "primary_strategy": getattr(rs, "primary_strategy", None),
        "source_strategies": [_value(s) for s in srcs],
        "net_size": float(getattr(rs, "net_size", 0.0) or 0.0),
    }


def emit_signal_router_emissions(
    routed: Iterable[Any],
    *,
    data_dir: Optional[Path] = None,
    ts_utc: Optional[str] = None,
) -> None:
    """COMPANION #2 writer. Best-effort + isolated — NEVER raises into the host.

    Appends one ``soak_signal_router.v1`` row per emitted RoutedSignal to
    ``data/soak/signal_router_emissions_<YYYYMMDD>.ndjson``. An empty/zero-signal
    route appends nothing.
    """
    try:
        if not writers_enabled():
            return
        rows = list(routed or [])
        if not rows:
            return
        row_ts = ts_utc or utc_now_iso()
        dd = Path(data_dir) if data_dir is not None else _default_data_dir()
        out_path = _soak_out_path(dd, "signal_router_emissions", row_ts)
        for rs in rows:
            _append_row(out_path, build_signal_router_row(rs, row_ts))
    except Exception as exc:  # noqa: BLE001 — best-effort, must not propagate
        LOG.warning("soak signal_router emit failed (best-effort, ignored): %s", exc)


# ----------------------------------------------------------------------------
# COMPANION #3 — ENTRY-intent audit log
# ----------------------------------------------------------------------------
def classify_intent_type(*, is_exit: bool, is_flip: bool, side: Any) -> str:
    """Classify an execution intent as ``ENTRY`` / ``EXIT`` / ``FLIP``.

    Mirrors the live_loop decision tree (PA §4.3): an exit-meta intent or an
    explicit ``EXIT``/``CLOSE`` side is EXIT; a flip is FLIP; everything else
    (the ``mark_position_open`` branch) is a fresh ENTRY. Criterion 3 counts only
    ``ENTRY``. Centralized here so the host and the tests grade identically.
    """
    side_str = str(_value(side) or "").upper()
    if is_exit or side_str in {"EXIT", "CLOSE"}:
        return "EXIT"
    if is_flip:
        return "FLIP"
    return "ENTRY"


def emit_entry_intent_audit(
    *,
    intent_type: str,
    symbol: Any,
    side: Any,
    strategy: Any,
    admitted: bool,
    data_dir: Optional[Path] = None,
    ts_utc: Optional[str] = None,
) -> None:
    """COMPANION #3 writer. Best-effort + isolated — NEVER raises into the host.

    Appends one ``soak_entry_intent.v1`` row to
    ``data/soak/entry_intent_audit_<YYYYMMDD>.ndjson``.
    ``intent_type ∈ {ENTRY, EXIT, FLIP}``; criterion 3 counts only ``ENTRY``.
    """
    try:
        if not writers_enabled():
            return
        row_ts = ts_utc or utc_now_iso()
        dd = Path(data_dir) if data_dir is not None else _default_data_dir()
        out_path = _soak_out_path(dd, "entry_intent_audit", row_ts)
        row = {
            "schema_version": SCHEMA_ENTRY_INTENT,
            "ts_utc": row_ts,
            "intent_type": intent_type,
            "symbol": _value(symbol),
            "side": _value(side),
            "strategy": _value(strategy),
            "admitted": bool(admitted),
        }
        _append_row(out_path, row)
    except Exception as exc:  # noqa: BLE001 — best-effort, must not propagate
        LOG.warning("soak entry_intent emit failed (best-effort, ignored): %s", exc)
