"""
chad/utils/epoch.py

Shared Paper-Epoch boundary loader for runtime publishers.

CHAD's SCR / strategy_health / winner_scaler publishers compute from
ledger files under data/trades/ and data/fills/. After an epoch reset
(operator-driven boundary, e.g. Paper Epoch 2 on 2026-05-04), the
publishers must restrict their performance pool to records whose
realised time falls strictly on or after ``epoch_started_at_utc``.

The boundary is read from ``runtime/epoch_state.json``:

    {
      "schema_version": "epoch_state.v1",
      "active_epoch": "CHAD_v8.9_Paper_Epoch_2",
      "epoch_started_at_utc": "2026-05-04T00:54:30Z",
      "paper_only": true,
      "ready_for_live": false,
      "previous_epoch_archive": "...",
      "quarantine_manifest": "..."
    }

Fail-safe behavior (legacy preserved):
  * No ``runtime/epoch_state.json``         -> ``load_epoch_state`` returns None.
  * Corrupt JSON / wrong shape              -> warning logged, returns None.
  * Missing/unparseable ``epoch_started_at_utc`` -> warning logged, returns None.

Returning None means the publisher must apply its legacy (pre-epoch)
behavior: the boundary is opt-in by file presence so existing test
fixtures and historical replays continue to work unchanged.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

LOG = logging.getLogger("chad.utils.epoch")

EPOCH_STATE_FILENAME = "epoch_state.json"
SCHEMA_VERSION = "epoch_state.v1"


@dataclass(frozen=True)
class EpochState:
    """In-memory view of ``runtime/epoch_state.json``."""

    active_epoch: str
    epoch_started_at: datetime  # tz-aware UTC
    epoch_started_at_raw: str
    paper_only: bool
    ready_for_live: bool
    previous_epoch_archive: str
    quarantine_manifest: str
    source_path: Path


def _runtime_dir(runtime_dir: Optional[Path] = None) -> Path:
    if runtime_dir is not None:
        return Path(runtime_dir)
    env = os.environ.get("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "runtime"


def _parse_iso8601_utc(s: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a tz-aware UTC datetime.

    Accepts trailing ``Z`` (UTC zulu) and offset-aware forms. Returns
    ``None`` on any parse failure — callers must treat None as "no
    boundary known".
    """
    if not isinstance(s, str) or not s:
        return None
    txt = s.strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_epoch_state(runtime_dir: Optional[Path] = None) -> Optional[EpochState]:
    """Return the active ``EpochState`` or ``None`` if no boundary exists.

    Fail-safe: any error path (missing file / corrupt JSON / missing
    fields / unparseable timestamp) returns ``None`` and emits a
    warning. The caller is expected to fall back to legacy behavior.
    """
    rdir = _runtime_dir(runtime_dir)
    path = rdir / EPOCH_STATE_FILENAME
    if not path.is_file():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
        doc: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("epoch_state_unreadable path=%s err=%s — legacy behavior", path, exc)
        return None
    if not isinstance(doc, Mapping):
        LOG.warning("epoch_state_bad_shape path=%s — legacy behavior", path)
        return None

    started_raw = doc.get("epoch_started_at_utc")
    started_dt = _parse_iso8601_utc(started_raw) if isinstance(started_raw, str) else None
    if started_dt is None:
        LOG.warning(
            "epoch_state_missing_or_bad_timestamp path=%s value=%r — legacy behavior",
            path,
            started_raw,
        )
        return None

    return EpochState(
        active_epoch=str(doc.get("active_epoch") or ""),
        epoch_started_at=started_dt,
        epoch_started_at_raw=str(started_raw),
        paper_only=bool(doc.get("paper_only", True)),
        ready_for_live=bool(doc.get("ready_for_live", False)),
        previous_epoch_archive=str(doc.get("previous_epoch_archive") or ""),
        quarantine_manifest=str(doc.get("quarantine_manifest") or ""),
        source_path=path,
    )


def record_realized_time(record: Mapping[str, Any]) -> Optional[datetime]:
    """Best-effort UTC timestamp for an NDJSON trade or fill record.

    Preference order:
      1. ``payload.exit_time_utc``  (closed trade realised)
      2. ``payload.fill_time_utc``  (fill stamp)
      3. ``payload.entry_time_utc`` (trade open)
      4. top-level ``timestamp_utc`` (record append time)

    Returns ``None`` if no usable timestamp is found. Callers using
    epoch boundary filtering should treat ``None`` as "include" — i.e.
    when in doubt, do not silently drop records on a missing stamp; let
    other hygiene (quarantine, untrusted, nonfinite) handle them.
    """
    if not isinstance(record, Mapping):
        return None

    payload = record.get("payload")
    if isinstance(payload, Mapping):
        for key in ("exit_time_utc", "fill_time_utc", "entry_time_utc"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                dt = _parse_iso8601_utc(val)
                if dt is not None:
                    return dt

    top = record.get("timestamp_utc")
    if isinstance(top, str) and top:
        return _parse_iso8601_utc(top)

    return None


def is_pre_epoch(record: Mapping[str, Any], cutoff_utc: datetime) -> bool:
    """Return True iff *record* has a usable timestamp strictly before *cutoff_utc*.

    Records whose timestamp cannot be determined return False (= not
    pre-epoch); they pass through to the publisher's other filters.
    """
    ts = record_realized_time(record)
    if ts is None:
        return False
    return ts < cutoff_utc


__all__ = [
    "EpochState",
    "EPOCH_STATE_FILENAME",
    "SCHEMA_VERSION",
    "load_epoch_state",
    "record_realized_time",
    "is_pre_epoch",
]
