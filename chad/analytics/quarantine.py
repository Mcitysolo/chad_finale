"""
chad/analytics/quarantine.py
============================

Sidecar quarantine loader for placeholder-fill pollution.

Loads ``data/fills/quarantine_*.json`` and ``data/trades/quarantine_*.json``
sidecar manifests so consumers can filter poisoned records WITHOUT mutating
the append-only NDJSON ledgers under ``data/fills/`` and ``data/trades/``.

This module composes with the existing operator-managed manifest loader at
``chad.utils.quarantine`` — that helper handles ``runtime/quarantine_manifest_*.json``
and live ``pnl_untrusted`` scans of ``FILLS_*.ndjson``. The sidecar loader
here is the *forensic* surface: it pins specific historical records that
slipped through the upstream guard before it was added.

Public API
----------
- ``is_quarantined_fill(source_file, sequence_id, record_hash, fill_id)``
- ``is_quarantined_trade(source_file, sequence_id, record_hash)``
- ``load_fills_sidecar()`` / ``load_trades_sidecar()`` for inspection
- ``get_sidecar_exclusion_sets()`` returns ``(fill_ids, trade_hashes)`` so
  downstream consumers (trade_stats_engine, profit_lock, …) can union with
  ``chad.utils.quarantine.get_exclusion_sets``.

Failure mode: any unreadable / malformed sidecar is logged once at WARNING
and skipped. The loader fails *open* — a corrupt sidecar must not prevent
the trading lane from running.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Set, Tuple

LOG = logging.getLogger("chad.analytics.quarantine")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILLS_DIR = REPO_ROOT / "data" / "fills"
DEFAULT_TRADES_DIR = REPO_ROOT / "data" / "trades"

_FILLS_SIDECAR_GLOB = "quarantine_*.json"
_TRADES_SIDECAR_GLOB = "quarantine_*.json"


def _resolve_fills_dir(fills_dir: Optional[Path] = None) -> Path:
    if fills_dir is not None:
        return Path(fills_dir)
    env = os.environ.get("CHAD_FILLS_DIR", "").strip()
    if env:
        return Path(env)
    return DEFAULT_FILLS_DIR


def _resolve_trades_dir(trades_dir: Optional[Path] = None) -> Path:
    if trades_dir is not None:
        return Path(trades_dir)
    env = os.environ.get("CHAD_TRADES_DIR", "").strip()
    if env:
        return Path(env)
    return DEFAULT_TRADES_DIR


def _load_sidecar_files(folder: Path, pattern: str) -> List[Mapping[str, object]]:
    """Return parsed JSON docs for every sidecar matching *pattern*.

    Malformed / unreadable files are logged at WARNING and skipped.
    """
    docs: List[Mapping[str, object]] = []
    if not folder.is_dir():
        return docs
    for path in sorted(folder.glob(pattern)):
        try:
            raw = path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning(
                "quarantine_sidecar_unreadable path=%s err=%s — skipping", path, exc
            )
            continue
        if not isinstance(doc, Mapping):
            LOG.warning(
                "quarantine_sidecar_bad_shape path=%s — skipping", path
            )
            continue
        docs.append(doc)
    return docs


def load_fills_sidecar(
    fills_dir: Optional[Path] = None,
) -> List[Mapping[str, object]]:
    """Return every record from every ``data/fills/quarantine_*.json`` sidecar.

    Each returned record has at minimum ``source_file``, ``sequence_id``,
    ``record_hash``, ``fill_id``, ``reason`` (per the schema written by
    the operator). Records from sidecars with a missing ``invalid_fills``
    key contribute nothing.
    """
    out: List[Mapping[str, object]] = []
    folder = _resolve_fills_dir(fills_dir)
    for doc in _load_sidecar_files(folder, _FILLS_SIDECAR_GLOB):
        items = doc.get("invalid_fills")
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, Mapping):
                    out.append(entry)
    return out


def load_trades_sidecar(
    trades_dir: Optional[Path] = None,
) -> List[Mapping[str, object]]:
    """Return every record from every ``data/trades/quarantine_*.json`` sidecar."""
    out: List[Mapping[str, object]] = []
    folder = _resolve_trades_dir(trades_dir)
    for doc in _load_sidecar_files(folder, _TRADES_SIDECAR_GLOB):
        items = doc.get("invalid_trades")
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, Mapping):
                    out.append(entry)
    return out


def is_quarantined_fill(
    source_file: Optional[str] = None,
    sequence_id: Optional[int] = None,
    record_hash: Optional[str] = None,
    fill_id: Optional[str] = None,
    *,
    fills_dir: Optional[Path] = None,
) -> bool:
    """Return True if *any* of the supplied identifiers matches a sidecar entry.

    Match precedence (any one suffices):
      1. ``record_hash`` exact match
      2. ``fill_id`` exact match
      3. ``(source_file, sequence_id)`` pair exact match
    """
    if not (record_hash or fill_id or (source_file and sequence_id is not None)):
        return False
    for entry in load_fills_sidecar(fills_dir=fills_dir):
        if record_hash and entry.get("record_hash") == record_hash:
            return True
        if fill_id and entry.get("fill_id") == fill_id:
            return True
        if (
            source_file
            and sequence_id is not None
            and entry.get("source_file") == source_file
            and entry.get("sequence_id") == sequence_id
        ):
            return True
    return False


def is_quarantined_trade(
    source_file: Optional[str] = None,
    sequence_id: Optional[int] = None,
    record_hash: Optional[str] = None,
    *,
    trades_dir: Optional[Path] = None,
) -> bool:
    """Return True if *any* of the supplied identifiers matches a sidecar entry.

    Match precedence (any one suffices):
      1. ``record_hash`` exact match
      2. ``(source_file, sequence_id)`` pair exact match
    """
    if not (record_hash or (source_file and sequence_id is not None)):
        return False
    for entry in load_trades_sidecar(trades_dir=trades_dir):
        if record_hash and entry.get("record_hash") == record_hash:
            return True
        if (
            source_file
            and sequence_id is not None
            and entry.get("source_file") == source_file
            and entry.get("sequence_id") == sequence_id
        ):
            return True
    return False


def get_sidecar_exclusion_sets(
    fills_dir: Optional[Path] = None,
    trades_dir: Optional[Path] = None,
) -> Tuple[Set[str], Set[str]]:
    """Return ``(invalid_fill_ids, invalid_trade_hashes)`` from sidecars.

    ``invalid_fill_ids`` contains every ``fill_id`` listed in any
    ``data/fills/quarantine_*.json`` sidecar. The trade-side derivation
    intentionally also seeds ``invalid_fill_ids`` from each trade
    sidecar's ``fill_ids`` array so downstream consumers that match
    derived closed trades by ``payload.fill_ids`` (e.g.
    ``chad.analytics.trade_stats_engine``) automatically exclude phantom
    closed trades whose opening fills were placeholders.

    ``invalid_trade_hashes`` contains every ``record_hash`` listed in any
    ``data/trades/quarantine_*.json`` sidecar.

    Fail-open: any unreadable sidecar contributes nothing (already logged
    in ``_load_sidecar_files``).
    """
    fill_ids: Set[str] = set()
    trade_hashes: Set[str] = set()

    for entry in load_fills_sidecar(fills_dir=fills_dir):
        v = entry.get("fill_id")
        if isinstance(v, str) and v:
            fill_ids.add(v)

    for entry in load_trades_sidecar(trades_dir=trades_dir):
        v = entry.get("record_hash")
        if isinstance(v, str) and v:
            trade_hashes.add(v)
        # Trade sidecar entries may carry the underlying fill_ids — adding
        # them here means a consumer that filters by payload.fill_ids
        # (closed trades) will also drop the phantom record even if it
        # missed the trade record_hash match.
        fids = entry.get("fill_ids")
        if isinstance(fids, list):
            for f in fids:
                if isinstance(f, str) and f:
                    fill_ids.add(f)

    return fill_ids, trade_hashes


__all__ = [
    "is_quarantined_fill",
    "is_quarantined_trade",
    "load_fills_sidecar",
    "load_trades_sidecar",
    "get_sidecar_exclusion_sets",
]
