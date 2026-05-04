"""
chad/utils/quarantine.py

Shared quarantine awareness for runtime publishers.

Loads ``runtime/quarantine_manifest_*.json`` and returns the union of
invalid fill IDs and invalid trade record_hashes. Publishers that
derive runtime state from data/fills/ or data/trades/ ledgers must
consult this helper at read time so quarantined evidence cannot
re-pollute pnl_state, scr_state, strategy_health, or winner_scaling.

Manifest schema (advisory only — original ledgers remain unmodified):

    {
      "quarantined_at_utc": "...",
      "reason": "...",
      "invalid_fills": [{"fill_id": "...", ...}, ...],
      "invalid_trades": [{"record_hash": "...", ...}, ...]
    }

Multiple manifests are unioned via ``runtime/quarantine_manifest_*.json``
glob, so additional manifests can be dropped in by operators without
re-deploying code.

Fail-safe behavior:
  * No manifest files       -> empty sets (helper returns silently).
  * Corrupt JSON / bad shape -> log warning, return empty sets, do NOT
                                crash the publisher.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set, Tuple

LOG = logging.getLogger("chad.utils.quarantine")

# Default manifest glob — operator-friendly: drop a new file in
# runtime/ and the union expands automatically.
_MANIFEST_GLOB = "quarantine_manifest_*.json"


def _runtime_dir(runtime_dir: Optional[Path] = None) -> Path:
    if runtime_dir is not None:
        return Path(runtime_dir)
    env = os.environ.get("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "runtime"


def _iter_manifest_paths(runtime_dir: Path) -> Iterable[Path]:
    if not runtime_dir.is_dir():
        return ()
    return sorted(runtime_dir.glob(_MANIFEST_GLOB))


def _extract_strings(items: object, key: str) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(items, list):
        return out
    for entry in items:
        if isinstance(entry, Mapping):
            v = entry.get(key)
            if isinstance(v, str) and v:
                out.add(v)
        elif isinstance(entry, str) and entry:
            out.add(entry)
    return out


def get_quarantine_sets(
    runtime_dir: Optional[Path] = None,
) -> Tuple[Set[str], Set[str]]:
    """Return ``(invalid_fill_ids, invalid_trade_hashes)``.

    Loads the union of every ``runtime/quarantine_manifest_*.json``
    file. Missing directory or no manifests -> ``(set(), set())``.
    Corrupt or malformed manifests are logged once and ignored — the
    publisher must continue to run rather than crash.
    """
    fill_ids: Set[str] = set()
    trade_hashes: Set[str] = set()

    rdir = _runtime_dir(runtime_dir)
    for manifest_path in _iter_manifest_paths(rdir):
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning(
                "quarantine_manifest_unreadable path=%s err=%s — skipping",
                manifest_path,
                exc,
            )
            continue
        if not isinstance(doc, Mapping):
            LOG.warning(
                "quarantine_manifest_bad_shape path=%s — skipping",
                manifest_path,
            )
            continue
        fill_ids.update(_extract_strings(doc.get("invalid_fills"), "fill_id"))
        trade_hashes.update(_extract_strings(doc.get("invalid_trades"), "record_hash"))

    return fill_ids, trade_hashes


def is_record_quarantined(
    record: Mapping[str, object],
    invalid_fill_ids: Set[str],
    invalid_trade_hashes: Set[str],
) -> bool:
    """Return True when *record* should be excluded by publishers.

    A record is quarantined if any of:
      * top-level ``record_hash`` is in *invalid_trade_hashes*
      * ``payload.fill_id`` is in *invalid_fill_ids*
      * payload (or top-level) carries ``pnl_untrusted=True`` or a
        ``"pnl_untrusted"`` tag
    """
    if not isinstance(record, Mapping):
        return False

    rh = record.get("record_hash")
    if isinstance(rh, str) and rh in invalid_trade_hashes:
        return True

    payload = record.get("payload")
    if isinstance(payload, Mapping):
        fid = payload.get("fill_id")
        if isinstance(fid, str) and fid in invalid_fill_ids:
            return True
        if payload.get("pnl_untrusted") is True:
            return True
        tags = payload.get("tags")
        if isinstance(tags, list) and any(
            str(t).strip().lower() == "pnl_untrusted" for t in tags
        ):
            return True
        extra = payload.get("extra")
        if isinstance(extra, Mapping) and extra.get("pnl_untrusted") is True:
            return True

    if record.get("pnl_untrusted") is True:
        return True

    return False


__all__ = [
    "get_quarantine_sets",
    "is_record_quarantined",
]
