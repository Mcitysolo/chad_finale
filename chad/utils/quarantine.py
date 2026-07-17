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

# Default fills glob — used by the untrusted-fill scan helper.
_FILLS_GLOB = "FILLS_*.ndjson"


def _runtime_dir(runtime_dir: Optional[Path] = None) -> Path:
    if runtime_dir is not None:
        return Path(runtime_dir)
    env = os.environ.get("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "runtime"


def _fills_dir(fills_dir: Optional[Path] = None) -> Path:
    if fills_dir is not None:
        return Path(fills_dir)
    env = os.environ.get("CHAD_FILLS_DIR", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "fills"


def _trade_closer_state_path(runtime_dir: Optional[Path] = None) -> Path:
    return _runtime_dir(runtime_dir) / "trade_closer_state.json"


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


def _payload_is_untrusted(payload: Mapping[str, object]) -> bool:
    """Return True if *payload* carries any pnl_untrusted marker.

    Markers checked (any one suffices):
      * ``payload.pnl_untrusted == True``
      * ``payload.extra.pnl_untrusted == True``
      * any tag in ``payload.tags`` equals ``"pnl_untrusted"`` (case-insensitive)
    """
    if not isinstance(payload, Mapping):
        return False
    if payload.get("pnl_untrusted") is True:
        return True
    extra = payload.get("extra")
    if isinstance(extra, Mapping) and extra.get("pnl_untrusted") is True:
        return True
    tags = payload.get("tags")
    if isinstance(tags, list) and any(
        str(t).strip().lower() == "pnl_untrusted" for t in tags
    ):
        return True
    return False


def get_untrusted_fill_ids_from_fills(
    fills_dir: Optional[Path] = None,
) -> Set[str]:
    """Scan ``data/fills/FILLS_*.ndjson`` and return the set of
    ``fill_id`` values whose record carries any ``pnl_untrusted`` marker.

    A fill counts as untrusted when the payload (or top-level record)
    has ``pnl_untrusted=True``, has a ``"pnl_untrusted"`` tag, or has
    ``extra.pnl_untrusted=True``. Original ledgers are not mutated.

    Fail-safe: missing directory or unreadable files yield an empty
    set; corrupt JSON lines are skipped silently. The publisher must
    keep running rather than crash.
    """
    out: Set[str] = set()
    fdir = _fills_dir(fills_dir)
    if not fdir.is_dir():
        return out
    for path in sorted(fdir.glob(_FILLS_GLOB)):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, Mapping):
                        continue
                    payload = rec.get("payload")
                    if not isinstance(payload, Mapping):
                        payload = rec
                    untrusted = _payload_is_untrusted(payload) or (
                        rec.get("pnl_untrusted") is True
                    )
                    if not untrusted:
                        continue
                    fid = payload.get("fill_id")
                    if isinstance(fid, str) and fid:
                        out.add(fid)
        except OSError as exc:
            LOG.warning(
                "untrusted_fills_scan_unreadable path=%s err=%s — skipping",
                path,
                exc,
            )
            continue
    return out


# B2 (FLIP-UNBLOCK 2026-07-17): the FIFO-lot backstop.
#
# The fills scan above can only quarantine ids that EXIST in data/fills/. The
# Epoch-3 reconciler's seed lots are minted straight into
# runtime/trade_closer_state.json with ids like ``RECON_ADOPT_UNH_<stamp>`` and
# never pass through a fills file — so a `get_exclusion_sets()` returning 4,782
# ids matched exactly ZERO of them. Both the flag path and the id backstop
# missed, and a seed-lot close banked fabricated PnL as clean evidence.
#
# This scan closes the backstop at its source: the lots themselves. Marker-driven
# (meta.pnl_untrusted / meta.scoring_excluded) rather than prefix-driven, so any
# future minter that stamps the markers is covered without touching this code.
# The RECON_ADOPT_ prefix is honoured as a belt-and-braces fallback for lots that
# predate the markers.
_SEED_LOT_ID_PREFIXES: Tuple[str, ...] = ("RECON_ADOPT_",)
_LOT_EXCLUSION_MARKER_KEYS: Tuple[str, ...] = ("pnl_untrusted", "scoring_excluded")


def _lot_is_untrusted(lot: Mapping[str, object]) -> bool:
    meta = lot.get("meta")
    if isinstance(meta, Mapping):
        for key in _LOT_EXCLUSION_MARKER_KEYS:
            if meta.get(key) is True:
                return True
    fid = lot.get("fill_id")
    if isinstance(fid, str):
        return any(fid.startswith(p) for p in _SEED_LOT_ID_PREFIXES)
    return False


def get_untrusted_fill_ids_from_fifo_lots(
    runtime_dir: Optional[Path] = None,
) -> Set[str]:
    """Return ``fill_id``s of FIFO lots in trade_closer_state.json marked unscoreable.

    These ids are what a derived closed trade carries in ``payload.fill_ids``, so
    quarantining them here makes the existing fill_ids check in
    ``trade_stats_engine._parse_ledger_file`` reject a seed-lot round-trip even if
    every flag-based defense upstream were bypassed.

    Fail-safe: a missing/corrupt state file yields an empty set — a publisher must
    keep running rather than crash.
    """
    out: Set[str] = set()
    path = _trade_closer_state_path(runtime_dir)
    try:
        if not path.is_file():
            return out
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("fifo_lot_scan_unreadable path=%s err=%s — skipping", path, exc)
        return out
    if not isinstance(data, Mapping):
        return out
    for row in data.get("queues") or []:
        if not isinstance(row, Mapping):
            continue
        for lot in row.get("lots") or []:
            if not isinstance(lot, Mapping):
                continue
            fid = lot.get("fill_id")
            if isinstance(fid, str) and fid and _lot_is_untrusted(lot):
                out.add(fid)
    return out


def get_exclusion_sets(
    runtime_dir: Optional[Path] = None,
    fills_dir: Optional[Path] = None,
    trades_dir: Optional[Path] = None,
) -> Tuple[Set[str], Set[str]]:
    """Return ``(invalid_fill_ids, invalid_trade_hashes)`` as the union
    of:
      * ``runtime/quarantine_manifest_*.json`` (operator-managed)
      * ``data/fills/FILLS_*.ndjson`` rows flagged ``pnl_untrusted``
        (live source-of-truth — Epoch 2 untrusted fills may not yet
        be in any manifest, so the fills scan is required to keep
        derived closed trades from re-entering SCR/pnl_state).
      * ``data/fills/quarantine_*.json`` and ``data/trades/quarantine_*.json``
        sidecars (forensic pin for historical pollution that slipped
        through before the upstream guard was added).

    Fail-safe: each loader is independent — a failure in one branch
    does not prevent the others from contributing.
    """
    fill_ids, trade_hashes = get_quarantine_sets(runtime_dir=runtime_dir)
    try:
        fill_ids = fill_ids | get_untrusted_fill_ids_from_fills(fills_dir=fills_dir)
    except Exception as exc:  # noqa: BLE001 — must never break publishers
        LOG.warning("untrusted_fills_scan_failed err=%s — using manifest only", exc)
    try:
        # B2: seed lots exist ONLY in trade_closer_state.json — no fills row will
        # ever carry a RECON_ADOPT_* id, so the fills scan above cannot see them.
        fill_ids = fill_ids | get_untrusted_fill_ids_from_fifo_lots(runtime_dir=runtime_dir)
    except Exception as exc:  # noqa: BLE001 — must never break publishers
        LOG.warning("fifo_lot_scan_failed err=%s — seed lots not pinned", exc)
    try:
        # Local import keeps a chad.utils -> chad.analytics dependency
        # contained to this single function instead of the module top.
        from chad.analytics.quarantine import get_sidecar_exclusion_sets

        side_fill_ids, side_trade_hashes = get_sidecar_exclusion_sets(
            fills_dir=fills_dir,
            trades_dir=trades_dir,
        )
        fill_ids = fill_ids | side_fill_ids
        trade_hashes = trade_hashes | side_trade_hashes
    except Exception as exc:  # noqa: BLE001 — must never break publishers
        LOG.warning("quarantine_sidecar_load_failed err=%s — using base sets only", exc)
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
      * any element of ``payload.fill_ids`` (closed-trade derived
        records) is in *invalid_fill_ids* — this catches derived
        closed trades that reference untrusted opening or closing
        fills, even when the closed-trade row itself carries no
        ``pnl_untrusted`` marker
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
        fids = payload.get("fill_ids")
        if isinstance(fids, list):
            for f in fids:
                if isinstance(f, str) and f in invalid_fill_ids:
                    return True
        if _payload_is_untrusted(payload):
            return True

    if record.get("pnl_untrusted") is True:
        return True

    return False


__all__ = [
    "get_quarantine_sets",
    "get_untrusted_fill_ids_from_fills",
    "get_exclusion_sets",
    "is_record_quarantined",
]
