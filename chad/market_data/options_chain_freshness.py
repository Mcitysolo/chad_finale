"""Options-chain freshness validator (OPTIONS-CHAIN-1).

A small, dependency-free read of two artefacts:
  - runtime/options_chains_cache.json          (the cache; success path)
  - runtime/options_chain_refresh_failure.json (the failure artefact)

This module is the single source of truth for:
  - "is the chain cache fresh enough to trust?"  → ``is_chain_fresh()``
  - "is there a recent, unresolved refresh failure?" → ``is_failure_artifact_fresh()``
  - aggregated verdict for strategy fail-closed → ``chain_usability()``

It is read-only. It never writes to runtime/.

Used by:
  - chad/strategies/alpha_options.py and chad/strategies/omega_momentum_options.py
    (fail-closed when chain is stale or failure artifact is fresh)
  - chad/ops/health_monitor_rules.py (loud alert when failure artifact is fresh)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CACHE_PATH = RUNTIME_DIR / "options_chains_cache.json"
FAILURE_ARTIFACT_PATH = RUNTIME_DIR / "options_chain_refresh_failure.json"

DEFAULT_CACHE_MAX_AGE_SECONDS = 26 * 3600  # 26 h (matches R17 rule's weekday gate)
DEFAULT_FAILURE_FRESH_SECONDS = 2 * 3600   # 2 h: a failure < 2 h old still blocks usage


@dataclass
class FreshnessVerdict:
    """Aggregated chain-cache verdict for strategy gating."""

    usable: bool
    reason: str
    cache_exists: bool
    cache_age_seconds: float | None
    cache_error: str | None
    cache_chains_count: int
    failure_artifact_exists: bool
    failure_artifact_age_seconds: float | None
    failure_artifact_reason: str | None


def _parse_iso_ts(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_chain_fresh(
    cache_path: Path = CACHE_PATH,
    max_age_seconds: float = DEFAULT_CACHE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Return (fresh, reason, details).

    Fresh iff: file exists, error field is empty, chains is non-empty dict,
    and ts_utc is within max_age_seconds of now.
    """
    now = now or datetime.now(timezone.utc)
    doc = _load_json(cache_path)
    details: dict[str, Any] = {
        "cache_exists": doc is not None,
        "cache_error": None,
        "cache_chains_count": 0,
        "cache_age_seconds": None,
    }
    if doc is None:
        return False, "cache_missing", details
    err = doc.get("error")
    if isinstance(err, str) and err.strip():
        details["cache_error"] = err
        return False, "cache_error_present", details
    chains = doc.get("chains") if isinstance(doc.get("chains"), dict) else {}
    details["cache_chains_count"] = len(chains)
    if not chains:
        return False, "cache_empty_chains", details
    ts = _parse_iso_ts(doc.get("ts_utc"))
    if ts is None:
        return False, "cache_ts_unparseable", details
    age = (now - ts).total_seconds()
    details["cache_age_seconds"] = age
    if age > max_age_seconds:
        return False, "cache_stale", details
    return True, "fresh", details


def is_failure_artifact_fresh(
    failure_path: Path = FAILURE_ARTIFACT_PATH,
    fresh_window_seconds: float = DEFAULT_FAILURE_FRESH_SECONDS,
    now: datetime | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Return (is_fresh_failure, details).

    Fresh iff: failure artefact file exists AND its ts_utc is within
    fresh_window_seconds of now. A stale failure artefact (older than the
    window) is treated as "old news — the cache has since recovered or the
    refresh is in a new cycle".
    """
    now = now or datetime.now(timezone.utc)
    details: dict[str, Any] = {
        "failure_artifact_exists": False,
        "failure_artifact_age_seconds": None,
        "failure_artifact_reason": None,
    }
    doc = _load_json(failure_path)
    if doc is None:
        return False, details
    details["failure_artifact_exists"] = True
    details["failure_artifact_reason"] = doc.get("blocked_reason") or doc.get("error_type")
    ts = _parse_iso_ts(doc.get("ts_utc"))
    if ts is None:
        return False, details
    age = (now - ts).total_seconds()
    details["failure_artifact_age_seconds"] = age
    return age <= fresh_window_seconds, details


def chain_usability(
    cache_path: Path = CACHE_PATH,
    failure_path: Path = FAILURE_ARTIFACT_PATH,
    cache_max_age_seconds: float = DEFAULT_CACHE_MAX_AGE_SECONDS,
    failure_fresh_window_seconds: float = DEFAULT_FAILURE_FRESH_SECONDS,
    now: datetime | None = None,
) -> FreshnessVerdict:
    """Aggregate verdict consumed by strategies for fail-closed gating.

    The chain is usable iff:
      * the cache is fresh (see ``is_chain_fresh``), AND
      * there is no fresh failure artefact (see ``is_failure_artifact_fresh``).

    A fresh failure artefact overrides a non-stale cache because it indicates
    the most recent refresh attempt explicitly failed.
    """
    now = now or datetime.now(timezone.utc)
    fresh, cache_reason, cache_details = is_chain_fresh(
        cache_path=cache_path,
        max_age_seconds=cache_max_age_seconds,
        now=now,
    )
    fresh_failure, failure_details = is_failure_artifact_fresh(
        failure_path=failure_path,
        fresh_window_seconds=failure_fresh_window_seconds,
        now=now,
    )
    usable = fresh and not fresh_failure
    if fresh_failure:
        reason = "fresh_failure_artifact"
    elif not fresh:
        reason = cache_reason
    else:
        reason = "usable"
    return FreshnessVerdict(
        usable=usable,
        reason=reason,
        cache_exists=cache_details["cache_exists"],
        cache_age_seconds=cache_details["cache_age_seconds"],
        cache_error=cache_details["cache_error"],
        cache_chains_count=cache_details["cache_chains_count"],
        failure_artifact_exists=failure_details["failure_artifact_exists"],
        failure_artifact_age_seconds=failure_details["failure_artifact_age_seconds"],
        failure_artifact_reason=failure_details["failure_artifact_reason"],
    )
