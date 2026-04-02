#!/usr/bin/env python3
"""
CHAD — Calendar State Publisher (SSOT v5.0)

Purpose
-------
Publish /runtime/calendar_state.json with strict top-level freshness metadata:

  - ts_utc (ISO8601, UTC)
  - ttl_seconds (int)

This file is also written opportunistically by chad/core/paper_shadow_runner.py as a
contract-hours cache (items keyed by "SYMBOL::EXCHANGE::CURRENCY"). That legacy writer
does not include ttl_seconds.

This publisher:
  - Preserves and sanitizes existing cache items.
  - Adds schema_version + ttl_seconds + health summary.
  - Writes atomically via chad.utils.runtime_json.write_runtime_state_json.
  - Avoids edge-case corruption via an OS file lock.
  - Never fabricates IBKR hours. It only reports what exists and how fresh it is.

CLI
---
Default: run once and exit 0.

Environment
-----------
CHAD_ROOT                         (default: inferred from file location)
CHAD_RUNTIME_DIR                  (default: $CHAD_ROOT/runtime)
CHAD_CALENDAR_STATE_TTL_SECONDS   (default: 300)
CHAD_CALENDAR_ITEM_TTL_SECONDS    (default: 3600)
CHAD_CALENDAR_MAX_ITEMS           (default: 2000)
CHAD_CALENDAR_PRUNE_MAX_AGE_SECONDS (default: 1209600)  # 14 days

Exit codes
----------
0  success (file written and self-validated)
2  configuration error
1  unexpected runtime error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

# Linux-only but correct for systemd Ubuntu hosts.
import fcntl  # type: ignore

from chad.utils.runtime_json import write_runtime_state_json

LOG = logging.getLogger("chad.calendar_state_publisher")


class ConfigError(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    # Keep 'Z' form for consistency with other CHAD runtime files.
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_int(name: str, default: int, *, min_v: int, max_v: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except Exception as e:
        raise ConfigError(f"{name} must be int; got {raw!r}") from e
    if v < min_v or v > max_v:
        raise ConfigError(f"{name} out of range [{min_v},{max_v}]: {v}")
    return v


def _default_root_from_file() -> Path:
    # .../chad/ops/calendar_state_publisher.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _safe_resolve_dir(p: Path) -> Path:
    rp = p.expanduser().resolve()
    if not rp.exists():
        raise ConfigError(f"required directory does not exist: {rp}")
    if not rp.is_dir():
        raise ConfigError(f"not a directory: {rp}")
    return rp


def _safe_resolve_output(runtime_dir: Path, filename: str) -> Path:
    out = (runtime_dir / filename).resolve()
    # Guardrail: output must remain within runtime_dir (no path traversal / symlink surprises).
    if runtime_dir.resolve() not in out.parents:
        raise ConfigError(f"output path escapes runtime_dir: {out}")
    return out


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        LOG.warning("failed to read/parse json; treating as empty. path=%s err=%r", path, e)
        return {}


def _parse_ts(ts: str) -> Optional[datetime]:
    s = str(ts or "").strip()
    if not s:
        return None
    try:
        # Accept both "...Z" and "...+00:00"
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _sanitize_items(
    items: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Keep only well-formed items:
      item[key] must be dict
      fetched_ts_utc must be string parseable (optional but tracked)
      liquidHours/tradingHours coerced to strings
    """
    kept: Dict[str, Dict[str, Any]] = {}
    counters = {
        "input_non_dict": 0,
        "dropped_non_dict_item": 0,
        "dropped_bad_key": 0,
        "kept": 0,
        "missing_fetched_ts": 0,
        "bad_fetched_ts": 0,
    }

    if not isinstance(items, dict):
        counters["input_non_dict"] += 1
        return kept, counters

    for k, v in items.items():
        if not isinstance(k, str) or "::" not in k:
            counters["dropped_bad_key"] += 1
            continue
        if not isinstance(v, dict):
            counters["dropped_non_dict_item"] += 1
            continue

        fetched_ts = str(v.get("fetched_ts_utc") or "")
        parsed = _parse_ts(fetched_ts)
        if not fetched_ts:
            counters["missing_fetched_ts"] += 1
        elif parsed is None:
            counters["bad_fetched_ts"] += 1

        kept[k] = {
            "fetched_ts_utc": fetched_ts if fetched_ts else None,
            "liquidHours": str(v.get("liquidHours") or ""),
            "tradingHours": str(v.get("tradingHours") or ""),
        }
        counters["kept"] += 1

    return kept, counters


def _prune_items(
    items: MutableMapping[str, Dict[str, Any]],
    *,
    now: datetime,
    prune_max_age_seconds: int,
    max_items: int,
) -> Dict[str, Any]:
    """
    Prune items that are extremely old (defensive hygiene) and cap total item count.

    This does NOT enforce the *logical* cache TTL used by paper_shadow_runner (that is per-key).
    This only prevents unbounded growth / junk accumulation.
    """
    now_utc = now.astimezone(timezone.utc)

    def age_s(it: Mapping[str, Any]) -> float:
        ts = _parse_ts(str(it.get("fetched_ts_utc") or ""))
        if ts is None:
            return float("inf")
        return max(0.0, (now_utc - ts).total_seconds())

    # Drop ultra-old
    dropped_old = 0
    for key in list(items.keys()):
        if age_s(items[key]) > float(prune_max_age_seconds):
            items.pop(key, None)
            dropped_old += 1

    # Cap size: drop oldest first (including unknown timestamps)
    dropped_cap = 0
    if len(items) > max_items:
        ranked = sorted(items.items(), key=lambda kv: age_s(kv[1]), reverse=True)
        for key, _ in ranked[: max(0, len(items) - max_items)]:
            items.pop(key, None)
            dropped_cap += 1

    return {"dropped_old": dropped_old, "dropped_cap": dropped_cap, "remaining": len(items)}


def _compute_freshness_summary(
    items: Mapping[str, Mapping[str, Any]],
    *,
    now: datetime,
    item_ttl_seconds: int,
) -> Dict[str, Any]:
    now_utc = now.astimezone(timezone.utc)
    ages: list[float] = []
    stale = 0
    fresh = 0
    unknown = 0

    for v in items.values():
        ts = _parse_ts(str(v.get("fetched_ts_utc") or ""))
        if ts is None:
            unknown += 1
            continue
        age = max(0.0, (now_utc - ts).total_seconds())
        ages.append(age)
        if age <= float(item_ttl_seconds):
            fresh += 1
        else:
            stale += 1

    return {
        "items_total": len(items),
        "items_fresh": fresh,
        "items_stale": stale,
        "items_unknown_ts": unknown,
        "age_seconds_min": min(ages) if ages else None,
        "age_seconds_max": max(ages) if ages else None,
        "freshness_ok": bool(fresh > 0),
        "item_ttl_seconds": int(item_ttl_seconds),
    }


@dataclass(frozen=True)
class PublisherConfig:
    root: Path
    runtime_dir: Path
    output_path: Path
    ttl_seconds: int
    item_ttl_seconds: int
    max_items: int
    prune_max_age_seconds: int
    lock_path: Path


def _build_config(argv: Optional[list[str]] = None) -> PublisherConfig:
    p = argparse.ArgumentParser(prog="chad.ops.calendar_state_publisher", add_help=True)
    p.add_argument("--root", default=os.getenv("CHAD_ROOT", "").strip() or None)
    p.add_argument("--runtime-dir", default=os.getenv("CHAD_RUNTIME_DIR", "").strip() or None)
    p.add_argument("--ttl-seconds", type=int, default=None)
    p.add_argument("--item-ttl-seconds", type=int, default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--prune-max-age-seconds", type=int, default=None)
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    root = Path(args.root) if args.root else _default_root_from_file()
    root = _safe_resolve_dir(root)

    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else (root / "runtime")
    runtime_dir = _safe_resolve_dir(runtime_dir)

    ttl_seconds = int(args.ttl_seconds) if args.ttl_seconds is not None else _env_int(
        "CHAD_CALENDAR_STATE_TTL_SECONDS", 300, min_v=10, max_v=86400
    )
    item_ttl_seconds = int(args.item_ttl_seconds) if args.item_ttl_seconds is not None else _env_int(
        "CHAD_CALENDAR_ITEM_TTL_SECONDS", 3600, min_v=10, max_v=7 * 86400
    )
    max_items = int(args.max_items) if args.max_items is not None else _env_int(
        "CHAD_CALENDAR_MAX_ITEMS", 2000, min_v=1, max_v=200000
    )
    prune_max_age_seconds = (
        int(args.prune_max_age_seconds)
        if args.prune_max_age_seconds is not None
        else _env_int("CHAD_CALENDAR_PRUNE_MAX_AGE_SECONDS", 14 * 86400, min_v=60, max_v=365 * 86400)
    )

    output_path = _safe_resolve_output(runtime_dir, "calendar_state.json")
    lock_path = _safe_resolve_output(runtime_dir, ".calendar_state_publisher.lock")

    return PublisherConfig(
        root=root,
        runtime_dir=runtime_dir,
        output_path=output_path,
        ttl_seconds=ttl_seconds,
        item_ttl_seconds=item_ttl_seconds,
        max_items=max_items,
        prune_max_age_seconds=prune_max_age_seconds,
        lock_path=lock_path,
    )


def _acquire_lock(lock_path: Path) -> Any:
    """
    Exclusive non-blocking lock. If another publisher run is active, exit cleanly.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.seek(0)
        f.truncate()
        f.write(_iso_utc(_utcnow()))
        f.flush()
        os.fsync(f.fileno())
        return f
    except BlockingIOError:
        f.close()
        raise
    except Exception:
        f.close()
        raise


def publish_once(cfg: PublisherConfig) -> None:
    now = _utcnow()

    # Merge existing cache written by paper_shadow_runner (if present).
    existing = _read_json_file(cfg.output_path)
    existing_items_raw = existing.get("items", {})
    items, sanitize_counts = _sanitize_items(existing_items_raw)

    prune_stats = _prune_items(
        items,
        now=now,
        prune_max_age_seconds=cfg.prune_max_age_seconds,
        max_items=cfg.max_items,
    )

    freshness = _compute_freshness_summary(items, now=now, item_ttl_seconds=cfg.item_ttl_seconds)

    payload: Dict[str, Any] = {
        "schema_version": "calendar_state.v1",
        "paths": {
            "repo_root": str(cfg.root),
            "runtime_dir": str(cfg.runtime_dir),
            "source_file": str(cfg.output_path),
        },
        "notes": (
            "Publisher ensures SSOT v5 metadata (ts_utc + ttl_seconds) without fabricating hours. "
            "Items are preserved/sanitized from the legacy writer in paper_shadow_runner."
        ),
        "items": items,
        "health": {
            "sanitize": sanitize_counts,
            "prune": prune_stats,
            "freshness": freshness,
        },
    }

    # Atomic write + injected ts_utc + ttl_seconds
    write_runtime_state_json(cfg.output_path, payload, ttl_seconds=int(cfg.ttl_seconds), inject_ts=True)

    # Post-write self-check (prevents “green but wrong”)
    written = _read_json_file(cfg.output_path)
    ttl = written.get("ttl_seconds", None)
    ts = written.get("ts_utc", None)
    if not isinstance(ttl, int):
        raise RuntimeError(f"post_write_validation_failed: ttl_seconds not int (got {ttl!r})")
    if not isinstance(ts, str) or not ts.strip():
        raise RuntimeError("post_write_validation_failed: ts_utc missing/blank")

    LOG.info(
        "published calendar_state=%s ttl=%ss items=%d freshness_ok=%s",
        str(cfg.output_path),
        cfg.ttl_seconds,
        len(items),
        bool(written.get("health", {}).get("freshness", {}).get("freshness_ok", False)),
    )


def main(argv: Optional[list[str]] = None) -> int:
    try:
        cfg = _build_config(argv)

        try:
            lock_fh = _acquire_lock(cfg.lock_path)
        except BlockingIOError:
            LOG.warning("another publisher instance is running; exiting cleanly")
            return 0

        try:
            publish_once(cfg)
            return 0
        finally:
            try:
                lock_fh.close()
            except Exception:
                pass

    except ConfigError as e:
        LOG.error("configuration error: %s", e)
        return 2
    except Exception as e:
        LOG.exception("fatal error: %r", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
