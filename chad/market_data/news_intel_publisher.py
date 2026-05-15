"""Phase B Item 1 — News intel publisher.

Writes ``runtime/news_intel.json`` from the catalyst news provider for the
equity/ETF subset of the active universe. Futures and crypto symbols are
skipped because the catalyst gate never gates those asset classes.

The CLI is the only entry point. The publisher fails open: empty payloads
are written rather than raising so the entry-only catalyst gate keeps
allowing trades.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from chad.market_data.catalyst_news_provider import (
    CatalystIntel,
    build_catalyst_intel,
    get_catalyst_intel,
)

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_TTL_SECONDS = 1800
SCHEMA_VERSION = "news_intel.v1"

KNOWN_FUTURES_SYMBOLS = frozenset(
    {"MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "ES", "NQ", "GC", "CL", "RTY", "MYM", "M2K", "M6E"}
)


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_equity_or_etf(symbol: str) -> bool:
    sym = (symbol or "").strip().upper()
    if not sym:
        return False
    if sym in KNOWN_FUTURES_SYMBOLS:
        return False
    if "-USD" in sym:
        return False
    return True


def filter_universe(symbols: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = (raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        if not _is_equity_or_etf(sym):
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _load_universe_symbols() -> List[str]:
    try:
        from chad.utils.universe_provider import load_active_universe
        result = load_active_universe()
        return list(result.symbols or [])
    except Exception:
        return []


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _intel_to_payload(intel: CatalystIntel) -> Dict[str, Any]:
    return {
        "has_catalyst": bool(intel.has_catalyst),
        "catalyst_strength": intel.catalyst_strength,
        "catalyst_direction": intel.catalyst_direction,
        "news_count": int(intel.news_count),
        "catalyst_count": int(intel.catalyst_count),
        "latest_headline": intel.latest_headline,
        "latest_ts_utc": intel.latest_ts_utc,
        "catalyst_categories": list(intel.catalyst_categories),
        "source_provider": intel.source_provider,
    }


def build_payload(
    intel_map: Dict[str, CatalystIntel],
    *,
    lookback_hours: int,
    provider_status: str,
    status: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Dict[str, Any]:
    symbols_payload: Dict[str, Dict[str, Any]] = {}
    high_count = 0
    with_catalyst = 0
    breakdown = {"polygon": 0, "yahoo": 0, "none": 0, "test_no_fetch": 0}

    for sym, intel in sorted(intel_map.items()):
        symbols_payload[sym] = _intel_to_payload(intel)
        if intel.has_catalyst:
            with_catalyst += 1
        if intel.catalyst_strength == "high":
            high_count += 1
        provider = intel.source_provider if intel.source_provider in breakdown else "none"
        breakdown[provider] = breakdown.get(provider, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now_z(),
        "ttl_seconds": int(ttl_seconds),
        "lookback_hours": int(lookback_hours),
        "source": {
            "primary": "polygon",
            "fallback": "yahoo",
            "provider_status": provider_status,
        },
        "status": status,
        "symbols": symbols_payload,
        "summary": {
            "symbols_processed": len(intel_map),
            "symbols_with_catalyst": with_catalyst,
            "high_strength_count": high_count,
            "provider_breakdown": breakdown,
        },
    }


def _resolve_provider_status(intel_map: Dict[str, CatalystIntel]) -> str:
    if not intel_map:
        return "empty"
    providers = {i.source_provider for i in intel_map.values()}
    if "polygon" in providers:
        return "real"
    if "yahoo" in providers:
        return "fallback"
    return "empty"


def _resolve_status(intel_map: Dict[str, CatalystIntel]) -> str:
    if not intel_map:
        return "empty"
    return "ok"


def _empty_intel(symbols: List[str], source_provider: str) -> Dict[str, CatalystIntel]:
    return {
        sym: build_catalyst_intel(sym, [], source_provider=source_provider)
        for sym in symbols
    }


def run_publish(
    *,
    runtime_dir: Path,
    lookback_hours: int,
    dry_run: bool,
    no_fetch_test_mode: bool,
) -> Dict[str, Any]:
    raw_symbols = _load_universe_symbols()
    eq_symbols = filter_universe(raw_symbols)

    if no_fetch_test_mode:
        intel_map = _empty_intel(eq_symbols, source_provider="test_no_fetch")
        provider_status = "test_no_fetch"
        status = "ok" if eq_symbols else "empty"
    else:
        intel_map = get_catalyst_intel(eq_symbols, lookback_hours=lookback_hours)
        provider_status = _resolve_provider_status(intel_map)
        status = _resolve_status(intel_map)

    payload = build_payload(
        intel_map,
        lookback_hours=lookback_hours,
        provider_status=provider_status,
        status=status,
    )

    if not dry_run:
        out_path = runtime_dir / "news_intel.json"
        _atomic_write_json(out_path, payload)

    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Publish runtime/news_intel.json (Polygon primary, Yahoo fallback)."
    )
    ap.add_argument(
        "--runtime-dir", default=str(DEFAULT_RUNTIME_DIR),
        help="Output directory (default: %(default)s)",
    )
    ap.add_argument(
        "--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS,
        help="News lookback horizon (default: %(default)d)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Compute payload and print it; do not write to disk.",
    )
    ap.add_argument(
        "--no-fetch-test-mode", action="store_true",
        help="Skip all external API calls and emit a deterministic empty payload.",
    )
    args = ap.parse_args(argv)

    try:
        payload = run_publish(
            runtime_dir=Path(args.runtime_dir).resolve(),
            lookback_hours=int(args.lookback_hours),
            dry_run=bool(args.dry_run),
            no_fetch_test_mode=bool(args.no_fetch_test_mode),
        )
    except Exception as exc:
        print(f"[news_intel_publisher] fatal: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            f"[news_intel_publisher] wrote {Path(args.runtime_dir) / 'news_intel.json'}"
        )
        print(f"  status={payload.get('status')}")
        print(f"  provider_status={payload.get('source', {}).get('provider_status')}")
        print(f"  symbols_processed={summary.get('symbols_processed')}")
        print(f"  symbols_with_catalyst={summary.get('symbols_with_catalyst')}")
        print(f"  high_strength_count={summary.get('high_strength_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
