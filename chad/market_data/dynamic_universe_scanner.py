"""Phase D Item 1A — dynamic universe candidate scanner publisher.

Reads the canonical active universe via
``chad.utils.universe_provider.load_active_universe()`` and existing runtime
intelligence feeds (relative_strength.json, volume_scan.json, news_intel.json,
earnings_intel.json, event_risk.json) to publish a ranked equity/ETF candidate
list to ``runtime/dynamic_universe_candidates.json`` under schema
``dynamic_universe_candidates.v1``.

This module is dashboard/reporting-first. It MUST NOT:
  * write or replace the canonical runtime universe artifact;
  * mutate the operator-controlled static universe configuration;
  * import strategies or execution modules;
  * call broker APIs;
  * interact with systemd.

Failure mode is fail-open: missing or stale intel feeds degrade each candidate
to ``data_available=false`` with warnings, never raising out of
``build_payload`` / ``publish``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chad.utils.universe_provider import load_active_universe


DEFAULT_REPO_ROOT = Path("/home/ubuntu/chad_finale")
DEFAULT_RUNTIME_DIR = DEFAULT_REPO_ROOT / "runtime"

SCHEMA_VERSION = "dynamic_universe_candidates.v1"
OUTPUT_FILENAME = "dynamic_universe_candidates.json"
DEFAULT_TTL_SECONDS = 300
DEFAULT_MAX_CANDIDATES = 25

# Freshness windows (mtime-based) per intel feed. Aligned with each producer's
# native cadence so the scanner classifies a feed "fresh" when it has been
# refreshed within the producer's own publish window plus a small grace
# margin. A "stale" feed still feeds the score (fail-open) but the source
# block records the staleness so dashboard consumers can flag it.
FRESHNESS_SECONDS: Dict[str, int] = {
    "relative_strength": 5400,   # daily-bars producer runs ~hourly
    "volume_scan": 900,          # 5min cadence
    "news_intel": 3600,          # 30min cadence
    "earnings_intel": 43200,     # 6h cadence
    "event_risk": 7200,          # 30min cadence
}

# Equity/ETF allow-list filter. Futures roots are excluded by exact match;
# crypto pairs are excluded by suffix.
KNOWN_FUTURES_ROOTS = frozenset(
    {"MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL", "MYM", "M2K"}
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_seconds() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Runtime IO
# ---------------------------------------------------------------------------


def _load_json_runtime(runtime_dir: Path, filename: str) -> Tuple[Optional[Dict[str, Any]], float]:
    """Read a runtime JSON document and its mtime age in seconds.

    Returns ``(payload_or_none, age_seconds_or_-1)``. A missing or malformed
    file returns ``(None, -1.0)`` and never raises.
    """
    path = runtime_dir / filename
    try:
        if not path.is_file():
            return None, -1.0
        mtime = path.stat().st_mtime
        age = max(0.0, _now_seconds() - mtime)
        raw = path.read_text(encoding="utf-8")
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            return None, age
        return doc, age
    except (OSError, ValueError, TypeError):
        return None, -1.0


def _freshness(age_seconds: float, max_age: int) -> Optional[bool]:
    """Classify a feed as fresh (True), stale (False), or unknown (None)."""
    if age_seconds is None or age_seconds < 0:
        return None
    return age_seconds <= float(max_age)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Symbol classification
# ---------------------------------------------------------------------------


def _normalize_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _is_equity_or_etf_symbol(symbol: str) -> bool:
    """True iff ``symbol`` is a v1-eligible equity / ETF ticker.

    Excludes:
      * empty / non-string symbols
      * crypto pairs (``*-USD``)
      * known futures roots (MES, MNQ, MCL, MGC, ZN, ZB, M6E, SIL, MYM, M2K)
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        return False
    if sym.endswith("-USD"):
        return False
    if sym in KNOWN_FUTURES_ROOTS:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-component scoring
# ---------------------------------------------------------------------------


def _score_rs(record: Optional[Dict[str, Any]]) -> Tuple[float, List[str], List[str], str, Optional[float]]:
    """Score the relative-strength contribution.

    Returns ``(score_delta, reasons, warnings, rs_class, excess_vs_spy_5d)``.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    if not isinstance(record, dict):
        return 0.05, reasons, ["rs_unknown"], "unknown", None

    rs_class_raw = record.get("rs_class")
    rs_class = str(rs_class_raw).lower() if isinstance(rs_class_raw, str) else "unknown"

    excess = record.get("excess_vs_spy_5d")
    excess_f: Optional[float]
    try:
        excess_f = float(excess) if excess is not None else None
    except (TypeError, ValueError):
        excess_f = None

    if rs_class == "strong":
        return 0.35, ["rs_strong"], warnings, "strong", excess_f
    if rs_class == "neutral":
        return 0.15, ["rs_neutral"], warnings, "neutral", excess_f
    if rs_class == "weak":
        return 0.0, reasons, ["rs_weak"], "weak", excess_f
    return 0.05, reasons, ["rs_unknown"], "unknown", excess_f


def _score_rvol(record: Optional[Dict[str, Any]]) -> Tuple[float, List[str], List[str], str, Optional[float]]:
    """Score the relative-volume contribution.

    Returns ``(score_delta, reasons, warnings, rvol_class, rvol)``.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    if not isinstance(record, dict):
        return 0.03, reasons, ["rvol_unavailable"], "unknown", None

    rvol_class_raw = record.get("rvol_class")
    rvol_class = str(rvol_class_raw).lower() if isinstance(rvol_class_raw, str) else "unknown"

    rvol = record.get("rvol")
    rvol_f: Optional[float]
    try:
        rvol_f = float(rvol) if rvol is not None else None
    except (TypeError, ValueError):
        rvol_f = None

    if rvol_class == "high":
        return 0.30, ["rvol_high"], warnings, "high", rvol_f
    if rvol_class == "above":
        return 0.20, ["rvol_above"], warnings, "above", rvol_f
    if rvol_class == "normal":
        return 0.10, ["rvol_normal"], warnings, "normal", rvol_f
    if rvol_class == "low":
        return 0.0, reasons, ["rvol_low"], "low", rvol_f
    if rvol_class == "unavailable":
        return 0.03, reasons, ["rvol_unavailable"], "unavailable", rvol_f
    return 0.03, reasons, ["rvol_unavailable"], "unknown", rvol_f


def _score_catalyst(record: Optional[Dict[str, Any]]) -> Tuple[
    float, List[str], List[str], str, str, bool
]:
    """Score the news-catalyst contribution.

    Returns
        ``(score_delta, reasons, warnings, catalyst_strength, catalyst_direction, confirmed)``.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    if not isinstance(record, dict):
        return 0.0, reasons, warnings, "unknown", "unknown", False

    strength_raw = record.get("catalyst_strength")
    strength = str(strength_raw).lower() if isinstance(strength_raw, str) else "unknown"
    direction_raw = record.get("catalyst_direction")
    direction = str(direction_raw).lower() if isinstance(direction_raw, str) else "unknown"
    confirmed = bool(record.get("confirmed_gate_relevant"))

    if confirmed and strength == "high":
        return 0.25, ["confirmed_high_catalyst"], warnings, "high", direction, True
    if confirmed and strength == "medium":
        return 0.18, ["confirmed_medium_catalyst"], warnings, "medium", direction, True
    if confirmed and strength == "low":
        return 0.08, ["confirmed_low_catalyst"], warnings, "low", direction, True
    # unconfirmed or no catalyst contributes nothing
    return 0.0, reasons, warnings, strength if strength != "unknown" else "none", direction, confirmed


def _score_liquidity(record: Optional[Dict[str, Any]]) -> Tuple[float, List[str], List[str], str]:
    """Score the liquidity contribution.

    v1 reads ``liquidity_class`` from whichever feed record carries it. The
    existing publishers do not yet emit a per-symbol ``liquidity_class``, so
    in practice this will fall to ``UNKNOWN`` for most rows — that is
    intentional fail-open behaviour and matches the design doc §7.1 band 4.

    Returns ``(score_delta, reasons, warnings, liquidity_class)``.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    if not isinstance(record, dict):
        return 0.03, reasons, warnings, "unknown"

    raw = record.get("liquidity_class")
    cls = str(raw).upper() if isinstance(raw, str) and raw else ""

    if cls == "LARGE":
        return 0.10, reasons, warnings, "LARGE"
    if cls == "STANDARD":
        return 0.06, reasons, warnings, "STANDARD"
    if cls == "THIN":
        return 0.0, reasons, ["thin_liquidity"], "THIN"
    if cls == "UNKNOWN":
        return 0.03, reasons, warnings, "UNKNOWN"
    return 0.03, reasons, warnings, "unknown"


def _earnings_warning(record: Optional[Dict[str, Any]]) -> Tuple[float, List[str], Optional[int]]:
    """Translate earnings proximity into a score multiplier and warning bands.

    Returns ``(multiplier, warnings, earnings_days)``.
    """
    warnings: List[str] = []
    if not isinstance(record, dict):
        return 1.0, warnings, None

    raw = record.get("days_to_next_earnings")
    if raw is None:
        return 1.0, warnings, None
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return 1.0, warnings, None

    if 0 <= days <= 2:
        return 0.5, ["earnings_within_2d"], days
    if 3 <= days <= 7:
        return 1.0, ["earnings_within_7d"], days
    return 1.0, warnings, days


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _feed_record(feed: Optional[Dict[str, Any]], symbol: str) -> Optional[Dict[str, Any]]:
    if not isinstance(feed, dict):
        return None
    syms = feed.get("symbols")
    if not isinstance(syms, dict):
        return None
    rec = syms.get(symbol)
    return rec if isinstance(rec, dict) else None


def build_candidate(
    symbol: str,
    feeds: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a single candidate record. Pure: no IO."""
    sym = _normalize_symbol(symbol)

    rs_rec = _feed_record(feeds.get("relative_strength"), sym)
    rvol_rec = _feed_record(feeds.get("volume_scan"), sym)
    news_rec = _feed_record(feeds.get("news_intel"), sym)
    earnings_rec = _feed_record(feeds.get("earnings_intel"), sym)

    rs_delta, rs_reasons, rs_warnings, rs_class, rs_excess = _score_rs(rs_rec)
    rv_delta, rv_reasons, rv_warnings, rvol_class, rvol_value = _score_rvol(rvol_rec)
    cat_delta, cat_reasons, cat_warnings, cat_strength, cat_direction, confirmed = _score_catalyst(news_rec)

    # Liquidity: prefer volume_scan record (it carries volume metadata), fall
    # back to news_intel record. Either may set ``liquidity_class``; neither
    # is required.
    liq_source = rvol_rec if rvol_rec is not None else news_rec
    liq_delta, liq_reasons, liq_warnings, liq_class = _score_liquidity(liq_source)

    multiplier, earn_warnings, earnings_days = _earnings_warning(earnings_rec)

    raw_score = rs_delta + rv_delta + cat_delta + liq_delta
    score = _clamp01(raw_score)
    score = _clamp01(score * multiplier)

    reasons: List[str] = []
    reasons.extend(rs_reasons)
    reasons.extend(rv_reasons)
    reasons.extend(cat_reasons)
    reasons.extend(liq_reasons)

    warnings: List[str] = []
    warnings.extend(rs_warnings)
    warnings.extend(rv_warnings)
    warnings.extend(cat_warnings)
    warnings.extend(liq_warnings)
    warnings.extend(earn_warnings)

    data_available = bool(
        rs_rec is not None
        or rvol_rec is not None
        or news_rec is not None
        or earnings_rec is not None
    )

    return {
        "symbol": sym,
        "score": round(score, 4),
        "reasons": reasons,
        "warnings": warnings,
        "rs_class": rs_class,
        "rs_excess_vs_spy_5d": (round(rs_excess, 6) if rs_excess is not None else None),
        "rvol_class": rvol_class,
        "rvol": (round(rvol_value, 4) if rvol_value is not None else None),
        "catalyst_strength": cat_strength,
        "catalyst_direction": cat_direction,
        "confirmed_gate_relevant": bool(confirmed),
        "earnings_days": earnings_days,
        "liquidity_class": liq_class,
        "data_available": data_available,
    }


# ---------------------------------------------------------------------------
# Top-level payload
# ---------------------------------------------------------------------------


def _resolve_active_universe() -> Tuple[List[str], Optional[str], Optional[bool]]:
    """Return ``(symbols, source_type, stale)`` from the canonical loader.

    On loader failure returns ``([], None, None)`` — the scanner publishes
    an empty candidate list rather than raising.
    """
    try:
        result = load_active_universe()
    except Exception:
        return [], None, None
    return list(result.symbols or []), str(result.source_type), bool(result.stale)


def build_payload(
    runtime_dir: Path,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    now_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the candidate payload. Pure relative to caller-provided dir."""

    universe_syms, universe_source_type, universe_stale = _resolve_active_universe()
    universe_available = bool(universe_syms)

    # Load each intel feed once.
    rs_doc, rs_age = _load_json_runtime(runtime_dir, "relative_strength.json")
    rv_doc, rv_age = _load_json_runtime(runtime_dir, "volume_scan.json")
    news_doc, news_age = _load_json_runtime(runtime_dir, "news_intel.json")
    earn_doc, earn_age = _load_json_runtime(runtime_dir, "earnings_intel.json")
    evt_doc, evt_age = _load_json_runtime(runtime_dir, "event_risk.json")

    inputs_source = {
        "active_universe": {
            "available": universe_available,
            "source_type": universe_source_type,
            "stale": (universe_stale if universe_available else None),
        },
        "news_intel": {
            "available": news_doc is not None,
            "fresh": _freshness(news_age, FRESHNESS_SECONDS["news_intel"]) if news_doc is not None else None,
        },
        "relative_strength": {
            "available": rs_doc is not None,
            "fresh": _freshness(rs_age, FRESHNESS_SECONDS["relative_strength"]) if rs_doc is not None else None,
        },
        "volume_scan": {
            "available": rv_doc is not None,
            "fresh": _freshness(rv_age, FRESHNESS_SECONDS["volume_scan"]) if rv_doc is not None else None,
        },
        "earnings_intel": {
            "available": earn_doc is not None,
            "fresh": _freshness(earn_age, FRESHNESS_SECONDS["earnings_intel"]) if earn_doc is not None else None,
        },
        "event_risk": {
            "available": evt_doc is not None,
            "fresh": _freshness(evt_age, FRESHNESS_SECONDS["event_risk"]) if evt_doc is not None else None,
        },
    }

    feeds: Dict[str, Optional[Dict[str, Any]]] = {
        "relative_strength": rs_doc,
        "volume_scan": rv_doc,
        "news_intel": news_doc,
        "earnings_intel": earn_doc,
    }

    eligible: List[str] = []
    seen: set[str] = set()
    for raw in universe_syms:
        sym = _normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        if not _is_equity_or_etf_symbol(sym):
            continue
        seen.add(sym)
        eligible.append(sym)

    candidates_all: List[Dict[str, Any]] = [
        build_candidate(sym, feeds) for sym in eligible
    ]

    candidates_all.sort(key=lambda c: (-float(c["score"]), str(c["symbol"])))

    n = max(0, int(max_candidates))
    top = candidates_all[:n]
    for i, cand in enumerate(top, start=1):
        cand["rank"] = i

    # Summary counts span the published top-N (consumers see only those).
    strong_rs = sum(1 for c in top if c.get("rs_class") == "strong")
    high_rvol = sum(1 for c in top if c.get("rvol_class") == "high")
    confirmed_cat = sum(
        1
        for c in top
        if c.get("confirmed_gate_relevant")
        and c.get("catalyst_strength") in {"high", "medium", "low"}
    )
    earnings_warn = sum(
        1
        for c in top
        if any(w.startswith("earnings_within_") for w in c.get("warnings", []))
    )

    any_data = any(c.get("data_available") for c in top)
    inputs_missing = sum(
        1
        for key in ("news_intel", "relative_strength", "volume_scan")
        if not inputs_source[key]["available"]
    )

    if not top:
        status: str = "empty"
        provider_status = "empty"
    elif not any_data:
        status = "partial"
        provider_status = "partial"
    elif inputs_missing >= 2:
        status = "partial"
        provider_status = "partial"
    else:
        status = "ok"
        provider_status = "real"

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": now_utc or _iso_utc_now(),
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "status": status,
        "source": {
            "provider": "local_runtime_intelligence",
            "provider_status": provider_status,
            "inputs": inputs_source,
        },
        "candidates": top,
        "summary": {
            "symbols_considered": len(eligible),
            "candidates_published": len(top),
            "strong_rs_count": int(strong_rs),
            "high_rvol_count": int(high_rvol),
            "confirmed_catalyst_count": int(confirmed_cat),
            "earnings_warning_count": int(earnings_warn),
        },
    }
    return payload


def publish(
    runtime_dir: Path,
    *,
    dry_run: bool = False,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Tuple[Dict[str, Any], bool]:
    """Build and (optionally) atomically write the candidate payload.

    Returns ``(payload, wrote)``. ``wrote`` is True iff the runtime file was
    written. The output filename is ``dynamic_universe_candidates.json``.
    The scanner intentionally never writes ``universe.json``.
    """
    payload = build_payload(runtime_dir, max_candidates=max_candidates)
    if dry_run:
        return payload, False
    out_path = runtime_dir / OUTPUT_FILENAME
    _atomic_write_json(out_path, payload)
    return payload, True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Publish runtime/dynamic_universe_candidates.json — Phase D v1 "
            "ranked equity/ETF candidates derived from existing runtime "
            "intelligence feeds. Does NOT modify the active universe."
        )
    )
    ap.add_argument(
        "--runtime-dir",
        default=str(DEFAULT_RUNTIME_DIR),
        help="Runtime directory containing intel feeds (default: %(default)s)",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum number of candidates to emit (default: %(default)d)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute payload and print to stdout; do not write to disk.",
    )
    args = ap.parse_args(argv)

    try:
        payload, wrote = publish(
            Path(args.runtime_dir).resolve(),
            dry_run=bool(args.dry_run),
            max_candidates=int(args.max_candidates),
        )
    except Exception as exc:
        print(
            f"[dynamic_universe_scanner] fatal: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            "[dynamic_universe_scanner] wrote "
            f"{Path(args.runtime_dir) / OUTPUT_FILENAME}"
            if wrote
            else "[dynamic_universe_scanner] no write (dry-run)"
        )
        print(f"  status={payload.get('status')}")
        print(
            "  provider_status="
            f"{payload.get('source', {}).get('provider_status')}"
        )
        print(f"  symbols_considered={summary.get('symbols_considered')}")
        print(f"  candidates_published={summary.get('candidates_published')}")
        print(f"  strong_rs_count={summary.get('strong_rs_count')}")
        print(f"  high_rvol_count={summary.get('high_rvol_count')}")
        print(
            "  confirmed_catalyst_count="
            f"{summary.get('confirmed_catalyst_count')}"
        )
        print(
            "  earnings_warning_count="
            f"{summary.get('earnings_warning_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
