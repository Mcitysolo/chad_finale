"""FMP earnings / analyst / filing intelligence publisher.

Phase C — earnings intelligence layer.

Publishes ``runtime/earnings_intel.json`` from the confirmed-working FMP
stable endpoints (earnings-calendar, price-target-consensus,
analyst-estimates ?period=annual, sec-filings-search/symbol).

This module is publisher-only:
    - No strategy imports.
    - No execution / broker imports.
    - No systemd interaction.
    - No mutation of ``runtime/earnings_state.json`` (the stale bootstrap
      file with no active consumer remains untouched).
    - FMP stable news endpoint is intentionally never used — the
      current plan restricts it.

Failure model: fail-open per symbol *and* per endpoint. Individual
endpoint failures record a ``provider_errors`` entry for that symbol but
never raise to the caller. The publisher only returns a non-zero exit
code on a genuinely unhandled exception in ``main``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chad.market_data import market_intel_provider
from chad.market_data.fmp_client import (
    FMPAnalystEstimate,
    FMPEarningsEvent,
    FMPPriceTargetConsensus,
    FMPSecFiling,
)

DEFAULT_REPO_ROOT = Path("/home/ubuntu/chad_finale")
DEFAULT_RUNTIME_DIR = DEFAULT_REPO_ROOT / "runtime"
DEFAULT_CONFIG_PATH = DEFAULT_REPO_ROOT / "config" / "universe.json"

SCHEMA_VERSION = "earnings_intel.v1"
OUTPUT_FILE_NAME = "earnings_intel.json"

DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_FORWARD_DAYS = 45
DEFAULT_TTL_SECONDS = 21600  # 6h — matches the refresh timer cadence

DEFAULT_FALLBACK_SYMBOLS: Tuple[str, ...] = (
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
    "GOOGL", "AMZN", "META", "AVGO", "LLY", "BAC",
)

KNOWN_FUTURES_SYMBOLS = frozenset(
    {"MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "ES", "NQ", "GC", "CL",
     "RTY", "MYM", "M2K", "M6E", "SIL"}
)

ENDPOINTS_USED: Tuple[str, ...] = (
    "earnings-calendar",
    "price-target-consensus",
    "analyst-estimates annual",
    "sec-filings-search",
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _iso_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _parse_iso_date(raw: str) -> Optional[date]:
    s = (raw or "").strip()
    if not s:
        return None
    # FMP returns dates as YYYY-MM-DD; some payloads include time too.
    candidate = s.split("T", 1)[0].split(" ", 1)[0]
    try:
        return datetime.strptime(candidate, "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Symbol filtering
# ---------------------------------------------------------------------------


def _is_equity_or_etf(symbol: str) -> bool:
    sym = (symbol or "").strip().upper()
    if not sym:
        return False
    if "-USD" in sym:
        return False
    if sym in KNOWN_FUTURES_SYMBOLS:
        return False
    return True


def filter_universe(symbols: Iterable[str]) -> List[str]:
    """Keep only equity / ETF tickers, uppercased, deduped, in input order."""
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


# ---------------------------------------------------------------------------
# Universe resolution
# ---------------------------------------------------------------------------


def _load_universe_symbols() -> List[str]:
    """Resolve the active equity universe.

    Preference order: the universe_provider helper used by sister Phase B
    publishers, then ``config/universe.json``'s bare symbol list, then the
    hardcoded safe fallback.
    """
    try:
        from chad.utils.universe_provider import load_active_universe
        result = load_active_universe()
        symbols = list(result.symbols or [])
        if symbols:
            return symbols
    except Exception:
        pass

    try:
        raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            syms = raw.get("symbols")
            if isinstance(syms, list):
                return [str(s) for s in syms]
    except Exception:
        pass

    return list(DEFAULT_FALLBACK_SYMBOLS)


def _resolve_symbols(cli_symbols: Optional[str]) -> List[str]:
    if cli_symbols:
        return filter_universe(cli_symbols.split(","))
    return filter_universe(_load_universe_symbols())


# ---------------------------------------------------------------------------
# Per-endpoint extractors
# ---------------------------------------------------------------------------


def _fetch_earnings_window(
    date_from: str, date_to: str,
) -> Tuple[Optional[List[FMPEarningsEvent]], Optional[str]]:
    """Fetch the full earnings-calendar window once.

    Returns ``(events, error)``. The window is shared across all symbols
    because the FMP calendar endpoint returns multi-symbol events for the
    window — calling it once is cheap, calling it once per symbol is not.
    """
    try:
        events = market_intel_provider.fetch_earnings_calendar(date_from, date_to)
    except Exception as exc:  # pragma: no cover — provider already fail-open
        return None, f"earnings_calendar_exception:{type(exc).__name__}"
    if events is None:
        return None, "earnings_calendar_none"
    return list(events), None


def _split_earnings_for_symbol(
    sym: str,
    events: List[FMPEarningsEvent],
    today: date,
) -> Tuple[Optional[FMPEarningsEvent], Optional[FMPEarningsEvent]]:
    """Pick (next_event_on_or_after_today, last_event_before_today) for symbol."""
    next_event: Optional[Tuple[date, FMPEarningsEvent]] = None
    last_event: Optional[Tuple[date, FMPEarningsEvent]] = None
    for ev in events:
        if (ev.symbol or "").upper() != sym:
            continue
        d = _parse_iso_date(ev.date)
        if d is None:
            continue
        if d >= today:
            if next_event is None or d < next_event[0]:
                next_event = (d, ev)
        else:
            if last_event is None or d > last_event[0]:
                last_event = (d, ev)
    return (
        next_event[1] if next_event else None,
        last_event[1] if last_event else None,
    )


def _select_annual_estimate(
    estimates: List[FMPAnalystEstimate],
    today: date,
) -> Optional[FMPAnalystEstimate]:
    """Pick the nearest future annual estimate, falling back to most recent."""
    if not estimates:
        return None
    future: List[Tuple[date, FMPAnalystEstimate]] = []
    past: List[Tuple[date, FMPAnalystEstimate]] = []
    undated: List[FMPAnalystEstimate] = []
    for est in estimates:
        d = _parse_iso_date(est.date)
        if d is None:
            undated.append(est)
            continue
        if d >= today:
            future.append((d, est))
        else:
            past.append((d, est))
    if future:
        future.sort(key=lambda t: t[0])
        return future[0][1]
    if past:
        past.sort(key=lambda t: t[0], reverse=True)
        return past[0][1]
    return undated[0] if undated else None


def _latest_filing(filings: List[FMPSecFiling]) -> Optional[FMPSecFiling]:
    if not filings:
        return None
    best: Optional[FMPSecFiling] = None
    best_key: Tuple[str, str] = ("", "")
    for f in filings:
        key = (f.accepted_date or "", f.filing_date or "")
        if best is None or key > best_key:
            best = f
            best_key = key
    return best


# ---------------------------------------------------------------------------
# Per-symbol record builder
# ---------------------------------------------------------------------------


def _empty_symbol_record() -> Dict[str, Any]:
    return {
        "next_earnings_date": None,
        "days_to_next_earnings": None,
        "last_earnings_date": None,
        "eps_estimated": None,
        "eps_actual": None,
        "revenue_estimated": None,
        "revenue_actual": None,
        "price_target_high": None,
        "price_target_low": None,
        "price_target_consensus": None,
        "price_target_median": None,
        "annual_revenue_avg_estimate": None,
        "annual_eps_avg_estimate": None,
        "latest_filing_date": None,
        "latest_filing_type": None,
        "sec_filings_count": 0,
        "data_available": False,
        "provider_errors": [],
    }


def build_symbol_record(
    sym: str,
    *,
    earnings_events: List[FMPEarningsEvent],
    earnings_error: Optional[str],
    price_targets: Optional[List[FMPPriceTargetConsensus]],
    pt_error: Optional[str],
    analyst_estimates: Optional[List[FMPAnalystEstimate]],
    ae_error: Optional[str],
    sec_filings: Optional[List[FMPSecFiling]],
    sec_error: Optional[str],
    today: date,
) -> Dict[str, Any]:
    """Assemble the per-symbol record. Never raises."""
    rec = _empty_symbol_record()

    # --- earnings calendar -------------------------------------------------
    if earnings_error:
        rec["provider_errors"].append("earnings-calendar")
    next_ev, last_ev = _split_earnings_for_symbol(sym, earnings_events, today)
    if next_ev is not None:
        rec["next_earnings_date"] = next_ev.date or None
        d = _parse_iso_date(next_ev.date)
        if d is not None:
            rec["days_to_next_earnings"] = int((d - today).days)
        rec["eps_estimated"] = next_ev.eps_estimated
        rec["revenue_estimated"] = next_ev.revenue_estimated
        if next_ev.eps_actual is not None:
            rec["eps_actual"] = next_ev.eps_actual
        if next_ev.revenue_actual is not None:
            rec["revenue_actual"] = next_ev.revenue_actual
    if last_ev is not None:
        rec["last_earnings_date"] = last_ev.date or None
        if rec["eps_actual"] is None:
            rec["eps_actual"] = last_ev.eps_actual
        if rec["revenue_actual"] is None:
            rec["revenue_actual"] = last_ev.revenue_actual
        if rec["eps_estimated"] is None:
            rec["eps_estimated"] = last_ev.eps_estimated
        if rec["revenue_estimated"] is None:
            rec["revenue_estimated"] = last_ev.revenue_estimated

    # --- price-target consensus -------------------------------------------
    if pt_error:
        rec["provider_errors"].append("price-target-consensus")
    elif not price_targets:
        rec["provider_errors"].append("price-target-consensus")
    else:
        pt = price_targets[0]
        rec["price_target_high"] = pt.target_high
        rec["price_target_low"] = pt.target_low
        rec["price_target_consensus"] = pt.target_consensus
        rec["price_target_median"] = pt.target_median

    # --- analyst estimates (annual) ---------------------------------------
    if ae_error:
        rec["provider_errors"].append("analyst-estimates")
    elif not analyst_estimates:
        rec["provider_errors"].append("analyst-estimates")
    else:
        est = _select_annual_estimate(list(analyst_estimates), today)
        if est is not None:
            rec["annual_revenue_avg_estimate"] = est.revenue_avg
            rec["annual_eps_avg_estimate"] = est.eps_avg

    # --- SEC filings -------------------------------------------------------
    if sec_error:
        rec["provider_errors"].append("sec-filings")
    elif not sec_filings:
        rec["provider_errors"].append("sec-filings")
    else:
        rec["sec_filings_count"] = int(len(sec_filings))
        latest = _latest_filing(list(sec_filings))
        if latest is not None:
            rec["latest_filing_date"] = (
                latest.accepted_date or latest.filing_date or None
            )
            rec["latest_filing_type"] = latest.form_type or None

    rec["data_available"] = bool(
        rec["next_earnings_date"]
        or rec["last_earnings_date"]
        or rec["price_target_consensus"] is not None
        or rec["annual_revenue_avg_estimate"] is not None
        or rec["annual_eps_avg_estimate"] is not None
        or rec["sec_filings_count"] > 0
    )
    return rec


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def _safe_fetch_price_targets(
    sym: str,
) -> Tuple[Optional[List[FMPPriceTargetConsensus]], Optional[str]]:
    try:
        result = market_intel_provider.fetch_price_target_consensus(sym)
    except Exception as exc:  # pragma: no cover — provider fail-open
        return None, f"price_target_exception:{type(exc).__name__}"
    if result is None:
        return None, "price_target_none"
    return list(result), None


def _safe_fetch_analyst_estimates(
    sym: str,
) -> Tuple[Optional[List[FMPAnalystEstimate]], Optional[str]]:
    try:
        result = market_intel_provider.fetch_analyst_estimates_annual(sym)
    except Exception as exc:  # pragma: no cover
        return None, f"analyst_estimates_exception:{type(exc).__name__}"
    if result is None:
        return None, "analyst_estimates_none"
    return list(result), None


def _safe_fetch_sec_filings(
    sym: str, date_from: str, date_to: str,
) -> Tuple[Optional[List[FMPSecFiling]], Optional[str]]:
    try:
        result = market_intel_provider.fetch_sec_filings(sym, date_from, date_to)
    except Exception as exc:  # pragma: no cover
        return None, f"sec_filings_exception:{type(exc).__name__}"
    if result is None:
        return None, "sec_filings_none"
    return list(result), None


def build_payload(
    symbols: List[str],
    *,
    lookback_days: int,
    forward_days: int,
    today: Optional[date] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Dict[str, Any]:
    """Compose the full publisher payload — no disk I/O, no raises."""
    today = today or _today_utc()
    date_from = today - timedelta(days=int(max(0, lookback_days)))
    date_to = today + timedelta(days=int(max(0, forward_days)))

    earnings_events, earnings_error = _fetch_earnings_window(
        _iso_date(date_from), _iso_date(date_to),
    )
    events_list: List[FMPEarningsEvent] = list(earnings_events or [])

    symbols_payload: Dict[str, Dict[str, Any]] = {}
    with_next = 0
    with_pt = 0
    with_ae = 0
    with_filings = 0

    for sym in symbols:
        pts, pt_err = _safe_fetch_price_targets(sym)
        ests, ae_err = _safe_fetch_analyst_estimates(sym)
        filings, sec_err = _safe_fetch_sec_filings(
            sym, _iso_date(date_from), _iso_date(date_to),
        )
        rec = build_symbol_record(
            sym,
            earnings_events=events_list,
            earnings_error=earnings_error,
            price_targets=pts,
            pt_error=pt_err,
            analyst_estimates=ests,
            ae_error=ae_err,
            sec_filings=filings,
            sec_error=sec_err,
            today=today,
        )
        symbols_payload[sym] = rec
        if rec["next_earnings_date"]:
            with_next += 1
        if rec["price_target_consensus"] is not None:
            with_pt += 1
        if rec["annual_revenue_avg_estimate"] is not None or rec["annual_eps_avg_estimate"] is not None:
            with_ae += 1
        if rec["sec_filings_count"] > 0:
            with_filings += 1

    available_count = sum(
        1 for r in symbols_payload.values() if r["data_available"]
    )
    if not symbols_payload:
        provider_status = "empty"
        status = "empty"
    elif available_count == 0:
        if earnings_error and all(
            "price-target-consensus" in r["provider_errors"]
            and "analyst-estimates" in r["provider_errors"]
            and "sec-filings" in r["provider_errors"]
            for r in symbols_payload.values()
        ):
            provider_status = "error"
            status = "error"
        else:
            provider_status = "empty"
            status = "empty"
    elif available_count == len(symbols_payload):
        provider_status = "real"
        status = "ok"
    else:
        provider_status = "partial"
        status = "partial"

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now_z(),
        "ttl_seconds": int(ttl_seconds),
        "status": status,
        "source": {
            "provider": "fmp_stable",
            "provider_status": provider_status,
            "endpoints": list(ENDPOINTS_USED),
        },
        "window": {
            "date_from": _iso_date(date_from),
            "date_to": _iso_date(date_to),
            "lookback_days": int(lookback_days),
            "forward_days": int(forward_days),
        },
        "symbols": symbols_payload,
        "summary": {
            "symbols_requested": len(symbols),
            "symbols_processed": len(symbols_payload),
            "symbols_with_next_earnings": int(with_next),
            "symbols_with_price_targets": int(with_pt),
            "symbols_with_analyst_estimates": int(with_ae),
            "symbols_with_sec_filings": int(with_filings),
        },
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def publish(
    *,
    runtime_dir: Path,
    symbols: List[str],
    lookback_days: int,
    forward_days: int,
    dry_run: bool,
) -> Dict[str, Any]:
    payload = build_payload(
        symbols,
        lookback_days=lookback_days,
        forward_days=forward_days,
    )
    if not dry_run:
        _atomic_write_json(runtime_dir / OUTPUT_FILE_NAME, payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Publish runtime/earnings_intel.json from FMP stable endpoints "
            "(earnings calendar, price targets, analyst estimates, SEC filings)."
        )
    )
    ap.add_argument(
        "--runtime-dir", default=str(DEFAULT_RUNTIME_DIR),
        help="Output directory (default: %(default)s).",
    )
    ap.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="Earnings/filings lookback window in days (default: %(default)d).",
    )
    ap.add_argument(
        "--forward-days", type=int, default=DEFAULT_FORWARD_DAYS,
        help="Earnings/filings forward window in days (default: %(default)d).",
    )
    ap.add_argument(
        "--symbols", default=None,
        help="Optional comma-separated symbol override (e.g. AAPL,NVDA,MSFT).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Compute payload and print JSON; do not write to disk.",
    )
    args = ap.parse_args(argv)

    try:
        symbols = _resolve_symbols(args.symbols)
        payload = publish(
            runtime_dir=Path(args.runtime_dir).resolve(),
            symbols=symbols,
            lookback_days=int(args.lookback_days),
            forward_days=int(args.forward_days),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(
            f"[fmp_earnings_intel_publisher] fatal: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            f"[fmp_earnings_intel_publisher] wrote "
            f"{Path(args.runtime_dir) / OUTPUT_FILE_NAME}"
        )
        print(f"  status={payload.get('status')}")
        print(
            f"  provider_status="
            f"{payload.get('source', {}).get('provider_status')}"
        )
        print(f"  symbols_requested={summary.get('symbols_requested')}")
        print(f"  symbols_processed={summary.get('symbols_processed')}")
        print(
            f"  with_next_earnings={summary.get('symbols_with_next_earnings')} "
            f"with_price_targets={summary.get('symbols_with_price_targets')} "
            f"with_analyst_estimates={summary.get('symbols_with_analyst_estimates')} "
            f"with_sec_filings={summary.get('symbols_with_sec_filings')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
