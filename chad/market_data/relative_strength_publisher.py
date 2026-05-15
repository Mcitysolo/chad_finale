"""Phase B Item 2 — Relative strength publisher.

Writes ``runtime/relative_strength.json`` from existing daily bars at
``data/bars/1d/{SYM}.json``. Computes per-symbol 5-day total returns
against SPY and QQQ benchmarks and emits a coarse classification used as
an additive, fail-open confidence modifier by the entry-only alpha gates.

The publisher fails open per symbol: missing/corrupt bar files yield
``data_available=false`` with ``rs_class="unknown"`` rather than raising.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_REPO_ROOT = Path("/home/ubuntu/chad_finale")
DEFAULT_RUNTIME_DIR = DEFAULT_REPO_ROOT / "runtime"
DEFAULT_DATA_DIR = DEFAULT_REPO_ROOT / "data" / "bars" / "1d"
DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_TTL_SECONDS = 90000
SCHEMA_VERSION = "relative_strength.v1"

BENCHMARK_SPY = "SPY"
BENCHMARK_QQQ = "QQQ"

KNOWN_FUTURES_SYMBOLS = frozenset(
    {"MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "ES", "NQ", "GC", "CL",
     "RTY", "MYM", "M2K", "M6E"}
)

_MARKET_THRESHOLD = 0.005
_EXCESS_STRONG = 0.03
_EXCESS_WEAK = -0.03
_RATIO_STRONG = 2.0
_RATIO_WEAK = 1.0
_RATIO_FLOOR = 0.001


@dataclass(frozen=True)
class RelativeStrengthRow:
    symbol: str
    return_5d: Optional[float]
    spy_return_5d: Optional[float]
    qqq_return_5d: Optional[float]
    excess_vs_spy_5d: Optional[float]
    excess_vs_qqq_5d: Optional[float]
    rs_vs_spy: Optional[float]
    rs_vs_qqq: Optional[float]
    rs_class: str
    data_available: bool
    bars_used: int
    last_bar_ts_utc: Optional[str]


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _is_equity_or_etf(symbol: str) -> bool:
    sym = _normalize_symbol(symbol)
    if not sym:
        return False
    if sym in KNOWN_FUTURES_SYMBOLS:
        return False
    if "-USD" in sym:
        return False
    return True


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
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _bar_path(symbol: str, data_dir: Path) -> Path:
    return data_dir / f"{_normalize_symbol(symbol)}.json"


def _load_bars(symbol: str, data_dir: Path) -> List[Dict[str, Any]]:
    """Return the bar list for ``symbol`` or [] on any failure."""
    path = _bar_path(symbol, data_dir)
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if isinstance(doc, dict):
        bars = doc.get("bars", doc.get("data"))
    elif isinstance(doc, list):
        bars = doc
    else:
        bars = None
    if not isinstance(bars, list):
        return []
    out: List[Dict[str, Any]] = []
    for rec in bars:
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _close_of(rec: Dict[str, Any]) -> Optional[float]:
    val = rec.get("close", rec.get("c"))
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:
        return None
    return f


def _ts_of(rec: Dict[str, Any]) -> Optional[str]:
    ts = rec.get("ts_utc", rec.get("t"))
    if isinstance(ts, str):
        return ts
    return None


def compute_return_5d(
    bars: List[Dict[str, Any]],
    lookback_days: int,
) -> Optional[float]:
    """Total close-to-close return over the trailing ``lookback_days`` bars.

    Requires at least ``lookback_days + 1`` bars. Returns None on any
    malformed/insufficient input.
    """
    if lookback_days <= 0:
        return None
    if len(bars) < lookback_days + 1:
        return None
    last_close = _close_of(bars[-1])
    prior_close = _close_of(bars[-(lookback_days + 1)])
    if last_close is None or prior_close is None:
        return None
    if prior_close <= 0:
        return None
    return (last_close - prior_close) / prior_close


def _classify_rs(
    sym: str,
    sym_return: Optional[float],
    spy_return: Optional[float],
    excess_vs_spy: Optional[float],
    rs_vs_spy: Optional[float],
) -> str:
    if sym_return is None or spy_return is None or excess_vs_spy is None:
        return "unknown"
    if _normalize_symbol(sym) == BENCHMARK_SPY:
        return "neutral"
    if excess_vs_spy >= _EXCESS_STRONG:
        return "strong"
    if excess_vs_spy <= _EXCESS_WEAK:
        return "weak"
    if spy_return > _MARKET_THRESHOLD and rs_vs_spy is not None:
        if rs_vs_spy >= _RATIO_STRONG:
            return "strong"
        if rs_vs_spy < _RATIO_WEAK:
            return "weak"
    return "neutral"


def _market_direction(spy_return: Optional[float]) -> str:
    if spy_return is None:
        return "unknown"
    if spy_return > _MARKET_THRESHOLD:
        return "up"
    if spy_return < -_MARKET_THRESHOLD:
        return "down"
    return "flat"


def _safe_ratio(
    sym_return: Optional[float],
    benchmark_return: Optional[float],
) -> Optional[float]:
    if sym_return is None or benchmark_return is None:
        return None
    if benchmark_return <= 0:
        return None
    if abs(benchmark_return) < _RATIO_FLOOR:
        return None
    return sym_return / benchmark_return


def _excess(
    sym_return: Optional[float],
    benchmark_return: Optional[float],
) -> Optional[float]:
    if sym_return is None or benchmark_return is None:
        return None
    return sym_return - benchmark_return


def compute_row(
    symbol: str,
    *,
    data_dir: Path,
    lookback_days: int,
    spy_return: Optional[float],
    qqq_return: Optional[float],
) -> RelativeStrengthRow:
    sym = _normalize_symbol(symbol)
    bars = _load_bars(sym, data_dir)
    bars_used = len(bars)
    last_ts = _ts_of(bars[-1]) if bars else None
    sym_return = compute_return_5d(bars, lookback_days)
    if sym_return is None:
        return RelativeStrengthRow(
            symbol=sym,
            return_5d=None,
            spy_return_5d=spy_return,
            qqq_return_5d=qqq_return,
            excess_vs_spy_5d=None,
            excess_vs_qqq_5d=None,
            rs_vs_spy=None,
            rs_vs_qqq=None,
            rs_class="unknown",
            data_available=False,
            bars_used=bars_used,
            last_bar_ts_utc=last_ts,
        )
    excess_spy = _excess(sym_return, spy_return)
    excess_qqq = _excess(sym_return, qqq_return)
    rs_spy = _safe_ratio(sym_return, spy_return)
    rs_qqq = _safe_ratio(sym_return, qqq_return)
    rs_class = _classify_rs(sym, sym_return, spy_return, excess_spy, rs_spy)
    return RelativeStrengthRow(
        symbol=sym,
        return_5d=sym_return,
        spy_return_5d=spy_return,
        qqq_return_5d=qqq_return,
        excess_vs_spy_5d=excess_spy,
        excess_vs_qqq_5d=excess_qqq,
        rs_vs_spy=rs_spy,
        rs_vs_qqq=rs_qqq,
        rs_class=rs_class,
        data_available=True,
        bars_used=bars_used,
        last_bar_ts_utc=last_ts,
    )


def _row_to_payload(row: RelativeStrengthRow) -> Dict[str, Any]:
    def _round(x: Optional[float], n: int) -> Optional[float]:
        return round(float(x), n) if x is not None else None

    return {
        "return_5d": _round(row.return_5d, 6),
        "spy_return_5d": _round(row.spy_return_5d, 6),
        "qqq_return_5d": _round(row.qqq_return_5d, 6),
        "excess_vs_spy_5d": _round(row.excess_vs_spy_5d, 6),
        "excess_vs_qqq_5d": _round(row.excess_vs_qqq_5d, 6),
        "rs_vs_spy": _round(row.rs_vs_spy, 6),
        "rs_vs_qqq": _round(row.rs_vs_qqq, 6),
        "rs_class": row.rs_class,
        "data_available": bool(row.data_available),
        "bars_used": int(row.bars_used),
        "last_bar_ts_utc": row.last_bar_ts_utc,
    }


def _select_symbols(
    extra_symbols: Optional[List[str]] = None,
) -> List[str]:
    syms = _load_universe_symbols()
    seen: set[str] = set()
    out: List[str] = [BENCHMARK_SPY, BENCHMARK_QQQ]
    seen.update(out)
    for raw in syms:
        sym = _normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        if not _is_equity_or_etf(sym):
            continue
        seen.add(sym)
        out.append(sym)
    for raw in extra_symbols or []:
        sym = _normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def build_payload(
    *,
    data_dir: Path,
    lookback_days: int,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    extra_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    symbols = _select_symbols(extra_symbols=extra_symbols)

    spy_bars = _load_bars(BENCHMARK_SPY, data_dir)
    qqq_bars = _load_bars(BENCHMARK_QQQ, data_dir)
    spy_return = compute_return_5d(spy_bars, lookback_days)
    qqq_return = compute_return_5d(qqq_bars, lookback_days)

    symbols_out: Dict[str, Dict[str, Any]] = {}
    strong = neutral = weak = unknown = 0
    nonbench_computed = 0
    for sym in symbols:
        row = compute_row(
            sym,
            data_dir=data_dir,
            lookback_days=lookback_days,
            spy_return=spy_return,
            qqq_return=qqq_return,
        )
        symbols_out[sym] = _row_to_payload(row)
        if row.rs_class == "strong":
            strong += 1
        elif row.rs_class == "weak":
            weak += 1
        elif row.rs_class == "neutral":
            neutral += 1
        else:
            unknown += 1
        if (
            row.data_available
            and sym not in (BENCHMARK_SPY, BENCHMARK_QQQ)
        ):
            nonbench_computed += 1

    spy_ok = spy_return is not None
    qqq_ok = qqq_return is not None
    if spy_ok and qqq_ok and nonbench_computed >= 1:
        provider_status = "real"
        status = "ok"
    elif spy_ok and qqq_ok:
        provider_status = "partial"
        status = "partial"
    elif not spy_ok or len(symbols_out) == 0:
        provider_status = "error"
        status = "error"
    else:
        provider_status = "partial"
        status = "partial"

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now_z(),
        "ttl_seconds": int(ttl_seconds),
        "lookback_days": int(lookback_days),
        "benchmark_spy_return_5d": (
            round(spy_return, 6) if spy_return is not None else None
        ),
        "benchmark_qqq_return_5d": (
            round(qqq_return, 6) if qqq_return is not None else None
        ),
        "market_direction": _market_direction(spy_return),
        "symbols": symbols_out,
        "summary": {
            "symbols_computed": nonbench_computed,
            "strong_count": strong,
            "neutral_count": neutral,
            "weak_count": weak,
            "unknown_count": unknown,
        },
        "source": {
            "provider": "daily_bars",
            "bar_path": str(data_dir),
            "provider_status": provider_status,
        },
        "status": status,
    }


def run_publish(
    *,
    runtime_dir: Path,
    data_dir: Path,
    lookback_days: int,
    dry_run: bool,
    extra_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = build_payload(
        data_dir=data_dir,
        lookback_days=lookback_days,
        extra_symbols=extra_symbols,
    )
    if not dry_run:
        out_path = runtime_dir / "relative_strength.json"
        _atomic_write_json(out_path, payload)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Publish runtime/relative_strength.json from daily bars in "
            "data/bars/1d/. Used as an additive, fail-open entry-only "
            "confidence modifier."
        )
    )
    ap.add_argument(
        "--runtime-dir", default=str(DEFAULT_RUNTIME_DIR),
        help="Output directory (default: %(default)s)",
    )
    ap.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Daily bars directory (default: %(default)s)",
    )
    ap.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="Trailing window in trading days (default: %(default)d)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Compute payload and print it; do not write to disk.",
    )
    args = ap.parse_args(argv)

    try:
        payload = run_publish(
            runtime_dir=Path(args.runtime_dir).resolve(),
            data_dir=Path(args.data_dir).resolve(),
            lookback_days=int(args.lookback_days),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[relative_strength_publisher] fatal: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            "[relative_strength_publisher] wrote "
            f"{Path(args.runtime_dir) / 'relative_strength.json'}"
        )
        print(f"  status={payload.get('status')}")
        print(
            "  provider_status="
            f"{payload.get('source', {}).get('provider_status')}"
        )
        print(f"  market_direction={payload.get('market_direction')}")
        print(f"  symbols_computed={summary.get('symbols_computed')}")
        print(f"  strong={summary.get('strong_count')}")
        print(f"  neutral={summary.get('neutral_count')}")
        print(f"  weak={summary.get('weak_count')}")
        print(f"  unknown={summary.get('unknown_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
