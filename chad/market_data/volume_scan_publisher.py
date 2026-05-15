"""Phase B Item 3 — Intraday relative-volume (RVOL) publisher.

Writes ``runtime/volume_scan.json`` with per-symbol relative volume so
``alpha_intraday`` can prioritize high-volume names as an additive,
fail-open confidence modifier.

RVOL formula::

    rvol = current_session_volume / (avg_daily_volume * fraction_elapsed)

Volume source:
  * Equities/ETFs: Polygon snapshot ``day.v`` from
    ``/v2/snapshot/locale/us/markets/stocks/tickers``. Off the hot path
    (5-minute timer); mirrors the off-path Polygon usage pattern of
    ``chad.market_data.catalyst_news_provider``.
  * Futures (MES/MNQ/etc.) and crypto (``*-USD``): marked
    ``rvol_class="unavailable"`` in v1. A proper futures-session model
    would be required to compute meaningful intraday RVOL.

The publisher is fail-open per symbol and never raises out of
``run_publish``. Outside market hours, all symbols are marked
``unavailable`` so the downstream gate yields a zero adjustment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

DEFAULT_REPO_ROOT = Path("/home/ubuntu/chad_finale")
DEFAULT_RUNTIME_DIR = DEFAULT_REPO_ROOT / "runtime"
DEFAULT_DATA_DIR = DEFAULT_REPO_ROOT / "data" / "bars" / "1d"
DEFAULT_TTL_SECONDS = 300
SCHEMA_VERSION = "volume_scan.v1"

POLYGON_SNAPSHOT_URL = (
    "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
)
POLYGON_TIMEOUT_SECONDS = 6.0
POLYGON_BATCH_SIZE = 50

SESSION_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
SESSION_MINUTES = 390

AVG_DAILY_LOOKBACK = 20
AVG_DAILY_MIN_BARS = 5

RVOL_HIGH_THRESHOLD = 3.0
RVOL_ABOVE_THRESHOLD = 1.5
RVOL_NORMAL_THRESHOLD = 0.7

KNOWN_FUTURES_SYMBOLS = frozenset(
    {"MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "ES", "NQ", "GC", "CL",
     "RTY", "MYM", "M2K", "M6E", "SIL"}
)


@dataclass(frozen=True)
class RvolRow:
    symbol: str
    current_volume: Optional[int]
    avg_daily_volume: Optional[float]
    expected_volume: Optional[float]
    rvol: Optional[float]
    rvol_class: str
    data_available: bool


# ---------------------------------------------------------------------------
# Time / session
# ---------------------------------------------------------------------------


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_state(
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """Return ``(market_open, minutes_into_session, fraction_elapsed)``.

    ``fraction_elapsed`` is clamped to ``[0.01, 1.0]`` once the market is
    open. Outside RTH on a weekday, or any weekend, all three values are
    returned as ``(False, None, None)``.
    """
    now = (now_utc or datetime.now(timezone.utc)).astimezone(SESSION_TZ)
    if now.weekday() >= 5:
        return (False, None, None)
    local_t = now.time()
    if local_t < SESSION_OPEN or local_t >= SESSION_CLOSE:
        return (False, None, None)
    open_dt = now.replace(
        hour=SESSION_OPEN.hour,
        minute=SESSION_OPEN.minute,
        second=0,
        microsecond=0,
    )
    minutes = (now - open_dt).total_seconds() / 60.0
    if minutes < 0:
        return (False, None, None)
    fraction = max(0.01, min(1.0, minutes / float(SESSION_MINUTES)))
    return (True, minutes, fraction)


# ---------------------------------------------------------------------------
# Symbol classification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------


def _load_universe_symbols() -> List[str]:
    try:
        from chad.utils.universe_provider import load_active_universe
        result = load_active_universe()
        return list(result.symbols or [])
    except Exception:
        return []


def _select_symbols(
    data_dir: Path,
    extra_symbols: Optional[List[str]] = None,
) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in _load_universe_symbols():
        sym = _normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    if not out and data_dir.exists():
        for p in sorted(data_dir.glob("*.json")):
            sym = _normalize_symbol(p.stem)
            if not sym or sym in seen:
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


# ---------------------------------------------------------------------------
# Daily bars / avg daily volume
# ---------------------------------------------------------------------------


def _bar_path(symbol: str, data_dir: Path) -> Path:
    return data_dir / f"{_normalize_symbol(symbol)}.json"


def _load_daily_bars(symbol: str, data_dir: Path) -> List[Dict[str, Any]]:
    path = _bar_path(symbol, data_dir)
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    bars = doc.get("bars") if isinstance(doc, dict) else None
    if not isinstance(bars, list):
        return []
    return [b for b in bars if isinstance(b, dict)]


def compute_avg_daily_volume(
    bars: List[Dict[str, Any]],
    *,
    lookback: int = AVG_DAILY_LOOKBACK,
    min_bars: int = AVG_DAILY_MIN_BARS,
) -> Optional[float]:
    """Mean of the trailing ``lookback`` valid daily volumes.

    Returns ``None`` if fewer than ``min_bars`` valid entries are found.
    """
    if lookback <= 0 or min_bars <= 0:
        return None
    window = bars[-lookback:] if len(bars) >= lookback else list(bars)
    volumes: List[float] = []
    for rec in window:
        v = rec.get("volume", rec.get("v"))
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f or f <= 0:
            continue
        volumes.append(f)
    if len(volumes) < min_bars:
        return None
    return sum(volumes) / len(volumes)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_rvol(rvol: Optional[float]) -> str:
    if rvol is None:
        return "unavailable"
    if rvol >= RVOL_HIGH_THRESHOLD:
        return "high"
    if rvol >= RVOL_ABOVE_THRESHOLD:
        return "above"
    if rvol >= RVOL_NORMAL_THRESHOLD:
        return "normal"
    return "low"


# ---------------------------------------------------------------------------
# Polygon snapshot fetch
# ---------------------------------------------------------------------------


def _read_polygon_key() -> Optional[str]:
    env_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if env_key:
        return env_key
    env_path = Path("/etc/chad/polygon.env")
    if not env_path.is_file():
        return None
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "POLYGON_API_KEY":
            v = v.strip().strip('"').strip("'")
            if v:
                return v
    return None


def fetch_polygon_snapshots(
    api_key: str,
    tickers: List[str],
    *,
    batch_size: int = POLYGON_BATCH_SIZE,
    timeout: float = POLYGON_TIMEOUT_SECONDS,
) -> Dict[str, int]:
    """Fetch ``day.v`` per ticker via the Polygon batch snapshot endpoint.

    Returns ``{ticker: current_session_volume}``. Symbols that the server
    omits or returns malformed are silently dropped (per-symbol fail-open).
    Network errors yield an empty dict for the affected batch.
    """
    out: Dict[str, int] = {}
    if not api_key or not tickers:
        return out
    upper_tickers = [_normalize_symbol(t) for t in tickers if t]
    upper_tickers = [t for t in upper_tickers if t]
    if not upper_tickers:
        return out
    for start in range(0, len(upper_tickers), batch_size):
        chunk = upper_tickers[start:start + batch_size]
        params = {"tickers": ",".join(chunk), "apiKey": api_key}
        url = f"{POLYGON_SNAPSHOT_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        entries = doc.get("tickers") if isinstance(doc, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ticker = _normalize_symbol(str(entry.get("ticker") or ""))
            if not ticker:
                continue
            day = entry.get("day")
            if not isinstance(day, dict):
                continue
            v = day.get("v", day.get("volume"))
            try:
                vol = int(float(v))
            except (TypeError, ValueError):
                continue
            if vol < 0:
                continue
            out[ticker] = vol
    return out


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _build_row(
    sym: str,
    *,
    current_volume: Optional[int],
    avg_daily_volume: Optional[float],
    fraction_elapsed: Optional[float],
    market_open: bool,
) -> RvolRow:
    if (
        not market_open
        or current_volume is None
        or avg_daily_volume is None
        or fraction_elapsed is None
    ):
        return RvolRow(
            symbol=sym,
            current_volume=current_volume,
            avg_daily_volume=avg_daily_volume,
            expected_volume=None,
            rvol=None,
            rvol_class="unavailable",
            data_available=False,
        )
    expected = avg_daily_volume * fraction_elapsed
    if expected <= 0:
        return RvolRow(
            symbol=sym,
            current_volume=current_volume,
            avg_daily_volume=avg_daily_volume,
            expected_volume=expected,
            rvol=None,
            rvol_class="unavailable",
            data_available=False,
        )
    rvol = float(current_volume) / float(expected)
    return RvolRow(
        symbol=sym,
        current_volume=current_volume,
        avg_daily_volume=avg_daily_volume,
        expected_volume=expected,
        rvol=rvol,
        rvol_class=classify_rvol(rvol),
        data_available=True,
    )


def _row_to_payload(row: RvolRow) -> Dict[str, Any]:
    def _round(x: Optional[float], n: int) -> Optional[float]:
        return round(float(x), n) if x is not None else None

    return {
        "current_volume": (
            int(row.current_volume) if row.current_volume is not None else None
        ),
        "avg_daily_volume": _round(row.avg_daily_volume, 2),
        "expected_volume": _round(row.expected_volume, 2),
        "rvol": _round(row.rvol, 4),
        "rvol_class": row.rvol_class,
        "data_available": bool(row.data_available),
    }


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


def build_payload(
    *,
    data_dir: Path,
    extra_symbols: Optional[List[str]] = None,
    now_utc: Optional[datetime] = None,
    test_mode: bool = False,
    polygon_fetcher: Optional[Callable[[str, List[str]], Dict[str, int]]] = None,
    polygon_api_key: Optional[str] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Dict[str, Any]:
    market_open, minutes_in, fraction = session_state(now_utc)
    symbols_all = _select_symbols(data_dir, extra_symbols=extra_symbols)

    equity_syms = [s for s in symbols_all if _is_equity_or_etf(s)]
    nonequity_syms = [s for s in symbols_all if not _is_equity_or_etf(s)]

    # Determine volume provider and fetch.
    provider: str
    provider_status: str
    snapshots: Dict[str, int] = {}

    if test_mode:
        provider = "test_no_fetch"
        provider_status = "test_no_fetch"
    elif not market_open:
        provider = "polygon_snapshot"
        provider_status = "unavailable"
    else:
        api_key = polygon_api_key if polygon_api_key is not None else (_read_polygon_key() or "")
        if polygon_fetcher is not None:
            try:
                snapshots = polygon_fetcher(api_key, equity_syms) or {}
            except Exception:
                snapshots = {}
        elif api_key:
            snapshots = fetch_polygon_snapshots(api_key, equity_syms)
        if api_key or polygon_fetcher is not None:
            provider = "polygon_snapshot"
        else:
            provider = "none"
        if snapshots:
            provider_status = (
                "real" if len(snapshots) >= max(1, len(equity_syms) // 2)
                else "partial"
            )
        elif market_open and (api_key or polygon_fetcher is not None):
            provider_status = "unavailable"
        else:
            provider_status = "unavailable"

    symbols_out: Dict[str, Dict[str, Any]] = {}
    high_count = 0
    high_syms: List[str] = []
    available_count = 0

    for sym in equity_syms:
        bars = _load_daily_bars(sym, data_dir)
        adv = compute_avg_daily_volume(bars)
        cur_vol: Optional[int] = snapshots.get(sym) if market_open and not test_mode else None
        row = _build_row(
            sym,
            current_volume=cur_vol,
            avg_daily_volume=adv,
            fraction_elapsed=fraction,
            market_open=market_open,
        )
        symbols_out[sym] = _row_to_payload(row)
        if row.data_available:
            available_count += 1
            if row.rvol_class == "high":
                high_count += 1
                high_syms.append(sym)

    for sym in nonequity_syms:
        bars = _load_daily_bars(sym, data_dir)
        adv = compute_avg_daily_volume(bars)
        # Futures/crypto: no intraday volume source in v1.
        symbols_out[sym] = _row_to_payload(RvolRow(
            symbol=sym,
            current_volume=None,
            avg_daily_volume=adv,
            expected_volume=None,
            rvol=None,
            rvol_class="unavailable",
            data_available=False,
        ))

    if test_mode:
        status = "unavailable"
    elif not market_open:
        status = "unavailable"
    elif available_count == 0:
        status = "unavailable"
    elif available_count < max(1, len(equity_syms) // 2):
        status = "partial"
    else:
        status = "ok"

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now_z(),
        "ttl_seconds": int(ttl_seconds),
        "market_open": bool(market_open),
        "minutes_into_session": (
            round(float(minutes_in), 2) if minutes_in is not None else None
        ),
        "fraction_elapsed": (
            round(float(fraction), 6) if fraction is not None else None
        ),
        "symbols": symbols_out,
        "summary": {
            "symbols_scanned": len(equity_syms),
            "high_rvol_count": high_count,
            "high_rvol_symbols": sorted(high_syms),
        },
        "source": {
            "volume_provider": provider,
            "avg_volume_source": "daily_bars",
            "provider_status": provider_status,
        },
        "status": status,
    }


# ---------------------------------------------------------------------------
# Atomic write + CLI runner
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def run_publish(
    *,
    runtime_dir: Path,
    data_dir: Path,
    dry_run: bool,
    test_mode: bool = False,
    extra_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = build_payload(
        data_dir=data_dir,
        extra_symbols=extra_symbols,
        test_mode=test_mode,
    )
    if not dry_run:
        out_path = runtime_dir / "volume_scan.json"
        _atomic_write_json(out_path, payload)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Publish runtime/volume_scan.json — per-symbol intraday RVOL "
            "used as an additive, fail-open entry-only confidence modifier."
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
        "--dry-run", action="store_true",
        help="Compute payload and print it; do not write to disk.",
    )
    ap.add_argument(
        "--no-fetch-test-mode", action="store_true",
        help="Skip external Polygon calls; emit deterministic unavailable payload.",
    )
    args = ap.parse_args(argv)

    try:
        payload = run_publish(
            runtime_dir=Path(args.runtime_dir).resolve(),
            data_dir=Path(args.data_dir).resolve(),
            dry_run=bool(args.dry_run),
            test_mode=bool(args.no_fetch_test_mode),
        )
    except Exception as exc:
        print(f"[volume_scan_publisher] fatal: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            "[volume_scan_publisher] wrote "
            f"{Path(args.runtime_dir) / 'volume_scan.json'}"
        )
        print(f"  status={payload.get('status')}")
        print(f"  market_open={payload.get('market_open')}")
        print(f"  provider_status={payload.get('source', {}).get('provider_status')}")
        print(f"  symbols_scanned={summary.get('symbols_scanned')}")
        print(f"  high_rvol_count={summary.get('high_rvol_count')}")
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_TTL_SECONDS",
    "RvolRow",
    "classify_rvol",
    "compute_avg_daily_volume",
    "session_state",
    "fetch_polygon_snapshots",
    "build_payload",
    "run_publish",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
