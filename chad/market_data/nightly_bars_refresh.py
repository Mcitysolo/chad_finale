"""
Unified nightly bar refresh — fetches daily OHLCV for all strategy universes.

Replaces polygon_daily_bars_backfill.py and ibkr_daily_bars_refresh.py.
Runnable as: python3 -m chad.market_data.nightly_bars_refresh
"""

import csv
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BARS_1D_DIR: Path = REPO_ROOT / "data" / "bars" / "1d"
UNIVERSE_PATH: Path = REPO_ROOT / "config" / "universe.json"

IB_HOST: str = "127.0.0.1"
IB_PORT: int = 4002
IB_CLIENT_ID: int = 9053

KRAKEN_PAIRS: Dict[str, str] = {
    "BTC-USD": "XXBTZUSD",
    "ETH-USD": "XETHZUSD",
    "SOL-USD": "SOLUSD",
}

KRAKEN_ENDPOINT: str = "https://api.kraken.com/0/public/OHLC"
VIX_CSV_URL: str = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def _ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(symbol: str, msg: str) -> None:
    print(f"[{_ts_now()}] {symbol}: {msg}", flush=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_universe() -> Tuple[List[str], List[Dict[str, str]]]:
    with open(UNIVERSE_PATH) as f:
        data = json.load(f)
    return data["symbols"], data["futures"]


def fetch_kraken_bars(symbol: str, pair_key: str) -> List[Dict[str, Any]]:
    """Fetch daily OHLCV bars from Kraken public API."""
    url = f"{KRAKEN_ENDPOINT}?pair={pair_key}&interval=1440"
    req = urllib.request.Request(url, headers={"User-Agent": "CHAD/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    if body.get("error"):
        raise RuntimeError(f"Kraken API error: {body['error']}")

    result_key: Optional[str] = None
    for k in body.get("result", {}):
        if k != "last":
            result_key = k
            break
    if result_key is None:
        raise RuntimeError(f"No result key found for {pair_key}")

    raw_bars = body["result"][result_key]
    bars: List[Dict[str, Any]] = []
    for bar in raw_bars:
        ts_utc = datetime.fromtimestamp(int(bar[0]), tz=timezone.utc).strftime("%Y-%m-%d")
        bars.append({
            "ts_utc": ts_utc,
            "open": float(bar[1]),
            "high": float(bar[2]),
            "low": float(bar[3]),
            "close": float(bar[4]),
            "volume": float(bar[6]),
        })

    bars.sort(key=lambda b: b["ts_utc"])
    return bars


def fetch_vix_bars() -> List[Dict[str, Any]]:
    """Fetch daily VIX bars from CBOE CSV."""
    req = urllib.request.Request(VIX_CSV_URL, headers={"User-Agent": "CHAD/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    all_rows: List[Dict[str, Any]] = []
    for row in reader:
        date_str = row["DATE"].strip()
        # MM/DD/YYYY -> YYYY-MM-DD
        parts = date_str.split("/")
        if len(parts) != 3:
            continue
        ts_utc = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        try:
            bar = {
                "ts_utc": ts_utc,
                "open": float(row["OPEN"]),
                "high": float(row["HIGH"]),
                "low": float(row["LOW"]),
                "close": float(row["CLOSE"]),
                "volume": 0.0,
            }
            all_rows.append(bar)
        except (ValueError, KeyError):
            continue

    all_rows.sort(key=lambda b: b["ts_utc"])
    return all_rows[-400:]


def _write_bar_file(symbol: str, bars: List[Dict[str, Any]], source: str) -> None:
    payload = {
        "symbol": symbol,
        "source": source,
        "timeframe": "1d",
        "ts_utc": _ts_now(),
        "ttl_seconds": 86400,
        "bars": bars,
    }
    out_path = BARS_1D_DIR / f"{symbol}.json"
    _atomic_write_json(out_path, payload)


def _run_kraken(results: Dict[str, bool]) -> None:
    for symbol, pair_key in KRAKEN_PAIRS.items():
        try:
            bars = fetch_kraken_bars(symbol, pair_key)
            _write_bar_file(symbol, bars, source="kraken")
            _log(symbol, f"wrote {len(bars)} bars, source=kraken")
            results[symbol] = True
        except Exception as e:
            _log(symbol, f"FAILED {e}")
            results[symbol] = False


def _run_vix(results: Dict[str, bool]) -> None:
    symbol = "VIX"
    try:
        bars = fetch_vix_bars()
        _write_bar_file(symbol, bars, source="cboe")
        _log(symbol, f"wrote {len(bars)} bars, source=cboe")
        results[symbol] = True
    except Exception as e:
        _log(symbol, f"FAILED {e}")
        results[symbol] = False


def _run_ibkr(
    equities: List[str],
    futures: List[Dict[str, str]],
    results: Dict[str, bool],
) -> bool:
    """Run IBKR fetches. Returns False if connection failed entirely."""
    try:
        from ib_insync import IB
        from chad.market_data.ibkr_historical_provider import IBKRHistoricalProvider
    except ImportError as e:
        _log("IBKR", f"FAILED import error: {e}")
        for sym in equities:
            results[sym] = False
        for fut in futures:
            results[fut["symbol"]] = False
        return False

    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
    except Exception as e:
        _log("IBKR", f"FAILED connection error: {e}")
        for sym in equities:
            results[sym] = False
        for fut in futures:
            results[fut["symbol"]] = False
        return False

    try:
        provider = IBKRHistoricalProvider(ib)

        # Equities via backfill_universe
        try:
            counts = provider.backfill_universe(equities, days=400, sec_type="STK")
            for sym in equities:
                if sym in counts and counts[sym] > 0:
                    _log(sym, f"wrote {counts[sym]} bars, source=ibkr")
                    results[sym] = True
                else:
                    _log(sym, "FAILED no bars returned")
                    results[sym] = False
        except Exception as e:
            _log("EQUITIES", f"FAILED batch error: {e}")
            for sym in equities:
                if sym not in results:
                    results[sym] = False

        # Futures individually
        for fut in futures:
            sym = fut["symbol"]
            try:
                bars = provider.fetch_daily_bars(sym, days=400, sec_type="FUT")
                if bars:
                    _log(sym, f"wrote {len(bars)} bars, source=ibkr")
                    results[sym] = True
                else:
                    _log(sym, "FAILED no bars returned")
                    results[sym] = False
            except Exception as e:
                _log(sym, f"FAILED {e}")
                results[sym] = False
    finally:
        ib.disconnect()

    return True


def main() -> int:
    equities, futures = _load_universe()
    results: Dict[str, bool] = {}

    ib_connected = _run_ibkr(equities, futures, results)
    _run_kraken(results)
    _run_vix(results)

    success = sum(1 for v in results.values() if v)
    failure = sum(1 for v in results.values() if not v)
    print(f"[{_ts_now()}] SUMMARY success={success} failure={failure}", flush=True)

    if not ib_connected and failure > 0:
        return 2
    if failure > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
