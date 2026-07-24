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


def _write_kraken_bars_state(crypto_results: Dict[str, bool]) -> None:
    """CBAR-S2 — freshness marker + state file for the EXS1 sentinel table.

    Emits the ``KRAKEN_BARS_REFRESH ok=N fail=M`` marker line (over the CRYPTO symbols only,
    NOT the shared results dict which also holds IBKR/VIX rows) and writes
    ``runtime/kraken_bars_state.json`` carrying its own ``ts_utc`` + ``ttl_seconds`` so the
    Exterminator Sentinel's ``check_stale_feeds`` can judge crypto-bar freshness once an
    operator adds the feed row to ``config/exterminator.json``. ttl=90000s (25h) gives a 1h
    grace over the 24h refresh cadence. Purely observability: a state-write failure is logged
    and swallowed so it can never fail the refresh run or block the bar writes above.
    """
    ok = sum(1 for v in crypto_results.values() if v)
    fail = sum(1 for v in crypto_results.values() if not v)
    print(f"[{_ts_now()}] KRAKEN_BARS_REFRESH ok={ok} fail={fail}", flush=True)
    try:
        payload = {
            "schema_version": "kraken_bars_state.v1",
            "ts_utc": _ts_now(),
            "ttl_seconds": 90000,
            "ok": ok,
            "fail": fail,
            "symbols": {s: ("ok" if v else "fail") for s, v in sorted(crypto_results.items())},
        }
        _atomic_write_json(REPO_ROOT / "runtime" / "kraken_bars_state.json", payload)
    except Exception as e:  # noqa: BLE001 — observability must never fail the refresh
        _log("KRAKEN_BARS_STATE", f"state write FAILED {e}")


def _read_last_bar_utc(symbol: str) -> Optional[str]:
    """Newest bar date recorded for ``symbol``, or None if unreadable.

    Read from the file on disk (not the in-memory result) so a symbol that
    FAILED this run still reports the age of whatever it last managed to
    write. A failed symbol with a fresh file and a failed symbol that has been
    dark for a week are very different problems.
    """
    try:
        with open(BARS_1D_DIR / f"{symbol}.json") as f:
            bars = json.load(f).get("bars") or []
        return str(bars[-1]["ts_utc"]) if bars else None
    except Exception:  # noqa: BLE001 — observability only
        return None


def _age_days(last_bar_utc: Optional[str], today: Optional[Any] = None) -> Optional[int]:
    if not last_bar_utc:
        return None
    try:
        last = datetime.strptime(last_bar_utc[:10], "%Y-%m-%d").date()
    except Exception:  # noqa: BLE001
        return None
    ref = today or datetime.now(timezone.utc).date()
    return (ref - last).days


def _write_bars_refresh_state(
    results: Dict[str, bool],
    futures_symbols: List[str],
    *,
    today: Optional[Any] = None,
) -> None:
    """W6A-3 — per-symbol status + age, so partial success is never silent.

    D1: the unit's exit code no longer reddens on a single bad symbol, so the
    per-symbol truth has to surface somewhere the sentinel can read. This is
    that surface.

    Carries ``age_vs_cohort_days`` — a symbol's bar age minus the FRESHEST age
    achieved in its group. That relative signature is what exposed the MCL
    defect: on 2026-07-23 MCL sat at age 2 while all nine peers were at 1,
    because it was pinned to a contract that had stopped trading. An absolute
    staleness threshold would have missed it (age 2 is well inside any sane
    weekend allowance); the +1 versus its peers is the tell.

    The reference is the group MINIMUM, deliberately not the median: a median
    is dragged along by the stale symbols themselves, so if several symbols go
    dark together the median moves with them and the lag silently vanishes —
    exactly when detection matters most. The freshest symbol in a group is the
    honest statement of "what was achievable this run".

    Grouping matters because cadences differ: Kraken crypto bars land same-day
    (age 0) while IBKR equity/futures bars are a day behind. Comparing across
    those sources would mark every IBKR symbol as lagging. Groups are scoped
    to crypto / futures / equity so only like is compared with like.

    Purely observability — a write failure is logged and swallowed so it can
    never fail the refresh or block bar writes.
    """
    try:
        # `today` is injectable so tests can pin it — a test that reads the wall
        # clock is a time bomb, which is exactly the defect W6A-5 converts.
        today = today or datetime.now(timezone.utc).date()
        futures_set = set(futures_symbols)

        def _group(sym: str) -> str:
            if sym in KRAKEN_PAIRS:
                return "crypto"
            if sym == "VIX":
                return "index"
            return "future" if sym in futures_set else "equity"

        rows: Dict[str, Dict[str, Any]] = {}
        for sym, ok in sorted(results.items()):
            last = _read_last_bar_utc(sym)
            rows[sym] = {
                "status": "ok" if ok else "fail",
                "last_bar_utc": last,
                "age_days": _age_days(last, today),
                "group": _group(sym),
            }

        # Freshest age per group — the reference every symbol is measured against.
        group_min: Dict[str, int] = {}
        for row in rows.values():
            age = row["age_days"]
            if age is None:
                continue
            g = row["group"]
            group_min[g] = age if g not in group_min else min(group_min[g], age)

        for row in rows.values():
            ref = group_min.get(row["group"])
            row["group_min_age_days"] = ref
            row["age_vs_cohort_days"] = (
                None if (row["age_days"] is None or ref is None)
                else row["age_days"] - ref
            )

        ok_count = sum(1 for v in results.values() if v)
        fail_count = len(results) - ok_count
        lagging = sorted(
            s for s, r in rows.items()
            if r["age_vs_cohort_days"] is not None and r["age_vs_cohort_days"] > 0
        )
        no_data = sorted(s for s, r in rows.items() if r["age_days"] is None)

        payload = {
            "schema_version": "bars_refresh_state.v1",
            "ts_utc": _ts_now(),
            # 25h — 1h grace over the 24h refresh cadence (mirrors kraken_bars_state.v1)
            "ttl_seconds": 90000,
            "ok": ok_count,
            "fail": fail_count,
            "total": len(results),
            "group_min_age_days": dict(sorted(group_min.items())),
            "lagging_symbols": lagging,
            "no_data_symbols": no_data,
            "symbols": rows,
        }
        _atomic_write_json(REPO_ROOT / "runtime" / "bars_refresh_state.json", payload)
        print(
            f"[{_ts_now()}] BARS_REFRESH_STATE ok={ok_count} fail={fail_count} "
            f"group_min_age={group_min} lagging={','.join(lagging) or '-'} "
            f"no_data={','.join(no_data) or '-'}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 — observability must never fail the refresh
        _log("BARS_REFRESH_STATE", f"state write FAILED {e}")


def _run_kraken(results: Dict[str, bool]) -> None:
    crypto_results: Dict[str, bool] = {}
    for symbol, pair_key in KRAKEN_PAIRS.items():
        try:
            bars = fetch_kraken_bars(symbol, pair_key)
            _write_bar_file(symbol, bars, source="kraken")
            _log(symbol, f"wrote {len(bars)} bars, source=kraken")
            results[symbol] = True
            crypto_results[symbol] = True
        except Exception as e:
            _log(symbol, f"FAILED {e}")
            results[symbol] = False
            crypto_results[symbol] = False
    _write_kraken_bars_state(crypto_results)


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
        from ib_async import IB
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
            counts = provider.backfill_universe(equities, days=400, sec_type="STK", include_minute=True)
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
            # 1-minute bars (additive, never affects exit code)
            try:
                provider.fetch_minute_bars(sym, days=5, sec_type="FUT")
                _log(sym, "1m bars fetched")
            except Exception as exc:
                _log(sym, f"1m FAILED {exc!r}")
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

    _write_bars_refresh_state(results, [f["symbol"] for f in futures])

    # W6A-3 (D1): per-symbol exit-status isolation.
    #
    # Per-symbol WORK was already isolated (one bad contract cannot abort the
    # loop), but the exit code was not: `failure > 0 -> return 1` reddened the
    # whole unit — and EXS5 — whenever a single symbol out of ~52 failed.
    #
    # The unit now fails only on SYSTEMIC failure: nothing attempted, the IBKR
    # connection dead, or every symbol failing. A partial failure exits 0 and
    # is reported through bars_refresh_state.v1 (per-symbol status, age, and
    # cohort-relative lag) rather than through a red unit.
    #
    # Partial success must never be SILENT — that is why the state row and the
    # BARS_REFRESH_STATE marker line are written unconditionally above. Exit 0
    # here means "the unit ran"; it does not mean "every symbol is healthy",
    # and the state row is the place that distinction is recorded.
    if not results:
        print(f"[{_ts_now()}] SUMMARY systemic=no_symbols_attempted", flush=True)
        return 1
    if not ib_connected and failure > 0:
        return 2
    if failure == len(results):
        print(f"[{_ts_now()}] SUMMARY systemic=all_symbols_failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
