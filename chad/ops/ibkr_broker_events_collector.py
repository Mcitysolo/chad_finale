#!/usr/bin/env python3
"""
CHAD — IBKR Broker Events Collector (SSOT v5.0 Evidence Layer)

This is the missing "broker_events" evidence producer required for SSOT v5 full semantics.

It produces an append-only, normalized broker event stream:
  data/broker_events/BROKER_EVENTS_IBKR_YYYYMMDD.ndjson

Event types produced (today):
  - fill  (from ib_insync Fill / Execution)
  - fee   (from ib_insync CommissionReport when available)

Hard guarantees
---------------
- NEVER places orders.
- Read-only against IBKR (fills/commissionReports only).
- Append-only NDJSON ledger.
- Exactly-once per event via durable dedupe (SQLite).
- Uses a single-process lock to prevent concurrent runs.
- Never logs secrets.
- Safe when IBKR is down: exits non-zero only on configuration errors; runtime errors are surfaced.

Design notes
------------
- This collector is a oneshot; pair it with a systemd timer (every 60s) for continuous evidence.
- It is intentionally strict about dedupe and canonicalization.
- It does NOT attempt to infer symbol mapping beyond what IBKR provides in fills/contracts.
- It is the source of truth for lifecycle evidence (gap/backlog flags are computed elsewhere).

Environment
-----------
CHAD_ROOT=/home/ubuntu/chad_finale            (preferred)
CHAD_DATA_DIR=/home/ubuntu/chad_finale/data   (optional)
CHAD_RUNTIME_DIR=/home/ubuntu/chad_finale/runtime (optional)
DEPLOYMENT_ID=primary                         (optional)

IBKR connection:
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=118        # use a dedicated id (avoid conflicts with other services)

Collector behavior:
CHAD_BROKER_EVENTS_LOOKBACK_SECONDS=7200      # first run lookback window (default 2h)
CHAD_BROKER_EVENTS_MAX_FETCH=2000             # cap fills returned per run (defensive)
CHAD_BROKER_EVENTS_DEDUPE_DB=data/exec_state/broker_events_ibkr.sqlite3 (default)
CHAD_BROKER_EVENTS_LOCK_PATH=runtime/.ibkr_broker_events.lock           (default)

Exit codes
----------
0  success (wrote 0+ new events or none found; still considered OK)
2  configuration error (missing deps/env)
1  runtime failure (IBKR down, DB failure, etc.)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fcntl  # linux lock

LOG = logging.getLogger("chad.ops.ibkr_broker_events_collector")



# IB_INSYNC_NOISE_FILTER_V1
def _install_ib_insync_noise_filter() -> None:
    """Suppress known-noisy ib_insync log line(s) in THIS process only."""
    class _DropCompletedOrdersTimeout(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            # Drop the single recurring noise line.
            if "completed orders request timed out" in msg:
                return False
            return True

    flt = _DropCompletedOrdersTimeout()
    # Apply to all handlers on the root logger (covers ib_insync logger output).
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.addFilter(flt)
        except Exception:
            pass


# ----------------------------
# Time / serialization
# ----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_json_dumps(obj: Any) -> str:
    # Canonical JSON for hashing and consistent NDJSON records
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jitter_sleep(base_s: float, *, jitter_frac: float = 0.25) -> None:
    j = base_s * jitter_frac
    time.sleep(max(0.05, base_s + random.uniform(-j, j)))


# ----------------------------
# Env helpers
# ----------------------------

def env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def repo_root() -> Path:
    root = env_str("CHAD_ROOT", "")
    if root:
        p = Path(root).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parents[2]


def runtime_dir(root: Path) -> Path:
    rd = env_str("CHAD_RUNTIME_DIR", "")
    if rd:
        return Path(rd).expanduser().resolve()
    return (root / "runtime").resolve()


def data_dir(root: Path) -> Path:
    dd = env_str("CHAD_DATA_DIR", "")
    if dd:
        return Path(dd).expanduser().resolve()
    return (root / "data").resolve()


# ----------------------------
# Locking
# ----------------------------

@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.seek(0)
        fh.truncate()
        fh.write(iso_utc(utc_now()))
        fh.flush()
        os.fsync(fh.fileno())
        yield
    except BlockingIOError:
        raise RuntimeError("lock_busy")
    finally:
        with contextlib.suppress(Exception):
            fh.close()


# ----------------------------
# Dedupe DB
# ----------------------------

class DedupeDB:
    """
    Exactly-once dedupe using SQLite.

    Primary key: event_key (stable string)
    """
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_events (
              event_key TEXT PRIMARY KEY,
              first_seen_ts_utc TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def seen(self, key: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seen_events WHERE event_key=? LIMIT 1", (key,))
        return cur.fetchone() is not None

    def mark(self, key: str, ts_utc: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_events(event_key, first_seen_ts_utc) VALUES (?, ?)",
            (key, ts_utc),
        )

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.commit()
        with contextlib.suppress(Exception):
            self._conn.close()


# ----------------------------
# IBKR connector (read-only)
# ----------------------------

@dataclass(frozen=True)
class IbkrConn:
    host: str
    port: int
    client_id: int
    connect_retries: int


def connect_ibkr(cfg: IbkrConn):
    try:
        from ib_async import IB  # type: ignore
    except Exception as e:
        raise RuntimeError("missing_dependency: ib_async") from e

    last: Optional[Exception] = None
    for attempt in range(cfg.connect_retries + 1):
        ib = IB()
        try:
            ib.connect(cfg.host, int(cfg.port), clientId=int(cfg.client_id), timeout=10)
            if not ib.isConnected():
                raise RuntimeError("ibkr_connected_false")
            return ib
        except Exception as e:
            last = e
            with contextlib.suppress(Exception):
                ib.disconnect()
            if attempt >= cfg.connect_retries:
                break
            jitter_sleep(0.6 * (2 ** attempt), jitter_frac=0.25)
    raise RuntimeError(f"ibkr_connect_failed: {last!r}")


# ----------------------------
# Canonical event normalization
# ----------------------------

def _safe_str(x: Any) -> str:
    try:
        s = str(x)
        return s.strip()
    except Exception:
        return ""


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not (v == v) or v in (float("inf"), float("-inf")):
            return None
        return v
    except Exception:
        return None


def _contract_symbol(fill: Any) -> str:
    # Fill -> execution -> contract
    c = getattr(fill, "contract", None)
    if c is None:
        return ""
    return _safe_str(getattr(c, "symbol", ""))


def _contract_currency(fill: Any) -> str:
    c = getattr(fill, "contract", None)
    if c is None:
        return ""
    return _safe_str(getattr(c, "currency", "")) or "USD"


def _exec_side(fill: Any) -> str:
    ex = getattr(fill, "execution", None)
    side = _safe_str(getattr(ex, "side", ""))
    return side.upper()


def _exec_qty(fill: Any) -> Optional[float]:
    ex = getattr(fill, "execution", None)
    return _safe_float(getattr(ex, "shares", None))


def _exec_price(fill: Any) -> Optional[float]:
    ex = getattr(fill, "execution", None)
    return _safe_float(getattr(ex, "price", None))


def _exec_time_utc(fill: Any) -> Optional[str]:
    t = getattr(fill, "time", None)
    if t is None:
        return None
    try:
        dt = t.astimezone(timezone.utc)
        return iso_utc(dt)
    except Exception:
        return None


def _exec_id(fill: Any) -> str:
    ex = getattr(fill, "execution", None)
    return _safe_str(getattr(ex, "execId", ""))


def _order_id(fill: Any) -> Optional[int]:
    ex = getattr(fill, "execution", None)
    try:
        return int(getattr(ex, "orderId", 0) or 0)
    except Exception:
        return None


def _perm_id(fill: Any) -> Optional[int]:
    ex = getattr(fill, "execution", None)
    try:
        return int(getattr(ex, "permId", 0) or 0)
    except Exception:
        return None


def _build_fill_event(*, deployment_id: str, venue: str, fill: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (event_key, event_payload)
    """
    ts_event = _exec_time_utc(fill) or iso_utc(utc_now())
    symbol = _contract_symbol(fill)
    currency = _contract_currency(fill)
    side = _exec_side(fill)
    qty = _exec_qty(fill)
    price = _exec_price(fill)
    exec_id = _exec_id(fill)
    order_id = _order_id(fill)
    perm_id = _perm_id(fill)

    payload: Dict[str, Any] = {
        "schema_version": "broker_event.v1",
        "ts_utc": ts_event,                     # event time (not write time)
        "deployment_id": deployment_id,
        "venue": venue,
        "event_type": "fill",
        "broker_order_id": order_id,
        "client_order_id": None,                # can be mapped later from idempotency db
        "event_id": exec_id or None,            # stable if provided
        "seq": None,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "currency": currency,
        "raw_ref": {
            "permId": perm_id,
        },
    }

    # Build stable event key
    if exec_id:
        event_key = f"{deployment_id}|{venue}|fill|execId:{exec_id}"
    else:
        # fallback stable hash from canonical fields
        event_key = f"{deployment_id}|{venue}|fill|hash:{sha256_hex(stable_json_dumps(payload))}"

    payload["hash_sha256"] = f"sha256:{sha256_hex(stable_json_dumps(payload))}"
    return event_key, payload


def _build_fee_event(*, deployment_id: str, venue: str, fill: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    IBKR CommissionReport is sometimes attached to Fill as fill.commissionReport.
    Produce a fee event when possible.
    """
    cr = getattr(fill, "commissionReport", None)
    if cr is None:
        return None

    ts_event = _exec_time_utc(fill) or iso_utc(utc_now())
    symbol = _contract_symbol(fill)
    currency = _contract_currency(fill)
    exec_id = _exec_id(fill)
    order_id = _order_id(fill)

    fee_amount = _safe_float(getattr(cr, "commission", None))
    if fee_amount is None:
        return None

    payload: Dict[str, Any] = {
        "schema_version": "broker_event.v1",
        "ts_utc": ts_event,
        "deployment_id": deployment_id,
        "venue": venue,
        "event_type": "fee",
        "broker_order_id": order_id,
        "client_order_id": None,
        "event_id": exec_id or None,
        "seq": None,
        "symbol": symbol,
        "side": None,
        "qty": None,
        "price": None,
        "fee_amount": fee_amount,
        "currency": currency,
        "raw_ref": {
            "execId": exec_id or None,
            "currency": _safe_str(getattr(cr, "currency", "")) or None,
            "realizedPNL": _safe_float(getattr(cr, "realizedPNL", None)),
        },
    }

    if exec_id:
        event_key = f"{deployment_id}|{venue}|fee|execId:{exec_id}"
    else:
        event_key = f"{deployment_id}|{venue}|fee|hash:{sha256_hex(stable_json_dumps(payload))}"

    payload["hash_sha256"] = f"sha256:{sha256_hex(stable_json_dumps(payload))}"
    return event_key, payload


# ----------------------------
# Ledger writer
# ----------------------------

class NdjsonAppender:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append_many(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            # ensure ledger file exists even when 0 events
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.touch(exist_ok=True)
            except Exception:
                pass
            return
        with self._path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(stable_json_dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())


# ----------------------------
# Main collector
# ----------------------------

@dataclass(frozen=True)
class CollectorConfig:
    root: Path
    runtime: Path
    data: Path
    deployment_id: str
    venue: str
    lookback_seconds: int
    max_fetch: int
    lock_path: Path
    dedupe_db_path: Path
    ibkr: IbkrConn


def build_cfg() -> CollectorConfig:
    root = repo_root()
    runtime = runtime_dir(root)
    data = data_dir(root)

    deployment_id = env_str("DEPLOYMENT_ID", "primary") or "primary"
    venue = "IBKR"

    lookback_seconds = env_int("CHAD_BROKER_EVENTS_LOOKBACK_SECONDS", 7200)
    max_fetch = env_int("CHAD_BROKER_EVENTS_MAX_FETCH", 2000)

    lock_path = Path(env_str("CHAD_BROKER_EVENTS_LOCK_PATH", str(runtime / ".ibkr_broker_events.lock"))).expanduser().resolve()

    dedupe_default = data / "exec_state" / "broker_events_ibkr.sqlite3"
    dedupe_db = Path(env_str("CHAD_BROKER_EVENTS_DEDUPE_DB", str(dedupe_default))).expanduser().resolve()

    ibkr = IbkrConn(
        host=env_str("IBKR_HOST", "127.0.0.1"),
        port=env_int("IBKR_PORT", 4002),
        client_id=env_int("IBKR_CLIENT_ID", 118),
        connect_retries=env_int("CHAD_IBKR_CONNECT_RETRIES", 2),
    )

    return CollectorConfig(
        root=root,
        runtime=runtime,
        data=data,
        deployment_id=deployment_id,
        venue=venue,
        lookback_seconds=lookback_seconds,
        max_fetch=max_fetch,
        lock_path=lock_path,
        dedupe_db_path=dedupe_db,
        ibkr=ibkr,
    )


def _ledger_path(data: Path) -> Path:
    day = utc_now().strftime("%Y%m%d")
    return (data / "broker_events" / f"BROKER_EVENTS_IBKR_{day}.ndjson").resolve()


def collect_once(cfg: CollectorConfig) -> int:
    # Lock to avoid concurrent collectors
    try:
        with exclusive_lock(cfg.lock_path):
            return _collect_once_locked(cfg)
    except RuntimeError as e:
        if str(e) == "lock_busy":
            LOG.warning("collector lock busy; skipping this run")
            return 0
        raise


def _collect_once_locked(cfg: CollectorConfig) -> int:
    cfg.data.mkdir(parents=True, exist_ok=True)
    (cfg.data / "broker_events").mkdir(parents=True, exist_ok=True)
    (cfg.data / "exec_state").mkdir(parents=True, exist_ok=True)

    db = DedupeDB(cfg.dedupe_db_path)
    ib = None
    appended: List[Dict[str, Any]] = []
    new_count = 0
    fee_count = 0

    try:
        ib = connect_ibkr(cfg.ibkr)

        # Pull fills and filter by lookback window
        cutoff = utc_now() - timedelta(seconds=int(cfg.lookback_seconds))
        try:
            fills = ib.fills() or []
        except Exception as e:
            raise RuntimeError(f"ibkr_fills_failed: {e!r}") from e

        # Cap defensively
        fills = fills[-cfg.max_fetch :] if len(fills) > cfg.max_fetch else fills

        # Normalize + dedupe
        for f in fills:
            ts = _exec_time_utc(f)
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                if dt < cutoff:
                    continue

            # Fill event
            ek, ev = _build_fill_event(deployment_id=cfg.deployment_id, venue=cfg.venue, fill=f)
            if not db.seen(ek):
                appended.append(ev)
                db.mark(ek, ev["ts_utc"])
                new_count += 1

            # Fee event (optional)
            fee = _build_fee_event(deployment_id=cfg.deployment_id, venue=cfg.venue, fill=f)
            if fee:
                ek2, ev2 = fee
                if not db.seen(ek2):
                    appended.append(ev2)
                    db.mark(ek2, ev2["ts_utc"])
                    fee_count += 1

        # NO_NEW_EVENTS_HEARTBEAT:
        # If no fills/fees exist, write a heartbeat record so the ledger is never empty.
        # This does NOT fabricate broker activity; it only proves the collector ran.
        if not appended:
            hb: Dict[str, Any] = {
                "schema_version": "broker_event.v1",
                "ts_utc": iso_utc(utc_now()),
                "deployment_id": cfg.deployment_id,
                "venue": cfg.venue,
                "event_type": "heartbeat",
                "broker_order_id": None,
                "client_order_id": None,
                "event_id": None,
                "seq": None,
                "symbol": None,
                "side": None,
                "qty": None,
                "price": None,
                "fee_amount": None,
                "currency": None,
                "raw_ref": {"reason": "NO_NEW_EVENTS"},
            }
            hb["hash_sha256"] = f"sha256:{sha256_hex(stable_json_dumps(hb))}"
            appended.append(hb)


        # Append to NDJSON
        app = NdjsonAppender(_ledger_path(cfg.data))
        app.append_many(appended)

        db.commit()

        LOG.info(
            "ibkr broker_events collected new=%d fee=%d appended_lines=%d ledger=%s",
            new_count,
            fee_count,
            len(appended),
            str(_ledger_path(cfg.data)),
        )
        return 0

    finally:
        with contextlib.suppress(Exception):
            db.close()
        if ib is not None:
            with contextlib.suppress(Exception):
                ib.disconnect()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Collect IBKR broker_events (fills + fees) into append-only NDJSON.")
    ap.add_argument("--once", action="store_true", help="Run once (default).")
    ap.add_argument("--log-level", default=env_str("CHAD_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )


    _install_ib_insync_noise_filter()
    try:
        cfg = build_cfg()
    except Exception as e:
        LOG.error("configuration error: %r", e)
        return 2

    try:
        return collect_once(cfg)
    except Exception as e:
        LOG.exception("collector runtime error: %r", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
