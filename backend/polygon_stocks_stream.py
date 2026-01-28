"""
Polygon Stocks NDJSON Streamer for CHAD

Purpose
-------
Continuously pulls latest trades from Polygon for a configured universe of
tickers and writes them as NDJSON lines to a date-rotated file:

    data/feeds/polygon_stocks_YYYYMMDD.ndjson

Each line is a JSON object with:
    - timestamp_utc
    - ticker
    - price
    - size
    - exchange
    - raw (the raw trade model converted to dict)

Configuration
-------------
1) API key:
   - Export POLYGON_API_KEY in the environment for the service, OR
   - Put it into /home/ubuntu/CHAD FINALE/secrets/polygon.env as:
         POLYGON_API_KEY=your_key_here
     and ensure your systemd unit loads it via EnvironmentFile.

2) Universe file:
   - /home/ubuntu/CHAD FINALE/control/polygon_universe.txt
   - One ticker per line, e.g.:
         SPY
         QQQ
         AAPL
         MSFT

Runtime
-------
- Intended to be run under systemd as chad-polygon-stocks.service
- Poll interval is configurable via POLYGON_POLL_SECONDS env (default 5s).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "runtime")).resolve()
FEED_STATE_PATH = RUNTIME_DIR / "feed_state.json"

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

def _write_feed_state(feed_name: str, last_update_ts_utc: str, freshness_seconds: float) -> None:
    payload = {
        "ts_utc": last_update_ts_utc,
        "feeds": {
            feed_name: {
                "last_update_ts_utc": last_update_ts_utc,
                "freshness_seconds": freshness_seconds,
            }
        }
    }
    _atomic_write_json(FEED_STATE_PATH, payload)
from typing import Any, Dict, List, Optional

from polygon import RESTClient


BASE_DIR = Path("/home/ubuntu/CHAD FINALE").resolve()
DATA_DIR = BASE_DIR / "data" / "feeds"
CONTROL_DIR = BASE_DIR / "control"
LOGS_DIR = BASE_DIR / "logs"
UNIVERSE_FILE = CONTROL_DIR / "polygon_universe.txt"

POLL_SECONDS = float(os.getenv("POLYGON_POLL_SECONDS", "5"))


@dataclass
class TradeRecord:
    timestamp_utc: str
    ticker: str
    price: float
    size: int
    exchange: Optional[int]
    raw: Dict[str, Any]


class GracefulExit(Exception):
    """Raised to trigger clean shutdown."""


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # type: ignore[override]
        raise GracefulExit(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "polygon_stocks.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Polygon streamer starting up")


def _load_api_key() -> str:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        # Try a dotenv-style secrets file if present
        secrets_env = BASE_DIR / "secrets" / "polygon.env"
        if secrets_env.exists():
            for line in secrets_env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("POLYGON_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        logging.error(
            "POLYGON_API_KEY not set. "
            "Set it in the environment or in secrets/polygon.env"
        )
        sys.exit(1)

    return api_key


def _load_universe() -> List[str]:
    if not UNIVERSE_FILE.exists():
        logging.error(
            "Universe file %s not found. "
            "Create it with one ticker per line (e.g. SPY, QQQ, AAPL).",
            UNIVERSE_FILE,
        )
        sys.exit(1)

    tickers: List[str] = []
    for line in UNIVERSE_FILE.read_text(encoding="utf-8").splitlines():
        t = line.strip().upper()
        if not t or t.startswith("#"):
            continue
        tickers.append(t)

    if not tickers:
        logging.error("Universe file %s is empty.", UNIVERSE_FILE)
        sys.exit(1)

    logging.info("Loaded %d tickers from %s", len(tickers), UNIVERSE_FILE)
    return tickers


def _open_ndjson_file(current_date: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"polygon_stocks_{current_date}.ndjson"
    path = DATA_DIR / filename
    # open in append mode, line-buffered
    return path.open("a", encoding="utf-8")


def _trade_to_record(ticker: str, trade: Any) -> TradeRecord:
    """
    Convert Polygon LastTrade model to our normalized TradeRecord.
    """
    ts = datetime.now(timezone.utc).isoformat()
    price = float(getattr(trade, "price", 0.0))
    size = int(getattr(trade, "size", 0))
    exchange = getattr(trade, "exchange", None)

    # trade.dict() is available on polygon models; fallback to __dict__
    try:
        raw = trade.__dict__
    except Exception:
        try:
            raw = trade.dict()
        except Exception:
            raw = {"repr": repr(trade)}

    return TradeRecord(
        timestamp_utc=ts,
        ticker=ticker,
        price=price,
        size=size,
        exchange=exchange,
        raw=raw,
    )


def main() -> None:
    _install_signal_handlers()
    _setup_logging()

    api_key = _load_api_key()
    tickers = _load_universe()

    client = RESTClient(api_key=api_key)
    logging.info(
        "Polygon RESTClient initialized for %d tickers (poll interval=%ss)",
        len(tickers),
        POLL_SECONDS,
    )

    current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_fh = _open_ndjson_file(current_date)
    logging.info("Writing NDJSON to %s", out_fh.name)

    try:
        while True:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y%m%d")

            # rotate file at midnight UTC
            if date_str != current_date:
                logging.info("Date changed %s -> %s, rotating output file", current_date, date_str)
                out_fh.close()
                current_date = date_str
                out_fh = _open_ndjson_file(current_date)
                logging.info("New NDJSON file: %s", out_fh.name)

            for ticker in tickers:
                try:
                    trade = client.get_last_trade(ticker)
                    rec = _trade_to_record(ticker, trade)
                    out_fh.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
                except Exception as exc:
                    logging.warning(
                        "Failed to fetch/write last trade for %s: %r", ticker, exc
                    )
                    continue

            out_fh.flush()
            # update feed_state freshness proof
            _write_feed_state('polygon_stocks', __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'), 0.0)
            time.sleep(POLL_SECONDS)
    except GracefulExit as exc:
        logging.info("Graceful shutdown: %s", exc)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down.")
    finally:
        try:
            out_fh.close()
        except Exception:
            pass
        logging.info("Polygon streamer stopped.")


class GracefulExit(Exception):
    pass


if __name__ == "__main__":
    main()
