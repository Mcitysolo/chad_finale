"""
CHAD Kraken WebSocket Client — Real-time crypto price feed (public, no API key).

Connects to wss://ws.kraken.com, subscribes to ticker channel,
and maintains an in-memory price cache written atomically to
runtime/kraken_prices.json every WRITE_INTERVAL_S seconds.

Usage:
    client = KrakenWSClient(symbols=["XBT/USD", "ETH/USD", "SOL/USD"])
    client.start()     # background thread
    client.get_price("BTC-USD")  # -> float or None
    client.stop()

Or as a standalone daemon:
    python3 -m chad.market_data.kraken_ws_client
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("chad.kraken_ws")

WS_URL = "wss://ws.kraken.com"

DEFAULT_SYMBOLS = ["XBT/USD", "ETH/USD", "SOL/USD"]

# Kraken uses XBT; CHAD normalizes to BTC
SYMBOL_MAP: Dict[str, str] = {
    "XBT/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "DOGE/USD": "DOGE-USD",
    "AVAX/USD": "AVAX-USD",
    "LINK/USD": "LINK-USD",
}

WRITE_INTERVAL_S = 10
RECONNECT_BASE_S = 1.0
RECONNECT_MAX_S = 60.0
STALE_THRESHOLD_S = 120

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/chad_finale/runtime"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def normalize_symbol(kraken_pair: str) -> str:
    """Convert Kraken pair name to CHAD normalized symbol."""
    return SYMBOL_MAP.get(kraken_pair, kraken_pair.replace("/", "-"))


def kraken_pair_from_chad(chad_symbol: str) -> Optional[str]:
    """Reverse lookup: CHAD symbol -> Kraken pair."""
    for k, v in SYMBOL_MAP.items():
        if v == chad_symbol:
            return k
    return None


@dataclass
class KrakenTick:
    symbol: str          # CHAD-normalized (e.g. BTC-USD)
    kraken_pair: str     # Original (e.g. XBT/USD)
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume_24h: float = 0.0
    ts_utc: str = ""

    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


class KrakenWSClient:
    """Thread-safe Kraken WebSocket ticker client with auto-reconnect."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        runtime_dir: Optional[Path] = None,
        write_interval_s: float = WRITE_INTERVAL_S,
    ):
        self._symbols = symbols or list(DEFAULT_SYMBOLS)
        self._runtime_dir = runtime_dir or RUNTIME_DIR
        self._write_interval = write_interval_s

        self._lock = threading.Lock()
        self._ticks: Dict[str, KrakenTick] = {}
        self._connected = False
        self._running = False
        self._ws: Any = None
        self._thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._reconnect_delay = RECONNECT_BASE_S
        self._last_message_ts = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start WebSocket connection and writer in background threads."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="kraken-ws")
        self._thread.start()
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True, name="kraken-writer")
        self._writer_thread.start()
        LOGGER.info("kraken_ws.started symbols=%s", self._symbols)

    def stop(self) -> None:
        """Cleanly stop the client."""
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5)
        self._connected = False
        LOGGER.info("kraken_ws.stopped")

    def get_price(self, symbol: str) -> Optional[float]:
        """Get last price for a CHAD-normalized symbol (e.g. BTC-USD)."""
        with self._lock:
            tick = self._ticks.get(symbol)
            if tick is None:
                return None
            return tick.last if tick.last > 0 else None

    def get_all_prices(self) -> Dict[str, float]:
        """Get all tracked prices as {symbol: last_price}."""
        with self._lock:
            return {
                sym: tick.last
                for sym, tick in self._ticks.items()
                if tick.last > 0
            }

    def get_tick(self, symbol: str) -> Optional[KrakenTick]:
        with self._lock:
            return self._ticks.get(symbol)

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    # WebSocket lifecycle
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """Reconnect loop with exponential backoff."""
        import websocket

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                LOGGER.warning("kraken_ws.run_error err=%s", exc)

            self._connected = False
            if not self._running:
                break

            LOGGER.info("kraken_ws.reconnecting delay=%.1fs", self._reconnect_delay)
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX_S)

    def _on_open(self, ws: Any) -> None:
        self._connected = True
        self._reconnect_delay = RECONNECT_BASE_S
        LOGGER.info("kraken_ws.connected")

        subscribe_msg = {
            "event": "subscribe",
            "pair": self._symbols,
            "subscription": {"name": "ticker"},
        }
        ws.send(json.dumps(subscribe_msg))
        LOGGER.info("kraken_ws.subscribed pairs=%s", self._symbols)

    def _on_message(self, ws: Any, message: str) -> None:
        self._last_message_ts = time.time()
        try:
            data = json.loads(message)
        except Exception:
            return

        # Kraken sends events as dicts and ticker updates as lists
        if isinstance(data, dict):
            # Heartbeat, system status, subscription status
            return

        if not isinstance(data, list) or len(data) < 4:
            return

        # Ticker format: [channelID, tickerData, channelName, pair]
        try:
            ticker_data = data[1]
            pair = str(data[3])
        except (IndexError, TypeError):
            return

        if not isinstance(ticker_data, dict):
            return

        chad_symbol = normalize_symbol(pair)
        now = _utc_now_iso()

        try:
            # Kraken ticker fields: a=ask, b=bid, c=last trade
            ask_arr = ticker_data.get("a", [])
            bid_arr = ticker_data.get("b", [])
            last_arr = ticker_data.get("c", [])
            vol_arr = ticker_data.get("v", [])

            tick = KrakenTick(
                symbol=chad_symbol,
                kraken_pair=pair,
                ask=float(ask_arr[0]) if ask_arr else 0.0,
                bid=float(bid_arr[0]) if bid_arr else 0.0,
                last=float(last_arr[0]) if last_arr else 0.0,
                volume_24h=float(vol_arr[1]) if len(vol_arr) > 1 else 0.0,
                ts_utc=now,
            )

            with self._lock:
                self._ticks[chad_symbol] = tick

        except (ValueError, TypeError, IndexError) as exc:
            LOGGER.debug("kraken_ws.parse_error pair=%s err=%s", pair, exc)

    def _on_error(self, ws: Any, error: Any) -> None:
        LOGGER.warning("kraken_ws.error err=%s", error)

    def _on_close(self, ws: Any, close_status_code: Any = None, close_msg: Any = None) -> None:
        self._connected = False
        LOGGER.info("kraken_ws.closed code=%s msg=%s", close_status_code, close_msg)

    # ------------------------------------------------------------------ #
    # Periodic writer
    # ------------------------------------------------------------------ #

    def _write_loop(self) -> None:
        """Periodically write price cache and feed state to runtime."""
        prices_path = self._runtime_dir / "kraken_prices.json"
        feed_state_path = self._runtime_dir / "kraken_feed_state.json"

        while self._running:
            time.sleep(self._write_interval)
            if not self._running:
                break

            try:
                now = _utc_now_iso()
                with self._lock:
                    prices = {sym: tick.last for sym, tick in self._ticks.items() if tick.last > 0}
                    ticks_snapshot = {sym: asdict(tick) for sym, tick in self._ticks.items()}

                payload = {
                    "connected": self._connected,
                    "prices": dict(sorted(prices.items())),
                    "ticks": ticks_snapshot,
                    "ts_utc": now,
                    "ttl_seconds": int(self._write_interval * 3),
                }
                _atomic_write_json(prices_path, payload)

                # Feed state
                age_s = time.time() - self._last_message_ts if self._last_message_ts > 0 else -1
                feed_state = {
                    "connected": self._connected,
                    "last_message_age_s": round(age_s, 1),
                    "symbols_tracked": sorted(prices.keys()),
                    "symbols_count": len(prices),
                    "stale": age_s > STALE_THRESHOLD_S if age_s >= 0 else True,
                    "ts_utc": now,
                }
                _atomic_write_json(feed_state_path, feed_state)

            except Exception as exc:
                LOGGER.warning("kraken_ws.write_error err=%s", exc)


# ------------------------------------------------------------------ #
# Standalone daemon entry point
# ------------------------------------------------------------------ #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    symbols_env = os.environ.get("CHAD_KRAKEN_WS_SYMBOLS", "")
    if symbols_env.strip():
        symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]
    else:
        symbols = list(DEFAULT_SYMBOLS)

    client = KrakenWSClient(symbols=symbols)
    client.start()

    LOGGER.info("kraken_ws_daemon.running symbols=%s", symbols)

    try:
        while True:
            time.sleep(60)
            prices = client.get_all_prices()
            LOGGER.info(
                "kraken_ws_daemon.heartbeat connected=%s symbols=%d prices=%s",
                client.connected,
                len(prices),
                {k: round(v, 2) for k, v in prices.items()},
            )
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()


if __name__ == "__main__":
    main()
