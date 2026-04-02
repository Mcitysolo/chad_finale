#!/usr/bin/env python3
"""
CHAD — Polygon Stocks Streamer (Production-Grade)

Primary responsibilities:
1) Connect to Polygon (WebSocket) with robust reconnect + backoff.
2) Publish runtime/feed_state.json with STRICT SSOT runtime contract:
   - always includes ts_utc + ttl_seconds (int)
   - atomic write via chad.utils.runtime_json.write_runtime_state_json
3) (Optional) Append raw events to an NDJSON ledger under data/feeds/ for audit/replay.

This module is designed to run under systemd as a long-lived service.

SSOT Alignment Notes
- This streamer is a "feed producer". If it stops or becomes stale, higher layers should fail closed.
- It does NOT perform trading, risk, policy, or broker calls.
- It never writes secrets. It never prints API keys.

Dependencies
- Requires `websockets` at runtime for WS streaming.
  If missing, the service exits non-zero with a clear error.

Environment Variables (safe defaults)
- CHAD_ROOT: repo root (default inferred)
- CHAD_RUNTIME_DIR: runtime dir (default: <CHAD_ROOT>/runtime)
- CHAD_DATA_DIR: data dir (default: <CHAD_ROOT>/data)
- CHAD_FEED_TTL_SECONDS: feed_state TTL (default: 180)
- CHAD_POLYGON_WS_URL: WebSocket URL (default: wss://socket.polygon.io/stocks)
- CHAD_POLYGON_API_KEY: Polygon API key (required)
- CHAD_POLYGON_SUBSCRIBE: comma-separated subscribe strings (default: "T.*,Q.*" meaning all trades + quotes)
- CHAD_POLYGON_LEDGER_ENABLED: "1" to enable NDJSON ledger (default: 1)
- CHAD_POLYGON_LEDGER_FLUSH_EVERY: flush every N messages (default: 50)
- CHAD_POLYGON_PING_SECONDS: ping interval seconds (default: 20)
- CHAD_POLYGON_MAX_BACKOFF_SECONDS: reconnect backoff cap (default: 30)

Exit Codes
- 0: clean shutdown
- 2: misconfiguration (missing required env, missing dependency)
- 3: runtime failure loop exceeded safety cap (rare)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from chad.utils.runtime_json import write_runtime_state_json

LOG = logging.getLogger("chad.polygon_stream")


# ----------------------------
# Utilities
# ----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _repo_root() -> Path:
    # Prefer SSOT-defined CHAD_ROOT when set by systemd.
    root = _env_str("CHAD_ROOT", "")
    if root:
        p = Path(root).expanduser()
        if p.is_dir():
            return p.resolve()
    # Fallback: backend/ -> repo root
    return Path(__file__).resolve().parents[1]


def _runtime_dir(repo_root: Path) -> Path:
    rd = _env_str("CHAD_RUNTIME_DIR", "")
    if rd:
        return Path(rd).expanduser().resolve()
    return (repo_root / "runtime").resolve()


def _data_dir(repo_root: Path) -> Path:
    dd = _env_str("CHAD_DATA_DIR", "")
    if dd:
        return Path(dd).expanduser().resolve()
    return (repo_root / "data").resolve()


def _split_csv(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


# ----------------------------
# Config + DI Interfaces
# ----------------------------

@dataclass(frozen=True)
class PolygonConfig:
    ws_url: str
    api_key: str
    subscribe: List[str]
    ping_seconds: int
    max_backoff_seconds: int


@dataclass(frozen=True)
class FeedStateConfig:
    ttl_seconds: int
    runtime_dir: Path


@dataclass(frozen=True)
class LedgerConfig:
    enabled: bool
    data_dir: Path
    flush_every: int


class EventSink(Protocol):
    async def handle(self, event: Dict[str, Any]) -> None: ...


class PolygonClient(Protocol):
    async def connect_and_stream(self, *, on_event: EventSink, stop: asyncio.Event) -> None: ...


# ----------------------------
# Feed State Publisher
# ----------------------------

class FeedStatePublisher:
    """
    Publishes runtime/feed_state.json with TTL enforced as an integer.
    Uses SSOT runtime_json writer (atomic, ts_utc + ttl_seconds).
    """
    def __init__(self, cfg: FeedStateConfig) -> None:
        self._cfg = cfg
        self._path = cfg.runtime_dir / "feed_state.json"

    def publish_ok(self, *, feed_name: str, last_update_ts_utc: str, freshness_seconds: float) -> None:
        payload: Dict[str, Any] = {
            "feeds": {
                feed_name: {
                    "last_update_ts_utc": last_update_ts_utc,
                    "freshness_seconds": float(max(0.0, freshness_seconds)),
                }
            },
        }
        # Atomic + inject ts/ttl
        write_runtime_state_json(self._path, payload, ttl_seconds=int(self._cfg.ttl_seconds), inject_ts=True)


# ----------------------------
# Optional NDJSON Ledger Sink
# ----------------------------

class NdjsonLedgerSink:
    """
    Append-only NDJSON writer with controlled flushing.
    Writes under: data/feeds/polygon_stocks/YYYYMMDD.ndjson

    This is optional but recommended for audit/replay.
    """
    def __init__(self, cfg: LedgerConfig) -> None:
        self._cfg = cfg
        self._cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._feeds_dir = (cfg.data_dir / "feeds" / "polygon_stocks")
        self._feeds_dir.mkdir(parents=True, exist_ok=True)

        self._count_since_flush = 0
        self._fh: Optional[Any] = None
        self._current_day: Optional[str] = None

    def _ensure_open(self) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self._fh is not None and self._current_day == day:
            return
        # rotate
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
        path = self._feeds_dir / f"POLYGON_STOCKS_{day}.ndjson"
        self._fh = open(path, "a", encoding="utf-8")  # append-only
        self._current_day = day
        self._count_since_flush = 0

    async def handle(self, event: Dict[str, Any]) -> None:
        if not self._cfg.enabled:
            return
        self._ensure_open()
        assert self._fh is not None
        self._fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._count_since_flush += 1
        if self._count_since_flush >= self._cfg.flush_every:
            self._fh.flush()
            self._count_since_flush = 0

    async def close(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.flush()
            with contextlib.suppress(Exception):
                self._fh.close()
        self._fh = None
        self._current_day = None


# ----------------------------
# Multiplex Sink (fan-out)
# ----------------------------

class MultiplexSink:
    def __init__(self, sinks: Sequence[EventSink]) -> None:
        self._sinks = list(sinks)

    async def handle(self, event: Dict[str, Any]) -> None:
        # Fan out sequentially; each sink must be fast/non-blocking.
        # If a sink fails, we log but do not crash the stream.
        for s in self._sinks:
            try:
                await s.handle(event)
            except Exception:
                LOG.exception("event sink failed: %s", s)


# ----------------------------
# Polygon WebSocket Client
# ----------------------------

class PolygonWebSocketClient:
    """
    Robust WS client:
    - Auth
    - Subscribe
    - Read loop
    - Ping keepalive
    - Exponential backoff + jitter on reconnect
    """
    def __init__(self, cfg: PolygonConfig) -> None:
        self._cfg = cfg

        # Import websockets lazily to give clear dependency failure.
        try:
            import websockets  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: websockets. Install it in the venv to use PolygonWebSocketClient."
            ) from e
        self._websockets = websockets

    async def connect_and_stream(self, *, on_event: EventSink, stop: asyncio.Event) -> None:
        ws_url = self._cfg.ws_url
        backoff = 1.0

        while not stop.is_set():
            try:
                LOG.info("connecting websocket url=%s", ws_url)
                async with self._websockets.connect(
                    ws_url,
                    ping_interval=self._cfg.ping_seconds,
                    close_timeout=5,
                    max_queue=1024,
                ) as ws:
                    backoff = 1.0  # reset after successful connect

                    # Auth
                    await ws.send(json.dumps({"action": "auth", "params": self._cfg.api_key}))
                    # Subscribe
                    if self._cfg.subscribe:
                        await ws.send(json.dumps({"action": "subscribe", "params": ",".join(self._cfg.subscribe)}))

                    LOG.info("websocket connected and subscribed=%s", self._cfg.subscribe)

                    # Read loop
                    async for msg in ws:
                        if stop.is_set():
                            break
                        # Polygon typically sends JSON arrays
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue

                        # Normalize into dict events for downstream sinks
                        # If Polygon returns list of events, emit each.
                        if isinstance(data, list):
                            for ev in data:
                                if isinstance(ev, dict):
                                    await on_event.handle(ev)
                        elif isinstance(data, dict):
                            await on_event.handle(data)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Reconnect with backoff + jitter
                LOG.warning("websocket error: %s", e)
                sleep_s = min(float(self._cfg.max_backoff_seconds), backoff) * (0.8 + random.random() * 0.4)
                LOG.info("reconnecting in %.2fs", sleep_s)
                await asyncio.sleep(sleep_s)
                backoff = min(float(self._cfg.max_backoff_seconds), backoff * 2.0)


# ----------------------------
# Feed Supervisor (glues everything together)
# ----------------------------

class FeedSupervisor:
    def __init__(self, *, client: PolygonClient, sink: EventSink, feed_state: FeedStatePublisher) -> None:
        self._client = client
        self._sink = sink
        self._feed_state = feed_state
        self._last_event_monotonic: Optional[float] = None
        self._last_event_ts_utc: Optional[str] = None

    async def _handle_event(self, ev: Dict[str, Any]) -> None:
        # update freshness clock first (even if sinks fail)
        now_m = time.monotonic()
        self._last_event_monotonic = now_m
        self._last_event_ts_utc = _utc_now_iso()

        # fan out
        await self._sink.handle(ev)

        # publish feed_state opportunistically on each event (cheap)
        freshness = 0.0
        self._feed_state.publish_ok(
            feed_name="polygon_stocks",
            last_update_ts_utc=self._last_event_ts_utc,
            freshness_seconds=freshness,
        )

    async def run(self, *, stop: asyncio.Event) -> None:
        # Wrap self._handle_event as an EventSink
        supervisor = self

        class _SinkAdapter:
            async def handle(self, event: Dict[str, Any]) -> None:
                await supervisor._handle_event(event)

        await self._client.connect_and_stream(on_event=_SinkAdapter(), stop=stop)


# ----------------------------
# Main
# ----------------------------

def _configure_logging() -> None:
    level = _env_str("CHAD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )


def _build_config(repo_root: Path) -> tuple[PolygonConfig, FeedStateConfig, LedgerConfig]:
    api_key = _env_str("CHAD_POLYGON_API_KEY", "")
    if not api_key:
        raise ValueError("CHAD_POLYGON_API_KEY is required")

    ws_url = _env_str("CHAD_POLYGON_WS_URL", "wss://socket.polygon.io/stocks")
    subscribe = _split_csv(_env_str("CHAD_POLYGON_SUBSCRIBE", "T.*,Q.*"))
    ping_seconds = _env_int("CHAD_POLYGON_PING_SECONDS", 20)
    max_backoff_seconds = _env_int("CHAD_POLYGON_MAX_BACKOFF_SECONDS", 30)

    runtime_dir = _runtime_dir(repo_root)
    data_dir = _data_dir(repo_root)

    feed_ttl = _env_int("CHAD_FEED_TTL_SECONDS", 180)

    ledger_enabled = _env_bool("CHAD_POLYGON_LEDGER_ENABLED", True)
    flush_every = _env_int("CHAD_POLYGON_LEDGER_FLUSH_EVERY", 50)

    return (
        PolygonConfig(
            ws_url=ws_url,
            api_key=api_key,
            subscribe=subscribe,
            ping_seconds=ping_seconds,
            max_backoff_seconds=max_backoff_seconds,
        ),
        FeedStateConfig(ttl_seconds=feed_ttl, runtime_dir=runtime_dir),
        LedgerConfig(enabled=ledger_enabled, data_dir=data_dir, flush_every=flush_every),
    )


async def _amain() -> int:
    _configure_logging()
    repo_root = _repo_root()

    try:
        poly_cfg, feed_cfg, ledger_cfg = _build_config(repo_root)
    except Exception as e:
        LOG.error("configuration error: %s", e)
        return 2

    # Ensure dirs exist
    feed_cfg.runtime_dir.mkdir(parents=True, exist_ok=True)
    ledger_cfg.data_dir.mkdir(parents=True, exist_ok=True)

    stop = asyncio.Event()

    def _handle_sig(_sig: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    feed_state = FeedStatePublisher(feed_cfg)
    ledger_sink = NdjsonLedgerSink(ledger_cfg)
    sink = MultiplexSink([ledger_sink])

    try:
        client = PolygonWebSocketClient(poly_cfg)
    except Exception as e:
        LOG.error("%s", e)
        return 2

    supervisor = FeedSupervisor(client=client, sink=sink, feed_state=feed_state)

    LOG.info(
        "starting polygon streamer runtime_dir=%s data_dir=%s subscribe=%s ttl=%ss ledger=%s",
        str(feed_cfg.runtime_dir),
        str(ledger_cfg.data_dir),
        poly_cfg.subscribe,
        feed_cfg.ttl_seconds,
        ledger_cfg.enabled,
    )

    try:
        await supervisor.run(stop=stop)
        return 0
    except asyncio.CancelledError:
        return 0
    except Exception:
        LOG.exception("fatal run error")
        return 3
    finally:
        with contextlib.suppress(Exception):
            await ledger_sink.close()


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
