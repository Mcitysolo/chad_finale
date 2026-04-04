#!/usr/bin/env python3
"""
chad/core/state_bus.py

Redis-backed state bus for CHAD inter-service communication.

Replaces JSON file polling for:
- STOP signal propagation (<100ms vs 60s)
- LiveGate state push to all consumers
- dynamic_caps broadcast
- Profit lock state changes

Fallback contract: if Redis is unavailable, all publish() calls log a warning
and return False. All subscribe() calls are no-ops. CHAD continues operating
on JSON file polling as before. Redis is enhancement, never dependency.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("chad.state_bus")

# Channel constants
CHANNEL_DYNAMIC_CAPS = "chad:dynamic_caps"
CHANNEL_LIVE_GATE = "chad:live_gate"
CHANNEL_STOP_SIGNAL = "chad:stop_signal"
CHANNEL_PROFIT_LOCK = "chad:profit_lock"
CHANNEL_FAST_LOOP = "chad:fast_loop"
CHANNEL_LIVE_READINESS = "chad:live_readiness"

KEY_PREFIX = "chad:"


class StateBus:
    """
    Redis-backed state bus with graceful fallback.

    All operations are fail-soft: Redis unavailability never crashes CHAD.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._db = db
        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._subscriber_threads: list = []
        self._messages_published = 0
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            self._redis = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis.ping()
            logger.info("state_bus.connected", extra={"host": self._host, "port": self._port})
        except Exception as exc:
            logger.warning("state_bus.connect_failed", extra={"error": str(exc)})
            self._redis = None

    def is_connected(self) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def ping_ms(self) -> float:
        if self._redis is None:
            return -1.0
        try:
            t0 = time.monotonic()
            self._redis.ping()
            return (time.monotonic() - t0) * 1000.0
        except Exception:
            return -1.0

    def memory_used_mb(self) -> float:
        if self._redis is None:
            return -1.0
        try:
            info = self._redis.info("memory")
            return float(info.get("used_memory", 0)) / (1024 * 1024)
        except Exception:
            return -1.0

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    def publish(self, channel: str, data: Any) -> bool:
        if self._redis is None:
            logger.warning("state_bus.publish_skip_no_redis", extra={"channel": channel})
            return False
        try:
            payload = json.dumps(data, default=str, separators=(",", ":"))
            self._redis.publish(channel, payload)
            self._messages_published += 1
            return True
        except Exception as exc:
            logger.warning("state_bus.publish_failed", extra={"channel": channel, "error": str(exc)})
            return False

    def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        if self._redis is None:
            logger.warning("state_bus.subscribe_skip_no_redis", extra={"channel": channel})
            return False
        try:
            import redis as redis_mod
            ps = self._redis.pubsub()
            ps.subscribe(channel)

            def _listener() -> None:
                try:
                    for msg in ps.listen():
                        if msg["type"] != "message":
                            continue
                        try:
                            data = json.loads(msg["data"])
                            callback(data)
                        except Exception as exc:
                            logger.warning(
                                "state_bus.callback_error",
                                extra={"channel": channel, "error": str(exc)},
                            )
                except Exception as exc:
                    logger.warning(
                        "state_bus.listener_error",
                        extra={"channel": channel, "error": str(exc)},
                    )

            t = threading.Thread(target=_listener, daemon=True, name=f"state_bus_{channel}")
            t.start()
            self._subscriber_threads.append(t)
            logger.info("state_bus.subscribed", extra={"channel": channel})
            return True
        except Exception as exc:
            logger.warning("state_bus.subscribe_failed", extra={"channel": channel, "error": str(exc)})
            return False

    # ------------------------------------------------------------------
    # Key/Value state (with TTL)
    # ------------------------------------------------------------------

    def set_state(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        if self._redis is None:
            return False
        try:
            full_key = f"{KEY_PREFIX}{key}" if not key.startswith(KEY_PREFIX) else key
            payload = json.dumps(value, default=str, separators=(",", ":"))
            self._redis.setex(full_key, ttl_seconds, payload)
            return True
        except Exception as exc:
            logger.warning("state_bus.set_state_failed", extra={"key": key, "error": str(exc)})
            return False

    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        if self._redis is None:
            return None
        try:
            full_key = f"{KEY_PREFIX}{key}" if not key.startswith(KEY_PREFIX) else key
            raw = self._redis.get(full_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("state_bus.get_state_failed", extra={"key": key, "error": str(exc)})
            return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def messages_published(self) -> int:
        return self._messages_published

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriber_threads)


# ------------------------------------------------------------------
# Publisher mixin
# ------------------------------------------------------------------

class StateBusPublisher:
    """Mixin providing typed publish methods for CHAD channels."""

    def __init__(self, bus: Optional[StateBus] = None) -> None:
        self._bus = bus

    def publish_dynamic_caps(self, caps_dict: Dict[str, Any]) -> bool:
        if self._bus is None:
            return False
        return self._bus.publish(CHANNEL_DYNAMIC_CAPS, caps_dict)

    def publish_live_gate(self, gate_decision: Dict[str, Any]) -> bool:
        if self._bus is None:
            return False
        return self._bus.publish(CHANNEL_LIVE_GATE, gate_decision)

    def publish_stop(self, reason: str) -> bool:
        if self._bus is None:
            return False
        return self._bus.publish(CHANNEL_STOP_SIGNAL, {
            "stop": True,
            "reason": reason,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def publish_profit_lock(self, state: Dict[str, Any]) -> bool:
        if self._bus is None:
            return False
        return self._bus.publish(CHANNEL_PROFIT_LOCK, state)

    def publish_fast_loop(self, state: Dict[str, Any]) -> bool:
        if self._bus is None:
            return False
        return self._bus.publish(CHANNEL_FAST_LOOP, state)

    def publish_live_readiness(self, state: Dict[str, Any]) -> bool:
        if self._bus is None:
            return False
        return self._bus.publish(CHANNEL_LIVE_READINESS, state)


# ------------------------------------------------------------------
# Subscriber mixin
# ------------------------------------------------------------------

class StateBusSubscriber:
    """Mixin providing typed subscribe methods for CHAD channels."""

    def __init__(self, bus: Optional[StateBus] = None) -> None:
        self._bus = bus

    def on_stop(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        if self._bus is None:
            return False
        return self._bus.subscribe(CHANNEL_STOP_SIGNAL, callback)

    def on_dynamic_caps(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        if self._bus is None:
            return False
        return self._bus.subscribe(CHANNEL_DYNAMIC_CAPS, callback)

    def on_live_gate(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        if self._bus is None:
            return False
        return self._bus.subscribe(CHANNEL_LIVE_GATE, callback)

    def on_profit_lock(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        if self._bus is None:
            return False
        return self._bus.subscribe(CHANNEL_PROFIT_LOCK, callback)


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_global_bus: Optional[StateBus] = None
_global_lock = threading.Lock()


def get_state_bus() -> StateBus:
    """Get or create the global StateBus singleton."""
    global _global_bus
    if _global_bus is None:
        with _global_lock:
            if _global_bus is None:
                _global_bus = StateBus()
    return _global_bus


def get_publisher() -> StateBusPublisher:
    """Get a StateBusPublisher backed by the global bus."""
    return StateBusPublisher(get_state_bus())


def get_subscriber() -> StateBusSubscriber:
    """Get a StateBusSubscriber backed by the global bus."""
    return StateBusSubscriber(get_state_bus())


def write_redis_state_json(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Write runtime/redis_state.json with current Redis health info.

    Called periodically (e.g., every 60s from orchestrator fast loop).
    """
    import os
    from pathlib import Path

    bus = get_state_bus()
    state: Dict[str, Any] = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "connected": bus.is_connected(),
        "ping_ms": round(bus.ping_ms(), 2),
        "memory_used_mb": round(bus.memory_used_mb(), 2),
        "channels_active": bus.subscriber_count,
        "messages_published": bus.messages_published,
    }

    out_path = Path(path) if path else Path(__file__).resolve().parents[2] / "runtime" / "redis_state.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    data = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

    return state
