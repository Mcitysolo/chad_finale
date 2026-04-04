#!/usr/bin/env python3
"""
chad/tests/test_state_bus.py

Tests for Redis-backed state bus.

Uses mock Redis for unit tests (no live Redis required).
Covers: publish/subscribe, fallback, TTL, namespacing, stop signal.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Mock Redis classes
# ---------------------------------------------------------------------------

class MockPubSub:
    """Minimal mock of redis.client.PubSub."""

    def __init__(self) -> None:
        self._channels: list = []
        self._messages: list = []
        self._lock = threading.Lock()

    def subscribe(self, channel: str) -> None:
        self._channels.append(channel)

    def listen(self):
        # Yield subscription confirmation then queued messages
        for ch in self._channels:
            yield {"type": "subscribe", "channel": ch, "data": 1}
        while True:
            with self._lock:
                if self._messages:
                    yield self._messages.pop(0)
                    continue
            time.sleep(0.01)
            # Stop after a short time to avoid infinite loop in tests
            break

    def inject_message(self, channel: str, data: str) -> None:
        with self._lock:
            self._messages.append({"type": "message", "channel": channel, "data": data})


class MockRedis:
    """Minimal mock of redis.Redis for unit tests."""

    def __init__(self, fail_connect: bool = False) -> None:
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}
        self._pubsubs: List[MockPubSub] = []
        self._published: List[tuple] = []
        self._fail_connect = fail_connect

    def ping(self) -> bool:
        if self._fail_connect:
            raise ConnectionError("mock connection refused")
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = time.monotonic() + ttl

    def get(self, key: str) -> Optional[str]:
        if key in self._ttls and time.monotonic() > self._ttls[key]:
            del self._store[key]
            del self._ttls[key]
            return None
        return self._store.get(key)

    def publish(self, channel: str, data: str) -> int:
        self._published.append((channel, data))
        # Inject into all pubsubs subscribed to this channel
        for ps in self._pubsubs:
            if channel in ps._channels:
                ps.inject_message(channel, data)
        return 1

    def pubsub(self) -> MockPubSub:
        ps = MockPubSub()
        self._pubsubs.append(ps)
        return ps

    def info(self, section: str = "") -> dict:
        return {"used_memory": 1024 * 1024 * 10}  # 10 MB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_global_bus():
    """Reset the global singleton between tests."""
    import chad.core.state_bus as sb
    sb._global_bus = None
    yield
    sb._global_bus = None


@pytest.fixture
def mock_redis():
    return MockRedis()


@pytest.fixture
def bus_with_mock(mock_redis):
    """Create a StateBus with a mocked Redis connection."""
    from chad.core.state_bus import StateBus
    bus = StateBus.__new__(StateBus)
    bus._host = "127.0.0.1"
    bus._port = 6379
    bus._db = 0
    bus._redis = mock_redis
    bus._pubsub = None
    bus._subscriber_threads = []
    bus._messages_published = 0
    return bus


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStateBusConnection:
    def test_is_connected_with_mock(self, bus_with_mock):
        assert bus_with_mock.is_connected() is True

    def test_is_connected_when_redis_down(self):
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0
        assert bus.is_connected() is False

    def test_ping_ms_returns_positive(self, bus_with_mock):
        ms = bus_with_mock.ping_ms()
        assert ms >= 0.0

    def test_ping_ms_when_disconnected(self):
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0
        assert bus.ping_ms() == -1.0

    def test_memory_used_mb(self, bus_with_mock):
        mb = bus_with_mock.memory_used_mb()
        assert mb > 0.0  # Mock returns 10 MB


class TestPublishSubscribe:
    def test_publish_returns_true(self, bus_with_mock, mock_redis):
        result = bus_with_mock.publish("chad:test", {"key": "value"})
        assert result is True
        assert bus_with_mock.messages_published == 1
        assert len(mock_redis._published) == 1
        channel, data = mock_redis._published[0]
        assert channel == "chad:test"
        parsed = json.loads(data)
        assert parsed["key"] == "value"

    def test_publish_returns_false_when_disconnected(self):
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0
        result = bus.publish("chad:test", {"key": "value"})
        assert result is False

    def test_subscribe_returns_false_when_disconnected(self):
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0
        result = bus.subscribe("chad:test", lambda d: None)
        assert result is False


class TestKeyValueState:
    def test_set_and_get_state(self, bus_with_mock):
        bus_with_mock.set_state("test_key", {"foo": "bar"}, ttl_seconds=60)
        result = bus_with_mock.get_state("test_key")
        assert result == {"foo": "bar"}

    def test_key_namespacing(self, bus_with_mock, mock_redis):
        bus_with_mock.set_state("test_key", {"a": 1})
        # Key should be stored with chad: prefix
        assert "chad:test_key" in mock_redis._store

    def test_already_prefixed_key(self, bus_with_mock, mock_redis):
        bus_with_mock.set_state("chad:test_key", {"a": 1})
        # Should not double-prefix
        assert "chad:test_key" in mock_redis._store
        assert "chad:chad:test_key" not in mock_redis._store

    def test_get_missing_key(self, bus_with_mock):
        result = bus_with_mock.get_state("nonexistent")
        assert result is None

    def test_ttl_expiry(self, bus_with_mock, mock_redis):
        # Set with very short TTL
        bus_with_mock.set_state("expire_me", {"temp": True}, ttl_seconds=1)
        # Should exist now
        assert bus_with_mock.get_state("expire_me") == {"temp": True}
        # Force expiry in mock
        mock_redis._ttls["chad:expire_me"] = time.monotonic() - 1
        assert bus_with_mock.get_state("expire_me") is None

    def test_get_state_when_disconnected(self):
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0
        assert bus.get_state("any") is None

    def test_set_state_when_disconnected(self):
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0
        assert bus.set_state("any", {"x": 1}) is False


class TestPublisherMixin:
    def test_publish_dynamic_caps(self, bus_with_mock, mock_redis):
        from chad.core.state_bus import StateBusPublisher, CHANNEL_DYNAMIC_CAPS
        pub = StateBusPublisher(bus_with_mock)
        result = pub.publish_dynamic_caps({"total_equity": 994000})
        assert result is True
        assert mock_redis._published[0][0] == CHANNEL_DYNAMIC_CAPS

    def test_publish_live_gate(self, bus_with_mock, mock_redis):
        from chad.core.state_bus import StateBusPublisher, CHANNEL_LIVE_GATE
        pub = StateBusPublisher(bus_with_mock)
        result = pub.publish_live_gate({"mode": "DENY_ALL"})
        assert result is True
        assert mock_redis._published[0][0] == CHANNEL_LIVE_GATE

    def test_publish_stop(self, bus_with_mock, mock_redis):
        from chad.core.state_bus import StateBusPublisher, CHANNEL_STOP_SIGNAL
        pub = StateBusPublisher(bus_with_mock)
        result = pub.publish_stop("manual_halt")
        assert result is True
        channel, data = mock_redis._published[0]
        assert channel == CHANNEL_STOP_SIGNAL
        parsed = json.loads(data)
        assert parsed["stop"] is True
        assert parsed["reason"] == "manual_halt"

    def test_publish_profit_lock(self, bus_with_mock, mock_redis):
        from chad.core.state_bus import StateBusPublisher, CHANNEL_PROFIT_LOCK
        pub = StateBusPublisher(bus_with_mock)
        result = pub.publish_profit_lock({"locked": True})
        assert result is True
        assert mock_redis._published[0][0] == CHANNEL_PROFIT_LOCK

    def test_publisher_none_bus(self):
        from chad.core.state_bus import StateBusPublisher
        pub = StateBusPublisher(None)
        assert pub.publish_dynamic_caps({}) is False
        assert pub.publish_stop("test") is False


class TestSubscriberMixin:
    def test_subscriber_none_bus(self):
        from chad.core.state_bus import StateBusSubscriber
        sub = StateBusSubscriber(None)
        assert sub.on_stop(lambda d: None) is False
        assert sub.on_dynamic_caps(lambda d: None) is False
        assert sub.on_live_gate(lambda d: None) is False
        assert sub.on_profit_lock(lambda d: None) is False


class TestStopSignalPropagation:
    def test_stop_signal_publish(self, bus_with_mock, mock_redis):
        from chad.core.state_bus import StateBusPublisher, CHANNEL_STOP_SIGNAL
        pub = StateBusPublisher(bus_with_mock)
        pub.publish_stop("risk_limit_breached")

        assert len(mock_redis._published) == 1
        channel, raw = mock_redis._published[0]
        assert channel == CHANNEL_STOP_SIGNAL
        data = json.loads(raw)
        assert data["stop"] is True
        assert data["reason"] == "risk_limit_breached"
        assert "ts_utc" in data


class TestFallbackBehavior:
    def test_fallback_publish_logs_warning(self, caplog):
        """When Redis is down, publish returns False and logs warning."""
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0

        import logging
        with caplog.at_level(logging.WARNING, logger="chad.state_bus"):
            result = bus.publish("chad:test", {"x": 1})
        assert result is False

    def test_fallback_subscribe_noop(self):
        """When Redis is down, subscribe returns False (no-op)."""
        from chad.core.state_bus import StateBus
        bus = StateBus.__new__(StateBus)
        bus._redis = None
        bus._subscriber_threads = []
        bus._messages_published = 0

        called = []
        result = bus.subscribe("chad:test", lambda d: called.append(d))
        assert result is False
        assert len(called) == 0


class TestWriteRedisStateJson:
    def test_write_redis_state_json(self, bus_with_mock, tmp_path):
        import chad.core.state_bus as sb
        sb._global_bus = bus_with_mock

        out = tmp_path / "redis_state.json"
        state = sb.write_redis_state_json(str(out))

        assert out.exists()
        assert state["connected"] is True
        assert state["ping_ms"] >= 0
        assert "messages_published" in state

        content = json.loads(out.read_text())
        assert content["connected"] is True


class TestGlobalSingleton:
    def test_get_state_bus_creates_singleton(self):
        """get_state_bus() should return the same instance."""
        import chad.core.state_bus as sb
        bus1 = sb.get_state_bus()
        bus2 = sb.get_state_bus()
        assert bus1 is bus2

    def test_get_publisher(self):
        from chad.core.state_bus import get_publisher
        pub = get_publisher()
        assert pub is not None

    def test_get_subscriber(self):
        from chad.core.state_bus import get_subscriber
        sub = get_subscriber()
        assert sub is not None
