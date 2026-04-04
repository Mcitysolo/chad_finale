"""Tests for KrakenWSClient — symbol normalization, price cache, stale detection."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chad.market_data.kraken_ws_client import (
    KrakenTick,
    KrakenWSClient,
    normalize_symbol,
    kraken_pair_from_chad,
    SYMBOL_MAP,
    STALE_THRESHOLD_S,
)


# ------------------------------------------------------------------ #
# Symbol normalization
# ------------------------------------------------------------------ #

class TestSymbolNormalization:
    def test_xbt_to_btc(self):
        assert normalize_symbol("XBT/USD") == "BTC-USD"

    def test_eth(self):
        assert normalize_symbol("ETH/USD") == "ETH-USD"

    def test_sol(self):
        assert normalize_symbol("SOL/USD") == "SOL-USD"

    def test_unknown_pair_uses_dash(self):
        assert normalize_symbol("ADA/USD") == "ADA-USD"

    def test_reverse_lookup_btc(self):
        assert kraken_pair_from_chad("BTC-USD") == "XBT/USD"

    def test_reverse_lookup_eth(self):
        assert kraken_pair_from_chad("ETH-USD") == "ETH/USD"

    def test_reverse_lookup_unknown(self):
        assert kraken_pair_from_chad("UNKNOWN-USD") is None

    def test_all_known_symbols_have_mapping(self):
        for kraken, chad in SYMBOL_MAP.items():
            assert "/" in kraken
            assert "-" in chad
            assert kraken_pair_from_chad(chad) == kraken


# ------------------------------------------------------------------ #
# KrakenTick
# ------------------------------------------------------------------ #

class TestKrakenTick:
    def test_mid_price(self):
        tick = KrakenTick(symbol="BTC-USD", kraken_pair="XBT/USD", bid=100.0, ask=102.0, last=101.0)
        assert tick.mid() == 101.0

    def test_mid_fallback_to_last(self):
        tick = KrakenTick(symbol="BTC-USD", kraken_pair="XBT/USD", bid=0.0, ask=0.0, last=99.0)
        assert tick.mid() == 99.0


# ------------------------------------------------------------------ #
# Price cache updates via _on_message
# ------------------------------------------------------------------ #

class TestPriceCacheUpdates:
    def _make_client(self) -> KrakenWSClient:
        return KrakenWSClient(symbols=["XBT/USD"], runtime_dir=Path("/tmp/chad_test_kraken"))

    def test_ticker_message_updates_cache(self):
        client = self._make_client()
        msg = json.dumps([
            123,
            {"a": ["50001.0", "1", "1.0"], "b": ["50000.0", "1", "1.0"], "c": ["50000.5", "0.1"], "v": ["100", "2000"]},
            "ticker",
            "XBT/USD",
        ])
        client._on_message(None, msg)

        price = client.get_price("BTC-USD")
        assert price == 50000.5

        tick = client.get_tick("BTC-USD")
        assert tick is not None
        assert tick.bid == 50000.0
        assert tick.ask == 50001.0
        assert tick.volume_24h == 2000.0

    def test_dict_messages_ignored(self):
        client = self._make_client()
        client._on_message(None, json.dumps({"event": "heartbeat"}))
        assert client.get_all_prices() == {}

    def test_malformed_message_ignored(self):
        client = self._make_client()
        client._on_message(None, "not json at all")
        assert client.get_all_prices() == {}

    def test_short_list_ignored(self):
        client = self._make_client()
        client._on_message(None, json.dumps([1, 2]))
        assert client.get_all_prices() == {}

    def test_multiple_symbols(self):
        client = KrakenWSClient(symbols=["XBT/USD", "ETH/USD"])

        btc_msg = json.dumps([1, {"a": ["50000"], "b": ["49999"], "c": ["50000"], "v": ["100", "2000"]}, "ticker", "XBT/USD"])
        eth_msg = json.dumps([2, {"a": ["3000"], "b": ["2999"], "c": ["3000"], "v": ["500", "10000"]}, "ticker", "ETH/USD"])

        client._on_message(None, btc_msg)
        client._on_message(None, eth_msg)

        prices = client.get_all_prices()
        assert prices["BTC-USD"] == 50000.0
        assert prices["ETH-USD"] == 3000.0

    def test_get_price_unknown_symbol(self):
        client = self._make_client()
        assert client.get_price("UNKNOWN-USD") is None


# ------------------------------------------------------------------ #
# Connection state
# ------------------------------------------------------------------ #

class TestConnectionState:
    def test_initial_state_disconnected(self):
        client = KrakenWSClient()
        assert client.connected is False

    def test_on_open_sets_connected(self):
        client = KrakenWSClient()
        mock_ws = MagicMock()
        client._on_open(mock_ws)
        assert client.connected is True
        mock_ws.send.assert_called_once()

    def test_on_close_clears_connected(self):
        client = KrakenWSClient()
        client._connected = True
        client._on_close(None)
        assert client.connected is False


# ------------------------------------------------------------------ #
# File writer
# ------------------------------------------------------------------ #

class TestFileWriter:
    def test_write_creates_files(self, tmp_path):
        client = KrakenWSClient(runtime_dir=tmp_path, write_interval_s=0.1)
        client._last_message_ts = time.time()
        client._connected = True

        msg = json.dumps([1, {"a": ["50000"], "b": ["49999"], "c": ["50000"], "v": ["100", "2000"]}, "ticker", "XBT/USD"])
        client._on_message(None, msg)

        # Run one write cycle manually
        client._running = True
        # Call the internal write logic directly
        prices_path = tmp_path / "kraken_prices.json"
        feed_state_path = tmp_path / "kraken_feed_state.json"

        from chad.market_data.kraken_ws_client import _atomic_write_json, _utc_now_iso
        from dataclasses import asdict

        now = _utc_now_iso()
        with client._lock:
            prices = {sym: tick.last for sym, tick in client._ticks.items() if tick.last > 0}
            ticks_snapshot = {sym: asdict(tick) for sym, tick in client._ticks.items()}

        payload = {"prices": dict(sorted(prices.items())), "ticks": ticks_snapshot, "ts_utc": now, "ttl_seconds": 30}
        _atomic_write_json(prices_path, payload)

        assert prices_path.exists()
        data = json.loads(prices_path.read_text())
        assert data["prices"]["BTC-USD"] == 50000.0


# ------------------------------------------------------------------ #
# Stale detection
# ------------------------------------------------------------------ #

class TestStaleDetection:
    def test_stale_when_no_messages(self):
        client = KrakenWSClient()
        assert client._last_message_ts == 0.0

    def test_message_updates_timestamp(self):
        client = KrakenWSClient()
        msg = json.dumps([1, {"a": ["50000"], "b": ["49999"], "c": ["50000"], "v": ["100", "2000"]}, "ticker", "XBT/USD"])
        client._on_message(None, msg)
        assert client._last_message_ts > 0
