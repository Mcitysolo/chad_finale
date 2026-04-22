"""Tests for Phase-8 Session 5 VoteCollector (S1)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chad.analytics.vote_collector import VoteCollector


@dataclass
class _FakeIntent:
    symbol: str
    side: str
    signal_family: str = "momentum"


def test_min_votes_1_immediate_passthrough():
    collector = VoteCollector(min_votes=1, window_seconds=60)
    released = collector.submit(_FakeIntent("SPY", "BUY", "momentum"))
    assert len(released) == 1
    assert released[0].symbol == "SPY"
    assert collector.pending_count() == 0


def test_min_votes_2_holds_until_second_distinct_family():
    collector = VoteCollector(min_votes=2, window_seconds=60)
    first = collector.submit(_FakeIntent("SPY", "BUY", "momentum"))
    assert first == []
    assert collector.pending_count() == 1

    # Same family resubmission should NOT satisfy quorum.
    second_same = collector.submit(_FakeIntent("SPY", "BUY", "momentum"))
    assert second_same == []
    assert collector.pending_count() == 2

    # A distinct family releases both queued intents.
    third = collector.submit(_FakeIntent("SPY", "BUY", "mean_reversion"))
    assert len(third) == 3
    assert collector.pending_count() == 0


def test_expired_intents_discarded():
    collector = VoteCollector(min_votes=2, window_seconds=10)
    # Submit at t=0.
    collector.submit(_FakeIntent("SPY", "BUY", "momentum"), now=0.0)
    # Advance past the window.
    dropped = collector.flush_expired(now=100.0)
    assert len(dropped) == 1
    assert collector.pending_count() == 0


def test_different_symbols_dont_share_votes():
    collector = VoteCollector(min_votes=2, window_seconds=60)
    collector.submit(_FakeIntent("SPY", "BUY", "momentum"))
    collector.submit(_FakeIntent("QQQ", "BUY", "mean_reversion"))
    assert collector.pending_count() == 2
    # A third family voting on SPY should release only SPY's bucket.
    released = collector.submit(_FakeIntent("SPY", "BUY", "volatility"))
    released_symbols = {r.symbol for r in released}
    assert released_symbols == {"SPY"}
    assert collector.pending_count() == 1  # QQQ remains pending


def test_different_sides_dont_share_votes():
    collector = VoteCollector(min_votes=2, window_seconds=60)
    collector.submit(_FakeIntent("SPY", "BUY", "momentum"))
    released = collector.submit(_FakeIntent("SPY", "SELL", "mean_reversion"))
    # BUY and SELL are separate buckets — neither has 2 distinct families.
    assert released == []
    assert collector.pending_count() == 2


def test_stale_votes_purged_before_counting():
    """Stale votes inside the same bucket must not count toward a fresh quorum."""
    collector = VoteCollector(min_votes=2, window_seconds=10)
    collector.submit(_FakeIntent("SPY", "BUY", "momentum"), now=0.0)
    # A fresh momentum vote at t=50 — the first is stale and should be purged.
    released = collector.submit(_FakeIntent("SPY", "BUY", "momentum"), now=50.0)
    assert released == []
    assert collector.pending_count() == 1  # only the fresh one survived


def test_side_case_insensitive():
    """IBKR 'BUY' and Kraken 'buy' should bucket together for the same side."""
    collector = VoteCollector(min_votes=2, window_seconds=60)
    collector.submit(_FakeIntent("BTC-USD", "BUY", "momentum"))
    released = collector.submit(_FakeIntent("BTC-USD", "buy", "mean_reversion"))
    assert len(released) == 2


def test_clear_resets_state():
    collector = VoteCollector(min_votes=3, window_seconds=60)
    collector.submit(_FakeIntent("SPY", "BUY", "momentum"))
    collector.submit(_FakeIntent("QQQ", "SELL", "trend"))
    assert collector.pending_count() == 2
    collector.clear()
    assert collector.pending_count() == 0


def test_null_intent_safely_returns_empty():
    collector = VoteCollector(min_votes=2)
    assert collector.submit(None) == []
    assert collector.pending_count() == 0
