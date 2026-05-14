"""Phase A Item 5 regression tests — float-aware liquidity gate."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import pytest

from chad.utils import liquidity as lq
from chad.utils.liquidity import (
    LiquidityClass,
    THIN_MIN_CONFIDENCE,
    blocks_thin_entry,
    classify,
    clear_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_uniform_bars(
    root: Path,
    symbol: str,
    *,
    close: float,
    volume: float,
    n: int = 20,
) -> Path:
    dir_path = root / "data" / "daily_bars"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{symbol}.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(n):
            fh.write(json.dumps({"close": float(close), "volume": float(volume)}) + "\n")
    return path


def _write_bars(
    root: Path,
    symbol: str,
    rows: Iterable[Tuple[float, float]],
) -> Path:
    dir_path = root / "data" / "daily_bars"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{symbol}.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        for close_val, volume_val in rows:
            fh.write(json.dumps({"close": float(close_val), "volume": float(volume_val)}) + "\n")
    return path


@pytest.fixture(autouse=True)
def _reset_liquidity_cache() -> None:
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Test 1 — missing bar file fails open (UNKNOWN)
# ---------------------------------------------------------------------------


def test_classify_missing_file_returns_unknown(tmp_path: Path) -> None:
    cls = classify("NOPE", repo_root=tmp_path)
    assert cls is LiquidityClass.UNKNOWN


# ---------------------------------------------------------------------------
# Test 2 — LARGE classification (ADDV >= $500M)
# ---------------------------------------------------------------------------


def test_classify_large(tmp_path: Path) -> None:
    # close * volume = 500 * 5_000_000 = $2.5B per day
    _write_uniform_bars(tmp_path, "BIG", close=500.0, volume=5_000_000.0)
    assert classify("BIG", repo_root=tmp_path) is LiquidityClass.LARGE


# ---------------------------------------------------------------------------
# Test 3 — STANDARD classification ($50M <= ADDV < $500M)
# ---------------------------------------------------------------------------


def test_classify_standard(tmp_path: Path) -> None:
    # close * volume = 100 * 1_000_000 = $100M per day
    _write_uniform_bars(tmp_path, "MID", close=100.0, volume=1_000_000.0)
    assert classify("MID", repo_root=tmp_path) is LiquidityClass.STANDARD


# ---------------------------------------------------------------------------
# Test 4 — THIN classification (ADDV < $50M)
# ---------------------------------------------------------------------------


def test_classify_thin(tmp_path: Path) -> None:
    # close * volume = 5 * 100_000 = $500K per day
    _write_uniform_bars(tmp_path, "TINY", close=5.0, volume=100_000.0)
    assert classify("TINY", repo_root=tmp_path) is LiquidityClass.THIN


# ---------------------------------------------------------------------------
# Test 5 — blocks_thin_entry returns False for LARGE
# ---------------------------------------------------------------------------


def test_blocks_thin_entry_large_passes(tmp_path: Path) -> None:
    _write_uniform_bars(tmp_path, "BIG", close=500.0, volume=5_000_000.0)
    blocked, cls, required = blocks_thin_entry(
        "BIG", confidence=0.55, repo_root=tmp_path,
    )
    assert blocked is False
    assert cls is LiquidityClass.LARGE
    assert required == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Test 6 — blocks_thin_entry returns True for THIN when confidence < 0.80
# ---------------------------------------------------------------------------


def test_blocks_thin_entry_thin_low_conf_blocks(tmp_path: Path) -> None:
    _write_uniform_bars(tmp_path, "TINY", close=5.0, volume=100_000.0)
    blocked, cls, required = blocks_thin_entry(
        "TINY", confidence=0.70, repo_root=tmp_path,
    )
    assert blocked is True
    assert cls is LiquidityClass.THIN
    assert required == pytest.approx(THIN_MIN_CONFIDENCE)


# ---------------------------------------------------------------------------
# Test 7 — blocks_thin_entry returns False for THIN when confidence >= 0.80
# ---------------------------------------------------------------------------


def test_blocks_thin_entry_thin_high_conf_passes(tmp_path: Path) -> None:
    _write_uniform_bars(tmp_path, "TINY", close=5.0, volume=100_000.0)
    blocked, cls, required = blocks_thin_entry(
        "TINY", confidence=0.85, repo_root=tmp_path,
    )
    assert blocked is False
    assert cls is LiquidityClass.THIN
    assert required == pytest.approx(THIN_MIN_CONFIDENCE)


# ---------------------------------------------------------------------------
# Test 8 — blocks_thin_entry returns False for UNKNOWN (fail-open)
# ---------------------------------------------------------------------------


def test_blocks_thin_entry_unknown_fails_open(tmp_path: Path) -> None:
    blocked, cls, required = blocks_thin_entry(
        "MISSING", confidence=0.20, repo_root=tmp_path,
    )
    assert blocked is False
    assert cls is LiquidityClass.UNKNOWN
    assert required == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Test 9 — classify cache short-circuits repeated lookups
# ---------------------------------------------------------------------------


def test_classify_cache_calls_compute_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_uniform_bars(tmp_path, "FOO", close=500.0, volume=5_000_000.0)
    counter = {"n": 0}
    original = lq._compute_addv

    def wrapped(symbol, repo_root=None):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return original(symbol, repo_root=repo_root)

    monkeypatch.setattr(lq, "_compute_addv", wrapped)
    first = classify("FOO", repo_root=tmp_path)
    second = classify("FOO", repo_root=tmp_path)
    assert first is second is LiquidityClass.LARGE
    assert counter["n"] == 1


# ---------------------------------------------------------------------------
# Test 10 — cache key includes repo_root (no cross-root contamination)
# ---------------------------------------------------------------------------


def test_cache_key_includes_repo_root(tmp_path: Path) -> None:
    root_a = tmp_path / "A"
    root_b = tmp_path / "B"
    _write_uniform_bars(root_a, "FOO", close=500.0, volume=5_000_000.0)  # LARGE
    _write_uniform_bars(root_b, "FOO", close=5.0, volume=100_000.0)      # THIN
    assert classify("FOO", repo_root=root_a) is LiquidityClass.LARGE
    assert classify("FOO", repo_root=root_b) is LiquidityClass.THIN


# ---------------------------------------------------------------------------
# Test 11 — malformed NDJSON lines are skipped safely
# ---------------------------------------------------------------------------


def test_malformed_ndjson_line_skipped(tmp_path: Path) -> None:
    dir_path = tmp_path / "data" / "daily_bars"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "FOO.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("{not valid json}\n")
        for _ in range(10):
            fh.write(json.dumps({"close": 500.0, "volume": 5_000_000.0}) + "\n")
    cls = classify("FOO", repo_root=tmp_path)
    assert cls is LiquidityClass.LARGE


# ---------------------------------------------------------------------------
# Test 12 — alpha_intraday._build_signal does NOT gate futures (MES)
# ---------------------------------------------------------------------------


def test_alpha_intraday_futures_not_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    from chad.strategies import alpha_intraday as ai
    from chad.types import SignalSide

    def boom(*_a, **_kw):  # type: ignore[no-untyped-def]
        raise AssertionError("blocks_thin_entry must not be called for futures")

    monkeypatch.setattr(ai, "blocks_thin_entry", boom)
    sig = ai._build_signal(
        "MES",
        SignalSide.BUY,
        confidence=0.85,
        trigger="vol_explosion",
        timeframe="1m",
        atr=2.0,
        tier_max_risk_usd=None,
    )
    assert sig is not None
    assert sig.meta["liquidity_class"] == "unknown"
    assert sig.meta["liquidity_required_confidence"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Test 13 — alpha_intraday._build_signal blocks thin equity below floor
# ---------------------------------------------------------------------------


def test_alpha_intraday_thin_equity_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from chad.strategies import alpha_intraday as ai
    from chad.types import SignalSide

    def fake_block(symbol, conf, **_kw):  # type: ignore[no-untyped-def]
        return True, LiquidityClass.THIN, THIN_MIN_CONFIDENCE

    monkeypatch.setattr(ai, "blocks_thin_entry", fake_block)
    sig = ai._build_signal(
        "AAPL",
        SignalSide.BUY,
        confidence=0.55,
        trigger="vol_explosion",
        timeframe="1m",
        atr=2.0,
    )
    assert sig is None


# ---------------------------------------------------------------------------
# Test 14 — alpha_handler does not raise with empty bars/prices
# ---------------------------------------------------------------------------


def test_alpha_handler_no_raise_on_empty() -> None:
    from chad.strategies.alpha import alpha_handler
    from chad.types import LegendConsensus, MarketContext, PortfolioSnapshot

    now = datetime.now(timezone.utc)
    ctx = MarketContext(
        now=now,
        ticks={},
        legend=LegendConsensus(as_of=now, weights={}),
        portfolio=PortfolioSnapshot(timestamp=now, cash=100_000.0, positions={}),
        bars={},
    )
    result = alpha_handler(ctx)
    assert result == []


# ---------------------------------------------------------------------------
# Test 15 — liquidity_class meta present on alpha_intraday equity signal
# ---------------------------------------------------------------------------


def test_alpha_intraday_equity_signal_carries_liquidity_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chad.strategies import alpha_intraday as ai
    from chad.types import SignalSide

    def fake_block(symbol, conf, **_kw):  # type: ignore[no-untyped-def]
        return False, LiquidityClass.STANDARD, 0.5

    monkeypatch.setattr(ai, "blocks_thin_entry", fake_block)
    sig = ai._build_signal(
        "AAPL",
        SignalSide.BUY,
        confidence=0.85,
        trigger="vol_explosion",
        timeframe="1m",
        atr=2.0,
    )
    assert sig is not None
    assert sig.meta["liquidity_class"] == "standard"
    assert sig.meta["liquidity_required_confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 16 — classify from JSON object format ({"bars": [...]})
# ---------------------------------------------------------------------------


def test_classify_json_object_bars_format(tmp_path: Path) -> None:
    dir_path = tmp_path / "data" / "bars" / "1d"
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "bars": [
            {"close": 500.0, "volume": 5_000_000.0} for _ in range(25)
        ],
        "symbol": "BIG",
        "timeframe": "1d",
    }
    (dir_path / "BIG.json").write_text(json.dumps(payload), encoding="utf-8")
    assert classify("BIG", repo_root=tmp_path) is LiquidityClass.LARGE


# ---------------------------------------------------------------------------
# Test 17 — classify from JSON object with "vol" volume key
# ---------------------------------------------------------------------------


def test_classify_json_object_vol_key(tmp_path: Path) -> None:
    dir_path = tmp_path / "data" / "bars" / "1d"
    dir_path.mkdir(parents=True, exist_ok=True)
    # close * vol = 100 * 1_000_000 = $100M -> STANDARD
    payload = {
        "bars": [
            {"close": 100.0, "vol": 1_000_000.0} for _ in range(25)
        ],
        "symbol": "MID",
        "timeframe": "1d",
    }
    (dir_path / "MID.json").write_text(json.dumps(payload), encoding="utf-8")
    assert classify("MID", repo_root=tmp_path) is LiquidityClass.STANDARD


# ---------------------------------------------------------------------------
# Test 18 — real SPY bar file classifies as LARGE or STANDARD (not UNKNOWN)
# ---------------------------------------------------------------------------


def test_real_spy_bar_file_classified() -> None:
    clear_cache()
    result = classify("SPY")
    assert result in (LiquidityClass.LARGE, LiquidityClass.STANDARD), (
        f"SPY classified as {result} — real bar data not being read"
    )
