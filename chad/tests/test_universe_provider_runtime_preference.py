"""P1: chad.utils.universe_provider must prefer the live-screened
runtime/universe.json over the static config/universe.json, falling back
cleanly when runtime is missing, stale, or malformed.

These tests pin the contract that operational consumers (price cache
refresh, bar provider, portfolio fallback tickers) rely on.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from chad.utils.universe_provider import (
    DEFAULT_FALLBACK,
    RUNTIME_FRESHNESS_SECONDS,
    UniverseConfig,
    get_trade_universe,
    load_active_universe,
)


def _write(p: Path, payload) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def _set_mtime(p: Path, mtime: float) -> None:
    import os
    os.utime(p, (mtime, mtime))


@pytest.fixture
def runtime_path(tmp_path) -> Path:
    return tmp_path / "runtime" / "universe.json"


@pytest.fixture
def config_path(tmp_path) -> Path:
    return tmp_path / "config" / "universe.json"


@pytest.fixture
def cfg() -> UniverseConfig:
    # STATIC mode with explicit static_symbols so we can tell legacy fallback
    # from runtime preference apart in assertions.
    return UniverseConfig(
        mode="STATIC",
        top_n=0,
        static_symbols=["STATIC_A", "STATIC_B", "STATIC_C"],
        fallback=list(DEFAULT_FALLBACK),
    )


def test_universe_provider_prefers_runtime_when_fresh(
    runtime_path, config_path, cfg
):
    """Fresh runtime/universe.json wins over static config."""
    _write(runtime_path, {"symbols": ["RTM1", "RTM2", "RTM3"]})
    _write(config_path, {"symbols": ["CFG1", "CFG2"]})

    result = load_active_universe(
        cfg=cfg, runtime_path=runtime_path, config_path=config_path,
    )

    assert result.symbols == ["RTM1", "RTM2", "RTM3"]
    assert result.source_type == "runtime"
    assert result.source_path == str(runtime_path)
    assert result.symbol_count == 3
    assert result.stale is False
    assert result.reason == "fresh"

    # Compatibility wrapper preserves the list-only contract.
    assert get_trade_universe(
        cfg=cfg, runtime_path=runtime_path,
    ) == ["RTM1", "RTM2", "RTM3"]


def test_universe_provider_falls_back_when_stale(
    runtime_path, config_path, cfg, caplog
):
    """Stale runtime → fall back, reason='stale', warning logged."""
    _write(runtime_path, {"symbols": ["RTM1", "RTM2"]})
    # Force mtime far enough in the past to exceed the freshness window.
    _set_mtime(runtime_path, time.time() - (RUNTIME_FRESHNESS_SECONDS + 600))

    with caplog.at_level(logging.WARNING, logger="chad.utils.universe_provider"):
        result = load_active_universe(
            cfg=cfg, runtime_path=runtime_path, config_path=config_path,
        )

    assert result.source_type != "runtime"
    assert result.reason == "stale"
    assert result.stale is True
    # static_symbols on cfg should win first.
    assert result.symbols == ["STATIC_A", "STATIC_B", "STATIC_C"]
    assert any(
        "using_static_fallback" in rec.message and "reason=stale" in rec.message
        for rec in caplog.records
    ), "expected universe_provider.using_static_fallback warning with reason=stale"


def test_universe_provider_falls_back_when_missing(
    runtime_path, config_path, cfg, caplog
):
    """No runtime file at all → reason='missing', config/universe.json wins."""
    assert not runtime_path.exists()
    # Empty UniverseConfig (no static_symbols) so the bare config-file shape
    # is exercised and we can prove it is read.
    bare_cfg = UniverseConfig(
        mode="STATIC", top_n=0,
        static_symbols=list(DEFAULT_FALLBACK),
        fallback=list(DEFAULT_FALLBACK),
    )
    _write(config_path, {"symbols": ["CFG1", "CFG2", "CFG3", "CFG4"]})

    with caplog.at_level(logging.WARNING, logger="chad.utils.universe_provider"):
        result = load_active_universe(
            cfg=bare_cfg, runtime_path=runtime_path, config_path=config_path,
        )

    assert result.reason == "missing"
    assert result.source_type == "config"
    assert result.symbols == ["CFG1", "CFG2", "CFG3", "CFG4"]
    assert result.symbol_count == 4
    assert any(
        "reason=missing" in rec.message for rec in caplog.records
    ), "expected reason=missing in fallback warning"


def test_universe_provider_falls_back_when_malformed(
    runtime_path, config_path, cfg, caplog
):
    """Invalid JSON in runtime → reason='malformed', warning logged."""
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text("{not_valid_json", encoding="utf-8")
    _write(config_path, {"symbols": ["CFG1"]})

    with caplog.at_level(logging.WARNING, logger="chad.utils.universe_provider"):
        result = load_active_universe(
            cfg=cfg, runtime_path=runtime_path, config_path=config_path,
        )

    assert result.reason == "malformed"
    assert result.source_type != "runtime"
    assert any(
        "reason=malformed" in rec.message for rec in caplog.records
    )


def test_operational_consumer_uses_runtime_subset_if_applicable(
    runtime_path, config_path
):
    """price_cache_refresh._load_universe must return the runtime subset.

    Pinning the wiring: the central provider's preference for runtime must
    flow through to the operational consumer that drives price snapshots.
    """
    _write(runtime_path, {"symbols": ["AAPL", "MSFT", "NVDA"]})
    _write(config_path, {"symbols": ["AAPL", "MSFT", "NVDA", "BAC", "GOOGL"]})

    # Patch the module-level paths the central provider points at.
    import chad.utils.universe_provider as up
    import chad.market_data.price_cache_refresh as pcr

    orig_runtime = up.DEFAULT_RUNTIME_PATH
    orig_config = up.DEFAULT_CONFIG_PATH
    up.DEFAULT_RUNTIME_PATH = runtime_path
    up.DEFAULT_CONFIG_PATH = config_path
    try:
        syms = pcr._load_universe()
    finally:
        up.DEFAULT_RUNTIME_PATH = orig_runtime
        up.DEFAULT_CONFIG_PATH = orig_config

    assert syms == ["AAPL", "MSFT", "NVDA"], (
        "operational consumer must receive the runtime-screened subset, "
        f"got {syms}"
    )


def test_runtime_universe_supports_alternate_shapes(
    runtime_path, config_path, cfg
):
    """{"universe":[...]} and {"tickers":[...]} are also valid shapes."""
    _write(runtime_path, {"universe": ["U1", "U2"]})
    _write(config_path, {"symbols": ["CFG"]})
    r1 = load_active_universe(
        cfg=cfg, runtime_path=runtime_path, config_path=config_path,
    )
    assert r1.symbols == ["U1", "U2"]
    assert r1.source_type == "runtime"

    _write(runtime_path, {"tickers": ["T1", "T2", "T3"]})
    r2 = load_active_universe(
        cfg=cfg, runtime_path=runtime_path, config_path=config_path,
    )
    assert r2.symbols == ["T1", "T2", "T3"]
    assert r2.source_type == "runtime"
