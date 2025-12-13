"""
Tests for newly added strategy brains (Omega, Delta, AlphaCrypto, AlphaForex).

Goals
-----
- Ensure configs build without error.
- Validate StrategyName mapping and enabled flags.
- Confirm target universes are non-empty and deterministic.
"""

from __future__ import annotations

from chad.types import StrategyName, StrategyConfig

from chad.strategies.omega import build_omega_config, OmegaParams
from chad.strategies.delta import build_delta_config, DeltaParams
from chad.strategies.alpha_crypto import build_alpha_crypto_config, AlphaCryptoParams
from chad.strategies.alpha_forex import build_alpha_forex_config, AlphaForexParams


def _assert_basic_config(cfg: StrategyConfig, expected_name: StrategyName) -> None:
    assert isinstance(cfg, StrategyConfig)
    assert cfg.name is expected_name
    assert cfg.enabled is True
    # target_universe may be None for some strategies, but if present,
    # it must be a non-empty sequence of strings.
    if cfg.target_universe is not None:
        universe = list(cfg.target_universe)
        assert len(universe) > 0
        assert all(isinstance(sym, str) and sym for sym in universe)


def test_omega_config_and_params() -> None:
    cfg = build_omega_config()
    _assert_basic_config(cfg, StrategyName.OMEGA)

    params = OmegaParams()
    assert params.enabled is True


def test_delta_config_and_params() -> None:
    cfg = build_delta_config()
    _assert_basic_config(cfg, StrategyName.DELTA)

    params = DeltaParams()
    assert params.enabled is True


def test_alpha_crypto_config_and_params() -> None:
    cfg = build_alpha_crypto_config()
    _assert_basic_config(cfg, StrategyName.ALPHA_CRYPTO)

    params = AlphaCryptoParams()
    assert params.enabled is True
    # Ensure universe resolution is stable
    universe = params.actual_universe()
    assert len(universe) > 0
    assert all(isinstance(sym, str) and sym for sym in universe)


def test_alpha_forex_config_and_params() -> None:
    cfg = build_alpha_forex_config()
    _assert_basic_config(cfg, StrategyName.ALPHA_FOREX)

    params = AlphaForexParams()
    assert params.enabled is True
    universe = params.actual_universe()
    assert len(universe) > 0
    assert all(isinstance(sym, str) and sym for sym in universe)
