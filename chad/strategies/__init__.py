#!/usr/bin/env python3
"""
chad/strategies/__init__.py

Canonical strategy registry for CHAD.

This module provides a single, authoritative place to register all strategy
brains with the StrategyEngine. It is intentionally:

- Strongly typed and dependency-injected.
- Deterministic and side-effect free at import time.
- Easy to extend when new brains are added.
- Safe: registration order is explicit and controlled.

Strategies Covered
------------------
- ALPHA         – Intraday stocks/ETFs
- BETA          – Long-term legend/ETF
- GAMMA         – Swing / momentum
- OMEGA         – Hedge / macro
- DELTA         – Execution intelligence / meta-signals
- ALPHA_CRYPTO  – Intraday crypto
- ALPHA_FOREX   – Intraday FX
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Protocol

from chad.engine import StrategyEngine
from chad.types import StrategyConfig, StrategyName

from .alpha import build_alpha_config, alpha_handler
from .beta import build_beta_config, beta_handler
from .gamma import build_gamma_config, gamma_handler
from .omega import build_omega_config, omega_handler
from .delta import build_delta_config, delta_handler
from .alpha_crypto import build_alpha_crypto_config, alpha_crypto_handler
from .alpha_forex import build_alpha_forex_config, alpha_forex_handler


# ---------------------------------------------------------------------------
# Protocols & Dataclasses
# ---------------------------------------------------------------------------


class StrategyHandler(Protocol):
    """
    Protocol describing the callable signature for strategy handlers.

    Handlers are intentionally typed loosely on `ctx` and `params` so that
    individual strategies can evolve their parameter objects without forcing
    changes here. The return type is `List` to remain compatible with the
    existing engine contract (usually a list of TradeSignal-like objects).
    """

    def __call__(self, ctx: object, params: object) -> List:  # pragma: no cover - structural
        ...


@dataclass(frozen=True)
class StrategyRegistration:
    """
    Immutable registration record for a strategy.

    Attributes
    ----------
    name:
        StrategyName enum value for this brain.
    build_config:
        Zero-argument callable producing the StrategyConfig for this brain.
    handler:
        Callable implementing the strategy logic.
    """

    name: StrategyName
    build_config: Callable[[], StrategyConfig]
    handler: StrategyHandler


# ---------------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------------


def _build_registry() -> Dict[StrategyName, StrategyRegistration]:
    """
    Construct the canonical registry mapping for all core strategies.

    This function is kept private so that the public API exposes only
    high-level helpers (`register_core_strategies` and
    `iter_strategy_registrations`), avoiding accidental mutation.
    """
    registrations: Dict[StrategyName, StrategyRegistration] = {
        StrategyName.ALPHA: StrategyRegistration(
            name=StrategyName.ALPHA,
            build_config=build_alpha_config,
            handler=alpha_handler,
        ),
        StrategyName.BETA: StrategyRegistration(
            name=StrategyName.BETA,
            build_config=build_beta_config,
            handler=beta_handler,
        ),
        StrategyName.GAMMA: StrategyRegistration(
            name=StrategyName.GAMMA,
            build_config=build_gamma_config,
            handler=gamma_handler,
        ),
        StrategyName.OMEGA: StrategyRegistration(
            name=StrategyName.OMEGA,
            build_config=build_omega_config,
            handler=omega_handler,
        ),
        StrategyName.DELTA: StrategyRegistration(
            name=StrategyName.DELTA,
            build_config=build_delta_config,
            handler=delta_handler,
        ),
        StrategyName.ALPHA_CRYPTO: StrategyRegistration(
            name=StrategyName.ALPHA_CRYPTO,
            build_config=build_alpha_crypto_config,
            handler=alpha_crypto_handler,
        ),
        StrategyName.ALPHA_FOREX: StrategyRegistration(
            name=StrategyName.ALPHA_FOREX,
            build_config=build_alpha_forex_config,
            handler=alpha_forex_handler,
        ),
    }
    return registrations


_REGISTRY: Dict[StrategyName, StrategyRegistration] = _build_registry()


def iter_strategy_registrations() -> Iterable[StrategyRegistration]:
    """
    Iterate over all registered strategies in deterministic order.

    Returns
    -------
    Iterable[StrategyRegistration]
        An iterable of StrategyRegistration instances. Iteration order is
        defined by StrategyName enumeration order to keep behaviour stable.
    """
    # Respect enum order for reproducible registration.
    for name in StrategyName:
        reg = _REGISTRY.get(name)
        if reg is not None:
            yield reg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_core_strategies(engine: StrategyEngine) -> None:
    """
    Register all core strategy brains with the provided StrategyEngine.

    The engine is expected to expose a `register(config, handler)` method
    compatible with the StrategyConfig + handler defined here.

    Parameters
    ----------
    engine : StrategyEngine
        Instance of the strategy engine into which all brains will be
        registered.

    Behaviour
    ---------
    - Iterates over all StrategyName values in a deterministic order.
    - For each known strategy, builds its StrategyConfig and registers its
      handler with the engine.
    - Ignores StrategyName values that do not have a corresponding
      StrategyRegistration (future-proofing).

    This function performs no I/O and has no side effects beyond calling
    `engine.register` with deterministic arguments.
    """
    for reg in iter_strategy_registrations():
        config = reg.build_config()
        engine.register(config, reg.handler)
