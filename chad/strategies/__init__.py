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
- ALPHA_FUTURES – Futures momentum
- GAMMA_FUTURES – Futures mean-reversion
- OMEGA_MACRO      – Macro regime futures
- GAMMA_REVERSION  – ETF mean reversion
- ALPHA_OPTIONS    – Vertical spread options
- OMEGA_VOL       – Volatility regime alpha
- DELTA_PAIRS     – Market-neutral pairs trading
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
from .alpha_crypto import build_alpha_crypto_config, alpha_crypto_handler, AlphaCryptoParams
from .alpha_intraday import build_alpha_intraday_config, alpha_intraday_handler
# DEFERRED: alpha_forex import disabled with its registration (see _build_registry)
# from .alpha_forex import build_alpha_forex_config, alpha_forex_handler, AlphaForexParams
from chad.strategies.alpha_futures import build_alpha_futures_signals
from .alpha_futures import alpha_futures_handler
from .alpha_futures_config import build_alpha_futures_config
from .gamma_futures import gamma_futures_handler
from .gamma_futures_config import build_gamma_futures_config
from .omega_macro import omega_macro_handler
from .omega_macro_config import build_omega_macro_config
from .gamma_reversion import gamma_reversion_handler
from .gamma_reversion_config import build_gamma_reversion_config
from .alpha_options import alpha_options_handler
from .alpha_options_config import build_alpha_options_config
from .omega_vol import omega_vol_handler
from .omega_vol_config import build_omega_vol_config
from .delta_pairs import delta_pairs_handler
from .delta_pairs_config import build_delta_pairs_config
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
            handler=(lambda ctx: alpha_crypto_handler(ctx, AlphaCryptoParams())),
        ),
        StrategyName.ALPHA_INTRADAY: StrategyRegistration(
            name=StrategyName.ALPHA_INTRADAY,
            build_config=build_alpha_intraday_config,
            handler=alpha_intraday_handler,
        ),
        # DEFERRED: alpha_forex — FX universe (EUR-USD, GBP-USD, USD-CAD,
        # USD-JPY) not mapped to active bar/price context. M6E (Micro EUR/FX
        # future) is in universe but the symbol translation layer is not
        # implemented. Produces 0 signals every cycle, polluting audit output.
        # Re-enable when FX universe is formally defined.
        # StrategyName.ALPHA_FOREX: StrategyRegistration(
        #     name=StrategyName.ALPHA_FOREX,
        #     build_config=build_alpha_forex_config,
        #     handler=(lambda ctx: alpha_forex_handler(ctx, AlphaForexParams())),
        # ),
        StrategyName.ALPHA_FUTURES: StrategyRegistration(
            name=StrategyName.ALPHA_FUTURES,
            build_config=build_alpha_futures_config,
            handler=alpha_futures_handler,
        ),
        StrategyName.GAMMA_FUTURES: StrategyRegistration(
            name=StrategyName.GAMMA_FUTURES,
            build_config=build_gamma_futures_config,
            handler=gamma_futures_handler,
        ),
        StrategyName.OMEGA_MACRO: StrategyRegistration(
            name=StrategyName.OMEGA_MACRO,
            build_config=build_omega_macro_config,
            handler=omega_macro_handler,
        ),
        StrategyName.GAMMA_REVERSION: StrategyRegistration(
            name=StrategyName.GAMMA_REVERSION,
            build_config=build_gamma_reversion_config,
            handler=gamma_reversion_handler,
        ),
        StrategyName.ALPHA_OPTIONS: StrategyRegistration(
            name=StrategyName.ALPHA_OPTIONS,
            build_config=build_alpha_options_config,
            handler=alpha_options_handler,
        ),
        StrategyName.OMEGA_VOL: StrategyRegistration(
            name=StrategyName.OMEGA_VOL,
            build_config=build_omega_vol_config,
            handler=omega_vol_handler,
        ),
        StrategyName.DELTA_PAIRS: StrategyRegistration(
            name=StrategyName.DELTA_PAIRS,
            build_config=build_delta_pairs_config,
            handler=delta_pairs_handler,
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
