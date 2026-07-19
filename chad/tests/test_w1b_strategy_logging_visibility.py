"""W1B-2 — strategy-module INFO logging reaches journald.

The live-loop process's only logging bootstrap is run_loop(), whose setup is
extracted into chad.core.live_loop._configure_live_loop_logging(). It binds the
stderr handler + INFO level to the ``chad`` root logger, so every ``chad.*``
logger -- including ``chad.strategies.*`` (e.g. ``chad.strategies.alpha_crypto``
emitting CRYPTO_EXPLORATION_HANDLER_PASS) -- inherits INFO and reaches the
handler by propagation. Before the fix the handler bound to ``chad.live_loop``
(a *sibling* of ``chad.strategies``) while the root stayed at the default
WARNING, so ``chad.strategies.*`` .info() was dropped at source.

We assert the *effective configuration*, not captured records: caplog attaches
its own root handler and forces levels, which masks the exact source-level drop
this fix addresses.
"""

from __future__ import annotations

import logging

import pytest

from chad.core.live_loop import _configure_live_loop_logging

_MANAGED = ("chad", "chad.live_loop", "chad.strategies", "chad.strategies.alpha_crypto")


@pytest.fixture
def restore_chad_logging():
    """Snapshot/restore the managed ``chad.*`` loggers so the bootstrap side
    effects (a handler + INFO level on ``chad``) never leak into other tests,
    and set the pre-bootstrap default state so assertions are deterministic
    under a full-suite run."""
    saved = {
        n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level)
        for n in _MANAGED
    }
    # Pre-bootstrap default: root ``chad`` at WARNING with no handler; the rest
    # NOTSET so effective level is inherited from ``chad``.
    logging.getLogger("chad").handlers[:] = []
    logging.getLogger("chad").setLevel(logging.WARNING)
    logging.getLogger("chad.live_loop").handlers[:] = []
    logging.getLogger("chad.live_loop").setLevel(logging.NOTSET)
    logging.getLogger("chad.strategies").setLevel(logging.NOTSET)
    logging.getLogger("chad.strategies.alpha_crypto").setLevel(logging.NOTSET)
    try:
        yield
    finally:
        for n in _MANAGED:
            handlers, level = saved[n]
            logging.getLogger(n).handlers[:] = handlers
            logging.getLogger(n).setLevel(level)


def test_strategy_info_enabled_after_bootstrap(restore_chad_logging) -> None:
    strat = logging.getLogger("chad.strategies.alpha_crypto")

    # Pre-condition (the bug): with root at WARNING and no ``chad`` handler,
    # strategy INFO is disabled at source.
    assert strat.getEffectiveLevel() == logging.WARNING
    assert strat.isEnabledFor(logging.INFO) is False

    _configure_live_loop_logging()

    # Post-condition: strategy INFO is enabled and a handler is reachable by
    # propagation up to ``chad``.
    assert strat.getEffectiveLevel() == logging.INFO
    assert strat.isEnabledFor(logging.INFO) is True
    assert logging.getLogger("chad").handlers, "handler must sit on the 'chad' root"


def test_bootstrap_is_idempotent_and_single_handler(restore_chad_logging) -> None:
    _configure_live_loop_logging()
    _configure_live_loop_logging()
    # A second bootstrap must not stack a second handler (would double-emit).
    assert len(logging.getLogger("chad").handlers) == 1
    # And it must NOT leave a competing handler on the ``chad.live_loop`` sibling.
    assert logging.getLogger("chad.live_loop").handlers == []


def test_propagation_chain_intact(restore_chad_logging) -> None:
    _configure_live_loop_logging()
    # The fix depends on propagation from chad.strategies.alpha_crypto up to the
    # single ``chad`` handler; every logger in that chain must propagate.
    for name in ("chad.strategies.alpha_crypto", "chad.strategies", "chad"):
        assert logging.getLogger(name).propagate is True, f"{name} must propagate"


def test_returns_the_chad_logger(restore_chad_logging) -> None:
    returned = _configure_live_loop_logging()
    assert returned is logging.getLogger("chad")
    assert returned.level == logging.INFO
