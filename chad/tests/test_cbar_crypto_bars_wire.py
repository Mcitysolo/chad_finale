"""CBAR S1 — the "real wire": exploration bypass at alpha_crypto's INTERNAL regime short-circuit.

Context (from the C1 audit): daily crypto bars already exist and are refreshed nightly by
``chad/market_data/nightly_bars_refresh.py::_run_kraken`` (Kraken public OHLC), so alpha_crypto
is NOT starved for data. Its silence is the INTERNAL regime short-circuit
(``alpha_crypto_handler`` returns [] in ranging/adverse BEFORE any signal computes). S1 extends
the CEW1 exploration flag through that short-circuit: under the SAME double fail-closed
conditions as CEW1-W2 (flag on AND CHAD_EXECUTION_MODE=paper AND CHAD_KRAKEN_MODE!=live) the
handler runs its full momentum/breakout path in ALL regimes; flag off — or either refusal — is
byte-identical stock gating. It reuses the ONE authority
(regime_activation.crypto_exploration_state), never a forked flag-evaluation path.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from chad.strategies import alpha_crypto as ac
from chad.strategies.alpha_crypto import (
    AlphaCryptoParams,
    MARKER_EXPLORATION_HANDLER_PASS,
    alpha_crypto_handler,
)
from chad.portfolio import regime_activation as ra

_HANDLER_LOGGER = "chad.strategies.alpha_crypto"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _uptrend_bars(n: int = 30, step: float = 0.012) -> List[dict]:
    """A clean geometric uptrend that clears every momentum gate (price>sma20, up-confirm,
    3d>1.5%). Same series both flag states, so a signal appearing only under the flag proves
    the regime early-return is no longer the exit point."""
    bars: List[dict] = []
    px = 100.0
    for i in range(n):
        o = px
        px *= (1.0 + step)
        c = px
        bars.append({"ts_utc": f"2026-06-{i + 1:02d}", "open": o, "high": c * 1.001,
                     "low": o * 0.999, "close": c, "volume": 1000.0})
    return bars


def _flat_bars(n: int = 30) -> List[dict]:
    """Enough history to clear the no-data gate (len>=22) but flat -> the momentum gates emit
    0. Used to show the handler ran PAST the regime/no-data early-returns and exited on a
    downstream gate instead (fixture-may-emit-0 case)."""
    return [{"ts_utc": f"2026-06-{i + 1:02d}", "open": 100.0, "high": 100.0, "low": 100.0,
             "close": 100.0, "volume": 1000.0} for i in range(n)]


class _Ctx:
    def __init__(self, regime: str, bars: Dict[str, List[dict]]):
        self.regime = regime
        self.prices: Dict[str, float] = {}
        self.bars = bars


def _params() -> AlphaCryptoParams:
    # Pin the universe to BTC-USD and disable the CAD lane so the handler is fully hermetic
    # (never falls back to reading the real data/bars/1d/*.json off disk).
    return AlphaCryptoParams(universe=["BTC-USD"], enable_cad_pairs=False)


def _clear_env(monkeypatch) -> None:
    for k in ("CHAD_CRYPTO_EXPLORATION", "CHAD_EXECUTION_MODE", "CHAD_KRAKEN_MODE"):
        monkeypatch.delenv(k, raising=False)


def _marker_logged(caplog) -> bool:
    return any(MARKER_EXPLORATION_HANDLER_PASS in r.getMessage() for r in caplog.records)


def _stock_ranging_logged(caplog) -> bool:
    return any("reason: regime_ranging" in r.getMessage() for r in caplog.records)


# ===========================================================================
# S1 — exploration bypass at the INTERNAL regime short-circuit
# ===========================================================================
def test_s1_flag_off_ranging_is_byte_identical_stock(monkeypatch, caplog):
    _clear_env(monkeypatch)
    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        sigs = alpha_crypto_handler(_Ctx("ranging", {"BTC-USD": _uptrend_bars()}), _params())
    assert sigs == []                          # stock: silent in ranging even with a clean trend
    assert _stock_ranging_logged(caplog)       # identical stock log line
    assert not _marker_logged(caplog)          # bypass did NOT engage


def test_s1_flag_on_paper_ranging_bypasses_and_emits(monkeypatch, caplog):
    monkeypatch.setenv("CHAD_CRYPTO_EXPLORATION", "1")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        sigs = alpha_crypto_handler(_Ctx("ranging", {"BTC-USD": _uptrend_bars()}), _params())
    # Same uptrend bars as the flag-off test above emit a signal here: the flag, not the data,
    # is what flips the outcome. And the loud marker proves the internal bypass engaged.
    assert len(sigs) == 1 and sigs[0].symbol == "BTC-USD"
    assert _marker_logged(caplog)
    assert not _stock_ranging_logged(caplog)


def test_s1_flag_on_ranging_runs_full_path_even_when_momentum_emits_zero(monkeypatch, caplog):
    # The required "fixture may still emit 0" case: flat bars produce 0 signals, but the
    # marker proves the regime/no-data early-returns were NOT the exit point — the handler ran
    # through to a downstream momentum gate.
    monkeypatch.setenv("CHAD_CRYPTO_EXPLORATION", "1")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        sigs = alpha_crypto_handler(_Ctx("ranging", {"BTC-USD": _flat_bars()}), _params())
    assert sigs == []                          # emits 0 on the momentum gate...
    assert _marker_logged(caplog)              # ...but the regime early-return was NOT the exit
    assert not _stock_ranging_logged(caplog)


def test_s1_live_exec_mode_refused_no_bypass(monkeypatch, caplog):
    monkeypatch.setenv("CHAD_CRYPTO_EXPLORATION", "1")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "live")     # axis-1 fail-closed
    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        sigs = alpha_crypto_handler(_Ctx("ranging", {"BTC-USD": _uptrend_bars()}), _params())
    assert sigs == []
    assert _stock_ranging_logged(caplog) and not _marker_logged(caplog)


def test_s1_kraken_lane_live_refused_no_bypass(monkeypatch, caplog):
    monkeypatch.setenv("CHAD_CRYPTO_EXPLORATION", "1")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    monkeypatch.setenv("CHAD_KRAKEN_MODE", "live")        # axis-2 fail-closed
    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        sigs = alpha_crypto_handler(_Ctx("ranging", {"BTC-USD": _uptrend_bars()}), _params())
    assert sigs == []
    assert _stock_ranging_logged(caplog) and not _marker_logged(caplog)


def test_s1_non_ranging_regime_unaffected_by_flag(monkeypatch, caplog):
    # The bypass only touches the ranging/adverse short-circuit. A trending regime behaves
    # identically flag-on vs flag-off, and never logs the bypass marker.
    bars = {"BTC-USD": _uptrend_bars()}
    _clear_env(monkeypatch)
    off = alpha_crypto_handler(_Ctx("trending_bull", bars), _params())
    monkeypatch.setenv("CHAD_CRYPTO_EXPLORATION", "1")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        on = alpha_crypto_handler(_Ctx("trending_bull", bars), _params())
    assert [s.symbol for s in off] == [s.symbol for s in on]   # flag is inert outside ranging
    assert not _marker_logged(caplog)


def test_s1_delegates_to_single_cew1_authority(monkeypatch):
    # Prove the handler's bypass decision flows THROUGH regime_activation.crypto_exploration_state
    # (one authority) rather than a private forked copy: monkeypatch that function and watch the
    # helper obey it with the env cleared entirely.
    _clear_env(monkeypatch)
    monkeypatch.setattr(ra, "crypto_exploration_state", lambda env=None: (True, "active"))
    assert ac._exploration_bypass_active() is True
    monkeypatch.setattr(ra, "crypto_exploration_state", lambda env=None: (False, "off"))
    assert ac._exploration_bypass_active() is False


def test_s1_bypass_helper_fails_closed_on_authority_error(monkeypatch):
    # A broken authority import/eval can only ever restore stock behavior, never widen it.
    def _boom(env=None):
        raise RuntimeError("authority unavailable")
    monkeypatch.setattr(ra, "crypto_exploration_state", _boom)
    assert ac._exploration_bypass_active() is False


def test_s1_module_does_not_fork_flag_evaluation():
    # Structural: the strategy must not read the flag/mode env vars itself (that lives solely in
    # the authority). alpha_crypto never imports os, and touches os.environ nowhere.
    import inspect
    src = inspect.getsource(ac)
    assert "os.environ" not in src and "os.getenv" not in src, (
        "alpha_crypto must not evaluate the exploration flag itself — one authority only"
    )
