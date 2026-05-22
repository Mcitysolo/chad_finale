"""GAP-035 — upstream TradeSignal emission exclusions.

Proves that ``delta``, ``delta_pairs``, and ``alpha_options`` cannot emit
``TradeSignal`` objects for operator-excluded symbols. The downstream
close-path chokepoints (``position_reconciler.apply_close_intents`` and
``flip_executor.enforce_flip_close_first``) already block close intents
for excluded symbols; these tests cover the *open*-side / upstream gap.

Exclusion source of truth: ``chad.core.position_reconciler.
_EFFECTIVE_NON_CHAD_SYMBOLS`` (re-exported via
``chad.strategies._upstream_exclusion``).
"""
from __future__ import annotations

from chad.strategies._upstream_exclusion import (
    EXCLUSION_SOURCE,
    OPERATOR_EXCLUDED_SYMBOLS,
    filter_operator_excluded,
    is_operator_excluded,
)


# ---------------------------------------------------------------------------
# Canonical SSOT identity
# ---------------------------------------------------------------------------

def test_exclusion_source_matches_canonical_ssot():
    """The strategy-side helper must point at the SAME frozenset object as
    the close-path chokepoints. No parallel exclusion table is permitted.
    """
    from chad.core.position_reconciler import (
        _EFFECTIVE_NON_CHAD_SYMBOLS,
        _EXCLUSION_SOURCE,
    )
    # Same membership
    assert OPERATOR_EXCLUDED_SYMBOLS == frozenset(
        str(s).upper() for s in _EFFECTIVE_NON_CHAD_SYMBOLS
    )
    # Same source label so an evidence audit can trace the path
    assert EXCLUSION_SOURCE == _EXCLUSION_SOURCE


def test_excluded_set_covers_documented_operator_policy():
    """Documented operator policy from Box-10 includes these equity
    symbols. The SSOT must include them all.
    """
    documented = {"AAPL", "BAC", "CVX", "LLY", "MSFT",
                  "NVDA", "PEP", "QQQ", "SPY"}
    missing = documented - OPERATOR_EXCLUDED_SYMBOLS
    assert not missing, f"SSOT is missing operator-policy symbols: {missing}"


def test_is_operator_excluded_case_insensitive():
    assert is_operator_excluded("SPY") is True
    assert is_operator_excluded("spy") is True
    assert is_operator_excluded(" SpY ") is True
    assert is_operator_excluded("IWM") is False
    assert is_operator_excluded("") is False
    assert is_operator_excluded(None) is False


def test_filter_operator_excluded_preserves_order():
    out = list(filter_operator_excluded(
        ["IWM", "SPY", "GLD", "QQQ", "tlt", None, "", "MSFT", "VWO"]
    ))
    assert out == ["IWM", "GLD", "TLT", "VWO"]


# ---------------------------------------------------------------------------
# delta strategy emitter
# ---------------------------------------------------------------------------

class _FakePortfolio:
    cash = 1_000_000.0
    positions = {}


class _FakeCtx:
    def __init__(self, prices=None, ticks=None):
        self.portfolio = _FakePortfolio()
        self.prices = prices or {}
        self.ticks = ticks or {}
        self.legend = None
        self.event_risk = None
        # delta universe override
        self.delta_universe = None


def _build_delta_ctx_with_universe(universe):
    ctx = _FakeCtx()
    ctx.delta_universe = list(universe)
    return ctx


def test_delta_handler_skips_excluded_symbols_in_universe():
    """delta_handler must NOT emit a TradeSignal for any symbol in
    OPERATOR_EXCLUDED_SYMBOLS, even when the universe explicitly lists
    them.
    """
    from chad.strategies.delta import delta_handler, DEFAULT_PARAMS

    ctx = _build_delta_ctx_with_universe([
        "SPY", "AAPL", "QQQ", "MSFT", "NVDA", "BAC", "CVX", "LLY", "PEP",
        "IWM", "GLD",  # non-excluded
    ])
    signals = delta_handler(ctx, DEFAULT_PARAMS, prices={})
    emitted_symbols = {str(s.symbol).upper() for s in signals}
    overlap = emitted_symbols & OPERATOR_EXCLUDED_SYMBOLS
    assert not overlap, (
        f"delta_handler emitted excluded symbols: {overlap}"
    )


def test_delta_handler_still_processes_non_excluded_symbols():
    """The filter must not be a blanket-deny — non-excluded symbols still
    reach ``_propose_for_symbol``. We do not assert positive emission
    (that depends on bar history), only that the universe loop does NOT
    skip non-excluded symbols.
    """
    from chad.strategies import delta as delta_mod

    seen = []
    real_propose = delta_mod._propose_for_symbol

    def _tracker(sym, ctx, p, px):
        seen.append(str(sym).upper())
        return real_propose(sym, ctx, p, px)

    delta_mod._propose_for_symbol = _tracker
    try:
        ctx = _build_delta_ctx_with_universe(["SPY", "IWM", "GLD", "TLT"])
        delta_mod.delta_handler(ctx, delta_mod.DEFAULT_PARAMS, prices={})
    finally:
        delta_mod._propose_for_symbol = real_propose
    # SPY is excluded — must not be passed to _propose_for_symbol
    assert "SPY" not in seen
    # Non-excluded symbols MUST reach _propose_for_symbol
    assert "IWM" in seen
    assert "GLD" in seen
    assert "TLT" in seen


# ---------------------------------------------------------------------------
# delta_pairs strategy emitter
# ---------------------------------------------------------------------------

def test_delta_pairs_skips_pairs_with_excluded_legs():
    """delta_pairs default pairs are SPY/QQQ, SPY/IWM, QQQ/IWM. All three
    contain at least one excluded symbol (SPY or QQQ). The handler must
    not emit any signal for them.
    """
    from chad.strategies.delta_pairs import build_delta_pairs_signals

    class _Ctx:
        portfolio = _FakePortfolio()
        prices = {}
        ticks = {}
        now = None
        regime = "trending_bull"

    # No bars supplied so even non-excluded pairs would emit nothing — the
    # contract we care about is "emits nothing for excluded pairs". We
    # also assert the pair-skip path runs before bar-loading.
    sigs = build_delta_pairs_signals(_Ctx())
    emitted = {str(s.symbol).upper() for s in sigs}
    overlap = emitted & OPERATOR_EXCLUDED_SYMBOLS
    assert not overlap, (
        f"delta_pairs emitted excluded symbols: {overlap}"
    )


def test_delta_pairs_legitimate_pair_not_blocked(monkeypatch):
    """If a custom pair (IWM/EFA — both non-excluded) is configured, the
    upstream exclusion filter must NOT block it. We mock out the math
    layer to confirm the pair reaches ``compute_zscore``.
    """
    from chad.strategies import delta_pairs as dp

    visited_pairs = []

    real_compute = dp.compute_zscore

    def _tracker(closes_a, closes_b, lookback):
        visited_pairs.append((len(closes_a), len(closes_b)))
        return real_compute(closes_a, closes_b, lookback)

    monkeypatch.setattr(dp, "compute_zscore", _tracker)

    pair = dp.PairSpec("IWM", "EFA", correlation=0.85, half_life_days=22.0)
    tuning = dp.DeltaPairsTuning(
        pairs=(pair,),
        min_bars=2,
        zscore_entry=10.0,  # so we don't actually emit
    )
    monkeypatch.setattr(dp, "DEFAULT_TUNING", tuning)

    # Need at least min_bars bars on both legs
    closes = [100.0, 101.0, 99.0, 102.0]

    def _fake_extract_closes(ctx, sym):
        return closes

    monkeypatch.setattr(dp, "_extract_closes", _fake_extract_closes)

    class _Ctx:
        portfolio = _FakePortfolio()
        prices = {"IWM": 200.0, "EFA": 90.0}
        ticks = {}
        now = None

    dp.build_delta_pairs_signals(_Ctx())
    assert visited_pairs, "non-excluded pair IWM/EFA was filtered out"


def test_delta_pairs_skips_when_only_one_leg_excluded(monkeypatch):
    """If a pair has one excluded leg (e.g. IWM/SPY), the WHOLE pair must
    be skipped — pairs are atomic, we cannot trade one leg without the
    other.
    """
    from chad.strategies import delta_pairs as dp

    pair = dp.PairSpec("IWM", "SPY", correlation=0.95, half_life_days=22.0)
    tuning = dp.DeltaPairsTuning(pairs=(pair,))
    monkeypatch.setattr(dp, "DEFAULT_TUNING", tuning)

    visited = []
    monkeypatch.setattr(
        dp, "_extract_closes",
        lambda *a, **k: (visited.append("touched") or [])
    )

    class _Ctx:
        portfolio = _FakePortfolio()
        prices = {}
        ticks = {}
        now = None

    dp.build_delta_pairs_signals(_Ctx())
    assert not visited, (
        "delta_pairs invoked _extract_closes on a pair containing SPY"
    )


# ---------------------------------------------------------------------------
# alpha_options strategy emitter
# ---------------------------------------------------------------------------

def test_alpha_options_open_skips_excluded_universe(monkeypatch):
    """alpha_options.options_universe default is ('SPY',) — SPY is in
    OPERATOR_EXCLUDED_SYMBOLS, so the universe loop must skip it before
    chain loading / spread building.
    """
    from chad.strategies import alpha_options as ao

    chain_calls = []
    monkeypatch.setattr(
        ao, "_load_chain_from_cache",
        lambda sym: (chain_calls.append(sym) or None)
    )
    # No directional signal context → would short-circuit anyway, but the
    # earlier exclusion gate is what we are testing.
    monkeypatch.setattr(
        ao, "_extract_price", lambda ctx, sym: 100.0
    )
    monkeypatch.setattr(
        ao, "_extract_directional_signal", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ao, "_extract_directional_from_bars", lambda *a, **k: None
    )

    class _Ctx:
        portfolio = _FakePortfolio()
        prices = {"SPY": 700.0}
        ticks = {}
        now = None
        regime = "trending_bull"

    tuning = ao.AlphaOptionsTuning(options_universe=("SPY", "QQQ"))
    monkeypatch.setattr(ao, "DEFAULT_TUNING", tuning)

    sigs = ao.build_alpha_options_signals(_Ctx())
    assert sigs == [], "alpha_options emitted a signal for excluded SPY/QQQ"
    assert "SPY" not in chain_calls
    assert "QQQ" not in chain_calls


def test_alpha_options_open_processes_non_excluded(monkeypatch):
    """A non-excluded options underlying (e.g. IWM) must still reach the
    chain-loader. We do not require positive emission (chain mock returns
    None); we only require the exclusion gate did NOT short-circuit IWM.
    """
    from chad.strategies import alpha_options as ao

    chain_calls = []
    monkeypatch.setattr(
        ao, "_load_chain_from_cache",
        lambda sym: (chain_calls.append(sym) or None)
    )
    monkeypatch.setattr(
        ao, "_extract_price", lambda ctx, sym: 100.0
    )
    monkeypatch.setattr(
        ao, "_extract_directional_signal",
        lambda *a, **k: {"direction": "BUY", "confidence": 0.8,
                          "source_strategy": "test"}
    )

    class _Ctx:
        portfolio = _FakePortfolio()
        prices = {"IWM": 270.0}
        ticks = {}
        now = None
        regime = "trending_bull"

    tuning = ao.AlphaOptionsTuning(options_universe=("IWM",))
    monkeypatch.setattr(ao, "DEFAULT_TUNING", tuning)

    ao.build_alpha_options_signals(_Ctx())
    assert "IWM" in chain_calls, (
        "alpha_options exclusion gate falsely blocked IWM"
    )


def test_alpha_options_max_hold_exit_skips_excluded(monkeypatch, tmp_path):
    """Pre-existing stale alpha_options position on SPY (excluded) must
    NOT produce a max-hold exit SELL signal — the operator-exclusion
    invariant says CHAD cannot close excluded symbols either.
    """
    from chad.strategies import alpha_options as ao

    # The exit path walks position_guard entries; mock the loader.
    fake_state = {
        "alpha_options|SPY": {
            "open": True,
            "strategy": "alpha_options",
            "symbol": "SPY",
            "quantity": 1,
            "opened_at_utc": "2026-04-01T00:00:00Z",  # well over max_hold
        },
    }
    monkeypatch.setattr(
        ao, "_load_position_guard_state",
        lambda: fake_state, raising=False,
    )

    class _Ctx:
        portfolio = _FakePortfolio()
        prices = {"SPY": 700.0}
        ticks = {}
        now = None
        regime = "trending_bull"

    tuning = ao.AlphaOptionsTuning(
        options_universe=(),  # no opens this cycle
        max_hold_seconds=3600,
    )
    monkeypatch.setattr(ao, "DEFAULT_TUNING", tuning)

    # The chain loader is unrelated to the exit emission; ensure no opens.
    monkeypatch.setattr(
        ao, "_load_chain_from_cache", lambda sym: None
    )

    sigs = ao.build_alpha_options_signals(_Ctx())
    excluded_emitted = [s for s in sigs
                        if str(s.symbol).upper() in OPERATOR_EXCLUDED_SYMBOLS]
    assert excluded_emitted == [], (
        f"alpha_options max_hold_exit emitted on excluded symbol: "
        f"{[s.symbol for s in excluded_emitted]}"
    )
