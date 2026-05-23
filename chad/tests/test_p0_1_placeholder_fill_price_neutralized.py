"""
P0-1 — Delta/IWM $100 placeholder source remediation (2026-05-23)

These tests pin the chokepoint behavior of
``chad.execution.paper_exec_evidence_writer.normalize_paper_fill_evidence``:

When the trust-boundary deviation guard or the
placeholder-without-price-cache guard fires, the resulting PaperExecEvidence
must no longer carry numeric ``fill_price=100.0`` / ``expected_price=100.0``
at the top level. The forensic original values are preserved under
``ev.extra["placeholder_*"]`` so audit trails remain reconstructible, but
no consumer can mistake the row for a real $100 fill.

This is the upstream-emitter remediation half of P0-1 (the downstream
defense — trade_closer skip + SCR skip + profit_lock skip — remains
intact and is exercised elsewhere).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import pytest

from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
    _PLACEHOLDER_FILL_PRICE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_price_cache(tmp_path: Path, prices: dict) -> Path:
    """Write a minimal runtime/price_cache.json under *tmp_path* and patch
    the module-level PRICE_CACHE_PATH so the normalizer's lookup hits it."""
    cache_dir = tmp_path / "runtime"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "price_cache.json"
    cache_file.write_text(json.dumps({"prices": prices}), encoding="utf-8")
    return cache_file


@pytest.fixture
def patched_price_cache(monkeypatch, tmp_path):
    """Yields a helper that points _lookup_paper_fill_price at a temp cache."""

    def _apply(prices: dict) -> Path:
        cache_file = _seed_price_cache(tmp_path, prices)
        import chad.execution.paper_exec_evidence_writer as mod
        monkeypatch.setattr(mod, "PRICE_CACHE_PATH", cache_file, raising=True)
        return cache_file

    return _apply


def _build_ev(
    *,
    symbol: str,
    side: str,
    qty: float,
    fill_price: float,
    expected_price: float,
    asset_class: str = "equity",
    status: str = "filled",
    strategy: str = "delta",
) -> PaperExecEvidence:
    return PaperExecEvidence(
        symbol=symbol,
        side=side,
        quantity=qty,
        fill_price=fill_price,
        expected_price=expected_price,
        strategy=strategy,
        source_strategies=[strategy],
        broker="ibkr_paper",
        status=status,
        asset_class=asset_class,
        is_live=False,
        fill_time_utc="2026-05-23T00:15:24Z",
    )


# ---------------------------------------------------------------------------
# 1) delta/IWM placeholder with cache present → 5b deviation guard fires
# ---------------------------------------------------------------------------


def test_delta_iwm_placeholder_zeroes_top_level_fill_price(patched_price_cache):
    """The reported P0-1 case: delta SELL IWM @ $100 with price_cache=$285.21.
    Expectation: row is still rejected/untrusted, BUT top-level fill_price,
    notional, and expected_price are zero. The original placeholder values
    are preserved under ev.extra["placeholder_*"]."""
    patched_price_cache({"IWM": 285.21})

    ev = _build_ev(
        symbol="IWM", side="SELL", qty=10.0,
        fill_price=100.0, expected_price=100.0,
    )
    normalize_paper_fill_evidence(ev)

    # Trust boundary still flips to rejected.
    assert ev.status == "rejected"
    assert ev.reject is True

    # Numeric $100 fingerprint must be GONE from top-level fields.
    assert ev.fill_price == 0.0
    assert ev.expected_price == 0.0
    assert ev.notional == 0.0

    # extra carries the explicit trust marker.
    assert isinstance(ev.extra, dict)
    assert ev.extra.get("trust_state") == "PLACEHOLDER"
    assert ev.extra.get("pnl_untrusted") is True
    reason = ev.extra.get("pnl_untrusted_reason", "")
    assert "placeholder_no_broker_confirmed_fill_price" in reason

    # Forensic originals preserved.
    assert ev.extra.get("placeholder_fill_price") == 100.0
    assert ev.extra.get("placeholder_expected_price") == 100.0
    assert ev.extra.get("placeholder_price_cache") == 285.21

    # Tags include the placeholder marker.
    assert "placeholder" in ev.tags
    assert "pnl_untrusted" in ev.tags


# ---------------------------------------------------------------------------
# 2) reconciler/IWM placeholder — same chokepoint, different attribution
# ---------------------------------------------------------------------------


def test_reconciler_iwm_placeholder_also_zeroed(patched_price_cache):
    patched_price_cache({"IWM": 283.6})

    ev = _build_ev(
        symbol="IWM", side="SELL", qty=10.0,
        fill_price=100.0, expected_price=100.0,
        strategy="reconciler",
    )
    normalize_paper_fill_evidence(ev)

    assert ev.fill_price == 0.0
    assert ev.expected_price == 0.0
    assert ev.notional == 0.0
    assert ev.status == "rejected"
    assert ev.reject is True
    assert ev.extra.get("trust_state") == "PLACEHOLDER"


# ---------------------------------------------------------------------------
# 3) SPY placeholder with cache → 5b deviation guard zeroes
# ---------------------------------------------------------------------------


def test_spy_placeholder_with_cache_zeroes(patched_price_cache):
    """Mirror case from the 2026-05-08 SPY incident pattern."""
    patched_price_cache({"SPY": 720.0})

    ev = _build_ev(
        symbol="SPY", side="BUY", qty=1.0,
        fill_price=100.0, expected_price=100.0,
        strategy="delta",
    )
    normalize_paper_fill_evidence(ev)

    assert ev.fill_price == 0.0
    assert ev.expected_price == 0.0
    assert ev.notional == 0.0
    assert ev.extra.get("trust_state") == "PLACEHOLDER"
    assert ev.extra.get("placeholder_fill_price") == 100.0
    assert ev.extra.get("placeholder_price_cache") == 720.0


# ---------------------------------------------------------------------------
# 4) IWM placeholder WITHOUT price_cache → 5a "liquid equity" guard fires
# ---------------------------------------------------------------------------


def test_iwm_placeholder_without_price_cache_zeroes_via_liquid_allowlist(
    patched_price_cache,
):
    """When price_cache has no IWM entry but symbol is a known liquid ETF,
    the 5a guard recognises the canonical 100.0/100.0 placeholder shape
    and demotes accordingly. Same top-level zeroing behaviour as 5b."""
    patched_price_cache({})  # empty cache

    ev = _build_ev(
        symbol="IWM", side="SELL", qty=10.0,
        fill_price=100.0, expected_price=100.0,
        strategy="delta",
    )
    normalize_paper_fill_evidence(ev)

    assert ev.fill_price == 0.0
    assert ev.expected_price == 0.0
    assert ev.notional == 0.0
    assert ev.status == "rejected"
    assert ev.reject is True
    assert ev.extra.get("trust_state") == "PLACEHOLDER"
    # 5a reason marker is the legacy "placeholder_price_without_price_cache"
    assert ev.extra.get("pnl_untrusted_reason") == "placeholder_price_without_price_cache"
    assert ev.extra.get("placeholder_fill_price") == 100.0
    assert ev.extra.get("placeholder_price_cache") == 0.0


# ---------------------------------------------------------------------------
# 5) Real broker-confirmed fill is untouched
# ---------------------------------------------------------------------------


def test_real_broker_fill_untouched_by_placeholder_guards(patched_price_cache):
    """An IWM BUY at $285 with price_cache=$285 must NOT be zeroed."""
    patched_price_cache({"IWM": 285.21})

    ev = _build_ev(
        symbol="IWM", side="BUY", qty=10.0,
        fill_price=285.20, expected_price=285.20,
        strategy="delta",
    )
    normalize_paper_fill_evidence(ev)

    # Real fill survives intact.
    assert ev.fill_price == 285.20
    assert ev.expected_price == 285.20
    assert ev.reject is False
    assert (ev.status or "").lower() != "rejected"
    # No placeholder markers stamped.
    extra = ev.extra if isinstance(ev.extra, dict) else {}
    assert extra.get("trust_state") != "PLACEHOLDER"
    assert "placeholder" not in (ev.tags or ())


# ---------------------------------------------------------------------------
# 6) The 5b deviation guard at the boundary (51% deviation) still zeroes
# ---------------------------------------------------------------------------


def test_5b_just_above_50_pct_deviation_zeroes(patched_price_cache):
    """A fill at $48 when cache says $100 (52% deviation) is also untrusted."""
    patched_price_cache({"AAPL": 100.0})

    ev = _build_ev(
        symbol="AAPL", side="BUY", qty=5.0,
        fill_price=48.0, expected_price=48.0,
        strategy="delta",
    )
    normalize_paper_fill_evidence(ev)

    assert ev.fill_price == 0.0
    assert ev.expected_price == 0.0
    assert ev.notional == 0.0
    assert ev.extra.get("trust_state") == "PLACEHOLDER"
    # Forensic record preserves the original deviating values.
    assert ev.extra.get("placeholder_fill_price") == 48.0
    assert ev.extra.get("placeholder_price_cache") == 100.0


# ---------------------------------------------------------------------------
# 7) The 50%-boundary case (exactly 50%) does NOT trigger zeroing
# ---------------------------------------------------------------------------


def test_5b_at_50_pct_deviation_does_not_zero(patched_price_cache):
    """The guard is "> 0.50" strictly, so exactly 50% deviation passes."""
    patched_price_cache({"IWM": 200.0})

    ev = _build_ev(
        symbol="IWM", side="BUY", qty=1.0,
        fill_price=100.0, expected_price=100.0,
        strategy="delta",
    )
    normalize_paper_fill_evidence(ev)

    # 50% exactly is not >50%, so no zeroing.
    assert ev.fill_price == 100.0
    assert ev.expected_price == 100.0


# ---------------------------------------------------------------------------
# 8) SCR effective_trades-style consumer cannot see numeric 100.0
# ---------------------------------------------------------------------------


def test_scr_consumer_sees_zero_fill_price_after_normalize(patched_price_cache):
    """Pin the consumer-side contract: after normalize, any reader that
    naively does ``payload['fill_price'] > 0`` to detect a real fill must
    NOT see the placeholder row as trusted."""
    patched_price_cache({"IWM": 285.21})

    ev = _build_ev(
        symbol="IWM", side="SELL", qty=10.0,
        fill_price=100.0, expected_price=100.0,
        strategy="delta",
    )
    normalize_paper_fill_evidence(ev)

    # Simulate the naive consumer.
    is_trusted_numeric = (
        ev.fill_price > 0.0
        and (ev.status or "").lower() in ("filled", "paper_fill")
        and not (ev.extra or {}).get("pnl_untrusted", False)
    )
    assert is_trusted_numeric is False
