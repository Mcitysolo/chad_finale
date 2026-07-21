"""W2BS (Q2) — tonight's shadow receipt gamma|MSFT|SELL is a RAW pre-filter urge
that the D4 guardrail eats before execution, with the operator-exclusion SSOT as
an independent second line.

Fixture = the REAL 2026-07-21 ctx-positions shadow receipt. The live shadow (up
since 00:55) recorded ``added_sells=[["gamma","MSFT","SELL"]]``: injecting the
CHAD-netted book woke gamma's dormant native equity exit on MSFT. This test pins
WHY that is safe, tying the live evidence to the code path:

  * Q1 — the shadow diff is computed on the UNFILTERED counterfactual, so the
    receipt's ``added_sells`` is a RAW urge, not "what would reach execution".
    Proven here by the filter being INERT in shadow/off (the receipt could only
    record the urge because the guardrail does not act in shadow).
  * Q2a — ``filter_overlay_owned_exits(mode="on")`` DROPS the MSFT SELL: it is a
    strategy EQUITY sell, and the ACTIVE exit overlay is the sole equity/ETF exit
    authority (D4). So on an ON cycle the urge never becomes an order.
  * Q2b — defense in depth: MSFT ∈ the operator-exclusion SSOT
    (``_EFFECTIVE_NON_CHAD_SYMBOLS``), which the GAP-001 corrected-scope
    chokepoints all consult. Even if the D4 filter missed it, the ONLY close
    path (``apply_close_intents``) refuses any MSFT close before it can touch the
    adapter (idem for the reconciler close-intent generator and the BG11 flip
    path). Proven behaviorally: the adapter is never called for an MSFT close.

Hermetic: prefers the on-disk live receipt when present, else the verbatim
embedded copy — so the test is portable AND genuinely tied to tonight's record.
"""
from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

from chad.core import context_positions as cp
from chad.core.ctx_positions_shadow import signal_keys
from chad.core.position_reconciler import (
    _EFFECTIVE_NON_CHAD_SYMBOLS,
    apply_close_intents,
)
from chad.types import AssetClass, SignalSide

# The verbatim record captured from the live 2026-07-21 shadow receipt
# (data/ctx_positions_shadow/ctx_positions_20260721.ndjson), preserved so the
# regression survives log rotation / a fresh checkout.
_TONIGHT_RECEIPT = {
    "added": [["gamma", "MSFT", "SELL"]],
    "added_sells": [["gamma", "MSFT", "SELL"]],
    "added_sells_count": 1,
    "mode": "shadow",
    "n_injected": 8,
    "n_off": 7,
    "n_on": 4,
    "positions_status": "KNOWN",
    "removed": [["gamma", "AAPL", "BUY"], ["gamma", "BAC", "BUY"],
                ["gamma", "UNH", "BUY"], ["gamma", "V", "BUY"]],
    "schema_version": "ctx_positions_shadow.v1",
    "ts_utc": "2026-07-21T01:44:47Z",
    "unchanged": 3,
}

_LIVE_RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "data" / "ctx_positions_shadow" / "ctx_positions_20260721.ndjson"
)

_Sig = namedtuple("Sig", "strategy symbol side asset_class")


def _receipt() -> dict:
    """Tonight's shadow record: the last on-disk line that flags the MSFT sell,
    else the embedded verbatim copy. Either way it is real 2026-07-21 evidence."""
    if _LIVE_RECEIPT.is_file():
        for line in reversed(_LIVE_RECEIPT.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("added_sells"):
                return rec
    return _TONIGHT_RECEIPT


def _msft_sell_from_receipt():
    """Reconstruct the exact TradeSignal the receipt's added_sells tuple stands
    for: gamma / MSFT / SELL, equity (MSFT classifies EQUITY, D6)."""
    rec = _receipt()
    strat, sym, side = rec["added_sells"][0]
    assert (strat, sym, side) == ("gamma", "MSFT", "SELL")
    sig = _Sig(strat, sym, SignalSide.SELL, AssetClass.EQUITY)
    # The reconstructed signal maps back to the receipt tuple 1:1 (tie evidence
    # to code): the shadow recorder's own key extractor agrees.
    assert signal_keys({strat: [sig]}) == {(strat, sym, side)}
    return sig


# --------------------------------------------------------------------------- #
# The receipt itself — the raw urge exists and is a single equity SELL.
# --------------------------------------------------------------------------- #

def test_receipt_flags_exactly_the_msft_sell():
    rec = _receipt()
    assert rec["mode"] == "shadow"                       # acted on nothing
    assert rec["added_sells"] == [["gamma", "MSFT", "SELL"]]
    assert rec["added_sells_count"] == 1
    # MSFT is the ONLY added sell; the other diffs are removed gamma BUYs (the
    # churn cure), never new sells.
    assert all(k[2] != "SELL" for k in rec["removed"])


# --------------------------------------------------------------------------- #
# Q2a — the D4 guardrail eats the MSFT sell (equity/ETF sells belong to overlay).
# --------------------------------------------------------------------------- #

def test_guardrail_eats_msft_sell_on():
    sig = _msft_sell_from_receipt()
    assert cp.is_overlay_owned_exit(sig) is True         # equity SELL == overlay-owned
    kept, dropped = cp.filter_overlay_owned_exits([sig], mode="on")
    assert kept == [] and dropped == [sig]               # never reaches execution


# --------------------------------------------------------------------------- #
# Q1 — the filter is INERT in shadow/off, which is exactly WHY the receipt could
# record the urge: the shadow diff is the RAW pre-filter counterfactual.
# --------------------------------------------------------------------------- #

def test_urge_is_raw_pre_filter_inert_in_shadow_and_off():
    sig = _msft_sell_from_receipt()
    for mode in ("shadow", "off"):
        kept, dropped = cp.filter_overlay_owned_exits([sig], mode=mode)
        assert kept == [sig] and dropped == []           # survives -> recorded as a raw urge


# --------------------------------------------------------------------------- #
# Q2b — defense in depth: the operator-exclusion SSOT independently blocks MSFT.
# --------------------------------------------------------------------------- #

def test_msft_is_operator_excluded_ssot():
    assert "MSFT" in _EFFECTIVE_NON_CHAD_SYMBOLS         # GAP-001 SSOT all chokepoints read


def test_only_close_path_refuses_msft_before_the_adapter():
    """apply_close_intents (the ONLY close path) skips an MSFT close on the
    operator-exclusion guard BEFORE it can submit to the adapter — so even a
    guardrail miss cannot open/close CHAD inventory in MSFT."""
    calls: list = []

    class _FakeAdapter:
        def submit_strategy_trade_intents(self, intents):
            calls.append(intents)                        # must never be reached for MSFT
            return []

    apply_close_intents(
        [{"symbol": "MSFT", "close_side": "SELL", "quantity": 16.0,
          "strategy": "gamma", "reason": "test_excluded", "position_key": "gamma|MSFT"}],
        _FakeAdapter(),
    )
    assert calls == []                                   # adapter untouched -> close refused
