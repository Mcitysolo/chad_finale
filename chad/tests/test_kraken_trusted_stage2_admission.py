"""U1 CRYPTO-TRUST — Stage-2 admission proof.

Drives a real trusted Kraken round-trip through the REAL trade_history writer
(redirected to tmp_path) and the REAL Stage-2 adapter, proving:
  * a kraken_paper_v1 round-trip is ADMITTED (passes the trust filters);
  * it is counted as SIMULATED_AGAINST_LIVE_TICKS crypto SEPARATELY from
    broker-confirmed evidence (the honesty guard);
  * it is NOT excluded as validate_only / pnl_untrusted.
"""

from __future__ import annotations

import chad.analytics.trade_result_logger as trl
from chad.core import kraken_trusted_fill_engine as tfe
from chad.execution.kraken_trading_config import load_kraken_trading_config
from chad.validation.trade_log_adapter import run_adapter

_NOW = 1_800_000_000.0
_CFG = load_kraken_trading_config()


class _FakeTickSource:
    def __init__(self, touch):
        self._touch = touch

    def get_touch(self, symbol, *, now_epoch):
        return self._touch


class _Intent:
    def __init__(self, side, volume=1.0):
        self.strategy = "alpha_crypto"
        self.pair = "SOLUSD"
        self.side = side
        self.volume = volume
        self.ordertype = "market"
        self.markers = ()
        self.idempotency_key = ""
        self.trace_id = ""


def _engine(tmp_path):
    return tfe.TrustedFillEngine(
        config=_CFG,
        tick_source=_FakeTickSource(tfe.Touch(bid=170.0, ask=170.2, last=170.1)),
        book=tfe.RoundTripBook(db_path=tmp_path / "book.sqlite3",
                               now_iso=lambda: "2026-07-12T00:00:00Z"),
        now_fn=lambda: _NOW,
        # Stage-2 reads only trade_history; skip the FILLS surface here and use
        # the REAL trade_history writer (log_trade_result -> patched TRADE_DIR).
        evidence_writer=lambda kw: "",
        trade_history_writer=None,
    )


def test_kraken_paper_v1_roundtrip_is_admitted_and_counted_separately(tmp_path, monkeypatch):
    trades_dir = tmp_path / "trades"
    monkeypatch.setattr(trl, "TRADE_DIR", trades_dir)

    eng = _engine(tmp_path)
    r_open = eng.process_intent(_Intent("buy", 1.0))    # open long
    r_close = eng.process_intent(_Intent("sell", 1.0))  # close -> realized trade_history row
    assert r_open["leg"] == "open" and r_close["leg"] == "close"
    assert len(r_close["realized"]) == 1

    result = run_adapter(trades_dir=trades_dir)
    m = result.manifest

    # ADMITTED — the trust gate did NOT reject it.
    assert m.admitted >= 1
    assert result.admitted, "trusted Kraken round-trip must reach the scorer seam"
    admitted = result.admitted[0]
    assert admitted.instrument_class.upper() == "CRYPTO"
    assert admitted.provenance.get("broker") == "kraken_paper"
    assert admitted.provenance.get("fee_model") == "kraken_paper_v1"

    # Honesty guard — simulated-crypto counted SEPARATELY.
    assert m.admitted_by_provenance.get("SIMULATED_AGAINST_LIVE_TICKS", 0) >= 1
    assert sum(v for k, v in m.admitted_by_instrument_class.items()
               if k.upper() == "CRYPTO") >= 1

    # NOT excluded as untrusted.
    assert m.excluded_by_reason.get("validate_only", 0) == 0
    assert m.excluded_by_reason.get("pnl_untrusted", 0) == 0


def test_legacy_untrusted_kraken_row_is_still_excluded(tmp_path, monkeypatch):
    """Guard: the OLD untrusted shape (validate_only + pnl_untrusted) must still
    be rejected — proving the admission above is earned by the trust labels."""
    trades_dir = tmp_path / "trades"
    monkeypatch.setattr(trl, "TRADE_DIR", trades_dir)
    trl.log_trade_result(trl.TradeResult(
        strategy="alpha_crypto", symbol="SOL-USD", side="SELL", quantity=1.0,
        fill_price=170.0, notional=170.0, pnl=0.0,
        entry_time_utc="2026-07-12T00:00:00Z", exit_time_utc="2026-07-12T00:00:00Z",
        is_live=False, broker="kraken_paper",
        tags=["kraken_paper", "validate_only", "paper_fill", "pnl_untrusted"],
        extra={"validate_only": True, "pnl_untrusted": True},
    ))
    m = run_adapter(trades_dir=trades_dir).manifest
    assert m.admitted == 0
    assert m.excluded_by_reason.get("validate_only", 0) >= 1
