"""Unit tests for chad.validation.trade_log_adapter (Phase 6, SSOT §1.3 / Part 6).

Synthetic in-memory ndjson fixtures only — nothing here reads the real ledger, the
network, a broker, or any runtime state. Every trust-filter exclusion reason, the happy
path, malformed rows (skip + count, never crash), empty input, window filtering, the
fail-closed structural guarantee, and determinism (byte-identical output) are covered.

The seam is verified concretely: an admitted trade's ``to_fill_mapping()`` must construct a
valid :class:`chad.validation.cost_model.Trade` (via ``from_fill``) and cost without error —
proving the adapter output plugs into the identical Stage-1 cost path (S4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.validation.cost_model import (
    DEFAULT_COST_CONFIG,
    InstrumentClass,
    LiquidityTier,
    Trade,
    apply_costs,
)
from chad.validation.trade_log_adapter import (
    EXCLUSION_REASONS,
    PLACEHOLDER_FILL_PRICE,
    SCHEMA_VERSION,
    AdmittedTrade,
    adapt_records,
    classify_instrument,
    is_placeholder_fill,
    iter_ledger_files,
    main,
    run_adapter,
    trust_exclusion,
)


# --------------------------------------------------------------------------- #
# Record builders (mirror the real trade_history_*.ndjson payload shape).
# --------------------------------------------------------------------------- #
def _rec(**payload_over):
    """A wrapped ledger record with a trusted-by-default payload; overrides merge in."""
    extra = payload_over.pop("extra", {})
    payload = {
        "broker": "paper_exec",
        "symbol": "AAPL",
        "asset_class": "equity",
        "side": "BUY",
        "strategy": "alpha_test",
        "quantity": 10.0,
        "fill_price": 190.0,
        "notional": 1900.0,
        "pnl": 25.0,
        "is_live": False,
        "regime": "risk_on",
        "entry_time_utc": "2026-07-03T10:00:00+00:00",
        "exit_time_utc": "2026-07-03T11:00:00+00:00",
        "tags": [],
        "extra": extra,
    }
    payload.update(payload_over)
    return {
        "payload": payload,
        "record_hash": payload_over.get("_hash", "h0"),
        "sequence_id": payload_over.get("_seq", 1),
        "timestamp_utc": "2026-07-03T11:00:00.5+00:00",
    }


def _stream(records, source="trade_history_20260703.ndjson"):
    return [(r, source) for r in records]


# --------------------------------------------------------------------------- #
# 1. Happy path — a trusted equity round-trip is admitted + mapped correctly.
# --------------------------------------------------------------------------- #
def test_happy_path_admits_and_maps():
    admitted, counters = adapt_records(_stream([_rec()]))
    assert counters["admitted"] == 1
    assert counters["rows_read"] == 1
    t = admitted[0]
    assert isinstance(t, AdmittedTrade)
    assert t.instrument_class == InstrumentClass.STK.value
    assert t.quantity == 10.0
    assert t.fill_price == 190.0
    assert t.notional == 1900.0
    assert t.gross_pnl == 25.0
    assert t.strategy == "alpha_test"
    assert t.provenance["symbol"] == "AAPL"
    assert t.provenance["source_file"] == "trade_history_20260703.ndjson"


def test_to_fill_mapping_builds_valid_costable_trade():
    """The seam: admitted → from_fill → apply_costs, no error, prices = fill (both legs)."""
    admitted, _ = adapt_records(_stream([_rec()]))
    mapping = admitted[0].to_fill_mapping()
    assert mapping["entry_price"] == mapping["exit_price"] == 190.0
    assert mapping["pnl"] == 25.0
    trade = Trade.from_fill(mapping)
    assert isinstance(trade, Trade)
    assert trade.instrument_class is InstrumentClass.STK
    assert trade.liquidity_tier is LiquidityTier.MID
    breakdown = apply_costs(trade, DEFAULT_COST_CONFIG)
    assert breakdown.total_cost >= 0.0
    # gross_pnl flows through; net = gross - costs (the S4 haircut on a real fill).
    assert breakdown.gross_pnl == 25.0
    assert breakdown.net_pnl == pytest.approx(25.0 - breakdown.total_cost)


# --------------------------------------------------------------------------- #
# 2. Every exclusion reason (fail-closed) — none reach `admitted`.
# --------------------------------------------------------------------------- #
def test_excludes_pnl_untrusted():
    admitted, c = adapt_records(_stream([_rec(extra={"pnl_untrusted": True,
                                                     "pnl_untrusted_reason": "scr_reset"})]))
    assert admitted == []
    assert c["excluded:pnl_untrusted"] == 1


def test_excludes_validate_only():
    admitted, c = adapt_records(_stream([_rec(broker="kraken_paper", symbol="SOL-USD",
                                              asset_class="crypto",
                                              extra={"validate_only": True})]))
    assert admitted == []
    assert c["excluded:validate_only"] == 1


def test_excludes_broker_rejected_by_status():
    admitted, c = adapt_records(_stream([_rec(status="rejected")]))
    assert admitted == []
    assert c["excluded:broker_rejected"] == 1


def test_excludes_broker_rejected_by_tag():
    admitted, c = adapt_records(_stream([_rec(tags=["pnl_untrusted", "broker_rejected"])]))
    assert admitted == []
    # broker_rejected tag precedes the pnl_untrusted flag in the gate.
    assert c["excluded:broker_rejected"] == 1


def test_excludes_non_fill_status():
    admitted, c = adapt_records(_stream([_rec(status="PendingSubmit")]))
    assert admitted == []
    assert c["excluded:non_fill_status"] == 1


def test_admits_canonical_paper_fill_status():
    """A row that DOES carry the canonical trusted status is admitted."""
    admitted, c = adapt_records(_stream([_rec(status="paper_fill")]))
    assert c["admitted"] == 1


def test_excludes_100_placeholder_equity():
    admitted, c = adapt_records(_stream([_rec(
        fill_price=PLACEHOLDER_FILL_PRICE, notional=100.0, quantity=1.0,
        asset_class="equity", extra={"expected_price": 100.0})]))
    assert admitted == []
    assert c["excluded:placeholder_100"] == 1


def test_excludes_100_placeholder_by_trust_state_marker():
    admitted, c = adapt_records(_stream([_rec(extra={"trust_state": "PLACEHOLDER"})]))
    assert admitted == []
    assert c["excluded:placeholder_100"] == 1


def test_non_placeholder_100_priced_crypto_is_not_flagged():
    """A legit crypto at exactly $100 (not equity) is NOT a placeholder — admitted."""
    admitted, c = adapt_records(_stream([_rec(
        broker="kraken_paper", symbol="XYZ-USD", asset_class="crypto",
        fill_price=100.0, notional=1000.0, quantity=10.0,
        extra={"expected_price": 100.0})]))
    assert c["admitted"] == 1
    assert c["excluded:placeholder_100"] == 0


def test_every_reason_bucket_exists_in_counters():
    _, c = adapt_records(_stream([_rec()]))
    for reason in EXCLUSION_REASONS:
        assert f"excluded:{reason}" in c


# --------------------------------------------------------------------------- #
# 3. Malformed rows — skip + count, never crash.
# --------------------------------------------------------------------------- #
def test_malformed_none_record_counted_not_crash():
    admitted, c = adapt_records([(None, "f.ndjson"), (_rec(), "f.ndjson")])
    assert c["admitted"] == 1
    assert c["malformed"] == 1
    assert c["rows_read"] == 2


def test_malformed_missing_fields_counted():
    bad = _rec()
    del bad["payload"]["quantity"]
    admitted, c = adapt_records(_stream([bad]))
    assert admitted == []
    assert c["malformed"] == 1


@pytest.mark.parametrize("field,value", [
    ("quantity", 0.0), ("quantity", -5.0), ("fill_price", 0.0),
    ("fill_price", None), ("pnl", None), ("notional", -1.0),
])
def test_malformed_nonpositive_or_missing_numeric(field, value):
    admitted, c = adapt_records(_stream([_rec(**{field: value})]))
    assert admitted == []
    assert c["malformed"] == 1


def test_notional_derived_from_qty_price_when_missing():
    r = _rec()
    del r["payload"]["notional"]
    admitted, c = adapt_records(_stream([r]))
    assert c["admitted"] == 1
    assert admitted[0].notional == pytest.approx(10.0 * 190.0)


def test_nan_pnl_is_malformed_not_crash():
    admitted, c = adapt_records(_stream([_rec(pnl=float("nan"))]))
    assert admitted == []
    assert c["malformed"] == 1


# --------------------------------------------------------------------------- #
# 4. Empty input — a valid empty result, not an error.
# --------------------------------------------------------------------------- #
def test_empty_input_is_valid_empty():
    admitted, c = adapt_records([])
    assert admitted == []
    assert c["rows_read"] == 0
    assert c["admitted"] == 0


def test_run_adapter_empty_dir(tmp_path):
    result = run_adapter(trades_dir=tmp_path, generated_at="2026-07-10T00:00:00Z")
    assert result.admitted == []
    assert result.manifest.admitted == 0
    assert result.manifest.rows_read == 0
    assert result.manifest.schema_version == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# 5. Window filtering.
# --------------------------------------------------------------------------- #
def test_window_excludes_out_of_range():
    inside = _rec(entry_time_utc="2026-07-03T10:00:00+00:00")
    before = _rec(entry_time_utc="2026-07-01T10:00:00+00:00")
    after = _rec(entry_time_utc="2026-07-09T10:00:00+00:00")
    admitted, c = adapt_records(_stream([inside, before, after]),
                                since="2026-07-02", until="2026-07-04")
    assert c["admitted"] == 1
    assert c["out_of_window"] == 2
    assert c["rows_in_window"] == 1


def test_window_open_ended():
    admitted, c = adapt_records(_stream([_rec()]), since=None, until=None)
    assert c["admitted"] == 1
    assert c["out_of_window"] == 0


# --------------------------------------------------------------------------- #
# 6. Determinism + stable ordering.
# --------------------------------------------------------------------------- #
def test_stable_ordering_by_time_seq_hash():
    a = _rec(entry_time_utc="2026-07-03T12:00:00+00:00", _seq=2, _hash="b")
    b = _rec(entry_time_utc="2026-07-03T09:00:00+00:00", _seq=1, _hash="a")
    admitted, _ = adapt_records(_stream([a, b]))
    # earlier entry_time_utc sorts first, regardless of input order.
    assert admitted[0].provenance["entry_time_utc"] == "2026-07-03T09:00:00+00:00"


def test_run_adapter_byte_identical(tmp_path):
    src = tmp_path / "trades"
    src.mkdir()
    (src / "trade_history_20260703.ndjson").write_text(
        "\n".join(json.dumps(_rec(_seq=i, _hash=f"h{i}")) for i in range(3)) + "\n"
    )
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run_adapter(trades_dir=src, out_dir=out_a, generated_at="2026-07-10T00:00:00Z")
    run_adapter(trades_dir=src, out_dir=out_b, generated_at="2026-07-10T00:00:00Z")
    for name in ("stage2_trades_open_open.ndjson", "stage2_manifest_open_open.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


# --------------------------------------------------------------------------- #
# 7. Manifest content — sha256, counts, provenance.
# --------------------------------------------------------------------------- #
def test_manifest_records_sha256_and_counts(tmp_path):
    src = tmp_path / "trades"
    src.mkdir()
    f = src / "trade_history_20260703.ndjson"
    rows = [_rec(), _rec(extra={"validate_only": True}), _rec(status="rejected")]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = run_adapter(trades_dir=src, generated_at="2026-07-10T00:00:00Z")
    m = result.manifest
    assert m.rows_read == 3
    assert m.admitted == 1
    assert m.excluded_by_reason.get("validate_only") == 1
    assert m.excluded_by_reason.get("broker_rejected") == 1
    assert len(m.input_files) == 1
    assert len(m.input_files[0]["sha256"]) == 64
    assert m.input_files[0]["rows"] == 3
    assert m.date_range_admitted == "2026-07-03..2026-07-03"
    assert m.strategies_admitted == {"alpha_test": 1}


# --------------------------------------------------------------------------- #
# 8. File globbing excludes backups.
# --------------------------------------------------------------------------- #
def test_iter_ledger_files_excludes_backups(tmp_path):
    src = tmp_path / "trades"
    src.mkdir()
    (src / "trade_history_20260703.ndjson").write_text("{}\n")
    (src / "trade_history_20260703.ndjson.scr_reset_bak").write_text("{}\n")
    (src / "trade_history_20260703.ndjson.pre_phase5_bak").write_text("{}\n")
    (src / "quarantine_20260508.json").write_text("{}\n")
    files = iter_ledger_files(src)
    assert [f.name for f in files] == ["trade_history_20260703.ndjson"]


# --------------------------------------------------------------------------- #
# 9. Instrument classification + placeholder helper (unit).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload,expected", [
    ({"asset_class": "crypto"}, InstrumentClass.CRYPTO),
    ({"asset_class": "futures"}, InstrumentClass.FUT),
    ({"asset_class": "equity"}, InstrumentClass.STK),
    ({"broker": "kraken_paper", "symbol": "SOL-USD"}, InstrumentClass.CRYPTO),
    ({"symbol": "M6E"}, InstrumentClass.FUT),
    ({"symbol": "MESU6"}, InstrumentClass.FUT),
    ({"symbol": "BTC-USD"}, InstrumentClass.CRYPTO),
    ({"symbol": "AAPL"}, InstrumentClass.STK),
])
def test_classify_instrument(payload, expected):
    assert classify_instrument(payload, payload.get("extra", {})) is expected


def test_is_placeholder_fill_numeric_fingerprint():
    assert is_placeholder_fill(
        {"fill_price": 100.0, "asset_class": "equity"}, {"expected_price": 0.0}) is True
    # crypto at 100 is not the equity placeholder fingerprint
    assert is_placeholder_fill(
        {"fill_price": 100.0, "asset_class": "crypto"}, {"expected_price": 100.0}) is False


# --------------------------------------------------------------------------- #
# 10. Fail-closed structural guarantee — the self-check catches a leak.
# --------------------------------------------------------------------------- #
def test_trust_exclusion_admits_clean_row():
    assert trust_exclusion(_rec()) is None


def test_self_check_holds_on_admitted(monkeypatch):
    """A non-deterministic gate that admits an untrusted row on the first pass but excludes
    it on the post-admission re-check must trip the fail-closed AssertionError, never ship it."""
    import chad.validation.trade_log_adapter as adapter

    calls = {"n": 0}
    real_gate = adapter.trust_exclusion

    def flaky_gate(record):
        calls["n"] += 1
        # First call (admission gate) says clean; later re-check reveals the real exclusion.
        if calls["n"] == 1:
            return None
        return real_gate(record)

    monkeypatch.setattr(adapter, "trust_exclusion", flaky_gate)
    leaky = _rec(extra={"validate_only": True})  # genuinely untrusted
    with pytest.raises(AssertionError):
        adapter.adapt_records(_stream([leaky]))


# --------------------------------------------------------------------------- #
# 11. CLI.
# --------------------------------------------------------------------------- #
def test_cli_writes_artifacts_and_returns_zero(tmp_path, capsys):
    src = tmp_path / "trades"
    src.mkdir()
    (src / "trade_history_20260703.ndjson").write_text(json.dumps(_rec()) + "\n")
    out = tmp_path / "out"
    rc = main(["--since", "2026-07-01", "--until", "2026-07-31",
               "--trades-dir", str(src), "--out-dir", str(out),
               "--now", "2026-07-10T00:00:00Z"])
    assert rc == 0
    assert (out / "stage2_trades_2026-07-01_2026-07-31.ndjson").exists()
    assert (out / "stage2_manifest_2026-07-01_2026-07-31.json").exists()
    assert "ADMITTED (trusted, scorer-ready): 1" in capsys.readouterr().out


def test_cli_rejects_bad_date(tmp_path):
    assert main(["--since", "07/01/2026", "--trades-dir", str(tmp_path)]) == 2


def test_cli_rejects_inverted_window(tmp_path):
    assert main(["--since", "2026-07-10", "--until", "2026-07-01",
                 "--trades-dir", str(tmp_path)]) == 2
