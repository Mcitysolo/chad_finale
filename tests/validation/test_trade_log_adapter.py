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
        "schema_version": "closed_trade.v1",
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
               "--now", "2026-07-10T00:00:00Z", "--no-verify-chain"])  # synthetic hashes
    assert rc == 0
    assert (out / "stage2_trades_2026-07-01_2026-07-31.ndjson").exists()
    assert (out / "stage2_manifest_2026-07-01_2026-07-31.json").exists()
    assert "ADMITTED (trusted, scorer-ready): 1" in capsys.readouterr().out


def test_cli_rejects_bad_date(tmp_path):
    assert main(["--since", "07/01/2026", "--trades-dir", str(tmp_path)]) == 2


def test_cli_rejects_inverted_window(tmp_path):
    assert main(["--since", "2026-07-10", "--until", "2026-07-01",
                 "--trades-dir", str(tmp_path)]) == 2


# --------------------------------------------------------------------------- #
# 12. Row-level idempotency — a duplicate record_hash is scored once.
# --------------------------------------------------------------------------- #
def test_duplicate_record_hash_deduped():
    a = _rec(_hash="dup", _seq=1)
    b = _rec(_hash="dup", _seq=2)  # same content hash → duplicate
    c = _rec(_hash="uniq", _seq=3)
    admitted, counters = adapt_records(_stream([a, b, c]))
    assert counters["admitted"] == 2
    assert counters["duplicate"] == 1


def test_missing_record_hash_not_deduped():
    """Rows without a record_hash cannot be de-duplicated — both admitted (honest)."""
    a = _rec(_hash=None, _seq=1)
    a["record_hash"] = None
    b = _rec(_hash=None, _seq=2)
    b["record_hash"] = None
    admitted, counters = adapt_records(_stream([a, b]))
    assert counters["admitted"] == 2
    assert counters["duplicate"] == 0


# --------------------------------------------------------------------------- #
# 13. Fixed-schema manifest — every exclusion reason key is always present.
# --------------------------------------------------------------------------- #
def test_manifest_excluded_by_reason_is_fixed_schema(tmp_path):
    src = tmp_path / "trades"
    src.mkdir()
    (src / "trade_history_20260703.ndjson").write_text(json.dumps(_rec()) + "\n")
    m = run_adapter(trades_dir=src, generated_at="2026-07-10T00:00:00Z").manifest
    assert set(m.excluded_by_reason) == set(EXCLUSION_REASONS)
    assert all(v == 0 for v in m.excluded_by_reason.values())
    assert m.to_dict()["duplicate"] == 0


# --------------------------------------------------------------------------- #
# 14. Constant parity with the live write-path defenses (pins the documented copy).
# --------------------------------------------------------------------------- #
def test_trust_constants_match_live_write_path():
    """The adapter's trust constants are cited copies of the live evidence-writer defenses
    (isolation forbids importing that module inside chad.validation). Tests are NOT
    isolation-bound, so import both here and pin them — catches a future writer-side edit."""
    from chad.execution.paper_exec_evidence_writer import (  # noqa: PLC0415 (lazy by design)
        _PAPER_REJECTED_STATUSES,
        _PLACEHOLDER_FILL_PRICE,
        _STATUS_CANON,
    )
    from chad.validation.trade_log_adapter import REJECTED_STATUSES, TRUSTED_FILL_STATUSES

    assert REJECTED_STATUSES == _PAPER_REJECTED_STATUSES
    assert PLACEHOLDER_FILL_PRICE == _PLACEHOLDER_FILL_PRICE
    # Every canonical genuine-fill target must be an admissible trusted status.
    assert set(_STATUS_CANON.values()) <= TRUSTED_FILL_STATUSES


# --------------------------------------------------------------------------- #
# 15. W3A-1 — schema-mapping completeness (gross-pnl D3, multiplier, asset_class, holds).
# --------------------------------------------------------------------------- #
def test_gross_pnl_preferred_over_net(monkeypatch=None):
    """D3: when a row carries gross_pnl, the harness is costed on GROSS (not the net `pnl`),
    so the harness haircut is the single cost authority and post-EP1 rows are not double-charged."""
    admitted, _ = adapt_records(_stream([_rec(pnl=8.0, gross_pnl=12.0)]))
    assert len(admitted) == 1
    assert admitted[0].gross_pnl == 12.0
    assert admitted[0].provenance["pnl_field"] == "gross_pnl"


def test_pnl_fallback_when_gross_absent():
    """Pre-EP1 rows carry no gross_pnl → fall back to `pnl`, recorded as pnl_field='pnl'."""
    admitted, _ = adapt_records(_stream([_rec(pnl=25.0)]))
    assert admitted[0].gross_pnl == 25.0
    assert admitted[0].provenance["pnl_field"] == "pnl"


def test_contract_multiplier_is_used():
    """A futures row's contract_multiplier (the key real rows carry) reaches the cost mapping.
    (Futures are excluded by default under D2, so admit them explicitly here to test the map.)"""
    admitted, _ = adapt_records(
        _stream([_rec(symbol="MESU6", asset_class="future", contract_multiplier=5.0)]),
        include_futures=True,
    )
    assert admitted[0].multiplier == 5.0
    assert admitted[0].instrument_class == InstrumentClass.FUT.value


def test_meta_raw_asset_class_classifies_equity():
    """An equity with a futures-looking symbol but meta.raw_asset_class='equity' → STK, not FUT."""
    rec = _rec(symbol="ZN", meta={"raw_asset_class": "equity"})
    del rec["payload"]["asset_class"]  # force the meta path
    admitted, _ = adapt_records(_stream([rec]))
    assert admitted[0].instrument_class == InstrumentClass.STK.value


def test_inverted_duration_is_kept_and_counted():
    """A real netting/clock artifact (exit < entry) is KEPT (honest data) but flagged + counted;
    hold_hours is the absolute span, never a negative hold."""
    rec = _rec(
        entry_time_utc="2026-07-20T13:51:10+00:00",
        exit_time_utc="2026-07-20T13:50:56Z",  # 14s before entry — the real 07-20 artifact
    )
    admitted, counters = adapt_records(_stream([rec]))
    assert len(admitted) == 1  # kept, not excluded
    assert admitted[0].provenance["inverted_duration"] is True
    assert admitted[0].provenance["hold_hours"] >= 0.0
    assert counters["inverted_duration"] == 1


def test_timestamp_tz_forms_both_parse():
    """Both `…Z` and `…+00:00` forms yield a finite non-inverted hold (real rows mix them)."""
    rec = _rec(
        entry_time_utc="2026-07-20T10:00:00Z",
        exit_time_utc="2026-07-20T12:00:00+00:00",
    )
    admitted, counters = adapt_records(_stream([rec]))
    assert admitted[0].provenance["inverted_duration"] is False
    assert abs(admitted[0].provenance["hold_hours"] - 2.0) < 1e-9
    assert counters["inverted_duration"] == 0


def test_manifest_carries_w3a1_fields(tmp_path):
    """run_adapter's manifest exposes the pnl-field tally + inverted count for audit."""
    d = tmp_path / "trades"
    d.mkdir()
    lines = [
        json.dumps(_rec(pnl=1.0, gross_pnl=2.0, _hash="a", _seq=1)),
        json.dumps(_rec(pnl=3.0, _hash="b", _seq=2)),
    ]
    (d / "trade_history_20260703.ndjson").write_text("\n".join(lines) + "\n")
    result = run_adapter(trades_dir=d, generated_at="2026-07-22T00:00:00Z")
    md = result.manifest.to_dict()
    assert md["admitted_pnl_field_counts"] == {"gross_pnl": 1, "pnl": 1}
    assert md["inverted_duration_admitted"] == 0


# --------------------------------------------------------------------------- #
# 16. W3A-2 — trust-gate alignment (manual/warmup_sim/futures) + integrity gates (D2/D6).
# --------------------------------------------------------------------------- #
def test_manual_hand_trade_is_excluded():
    """D2: SCR excludes operator hand-trades; the adapter mirrors it (tag or strategy)."""
    assert trust_exclusion(_rec(tags=["paper", "manual"])) == "manual"
    assert trust_exclusion(_rec(strategy="manual")) == "manual"
    admitted, c = adapt_records(_stream([_rec(strategy="manual")]))
    assert admitted == [] and c["excluded:manual"] == 1


def test_warmup_sim_is_excluded():
    """D2: synthetic warmup rows must never reach the scorer (SCR parity)."""
    assert trust_exclusion(_rec(tags=["warmup_sim"])) == "warmup_sim"
    _, c = adapt_records(_stream([_rec(tags=["warmup_sim"])]))
    assert c["excluded:warmup_sim"] == 1


def test_futures_excluded_by_default_bug_b():
    """D2: a futures row is Bug-B-contaminated → excluded by default, admitted with the flag."""
    fut = _rec(symbol="MESU6", asset_class="future", contract_multiplier=5.0)
    admitted, c = adapt_records(_stream([fut]))
    assert admitted == [] and c["excluded:futures_bug_b"] == 1
    admitted2, c2 = adapt_records(_stream([_rec(symbol="MESU6", asset_class="future",
                                                contract_multiplier=5.0)]), include_futures=True)
    assert len(admitted2) == 1 and c2["excluded:futures_bug_b"] == 0


def test_non_closed_trade_schema_excluded():
    """D6: a legacy/foreign-writer row without schema_version=='closed_trade.v1' is excluded."""
    legacy = _rec()
    del legacy["payload"]["schema_version"]
    admitted, c = adapt_records(_stream([legacy]))
    assert admitted == [] and c["excluded:non_closed_trade"] == 1
    wrong = _rec(schema_version="open_fill.v0")
    _, c2 = adapt_records(_stream([wrong]))
    assert c2["excluded:non_closed_trade"] == 1


def test_verify_ledger_chain_intact_and_tampered(tmp_path):
    """D6: an intact synthetic chain yields no issues; a mutated payload / broken link is caught."""
    import hashlib as _h, json as _j
    from chad.validation.trade_log_adapter import verify_ledger_chain

    def _hash(core):
        return _h.sha256(_j.dumps(core, sort_keys=True, default=str).encode()).hexdigest()

    rows, prev = [], "GENESIS"
    for seq in (1, 2, 3):
        core = {"payload": {"schema_version": "closed_trade.v1", "pnl": float(seq)},
                "prev_hash": prev, "sequence_id": seq, "timestamp_utc": f"2026-07-20T0{seq}:00:00Z"}
        core["record_hash"] = _hash({k: core[k] for k in ("payload", "prev_hash", "sequence_id", "timestamp_utc")})
        rows.append(core)
        prev = core["record_hash"]
    p = tmp_path / "trade_history_20260720.ndjson"
    p.write_text("\n".join(_j.dumps(r) for r in rows) + "\n")
    assert verify_ledger_chain(p) == []  # intact

    rows[1]["payload"]["pnl"] = 999.0  # tamper a payload without re-hashing
    p.write_text("\n".join(_j.dumps(r) for r in rows) + "\n")
    issues = verify_ledger_chain(p)
    assert any("record_hash mismatch" in i for i in issues)


def test_run_adapter_verify_chain_fails_loud(tmp_path):
    """D6: run_adapter(verify_chain=True) refuses to admit from a broken-chain ledger."""
    from chad.validation.trade_log_adapter import LedgerChainError
    d = tmp_path / "trades"
    d.mkdir()
    # A row whose prev_hash is not GENESIS → broken link on the first row.
    bad = {"payload": {"schema_version": "closed_trade.v1"}, "prev_hash": "NOTGENESIS",
           "sequence_id": 1, "timestamp_utc": "2026-07-20T01:00:00Z", "record_hash": "x"}
    (d / "trade_history_20260720.ndjson").write_text(json.dumps(bad) + "\n")
    with pytest.raises(LedgerChainError):
        run_adapter(trades_dir=d, generated_at="2026-07-22T00:00:00Z", verify_chain=True)
    # default (verify_chain=False) does not raise — still admits/handles gracefully.
    run_adapter(trades_dir=d, generated_at="2026-07-22T00:00:00Z")


# --------------------------------------------------------------------------- #
# 17. W3A-3 — read-only SCR reconciliation cross-check (D2; fail-loud on absence).
# --------------------------------------------------------------------------- #
def _write_scr(tmp_path, effective, ts="2026-07-22T18:09:35Z"):
    p = tmp_path / "scr_state.json"
    p.write_text(json.dumps({"state": "WARMUP", "ts_utc": ts,
                             "stats": {"effective_trades": effective}}))
    return p


def test_load_scr_effective_reads_effective_trades(tmp_path):
    from chad.validation.trade_log_adapter import load_scr_effective
    eff, ts = load_scr_effective(_write_scr(tmp_path, 67))
    assert eff == 67 and ts == "2026-07-22T18:09:35Z"
    assert load_scr_effective(None) == (None, None)  # disabled → no read


def test_load_scr_effective_fails_loud_on_absence(tmp_path):
    from chad.validation.trade_log_adapter import ScrCrosscheckError, load_scr_effective
    with pytest.raises(ScrCrosscheckError):
        load_scr_effective(tmp_path / "does_not_exist.json")
    (tmp_path / "bad.json").write_text('{"no_stats": true}')
    with pytest.raises(ScrCrosscheckError):
        load_scr_effective(tmp_path / "bad.json")


def test_run_adapter_reconciliation_block(tmp_path):
    """The manifest surfaces the admitted-vs-SCR-effective delta (D2) when a path is supplied,
    including the pnl==0 laps the adapter keeps but SCR drops."""
    d = tmp_path / "trades"
    d.mkdir()
    lines = [
        json.dumps(_rec(pnl=5.0, gross_pnl=5.0, _hash="a", _seq=1)),
        json.dumps(_rec(pnl=0.0, gross_pnl=0.0, _hash="b", _seq=2)),  # 0-return: kept, SCR drops
    ]
    (d / "trade_history_20260703.ndjson").write_text("\n".join(lines) + "\n")
    scr = _write_scr(tmp_path, 1)  # SCR scored 1; adapter admits 2
    result = run_adapter(trades_dir=d, generated_at="2026-07-22T00:00:00Z", scr_state_path=scr)
    recon = result.manifest.to_dict()["scr_reconciliation"]
    assert recon["scr_effective_trades"] == 1
    assert recon["adapter_admitted"] == 2
    assert recon["delta_admitted_minus_scr"] == 1
    assert recon["adapter_admitted_pnl_zero"] == 1
    # Disabled by default → no reconciliation block.
    result2 = run_adapter(trades_dir=d, generated_at="2026-07-22T00:00:00Z")
    assert result2.manifest.to_dict()["scr_reconciliation"] is None


def test_run_adapter_scr_crosscheck_fails_loud(tmp_path):
    from chad.validation.trade_log_adapter import ScrCrosscheckError
    d = tmp_path / "trades"
    d.mkdir()
    (d / "trade_history_20260703.ndjson").write_text(json.dumps(_rec()) + "\n")
    with pytest.raises(ScrCrosscheckError):
        run_adapter(trades_dir=d, generated_at="2026-07-22T00:00:00Z",
                    scr_state_path=tmp_path / "absent_scr.json")
