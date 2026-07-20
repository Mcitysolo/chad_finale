"""W2A-1: Stage-2 trade-log adapter honours the operator quarantine manifest.

The adapter (``chad.validation.trade_log_adapter``) is the on-demand Stage-2 scorekeeper.
Before W2A it judged rows purely by in-band trust markers, so a genuine-looking equity
round-trip with NO ``pnl_untrusted`` marker (exactly the PFF1 harvester double-book
phantoms) would be ADMITTED into the verdict engine. SCR already drops such rows when their
``record_hash`` is pinned in ``runtime/quarantine_manifest_*.json`` (via
``chad.utils.quarantine.get_exclusion_sets``); W2A-1 makes the OTHER scorekeeper honour the
same manifest so both exclude the identical set.

Design invariant under test: the adapter reads the manifest as TEXT with a stdlib-only
reader (``load_quarantine_pins``) and does NOT import ``chad.utils`` — the harness
import-closure isolation (``tests/validation/test_isolation.py``) must stay green (asserted
there, not here). These tests pin behaviour: a pinned trust-clean row is excluded as
``quarantined``; ``runtime_dir=None`` is byte-identical to pre-W2A.
"""
from __future__ import annotations

import json
from pathlib import Path

from chad.validation.trade_log_adapter import (
    adapt_records,
    is_quarantined,
    load_quarantine_pins,
    run_adapter,
)


# --------------------------------------------------------------------------- #
# Fixtures: trust-CLEAN, admissible equity round-trips (no in-band untrust marker),
# shaped like the 6 PFF1 gamma UNH phantom closes. Without a manifest these ADMIT.
# --------------------------------------------------------------------------- #
def _clean_row(record_hash: str, *, qty: float, pnl: float, price: float = 425.0,
               fill_id: str = "", entry_date: str = "2026-07-20") -> dict:
    payload = {
        "schema_version": "closed_trade.v1",
        "strategy": "gamma",
        "symbol": "UNH",
        "side": "SELL",
        "pnl": pnl,
        "fill_price": price,
        "quantity": qty,
        "notional": abs(qty * price),
        "broker": "paper_exec",
        "is_live": False,
        "entry_time_utc": f"{entry_date}T13:50:56Z",
        "exit_time_utc": f"{entry_date}T13:55:45Z",
        "tags": ["paper", "closed", "gamma"],
        "extra": {},
    }
    if fill_id:
        payload["fill_ids"] = [fill_id]
    return {"payload": payload, "sequence_id": 1, "record_hash": record_hash}


# The six real phantom record hashes (trade_history_20260720.ndjson seq 2-7).
_SIX_HASHES = [
    "0c70922e1dc265aaf171f4fc8a7c17bf5f8d282221483fb94113b3ca1f087dd6",
    "7bd3a872261ff9a4965c222d8fa39dcae059a7236f768f8cee817eca577e635e",
    "e6f51f9923947089e7f5ed4e9b4dabc44f65ab29bb06b1adad6ca0fb663f10b9",
    "687133278be20e0d87854ce71f80efb0c39e8bfa7bf8000fe54b7fbbd47a29fc",
    "ead6c9f71dc97e01ea7d6ab8a5b468494d56c9bb2100ad86e6b310bd35a79747",
    "3d0627208074e0dd3817853a07f114f6e2eeafb582a4b3cbec9b339c652a429d",
]
_SIX_QTY = [5, 5, 32, 40, 80, 71]
_SIX_PNL = [12.05, -3.55, -22.72, -28.00, -56.00, -47.57]


def _six_rows() -> list:
    return [
        _clean_row(h, qty=q, pnl=p)
        for h, q, p in zip(_SIX_HASHES, _SIX_QTY, _SIX_PNL)
    ]


def _write_manifest(runtime_dir: Path, name: str, *, hashes=(), fill_ids=()) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "quarantined_at_utc": "2026-07-20T00:00:00Z",
        "reason": "pff1_phantom_double_book",
        "invalid_trades": [{"record_hash": h} for h in hashes],
        "invalid_fills": [{"fill_id": f} for f in fill_ids],
    }
    (runtime_dir / name).write_text(json.dumps(doc), encoding="utf-8")


# --------------------------------------------------------------------------- #
# The core proof: pinned trust-clean rows are excluded (quarantined), not admitted.
# --------------------------------------------------------------------------- #
def test_clean_rows_admit_without_manifest():
    """Baseline: with no pins the 6 trust-clean rows all ADMIT (this is the bug)."""
    admitted, counters = adapt_records([(r, "f") for r in _six_rows()])
    assert counters["admitted"] == 6
    assert counters["excluded:quarantined"] == 0


def test_pinned_rows_excluded_as_quarantined():
    """W2A-1: pinning the 6 record_hashes drops all 6 as `quarantined`, admits 0."""
    hashes = frozenset(_SIX_HASHES)
    admitted, counters = adapt_records(
        [(r, "f") for r in _six_rows()], quarantined_hashes=hashes
    )
    assert counters["admitted"] == 0
    assert counters["excluded:quarantined"] == 6
    assert admitted == []


def test_pin_is_surgical_only_pinned_dropped():
    """Pinning 6 of 7 drops exactly those 6; the unpinned trust-clean row still admits."""
    extra = _clean_row("f" * 64, qty=9, pnl=1.0)
    rows = _six_rows() + [extra]
    admitted, counters = adapt_records(
        [(r, "f") for r in rows], quarantined_hashes=frozenset(_SIX_HASHES)
    )
    assert counters["admitted"] == 1
    assert counters["excluded:quarantined"] == 6
    assert admitted[0].provenance["record_hash"] == "f" * 64


def test_quarantine_matches_by_fill_id_and_fill_ids():
    """A pin by fill_id catches a derived closed trade referencing it in fill_ids."""
    row = _clean_row("a" * 64, qty=5, pnl=1.0, fill_id="FILL_PINNED_1")
    admitted, counters = adapt_records(
        [(row, "f")], quarantined_fill_ids=frozenset({"FILL_PINNED_1"})
    )
    assert counters["admitted"] == 0
    assert counters["excluded:quarantined"] == 1


def test_is_quarantined_does_not_rederive_pnl_untrusted():
    """The quarantine gate is ONLY the operator pin — an unpinned pnl_untrusted row is
    NOT flagged by is_quarantined (that is trust_exclusion's job, not double-counted)."""
    row = _clean_row("b" * 64, qty=5, pnl=1.0)
    row["payload"]["extra"]["pnl_untrusted"] = True
    assert is_quarantined(row, frozenset(), frozenset()) is False
    assert is_quarantined(row, frozenset({"b" * 64}), frozenset()) is True


# --------------------------------------------------------------------------- #
# load_quarantine_pins — the stdlib manifest reader (record_hash + fill_id union).
# --------------------------------------------------------------------------- #
def test_load_pins_none_runtime_dir_is_empty():
    assert load_quarantine_pins(None) == (frozenset(), frozenset(), [])


def test_load_pins_reads_and_unions_multiple_manifests(tmp_path):
    _write_manifest(tmp_path, "quarantine_manifest_a.json", hashes=_SIX_HASHES[:3])
    _write_manifest(tmp_path, "quarantine_manifest_b.json",
                    hashes=_SIX_HASHES[3:], fill_ids=["FID1"])
    hashes, fill_ids, consulted = load_quarantine_pins(tmp_path)
    assert hashes == frozenset(_SIX_HASHES)
    assert fill_ids == frozenset({"FID1"})
    assert consulted == ["quarantine_manifest_a.json", "quarantine_manifest_b.json"]


def test_load_pins_failsafe_on_corrupt_and_missing(tmp_path):
    (tmp_path / "quarantine_manifest_bad.json").write_text("{not json", encoding="utf-8")
    _write_manifest(tmp_path, "quarantine_manifest_ok.json", hashes=["deadbeef"])
    hashes, fill_ids, consulted = load_quarantine_pins(tmp_path)
    # corrupt file contributes nothing; the good one still loads.
    assert hashes == frozenset({"deadbeef"})
    assert consulted == ["quarantine_manifest_ok.json"]
    # a runtime dir that does not exist -> empty, no raise.
    assert load_quarantine_pins(tmp_path / "nope") == (frozenset(), frozenset(), [])


# --------------------------------------------------------------------------- #
# run_adapter end-to-end: manifest honoured via runtime_dir; None is unchanged.
# --------------------------------------------------------------------------- #
def _write_ledger(trades_dir: Path, rows: list) -> None:
    trades_dir.mkdir(parents=True, exist_ok=True)
    with (trades_dir / "trade_history_20260720.ndjson").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_run_adapter_honours_manifest_end_to_end(tmp_path):
    trades = tmp_path / "data" / "trades"
    runtime = tmp_path / "runtime"
    _write_ledger(trades, _six_rows())
    _write_manifest(runtime, "quarantine_manifest_pff1_ghost_scrub.json", hashes=_SIX_HASHES)

    # Without runtime_dir: all 6 admit (pre-W2A behaviour, byte-identical).
    res_off = run_adapter(trades_dir=trades, generated_at="2026-07-20T00:00:00Z")
    assert res_off.manifest.admitted == 6
    assert res_off.manifest.excluded_by_reason["quarantined"] == 0

    # With runtime_dir: all 6 dropped as quarantined; manifest is listed in notes.
    res_on = run_adapter(
        trades_dir=trades, runtime_dir=runtime, generated_at="2026-07-20T00:00:00Z"
    )
    assert res_on.manifest.admitted == 0
    assert res_on.manifest.excluded_by_reason["quarantined"] == 6
    assert any("quarantine_manifest_pff1_ghost_scrub.json" in n for n in res_on.manifest.notes)


def test_run_adapter_no_manifest_present_notes_checked(tmp_path):
    trades = tmp_path / "data" / "trades"
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    _write_ledger(trades, _six_rows())
    res = run_adapter(trades_dir=trades, runtime_dir=runtime, generated_at="2026-07-20T00:00:00Z")
    assert res.manifest.admitted == 6
    assert res.manifest.excluded_by_reason["quarantined"] == 0
    assert any("no quarantine_manifest" in n for n in res.manifest.notes)
