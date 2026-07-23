"""W5A-6 — EXS7 pins the ledger-embedded sub-schemas (amended-R2 condition b).

The measurement blocks ride inside closed_trade.v1 rows (no top-level bump), so
EXS7 validates them against the newest ledger row: absent ⇒ OK (optional,
flag-gated), present-with-wrong-shape ⇒ break. Also pins that the WRITERS'
output matches the declared required_keys (so a config/writer drift is caught).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from chad.ops.exterminator_sentinel import ExterminatorSentinel

NOW = datetime(2026, 7, 23, 18, 0, 0, tzinfo=timezone.utc)
REAL_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[2] / "config" / "exterminator.json"


def _make(tmp_path):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    return ExterminatorSentinel(
        repo_root=tmp_path, runtime_dir=tmp_path / "runtime",
        data_dir=tmp_path / "data", config_path=REAL_CONFIG,
        reports_dir=tmp_path / "runtime" / "reports", clock=lambda: NOW,
    )


def _ledger(tmp_path, payload):
    d = tmp_path / "data" / "trades"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trade_history_20260723.ndjson").write_text(
        json.dumps({"payload": payload, "record_hash": "rh"}) + "\n")


def _base_payload(**extra):
    p = {"schema_version": "closed_trade.v1", "strategy": "gamma", "symbol": "PSQ",
         "pnl": 1.0}
    p.update(extra)
    return p


# --------------------------------------------------------------------------- #
# Config declares both sub-schemas
# --------------------------------------------------------------------------- #

def test_config_pins_both_subschemas():
    cfg = json.loads(REAL_CONFIG.read_text())
    embedded = cfg["schema_contracts"]["embedded"]
    assert "implementation_shortfall.v1" in embedded
    assert "mae_mfe.v1" in embedded
    assert embedded["implementation_shortfall.v1"]["field"] == "implementation_shortfall"
    assert embedded["mae_mfe.v1"]["field"] == "mae_mfe"


# --------------------------------------------------------------------------- #
# EXS7 validation of the embedded blocks
# --------------------------------------------------------------------------- #

def test_absent_blocks_ok(tmp_path):
    _ledger(tmp_path, _base_payload())  # no measurement blocks (flag off)
    r = _make(tmp_path).check_schema_breaks()
    # File-keyed contracts may fail on this empty tmp runtime — we only assert
    # the EMBEDDED sub-schemas produce NO break when the blocks are absent.
    assert not any("closed_trade." in str(b.get("file", "")) for b in r.evidence["breaks"])
    assert r.evidence["embedded_contracts"] == 2


def test_present_valid_block_ok(tmp_path):
    from chad.analytics.implementation_shortfall import compute_lap_is
    is_block = compute_lap_is(fill_ids=["a"], quantity=1.0, contract_multiplier=1.0,
                              broker="paper_exec", stop_width_usd=None, index={})
    _ledger(tmp_path, _base_payload(implementation_shortfall=is_block))
    r = _make(tmp_path).check_schema_breaks()
    assert not any(b.get("file") == "closed_trade.implementation_shortfall"
                   for b in r.evidence["breaks"])
    assert "implementation_shortfall.v1" in r.evidence["embedded_checked"]


def test_malformed_block_is_break(tmp_path):
    _ledger(tmp_path, _base_payload(implementation_shortfall={
        "schema_version": "implementation_shortfall.v1", "is_usd": 1.0}))  # missing keys
    r = _make(tmp_path).check_schema_breaks()
    assert r.status == "fail"
    b = next(b for b in r.evidence["breaks"]
             if b.get("file") == "closed_trade.implementation_shortfall")
    assert b["break"] == "required_keys_missing"


def test_wrong_subschema_version_is_break(tmp_path):
    _ledger(tmp_path, _base_payload(mae_mfe={
        "schema_version": "mae_mfe.v2", "mae_pct": None, "mae_usd": None,
        "mfe_pct": None, "mfe_usd": None, "excursion_source": "x"}))
    r = _make(tmp_path).check_schema_breaks()
    b = next(b for b in r.evidence["breaks"] if b.get("file") == "closed_trade.mae_mfe")
    assert b["break"] == "schema_version_unrecognised"


# --------------------------------------------------------------------------- #
# Writer output honors the declared required_keys (drift guard)
# --------------------------------------------------------------------------- #

def test_is_writer_matches_contract():
    from chad.analytics.implementation_shortfall import compute_lap_is
    cfg = json.loads(REAL_CONFIG.read_text())
    req = cfg["schema_contracts"]["embedded"]["implementation_shortfall.v1"]["required_keys"]
    block = compute_lap_is(fill_ids=["a"], quantity=1.0, contract_multiplier=1.0,
                           broker="paper_exec", stop_width_usd=None, index={})
    assert set(req) <= set(block), f"writer missing declared keys: {set(req) - set(block)}"


def test_mae_mfe_writer_matches_contract():
    from chad.analytics.excursion_recorder import read_lap_excursion
    import chad.analytics.excursion_recorder as ex
    from pathlib import Path
    import json as _json
    import tempfile

    cfg = json.loads(REAL_CONFIG.read_text())
    req = cfg["schema_contracts"]["embedded"]["mae_mfe.v1"]["required_keys"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "excursion_20260723.ndjson"
        p.write_text(_json.dumps({
            "schema_version": "mae_mfe.v1", "strategy": "gamma", "symbol": "PSQ",
            "opened_at_utc": "2026-07-23T10:00:00Z", "closed_detect_utc": "2026-07-23T16:00:00Z",
            "mae_pct": -0.05, "mae_usd": -50.0, "mfe_pct": 0.1, "mfe_usd": 100.0,
            "excursion_source": "watermark_bar_hilo"}) + "\n")
        ex._EXCURSION_CACHE.clear()
        block = read_lap_excursion(strategy="gamma", symbol="PSQ",
                                   entry_time_utc="2026-07-23T10:00:00Z",
                                   exit_time_utc="2026-07-23T16:00:00Z", evidence_dir=Path(d))
    assert set(req) <= set(block), f"writer missing declared keys: {set(req) - set(block)}"
