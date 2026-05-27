"""Unit tests for chad/execution/fill_validation.py and the
quarantine_placeholder_fills tool (HISTORICAL-PLACEHOLDER-1)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from chad.execution.fill_validation import (
    is_trusted_fake_placeholder,
    classify_placeholder,
    DEFAULT_SIGNALS,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fills"


def _load_ndjson(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- predicate level --------------------------------------------------------

def test_14row_fixture_all_rejected():
    rows = _load_ndjson(FIXTURE_DIR / "placeholder_14row_fixture.ndjson")
    assert len(rows) == 14
    flagged = [is_trusted_fake_placeholder(r) for r in rows]
    assert all(flagged), f"expected all 14 rejected; got {flagged}"


def test_legitimate_rows_not_rejected():
    rows = _load_ndjson(FIXTURE_DIR / "legitimate_trades.ndjson")
    for r in rows:
        assert not is_trusted_fake_placeholder(r), r["payload"]


def test_classifier_returns_multiple_signals_for_known_pattern():
    rows = _load_ndjson(FIXTURE_DIR / "placeholder_14row_fixture.ndjson")
    res = classify_placeholder(rows[0])
    assert res.is_placeholder
    # At least the delta_spy_100_pattern and the synthetic_expected_price
    # detectors should fire on this fixture row.
    assert "delta_spy_100_pattern" in res.signals_fired
    assert "synthetic_expected_price" in res.signals_fired


def test_pnl_untrusted_flag_alone_rejects():
    row = {"payload": {"pnl_untrusted": True, "fill_price": 500.0,
                       "symbol": "TSLA", "strategy": "alpha"}}
    assert is_trusted_fake_placeholder(row)


def test_broker_reject_flag_alone_rejects():
    row = {"payload": {"reject": True, "fill_price": 250.0,
                       "symbol": "MSFT", "strategy": "alpha"}}
    assert is_trusted_fake_placeholder(row)


def test_explicit_trusted_fake_marker_in_extra():
    row = {"payload": {"extra": {"trusted_fake": True},
                       "fill_price": 200.0, "symbol": "X", "strategy": "Y"}}
    assert is_trusted_fake_placeholder(row)


def test_status_rejected_is_caught():
    row = {"payload": {"status": "rejected", "fill_price": 100.0,
                       "symbol": "ZZZ", "strategy": "Y"}}
    assert is_trusted_fake_placeholder(row)


def test_placeholder_tag_alone_rejects():
    row = {"payload": {"tags": ["paper", "placeholder"],
                       "fill_price": 500.0, "symbol": "TSLA", "strategy": "alpha"}}
    assert is_trusted_fake_placeholder(row)


def test_signal_subset_can_be_configured():
    """Caller can opt out of the historical-pattern detector if they only
    want flag-based signals (e.g. for forward-only readers)."""
    row = _load_ndjson(FIXTURE_DIR / "placeholder_14row_fixture.ndjson")[0]
    only_flags = ("pnl_untrusted_flag", "broker_reject", "placeholder_marker")
    # The historical fixture lacks pnl_untrusted/reject/placeholder-tag, so
    # the flag-only chain returns False.
    assert not is_trusted_fake_placeholder(row, signals=only_flags)
    # Full chain catches it.
    assert is_trusted_fake_placeholder(row)


def test_unwrapping_payload_handles_both_shapes():
    wrapped = {"payload": {"fill_price": 100.0, "symbol": "SPY", "strategy": "delta",
                           "is_live": False, "order_type": "SIM",
                           "extra": {"expected_price": 100.0}}}
    bare = wrapped["payload"]
    assert is_trusted_fake_placeholder(wrapped)
    assert is_trusted_fake_placeholder(bare)


# --- quarantine tool --------------------------------------------------------

def test_quarantine_tool_check_is_read_only(tmp_path, capsys):
    src = FIXTURE_DIR / "placeholder_14row_fixture.ndjson"
    work = tmp_path / "fixture.ndjson"
    work.write_bytes(src.read_bytes())
    before = hashlib.sha256(work.read_bytes()).hexdigest()
    before_listing = sorted(p.name for p in tmp_path.iterdir())

    from chad.tools import quarantine_placeholder_fills as tool
    rc = tool.main(["--input", str(work), "--check"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["mode"] == "check"
    assert parsed["summary"]["placeholder_rows"] == 14

    after = hashlib.sha256(work.read_bytes()).hexdigest()
    after_listing = sorted(p.name for p in tmp_path.iterdir())
    assert before == after, "input file mutated by --check"
    assert before_listing == after_listing, "tmpdir contents changed during --check"


def test_quarantine_tool_apply_requires_reason(tmp_path, capsys):
    src = FIXTURE_DIR / "placeholder_14row_fixture.ndjson"
    work = tmp_path / "fixture.ndjson"
    work.write_bytes(src.read_bytes())

    from chad.tools import quarantine_placeholder_fills as tool
    rc = tool.main(["--input", str(work), "--apply"])  # no --operator-approve
    assert rc != 0
    err = capsys.readouterr().err
    assert "operator_approve_required" in err


def test_quarantine_tool_apply_with_reason_writes_backup_and_sidecar(tmp_path, capsys):
    src = FIXTURE_DIR / "placeholder_14row_fixture.ndjson"
    work = tmp_path / "fixture.ndjson"
    work.write_bytes(src.read_bytes())

    from chad.tools import quarantine_placeholder_fills as tool
    rc = tool.main([
        "--input", str(work),
        "--apply",
        "--operator-approve", "test-operator-reason",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    backup = Path(out["applied"]["backup_path"])
    sidecar = Path(out["applied"]["sidecar_path"])
    assert backup.is_file()
    assert sidecar.is_file()
    side_doc = json.loads(sidecar.read_text())
    assert side_doc["placeholder_rows"] == 14
    assert side_doc["operator_reason"] == "test-operator-reason"
    # Original file must be unchanged (defense-in-depth).
    assert work.read_bytes() == src.read_bytes()


def test_quarantine_tool_missing_input_returns_user_error(tmp_path, capsys):
    from chad.tools import quarantine_placeholder_fills as tool
    rc = tool.main(["--input", str(tmp_path / "nonexistent.ndjson"), "--check"])
    assert rc != 0


# --- integration with existing readers --------------------------------------

def test_validator_excludes_placeholders_when_used_by_a_reader(tmp_path):
    """Mimics the contract a future SCR/PnL/replay/report reader should
    follow: filter via is_trusted_fake_placeholder before aggregating."""
    rows = _load_ndjson(FIXTURE_DIR / "placeholder_14row_fixture.ndjson")
    rows += _load_ndjson(FIXTURE_DIR / "legitimate_trades.ndjson")
    kept = [r for r in rows if not is_trusted_fake_placeholder(r)]
    # 14 placeholders removed; 2 legitimate kept.
    assert len(kept) == 2
    syms = {r["payload"]["symbol"] for r in kept}
    assert syms == {"SPY", "AAPL"}
