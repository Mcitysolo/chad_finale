"""Phase B Item 5 — futures roll calendar tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.market_data.futures_roll_publisher import (
    DEFAULT_TTL_SECONDS,
    ROLL_CRITICAL_DAYS,
    ROLL_WARNING_DAYS,
    SCHEMA_VERSION,
    build_payload,
    build_symbol_record,
    next_quarterly_expiry,
    publish,
    third_friday,
)
from chad.utils.roll_gate import ROLL_FILE_TTL, RollGateResult, check_roll_gate


# ---------------------------------------------------------------------------
# Calendar math
# ---------------------------------------------------------------------------


def test_third_friday_march_2026() -> None:
    assert third_friday(2026, 3) == date(2026, 3, 20)


def test_third_friday_june_2026() -> None:
    assert third_friday(2026, 6) == date(2026, 6, 19)


def test_next_quarterly_expiry_before_june() -> None:
    assert next_quarterly_expiry(date(2026, 5, 15)) == date(2026, 6, 19)


# ---------------------------------------------------------------------------
# Payload + record shape
# ---------------------------------------------------------------------------


_EXPECTED_DEFAULT_SYMBOLS = {
    "MES", "MNQ", "MCL", "MGC", "MYM", "M2K", "M6E", "ZN", "ZB",
}


def test_build_payload_contains_default_symbols() -> None:
    payload = build_payload(today=date(2026, 5, 15))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["ttl_seconds"] == DEFAULT_TTL_SECONDS
    assert set(payload["symbols"].keys()) == _EXPECTED_DEFAULT_SYMBOLS


def test_mes_supported_quarterly_record() -> None:
    rec = build_symbol_record("MES", date(2026, 5, 15))
    assert rec["roll_supported"] is True
    assert rec["roll_pattern"] == "quarterly_3rd_friday"
    assert rec["current_expiry"] == "2026-06-19"
    assert rec["next_expiry"] == "2026-09-18"
    assert rec["days_to_expiry"] == 35
    assert rec["roll_warning"] is False
    assert rec["block_new_entries"] is False


def test_mcl_unsupported_record() -> None:
    rec = build_symbol_record("MCL", date(2026, 5, 15))
    assert rec["roll_supported"] is False
    assert rec["roll_pattern"] == "unsupported_v1"
    assert rec["block_new_entries"] is False
    assert rec["current_expiry"] is None
    assert rec["days_to_expiry"] is None


def test_roll_warning_true_when_within_window() -> None:
    # 4 days before June 19 expiry => warning True, critical False
    rec = build_symbol_record("MES", date(2026, 6, 15))
    assert rec["days_to_expiry"] == 4
    assert rec["roll_warning"] is True
    assert rec["roll_critical"] is False
    assert rec["block_new_entries"] is True


def test_roll_warning_false_at_ten_days() -> None:
    rec = build_symbol_record("MES", date(2026, 6, 9))
    assert rec["days_to_expiry"] == 10
    assert rec["roll_warning"] is False
    assert rec["block_new_entries"] is False


def test_roll_critical_true_when_within_two_days() -> None:
    rec = build_symbol_record("MES", date(2026, 6, 18))
    assert rec["days_to_expiry"] == 1
    assert rec["roll_warning"] is True
    assert rec["roll_critical"] is True
    assert rec["block_new_entries"] is True


def test_block_new_entries_only_when_supported_and_warning() -> None:
    # MGC (unsupported_v1) — even 0 days to anything must not block
    rec_unsup = build_symbol_record("MGC", date(2026, 6, 19))
    assert rec_unsup["block_new_entries"] is False
    # MES inside window — must block
    rec_sup = build_symbol_record("MES", date(2026, 6, 17))
    assert rec_sup["roll_supported"] is True
    assert rec_sup["roll_warning"] is True
    assert rec_sup["block_new_entries"] is True


# ---------------------------------------------------------------------------
# Roll gate fail-open semantics
# ---------------------------------------------------------------------------


def test_check_roll_gate_missing_file_fails_open(tmp_path: Path) -> None:
    result = check_roll_gate("MES", runtime_dir=tmp_path)
    assert isinstance(result, RollGateResult)
    assert result.blocked is False
    assert result.block_reason is None
    assert result.roll_supported is False


def test_check_roll_gate_stale_file_fails_open(tmp_path: Path) -> None:
    stale_ts = (
        datetime.now(timezone.utc) - timedelta(seconds=ROLL_FILE_TTL + 3600)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = build_payload(today=date(2026, 6, 17))
    payload["ts_utc"] = stale_ts
    (tmp_path / "futures_roll_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = check_roll_gate("MES", runtime_dir=tmp_path)
    assert result.blocked is False
    assert result.block_reason is None


def test_check_roll_gate_blocks_supported_symbol_in_window(tmp_path: Path) -> None:
    payload = build_payload(today=date(2026, 6, 17))
    # Fresh ts_utc so file is not stale
    payload["ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "futures_roll_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = check_roll_gate("MES", runtime_dir=tmp_path)
    assert result.blocked is True
    assert result.block_reason == "ROLL_WARNING_WINDOW"
    assert result.roll_supported is True
    assert result.roll_pattern == "quarterly_3rd_friday"
    assert result.days_to_expiry == 2


def test_check_roll_gate_unsupported_symbol_does_not_block(tmp_path: Path) -> None:
    payload = build_payload(today=date(2026, 6, 17))
    payload["ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "futures_roll_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = check_roll_gate("MCL", runtime_dir=tmp_path)
    assert result.blocked is False
    assert result.roll_supported is False
    assert result.block_reason is None


def test_check_roll_gate_supported_outside_window_does_not_block(
    tmp_path: Path,
) -> None:
    payload = build_payload(today=date(2026, 5, 15))
    payload["ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "futures_roll_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = check_roll_gate("MES", runtime_dir=tmp_path)
    assert result.blocked is False
    assert result.roll_supported is True
    assert result.block_reason is None
    assert result.days_to_expiry == 35


# ---------------------------------------------------------------------------
# alpha_futures wiring + deploy files
# ---------------------------------------------------------------------------


def test_alpha_futures_source_contains_roll_meta_keys() -> None:
    src = Path(
        "/home/ubuntu/chad_finale/chad/strategies/alpha_futures.py"
    ).read_text(encoding="utf-8")
    # Import present
    assert "from chad.utils.roll_gate import RollGateResult, check_roll_gate" in src
    # Entry-only gate call present and uses check_roll_gate
    assert "check_roll_gate(symbol)" in src
    # Meta keys present
    assert '"days_to_expiry": _roll.days_to_expiry' in src
    assert '"roll_pattern": _roll.roll_pattern' in src
    assert '"roll_supported": _roll.roll_supported' in src
    # Roll gate must sit after R:R gate and before the sizing CALL — the
    # sizing function is defined far above; locate its call site by looking
    # for the keyword argument used at the call.
    rr_idx = src.index("passes_rr_gate(_target_pts, _stop_pts)")
    roll_idx = src.index("check_roll_gate(symbol)")
    sizing_call_idx = src.index("contracts, alloc_wt, risk_budget_usd, risk_per_contract_usd = _compute_contract_size(")
    assert rr_idx < roll_idx < sizing_call_idx, (
        "Roll gate must be inserted between the R:R gate and the sizing call."
    )
    # And the gate must sit before TradeSignal construction on the entry
    # path. The exit path also returns a TradeSignal far above; use the
    # entry-path construction whose `meta=_entry_meta` keyword is unique.
    entry_signal_idx = src.index("meta=_entry_meta")
    assert roll_idx < entry_signal_idx


def test_deploy_service_and_timer_files_exist() -> None:
    svc = Path(
        "/home/ubuntu/chad_finale/deploy/chad-futures-roll-refresh.service"
    )
    tmr = Path(
        "/home/ubuntu/chad_finale/deploy/chad-futures-roll-refresh.timer"
    )
    assert svc.is_file()
    assert tmr.is_file()
    svc_txt = svc.read_text(encoding="utf-8")
    tmr_txt = tmr.read_text(encoding="utf-8")
    assert "ExecStart=" in svc_txt
    assert "futures_roll_publisher" in svc_txt
    assert "OnUnitActiveSec=86400" in tmr_txt


def test_dry_run_payload_schema(tmp_path: Path) -> None:
    payload, wrote = publish(runtime_dir=tmp_path, dry_run=True, today=date(2026, 5, 15))
    assert wrote is False
    # No file written
    assert not (tmp_path / "futures_roll_state.json").exists()
    # Schema invariants
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] == "ok"
    assert payload["source"]["provider"] == "static_cme_calendar"
    assert payload["source"]["provider_status"] == "real"
    summary = payload["summary"]
    assert summary["symbols_tracked"] == 9
    assert summary["supported_count"] == 4
    assert summary["unsupported_count"] == 5
    assert summary["roll_warning_count"] == 0
    assert summary["roll_critical_count"] == 0
    assert summary["blocked_symbols"] == []


# Sanity check on module-level constants (helps catch accidental edits)
def test_warning_and_critical_thresholds() -> None:
    assert ROLL_WARNING_DAYS == 5
    assert ROLL_CRITICAL_DAYS == 2
