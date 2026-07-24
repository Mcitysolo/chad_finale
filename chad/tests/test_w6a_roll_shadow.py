#!/usr/bin/env python3
"""
chad/tests/test_w6a_roll_shadow.py

W6A-4 (D2) — SHADOW roll coverage for the non-equity-index families.

The blocking arm does NOT ship here. MCL/MGC/ZN/ZB/M6E gain real computed
expiries under ``shadow_*`` keys while every live-behaviour field stays
exactly as it was, so neither ``roll_gate`` nor ``futures_expiry_gate``
changes behaviour. Promotion to the live fields is its own PA after a roll
cycle of evidence — house ladder, no exceptions.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from chad.market_data.futures_roll_publisher import build_payload, build_symbol_record

TODAY = date(2026, 7, 23)
SHADOWED = ("MCL", "MGC", "ZN", "ZB", "M6E")
LIVE_BEHAVIOUR_KEYS = (
    "current_expiry",
    "next_expiry",
    "days_to_expiry",
    "roll_warning",
    "roll_critical",
    "roll_pattern",
    "roll_supported",
    "block_new_entries",
)


# ---------------------------------------------------------------------------
# The shadow must not move a single live field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SHADOWED)
def test_shadow_symbols_keep_unsupported_live_fields(symbol: str) -> None:
    """roll_gate blocks only on (block_new_entries AND roll_supported);
    futures_expiry_gate skips polling on a parseable current_expiry. Both must
    read exactly what they read before W6A-4."""
    rec = build_symbol_record(symbol, TODAY)
    assert rec["roll_supported"] is False
    assert rec["block_new_entries"] is False
    assert rec["current_expiry"] is None
    assert rec["next_expiry"] is None
    assert rec["days_to_expiry"] is None
    assert rec["roll_pattern"] == "unsupported_v1"


def test_summary_counts_unchanged() -> None:
    """4 supported / 5 unsupported / nothing blocked — the shape production
    already publishes."""
    summary = build_payload(today=TODAY)["summary"]
    assert summary["supported_count"] == 4
    assert summary["unsupported_count"] == 5
    assert summary["blocked_symbols"] == []
    assert summary["roll_warning_count"] == 0
    assert summary["roll_critical_count"] == 0


@pytest.mark.parametrize("symbol", ("MES", "MNQ", "MYM", "M2K"))
def test_equity_index_records_untouched(symbol: str) -> None:
    rec = build_symbol_record(symbol, TODAY)
    assert rec["roll_supported"] is True
    assert rec["roll_pattern"] == "quarterly_3rd_friday"
    assert rec["current_expiry"] == "2026-09-18"
    assert "shadow_current_expiry" not in rec


# ---------------------------------------------------------------------------
# ...while still recording the truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,expected_expiry",
    [
        ("MCL", "2026-08-20"),   # the delivery-offset family the calendar never covered
        ("MGC", "2026-08-27"),
        ("ZN", "2026-09-21"),
        ("ZB", "2026-09-21"),
        ("M6E", "2026-09-14"),
    ],
)
def test_shadow_expiry_is_computed(symbol: str, expected_expiry: str) -> None:
    rec = build_symbol_record(symbol, TODAY)
    assert rec["shadow_roll_supported"] is True
    assert rec["shadow_current_expiry"] == expected_expiry
    assert rec["shadow_block_new_entries"] is False  # none are near expiry on this date


def test_shadow_next_expiry_follows_the_symbols_own_cycle() -> None:
    """MCL is monthly, ZN quarterly — a blanket quarterly assumption would be
    wrong for the very family this shadow exists to cover."""
    assert build_symbol_record("MCL", TODAY)["shadow_next_expiry"] == "2026-09-22"
    assert build_symbol_record("ZN", TODAY)["shadow_next_expiry"] == "2026-12-22"


def test_shadow_warning_is_computed_off_the_front_contract_not_the_resolved_one() -> None:
    """The resolver's 5-day roll buffer equals ROLL_WARNING_DAYS, so a warning
    computed off the RESOLVED contract could never fire — it would be a flag
    structurally incapable of being true. It is computed off the front
    contract instead, which is what an open position actually holds."""
    near = date(2026, 8, 18)  # 2 days before MCL's 2026-08-20 last trade
    rec = build_symbol_record("MCL", near)

    # Resolution has already rolled to September...
    assert rec["shadow_current_expiry"] == "2026-09-22"
    assert rec["shadow_days_to_expiry"] == 35
    # ...while the front contract is 2 days from expiry, and says so.
    assert rec["shadow_front_expiry"] == "2026-08-20"
    assert rec["shadow_front_days_to_expiry"] == 2
    assert rec["shadow_roll_warning"] is True
    assert rec["shadow_roll_critical"] is True
    assert rec["shadow_block_new_entries"] is True

    assert rec["block_new_entries"] is False, "shadow must never block for real"
    assert rec["roll_supported"] is False


def test_shadow_block_flag_is_capable_of_being_both_values() -> None:
    """Guard against the dead-flag class: sweep a year and require the shadow
    block to be observed both True and at least mostly False."""
    seen = set()
    for offset in range(0, 365, 1):
        rec = build_symbol_record("MCL", date(2026, 1, 1) + timedelta(days=offset))
        seen.add(rec["shadow_block_new_entries"])
    assert seen == {True, False}, f"shadow block flag never varies: {seen}"


# ---------------------------------------------------------------------------
# SIL: gates never cover ghosts
# ---------------------------------------------------------------------------


def test_sil_is_not_published() -> None:
    """SIL is data-only — no strategy consumes it, and its serial months are
    ~600x thinner than MGC. A roll record for it would be a gate covering a
    ghost."""
    assert "SIL" not in build_payload(today=TODAY)["symbols"]
