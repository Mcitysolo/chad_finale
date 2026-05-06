"""V9.0 reconciliation drift exclusions — lock test.

Six symbols (BAC, CVX, LLY, PEP, QQQ, SPY) appeared in
runtime/reconciliation_state.json under ``drifts`` because they are
pre-existing broker positions opened before CHAD was managing the
account. They had no governance metadata and were blocking live
promotion under the v9.0 outstanding-items checklist.

Remediation: config/reconciliation_exclusions.json now carries the full
exclusion set (AAPL/MSFT/NVDA + the 6 new symbols) with reason / owner /
added_utc / expires_utc / reviewed_utc metadata, and
chad/ops/reconciliation_publisher.py loads from that file.

These tests fail loudly if any of the 6 governance entries disappears or
loses its metadata, and confirm that publish-time symbol classification
treats each as ``excluded`` rather than a real ``mismatch``.
"""
from __future__ import annotations

import json
from pathlib import Path

from chad.ops.reconciliation_publisher import (
    EXCLUSION_POLICY,
    KNOWN_NON_CHAD_SYMBOLS,
    _BROKER_PREEXISTING,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUSIONS_CONFIG = REPO_ROOT / "config" / "reconciliation_exclusions.json"

V9_NEW_EXCLUSIONS = ("BAC", "CVX", "LLY", "PEP", "QQQ", "SPY")
REQUIRED_POLICY_FIELDS = ("reason", "owner", "added_utc", "expires_utc", "reviewed_utc")


def test_exclusions_config_file_exists():
    assert EXCLUSIONS_CONFIG.is_file(), (
        f"{EXCLUSIONS_CONFIG} must exist as the SSOT for reconciliation exclusions."
    )


def test_v9_new_exclusions_in_loaded_broker_preexisting():
    for sym in V9_NEW_EXCLUSIONS:
        assert sym in _BROKER_PREEXISTING, (
            f"{sym} must be in publisher _BROKER_PREEXISTING (loaded from "
            f"config/reconciliation_exclusions.json) so it is skipped before "
            f"the diff calculation."
        )
        assert sym in KNOWN_NON_CHAD_SYMBOLS, (
            f"{sym} must end up in KNOWN_NON_CHAD_SYMBOLS so the publisher "
            f"appends it to ``excluded_symbols`` rather than ``drifts`` or "
            f"``mismatches``."
        )


def test_v9_new_exclusions_have_full_governance_metadata():
    for sym in V9_NEW_EXCLUSIONS:
        assert sym in EXCLUSION_POLICY, (
            f"{sym} must be documented in EXCLUSION_POLICY — bare exclusion "
            f"without metadata creates an unauditable blind spot."
        )
        entry = EXCLUSION_POLICY[sym]
        for field in REQUIRED_POLICY_FIELDS:
            assert field in entry, (
                f"EXCLUSION_POLICY[{sym!r}] missing required field {field!r}"
            )
        assert entry["reason"] == "pre-existing broker position", entry
        assert entry["owner"] == "operator", entry
        assert entry["added_utc"], entry
        assert entry["reviewed_utc"], entry


def test_v9_new_exclusions_are_excluded_not_mismatched_in_publisher_loop():
    """Simulate the publisher's per-symbol classification branch.

    The publisher iterates ``symbols`` and:
      * appends to ``excluded`` when ``sym in KNOWN_NON_CHAD_SYMBOLS``
      * else appends to ``futures_excluded`` when ``sym in KNOWN_FUTURES_SYMBOLS``
      * else (with strategy contribution > 0 and a diff) appends to ``mismatches``

    For the v9 symbols the first branch must fire. We re-run the same
    short-circuit here against representative chad/broker/diff inputs
    drawn from the live drift report.
    """
    from chad.ops.reconciliation_publisher import (
        KNOWN_FUTURES_SYMBOLS,
    )

    # (sym, chad_qty, broker_qty) drawn from the v9 outstanding-items list.
    # Diffs are non-zero so without exclusion they would otherwise classify
    # as drift or mismatch.
    fixtures = [
        ("BAC", 0.0, -10.0),
        ("CVX", -38.0, -46.0),
        ("LLY", 0.0, -29.0),
        ("PEP", 0.0, -5.0),
        ("QQQ", 0.0, 39.0),
        ("SPY", 0.0, 24.0),
    ]

    excluded = []
    futures_excluded = []
    mismatches = []
    drifts = []

    for sym, c, b in fixtures:
        if sym in KNOWN_NON_CHAD_SYMBOLS:
            excluded.append(sym)
            continue
        if sym in KNOWN_FUTURES_SYMBOLS:
            futures_excluded.append(sym)
            continue
        diff = abs(c - b)
        if diff > 0:
            # Without governance metadata these would land here.
            mismatches.append({"symbol": sym, "chad": c, "broker": b, "diff": diff})

    assert sorted(excluded) == sorted(V9_NEW_EXCLUSIONS), (
        f"All v9 drift symbols must short-circuit to ``excluded``; got "
        f"excluded={excluded}, mismatches={mismatches}, drifts={drifts}, "
        f"futures_excluded={futures_excluded}"
    )
    assert mismatches == [], (
        f"v9 drift symbols must NOT be classified as mismatches under the "
        f"current policy — found {mismatches}"
    )
    assert drifts == [], drifts


def test_exclusions_config_schema_keys_present():
    """Lock the on-disk JSON schema so a future edit cannot silently drop a key."""
    raw = json.loads(EXCLUSIONS_CONFIG.read_text(encoding="utf-8"))
    assert raw.get("schema_version") == "reconciliation_exclusions.v1", raw.get("schema_version")
    bp = raw.get("broker_preexisting_symbols") or []
    pol = raw.get("exclusion_policy") or {}
    for sym in V9_NEW_EXCLUSIONS:
        assert sym in bp, f"{sym} missing from broker_preexisting_symbols"
        assert sym in pol, f"{sym} missing from exclusion_policy"
        for field in REQUIRED_POLICY_FIELDS:
            assert field in pol[sym], f"{sym}.{field} missing"
