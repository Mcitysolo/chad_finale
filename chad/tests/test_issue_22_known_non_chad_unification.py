"""ISSUE-22 regression guard.

Original defect (commit 92b48df message + reports/pre_step_13_restart_plan_20260420.json):
divergent ``KNOWN_NON_CHAD_SYMBOLS`` hardcoded sets between
``chad-reconciliation`` (``ops/reconcile_positions.py``) and
``chad-reconciliation-publisher`` (``chad/ops/reconciliation_publisher.py``)
caused a dual-writer race on ``runtime/reconciliation_state.json`` whenever
both timers were active, with last-writer-wins flipping the dashboard
status because the two services excluded different symbol sets.

Remediation already in tree:
* ``chad/core/position_reconciler.py`` is the canonical source.
* ``chad/ops/reconciliation_publisher.py`` imports the canonical set and
  unions it with an explicit, policy-tracked publisher-only augmentation.
* ``ops/reconcile_positions.py`` carries no exclusion list; the
  ``chad-reconciliation`` service was contained, eliminating the dual
  writer.

These tests fail loudly if a future change reintroduces the divergence.
"""
from __future__ import annotations

from pathlib import Path

from chad.core.position_reconciler import KNOWN_NON_CHAD_SYMBOLS as RECONCILER_SET
from chad.ops.reconciliation_publisher import (
    EXCLUSION_POLICY,
    KNOWN_NON_CHAD_SYMBOLS as PUBLISHER_SET,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECONCILE_PRODUCER = REPO_ROOT / "ops" / "reconcile_positions.py"


def test_publisher_inherits_canonical_non_chad_set():
    assert RECONCILER_SET <= PUBLISHER_SET, (
        "reconciliation_publisher.KNOWN_NON_CHAD_SYMBOLS must be a superset "
        "of position_reconciler.KNOWN_NON_CHAD_SYMBOLS — divergence would "
        "reintroduce ISSUE-22."
    )


def test_publisher_only_extras_are_policy_tracked():
    publisher_only = PUBLISHER_SET - RECONCILER_SET
    missing_policy = sorted(sym for sym in publisher_only if sym not in EXCLUSION_POLICY)
    assert not missing_policy, (
        "Publisher-only exclusions must be documented in EXCLUSION_POLICY: "
        f"{missing_policy}"
    )


def test_legacy_producer_has_no_separate_exclusion_list():
    src = RECONCILE_PRODUCER.read_text(encoding="utf-8")
    assert "KNOWN_NON_CHAD_SYMBOLS" not in src, (
        "ops/reconcile_positions.py must not redeclare KNOWN_NON_CHAD_SYMBOLS; "
        "the publisher's canonical import is the single source of truth."
    )
