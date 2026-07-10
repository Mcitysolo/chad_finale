"""G3C-HF regression: the margin_shadow test-leak class is now blocked.

Covers both structural fixes for the 2026-07-10 incident where a fixture using a fixed
future epoch (NOW=1_800_000_000 → ts_utc 2027-01-15) wrote through the REAL evidence path,
creating data/margin_shadow/margin_shadow_20270115.ndjson:

  * fix 2(a) — build_default_shadow_gate() refuses to compose the real evidence path under
    pytest unless evidence_dir is explicit (fails loudly instead of silently leaking);
  * fix 2(b) — the conftest repo-write guard blocks ANY create/modify under the repo's
    data/ | runtime/ tree (defense-in-depth for the whole leak class), while tmp_path
    writes are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chad.execution.margin_shadow_gate import (
    build_default_shadow_gate,
    default_evidence_dir,
    order_view_from_intent,
)
from chad.testing.repo_write_guard import RepoWriteLeakError, expect_blocked

REPO = Path("/home/ubuntu/chad_finale")
NOW = 1_800_000_000.0  # the exact leak epoch → 2027-01-15T08:00:00Z


class _OV:
    """The minimal intent duck-type the original offending fixture used."""

    symbol = "AAPL"
    side = "BUY"
    asset_class = "equity"
    quantity = 10.0
    currency = "USD"
    notional_estimate = 1900.0
    limit_price = 190.0
    meta: dict = {}
    strategy = "alpha"


# --------------------------------------------------------------------------- #
# fix 2(a): the factory is REQUIRED-explicit about evidence_dir under pytest.
# --------------------------------------------------------------------------- #
def test_build_default_gate_without_evidence_dir_raises_under_pytest():
    """The exact offending construction — build_default_shadow_gate(repo_root=REPO) with no
    evidence_dir — now FAILS LOUDLY under pytest instead of composing the real path."""
    with pytest.raises(RuntimeError, match="evidence_dir is REQUIRED-explicit"):
        build_default_shadow_gate(repo_root=REPO)


def test_original_leak_scenario_now_lands_in_tmp_only(tmp_path):
    """Re-run the exact offending evaluate() but with evidence_dir injected at tmp_path: the
    single row lands under tmp (named for the 2027-01-15 epoch), and the real
    data/margin_shadow tree is never touched (the conftest guard would have failed us)."""
    g = build_default_shadow_gate(repo_root=REPO, evidence_dir=tmp_path / "ev")
    assert g is not None
    v = g.evaluate(order_view_from_intent(_OV(), order_id="k1"), now_epoch=NOW)
    assert v.reason == "STALE_OR_MISSING_MARGIN_DATA"
    files = list((tmp_path / "ev").glob("margin_shadow_*.ndjson"))
    assert len(files) == 1
    assert files[0].name == "margin_shadow_20270115.ndjson"
    # And nothing was written to the real evidence tree by this test.
    assert not (default_evidence_dir(REPO) / "margin_shadow_20270115.ndjson").exists()


# --------------------------------------------------------------------------- #
# fix 2(b): the conftest guard blocks the whole leak class.
# --------------------------------------------------------------------------- #
def test_guard_blocks_direct_write_under_data():
    target = REPO / "data" / "margin_shadow" / "__guard_probe_should_not_exist__.ndjson"
    with expect_blocked():
        with open(target, "a", encoding="utf-8") as fh:  # exact primitive the leak used
            fh.write("nope\n")
    assert not target.exists()


def test_guard_blocks_write_under_runtime():
    target = REPO / "runtime" / "__guard_probe_should_not_exist__.json"
    with expect_blocked():
        target.write_text("{}", encoding="utf-8")  # pathlib → io.open, also guarded
    assert not target.exists()


def test_guard_blocks_mkdir_under_data():
    target = REPO / "data" / "__guard_probe_dir_should_not_exist__"
    with expect_blocked():
        target.mkdir(parents=True, exist_ok=True)
    assert not target.exists()


def test_guard_reraises_even_when_caller_swallows(tmp_path):
    """A best-effort writer that swallows the block (like _safe_write_evidence) still leaves a
    recorded attempt — the per-test teardown check is what fails such a test. Here we assert
    the attempt is recorded (then consume it so THIS test still passes)."""
    from chad.testing import repo_write_guard as guard

    target = REPO / "runtime" / "__guard_swallow_probe__.json"
    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("x")
    except RepoWriteLeakError:
        pass  # swallow, exactly like a best-effort writer
    attempts = guard.take_attempts()  # consume so teardown sees a clean slate
    assert any("__guard_swallow_probe__" in p for p, _m, _prim in attempts)
    assert not target.exists()


# --------------------------------------------------------------------------- #
# The guard must NOT interfere with legitimate tmp_path writes.
# --------------------------------------------------------------------------- #
def test_guard_allows_tmp_path_writes(tmp_path):
    p = tmp_path / "fine.txt"
    p.write_text("ok", encoding="utf-8")
    with open(tmp_path / "also_fine.ndjson", "a", encoding="utf-8") as fh:
        fh.write("{}\n")
    assert p.read_text(encoding="utf-8") == "ok"


def test_guard_allows_noop_mkdir_on_existing_guarded_dir():
    """Regression: mkdir(exist_ok=True) on an already-existing guarded dir (the ubiquitous
    `parent.mkdir(parents=True, exist_ok=True)` on runtime/) creates/modifies nothing and must
    NOT be flagged — else it storms the whole suite with false positives (G3C-HF first pass)."""
    existing = REPO / "runtime"
    assert existing.exists()
    existing.mkdir(parents=True, exist_ok=True)  # no-op; would raise RepoWriteLeakError if buggy


def test_baseline_blocks_all_but_records_only_new_sinks():
    """Ratchet semantics: EVERY guarded write is blocked (never hits disk — live-safe). A
    grandfathered pre-existing sink is blocked-but-NOT-recorded (best-effort writers pass);
    data/margin_shadow (the incident) and any NEW sink are blocked AND recorded (fail the
    test). Pure-logic assertions — no filesystem writes."""
    from chad.testing import repo_write_guard as guard

    def _resolved(rel):
        return guard._resolve(REPO / rel)

    # Incident + brand-new sinks: guarded and NOT baselined → recorded → fail.
    for rel in ("data/margin_shadow/margin_shadow_20270115.ndjson",
                "data/some_new_evidence/x.ndjson", "runtime/some_new_state.json"):
        assert guard._is_guarded(REPO / rel) is not None, rel
        assert guard._is_baselined(_resolved(rel)) is False, rel

    # Grandfathered pre-existing sinks: still guarded (blocked) but baselined → not recorded.
    for rel in (
        "runtime/claude_usage.json",
        "runtime/execution_environment.json.tmp.123",       # atomic-write temp sibling
        "runtime/locks/FILLS_20260710.ndjson.lock",         # under a grandfathered dir
        "data/traces/decision_trace_20260710.ndjson",
        "data/slippage/SLIPPAGE_20260710.ndjson",
        "data/signal_decay/alpha_crypto_decay.ndjson",
    ):
        assert guard._is_guarded(REPO / rel) is not None, rel
        assert guard._is_baselined(_resolved(rel)) is True, rel


def test_grandfathered_write_is_blocked_but_not_recorded():
    """Behavioral: a write to a grandfathered runtime STATE file is BLOCKED (never reaches disk,
    so the live service's runtime/ is untouched) yet records nothing (so best-effort writers that
    swallow the block keep the test green)."""
    from chad.testing import repo_write_guard as guard

    target = REPO / "runtime" / "claude_usage.json"  # grandfathered
    with pytest.raises(guard.RepoWriteLeakError):
        with open(target, "w", encoding="utf-8"):  # blocked BEFORE truncation → file intact
            pass
    assert guard.take_attempts() == []  # baselined → nothing recorded → no teardown failure


def test_guard_allows_reads_under_data(tmp_path):
    """Reads under the guarded tree are never blocked — config lives under the repo too."""
    cfg = REPO / "config" / "margin_block.json"
    if cfg.exists():
        with open(cfg, "r", encoding="utf-8") as fh:
            assert fh.read(1) is not None
