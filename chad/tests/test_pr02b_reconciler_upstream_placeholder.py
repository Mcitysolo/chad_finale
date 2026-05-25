"""
PR-02b — Reconciler upstream $100 placeholder fix (2026-05-25)

These tests pin the new behaviour of
``chad.core.position_reconciler._resolve_close_fill_price`` and the
matching ``apply_close_intents`` abstain branch.

Background: PR-02 silenced the *delta* upstream producer (commit 139d275)
of the canonical fill_price=100.0 placeholder. The remaining producer was
the reconciler's synthesized close-fill path — ``apply_close_intents``
used the legacy unchecked ``_load_price`` helper, so a stale/missing
price_cache let a $100 record land in data/fills/FILLS_*.ndjson tagged
``strategy="reconciler"``. PR-02b adds a cascade
(broker fill → fresh cache → abstain) and a hard abstain when no usable
price is available.

T1 — reconciler abstains when no real price source resolves.
T2 — reconciler uses fresh price_cache when available.
T3 — reconciler uses last broker-confirmed fill when price_cache is stale.
T4 — GAP-058 writer-level defense gate is preserved (regression lock).
T5 — PR-02 delta abstain helper still rejects invalid prices (smoke).
T6 — PR-09 positions_truth schema fields still present (smoke).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_reconciler_paths(monkeypatch, tmp_path):
    """Redirect the reconciler's _PRICE_CACHE_PATH and _FILLS_DIR to
    isolated temp locations so each test owns its own price/broker-fill
    universe and cannot leak into runtime/data/."""
    from chad.core import position_reconciler as pr

    cache_dir = tmp_path / "runtime"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "price_cache.json"

    fills_dir = tmp_path / "data" / "fills"
    fills_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pr, "_PRICE_CACHE_PATH", cache_file, raising=True)
    monkeypatch.setattr(pr, "_FILLS_DIR", fills_dir, raising=True)
    return cache_file, fills_dir


def _write_cache(cache_file: Path, prices: dict, *, age_seconds: float = 0.0,
                 ttl_seconds: float = 300.0) -> None:
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    cache_file.write_text(
        json.dumps(
            {
                "prices": prices,
                "ts_utc": ts.isoformat().replace("+00:00", "Z"),
                "ttl_seconds": ttl_seconds,
            }
        ),
        encoding="utf-8",
    )


def _append_fill(fills_dir: Path, *, day: str, symbol: str, fill_price: float,
                 status: str = "paper_fill", reject: bool = False,
                 extra: dict | None = None, ts: str | None = None) -> Path:
    f = fills_dir / f"FILLS_{day}.ndjson"
    payload = {
        "symbol": symbol,
        "side": "BUY",
        "quantity": 10.0,
        "fill_price": fill_price,
        "status": status,
        "reject": reject,
        "strategy": "delta",
        "extra": extra or {},
        "fill_time_utc": ts or datetime.now(timezone.utc).isoformat(),
    }
    with f.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"payload": payload}) + "\n")
    return f


# ---------------------------------------------------------------------------
# T1 — reconciler abstain on no real price
# ---------------------------------------------------------------------------


def test_t1_reconciler_abstains_when_no_real_price_available(
    patched_reconciler_paths, monkeypatch, caplog
):
    """No broker fills, no cache → _resolve_close_fill_price returns 0.0
    AND apply_close_intents emits the documented abstain log line and
    writes NO PaperExecEvidence."""
    cache_file, fills_dir = patched_reconciler_paths
    # Empty cache file path (does not exist) AND empty fills dir.

    from chad.core import position_reconciler as pr

    # Resolver returns 0.0 (abstain signal).
    assert pr._resolve_close_fill_price("IWM") == 0.0

    written: List[dict] = []

    def _fake_write(ev):
        written.append({"symbol": getattr(ev, "symbol", None)})
        return {"fill_id": "fake", "fills_path": ""}

    # Patch both write and normalize so the test exercises only the
    # abstain branch in apply_close_intents.
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.write_paper_exec_evidence",
        _fake_write,
        raising=True,
    )

    class _FakeOrder:
        symbol = "IWM"
        side = "SELL"
        quantity = 10.0
        status = "paper_fill"
        submitted_at = None
        asset_class = "EQUITY"

    class _FakeAdapter:
        def submit_strategy_trade_intents(self, intents):
            return [_FakeOrder()]

    # Stop the post-confirmation branch from poking position_guard — the
    # guard is real-codebase state and not what this test is asserting on.
    monkeypatch.setattr(
        "chad.core.position_guard.is_fill_confirmed",
        lambda f: True,
        raising=True,
    )

    close_intent = {
        "symbol": "IWM",
        "action": "CLOSE",
        "open_side": "BUY",
        "close_side": "SELL",
        "quantity": 10.0,
        "reason": "reconciler_flip_test",
        "position_key": "reconciler|IWM",
        "strategy": "reconciler",
    }

    import logging
    caplog.set_level(logging.WARNING, logger="chad.core.position_reconciler")
    pr.apply_close_intents([close_intent], _FakeAdapter())

    assert written == [], (
        "T1 regression: reconciler must NOT write evidence when no real "
        "price source resolves — got %r" % written
    )
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "RECONCILER_CLOSE_ABSTAIN_NO_PRICE" in msgs, (
        "T1: abstain branch must emit RECONCILER_CLOSE_ABSTAIN_NO_PRICE "
        "with the symbol — got: %s" % msgs
    )
    assert "IWM" in msgs


# ---------------------------------------------------------------------------
# T2 — reconciler uses fresh price_cache when available
# ---------------------------------------------------------------------------


def test_t2_reconciler_uses_fresh_price_cache_not_100(
    patched_reconciler_paths,
):
    cache_file, _fills_dir = patched_reconciler_paths
    _write_cache(cache_file, {"IWM": 283.8}, age_seconds=10.0, ttl_seconds=300.0)

    from chad.core import position_reconciler as pr

    px = pr._resolve_close_fill_price("IWM")
    assert px == 283.8, "T2: must surface the fresh cached price, not 100.0"
    assert px != 100.0


def test_t2b_reconciler_rejects_stale_cache(patched_reconciler_paths):
    """Stale cache (age > ttl_seconds) MUST return 0.0 — the missing
    freshness check on the legacy _load_price was the root-cause."""
    cache_file, _fills_dir = patched_reconciler_paths
    _write_cache(cache_file, {"IWM": 283.8}, age_seconds=900.0, ttl_seconds=300.0)

    from chad.core import position_reconciler as pr

    assert pr._load_fresh_cache_price("IWM") == 0.0
    # With no broker fills either, the cascade abstains.
    assert pr._resolve_close_fill_price("IWM") == 0.0


# ---------------------------------------------------------------------------
# T3 — reconciler uses last broker fill when price_cache is stale/missing
# ---------------------------------------------------------------------------


def test_t3_reconciler_uses_last_broker_fill_when_cache_stale(
    patched_reconciler_paths,
):
    cache_file, fills_dir = patched_reconciler_paths
    # Stale cache → tier-2 fails.
    _write_cache(cache_file, {"IWM": 283.8}, age_seconds=900.0, ttl_seconds=300.0)
    # A real recent broker-confirmed fill for IWM at $285.20.
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    _append_fill(fills_dir, day=today, symbol="IWM", fill_price=285.20)

    from chad.core import position_reconciler as pr

    px = pr._resolve_close_fill_price("IWM")
    assert px == 285.20, "T3: broker-confirmed fill must win over stale cache"
    assert px != 100.0


def test_t3b_broker_fill_resolver_rejects_untrusted_and_rejected(
    patched_reconciler_paths,
):
    """Rejected fills, pnl_untrusted fills, and fill_price <= 0 fills
    must NEVER be surfaced as tier-1 truth."""
    cache_file, fills_dir = patched_reconciler_paths
    cache_file.unlink(missing_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    # Three poison candidates — all must be skipped.
    _append_fill(fills_dir, day=today, symbol="IWM",
                 fill_price=100.0, status="rejected", reject=True)
    _append_fill(fills_dir, day=today, symbol="IWM",
                 fill_price=999.0, status="paper_fill",
                 extra={"pnl_untrusted": True})
    _append_fill(fills_dir, day=today, symbol="IWM",
                 fill_price=0.0, status="paper_fill")

    from chad.core import position_reconciler as pr

    assert pr._load_recent_broker_fill_price("IWM") == 0.0
    assert pr._resolve_close_fill_price("IWM") == 0.0


# ---------------------------------------------------------------------------
# T4 — GAP-058 writer defense gate preserved (regression lock)
# ---------------------------------------------------------------------------


def test_t4_writer_defense_gate_still_demotes_100_dollar_placeholder(
    monkeypatch, tmp_path,
):
    """Direct injection of fill_price=100.0 through
    paper_exec_evidence_writer.normalize_paper_fill_evidence must still
    demote the row to status='rejected' with pnl_untrusted=True. The
    GAP-058 / P0-1 defense gates are downstream of PR-02b and MUST
    remain unchanged."""
    import chad.execution.paper_exec_evidence_writer as wmod
    from chad.execution.paper_exec_evidence_writer import (
        PaperExecEvidence,
        normalize_paper_fill_evidence,
    )

    # Point the writer's cache at a temp file with the real IWM price.
    cache = tmp_path / "price_cache.json"
    cache.write_text(json.dumps({"prices": {"IWM": 283.6}}), encoding="utf-8")
    monkeypatch.setattr(wmod, "PRICE_CACHE_PATH", cache, raising=True)

    ev = PaperExecEvidence(
        symbol="IWM",
        side="SELL",
        quantity=10.0,
        fill_price=100.0,
        expected_price=100.0,
        strategy="reconciler",
        source_strategies=["reconciler"],
        broker="ibkr_paper",
        status="paper_fill",
        asset_class="equity",
        is_live=False,
        fill_time_utc="2026-05-25T00:00:00Z",
    )
    normalize_paper_fill_evidence(ev)

    assert ev.reject is True
    assert ev.status == "rejected"
    assert ev.fill_price == 0.0
    assert ev.expected_price == 0.0
    assert isinstance(ev.extra, dict)
    assert ev.extra.get("pnl_untrusted") is True
    assert ev.extra.get("trust_state") == "PLACEHOLDER"
    assert ev.extra.get("placeholder_fill_price") == 100.0
    assert "pnl_untrusted" in ev.tags


# ---------------------------------------------------------------------------
# T5 — PR-02 delta abstain helper unchanged (smoke)
# ---------------------------------------------------------------------------


def test_t5_pr02_delta_resolve_positive_price_still_rejects_invalid(
    monkeypatch, tmp_path,
):
    """PR-02 (commit 139d275) added a price-validation chokepoint at the
    top of chad.strategies.delta. PR-02b is in a different file (the
    reconciler), but a smoke import + invalid-price reject keeps the
    PR-02 contract visible from the same regression run.

    Use CHAD_PRICE_CACHE_PATH to point the PR-02 helper at an empty
    cache so the cross-check cannot accidentally rescue a bad input
    from runtime/price_cache.json during the test."""
    empty_cache = tmp_path / "price_cache.json"
    empty_cache.write_text(json.dumps({"prices": {}}), encoding="utf-8")
    monkeypatch.setenv("CHAD_PRICE_CACHE_PATH", str(empty_cache))

    from chad.strategies.delta import _resolve_positive_price

    # Minimal ctx duck-type: no ticks, no bars -> only the prices map
    # contributes a candidate to _resolve_positive_price.
    bad_ctx = type("C", (), {"ticks": {}, "bars": {}})()
    for bad in (None, 0, 0.0, -1.0, float("nan"), float("inf")):
        prices = {"IWM": bad}
        out = _resolve_positive_price(bad_ctx, "IWM", prices)
        assert out is None, (
            "T5: PR-02 helper must abstain on invalid input %r — got %r"
            % (bad, out)
        )


# ---------------------------------------------------------------------------
# T6 — PR-09 positions_truth schema fields still exposed (smoke)
# ---------------------------------------------------------------------------


def test_t6_pr09_positions_truth_exposes_broker_authority_fields():
    """PR-09 (commit 2d454ed) separated broker-authority truth from
    replay diagnostics. positions_truth.json MUST keep both
    ``broker_authority_status`` and ``replay_diagnostic_status`` at the
    top level. Smoke-read the live runtime artifact (read-only — no
    mutation). Skip cleanly if the artifact does not yet exist."""
    p = Path("/home/ubuntu/chad_finale/runtime/positions_truth.json")
    if not p.is_file():
        pytest.skip("runtime/positions_truth.json missing — PR-09 smoke skipped")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "broker_authority_status" in data, (
        "T6: PR-09 contract broken — broker_authority_status missing"
    )
    assert "replay_diagnostic_status" in data, (
        "T6: PR-09 contract broken — replay_diagnostic_status missing"
    )
