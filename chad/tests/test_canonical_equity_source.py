"""GAP-007 / Box 034 — canonical equity source parity invariants.

Tests the runtime-invariant declarations made in
``ops/pending_actions/BOX-034_canonical_equity_source_policy.md``:

  - §3a: ``runtime/pnl_state.json::account_equity`` is the operator/dashboard truth.
  - §3b: ``runtime/portfolio_snapshot.json`` (sum of ibkr+kraken+coinbase) is
         the ops/risk truth.
  - §4a: when both files are fresh, their values agree within 0.05 %.
  - §4d: the profit-lock composite provider chain must not include
         the daily ``equity_history.ndjson`` ledger.

Tests are skipped (rather than failed) when a canonical runtime file is
absent — they are intended to run in environments with a populated
runtime directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "runtime"


def _load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def test_pnl_state_schema_matches_canonical_section_3a() -> None:
    """§3a: pnl_state.json carries the canonical operator account_equity."""
    pnl_path = RUNTIME / "pnl_state.json"
    pnl = _load(pnl_path)
    if pnl is None:
        pytest.skip(f"{pnl_path} not present in this environment")
    assert pnl.get("schema_version") == "pnl_state.v1", (
        f"pnl_state schema must be pnl_state.v1; got {pnl.get('schema_version')!r}"
    )
    # The canonical key MUST be present (even when value is None for the
    # "equity unknown" case, the key itself is part of the schema).
    assert "account_equity" in pnl, "pnl_state.account_equity key missing"
    assert "ts_utc" in pnl and isinstance(pnl["ts_utc"], str), (
        "pnl_state.ts_utc missing or non-string"
    )
    assert "ttl_seconds" in pnl, "pnl_state.ttl_seconds missing"
    # §3a expected TTL ≤ 120s (currently 60s).
    assert int(pnl["ttl_seconds"]) <= 120, (
        f"pnl_state.ttl_seconds={pnl['ttl_seconds']} exceeds operator-freshness "
        "expectation of 120s; see policy §3a"
    )


def test_portfolio_snapshot_schema_matches_canonical_section_3b() -> None:
    """§3b: portfolio_snapshot.json is the ops/risk canonical."""
    snap_path = RUNTIME / "portfolio_snapshot.json"
    snap = _load(snap_path)
    if snap is None:
        pytest.skip(f"{snap_path} not present in this environment")
    for key in ("ibkr_equity", "kraken_equity", "coinbase_equity", "ts_utc", "ttl_seconds"):
        assert key in snap, (
            f"portfolio_snapshot.{key} missing; canonical §3b requires this key"
        )
    # §3b expected TTL ≤ 600s (currently 300s).
    assert int(snap["ttl_seconds"]) <= 600, (
        f"portfolio_snapshot.ttl_seconds={snap['ttl_seconds']} exceeds ops-truth "
        "freshness expectation of 600s; see policy §3b"
    )
    for venue_key in ("ibkr_equity", "kraken_equity", "coinbase_equity"):
        val = snap.get(venue_key)
        assert val is None or isinstance(val, (int, float)), (
            f"portfolio_snapshot.{venue_key} must be numeric or null; got {type(val).__name__}"
        )


def test_canonical_sources_agree_within_skew_tolerance() -> None:
    """§4a: when both canonical files exist with non-null equity values,
    pnl_state.account_equity must agree with the sum of per-venue
    portfolio_snapshot equities within max(1.0 USD, 0.05 % of total)."""
    pnl = _load(RUNTIME / "pnl_state.json")
    snap = _load(RUNTIME / "portfolio_snapshot.json")
    if pnl is None or snap is None:
        pytest.skip("Both canonical files required for skew check")
    pnl_eq = pnl.get("account_equity")
    if pnl_eq is None:
        pytest.skip("pnl_state.account_equity is null in this environment")
    try:
        snap_total = float(snap.get("ibkr_equity", 0.0) or 0.0) \
                   + float(snap.get("kraken_equity", 0.0) or 0.0) \
                   + float(snap.get("coinbase_equity", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"portfolio_snapshot legs not numeric: {exc}")
    if snap_total <= 0.0:
        pytest.skip("portfolio_snapshot total is non-positive in this environment")
    tolerance = max(1.0, 0.0005 * snap_total)
    drift = abs(float(pnl_eq) - snap_total)
    assert drift <= tolerance, (
        f"Canonical equity skew exceeded tolerance: pnl_state.account_equity="
        f"{pnl_eq} vs portfolio_snapshot total={snap_total:.2f}; drift={drift:.4f} > tolerance={tolerance:.4f}. "
        "See ops/pending_actions/BOX-034_canonical_equity_source_policy.md §4a."
    )


def test_profit_lock_provider_chain_excludes_daily_equity_history() -> None:
    """§4d: the profit-lock composite provider must NOT read from the
    daily equity_history.ndjson ledger (which would mix stale daily
    snapshots into live equity decisions)."""
    pl_path = REPO_ROOT / "chad" / "risk" / "profit_lock.py"
    source = pl_path.read_text(encoding="utf-8")
    # Find the _build_default_equity_provider function body.
    start = source.find("def _build_default_equity_provider")
    assert start != -1, "could not locate _build_default_equity_provider in profit_lock.py"
    end = source.find("\n\n\n", start)
    body = source[start: end if end != -1 else len(source)]
    assert "equity_history" not in body, (
        "profit_lock composite provider must not include equity_history.ndjson; "
        "see ops/pending_actions/BOX-034_canonical_equity_source_policy.md §4d"
    )


def test_policy_artifact_exists() -> None:
    """The Box-034 policy artifact MUST exist; it is the SSOT for §3 declarations."""
    policy_path = REPO_ROOT / "ops" / "pending_actions" / "BOX-034_canonical_equity_source_policy.md"
    assert policy_path.is_file(), (
        f"Box-034 policy artifact missing at {policy_path} — this is the "
        "canonical declaration referenced by every consumer in §3."
    )
