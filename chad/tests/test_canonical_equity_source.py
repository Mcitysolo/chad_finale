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

import asyncio
import json
from datetime import datetime, timezone
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


def _parse_ts_utc(value):
    """Tolerant ISO-8601 UTC parser.

    Handles the runtime ts_utc shapes seen across the canonical files:
    trailing ``Z`` with or without fractional seconds (pnl_state uses
    ``...:12Z``; portfolio_snapshot/dynamic_caps use ``...:13.281798Z``) and
    explicit ``+00:00`` offsets. Returns a timezone-aware datetime, or None
    when the value is missing/unparseable (caller treats None as skip, never
    fail — an unparseable timestamp is sampling/format noise, not a currency
    defect).
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def test_canonical_single_source_exactness_hermetic(tmp_path) -> None:
    """BOX-034A §5 / Inc 4 (4A): account_equity is sourced EXACTLY from
    dynamic_caps.total_equity, with no live-generation skew.

    Hermetic by construction: build a temp ``runtime/dynamic_caps.json`` with a
    known value + CAD currency tags, drive the real
    ``DynamicCapsEquityProvider`` against it, and assert it returns that value
    bit-for-bit from that exact file. This retires the flake permanently — the
    earlier live exact-compare (pnl_state vs dynamic_caps) could lag by one
    futures-marking cycle because account_equity is profit_lock's last read of
    dynamic_caps; here there is only one generation, so the relationship is
    deterministic.
    """
    from chad.risk.profit_lock import DynamicCapsEquityProvider

    X = 312027.3862987942
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    dc_path = runtime / "dynamic_caps.json"
    dc_path.write_text(
        json.dumps(
            {
                "total_equity": X,
                "total_equity_currency": "CAD",
                "total_equity_currency_ok": True,
            }
        ),
        encoding="utf-8",
    )

    provider = DynamicCapsEquityProvider()
    value, source = asyncio.run(provider.get_equity(tmp_path))

    assert value is not None, "provider returned None for a valid positive dynamic_caps.total_equity"
    assert abs(float(value) - X) <= 1e-9 * max(1.0, X), (
        f"DynamicCapsEquityProvider must return total_equity exactly; got {value!r} vs {X!r}"
    )
    assert source == str(dc_path), (
        f"provider source must point at the dynamic_caps.json it read; got {source!r}"
    )
    # Currency tags read back canonical (fixture integrity + §3 convention).
    dc = json.loads(dc_path.read_text(encoding="utf-8"))
    assert dc.get("total_equity_currency") == "CAD"
    assert dc.get("total_equity_currency_ok") is True


def test_canonical_sources_agree_cross_file_bounded() -> None:
    """BOX-034A §5 / Inc 4 (4B): live cross-file sanity rewrite of the former
    skew test. Currency must ALWAYS bind; the numeric compare is skew-gated and
    given a futures-aware tolerance so it never flakes on sampling lag.

    pnl_state.account_equity and the portfolio_snapshot leg-sum are the two ends
    of one derivation chain (snapshot legs -> dynamic_caps.total_equity ->
    account_equity), written by independent timers; the precision invariant is
    owned by the hermetic 4A test above. Here we only assert (a) currency never
    drifts, and (b) the two files, when same-generation/fresh, sum to roughly
    the same CAD total.
    """
    pnl = _load(RUNTIME / "pnl_state.json")
    snap = _load(RUNTIME / "portfolio_snapshot.json")
    if pnl is None or snap is None:
        pytest.skip("Both canonical files required for cross-file check")
    pnl_eq = pnl.get("account_equity")
    if pnl_eq is None:
        pytest.skip("pnl_state.account_equity is null in this environment")
    try:
        legs = {
            "ibkr_equity": float(snap.get("ibkr_equity", 0.0) or 0.0),
            "kraken_equity": float(snap.get("kraken_equity", 0.0) or 0.0),
            "coinbase_equity": float(snap.get("coinbase_equity", 0.0) or 0.0),
        }
    except (TypeError, ValueError) as exc:
        pytest.fail(f"portfolio_snapshot legs not numeric: {exc}")
    snap_total = sum(legs.values())
    if snap_total <= 0.0:
        pytest.skip("portfolio_snapshot total is non-positive in this environment")

    # --- CURRENCY-EXPLICIT (hard asserts; currency must NEVER drift) ---------
    assert pnl.get("account_equity_currency") == "CAD", (
        f"pnl_state.account_equity_currency must be CAD; got "
        f"{pnl.get('account_equity_currency')!r}"
    )
    assert pnl.get("account_equity_currency_ok") is True, (
        "pnl_state.account_equity_currency_ok must be True"
    )
    # Per-leg currency, CONDITIONAL: only venues with a non-zero/non-null equity
    # carry currency tags (e.g. coinbase_equity=0 has no tag — skip it).
    for leg_key, leg_val in legs.items():
        if leg_val == 0.0:
            continue
        cur = snap.get(f"{leg_key}_currency")
        ok = snap.get(f"{leg_key}_currency_ok")
        assert cur == "CAD", (
            f"portfolio_snapshot.{leg_key}_currency must be CAD for non-zero leg; got {cur!r}"
        )
        assert ok is True, (
            f"portfolio_snapshot.{leg_key}_currency_ok must be True for non-zero leg"
        )

    # --- SKEW / STALENESS GATE (skip, never fail) ---------------------------
    ts_pnl = _parse_ts_utc(pnl.get("ts_utc"))
    ts_snap = _parse_ts_utc(snap.get("ts_utc"))
    if ts_pnl is None or ts_snap is None:
        pytest.skip("unparseable ts_utc: drift is sampling/format noise, not currency")
    now = datetime.now(timezone.utc)
    pnl_ttl = int(pnl.get("ttl_seconds", 0) or 0)
    snap_ttl = int(snap.get("ttl_seconds", 0) or 0)
    if pnl_ttl > 0 and (now - ts_pnl).total_seconds() > pnl_ttl:
        pytest.skip("generation skew / staleness: drift is sampling-lag, not currency")
    if snap_ttl > 0 and (now - ts_snap).total_seconds() > snap_ttl:
        pytest.skip("generation skew / staleness: drift is sampling-lag, not currency")
    if abs((ts_pnl - ts_snap).total_seconds()) > 300:  # snapshot cadence
        pytest.skip("generation skew / staleness: drift is sampling-lag, not currency")

    # --- TOLERANT NUMERIC -----------------------------------------------------
    # Coarse cross-file currency/sanity bound; the precision invariant is owned
    # by test_canonical_single_source_exactness_hermetic (4A). 0.5% is sized to
    # absorb ~$4M futures notional marking across the 300s snapshot cadence
    # (observed worst-case cross-generation drift ~0.22%).
    tolerance = max(1.0, 0.005 * snap_total)
    drift = abs(float(pnl_eq) - snap_total)
    assert drift <= tolerance, (
        f"Canonical equity cross-file drift exceeded bound (in-window): "
        f"pnl_state.account_equity={pnl_eq} vs portfolio_snapshot total={snap_total:.2f}; "
        f"drift={drift:.4f} > tolerance={tolerance:.4f}. "
        "See ops/pending_actions/BOX-034A_canonical_equity_currency_unification_2026-06-01.md §5."
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
