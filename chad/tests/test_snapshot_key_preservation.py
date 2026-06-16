"""
Snapshot key-preservation contract — every writer of
runtime/portfolio_snapshot.json must PRESERVE keys it does not own.

The publisher (chad/ops/portfolio_snapshot_publisher.py) authors an
authoritative USD block:

    total_equity_usd_authoritative
    usd_ok
    usdcad_rate_used

…which the tier-manager reads. The three collectors that also write the
snapshot — the IBKR v2 collector, the Kraken merge, and the Coinbase
collector — previously rebuilt a fresh, fixed-schema payload and wrote it
wholesale, silently erasing that block (and any other unknown key) between
publisher runs.

These tests seed a snapshot carrying the base keys PLUS the publisher's
authoritative block and an arbitrary probe key (``__preserve_probe__``),
invoke each writer's merge/write path with injected equity (no live IB), and
assert that AFTER each writer:

    * the writer's OWN field updated, and
    * all four seeded keys survive UNCHANGED.

A fourth test confirms the publisher itself preserves keys it does not own
(``__preserve_probe__`` + the collector-authored ``ibkr_equity``) while it
(legitimately) recomputes the authoritative block it owns.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest

# The four keys that MUST survive a collector run unchanged (the three
# publisher-authored authoritative-USD keys + an arbitrary future/unknown key).
AUTHORITATIVE_TOTAL = 141545.23
USD_OK = True
USDCAD_RATE = 1.3994
PROBE_KEY = "__preserve_probe__"
PROBE_VALUE = 123


def _seed_snapshot(path: Path) -> Dict[str, Any]:
    """Write a realistic base snapshot PLUS the authoritative block + probe."""
    seed: Dict[str, Any] = {
        # base keys authored by the collectors
        "ibkr_equity": 219607.0,
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "coinbase_equity": 0.0,
        "kraken_equity": 50.0,
        "kraken_equity_currency": "CAD",
        "kraken_equity_currency_ok": True,
        "ibkr_equity_usd_display": 156000.0,
        "ts_utc": "2026-06-15T00:00:00.000000Z",
        "ttl_seconds": 300,
        # publisher-authored authoritative USD block (must survive collectors)
        "total_equity_usd_authoritative": AUTHORITATIVE_TOTAL,
        "usd_ok": USD_OK,
        "usdcad_rate_used": USDCAD_RATE,
        # arbitrary unknown / future key (must survive every writer)
        PROBE_KEY: PROBE_VALUE,
    }
    path.write_text(json.dumps(seed, indent=2, sort_keys=True), encoding="utf-8")
    return seed


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_four_keys_survive(out: Dict[str, Any]) -> None:
    """The four seeded non-owned keys must be present and unchanged."""
    assert out["total_equity_usd_authoritative"] == AUTHORITATIVE_TOTAL
    assert out["usd_ok"] is USD_OK
    assert out["usdcad_rate_used"] == USDCAD_RATE
    assert out[PROBE_KEY] == PROBE_VALUE


# ---------------------------------------------------------------------------
# Writer 1: IBKR v2 collector (chad-ibkr-collector.timer, every 120s)
# ---------------------------------------------------------------------------


def test_ibkr_v2_collector_preserves_unknown_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chad.portfolio.ibkr_portfolio_collector_v2 import (
        IBKRConnectionConfig,
        IBKRPortfolioCollector,
    )

    monkeypatch.setenv("CHAD_SKIP_IB_CONNECT", "1")
    monkeypatch.setenv("CHAD_BASE_CURRENCY", "CAD")

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)

    cfg = IBKRConnectionConfig(host="127.0.0.1", port=4002, client_id=99, account_id="DUTEST")
    collector = IBKRPortfolioCollector(cfg)
    # Inject equity in place of any IB connection (base currency CAD -> authored).
    collector.get_net_liquidation_with_currency = lambda: (333333.33, "CAD")  # type: ignore[assignment]

    collector.update_snapshot(snapshot_path=snap)
    out = _load(snap)

    # Own field updated.
    assert out["ibkr_equity"] == 333333.33
    assert out["ibkr_equity_currency"] == "CAD"
    assert out["ibkr_equity_currency_ok"] is True
    # All four seeded non-owned keys survive unchanged.
    _assert_four_keys_survive(out)


def test_ibkr_v2_collector_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chad.portfolio.ibkr_portfolio_collector_v2 import (
        IBKRConnectionConfig,
        IBKRPortfolioCollector,
    )

    monkeypatch.setenv("CHAD_SKIP_IB_CONNECT", "1")
    monkeypatch.setenv("CHAD_BASE_CURRENCY", "CAD")

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)

    cfg = IBKRConnectionConfig(host="127.0.0.1", port=4002, client_id=99, account_id="DUTEST")
    collector = IBKRPortfolioCollector(cfg)
    collector.get_net_liquidation_with_currency = lambda: (333333.33, "CAD")  # type: ignore[assignment]

    collector.update_snapshot(snapshot_path=snap)
    first = _load(snap)
    collector.update_snapshot(snapshot_path=snap)
    second = _load(snap)

    # Running twice yields the same preserved result (ts_utc aside).
    first.pop("ts_utc", None)
    second.pop("ts_utc", None)
    assert first == second
    _assert_four_keys_survive(second)


# ---------------------------------------------------------------------------
# Writer 2: Kraken merge (chad-kraken merge path)
# ---------------------------------------------------------------------------


def test_kraken_merge_preserves_unknown_keys(tmp_path: Path) -> None:
    from chad.portfolio.merge_kraken_into_snapshot import merge_kraken_into_snapshot

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)

    # Inject Kraken equity via a balances file (ZCAD treated 1:1 CAD).
    kraken_bal = tmp_path / "kraken_balances.json"
    kraken_bal.write_text(json.dumps({"balances": {"ZCAD": 77.0}}), encoding="utf-8")

    merge_kraken_into_snapshot(snapshot_path=snap, kraken_balances_path=kraken_bal)
    out = _load(snap)

    # Own field updated.
    assert out["kraken_equity"] == 77.0
    # All four seeded non-owned keys survive unchanged.
    _assert_four_keys_survive(out)


def test_kraken_merge_idempotent(tmp_path: Path) -> None:
    from chad.portfolio.merge_kraken_into_snapshot import merge_kraken_into_snapshot

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)
    kraken_bal = tmp_path / "kraken_balances.json"
    kraken_bal.write_text(json.dumps({"balances": {"ZCAD": 77.0}}), encoding="utf-8")

    merge_kraken_into_snapshot(snapshot_path=snap, kraken_balances_path=kraken_bal)
    first = _load(snap)
    merge_kraken_into_snapshot(snapshot_path=snap, kraken_balances_path=kraken_bal)
    second = _load(snap)

    assert first == second
    _assert_four_keys_survive(second)


# ---------------------------------------------------------------------------
# Writer 3: Coinbase collector
# ---------------------------------------------------------------------------


def test_coinbase_collector_preserves_unknown_keys(tmp_path: Path) -> None:
    from chad.portfolio.coinbase_portfolio_collector import PortfolioSnapshotWriter

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)

    writer = PortfolioSnapshotWriter(snap)
    writer.write_with_coinbase_equity(coinbase_equity_usd=Decimal("888.88"))
    out = _load(snap)

    # Own field updated; ibkr_equity preserved (read-through, not erased).
    assert out["coinbase_equity"] == pytest.approx(888.88)
    assert out["ibkr_equity"] == pytest.approx(219607.0)
    # All four seeded non-owned keys survive unchanged.
    _assert_four_keys_survive(out)


def test_coinbase_collector_idempotent(tmp_path: Path) -> None:
    from chad.portfolio.coinbase_portfolio_collector import PortfolioSnapshotWriter

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)

    writer = PortfolioSnapshotWriter(snap)
    writer.write_with_coinbase_equity(coinbase_equity_usd=Decimal("888.88"))
    first = _load(snap)
    writer.write_with_coinbase_equity(coinbase_equity_usd=Decimal("888.88"))
    second = _load(snap)

    assert first == second
    _assert_four_keys_survive(second)


# ---------------------------------------------------------------------------
# Writer 4: Publisher — already preserves unknown keys (VERIFY, do not alter).
# Exercised via the module's own injection seams (no live IB). The publisher
# OWNS the authoritative block, so it (legitimately) recomputes it; it must
# still preserve keys it does NOT own (the probe + collector-authored ibkr_equity).
# ---------------------------------------------------------------------------


def test_publisher_preserves_unowned_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chad.ops.portfolio_snapshot_publisher as pub

    snap = tmp_path / "portfolio_snapshot.json"
    _seed_snapshot(snap)

    # Redirect output + inject the IB-dependent inputs (clean injection seams,
    # not weakened assertions): a live USDCAD rate, IBKR USD equity, Kraken USD.
    monkeypatch.setattr(pub, "RUNTIME_DIR", tmp_path, raising=True)
    monkeypatch.setattr(pub, "OUT_PATH", snap, raising=True)
    monkeypatch.setattr(pub, "_get_live_usdcad_rate", lambda *a, **k: USDCAD_RATE, raising=True)
    monkeypatch.setattr(pub, "_ibkr_equity_usd", lambda usdcad: 156000.0, raising=True)
    monkeypatch.setattr(pub, "_read_kraken_usd_equity", lambda: 0.0, raising=True)

    rc = pub.main()
    assert rc == 0
    out = _load(snap)

    # Keys the publisher does NOT own survive untouched.
    assert out[PROBE_KEY] == PROBE_VALUE
    assert out["ibkr_equity"] == pytest.approx(219607.0)
    # Keys the publisher OWNS are (re)authored from the injected live inputs.
    assert out["usdcad_rate_used"] == USDCAD_RATE
    assert out["usd_ok"] is True
    assert out["total_equity_usd_authoritative"] == pytest.approx(156000.0)
