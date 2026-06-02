"""
BOX-034B Step 2 — collector preserves publisher-authored snapshot keys.

`IBKRPortfolioCollector.update_snapshot` builds a hand-built payload written
wholesale, so any prior key it does not re-list is dropped. Previously this
silently stripped the publisher-authored kraken currency tags
(`kraken_equity_currency` / `kraken_equity_currency_ok`) and the cosmetic
`ibkr_equity_usd_display` on every 2-min collector cycle, preventing a stable
`total_equity_currency_ok`.

The fix carries these forward verbatim (PRESERVE, never author), present-only
so a cold/empty snapshot omits them rather than emitting null.

These tests avoid any IB connection by stubbing
`get_net_liquidation_with_currency` on the instance.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.portfolio.ibkr_portfolio_collector_v2 import (
    IBKRConnectionConfig,
    IBKRPortfolioCollector,
)


def _make_collector() -> IBKRPortfolioCollector:
    cfg = IBKRConnectionConfig(host="127.0.0.1", port=4002, client_id=99, account_id="DUTEST")
    collector = IBKRPortfolioCollector(cfg)
    # Stub the only IB call update_snapshot makes — return CAD so ibkr_equity
    # is freshly authored (currency_ok=True path).
    collector.get_net_liquidation_with_currency = lambda: (123456.78, "CAD")  # type: ignore[assignment]
    return collector


def _write_prior(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_carries_kraken_tags_and_usd_display_verbatim(tmp_path: Path) -> None:
    snap = tmp_path / "portfolio_snapshot.json"
    _write_prior(
        snap,
        {
            "ibkr_equity": 999.0,
            "kraken_equity": 255.55,
            "kraken_equity_currency": "CAD",
            "kraken_equity_currency_ok": True,
            "ibkr_equity_usd_display": 184.58,
        },
    )

    _make_collector().update_snapshot(snapshot_path=snap)

    out = json.loads(snap.read_text(encoding="utf-8"))
    # Publisher-authored keys carried verbatim.
    assert out["kraken_equity_currency"] == "CAD"
    assert out["kraken_equity_currency_ok"] is True
    assert out["ibkr_equity_usd_display"] == 184.58
    # Kraken value still preserved (read-through, unchanged).
    assert out["kraken_equity"] == 255.55


def test_preserves_fail_closed_false_flag(tmp_path: Path) -> None:
    snap = tmp_path / "portfolio_snapshot.json"
    _write_prior(
        snap,
        {
            "ibkr_equity": 999.0,
            "kraken_equity": 250.0,
            "kraken_equity_currency_ok": False,  # fail-closed flag from publisher
        },
    )

    _make_collector().update_snapshot(snapshot_path=snap)

    out = json.loads(snap.read_text(encoding="utf-8"))
    assert out["kraken_equity_currency_ok"] is False
    # Not present in prior -> not fabricated.
    assert "kraken_equity_currency" not in out
    assert "ibkr_equity_usd_display" not in out


def test_cold_empty_snapshot_omits_keys_no_null(tmp_path: Path) -> None:
    # No prior file at all -> data == {}.
    snap = tmp_path / "portfolio_snapshot.json"

    _make_collector().update_snapshot(snapshot_path=snap)

    out = json.loads(snap.read_text(encoding="utf-8"))
    # None of the three present-only keys appear (no null fabrication).
    assert "kraken_equity_currency" not in out
    assert "kraken_equity_currency_ok" not in out
    assert "ibkr_equity_usd_display" not in out


def test_ibkr_authoring_unaffected(tmp_path: Path) -> None:
    snap = tmp_path / "portfolio_snapshot.json"
    _write_prior(
        snap,
        {
            "ibkr_equity": 111.0,
            "ibkr_equity_currency": "USD",  # stale/wrong — must be overwritten
            "ibkr_equity_currency_ok": False,
            "kraken_equity": 250.0,
            "kraken_equity_currency": "CAD",
        },
    )

    _make_collector().update_snapshot(snapshot_path=snap)

    out = json.loads(snap.read_text(encoding="utf-8"))
    # ibkr_equity + tags freshly authored (CAD, ok=True), NOT read-through.
    assert out["ibkr_equity"] == 123456.78
    assert out["ibkr_equity_currency"] == "CAD"
    assert out["ibkr_equity_currency_ok"] is True
    # kraken tag still carried verbatim alongside.
    assert out["kraken_equity_currency"] == "CAD"
