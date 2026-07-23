"""W4B-8c (INCIDENT-0723 D3) — tier-1 close-price cascade reads real fills only.

_load_recent_broker_fill_price is documented as the "broker-confirmed fill"
tier of the PR-02b cascade, but its old filter only rejected
_REJECTED_FILL_STATUSES — so dry_run / market_closed exhaust rows (98% of
FILLS_20260723) qualified as "broker-confirmed" prices for real closes.
These tests pin the allow-list: only genuine fill statuses may price a close.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chad.core import position_reconciler as pr


@pytest.fixture()
def fills_dir(tmp_path, monkeypatch):
    d = tmp_path / "fills"
    d.mkdir()
    monkeypatch.setattr(pr, "_FILLS_DIR", d, raising=True)
    # An absent/stale cache so tier-2 can never answer for these tests.
    monkeypatch.setattr(pr, "_PRICE_CACHE_PATH", tmp_path / "no_cache.json",
                        raising=True)
    return d


def _append(d: Path, *, symbol: str, price: float, status: str,
            day: str = "20260723") -> None:
    payload = {
        "symbol": symbol,
        "side": "SELL",
        "quantity": 5.0,
        "fill_price": price,
        "status": status,
        "reject": False,
        "fill_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (d / f"FILLS_{day}.ndjson").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"payload": payload}) + "\n")


@pytest.mark.parametrize("status", ["dry_run", "market_closed", "Submitted",
                                    "PendingCancel", "duplicate_blocked"])
def test_exhaust_rows_never_price_a_close(fills_dir, status):
    """The incident shape: only exhaust rows exist → the tier must abstain
    (0.0), never 'confirm' a price off a row that never traded."""
    _append(fills_dir, symbol="PSQ", price=26.2, status=status)
    assert pr._load_recent_broker_fill_price("PSQ") == 0.0
    assert pr._resolve_close_fill_price("PSQ") == 0.0  # full-cascade abstain


@pytest.mark.parametrize("status", ["paper_fill", "filled", "Filled", "fill"])
def test_genuine_fill_statuses_price_the_close(fills_dir, status):
    """Real fills (any historical casing) remain tier-1 price sources."""
    _append(fills_dir, symbol="PSQ", price=26.2, status=status)
    assert pr._load_recent_broker_fill_price("PSQ") == 26.2


def test_exhaust_row_does_not_shadow_older_real_fill(fills_dir):
    """A newer dry_run row must not shadow (or block) the older real fill —
    the walk skips it and finds the genuine price."""
    _append(fills_dir, symbol="PSQ", price=26.2, status="paper_fill",
            day="20260721")
    _append(fills_dir, symbol="PSQ", price=99.9, status="dry_run",
            day="20260723")
    assert pr._load_recent_broker_fill_price("PSQ") == 26.2
