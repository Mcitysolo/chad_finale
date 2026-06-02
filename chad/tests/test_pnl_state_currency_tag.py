"""
BOX-034A Inc 3 Step 1b — pnl_state currency propagation.

The profit_lock side-write of runtime/pnl_state.json additively carries
``account_equity_currency`` / ``account_equity_currency_ok`` onto the
operator-facing account_equity. The ok flag is True ONLY when the equity came
from dynamic_caps (matched by equity_source), the dynamic_caps payload tagged
``total_equity_currency_ok`` True, and equity is known. Everything else is
fail-closed (False); the currency falls back to the configured base currency.

These are ADDITIVE-write assertions only — no reader enforcement (later step).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from chad.risk.profit_lock import (
    DynamicCapsEquityProvider,
    FileEquityProvider,
    CompositeEquityProvider,
    ProfitLockConfig,
    ProfitLockEngine,
    PnlProvider,
)


class _StubPnlProvider(PnlProvider):
    """Deterministic realized-PnL stub so tests don't depend on data/ trades."""

    def __init__(self, pnl: float = 100.0) -> None:
        self._pnl = pnl

    async def get_realized_pnl(
        self, repo_root: Path, days: int = 0
    ) -> Tuple[float, int, List[str]]:
        return float(self._pnl), 1, ["stub"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_side_write(repo_root: Path) -> dict:
    """Run write_state and return the resulting pnl_state.json payload."""
    engine = ProfitLockEngine(
        repo_root=repo_root,
        equity_provider=CompositeEquityProvider(
            [
                DynamicCapsEquityProvider(),
                FileEquityProvider(
                    "positions_snapshot.json",
                    ("net_liquidation", "equity", "total_equity"),
                ),
            ]
        ),
        pnl_provider=_StubPnlProvider(),
        config=ProfitLockConfig(),
    )
    out_path = repo_root / "runtime" / "profit_lock_state.json"
    asyncio.run(engine.write_state(out_path))
    pnl_path = repo_root / "runtime" / "pnl_state.json"
    return json.loads(pnl_path.read_text(encoding="utf-8"))


def test_dynamic_caps_source_currency_ok_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {
            "total_equity": 150_000.0,
            "total_equity_currency": "CAD",
            "total_equity_currency_ok": True,
        },
    )
    pnl = _run_side_write(tmp_path)
    assert pnl["equity_known"] is True
    assert pnl["account_equity_currency"] == "CAD"
    assert pnl["account_equity_currency_ok"] is True


def test_dynamic_caps_source_but_currency_ok_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {
            "total_equity": 150_000.0,
            "total_equity_currency": "CAD",
            "total_equity_currency_ok": False,
        },
    )
    pnl = _run_side_write(tmp_path)
    assert pnl["account_equity_currency"] == "CAD"
    assert pnl["account_equity_currency_ok"] is False


def test_fallback_source_currency_ok_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    # dynamic_caps yields no usable equity (<= 0) -> chain falls through to the
    # FileEquityProvider, so equity_source is NOT the dynamic_caps path.
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 0.0, "total_equity_currency": "CAD", "total_equity_currency_ok": True},
    )
    _write_json(
        tmp_path / "runtime" / "positions_snapshot.json",
        {"net_liquidation": 200_000.0},
    )
    pnl = _run_side_write(tmp_path)
    assert pnl["equity_known"] is True
    assert pnl["account_equity"] == pytest.approx(200_000.0)
    # equity came from a fallback file -> currency unknown -> fail-closed
    assert pnl["account_equity_currency"] == "CAD"  # base
    assert pnl["account_equity_currency_ok"] is False


def test_equity_unknown_currency_ok_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    # No provider yields equity -> equity unknown, account_equity None.
    pnl = _run_side_write(tmp_path)
    assert pnl["equity_known"] is False
    assert pnl["account_equity"] is None
    assert pnl["account_equity_currency"] == "CAD"
    assert pnl["account_equity_currency_ok"] is False


def test_dynamic_caps_missing_currency_key_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    # dynamic_caps is the source but predates Step 1a (no currency keys).
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 150_000.0},
    )
    pnl = _run_side_write(tmp_path)
    assert pnl["equity_known"] is True
    # currency key absent -> falls back to base; ok absent -> fail-closed False
    assert pnl["account_equity_currency"] == "CAD"
    assert pnl["account_equity_currency_ok"] is False


def test_base_currency_override_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAD_BASE_CURRENCY", "usd")  # lower-case to verify normalization
    # Fallback source so currency comes from base (not the dynamic_caps tag).
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 0.0},
    )
    _write_json(
        tmp_path / "runtime" / "positions_snapshot.json",
        {"net_liquidation": 200_000.0},
    )
    pnl = _run_side_write(tmp_path)
    assert pnl["account_equity_currency"] == "USD"
    assert pnl["account_equity_currency_ok"] is False
