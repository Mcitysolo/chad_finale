"""
BOX-034A Inc 3 Step 2 — WARN-MODE currency assertions (warn-before-enforce).

Each of the four equity consumers (plus the untagged-fallback flag) emits a
distinct, greppable ``CURRENCY_WARN_*`` WARNING when the consumed equity's
currency is unverified or != base, and is SILENT when currency == CAD and the
``_ok`` flag is True. These checks are WARN-ONLY: they must never raise, alter
the equity value, or change control flow. Enforce is a later step.

Markers under test:
  - CURRENCY_WARN_DYNCAPS_PROVIDER     (DynamicCapsEquityProvider.get_equity)
  - CURRENCY_WARN_PNL_STATE            (profit_lock pnl_state side-write)
  - CURRENCY_WARN_SNAPSHOT_LEG         (orchestrator per-leg snapshot)
  - CURRENCY_WARN_TOTAL_EQUITY_OK_FALSE(orchestrator derived total ok)
  - CURRENCY_WARN_RISK_CAP_UNVERIFIED  (orchestrator post-build_payload)
  - CURRENCY_WARN_UNTAGGED_FALLBACK    (CompositeEquityProvider fall-through)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from chad.risk.profit_lock import (
    CompositeEquityProvider,
    DynamicCapsEquityProvider,
    FileEquityProvider,
    PnlProvider,
    ProfitLockConfig,
    ProfitLockEngine,
)
from chad.core.orchestrator import Orchestrator, OrchestratorSettings


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _StubPnlProvider(PnlProvider):
    def __init__(self, pnl: float = 100.0) -> None:
        self._pnl = pnl

    async def get_realized_pnl(
        self, repo_root: Path, days: int = 0
    ) -> Tuple[float, int, List[str]]:
        return float(self._pnl), 1, ["stub"]


def _run_side_write(repo_root: Path) -> dict:
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
    return json.loads((repo_root / "runtime" / "pnl_state.json").read_text(encoding="utf-8"))


def _refresh(tmp_path: Path, snapshot: Dict[str, Any], monkeypatch) -> Path:
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "10.0")
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    snapshot_path = tmp_path / "portfolio_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"
    _write_json(snapshot_path, snapshot)
    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    Orchestrator(settings=settings).refresh_dynamic_caps()
    return caps_path


# --------------------------------------------------------------------------- #
# Consumer 1 — CURRENCY_WARN_DYNCAPS_PROVIDER
# --------------------------------------------------------------------------- #
def test_dyncaps_provider_warns_when_ok_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 150_000.0, "total_equity_currency": "CAD", "total_equity_currency_ok": False},
    )
    with caplog.at_level(logging.WARNING):
        value, _src = asyncio.run(DynamicCapsEquityProvider().get_equity(tmp_path))
    # control flow / value unaffected by the warn
    assert value == pytest.approx(150_000.0)
    assert "CURRENCY_WARN_DYNCAPS_PROVIDER" in caplog.text


def test_dyncaps_provider_silent_when_ok_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 150_000.0, "total_equity_currency": "CAD", "total_equity_currency_ok": True},
    )
    with caplog.at_level(logging.WARNING):
        value, _src = asyncio.run(DynamicCapsEquityProvider().get_equity(tmp_path))
    assert value == pytest.approx(150_000.0)
    assert "CURRENCY_WARN_DYNCAPS_PROVIDER" not in caplog.text


# --------------------------------------------------------------------------- #
# Consumer 2 — CURRENCY_WARN_PNL_STATE
# --------------------------------------------------------------------------- #
def test_pnl_state_warns_when_ok_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 150_000.0, "total_equity_currency": "CAD", "total_equity_currency_ok": False},
    )
    with caplog.at_level(logging.WARNING):
        pnl = _run_side_write(tmp_path)
    # write still succeeds with fail-closed currency (control flow unaffected)
    assert pnl["account_equity_currency_ok"] is False
    assert "CURRENCY_WARN_PNL_STATE" in caplog.text


def test_pnl_state_silent_when_ok_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 150_000.0, "total_equity_currency": "CAD", "total_equity_currency_ok": True},
    )
    with caplog.at_level(logging.WARNING):
        pnl = _run_side_write(tmp_path)
    assert pnl["account_equity_currency_ok"] is True
    assert "CURRENCY_WARN_PNL_STATE" not in caplog.text


# --------------------------------------------------------------------------- #
# Consumer 3 — CURRENCY_WARN_SNAPSHOT_LEG + CURRENCY_WARN_TOTAL_EQUITY_OK_FALSE
# --------------------------------------------------------------------------- #
def test_snapshot_leg_warns_for_untrusted_active_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        caps_path = _refresh(
            tmp_path,
            {
                "ibkr_equity": 100_000.0,
                "ibkr_equity_currency_ok": True,
                "coinbase_equity": 0.0,
                "kraken_equity": 250.0,
                "kraken_equity_currency_ok": False,  # active + untrusted
            },
            monkeypatch,
        )
    # cycle completed and wrote caps (control flow unaffected)
    assert caps_path.is_file()
    assert "CURRENCY_WARN_SNAPSHOT_LEG leg=kraken" in caplog.text
    # ibkr leg is active + ok -> not warned
    assert "CURRENCY_WARN_SNAPSHOT_LEG leg=ibkr" not in caplog.text
    # derived total is False (one active leg untrusted)
    assert "CURRENCY_WARN_TOTAL_EQUITY_OK_FALSE" in caplog.text


def test_snapshot_leg_silent_when_all_active_legs_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        _refresh(
            tmp_path,
            {
                "ibkr_equity": 100_000.0,
                "ibkr_equity_currency_ok": True,
                "coinbase_equity": 0.0,
                "kraken_equity": 250.0,
                "kraken_equity_currency_ok": True,
            },
            monkeypatch,
        )
    assert "CURRENCY_WARN_SNAPSHOT_LEG" not in caplog.text
    assert "CURRENCY_WARN_TOTAL_EQUITY_OK_FALSE" not in caplog.text


# --------------------------------------------------------------------------- #
# Consumer 4 — CURRENCY_WARN_RISK_CAP_UNVERIFIED
# --------------------------------------------------------------------------- #
def test_risk_cap_warns_when_total_ok_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        caps_path = _refresh(
            tmp_path,
            {
                "ibkr_equity": 100_000.0,
                "ibkr_equity_currency_ok": False,  # untrusted -> total ok False
                "coinbase_equity": 0.0,
            },
            monkeypatch,
        )
    # risk cap was still computed & persisted (warn-only, no control-flow change)
    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["portfolio_risk_cap"] == pytest.approx(10_000.0)
    assert "CURRENCY_WARN_RISK_CAP_UNVERIFIED" in caplog.text


def test_risk_cap_silent_when_total_ok_true_base_cad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        caps_path = _refresh(
            tmp_path,
            {
                "ibkr_equity": 100_000.0,
                "ibkr_equity_currency_ok": True,
                "coinbase_equity": 0.0,
            },
            monkeypatch,
        )
    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity_currency_ok"] is True
    assert "CURRENCY_WARN_RISK_CAP_UNVERIFIED" not in caplog.text


# --------------------------------------------------------------------------- #
# Consumer 5 — CURRENCY_WARN_UNTAGGED_FALLBACK
# --------------------------------------------------------------------------- #
def test_untagged_fallback_warns_when_file_provider_resolves(
    tmp_path: Path, caplog
) -> None:
    # dynamic_caps yields no usable equity (<=0) -> chain falls to FileEquityProvider
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 0.0, "total_equity_currency": "CAD", "total_equity_currency_ok": True},
    )
    _write_json(tmp_path / "runtime" / "positions_snapshot.json", {"net_liquidation": 200_000.0})
    provider = CompositeEquityProvider(
        [
            DynamicCapsEquityProvider(),
            FileEquityProvider("positions_snapshot.json", ("net_liquidation",)),
        ]
    )
    with caplog.at_level(logging.WARNING):
        value, _src = asyncio.run(provider.get_equity(tmp_path))
    assert value == pytest.approx(200_000.0)
    assert "CURRENCY_WARN_UNTAGGED_FALLBACK" in caplog.text


def test_untagged_fallback_silent_when_dyncaps_resolves(
    tmp_path: Path, caplog
) -> None:
    _write_json(
        tmp_path / "runtime" / "dynamic_caps.json",
        {"total_equity": 150_000.0, "total_equity_currency": "CAD", "total_equity_currency_ok": True},
    )
    provider = CompositeEquityProvider(
        [
            DynamicCapsEquityProvider(),
            FileEquityProvider("positions_snapshot.json", ("net_liquidation",)),
        ]
    )
    with caplog.at_level(logging.WARNING):
        value, _src = asyncio.run(provider.get_equity(tmp_path))
    assert value == pytest.approx(150_000.0)
    assert "CURRENCY_WARN_UNTAGGED_FALLBACK" not in caplog.text
