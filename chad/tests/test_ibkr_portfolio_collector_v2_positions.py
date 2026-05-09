"""
Tests for `python -m chad.portfolio.ibkr_portfolio_collector_v2 positions`.

Covers the positions-only CLI subcommand wired into chad-positions-snapshot
systemd units. The subcommand must:

  * write runtime/positions_snapshot.json atomically with the documented
    schema (positions[], positions_count, ts_utc, ttl_seconds, source),
  * never touch runtime/portfolio_snapshot.json,
  * use a read-only IB connection (collector_v2.collect_positions sets
    `readonly=True` on connect — verified by the stub),
  * return exit code 0 on success and a non-zero code on collector failure.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest

from chad.portfolio import ibkr_portfolio_collector_v2 as collector_module


# ---------------------------------------------------------------------------
# Minimal IB stub — emulates only what collect_positions touches.
# ---------------------------------------------------------------------------


@dataclass
class _FakeContract:
    conId: int
    symbol: str
    secType: str
    currency: str = "USD"


@dataclass
class _FakePosition:
    contract: _FakeContract
    position: float
    avgCost: float


class _FakeIB:
    """Records connect args + returns the seeded positions list."""

    def __init__(self, positions: List[_FakePosition]) -> None:
        self._positions = positions
        self.connect_calls: list[dict] = []
        self.disconnected = False

    def connect(
        self,
        host: str,
        port: int,
        *,
        clientId: int,  # noqa: N803 — IB API spelling
        readonly: bool,
        timeout: float,
    ) -> None:
        self.connect_calls.append(
            {
                "host": host,
                "port": port,
                "clientId": clientId,
                "readonly": readonly,
                "timeout": timeout,
            }
        )

    def positions(self) -> List[_FakePosition]:
        return list(self._positions)

    def disconnect(self) -> None:
        self.disconnected = True


def _install_fake_ib_module(
    monkeypatch: pytest.MonkeyPatch, fake_ib: _FakeIB
) -> None:
    """Install a fake `ib_async` module so the collector picks it up via its
    local `from ib_async import IB` import inside collect_positions."""
    fake_module = types.ModuleType("ib_async")

    def _factory(*args: Any, **kwargs: Any) -> _FakeIB:
        return fake_ib

    fake_module.IB = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ib_async", fake_module)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _seeded_positions() -> List[_FakePosition]:
    return [
        _FakePosition(
            contract=_FakeContract(conId=756733, symbol="SPY", secType="STK"),
            position=2.0,
            avgCost=690.56,
        ),
        _FakePosition(
            contract=_FakeContract(conId=15547841, symbol="TLT", secType="STK"),
            position=-14.0,
            avgCost=86.21,
        ),
        _FakePosition(
            contract=_FakeContract(
                conId=770561194, symbol="MES", secType="FUT", currency="USD"
            ),
            position=-1.0,
            avgCost=31845.63,
        ),
    ]


def test_collect_positions_writes_snapshot_with_expected_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """collect_positions writes positions_snapshot.json atomically with the
    documented schema and seeded broker positions."""
    fake_ib = _FakeIB(_seeded_positions())
    _install_fake_ib_module(monkeypatch, fake_ib)

    cfg = collector_module.IBKRConnectionConfig(
        host="127.0.0.1", port=4002, client_id=9041, account_id="DUK902770"
    )
    collector = collector_module.IBKRPortfolioCollector(cfg)

    out_path = tmp_path / "positions_snapshot.json"
    returned = collector.collect_positions(positions_path=out_path)

    assert returned == out_path, "collect_positions must return the written path"
    assert out_path.is_file(), "positions_snapshot.json must exist after run"

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    # Schema
    for required in ("positions", "positions_count", "ts_utc", "ttl_seconds", "source"):
        assert required in payload, f"missing required field {required!r}"
    assert payload["source"] == "ibkr_portfolio_collector_v2"
    assert payload["positions_count"] == 3
    assert payload["ttl_seconds"] == collector_module.POSITIONS_SNAPSHOT_TTL_SECONDS

    # Per-position fields
    syms = {p["symbol"]: p for p in payload["positions"]}
    assert set(syms) == {"SPY", "TLT", "MES"}
    assert syms["TLT"]["position"] == -14.0
    assert syms["TLT"]["secType"] == "STK"
    assert syms["MES"]["secType"] == "FUT"
    assert syms["SPY"]["conId"] == 756733


def test_collect_positions_uses_readonly_connect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The connection must be read-only and use the configured clientId."""
    fake_ib = _FakeIB(_seeded_positions())
    _install_fake_ib_module(monkeypatch, fake_ib)

    cfg = collector_module.IBKRConnectionConfig(
        host="127.0.0.1", port=4002, client_id=9041, account_id="DUK902770"
    )
    collector_module.IBKRPortfolioCollector(cfg).collect_positions(
        positions_path=tmp_path / "positions_snapshot.json"
    )

    assert fake_ib.connect_calls, "expected exactly one connect call"
    call = fake_ib.connect_calls[0]
    assert call["readonly"] is True, "collect_positions must use readonly connect"
    assert call["clientId"] == 9041, "must propagate configured clientId"
    assert fake_ib.disconnected is True, "must disconnect after read"


def test_cli_positions_subcommand_writes_only_positions_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`python -m ... positions --snapshot-path X` writes positions snapshot
    and DOES NOT touch portfolio_snapshot.json."""
    fake_ib = _FakeIB(_seeded_positions())
    _install_fake_ib_module(monkeypatch, fake_ib)

    pos_path = tmp_path / "positions_snapshot.json"
    portfolio_path = tmp_path / "portfolio_snapshot.json"
    pre_existing = {"ibkr_equity": 99999.0, "kraken_equity": 1.23}
    portfolio_path.write_text(json.dumps(pre_existing), encoding="utf-8")
    portfolio_mtime_before = portfolio_path.stat().st_mtime_ns

    rc = collector_module.main(
        ["positions", "--snapshot-path", str(pos_path)]
    )
    assert rc == 0, "positions subcommand must exit 0 on success"
    assert pos_path.is_file()

    payload = json.loads(pos_path.read_text(encoding="utf-8"))
    assert payload["positions_count"] == 3

    # Critical: portfolio_snapshot.json must remain untouched (no race with
    # chad-portfolio-snapshot.service which owns the equity file).
    assert portfolio_path.stat().st_mtime_ns == portfolio_mtime_before
    assert json.loads(portfolio_path.read_text(encoding="utf-8")) == pre_existing


def test_cli_positions_subcommand_propagates_collector_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the collector raises, the CLI must return a non-zero exit code."""

    class _ExplodingIB(_FakeIB):
        def positions(self) -> List[_FakePosition]:
            raise RuntimeError("simulated IB outage")

    fake_ib = _ExplodingIB([])
    _install_fake_ib_module(monkeypatch, fake_ib)

    pos_path = tmp_path / "positions_snapshot.json"
    rc = collector_module.main(
        ["positions", "--snapshot-path", str(pos_path)]
    )
    assert rc != 0, "must exit non-zero when the collector raises"
    assert not pos_path.exists(), "no snapshot must be written on failure"


def test_cli_positions_no_path_uses_default_runtime_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without --snapshot-path the CLI uses the collector's default location;
    we redirect that default to a tmp dir so the test does not write to the
    repo's real runtime/."""
    fake_ib = _FakeIB(_seeded_positions())
    _install_fake_ib_module(monkeypatch, fake_ib)

    target = tmp_path / "positions_snapshot.json"
    monkeypatch.setattr(
        collector_module.IBKRPortfolioCollector,
        "_default_positions_path",
        staticmethod(lambda: target),
    )

    rc = collector_module.main(["positions"])
    assert rc == 0
    assert target.is_file()
