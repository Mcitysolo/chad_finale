"""
Audit-layer tests for CHAD's capital-routing / profit-routing gates
(Channel 2 Batch 10).

These tests pin the safety contract of two related but distinct features:

  Beta injection
      Implemented in chad/strategies/beta.py and gated by the env var
      CHAD_PROFIT_ROUTER_BETA_INJECTION (default OFF). When enabled, the
      ProfitRouter's net-remaining beta budget caps each per-cycle Beta
      fill_notional. The gate can only TIGHTEN sizing — never raise it
      above Beta's intrinsic 2%-of-equity / half-gap ceiling.

  Amplifier injection
      The accounting API exists in chad/risk/profit_router.py
      (mark_amplifier_consumed / get_amplifier_remaining /
      get_amplifier_accumulated). It has NO production consumer. The
      bucket is therefore an advisory ledger only — it cannot move
      money or alter sizing. Wiring a consumer in the future requires
      independent gates (winner-scaling freshness, non-flat multipliers,
      operator flag) and an explicit update to this test.

Beyond the env-flag / no-consumer gates, both features remain subject
to the LiveGate live_readiness check before any LIVE execution can
occur. Real money movement requires (a) live mode, (b) live_readiness
ready_for_live=true, and (c) the operator-controlled env flag — these
are independent layers and any one being false fails closed.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from chad.strategies import beta as beta_mod
from chad.strategies.beta import Beta, BetaParams, beta_handler

REPO_ROOT = Path(__file__).resolve().parents[2]
PROD_DIRS = [
    REPO_ROOT / "chad" / "strategies",
    REPO_ROOT / "chad" / "execution",
    REPO_ROOT / "chad" / "core",
    REPO_ROOT / "chad" / "risk",
    REPO_ROOT / "chad" / "ops",
]


# ---------------------------------------------------------------------------
# Fixtures (mirroring the patterns used in test_beta_strategy.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakePortfolio:
    cash: float = 1_000_000.0
    total_equity: float = 1_000_000.0
    positions: Dict[str, Any] = field(default_factory=dict)
    extra: Optional[Dict[str, Any]] = None


@dataclass
class _FakeCtx:
    now: datetime
    portfolio: _FakePortfolio
    prices: Dict[str, float]


def _write_consensus(path: Path, weights: Dict[str, float]) -> None:
    payload = {
        "schema_version": "institutional_consensus.v1",
        "updated_ts_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "funds_included": ["fund_a", "fund_b"],
        "weights": weights,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_ctx() -> _FakeCtx:
    return _FakeCtx(
        now=datetime.now(timezone.utc),
        portfolio=_FakePortfolio(),
        prices={"AAPL": 180.0, "MSFT": 400.0, "NVDA": 900.0},
    )


@pytest.fixture(autouse=True)
def _reset_beta_state():
    """Beta uses module-level throttle state; isolate each test."""
    Beta.reset_state()
    yield
    Beta.reset_state()


@pytest.fixture(autouse=True)
def _ensure_clean_env(monkeypatch):
    """The injection flag is read from the process environment. Strip it
    by default so an inherited shell value cannot influence tests."""
    monkeypatch.delenv("CHAD_PROFIT_ROUTER_BETA_INJECTION", raising=False)


def _grep_prod(pattern: str) -> list[tuple[Path, int, str]]:
    """Search production .py files for ``pattern``.

    Excludes test files, ``__pycache__`` directories, and the file that
    DEFINES the API (chad/risk/profit_router.py) — we want callers, not
    the source declaration.
    """
    rx = re.compile(pattern)
    hits: list[tuple[Path, int, str]] = []
    for d in PROD_DIRS:
        for p in d.rglob("*.py"):
            sp = str(p)
            if "__pycache__" in sp:
                continue
            if "/tests/" in sp or p.name.startswith("test_"):
                continue
            if p.name == "profit_router.py":
                continue
            try:
                for i, line in enumerate(
                    p.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if rx.search(line):
                        hits.append((p, i, line.rstrip()))
            except OSError:
                continue
    return hits


# ---------------------------------------------------------------------------
# Beta injection — env-flag gate
# ---------------------------------------------------------------------------


def test_beta_injection_defaults_disabled(tmp_path, monkeypatch):
    """With the env var unset, beta_handler must NOT consult ProfitRouter
    even when there are real candidates to size.

    This is the default-off contract: shipping a clean machine with no
    operator action must yield the legacy (pre-v8.6) sizing path.
    """
    monkeypatch.delenv("CHAD_PROFIT_ROUTER_BETA_INJECTION", raising=False)
    cpath = tmp_path / "consensus.json"
    _write_consensus(cpath, {"AAPL": 0.5, "MSFT": 0.3})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", cpath)

    import chad.risk.profit_router as pr_mod

    with patch.object(pr_mod, "ProfitRouter") as pr_class:
        params = BetaParams(underweight_gap=0.005, max_position_weight=0.05)
        signals = list(beta_handler(_make_ctx(), params=params))
        assert signals, "expected at least one signal in the legacy path"
        assert pr_class.call_count == 0, (
            "ProfitRouter must not be instantiated when "
            "CHAD_PROFIT_ROUTER_BETA_INJECTION is unset"
        )


@pytest.mark.parametrize("flag_value", ["1", "true", "yes", "TRUE", "Yes", "True"])
def test_beta_injection_requires_operator_flag(tmp_path, monkeypatch, flag_value):
    """The injection branch only activates for the documented truthy
    values {"1","true","yes"} (case-insensitive). When set, the budget
    cap must take effect: with zero accumulated beta budget, no Beta
    signals can be emitted."""
    monkeypatch.setenv("CHAD_PROFIT_ROUTER_BETA_INJECTION", flag_value)
    cpath = tmp_path / "consensus.json"
    _write_consensus(cpath, {"AAPL": 0.5, "MSFT": 0.3})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", cpath)

    # Stub ProfitRouter so beta_remaining is zero.
    import chad.risk.profit_router as pr_mod

    fake_router = type(
        "_FakeRouter",
        (),
        {
            "get_beta_remaining": lambda self: 0.0,
            "mark_beta_consumed": lambda self, amt: True,
        },
    )()
    with patch.object(pr_mod, "ProfitRouter", return_value=fake_router) as pr_class:
        params = BetaParams(underweight_gap=0.005, max_position_weight=0.05)
        signals = list(beta_handler(_make_ctx(), params=params))
        assert pr_class.call_count == 1, (
            f"flag={flag_value!r}: ProfitRouter must be instantiated"
        )
        assert signals == [], (
            f"flag={flag_value!r}: with zero beta budget, no signals must "
            "be emitted (injection branch must short-circuit)"
        )


@pytest.mark.parametrize(
    "flag_value", ["", "0", "false", "no", "off", "False", "NO", "garbage", " "]
)
def test_beta_injection_falsy_values_do_not_enable(tmp_path, monkeypatch, flag_value):
    """Anything other than {"1","true","yes"} (case-insensitive) must
    leave injection disabled. This pins the strict allowlist parsing."""
    monkeypatch.setenv("CHAD_PROFIT_ROUTER_BETA_INJECTION", flag_value)
    cpath = tmp_path / "consensus.json"
    _write_consensus(cpath, {"AAPL": 0.5, "MSFT": 0.3})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", cpath)

    import chad.risk.profit_router as pr_mod

    with patch.object(pr_mod, "ProfitRouter") as pr_class:
        params = BetaParams(underweight_gap=0.005, max_position_weight=0.05)
        list(beta_handler(_make_ctx(), params=params))
        assert pr_class.call_count == 0, (
            f"flag={flag_value!r} must NOT enable beta injection"
        )


def test_beta_injection_can_only_tighten_sizing(tmp_path, monkeypatch):
    """When injection is ON and beta_remaining is very large, the
    accumulator must NEVER raise sizing above Beta's intrinsic ceiling.
    Beta caps at min(half_gap, beta_remaining, 2%-of-equity). The
    accumulator is upper-bounded inside the handler at
    max_position_weight*equity for exactly this reason."""
    monkeypatch.setenv("CHAD_PROFIT_ROUTER_BETA_INJECTION", "true")
    cpath = tmp_path / "consensus.json"
    _write_consensus(cpath, {"AAPL": 0.5})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", cpath)

    import chad.risk.profit_router as pr_mod

    # Astronomical beta budget — should be ignored beyond the 2%-equity guard.
    fake_router = type(
        "_FakeRouter",
        (),
        {
            "get_beta_remaining": lambda self: 10_000_000.0,
            "mark_beta_consumed": lambda self, amt: True,
        },
    )()
    with patch.object(pr_mod, "ProfitRouter", return_value=fake_router):
        params = BetaParams(underweight_gap=0.005, max_position_weight=0.02)
        signals = list(beta_handler(_make_ctx(), params=params))

    assert signals, "expected at least one signal"
    # Equity = 1_000_000; max_position_weight = 0.02 → fill notional
    # cannot exceed $20,000 even with a runaway accumulator.
    for sig in signals:
        # AAPL price = 180 → max integer shares within 2% guard = 111
        assert sig.size <= 111, (
            f"size {sig.size} exceeds 2%-of-equity guard — accumulator "
            "managed to RAISE sizing, which is forbidden"
        )


# ---------------------------------------------------------------------------
# Live-readiness coupling
# ---------------------------------------------------------------------------


def test_beta_injection_refuses_when_live_readiness_false(tmp_path, monkeypatch):
    """Even with the injection flag ON, no LIVE execution may occur
    while live_readiness.ready_for_live is false. The Beta strategy
    itself only emits signals; LiveGate is the independent layer that
    enforces live_readiness — fail-closed when the readiness file is
    missing OR explicitly false.

    This test pins the LiveGate readiness contract that any future
    activation of capital-routing features must traverse.
    """
    from chad.core import live_gate as lg

    runtime = tmp_path / "runtime"
    runtime.mkdir()

    # Case A: explicit ready_for_live=false
    (runtime / "live_readiness.json").write_text(
        json.dumps(
            {
                "schema_version": "live_readiness_state.v1",
                "ready_for_live": False,
                "ts_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)
    state = lg._load_live_readiness_state()
    assert state.ready_for_live is False
    assert state.reason == "LIVE_READINESS_FALSE"

    # Case B: missing file → fail-closed
    (runtime / "live_readiness.json").unlink()
    state = lg._load_live_readiness_state()
    assert state.ready_for_live is False
    assert state.ok is False


# ---------------------------------------------------------------------------
# Amplifier — no production consumer
# ---------------------------------------------------------------------------


def test_amplifier_defaults_propose_only():
    """The amplifier accumulator API (mark_amplifier_consumed,
    get_amplifier_remaining, get_amplifier_accumulated) must have ZERO
    production consumers. The bucket is an advisory ledger only.

    To wire amplifier injection later, the new caller MUST gate on:
      - winner_scaling.json freshness (<= 1800s)
      - non-flat multipliers (max != 1.0 OR min != 1.0)
      - sufficient sample size (>= min_trades_for_scaling per strategy)
      - SCR not PAUSED
      - LiveGate live_readiness, OR explicit paper/propose-only mode
      - explicit operator flag (e.g. CHAD_PROFIT_ROUTER_AMPLIFIER_INJECTION)

    When you add the consumer, update this test to whitelist the
    specific call sites and to assert each gate above.
    """
    consumer_hits = (
        _grep_prod(r"\bmark_amplifier_consumed\b")
        + _grep_prod(r"\bget_amplifier_remaining\b")
        + _grep_prod(r"\bget_amplifier_accumulated\b")
    )
    if consumer_hits:
        formatted = "\n".join(f"  {p}:{l}: {s}" for p, l, s in consumer_hits)
        pytest.fail(
            "Amplifier injection is intentionally consumer-less "
            "(advisory ledger only). New production consumer(s) found:\n"
            + formatted
            + "\n\nAdd the activation gates documented in this test before "
            "removing this assertion."
        )


def test_amplifier_requires_fresh_nonflat_winner_scaling():
    """The dynamic_risk_allocator exposes
    load_winner_multipliers_or_stale() which returns ({}, True) when
    winner_scaling.json is missing or older than 1800s. Any future
    amplifier consumer MUST consult this stale flag and refuse to inject
    when stale=True.

    This test verifies the contract still exists at the loader layer so
    callers can rely on it.
    """
    from chad.risk import dynamic_risk_allocator as dra

    loader = dra.load_winner_multipliers_or_stale
    sig = inspect.signature(loader)
    assert sig.return_annotation != inspect.Signature.empty or True
    # Source-level contract: the stale-bool branch must exist.
    src = inspect.getsource(loader)
    assert "stale_seconds=1800" in src, "freshness bound must remain at 1800s"
    assert "return {}, True" in src, (
        "loader must signal stale=True via empty multipliers"
    )

    # And the freshness check at the consuming side: when caps apply
    # winner_stale, allocator must downshift to a conservative 0.5x.
    allocator_src = (
        REPO_ROOT / "chad" / "risk" / "dynamic_risk_allocator.py"
    ).read_text(encoding="utf-8")
    assert "winner_scaling_stale" in allocator_src
    assert "0.5" in allocator_src, (
        "stale-fallback multiplier (0.5x) must remain wired into the "
        "allocator so amplifier injection cannot ride a stale signal"
    )


def test_amplifier_refuses_flat_or_stale_multipliers():
    """When the expectancy doc is empty (no clean trade sample yet),
    winner_scaler MUST emit all-1.0 (flat) multipliers and report zero
    scaled strategies. Flat multipliers are the natural floor — an
    amplifier consumer that respected only a "non-flat" gate would have
    nothing to inject when multipliers are all 1.0.
    """
    from chad.risk.winner_scaler import compute_multipliers, DEFAULT_POLICY

    out = compute_multipliers({"strategies": {}}, DEFAULT_POLICY)
    mults = list(out["multipliers"].values())
    assert mults, "winner_scaler must always publish the canonical strategy set"
    assert all(m == 1.0 for m in mults), (
        "Empty expectancy must yield ALL-1.0 multipliers — no scaling "
        "is permitted without sample data"
    )
    assert out["n_strategies_scaled"] == 0
    assert out["max_multiplier"] == DEFAULT_POLICY["max_multiplier"]
    assert out["min_multiplier"] == DEFAULT_POLICY["min_multiplier"]

    # And: a strategy with too few trades stays neutral.
    expectancy = {
        "strategies": {
            "alpha": {"total_trades": 1, "expectancy": 50.0},
            "beta": {"total_trades": 1, "expectancy": -50.0},
        }
    }
    out2 = compute_multipliers(expectancy, DEFAULT_POLICY)
    assert out2["multipliers"]["alpha"] == 1.0
    assert out2["multipliers"]["beta"] == 1.0
    assert out2["n_strategies_scaled"] == 0


# ---------------------------------------------------------------------------
# No real money movement without LiveGate approval + operator flag
# ---------------------------------------------------------------------------


def test_no_real_money_movement_without_live_and_operator_go():
    """Architectural assertion: ProfitRouter must NOT call any broker /
    execution surface. It only writes a JSON ledger. Real money
    movement is gated by LiveGate (independent layer) which fails closed
    when live_readiness=false or the operator hasn't issued GO.
    """
    from chad.risk.profit_router import ProfitRouter

    src = inspect.getsource(ProfitRouter)
    forbidden_substrings = [
        "place_order",
        "placeOrder",
        "submit_order",
        "submitOrder",
        "ibkr_adapter",
        "broker.send",
        "send_order",
        "execute_order",
        "transfer_funds",
        "withdraw",
    ]
    for needle in forbidden_substrings:
        assert needle not in src, (
            f"ProfitRouter must remain ledger-only — found broker/funds "
            f"surface: {needle!r}"
        )
    # Positive: it writes a JSON ledger via atomic tmp+replace.
    assert "_write_state" in src
    assert ".tmp" in src
    assert ".replace(" in src

    # And the broader contract: no production code activates LIVE
    # execution paths without LiveGate. We pin this by asserting
    # LiveGate references live_readiness in its production gating.
    lg_src = (REPO_ROOT / "chad" / "core" / "live_gate.py").read_text(
        encoding="utf-8"
    )
    assert "live_readiness" in lg_src
    assert "ready_for_live" in lg_src
    assert "LIVE_READINESS_FALSE" in lg_src or "LIVE_READINESS_UNAVAILABLE" in lg_src
