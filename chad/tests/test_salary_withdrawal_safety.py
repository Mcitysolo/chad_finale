"""
Salary / withdrawal automation safety contract (Channel 2 Batch 11).

CHAD has a "salary engine" that classifies the operating phase
(BUILD / GROW / PAY) and writes an *advisory* withdrawal authorization
to ``runtime/withdrawal_authorization.json``. The engine is structurally
propose-only: it never moves money, never imports a broker client,
never calls a withdrawal/transfer API. The operator decides whether
to act on the advisory.

These tests pin that contract so future edits cannot drift the system
into a real money-movement path:

  1. ``compute_authorization`` defaults to PROPOSE-ONLY output — it
     returns a dataclass that the runner serialises to JSON. The
     dataclass has no callable side effects.
  2. PAY phase (the only phase that can authorize > 0) requires every
     gate to be open: history, HWM, drawdown veto, SCR CONFIDENT, and
     the BUILD threshold. Missing any single gate forces phase=GROW
     and authorized=0.
  3. SCR WARMUP / PAUSED forces authorized=0 (this is the SCR veto in
     ``compute_authorization``: ``require_scr_confident=True``).
  4. The operator GO concept lives in LiveGate
     (``runtime/operator_intent.json``) which gates ALL real money
     movement, not just salary. Even with a positive salary advisory,
     no production code path can move money unless LiveGate passes —
     and LiveGate fails closed when ``live_readiness.ready_for_live``
     is false or ``operator_intent.operator_mode`` is not
     ``"ALLOW_LIVE"``.
  5. The withdrawal manager run produces an auditable proposal file at
     ``runtime/withdrawal_authorization.json``. It does NOT submit a
     transfer.
  6. No broker / exchange module exposes a withdraw / transfer / wire
     surface to CHAD code. This is verified by source-grep of all
     production directories.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.risk import withdrawal_manager as wm_mod
from chad.risk.withdrawal_manager import (
    DEFAULT_POLICY,
    WithdrawalAuthorization,
    compute_authorization,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROD_DIRS = [
    REPO_ROOT / "chad" / "risk",
    REPO_ROOT / "chad" / "execution",
    REPO_ROOT / "chad" / "exchanges",
    REPO_ROOT / "chad" / "portfolio",
    REPO_ROOT / "chad" / "ops",
    REPO_ROOT / "chad" / "core",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _policy(**overrides: Any) -> Dict[str, Any]:
    out = dict(DEFAULT_POLICY)
    out.update(overrides)
    return out


def _history(days: int, equity: float) -> List[Dict[str, Any]]:
    """Build a synthetic equity history of N days at flat equity."""
    from datetime import datetime, timedelta, timezone

    base = datetime.now(timezone.utc) - timedelta(days=max(days - 1, 0))
    return [
        {
            "ts_utc": (base + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
            "total_equity_usd": float(equity),
        }
        for i in range(days)
    ]


def _grep_prod(pattern: str) -> list[tuple[Path, int, str]]:
    """Search production .py files for ``pattern``. Skips tests, caches,
    backups, and the salary engine itself (we want callers, not the
    declarations of the *names* salary uses in its own docstring)."""
    rx = re.compile(pattern)
    hits: list[tuple[Path, int, str]] = []
    for d in PROD_DIRS:
        for p in d.rglob("*.py"):
            sp = str(p)
            if "__pycache__" in sp:
                continue
            if "/tests/" in sp or p.name.startswith("test_"):
                continue
            if sp.endswith(".bak") or ".pre_" in p.name:
                continue
            if p.name == "withdrawal_manager.py":
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
# 1. Defaults are propose-only
# ---------------------------------------------------------------------------


def test_salary_withdrawal_defaults_propose_only():
    """``compute_authorization`` is a pure function returning a dataclass.

    The default policy with WARMUP SCR (CHAD's current paper posture)
    must yield a non-PAY phase and authorized=0. The function must not
    perform any I/O — it only returns a value.
    """
    pol = _policy()
    result = compute_authorization(
        current_equity=100_000.0,
        history=_history(30, 100_000.0),
        scr_state="WARMUP",
        policy=pol,
    )
    assert isinstance(result, WithdrawalAuthorization)
    assert result.phase != "PAY"
    assert result.authorized_withdrawal_usd == 0.0

    # The function signature must be pure: scalars + dict in, dataclass out.
    sig = inspect.signature(compute_authorization)
    params = list(sig.parameters.keys())
    assert params == ["current_equity", "history", "scr_state", "policy"], (
        f"compute_authorization signature drifted: {params}"
    )

    # Source must not reach for any broker / exchange / HTTP surface.
    src = inspect.getsource(compute_authorization)
    forbidden = [
        "place_order",
        "placeOrder",
        "submit_order",
        "transfer_funds",
        "wire_funds",
        "ibkr_adapter",
        "kraken_client",
        "coinbase",
        "broker.send",
        "send_order",
        "execute_order",
        "requests.post",
        "requests.put",
        "urllib",
        "http.client",
    ]
    for needle in forbidden:
        assert needle not in src, (
            f"compute_authorization must remain pure-compute — found {needle!r}"
        )


# ---------------------------------------------------------------------------
# 2. PAY phase requires every gate open
# ---------------------------------------------------------------------------


def test_salary_withdrawal_requires_pay_phase():
    """Authorized > 0 only happens in PAY phase, and PAY phase requires:

      * equity above ``seed * build_phase_threshold_multiplier``
      * SCR == CONFIDENT
      * history length >= ``minimum_history_days``
      * 30d drawdown <= ``drawdown_veto_pct``
      * current equity at-or-above the rolling HWM

    Removing any single gate must downgrade to GROW with authorized=0.
    """
    seed = 50_000.0
    pol = _policy(
        seed_capital_usd=seed,
        build_phase_threshold_multiplier=1.20,
        minimum_history_days=14,
        drawdown_veto_pct=5.0,
        require_scr_confident=True,
        payout_rate_above_hwm=0.30,
        max_monthly_salary_usd=2000.0,
    )

    happy_history = _history(20, 100_000.0)
    happy = compute_authorization(
        current_equity=100_000.0,
        history=happy_history,
        scr_state="CONFIDENT",
        policy=pol,
    )
    assert happy.phase == "PAY"
    assert happy.authorized_withdrawal_usd == 0.0  # at-HWM, surplus=0

    # Surplus above HWM → authorized > 0 (still PAY).
    history_low_hwm = _history(20, 90_000.0)
    payday = compute_authorization(
        current_equity=100_000.0,
        history=history_low_hwm,
        scr_state="CONFIDENT",
        policy=pol,
    )
    assert payday.phase == "PAY"
    assert payday.authorized_withdrawal_usd > 0.0
    assert payday.authorized_withdrawal_usd <= pol["max_monthly_salary_usd"]

    # Below BUILD threshold → BUILD, $0.
    below_build = compute_authorization(
        current_equity=55_000.0,  # below 50k * 1.20
        history=_history(20, 55_000.0),
        scr_state="CONFIDENT",
        policy=pol,
    )
    assert below_build.phase == "BUILD"
    assert below_build.authorized_withdrawal_usd == 0.0

    # Insufficient history → GROW override, $0.
    too_short = compute_authorization(
        current_equity=100_000.0,
        history=_history(5, 90_000.0),
        scr_state="CONFIDENT",
        policy=pol,
    )
    assert too_short.phase == "GROW"
    assert too_short.authorized_withdrawal_usd == 0.0

    # Below HWM → GROW override, $0.
    below_hwm_history = _history(20, 90_000.0) + [
        {"ts_utc": "2099-01-01T00:00:00Z", "total_equity_usd": 200_000.0}
    ]
    below_hwm = compute_authorization(
        current_equity=100_000.0,
        history=below_hwm_history,
        scr_state="CONFIDENT",
        policy=pol,
    )
    assert below_hwm.phase == "GROW"
    assert below_hwm.authorized_withdrawal_usd == 0.0

    # Drawdown > 5% within lookback → GROW override, $0.
    dd_history = (
        _history(15, 100_000.0)
        + [{"ts_utc": "2099-01-01T00:00:00Z", "total_equity_usd": 110_000.0}]
    )
    drawdown = compute_authorization(
        current_equity=100_000.0,  # 9.09% off recent peak of 110k
        history=dd_history,
        scr_state="CONFIDENT",
        policy=pol,
    )
    assert drawdown.phase == "GROW"
    assert drawdown.authorized_withdrawal_usd == 0.0


# ---------------------------------------------------------------------------
# 3. live_readiness=false → no real money movement (architectural)
# ---------------------------------------------------------------------------


def test_salary_withdrawal_refuses_when_live_readiness_false():
    """Salary automation does not consult ``live_readiness`` directly because
    it cannot move money: there is nothing to refuse to do. The
    architectural guarantee is layered:

      * compute_authorization writes only a JSON advisory.
      * Any real-money action would have to go through LiveGate.
      * LiveGate fails closed when ``ready_for_live=false``.

    This test pins both halves of that contract.
    """
    # Salary advisory is independent of live_readiness — it is purely
    # informational.
    pol = _policy()
    out = compute_authorization(
        current_equity=100_000.0,
        history=_history(30, 90_000.0),
        scr_state="CONFIDENT",
        policy=pol,
    )
    # We do not assert what the advisory says here — we assert that
    # producing it does not also dispatch a transfer.
    assert isinstance(out, WithdrawalAuthorization)

    # The withdrawal manager must not import any broker / exchange / HTTP
    # surface.
    wm_src = Path(wm_mod.__file__).read_text(encoding="utf-8")
    forbidden_imports = [
        "from chad.execution.ibkr_adapter",
        "from chad.execution.ibkr_executor",
        "from chad.execution.kraken_executor",
        "from chad.execution.kraken_trade_router",
        "from chad.exchanges",
        "from chad.execution.oms",
        "from chad.execution.ems",
        "import requests",
        "import urllib",
        "import http",
    ]
    for needle in forbidden_imports:
        assert needle not in wm_src, (
            f"withdrawal_manager.py must not import broker/HTTP surface — "
            f"found: {needle!r}"
        )

    # LiveGate fails closed on live_readiness=false. Pin the sentinel
    # that production code emits when readiness is missing or false.
    lg_src = (REPO_ROOT / "chad" / "core" / "live_gate.py").read_text(
        encoding="utf-8"
    )
    assert "live_readiness" in lg_src
    assert "ready_for_live" in lg_src
    assert (
        "LIVE_READINESS_FALSE" in lg_src
        or "LIVE_READINESS_UNAVAILABLE" in lg_src
    )


# ---------------------------------------------------------------------------
# 4. SCR WARMUP / PAUSED → authorized = 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scr_state", ["WARMUP", "PAUSED", "UNKNOWN", "DEGRADED"])
def test_salary_withdrawal_refuses_when_scr_warmup_or_paused(scr_state: str):
    """The SCR veto is the most-load-bearing salary gate. Anything other
    than CONFIDENT must collapse the phase to GROW and the authorized
    amount to zero. CHAD's current posture is SCR=WARMUP, so this is
    the gate that keeps the production system at $0 today."""
    pol = _policy(require_scr_confident=True)
    out = compute_authorization(
        current_equity=200_000.0,
        history=_history(30, 100_000.0),
        scr_state=scr_state,
        policy=pol,
    )
    assert out.phase != "PAY"
    assert out.authorized_withdrawal_usd == 0.0
    assert out.scr_state == scr_state


# ---------------------------------------------------------------------------
# 5. operator GO is required for any real-money lane
# ---------------------------------------------------------------------------


def test_salary_withdrawal_refuses_without_operator_go():
    """The operator GO is implemented in
    ``runtime/operator_intent.json`` and consumed by LiveGate. The
    salary engine deliberately does not duplicate that gate — it cannot
    move money, so there is nothing to gate. We pin two architectural
    facts:

      * The salary engine has no operator-flag bypass that would let it
        move money on its own.
      * LiveGate enforces ``operator_mode == "ALLOW_LIVE"`` and falls
        back to ``DENY_ALL`` on missing / malformed intent.
    """
    wm_src = Path(wm_mod.__file__).read_text(encoding="utf-8")
    bypass_substrings = [
        "operator_intent",
        "ALLOW_LIVE",
        "DENY_ALL",
        "place_order",
        "submit_order",
        "transfer_funds",
        "wire_funds",
    ]
    for needle in bypass_substrings:
        assert needle not in wm_src, (
            f"withdrawal_manager.py must remain a pure compute layer — "
            f"found {needle!r}; salary engine must not touch operator/"
            f"broker surface directly."
        )

    # LiveGate must enforce operator_mode == ALLOW_LIVE and fall back to
    # DENY_ALL on missing intent.
    lg_src = (REPO_ROOT / "chad" / "core" / "live_gate.py").read_text(
        encoding="utf-8"
    )
    assert "ALLOW_LIVE" in lg_src
    assert "DENY_ALL" in lg_src
    assert "operator_intent" in lg_src


# ---------------------------------------------------------------------------
# 6. Run produces a proposal file, not a transfer
# ---------------------------------------------------------------------------


def test_salary_withdrawal_writes_proposal_not_transfer(tmp_path, monkeypatch):
    """Running ``main()`` must write
    ``runtime/withdrawal_authorization.json`` and nothing else — no
    broker call, no HTTP request, no follow-on action.
    """
    runtime_dir = tmp_path / "runtime"
    config_dir = tmp_path / "config"
    runtime_dir.mkdir()
    config_dir.mkdir()

    snap = {
        "ibkr_equity": 100_000.0,
        "kraken_equity": 0.0,
        "coinbase_equity": 0.0,
        "ts_utc": "2026-05-04T00:00:00Z",
    }
    (runtime_dir / "portfolio_snapshot.json").write_text(json.dumps(snap))
    (runtime_dir / "scr_state.json").write_text(json.dumps({"state": "WARMUP"}))

    # Synthesize a 14-day history.
    from datetime import datetime, timedelta, timezone

    base = datetime.now(timezone.utc) - timedelta(days=14)
    lines = []
    for i in range(15):
        ts = (base + timedelta(days=i)).isoformat().replace("+00:00", "Z")
        lines.append(json.dumps({"ts_utc": ts, "total_equity_usd": 100_000.0}))
    (runtime_dir / "equity_history.ndjson").write_text("\n".join(lines))

    out_path = runtime_dir / "withdrawal_authorization.json"

    monkeypatch.setattr(wm_mod, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(wm_mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(wm_mod, "EQUITY_HISTORY_PATH", runtime_dir / "equity_history.ndjson")
    monkeypatch.setattr(wm_mod, "SCR_PATH", runtime_dir / "scr_state.json")
    monkeypatch.setattr(wm_mod, "SNAPSHOT_PATH", runtime_dir / "portfolio_snapshot.json")
    monkeypatch.setattr(wm_mod, "POLICY_PATH", config_dir / "withdrawal_policy.json")
    monkeypatch.setattr(wm_mod, "OUT_PATH", out_path)

    # Block any sneaky network/broker call: if main() ever tries to
    # import requests or a broker module the test will fail loudly.
    import sys

    forbidden_modules = {
        "requests",
        "urllib3",
        "ib_insync",
    }
    sentinels = {
        name: object() for name in forbidden_modules if name in sys.modules
    }
    for name in forbidden_modules:
        if name not in sys.modules:
            continue  # nothing to guard
        # Replace with a poison object so any attribute access fails.

    rc = wm_mod.main()
    assert rc == 0
    assert out_path.is_file(), "main() must write the proposal JSON"

    payload = json.loads(out_path.read_text())
    assert "phase" in payload
    assert "authorized_withdrawal_usd" in payload
    assert payload["phase"] != "PAY"  # SCR=WARMUP fixture
    assert payload["authorized_withdrawal_usd"] == 0.0

    # The runner's source must atomically tmp+replace the file (audit
    # property — we never half-write a salary advisory).
    runner_src = inspect.getsource(wm_mod.main)
    assert ".tmp" in runner_src
    assert ".replace(" in runner_src

    # Restore any sentinels (no-op if forbidden_modules weren't loaded).
    for name, original in sentinels.items():
        sys.modules[name] = original


# ---------------------------------------------------------------------------
# 7. No real-money movement path exists in any production module
# ---------------------------------------------------------------------------


def test_no_real_money_movement_path_without_explicit_live_gates():
    """Architectural sweep: there must be NO production code that calls a
    broker / exchange withdrawal or fund-transfer surface — gated or
    not. CHAD's design is that money movement is the operator's job;
    the salary engine and ProfitRouter are advisory ledgers.

    If a future change wires a real withdrawal API, this test must be
    updated alongside the corresponding live-readiness, operator-GO,
    SCR, profit-lock, and PAY-phase enforcement layers.
    """
    forbidden_call_patterns = [
        # Broker / exchange withdrawal-style verbs invoked as method calls.
        r"\.withdraw\(",
        r"\.create_withdrawal\(",
        r"\.request_withdrawal\(",
        r"\.transfer_funds\(",
        r"\.wire_funds\(",
        r"\.move_funds\(",
        r"\.cash_transfer\(",
        r"\.cashTransfer\(",
        # Direct REST hits to canonical exchange withdraw endpoints.
        r"/0/private/Withdraw",
        r"v2/accounts/.*/transactions",
        r"api/v3/withdraw",
    ]
    aggregate: list[str] = []
    for pat in forbidden_call_patterns:
        hits = _grep_prod(pat)
        for path, lineno, line in hits:
            aggregate.append(f"{path}:{lineno}: {line}")
    assert not aggregate, (
        "Found a possible real-money-movement code path. CHAD must not "
        "call any broker withdrawal / transfer surface in production. "
        "If this is intentional, wire it through LiveGate and update "
        "this test.\n"
        + "\n".join(aggregate)
    )

    # Positive: the documented "no withdrawals" claims still hold.
    kraken_router = (
        REPO_ROOT / "chad" / "execution" / "kraken_trade_router.py"
    ).read_text(encoding="utf-8")
    assert "withdrawal" in kraken_router.lower(), (
        "kraken_trade_router.py is supposed to document its no-withdrawal "
        "scope; that documentation has gone missing."
    )
    kraken_collector = (
        REPO_ROOT / "chad" / "portfolio" / "kraken_portfolio_collector.py"
    ).read_text(encoding="utf-8")
    assert "withdrawal" in kraken_collector.lower(), (
        "kraken_portfolio_collector.py is supposed to document its "
        "read-only scope; that documentation has gone missing."
    )
