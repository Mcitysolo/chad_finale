"""
Tier 3B Phase D Item 2 — probe_bag_quotes.py CLI behavior and safety tests.

Validates:
  * default mode is dry-run-fake and prints a quote_check block
  * synthetic fake quotes produce ``ok=True``
  * conflicting flags exit non-zero
  * ``CHAD_EXECUTION_MODE=live`` refusal without ``--live-readonly``
  * expiry / strike / right validators propagate through the spec builder
  * ticker → quote dataclass conversion drops NaN / negative values
  * the live branch can be exercised via a monkeypatched ``ib_async``
  * the script imports no IBKR adapter / strategy / execution surface and
    contains no ``placeOrder`` call
  * the test module itself does not pull live broker dependencies
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = REPO_ROOT / "scripts" / "probe_bag_quotes.py"


def _load_probe_module() -> types.ModuleType:
    """Load ``scripts/probe_bag_quotes.py`` as a module for direct calls
    into ``run_fake``/``run_live`` without spawning a subprocess."""
    spec = importlib.util.spec_from_file_location(
        "_probe_bag_quotes_under_test", PROBE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run_probe(
    args: list[str], extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(PROBE_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )


_BASE_ARGS = [
    "--symbol", "SPY",
    "--expiry", "20260618",
    "--long-strike", "737",
    "--short-strike", "744",
    "--long-right", "C",
    "--short-right", "C",
    "--limit-price", "3.50",
]


# --------------------------------------------------------------------------- #
# 1. default mode                                                             #
# --------------------------------------------------------------------------- #


def test_default_mode_is_dry_run_fake_and_exits_zero() -> None:
    result = _run_probe(_BASE_ARGS)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run_fake"
    assert payload["live_readonly"] is False


# --------------------------------------------------------------------------- #
# 2. JSON shape                                                               #
# --------------------------------------------------------------------------- #


def test_dry_run_fake_produces_quote_check_block() -> None:
    result = _run_probe(_BASE_ARGS)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "quote_check" in payload
    qc = payload["quote_check"]
    for key in (
        "ok",
        "source",
        "mid_debit",
        "limit_price",
        "deviation_abs",
        "deviation_pct",
        "max_allowed_deviation",
        "warnings",
        "errors",
    ):
        assert key in qc, f"quote_check missing key {key!r}"


# --------------------------------------------------------------------------- #
# 3. happy path: ok=True for synthetic quotes                                 #
# --------------------------------------------------------------------------- #


def test_dry_run_fake_ok_for_sane_synthetic_quotes() -> None:
    result = _run_probe(_BASE_ARGS)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["quote_check"]["ok"] is True
    # Default --combo-quote-mode is "legs", so leg_mid is the source.
    assert payload["quote_check"]["source"] == "leg_mid"
    assert payload["quote_check"]["mid_debit"] == pytest.approx(3.50, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4. conflicting flags                                                        #
# --------------------------------------------------------------------------- #


def test_conflicting_dry_run_fake_and_live_readonly_exits_one() -> None:
    result = _run_probe(_BASE_ARGS + ["--dry-run-fake", "--live-readonly"])
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "conflicting_mode_flags" in payload["errors"]


# --------------------------------------------------------------------------- #
# 5. CHAD_EXECUTION_MODE=live refusal                                         #
# --------------------------------------------------------------------------- #


def test_live_execution_mode_without_live_readonly_refuses() -> None:
    result = _run_probe(_BASE_ARGS, extra_env={"CHAD_EXECUTION_MODE": "live"})
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "probe_refused_live_mode_without_live_readonly"
    assert payload["CHAD_EXECUTION_MODE"] == "live"


# --------------------------------------------------------------------------- #
# 6. spec builder rejects invalid expiry                                      #
# --------------------------------------------------------------------------- #


def test_build_spec_from_args_rejects_invalid_expiry() -> None:
    mod = _load_probe_module()
    args = mod.parse_args(
        [
            "--symbol", "SPY",
            "--expiry", "2026-06-18",   # bad: hyphenated, fails ^\d{8}$
            "--long-strike", "737",
            "--short-strike", "744",
            "--long-right", "C",
            "--short-right", "C",
            "--limit-price", "3.50",
        ]
    )
    with pytest.raises(ValueError) as exc_info:
        mod.build_spec_from_args(args)
    assert "expiry" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# 7. JSON output has no runtime mutation fields                               #
# --------------------------------------------------------------------------- #


def test_output_has_no_runtime_mutation_fields() -> None:
    result = _run_probe(_BASE_ARGS)
    assert result.returncode == 0, result.stderr
    # The output JSON must not promise any side-effect-y artifacts.
    forbidden_subs = (
        "runtime/",
        "price_cache.json",
        "positions_snapshot.json",
        "placeOrder",
        "submit_intent",
    )
    for sub in forbidden_subs:
        assert sub not in result.stdout, (
            f"output contains forbidden substring {sub!r}"
        )


# --------------------------------------------------------------------------- #
# 8-10. quote_mode coverage                                                    #
# --------------------------------------------------------------------------- #


def test_fake_mode_supports_quote_mode_legs() -> None:
    result = _run_probe(_BASE_ARGS + ["--combo-quote-mode", "legs"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["quote_mode"] == "legs"
    assert payload["quotes"]["long_leg"] is not None
    assert payload["quotes"]["short_leg"] is not None
    assert payload["quotes"]["combo"] is None
    assert payload["quote_check"]["source"] == "leg_mid"


def test_fake_mode_supports_quote_mode_combo() -> None:
    result = _run_probe(_BASE_ARGS + ["--combo-quote-mode", "combo"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["quote_mode"] == "combo"
    assert payload["quotes"]["long_leg"] is None
    assert payload["quotes"]["short_leg"] is None
    assert payload["quotes"]["combo"] is not None
    assert payload["quote_check"]["source"] == "combo_mid"


def test_fake_mode_supports_quote_mode_both() -> None:
    result = _run_probe(_BASE_ARGS + ["--combo-quote-mode", "both"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["quote_mode"] == "both"
    assert payload["quotes"]["long_leg"] is not None
    assert payload["quotes"]["short_leg"] is not None
    assert payload["quotes"]["combo"] is not None
    # combo mid is preferred over leg mid when both are available.
    assert payload["quote_check"]["source"] == "combo_mid"


# --------------------------------------------------------------------------- #
# 11-12. ticker → quote conversions drop NaN / negative                        #
# --------------------------------------------------------------------------- #


class _FakeTicker:
    def __init__(self, bid=None, ask=None, last=None) -> None:
        self.bid = bid
        self.ask = ask
        self.last = last


def test_ticker_to_leg_quote_converts_nan_and_negative_to_none() -> None:
    mod = _load_probe_module()
    from chad.options.spread_spec import OptionsSpreadSpec

    spec = OptionsSpreadSpec(
        symbol="SPY",
        expiry="20260618",
        long_strike=737.0,
        short_strike=744.0,
        long_right="C",
        short_right="C",
    )
    nan_ticker = _FakeTicker(bid=float("nan"), ask=-1.0, last=0.0)
    leg = mod._ticker_to_leg_quote(spec, "long", nan_ticker, "ibkr_delayed")
    assert leg.bid is None
    assert leg.ask is None
    assert leg.last is None
    # Sanity: positive numbers survive.
    ok_ticker = _FakeTicker(bid=1.0, ask=1.10, last=1.05)
    leg2 = mod._ticker_to_leg_quote(spec, "short", ok_ticker, "ibkr_delayed")
    assert leg2.bid == 1.0
    assert leg2.ask == 1.10
    assert leg2.last == 1.05
    assert leg2.strike == 744.0
    assert leg2.right == "C"


def test_ticker_to_combo_quote_converts_nan_and_negative_to_none() -> None:
    mod = _load_probe_module()
    bad_ticker = _FakeTicker(bid=float("nan"), ask=-0.5, last=float("inf"))
    combo = mod._ticker_to_combo_quote(bad_ticker, "ibkr_delayed")
    assert combo.bid is None
    assert combo.ask is None
    assert combo.last is None
    ok_ticker = _FakeTicker(bid=1.55, ask=1.65, last=None)
    combo2 = mod._ticker_to_combo_quote(ok_ticker, "ibkr_delayed")
    assert combo2.bid == 1.55
    assert combo2.ask == 1.65
    assert combo2.last is None


# --------------------------------------------------------------------------- #
# 13. live mode is monkeypatchable                                             #
# --------------------------------------------------------------------------- #


def test_run_live_is_monkeypatchable_without_real_ibkr() -> None:
    """Demonstrate ``run_live`` can be exercised against a fake ``ib_async``
    module without any network or real broker dependency."""

    import asyncio

    # Build a fake ib_async module and install it on sys.modules so the
    # lazy import inside run_live picks it up. We restore sys.modules at
    # the end of the test.
    saved = {}
    for name in ("ib_async",):
        if name in sys.modules:
            saved[name] = sys.modules[name]

    try:
        fake_mod = types.ModuleType("ib_async")

        class _FakeIB:
            def __init__(self) -> None:
                self.connected = False

            async def connectAsync(self, host, port, clientId):  # noqa: D401
                self.connected = True

            def reqMarketDataType(self, code):  # noqa: D401
                self.last_mdt = code

            def qualifyContracts(self, *contracts):
                # Stamp synthetic conIds.
                for i, c in enumerate(contracts, start=100):
                    c.conId = i
                return list(contracts)

            def reqMktData(self, contract, *args, **kwargs):
                # Return a fake ticker that produces a passing combo mid.
                return _FakeTicker(bid=3.45, ask=3.55, last=None)

            def cancelMktData(self, contract):
                pass

            def disconnect(self):
                self.connected = False

        class _FakeContract:
            def __init__(self) -> None:
                self.symbol = ""
                self.secType = ""
                self.currency = ""
                self.exchange = ""
                self.comboLegs = []

        class _FakeComboLeg:
            def __init__(self) -> None:
                self.conId = 0
                self.ratio = 1
                self.action = ""
                self.exchange = ""

        class _FakeOption:
            def __init__(
                self,
                symbol,
                lastTradeDateOrContractMonth,
                strike,
                right,
                exchange,
                currency,
            ) -> None:
                self.symbol = symbol
                self.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth
                self.strike = strike
                self.right = right
                self.exchange = exchange
                self.currency = currency
                self.conId = 0

        fake_mod.IB = _FakeIB
        fake_mod.Contract = _FakeContract
        fake_mod.ComboLeg = _FakeComboLeg
        fake_mod.Option = _FakeOption
        sys.modules["ib_async"] = fake_mod

        mod = _load_probe_module()
        args = mod.parse_args(
            _BASE_ARGS
            + [
                "--live-readonly",
                "--combo-quote-mode", "combo",
                "--timeout-seconds", "0.1",
            ]
        )
        payload = asyncio.run(mod.run_live(args))

        assert payload["mode"] == "live_readonly"
        assert payload["live_readonly"] is True
        # The fake broker returned a usable combo mid that matches limit_price
        # within the default tolerance.
        assert payload["quote_check"]["source"] == "combo_mid"
        assert payload["quote_check"]["ok"] is True
        assert payload["quote_check"]["mid_debit"] == pytest.approx(3.50, abs=1e-6)
    finally:
        # Restore sys.modules as we found it.
        for name in ("ib_async",):
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)


# --------------------------------------------------------------------------- #
# 14. source safety: no placeOrder call                                       #
# --------------------------------------------------------------------------- #


def test_probe_source_has_no_placeorder_call() -> None:
    """The probe must not have any call whose attribute is ``placeOrder``.

    We parse the AST so docstring text describing what the script does
    *not* do is ignored.
    """
    src = PROBE_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_calls = {
        "placeOrder",
        "placeOrderAsync",
        "submit_intent",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            attr = None
            if isinstance(func, ast.Name):
                attr = func.id
            elif isinstance(func, ast.Attribute):
                attr = func.attr
            if attr is not None:
                assert attr not in forbidden_calls, (
                    f"probe script must not call {attr!r}"
                )


# --------------------------------------------------------------------------- #
# 15. source safety: no chad.execution / chad.strategies imports              #
# --------------------------------------------------------------------------- #


def test_probe_source_does_not_import_execution_or_strategies() -> None:
    src = PROBE_SCRIPT.read_text(encoding="utf-8")
    # Regex sweeps target real import statements, not docstring prose that
    # documents what the probe deliberately does not do. (``IbkrAdapter`` is
    # checked separately via AST so docstring mentions are ignored.)
    banned_import_patterns = [
        r"^\s*import\s+chad\.execution",
        r"^\s*from\s+chad\.execution\b",
        r"^\s*import\s+chad\.strategies",
        r"^\s*from\s+chad\.strategies\b",
    ]
    for pat in banned_import_patterns:
        assert (
            re.search(pat, src, flags=re.MULTILINE) is None
        ), f"forbidden pattern {pat!r} found in probe_bag_quotes.py"

    # AST-level sweep for import statements + code-level name references.
    # This ignores docstring prose entirely.
    tree = ast.parse(src)
    name_uses: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")
                assert root[:2] != ["chad", "execution"], (
                    f"probe imports chad.execution module {alias.name!r}"
                )
                assert root[:2] != ["chad", "strategies"], (
                    f"probe imports chad.strategies module {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("chad.execution"), (
                f"probe imports from {mod!r}"
            )
            assert not mod.startswith("chad.strategies"), (
                f"probe imports from {mod!r}"
            )
        elif isinstance(node, ast.Name):
            name_uses.append(node.id)
        elif isinstance(node, ast.Attribute):
            name_uses.append(node.attr)
    assert "IbkrAdapter" not in name_uses, (
        "probe must not reference IbkrAdapter at code level"
    )


# --------------------------------------------------------------------------- #
# 16. test module makes no live network calls                                  #
# --------------------------------------------------------------------------- #


def test_test_module_makes_no_live_network_calls() -> None:
    """Audit this test module itself: it must not import live broker
    libraries at top level, and it must not contain real socket / network
    helpers that would reach IB Gateway. The monkeypatched ``ib_async``
    used in test 13 is injected into ``sys.modules`` from inside the test
    body — a top-level ``import ib_async`` would be rejected here."""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in {"ib_async", "ib_insync"}, (
                    f"test module must not import {alias.name!r}"
                )
                assert "ibkr_adapter" not in alias.name, (
                    f"test must not import IBKR adapter ({alias.name!r})"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0] if mod else ""
            assert root not in {"ib_async", "ib_insync"}, (
                f"test must not import {mod!r}"
            )
            assert "ibkr_adapter" not in mod, (
                f"test must not import IBKR adapter ({mod!r})"
            )

    # Reject any obvious socket / urllib / requests usage that could reach
    # a live broker. We accept only ``subprocess`` (used to drive the
    # probe under test).
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for n in names:
                root = n.split(".")[0]
                assert root not in {
                    "socket",
                    "urllib",
                    "requests",
                    "httpx",
                    "aiohttp",
                }, f"test must not import network library {n!r}"
